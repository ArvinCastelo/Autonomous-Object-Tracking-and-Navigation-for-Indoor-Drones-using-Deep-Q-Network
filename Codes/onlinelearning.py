"""
Random exploration data‑collection script for a Crazyflie‑style quadrotor in Webots.

Features
--------
* **Robust start‑up & take‑off:** waits for valid GPS, then rises smoothly to a safe
  hover height.
* **Action repeat:** each discrete action is applied for `ACTION_REPEAT` controller
  steps to avoid flickering inputs and to give the drone time to respond.
* **Safety monitors:** terminates the episode if the drone gets too close to an
  obstacle (`front_dist`), tips over, or hits altitude limits.  These rules can
  be adapted for your world.
* **CSV logging:** every transition
  \(state, action, reward, next_state, done) is appended to a timestamped file
  in the `dataset/` directory, ready for offline RL algorithms (e.g. DQN).

The script uses the minimal, debug‑friendly `WebotsDroneAPI` that comes with
this repo (see *webots_drone_api.py*).  Make sure you have that file in the same
folder or on your Python path.
"""

import csv
import os
import random
from datetime import datetime
import math
import numpy as np
import matplotlib.pyplot as plt

from discrete_controller import DiscreteController
from webots_drone_api import WebotsDroneAPI

from pid_controller import pid_velocity_fixed_height_controller

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# ---------------------------------------------------------------------------
# Hyper‑parameters (tweak as you like)
# ---------------------------------------------------------------------------
TIMESTEP           = 32          # [ms] – must match World > basicTimeStep
MAX_EPISODE_STEPS  = 500      # safety stop (≈ 13 min at 32 ms)
ACTION_REPEAT      = 5           # how many control cycles per action
SAFE_Z             = 0.8         # [m] – target hover height after take‑off
DATA_DIR           = "dataset9"   # where rollouts are stored
WALL_THRESHOLD     = 0.3        # meters
AVOID_DURATION     = 10          # how many steps to avoid after trigger

# Hyperparameters for DQN
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-4
TAU = 0.005
INPUT_DIM = 15  # 13 state vars + 2 YOLO flags
ACTION_DIM = 13  # Actions in DiscreteController
BUFFER_CAPACITY = 10000

# Hyperparameters for Plotting
PLOT_DIR = "learning_curves"
SAVE_PLOT_EVERY = 10

# ---------------------------------------------------------------------------

# Utility helpers
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, transition):
        self.buffer.append(transition)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)


# Ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def csv_header():
    return [
        "t",
        "x", "y", "z", "roll", "pitch", "yaw",
        "front_dist", "left_dist", "right_dist", "back_dist",
        "vx", "vy", "vz",
        "action_id", "reward", "done",
        "next_x", "next_y", "next_z", "next_roll", "next_pitch", "next_yaw",
        "next_front_dist", "next_left_dist", "next_right_dist", "next_back_dist",
        "next_vx", "next_vy", "next_vz",
        "yolo_detect_any", "yolo_detect_target"
    ]

# ---------------------------------------------------------------------------
# Simple reward & termination
# ---------------------------------------------------------------------------

def compute_reward(state: dict, drone: WebotsDroneAPI):
    """
    Return (reward, done_flag, detected_something) where:
    - reward: numeric reward
    - done_flag: whether the episode should terminate
    - detected_something: True if anything (not necessarily the target) was detected by YOLO
    - detected_target: True if the target is detected
    """

    # crash checks
    if state["front_dist"] < WALL_THRESHOLD or state["back_dist"] < WALL_THRESHOLD \
        or state["right_dist"] < WALL_THRESHOLD or state["left_dist"] < WALL_THRESHOLD:
        return -1.0, True, False, False

    if abs(state["roll"]) > 1.0 or abs(state["pitch"]) > 1.0:
        return -1.0, True, False, False

    image = drone.get_camera_image()
    results = model.predict(image, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("[YOLO] No detections.")
        return 0.0, False, False, False

    print(f"[YOLO] Found {len(boxes)} objects.")
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        print(f"  - Detected {label} ({conf:.2f})")
        if label == "bottle" and conf > 0.25:
            print("[YOLO] Detected TARGET: bottle!")
            return +10.0, True, True, True

    # Detected something, but not the target
    return +0.05, False, True, False

# ---------------------------------------------------------------------------
# state to tensor
# ---------------------------------------------------------------------------
# Initialize DQN and buffer
policy_net = QNetwork(INPUT_DIM, ACTION_DIM)
target_net = QNetwork(INPUT_DIM, ACTION_DIM)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

def state_to_tensor(state):
    return torch.tensor([
        state['x'], state['y'], state['z'],
        state['roll'], state['pitch'], state['yaw'],
        state['front_dist'], state['left_dist'], state['right_dist'], state['back_dist'],
        state['vx'], state['vy'], state['vz'],
        state['yolo_detect_any'], state['yolo_detect_target']
    ], dtype=torch.float32)

# ---------------------------------------------------------------------------
# selection with epsilon-greedy
# ---------------------------------------------------------------------------
def select_action(state, available_ids, steps_done):
    epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    if random.random() > epsilon:
        with torch.no_grad():
            state_tensor = state_to_tensor(state)
            q_values = policy_net(state_tensor)
            valid_q = [q_values[aid] for aid in available_ids]
            action_id = available_ids[torch.argmax(torch.stack(valid_q)).item()]
    else:
        action_id = random.choice(available_ids)
    return action_id, epsilon

# ---------------------------------------------------------------------------
# update model
# ---------------------------------------------------------------------------
def update_model():
    if len(replay_buffer) < BATCH_SIZE:
        return 0.0
    transitions = replay_buffer.sample(BATCH_SIZE)
    state_batch = torch.tensor([t[0] for t in transitions], dtype=torch.float32)
    action_batch = torch.tensor([t[1] for t in transitions], dtype=torch.long)
    reward_batch = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
    next_state_batch = torch.tensor([t[3] for t in transitions], dtype=torch.float32)
    done_batch = torch.tensor([t[4] for t in transitions], dtype=torch.float32)

    current_q = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    next_q = target_net(next_state_batch).max(1)[0].detach()
    expected_q = reward_batch + (1 - done_batch) * GAMMA * next_q

    loss = nn.functional.mse_loss(current_q.squeeze(), expected_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    # Soft update target network
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
    return loss.item()

episode_rewards = []  # Store cumulative rewards per episode
rolling_avg = []      # Store rolling average rewards
window_size = 10      # For rolling average calculation
LOSS_PLOT_WINDOW = 10
# Add these global variables
episode_losses = []
loss_rolling_avg = []

def plot_learning_curves(episode_rewards, rolling_avg, losses, loss_avg, current_episode):
    ensure_dir(PLOT_DIR)
    plt.figure(figsize=(12, 10))

    # Reward plot (top)
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards, 'b-', alpha=0.3, label='Episode Reward')
    if rolling_avg:
        x_vals = np.arange(window_size - 1, window_size - 1 + len(rolling_avg))
        plt.plot(x_vals, rolling_avg, 'r-', linewidth=2, label=f'{window_size}-episode Avg')
    plt.title(f'Training Progress (Episode {current_episode})')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=10, color='g', linestyle='--', alpha=0.5, label='Target Reward')

    # Loss plot (bottom)
    plt.subplot(2, 1, 2)
    plt.plot(losses, 'm-', alpha=0.3, label='Batch Loss')
    if loss_avg:
        x_vals = np.arange(LOSS_PLOT_WINDOW - 1, LOSS_PLOT_WINDOW - 1 + len(loss_avg))
        plt.plot(x_vals, loss_avg, 'c-', linewidth=2, label=f'{LOSS_PLOT_WINDOW}-batch Avg')
    plt.xlabel('Training Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale often helps visualize loss trends

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f'training_progress_ep{current_episode}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved training progress to {plot_path}")
# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

# Main method
def main():
    num_episodes = 300
    steps_done = 0

    # Initialize Supervisor and drone API ONCE outside the loop
    drone = WebotsDroneAPI(TIMESTEP)
    supervisor = drone.robot
    ctrl = DiscreteController()
    drone_node = supervisor.getSelf()
    # ---------- Reset Drone Position ---------- #
    drone_node = supervisor.getSelf()  # Get drone node once
    initial_translation = [0, 0, 0.715772]



    for episode in range(num_episodes):
        print(f"\n[INFO] Starting Episode {episode + 1}/{num_episodes}")
        episode_reward = 0
        # ---------- log file ---------- #
        ensure_dir(DATA_DIR)
        filename = os.path.join(DATA_DIR, f"dataset_{datetime.now():%Y%m%d_%H%M%S}_{episode}.csv")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header())

            # Reset simulation
            drone_node.getField("translation").setSFVec3f(initial_translation)
            # Reset simulation state
            supervisor.simulationReset()
            supervisor.step(TIMESTEP)  # Step to apply reset
            drone.pid = pid_velocity_fixed_height_controller()  # Reset PID
            drone._valid_altitude = False  # Force re-initialize altitude
            drone.reset()

            no_detection_steps = 0
            NO_DETECTION_LIMIT = 10  # steps without detection before penalizing

            # ---------- wait for valid GPS ---------- #
            print("[INFO] Waiting for GPS altitude to become valid …")
            while True:
                s = drone.get_state()
                if not math.isnan(s["z"]):
                    break
                drone.step()
            print("[INFO]   → sensors ready")

            # ---------- take‑off ---------- #
            print("[INFO] Taking off …")
            takeoff_start = drone.robot.getTime()
            while True:
                s = drone.get_state()
                if s["z"] >= SAFE_Z - 0.05:
                    break
                if drone.robot.getTime() - takeoff_start > 30:
                    print("[ERROR] Takeoff timeout!")
                    break
                drone.send_command(0.0, 0.0, 0.0, +0.8)
                drone.step()
            print(f"[INFO] Hover reached  {s['z']:.2f} m – starting exploration")

            # ---------- random exploration ---------- #
            step_idx = 0
            done = False
            state = s
            state['yolo_detect_any'] = 0  # Initialize detection flags
            state['yolo_detect_target'] = 0

            # Move model saving outside the step loop
            if episode % 10 == 0 and step_idx == 0:  # Save at start of every 10th episode
                torch.save(policy_net.state_dict(), f"policy_net_episode_{episode}.pth")

            while step_idx < MAX_EPISODE_STEPS and not done:

                # --- Smart Obstacle Avoidance --- #
                available_ids = []
                if state["front_dist"] > WALL_THRESHOLD:
                    available_ids.append(ctrl.forward)
                if state["back_dist"] > WALL_THRESHOLD:
                    available_ids.append(ctrl.backward)
                if state["left_dist"] > WALL_THRESHOLD:
                    available_ids.append(ctrl.left)
                if state["right_dist"] > WALL_THRESHOLD:
                    available_ids.append(ctrl.right)


                # Always allow hover, ascend, descend
                available_ids.extend([ctrl.hover,
                                      ctrl.ascend,
                                      ctrl.descend,
                                      ctrl.yaw_left,
                                      ctrl.yaw_right,
                                      ctrl.halfspinleft,
                                      ctrl.halfspinright,
                                      ctrl.spright,
                                      ctrl.spleft])

                # Select action using epsilon-greedy policy
                action_id, epsilon = select_action(state, available_ids, steps_done)
                steps_done += 1
                vx, vy, yaw_rate, dz = ctrl.get_command(action_id)


                for i in range(ACTION_REPEAT):
                    if done:
                        break

                    # Add brief hover before action to stabilize
                    if i == 0:
                        for _ in range(int(1 / (TIMESTEP / 1000))):
                            drone.send_command(0.0, 0.0, 0.0, 0.0)
                            drone.step()

                    drone.send_command(vx, vy, yaw_rate, dz)
                    drone.step()

                    next_state = drone.get_state()
                    reward, done, detected_something, detected_target = compute_reward(next_state, drone)

                    episode_reward += reward

                    next_state['yolo_detect_any'] = int(detected_something)
                    next_state['yolo_detect_target'] = int(detected_target)

                    # Store transition in replay buffer
                    transition = (
                        state_to_tensor(state).tolist(),
                        action_id,
                        reward,
                        state_to_tensor(next_state).tolist(),
                        done
                    )
                    replay_buffer.push(transition)

                    # Train the model
                    loss = update_model()  # Now returns the loss value
                    episode_losses.append(loss)

                    next_vz = (next_state["z"] - state["z"]) / (TIMESTEP / 1000)

                    # penalize if no detection at all for a while
                    if not detected_something:
                        no_detection_steps += 1
                        if no_detection_steps >= NO_DETECTION_LIMIT:
                            print("[INFO] Penalized: no object detected for a while.")
                            reward -= 0.1
                            no_detection_steps = 0
                    else:
                        no_detection_steps = 0

                    writer.writerow([
                        step_idx,
                        state["x"], state["y"], state["z"], state["roll"], state["pitch"], state["yaw"],
                        state["front_dist"], state["left_dist"], state["right_dist"], state["back_dist"],
                        state["vx"], state["vy"], state["vz"],
                        action_id, reward, int(done),
                        next_state["x"], next_state["y"], next_state["z"], next_state["roll"], next_state["pitch"], next_state["yaw"],
                        next_state["front_dist"], next_state["left_dist"], next_state["right_dist"], next_state["back_dist"],
                        next_state["vx"], next_state["vy"], next_vz,
                        int(detected_something), int(detected_target)
                    ])
                    state = next_state
                    step_idx += 1

                    # Termination conditions
                    if reward == 10.0:  # Target found
                        print("[SUCCESS] Target found!")
                        done = True
                    elif reward <= -1.0:  # Crash
                        print("[FAIL] Episode terminated by collision.")
                        done = True

                    # Save model periodically
                if episode % 10 == 0:
                    torch.save(policy_net.state_dict(), f"policy_net_episode_{episode}.pth")
                    print(f"[INFO] Model saved at episode {episode}")

                print(f"[INFO] Episode {episode + 1} completed. Total steps: {steps_done}")


            print("[INFO] Episode terminated by reaching maximum transition.")
            print(
                f"Ep {episode}: Reward={episode_reward:.1f} (Avg={np.mean(episode_rewards[-10:]):.1f}, Eps={epsilon:.3f}")
            print(f"[INFO] Dataset saved to {filename}  (transitions: {step_idx})")
        # Store episode metrics
        episode_rewards.append(episode_reward)

        # Calculate rolling average
        if len(episode_rewards) >= window_size:
            rolling_avg.append(np.mean(episode_rewards[-window_size:]))

        if len(episode_losses) >= LOSS_PLOT_WINDOW:
            loss_rolling_avg.append(np.mean(episode_losses[-LOSS_PLOT_WINDOW:]))

        # Save learning curve periodically
        if episode % SAVE_PLOT_EVERY == 0 or episode == num_episodes - 1:
            plot_learning_curves(episode_rewards, rolling_avg, episode_losses, loss_rolling_avg, episode)

if __name__ == "__main__":
    main()

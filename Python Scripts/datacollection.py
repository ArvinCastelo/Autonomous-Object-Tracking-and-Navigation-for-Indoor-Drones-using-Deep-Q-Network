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

from discrete_controller import DiscreteController
from webots_drone_api import WebotsDroneAPI

from pid_controller import pid_velocity_fixed_height_controller

from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# ---------------------------------------------------------------------------
# Hyper‑parameters (tweak as you like)
# ---------------------------------------------------------------------------
TIMESTEP           = 32          # [ms] – must match World > basicTimeStep
MAX_EPISODE_STEPS  = 500      # safety stop (≈ 13 min at 32 ms)
ACTION_REPEAT      = 5           # how many control cycles per action
SAFE_Z             = 0.8         # [m] – target hover height after take‑off
DATA_DIR           = "dataset7_1"   # where rollouts are stored
WALL_THRESHOLD     = 0.3        # meters
AVOID_DURATION     = 10          # how many steps to avoid after trigger
# ---------------------------------------------------------------------------

# Utility helpers
# ---------------------------------------------------------------------------

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
# Main routine
# ---------------------------------------------------------------------------

# Main method
def main():
    num_episodes = 1000

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
        # ---------- log file ---------- #
        ensure_dir(DATA_DIR)
        filename = os.path.join(DATA_DIR, f"dataset_{datetime.now():%Y%m%d_%H%M%S}_{episode}.csv")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header())

            drone_node.getField("translation").setSFVec3f(initial_translation)

            # Reset simulation state (critical!)
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

            avoidance_counter = 0

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

                # Pick a random safe action
                action_id = random.choice(available_ids)
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


                    next_vz = (next_state["z"] - state["z"]) / (TIMESTEP / 1000)
                    prev_z = state["z"]
                    vz = (state["z"] - prev_z) / (TIMESTEP / 1000)

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


                    if reward == 5:
                        print("[INFO] Episode terminated by finding the target.")
                        done = True


                    if reward <= -1:
                        print("[INFO] Episode terminated by collision.")
                        done = True



            print("[INFO] Episode terminated by reaching maximum transition.")
            print(f"[INFO] Dataset saved to {filename}  (transitions: {step_idx})")


if __name__ == "__main__":
    main()

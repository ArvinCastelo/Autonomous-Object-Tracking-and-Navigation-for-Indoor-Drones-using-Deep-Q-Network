# train.py
import pandas as pd
import torch
from dqn_agent import DQNAgent
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
STATE_COLS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw',
              'front_dist', 'left_dist', 'right_dist', 'back_dist',
              'vx', 'vy', 'vz']
NEXT_STATE_COLS = ['next_' + col for col in STATE_COLS]

ACTION_COL = 'action_id'
REWARD_COL = 'reward'
DONE_COL = 'done'

BATCHES = 5000
TARGET_UPDATE = 50 # larger TARGET_UPDATE = slower updte, more stable

# Load dataset
path = "../full_data/dataset_20250420_191946_merged_epsilon.csv"  # Replace with your actual dataset path
df = pd.read_csv(path)

# Convert transitions
states = df[STATE_COLS].values
actions = df[ACTION_COL].values
rewards = df[REWARD_COL].values
next_states = df[NEXT_STATE_COLS].values
dones = df[DONE_COL].astype(bool).values

# Train/val split (optional)
train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=0.1, random_state=42)

# Init agent
state_dim = states.shape[1]
action_dim = len(np.unique(actions))
agent = DQNAgent(state_dim, action_dim, device="cpu")

# Training loop
losses = []
avg_rewards = []

for step in range(BATCHES):
    idx = np.random.choice(train_idx)
    agent.push_transition(states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx])

    loss = agent.train_step()
    if loss is not None:
        losses.append(loss)

    # Update target network
    if step % TARGET_UPDATE == 0:
        agent.update_target()

    # Logging
    if step % 100 == 0:
        # larger window_size, smoother reward curve, stablizing loss curve
        recent = df.iloc[train_idx[max(0, step - 2000):step]]
        avg_rewards.append(recent[REWARD_COL].mean())
        if loss is not None:
            print(f"Step {step} | Loss: {loss:.4f} | Avg Reward: {avg_rewards[-1]:.4f}")
        else:
            print(f"Step {step} | Loss: ---- | Avg Reward: {avg_rewards[-1]:.4f}")

# Save model
agent.save("dqn_model.pth")



fig = plt.figure(figsize=(12, 10))

# Plot rewards
plt.subplot(2, 1, 1)
# plt.plot(avg_rewards)
plt.plot(avg_rewards, 'b-', alpha=0.3, label='Average Reward')
plt.title("Average Reward")
# plt.xlabel("Steps")
plt.ylabel("Avg Reward")
plt.grid(True)
plt.legend()
# plt.axhline(y=10, color='g', linestyle='--', alpha=0.5, label='Target Reward')
# plt.savefig("plots/avg_reward_curve.png")

# Plot loss
plt.subplot(2, 1, 2)
# plt.plot(losses)
plt.plot(losses, 'm-', alpha=0.3, label='Batch Loss')
plt.title("DQN Training Loss")
plt.xlabel("Steps")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.legend()

fig.suptitle('Offline Learning Evaluation - Epsilon-Greedy Actions', fontsize=14)
plt.tight_layout()
plt.savefig("plots/offline_epsilon_curve.png")




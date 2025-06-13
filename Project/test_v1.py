import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm.notebook import tqdm

# 设置随机种子
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# ✅ 创建环境
env = gym.make('Acrobot-v1')
env.reset(seed=SEED)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"状态维度: {STATE_DIM}, 动作空间: {ACTION_DIM}")

# ✅ 定义 Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

q_net = QNetwork(STATE_DIM, ACTION_DIM).to(device)
target_net = QNetwork(STATE_DIM, ACTION_DIM).to(device)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

# ✅ 定义 ReplayBuffer 类
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)

# ✅ 参数设置
BATCH_SIZE = 64
GAMMA = 0.99
REPLAY_CAPACITY = 10000
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

# epsilon-greedy 策略
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return q_net(state).argmax(dim=1).item()

# ✅ 训练函数
def train():
    if len(replay_buffer) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    q_values = q_net(states).gather(1, actions)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ✅ 动态绘图函数
plt.ion()
fig, ax = plt.subplots()
def update_plot(rewards):
    clear_output(wait=True)
    ax.clear()
    ax.plot(rewards, label='Episode Reward')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('DQN Training on Acrobot-v1')
    ax.legend()
    plt.pause(0.001)

# ✅ 主训练循环
EPISODES = 300
TARGET_UPDATE = 20
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

epsilon = EPSILON_START
episode_rewards = []

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        shaped_reward = reward
        if done and terminated:
            shaped_reward += 8  # ✅ 成功到达目标后给予奖励加成

        replay_buffer.push(state, action, shaped_reward, next_state, done)
        state = next_state
        total_reward += shaped_reward
        train()

    episode_rewards.append(total_reward)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(q_net.state_dict())
        torch.cuda.empty_cache()

    update_plot(episode_rewards)
    print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

# ✅ 保存最终的训练图像
plt.ioff()
fig.savefig("dqn_acrobot_training_rewards.png")
print("训练曲线已保存为 dqn_acrobot_training_rewards.png")

# ✅ 保存模型
torch.save(q_net.state_dict(), "dqn_acrobot_model.pth")
print("✅ 模型已保存至 dqn_acrobot_model.pth")

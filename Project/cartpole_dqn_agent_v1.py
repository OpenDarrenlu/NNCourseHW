import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random

# 检查是否可以使用GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构建Q网络结构
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# 经验回放缓冲区
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# CartPole DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = device
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        self.buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.tensor([action]).to(device)
        reward = torch.tensor([reward]).to(device)
        done = torch.tensor([done]).to(device)
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        self.buffer.push(state, action, next_state, reward, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.cat(batch.done)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (1 - done_batch.float()) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values.squeeze(), expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 软更新 target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(0.005 * policy_param.data + 0.995 * target_param.data)

# 训练函数
def train():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    episodes = 500
    max_steps = 800
    solved_reward = 500

    scores = []
    moving_avg = []
    convergence_episode = None

    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.store_transition(state, action, next_state, reward, done)
            agent.update()
            state = next_state
            total_reward += reward

            if done or truncated:
                break

        scores.append(total_reward)
        avg_score = np.mean(scores[-50:])
        moving_avg.append(avg_score)
        print(f"Episode: {episode+1}, Total reward: {total_reward}, Avg reward: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")

        if convergence_episode is None and avg_score >= solved_reward:
            convergence_episode = episode + 1

        if avg_score >= solved_reward:
            print(f"Solved in {episode} episodes!")
            torch.save(agent.policy_net.state_dict(), f"dqn_cartpole_{episode}.pth")
            break

    if convergence_episode is None:
        torch.save(agent.policy_net.state_dict(), "transformer_dqn_cartpole_no_convergence.pth")
    env.close()
    
    eval_episodes = 50
    final_avg_reward = np.mean(scores[-eval_episodes:]) if len(scores) >= eval_episodes else np.mean(scores)
    stability_std = np.std(scores[-eval_episodes:]) if len(scores) >= eval_episodes else np.std(scores)

    print("\n=== 训练完成 ===")
    print(f"平均奖励值（最后{eval_episodes}轮）: {final_avg_reward:.2f}")
    print(f"收敛速度（达到平均奖励{solved_reward}所需轮数）: {convergence_episode if convergence_episode else '未收敛'}")
    print(f"稳定性（最后100轮标准差）: {stability_std:.2f}")

    import matplotlib.pyplot as plt
    plt.plot(scores, label='Total Reward')
    plt.plot(moving_avg, label='Moving Avg (10)')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Curve (Transformer-DQN)")
    plt.legend()
    plt.grid(True)
    plt.savefig("dqn_v1.png")
    plt.show()

if __name__ == "__main__":
    train()

'''
=== 训练完成 ===
平均奖励值（最后50轮）: 267.36
收敛速度（达到平均奖励500所需轮数）: 271
稳定性（最后100轮标准差）: 121.65
'''
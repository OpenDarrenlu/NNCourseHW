# ✅ 强化学习：使用序列输入的 Transformer-DQN 完整实现（以 CartPole 为例）
# 该实现中：
# - 使用了最近 `seq_len` 步的状态作为 Transformer 输入序列
# - 使用 Transformer Decoder-only 架构作为 Q 网络
# - 保持 DQN 框架不变，仅改动模型结构与数据输入方式

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer Q 网络（Decoder-only）
class TransformerDQN(nn.Module):
    def __init__(self, state_dim, action_dim, seq_len, d_model=128, nhead=4, num_layers=1):
        super(TransformerDQN, self).__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.embedding = nn.Linear(state_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model))  # learnable positional embedding

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.query_token = nn.Parameter(torch.randn(1, d_model))  # learnable query
        self.output = nn.Linear(d_model, action_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, state_dim]
        batch_size = x.size(0)
        x = self.embedding(x) + self.pos_embedding.unsqueeze(0)  # [B, seq_len, d_model]
        memory = x  # [B, seq_len, d_model]

        query = self.query_token.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, d_model]
        output = self.transformer_decoder(query, memory)  # [B, 1, d_model]
        return self.output(output.squeeze(1))  # [B, action_dim]

# 经验回放
Transition = namedtuple('Transition', ('state_seq', 'action', 'next_state_seq', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# DQN Agent with Transformer Q network
class DQNAgent:
    def __init__(self, state_dim, action_dim, seq_len):
        self.device = device
        self.seq_len = seq_len
        self.policy_net = TransformerDQN(state_dim, action_dim, seq_len).to(device)
        self.target_net = TransformerDQN(state_dim, action_dim, seq_len).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        self.buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.990
        self.epsilon_min = 0.05

    def select_action(self, state_seq):
        state_seq = torch.FloatTensor(state_seq).unsqueeze(0).to(device)  # [1, seq_len, state_dim]
        if random.random() < self.epsilon:
            return random.randint(0, self.policy_net.output.out_features - 1)
        else:
            with torch.no_grad():
                return self.policy_net(state_seq).argmax().item()

    def store_transition(self, state_seq, action, next_state_seq, reward, done):
        action = torch.tensor([action]).to(device)
        reward = torch.tensor([reward]).to(device)
        done = torch.tensor([done]).to(device)
        state_seq = torch.FloatTensor(state_seq).to(device)
        next_state_seq = torch.FloatTensor(next_state_seq).to(device)
        self.buffer.push(state_seq, action, next_state_seq, reward, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state_seq)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.stack(batch.next_state_seq)
        done_batch = torch.cat(batch.done)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (1 - done_batch.float()) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values.squeeze(), expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(0.005 * policy_param.data + 0.995 * target_param.data)

# 训练函数
def train():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    seq_len = 20  # 用最近 4 个状态作为序列

    agent = DQNAgent(state_dim, action_dim, seq_len)
    episodes = 300
    max_steps = 800
    solved_reward = 500

    scores = []
    moving_avg = []
    convergence_episode = None

    for episode in range(episodes):
        state = env.reset()[0]
        state_buffer = deque([np.zeros(state_dim) for _ in range(seq_len - 1)], maxlen=seq_len - 1)
        state_buffer.append(state)
        total_reward = 0

        for step in range(max_steps):
            state_seq = list(state_buffer) + [state]
            action = agent.select_action(state_seq)
            next_state, reward, done, truncated, _ = env.step(action)

            next_state_buffer = deque(state_buffer, maxlen=seq_len - 1)
            next_state_buffer.append(next_state)
            next_state_seq = list(next_state_buffer) + [next_state]

            agent.store_transition(state_seq, action, next_state_seq, reward, done)
            agent.update()

            state = next_state
            state_buffer = next_state_buffer
            total_reward += reward
            if done or truncated:
                break

        scores.append(total_reward)
        avg_score = np.mean(scores[-10:])
        moving_avg.append(avg_score)

        print(f"Episode: {episode+1}, Total reward: {total_reward}, Avg reward: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")

        if convergence_episode is None and avg_score >= solved_reward:
            convergence_episode = episode + 1

        if avg_score >= solved_reward:
            print(f"Solved in {episode+1} episodes!")
            torch.save(agent.policy_net.state_dict(), f"transformer_dqn_cartpole_{episode+1}.pth")
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

    plt.plot(scores, label='Total Reward')
    plt.plot(moving_avg, label='Moving Avg (10)')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Curve (Transformer-DQN)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Transformer.png")
    # plt.show()

if __name__ == "__main__":
    train()


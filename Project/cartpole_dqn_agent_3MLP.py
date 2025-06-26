import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
# import F
import torch.nn.functional as F

# import swanlab

# run = swanlab.init(project="DQN-CartPole")

# 检查是否可以使用GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 奖励权重参数
ANGLE_WEIGHT = 0.4       # 角度惩罚权重
POSITION_WEIGHT = 0.1    # 位置惩罚权重
VELOCITY_WEIGHT = 0.05   # 速度惩罚权重
ANGLE_THRESHOLD = 0.05   # 角度阈值（低于此值不惩罚）
# 构建Q网络结构
dim = 256
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, action_dim)
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

        self.lr = 2e-3
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()  # 更稳定的 Huber Loss

        self.total_updates = 0
        self.warmup_steps = 200  # 1000次update前线性warmup
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=20000, eta_min=1e-5)  # 可调T_max

        self.buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0


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
        tau = 0.01
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)


# 训练函数
def train():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    episodes = 500
    max_steps = 800
    solved_reward = 500
    flag_resume = False
    reward_threshold = 450
    
    scores = [0]
    moving_avg = []

    cur_state = 1
    good_before = False
    good_before_cnt = 0
    state_epsd_cnt = 0
    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            position = next_state[0]  # 小车位置
            velocity = next_state[1]  # 小车速度
            angle = next_state[2]     # 杆子角度
            ang_vel = next_state[3]    # 角速度
            
            # 1. 角度惩罚：鼓励杆子保持竖直
            # 使用非线性函数增加大角度的惩罚
            angle_penalty = ANGLE_WEIGHT * (angle**2)
            
            # 2. 位置惩罚：鼓励小车保持在中心附近
            position_penalty = POSITION_WEIGHT * (position**2)
            
            # 3. 速度惩罚：减少不必要的移动
            velocity_penalty = VELOCITY_WEIGHT * abs(velocity)
            
            # 4. 组合奖励
            modified_reward = reward - angle_penalty - position_penalty - velocity_penalty
            
            # 5. 添加角度稳定奖励（当角度接近竖直时给予小奖励）
            if abs(angle) < ANGLE_THRESHOLD:
                modified_reward += 0.05 * (1 - abs(angle)/ANGLE_THRESHOLD)
                
            agent.store_transition(state, action, next_state, modified_reward, done)
            agent.update()
            
            # # warmup: 线性增加学习率
            # agent.total_updates += 1
            # if agent.total_updates < agent.warmup_steps:
            #     warmup_lr = agent.lr * agent.total_updates / agent.warmup_steps
            #     for param_group in agent.optimizer.param_groups:
            #         param_group['lr'] = warmup_lr
            # else:
            #     agent.scheduler.step()
            
            state = next_state
            total_reward += reward

            if done or truncated:
                break
        
        ######### change lr ###########
        if total_reward < 50:
            next_state = 1
            if scores[-1] == 500:
                agent.lr = 0.5e-3
            elif good_before:
                agent.lr = 2e-3
            else:
                2e-3
        elif 50 <= total_reward < 200:
            next_state = 1
            agent.lr = 2e-3
        elif 200 <= total_reward < 400:
            next_state = 2
            agent.lr = 1e-3
        else:
            next_state = 3
            agent.lr = 1e-4
        
        if total_reward > 350:
            good_before = True
        if cur_state == next_state:
            state_epsd_cnt += 1
        else:
            state_epsd_cnt = 0
        # if good_before:
        #     if state_epsd_cnt > 40:
        #         agent.lr = 0.001
        #         print("重置")
        #         state_epsd_cnt = 10
        # else:
        #     if state_epsd_cnt > 80:
        #         agent.lr = 0.001
        #         print("重置")
        #         state_epsd_cnt = 10
        # if scores[-1] < 400 and total_reward == 500:
        #     good_before_cnt += 1
        # if good_before_cnt > 6:
        #     agent.lr = 0.001
        #     print("重置")
        #     state_epsd_cnt = 0
        #     good_before_cnt = 0
        #     good_before = False
        cur_state = next_state
        ##############################

        scores.append(total_reward)
        avg_score_10 = np.mean(scores[-10:])
        avg_score_3 = np.mean(scores[-3:])
        print(f"Episode: {episode+1}, Total reward: {total_reward}, Avg reward: {avg_score_10:.2f}, Epsilon: {agent.epsilon:.2f}")
        # swanlab.log({"Total reward": total_reward, "Avg reward": avg_score, "cur_state":cur_state, "state_epsd_cnt":state_epsd_cnt, "LR": agent.optimizer.param_groups[0]['lr']})

        if avg_score_10 >= solved_reward and avg_score_3 >= 500:
            print(f"Solved in {episode+1} episodes!")
            torch.save(agent.policy_net.state_dict(), f"dqn_cartpole_{episode+1}.pth")
            break
        if np.mean(scores[-5:]) > reward_threshold:
            flag_resume = True
            torch.save(agent.policy_net.state_dict(), "dqn_cartpole.pth")
            # if (total_reward > reward_threshold + 100):
            #     reward_threshold += 100
            #     print(f"Reward threshold updated to {reward_threshold}")
        if flag_resume and total_reward < reward_threshold-200:
            print("Resume training..., rewardThreshold:", reward_threshold)
            agent.policy_net.load_state_dict(torch.load("dqn_cartpole.pth"))
            flag_resume = False
        moving_avg.append(avg_score_10)
    env.close()
    
    eval_episodes = 50
    final_avg_reward = np.mean(scores[-eval_episodes:]) if len(scores) >= eval_episodes else np.mean(scores)
    stability_std = np.std(scores[-eval_episodes:]) if len(scores) >= eval_episodes else np.std(scores)

    print("\n=== 训练完成 ===")
    print(f"平均奖励值（最后{eval_episodes}轮）: {final_avg_reward:.2f}")
    print(f"收敛速度（达到平均奖励{solved_reward}所需轮数）: {episode}")
    print(f"稳定性（最后100轮标准差）: {stability_std:.2f}")
    from matplotlib import pyplot as plt
    plt.plot(scores, label='Total Reward')
    plt.plot(moving_avg, label='Moving Avg (10)')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Curve (3MLP)")
    plt.legend()
    plt.grid(True)
    plt.savefig("3MLP.png")
    # plt.show()
    return episode

if __name__ == "__main__":
    for i in range(1):
        finish_episode = train() + 1
        # swanlab.log({"finish_episode":finish_episode})

# resMLP
'''
=== 训练完成 ===
平均奖励值（最后50轮）: 290.86
收敛速度（达到平均奖励500所需轮数）: 66
稳定性（最后100轮标准差）: 133.04
'''

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os

import matplotlib.pyplot as plt
import csv

# ========== 配置开关 ==========
LOAD_MODEL = False  # 是否加载预训练模型
TEST_MODE = False   # 是否只进行测试（不训练）
# =============================

# 设置超参数
ENV_NAME = "CartPole-v1"
LEARNING_RATE = 0.005
GAMMA = 0.99  # 折扣因子
MAX_EPISODES = 500  # 最大训练回合数
MODEL_SAVE_PATH = "actor_critic_cartpole.pth"  # 模型保存路径
n_number = 64  # 隐藏层尺寸

# 奖励权重参数
ANGLE_WEIGHT = 0.2       # 角度惩罚权重
POSITION_WEIGHT = 0.1    # 位置惩罚权重
VELOCITY_WEIGHT = 0.01   # 速度惩罚权重
ANGLE_THRESHOLD = 0.05   # 角度阈值（低于此值不惩罚）

class ActorCritic(nn.Module):
    """Actor-Critic神经网络模型"""
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor层 - 输出动作概率
        # Actor层 - 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # Critic层 - 输出状态值
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, state):
        # 前向传播
        # 演员：获取动作概率分布
        action_probs = F.softmax(self.actor(state), dim=-1)
        
        # 评论家：获取状态值
        state_value = self.critic(state)
        
        return action_probs, state_value

class Agent:
    """Actor-Critic智能体"""
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型和优化器
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # 存储每个时间步的数据
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
    
    def select_action(self, state, deterministic=False):
        """根据状态选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 获取动作概率和状态值
        action_probs, state_value = self.model(state)
        
        # 创建分类分布
        dist = Categorical(action_probs)
        
        # 根据模式选择动作：确定性选择最大概率动作，或随机采样
        if deterministic:
            action = torch.argmax(action_probs)
        else:
            action = dist.sample()
        
        # 存储选择动作的对数概率和状态值
        self.log_probs.append(dist.log_prob(action))
        self.values.append(state_value)
        
        return action.item()
    
    def update(self):
        """更新模型参数"""
        # 计算回报
        returns = []
        R = 0
        # 从后往前计算每个时间步的回报
        for reward, mask in zip(reversed(self.rewards), reversed(self.masks)):
            R = reward + GAMMA * R * mask
            returns.insert(0, R)
        
        # 转换为张量 - 修复：确保使用float32类型
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        log_probs = torch.cat(self.log_probs).to(self.device)
        values = torch.cat(self.values).squeeze().to(self.device)
        
        # 计算优势函数
        advantages = returns - values.detach()
        
        # 计算Actor损失 (策略梯度)
        actor_loss = -(log_probs * advantages).mean()
        
        # 计算Critic损失 (值函数近似)
        critic_loss = F.mse_loss(values, returns)
        
        # 总损失
        total_loss = actor_loss + critic_loss
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # 清空数据缓存
        self.clear_memory()
        
        return total_loss.item()
    
    def clear_memory(self):
        """清空存储的数据"""
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
    
    def store_transition(self, reward, done):
        """存储奖励和终止标志"""
        # 如果回合结束，mask为0，否则为1
        mask = 0.0 if done else 1.0
        self.rewards.append(reward)
        self.masks.append(mask)
    
    def save_model(self, path):
        """保存模型参数"""
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存至: {path}")
    
    def load_model(self, path):
        """加载模型参数"""
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            # 不再在这里设置eval()模式
            print(f"成功加载模型: {path}")
        else:
            print(f"警告: 模型文件 {path} 不存在，跳过加载")
            
    def set_train_mode(self):
        """设置模型为训练模式"""
        self.model.train()
        print("模型设置为训练模式")
        
    def set_eval_mode(self):
        """设置模型为评估模式"""
        self.model.eval()
        print("模型设置为评估模式")

def test_model(env, agent, episodes=10):
    """测试训练好的模型"""
    # 确保在评估模式下
    agent.set_eval_mode()
    
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_reward = 0
        done = False
        
        while not done:
            env.render()  # 渲染环境
                
            # 使用确定性策略选择动作
            action = agent.select_action(state, deterministic=True)
            
            # 执行动作
            result = env.step(action)
            if len(result) == 4:  # 旧版gym
                next_state, reward, done, _ = result
            else:  # 新版gym (>=0.26.0)
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            
            if done:
                total_rewards.append(episode_reward)
                print(f"测试回合 {episode+1}/{episodes}, 奖励: {episode_reward}")
                break
    
    env.close()
    avg_reward = np.mean(total_rewards)
    print(f"测试完成! 平均奖励: {avg_reward:.2f}, 最大奖励: {max(total_rewards)}")
    return avg_reward

def main():
    # 创建环境
    env = gym.make(ENV_NAME)
    
    # 获取环境维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 创建智能体
    agent = Agent(state_dim, action_dim)
    
    # 如果设置了加载模型，则加载预训练模型
    if LOAD_MODEL:
        agent.load_model(MODEL_SAVE_PATH)
    
    # 如果是测试模式，直接测试并退出
    if TEST_MODE:
        print("进入测试模式...")
        test_model(env, agent, episodes=10)
        return
    
    # 确保模型在训练模式
    agent.set_train_mode()
    
    # 训练统计
    total_rewards = []
    losses = []
    best_avg_reward = -np.inf
    
    early_stop = False
    for episode in range(MAX_EPISODES):
        # 重置环境
        state = env.reset()
        # 处理不同gym版本的返回值差异
        if isinstance(state, tuple):
            state = state[0]  # 新版gym返回(state, info)
        
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            result = env.step(action)
            
            # 处理不同gym版本的返回值差异
            if len(result) == 4:  # 旧版gym
                next_state, reward, done, _ = result
            else:  # 新版gym (>=0.26.0)
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            # ====== 修改奖励函数 ======
            # 获取状态信息
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
            # ========================
            
            # 存储奖励和终止标志
            agent.store_transition(modified_reward, done)
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward  # 注意：这里使用原始奖励统计回合总奖励
            
            # 回合结束时更新模型
            if done:
                loss = agent.update()
                losses.append(loss)
                total_rewards.append(episode_reward)
                
                # 计算最近100个回合的平均奖励
                avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
                
                # 更新最佳平均奖励
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    # 保存当前最佳模型
                    #agent.save_model(MODEL_SAVE_PATH)
                
                # 打印训练进度
                print(f"Episode: {episode+1}/{MAX_EPISODES}, Reward: {episode_reward}, Avg Reward: {avg_reward:.2f}, Best Avg: {best_avg_reward:.2f}, Loss: {loss:.4f}")
                
                # 提前停止条件：连续20个回合平均奖励达到495+
                # if len(total_rewards) >= 20 and np.mean(total_rewards[-20:]) >= 495:
                #     print(f"提前停止! 连续20回合平均奖励达到495+")
                #     early_stop = True
                #     break

        if early_stop:
            break

    env.close()
    print("训练完成!")
    
    # 保存最终模型
    agent.save_model("final_" + MODEL_SAVE_PATH)
    
    # 保存奖励数据
    filename = "rewards.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(total_rewards)
    
    # 测试训练好的模型
    # print("训练后测试模型...")
    # test_model(gym.make(ENV_NAME), agent, episodes=5)


if __name__ == "__main__":
    main()
import gym
import torch
import numpy as np
from cartpole_dqn_agent_v2 import DuelingDQN  # 假设上面的 DQN 类保存在 dqn_agent.py 中

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建环境
env = gym.make('CartPole-v1', max_episode_steps=20000, render_mode='human')  # 使用 human 渲染模式

# 获取状态和动作空间维度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 初始化网络
policy_net = DuelingDQN(state_dim, action_dim).to(device)
# policy_net.load_state_dict(torch.load("dqn_cartpole_100.pth"))
# policy_net.load_state_dict(torch.load("dqn_cartpole_300.pth"))
# policy_net.load_state_dict(torch.load("dqn_cartpole_500.pth"))
policy_net.load_state_dict(torch.load("dqn_cartpole_v2_169_500.pth"))
policy_net.eval()  # 设置为评估模式

# 测试函数
def test_model(episodes=5):
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # 转换为 tensor
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action = policy_net(state_tensor).argmax().item()

            next_state, reward, done, truncated, _ = env.step(action)

            state = next_state
            total_reward += reward

            # 可以适当延时让动画更清晰（可选）
            # import time; time.sleep(0.02)

        print(f"Episode {episode+1}: Total reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    test_model()
# import gym
# import torch
# import numpy as np
# from cartpole_dqn_agent import DQN  # 替换为你自己的网络定义文件

# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 创建环境（可调整最大步数）
# env = gym.make('CartPole-v1', render_mode='human', max_episode_steps=1000)

# # 获取状态和动作空间维度
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n

# # 初始化并加载模型
# policy_net = DQN(state_dim, action_dim).to(device)
# # policy_net.load_state_dict(torch.load("dqn_cartpole_300.pth"))
# policy_net.load_state_dict(torch.load("dqn_cartpole.pth"))
# policy_net.eval()

# # 测试函数，包含扰动
# def test_with_perturbation(episodes=5, perturb_interval=50, perturb_magnitude=0.5):
#     for episode in range(episodes):
#         state, _ = env.reset()
#         total_reward = 0
#         done = False
#         truncated = False
#         step_count = 0

#         while not (done or truncated):
#             # 添加扰动：每隔一定步数，给小车位移一个扰动
#             if step_count % perturb_interval == 0 and step_count > 0:
#                 print(f"[扰动] 第 {step_count} 步，对小车施加扰动: ±{perturb_magnitude}")
#                 state[0] += perturb_magnitude * np.random.choice([-1, 1])  # 随机向左或右扰动

#             # 选择动作
#             state_tensor = torch.FloatTensor(state).to(device)
#             with torch.no_grad():
#                 action = policy_net(state_tensor).argmax().item()

#             # 执行动作
#             next_state, reward, done, truncated, _ = env.step(action)

#             state = next_state
#             total_reward += reward
#             step_count += 1

#         print(f"Episode {episode+1}: Total reward = {total_reward}, Steps = {step_count}")

#     env.close()

# if __name__ == "__main__":
#     test_with_perturbation(episodes=5, perturb_interval=50, perturb_magnitude=0.5)
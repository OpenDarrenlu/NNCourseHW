import gym
import torch
import numpy as np
# from cartpole_dqn_agent_v2 import DuelingDQN 
from cartpole_dqn_agent_v1 import DQN

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建环境
env = gym.make('CartPole-v1', max_episode_steps=500, render_mode='human')  # 使用 human 渲染模式
# env = gym.make('CartPole-v1', max_episode_steps=500)

# 获取状态和动作空间维度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 初始化网络
policy_net = DQN(state_dim, action_dim).to(device)
policy_net.load_state_dict(torch.load("dqn_cartpole.pth"))
policy_net.eval()

# 加扰动测试函数
def test_model_with_perturbation(episodes=5, perturb_interval=50, perturb_magnitude=0.5):
    scores = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        step_count = 0

        while not (done or truncated):
            # 每隔 perturb_interval 步扰动一次 state[0]（位置）
            if step_count > 0 and step_count % perturb_interval == 0:
                delta = perturb_magnitude * np.random.choice([-1, 1])
                tmp = np.random.choice([0,1,2,3])
                print(state)
                state[tmp] += delta
                print(f"[扰动] Step {step_count}: 对小车位置 state[{tmp}] 加扰动 {delta:.2f}")

            # 选择动作
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action = policy_net(state_tensor).argmax().item()

            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)

            state = next_state
            total_reward += reward
            step_count += 1

        scores.append(total_reward)
        print(f"Episode {episode+1}: Total reward: {total_reward}")

    env.close()
    # 平均奖励值
    print(f"\nAverage score over {episodes} episodes: {np.mean(scores):.2f}")
    # 稳定性
    print(f"Standard deviation of scores: {np.std(scores):.2f}")

if __name__ == "__main__":
    # 打印各个维度的最大扰动
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(env.observation_space.high)
    print(env.observation_space.low)
    test_model_with_perturbation(
        episodes=100,
        perturb_interval=50,     # 每 50 步干扰一次
        perturb_magnitude=5    # 干扰幅度
    )

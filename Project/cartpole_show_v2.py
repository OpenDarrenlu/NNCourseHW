import gym
import torch
import numpy as np
from collections import deque
from cartpole_dqn_agent_v2 import TransformerDQN  # 你应把训练代码中的模型部分单独存成 transformer_dqn_model.py

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 环境设置
env = gym.make('CartPole-v1', max_episode_steps=1000, render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
seq_len = 20  # 与训练时保持一致

# 初始化模型
model = TransformerDQN(state_dim, action_dim, seq_len).to(device)
model.load_state_dict(torch.load("transformer_dqn_cartpole_no_convergence.pth"))  # 替换为你的模型文件名
model.eval()

# 标准测试
def test_model(episodes=5, render=False):
    for episode in range(episodes):
        state, _ = env.reset()
        state_buffer = deque([np.zeros(state_dim) for _ in range(seq_len - 1)], maxlen=seq_len - 1)
        state_buffer.append(state)

        total_reward = 0
        done, truncated = False, False

        while not (done or truncated):
            state_seq = list(state_buffer) + [state]
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(device)  # [1, seq_len, state_dim]

            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_state, reward, done, truncated, _ = env.step(action)
            if render:
                env.render()

            # 更新序列
            state = next_state
            state_buffer.append(state)
            total_reward += reward

        print(f"[Test] Episode {episode+1}: Total reward = {total_reward}")

    env.close()

# 扰动测试
def test_with_perturbation(episodes=5, perturb_interval=50, perturb_magnitude=0.5):
    for episode in range(episodes):
        state, _ = env.reset()
        state_buffer = deque([np.zeros(state_dim) for _ in range(seq_len - 1)], maxlen=seq_len - 1)
        state_buffer.append(state)

        total_reward = 0
        step_count = 0
        done, truncated = False, False

        while not (done or truncated):
            if step_count > 0 and step_count % perturb_interval == 0:
                perturb = perturb_magnitude * np.random.choice([-1, 1])
                print(f"[扰动] 第{step_count}步，对位置添加扰动: {perturb}")
                state[0] += perturb

            state_seq = list(state_buffer) + [state]
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(device)

            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_state, reward, done, truncated, _ = env.step(action)

            state = next_state
            state_buffer.append(state)
            total_reward += reward
            step_count += 1

        print(f"[扰动] Episode {episode+1}: Total reward = {total_reward}, Steps = {step_count}")

    env.close()

if __name__ == "__main__":
    # 普通测试
    test_model(episodes=5, render=False)

    # 扰动测试（可选）
    # test_with_perturbation(episodes=5, perturb_interval=50, perturb_magnitude=0.5)

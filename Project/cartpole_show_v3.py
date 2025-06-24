import gym
import torch
import numpy as np
import keyboard  # 导入键盘监听库
import time
from cartpole_dqn_agent_v3 import DQN

# --- 主要修改部分 ---

def test_model_with_disturbance(episodes=5, model_path="dqn_cartpole.pth", disturbance_magnitude=1.5):
    """
    测试DQN模型，并允许通过键盘施加干扰。

    Args:
        episodes (int): 测试的总回合数。
        model_path (str): 加载的预训练模型文件路径。
        disturbance_magnitude (float): 每次施加干扰的强度（即瞬间改变的速度大小）。
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建环境，使用 human 模式进行可视化
    env = gym.make('CartPole-v1', render_mode='human')

    # 获取状态和动作空间维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 初始化网络并加载预训练权重
    policy_net = DQN(state_dim, action_dim).to(device)
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"错误：找不到模型文件 '{model_path}'。请确保文件路径正确。")
        env.close()
        return
        
    policy_net.eval()  # 设置为评估模式

    print("\n--- 模型抗干扰能力测试 ---")
    print(f"模型: {model_path}")
    print(f"干扰强度: {disturbance_magnitude}")
    print("测试开始后，在窗口激活状态下：")
    print("  - 按下 [左方向键] 给小车一个向左的推力。")
    print("  - 按下 [右方向键] 给小车一个向右的推力。")
    print("---------------------------------")
    time.sleep(3) # 准备时间

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            env.render() # 确保每一帧都渲染

            # --- 核心：检测键盘输入并施加干扰 ---
            # env.unwrapped.state可以直接访问和修改内部状态: [车位置, 车速度, 杆角度, 杆尖速度]
            current_state = env.unwrapped.state
            
            try:
                if keyboard.is_pressed('left arrow'):
                    # 施加向左的力 (减少小车速度)
                    new_velocity = current_state[1] - disturbance_magnitude
                    env.unwrapped.state = (current_state[0], new_velocity, current_state[2], current_state[3])
                    print("施加向左干扰! <---")
                    # 短暂延时防止连续触发
                    time.sleep(0.2) 
                    
                elif keyboard.is_pressed('right arrow'):
                    # 施加向右的力 (增加小车速度)
                    new_velocity = current_state[1] + disturbance_magnitude
                    env.unwrapped.state = (current_state[0], new_velocity, current_state[2], current_state[3])
                    print("---> 施加向右干扰!")
                    time.sleep(0.2)
            except Exception as e:
                # 在某些环境下 keyboard 库可能会报错，这里做个保护
                # print(f"无法监听键盘，请尝试使用sudo权限运行: {e}")
                # 一旦出错，就不再尝试监听，避免刷屏
                disturbance_magnitude = 0 # 禁用干扰

            # DQN Agent决策
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy_net(state_tensor).argmax().item()

            # 与环境交互
            next_state, reward, done, truncated, _ = env.step(action)
            
            state = next_state
            total_reward += reward
            
            # 稍微减慢速度，方便观察和操作
            time.sleep(0.01)

        print(f"回合 {episode+1}: 总奖励 = {total_reward}")

    env.close()
    print("\n测试完成。")

if __name__ == "__main__":
    # 你可以修改这里的参数来测试不同的模型或干扰强度
    import sys

    if len(sys.argv) < 2:
        print("请提供一个参数！")
        sys.exit(1)

    arg1 = sys.argv[1]

    test_model_with_disturbance(
        episodes=5, 
        model_path=f"dqn_cartpole_{arg1}.pth", 
        disturbance_magnitude=0.5 
    )

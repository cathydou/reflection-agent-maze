import numpy as np
import matplotlib.pyplot as plt
from dynamic_maze_env import DynamicMazeEnv
from reflection_agent import ReflectionAgent
from baseline_confidence_agent import BaselineConfidenceAgent

def run_experiment(num_episodes=100, max_steps=200):
    # 环境参数
    env_params = {
        'size': 10,
        'obstacle_ratio': 0.25,
        'change_frequency': 18
    }
    
    # 创建环境和智能体
    env = DynamicMazeEnv(**env_params)
    reflection_agent = ReflectionAgent(env.action_space)
    
    # 记录性能指标
    history = {
        'steps': [],
        'success': [],
        'shortest_paths': [],
        'efficiency': [],
        'goal_distances': []
    }
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        reflection_agent.set_goal_position(env.goal_pos)
        episode_steps = 0
        
        while episode_steps < max_steps:
            # 获取最短路径
            shortest_path = env.get_shortest_path_length(
                tuple(state), tuple(env.goal_pos)
            )
            
            # 选择动作
            action = reflection_agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # 学习
            reflection_agent.learn(
                state, action, reward, next_state, done,
                episode_steps, shortest_path
            )
            
            state = next_state
            episode_steps += 1
            
            if done:
                break
        
        # 记录本轮结果
        history['steps'].append(episode_steps)
        history['success'].append(int(done))
        history['shortest_paths'].append(shortest_path)
        history['efficiency'].append(
            shortest_path / episode_steps if episode_steps > 0 else 0
        )
        history['goal_distances'].append(
            np.linalg.norm(env.goal_pos - np.array(state))
        )
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            success_rate = sum(history['success'][-10:]) / 10
            avg_steps = np.mean(history['steps'][-10:])
            print(f"\nEpisode {episode + 1}")
            print(f"Recent Success Rate: {success_rate:.2f}")
            print(f"Average Steps: {avg_steps:.2f}")
    
    return history

def plot_results(history):
    plt.figure(figsize=(15, 10))
    
    # 成功率（移动平均）
    plt.subplot(2, 2, 1)
    success_ma = np.convolve(history['success'], 
                            np.ones(10)/10, 
                            mode='valid')
    plt.plot(success_ma)
    plt.title('Success Rate (10-episode moving average)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    
    # 步数
    plt.subplot(2, 2, 2)
    plt.plot(history['steps'])
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # 路径效率
    plt.subplot(2, 2, 3)
    plt.plot(history['efficiency'])
    plt.title('Path Efficiency (Shortest/Actual)')
    plt.xlabel('Episode')
    plt.ylabel('Efficiency')
    
    # 目标距离
    plt.subplot(2, 2, 4)
    plt.plot(history['goal_distances'])
    plt.title('Final Distance to Goal')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png')
    plt.close()

def analyze_learning_progress(history):
    # 分析学习进展
    episodes = len(history['success'])
    first_quarter = np.mean(history['success'][:episodes//4])
    last_quarter = np.mean(history['success'][-episodes//4:])
    
    print("\nLearning Progress Analysis:")
    print(f"First Quarter Success Rate: {first_quarter:.2f}")
    print(f"Last Quarter Success Rate: {last_quarter:.2f}")
    print(f"Improvement: {(last_quarter - first_quarter) * 100:.1f}%")
    
    print("\nPath Efficiency:")
    print(f"Average: {np.mean(history['efficiency']):.2f}")
    print(f"Best: {np.max(history['efficiency']):.2f}")

if __name__ == "__main__":
    # 运行实验
    print("Starting experiment...")
    history = run_experiment(num_episodes=100)
    
    # 绘制结果
    plot_results(history)
    
    # 分析学习进展
    analyze_learning_progress(history) 
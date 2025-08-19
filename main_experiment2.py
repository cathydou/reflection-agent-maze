import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import pygame

# 确保当前目录在Python路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dynamic_maze_env import DynamicMazeEnv
from baseline_confidence_agent import BaselineConfidenceAgent
from reflection_agent import ReflectionAgent

class ExperimentAnalyzer:
    """实验数据分析器"""
    def __init__(self):
        self.episode_data = defaultdict(list)

    def record_episode_data(self, agent_type, episode, data):
        """记录每个episode的数据"""
        self.episode_data[agent_type].append({
            'episode': episode,
            'rewards': data['reward'],
            'steps': data['steps'],
            'success': data['success'],  
            'path_efficiency': data['path_efficiency'],
            'stability': data['stability']
        })

def run_experiment(env, agent, num_episodes, analyzer, agent_type, threshold_params=None):
    """运行实验并记录性能指标"""
    episode_rewards = []
    episode_steps = []
    success_rates = []
    episode_metrics = defaultdict(list)

    # 如果提供了阈值参数，设置到反思智能体
    if threshold_params and hasattr(agent, 'set_thresholds'):
        agent.set_thresholds(*threshold_params)

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        trajectory = []
        
        while not done and steps < env.max_steps:
            steps += 1
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            trajectory.append((state, action, reward, next_state, done))
            shortest_path = env.get_optimal_path_length()
            path_efficiency = shortest_path / max(steps, 1) if shortest_path != float('inf') else 0
            
            agent.learn(state, action, reward, next_state, done, steps, shortest_path)
            
            state = next_state
            episode_reward += reward

        # 记录episode指标
        success = done and np.array_equal(state, env.goal_pos)
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        success_rates.append(1 if success else 0)
        
        # 计算稳定性
        stability = 0
        if len(trajectory) > 1:
            reward_changes = [abs(t[2] - p[2]) for t, p in zip(trajectory[1:], trajectory[:-1])]
            stability = 1.0 / (1.0 + np.mean(reward_changes))
        
        # 记录额外指标
        episode_metrics['path_efficiency'].append(path_efficiency)
        episode_metrics['reward_stability'].append(stability)
        episode_metrics['environment_changes'].append(env.episode_data['environment_updates'])
        episode_metrics['goal_changes'].append(env.episode_data['goal_changes'])

        # 记录到分析器
        analyzer.record_episode_data(agent_type, episode, {
            'reward': episode_reward,
            'steps': steps,
            'success': success,
            'path_efficiency': path_efficiency,
            'stability': stability
        })

    return {
        'rewards': episode_rewards,
        'steps': episode_steps,
        'success_rates': success_rates,
        'metrics': episode_metrics
    }

def calculate_final_metrics(results):
    """计算最终指标统计"""
    metrics = {
        'success_rate': np.mean([np.mean(r['success_rates']) for r in results]),
        'avg_steps': np.mean([np.mean(r['steps']) for r in results]),
        'avg_reward': np.mean([np.mean(r['rewards']) for r in results]),
        'path_efficiency': np.mean([np.mean(r['metrics']['path_efficiency']) for r in results]),
        'stability': np.mean([np.mean(r['metrics']['reward_stability']) for r in results])
    }
    return metrics

def print_metrics(metrics):
    """打印指标"""
    print(f"Success Rate: {metrics['success_rate']:.3f}")
    print(f"Average Steps: {metrics['avg_steps']:.2f}")
    print(f"Average Reward: {metrics['avg_reward']:.2f}")
    print(f"Path Efficiency: {metrics['path_efficiency']:.3f}")
    print(f"Reward Stability: {metrics['stability']:.3f}")

def plot_experiment_results(all_results, results_dir, timestamp):
    """绘制实验结果图表"""
    # 使用默认样式而不是seaborn
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Dynamic Maze Environment: Baseline vs Reflection Agent Performance')

    # 设置子图之间的间距
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.25, wspace=0.35)

    # 提取基线数据
    baseline_rewards = np.mean([r['rewards'] for r in all_results['baseline']], axis=0)
    baseline_steps = np.mean([r['steps'] for r in all_results['baseline']], axis=0)
    baseline_success = np.mean([r['success_rates'] for r in all_results['baseline']], axis=0)
    baseline_efficiency = np.mean([r['metrics']['path_efficiency'] for r in all_results['baseline']], axis=0)
    baseline_changes = np.mean([r['metrics']['environment_changes'] for r in all_results['baseline']], axis=0)
    baseline_stability = np.mean([r['metrics']['reward_stability'] for r in all_results['baseline']], axis=0)

    # 计算标准差用于置信区间
    baseline_rewards_std = np.std([r['rewards'] for r in all_results['baseline']], axis=0)
    baseline_steps_std = np.std([r['steps'] for r in all_results['baseline']], axis=0)
    baseline_success_std = np.std([r['success_rates'] for r in all_results['baseline']], axis=0)
    baseline_efficiency_std = np.std([r['metrics']['path_efficiency'] for r in all_results['baseline']], axis=0)
    baseline_changes_std = np.std([r['metrics']['environment_changes'] for r in all_results['baseline']], axis=0)
    baseline_stability_std = np.std([r['metrics']['reward_stability'] for r in all_results['baseline']], axis=0)

    # 对于第一个反思智能体配置
    first_reflection_config = list(all_results['reflection'].keys())[0]
    reflection_results = all_results['reflection'][first_reflection_config]
    
    reflection_rewards = np.mean([r['rewards'] for r in reflection_results], axis=0)
    reflection_steps = np.mean([r['steps'] for r in reflection_results], axis=0)
    reflection_success = np.mean([r['success_rates'] for r in reflection_results], axis=0)
    reflection_efficiency = np.mean([r['metrics']['path_efficiency'] for r in reflection_results], axis=0)
    reflection_changes = np.mean([r['metrics']['environment_changes'] for r in reflection_results], axis=0)
    reflection_stability = np.mean([r['metrics']['reward_stability'] for r in reflection_results], axis=0)

    reflection_rewards_std = np.std([r['rewards'] for r in reflection_results], axis=0)
    reflection_steps_std = np.std([r['steps'] for r in reflection_results], axis=0)
    reflection_success_std = np.std([r['success_rates'] for r in reflection_results], axis=0)
    reflection_efficiency_std = np.std([r['metrics']['path_efficiency'] for r in reflection_results], axis=0)
    reflection_changes_std = np.std([r['metrics']['environment_changes'] for r in reflection_results], axis=0)
    reflection_stability_std = np.std([r['metrics']['reward_stability'] for r in reflection_results], axis=0)

    episodes = range(len(baseline_rewards))

    def plot_with_confidence(ax, x, y1, y2, std1, std2, title, ylabel):
        ax.plot(x, y1, 'b-', label='Baseline', linewidth=1.5)
        ax.fill_between(x, y1 - std1, y1 + std1, color='blue', alpha=0.1)
        ax.plot(x, y2, 'r-', label='Reflection', linewidth=1.5)
        ax.fill_between(x, y2 - std2, y2 + std2, color='red', alpha=0.1)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    # 绘制累积奖励
    cumulative_baseline_rewards = np.cumsum(baseline_rewards)
    cumulative_reflection_rewards = np.cumsum(reflection_rewards)
    plot_with_confidence(axes[0, 0], episodes, cumulative_baseline_rewards, cumulative_reflection_rewards,
                        baseline_rewards_std, reflection_rewards_std, 'Cumulative Rewards', 'Reward')

    # 成功率（移动平均）
    window = 10
    baseline_success_ma = np.convolve(baseline_success, np.ones(window)/window, mode='valid')
    reflection_success_ma = np.convolve(reflection_success, np.ones(window)/window, mode='valid')
    baseline_success_std_ma = np.convolve(baseline_success_std, np.ones(window)/window, mode='valid')
    reflection_success_std_ma = np.convolve(reflection_success_std, np.ones(window)/window, mode='valid')
    plot_with_confidence(axes[0, 1], episodes[window-1:], baseline_success_ma, reflection_success_ma,
                        baseline_success_std_ma, reflection_success_std_ma, 
                        'Success Rate (Moving Average)', 'Success Rate')

    # 绘制平均步数
    plot_with_confidence(axes[0, 2], episodes, baseline_steps, reflection_steps,
                        baseline_steps_std, reflection_steps_std, 'Average Steps per Episode', 'Steps')

    # 绘制路径效率
    plot_with_confidence(axes[1, 0], episodes, baseline_efficiency, reflection_efficiency,
                        baseline_efficiency_std, reflection_efficiency_std, 'Path Efficiency', 'Efficiency')

    # 绘制环境变化
    plot_with_confidence(axes[1, 1], episodes, baseline_changes, reflection_changes,
                        baseline_changes_std, reflection_changes_std, 'Environment Changes', 'Number of Changes')

    # 绘制奖励稳定性
    plot_with_confidence(axes[1, 2], episodes, baseline_stability, reflection_stability,
                        baseline_stability_std, reflection_stability_std, 'Reward Stability', 'Stability')

    # 保存图表
    plot_path = os.path.join(results_dir, f'performance_plots_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def start_visualization(size):
    """初始化可视化"""
    from maze_visualization import MazeVisualization
    viz = MazeVisualization(width=size * 50, height=size * 50 + 200)  # 额外空间用于指标面板
    return viz

def main():
    """主程序"""
    # 环境参数
    env_params = {
        'size': 10,
        'obstacle_ratio': 0.25,
        'change_frequency': 18
    }
    max_steps = 200
    
    # 实验参数
    num_episodes = 100
    base_seed = 42
    
    # 初始化可视化
    viz = start_visualization(env_params['size'])
    
    # 创建环境实例
    env = DynamicMazeEnv(**env_params, seed=base_seed)
    env.max_steps = max_steps
    
    # 创建智能体
    baseline_agent = BaselineConfidenceAgent(env.action_space)
    reflection_agent = ReflectionAgent(env.action_space)
    reflection_agent.confidence_threshold = 0.25  # 设置默认阈值
    reflection_agent.adaptation_threshold = 0.45  # 设置默认阈值
    
    # 设置目标位置
    reflection_agent.set_goal_position(env.goal_pos)
    
    # 统计数据
    baseline_stats = {'successes': 0, 'total_steps': 0, 'total_rewards': 0}
    reflection_stats = {'successes': 0, 'total_steps': 0, 'total_rewards': 0}
    
    try:
        episode = 0
        while episode < num_episodes and viz.running:
            # 重置环境
            baseline_state, _ = env.reset(seed=base_seed + episode)
            reflection_state = baseline_state.copy()
            
            baseline_reward = 0
            reflection_reward = 0
            steps = 0
            done = False
            
            while not done and steps < max_steps and viz.running:
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        viz.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            viz.paused = not viz.paused
                
                if viz.paused:
                    continue
                
                # 获取动作
                baseline_action = baseline_agent.select_action(baseline_state)
                reflection_action = reflection_agent.select_action(reflection_state)
                
                # 执行动作
                baseline_next_state, b_reward, b_done, _, _ = env.step(baseline_action)
                current_pos = np.array(env.current_pos)
                env.current_pos = reflection_state
                reflection_next_state, r_reward, r_done, _, _ = env.step(reflection_action)
                env.current_pos = current_pos
                
                # 更新累积奖励
                baseline_reward += b_reward
                reflection_reward += r_reward
                
                # 学习
                baseline_agent.learn(baseline_state, baseline_action, b_reward, 
                                  baseline_next_state, b_done, steps, 
                                  env.get_optimal_path_length())
                reflection_agent.learn(reflection_state, reflection_action, r_reward, 
                                    reflection_next_state, r_done, steps,
                                    env.get_optimal_path_length())
                
                # Update states
                baseline_state = baseline_next_state
                reflection_state = reflection_next_state
                steps += 1
                done = b_done or r_done
                
                # Update visualization
                viz.current_maze = env.maze
                viz.baseline_pos = baseline_state
                viz.reflection_pos = reflection_state
                viz.goal_pos = env.goal_pos
                
                # Draw and update display
                viz.draw()
                
                # Control update rate
                pygame.time.wait(100)  # 100ms delay for better visualization
            
            # Calculate results
            baseline_success = b_done and np.array_equal(baseline_state, env.goal_pos)
            reflection_success = r_done and np.array_equal(reflection_state, env.goal_pos)
            
            # Update statistics
            baseline_stats['total_steps'] += steps
            baseline_stats['total_rewards'] += baseline_reward
            if baseline_success:
                baseline_stats['successes'] += 1
                
            reflection_stats['total_steps'] += steps
            reflection_stats['total_rewards'] += reflection_reward
            if reflection_success:
                reflection_stats['successes'] += 1
            
            episode += 1
            print(f"Episode {episode} completed - Baseline success: {baseline_success}, Reflection success: {reflection_success}")
            
            # Print statistics every 10 episodes
            if episode % 10 == 0:
                baseline_success_rate = baseline_stats['successes'] / episode
                reflection_success_rate = reflection_stats['successes'] / episode
                baseline_avg_steps = baseline_stats['total_steps'] / episode
                reflection_avg_steps = reflection_stats['total_steps'] / episode
                baseline_avg_reward = baseline_stats['total_rewards'] / episode
                reflection_avg_reward = reflection_stats['total_rewards'] / episode
                
                print(f"\n=== Episode {episode} Statistics ===")
                print(f"Baseline Agent (Baseline):")
                print(f"  Success Rate: {baseline_success_rate:.3f}")
                print(f"  Average Steps: {baseline_avg_steps:.1f}")
                print(f"  Average Reward: {baseline_avg_reward:.2f}")
                print(f"Reflection Agent (Reflection):")
                print(f"  Success Rate: {reflection_success_rate:.3f}")
                print(f"  Average Steps: {reflection_avg_steps:.1f}")
                print(f"  Average Reward: {reflection_avg_reward:.2f}")
                print("=" * 30)
            
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    finally:
        # Print final statistics
        if episode > 0:
            baseline_success_rate = baseline_stats['successes'] / episode
            reflection_success_rate = reflection_stats['successes'] / episode
            baseline_avg_steps = baseline_stats['total_steps'] / episode
            reflection_avg_steps = reflection_stats['total_steps'] / episode
            baseline_avg_reward = baseline_stats['total_rewards'] / episode
            reflection_avg_reward = reflection_stats['total_rewards'] / episode
            
            print(f"\n{'='*50}")
            print(f"Final Experimental Results (Total {episode} episodes)")
            print(f"{'='*50}")
            print(f"Baseline Agent:")
            print(f"  Total Successes: {baseline_stats['successes']}")
            print(f"  Success Rate: {baseline_success_rate:.3f} ({baseline_success_rate*100:.1f}%)")
            print(f"  Average Steps: {baseline_avg_steps:.1f}")
            print(f"  Average Reward: {baseline_avg_reward:.2f}")
            print(f"\nReflection Agent:")
            print(f"  Total Successes: {reflection_stats['successes']}")
            print(f"  Success Rate: {reflection_success_rate:.3f} ({reflection_success_rate*100:.1f}%)")
            print(f"  Average Steps: {reflection_avg_steps:.1f}")
            print(f"  Average Reward: {reflection_avg_reward:.2f}")
            print(f"\nPerformance Comparison:")
            success_improvement = (reflection_success_rate - baseline_success_rate) * 100
            step_improvement = ((baseline_avg_steps - reflection_avg_steps) / baseline_avg_steps) * 100
            reward_improvement = ((reflection_avg_reward - baseline_avg_reward) / abs(baseline_avg_reward)) * 100
            print(f"  Success Rate Improvement: {success_improvement:+.1f}%")
            print(f"  Step Efficiency Improvement: {step_improvement:+.1f}%")
            print(f"  Reward Improvement: {reward_improvement:+.1f}%")
            print(f"{'='*50}")
        
        viz.running = False
        pygame.quit()

if __name__ == "__main__":
    main()
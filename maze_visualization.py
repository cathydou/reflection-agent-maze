import pygame
import sys
import numpy as np
from dynamic_maze_env import DynamicMazeEnv
from baseline_confidence_agent import BaselineConfidenceAgent
from reflection_agent import ReflectionAgent

class MazeVisualization:
    def __init__(self, width=800, height=1000):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Dynamic Maze Environment")
        
        # Grid settings
        self.grid_size = 10
        self.cell_size = min(width, height - 200) // self.grid_size
        self.grid_offset_x = (width - self.cell_size * self.grid_size) // 2
        self.grid_offset_y = (height - 200 - self.cell_size * self.grid_size) // 2
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (240, 240, 240)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        
        # Font setup
        pygame.font.init()
        self.font = pygame.font.Font(None, 36)
        
        # Animation settings
        self.paused = False
        self.running = True
        
        # Current state
        self.current_maze = None
        self.baseline_pos = None
        self.reflection_pos = None
        self.goal_pos = None
        
    def draw_grid(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = self.grid_offset_x + j * self.cell_size
                y = self.grid_offset_y + i * self.cell_size
                
                # Draw cell
                cell_color = self.BLACK if self.current_maze[i][j] else self.WHITE
                pygame.draw.rect(self.screen, cell_color, 
                               (x, y, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, self.GRAY, 
                               (x, y, self.cell_size, self.cell_size), 1)
                
                # Draw agents and goal
                if self.baseline_pos is not None and i == self.baseline_pos[0] and j == self.baseline_pos[1]:
                    pygame.draw.circle(self.screen, self.RED, 
                                    (x + self.cell_size//2, y + self.cell_size//2), 
                                    self.cell_size//3)
                
                if self.reflection_pos is not None and i == self.reflection_pos[0] and j == self.reflection_pos[1]:
                    pygame.draw.circle(self.screen, self.BLUE, 
                                    (x + self.cell_size//2, y + self.cell_size//2), 
                                    self.cell_size//3)
                
                if self.goal_pos is not None and i == self.goal_pos[0] and j == self.goal_pos[1]:
                    pygame.draw.circle(self.screen, self.GREEN, 
                                    (x + self.cell_size//2, y + self.cell_size//2), 
                                    self.cell_size//3)
    
    def draw(self):
        self.screen.fill(self.WHITE)
        if self.current_maze is not None:
            self.draw_grid()
        pygame.display.flip()

def main():
    # 环境参数
    env_params = {
        'size': 10,
        'obstacle_ratio': 0.25,
        'change_frequency': 18
    }
    max_steps = 200
    num_episodes = 100
    
    # 创建可视化和环境
    viz = MazeVisualization()
    env = DynamicMazeEnv(**env_params)
    baseline_agent = BaselineConfidenceAgent(env.action_space)
    reflection_agent = ReflectionAgent(env.action_space)
    
    # 初始化状态
    state, _ = env.reset()
    baseline_state = state.copy()
    reflection_state = state.copy()
    reflection_agent.set_goal_position(env.goal_pos)
    episode_steps = 0
    
    # 初始化统计数据
    baseline_stats = {'successes': 0, 'failures': 0}
    reflection_stats = {'successes': 0, 'failures': 0}
    
    try:
        episode = 0
        while episode < num_episodes and viz.running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    viz.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        viz.paused = not viz.paused
            
            if not viz.paused:
                # 获取动作
                baseline_action = baseline_agent.select_action(baseline_state)
                reflection_action = reflection_agent.select_action(reflection_state)
                
                # 执行动作
                baseline_next_state, b_reward, b_done, _, _ = env.step(baseline_action)
                current_pos = np.array(env.current_pos)
                env.current_pos = reflection_state
                reflection_next_state, r_reward, r_done, _, _ = env.step(reflection_action)
                env.current_pos = current_pos
                
                # 更新状态
                baseline_state = baseline_next_state
                reflection_state = reflection_next_state
                episode_steps += 1
                
                # 计算最短路径
                shortest_path = env.get_optimal_path_length()
                
                # 学习
                baseline_agent.learn(baseline_state, baseline_action, b_reward, 
                                  baseline_next_state, b_done, episode_steps, shortest_path)
                reflection_agent.learn(reflection_state, reflection_action, r_reward, 
                                    reflection_next_state, r_done, episode_steps, shortest_path)
                
                # 更新显示
                viz.current_maze = env.maze
                viz.baseline_pos = baseline_state
                viz.reflection_pos = reflection_state
                viz.goal_pos = env.goal_pos
                
                # 绘制和更新显示
                viz.draw()
                pygame.time.wait(100)  # 添加延迟使移动可见
                
                # 检查是否需要重置
                if b_done or r_done or episode_steps >= max_steps:
                    print(f"\nEpisode {episode} completed:")
                    print(f"Steps taken: {episode_steps}")
                    print(f"Baseline position: {baseline_state}")
                    print(f"Baseline goal reached: {b_done and np.array_equal(baseline_state, env.goal_pos)}")
                    print(f"Reflection position: {reflection_state}")
                    print(f"Reflection goal reached: {r_done and np.array_equal(reflection_state, env.goal_pos)}")
                    print(f"Goal position: {env.goal_pos}")
                    print(f"Max steps reached: {episode_steps >= max_steps}\n")
                    
                    # 更新统计
                    if b_done and np.array_equal(baseline_state, env.goal_pos):
                        baseline_stats['successes'] += 1
                    if r_done and np.array_equal(reflection_state, env.goal_pos):
                        reflection_stats['successes'] += 1
                    
                    # 重置环境和状态
                    state, _ = env.reset()
                    baseline_state = state.copy()
                    reflection_state = state.copy()
                    reflection_agent.set_goal_position(env.goal_pos)
                    episode_steps = 0
                    episode += 1
                    
                    # 每10轮打印统计信息
                    if episode % 10 == 0:
                        b_success_rate = baseline_stats['successes'] / episode
                        r_success_rate = reflection_stats['successes'] / episode
                        print(f"\nEpisode {episode} Statistics:")
                        print(f"Baseline Success Rate: {b_success_rate:.2f}")
                        print(f"Reflection Success Rate: {r_success_rate:.2f}")
    
    except Exception as e:
        print(f"Error occurred: {e}")
        raise
    
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()

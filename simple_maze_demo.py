#!/usr/bin/env python3
"""
简化的迷宫演示程序
展示小球在迷宫中的移动过程
"""

import pygame
import numpy as np
import sys
import os

# 确保当前目录在Python路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dynamic_maze_env import DynamicMazeEnv
from baseline_confidence_agent import BaselineConfidenceAgent
from reflection_agent import ReflectionAgent

class SimpleMazeVisualization:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("迷宫演示 - 小球走迷宫")
        
        # 网格设置
        self.grid_size = 10
        self.cell_size = min(width, height - 100) // self.grid_size
        self.grid_offset_x = (width - self.cell_size * self.grid_size) // 2
        self.grid_offset_y = (height - 100 - self.cell_size * self.grid_size) // 2
        
        # 颜色
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.RED = (255, 0, 0)      # 基线智能体
        self.BLUE = (0, 0, 255)     # 反思智能体
        self.GREEN = (0, 255, 0)    # 目标
        self.YELLOW = (255, 255, 0) # 墙壁
        
        # 字体设置
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        
        # 运行状态
        self.running = True
        self.paused = False
        
    def draw_maze(self, maze, agent_pos, goal_pos):
        """绘制迷宫"""
        self.screen.fill(self.WHITE)
        
        # 绘制网格
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = self.grid_offset_x + j * self.cell_size
                y = self.grid_offset_y + i * self.cell_size
                
                # 绘制单元格
                if maze[i][j]:  # 墙壁
                    pygame.draw.rect(self.screen, self.YELLOW, 
                                   (x, y, self.cell_size, self.cell_size))
                else:  # 空地
                    pygame.draw.rect(self.screen, self.WHITE, 
                                   (x, y, self.cell_size, self.cell_size))
                
                # 绘制网格线
                pygame.draw.rect(self.screen, self.GRAY, 
                               (x, y, self.cell_size, self.cell_size), 1)
                
                # 绘制智能体（小球）
                if i == agent_pos[0] and j == agent_pos[1]:
                    pygame.draw.circle(self.screen, self.RED, 
                                    (x + self.cell_size//2, y + self.cell_size//2), 
                                    self.cell_size//3)
                
                # 绘制目标
                if i == goal_pos[0] and j == goal_pos[1]:
                    pygame.draw.circle(self.screen, self.GREEN, 
                                    (x + self.cell_size//2, y + self.cell_size//2), 
                                    self.cell_size//3)
        
        # 绘制说明文字
        info_text = [
            "迷宫演示 - 小球走迷宫",
            "红色小球: 智能体",
            "绿色圆圈: 目标",
            "黄色方块: 墙壁",
            "空格键: 暂停/继续",
            "ESC键: 退出"
        ]
        
        for i, text in enumerate(info_text):
            text_surface = self.font.render(text, True, self.BLACK)
            self.screen.blit(text_surface, (10, 10 + i * 25))
        
        pygame.display.flip()
    
    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
        return self.running

def main():
    """主程序"""
    print("启动迷宫演示程序...")
    print("程序将显示一个pygame窗口，展示小球在迷宫中的移动")
    print("按空格键暂停/继续，按ESC键退出")
    
    # 创建可视化
    viz = SimpleMazeVisualization()
    
    # 创建环境
    env = DynamicMazeEnv(size=10, obstacle_ratio=0.25, change_frequency=20)
    env.max_steps = 100
    
    # 创建智能体
    agent = ReflectionAgent(env.action_space)
    agent.confidence_threshold = 0.25
    agent.adaptation_threshold = 0.45
    
    # 设置目标位置
    agent.set_goal_position(env.goal_pos)
    
    try:
        episode = 0
        while episode < 5 and viz.running:  # 只运行5个episode
            print(f"\n开始第 {episode + 1} 个episode")
            
            # 重置环境
            state, _ = env.reset()
            steps = 0
            done = False
            
            while not done and steps < env.max_steps and viz.running:
                # 处理事件
                if not viz.handle_events():
                    break
                
                if viz.paused:
                    pygame.time.wait(100)
                    continue
                
                # 智能体选择动作
                action = agent.select_action(state)
                
                # 执行动作
                next_state, reward, done, _, _ = env.step(action)
                
                # 学习
                agent.learn(state, action, reward, next_state, done, steps, 
                          env.get_optimal_path_length())
                
                # 更新状态
                state = next_state
                steps += 1
                
                # 更新可视化
                viz.draw_maze(env.maze, state, env.goal_pos)
                
                # 控制速度
                pygame.time.wait(200)  # 200ms延迟，使移动更容易观察
                
                # 检查是否到达目标
                if done and np.array_equal(state, env.goal_pos):
                    print(f"Episode {episode + 1} 成功完成！步数: {steps}")
                elif steps >= env.max_steps:
                    print(f"Episode {episode + 1} 超时，步数: {steps}")
            
            episode += 1
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("程序结束")
        viz.running = False
        pygame.quit()

if __name__ == "__main__":
    main()

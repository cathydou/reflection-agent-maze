import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging

class DynamicMazeEnv(gym.Env):
    """动态迷宫环境"""
    
    # 定义动作空间
    ACTIONS = {
        0: np.array([-1, 0]),  # 上
        1: np.array([1, 0]),   # 下
        2: np.array([0, -1]),  # 左
        3: np.array([0, 1])    # 右
    }
    
    def __init__(self, size=10, obstacle_ratio=0.3, change_frequency=20, seed=None):
        super().__init__()
        
        # 环境参数
        self.size = size
        self.obstacle_ratio = obstacle_ratio
        self.change_frequency = change_frequency
        self.max_steps = 200  # 添加最大步数限制
        
        # 动作和观察空间
        self.action_space = gym.spaces.Discrete(4)  # 上下左右四个动作
        self.observation_space = gym.spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)
        
        # 奖励设置 - 移到这里
        self.STEP_PENALTY = -0.1
        self.COLLISION_PENALTY = -1.0
        self.GOAL_REWARD = 10.0
        
        # 初始化随机数生成器
        self.np_random = np.random.default_rng(seed)
        
        # 位置相关
        self.current_pos = None
        self.previous_pos = None
        self.goal_pos = None
        self.obstacle_positions = set()
        
        # 环境动态变化相关
        self._steps = 0
        self.last_change_step = 0
        
        # 记录每个episode的数据
        self.episode_data = {
            'environment_updates': 0,  # 环境更新次数
            'goal_changes': 0,         # 目标位置变化次数
            'obstacle_changes': 0,     # 障碍物变化次数
            'path_length': 0           # 最优路径长度
        }
        
        # 初始化环境
        self.reset(seed=seed)
        
        # 迷宫相关属性
        self.maze = None
        
    def seed(self, seed=None):
        """设置随机种子"""
        if seed is not None:
            np.random.seed(seed)
        return [seed]
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 生成新迷宫
        self.maze = self.generate_maze()
        
        # 设置起点（左上角区域）
        self.current_pos = self.find_empty_position()
        while self.current_pos[0] > self.size//3 or self.current_pos[1] > self.size//3:
            self.current_pos = self.find_empty_position()
        
        # 确保智能体不会被封死
        self._ensure_agent_not_trapped()
        
        # 设置终点（右下角区域）
        self.goal_pos = self.find_empty_position()
        while (self.goal_pos[0] < 2*self.size//3 or 
               self.goal_pos[1] < 2*self.size//3 or
               np.sum(np.abs(self.goal_pos - self.current_pos)) < self.size):
            self.goal_pos = self.find_empty_position()
        
        # 重置步数和位置记录
        self._steps = 0
        self.previous_pos = self.current_pos.copy()
        
        return self.current_pos, {}
        
    def step(self, action):
        """执行一步动作"""
        self._steps += 1
        
        # 获取动作对应的方向
        direction = self.ACTIONS[action]
        next_pos = self.current_pos + direction
        
        # 检查是否超出边界
        if (next_pos < 0).any() or (next_pos >= self.size).any():
            return self.current_pos, self.COLLISION_PENALTY, False, {}, {}
        
        # 检查是否撞墙
        if self.maze[tuple(next_pos)] == 1:
            return self.current_pos, self.COLLISION_PENALTY, False, {}, {}
        
        # 更新位置
        self.current_pos = next_pos
        
        # 检查是否到达目标
        reached_goal = np.array_equal(self.current_pos, self.goal_pos)
        done = reached_goal or self._steps >= self.max_steps
        
        # 计算到目标的距离
        old_distance = np.sum(np.abs(self.previous_pos - self.goal_pos))
        new_distance = np.sum(np.abs(self.current_pos - self.goal_pos))
        
        # 奖励结构
        if reached_goal:
            reward = self.GOAL_REWARD
        elif self.maze[tuple(next_pos)] == 1:  # 撞墙
            reward = self.COLLISION_PENALTY
        else:
            # 根据是否接近目标给予奖励
            reward = self.STEP_PENALTY + (old_distance - new_distance)
        
        # 动态更新环境
        if self._steps % self.change_frequency == 0:
            self.update_environment()
        
        self.previous_pos = self.current_pos.copy()
        
        return self.current_pos, reward, done, {}, {}
        
    def generate_maze(self):
        """生成迷宫"""
        maze = np.zeros((self.size, self.size), dtype=np.int32)
        num_obstacles = int(self.size * self.size * self.obstacle_ratio)
        
        # 随机放置障碍物
        obstacle_positions = self.np_random.choice(
            self.size * self.size,
            size=num_obstacles,
            replace=False
        )
        
        for pos in obstacle_positions:
            row = pos // self.size
            col = pos % self.size
            maze[row, col] = 1
        
        return maze
    
    def find_empty_position(self):
        """找到一个合适的空位置"""
        while True:
            pos = np.array([
                self.np_random.integers(0, self.size),
                self.np_random.integers(0, self.size)
            ])
            if self.maze[tuple(pos)] == 0:  # 确保位置是空的
                return pos
    
    def update_environment(self):
        """更新环境"""
        # 保存旧的目标位置
        old_goal = tuple(self.goal_pos)  # 转换为元组以便比较
        
        # 随机更新一些障碍物
        num_changes = self.np_random.integers(1, 4)
        for _ in range(num_changes):
            x = self.np_random.integers(0, self.size)
            y = self.np_random.integers(0, self.size)
            pos = np.array([x, y])
            
            # 使用 np.array_equal 进行数组比较
            if not (np.array_equal(pos, self.current_pos) or 
                   np.array_equal(pos, self.goal_pos)):
                self.maze[x, y] = 1 - self.maze[x, y]
        
        # 确保路径存在
        self._ensure_path_exists()
        
        # 确保智能体不会被封死
        self._ensure_agent_not_trapped()
        
        # 检查目标是否改变
        if not np.array_equal(old_goal, self.goal_pos):
            self.episode_data['goal_changes'] += 1
        
        self.episode_data['environment_updates'] += 1
        
    def _set_goal(self):
        """设置目标位置"""
        while True:
            x = self.np_random.integers(0, self.size)
            y = self.np_random.integers(0, self.size)
            candidate_pos = np.array([x, y])
            
            # 使用 np.array_equal 进行数组比较
            if (not np.array_equal(candidate_pos, self.current_pos) and 
                self.maze[x, y] == 0):  # 确保目标不在障碍物上
                self.goal_pos = candidate_pos
                break
                
    def _calculate_reward(self, new_pos, hit_obstacle):
        """计算奖励"""
        if hit_obstacle:
            return self.reward_scale['obstacle']
            
        if new_pos == self.goal_pos:
            return self.reward_scale['goal']
            
        # 计算与目标的曼哈顿距离变化
        old_dist = abs(self.current_pos[0] - self.goal_pos[0]) + abs(self.current_pos[1] - self.goal_pos[1])
        new_dist = abs(new_pos[0] - self.goal_pos[0]) + abs(new_pos[1] - self.goal_pos[1])
        
        if new_dist < old_dist:
            return self.reward_scale['approach']
        elif new_dist > old_dist:
            return self.reward_scale['away']
        return 0
        
    def bfs(self, start, goal, maze):
        """广度优先搜索寻找路径"""
        if np.array_equal(start, goal):
            return True, [start]
        
        visited = set()
        queue = [(start, [start])]
        visited.add(tuple(start))
        
        while queue:
            current, path = queue.pop(0)
            
            for action in self.ACTIONS.values():
                next_pos = current + action
                next_pos_tuple = tuple(next_pos)
                
                # 检查是否有效
                if (next_pos < 0).any() or (next_pos >= self.size).any():
                    continue
                if maze[tuple(next_pos)] == 1:  # 是障碍物
                    continue
                if next_pos_tuple in visited:
                    continue
                
                # 检查是否到达目标
                if np.array_equal(next_pos, goal):
                    return True, path + [next_pos]
                
                # 添加到队列
                visited.add(next_pos_tuple)
                queue.append((next_pos, path + [next_pos]))
        
        return False, []

    def _ensure_path_exists(self):
        """确保存在从当前位置到目标的路径"""
        path_exists, _ = self.bfs(self.current_pos, self.goal_pos, self.maze)
        if not path_exists:
            # 如果没有路径，移除一些障碍物直到有路径
            while not path_exists:
                # 随机选择一个障碍物移除
                obstacles = np.where(self.maze == 1)
                if len(obstacles[0]) == 0:  # 没有障碍物了
                    break
                idx = self.np_random.integers(0, len(obstacles[0]))
                x, y = obstacles[0][idx], obstacles[1][idx]
                pos = np.array([x, y])
                
                # 不移除起点和终点位置的障碍物
                if not (np.array_equal(pos, self.current_pos) or 
                       np.array_equal(pos, self.goal_pos)):
                    self.maze[x, y] = 0
                    path_exists, _ = self.bfs(self.current_pos, self.goal_pos, self.maze)

    def get_optimal_path_length(self):
        """获取最短路径长度"""
        from queue import Queue
        
        if np.array_equal(self.current_pos, self.goal_pos):
            return 0
            
        visited = set()
        q = Queue()
        q.put((tuple(self.current_pos), 0))
        visited.add(tuple(self.current_pos))
        
        while not q.empty():
            pos, dist = q.get()
            
            for action in self.ACTIONS.values():
                next_pos = tuple(np.array(pos) + action)
                
                if (next_pos[0] < 0 or next_pos[0] >= self.size or
                    next_pos[1] < 0 or next_pos[1] >= self.size or
                    self.maze[next_pos] == 1 or
                    next_pos in visited):
                    continue
                
                if np.array_equal(next_pos, tuple(self.goal_pos)):
                    return dist + 1
                
                visited.add(next_pos)
                q.put((next_pos, dist + 1))
        
        return float('inf')  # 如果没有路径

    def get_current_metrics(self):
        """返回当前环境的指标"""
        return {
            'goal_changes': self.episode_data['goal_changes'],
            'obstacle_changes': self.episode_data['obstacle_changes'],
            'path_length': self.get_optimal_path_length(),
            'environment_stability': 1.0 / (1.0 + self.episode_data['environment_updates']),
            'current_distance': (abs(self.current_pos[0] - self.goal_pos[0]) + 
                               abs(self.current_pos[1] - self.goal_pos[1])),
            'obstacle_density': len(self.obstacle_positions) / (self.size * self.size),
            'steps_since_last_change': self._steps - self.last_change_step
        }

    def _ensure_agent_not_trapped(self):
        """确保智能体不会被障碍物封死"""
        # 检查智能体周围是否至少有一个可移动的方向
        has_valid_move = False
        for action in self.ACTIONS.values():
            next_pos = self.current_pos + action
            # 检查是否有效移动
            if ((next_pos >= 0).all() and (next_pos < self.size).all() and 
                self.maze[tuple(next_pos)] == 0):
                has_valid_move = True
                break
        
        # 如果没有有效移动，清除周围的一个障碍物
        if not has_valid_move:
            # 优先考虑右边和下面的方向（对左上角尤其重要）
            priority_actions = [1, 3, 0, 2]  # 下、右、上、左
            for action_idx in priority_actions:
                action = self.ACTIONS[action_idx]
                next_pos = self.current_pos + action
                # 检查位置是否在边界内
                if ((next_pos >= 0).all() and (next_pos < self.size).all()):
                    # 如果是障碍物，移除它
                    if self.maze[tuple(next_pos)] == 1:
                        self.maze[tuple(next_pos)] = 0
                        return  # 只需要清除一个障碍物
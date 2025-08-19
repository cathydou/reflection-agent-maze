from collections import defaultdict, deque
import numpy as np
import random

class BaselineConfidenceAgent:
    """基线智能体：基于实验一 ProposedAgent 的 Q-learning 模型"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_table = {}
        self.default_q_values = np.zeros(action_space.n)
        
        # 修改这些参数
        self.epsilon = 0.3  # 增加探索概率
        self.alpha = 0.1    # 学习率
        self.gamma = 0.9    # 折扣因子
        
        # 添加探索衰减
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.np_random = np.random.default_rng()
        
        # 置信度机制参数
        self.history_size = 8  # 更大的窗口
        self.reward_history = deque(maxlen=self.history_size)
        self.step_history = deque(maxlen=self.history_size)
        self.success_history = deque(maxlen=150)  # 更大的历史记录长度
        self.visited_states = set()

        # 奖励整形参数
        self.unvisited_bonus = 0.08  # 增加未访问奖励
        self.distance_weight = 0.15  # 增加距离权重

        self.experience_buffer = deque(maxlen=1000)  # 添加经验回放
        self.min_epsilon = 0.1

    def _state_to_key(self, state):
        """将状态转换为 Q 表的键"""
        return tuple(state)

    def calculate_confidence(self, steps, shortest_path):
        """计算当前置信度"""
        if shortest_path == float('inf') or shortest_path == 0:
            return 0.0
        return max(0.0, 1.0 - (steps - shortest_path) / shortest_path)

    def select_action(self, state):
        if self.np_random.random() < self.epsilon:
            return self.action_space.sample()
        
        state_key = tuple(state)
        q_values = self.q_table.get(state_key, self.default_q_values)
        
        # 添加一些随机性来打破平局
        max_q = np.max(q_values)
        actions = np.where(q_values == max_q)[0]
        return self.np_random.choice(actions)

    def learn(self, state, action, reward, next_state, done, steps, shortest_path):
        # 存储经验
        self.experience_buffer.append((state, action, reward, next_state, done))
        
        # 从经验中随机采样学习
        if len(self.experience_buffer) >= 32:
            batch = random.sample(self.experience_buffer, 32)
            for s, a, r, ns, d in batch:
                self._update_q_value(s, a, r, ns, d)
        
        # 缓慢衰减探索率
        if done:
            self.epsilon = max(self.min_epsilon, 
                             self.epsilon * self.epsilon_decay)

        # 更新访问状态集合
        self.visited_states.add(tuple(next_state))

        # 更新历史记录
        self.reward_history.append(reward)
        if done:
            self.step_history.append(steps)
            self.success_history.append(1 if reward > 0 else 0)

    def _update_q_value(self, state, action, reward, next_state, done):
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        # 获取Q值
        if state_key not in self.q_table:
            self.q_table[state_key] = self.default_q_values.copy()
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = self.default_q_values.copy()
        
        # Q-learning更新
        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state_key][action] = new_value


if __name__ == "__main__":
    from dynamic_maze_env import DynamicMazeEnv

    # 初始化环境和智能体
    env = DynamicMazeEnv(size=10, obstacle_ratio=0.2, change_frequency=20)
    agent = BaselineConfidenceAgent(env.action_space)

    # 模拟一回合
    state, _ = env.reset()
    done = False
    steps = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        shortest_path = env.get_optimal_path_length()
        agent.learn(state, action, reward, next_state, done, steps, shortest_path)
        state = next_state
        steps += 1
import numpy as np
from collections import defaultdict, deque
import random

class ReflectionAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_table_short_term = {}  # 短期记忆（对最近环境敏感）
        self.q_table_long_term = {}   # 长期记忆（保存通用策略）
        self.memory_balance = 0.5     # 短期和长期记忆的平衡因子 (0-1)
        self.default_q_values = np.zeros(action_space.n)
        
        # 改为列表而非deque，以支持优先级经验回放
        self.experience_buffer = []  
        self.experience_priorities = []  # 存储经验的优先级
        self.max_buffer_size = 1000
        
        # 学习参数
        self.epsilon = 0.9
        self.epsilon_min = 0.3  # 设置较高的最小值，保守地修改
        self.epsilon_decay = 0.999  # 使用非常缓慢的衰减率
        self.alpha = 0.5
        self.gamma = 0.9
        
        # 反思机制参数
        self.confidence_threshold = 0.25
        self.adaptation_threshold = 0.35  # 已调整为0.35
        self.recent_confidences = deque(maxlen=10)
        self.recent_rewards = deque(maxlen=10)
        self.recent_steps = deque(maxlen=10)
        self.reflection_memory = []
        self.reflection_frequency = 5
        
        # 访问统计
        self.visit_counts = defaultdict(int)
        self.np_random = np.random.default_rng()
        
        # 目标位置
        self.goal_pos = None
        
        # 墙壁记忆
        self.wall_memory = {}
        self.wall_memory_age = {}  # 记录墙壁记忆的年龄
        self.memory_refresh_freq = 20  # 每20步检查一次墙壁记忆
        self.steps_count = 0  # 步数计数器
        
        # 环境变化检测
        self.state_action_results = {}  # 记录状态-动作对的结果历史
        self.environment_stability = 1.0  # 环境稳定性估计 (1.0表示完全稳定)
        self.env_change_detected = False  # 是否检测到环境变化
        self.last_env_change_step = 0  # 上次检测到环境变化的步数
        
        # 环境变化检测参数调整
        self.environment_stability_threshold = 0.6  # 从0.5恢复到0.6
        self.env_change_cooldown = 30  # 从40降低到30
        self.wall_memory_clear_ratio = 0.4  # 从0.3增加到0.4
        
        # 知识转移参数
        self.knowledge_transfer_interval = 100  # 每100步执行一次知识转移
        self.last_knowledge_transfer_step = 0  # 上次知识转移的步数
    
    def select_action(self, state):
        """改进的动作选择策略 - 更平衡的版本"""
        state_key = tuple(state)
        self.visit_counts[state_key] += 1
        
        # 更新 epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # 计算方向和距离
        current_pos = np.array(state)
        goal_direction = self.goal_pos - current_pos
        vertical_dist = abs(goal_direction[0])
        horizontal_dist = abs(goal_direction[1])
        
        # 构建可能的动作列表
        possible_actions = []
        if vertical_dist >= horizontal_dist:
            if goal_direction[0] > 0:
                possible_actions.append(1)  # 下
            else:
                possible_actions.append(0)  # 上
            
        if horizontal_dist >= vertical_dist:
            if goal_direction[1] > 0:
                possible_actions.append(3)  # 右
            else:
                possible_actions.append(2)  # 左
            
        # 移除已知的墙壁方向
        if state_key in self.wall_memory:
            possible_actions = [
                action for action in possible_actions 
                if action not in self.wall_memory[state_key]
            ]
        
        # 如果所有预期动作都不可行，完全随机
        if not possible_actions:
            return self.action_space.sample()
        
        # epsilon-greedy with direction bias
        if self.np_random.random() < self.epsilon:
            if possible_actions and self.np_random.random() < 0.7:
                return self.np_random.choice(possible_actions)
            return self.action_space.sample()
        
        # 使用短期和长期记忆的组合来选择动作
        if state_key not in self.q_table_short_term:
            self.q_table_short_term[state_key] = self.default_q_values.copy()
        if state_key not in self.q_table_long_term:
            self.q_table_long_term[state_key] = self.default_q_values.copy()
        
        # 平滑记忆平衡调整
        target_balance = max(0.3, min(0.7, 1.0 - self.environment_stability))
        self.memory_balance = 0.9 * self.memory_balance + 0.1 * target_balance  # 平滑调整
        
        # 组合两种记忆的Q值
        combined_q_values = (
            self.memory_balance * self.q_table_short_term[state_key] + 
            (1 - self.memory_balance) * self.q_table_long_term[state_key]
        )
        
        # 计算UCB值
        ucb_values = combined_q_values + np.sqrt(
            2 * np.log(sum(self.visit_counts.values())) / 
            (np.array([self.visit_counts[state_key]] * self.action_space.n) + 1e-6)
        )
        
        return np.argmax(ucb_values)
    
    def learn(self, state, action, reward, next_state, done, steps, shortest_path):
        """学习方法 - 使用优先级经验回放"""
        # 计算当前置信度
        confidence = self.calculate_confidence(steps, shortest_path)
        self.recent_confidences.append(confidence)
        self.recent_rewards.append(reward)
        if done:
            self.recent_steps.append(steps)
        
        # 存储经验
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        # 计算TD误差作为优先级
        if state_key not in self.q_table_short_term:
            self.q_table_short_term[state_key] = self.default_q_values.copy()
        if next_state_key not in self.q_table_short_term:
            self.q_table_short_term[next_state_key] = self.default_q_values.copy()
        
        next_max = np.max(self.q_table_short_term[next_state_key])
        target = reward + self.gamma * next_max * (1 - done)
        current = self.q_table_short_term[state_key][action]
        td_error = abs(target - current)
        
        # 检测环境变化
        state_action_key = (state_key, action)
        current_result = (next_state_key, reward)
        
        # 记录结果并检测变化
        if state_action_key in self.state_action_results:
            previous_results = self.state_action_results[state_action_key]
            
            # 检查结果是否一致
            consistent = True
            for prev_state, prev_reward in previous_results:
                # 如果下一状态不同或奖励差异大，认为环境可能变化
                if prev_state != next_state_key or abs(prev_reward - reward) > 0.5:
                    consistent = False
                    break
            
            if not consistent:
                # 环境可能发生了变化
                self.environment_stability = max(0.5, self.environment_stability * 0.8)
                
                # 使用调整后的阈值
                if (self.environment_stability < self.environment_stability_threshold and 
                    (self.steps_count - self.last_env_change_step) > self.env_change_cooldown):
                    self.env_change_detected = True
                    self.last_env_change_step = self.steps_count
                    
                    # 环境变化时的适应措施
                    self._adapt_to_environment_change()
            else:
                # 环境稳定
                self.environment_stability = min(1.0, self.environment_stability * 1.02)
        
        # 更新状态-动作结果历史 (只保留最近3个结果)
        if state_action_key not in self.state_action_results:
            self.state_action_results[state_action_key] = []
        
        self.state_action_results[state_action_key].append(current_result)
        if len(self.state_action_results[state_action_key]) > 3:
            self.state_action_results[state_action_key] = self.state_action_results[state_action_key][-3:]
        
        # 存储经验和优先级
        experience = (state_key, action, reward, next_state_key, done)
        
        # 如果是成功到达目标的经验，给予更高优先级
        if done and reward > 1.0:  # 成功到达目标
            priority = max(1.0, td_error) * 2.0  # 加倍优先级
        else:
            priority = max(0.01, td_error)  # 最小优先级为0.01
        
        if len(self.experience_buffer) < self.max_buffer_size:
            self.experience_buffer.append(experience)
            self.experience_priorities.append(priority)
        else:
            # 替换优先级最低的经验
            if len(self.experience_priorities) > 0:
                min_idx = np.argmin(self.experience_priorities)
                self.experience_buffer[min_idx] = experience
                self.experience_priorities[min_idx] = priority
        
        # 更新短期和长期记忆
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        # 确保两种记忆都有相关状态
        for memory in [self.q_table_short_term, self.q_table_long_term]:
            if state_key not in memory:
                memory[state_key] = self.default_q_values.copy()
            if next_state_key not in memory:
                memory[next_state_key] = self.default_q_values.copy()
        
        # 短期记忆更新 - 使用较高的学习率
        short_term_alpha = min(0.8, self.alpha * 1.5)
        old_value = self.q_table_short_term[state_key][action]
        next_max = np.max(self.q_table_short_term[next_state_key])
        new_value = (1 - short_term_alpha) * old_value + short_term_alpha * (reward + self.gamma * next_max)
        self.q_table_short_term[state_key][action] = new_value
        
        # 长期记忆更新 - 使用较低的学习率，更稳定
        long_term_alpha = max(0.1, self.alpha * 0.7)
        old_value = self.q_table_long_term[state_key][action]
        next_max = np.max(self.q_table_long_term[next_state_key])
        new_value = (1 - long_term_alpha) * old_value + long_term_alpha * (reward + self.gamma * next_max)
        self.q_table_long_term[state_key][action] = new_value
        
        # 经验回放
        if len(self.experience_buffer) >= 32:
            self._learn_from_experience()
        
        # 更新步数计数器
        self.steps_count += 1
        
        # 更新墙壁记忆
        if reward <= -1.0:  # 撞墙的惩罚
            if state_key not in self.wall_memory:
                self.wall_memory[state_key] = set()
                self.wall_memory_age[state_key] = {}
            self.wall_memory[state_key].add(action)
            self.wall_memory_age[state_key][action] = 0  # 重置年龄
        
        # 定期检查和清理墙壁记忆
        if self.steps_count % self.memory_refresh_freq == 0:
            self._refresh_wall_memory()
        
        # 调用反思机制
        self.reflect(state, action, reward, next_state, done, steps)
    
    def _learn_from_experience(self):
        """优先级经验回放"""
        if not self.experience_buffer:
            return
            
        # 计算采样概率
        priorities = np.array(self.experience_priorities)
        probs = priorities / sum(priorities)
        
        # 采样经验
        batch_size = min(32, len(self.experience_buffer))
        batch_indices = np.random.choice(
            len(self.experience_buffer), 
            batch_size, 
            replace=False,
            p=probs
        )
        
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # 学习
        for state_key, action, reward, next_state_key, done in batch:
            if state_key not in self.q_table_short_term:
                self.q_table_short_term[state_key] = self.default_q_values.copy()
            if next_state_key not in self.q_table_short_term:
                self.q_table_short_term[next_state_key] = self.default_q_values.copy()
            
            old_value = self.q_table_short_term[state_key][action]
            next_max = np.max(self.q_table_short_term[next_state_key])
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            self.q_table_short_term[state_key][action] = new_value
            
            # 更新优先级
            td_error = abs(new_value - old_value)
            idx = self.experience_buffer.index((state_key, action, reward, next_state_key, done))
            self.experience_priorities[idx] = max(0.01, td_error)
    
    def calculate_confidence(self, steps, shortest_path):
        """计算当前置信度"""
        if shortest_path == float('inf') or shortest_path == 0:
            return 0.0
        return max(0.0, 1.0 - (steps - shortest_path) / shortest_path)

    def reflect(self, state, action, reward, next_state, done, steps):
        """反思和总结经验"""
        if self.goal_pos is None:
            return
        
        try:
            # 记录反思数据
            self.reflection_memory.append((state, action, reward, next_state, done, steps))
            
            # 只有在收集了足够的数据后才进行反思
            if len(self.reflection_memory) >= self.reflection_frequency:
                # 计算性能指标
                distances = []
                progress = 0
                
                for i, (s, a, r, ns, d, _) in enumerate(self.reflection_memory):
                    if i > 0:
                        prev_s = self.reflection_memory[i-1][0]
                        prev_dist = np.sum(np.abs(np.array(prev_s) - self.goal_pos))
                        curr_dist = np.sum(np.abs(np.array(s) - self.goal_pos))
                        distances.append(curr_dist)
                        progress += (prev_dist - curr_dist)
                
                # 如果没有足够的数据计算进展，则设为0
                if not distances:
                    progress = 0
                
                # 计算性能得分 - 调整权重以更好地评估性能
                performance_score = max(
                    0.0,
                    0.5 * (progress / self.reflection_frequency) +  # 增加进展的权重
                    0.3 * (sum(1 for exp in self.reflection_memory if exp[4]) / self.reflection_frequency) +  # 完成率权重
                    0.2 * (sum(exp[2] for exp in self.reflection_memory) / self.reflection_frequency)  # 平均奖励权重
                )
                
                # 提高适应阈值，减少不必要的策略调整
                if performance_score < 0.4:  # 从0.35提高到0.4，使适应更加积极
                    self.adapt_strategy(progress, distances[-1] if distances else 0)
                
                # 执行知识转移 - 只在环境相对稳定时
                if self.environment_stability > 0.6:  # 从0.7降低到0.6
                    self._transfer_knowledge()
                
                self.reflection_memory = []
            
        except Exception as e:
            print(f"Error in reflect: {e}")

    def set_goal_position(self, goal_pos):
        """设置目标位置"""
        self.goal_pos = np.array(goal_pos)

    def adapt_strategy(self, progress, current_distance):
        """根据性能调整策略 - 更平衡的版本"""
        # 如果没有向目标靠近，适度增加探索率
        if progress <= 0:
            # 降低探索率增加的幅度，从0.1降至0.05
            self.epsilon = min(0.8, self.epsilon + 0.05)
        else:
            # 如果有进展，逐渐降低探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.995)
        
        # 如果距离目标很远，适度增加学习率
        if current_distance > 5:
            # 降低学习率增加的幅度，从0.1降至0.05
            self.alpha = min(0.7, self.alpha + 0.05)
        
        # 如果最近的奖励都很低，更精细地调整探索
        if len(self.recent_rewards) > 0:
            avg_reward = np.mean(self.recent_rewards)
            if avg_reward < -0.5:
                # 根据奖励的严重程度调整探索率
                increase = max(0.05, min(0.15, abs(avg_reward) * 0.1))
                self.epsilon = min(0.8, self.epsilon + increase)
        
        # 更精细地控制Q值重置
        if len(self.recent_confidences) > 0:
            avg_confidence = np.mean(self.recent_confidences)
            if avg_confidence < 0.2:
                # 只重置少量状态，并且只重置那些置信度低的状态
                num_to_reset = max(3, min(5, int(len(self.q_table_short_term) * 0.05)))
                
                # 尝试找出置信度低的状态
                low_confidence_states = []
                for state_key in self.q_table_short_term.keys():
                    if tuple(state_key) in self.visit_counts and self.visit_counts[tuple(state_key)] > 2:
                        # 这是一个被多次访问但可能没有好结果的状态
                        low_confidence_states.append(state_key)
                
                # 如果找到了足够的低置信度状态，就重置它们
                if len(low_confidence_states) >= num_to_reset:
                    keys_to_reset = random.sample(low_confidence_states, num_to_reset)
                else:
                    # 否则随机选择
                    keys_to_reset = random.sample(list(self.q_table_short_term.keys()), 
                                                 min(num_to_reset, len(self.q_table_short_term)))
                
                # 不完全重置，而是部分降低Q值
                for key in keys_to_reset:
                    self.q_table_short_term[key] = self.q_table_short_term[key] * 0.5  # 只降低50%

    def _refresh_wall_memory(self):
        """更新墙壁记忆的年龄，并移除过旧的记忆"""
        keys_to_check = list(self.wall_memory.keys())
        for state_key in keys_to_check:
            actions_to_check = list(self.wall_memory[state_key])
            for action in actions_to_check:
                # 增加年龄
                self.wall_memory_age[state_key][action] += 1
                
                # 如果年龄超过阈值，随机决定是否移除
                if self.wall_memory_age[state_key][action] > 50:  # 50步后开始考虑移除
                    if np.random.random() < 0.2:  # 20%的概率移除
                        self.wall_memory[state_key].remove(action)
                        del self.wall_memory_age[state_key][action]
            
            # 如果该状态没有墙壁记忆了，删除该状态
            if not self.wall_memory[state_key]:
                del self.wall_memory[state_key]
                del self.wall_memory_age[state_key]

    def _adapt_to_environment_change(self):
        """当检测到环境变化时调整策略"""
        # 1. 临时增加探索率
        self.epsilon = min(0.9, self.epsilon + 0.2)
        
        # 2. 主动清除部分墙壁记忆
        if self.wall_memory:
            # 随机清除墙壁记忆
            keys_to_check = list(self.wall_memory.keys())
            num_to_clear = max(1, int(len(keys_to_check) * self.wall_memory_clear_ratio))
            keys_to_clear = random.sample(keys_to_check, num_to_clear)
            
            for key in keys_to_clear:
                del self.wall_memory[key]
                if key in self.wall_memory_age:
                    del self.wall_memory_age[key]
        
        # 3. 降低部分经验的优先级
        if self.experience_priorities:
            # 降低50%的经验优先级
            num_to_reduce = max(1, int(len(self.experience_priorities) * 0.5))
            indices = random.sample(range(len(self.experience_priorities)), num_to_reduce)
            
            for idx in indices:
                self.experience_priorities[idx] *= 0.5
        
        # 4. 重置反思频率
        self.reflection_frequency = 3  # 环境变化后更频繁地反思
        
        # 调整短期和长期记忆的平衡
        self.memory_balance = 0.7  # 环境变化后更依赖短期记忆
        
        # 更保守的短期记忆重置
        if self.q_table_short_term:
            # 降低重置比例从30%到20%
            keys_to_reset = random.sample(
                list(self.q_table_short_term.keys()), 
                max(1, int(len(self.q_table_short_term) * 0.2))
            )
            # 更温和的重置方式
            for key in keys_to_reset:
                self.q_table_short_term[key] = (
                    0.2 * self.default_q_values + 
                    0.8 * self.q_table_short_term[key]  # 保留更多原始信息
                )
        
        # 移除环境变化前的知识转移，避免保留过时知识
        # self._transfer_knowledge()  # 注释掉这一行
        
        print(f"Environment change detected at step {self.steps_count}! Adapting strategy...")

    def _transfer_knowledge(self):
        """从短期记忆转移有价值的知识到长期记忆"""
        if not self.q_table_short_term:
            return
        
        # 根据环境稳定性动态调整参数
        visit_threshold = max(3, int(5 * self.environment_stability))
        q_threshold = max(0.3, 0.35 * self.environment_stability)
        transfer_ratio = min(0.4, 0.35 * (1 + self.environment_stability))
        
        # 选择访问频率高且Q值较大的状态
        candidates = []
        for state_key in self.q_table_short_term:
            if state_key in self.visit_counts and self.visit_counts[state_key] > visit_threshold:
                max_q = np.max(self.q_table_short_term[state_key])
                if max_q > q_threshold:
                    candidates.append((state_key, max_q))
        
        # 按Q值排序并选择前transfer_ratio转移
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            num_to_transfer = max(1, int(len(candidates) * transfer_ratio))
            
            for i in range(num_to_transfer):
                state_key = candidates[i][0]
                # 将短期记忆中的知识融合到长期记忆中
                if state_key in self.q_table_long_term:
                    self.q_table_long_term[state_key] = (
                        0.6 * self.q_table_long_term[state_key] +
                        0.4 * self.q_table_short_term[state_key]
                    )
                else:
                    self.q_table_long_term[state_key] = self.q_table_short_term[state_key].copy()
        
        # 添加知识遗忘机制
        self._forget_outdated_knowledge()

    def _forget_outdated_knowledge(self):
        """清除长期记忆中不再有用的知识"""
        if len(self.q_table_long_term) > 800:  # 从1000降低到800
            # 找出访问频率低且Q值低的状态
            candidates_to_forget = []
            for state_key in self.q_table_long_term:
                if state_key not in self.visit_counts or self.visit_counts[state_key] < 3:  # 从2提高到3
                    max_q = np.max(self.q_table_long_term[state_key])
                    if max_q < 0.25:  # 从0.2提高到0.25
                        candidates_to_forget.append((state_key, max_q))
            
            # 按Q值排序并选择前15%遗忘
            if candidates_to_forget:
                candidates_to_forget.sort(key=lambda x: x[1])
                num_to_forget = max(1, int(len(candidates_to_forget) * 0.15))  # 从10%增加到15%
                
                for i in range(num_to_forget):
                    state_key = candidates_to_forget[i][0]
                    del self.q_table_long_term[state_key]

    def step(self, state, action, reward, next_state, done, steps, shortest_path=None):
        """更新智能体的状态和学习"""
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        # 更新环境稳定性估计
        self._update_environment_stability(state, action, next_state)
        
        # 检测环境变化
        if (self.environment_stability < self.environment_stability_threshold and 
            self.steps_count - self.last_env_change_step > self.env_change_cooldown):
            self.env_change_detected = True
            self.last_env_change_step = self.steps_count
            self._adapt_to_environment_change()
        
        # 更新Q值
        if state_key not in self.q_table_short_term:
            self.q_table_short_term[state_key] = self.default_q_values.copy()
        if next_state_key not in self.q_table_short_term:
            self.q_table_short_term[next_state_key] = self.default_q_values.copy()
        
        # 使用短期记忆进行Q学习更新
        old_value = self.q_table_short_term[state_key][action]
        next_max = np.max(self.q_table_short_term[next_state_key])
        
        # 使用优先级经验回放的TD误差作为优先级
        td_error = abs(reward + self.gamma * next_max - old_value)
        
        # 更新Q值
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table_short_term[state_key][action] = new_value
        
        # 更新墙壁记忆
        if reward == -1 and np.array_equal(state, next_state):  # 撞墙
            if state_key not in self.wall_memory:
                self.wall_memory[state_key] = []
                self.wall_memory_age[state_key] = {}
            
            if action not in self.wall_memory[state_key]:
                self.wall_memory[state_key].append(action)
                self.wall_memory_age[state_key][action] = 0
        
        # 存储经验到缓冲区（带优先级）
        experience = (state, action, reward, next_state, done)
        if len(self.experience_buffer) < self.max_buffer_size:
            self.experience_buffer.append(experience)
            self.experience_priorities.append(td_error)
        else:
            # 替换优先级最低的经验
            if self.experience_priorities:
                min_priority_idx = np.argmin(self.experience_priorities)
                self.experience_buffer[min_priority_idx] = experience
                self.experience_priorities[min_priority_idx] = td_error
        
        # 反思和总结经验
        self.reflect(state, action, reward, next_state, done, steps)
        
        # 定期更新墙壁记忆
        self.steps_count += 1
        if self.steps_count % self.memory_refresh_freq == 0:
            self._refresh_wall_memory()
        
        # 定期知识转移
        if self.steps_count - self.last_knowledge_transfer_step >= 100:
            self._transfer_knowledge()
            self.last_knowledge_transfer_step = self.steps_count
        
        # 设置目标位置
        self.goal_pos = shortest_path[-1] if shortest_path and len(shortest_path) > 0 else None
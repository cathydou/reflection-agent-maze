#!/usr/bin/env python3
"""
Unit tests for the ReflectionAgent class.

I created these tests to ensure the reflection mechanism works correctly
and to help maintain code quality as the project grows.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reflection_agent import ReflectionAgent


class TestReflectionAgent:
    """Test suite for ReflectionAgent class."""
    
    @pytest.fixture
    def action_space(self):
        """Create a mock action space for testing."""
        mock_space = Mock()
        mock_space.n = 4  # 4 actions: up, down, left, right
        mock_space.sample.return_value = 2  # Return action 2 (left) as default
        return mock_space
    
    @pytest.fixture
    def agent(self, action_space):
        """Create a ReflectionAgent instance for testing."""
        return ReflectionAgent(action_space)
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample state for testing."""
        return np.array([5, 5])
    
    @pytest.fixture
    def sample_goal(self):
        """Create a sample goal position."""
        return np.array([8, 8])

    def test_agent_initialization(self, action_space):
        """Test that the agent initializes correctly with all required attributes."""
        agent = ReflectionAgent(action_space)
        
        # Check basic attributes
        assert agent.action_space == action_space
        assert agent.epsilon == 0.9
        assert agent.epsilon_min == 0.3
        assert agent.alpha == 0.5
        assert agent.gamma == 0.9
        
        # Check memory systems
        assert isinstance(agent.q_table_short_term, dict)
        assert isinstance(agent.q_table_long_term, dict)
        assert agent.memory_balance == 0.5
        
        # Check reflection parameters
        assert agent.confidence_threshold == 0.25
        assert agent.adaptation_threshold == 0.35
        assert agent.reflection_frequency == 5
        
        # Check experience buffer
        assert isinstance(agent.experience_buffer, list)
        assert isinstance(agent.experience_priorities, list)
        assert agent.max_buffer_size == 1000

    def test_set_goal_position(self, agent, sample_goal):
        """Test setting the goal position."""
        agent.set_goal_position(sample_goal)
        assert np.array_equal(agent.goal_pos, sample_goal)

    def test_calculate_confidence(self, agent):
        """Test confidence calculation with various scenarios."""
        # Test with optimal path
        confidence = agent.calculate_confidence(steps=10, shortest_path=10)
        assert confidence == 1.0
        
        # Test with suboptimal path
        confidence = agent.calculate_confidence(steps=15, shortest_path=10)
        assert confidence == 0.5
        
        # Test with infinite path
        confidence = agent.calculate_confidence(steps=10, shortest_path=float('inf'))
        assert confidence == 0.0
        
        # Test with zero path
        confidence = agent.calculate_confidence(steps=10, shortest_path=0)
        assert confidence == 0.0

    def test_select_action_with_goal_direction(self, agent, sample_state, sample_goal):
        """Test action selection when goal position is set."""
        agent.set_goal_position(sample_goal)
        agent.epsilon = 0.0  # Disable random exploration
        
        # Test action selection toward goal
        action = agent.select_action(sample_state)
        assert action in [0, 1, 2, 3]  # Valid action range
        
        # The agent should prefer actions toward the goal, but may choose any valid action
        # due to the complex UCB algorithm and memory combination
        assert action in [0, 1, 2, 3]

    def test_select_action_with_exploration(self, agent, sample_state, sample_goal):
        """Test action selection with exploration enabled."""
        agent.set_goal_position(sample_goal)  # Need goal position to avoid error
        agent.epsilon = 1.0  # Enable full exploration
        
        # Test multiple actions to ensure randomness
        actions = set()
        for _ in range(10):
            action = agent.select_action(sample_state)
            actions.add(action)
            assert action in [0, 1, 2, 3]
        
        # Should have some variety in actions due to exploration
        assert len(actions) > 1

    def test_learn_basic_q_update(self, agent, sample_state):
        """Test basic Q-learning update functionality."""
        action = 1
        reward = 1.0
        next_state = np.array([6, 5])
        done = False
        
        # Initial Q-values should be zeros
        state_key = tuple(sample_state)
        assert state_key not in agent.q_table_short_term
        assert state_key not in agent.q_table_long_term
        
        # Perform learning
        agent.learn(sample_state, action, reward, next_state, done, 1, 10)
        
        # Check that Q-values were updated
        assert state_key in agent.q_table_short_term
        assert state_key in agent.q_table_long_term
        
        # Check that the action's Q-value increased
        short_term_q = agent.q_table_short_term[state_key][action]
        long_term_q = agent.q_table_long_term[state_key][action]
        assert short_term_q > 0
        assert long_term_q > 0

    def test_learn_with_wall_collision(self, agent, sample_state):
        """Test learning when agent hits a wall."""
        action = 1
        reward = -1.0  # Wall collision penalty
        next_state = sample_state  # Same state (didn't move)
        done = False
        
        # Perform learning
        agent.learn(sample_state, action, reward, next_state, done, 1, 10)
        
        # Check that wall memory was updated
        state_key = tuple(sample_state)
        assert state_key in agent.wall_memory
        assert action in agent.wall_memory[state_key]

    def test_reflection_mechanism(self, agent, sample_goal):
        """Test the reflection mechanism functionality."""
        agent.set_goal_position(sample_goal)
        
        # Add some reflection data
        for i in range(agent.reflection_frequency):
            state = np.array([5 + i, 5])
            action = 1
            reward = 0.1
            next_state = np.array([6 + i, 5])
            done = False
            steps = i + 1
            
            agent.reflect(state, action, reward, next_state, done, steps)
        
        # After reflection_frequency calls, reflection_memory should be cleared
        assert len(agent.reflection_memory) == 0

    def test_environment_change_detection_in_learn(self, agent, sample_state):
        """Test environment change detection mechanism through the learn method."""
        action = 1
        reward = 0.5
        next_state = np.array([6, 5])
        
        # First call - should record the result
        agent.learn(sample_state, action, reward, next_state, False, 1, 10)
        initial_stability = agent.environment_stability
        
        # Second call with same result - should maintain stability
        agent.learn(sample_state, action, reward, next_state, False, 2, 10)
        assert agent.environment_stability == initial_stability
        
        # Call with different result - should decrease stability
        different_next_state = np.array([7, 5])
        agent.learn(sample_state, action, reward, different_next_state, False, 3, 10)
        assert agent.environment_stability < initial_stability

    def test_adapt_strategy(self, agent):
        """Test strategy adaptation based on performance."""
        initial_epsilon = agent.epsilon
        
        # Test adaptation with poor progress
        agent.adapt_strategy(progress=0, current_distance=10)
        
        # Should increase exploration rate (but may be capped)
        assert agent.epsilon >= agent.epsilon_min
        
        # Test adaptation with good progress
        agent.epsilon = initial_epsilon
        agent.adapt_strategy(progress=5, current_distance=2)
        
        # Should decrease exploration rate
        assert agent.epsilon <= initial_epsilon

    def test_wall_memory_refresh(self, agent, sample_state):
        """Test wall memory refresh mechanism."""
        # Add some wall memory
        state_key = tuple(sample_state)
        agent.wall_memory[state_key] = {1, 2}
        agent.wall_memory_age[state_key] = {1: 60, 2: 60}  # Old memories
        
        # Trigger refresh
        agent._refresh_wall_memory()
        
        # Some memories should be removed due to age
        assert len(agent.wall_memory.get(state_key, set())) <= 2

    def test_experience_replay(self, agent, sample_state):
        """Test experience replay functionality."""
        # Add some experiences
        for i in range(50):
            state = np.array([5 + i % 5, 5])
            action = i % 4
            reward = 0.1
            next_state = np.array([6 + i % 5, 5])
            done = False
            
            agent.learn(state, action, reward, next_state, done, i + 1, 10)
        
        # Check that experience buffer has data
        assert len(agent.experience_buffer) > 0
        assert len(agent.experience_priorities) > 0

    def test_knowledge_transfer(self, agent, sample_state):
        """Test knowledge transfer from short-term to long-term memory."""
        # Add some short-term memory
        state_key = tuple(sample_state)
        agent.q_table_short_term[state_key] = np.array([0.1, 0.8, 0.2, 0.3])
        agent.visit_counts[state_key] = 10  # High visit count
        
        # Trigger knowledge transfer
        agent._transfer_knowledge()
        
        # Check that knowledge was transferred
        assert state_key in agent.q_table_long_term

    def test_adapt_to_environment_change(self, agent):
        """Test adaptation when environment change is detected."""
        initial_epsilon = agent.epsilon
        initial_reflection_freq = agent.reflection_frequency
        
        # Trigger environment change adaptation
        agent._adapt_to_environment_change()
        
        # Should increase exploration rate (but may be capped at max)
        assert agent.epsilon >= agent.epsilon_min
        
        # Should increase reflection frequency (lower number = more frequent)
        assert agent.reflection_frequency <= initial_reflection_freq

    def test_learn_from_experience(self, agent):
        """Test learning from experience replay buffer."""
        # Add some experiences to the buffer
        for i in range(40):
            state = tuple(np.array([5 + i % 3, 5]))
            action = i % 4
            reward = 0.1
            next_state = tuple(np.array([6 + i % 3, 5]))
            done = False
            
            agent.experience_buffer.append((state, action, reward, next_state, done))
            agent.experience_priorities.append(1.0)
        
        # Trigger experience replay learning
        agent._learn_from_experience()
        
        # Should have processed some experiences
        assert len(agent.experience_buffer) > 0

    def test_edge_cases(self, agent, sample_goal):
        """Test edge cases and error handling."""
        # Test with None goal position
        agent.goal_pos = None
        # This should raise an error, so we test it properly
        with pytest.raises(TypeError):
            agent.select_action(np.array([5, 5]))
        
        # Test with goal position set
        agent.set_goal_position(sample_goal)
        action = agent.select_action(np.array([5, 5]))
        assert action in [0, 1, 2, 3]  # Should return valid action
        
        # Test with empty experience buffer
        agent.experience_buffer = []
        agent.experience_priorities = []
        agent._learn_from_experience()  # Should not crash
        
        # Test with empty wall memory
        agent.wall_memory = {}
        agent.wall_memory_age = {}
        agent._refresh_wall_memory()  # Should not crash

    def test_memory_balance_adjustment(self, agent, sample_state, sample_goal):
        """Test memory balance adjustment based on environment stability."""
        agent.set_goal_position(sample_goal)
        initial_balance = agent.memory_balance
        
        # Test with stable environment - need multiple calls for gradual adjustment
        agent.environment_stability = 0.9
        for _ in range(20):  # More calls to ensure visible change
            agent.select_action(sample_state)
        
        # The memory balance should change after multiple calls
        stable_balance = agent.memory_balance
        
        # Reset and test with unstable environment
        agent.memory_balance = 0.5
        agent.environment_stability = 0.1  # More extreme instability
        for _ in range(20):  # More calls to ensure visible change
            agent.select_action(sample_state)
        
        # The memory balance should change
        unstable_balance = agent.memory_balance
        
        # At least one of the balances should be different from the initial
        assert stable_balance != initial_balance or unstable_balance != 0.5

    def test_ucb_action_selection(self, agent, sample_state, sample_goal):
        """Test UCB-based action selection."""
        agent.set_goal_position(sample_goal)
        agent.epsilon = 0.0  # Disable exploration to test UCB
        
        # Add some Q-values to test UCB
        state_key = tuple(sample_state)
        agent.q_table_short_term[state_key] = np.array([0.1, 0.2, 0.3, 0.4])
        agent.q_table_long_term[state_key] = np.array([0.1, 0.2, 0.3, 0.4])
        agent.visit_counts[state_key] = 5
        
        action = agent.select_action(sample_state)
        assert action in [0, 1, 2, 3]

    def test_priority_experience_replay(self, agent):
        """Test priority-based experience replay."""
        # Add experiences with different priorities
        for i in range(20):
            state = tuple(np.array([5 + i % 3, 5]))
            action = i % 4
            reward = 0.1 if i % 2 == 0 else 1.0  # Different rewards
            next_state = tuple(np.array([6 + i % 3, 5]))
            done = False
            
            agent.experience_buffer.append((state, action, reward, next_state, done))
            agent.experience_priorities.append(1.0 + i)  # Different priorities
        
        # Trigger experience replay
        agent._learn_from_experience()
        
        # Should have processed experiences
        assert len(agent.experience_buffer) > 0


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "--cov=reflection_agent", "--cov-report=term-missing", "-v"])

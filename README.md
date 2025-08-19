# 🧠 Reflection Agent: Intelligent Reflection Mechanism in Dynamic Environments

## 📖 Project Introduction

This project implements a reflective agent (Reflection Agent) that can autonomously learn, adapt, and optimize strategies in dynamically changing maze environments. By introducing reflection mechanisms, the agent significantly improves performance in complex environments.

## 🎯 Core Features

### 🔄 Reflection Mechanism
- **Real-time Performance Evaluation**: Self-reflection every 5 steps
- **Dynamic Strategy Adjustment**: Automatically adjust exploration rate and learning rate based on performance
- **Environment Change Detection**: Intelligently identify environment changes and adapt quickly

### 🧠 Dual Memory System
- **Short-term Memory**: Quickly adapt to environment changes
- **Long-term Memory**: Preserve stable general strategies
- **Intelligent Balance**: Dynamically adjust memory weights based on environment stability

### 🎮 Intelligent Decision Making
- **Direction Priority**: Prioritize directions toward the goal
- **Wall Memory**: Remember and avoid known walls
- **UCB Algorithm**: Optimal decision making balancing exploration and exploitation

## 📊 Experimental Results

In a 100-episode dynamic maze experiment:

| Metric | Baseline Agent | Reflection Agent | Improvement |
|--------|----------------|------------------|-------------|
| **Success Rate** | 2.0% | **40.0%** | **+38.0%** |
| **Average Reward** | -97.17 | **30.52** | **+131.4%** |
| **Total Successes** | 2 times | **40 times** | **20x** |

## 🏗️ Project Structure

```
reflection-agent/
├── README.md                 # Project documentation
├── requirements.txt          # Dependency list
├── main_experiment2.py      # Main experiment program
├── reflection_agent.py      # Reflection agent implementation
├── baseline_agent.py        # Baseline agent implementation
├── dynamic_maze_env.py      # Dynamic maze environment
├── maze_visualization.py    # Visualization interface
├── results/                 # Experimental results
│   ├── performance_plots/   # Performance charts
│   └── training_logs/       # Training logs
└── docs/                    # Documentation
    ├── methodology.md       # Methodology
    └── results_analysis.md  # Results analysis
```

## 🚀 Quick Start

### Requirements
- Python 3.8+
- Pygame
- NumPy
- Matplotlib

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Experiments
```bash
# Run complete comparison experiment
python main_experiment2.py

# Run visualization demo
python maze_visualization.py
```

## 🔬 Technical Details

### Reflection Mechanism Parameters
- **Reflection Frequency**: Every 5 steps
- **Confidence Threshold**: 0.25
- **Adaptation Threshold**: 0.45
- **Environment Stability Threshold**: 0.6

### Learning Parameters
- **Learning Rate**: 0.5 (dynamic adjustment)
- **Discount Factor**: 0.9
- **Exploration Rate**: 0.9 → 0.3 (dynamic decay)
- **Experience Buffer Size**: 1000

## 📈 Performance Analysis

### Environment Change Detection
The agent can detect real-time changes in maze structure, including:
- Wall position changes
- Reward function changes
- Goal position changes

### Strategy Adaptation
When environment changes are detected, the agent will:
1. Increase exploration rate
2. Clear partial outdated memories
3. Adjust learning parameters
4. Reflect more frequently

## 🎨 Visualization

The project includes a complete visualization interface showing:
- Maze layout
- Agent movement trajectories
- Real-time performance metrics
- Environment change detection

## 📚 Related Research

This project explores cutting-edge concepts in AI:
- **Meta-Learning**
- **Reflective Learning**
- **Dynamic Environment Adaptation**
- **Multi-Memory Systems**

## 🤝 Contributing

Welcome to submit Issues and Pull Requests! Please ensure:
1. Code follows PEP 8 standards
2. Add appropriate comments and documentation
3. Test new features
4. Update related documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 👨‍💻 Author

**Zhou Chenyu** - Reflection Agent Research

## 🙏 Acknowledgments

Thanks to all researchers and developers who contributed to this project.

---

⭐ **If this project helps you, please give it a star!**

# 🧠 Reflection Agent: Intelligent Reflection Mechanism in Dynamic Environments

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/cathydou/reflection-agent-maze.svg)](https://github.com/cathydou/reflection-agent-maze/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/cathydou/reflection-agent-maze.svg)](https://github.com/cathydou/reflection-agent-maze/network)
[![GitHub issues](https://img.shields.io/github/issues/cathydou/reflection-agent-maze.svg)](https://github.com/cathydou/reflection-agent-maze/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/cathydou/reflection-agent-maze.svg)](https://github.com/cathydou/reflection-agent-maze/pulls)
[![Research](https://img.shields.io/badge/Research-Paper-blue.svg)](papers/README.md)
[![Demo](https://img.shields.io/badge/Demo-Videos-red.svg)](demos/README.md)

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

## 📚 Research Papers & Demos

### 📄 Academic Papers
- **[Reflection Agent: Dynamic Environment Adaptation through Meta-Learning](papers/README.md)** - Complete research paper with theoretical foundations
- **[Implementation Details and Experimental Results](papers/README.md)** - Comprehensive analysis and results

### 🎥 Demo Videos
- **[Agent Comparison Demo](demos/README.md)** - Side-by-side performance comparison
- **[Technical Deep Dive](demos/README.md)** - Detailed mechanism explanation
- **[Live Experiment Run](demos/README.md)** - Complete 100-episode experiment

### 🔬 Related Research Areas
This project explores cutting-edge concepts in AI:
- **Meta-Learning**
- **Reflective Learning**
- **Dynamic Environment Adaptation**
- **Multi-Memory Systems**

## 🤝 Contributing

We welcome contributions from the community! Here are several ways you can contribute:

### 🐛 Report Bugs
- Open an issue with a detailed description of the bug
- Include steps to reproduce and expected behavior
- Add labels like `bug`, `help wanted`, or `good first issue`

### 💡 Suggest Features
- Propose new ideas in issues
- Discuss implementation approaches
- Help prioritize feature development

### 🔧 Submit Code
- Fork the repository
- Create a feature branch (`git checkout -b feature/amazing-feature`)
- Commit your changes (`git commit -m 'Add amazing feature'`)
- Push to the branch (`git push origin feature/amazing-feature`)
- Open a Pull Request

### 📚 Improve Documentation
- Fix typos and improve clarity
- Add examples and tutorials
- Translate to other languages

### 🧪 Help with Testing
- Test on different platforms
- Report performance issues
- Suggest test cases

**Please ensure:**
1. Code follows PEP 8 standards
2. Add appropriate comments and documentation
3. Test new features thoroughly
4. Update related documentation

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{zhou2024reflection,
  title={Reflection Agent: Intelligent Reflection Mechanism in Dynamic Environments},
  author={Zhou Chenyu},
  year={2024},
  url={https://github.com/cathydou/reflection-agent-maze}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 👨‍💻 Author

**Zhou Chenyu** - Reflection Agent Research

## 🙏 Acknowledgments

Thanks to all researchers and developers who contributed to this project.

---

⭐ **If this project helps you, please give it a star!**

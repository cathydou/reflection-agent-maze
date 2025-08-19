import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_training_history():
    # 读取训练历史
    df = pd.read_csv('training_history.txt')
    
    # 计算移动平均
    window = 10
    df['success_rate_ma'] = df['Success'].rolling(window).mean()
    df['efficiency'] = df['ShortestPath'] / df['Steps']
    df['efficiency_ma'] = df['efficiency'].rolling(window).mean()
    
    # 绘制图表
    plt.figure(figsize=(15, 10))
    
    # 成功率
    plt.subplot(2, 2, 1)
    plt.plot(df['Episode'], df['success_rate_ma'])
    plt.title('Success Rate (10-episode moving average)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    
    # 步数效率
    plt.subplot(2, 2, 2)
    plt.plot(df['Episode'], df['efficiency_ma'])
    plt.title('Path Efficiency (Shortest/Actual Steps)')
    plt.xlabel('Episode')
    plt.ylabel('Efficiency')
    
    # 最终距离
    plt.subplot(2, 2, 3)
    plt.plot(df['Episode'], df['FinalDistance'])
    plt.title('Final Distance to Goal')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    
    plt.tight_layout()
    plt.savefig('training_analysis.png')
    
    # 打印统计信息
    print("\nTraining Statistics:")
    print(f"Overall Success Rate: {df['Success'].mean():.2f}")
    print(f"Average Steps per Episode: {df['Steps'].mean():.2f}")
    print(f"Average Path Efficiency: {df['efficiency'].mean():.2f}")
    
    # 分析学习进展
    first_quarter = df['Success'][:len(df)//4].mean()
    last_quarter = df['Success'][3*len(df)//4:].mean()
    print(f"\nLearning Progress:")
    print(f"First Quarter Success Rate: {first_quarter:.2f}")
    print(f"Last Quarter Success Rate: {last_quarter:.2f}") 
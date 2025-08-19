#!/bin/bash

# 论文上传脚本
echo "🚀 开始上传论文到GitHub..."

# 检查论文文件是否存在
if [ ! -f "reflection_agent_paper.pdf" ]; then
    echo "❌ 错误：找不到 reflection_agent_paper.pdf 文件"
    echo "请确保论文文件在当前目录中"
    exit 1
fi

# 复制论文到papers目录
echo "📄 复制论文文件..."
cp reflection_agent_paper.pdf papers/reflection_agent_paper_2024.pdf

# 检查复制是否成功
if [ $? -eq 0 ]; then
    echo "✅ 论文文件复制成功"
else
    echo "❌ 论文文件复制失败"
    exit 1
fi

# 添加到git
echo "📝 添加到git..."
git add papers/

# 提交更改
echo "💾 提交更改..."
git commit -m "Add main research paper: Reflection Agent implementation"

# 推送到GitHub
echo "🚀 推送到GitHub..."
git push origin main

# 检查推送是否成功
if [ $? -eq 0 ]; then
    echo "🎉 论文上传成功！"
    echo "📖 现在可以在 https://github.com/cathydou/reflection-agent-maze 查看你的论文"
else
    echo "❌ 推送到GitHub失败，请检查网络连接"
    exit 1
fi

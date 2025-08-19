# 📤 File Upload Guide

This guide explains how to add your research papers and demo videos to the project.

## 📄 Adding Research Papers

### 1. Paper Files
Place your paper files in the `papers/` directory:
```
papers/
├── README.md                    # This file (already created)
├── reflection_agent_paper.pdf   # Your main research paper
├── implementation_details.pdf   # Implementation analysis
└── supplementary_materials/     # Additional resources
    ├── appendix.pdf
    ├── datasets/
    └── code_examples/
```

### 2. File Naming Convention
- Use descriptive names: `reflection_agent_paper_2024.pdf`
- Include year in filename
- Use underscores instead of spaces
- Keep names under 50 characters

### 3. File Formats
- **PDF**: For research papers (recommended)
- **DOCX**: For editable versions
- **TEX**: For LaTeX source files
- **MD**: For markdown versions

## 🎥 Adding Demo Videos

### 1. Video Files
Place your video files in the `demos/videos/` directory:
```
demos/
├── README.md                    # Demo documentation (already created)
├── videos/                      # Video files
│   ├── comparison_demo.mp4      # Agent comparison
│   ├── technical_deep_dive.mp4  # Technical explanation
│   └── live_experiment.mp4      # Live experiment run
└── screenshots/                 # Key moments
    ├── episode_10_results.png
    ├── environment_change.png
    └── final_statistics.png
```

### 2. Video Specifications
- **Format**: MP4 (H.264 codec)
- **Resolution**: 1080p (1920x1080) or higher
- **Frame Rate**: 30fps
- **Audio**: Clear narration with good quality
- **Duration**: 5-15 minutes per video

### 3. Video Content Structure
Each demo video should include:
- **Introduction** (30 seconds)
- **Main content** (4-14 minutes)
- **Summary** (30 seconds)
- **Call to action** (15 seconds)

## 📸 Adding Screenshots

### 1. Screenshot Types
- **Performance graphs**: Learning curves, success rates
- **Interface snapshots**: Key moments in experiments
- **Results tables**: Final statistics and comparisons
- **Process diagrams**: Algorithm flowcharts

### 2. Screenshot Specifications
- **Format**: PNG or JPG
- **Resolution**: At least 1920x1080
- **Quality**: High quality, clear text
- **Annotations**: Add arrows, circles, or text to highlight key points

## 🔗 Adding External Links

### 1. Video Platforms
If you upload videos to external platforms, add links in `demos/README.md`:
```markdown
## 🔗 Video Links

- **YouTube**: [Your Channel Link]
- **Bilibili**: [B站链接]
- **Direct Download**: Available in this repository
```

### 2. Paper Repositories
If you publish papers on external platforms:
```markdown
## 🔗 Paper Links

- **arXiv**: [arXiv link]
- **ResearchGate**: [ResearchGate link]
- **Google Scholar**: [Google Scholar link]
```

## 📋 Upload Checklist

Before uploading files:

- [ ] **Papers**: PDF format, clear title, proper citations
- [ ] **Videos**: MP4 format, good audio, clear narration
- [ ] **Screenshots**: High quality, annotated if needed
- [ ] **Documentation**: Updated README files
- [ ] **File sizes**: Reasonable sizes (papers < 10MB, videos < 100MB)
- [ ] **Permissions**: You have rights to share these materials

## 🚀 Upload Commands

### For Papers
```bash
# Copy paper files
cp your_paper.pdf papers/reflection_agent_paper_2024.pdf

# Add to git
git add papers/
git commit -m "Add research paper: Reflection Agent implementation"
git push origin main
```

### For Videos
```bash
# Copy video files
cp your_demo.mp4 demos/videos/comparison_demo.mp4

# Add to git
git add demos/
git commit -m "Add demo video: Agent comparison demonstration"
git push origin main
```

## 📞 Need Help?

If you encounter issues uploading files:
1. Check file sizes (GitHub has limits)
2. Verify file formats are supported
3. Open an issue for technical problems
4. Contact the maintainer for large files

## 🎯 Best Practices

1. **Keep files organized** in appropriate directories
2. **Use descriptive names** for easy identification
3. **Update documentation** when adding new files
4. **Check file permissions** before uploading
5. **Test links** after adding external references

# 🚀 MRT-43K: A Large-Scale Benchmark for Data-Centric Mars Terrain Classification

[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

Official implementation of the paper: **"MRT-43K: A Large-Scale Benchmark for Data-Centric Mars Terrain Classification"**. 

This repository provides access to the **MRT-43K dataset**, a large-scale benchmark containing **43,000 images** for Martian terrain classification, along with scripts for feature extraction, entropy-based data analysis, and model training.

---

## 📸 Overview

**MRT-43K** is one of the largest publicly available datasets for Martian terrain classification. Moving beyond traditional model-centric approaches, this project emphasizes **Data-Centric AI**. We utilize Self-Attention mechanisms and entropy calculation to evaluate data quality and uncertainty, significantly improving model robustness in extreme extraterrestrial environments.

---

## 📂 Dataset Access

The dataset includes raw images and pre-formatted CSV files for training, validation, and testing.

* **Download Link**: [Baidu Netdisk](https://pan.baidu.com/s/1eWOo-51w72GVeoSh8wF4Bw?pwd=sxun)
* **Access Code**: `sxun`

**Recommended Data Structure:**
```text
data/
├── images/             # Folder containing all .jpg / .png image files
├── train_data.csv      # Training metadata (columns: image_path, label)
└── val_data.csv        # Validation metadata
```
---

## ✍️ Citation
If you find the MRT-43K dataset or this codebase useful for your research, please cite our work:
```text


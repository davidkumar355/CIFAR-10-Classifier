<div align="center">
<img width="1904" height="902" alt="image" src="https://github.com/user-attachments/assets/a7d1998b-bb67-485a-8597-3d502743b698" />

# CIFAR-10 Image Classification: Custom CNN & Transfer Learning
*Benchmarking custom architectures against DenseNet121 and EfficientNetV2M.*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)



</div>

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Model Performance](#-model-performance)
- [Getting Started / Quickstart](#-getting-started--quickstart)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Contribution](#-contribution)
- [License & Acknowledgements](#-license--acknowledgements)

---

## 📖 Overview

This project tackles the CIFAR-10 image classification challenge by developing and comparing multiple Convolutional Neural Network (CNN) architectures. It establishes a baseline using a custom-built CNN and subsequently leverages transfer learning with **DenseNet121** and **EfficientNetV2M** to achieve up to **94.5% validation accuracy**. The pipeline features robust data augmentation, dynamic learning rate scheduling (Cosine Decay and ReduceLROnPlateau), and **Grad-CAM** heatmaps to interpret model predictions visually.

<div align="center">
  <br>
  <br>
</div>

---

## ✨ Features

* 🧠 **Architectural Benchmarking:** Compares a Custom CNN (2 blocks + Dense head) against ImageNet-pretrained DenseNet121 and EfficientNetV2M.
* 📈 **Advanced Training Pipeline:** Integrates `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, and `CosineDecay` for optimal convergence and overfitting prevention.
* 🔍 **Model Interpretability (XAI):** Implements Grad-CAM to generate activation heatmaps using OpenCV, explaining *why* the model makes specific predictions.
* 📐 **Dynamic Upsampling & Augmentation:** Uses native Keras layers (`RandomFlip`, `RandomRotation`, `UpSampling2D`) to scale 32x32 CIFAR images to 96x96 and 224x224 for transfer learning compatibility.

---

## 📊 Model Performance

| Architecture | Best Accuracy | Best Val Loss | Epochs to Converge |
|---|---|---|---|
| **Custom CNN** | ~81.8% | 0.5484 | 33 |
| **DenseNet121** | 94.4% | 0.2089 | 22 |
| **EfficientNetV2M** | **94.5%** | **0.6495** | 20 (Phase 2) |

*Note: The Custom CNN struggles with structural overlap (e.g., Cat vs. Dog), which is heavily mitigated by the deeper feature extraction in DenseNet and EfficientNet.*

---

## 🚀 Getting Started / Quickstart

### Prerequisites
* **Python** (v3.8 or higher)
* **TensorFlow** (v2.x with GPU support recommended)
* **Jupyter Notebook**

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/cifar10-cnn-transfer.git](https://github.com/yourusername/cifar10-cnn-transfer.git)
   cd cifar10-cnn-transfer
2. python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. pip install -r requirements.txt

### Usage
jupyter notebook


###  Project Structure
cifar10-cnn-transfer/
├── models/                 # Serialized model weights and histories
│   ├── best_customCNN_model.keras
│   ├── best_EfficientNetV2M_model.keras
│   └── densenet_history.pkl
├── notebooks/              
│   └── CIFAR_CNN_Model2.ipynb  # Core pipeline notebook 
├── .gitignore              
├── README.md               
└── requirements.txt        # numpy, pandas, matplotlib, tensorflow, opencv-python, scikit-learn

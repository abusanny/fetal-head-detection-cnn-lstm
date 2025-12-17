# Fetal Head Detection using CNN-LSTM

## Overview

This repository contains a deep learning model for detecting and classifying fetal heads in ultrasound images using a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The model utilizes VGG19 as a backbone for feature extraction and processes temporal sequences of ultrasound frames.

## Key Features

- **VGG19 CNN Backbone**: Pre-trained feature extraction for medical images
- **LSTM Temporal Modeling**: Captures temporal dependencies across ultrasound sequences
- **Binary Focal Loss**: Handles class imbalance in medical datasets
- **K-fold Cross-Validation**: Robust evaluation with multiple train-test splits
- **Comprehensive Preprocessing**: Image normalization, cropping, and padding
- **Batch Normalization & Regularization**: Prevents overfitting with dropout layers

## Dataset

- **Fetal Ultrasound Phantom Data**: 72 scans (5S, 10S, 15S, 20S, 25S, 30S configurations)
- **Total Images**: 24,036 images across all scans
- **Class Distribution**: 5,228 Head images, 18,833 Not-Head images (imbalanced)
- **Train/Val Split**: 9:2 ratio per scan
- **Image Size**: 224x224 pixels (resized)
- **Sequence Length**: 8 frames per temporal window

## Model Architecture

### Input Layer
- Temporal sequences: (Batch, 8, 224, 224, 3)

### Feature Extraction
- TimeDistributed VGG19 (frozen weights from ImageNet)
- Output: (Batch, 8, 512) feature vectors

### Temporal Modeling  
- LSTM Layer 1: 256 units, 40% recurrent dropout
- Batch Normalization, Dropout: 40%
- LSTM Layer 2: 128 units, 40% recurrent dropout
- Batch Normalization, Dropout: 40%

### Classification Head (Many-to-Many)
- TimeDistributed Dense Layer: 64 units (ReLU)
- Dropout: 30%
- TimeDistributed Output: 1 unit (Sigmoid) for binary classification

## Installation

```bash
git clone https://github.com/abusanny/fetal-head-detection-cnn-lstm.git
cd fetal-head-detection-cnn-lstm
pip install -r requirements.txt
```

## Training Configuration

- **Optimizer**: Adam (learning rate: 1e-5)
- **Loss Function**: Binary Focal Loss (gamma=2.0, alpha=0.79)
- **Metrics**: Accuracy, Precision, Recall, ROC-AUC, PR-AUC
- **Batch Size**: 8
- **Epochs**: 50
- **Early Stopping**: Patience=30-80 epochs
- **Total Trainable Parameters**: ~993,665

## Key Results

### Validation Performance (72 Scans)
- **Heads**: 4,088 sequences
- **Not-Heads**: 14,919 sequences  
- **Total Sequences**: 4,525
- **Accuracy**: ~90.6%
- **ROC-AUC**: Strong discrimination capability

## Technologies Used

- **Deep Learning**: TensorFlow 2.13+, Keras
- **Data Processing**: NumPy, Pandas, OpenCV
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Python 3.10+, WSL2 with NVIDIA GPU

## Author

**Abu Sanny**  
Research Assistant at Utsaah Lab  
Specializing in Medical Image Analysis and Deep Learning

## License

This project is open source and available under the MIT License.

## Citation

If you use this code in your research:

```bibtex
@github{abusanny2025fetal,
  author = {Abu Sanny},
  title = {Fetal Head Detection using CNN-LSTM},
  year = {2025},
  url = {https://github.com/abusanny/fetal-head-detection-cnn-lstm}
}
```

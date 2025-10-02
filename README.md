🦷 **Automated classification system for dental caries using ConvNeXt and YOLO11 ensemble**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This project implements an automated dental caries classification system using an ensemble of Convolutional Neural Networks (CNNs). The system combines two state-of-the-art architectures:

- **ConvNeXt Base**: Specialized for fine-grained image classification
- **YOLO11**: Adapted for classification with enhanced feature extraction

### Key Results

- **Ensemble Accuracy**: 90.3%
- **ConvNeXt Individual**: 81.9%
- **YOLO11 Individual**: 83.3%
- **Improvement**: +7% over best individual model

## 🎯 Features

- ✅ Five-class dental caries classification (bc, c4, c5, c6, hg)
- ✅ Multiple ensemble methods (Weighted Average, Max Voting, Geometric Mean, Harmonic Mean)
- ✅ Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score)
- ✅ Detailed visualization tools (Confusion matrices, radar charts, performance graphs)
- ✅ Production-ready inference pipeline

## 🏗️ Architecture

### Dataset Classes

| Class | Description | Severity |
|-------|-------------|----------|
| bc | Initial caries (ICDAS 1-3) | Early stage |
| c4 | Dark shadow in dentin (ICDAS 4) | Moderate |
| c5 | Exposed dentin cavity <50% (ICDAS 5) | Advanced |
| c6 | Exposed dentin cavity >50% (ICDAS 6) | Severe |
| hg | Healthy tooth (ICDAS 0) | No caries |

### Ensemble Methods

1. **Weighted Average**: 60% ConvNeXt + 40% YOLO11
2. **Max Voting**: Highest confidence selection
3. **Geometric Mean**: Multiplicative combination
4. **Harmonic Mean**: Conservative consensus (Best: 90.3%)

## 📊 Results

### Overall Performance

| Method | Accuracy | Improvement |
|--------|----------|-------------|
| ConvNeXt | 81.9% | Baseline |
| YOLO11 | 83.3% | +1.4% |
| Weighted Average | 86.1% | +4.2% |
| Max Voting | 88.9% | +7.0% |
| Geometric Mean | 88.9% | +7.0% |
| **Harmonic Mean** | **90.3%** | **+8.4%** |

### Per-Class Performance (Harmonic Mean)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| bc | 0.89 | 0.88 | 0.88 |
| c4 | 1.00 | 0.92 | 0.96 |
| c5 | 0.92 | 1.00 | 0.96 |
| c6 | 0.88 | 1.00 | 0.93 |
| hg | 0.92 | 0.83 | 0.87 |

 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Advisor: Prof. Lucas Ferrari de Oliveira for guidance and support
UFPR: Universidade Federal do Paraná for providing resources
Community: PyTorch and Ultralytics teams for excellent frameworks

📧 Contact
Alex Matsuo

GitHub: @alexmatsuo
LinkedIn: https://www.linkedin.com/in/alex-matsuo/
Email: gmalexmatsuo@gmail.com

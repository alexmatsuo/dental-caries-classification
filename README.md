# Dental Caries Classification Using Deep Learning Ensemble

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

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/alexmatsuo/dental-caries-classification.git
cd dental-caries-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Train ConvNeXt Model

```bash
python src/convnext.py
```

#### 2. Train YOLO11 Model

```bash
python src/yolo.py
```

#### 3. Evaluate Individual Models

```bash
# Evaluate ConvNeXt
python src/convnexteval.py
```

#### 4. Run Ensemble Evaluation

```bash
python src/eval.py
```

#### 5. Predict Single Image

```python
from src.eval import DentalCariesEnsemble

# Initialize ensemble
ensemble = DentalCariesEnsemble(
    convnext_path='best_convnext.pth',
    yolo_path='best.pt'
)

# Make prediction
result = ensemble.ensemble_predict(
    'path/to/image.jpg',
    method='harmonic_mean'
)

print(f"Predicted class: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📁 Project Structure

```
dental-caries-classification/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── docs/
│   ├── TCC_Alex_Matsuo.pdf
│   └── images/
├── src/
│   ├── convnext.py
│   ├── yolo.py
│   ├── convnexteval.py
│   ├── eval.py
│   └── en3.py
├── models/
│   └── README.md
├── dataset/
│   └── README.md
└── results/
    ├── confusion_matrices/
    ├── performance_graphs/
    └── sample_predictions/
```

## 📁 Dataset Structure

```
dataset/
├── train/
│   ├── bc/
│   ├── c4/
│   ├── c5/
│   ├── c6/
│   └── hg/
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

**Dataset Statistics:**
- Total images: 600
- Training: 75% (450 images)
- Validation: 13% (78 images)
- Test: 12% (72 images)
- Image size: 224×224 pixels

## 🔬 Methodology

### Data Preprocessing

- **Resize**: 256×256 → Center crop 224×224
- **Normalization**: ImageNet mean/std
- **Augmentation** (training only):
  - Random rotation (±15°)
  - Random horizontal flip
  - Random resized crop
  - HSV jittering

### Training Configuration

#### ConvNeXt
- **Optimizer**: AdamW (lr=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Epochs**: 20
- **Batch size**: 16
- **Loss**: CrossEntropyLoss

#### YOLO11
- **Optimizer**: AdamW (lr=1e-3)
- **Scheduler**: Cosine with warmup
- **Epochs**: 50
- **Batch size**: 32
- **Augmentation**: Extensive (mosaic, mixup, HSV)

### Ensemble Strategy

The Harmonic Mean method achieved the best results by:
1. Requiring strong consensus between models
2. Penalizing large disagreements
3. Maintaining high confidence only when both models agree

**Formula:**
```
HarmonicMean = 2 × (p_convnext × p_yolo) / (p_convnext + p_yolo + ε)
```

## 📈 Visualizations

All generated visualizations are available in the `results/` directory:

- Confusion matrices (validation and test sets)
- Per-class accuracy comparisons
- Precision-Recall-F1 radar charts
- Sample predictions with confidence scores
- Model performance summary

## 🎓 Academic Context

This project was developed as a Bachelor's thesis (TCC) in Computer Science at:

**Universidade Federal do Paraná (UFPR)**
- **Student**: Alex Matsuo
- **Advisor**: Prof. Lucas Ferrari de Oliveira
- **Year**: 2025
- **Department**: Departamento de Informática (DInf)

### Abstract

This work presents an automated system for dental caries classification using deep learning ensemble techniques. By combining ConvNeXt and YOLO11 architectures, the system achieves 90.3% accuracy across five caries categories, demonstrating the potential of hybrid AI systems in dental diagnosis assistance.

**Full thesis**: [docs/TCC_Alex_Matsuo.pdf](docs/TCC_Alex_Matsuo.pdf)

## 📚 References

### Key Papers

1. **ConvNeXt**: Liu et al. (2022). "A ConvNet for the 2020s"
2. **YOLO11**: Khanam & Hussain (2024). "YOLOv11: An Overview"
3. **Ensemble Learning**: Dietterich (2000). "Ensemble Methods in Machine Learning"
4. **Dental AI**: Schwendicke et al. (2019). "CNNs for Dental Image Diagnostics"

### Citation

If you use this work, please cite:

```bibtex
@thesis{matsuo2025dental,
  title={Classificação de Cáries Dentárias Utilizando Ensemble de Redes Neurais Convolucionais},
  author={Matsuo, Alex},
  year={2025},
  school={Universidade Federal do Paraná},
  type={Bachelor's Thesis}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Advisor**: Prof. Lucas Ferrari de Oliveira for guidance and support
- **UFPR**: Universidade Federal do Paraná for providing resources
- **Community**: PyTorch and Ultralytics teams for excellent frameworks

## 📧 Contact

**Alex Matsuo**
- GitHub: [@alexmatsuo](https://github.com/alexmatsuo)
- LinkedIn: [linkedin.com/in/alex-matsuo](https://www.linkedin.com/in/alex-matsuo/)
- Email: gmalexmatsuo@gmail.com

---

⭐ If you find this project useful, please consider giving it a star!


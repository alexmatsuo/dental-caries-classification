import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configuration
DATA_DIR = 'dataset'
BATCH_SIZE = 16
NUM_CLASSES = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'best_convnext.pth'

# Transform (same as validation in your training script)
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, 'val'),
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, 'test'),
    transform=eval_transform
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = val_dataset.classes

# Initialize model
model = timm.create_model('convnext_base', pretrained=False, num_classes=NUM_CLASSES)
model = model.to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def evaluate_with_metrics(loader, dataset_name):
    """Enhanced evaluation function that collects detailed metrics"""
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f'Evaluating {dataset_name}'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Per-class accuracy calculation
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracy = 100 * correct / total
    print(f'\n{dataset_name} Overall Accuracy: {accuracy:.2f}%')
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'class_correct': class_correct,
        'class_total': class_total
    }

def create_individual_graphs(val_results, test_results):
    """Create individual graphs saved as separate files"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Calculate per-class accuracies
    val_class_acc = [100 * val_results['class_correct'][i] / val_results['class_total'][i] 
                     if val_results['class_total'][i] > 0 else 0 for i in range(NUM_CLASSES)]
    test_class_acc = [100 * test_results['class_correct'][i] / test_results['class_total'][i] 
                      if test_results['class_total'][i] > 0 else 0 for i in range(NUM_CLASSES)]
    
    # Calculate precision, recall, F1-score with proper label handling
    # First, let's check which classes are actually present in the test set
    unique_labels = np.unique(test_results['labels'])
    print(f"\nUnique labels in test set: {unique_labels}")
    print(f"Class names: {class_names}")
    
    # Calculate metrics ensuring all classes are included
    precision, recall, f1, support = precision_recall_fscore_support(
        test_results['labels'], 
        test_results['predictions'], 
        labels=list(range(NUM_CLASSES)),  # Explicitly specify all class labels
        average=None,
        zero_division=0  # Handle classes with no predictions
    )
    
    # Debug print
    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")
    
    # 1. Overall Accuracy Comparison
    plt.figure(figsize=(8, 6))
    accuracies = [val_results['accuracy'], test_results['accuracy']]
    datasets = ['Validation', 'Test']
    colors = ['#3498db', '#e74c3c']
    
    plt.bar(datasets, accuracies, color=colors, alpha=0.8)
    plt.title('Overall Accuracy Comparison', fontweight='bold', fontsize=14)
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convnext_overall_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-Class Accuracy Comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, val_class_acc, width, label='Validation', alpha=0.8)
    plt.bar(x + width/2, test_class_acc, width, label='Test', alpha=0.8)
    plt.title('Per-Class Accuracy Comparison', fontweight='bold', fontsize=14)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Classes')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convnext_per_class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Sample Distribution
    plt.figure(figsize=(10, 6))
    val_samples = [val_results['class_total'][i] for i in range(NUM_CLASSES)]
    test_samples = [test_results['class_total'][i] for i in range(NUM_CLASSES)]
    
    plt.bar(x - width/2, val_samples, width, label='Validation', alpha=0.8)
    plt.bar(x + width/2, test_samples, width, label='Test', alpha=0.8)
    plt.title('Sample Distribution by Class', fontweight='bold', fontsize=14)
    plt.ylabel('Number of Samples')
    plt.xlabel('Classes')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i in range(len(class_names)):
        plt.text(i - width/2, val_samples[i] + 0.5, str(val_samples[i]), ha='center', va='bottom')
        plt.text(i + width/2, test_samples[i] + 0.5, str(test_samples[i]), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('convnext_sample_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Precision, Recall, F1-Score for Test Set (FIXED)
    plt.figure(figsize=(12, 8))
    x = np.arange(len(class_names))
    width = 0.25
    
    # Create bars
    bars1 = plt.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#3498db')
    bars2 = plt.bar(x, recall, width, label='Recall', alpha=0.8, color='#e74c3c')
    bars3 = plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='#2ecc71')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only label non-zero values
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.title('Test Set: Precision, Recall, F1-Score by Class', fontweight='bold', fontsize=14)
    plt.ylabel('Score')
    plt.xlabel('Classes')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.15)  # Give some space for labels
    
    # Add support numbers below x-axis
    for i, s in enumerate(support):
        plt.text(i, -0.1, f'n={int(s)}', ha='center', transform=plt.gca().get_xaxis_transform(), fontsize=9)
    
    plt.tight_layout()
    plt.savefig('convnext_precision_recall_f1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Test Set Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(test_results['labels'], test_results['predictions'])
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotation text with both count and percentage
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Test Set Confusion Matrix', fontweight='bold', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('convnext_test_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Model Performance Summary
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    
    # Calculate macro and weighted averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    summary_text = f"""ConvNeXt Model Performance Summary

Validation Accuracy: {val_results['accuracy']:.2f}%
Test Accuracy: {test_results['accuracy']:.2f}%

Best Performing Class:
{class_names[np.argmax(test_class_acc)]} ({max(test_class_acc):.1f}%)

Most Challenging Class:
{class_names[np.argmin(test_class_acc)]} ({min(test_class_acc):.1f}%)

Macro Averages:
  Precision: {macro_precision:.3f}
  Recall: {macro_recall:.3f}
  F1-Score: {macro_f1:.3f}

Weighted Averages:
  Precision: {weighted_precision:.3f}
  Recall: {weighted_recall:.3f}
  F1-Score: {weighted_f1:.3f}

Total Test Samples: {len(test_results['labels'])}
Total Validation Samples: {len(val_results['labels'])}

Per-Class Test Performance:
{chr(10).join([f"  {class_names[i]}: Acc={test_class_acc[i]:.1f}%, P={precision[i]:.2f}, R={recall[i]:.2f}, F1={f1[i]:.2f}, n={int(support[i])}" for i in range(len(class_names))])}
"""
    
    plt.text(0.1, 0.95, summary_text, transform=plt.gca().transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    plt.title('ConvNeXt Model Performance Summary', fontweight='bold', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('convnext_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate detailed confusion matrices
    create_detailed_confusion_matrices(val_results, test_results)
    
    # Create class-wise performance radar chart
    create_radar_chart(precision, recall, f1, class_names)
    
    # Create additional diagnostic plots
    create_diagnostic_plots(test_results, precision, recall, f1, support, class_names)

def create_detailed_confusion_matrices(val_results, test_results):
    """Create detailed confusion matrices as separate files"""
    # Validation confusion matrix
    plt.figure(figsize=(10, 8))
    cm_val = confusion_matrix(val_results['labels'], val_results['predictions'])
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Validation Set Confusion Matrix', fontweight='bold', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('convnext_validation_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_radar_chart(precision, recall, f1, class_names):
    """Create a radar chart for class-wise performance metrics"""
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    
    # Number of variables
    N = len(class_names)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add values
    precision_plot = list(precision) + [precision[0]]
    recall_plot = list(recall) + [recall[0]]
    f1_plot = list(f1) + [f1[0]]
    
    # Plot
    ax.plot(angles, precision_plot, 'o-', linewidth=2, label='Precision', alpha=0.8)
    ax.fill(angles, precision_plot, alpha=0.25)
    ax.plot(angles, recall_plot, 'o-', linewidth=2, label='Recall', alpha=0.8)
    ax.fill(angles, recall_plot, alpha=0.25)
    ax.plot(angles, f1_plot, 'o-', linewidth=2, label='F1-Score', alpha=0.8)
    ax.fill(angles, f1_plot, alpha=0.25)
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1)
    ax.set_title('Class-wise Performance Metrics (Test Set)', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('convnext_performance_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_diagnostic_plots(test_results, precision, recall, f1, support, class_names):
    """Create additional diagnostic plots to verify metrics"""
    
    # 1. Metrics comparison table
    plt.figure(figsize=(10, 6))
    plt.axis('tight')
    plt.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Class', 'Support', 'Precision', 'Recall', 'F1-Score', 'Accuracy']
    
    for i, class_name in enumerate(class_names):
        class_acc = 100 * test_results['class_correct'][i] / test_results['class_total'][i] if test_results['class_total'][i] > 0 else 0
        row = [
            class_name,
            f"{int(support[i])}",
            f"{precision[i]:.3f}",
            f"{recall[i]:.3f}",
            f"{f1[i]:.3f}",
            f"{class_acc:.1f}%"
        ]
        table_data.append(row)
    
    # Add summary row
    table_data.append(['', '', '', '', '', ''])
    table_data.append([
        'Macro Avg',
        f"{int(np.sum(support))}",
        f"{np.mean(precision):.3f}",
        f"{np.mean(recall):.3f}",
        f"{np.mean(f1):.3f}",
        f"{test_results['accuracy']:.1f}%"
    ])
    
    table = plt.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style the macro avg row
    for i in range(len(headers)):
        table[(len(table_data), i)].set_facecolor('#e8f4f8')
        table[(len(table_data), i)].set_text_props(weight='bold')
    
    plt.title('Detailed Performance Metrics by Class', fontweight='bold', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('convnext_metrics_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_predictions(loader, dataset_name, num_samples=8):
    """Visualize sample predictions"""
    images, labels = next(iter(loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    _, preds = torch.max(outputs, 1)

    # Denormalize for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    images = inv_normalize(images)

    plt.figure(figsize=(16, 8))
    for i in range(min(num_samples, len(images))):
        ax = plt.subplot(2, 4, i+1)
        ax.axis('off')
        
        # Get confidence
        confidence = probs[i][preds[i]].item()
        correct = preds[i] == labels[i]
        
        title_color = 'green' if correct else 'red'
        ax.set_title(f'Pred: {class_names[preds[i]]} ({confidence:.2f})\nTrue: {class_names[labels[i]]}',
                    color=title_color, fontweight='bold')
        
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        plt.imshow(img)
    
    plt.suptitle(f'ConvNeXt Sample Predictions - {dataset_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'convnext_sample_predictions_{dataset_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main execution
if __name__ == "__main__":
    print("Starting ConvNeXt Model Evaluation with Comprehensive Graphs...")
    print(f"Using device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    
    # Evaluate both datasets
    val_results = evaluate_with_metrics(val_loader, 'Validation')
    test_results = evaluate_with_metrics(test_loader, 'Test')
    
    # Print classification reports
    print("\n" + "="*50)
    print("VALIDATION SET CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(val_results['labels'], val_results['predictions'], 
                              target_names=class_names, zero_division=0))
    
    print("\n" + "="*50)
    print("TEST SET CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(test_results['labels'], test_results['predictions'], 
                              target_names=class_names, zero_division=0))
    
    # Create all individual graphs
    create_individual_graphs(val_results, test_results)
    
    # Visualize sample predictions
    visualize_predictions(val_loader, 'Validation')
    visualize_predictions(test_loader, 'Test')
    
    print("\nEvaluation complete! Generated individual graph files:")
    print("- convnext_overall_accuracy.png")
    print("- convnext_per_class_accuracy.png")
    print("- convnext_sample_distribution.png")
    print("- convnext_precision_recall_f1.png")
    print("- convnext_test_confusion_matrix.png")
    print("- convnext_performance_summary.png")
    print("- convnext_validation_confusion_matrix.png")
    print("- convnext_performance_radar_chart.png")
    print("- convnext_metrics_table.png")
    print("- convnext_sample_predictions_validation.png")
    print("- convnext_sample_predictions_test.png")

# YOLO11 Dental Caries Classification - Working Version
# Uses direct dataset path instead of YAML

import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import shutil

# Configuration
CLASSES = ["bc", "c4", "c5", "c6", "hg"]
MODEL = "yolo11n-cls.pt"
EPOCHS = 50
BATCH_SIZE = 32

# Device configuration
if torch.cuda.is_available():
    DEVICE = 0
    DEVICE_NAME = f"GPU - {torch.cuda.get_device_name(0)}"
else:
    DEVICE = 'cpu'
    DEVICE_NAME = 'CPU'

def verify_dataset():
    """Verify dataset structure"""
    print(f"\n{'='*60}")
    print("Verifying Dataset Structure")
    print(f"{'='*60}")
    
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    print(f"Dataset path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder not found at {dataset_path}")
        return False
    
    # Check each split
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"❌ {split} folder not found")
            return False
        
        # Count images per class
        total_images = 0
        print(f"\n{split.upper()}:")
        for class_name in CLASSES:
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"  {class_name}: {len(images)} images")
                total_images += len(images)
            else:
                print(f"  {class_name}: NOT FOUND")
        print(f"  Total: {total_images} images")
    
    print(f"{'='*60}\n")
    return True

def train_yolo11():
    """Train YOLO11 using direct dataset path"""
    print(f"\n{'='*60}")
    print("Starting YOLO11 Training")
    print(f"{'='*60}")
    print(f"Device: {DEVICE_NAME}")
    print(f"Model: {MODEL}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"{'='*60}\n")
    
    # Get absolute path to dataset
    current_dir = os.getcwd()
    dataset_path = os.path.join(current_dir, 'dataset')
    
    # Initialize model
    model = YOLO(MODEL)
    
    # Train using the dataset directory directly
    try:
        results = model.train(
            data=dataset_path,  # Pass the dataset directory directly
            epochs=EPOCHS,
            imgsz=224,
            batch=BATCH_SIZE,
            device=DEVICE,
            project='dental-caries-yolo11',
            name='run',
            exist_ok=True,
            patience=20,
            save=True,
            verbose=True,
            pretrained=True,
            
            # Optimizer
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15,
            translate=0.1,
            scale=0.2,
            shear=5,
            perspective=0.0003,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.5,
            mixup=0.2,
            
            # Training settings
            dropout=0.3,
            val=True,
            amp=True,
            cos_lr=True,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            close_mosaic=10,
            
            # Workers
            workers=8 if DEVICE != 'cpu' else 4
        )
        
        return model, results
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        print("\nTrying alternative approach...")
        
        # Alternative: Create a temporary YAML in a different location
        temp_yaml_path = os.path.join(current_dir, 'temp_data_config.yaml')
        
        yaml_content = f"""path: {dataset_path}
train: train
val: val
test: test
nc: 6
names: ['c2', 'c3', 'c4', 'c5', 'c6', 'hg']
"""
        
        with open(temp_yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created temporary config at: {temp_yaml_path}")
        
        try:
            results = model.train(
                data=temp_yaml_path,
                epochs=EPOCHS,
                imgsz=224,
                batch=BATCH_SIZE,
                device=DEVICE,
                project='dental-caries-yolo11',
                name='run',
                exist_ok=True,
                patience=20,
                save=True,
                verbose=True,
                pretrained=True,
                optimizer='AdamW',
                lr0=0.001,
                lrf=0.01,
                dropout=0.3,
                val=True,
                amp=True,
                workers=8 if DEVICE != 'cpu' else 4
            )
            
            # Clean up temp file
            if os.path.exists(temp_yaml_path):
                os.remove(temp_yaml_path)
            
            return model, results
            
        except Exception as e2:
            print(f"❌ Alternative approach also failed: {str(e2)}")
            return None, None

def evaluate_model(model_path=None):
    """Evaluate the trained model"""
    if model_path is None:
        # Try to find the best model
        possible_paths = [
            'dental-caries-yolo11/run/weights/best.pt',
            'dental-caries-yolo11/run2/weights/best.pt',
            'dental-caries-yolo11/run3/weights/best.pt',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("❌ No trained model found")
            return
    
    print(f"\n{'='*60}")
    print(f"Evaluating Model: {model_path}")
    print(f"{'='*60}\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Test on test set
    test_dir = os.path.join('dataset', 'test')
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Running predictions on test set...")
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in tqdm(images, desc=f'{class_name}'):
            img_path = os.path.join(class_dir, img_name)
            
            # Predict
            results = model.predict(img_path, verbose=False)
            
            if results and len(results) > 0:
                probs = results[0].probs
                if probs is not None:
                    pred_class = probs.top1
                    all_probs.append(probs.data.cpu().numpy())
                    all_preds.append(pred_class)
                    all_labels.append(class_idx)
    
    if len(all_preds) == 0:
        print("❌ No predictions were made")
        return
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = np.mean(all_preds == all_labels)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(CLASSES):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = np.mean(all_preds[mask] == all_labels[mask])
            print(f"  {class_name}: {class_acc*100:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix - Test Set\nAccuracy: {accuracy*100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    # Save confusion matrix
    save_path = 'confusion_matrix_test.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to: {save_path}")
    plt.close()
    
    # Save results to text file
    with open('test_results.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy*100:.2f}%\n\n")
        f.write("Per-class Accuracy:\n")
        for i, class_name in enumerate(CLASSES):
            mask = all_labels == i
            if mask.sum() > 0:
                class_acc = np.mean(all_preds[mask] == all_labels[mask])
                f.write(f"  {class_name}: {class_acc*100:.2f}%\n")
        f.write("\n" + classification_report(all_labels, all_preds, target_names=CLASSES))
    
    print("✅ Results saved to: test_results.txt")

def predict_image(image_path, model_path=None):
    """Make prediction on a single image"""
    if model_path is None:
        model_path = 'dental-caries-yolo11/run/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found at {image_path}")
        return
    
    model = YOLO(model_path)
    results = model.predict(image_path, verbose=False)
    
    if results and len(results) > 0:
        probs = results[0].probs
        if probs is not None:
            print(f"\nPrediction for: {image_path}")
            print("-" * 50)
            
            # Top prediction
            top1_idx = probs.top1
            top1_conf = probs.data[top1_idx].item()
            
            print(f"Predicted class: {CLASSES[top1_idx]}")
            print(f"Confidence: {top1_conf:.2%}")
            
            # All probabilities
            print("\nAll probabilities:")
            probs_numpy = probs.data.cpu().numpy()
            for idx, class_name in enumerate(CLASSES):
                prob = probs_numpy[idx]
                bar = '█' * int(prob * 30)
                print(f"  {class_name}: {prob:6.2%} {bar}")

def main():
    """Main execution"""
    print("YOLO11 Dental Caries Classification")
    print(f"Working directory: {os.getcwd()}")
    
    # Step 1: Verify dataset
    if not verify_dataset():
        print("\n❌ Please ensure your dataset is in the 'dataset' folder with proper structure:")
        print("dataset/")
        print("├── train/")
        print("│   ├── bc/")
        print("│   ├── c4/")
        print("│   ├── c5/")
        print("│   ├── c6/")
        print("│   └── hg/")
        print("├── val/")
        print("└── test/")
        return
    
    # Step 2: Train model
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    model, results = train_yolo11()
    
    if model is None:
        print("\n❌ Training failed. Please check the error messages above.")
        return
    
    print("\n✅ Training completed successfully!")
    
    # Step 3: Evaluate model
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    evaluate_model()
    
    print("\n" + "="*60)
    print("✅ All tasks completed!")
    print("="*60)

if __name__ == "__main__":
    main()

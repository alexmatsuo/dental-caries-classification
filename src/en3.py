import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from ultralytics import YOLO
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import json

class DentalCariesEnsemble:
    """
    Ensemble model combining ConvNeXt and YOLO11 for dental caries classification
    """
    
    def __init__(self, convnext_path, yolo_path, num_classes=5, device='cuda'):
        """
        Initialize the ensemble model
        
        Args:
            convnext_path: Path to trained ConvNeXt model weights
            yolo_path: Path to trained YOLO11 model
            num_classes: Number of classes (5 for dental caries)
            device: Device to run inference on
        """
        self.num_classes = num_classes
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.classes = ["bc", "c4", "c5", "c6", "hg"]
        
        # Load ConvNeXt model
        self.convnext_model = self._load_convnext(convnext_path)
        
        # Load YOLO11 model
        self.yolo_model = self._load_yolo(yolo_path)
        
        # Image preprocessing for ConvNeXt
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Ensemble weights (can be learned or manually set)
        self.ensemble_weights = {
            'convnext': 0.6,  # ConvNeXt typically stronger for fine-grained classification
            'yolo': 0.4       # YOLO good for feature extraction and object detection
        }
        
        # Ensemble methods
        self.ensemble_methods = [
            'weighted_average',
            'max_voting',
            'geometric_mean',
            'harmonic_mean',
            'learned_weights'
        ]
    
    def _load_convnext(self, model_path):
        """Load the trained ConvNeXt model"""
        try:
            model = timm.create_model('convnext_base', pretrained=False, num_classes=self.num_classes)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            print(f"✅ ConvNeXt model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"❌ Error loading ConvNeXt model: {e}")
            return None
    
    def _load_yolo(self, model_path):
        """Load the trained YOLO11 model"""
        try:
            model = YOLO(model_path)
            print(f"✅ YOLO11 model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"❌ Error loading YOLO11 model: {e}")
            return None
    
    def predict_convnext(self, image_path):
        """Get prediction from ConvNeXt model"""
        if self.convnext_model is None:
            return None
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.convnext_model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
            return probabilities.cpu().numpy()[0]
        except Exception as e:
            print(f"Error in ConvNeXt prediction: {e}")
            return None
    
    def predict_yolo(self, image_path):
        """Get prediction from YOLO11 model"""
        if self.yolo_model is None:
            return None
        
        try:
            # Get prediction
            results = self.yolo_model.predict(image_path, verbose=False)
            
            if results and len(results) > 0:
                probs = results[0].probs
                if probs is not None:
                    return probs.data.cpu().numpy()
            
            return None
        except Exception as e:
            print(f"Error in YOLO prediction: {e}")
            return None
    
    def ensemble_predict(self, image_path, method='weighted_average'):
        """
        Make ensemble prediction using specified method
        
        Args:
            image_path: Path to input image
            method: Ensemble method to use
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Get individual model predictions
        convnext_probs = self.predict_convnext(image_path)
        yolo_probs = self.predict_yolo(image_path)
        
        if convnext_probs is None and yolo_probs is None:
            return None
        
        # Handle case where only one model works
        if convnext_probs is None:
            ensemble_probs = yolo_probs
            method_used = "yolo_only"
        elif yolo_probs is None:
            ensemble_probs = convnext_probs
            method_used = "convnext_only"
        else:
            # Apply ensemble method
            if method == 'weighted_average':
                ensemble_probs = (self.ensemble_weights['convnext'] * convnext_probs + 
                                self.ensemble_weights['yolo'] * yolo_probs)
            elif method == 'max_voting':
                # Take the class with highest confidence from either model
                convnext_pred = np.argmax(convnext_probs)
                yolo_pred = np.argmax(yolo_probs)
                if convnext_probs[convnext_pred] >= yolo_probs[yolo_pred]:
                    ensemble_probs = convnext_probs
                else:
                    ensemble_probs = yolo_probs
            elif method == 'geometric_mean':
                ensemble_probs = np.sqrt(convnext_probs * yolo_probs)
                ensemble_probs = ensemble_probs / np.sum(ensemble_probs)  # Normalize
            elif method == 'harmonic_mean':
                # Avoid division by zero
                epsilon = 1e-8
                ensemble_probs = 2 * (convnext_probs * yolo_probs) / (convnext_probs + yolo_probs + epsilon)
                ensemble_probs = ensemble_probs / np.sum(ensemble_probs)  # Normalize
            else:
                # Default to weighted average
                ensemble_probs = (self.ensemble_weights['convnext'] * convnext_probs + 
                                self.ensemble_weights['yolo'] * yolo_probs)
            
            method_used = method
        
        # Get prediction
        predicted_class = np.argmax(ensemble_probs)
        confidence = ensemble_probs[predicted_class]
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': self.classes[predicted_class],
            'confidence': confidence,
            'all_probabilities': ensemble_probs,
            'convnext_probs': convnext_probs,
            'yolo_probs': yolo_probs,
            'method_used': method_used
        }
    
    def evaluate_ensemble(self, test_dir, methods=['weighted_average']):
        results = {}
        
        # Collect all test images
        test_images = []
        true_labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(test_dir, class_name)
            if os.path.exists(class_dir):
                images = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for img_name in images:
                    img_path = os.path.join(class_dir, img_name)
                    test_images.append(img_path)
                    true_labels.append(class_idx)
        
        print(f"Found {len(test_images)} test images")
        
        # ===== Evaluate ConvNeXt Individually =====
        print("\nEvaluating ConvNeXt individually...")
        convnext_predictions = []
        convnext_probs = []
        
        for img_path in tqdm(test_images, desc='Testing ConvNeXt'):
            probs = self.predict_convnext(img_path)
            if probs is not None:
                convnext_predictions.append(np.argmax(probs))
                convnext_probs.append(probs)
            else:
                convnext_predictions.append(0)  # Default to class 0 if prediction fails
                convnext_probs.append(np.ones(self.num_classes) / self.num_classes)
        
        convnext_accuracy = accuracy_score(true_labels, convnext_predictions)
        print(f"ConvNeXt Accuracy: {convnext_accuracy:.4f}")
        
        # Store ConvNeXt results
        results['convnext'] = {
            'accuracy': convnext_accuracy,
            'predictions': convnext_predictions,
            'true_labels': true_labels,
            'probabilities': convnext_probs
        }
        
        # ===== Evaluate YOLO Individually =====
        print("\nEvaluating YOLO individually...")
        yolo_predictions = []
        yolo_probs = []
        
        for img_path in tqdm(test_images, desc='Testing YOLO'):
            probs = self.predict_yolo(img_path)
            if probs is not None:
                yolo_predictions.append(np.argmax(probs))
                yolo_probs.append(probs)
            else:
                yolo_predictions.append(0)  # Default to class 0 if prediction fails
                yolo_probs.append(np.ones(self.num_classes) / self.num_classes)
        
        yolo_accuracy = accuracy_score(true_labels, yolo_predictions)
        print(f"YOLO Accuracy: {yolo_accuracy:.4f}")
        
        # Store YOLO results
        results['yolo'] = {
            'accuracy': yolo_accuracy,
            'predictions': yolo_predictions,
            'true_labels': true_labels,
            'probabilities': yolo_probs
        }
        
        # ===== Evaluate Ensemble Methods =====
        for method in methods:
            print(f"\nEvaluating ensemble method: {method}")
            
            predictions = []
            all_probs = []
            individual_results = []
            
            for img_path in tqdm(test_images, desc=f'Testing {method}'):
                result = self.ensemble_predict(img_path, method=method)
                
                if result is not None:
                    predictions.append(result['predicted_class'])
                    all_probs.append(result['all_probabilities'])
                    individual_results.append(result)
                else:
                    predictions.append(0)
                    all_probs.append(np.ones(self.num_classes) / self.num_classes)
            
            accuracy = accuracy_score(true_labels, predictions)
            
            # Per-class accuracy
            per_class_acc = {}
            for i, class_name in enumerate(self.classes):
                mask = np.array(true_labels) == i
                if mask.sum() > 0:
                    class_acc = np.mean(np.array(predictions)[mask] == np.array(true_labels)[mask])
                    per_class_acc[class_name] = class_acc
            
            results[method] = {
                'accuracy': accuracy,
                'per_class_accuracy': per_class_acc,
                'predictions': predictions,
                'true_labels': true_labels,
                'probabilities': all_probs,
                'individual_results': individual_results
            }
            
            print(f"Accuracy: {accuracy:.4f}")
        
        return results
    
    def compare_methods(self, test_dir):
        """Compare all ensemble methods and individual models"""
        print("Comparing all methods...")
        
        results = self.evaluate_ensemble(test_dir, self.ensemble_methods[:4])  # Exclude learned_weights
        
        # Create comparison DataFrame
        comparison_data = []
        
        # Add ConvNeXt and YOLO results
        for model in ['convnext', 'yolo']:
            row = {
                'Method': model.upper(),
                'Overall_Accuracy': results[model]['accuracy']
            }
            comparison_data.append(row)
        
        # Add ensemble methods
        for method, result in results.items():
            if method not in ['convnext', 'yolo']:  # Skip individual models (already added)
                row = {
                    'Method': method.replace('_', ' ').title(),
                    'Overall_Accuracy': result['accuracy']
                }
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display results
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(comparison_df.round(4))
        
        # Find best method
        best_method = comparison_df.loc[comparison_df['Overall_Accuracy'].idxmax(), 'Method']
        best_accuracy = comparison_df['Overall_Accuracy'].max()
        
        print(f"\n🏆 Best Method: {best_method} (Accuracy: {best_accuracy:.4f})")
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        
        # Overall accuracy comparison
        methods = comparison_df['Method']
        accuracies = comparison_df['Overall_Accuracy']
        
        bars = plt.bar(methods, accuracies, color=['red', 'blue', 'skyblue', 'lightgreen', 'gold'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return results, best_method
    
    def create_confusion_matrices(self, results, save_dir='ensemble_results'):
        """Create confusion matrices for all methods and individual models"""
        os.makedirs(save_dir, exist_ok=True)
        
        for method, result in results.items():
            plt.figure(figsize=(8, 6))
            
            cm = confusion_matrix(result['true_labels'], result['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes, yticklabels=self.classes)
            plt.title(f'Confusion Matrix - {method.upper()}\n'
                    f'Accuracy: {result["accuracy"]:.3f}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            save_path = os.path.join(save_dir, f'confusion_matrix_{method}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Confusion matrix saved: {save_path}")
    
    def save_results(self, results, filename='ensemble_results.json'):
        """Save evaluation results to JSON file with proper type conversion"""
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj

        json_results = {}
        
        for method, result in results.items():
            if method in ['convnext', 'yolo']:
                # Handle individual model results (no per_class_accuracy)
                json_results[method] = {
                    'accuracy': convert(result['accuracy']),
                    'predictions': convert(result['predictions']),
                    'true_labels': convert(result['true_labels'])
                }
            else:
                # Handle ensemble method results (with per_class_accuracy)
                json_results[method] = {
                    'accuracy': convert(result['accuracy']),
                    'per_class_accuracy': convert(result['per_class_accuracy']),
                    'predictions': convert(result['predictions']),
                    'true_labels': convert(result['true_labels'])
                }
        
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {filename}")
    
    def predict_single_image(self, image_path, show_details=True):
        """Make prediction on a single image with detailed output"""
        print(f"\nPredicting: {image_path}")
        print("-" * 60)
        
        # Get predictions from all methods
        methods_to_test = ['weighted_average', 'max_voting', 'geometric_mean']
        
        for method in methods_to_test:
            result = self.ensemble_predict(image_path, method=method)
            
            if result is not None:
                print(f"\n{method.replace('_', ' ').title()}:")
                print(f"  Predicted: {result['predicted_label']} (confidence: {result['confidence']:.3f})")
                
                if show_details and result['convnext_probs'] is not None and result['yolo_probs'] is not None:
                    print("  Individual model probabilities:")
                    print("    ConvNeXt | YOLO11  | Class")
                    print("    ---------|---------|--------")
                    for i, class_name in enumerate(self.classes):
                        convnext_prob = result['convnext_probs'][i]
                        yolo_prob = result['yolo_probs'][i]
                        print(f"    {convnext_prob:.3f}    | {yolo_prob:.3f}   | {class_name}")


def main():
    """Main execution function"""
    print("Dental Caries Ensemble Classification")
    print("=" * 60)
    
    # Configuration
    CONVNEXT_PATH = 'best_convnext.pth'  # Path to trained ConvNeXt model
    YOLO_PATH = 'best.pt'  # Path to trained YOLO11 model
    TEST_DIR = 'dataset/test'  # Test dataset directory
    
    # Check if model files exist
    if not os.path.exists(CONVNEXT_PATH):
        print(f"❌ ConvNeXt model not found at {CONVNEXT_PATH}")
        print("Please train the ConvNeXt model first using convnext.py")
        return
    
    if not os.path.exists(YOLO_PATH):
        print(f"❌ YOLO11 model not found at {YOLO_PATH}")
        print("Please train the YOLO11 model first using yolo.py")
        return
    
    # Initialize ensemble
    ensemble = DentalCariesEnsemble(
        convnext_path=CONVNEXT_PATH,
        yolo_path=YOLO_PATH,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Test on a single image first (if available)
    sample_images = []
    if os.path.exists(TEST_DIR):
        for class_name in ensemble.classes:
            class_dir = os.path.join(TEST_DIR, class_name)
            if os.path.exists(class_dir):
                images = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    sample_images.append(os.path.join(class_dir, images[0]))
    
    if sample_images:
        print("\nTesting on sample images:")
        for img_path in sample_images[:2]:  # Test on first 2 samples
            ensemble.predict_single_image(img_path)
    
    # Evaluate ensemble methods
    if os.path.exists(TEST_DIR):
        print(f"\n{'='*60}")
        print("EVALUATING ENSEMBLE METHODS")
        print(f"{'='*60}")
        
        results, best_method = ensemble.compare_methods(TEST_DIR)
        
        # Create confusion matrices
        ensemble.create_confusion_matrices(results)
        
        # Save results
        ensemble.save_results(results)
        
        print(f"\n✅ Ensemble evaluation completed!")
        print(f"🏆 Best method: {best_method}")
        print(f"📊 Results saved in 'ensemble_results' directory")
        
    else:
        print(f"❌ Test directory not found at {TEST_DIR}")
        print("Please ensure your dataset has a 'test' folder with the proper structure")


if __name__ == "__main__":
    main()
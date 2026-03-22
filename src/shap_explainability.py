"""
SHAP Explainability Integration for Signature-Based Biometric Authentication
Implements SHAP (SHapley Additive exPlanations) for feature importance analysis
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import os
import json


class SHAPExplainer:
    def __init__(self, model, background_data=None, feature_names=None):
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.base_value = None
        
    def create_explainer(self, explainer_type='deep', background_samples=100):
        """Create SHAP explainer based on model type"""
        
        if self.background_data is not None:
            # Sample background data for efficiency
            if len(self.background_data) > background_samples:
                indices = np.random.choice(len(self.background_data), 
                                         size=background_samples, 
                                         replace=False)
                background = self.background_data[indices]
            else:
                background = self.background_data
        else:
            # Create synthetic background if none provided
            if hasattr(self.model, 'input_shape'):
                input_shape = self.model.input_shape[1:]  # Remove batch dimension
            else:
                input_shape = (224, 224, 1)  # Default shape
                
            background = np.random.random((background_samples,) + input_shape)
        
        if explainer_type.lower() == 'deep':
            # For deep learning models (CNN)
            try:
                self.explainer = shap.DeepExplainer(self.model, background)
            except Exception as e:
                print(f"DeepExplainer failed: {e}")
                print("Falling back to KernelExplainer...")
                self.explainer = shap.KernelExplainer(self.model.predict, background)
        
        elif explainer_type.lower() == 'kernel':
            # Universal explainer (works with any model)
            self.explainer = shap.KernelExplainer(self.model.predict, background)
            
        elif explainer_type.lower() == 'gradient':
            # For gradient-based explanations
            try:
                self.explainer = shap.GradientExplainer(self.model, background)
            except Exception as e:
                print(f"GradientExplainer failed: {e}")
                print("Falling back to KernelExplainer...")
                self.explainer = shap.KernelExplainer(self.model.predict, background)
                
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
            
        print(f"SHAP {explainer_type} explainer created successfully!")
        return self.explainer
        
    def explain_predictions(self, X_test, max_samples=50):
        """Generate SHAP explanations for test samples"""
        if self.explainer is None:
            raise ValueError("Explainer not created! Call create_explainer first.")
            
        # Limit samples for efficiency
        if len(X_test) > max_samples:
            indices = np.random.choice(len(X_test), size=max_samples, replace=False)
            X_explain = X_test[indices]
        else:
            X_explain = X_test
            
        print(f"Generating SHAP explanations for {len(X_explain)} samples...")
        
        try:
            # Generate SHAP values
            self.shap_values = self.explainer.shap_values(X_explain)
            
            # Handle different output formats
            if isinstance(self.shap_values, list):
                # Multi-class classification
                if len(self.shap_values) == 2:
                    # Binary classification - use positive class
                    self.shap_values = self.shap_values[1]
                else:
                    # Multi-class - use first class for now
                    self.shap_values = self.shap_values[0]
                    
            # Get base value (expected value)
            if hasattr(self.explainer, 'expected_value'):
                self.base_value = self.explainer.expected_value
                if isinstance(self.base_value, (list, np.ndarray)):
                    self.base_value = self.base_value[0] if len(self.base_value) > 0 else 0
            else:
                self.base_value = 0
                
            print("SHAP explanations generated successfully!")
            
        except Exception as e:
            print(f"Error generating SHAP explanations: {e}")
            return None
            
        return self.shap_values
        
    def calculate_feature_importance(self, aggregate_method='mean'):
        """Calculate global feature importance from SHAP values"""
        if self.shap_values is None:
            raise ValueError("SHAP values not generated! Call explain_predictions first.")
            
        # Flatten SHAP values if they are for images
        if len(self.shap_values.shape) > 2:
            # For images, flatten to (samples, features)
            flattened_shap = self.shap_values.reshape(self.shap_values.shape[0], -1)
        else:
            flattened_shap = self.shap_values
            
        # Calculate importance based on aggregation method
        if aggregate_method == 'mean':
            importance = np.mean(np.abs(flattened_shap), axis=0)
        elif aggregate_method == 'sum':
            importance = np.sum(np.abs(flattened_shap), axis=0)
        elif aggregate_method == 'max':
            importance = np.max(np.abs(flattened_shap), axis=0)
        elif aggregate_method == 'std':
            importance = np.std(flattened_shap, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate_method}")
            
        return importance
        
    def prune_features_by_importance(self, features, threshold_percentile=50, 
                                   min_features=10):
        """Prune features based on SHAP importance"""
        if self.shap_values is None:
            raise ValueError("SHAP values not generated! Call explain_predictions first.")
            
        importance = self.calculate_feature_importance()
        
        # Determine threshold
        if threshold_percentile > 0:
            threshold = np.percentile(importance, threshold_percentile)
        else:
            threshold = 0
            
        # Find important features
        important_indices = np.where(importance >= threshold)[0]
        
        # Ensure minimum number of features
        if len(important_indices) < min_features:
            # Select top features if below minimum
            top_indices = np.argsort(importance)[-min_features:]
            important_indices = top_indices
            
        print(f"Feature pruning: {len(important_indices)}/{len(importance)} features retained")
        print(f"Importance threshold: {threshold:.6f}")
        
        # Prune features
        if len(features.shape) > 2:
            # For image features, need to reshape and select
            flattened_features = features.reshape(features.shape[0], -1)
            pruned_features = flattened_features[:, important_indices]
        else:
            pruned_features = features[:, important_indices]
            
        pruning_info = {
            'original_feature_count': features.shape[-1] if len(features.shape) == 2 else np.prod(features.shape[1:]),
            'pruned_feature_count': len(important_indices),
            'threshold': threshold,
            'threshold_percentile': threshold_percentile,
            'important_indices': important_indices.tolist(),
            'feature_importance': importance.tolist()
        }
        
        return pruned_features, pruning_info
        
    def plot_feature_importance(self, top_k=20, save_path=None):
        """Plot global feature importance"""
        if self.shap_values is None:
            raise ValueError("SHAP values not generated! Call explain_predictions first.")
            
        importance = self.calculate_feature_importance()
        
        # Get top-k most important features
        top_indices = np.argsort(importance)[-top_k:]
        top_importance = importance[top_indices]
        
        # Create feature names if not provided
        if self.feature_names is not None:
            if len(self.feature_names) >= len(top_indices):
                feature_names = [self.feature_names[i] for i in top_indices]
            else:
                feature_names = [f'Feature_{i}' for i in top_indices]
        else:
            feature_names = [f'Feature_{i}' for i in top_indices]
            
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_importance)), top_importance)
        plt.yticks(range(len(top_importance)), feature_names)
        plt.xlabel('SHAP Feature Importance')
        plt.title(f'Top {top_k} Most Important Features')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_summary_plot(self, X_test, save_path=None, plot_type='dot'):
        """Create SHAP summary plot"""
        if self.shap_values is None:
            raise ValueError("SHAP values not generated! Call explain_predictions first.")
            
        try:
            plt.figure(figsize=(12, 8))
            
            if len(self.shap_values.shape) > 2:
                # For image data, flatten for summary plot
                shap_flat = self.shap_values.reshape(self.shap_values.shape[0], -1)
                X_flat = X_test[:len(self.shap_values)].reshape(X_test[:len(self.shap_values)].shape[0], -1)
                
                # Sample features for visualization (too many features make plot unreadable)
                n_features = min(50, shap_flat.shape[1])
                indices = np.random.choice(shap_flat.shape[1], size=n_features, replace=False)
                
                shap.summary_plot(
                    shap_flat[:, indices], 
                    X_flat[:, indices], 
                    feature_names=[f'Pixel_{i}' for i in indices],
                    plot_type=plot_type,
                    show=False
                )
            else:
                # For regular tabular data
                feature_names = self.feature_names if self.feature_names else None
                shap.summary_plot(
                    self.shap_values, 
                    X_test[:len(self.shap_values)], 
                    feature_names=feature_names,
                    plot_type=plot_type,
                    show=False
                )
                
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating summary plot: {e}")
            
    def plot_waterfall_plot(self, instance_idx=0, save_path=None):
        """Create waterfall plot for a specific instance"""
        if self.shap_values is None:
            raise ValueError("SHAP values not generated! Call explain_predictions first.")
            
        try:
            if len(self.shap_values.shape) > 2:
                # For image data, show top contributing pixels
                shap_flat = self.shap_values[instance_idx].flatten()
                top_indices = np.argsort(np.abs(shap_flat))[-20:]  # Top 20 pixels
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(top_indices)), shap_flat[top_indices])
                plt.yticks(range(len(top_indices)), [f'Pixel_{i}' for i in top_indices])
                plt.xlabel('SHAP Value')
                plt.title(f'Top Contributing Pixels - Instance {instance_idx}')
                
            else:
                # For tabular data
                shap.waterfall_plot(
                    shap.Explanation(
                        values=self.shap_values[instance_idx],
                        base_values=self.base_value,
                        feature_names=self.feature_names
                    ),
                    show=False
                )
                
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating waterfall plot: {e}")
            
    def plot_force_plot(self, instance_idx=0, save_path=None):
        """Create force plot for a specific instance"""
        if self.shap_values is None:
            raise ValueError("SHAP values not generated! Call explain_predictions first.")
            
        try:
            if len(self.shap_values.shape) <= 2:
                force_plot = shap.force_plot(
                    self.base_value,
                    self.shap_values[instance_idx],
                    feature_names=self.feature_names,
                    matplotlib=True,
                    show=False
                )
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()
            else:
                print("Force plots not suitable for high-dimensional data like images")
                
        except Exception as e:
            print(f"Error creating force plot: {e}")
            
    def visualize_image_explanations(self, X_test, instance_idx=0, save_path=None):
        """Visualize SHAP explanations for image data"""
        if self.shap_values is None:
            raise ValueError("SHAP values not generated! Call explain_predictions first.")
            
        if len(self.shap_values.shape) <= 2:
            print("Image visualization requires image-shaped SHAP values")
            return
            
        try:
            # Get original image and SHAP values
            original_image = X_test[instance_idx]
            shap_image = self.shap_values[instance_idx]
            
            # Ensure correct dimensions
            if len(original_image.shape) == 3 and original_image.shape[-1] == 1:
                original_image = original_image.squeeze()
            if len(shap_image.shape) == 3 and shap_image.shape[-1] == 1:
                shap_image = shap_image.squeeze()
                
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original_image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # SHAP explanation (positive and negative contributions)
            im1 = axes[1].imshow(shap_image, cmap='RdBu', center=0)
            axes[1].set_title('SHAP Explanation')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Overlay explanation on original
            axes[2].imshow(original_image, cmap='gray', alpha=0.7)
            im2 = axes[2].imshow(np.abs(shap_image), cmap='hot', alpha=0.3)
            axes[2].set_title('Explanation Overlay')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating image explanation visualization: {e}")
            
    def save_explanations(self, filepath):
        """Save SHAP explanations to file"""
        if self.shap_values is None:
            print("No SHAP values to save!")
            return
            
        explanations = {
            'shap_values': self.shap_values.tolist() if isinstance(self.shap_values, np.ndarray) else self.shap_values,
            'base_value': float(self.base_value) if self.base_value is not None else None,
            'feature_importance': self.calculate_feature_importance().tolist(),
            'feature_names': self.feature_names if self.feature_names else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(explanations, f, indent=2)
            
        print(f"SHAP explanations saved to {filepath}")
        
    def generate_explanation_report(self, X_test, y_test=None, save_dir=None):
        """Generate comprehensive explanation report"""
        if self.shap_values is None:
            raise ValueError("SHAP values not generated! Call explain_predictions first.")
            
        report = {
            'summary': {
                'num_samples_explained': len(self.shap_values),
                'num_features': self.shap_values.shape[1] if len(self.shap_values.shape) == 2 else np.prod(self.shap_values.shape[1:]),
                'base_value': float(self.base_value) if self.base_value is not None else None
            },
            'feature_importance': {
                'global_importance': self.calculate_feature_importance().tolist(),
                'top_10_features': np.argsort(self.calculate_feature_importance())[-10:].tolist()
            }
        }
        
        # Add individual explanations for first few samples
        report['individual_explanations'] = []
        for i in range(min(5, len(self.shap_values))):
            if len(self.shap_values.shape) > 2:
                # For images, summarize by regions
                explanation = {
                    'sample_idx': i,
                    'shap_values_summary': {
                        'mean': float(np.mean(self.shap_values[i])),
                        'std': float(np.std(self.shap_values[i])),
                        'min': float(np.min(self.shap_values[i])),
                        'max': float(np.max(self.shap_values[i]))
                    }
                }
            else:
                explanation = {
                    'sample_idx': i,
                    'shap_values': self.shap_values[i].tolist()
                }
                
            if y_test is not None:
                explanation['true_label'] = int(y_test[i])
                
            report['individual_explanations'].append(explanation)
            
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            report_path = os.path.join(save_dir, 'shap_explanation_report.json')
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            print(f"Explanation report saved to {report_path}")
            
        return report


class SHAPForTabularFeatures:
    """SHAP explainer specifically for tabular features (after CNN feature extraction)"""
    
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
    def create_tree_explainer(self, X_background):
        """Create TreeExplainer for tree-based models (XGBoost, LightGBM)"""
        self.explainer = shap.TreeExplainer(self.model)
        print("SHAP TreeExplainer created for tabular features!")
        
    def create_kernel_explainer(self, X_background, background_samples=100):
        """Create KernelExplainer for any model"""
        if len(X_background) > background_samples:
            indices = np.random.choice(len(X_background), size=background_samples, replace=False)
            background = X_background[indices]
        else:
            background = X_background
            
        self.explainer = shap.KernelExplainer(self.model.predict, background)
        print("SHAP KernelExplainer created for tabular features!")
        
    def explain_predictions(self, X_test, max_samples=100):
        """Generate SHAP explanations for tabular features"""
        if self.explainer is None:
            raise ValueError("Explainer not created!")
            
        if len(X_test) > max_samples:
            indices = np.random.choice(len(X_test), size=max_samples, replace=False)
            X_explain = X_test[indices]
        else:
            X_explain = X_test
            
        shap_values = self.explainer.shap_values(X_explain)
        
        # Handle different output formats
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Use positive class for binary classification
            
        return shap_values, X_explain


if __name__ == "__main__":
    print("SHAP Explainability module ready!")
    print("Available explainers:")
    print("- SHAPExplainer: For CNN models and image data")
    print("- SHAPForTabularFeatures: For extracted features and tree-based models")
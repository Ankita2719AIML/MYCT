"""
Complete Training Pipeline for Signature-Based Biometric Authentication
Integrates all components: preprocessing, CNN, autoencoder, SHAP, Grad-CAM, PCA, GBM
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all components
try:
    from data_preprocessing import SignaturePreprocessor, generate_sample_data
    from cnn_feature_extractor import SignatureCNN
    from autoencoder_forgery_detection import SignatureAutoencoder
    from shap_explainability import SHAPExplainer, SHAPForTabularFeatures
    from gradcam_visualization import GradCAM
    from pca_gbm_classifier import PCAFeatureReducer, GBMClassifier
    from evaluation_metrics import BiometricEvaluator
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are in the correct location.")
    raise

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


class SignatureAuthenticationPipeline:
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.components = {}
        self.results = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _get_default_config(self):
        """Get default configuration for the pipeline"""
        return {
            'data': {
                'dataset_path': 'C:/Users/priya/OneDrive/Desktop/MYCT_TASK/archive/MYCT_pics_updated',
                'image_size': (224, 224),
                'validation_split': 0.2,
                'test_split': 0.2,
                'random_state': 42
            },
            'preprocessing': {
                'save_intermediate': True,
                'noise_reduction': True,
                'orientation_correction': True
            },
            'cnn': {
                'epochs': 2,
                'batch_size': 32,
                'learning_rate': 0.001,
                'early_stopping_patience': 10
            },
            'autoencoder': {
                'epochs': 2,
                'batch_size': 32,
                'learning_rate': 0.001,
                'latent_dim': 128,
                'anomaly_threshold_percentile': 95
            },
            'explainability': {
                'shap_samples': 50,
                'gradcam_layer': None,  # Auto-detect
                'feature_importance_threshold': 50  # percentile
            },
            'pca': {
                'variance_threshold': 0.95,
                'scale_features': True
            },
            'gbm': {
                'model_type': 'xgboost',
                'hyperparameter_tuning': True,
                'cv_folds': 5
            },
            'output': {
                'save_models': True,
                'save_visualizations': True,
                'save_results': True,
                'results_dir': 'C:/Users/priya/OneDrive/Desktop/MYCT_TASK/results'
            }
        }
        
    def initialize_components(self):
        """Initialize all pipeline components"""
        print("Initializing pipeline components...")
        
        # Ensure config has required keys
        if 'data' not in self.config:
            self.config = self._get_default_config()
        
        # Data preprocessor
        self.components['preprocessor'] = SignaturePreprocessor(
            target_size=tuple(self.config['data']['image_size'])
        )
        
        # CNN feature extractor
        image_shape = tuple(self.config['data']['image_size']) + (1,)
        self.components['cnn'] = SignatureCNN(
            input_shape=image_shape,
            num_classes=2
        )
        
        # Autoencoder for forgery detection
        self.components['autoencoder'] = SignatureAutoencoder(
            input_shape=image_shape,
            latent_dim=self.config['autoencoder']['latent_dim']
        )
        
        # PCA feature reducer
        self.components['pca'] = PCAFeatureReducer(
            variance_threshold=self.config['pca']['variance_threshold']
        )
        
        # GBM classifier
        self.components['gbm'] = GBMClassifier(
            model_type=self.config['gbm']['model_type']
        )
        
        # Evaluator
        self.components['evaluator'] = BiometricEvaluator()
        
        print("All components initialized successfully!")
        
    def load_and_preprocess_data(self):
        """Load and preprocess the signature dataset"""
        print("Loading and preprocessing data...")
        
        dataset_path = self.config['data']['dataset_path']
        
        # Check if dataset exists and has data
        train_path = os.path.join(dataset_path, 'train')
        if not os.path.exists(train_path) or not any(
            os.listdir(os.path.join(train_path, category)) 
            for category in ['genuine', 'forged'] 
            if os.path.exists(os.path.join(train_path, category))
        ):
            print("Dataset not found or empty. Generating sample data...")
            self._generate_sample_dataset()
        
        # Load training data
        X_train, y_train = self.components['preprocessor'].preprocess_dataset(
            train_path,
            output_dir=os.path.join(self.config['output']['results_dir'], 'preprocessed_train') 
            if self.config['preprocessing']['save_intermediate'] else None,
            save_intermediate=self.config['preprocessing']['save_intermediate']
        )
        
        # Load validation data
        val_path = os.path.join(dataset_path, 'val')
        if os.path.exists(val_path):
            X_val, y_val = self.components['preprocessor'].preprocess_dataset(val_path)
        else:
            # Split training data
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=self.config['data']['validation_split'],
                random_state=self.config['data']['random_state'],
                stratify=y_train
            )
        
        # Load test data
        test_path = os.path.join(dataset_path, 'test')
        if os.path.exists(test_path):
            X_test, y_test = self.components['preprocessor'].preprocess_dataset(test_path)
        else:
            # Split from validation data
            X_val, X_test, y_val, y_test = train_test_split(
                X_val, y_val,
                test_size=self.config['data']['test_split'] / (1 - self.config['data']['validation_split']),
                random_state=self.config['data']['random_state'],
                stratify=y_val
            )
        
        print(f"Data loaded successfully!")
        print(f"Training samples: {len(X_train)} (Genuine: {np.sum(y_train)}, Forged: {len(y_train) - np.sum(y_train)})")
        print(f"Validation samples: {len(X_val)} (Genuine: {np.sum(y_val)}, Forged: {len(y_val) - np.sum(y_val)})")
        print(f"Test samples: {len(X_test)} (Genuine: {np.sum(y_test)}, Forged: {len(y_test) - np.sum(y_test)})")
        
        # Store data
        self.data = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
        
        return self.data
        
    def _generate_sample_dataset(self):
        """Generate sample dataset if original is not available"""
        dataset_path = self.config['data']['dataset_path']
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            for category in ['genuine', 'forged']:
                os.makedirs(os.path.join(dataset_path, split, category), exist_ok=True)
        
        # Generate sample data
        generate_sample_data(os.path.join(dataset_path, 'train'), num_genuine=100, num_forged=100)
        generate_sample_data(os.path.join(dataset_path, 'val'), num_genuine=30, num_forged=30)
        generate_sample_data(os.path.join(dataset_path, 'test'), num_genuine=30, num_forged=30)
        
        print("Sample dataset generated successfully!")
        
    def train_cnn_feature_extractor(self):
        """Train or load CNN for feature extraction"""
        models_dir = os.path.join(self.config['output']['results_dir'], 'models')
        cnn_model_path = os.path.join(models_dir, 'cnn_model.h5')
        
        # Check if pre-trained CNN model exists
        if os.path.exists(cnn_model_path):
            print("Loading existing CNN model...")
            try:
                self.components['cnn'].load_model(cnn_model_path)
                
                # Evaluate the loaded model
                cnn_results = self.components['cnn'].evaluate_model(
                    self.data['X_test'], self.data['y_test']
                )
                
                self.results['cnn_training_history'] = {
                    'model_loaded': True,
                    'model_path': cnn_model_path
                }
                self.results['cnn_evaluation'] = cnn_results
                
                print(f"CNN model loaded successfully! Test accuracy: {cnn_results['accuracy']:.4f}")
                return None
                
            except Exception as e:
                print(f"Failed to load existing CNN model: {e}")
                print("Training new CNN model...")
        else:
            print("No existing CNN model found. Training new CNN model...")
        
        # Train new CNN model
        self.components['cnn'].compile_model(
            learning_rate=self.config['cnn']['learning_rate']
        )
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Train CNN
        history = self.components['cnn'].train_model(
            self.data['X_train'], self.data['y_train'],
            self.data['X_val'], self.data['y_val'],
            epochs=self.config['cnn']['epochs'],
            batch_size=self.config['cnn']['batch_size'],
            save_path=cnn_model_path if self.config['output']['save_models'] else None
        )
        
        # Evaluate CNN
        cnn_results = self.components['cnn'].evaluate_model(
            self.data['X_test'], self.data['y_test']
        )
        
        # Store results
        self.results['cnn_training_history'] = {
            'epochs': len(history.history['accuracy']),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'model_trained': True
        }
        
        self.results['cnn_evaluation'] = cnn_results
        
        # Export architecture summary
        arch_path = os.path.join(self.config['output']['results_dir'], 'reports', 'cnn_architecture_summary.json')
        os.makedirs(os.path.dirname(arch_path), exist_ok=True)
        try:
            self.components['cnn'].export_architecture_summary(arch_path)
        except Exception as e:
            print(f"Failed to export CNN architecture: {e}")
        
        # Visualize training history
        if self.config['output']['save_visualizations']:
            viz_dir = os.path.join(self.config['output']['results_dir'], 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            self.components['cnn'].visualize_training_history(
                history,
                save_path=os.path.join(viz_dir, 'cnn_training_history.png')
            )
        
        print("CNN training completed successfully!")
        return history
        
    def train_autoencoder_forgery_detector(self):
        """Train or load autoencoder for forgery detection"""
        models_dir = os.path.join(self.config['output']['results_dir'], 'models')
        autoencoder_model_path = os.path.join(models_dir, 'autoencoder.h5')
        
        # Check if pre-trained autoencoder exists
        if os.path.exists(autoencoder_model_path):
            print("Loading existing autoencoder model...")
            try:
                self.components['autoencoder'].load_autoencoder(autoencoder_model_path)
                
                # Set threshold for the loaded model
                genuine_indices = np.where(self.data['y_train'] == 1)[0]
                X_genuine = self.data['X_train'][genuine_indices]
                self.components['autoencoder'].determine_threshold(
                    X_genuine,
                    percentile=self.config['autoencoder']['anomaly_threshold_percentile']
                )
                
                # Evaluate forgery detection
                forgery_results = self.components['autoencoder'].evaluate_forgery_detection(
                    self.data['X_test'][self.data['y_test'] == 1],  # Genuine test samples
                    self.data['X_test'][self.data['y_test'] == 0]   # Forged test samples
                )
                
                self.results['autoencoder_training'] = {
                    'model_loaded': True,
                    'model_path': autoencoder_model_path,
                    'threshold': float(self.components['autoencoder'].threshold)
                }
                self.results['forgery_detection'] = forgery_results
                
                print(f"Autoencoder loaded successfully! Detection accuracy: {forgery_results['accuracy']:.4f}")
                return None
                
            except Exception as e:
                print(f"Failed to load existing autoencoder: {e}")
                print("Training new autoencoder...")
        else:
            print("No existing autoencoder found. Training new autoencoder...")
        
        # Train new autoencoder
        # Extract only genuine signatures for autoencoder training
        genuine_indices = np.where(self.data['y_train'] == 1)[0]
        X_genuine = self.data['X_train'][genuine_indices]
        
        # Use some genuine validation samples
        genuine_val_indices = np.where(self.data['y_val'] == 1)[0]
        X_genuine_val = self.data['X_val'][genuine_val_indices] if len(genuine_val_indices) > 0 else None
        
        # Compile and train autoencoder
        self.components['autoencoder'].compile_autoencoder(
            learning_rate=self.config['autoencoder']['learning_rate']
        )
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        history = self.components['autoencoder'].train_autoencoder(
            X_genuine, X_genuine_val,
            epochs=self.config['autoencoder']['epochs'],
            batch_size=self.config['autoencoder']['batch_size'],
            save_path=autoencoder_model_path if self.config['output']['save_models'] else None
        )
        
        # Determine threshold using validation data
        self.components['autoencoder'].determine_threshold(
            X_genuine_val if X_genuine_val is not None else X_genuine,
            percentile=self.config['autoencoder']['anomaly_threshold_percentile']
        )
        
        # Evaluate forgery detection
        forgery_results = self.components['autoencoder'].evaluate_forgery_detection(
            self.data['X_test'][self.data['y_test'] == 1],  # Genuine test samples
            self.data['X_test'][self.data['y_test'] == 0]   # Forged test samples
        )
        
        self.results['autoencoder_training'] = {
            'epochs': len(history.history['loss']),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
            'threshold': float(self.components['autoencoder'].threshold)
        }
        
        self.results['forgery_detection'] = forgery_results
        
        # Visualizations
        if self.config['output']['save_visualizations']:
            viz_dir = os.path.join(self.config['output']['results_dir'], 'visualizations')
            
            # Reconstruction visualization
            self.components['autoencoder'].visualize_reconstruction(
                self.data['X_test'][:5],
                save_path=os.path.join(viz_dir, 'autoencoder_reconstruction.png')
            )
            
            # Error distribution
            self.components['autoencoder'].plot_reconstruction_error_distribution(
                self.data['X_test'][self.data['y_test'] == 1],
                self.data['X_test'][self.data['y_test'] == 0],
                save_path=os.path.join(viz_dir, 'reconstruction_error_distribution.png')
            )
        
        print("Autoencoder training completed successfully!")
        return history
        
    def extract_features_and_apply_explainability(self):
        """Extract CNN features and apply explainability methods"""
        print("Extracting features and applying explainability...")
        
        # Extract deep features using CNN
        print("Extracting deep features...")
        train_features = self.components['cnn'].extract_features(self.data['X_train'])
        val_features = self.components['cnn'].extract_features(self.data['X_val'])
        test_features = self.components['cnn'].extract_features(self.data['X_test'])
        
        # Store features
        self.extracted_features = {
            'train': train_features,
            'val': val_features,
            'test': test_features
        }
        
        print(f"Feature extraction completed. Feature dimension: {train_features.shape[1]}")
        
        # Apply SHAP explainability
        print("Applying SHAP explainability...")
        try:
            # For CNN model (image explanations)
            shap_explainer = SHAPExplainer(
                self.components['cnn'].model,
                background_data=self.data['X_train'][:20]  # Small background sample
            )
            
            shap_explainer.create_explainer(explainer_type='gradient', background_samples=20)
            shap_values = shap_explainer.explain_predictions(
                self.data['X_test'][:self.config['explainability']['shap_samples']]
            )
            
            if shap_values is not None:
                # Feature pruning based on SHAP importance
                pruned_features, pruning_info = shap_explainer.prune_features_by_importance(
                    test_features,
                    threshold_percentile=self.config['explainability']['feature_importance_threshold']
                )
                
                self.results['shap_analysis'] = {
                    'num_samples_explained': len(shap_values) if shap_values is not None else 0,
                    'feature_pruning': pruning_info
                }
                
                # Generate explanation report
                explanation_report = shap_explainer.generate_explanation_report(
                    self.data['X_test'][:self.config['explainability']['shap_samples']],
                    self.data['y_test'][:self.config['explainability']['shap_samples']],
                    save_dir=os.path.join(self.config['output']['results_dir'], 'explanations')
                    if self.config['output']['save_results'] else None
                )
                
                self.results['shap_report'] = explanation_report
                
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            self.results['shap_analysis'] = {'error': str(e)}
        
        # Apply Grad-CAM
        print("Applying Grad-CAM visualization...")
        try:
            gradcam = GradCAM(
                self.components['cnn'].model,
                layer_name=self.config['explainability']['gradcam_layer']
            )
            
            if self.config['output']['save_visualizations']:
                viz_dir = os.path.join(self.config['output']['results_dir'], 'visualizations')
                
                # Save GradCAM results
                gradcam_results = gradcam.save_gradcam_results(
                    self.data['X_test'],
                    self.data['y_test'],
                    os.path.join(viz_dir, 'gradcam_results'),
                    num_samples=10
                )
                
                # Analyze attention patterns
                attention_analysis = gradcam.analyze_attention_patterns(
                    self.data['X_test'][:20],
                    self.data['y_test'][:20],
                    save_dir=viz_dir
                )
                
                self.results['gradcam_analysis'] = {
                    'num_samples_analyzed': len(gradcam_results['samples']),
                    'attention_patterns': attention_analysis
                }
                
        except Exception as e:
            print(f"Grad-CAM analysis failed: {e}")
            self.results['gradcam_analysis'] = {'error': str(e)}
        
        print("Feature extraction and explainability analysis completed!")
        return self.extracted_features
        
    def apply_pca_and_train_gbm(self):
        """Apply PCA dimensionality reduction and train GBM classifier"""
        print("Applying PCA and training GBM classifier...")
        
        models_dir = os.path.join(self.config['output']['results_dir'], 'models')
        pca_model_path = os.path.join(models_dir, 'pca_model.pkl')
        gbm_model_path = os.path.join(models_dir, 'gbm_model.pkl')
        
        # Check if PCA model exists
        if os.path.exists(pca_model_path):
            print("Loading existing PCA model...")
            try:
                self.components['pca'].load_pca(pca_model_path)
            except Exception as e:
                print(f"Failed to load PCA model: {e}")
                print("Training new PCA model...")
                self.components['pca'].fit(
                    self.extracted_features['train'],
                    scale_features=self.config['pca']['scale_features']
                )
        else:
            print("Training new PCA model...")
            self.components['pca'].fit(
                self.extracted_features['train'],
                scale_features=self.config['pca']['scale_features']
            )
        
        # Transform features
        train_pca = self.components['pca'].transform(
            self.extracted_features['train'],
            scale_features=self.config['pca']['scale_features']
        )
        val_pca = self.components['pca'].transform(
            self.extracted_features['val'],
            scale_features=self.config['pca']['scale_features']
        )
        test_pca = self.components['pca'].transform(
            self.extracted_features['test'],
            scale_features=self.config['pca']['scale_features']
        )
        
        print(f"PCA applied. Original features: {self.extracted_features['train'].shape[1]}, "
              f"Reduced features: {train_pca.shape[1]}")
        
        # Visualize PCA
        if self.config['output']['save_visualizations']:
            viz_dir = os.path.join(self.config['output']['results_dir'], 'visualizations')
            self.components['pca'].plot_explained_variance(
                save_path=os.path.join(viz_dir, 'pca_explained_variance.png')
            )
        
        # Store PCA-transformed features for evaluation FIRST
        self.pca_features = {
            'train': train_pca,
            'val': val_pca,
            'test': test_pca
        }
        
        # Check if GBM model exists
        if os.path.exists(gbm_model_path):
            print("Loading existing GBM model...")
            try:
                self.components['gbm'].load_model(gbm_model_path)
                
                # Evaluate the loaded model
                gbm_results = self.components['gbm'].evaluate_model(test_pca, self.data['y_test'])
                self.results['gbm_evaluation'] = gbm_results
                
                print(f"GBM model loaded successfully! Test accuracy: {gbm_results['accuracy']:.4f}")
                
            except Exception as e:
                print(f"Failed to load GBM model: {e}")
                print("Training new GBM model...")
                self._train_new_gbm(train_pca)
        else:
            print("Training new GBM model...")
            self._train_new_gbm(train_pca)
        
        print("PCA and GBM processing completed successfully!")
        return self.pca_features
        
    def _train_new_gbm(self, train_pca):
        """Helper method to train new GBM model"""
        if self.config['gbm']['hyperparameter_tuning']:
            # Hyperparameter tuning
            best_params = self.components['gbm'].hyperparameter_tuning(
                train_pca, self.data['y_train'],
                cv=self.config['gbm']['cv_folds']
            )
            self.results['gbm_best_params'] = best_params
        else:
            # Train with default parameters
            self.components['gbm'].fit(train_pca, self.data['y_train'])
        
        # Evaluate GBM
        gbm_results = self.components['gbm'].evaluate_model(
            self.pca_features['test'], self.data['y_test']
        )
        self.results['gbm_evaluation'] = gbm_results
        
        # Cross-validation
        cv_scores = self.components['gbm'].cross_validate(
            train_pca, self.data['y_train'],
            cv=self.config['gbm']['cv_folds']
        )
        self.results['gbm_cv_scores'] = cv_scores.tolist()
        
        # Visualizations
        if self.config['output']['save_visualizations']:
            viz_dir = os.path.join(self.config['output']['results_dir'], 'visualizations')
            
            # Feature importance
            self.components['gbm'].plot_feature_importance(
                top_k=20,
                save_path=os.path.join(viz_dir, 'gbm_feature_importance.png')
            )
            
            # ROC curve
            self.components['gbm'].plot_roc_curve(
                self.pca_features['test'], self.data['y_test'],
                save_path=os.path.join(viz_dir, 'gbm_roc_curve.png')
            )
            
            # Confusion matrix
            self.components['gbm'].plot_confusion_matrix(
                self.pca_features['test'], self.data['y_test'],
                save_path=os.path.join(viz_dir, 'gbm_confusion_matrix.png')
            )

        
    def comprehensive_evaluation(self):
        """Perform comprehensive system evaluation"""
        print("Performing comprehensive evaluation...")
        
        # Prepare model results for comparison
        model_results = {}
        
        # CNN evaluation
        cnn_pred = self.components['cnn'].predict(self.data['X_test'])
        cnn_pred_classes = np.argmax(cnn_pred, axis=1)
        cnn_pred_proba = cnn_pred[:, 1]  # Probability of genuine class
        
        model_results['CNN'] = {
            'y_true': self.data['y_test'],
            'y_pred': cnn_pred_classes,
            'y_pred_proba': cnn_pred_proba
        }
        
        # GBM evaluation
        gbm_pred = self.components['gbm'].predict(self.pca_features['test'])
        gbm_pred_proba = self.components['gbm'].predict_proba(self.pca_features['test'])[:, 1]
        
        model_results['GBM'] = {
            'y_true': self.data['y_test'],
            'y_pred': gbm_pred,
            'y_pred_proba': gbm_pred_proba
        }
        
        # Autoencoder evaluation (forgery detection)
        autoencoder_results = self.components['autoencoder'].detect_forgery(self.data['X_test'])
        # Convert to match other models (1=genuine, 0=forged)
        ae_pred = 1 - autoencoder_results['predictions']  # Flip since AE predicts forgery
        ae_scores = 1 - autoencoder_results['reconstruction_errors']  # Higher score = more genuine
        ae_scores = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min())  # Normalize
        
        model_results['Autoencoder'] = {
            'y_true': self.data['y_test'],
            'y_pred': ae_pred,
            'y_pred_proba': ae_scores
        }
        
        # Generate comprehensive performance report
        performance_report = self.components['evaluator'].generate_performance_report(
            model_results,
            save_dir=os.path.join(self.config['output']['results_dir'], 'evaluation')
            if self.config['output']['save_results'] else None
        )
        
        self.results['performance_report'] = performance_report
        
        # Generate comparison visualizations
        if self.config['output']['save_visualizations']:
            viz_dir = os.path.join(self.config['output']['results_dir'], 'visualizations')
            
            # ROC curves comparison
            self.components['evaluator'].plot_roc_curves(
                model_results,
                save_path=os.path.join(viz_dir, 'models_roc_comparison.png')
            )
            
            # DET curves
            self.components['evaluator'].plot_det_curves(
                model_results,
                save_path=os.path.join(viz_dir, 'models_det_comparison.png')
            )
            
            # Precision-Recall curves
            self.components['evaluator'].plot_precision_recall_curves(
                model_results,
                save_path=os.path.join(viz_dir, 'models_pr_comparison.png')
            )
            
            # Confusion matrices
            self.components['evaluator'].plot_confusion_matrices(
                model_results,
                save_path=os.path.join(viz_dir, 'models_confusion_matrices.png')
            )
            
            # Metrics comparison
            self.components['evaluator'].plot_metrics_comparison(
                model_results,
                metrics=['accuracy', 'precision', 'recall', 'f1_score'],
                save_path=os.path.join(viz_dir, 'models_metrics_comparison.png')
            )
        
        print("Comprehensive evaluation completed!")
        return performance_report
        
    def save_models(self):
        """Save all trained models"""
        if not self.config['output']['save_models']:
            return
            
        models_dir = os.path.join(self.config['output']['results_dir'], 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        print("Saving trained models...")
        
        # Save CNN
        self.components['cnn'].save_model(os.path.join(models_dir, 'cnn_model.h5'))
        
        # Save Autoencoder
        self.components['autoencoder'].save_autoencoder(os.path.join(models_dir, 'autoencoder.h5'))
        
        # Save PCA
        self.components['pca'].save_pca(os.path.join(models_dir, 'pca_model.pkl'))
        
        # Save GBM
        self.components['gbm'].save_model(os.path.join(models_dir, 'gbm_model.pkl'))
        
        print("All models saved successfully!")
        
    def save_results_json(self):
        """Save all results to JSON files"""
        if not self.config['output']['save_results']:
            return
            
        results_dir = self.config['output']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        # Add pipeline metadata
        pipeline_info = {
            'pipeline_info': {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'configuration': self.config,
                'dataset_info': {
                    'train_samples': len(self.data['X_train']),
                    'val_samples': len(self.data['X_val']),
                    'test_samples': len(self.data['X_test']),
                    'genuine_train': int(np.sum(self.data['y_train'])),
                    'forged_train': int(len(self.data['y_train']) - np.sum(self.data['y_train'])),
                    'image_shape': self.data['X_train'].shape[1:]
                }
            }
        }
        
        # Combine all results
        complete_results = {**pipeline_info, **self.results}
        
        # Save main results
        with open(os.path.join(results_dir, f'complete_results_{self.session_id}.json'), 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        # Save individual component results
        for component_name, component_results in self.results.items():
            if isinstance(component_results, dict):
                with open(os.path.join(results_dir, f'{component_name}_{self.session_id}.json'), 'w') as f:
                    json.dump(component_results, f, indent=2, default=str)
        
        print(f"Results saved to {results_dir}")
        
    def run_complete_pipeline(self):
        """Execute the complete signature authentication pipeline"""
        print("="*80)
        print("SIGNATURE-BASED BIOMETRIC AUTHENTICATION PIPELINE")
        print(f"Session ID: {self.session_id}")
        print("="*80)
        
        try:
            # Step 1: Initialize components
            self.initialize_components()
            
            # Step 2: Load and preprocess data
            self.load_and_preprocess_data()
            
            # Step 3: Train CNN feature extractor
            self.train_cnn_feature_extractor()
            
            # Step 4: Train autoencoder for forgery detection
            self.train_autoencoder_forgery_detector()
            
            # Step 5: Extract features and apply explainability
            self.extract_features_and_apply_explainability()
            
            # Step 6: Apply PCA and train GBM
            self.apply_pca_and_train_gbm()
            
            # Step 7: Comprehensive evaluation
            self.comprehensive_evaluation()
            
            # Step 8: Save models and results
            self.save_models()
            self.save_results_json()
            
            print("="*80)
            print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            print(f"Results saved in: {self.config['output']['results_dir']}")
            print("="*80)
            
            # Print summary
            self.print_summary()
            
        except Exception as e:
            print(f"Pipeline execution failed: {e}")
            raise
            
    def print_summary(self):
        """Print pipeline execution summary"""
        print("\nPIPELINE EXECUTION SUMMARY")
        print("-" * 50)
        
        if 'performance_report' in self.results:
            best_performers = self.results['performance_report'].get('best_performers', {})
            
            print(f"Best Model by Accuracy: {best_performers.get('accuracy', {}).get('model', 'N/A')} "
                  f"({best_performers.get('accuracy', {}).get('value', 0):.4f})")
            
            print(f"Best Model by F1-Score: {best_performers.get('f1_score', {}).get('model', 'N/A')} "
                  f"({best_performers.get('f1_score', {}).get('value', 0):.4f})")
            
            print(f"Best Model by ROC-AUC: {best_performers.get('roc_auc', {}).get('model', 'N/A')} "
                  f"({best_performers.get('roc_auc', {}).get('value', 0):.4f})")
            
            if 'eer' in best_performers:
                print(f"Lowest EER: {best_performers['eer']['model']} ({best_performers['eer']['value']:.4f})")
        
        # Target performance analysis
        if 'performance_report' in self.results and 'target_analysis' in self.results['performance_report']:
            print("\nTarget Performance Achievement:")
            for model, analysis in self.results['performance_report']['target_analysis'].items():
                status = "✓ PASSED" if analysis['overall_performance'] else "✗ FAILED"
                print(f"  {model}: {status}")
        
        print(f"\nSession ID: {self.session_id}")
        print(f"Total execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    # Create and run the complete pipeline
    pipeline = SignatureAuthenticationPipeline()
    pipeline.run_complete_pipeline()
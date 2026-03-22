"""
PCA Dimensionality Reduction and GBM Classification for Signature Authentication
Implements feature optimization and final classification components
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os


class PCAFeatureReducer:
    def __init__(self, n_components=None, variance_threshold=0.95):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, scale_features=True):
        """Fit PCA on training data"""
        X_processed = X.copy()
        
        # Scale features if requested
        if scale_features:
            X_processed = self.scaler.fit_transform(X_processed)
        
        # Determine number of components if not specified
        if self.n_components is None:
            # Use variance threshold to determine components
            self.n_components = self._determine_components(X_processed)
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_processed)
        self.is_fitted = True
        
        print(f"PCA fitted with {self.n_components} components")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        return self
        
    def _determine_components(self, X):
        """Determine optimal number of components based on variance threshold"""
        # Fit PCA with all components first
        pca_full = PCA()
        pca_full.fit(X)
        
        # Find number of components needed for variance threshold
        cumsum_ratio = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum_ratio >= self.variance_threshold) + 1
        
        # Ensure minimum and maximum bounds
        n_components = max(10, min(n_components, min(X.shape[1], X.shape[0] // 2)))
        
        return n_components
        
    def transform(self, X, scale_features=True):
        """Transform data using fitted PCA"""
        if not self.is_fitted:
            raise ValueError("PCA not fitted yet! Call fit() first.")
            
        X_processed = X.copy()
        
        # Scale features if requested
        if scale_features:
            X_processed = self.scaler.transform(X_processed)
            
        # Apply PCA transformation
        X_transformed = self.pca.transform(X_processed)
        
        return X_transformed
        
    def fit_transform(self, X, scale_features=True):
        """Fit PCA and transform data in one step"""
        self.fit(X, scale_features)
        return self.transform(X, scale_features)
        
    def inverse_transform(self, X_transformed, scale_features=True):
        """Inverse transform PCA data back to original space"""
        if not self.is_fitted:
            raise ValueError("PCA not fitted yet!")
            
        # Inverse PCA
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        
        # Inverse scaling if applied
        if scale_features:
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)
            
        return X_reconstructed
        
    def plot_explained_variance(self, save_path=None):
        """Plot explained variance ratio"""
        if not self.is_fitted:
            raise ValueError("PCA not fitted yet!")
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Individual explained variance
        axes[0].bar(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                    self.pca.explained_variance_ratio_)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Individual Explained Variance')
        
        # Cumulative explained variance
        cumsum_ratio = np.cumsum(self.pca.explained_variance_ratio_)
        axes[1].plot(range(1, len(cumsum_ratio) + 1), cumsum_ratio, 'bo-')
        axes[1].axhline(y=self.variance_threshold, color='r', linestyle='--', 
                       label=f'Threshold: {self.variance_threshold}')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance Ratio')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def get_component_importance(self):
        """Get importance of original features in principal components"""
        if not self.is_fitted:
            raise ValueError("PCA not fitted yet!")
            
        # Get loadings (components)
        components = self.pca.components_
        
        # Calculate feature importance as sum of absolute loadings
        feature_importance = np.sum(np.abs(components), axis=0)
        feature_importance = feature_importance / np.sum(feature_importance)
        
        return feature_importance
        
    def save_pca(self, filepath):
        """Save PCA model"""
        if not self.is_fitted:
            raise ValueError("PCA not fitted yet!")
            
        pca_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'n_components': self.n_components,
            'variance_threshold': self.variance_threshold,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(pca_data, filepath)
        print(f"PCA model saved to {filepath}")
        
    def load_pca(self, filepath):
        """Load PCA model"""
        pca_data = joblib.load(filepath)
        
        self.pca = pca_data['pca']
        self.scaler = pca_data['scaler']
        self.n_components = pca_data['n_components']
        self.variance_threshold = pca_data['variance_threshold']
        self.is_fitted = pca_data['is_fitted']
        
        print(f"PCA model loaded from {filepath}")


class GBMClassifier:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type.lower()
        self.model = None
        self.best_params = None
        self.is_fitted = False
        self.feature_importance = None
        
    def _get_model(self, **params):
        """Get model instance based on type"""
        if self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'random_state': 42
            }
            default_params.update(params)
            return xgb.XGBClassifier(**default_params)
            
        elif self.model_type == 'lightgbm':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': -1,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary',
                'random_state': 42
            }
            default_params.update(params)
            return lgb.LGBMClassifier(**default_params)
            
        elif self.model_type == 'sklearn':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'subsample': 0.8,
                'random_state': 42
            }
            default_params.update(params)
            return GradientBoostingClassifier(**default_params)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def fit(self, X_train, y_train, **params):
        """Fit GBM model"""
        self.model = self._get_model(**params)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
            
        print(f"{self.model_type.upper()} model trained successfully!")
        return self
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
            
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
            
        return self.model.predict_proba(X)
        
    def hyperparameter_tuning(self, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1):
        """Perform hyperparameter tuning"""
        
        if self.model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        elif self.model_type == 'lightgbm':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 63, 127],
                'min_child_samples': [10, 20, 30],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        else:  # sklearn
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            }
            
        # Perform grid search
        base_model = self._get_model()
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=cv, 
            scoring=scoring, 
            n_jobs=n_jobs,
            verbose=1
        )
        
        print("Starting hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.best_params
        
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
            
        # Predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # False Acceptance Rate (FAR) and False Rejection Rate (FRR)
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False negative rate
        
        # Equal Error Rate (EER)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        fnr = 1 - tpr
        eer_index = np.argmin(np.abs(fpr - fnr))
        eer = fpr[eer_index]
        
        # Classification report
        class_names = ['Forged', 'Genuine']
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'far': far,  # False Acceptance Rate
            'frr': frr,  # False Rejection Rate
            'eer': eer,  # Equal Error Rate
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_proba.tolist(),
            'true_labels': y_test.tolist()
        }
        
        return results
        
    def plot_feature_importance(self, feature_names=None, top_k=20, save_path=None):
        """Plot feature importance"""
        if self.feature_importance is None:
            print("Feature importance not available!")
            return
            
        # Get top-k important features
        if len(self.feature_importance) > top_k:
            top_indices = np.argsort(self.feature_importance)[-top_k:]
            top_importance = self.feature_importance[top_indices]
            
            if feature_names is not None:
                top_names = [feature_names[i] for i in top_indices]
            else:
                top_names = [f'Feature_{i}' for i in top_indices]
        else:
            top_importance = self.feature_importance
            top_names = feature_names if feature_names else [f'Feature_{i}' for i in range(len(top_importance))]
            
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_importance)), top_importance)
        plt.yticks(range(len(top_importance)), top_names)
        plt.xlabel('Feature Importance')
        plt.title(f'{self.model_type.upper()} - Top {len(top_importance)} Most Important Features')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_roc_curve(self, X_test, y_test, save_path=None):
        """Plot ROC curve"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
            
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_type.upper()} - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        """Plot confusion matrix"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
            
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Forged', 'Genuine'],
                    yticklabels=['Forged', 'Genuine'])
        plt.title(f'{self.model_type.upper()} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def cross_validate(self, X, y, cv=5, scoring='accuracy'):
        """Perform cross-validation"""
        if not self.is_fitted:
            # Use default model for cross-validation
            model = self._get_model()
        else:
            model = self.model
            
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        print(f"Cross-validation {scoring} scores: {scores}")
        print(f"Mean {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
        
    def save_model(self, filepath):
        """Save GBM model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
            
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'is_fitted': self.is_fitted,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        print(f"GBM model saved to {filepath}")
        
    def load_model(self, filepath):
        """Load GBM model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.best_params = model_data['best_params']
        self.is_fitted = model_data['is_fitted']
        self.feature_importance = model_data.get('feature_importance', None)
        
        print(f"GBM model loaded from {filepath}")


if __name__ == "__main__":
    print("PCA Feature Reducer and GBM Classifier modules ready!")
    print("Available models:")
    print("- XGBoost")
    print("- LightGBM") 
    print("- Scikit-learn GradientBoosting")
    
    # Example usage
    print("\nExample usage:")
    print("pca = PCAFeatureReducer(variance_threshold=0.95)")
    print("gbm = GBMClassifier(model_type='xgboost')")
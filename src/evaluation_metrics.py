"""
Comprehensive Evaluation Metrics for Signature-Based Biometric Authentication
Implements all required performance metrics and generates detailed reports
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import json
import os
from datetime import datetime
import pandas as pd


class BiometricEvaluator:
    def __init__(self):
        self.results = {}
        self.evaluation_timestamp = None
        
    def calculate_biometric_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive biometric authentication metrics"""
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1)  # Genuine = 1
        recall = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Biometric-specific metrics
        # False Acceptance Rate (FAR) - Forged accepted as genuine
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False Rejection Rate (FRR) - Genuine rejected as forged  
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # True Accept Rate (TAR) = 1 - FRR
        tar = 1 - frr
        
        # True Reject Rate (TRR) = 1 - FAR
        trr = 1 - far
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'far': float(far),  # False Acceptance Rate
            'frr': float(frr),  # False Rejection Rate
            'tar': float(tar),  # True Accept Rate
            'trr': float(trr),  # True Reject Rate
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_genuine': int(tp + fn),
            'total_forged': int(tn + fp),
            'confusion_matrix': cm.tolist()
        }
        
        # If probabilities are available, calculate additional metrics
        if y_pred_proba is not None:
            # ROC AUC
            roc_auc = auc(*roc_curve(y_true, y_pred_proba)[:2])
            
            # Average Precision (Area under PR curve)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            
            # Equal Error Rate (EER)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            fnr = 1 - tpr
            eer_index = np.argmin(np.abs(fpr - fnr))
            eer = fpr[eer_index]
            eer_threshold = thresholds[eer_index]
            
            metrics.update({
                'roc_auc': float(roc_auc),
                'average_precision': float(avg_precision),
                'eer': float(eer),  # Equal Error Rate
                'eer_threshold': float(eer_threshold)
            })
            
        return metrics
        
    def calculate_det_curve(self, y_true, y_scores):
        """Calculate Detection Error Tradeoff (DET) curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr  # False Negative Rate
        
        return fpr, fnr, thresholds
        
    def find_optimal_threshold(self, y_true, y_scores, criterion='eer'):
        """Find optimal threshold based on different criteria"""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        if criterion == 'eer':
            # Equal Error Rate - where FAR = FRR
            fnr = 1 - tpr
            eer_index = np.argmin(np.abs(fpr - fnr))
            optimal_threshold = thresholds[eer_index]
            
        elif criterion == 'youden':
            # Youden's J statistic - maximizes (TPR - FPR)
            j_scores = tpr - fpr
            youden_index = np.argmax(j_scores)
            optimal_threshold = thresholds[youden_index]
            
        elif criterion == 'f1':
            # Maximize F1 score
            f1_scores = []
            for thresh in thresholds:
                y_pred_thresh = (y_scores >= thresh).astype(int)
                f1 = f1_score(y_true, y_pred_thresh)
                f1_scores.append(f1)
            f1_index = np.argmax(f1_scores)
            optimal_threshold = thresholds[f1_index]
            
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
            
        return optimal_threshold
        
    def evaluate_system_components(self, component_results):
        """Evaluate individual system components"""
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        for component_name, results in component_results.items():
            if 'y_true' in results and 'y_pred' in results:
                y_true = results['y_true']
                y_pred = results['y_pred']
                y_pred_proba = results.get('y_pred_proba', None)
                
                metrics = self.calculate_biometric_metrics(y_true, y_pred, y_pred_proba)
                evaluation['components'][component_name] = metrics
                
        self.results = evaluation
        return evaluation
        
    def compare_models(self, model_results):
        """Compare multiple models performance"""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'ranking': {}
        }
        
        metrics_to_rank = ['accuracy', 'f1_score', 'roc_auc', 'eer']
        rankings = {metric: [] for metric in metrics_to_rank}
        
        for model_name, results in model_results.items():
            y_true = results['y_true']
            y_pred = results['y_pred']
            y_pred_proba = results.get('y_pred_proba', None)
            
            metrics = self.calculate_biometric_metrics(y_true, y_pred, y_pred_proba)
            comparison['models'][model_name] = metrics
            
            # Collect metrics for ranking
            for metric in metrics_to_rank:
                if metric in metrics:
                    rankings[metric].append((model_name, metrics[metric]))
        
        # Rank models by different metrics
        for metric in metrics_to_rank:
            if rankings[metric]:
                if metric == 'eer':  # Lower is better for EER
                    sorted_models = sorted(rankings[metric], key=lambda x: x[1])
                else:  # Higher is better for others
                    sorted_models = sorted(rankings[metric], key=lambda x: x[1], reverse=True)
                comparison['ranking'][metric] = sorted_models
                
        return comparison
        
    def plot_roc_curves(self, model_results, save_path=None, figsize=(10, 8)):
        """Plot ROC curves for multiple models"""
        plt.figure(figsize=figsize)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (model_name, results) in enumerate(model_results.items()):
            y_true = results['y_true']
            y_pred_proba = results.get('y_pred_proba', None)
            
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FAR)')
        plt.ylabel('True Positive Rate (1-FRR)')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_det_curves(self, model_results, save_path=None, figsize=(10, 8)):
        """Plot DET (Detection Error Tradeoff) curves"""
        plt.figure(figsize=figsize)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (model_name, results) in enumerate(model_results.items()):
            y_true = results['y_true']
            y_pred_proba = results.get('y_pred_proba', None)
            
            if y_pred_proba is not None:
                far, frr, _ = self.calculate_det_curve(y_true, y_pred_proba)
                
                plt.plot(far * 100, frr * 100, color=colors[i % len(colors)], lw=2,
                        label=f'{model_name}')
        
        # EER line (where FAR = FRR)
        plt.plot([0, 50], [0, 50], color='black', lw=1, linestyle='--', alpha=0.5)
        
        plt.xlabel('False Accept Rate (%)')
        plt.ylabel('False Reject Rate (%)')
        plt.title('DET Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.xlim([0, 50])
        plt.ylim([0, 50])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_precision_recall_curves(self, model_results, save_path=None, figsize=(10, 8)):
        """Plot Precision-Recall curves"""
        plt.figure(figsize=figsize)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (model_name, results) in enumerate(model_results.items()):
            y_true = results['y_true']
            y_pred_proba = results.get('y_pred_proba', None)
            
            if y_pred_proba is not None:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                avg_precision = average_precision_score(y_true, y_pred_proba)
                
                plt.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                        label=f'{model_name} (AP = {avg_precision:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrices(self, model_results, save_path=None, figsize=(15, 5)):
        """Plot confusion matrices for multiple models"""
        n_models = len(model_results)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(model_results.items()):
            y_true = results['y_true']
            y_pred = results['y_pred']
            
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Forged', 'Genuine'],
                       yticklabels=['Forged', 'Genuine'],
                       ax=axes[i])
            axes[i].set_title(f'{model_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_metrics_comparison(self, model_results, metrics=['accuracy', 'precision', 'recall', 'f1_score'],
                               save_path=None, figsize=(12, 8)):
        """Plot bar chart comparing different metrics across models"""
        
        # Collect data
        models = []
        metric_values = {metric: [] for metric in metrics}
        
        for model_name, results in model_results.items():
            y_true = results['y_true']
            y_pred = results['y_pred']
            y_pred_proba = results.get('y_pred_proba', None)
            
            model_metrics = self.calculate_biometric_metrics(y_true, y_pred, y_pred_proba)
            
            models.append(model_name)
            for metric in metrics:
                if metric in model_metrics:
                    metric_values[metric].append(model_metrics[metric])
                else:
                    metric_values[metric].append(0)
        
        # Create plot
        x = np.arange(len(models))
        width = 0.8 / len(metrics)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, metric_values[metric], width, 
                   label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_performance_report(self, model_results, save_dir=None):
        """Generate comprehensive performance report"""
        
        report = {
            'evaluation_summary': {
                'timestamp': datetime.now().isoformat(),
                'num_models_evaluated': len(model_results),
                'models': list(model_results.keys())
            },
            'individual_results': {},
            'best_performers': {},
            'statistical_summary': {}
        }
        
        # Evaluate each model
        all_metrics = []
        for model_name, results in model_results.items():
            y_true = results['y_true']
            y_pred = results['y_pred']
            y_pred_proba = results.get('y_pred_proba', None)
            
            metrics = self.calculate_biometric_metrics(y_true, y_pred, y_pred_proba)
            report['individual_results'][model_name] = metrics
            all_metrics.append(metrics)
        
        # Find best performers
        key_metrics = ['accuracy', 'f1_score', 'roc_auc', 'eer']
        for metric in key_metrics:
            values = [(name, report['individual_results'][name].get(metric, 0)) 
                     for name in model_results.keys()]
            
            if metric == 'eer':  # Lower is better
                best = min(values, key=lambda x: x[1])
            else:  # Higher is better
                best = max(values, key=lambda x: x[1])
                
            report['best_performers'][metric] = {
                'model': best[0],
                'value': best[1]
            }
        
        # Statistical summary
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            summary_stats = metrics_df.describe().to_dict()
            report['statistical_summary'] = summary_stats
        
        # Target performance analysis
        target_metrics = {
            'accuracy': 0.94,  # 94%+
            'far': 0.001,      # < 0.1%
            'frr': 0.001,      # < 0.1%
            'eer': 0.0015      # ~0.15%
        }
        
        report['target_analysis'] = {}
        for model_name, metrics in report['individual_results'].items():
            meets_targets = {}
            for target_metric, target_value in target_metrics.items():
                if target_metric in metrics:
                    if target_metric in ['accuracy']:  # Higher is better
                        meets_targets[target_metric] = metrics[target_metric] >= target_value
                    else:  # Lower is better
                        meets_targets[target_metric] = metrics[target_metric] <= target_value
                else:
                    meets_targets[target_metric] = False
                    
            report['target_analysis'][model_name] = {
                'meets_targets': meets_targets,
                'overall_performance': all(meets_targets.values())
            }
        
        # Save report
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save JSON report
            report_path = os.path.join(save_dir, 'performance_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save CSV summary
            if all_metrics:
                metrics_df = pd.DataFrame(all_metrics)
                metrics_df.index = list(model_results.keys())
                metrics_df.to_csv(os.path.join(save_dir, 'metrics_summary.csv'))
            
            print(f"Performance report saved to {save_dir}")
        
        return report
        
    def save_evaluation_results(self, filepath):
        """Save evaluation results to JSON file"""
        if self.results:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"Evaluation results saved to {filepath}")
        else:
            print("No evaluation results to save!")


if __name__ == "__main__":
    print("Biometric Evaluation System ready!")
    print("Available metrics:")
    print("- Accuracy, Precision, Recall, F1-Score")
    print("- False Acceptance Rate (FAR)")
    print("- False Rejection Rate (FRR)")  
    print("- Equal Error Rate (EER)")
    print("- ROC AUC, Average Precision")
    print("- True Accept Rate (TAR), True Reject Rate (TRR)")
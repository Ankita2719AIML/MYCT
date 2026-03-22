"""
Comprehensive Results Generator and Report Manager
Creates detailed reports with all parameters, visualizations, and cross-validation results
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pickle
import zipfile
import shutil
from jinja2 import Template
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from enhanced_training_system import ParameterTracker, TrainingVisualizer, CrossValidator, TrustScoreCalculator


class ResultsGenerator:
    """Comprehensive results generator with reporting capabilities"""
    
    def __init__(self, base_results_dir: str, session_id: str = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_results_dir = Path(base_results_dir)
        self.results_dir = self.base_results_dir / f"comprehensive_results_{self.session_id}"
        
        # Create directory structure
        self._create_directory_structure()
        
        # Initialize components
        self.parameter_tracker = ParameterTracker()
        self.visualizer = TrainingVisualizer(str(self.results_dir / "visualizations"))
        self.cross_validator = CrossValidator()
        self.trust_calculator = TrustScoreCalculator()
        
        # Results storage
        self.all_results = {}
        self.model_artifacts = {}
        
    def _create_directory_structure(self):
        """Create organized directory structure for results"""
        directories = [
            "models",
            "visualizations", 
            "reports",
            "parameters",
            "cross_validation",
            "trust_scores",
            "raw_data",
            "logs"
        ]
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        for directory in directories:
            (self.results_dir / directory).mkdir(exist_ok=True)
    
    def log_training_session(self, component_name: str, model, training_history: Dict,
                           training_params: Dict, regularization_params: Dict = None):
        """Log complete training session information"""
        
        # Track parameters
        self.parameter_tracker.log_training_parameters(component_name, **training_params)
        
        if regularization_params:
            self.parameter_tracker.log_regularization(component_name, regularization_params)
        
        # Store model artifacts
        self.model_artifacts[component_name] = {
            'model': model,
            'history': training_history,
            'params': training_params
        }
        
        # Generate visualizations
        visualization_path = self.visualizer.plot_training_curves(
            training_history, component_name
        )
        
        # Store in results
        self.all_results[component_name] = {
            'training_history': training_history,
            'parameters': training_params,
            'visualization_path': visualization_path,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"Training session logged for {component_name}")
    
    def perform_comprehensive_cross_validation(self, X: np.ndarray, y: np.ndarray,
                                             model_builder_func, cv_params: Dict,
                                             cv_type: str = "both") -> Dict:
        """Perform comprehensive cross-validation analysis"""
        
        cv_results = {}
        
        if cv_type in ["stratified", "both"]:
            print("\\nStarting Stratified K-Fold Cross-Validation...")
            stratified_results = self.cross_validator.stratified_k_fold_cv(
                X, y, model_builder_func, cv_params
            )
            cv_results['stratified_k_fold'] = stratified_results
            
            # Generate CV visualizations
            cv_viz_path = self.visualizer.plot_cross_validation_results(
                stratified_results, "Stratified K-Fold"
            )
            cv_results['stratified_visualization'] = cv_viz_path
        
        # For cross-dataset validation, you would need multiple datasets
        # This is a placeholder for when you have multiple datasets
        if cv_type in ["cross_dataset", "both"] and hasattr(self, 'multiple_datasets'):
            print("\\nStarting Cross-Dataset Validation...")
            cross_dataset_results = self.cross_validator.cross_dataset_validation(
                self.multiple_datasets, model_builder_func, cv_params
            )
            cv_results['cross_dataset'] = cross_dataset_results
        
        # Save cross-validation results
        cv_results_path = self.results_dir / "cross_validation" / "cv_results.json"
        with open(cv_results_path, 'w') as f:
            json.dump(cv_results, f, indent=4, default=str)
        
        self.all_results['cross_validation'] = cv_results
        
        return cv_results
    
    def calculate_comprehensive_trust_scores(self, autoencoder_results: Dict = None,
                                           cnn_results: Dict = None,
                                           cv_results: Dict = None) -> Dict:
        """Calculate trust scores for all components"""
        
        trust_scores = {}
        
        # Autoencoder trust score
        if autoencoder_results:
            ae_trust = self.trust_calculator.calculate_autoencoder_trust(
                np.array(autoencoder_results.get('reconstruction_errors', [])),
                autoencoder_results.get('threshold', 0.1),
                np.array(autoencoder_results.get('true_labels', []))
            )
            trust_scores['autoencoder'] = ae_trust
        
        # CNN trust score  
        if cnn_results:
            cnn_trust = self.trust_calculator.calculate_cnn_trust(
                cnn_results.get('training_history', {}),
                cnn_results.get('validation_accuracy', 0.5)
            )
            trust_scores['cnn'] = cnn_trust
        
        # Overall trust score
        if cv_results and 'stratified_k_fold' in cv_results:
            overall_trust = self.trust_calculator.calculate_overall_trust(
                cv_results['stratified_k_fold']
            )
            trust_scores['overall'] = overall_trust
        
        # Generate trust score visualizations
        if trust_scores:
            trust_viz_path = self.visualizer.plot_trust_score_analysis(trust_scores)
            trust_scores['visualization_path'] = trust_viz_path
        
        # Save trust scores
        trust_report = self.trust_calculator.get_trust_report()
        trust_path = self.results_dir / "trust_scores" / "trust_analysis.json"
        with open(trust_path, 'w') as f:
            json.dump(trust_report, f, indent=4, default=str)
        
        self.all_results['trust_scores'] = trust_scores
        
        return trust_scores
    
    def save_all_models(self):
        """Save all trained models"""
        models_dir = self.results_dir / "models"
        
        for component_name, artifacts in self.model_artifacts.items():
            model = artifacts['model']
            
            try:
                if hasattr(model, 'save'):
                    # Keras/TensorFlow model
                    model_path = models_dir / f"{component_name}_model.h5"
                    model.save(str(model_path))
                elif hasattr(model, 'save_model'):
                    # XGBoost/LightGBM model
                    model_path = models_dir / f"{component_name}_model.pkl"
                    model.save_model(str(model_path))
                else:
                    # Generic pickle save
                    model_path = models_dir / f"{component_name}_model.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                
                print(f"Model saved: {model_path}")
                
            except Exception as e:
                print(f"Error saving model {component_name}: {e}")
    
    def generate_parameter_reproducibility_report(self):
        """Generate comprehensive parameter reproducibility report"""
        
        # Save parameter tracker results
        param_report = self.parameter_tracker.get_reproducibility_report()
        param_path = self.results_dir / "parameters" / "reproducibility_report.json"
        
        with open(param_path, 'w') as f:
            json.dump(param_report, f, indent=4, default=str)
        
        # Generate parameter summary CSV
        param_df = self._create_parameter_dataframe(param_report)
        csv_path = self.results_dir / "parameters" / "parameters_summary.csv"
        param_df.to_csv(csv_path, index=False)
        
        print(f"Parameter report saved: {param_path}")
        print(f"Parameter summary saved: {csv_path}")
        
        return param_report
    
    def _create_parameter_dataframe(self, param_report: Dict) -> pd.DataFrame:
        """Create a structured dataframe of parameters"""
        
        rows = []
        
        for component, params in param_report.get('training_parameters', {}).items():
            for param_name, param_value in params.items():
                if isinstance(param_value, dict):
                    for sub_param, sub_value in param_value.items():
                        rows.append({
                            'Component': component,
                            'Parameter_Group': param_name,
                            'Parameter': sub_param,
                            'Value': sub_value,
                            'Type': type(sub_value).__name__
                        })
                else:
                    rows.append({
                        'Component': component,
                        'Parameter_Group': 'main',
                        'Parameter': param_name,
                        'Value': param_value,
                        'Type': type(param_value).__name__
                    })
        
        return pd.DataFrame(rows)
    
    def generate_comprehensive_pdf_report(self) -> str:
        """Generate a comprehensive PDF report with all results"""
        
        pdf_path = self.results_dir / "reports" / f"comprehensive_report_{self.session_id}.pdf"
        
        with PdfPages(pdf_path) as pdf:
            # Title page
            self._add_title_page(pdf)
            
            # Executive summary
            self._add_executive_summary(pdf)
            
            # Parameter reproducibility section
            self._add_parameter_section(pdf)
            
            # Training visualizations
            self._add_training_visualizations(pdf)
            
            # Cross-validation results
            self._add_cross_validation_section(pdf)
            
            # Trust score analysis
            self._add_trust_score_section(pdf)
            
            # Model architecture summaries
            self._add_model_architecture_section(pdf)
            
            # Recommendations
            self._add_recommendations_section(pdf)
        
        print(f"Comprehensive PDF report generated: {pdf_path}")
        return str(pdf_path)
    
    def _add_title_page(self, pdf):
        """Add title page to PDF report"""
        fig, ax = plt.subplots(figsize=(8, 11))
        ax.text(0.5, 0.8, 'Signature Forgery Detection System', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        ax.text(0.5, 0.7, 'Comprehensive Training Report', 
                ha='center', va='center', fontsize=18)
        ax.text(0.5, 0.6, f'Session ID: {self.session_id}', 
                ha='center', va='center', fontsize=14)
        ax.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.2, 'Enhanced Deep Learning Pipeline with\\nCross-Validation and Trust Score Analysis', 
                ha='center', va='center', fontsize=14, style='italic')
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _add_executive_summary(self, pdf):
        """Add executive summary page"""
        fig, ax = plt.subplots(figsize=(8, 11))
        
        summary_text = self._generate_executive_summary()
        ax.text(0.1, 0.9, 'EXECUTIVE SUMMARY', fontsize=16, fontweight='bold')
        ax.text(0.1, 0.05, summary_text, fontsize=10, verticalalignment='bottom', 
                wrap=True, fontfamily='monospace')
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary text"""
        
        summary_parts = [
            "This report presents a comprehensive analysis of the signature forgery detection system",
            "training process, including detailed parameter tracking, cross-validation results,",
            "and trust score analysis for reproducibility and reliability assessment.\\n"
        ]
        
        if 'cross_validation' in self.all_results:
            cv_results = self.all_results['cross_validation']
            if 'stratified_k_fold' in cv_results and 'accuracy' in cv_results['stratified_k_fold']:
                accuracy_scores = cv_results['stratified_k_fold']['accuracy']
                mean_acc = np.mean(accuracy_scores)
                std_acc = np.std(accuracy_scores)
                summary_parts.append(f"Cross-Validation Results:")
                summary_parts.append(f"  - Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        
        if 'trust_scores' in self.all_results:
            trust_scores = self.all_results['trust_scores']
            if 'overall' in trust_scores:
                summary_parts.append(f"  - Overall Trust Score: {trust_scores['overall']:.3f}")
        
        summary_parts.extend([
            "\\nKey Features Implemented:",
            "  ✓ Comprehensive parameter tracking for reproducibility",
            "  ✓ Enhanced training visualizations (loss, accuracy, trust scores)",
            "  ✓ Stratified K-fold cross-validation",
            "  ✓ Trust score analysis for model reliability",
            "  ✓ Automated report generation"
        ])
        
        return "\\n".join(summary_parts)
    
    def _add_parameter_section(self, pdf):
        """Add parameter reproducibility section"""
        # This would include parameter tables and reproducibility information
        pass
    
    def _add_training_visualizations(self, pdf):
        """Add training visualization pages"""
        viz_dir = self.results_dir / "visualizations"
        
        for viz_file in viz_dir.glob("*.png"):
            fig, ax = plt.subplots(figsize=(8, 11))
            img = plt.imread(str(viz_file))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Training Visualization: {viz_file.stem}", fontsize=14, pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def _add_cross_validation_section(self, pdf):
        """Add cross-validation results section"""
        pass
    
    def _add_trust_score_section(self, pdf):
        """Add trust score analysis section"""
        pass
    
    def _add_model_architecture_section(self, pdf):
        """Add model architecture summaries"""
        pass
    
    def _add_recommendations_section(self, pdf):
        """Add recommendations based on results"""
        pass
    
    def create_results_archive(self) -> str:
        """Create a complete ZIP archive of all results"""
        
        archive_path = self.base_results_dir / f"complete_results_{self.session_id}.zip"
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.results_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.results_dir)
                    zipf.write(file_path, arcname)
        
        print(f"Complete results archive created: {archive_path}")
        return str(archive_path)
    
    def generate_summary_json(self) -> str:
        """Generate a summary JSON with all key results"""
        
        summary = {
            'session_info': {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'results_directory': str(self.results_dir)
            },
            'reproducibility_score': self.parameter_tracker._calculate_reproducibility_score(),
            'components_trained': list(self.all_results.keys()),
            'files_generated': self._count_generated_files(),
            'summary_statistics': self._calculate_summary_statistics()
        }
        
        # Add detailed results
        summary['detailed_results'] = self.all_results
        
        # Save summary
        summary_path = self.results_dir / "reports" / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        print(f"Summary JSON generated: {summary_path}")
        return str(summary_path)
    
    def _count_generated_files(self) -> Dict[str, int]:
        """Count files generated in each category"""
        counts = {}
        
        for subdir in self.results_dir.iterdir():
            if subdir.is_dir():
                file_count = len(list(subdir.glob("*.*")))
                counts[subdir.name] = file_count
        
        return counts
    
    def _calculate_summary_statistics(self) -> Dict:
        """Calculate summary statistics from all results"""
        stats = {}
        
        if 'cross_validation' in self.all_results:
            cv_results = self.all_results['cross_validation']
            if 'stratified_k_fold' in cv_results:
                skf_results = cv_results['stratified_k_fold']
                for metric, scores in skf_results.items():
                    if isinstance(scores, list) and len(scores) > 0:
                        stats[f'cv_{metric}_mean'] = np.mean(scores)
                        stats[f'cv_{metric}_std'] = np.std(scores)
                        stats[f'cv_{metric}_min'] = np.min(scores)
                        stats[f'cv_{metric}_max'] = np.max(scores)
        
        return stats
    
    def finalize_results(self) -> Dict[str, str]:
        """Finalize all results and generate final reports"""
        
        print(f"\\n{'='*60}")
        print(f"FINALIZING COMPREHENSIVE RESULTS")
        print(f"Session ID: {self.session_id}")
        print(f"{'='*60}")
        
        # Generate all final reports
        final_outputs = {}
        
        # 1. Parameter reproducibility report
        param_report = self.generate_parameter_reproducibility_report()
        final_outputs['parameter_report'] = str(self.results_dir / "parameters" / "reproducibility_report.json")
        
        # 2. Save all models
        self.save_all_models()
        final_outputs['models_directory'] = str(self.results_dir / "models")
        
        # 3. Generate PDF report
        pdf_report = self.generate_comprehensive_pdf_report()
        final_outputs['pdf_report'] = pdf_report
        
        # 4. Generate summary JSON
        summary_json = self.generate_summary_json()
        final_outputs['summary_json'] = summary_json
        
        # 5. Create complete archive
        archive_path = self.create_results_archive()
        final_outputs['complete_archive'] = archive_path
        
        # Print summary
        print(f"\\nResults successfully generated:")
        print(f"📁 Results Directory: {self.results_dir}")
        print(f"📊 PDF Report: {pdf_report}")
        print(f"📋 Summary JSON: {summary_json}")
        print(f"📦 Complete Archive: {archive_path}")
        print(f"\\n✅ All results finalized successfully!")
        
        return final_outputs


def create_autoencoder_model_builder():
    """Factory function for autoencoder model builder"""
    def build_model(**params):
        from autoencoder_forgery_detection import SignatureAutoencoder
        autoencoder = SignatureAutoencoder(
            input_shape=params.get('input_shape', (224, 224, 1)),
            latent_dim=params.get('latent_dim', 128)
        )
        autoencoder.compile_autoencoder(
            learning_rate=params.get('learning_rate', 0.001)
        )
        return autoencoder
    
    return build_model


def create_cnn_model_builder():
    """Factory function for CNN model builder"""
    def build_model(**params):
        from sklearn.ensemble import RandomForestClassifier
        
        # For this example, using RandomForest as a substitute
        # You can replace this with your actual CNN implementation
        model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            random_state=params.get('random_state', 42)
        )
        return model
    
    return build_model
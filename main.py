"""
Main Execution Script for Signature-Based Biometric Authentication System
Run this script to execute the complete pipeline with all components
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from training_pipeline import SignatureAuthenticationPipeline


def load_config_from_file(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found!")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing configuration file: {e}")
        return None


def create_sample_config(save_file=False):
    """Create a sample configuration file"""
    config = {
        "data": {
            "dataset_path": "C:/Users/priya/OneDrive/Desktop/MYCT_TASK/archive/MYCT_pics_updated",
            "image_size": [224, 224],
            "validation_split": 0.2,
            "test_split": 0.2,
            "random_state": 42
        },
        "preprocessing": {
            "save_intermediate": True,
            "noise_reduction": True,
            "orientation_correction": True
        },
        "cnn": {
            "epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.001,
            "early_stopping_patience": 10
        },
        "autoencoder": {
            "epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.001,
            "latent_dim": 128,
            "anomaly_threshold_percentile": 95
        },
        "explainability": {
            "shap_samples": 50,
            "gradcam_layer": None,
            "feature_importance_threshold": 50
        },
        "pca": {
            "variance_threshold": 0.95,
            "scale_features": True
        },
        "gbm": {
            "model_type": "xgboost",
            "hyperparameter_tuning": True,
            "cv_folds": 5
        },
        "output": {
            "save_models": True,
            "save_visualizations": True,
            "save_results": True,
            "results_dir": "C:/Users/priya/OneDrive/Desktop/MYCT_TASK/results"
        }
    }
    
    if save_file:
        config_path = "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Sample configuration file created: {config_path}")
    
    return config


def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        SIGNATURE-BASED BIOMETRIC AUTHENTICATION SYSTEM                      ║
║                     Using Deep Learning & Explainable AI                    ║
║                                                                              ║
║  Features:                                                                   ║
║  • CNN Feature Extraction                                                    ║
║  • Autoencoder Forgery Detection                                             ║
║  • SHAP Explainability                                                       ║
║  • Grad-CAM Visualization                                                    ║
║  • PCA Dimensionality Reduction                                              ║
║  • Gradient Boosting Classification                                          ║
║  • Comprehensive Evaluation                                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_system_info():
    """Print system information"""
    import tensorflow as tf
    import numpy as np
    import sklearn
    
    print("\nSystem Information:")
    print("-" * 40)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"Scikit-learn Version: {sklearn.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU') != []}")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 40)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Signature-Based Biometric Authentication System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run with default configuration
  python main.py --config config.json     # Run with custom configuration
  python main.py --create-config         # Create sample configuration file
  python main.py --quick                 # Run with reduced parameters for testing
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create a sample configuration file'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run with reduced parameters for quick testing'
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        help='Override dataset path in configuration'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        help='Override results directory in configuration'
    )
    
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Disable visualization generation'
    )
    
    parser.add_argument(
        '--no-models',
        action='store_true',
        help='Disable model saving'
    )
    
    parser.add_argument(
        '--use-existing-models',
        action='store_true',
        help='Use existing trained models if available'
    )
    
    args = parser.parse_args()
    
    print_banner()
    print_system_info()
    
    # Handle create-config option
    if args.create_config:
        create_sample_config(save_file=True)
        return
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = load_config_from_file(args.config)
        if config is None:
            return
        print(f"Configuration loaded from: {args.config}")
    else:
        config = None  # Use default configuration
        print("Using default configuration")
    
    # Apply quick testing parameters
    if args.quick:
        if config is None:
            config = create_sample_config()  # Use full default config as base
        
        # Reduce epochs and samples for quick testing
        config.setdefault('cnn', {}).update({
            'epochs': 2,
            'batch_size': 16
        })
        config.setdefault('autoencoder', {}).update({
            'epochs': 2,
            'batch_size': 16
        })
        config.setdefault('explainability', {}).update({
            'shap_samples': 10
        })
        config.setdefault('gbm', {}).update({
            'hyperparameter_tuning': False
        })
        
        print("Quick mode enabled: Reduced parameters for fast execution")
    
    # Apply command line overrides
    if config is None:
        config = {}
    
    if args.dataset_path:
        config.setdefault('data', {})['dataset_path'] = args.dataset_path
        print(f"Dataset path override: {args.dataset_path}")
    
    if args.results_dir:
        config.setdefault('output', {})['results_dir'] = args.results_dir
        print(f"Results directory override: {args.results_dir}")
    
    if args.no_visualization:
        config.setdefault('output', {})['save_visualizations'] = False
        print("Visualization generation disabled")
    
    if args.no_models:
        config.setdefault('output', {})['save_models'] = False
        print("Model saving disabled")
    
    if args.use_existing_models:
        config.setdefault('training', {})['use_existing_models'] = True
        print("Will use existing trained models if available")
    
    # Initialize and run pipeline
    try:
        print("\nInitializing pipeline...")
        pipeline = SignatureAuthenticationPipeline(config)
        
        print("Starting pipeline execution...")
        pipeline.run_complete_pipeline()
        
        print("\n" + "="*80)
        print("EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Print final results summary
        if 'performance_report' in pipeline.results:
            print("\nFINAL PERFORMANCE SUMMARY:")
            print("-" * 50)
            
            best_performers = pipeline.results['performance_report'].get('best_performers', {})
            
            for metric, info in best_performers.items():
                print(f"{metric.upper()}: {info.get('model', 'N/A')} ({info.get('value', 0):.4f})")
            
            # Check if target performance is met
            target_analysis = pipeline.results['performance_report'].get('target_analysis', {})
            models_meeting_targets = sum(1 for analysis in target_analysis.values() 
                                       if analysis.get('overall_performance', False))
            
            print(f"\nModels meeting all target criteria: {models_meeting_targets}/{len(target_analysis)}")
            
            if models_meeting_targets > 0:
                print("🎉 Target performance achieved!")
            else:
                print("⚠️  Target performance not fully achieved. Consider parameter tuning.")
        
        print(f"\nResults location: {pipeline.config['output']['results_dir']}")
        print(f"Session ID: {pipeline.session_id}")
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\nExecution failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
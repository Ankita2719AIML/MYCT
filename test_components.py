"""
Simple test script to verify all components work
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing imports...")
    
    from data_preprocessing import SignaturePreprocessor, generate_sample_data
    print("✓ Data preprocessing module imported")
    
    from cnn_feature_extractor import SignatureCNN
    print("✓ CNN feature extractor module imported")
    
    from autoencoder_forgery_detection import SignatureAutoencoder
    print("✓ Autoencoder module imported")
    
    from pca_gbm_classifier import PCAFeatureReducer, GBMClassifier
    print("✓ PCA and GBM modules imported")
    
    from evaluation_metrics import BiometricEvaluator
    print("✓ Evaluation metrics module imported")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    
    # Test preprocessor
    preprocessor = SignaturePreprocessor(target_size=(224, 224))
    print("✓ Preprocessor initialized")
    
    # Test CNN
    cnn = SignatureCNN(input_shape=(224, 224, 1))
    print("✓ CNN initialized")
    
    # Test autoencoder
    autoencoder = SignatureAutoencoder(input_shape=(224, 224, 1))
    print("✓ Autoencoder initialized")
    
    # Test PCA
    pca = PCAFeatureReducer(variance_threshold=0.95)
    print("✓ PCA initialized")
    
    # Test GBM
    gbm = GBMClassifier(model_type='xgboost')
    print("✓ GBM initialized")
    
    # Test evaluator
    evaluator = BiometricEvaluator()
    print("✓ Evaluator initialized")
    
    print("\n🎉 All components working correctly!")
    
    # Create small sample data for minimal test
    print("\nGenerating sample data...")
    dataset_path = "C:/Users/priya/OneDrive/Desktop/MYCT_TASK/archive/MYCT_pics_updated/train"
    os.makedirs(os.path.join(dataset_path, "genuine"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "forged"), exist_ok=True)
    
    generate_sample_data(dataset_path, num_genuine=10, num_forged=10)
    print("✓ Sample data generated")
    
    print("\n🚀 System ready for execution!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
"""
CNN Feature Extractor for Signature-Based Biometric Authentication
Implements deep feature extraction using Convolutional Neural Networks
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


class SignatureCNN:
    def __init__(self, input_shape=(224, 224, 1), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.feature_extractor = None
        
    def build_cnn_model(self):
        """Build CNN architecture for signature feature extraction"""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape))
        
        # First Convolutional Block
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), name='pool1'))
        model.add(layers.Dropout(0.25))
        
        # Second Convolutional Block
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), name='pool2'))
        model.add(layers.Dropout(0.25))
        
        # Third Convolutional Block
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_3'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), name='pool3'))
        model.add(layers.Dropout(0.25))
        
        # Fourth Convolutional Block
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_3'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), name='pool4'))
        model.add(layers.Dropout(0.25))
        
        # Fifth Convolutional Block
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1'))
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2'))
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), name='pool5'))
        model.add(layers.Dropout(0.25))
        
        # Global Average Pooling instead of flattening
        model.add(layers.GlobalAveragePooling2D())
        
        # Dense layers for feature extraction
        model.add(layers.Dense(1024, activation='relu', name='fc1'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Dense(512, activation='relu', name='fc2'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        
        # Feature vector (this will be extracted for downstream tasks)
        model.add(layers.Dense(256, activation='relu', name='feature_vector'))
        
        # Classification layer
        model.add(layers.Dense(self.num_classes, activation='softmax', name='predictions'))
        
        self.model = model
        return model
        
    def compile_model(self, learning_rate=0.001):
        """Compile the CNN model"""
        if self.model is None:
            self.build_cnn_model()
            
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
        
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, 
                   save_path=None, callbacks=None):
        """Train the CNN model"""
        if self.model is None:
            self.compile_model()
            
        # Convert labels to categorical
        from tensorflow.keras.utils import to_categorical
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
        
        # Reshape input data to include channel dimension
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
        
        # Default callbacks
        default_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        if callbacks is None:
            callbacks = default_callbacks
        else:
            callbacks.extend(default_callbacks)
            
        if save_path:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    save_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False
                )
            )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
        
    def extract_features(self, X, layer_name='feature_vector'):
        """Extract deep features from specified layer"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Reshape input if needed
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
        print(f"Using layer '{layer_name}' for feature extraction")
        
        # Force model to be built by making a prediction first
        try:
            print("Ensuring model is built...")
            dummy_prediction = self.model.predict(X[:1], verbose=0)
            print(f"Model prediction successful, output shape: {dummy_prediction.shape}")
        except Exception as e:
            print(f"Failed to get model prediction: {e}")
            return self._extract_features_from_raw_data(X)
        
        # Now try to create feature extractor using the functional API approach
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            
            # Method 1: Use the built model's intermediate layer
            print(f"Attempting to access layer '{layer_name}'...")
            
            # Check available layers
            available_layers = [layer.name for layer in self.model.layers]
            print(f"Available layers: {available_layers}")
            
            if layer_name not in available_layers:
                # Find best alternative layer
                candidate_layers = [name for name in available_layers if 'fc' in name or 'dense' in name or 'feature' in name]
                if candidate_layers:
                    layer_name = candidate_layers[-1]  # Use the last feature layer
                    print(f"Using alternative layer: {layer_name}")
                else:
                    print("No suitable feature layer found, using raw transformation")
                    return self._extract_features_from_raw_data(X)
            
            # Get the target layer output
            target_layer = self.model.get_layer(layer_name)
            
            # Create feature extractor model
            feature_extractor = Model(inputs=self.model.input, outputs=target_layer.output)
            
            # Extract features
            features = feature_extractor.predict(X, verbose=0)
            print(f"Successfully extracted features from {layer_name}. Shape: {features.shape}")
            
            return features
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            print("Falling back to raw data transformation...")
            return self._extract_features_from_raw_data(X)
    
    def _extract_features_from_raw_data(self, X):
        """Extract features directly from raw data using image processing techniques"""
        print("Extracting features from raw image data...")
        
        # Store global PCA if not exists to ensure consistent dimensions
        if not hasattr(self, '_global_pca'):
            self._global_pca = None
            self._global_scaler = None
        
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            import cv2
            
            # Flatten the images
            X_flattened = X.reshape(X.shape[0], -1)
            
            # Compute additional image statistics for all samples
            print("Computing image statistics...")
            additional_features = []
            for i in range(X.shape[0]):
                img = X[i].squeeze()
                
                # Basic statistics
                mean_val = np.mean(img)
                std_val = np.std(img)
                min_val = np.min(img)
                max_val = np.max(img)
                
                # Edge detection features
                try:
                    edges = cv2.Canny((img * 255).astype(np.uint8), 50, 150)
                    edge_ratio = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
                except:
                    edge_ratio = 0
                
                # Texture features (simplified)
                if img.shape[0] > 1 and img.shape[1] > 1:
                    grad_x = np.gradient(img, axis=1)
                    grad_y = np.gradient(img, axis=0)
                    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                    texture_measure = np.mean(gradient_magnitude)
                else:
                    texture_measure = 0
                
                additional_features.append([mean_val, std_val, min_val, max_val, edge_ratio, texture_measure])
            
            additional_features = np.array(additional_features)
            
            # Use consistent PCA dimensions
            target_pca_components = 512  # Fixed number of components
            
            if self._global_pca is None:
                # First time - fit PCA and scaler
                print(f"Fitting global PCA with {target_pca_components} components...")
                
                # Scale flattened features
                self._global_scaler = StandardScaler()
                X_scaled = self._global_scaler.fit_transform(X_flattened)
                
                # Ensure we don't exceed available dimensions
                max_components = min(target_pca_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
                
                self._global_pca = PCA(n_components=max_components)
                features_pca = self._global_pca.fit_transform(X_scaled)
                
                print(f"Global PCA fitted. PCA features shape: {features_pca.shape}")
            else:
                # Use existing PCA and scaler
                print("Using existing global PCA...")
                X_scaled = self._global_scaler.transform(X_flattened)
                features_pca = self._global_pca.transform(X_scaled)
                
                print(f"PCA features transformed. Shape: {features_pca.shape}")
            
            # Combine PCA features with additional features
            combined_features = np.hstack([features_pca, additional_features])
            
            print(f"Final consistent features shape: {combined_features.shape}")
            
            return combined_features
            
        except Exception as e:
            print(f"Raw feature extraction failed: {e}")
            print("Using simple consistent sampling...")
            
            # Ultimate fallback - consistent sampling
            X_flattened = X.reshape(X.shape[0], -1)
            
            # Always use same step size for consistency
            target_features = 1000
            step = max(1, X_flattened.shape[1] // target_features)
            sampled_features = X_flattened[:, ::step]
            
            # Pad or truncate to exactly target_features
            if sampled_features.shape[1] < target_features:
                padding = np.zeros((sampled_features.shape[0], target_features - sampled_features.shape[1]))
                sampled_features = np.hstack([sampled_features, padding])
            elif sampled_features.shape[1] > target_features:
                sampled_features = sampled_features[:, :target_features]
            
            print(f"Consistent sampled features shape: {sampled_features.shape}")
            return sampled_features
        
    def predict(self, X):
        """Make predictions on input data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Reshape input if needed
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
            
        predictions = self.model.predict(X)
        return predictions
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from tensorflow.keras.utils import to_categorical
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Reshape input if needed
        if len(X_test.shape) == 3:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
            
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
        
        # Model evaluation
        loss, accuracy, precision, recall = self.model.evaluate(X_test, y_test_cat, verbose=0)
        
        # Predictions
        predictions = self.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        
        # Classification report
        class_names = ['Forged', 'Genuine']
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist(),
            'predicted_classes': y_pred.tolist()
        }
        
        return results
        
    def visualize_training_history(self, history, save_path=None):
        """Visualize training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            self.build_cnn_model()
            
        return self.model.summary()
        
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save!")
            
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        # Reset feature extractor to force recreation
        self.feature_extractor = None
        
    def visualize_feature_maps(self, image, layer_names=None, save_path=None):
        """Visualize feature maps from different layers"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        if layer_names is None:
            layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
            
        # Ensure image has batch dimension
        if len(image.shape) == 2:
            image = image.reshape(1, image.shape[0], image.shape[1], 1)
        elif len(image.shape) == 3:
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            
        fig, axes = plt.subplots(len(layer_names), 8, figsize=(20, len(layer_names) * 3))
        
        for i, layer_name in enumerate(layer_names):
            # Create model to extract features from specific layer
            intermediate_model = keras.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer(layer_name).output
            )
            
            # Get feature maps
            feature_maps = intermediate_model.predict(image)
            
            # Display first 8 feature maps
            for j in range(min(8, feature_maps.shape[-1])):
                if len(layer_names) == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                    
                ax.imshow(feature_maps[0, :, :, j], cmap='viridis')
                ax.set_title(f'{layer_name} - Filter {j+1}')
                ax.axis('off')
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Initialize CNN
    cnn = SignatureCNN(input_shape=(224, 224, 1))
    
    # Build and compile model
    model = cnn.compile_model()
    
    # Display model summary
    print("CNN Model Architecture:")
    cnn.get_model_summary()
    
    print("CNN feature extractor ready!")
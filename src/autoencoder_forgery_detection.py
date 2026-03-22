"""
Autoencoder for Forgery Detection in Signature-Based Biometric Authentication
Implements reconstruction error-based anomaly detection for forged signatures
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import os


class SignatureAutoencoder:
    def __init__(self, input_shape=(224, 224, 1), latent_dim=128):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.threshold = None
        
    def build_encoder(self):
        """Build encoder network"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Encoder layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Flatten and encode to latent space
        shape_before_flattening = keras.backend.int_shape(x)
        x = layers.Flatten()(x)
        
        # Bottleneck layer (latent space)
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent_space')(x)
        
        encoder = keras.Model(inputs, latent, name='encoder')
        
        self.encoder = encoder
        self.shape_before_flattening = shape_before_flattening
        
        return encoder
        
    def build_decoder(self):
        """Build decoder network"""
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        
        # Start with a small feature map and progressively upsample
        # Calculate reasonable starting dimensions
        start_size = 7  # Start with 7x7 feature maps
        start_filters = 256
        
        x = layers.Dense(start_size * start_size * start_filters, activation='relu')(latent_inputs)
        x = layers.Reshape((start_size, start_size, start_filters))(x)
        
        # Progressive upsampling to reach target size
        # From 7x7 -> 14x14
        x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        # From 14x14 -> 28x28
        x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        # From 28x28 -> 56x56
        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        # From 56x56 -> 112x112
        x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D((2, 2))(x)
        
        # Final output layer
        outputs = layers.Conv2D(self.input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
        
        # Resize to exact target dimensions if needed
        target_h, target_w = self.input_shape[0], self.input_shape[1]
        if outputs.shape[1] != target_h or outputs.shape[2] != target_w:
            outputs = layers.Lambda(lambda x: tf.image.resize(x, [target_h, target_w]))(outputs)
        
        decoder = keras.Model(latent_inputs, outputs, name='decoder')
        
        self.decoder = decoder
        return decoder
        
    def build_autoencoder(self):
        """Build complete autoencoder"""
        encoder = self.build_encoder()
        decoder = self.build_decoder()
        
        # Connect encoder and decoder
        inputs = keras.Input(shape=self.input_shape)
        encoded = encoder(inputs)
        decoded = decoder(encoded)
        
        autoencoder = keras.Model(inputs, decoded, name='autoencoder')
        
        self.autoencoder = autoencoder
        return autoencoder
        
    def compile_autoencoder(self, learning_rate=0.001):
        """Compile autoencoder"""
        if self.autoencoder is None:
            self.build_autoencoder()
            
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.autoencoder.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['mse', 'mae']
        )
        
        return self.autoencoder
        
    def train_autoencoder(self, X_genuine, X_val=None, epochs=100, batch_size=32, 
                         save_path=None, callbacks=None):
        """Train autoencoder on genuine signatures only"""
        if self.autoencoder is None:
            self.compile_autoencoder()
            
        # Reshape input data if needed
        if len(X_genuine.shape) == 3:
            X_genuine = X_genuine.reshape(X_genuine.shape[0], X_genuine.shape[1], X_genuine.shape[2], 1)
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
        
        # For autoencoders, input and output are the same (unsupervised)
        validation_data = None
        if X_val is not None:
            validation_data = (X_val, X_val)
            
        # Default callbacks
        default_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.2,
                patience=8,
                min_lr=0.00001
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
                    monitor='val_loss' if X_val is not None else 'loss',
                    save_best_only=True,
                    save_weights_only=False
                )
            )
        
        # Train autoencoder
        history = self.autoencoder.fit(
            X_genuine, X_genuine,  # Input and target are the same
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
        
    def calculate_reconstruction_error(self, X):
        """Calculate reconstruction error for input data"""
        if self.autoencoder is None:
            raise ValueError("Autoencoder not trained yet!")
            
        # Reshape input if needed
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
            
        # Get reconstructed images
        reconstructed = self.autoencoder.predict(X, verbose=0)
        
        # Calculate reconstruction error (Mean Squared Error)
        reconstruction_error = np.mean(np.square(X - reconstructed), axis=(1, 2, 3))
        
        return reconstruction_error, reconstructed
        
    def determine_threshold(self, X_genuine, percentile=95):
        """Determine threshold for anomaly detection based on genuine signatures"""
        reconstruction_errors, _ = self.calculate_reconstruction_error(X_genuine)
        
        # Use percentile of genuine reconstruction errors as threshold
        threshold = np.percentile(reconstruction_errors, percentile)
        self.threshold = threshold
        
        print(f"Anomaly detection threshold set to: {threshold:.6f}")
        
        return threshold
        
    def detect_forgery(self, X, threshold=None):
        """Detect forged signatures based on reconstruction error"""
        if threshold is None:
            if self.threshold is None:
                raise ValueError("Threshold not set! Call determine_threshold first or provide threshold.")
            threshold = self.threshold
            
        reconstruction_errors, reconstructed = self.calculate_reconstruction_error(X)
        
        # Classify as forgery if reconstruction error > threshold
        predictions = (reconstruction_errors > threshold).astype(int)
        
        results = {
            'reconstruction_errors': reconstruction_errors,
            'predictions': predictions,  # 1 = forgery, 0 = genuine
            'reconstructed_images': reconstructed,
            'threshold': threshold
        }
        
        return results
        
    def evaluate_forgery_detection(self, X_genuine, X_forged, threshold=None):
        """Evaluate forgery detection performance"""
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        # Combine genuine and forged samples
        X_combined = np.vstack([X_genuine, X_forged])
        y_true = np.hstack([np.zeros(len(X_genuine)), np.ones(len(X_forged))])  # 0=genuine, 1=forged
        
        # Get reconstruction errors
        reconstruction_errors, _ = self.calculate_reconstruction_error(X_combined)
        
        if threshold is None:
            if self.threshold is None:
                # Auto-determine threshold if not set
                genuine_errors, _ = self.calculate_reconstruction_error(X_genuine)
                threshold = self.determine_threshold(X_genuine)
            else:
                threshold = self.threshold
        
        # Make predictions
        y_pred = (reconstruction_errors > threshold).astype(int)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_true)
        
        # Classification report
        class_names = ['Genuine', 'Forged']
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, reconstruction_errors)
        except:
            roc_auc = 0.0
            
        # False Acceptance Rate (FAR) and False Rejection Rate (FRR)
        false_accepts = np.sum((y_true == 1) & (y_pred == 0))  # Forged classified as genuine
        false_rejects = np.sum((y_true == 0) & (y_pred == 1))  # Genuine classified as forged
        
        far = false_accepts / len(X_forged) if len(X_forged) > 0 else 0
        frr = false_rejects / len(X_genuine) if len(X_genuine) > 0 else 0
        
        # Equal Error Rate (EER) - approximate
        fpr, tpr, thresholds = roc_curve(y_true, reconstruction_errors)
        fnr = 1 - tpr
        eer_index = np.argmin(np.abs(fpr - fnr))
        eer = fpr[eer_index]
        
        results = {
            'accuracy': accuracy,
            'far': far,  # False Acceptance Rate
            'frr': frr,  # False Rejection Rate
            'eer': eer,  # Equal Error Rate
            'roc_auc': roc_auc,
            'threshold': threshold,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'reconstruction_errors': reconstruction_errors.tolist(),
            'predictions': y_pred.tolist(),
            'true_labels': y_true.tolist()
        }
        
        return results
        
    def visualize_reconstruction(self, images, save_path=None, num_samples=5):
        """Visualize original vs reconstructed images"""
        if self.autoencoder is None:
            raise ValueError("Autoencoder not trained yet!")
            
        # Reshape if needed
        if len(images.shape) == 3:
            images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
            
        # Get reconstructions
        reconstructed = self.autoencoder.predict(images[:num_samples], verbose=0)
        
        # Calculate reconstruction errors
        errors = np.mean(np.square(images[:num_samples] - reconstructed), axis=(1, 2, 3))
        
        fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
        
        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(images[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstructed image
            axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
            
            # Difference (error map)
            diff = np.abs(images[i].squeeze() - reconstructed[i].squeeze())
            axes[2, i].imshow(diff, cmap='hot')
            axes[2, i].set_title(f'Error: {errors[i]:.4f}')
            axes[2, i].axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_reconstruction_error_distribution(self, X_genuine, X_forged=None, save_path=None):
        """Plot distribution of reconstruction errors"""
        genuine_errors, _ = self.calculate_reconstruction_error(X_genuine)
        
        plt.figure(figsize=(12, 6))
        
        if X_forged is not None:
            forged_errors, _ = self.calculate_reconstruction_error(X_forged)
            
            plt.subplot(1, 2, 1)
            plt.hist(genuine_errors, bins=50, alpha=0.7, label='Genuine', color='green')
            plt.hist(forged_errors, bins=50, alpha=0.7, label='Forged', color='red')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Frequency')
            plt.title('Reconstruction Error Distribution')
            plt.legend()
            
            if self.threshold is not None:
                plt.axvline(self.threshold, color='black', linestyle='--', 
                           label=f'Threshold: {self.threshold:.4f}')
                plt.legend()
            
            # ROC curve
            plt.subplot(1, 2, 2)
            y_true = np.hstack([np.zeros(len(genuine_errors)), np.ones(len(forged_errors))])
            y_scores = np.hstack([genuine_errors, forged_errors])
            
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
        else:
            plt.hist(genuine_errors, bins=50, alpha=0.7, label='Genuine', color='green')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Frequency')
            plt.title('Genuine Signatures - Reconstruction Error Distribution')
            
            if self.threshold is not None:
                plt.axvline(self.threshold, color='black', linestyle='--', 
                           label=f'Threshold: {self.threshold:.4f}')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_autoencoder(self, filepath):
        """Save the trained autoencoder"""
        if self.autoencoder is not None:
            self.autoencoder.save(filepath)
            print(f"Autoencoder saved to {filepath}")
        else:
            print("No autoencoder to save!")
            
    def load_autoencoder(self, filepath):
        """Load a pre-trained autoencoder"""
        self.autoencoder = keras.models.load_model(filepath)
        print(f"Autoencoder loaded from {filepath}")
        
        # Extract encoder and decoder
        encoder_input = self.autoencoder.input
        encoder_output = self.autoencoder.get_layer('latent_space').output
        self.encoder = keras.Model(encoder_input, encoder_output, name='encoder')
        
        decoder_input = keras.Input(shape=(self.latent_dim,))
        decoder_output = self.autoencoder.layers[-1](self.autoencoder.layers[-2](decoder_input))
        self.decoder = keras.Model(decoder_input, decoder_output, name='decoder')


if __name__ == "__main__":
    # Initialize autoencoder
    autoencoder = SignatureAutoencoder(input_shape=(224, 224, 1), latent_dim=128)
    
    # Build and compile autoencoder
    model = autoencoder.compile_autoencoder()
    
    # Display model summary
    print("Autoencoder Architecture:")
    print(autoencoder.autoencoder.summary())
    
    print("Autoencoder for forgery detection ready!")
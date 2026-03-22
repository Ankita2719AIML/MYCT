"""
Grad-CAM Visualization for Signature-Based Biometric Authentication
Implements Gradient-weighted Class Activation Mapping for spatial explainability
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os


class GradCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = None
        
        # Auto-detect suitable layer if not provided
        if self.layer_name is None:
            self.layer_name = self._find_suitable_layer()
            
        print(f"GradCAM initialized with layer: {self.layer_name}")
        
    def _find_suitable_layer(self):
        """Automatically find a suitable convolutional layer for GradCAM"""
        # Look for the last convolutional layer
        conv_layers = []
        
        for layer in self.model.layers:
            if 'conv' in layer.name.lower():
                conv_layers.append(layer.name)
                
        if conv_layers:
            return conv_layers[-1]  # Return last conv layer
        else:
            # Fallback to any layer with 'conv' in name or suitable activation
            for layer in self.model.layers:
                if hasattr(layer, 'activation') and layer.output_shape is not None:
                    if len(layer.output_shape) == 4:  # (batch, height, width, channels)
                        return layer.name
                        
        # Final fallback
        return self.model.layers[-3].name if len(self.model.layers) > 3 else self.model.layers[-1].name
        
    def make_gradcam_heatmap(self, img_array, class_index=None, eps=1e-8):
        """Generate GradCAM heatmap for input image"""
        
        # Create a model that outputs both the predictions and the feature map
        # from the target layer
        if self.grad_model is None:
            try:
                last_conv_layer = self.model.get_layer(self.layer_name)
                self.grad_model = keras.Model(
                    [self.model.inputs], 
                    [last_conv_layer.output, self.model.output]
                )
            except ValueError as e:
                print(f"Error creating grad model: {e}")
                print("Available layers:", [layer.name for layer in self.model.layers])
                raise
        
        # Ensure input has batch dimension
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        elif len(img_array.shape) == 2:
            img_array = np.expand_dims(np.expand_dims(img_array, axis=0), axis=-1)
            
        # Compute the gradients
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            conv_outputs, predictions = self.grad_model(img_array)
            
            if class_index is None:
                # Use the predicted class
                class_index = tf.argmax(predictions[0])
                
            loss = predictions[:, class_index]
            
        # Get gradients of the loss w.r.t. the convolutional layer output
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients (importance weights)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy(), float(predictions[0, class_index])
        
    def overlay_heatmap_on_image(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """Overlay GradCAM heatmap on original image"""
        
        # Ensure image is in correct format
        if len(image.shape) == 3 and image.shape[-1] == 1:
            image = image.squeeze()
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)  # Convert to RGB
            
        # Normalize image to 0-255 range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to 0-255 range
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # Convert BGR to RGB (OpenCV uses BGR)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        overlayed = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        
        return overlayed, heatmap_colored
        
    def generate_gradcam_for_batch(self, images, class_indices=None):
        """Generate GradCAM heatmaps for a batch of images"""
        heatmaps = []
        confidences = []
        
        for i, img in enumerate(images):
            class_idx = class_indices[i] if class_indices is not None else None
            heatmap, confidence = self.make_gradcam_heatmap(img, class_idx)
            heatmaps.append(heatmap)
            confidences.append(confidence)
            
        return np.array(heatmaps), np.array(confidences)
        
    def visualize_gradcam(self, image, save_path=None, class_index=None, 
                         title="GradCAM Visualization"):
        """Visualize GradCAM explanation for a single image"""
        
        # Generate heatmap
        heatmap, confidence = self.make_gradcam_heatmap(image, class_index)
        
        # Create overlay
        overlayed, heatmap_colored = self.overlay_heatmap_on_image(image, heatmap)
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        display_img = image.squeeze() if len(image.shape) > 2 else image
        axes[0].imshow(display_img, cmap='gray')
        axes[0].set_title(f'Original Image\nConfidence: {confidence:.3f}')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='hot')
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')
        
        # Colored heatmap
        axes[2].imshow(heatmap_colored)
        axes[2].set_title('Colored Heatmap')
        axes[2].axis('off')
        
        # Overlay
        axes[3].imshow(overlayed)
        axes[3].set_title('Overlayed Result')
        axes[3].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return heatmap, overlayed
        
    def compare_genuine_vs_forged(self, genuine_images, forged_images, 
                                 save_path=None, num_samples=3):
        """Compare GradCAM visualizations for genuine vs forged signatures"""
        
        fig, axes = plt.subplots(2, num_samples * 4, figsize=(num_samples * 16, 8))
        
        categories = ['Genuine', 'Forged']
        image_sets = [genuine_images[:num_samples], forged_images[:num_samples]]
        
        for row, (category, images) in enumerate(zip(categories, image_sets)):
            for col, image in enumerate(images):
                # Generate GradCAM
                heatmap, confidence = self.make_gradcam_heatmap(image)
                overlayed, _ = self.overlay_heatmap_on_image(image, heatmap)
                
                # Plot original
                display_img = image.squeeze() if len(image.shape) > 2 else image
                axes[row, col*4].imshow(display_img, cmap='gray')
                axes[row, col*4].set_title(f'{category} Original\nConf: {confidence:.3f}')
                axes[row, col*4].axis('off')
                
                # Plot heatmap
                axes[row, col*4+1].imshow(heatmap, cmap='hot')
                axes[row, col*4+1].set_title(f'{category} Heatmap')
                axes[row, col*4+1].axis('off')
                
                # Plot overlay
                axes[row, col*4+2].imshow(overlayed)
                axes[row, col*4+2].set_title(f'{category} Overlay')
                axes[row, col*4+2].axis('off')
                
                # Plot thresholded heatmap
                threshold = np.percentile(heatmap, 70)
                thresholded = np.where(heatmap > threshold, heatmap, 0)
                axes[row, col*4+3].imshow(thresholded, cmap='hot')
                axes[row, col*4+3].set_title(f'{category} Key Regions')
                axes[row, col*4+3].axis('off')
        
        plt.suptitle('GradCAM Comparison: Genuine vs Forged Signatures', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_attention_patterns(self, images, labels, save_dir=None):
        """Analyze attention patterns across different classes"""
        
        genuine_indices = np.where(labels == 1)[0]
        forged_indices = np.where(labels == 0)[0]
        
        # Generate heatmaps for each class
        genuine_heatmaps = []
        forged_heatmaps = []
        
        print("Analyzing genuine signatures...")
        for idx in genuine_indices[:10]:  # Limit to 10 samples
            heatmap, _ = self.make_gradcam_heatmap(images[idx])
            genuine_heatmaps.append(heatmap)
            
        print("Analyzing forged signatures...")
        for idx in forged_indices[:10]:  # Limit to 10 samples
            heatmap, _ = self.make_gradcam_heatmap(images[idx])
            forged_heatmaps.append(heatmap)
            
        # Calculate average attention patterns
        avg_genuine = np.mean(genuine_heatmaps, axis=0)
        avg_forged = np.mean(forged_heatmaps, axis=0)
        
        # Visualize average patterns
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Average genuine attention
        im1 = axes[0].imshow(avg_genuine, cmap='hot')
        axes[0].set_title('Average Genuine Attention')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # Average forged attention
        im2 = axes[1].imshow(avg_forged, cmap='hot')
        axes[1].set_title('Average Forged Attention')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference map
        difference = avg_genuine - avg_forged
        im3 = axes[2].imshow(difference, cmap='RdBu', center=0)
        axes[2].set_title('Attention Difference\n(Genuine - Forged)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'attention_patterns_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        # Return analysis results
        analysis_results = {
            'avg_genuine_attention': avg_genuine,
            'avg_forged_attention': avg_forged,
            'attention_difference': difference,
            'genuine_attention_stats': {
                'mean': float(np.mean(avg_genuine)),
                'std': float(np.std(avg_genuine)),
                'max': float(np.max(avg_genuine)),
                'entropy': float(-np.sum(avg_genuine * np.log(avg_genuine + 1e-8)))
            },
            'forged_attention_stats': {
                'mean': float(np.mean(avg_forged)),
                'std': float(np.std(avg_forged)),
                'max': float(np.max(avg_forged)),
                'entropy': float(-np.sum(avg_forged * np.log(avg_forged + 1e-8)))
            }
        }
        
        return analysis_results
        
    def guided_gradcam(self, image, class_index=None):
        """Generate Guided GradCAM (combines GradCAM with Guided Backpropagation)"""
        
        # Generate regular GradCAM
        heatmap, confidence = self.make_gradcam_heatmap(image, class_index)
        
        # Generate guided backpropagation
        guided_grads = self._guided_backprop(image, class_index)
        
        # Combine GradCAM and guided backpropagation
        guided_gradcam = guided_grads * np.expand_dims(heatmap, axis=-1)
        
        return guided_gradcam, heatmap, confidence
        
    def _guided_backprop(self, image, class_index=None):
        """Generate guided backpropagation"""
        # This is a simplified version - full implementation would require
        # modifying ReLU activations during backpropagation
        
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        elif len(image.shape) == 2:
            image = np.expand_dims(np.expand_dims(image, axis=0), axis=-1)
            
        with tf.GradientTape() as tape:
            tape.watch(image)
            predictions = self.model(image)
            
            if class_index is None:
                class_index = tf.argmax(predictions[0])
                
            loss = predictions[:, class_index]
            
        # Get gradients
        grads = tape.gradient(loss, image)
        
        # Apply guided backprop (keep only positive gradients)
        grads = tf.maximum(grads, 0)
        
        return grads[0].numpy()
        
    def save_gradcam_results(self, images, labels, save_dir, num_samples=10):
        """Save GradCAM visualizations for multiple samples"""
        os.makedirs(save_dir, exist_ok=True)
        
        results = {
            'samples': [],
            'statistics': {
                'genuine_attention_mean': [],
                'forged_attention_mean': [],
                'confidence_genuine': [],
                'confidence_forged': []
            }
        }
        
        # Process samples from each class
        genuine_indices = np.where(labels == 1)[0][:num_samples]
        forged_indices = np.where(labels == 0)[0][:num_samples]
        
        all_indices = list(genuine_indices) + list(forged_indices)
        all_labels = ['Genuine'] * len(genuine_indices) + ['Forged'] * len(forged_indices)
        
        for i, (idx, label) in enumerate(zip(all_indices, all_labels)):
            image = images[idx]
            
            # Generate GradCAM
            heatmap, confidence = self.make_gradcam_heatmap(image)
            overlayed, heatmap_colored = self.overlay_heatmap_on_image(image, heatmap)
            
            # Save visualization
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            display_img = image.squeeze() if len(image.shape) > 2 else image
            axes[0].imshow(display_img, cmap='gray')
            axes[0].set_title(f'{label} Original')
            axes[0].axis('off')
            
            axes[1].imshow(heatmap, cmap='hot')
            axes[1].set_title('GradCAM Heatmap')
            axes[1].axis('off')
            
            axes[2].imshow(heatmap_colored)
            axes[2].set_title('Colored Heatmap')
            axes[2].axis('off')
            
            axes[3].imshow(overlayed)
            axes[3].set_title(f'Overlay (Conf: {confidence:.3f})')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'gradcam_sample_{i:03d}_{label.lower()}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Store results
            sample_result = {
                'sample_index': int(idx),
                'label': label,
                'confidence': float(confidence),
                'attention_mean': float(np.mean(heatmap)),
                'attention_std': float(np.std(heatmap)),
                'attention_max': float(np.max(heatmap))
            }
            
            results['samples'].append(sample_result)
            
            # Update statistics
            if label == 'Genuine':
                results['statistics']['genuine_attention_mean'].append(float(np.mean(heatmap)))
                results['statistics']['confidence_genuine'].append(float(confidence))
            else:
                results['statistics']['forged_attention_mean'].append(float(np.mean(heatmap)))
                results['statistics']['confidence_forged'].append(float(confidence))
                
        # Save results to JSON
        import json
        with open(os.path.join(save_dir, 'gradcam_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"GradCAM results saved to {save_dir}")
        return results


if __name__ == "__main__":
    print("Grad-CAM visualization module ready!")
    print("Available methods:")
    print("- make_gradcam_heatmap(): Generate heatmap for single image")
    print("- visualize_gradcam(): Complete visualization with overlay")
    print("- compare_genuine_vs_forged(): Side-by-side comparison")
    print("- analyze_attention_patterns(): Statistical analysis of attention patterns")
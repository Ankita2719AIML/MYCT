"""
Data Preprocessing Pipeline for Signature-Based Biometric Authentication
Handles image preprocessing, normalization, and noise reduction
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class SignaturePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def load_image(self, image_path):
        """Load image from file path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
            
    def convert_to_grayscale(self, image):
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return gray
        
    def resize_image(self, image):
        """Resize image to target size (224x224)"""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
    def remove_background_noise(self, image):
        """Remove background noise and enhance contrast"""
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
        
    def normalize_strokes(self, image):
        """Normalize stroke thickness and intensity"""
        # Apply morphological operations to normalize stroke thickness
        kernel = np.ones((2,2), np.uint8)
        
        # Opening (erosion followed by dilation) to remove noise
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Closing (dilation followed by erosion) to close gaps
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return closed
        
    def correct_orientation(self, image):
        """Correct image orientation and scale"""
        # Find contours to determine main signature orientation
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (main signature)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # Correct rotation if needed
            if angle < -45:
                angle = 90 + angle
                
            # Rotate image if angle is significant
            if abs(angle) > 5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return rotated
                
        return image
        
    def apply_gaussian_filter(self, image, sigma=1.0):
        """Apply Gaussian filtering to remove scanning artifacts"""
        return cv2.GaussianBlur(image, (5, 5), sigma)
        
    def normalize_pixel_values(self, image):
        """Normalize pixel values to [0, 1] range"""
        return image.astype(np.float32) / 255.0
        
    def preprocess_image(self, image_path, save_intermediate=False, output_dir=None):
        """Complete preprocessing pipeline for a single image"""
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None
            
        preprocessing_steps = {}
        
        # Step 1: Convert to grayscale
        gray = self.convert_to_grayscale(image)
        preprocessing_steps['grayscale'] = gray.copy()
        
        # Step 2: Resize to target size
        resized = self.resize_image(gray)
        preprocessing_steps['resized'] = resized.copy()
        
        # Step 3: Remove background noise and enhance contrast
        denoised = self.remove_background_noise(resized)
        preprocessing_steps['denoised'] = denoised.copy()
        
        # Step 4: Normalize stroke thickness
        normalized = self.normalize_strokes(denoised)
        preprocessing_steps['normalized'] = normalized.copy()
        
        # Step 5: Correct orientation
        oriented = self.correct_orientation(normalized)
        preprocessing_steps['oriented'] = oriented.copy()
        
        # Step 6: Apply Gaussian filtering
        filtered = self.apply_gaussian_filter(oriented)
        preprocessing_steps['filtered'] = filtered.copy()
        
        # Step 7: Normalize pixel values
        final_image = self.normalize_pixel_values(filtered)
        preprocessing_steps['final'] = final_image.copy()
        
        # Save intermediate steps if requested
        if save_intermediate and output_dir:
            self.save_intermediate_steps(preprocessing_steps, image_path, output_dir)
            
        return final_image
        
    def save_intermediate_steps(self, steps, original_path, output_dir):
        """Save intermediate preprocessing steps for visualization"""
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        
        for step_name, step_image in steps.items():
            if step_name != 'final':
                output_path = os.path.join(output_dir, f"{base_name}_{step_name}.png")
                cv2.imwrite(output_path, step_image)
            else:
                # For normalized final image, convert back to 0-255 range for saving
                output_path = os.path.join(output_dir, f"{base_name}_{step_name}.png")
                cv2.imwrite(output_path, (step_image * 255).astype(np.uint8))
                
    def preprocess_dataset(self, data_dir, output_dir=None, save_intermediate=False):
        """Preprocess entire dataset"""
        processed_data = []
        labels = []
        
        # Process each category (genuine/forged)
        for category in ['genuine', 'forged']:
            category_dir = os.path.join(data_dir, category)
            if not os.path.exists(category_dir):
                print(f"Directory {category_dir} does not exist, skipping...")
                continue
                
            # Process each image in category
            image_files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            print(f"Processing {len(image_files)} images in {category} category...")
            
            for image_file in image_files:
                image_path = os.path.join(category_dir, image_file)
                processed_image = self.preprocess_image(
                    image_path, 
                    save_intermediate=save_intermediate,
                    output_dir=os.path.join(output_dir, category) if output_dir else None
                )
                
                if processed_image is not None:
                    processed_data.append(processed_image)
                    labels.append(1 if category == 'genuine' else 0)
                    
        return np.array(processed_data), np.array(labels)
        
    def visualize_preprocessing_steps(self, image_path, save_path=None):
        """Visualize all preprocessing steps"""
        image = self.load_image(image_path)
        if image is None:
            return
            
        # Apply all preprocessing steps
        gray = self.convert_to_grayscale(image)
        resized = self.resize_image(gray)
        denoised = self.remove_background_noise(resized)
        normalized = self.normalize_strokes(denoised)
        oriented = self.correct_orientation(normalized)
        filtered = self.apply_gaussian_filter(oriented)
        final = self.normalize_pixel_values(filtered)
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Signature Preprocessing Pipeline', fontsize=16)
        
        steps = [
            ('Original', cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image),
            ('Grayscale', gray),
            ('Resized', resized),
            ('Denoised', denoised),
            ('Normalized', normalized),
            ('Oriented', oriented),
            ('Filtered', filtered),
            ('Final', final)
        ]
        
        for i, (title, img) in enumerate(steps):
            row = i // 4
            col = i % 4
            axes[row, col].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def generate_sample_data(output_dir, num_genuine=100, num_forged=100):
    """Generate sample signature-like images for demonstration"""
    os.makedirs(os.path.join(output_dir, 'genuine'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'forged'), exist_ok=True)
    
    def create_signature_like_image(is_genuine=True, width=300, height=150):
        """Create a synthetic signature-like image"""
        # Create blank image
        img = np.ones((height, width), dtype=np.uint8) * 255
        
        # Generate random signature-like strokes
        num_strokes = np.random.randint(3, 8)
        
        for _ in range(num_strokes):
            # Random start and end points
            start_x = np.random.randint(10, width - 50)
            start_y = np.random.randint(20, height - 20)
            
            # Create curved stroke
            points = []
            current_x, current_y = start_x, start_y
            
            stroke_length = np.random.randint(30, 100)
            for i in range(stroke_length):
                # Add some randomness to make it look more natural
                dx = np.random.normal(1, 0.3) if is_genuine else np.random.normal(1, 0.8)
                dy = np.random.normal(0, 0.5) if is_genuine else np.random.normal(0, 1.2)
                
                current_x += dx
                current_y += dy
                
                if 0 <= current_x < width and 0 <= current_y < height:
                    points.append((int(current_x), int(current_y)))
                    
            # Draw stroke
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(img, points[i], points[i+1], 0, thickness=np.random.randint(2, 4))
                    
        return img
    
    # Generate genuine signatures
    print("Generating genuine signatures...")
    for i in range(num_genuine):
        img = create_signature_like_image(is_genuine=True)
        cv2.imwrite(os.path.join(output_dir, 'genuine', f'genuine_{i:03d}.png'), img)
        
    # Generate forged signatures (more irregular)
    print("Generating forged signatures...")
    for i in range(num_forged):
        img = create_signature_like_image(is_genuine=False)
        cv2.imwrite(os.path.join(output_dir, 'forged', f'forged_{i:03d}.png'), img)


if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = SignaturePreprocessor()
    
    # Generate sample data if dataset is empty
    dataset_path = "C:/Users/priya/OneDrive/Desktop/MYCT_TASK/archive/MYCT_pics_updated"
    train_path = os.path.join(dataset_path, "train")
    
    # Check if dataset has images
    has_data = False
    for category in ['genuine', 'forged']:
        category_path = os.path.join(train_path, category)
        if os.path.exists(category_path):
            images = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                has_data = True
                break
                
    if not has_data:
        print("Dataset appears to be empty. Generating sample data...")
        generate_sample_data(train_path, num_genuine=50, num_forged=50)
        generate_sample_data(os.path.join(dataset_path, "val"), num_genuine=15, num_forged=15)
        generate_sample_data(os.path.join(dataset_path, "test"), num_genuine=15, num_forged=15)
        print("Sample data generated successfully!")
    
    print("Data preprocessing module ready!")
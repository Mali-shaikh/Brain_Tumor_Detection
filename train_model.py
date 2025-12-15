"""
Brain Tumor Detection System - Model Training Script
This script downloads the dataset, preprocesses images, and trains a CNN model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import cv2
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BrainTumorModel:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self, data_dir):

        """
        Load images from directory structure:
        data_dir/brain_tumor_dataset/        
            tumor/yes/
            no_tumor/no/
        """
        
        images = []
        labels = []
        
        # Define class mapping
        class_names = ['no', 'yes']
        
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} does not exist")
                continue
                
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                
                # Read and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                
                images.append(img)
                labels.append(class_idx)
        
        # Convert to numpy arrays
        images = np.array(images, dtype='float32') # EfficientNet expects [0, 255]
        labels = np.array(labels)
        
        print(f"Loaded {len(images)} images")
        print(f"Class distribution: No Tumor={np.sum(labels==0)}, Tumor={np.sum(labels==1)}")
        
        return images, labels
    
    def create_model(self):
        """Create CNN model using EfficientNetB0 with transfer learning"""
        
        # Load pre-trained EfficientNetB0 model
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size[0], self.img_size[1], 3)
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Create new model on top
        inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        # Data augmentation layers
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Pre-trained model
        x = base_model(x, training=False)
        
        # Custom classification layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print("Model created successfully!")
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Train the model"""
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        
        print("\n--- Phase 2: Fine-tuning ---")
        # Unfreeze the base model
        self.model.layers[4].trainable = True # The EfficientNetB0 layer is usually at index 4
        
        # It's better to unfreeze only the top layers
        # Let's find the base model layer specifically
        for layer in self.model.layers:
            if "efficientnet" in layer.name.lower():
                layer.trainable = True
                # Freeze all layers except the last 20
                for inner_layer in layer.layers[:-20]:
                    inner_layer.trainable = False
                break
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5), # Low learning rate
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        # Train fine-tuned model
        total_epochs = epochs + 10 # Add 10 more epochs
        
        # Ensure proper integer conversion and handling for initial_epoch
        last_epoch = 0
        if self.history and self.history.epoch:
            last_epoch = self.history.epoch[-1]
            
        history_fine = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            initial_epoch=last_epoch + 1,
            epochs=total_epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Merge histories
        # Merge histories
        # Map keys from fine-tuning to original keys
        # Usually fine-tuning adds _1 suffix if metrics are recompiled
        
        base_metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        
        for key in self.history.history:
            target_key = key
            
            # If key not in fine history, try to find the matching one with suffix
            if key not in history_fine.history:
                # Try finding a key that starts with the base key name (e.g. precision for precision_1)
                # This is a simple heuristic
                candidates = [k for k in history_fine.history.keys() if k.startswith(key + '_') or (key in k and 'val_' in k == 'val_' in key)]
                # If we have 'precision' and fine has 'precision_1', match them.
                # Actually, simpler: just iterate fine history keys and map them back if needed?
                # or just append indiscriminately if we can't match?
                # Let's try to match by metric type.
                pass

        # Robust merge strategy:
        # 1. Standard metrics (loss, accuracy) should match.
        # 2. Precision/Recall might have suffix.
        
        # We will assume order of metrics in compile matches.
        # But easier: just create a new combined dictionary mapping "standard" names to the lists.
        
        combined_history = {}
        
        # Heuristic mapping
        metric_types = ['loss', 'accuracy', 'precision', 'recall']
        
        for m_type in metric_types:
            # Find key in original
            orig_key = next((k for k in self.history.history.keys() if m_type in k and 'val_' not in k), None)
            val_orig_key = next((k for k in self.history.history.keys() if m_type in k and 'val_' in k), None)
            
            fine_key = next((k for k in history_fine.history.keys() if m_type in k and 'val_' not in k), None)
            val_fine_key = next((k for k in history_fine.history.keys() if m_type in k and 'val_' in k), None)
            
            if orig_key and fine_key:
                # Create a standardized key name (e.g., 'precision' instead of 'precision_1')
                std_key = m_type
                combined_history[std_key] = self.history.history[orig_key] + history_fine.history[fine_key]
                
            if val_orig_key and val_fine_key:
                std_key = f"val_{m_type}"
                combined_history[std_key] = self.history.history[val_orig_key] + history_fine.history[val_fine_key]

        # Update self.history.history with the clean combined version
        # We need to make sure self.history object still works for plotting which expects .history attribute
        self.history.history = combined_history
        
        # Save history to JSON for reporting
        import json
        with open('history.json', 'w') as f:
            json.dump(self.history.history, f)
            
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history saved as 'training_history.png'")
    
    def save_model(self, filepath='brain_tumor_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved as '{filepath}'")

# Main training pipeline
if __name__ == "__main__":
    print("Brain Tumor Detection Model Training")
    print("=" * 50)
    
    # Initialize model
    model_trainer = BrainTumorModel(img_size=(224, 224))
    
    # Load data (replace with your dataset path)
    DATA_DIR = "brain_tumor_dataset"  # Update this path
    
    print("\n1. Loading and preprocessing data...")
    images, labels = model_trainer.load_and_preprocess_data(DATA_DIR)
    
    # Split data
    print("\n2. Splitting data into train, validation, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Create model
    print("\n3. Creating model...")
    model_trainer.create_model()
    model_trainer.model.summary()
    
    # Train model
    print("\n4. Training model...")
    model_trainer.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)
    
    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    test_results = model_trainer.model.evaluate(X_test, y_test)
    
    test_loss = test_results[0]
    test_acc = test_results[1]
    test_prec = test_results[2]
    test_rec = test_results[3]
    
    # Calculate F1 Score manually
    if (test_prec + test_rec) > 0:
        test_f1 = 2 * (test_prec * test_rec) / (test_prec + test_rec)
    else:
        test_f1 = 0
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Precision: {test_prec:.4f} ({test_prec*100:.2f}%)")
    print(f"Test Recall: {test_rec:.4f} ({test_rec*100:.2f}%)")
    print(f"Test F1-Score: {test_f1:.4f} ({test_f1*100:.2f}%)")
    
    # Plot training history
    print("\n6. Plotting training history...")
    model_trainer.plot_training_history()
    
    # Save model
    print("\n7. Saving model...")
    model_trainer.save_model('brain_tumor_model.h5')
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("Model saved as 'brain_tumor_model.h5'")
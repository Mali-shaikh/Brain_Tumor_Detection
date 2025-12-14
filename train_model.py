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
        
        history_fine = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            initial_epoch=self.history.epoch[-1],
            epochs=total_epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Merge histories (optional, but good for plotting)
        # For simplicity, we'll just keep the latest history object for plotting the fine-tuning phase
        # or we could append. For now, let's just use the fine-tuning history as the final result.
        self.history = history_fine
        
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
    print(f"\nTest Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test Precision: {test_results[2]:.4f}")
    print(f"Test Recall: {test_results[3]:.4f}")
    
    # Plot training history
    print("\n6. Plotting training history...")
    model_trainer.plot_training_history()
    
    # Save model
    print("\n7. Saving model...")
    model_trainer.save_model('brain_tumor_model.h5')
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("Model saved as 'brain_tumor_model.h5'")
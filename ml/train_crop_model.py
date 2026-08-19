"""
SmartAgri AI - TensorFlow / Keras Model Training Script
Target: On-device Edge Crop Disease Classifier (TFLite)
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 12

def build_mobilenet_model(num_classes=NUM_CLASSES):
    """
    Builds a lightweight MobileNetV3 transfer learning model optimized for edge devices.
    """
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base layers for initial training

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1./127.5, offset=-1),  # Normalize to [-1, 1]
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax', name='crop_output')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    print("[SmartAgri AI] Initializing MobileNetV3 training pipeline...")
    model = build_mobilenet_model()
    model.summary()
    print("[SmartAgri AI] Ready to fit dataset (PlantVillage / Agronomic Vision Benchmark).")

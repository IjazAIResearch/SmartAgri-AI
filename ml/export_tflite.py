"""
SmartAgri AI - TensorFlow Lite Quantization & Export Script
Converts Keras h5/SavedModel to INT8 quantized .tflite format for mobile execution.
"""

import tensorflow as tf
import numpy as np

def convert_to_tflite(saved_model_dir, output_path="crop_health_model.tflite"):
    print(f"[SmartAgri AI] Loading Keras model from {saved_model_dir}...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Enable INT8 / FP16 dynamic range quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"[SmartAgri AI] Successfully exported quantized TFLite model to: {output_path}")

if __name__ == "__main__":
    print("[SmartAgri AI] TFLite Converter Utility ready.")

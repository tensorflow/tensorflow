#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

def create_fully_connected_model():
    """
    Create a simple fully connected model for testing the delegate
    """
    
    # Create a simple model with fully connected layers
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation=None)  # No activation for final layer
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    # Generate some dummy data for quantization
    x_train = np.random.rand(100, 32).astype(np.float32)
    y_train = np.random.rand(100, 4).astype(np.float32)
    
    # Train briefly to get some weights
    model.fit(x_train, y_train, epochs=1, verbose=0)
    
    return model

def create_int32_model():
    """
    Create and save an INT32 model for testing
    """
    print("Creating INT32 fully connected model...")
    
    model = create_fully_connected_model()
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []  # No quantization
    tflite_model = converter.convert()
    
    # Save the model
    with open('fully_connected_int32.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("✅ INT32 model saved as 'fully_connected_int32.tflite'")
    return tflite_model

def create_int8_model():
    """
    Create and save an INT8 quantized model for testing
    """
    print("Creating INT8 fully connected model...")
    
    model = create_fully_connected_model()
    
    # Create representative dataset for quantization
    def representative_dataset():
        for _ in range(100):
            yield [np.random.rand(1, 32).astype(np.float32)]
    
    # Convert to TFLite with INT8 quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save the model
    with open('fully_connected_int8.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("✅ INT8 model saved as 'fully_connected_int8.tflite'")
    return tflite_model

def main():
    print("🏗️  Creating test models for fully connected delegate...")
    
    # Create both models
    int32_model = create_int32_model()
    int8_model = create_int8_model()
    
    print(f"\n📊 Model sizes:")
    print(f"  INT32 model: {len(int32_model)} bytes")
    print(f"  INT8 model: {len(int8_model)} bytes")
    
    print(f"\n🧪 Use these models to test your delegate:")
    print(f"  python debug_inference.py fully_connected_int32.tflite <delegate_path>")
    print(f"  python debug_inference.py fully_connected_int8.tflite <delegate_path>")

if __name__ == "__main__":
    main()

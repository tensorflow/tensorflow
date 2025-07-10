#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import os
import sys

def inspect_tflite_model(model_path):
    """
    Inspect TFLite model structure and tensor details
    """
    print(f"📊 Inspecting TFLite model: {model_path}")
    
    try:
        # Load model without delegate first
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\n🔍 Model Structure:")
        print(f"  Total operations: {len(interpreter.get_signature_list())}")
        
        print(f"\n📥 Input Details:")
        for i, detail in enumerate(input_details):
            print(f"  Input {i}:")
            print(f"    Name: {detail['name']}")
            print(f"    Shape: {detail['shape']}")
            print(f"    Type: {detail['dtype']}")
            print(f"    Quantization: {detail['quantization']}")
        
        print(f"\n📤 Output Details:")
        for i, detail in enumerate(output_details):
            print(f"  Output {i}:")
            print(f"    Name: {detail['name']}")
            print(f"    Shape: {detail['shape']}")
            print(f"    Type: {detail['dtype']}")
            print(f"    Quantization: {detail['quantization']}")
            
        # Try to get tensor details for all tensors
        print(f"\n🔍 All Tensor Details:")
        try:
            tensor_details = interpreter.get_tensor_details()
            print(f"  Total tensors: {len(tensor_details)}")
            
            dynamic_tensors = []
            static_tensors = []
            fully_connected_related = []
            
            for i, detail in enumerate(tensor_details):
                # Look for fully connected related tensors
                if any(keyword in detail['name'].lower() for keyword in ['dense', 'fully_connected', 'matmul', 'fc']):
                    fully_connected_related.append(i)
                
                if detail['shape'] is not None and any(dim < 0 for dim in detail['shape']):
                    dynamic_tensors.append(i)
                    print(f"  Tensor #{i} (DYNAMIC): {detail['name']} - Shape: {detail['shape']} - Type: {detail['dtype']}")
                else:
                    static_tensors.append(i)
                    if i < 20:  # Show first 20 tensors
                        print(f"  Tensor #{i}: {detail['name']} - Shape: {detail['shape']} - Type: {detail['dtype']}")
            
            print(f"\n📊 Tensor Summary:")
            print(f"  Static tensors: {len(static_tensors)}")
            print(f"  Dynamic tensors: {len(dynamic_tensors)}")
            print(f"  Fully connected related: {len(fully_connected_related)}")
            
            if dynamic_tensors:
                print(f"  Dynamic tensor indices: {dynamic_tensors}")
            
            if fully_connected_related:
                print(f"  Fully connected tensor indices: {fully_connected_related}")
                for idx in fully_connected_related:
                    detail = tensor_details[idx]
                    print(f"    Tensor #{idx}: {detail['name']} - Shape: {detail['shape']} - Type: {detail['dtype']}")
            else:
                print(f"  ⚠️  No fully connected layers found in this model!")
                print(f"  This appears to be an LSTM-based model without fully connected layers.")
            
            # Look specifically for tensor #40
            if len(tensor_details) > 40:
                detail = tensor_details[40]
                print(f"\n🎯 Tensor #40 Details:")
                print(f"  Name: {detail['name']}")
                print(f"  Shape: {detail['shape']}")
                print(f"  Type: {detail['dtype']}")
                print(f"  Quantization: {detail['quantization']}")
                print(f"  This is likely a TensorArray from LSTM operations")
                
        except Exception as e:
            print(f"  Could not get tensor details: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")
        return False

def test_with_delegate(model_path, delegate_path):
    """
    Test model with delegate and capture logs
    """
    print(f"\n🧪 Testing with delegate: {delegate_path}")
    
    try:
        # Load delegate
        delegate = tf.lite.experimental.load_delegate(delegate_path)
        print("✅ Delegate loaded successfully")
        
        # Create interpreter with delegate
        interpreter = tf.lite.Interpreter(
            model_path=model_path, 
            experimental_delegates=[delegate]
        )
        
        # This is where the error might occur
        interpreter.allocate_tensors()
        print("✅ Tensors allocated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with delegate: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python debug_inference.py <model_path> <delegate_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    delegate_path = sys.argv[2]
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(delegate_path):
        print(f"❌ Delegate file not found: {delegate_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("🔍 TensorFlow Lite Model and Delegate Debug Tool")
    print("=" * 80)
    
    # First, inspect the model
    if inspect_tflite_model(model_path):
        print("\n" + "=" * 80)
        # Then test with delegate
        test_with_delegate(model_path, delegate_path)
    
    print("\n" + "=" * 80)
    print("🏁 Debug complete")

if __name__ == "__main__":
    main()

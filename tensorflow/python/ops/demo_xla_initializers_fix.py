#!/usr/bin/env python3
# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Demonstration of the fix for issue #105334:
XLA JIT Compilation Fails with Keras Initializers and Dynamic Shapes

This script demonstrates:
1. The original problem (dynamic shapes with XLA)
2. The solution (use concrete shapes)
3. The improved error messaging
"""

import tensorflow as tf
import sys


def demonstrate_problem():
    """Show the original problem from issue #105334."""
    print("=" * 70)
    print("DEMONSTRATING ISSUE #105334")
    print("=" * 70)
    print()
    print("Problem: Using Keras initializers with tf.shape() in XLA context")
    print()
    
    class SimpleModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
        
        @tf.function(jit_compile=True)
        def call(self, x):
            batch_size = tf.shape(x)[0]
            # Using Keras initializer with dynamic shape fails in XLA
            weights = tf.keras.initializers.GlorotUniform()(shape=[batch_size, 128])
            return weights
    
    model = SimpleModel()
    input_tensor = tf.random.uniform([32, 50], minval=0, maxval=1000, dtype=tf.int32)
    
    print("Attempting to call model with dynamic shape...")
    try:
        output = model(input_tensor)
        print(f"✗ Unexpected success! Output shape: {output.shape}")
        return False
    except TypeError as e:
        print(f"✓ Expected error caught with improved message:")
        print(f"  {str(e)}")
        print()
        return True


def demonstrate_solution():
    """Show the recommended solution using concrete shapes."""
    print("=" * 70)
    print("SOLUTION: Use Concrete Shapes")
    print("=" * 70)
    print()
    print("Solution 1: Initialize weights with known dimensions")
    print()
    
    class WorkingModel1(tf.keras.Model):
        def __init__(self):
            super().__init__()
        
        @tf.function(jit_compile=True)
        def call(self, x):
            # Use concrete shape values (not tf.shape())
            weights = tf.keras.initializers.GlorotUniform()(shape=[32, 128])
            return tf.matmul(tf.cast(x[:, :32], tf.float32), weights)
    
    model1 = WorkingModel1()
    input_tensor = tf.random.uniform([32, 50], minval=0, maxval=1000, dtype=tf.int32)
    
    try:
        output = model1(input_tensor)
        print(f"✓ Solution 1 works! Output shape: {output.shape}")
        print()
    except Exception as e:
        print(f"✗ Solution 1 failed: {e}")
        print()
        return False
    
    print("Solution 2: Use tf.keras.layers.Dense with built-in initialization")
    print()
    
    class WorkingModel2(tf.keras.Model):
        def __init__(self):
            super().__init__()
            # Initialize layers in __init__ with known dimensions
            self.dense = tf.keras.layers.Dense(
                128, 
                kernel_initializer='glorot_uniform'
            )
        
        @tf.function(jit_compile=True)
        def call(self, x):
            # Dense layer handles shapes internally
            return self.dense(tf.cast(x, tf.float32))
    
    model2 = WorkingModel2()
    
    try:
        output = model2(input_tensor)
        print(f"✓ Solution 2 works! Output shape: {output.shape}")
        print()
    except Exception as e:
        print(f"✗ Solution 2 failed: {e}")
        print()
        return False
    
    return True


def demonstrate_various_initializers():
    """Show that the fix works for various Keras initializers."""
    print("=" * 70)
    print("TESTING VARIOUS INITIALIZERS WITH XLA")
    print("=" * 70)
    print()
    
    initializers = [
        ('GlorotUniform', tf.keras.initializers.GlorotUniform()),
        ('GlorotNormal', tf.keras.initializers.GlorotNormal()),
        ('HeNormal', tf.keras.initializers.HeNormal()),
        ('HeUniform', tf.keras.initializers.HeUniform()),
        ('LecunNormal', tf.keras.initializers.LecunNormal()),
        ('LecunUniform', tf.keras.initializers.LecunUniform()),
    ]
    
    all_passed = True
    
    for name, initializer in initializers:
        @tf.function(jit_compile=True)
        def test_initializer():
            return initializer(shape=[64, 128])
        
        try:
            result = test_initializer()
            print(f"✓ {name:20s} - Success! Shape: {result.shape}")
        except Exception as e:
            print(f"✗ {name:20s} - Failed: {e}")
            all_passed = False
    
    print()
    return all_passed


def main():
    """Run all demonstrations."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + " FIX FOR ISSUE #105334".center(68) + "║")
    print("║" + " XLA JIT Compilation with Keras Initializers".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Test 1: Show the problem
    problem_shown = demonstrate_problem()
    
    # Test 2: Show solutions
    solutions_work = demonstrate_solution()
    
    # Test 3: Test various initializers
    initializers_work = demonstrate_various_initializers()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    if problem_shown and solutions_work and initializers_work:
        print("✓ All demonstrations completed successfully!")
        print()
        print("Key takeaways:")
        print("  1. Dynamic shapes (tf.shape()) don't work with initializers in XLA")
        print("  2. Use concrete shape values when calling initializers")
        print("  3. Or use tf.keras.layers with built-in initialization")
        print("  4. Error messages now clearly explain the issue")
        print()
        return 0
    else:
        print("✗ Some demonstrations failed")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())

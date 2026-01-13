# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Test to verify that returning a Layer object from call() raises an error.

This test verifies the fix for issue #108142 where tf.function/XLA compilation
fails when model.call() returns a Keras Layer instead of a Tensor, but eager
mode would run without error (though returning unexpected results).
"""

import tensorflow as tf


class TestModel(tf.keras.Model):
    """A model that incorrectly returns a Layer object instead of a Tensor."""

    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(64)
        self.d2 = tf.keras.layers.Dense(32)
        self.d3 = tf.keras.layers.Dense(16)

    def call(self, x):
        x = tf.nn.relu(self.d1(x))
        x = tf.nn.softmax(x, axis=1)
        x = tf.nn.softmax(x, axis=-2)
        x = tf.expand_dims(x, axis=-1)
        x = tf.nn.softmax(x, axis=2)
        # BUG: Returns a Layer object instead of a Tensor
        return tf.keras.layers.Lambda(lambda x: self.d3(x))


class CorrectModel(tf.keras.Model):
    """A model that correctly returns a Tensor."""

    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(64)
        self.d2 = tf.keras.layers.Dense(32)
        self.d3 = tf.keras.layers.Dense(16)

    def call(self, x):
        x = tf.nn.relu(self.d1(x))
        x = tf.nn.softmax(x, axis=1)
        x = self.d2(x)
        x = self.d3(x)
        return x


def test_layer_return_raises_error():
    """Test that returning a Layer object raises TypeError in eager mode."""
    model = TestModel()
    inputs = tf.random.normal([4, 8, 32])
    
    try:
        output = model(inputs)
        print("FAIL: Expected TypeError but got output:", output)
        return False
    except TypeError as e:
        if "Layer object" in str(e) and "instead of a Tensor" in str(e):
            print("PASS: Eager mode correctly raises TypeError")
            print(f"Error message: {e}")
            return True
        else:
            print("FAIL: Got TypeError but with unexpected message:", e)
            return False
    except Exception as e:
        print(f"FAIL: Got unexpected exception type {type(e)}: {e}")
        return False


def test_correct_model_works():
    """Test that a correct model still works."""
    model = CorrectModel()
    inputs = tf.random.normal([4, 8, 32])
    
    try:
        output = model(inputs)
        print(f"PASS: Correct model works, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"FAIL: Correct model raised exception: {e}")
        return False


def test_tf_function_with_correct_model():
    """Test that tf.function works with a correct model."""
    model = CorrectModel()
    inputs = tf.random.normal([4, 8, 32])
    
    @tf.function(jit_compile=True)
    def compiled_forward(x):
        return model(x)
    
    try:
        output = compiled_forward(inputs)
        print(f"PASS: tf.function with correct model works, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"FAIL: tf.function with correct model raised exception: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Layer return validation fix (Issue #108142)")
    print("=" * 60)
    
    results = []
    
    print("\n1. Testing that Layer return raises error in eager mode...")
    results.append(test_layer_return_raises_error())
    
    print("\n2. Testing that correct model still works...")
    results.append(test_correct_model_works())
    
    print("\n3. Testing tf.function with correct model...")
    results.append(test_tf_function_with_correct_model())
    
    print("\n" + "=" * 60)
    if all(results):
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

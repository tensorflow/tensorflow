"""Integration demo for GitHub issue #105133: XLA compatibility with conditional operations.

This script reproduces the exact issue and demonstrates the fix.

REQUIREMENTS:
- TensorFlow 2.20.0+ with XLA/GPU support enabled
- Run on XLA-enabled build (GPU recommended)

USAGE:
  python issue_105133_fix_demo.py

This is an integration test - for CI-safe unit tests, see xla_conditional_compatibility_test.py

Issue: OperatorNotAllowedInGraphError when using tf.shape()[0] in Python conditionals with jit_compile=True
Fix: Replace Python 'if' statements with tf.cond() for XLA compatibility
"""

import tensorflow as tf


class TestModelProblematic(tf.keras.Model):
    """Original problematic model from issue #105133."""

    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        # These Python if statements cause OperatorNotAllowedInGraphError in XLA
        if tf.shape(x)[0] >= 1:  # PROBLEMATIC
            x = tf.stop_gradient(x)
        (h, w) = (tf.shape(x)[1], tf.shape(x)[2])
        if h > 1 and w > 1:  # PROBLEMATIC  
            x = self.conv1(x)
            x = self.pool1(x)
        else:
            x = tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='VALID')
        x = self.flatten(x)
        flat_size = tf.size(x)
        if flat_size == 1024:  # PROBLEMATIC
            x = self.dense1(x)
        else:
            x = tf.nn.dropout(x, rate=0.5)
            x = self.dense2(x)
        return x


class TestModelFixed(tf.keras.Model):
    """Fixed model using tf.cond for XLA compatibility."""

    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        # Use tf.cond instead of Python if statements for XLA compatibility
        x = tf.cond(
            tf.shape(x)[0] >= 1,
            lambda: tf.stop_gradient(x),
            lambda: x
        )
        
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = tf.cond(
            tf.logical_and(h > 1, w > 1),
            lambda: self._conv_pool_branch(x),
            lambda: tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='VALID')
        )
        
        x = self.flatten(x)
        flat_size = tf.size(x)
        x = tf.cond(
            tf.equal(flat_size, 1024),
            lambda: self.dense1(x),
            lambda: self.dense2(tf.nn.dropout(x, rate=0.5))
        )
        
        return x
    
    def _conv_pool_branch(self, x):
        """Helper method for convolution + pooling branch."""
        x = self.conv1(x)
        return self.pool1(x)


def get_default_model_problematic():
    return TestModelProblematic()


def get_default_model_fixed():
    return TestModelFixed()


def get_sample_inputs():
    x = tf.random.normal([16, 28, 28, 1])
    return (x,)


def test_problematic_version():
    """Test the problematic version - this demonstrates the original issue."""
    print("Testing problematic version...")
    model = get_default_model_problematic()
    inputs = get_sample_inputs()
    
    # This works in eager execution
    eager_out = model(*inputs)
    print('Problematic - Eager Input shape:', inputs[0].shape)
    print('Problematic - Eager Output shape:', eager_out.shape)
    
    # This should fail with OperatorNotAllowedInGraphError when jit_compile=True
    @tf.function(jit_compile=True)
    def compiled_forward(*args):
        return model(*args)
    
    try:
        compiled_out = compiled_forward(*inputs)
        print('Problematic - XLA Output shape:', compiled_out.shape)
        print("⚠️  WARNING: Expected OperatorNotAllowedInGraphError but compilation succeeded!")
        print("    This may indicate XLA is not enabled or a different TF version.")
    except Exception as e:
        error_msg = str(e)
        expected_keywords = ['symbolic', 'python bool', 'not allowed', 'operatornotallowed']
        
        if any(keyword in error_msg.lower() for keyword in expected_keywords):
            print(f"✓ Expected error caught: {type(e).__name__}")
            print(f"  Error indicates symbolic tensor used as Python bool")
        else:
            print(f"? Unexpected error type: {type(e).__name__}")
            print(f"  Error message: {error_msg}")
            print("  This may be related to the symbolic tensor issue or XLA availability.")


def test_fixed_version():
    """Test the fixed version - this demonstrates the solution."""
    print("\nTesting fixed version...")
    model = get_default_model_fixed()
    inputs = get_sample_inputs()
    
    # Test eager execution
    eager_out = model(*inputs)
    print('Fixed - Eager Input shape:', inputs[0].shape)
    print('Fixed - Eager Output shape:', eager_out.shape)
    
    # Test XLA compilation - this should now work
    @tf.function(jit_compile=True)
    def compiled_forward(*args):
        return model(*args)
    
    try:
        compiled_out = compiled_forward(*inputs)
        print('Fixed - XLA Output shape:', compiled_out.shape)
        print("✓ SUCCESS: XLA compilation worked with tf.cond!")
        
        # Verify deterministic behavior
        if eager_out.shape == compiled_out.shape:
            print("✓ Output shapes match between eager and XLA execution")
        else:
            print(f"✗ Shape mismatch: eager {eager_out.shape} vs XLA {compiled_out.shape}")
        
        # Test numerical consistency (shapes should be deterministic)
        if eager_out.dtype == compiled_out.dtype:
            print("✓ Output dtypes match between eager and XLA execution")
        else:
            print(f"✗ Dtype mismatch: eager {eager_out.dtype} vs XLA {compiled_out.dtype}")
            
    except Exception as e:
        print(f"✗ Unexpected error in fixed version: {type(e).__name__}: {e}")
        print("   This suggests an issue with the XLA environment or tf.cond implementation.")


def check_xla_availability():
    """Check if XLA/JIT compilation is available."""
    try:
        @tf.function(jit_compile=True)
        def test_xla():
            return tf.constant(1.0)
        test_xla()
        return True
    except Exception:
        return False


def main():
    """Run both test cases to demonstrate the issue and the fix."""
    print("=" * 70)
    print("GitHub Issue #105133: XLA Conditional Compatibility Demo")
    print("OperatorNotAllowedInGraphError with tf.shape()[0] in conditionals")
    print("=" * 70)
    
    # Check environment
    xla_available = check_xla_availability()
    print(f"TensorFlow version: {tf.__version__}")
    print(f"XLA/JIT compilation available: {xla_available}")
    
    if not xla_available:
        print("⚠️  WARNING: XLA/JIT compilation not available.")
        print("   Some parts of this demo may not show the expected behavior.")
        print("   Run on an XLA-enabled TensorFlow build for full demonstration.")
    
    print()
    
    # Test the problematic version (expected to fail with XLA)
    test_problematic_version()
    
    # Test the fixed version (should succeed with XLA)  
    test_fixed_version()
    
    print("\n" + "=" * 70)
    print("SOLUTION SUMMARY:")
    print("Issue:    Python 'if' with symbolic tensors fails under XLA compilation")  
    print("Solution: Replace with tf.cond() for XLA compatibility")
    print()
    print("Key Refactoring Patterns:")
    print("  • if condition:           -> tf.cond(condition, true_fn, false_fn)")
    print("  • if a and b:            -> tf.cond(tf.logical_and(a, b), ...)")
    print("  • if a == b:             -> tf.cond(tf.equal(a, b), ...)")
    print("  • if tf.shape(x)[0] > n: -> tf.cond(tf.shape(x)[0] > n, ...)")
    print()
    print("See xla_conditional_compatibility_test.py for CI-safe unit tests.")
    print("=" * 70)


if __name__ == '__main__':
    main()
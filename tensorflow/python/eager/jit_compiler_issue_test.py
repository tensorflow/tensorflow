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

import tensorflow as tf

from tensorflow.python.platform import test

class JitCompileIntegrationTest(test.TestCase):

  def testKerasLayerWithTfFunctionInsideJitCompile(self):
    """Regression test for GitHub Issue #105728.
    
    Ensures that a Keras layer method decorated with @tf.function
    can be called inside a @tf.function(jit_compile=True) block
    without crashing due to symbolic tensor tracing errors.
    """
    
    class AttentionLayer(tf.keras.layers.Layer):
      def __init__(self):
        super(AttentionLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(10)

      @tf.function
      def internal_fn(self, inputs):
        # This function caused the crash because trace_type_builder
        # attempted to convert the symbolic input to a numpy array.
        return tf.matmul(inputs, inputs, transpose_b=True)

      def call(self, inputs):
        # Call the internal tf.function
        processed = self.internal_fn(inputs)
        return self.dense(processed)

    # Setup model and inputs
    model = AttentionLayer()
    inputs = tf.random.normal((8, 10))

    # Define the XLA-compiled forward pass
    @tf.function(jit_compile=True)
    def compiled_forward(x):
      return model(x)

    # This should run without NotImplementedError
    try:
      out = compiled_forward(inputs)
    except NotImplementedError as e:
      self.fail(f"XLA compilation failed with NotImplementedError: {e}")
    
    # Verify output shape
    self.assertEqual(out.shape, (8, 10))

if __name__ == '__main__':
  test.main()
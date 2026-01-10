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

from tensorflow.python.eager import def_function
from tensorflow.python.module import module
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class JitCompileIntegrationTest(test.TestCase):

  def testNestedTfFunctionInsideJitCompile(self):
    """Regression test for GitHub Issue #105728.
    
    Ensures that a method decorated with @tf.function can be called 
    inside a @tf.function(jit_compile=True) block without crashing.
    """
    
    # Use module.Module instead of tf.Module to avoid Bazel namespace issues
    class AttentionModule(module.Module):
      def __init__(self):
        super(AttentionModule, self).__init__()
        # Use variables.Variable instead of tf.Variable
        self.w = variables.Variable(tf.random.normal((8, 10)))
        self.b = variables.Variable(tf.zeros([10]))

      @def_function.function
      def internal_fn(self, inputs):
        # This function caused the crash because trace_type_builder
        # attempted to convert the symbolic input to a numpy array.
        return math_ops.matmul(inputs, inputs, transpose_b=True)

      def __call__(self, inputs):
        # Call the internal tf.function
        processed = self.internal_fn(inputs)
        return math_ops.matmul(processed, self.w) + self.b

    # Setup model and inputs
    model = AttentionModule()
    inputs = tf.random.normal((8, 10))

    # Define the XLA-compiled forward pass
    @def_function.function(jit_compile=True)
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

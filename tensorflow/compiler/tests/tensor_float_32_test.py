# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests that the PrecisionConfig is set if TF32 is disabled."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import sysconfig


class TensorFloat32ConvTest(xla_test.XLATestCase):

  def tearDown(self):
    super().tearDown()
    config.enable_tensor_float_32_execution(True)

  def _test_fn(self, fn, inputs):
    with ops.device('device:{}:0'.format(self.device)):
      # Test with TF32 disabled
      config.enable_tensor_float_32_execution(False)
      compiled_fn = def_function.function(fn, jit_compile=True)
      hlo_text = compiled_fn.experimental_get_compiler_ir(*inputs)(stage='hlo')
      self.assertIn('operand_precision={highest,highest}', hlo_text)

      # Test the output is sufficiently precise by comparing with FP64 results
      out = compiled_fn(*inputs)
      sys_details = sysconfig.get_build_info()
      if sys_details["is_rocm_build"]: # MIOpen does not support fp64 data type
          f32_out = compiled_fn(*[math_ops.cast(x, 'float32') for x in inputs])
          self.assertAllClose(out, f32_out, rtol=1e-5, atol=1e-5)
      else:
          f64_out = compiled_fn(*[math_ops.cast(x, 'float64') for x in inputs])
          self.assertAllClose(out, f64_out, rtol=1e-5, atol=1e-5)

      # Test with TF32 enabled. Recompile fn because enabling TF32 does not
      # reset function cache.
      config.enable_tensor_float_32_execution(True)
      compiled_fn = def_function.function(fn, jit_compile=True)
      hlo_text = compiled_fn.experimental_get_compiler_ir(*inputs)(stage='hlo')
      # operand_precision is not in HLO if it's the default value.
      self.assertNotIn('operand_precision', hlo_text)

  def test_matmul(self):
    x = array_ops.fill((1024, 1024), 1 + 2**-12)
    y = array_ops.fill((1024, 1024), 1.0)

    def matmul(x, y):
      return math_ops.matmul(x, y)

    self._test_fn(matmul, [x, y])

  def test_batch_matmul(self):
    x = array_ops.fill((2, 1024, 1024), 1 + 2**-12)
    y = array_ops.fill((2, 1024, 1024), 1.0)

    def batch_matmul(x, y):
      return math_ops.matmul(x, y)

    self._test_fn(batch_matmul, [x, y])

  def test_conv2d(self):
    x = array_ops.fill((2, 20, 20, 32), 1 + 2**-12)
    y = array_ops.fill((3, 3, 32, 32), 1.0)

    def conv2d(x, y):
      return nn_ops.conv2d(x, y, [1, 1, 1, 1], padding='SAME')

    self._test_fn(conv2d, [x, y])

  def test_conv2d_backprop_input(self):
    y = array_ops.fill((3, 3, 32, 32), 1 + 2**-12)
    out_backprop = array_ops.fill((2, 20, 20, 32), 1.0)

    def conv2d_backprop_input(y, out_backprop):
      return nn_ops.conv2d_backprop_input(
          (2, 20, 20, 32), y, out_backprop, [1, 1, 1, 1], padding='SAME'
      )

    self._test_fn(conv2d_backprop_input, [y, out_backprop])

  def test_conv2d_backprop_filter(self):
    x = array_ops.fill((2, 20, 20, 32), 1 + 2**-12)
    out_backprop = array_ops.fill((2, 20, 20, 32), 1.0)

    def conv2d_backprop_filter(x, out_backprop):
      return nn_ops.conv2d_backprop_filter(
          x, (3, 3, 32, 32), out_backprop, [1, 1, 1, 1], padding='SAME'
      )

    self._test_fn(conv2d_backprop_filter, [x, out_backprop])


if __name__ == '__main__':
  ops.enable_eager_execution()
  googletest.main()

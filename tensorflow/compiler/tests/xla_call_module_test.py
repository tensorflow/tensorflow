# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for XLA call module op wrapper."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class XlaCallModuleOpTest(xla_test.XLATestCase):

  def _assertOpOutputMatchesExpected(self,
                                     op,
                                     args,
                                     expected,
                                     equality_fn=None):
    """Asserts op(*args) == expected."""
    with self.session() as session:
      with self.test_scope():
        placeholders = [
            array_ops.placeholder(dtypes.as_dtype(arg.dtype), arg.shape)
            for arg in args
        ]
        feeds = {placeholders[i]: args[i] for i in range(0, len(args))}
        output = op(*placeholders)
      result = session.run(output, feeds)
      if not equality_fn:
        equality_fn = self.assertAllClose
      equality_fn(result, expected, rtol=1e-3)

  def test_basic(self):
    x = np.array([1., 2., 3.], dtype=np.float32)

    def f(x):
      # sin(cos(x))
      module = """
module @jit_f.0 {
  func.func public @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %0 = mhlo.cosine %arg0 : tensor<3xf32>
    %1 = mhlo.sine %0 : tensor<3xf32>
    return %1 : tensor<3xf32>
  }
}
"""
      return xla.call_module([x], module=module, Tout=[x.dtype], Sout=[x.shape])

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

  def test_multiple_args_results(self):
    x = np.array([1., 2., 3.], dtype=np.float32)
    y = np.array([11., 12., 13., 14.], dtype=np.float64)

    def f(x, y):
      # (sin(x), cos(y))
      module = """
module @jit_f.0 {
  func.func public @main(%arg0: tensor<3xf32>, %arg1: tensor<4xf64>) -> (tensor<3xf32>, tensor<4xf64>) {
    %0 = mhlo.sine %arg0 : tensor<3xf32>
    %1 = mhlo.cosine %arg1 : tensor<4xf64>
    return %0, %1 : tensor<3xf32>, tensor<4xf64>
  }
}
"""
      return xla.call_module([x, y],
                             module=module,
                             Tout=[x.dtype, y.dtype],
                             Sout=[x.shape, y.shape])

    self._assertOpOutputMatchesExpected(f, (x, y), (np.sin(x), np.cos(y)))

  def test_dim_var_basic(self):
    x = np.arange(6, dtype=np.float32).reshape((2, 3))

    def f(x):  # x: f32[2, b]
      # Module takes another argument which is the value of b
      # (sin(x), x.shape[1])
      module = """
module @jit_f.0 {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<i32>) {
    %0 = mhlo.sine %arg1 : tensor<2x?xf32>
    return %0, %arg0 : tensor<2x?xf32>, tensor<i32>
  }
}
"""
      return xla.call_module([x],
                             module=module,
                             Tout=[x.dtype, np.int32],
                             Sout=[(None, 3), ()],
                             dim_args_spec=['0.1'])

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(x), x.shape[1]))


if __name__ == '__main__':
  # This test is using Tensorflow sessions which are not compatible with eager
  # mode.
  ops.disable_eager_execution()
  googletest.main()

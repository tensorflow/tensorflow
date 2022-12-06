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

import unittest
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
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
    %0 = stablehlo.cosine %arg0 : tensor<3xf32>
    %1 = stablehlo.sine %0 : tensor<3xf32>
    return %1 : tensor<3xf32>
  }
}
"""
      return xla.call_module([x], version=2,
                             module=module, Tout=[x.dtype], Sout=[x.shape])

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

  def test_basic_mhlo(self):
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
      return xla.call_module([x], version=1,
                             module=module, Tout=[x.dtype], Sout=[x.shape])

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

  def test_basic_disalow_mhlo(self):
    """Disallows MHLO in newer versions of the op."""
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
      return xla.call_module([x], version=2,
                             module=module, Tout=[x.dtype], Sout=[x.shape])

    with self.assertRaisesRegex(
        errors.InvalidArgumentError, 'Cannot deserialize computation'):
      self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

  def test_compare(self):
    x = np.uint32(2)
    res = np.bool_(True)

    def f(x):
      # return x >= 1
      module = """
module @jit_f_jax.0 {
  func.func public @main(%arg0: tensor<ui32>) -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<ui32>
    %1 = "stablehlo.compare"(%arg0, %0) {compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
    return %1 : tensor<i1>
  }
}
"""
      return xla.call_module([x], version=2,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[res.shape])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_multiple_args_results(self):
    x = np.array([1., 2., 3.], dtype=np.float32)
    y = np.array([11., 12., 13., 14.], dtype=np.float64)

    def f(x, y):
      # (sin(x), cos(y))
      module = """
module @jit_f.0 {
  func.func public @main(%arg0: tensor<3xf32>, %arg1: tensor<4xf64>) -> (tensor<3xf32>, tensor<4xf64>) {
    %0 = stablehlo.sine %arg0 : tensor<3xf32>
    %1 = stablehlo.cosine %arg1 : tensor<4xf64>
    return %0, %1 : tensor<3xf32>, tensor<4xf64>
  }
}
"""
      return xla.call_module([x, y], version=2,
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
    %0 = stablehlo.sine %arg1 : tensor<2x?xf32>
    return %0, %arg0 : tensor<2x?xf32>, tensor<i32>
  }
}
"""
      return xla.call_module([x],
                             version=2,
                             module=module,
                             Tout=[x.dtype, np.int32],
                             Sout=[(None, 3), ()],
                             dim_args_spec=['0.1'])

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(x), x.shape[1]))

  def test_dim_var_basic_wrapped(self):
    """Like dim_arg_var_basic, but with the wrapper already added."""
    x = np.arange(6, dtype=np.float32).reshape((2, 3))

    def f(x):  # x: f32[2, b]
      # Module takes another argument which is the value of b
      # (sin(x), x.shape[1])
      module = """
module @jit_f.0 {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<i32>) {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 1 : i64} : (tensor<2x?xf32>) -> tensor<i32>
    %arg1_new = tensor.cast %arg1 : tensor<2x?xf32> to tensor<2x?xf32>
    %0, %1 = call @dyn_main(%arg0_new, %arg1_new) : (tensor<i32>, tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<i32>)
    return %0, %1 : tensor<2x?xf32>, tensor<i32>
  }
  func.func private @dyn_main(%arg0: tensor<i32>, %arg1: tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<i32>) {
    %0 = stablehlo.sine %arg1 : tensor<2x?xf32>
    return %0, %arg0 : tensor<2x?xf32>, tensor<i32>
  }
}
"""
      return xla.call_module([x], version=2,
                             module=module,
                             Tout=[x.dtype, np.int32],
                             Sout=[(None, 3), ()],
                             dim_args_spec=['0.1'])

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(x), x.shape[1]))

  def test_dynamic_iota(self):
    x = np.ones((3, 5), dtype=np.int32)
    res = np.arange(x.shape[0], dtype=np.int32)

    def f(x):  # x: f32[b, 5]
      # return np.arange(x.shape[0], dtype=np.int32)
      module = """
module @jit_fun.1 {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?x5xi32>) -> tensor<?xi32> {
    %0 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %1 = "stablehlo.dynamic_iota"(%0) {iota_dimension = 0 : i64} : (tensor<1xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }
}
"""
      return xla.call_module([x,], version=2,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[(None,)],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_dynamic_broadcast_in_dim(self):
    x = np.ones((3, 4), dtype=np.float32)
    y = np.ones((2, 3, 4), dtype=np.float32)
    res = (np.broadcast_to(x, y.shape), x + y)

    def f(x, y):  # x: f32[b, 4]  y: f32[2, b, 4]
      # return (np.broadcast_to(x, y.shape), x + y)
      module = """
module @jit_fun.0 {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?x4xf32>, %arg2: tensor<2x?x4xf32>) -> (tensor<2x?x4xf32>, tensor<2x?x4xf32>) {
    %0 = stablehlo.constant dense<2> : tensor<1xi32>
    %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<4> : tensor<1xi32>
    %4 = "stablehlo.concatenate"(%0, %2, %3) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = "stablehlo.dynamic_broadcast_in_dim"(%arg1, %4) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<2x?x4xf32>
    %6 = stablehlo.add %5, %arg2 : (tensor<2x?x4xf32>, tensor<2x?x4xf32>) -> tensor<2x?x4xf32>
    return %5, %6 : tensor<2x?x4xf32>, tensor<2x?x4xf32>
  }
}
"""
      return xla.call_module([x, y], version=2,
                             module=module,
                             Tout=[res[0].dtype, res[1].dtype],
                             Sout=[(2, None, 4), (2, None, 4)],
                             dim_args_spec=['1.1'])

    self._assertOpOutputMatchesExpected(f, (x, y), res)

  @unittest.skip('TODO(necula): test is flaky')
  def test_reduce(self):
    x = np.arange(5, dtype=np.int32)
    res = np.sum(x) * x.shape[0]

    def f(x):  # x: i32[b]
      module = """
module @jit_fun{
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?xi32>) -> tensor<i32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [0] : (tensor<?xi32>, tensor<i32>) -> tensor<i32>
     reducer(%arg2: tensor<i32>, %arg3: tensor<i32>)  {
      %4 = mhlo.add %arg2, %arg3 : tensor<i32>
      "mhlo.return"(%4) : (tensor<i32>) -> ()
    }
    %2 = mhlo.multiply %1, %arg0 : tensor<i32>
    return %2 : tensor<i32>
  }
}
"""
      return xla.call_module([x], version=1,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[res.shape],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_call(self):
    """A chain of calls."""
    x = np.ones((5,), dtype=np.float32)
    res = np.arange(x.shape[0], dtype=np.int32)

    def f(x):  # x: f32[b]
      module = """
module @jit_fun_3 {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?xf32>) -> tensor<?xi32> {
    %0 = call @f(%arg0, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
  func.func private @f(%arg0: tensor<i32>, %arg1: tensor<?xf32>) -> tensor<?xi32> {
    %0 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %1 = "stablehlo.dynamic_iota"(%0) {iota_dimension = 0 : i64} : (tensor<1xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }
}
"""
      return xla.call_module([x,], version=2,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[()],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_identity(self):
    x = np.ones((5,), dtype=np.float32)
    res = x

    def f(x):  # x: f32[b]
      module = """
module @jit_fun_3 {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    return %arg1 : tensor<?xf32>
  }
}
"""
      return xla.call_module([
          x,
      ],
                             version=2,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[()],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))


if __name__ == '__main__':
  # This test is using Tensorflow sessions which are not compatible with eager
  # mode.
  ops.disable_eager_execution()
  googletest.main()

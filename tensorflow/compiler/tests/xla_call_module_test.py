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

  def test_dim_var_basic_dim_arg_i64(self):
    x = np.arange(6, dtype=np.float32).reshape((2, 3))

    def f(x):  # x: f32[2, b]
      # Module takes another argument which is the value of b
      # (sin(x), x.shape[1])
      module = """
module @jit_f.0 {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<i64>) {
    %0 = stablehlo.sine %arg1 : tensor<2x?xf32>
    return %0, %arg0 : tensor<2x?xf32>, tensor<i64>
  }
}
"""
      return xla.call_module([x],
                             version=2,
                             module=module,
                             Tout=[x.dtype, np.int64],
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

  def test_dim_args_spec_errors(self):
    # x, y: f32[2, b, c]
    x = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    y = x

    # Module takes two prefix arguments with the values of b and c
    #   return (sin(x + y), x.shape[1])
    module = """
module @jit_f.0 {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<2x?x?xf32>, %arg3: tensor<2x?x?xf32>) -> (tensor<2x?x?xf32>, tensor<i32>) {
    %0 = stablehlo.add %arg2, %arg3 : tensor<2x?x?xf32>
    %1 = stablehlo.sine %0 : tensor<2x?x?xf32>
    return %1, %arg0 : tensor<2x?x?xf32>, tensor<i32>
  }
}
"""

    dim_args_spec = ['0.1', '0.2']
    def f(x, y):
      return xla.call_module([x, y],
                             version=2,
                             module=module,
                             Tout=[x.dtype, np.int32],
                             Sout=[(None, 3), ()],
                             dim_args_spec=dim_args_spec)
    self._assertOpOutputMatchesExpected(f, (x, y), (np.sin(x + y), x.shape[1]))

    dim_args_spec = ['0.0', '0.0', '0.0', '0.0']  # Too many dim_args_spec
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'The module should have 4 dimension arguments, '
        'but it has only 4 total arguments'):
      self._assertOpOutputMatchesExpected(f, (x, y),
                                          (np.sin(x + y), x.shape[1]))

    dim_args_spec = ['0.0', '0.0', '0.0']  # dim_args_spec refers to non-scalar
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'Module argument at index 2 should be a 0-dimensional integer-tensor '
        'dimension argument but has type'):
      self._assertOpOutputMatchesExpected(f, (x, y),
                                          (np.sin(x + y), x.shape[1]))

    dim_args_spec = []  # No dim_args_spec
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'Module main has dynamic shapes but no dim_args_spec was given'):
      self._assertOpOutputMatchesExpected(f, (x, y),
                                          (np.sin(x + y), x.shape[1]))

    dim_args_spec = ['1.0']  # Too few dim_args_spec
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'Incorrect number of arguments for XlaCallModule: 2. '
        'The module has 4 of which 1 were declared to be dimension arguments.'):
      self._assertOpOutputMatchesExpected(f, (x, y),
                                          (np.sin(x + y), x.shape[1]))

    dim_args_spec = ['0.b', '0.1']  # axis_idx not a number
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Syntax error in dim_args_spec '0.b'"):
      self._assertOpOutputMatchesExpected(f, (x, y),
                                          (np.sin(x + y), x.shape[1]))

    dim_args_spec = ['2.0', '0.1']  # arg_idx too large
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'Invalid argument index 2 when the number of non-dimension arguments '
        "is 2 in dim_arg_spec '2.0'"):
      self._assertOpOutputMatchesExpected(f, (x, y),
                                          (np.sin(x + y), x.shape[1]))

    dim_args_spec = ['0.3', '0.1']  # axis_idx too large
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'Invalid axis index 3 when the rank of non-dimension argument 0 '
        "is 3 in dim_arg_spec '0.3'"):
      self._assertOpOutputMatchesExpected(f, (x, y),
                                          (np.sin(x + y), x.shape[1]))

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

  @unittest.skip('TODO(burmako): Shape inference leaves dynamic_reshape')
  def test_dynamic_reshape(self):
    x = np.ones((4, 3), dtype=np.float32)
    res = x.reshape((-1,))

    def f(x):  # x: f32[b, 3]
      module = """
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?x3xf32>) -> tensor<?xf32> {
    %0 = stablehlo.constant dense<3> : tensor<i32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.dynamic_reshape %arg1, %2 : (tensor<?x3xf32>, tensor<1xi32>) -> tensor<?xf32>
    return %3 : tensor<?xf32>
  }
}
"""
      return xla.call_module([x],
                             module=module,
                             Tout=[res.dtype],
                             Sout=[(None,)],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  @unittest.skip('TODO(burmako): Shape inference adds tf.Cast')
  def test_dynamic_reshape_cast(self):
    x = np.ones((4, 2, 3), dtype=np.float32)
    res = np.sin(x).reshape((4, -1))

    def f(x):  # x: f32[b, 2, 3]
      module = """
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?x2x3xf32>) -> tensor<?x6xf32> {
    %0 = stablehlo.sine %arg1 : tensor<?x2x3xf32>
    %1 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.constant dense<6> : tensor<1xi32>
    %3 = stablehlo.concatenate %1, %2, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %4 = stablehlo.dynamic_reshape %0, %3 : (tensor<?x2x3xf32>, tensor<2xi32>) -> tensor<?x6xf32>
    return %4 : tensor<?x6xf32>
  }
}
"""
      return xla.call_module([x],
                             module=module,
                             Tout=[res.dtype],
                             Sout=[(None, 6)],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  @unittest.skip('TODO(burmako): Crash in simplifyDynamicGatherToGather()')
  def test_dynamic_gather(self):
    x = np.ones((3, 4), dtype=np.float32)
    idx = np.array([2, 2], np.int32)
    res = x[idx]

    def f(x):  # x: f32[b, 4]
      module = """
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?x4xf32>) -> tensor<?x2xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.constant dense<0> : tensor<1xi64>
    %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<2> : tensor<1xi32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = "stablehlo.dynamic_gather"(%arg1, %1, %4) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], start_index_map = [1]>, indices_are_sorted = true} : (tensor<?x4xf32>, tensor<1xi64>, tensor<2xi32>) -> tensor<?x2xf32>
    return %5 : tensor<?x2xf32>
  }
}
"""
      return xla.call_module([x],
                             module=module,
                             Tout=[res.dtype],
                             Sout=[(None, 2)],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  @unittest.skip('TODO(burmako): Shape inference leaves real_dynamic_slice')
  def test_real_dynamic_slice(self):
    x = np.ones((3, 4), dtype=np.float32)
    res = x[-1, :]  # TODO(necula): adjust this, if not the right result

    def f(x):  # x: f32[b, 4]
      module = """
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?x4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.constant dense<-1> : tensor<i32>
    %1 = stablehlo.add %arg0, %0 : tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<0> : tensor<1xi32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.constant dense<4> : tensor<1xi32>
    %7 = stablehlo.concatenate %5, %6, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %10 = stablehlo.constant dense<1> : tensor<2xi32>
    %11 = stablehlo.real_dynamic_slice %arg1, %4, %7, %10 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x4xf32>
    %12 = stablehlo.reshape %11 : (tensor<1x4xf32>) -> tensor<4xf32>
    return %12 : tensor<4xf32>
  }
}
"""
      return xla.call_module([x],
                             module=module,
                             Tout=[x.dtype],
                             Sout=[(4,)],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  @unittest.skip('TODO(burmako): Module verification with dynamic_update_slice')
  def test_dynamic_update_slice(self):
    x = np.ones((3, 4), dtype=np.float32)
    idx = np.int32(-2)
    res = x   # The update should be a nop

    def f(x, idx):  # x: f32[b, 4]  idx: i32
      module = """
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?x4xf32>, %arg2: tensor<i32>) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.compare  LT, %arg2, %0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = stablehlo.add %arg2, %arg0 : tensor<i32>
    %3 = stablehlo.select %1, %2, %arg2 : tensor<i1>, tensor<i32>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.dynamic_update_slice %arg1, %arg1, %3, %4 : (tensor<?x4xf32>, tensor<?x4xf32>, tensor<i32>, tensor<i32>) -> tensor<?x4xf32>
    return %5 : tensor<?x4xf32>
  }
} 
"""
      return xla.call_module([x, idx],
                             module=module,
                             Tout=[res.dtype],
                             Sout=[(None, 4)],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x, idx), (res,))

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

  @unittest.skip('TODO(burmako): tf.Cast added after reduce')
  def test_reduce_broadcast(self):
    x = np.broadcast_to(np.arange(3, dtype=np.float32).reshape(3, 1), (3, 5))
    res = np.any(x, axis=1)   # TODO(necula): not sure this should be the result

    def f(x):  # x: f32[b, 5]
      module = """
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?x5xf32>) -> tensor<?x1xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [1] : (tensor<?x5xf32>, tensor<f32>) -> tensor<?xf32>
     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
      %6 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }
    %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<1> : tensor<1xi32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = stablehlo.dynamic_broadcast_in_dim %1, %4, dims = [0] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>
    return %5 : tensor<?x1xf32>
  }
}
"""
      return xla.call_module([x,],
                             module=module,
                             Tout=[res.dtype],
                             Sout=[(None, 1)],
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
      return xla.call_module([x],
                             version=2,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[()],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  @unittest.skip('TODO(burmako): Shape inference failure for while')
  def test_while(self):
    """A while loop with carryied dynamic shapes."""
    x = np.ones((5,), dtype=np.float32)
    # Compute the result in Pyton first
    res0 = x
    for i in range(5):
      res0 += np.arange(x.shape[0], dtype=np.float32)
    res1 = np.int64(i)

    def f(x):  # x: f32[b]
      module = """
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?xf32>) -> (tensor<?xf32>, tensor<i64>) {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1:2 = stablehlo.while(%iterArg = %arg1, %iterArg_0 = %0) : tensor<?xf32>, tensor<i64>
     cond {
      %2 = stablehlo.constant dense<5> : tensor<i64>
      %3 = stablehlo.compare  LT, %iterArg_0, %2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
      %3 = stablehlo.dynamic_iota %2, dim = 0 : (tensor<1xi32>) -> tensor<?xf32>
      %4 = stablehlo.add %iterArg, %3 : tensor<?xf32>
      %5 = stablehlo.constant dense<1> : tensor<i64>
      %6 = stablehlo.add %iterArg_0, %5 : tensor<i64>
      stablehlo.return %4, %6 : tensor<?xf32>, tensor<i64>
    }
    return %1#0, %1#1 : tensor<?xf32>, tensor<i64>
  }
}
"""
      return xla.call_module([x,], version=2,
                             module=module,
                             Tout=[res0.dtype, res1.dtype],
                             Sout=[(None,), res1.shape],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res0, res1))


if __name__ == '__main__':
  # This test is using Tensorflow sessions which are not compatible with eager
  # mode.
  ops.disable_eager_execution()
  googletest.main()

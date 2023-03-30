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
from typing import Tuple
import unittest

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


def serialize(module_str: str) -> Tuple[str, int]:
  # TODO(b/274838200): error importing xla_extension in OSS
  # target_version = '0.9.0'  # TODO(gleasonk): use APIs to get this
  # return xla_extension.mlir.serialize_portable_artifact(
  #     module_str, target_version), 4
  return module_str, 3


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

  def testing_platform(self):
    """Current testing platform, one of CPU, GPU, TPU."""
    if self.device in ['CPU', 'XLA_CPU']:
      return 'CPU'
    elif self.device in ['GPU', 'XLA_GPU']:
      if test.is_built_with_rocm():
        return 'ROCM'
      else:
        return 'CUDA'
    elif self.device in ['TPU', 'XLA_TPU']:
      return 'TPU'
    else:
      assert False, f'Unexpected {self.device=}'

  def test_basic(self):
    x = np.array([1., 2., 3.], dtype=np.float32)

    def f(x):
      # sin(cos(x))
      module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.cosine %arg0 : tensor<3xf32>
    %1 = stablehlo.sine %0 : tensor<3xf32>
    return %1 : tensor<3xf32>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module, Tout=[x.dtype], Sout=[x.shape])

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

  def test_compare(self):
    x = np.uint32(2)
    res = np.bool_(True)

    def f(x):
      # return x >= 1
      module, version = serialize("""
module @jit_f_jax.0 {
  func.func public @main(%arg0: tensor<ui32>) -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<ui32>
    %1 = "stablehlo.compare"(%arg0, %0) {compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
    return %1 : tensor<i1>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[res.shape])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_multiple_args_results(self):
    x = np.array([1., 2., 3.], dtype=np.float32)
    y = np.array([11., 12., 13., 14.], dtype=np.float64)

    def f(x, y):
      # (sin(x), cos(y))
      module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: tensor<3xf32>, %arg1: tensor<4xf64>) -> (tensor<3xf32>, tensor<4xf64>) {
    %0 = stablehlo.sine %arg0 : tensor<3xf32>
    %1 = stablehlo.cosine %arg1 : tensor<4xf64>
    return %0, %1 : tensor<3xf32>, tensor<4xf64>
  }
}
""")
      return xla.call_module([x, y], version=version,
                             module=module,
                             Tout=[x.dtype, y.dtype],
                             Sout=[x.shape, y.shape])

    self._assertOpOutputMatchesExpected(f, (x, y), (np.sin(x), np.cos(y)))

  def test_dim_var_basic(self):
    x = np.arange(6, dtype=np.float32).reshape((2, 3))

    def f(x):  # x: f32[2, b]
      # Module takes another argument which is the value of b
      # (sin(x), x.shape[1])
      module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<i32>) {
    %0 = stablehlo.sine %arg1 : tensor<2x?xf32>
    return %0, %arg0 : tensor<2x?xf32>, tensor<i32>
  }
}
""")
      return xla.call_module([x], version=version,
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
      module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<i64>) {
    %0 = stablehlo.sine %arg1 : tensor<2x?xf32>
    return %0, %arg0 : tensor<2x?xf32>, tensor<i64>
  }
}
""")
      return xla.call_module([x],
                             module=module, version=version,
                             Tout=[x.dtype, np.int64],
                             Sout=[(None, 3), ()],
                             dim_args_spec=['0.1'])

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(x), x.shape[1]))

  def test_dim_var_basic_wrapped(self):
    """Like dim_arg_var_basic, but with the wrapper already added."""
    x = np.arange(6, dtype=np.float32).reshape((2, 3))

    def f(x):  # x: f32[2, b]
      # (sin(x), x.shape[1])
      module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg1: tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<i32>) {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 1 : i64} : (tensor<2x?xf32>) -> tensor<i32>
    %0, %1 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<i32>)
    return %0, %1 : tensor<2x?xf32>, tensor<i32>
  }
  func.func private @dyn_main(%arg0: tensor<i32>, %arg1: tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<i32>) {
    %0 = stablehlo.sine %arg1 : tensor<2x?xf32>
    return %0, %arg0 : tensor<2x?xf32>, tensor<i32>
  }
}
""")
      return xla.call_module([x],
                             module=module, version=version,
                             Tout=[x.dtype, np.int32],
                             Sout=[(None, 3), ()])

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(x), x.shape[1]))

  def test_dim_args_spec_errors(self):
    # x, y: f32[2, b, c]
    x = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    y = x

    # Module takes two prefix arguments with the values of b and c
    #   return (sin(x + y), x.shape[1])
    module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<2x?x?xf32>, %arg3: tensor<2x?x?xf32>) -> (tensor<2x?x?xf32>, tensor<i32>) {
    %0 = stablehlo.add %arg2, %arg3 : tensor<2x?x?xf32>
    %1 = stablehlo.sine %0 : tensor<2x?x?xf32>
    return %1, %arg0 : tensor<2x?x?xf32>, tensor<i32>
  }
}
""")

    dim_args_spec = ['0.1', '0.2']
    def f(x, y):
      return xla.call_module([x, y],
                             module=module, version=version,
                             Tout=[x.dtype, np.int32],
                             Sout=[(None, 3), ()],
                             dim_args_spec=dim_args_spec)
    self._assertOpOutputMatchesExpected(f, (x, y), (np.sin(x + y), x.shape[1]))

    dim_args_spec = ['0.0', '0.0', '0.0', '0.0']  # Too many dim_args_spec
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'The module should have 0 platform index arguments and '
        '4 dimension arguments, '
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

    dim_args_spec = ['1.0']  # Too few dim_args_spec
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'Incorrect number of arguments passed to XlaCallModule: 2. '
        'The module takes 4 arguments of which 0 platform index arguments '
        'and 1 dimension arguments.'):
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

  def test_platforms_basic(self):
    x = np.float32(0.)

    #  returns x + 2. on CPU, x + 3. on GPU and x + 4. on TPU
    module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg_platform_idx: tensor<i32>, %arg0: tensor<f32>) -> tensor<f32> {
    %to_add = "stablehlo.case"(%arg_platform_idx) ({
      %cpu_val = stablehlo.constant dense<2.> : tensor<f32>
      stablehlo.return %cpu_val : tensor<f32>
    }, {
      %gpu_val = stablehlo.constant dense<3.> : tensor<f32>
      stablehlo.return %gpu_val : tensor<f32>
    }, {
      %tpu_val = stablehlo.constant dense<4.> : tensor<f32>
      stablehlo.return %tpu_val : tensor<f32>
    }) : (tensor<i32>) -> tensor<f32>
    %0 = stablehlo.add %arg0, %to_add : tensor<f32>
    return %0 : tensor<f32>
  }
}
""")

    platforms = ['CPU', 'CUDA', 'ROCM', 'TPU']
    def f(x):
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[np.float32],
                             Sout=[()],
                             platforms=platforms)

    expected_value = x + dict(CPU=2., CUDA=3., ROCM=4., TPU=4.)[self.testing_platform()]
    self._assertOpOutputMatchesExpected(f, (x,), (expected_value,))

  def test_platforms_with_dim_vars(self):
    x = np.ones((3,), dtype=np.float32)
    y = np.arange(3., dtype=np.float32)

    #  returns x + x on CPU and x - x on TPU
    module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg_platform_idx: tensor<i32>, %arg_dim0: tensor<i32>, %arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %res = "stablehlo.case"(%arg_platform_idx) ({
      %0 = stablehlo.add %arg0, %arg1 : tensor<?xf32>
      stablehlo.return %0 : tensor<?xf32>
    }, {
      %1 = stablehlo.subtract %arg0, %arg1 : tensor<?xf32>
      stablehlo.return %1 : tensor<?xf32>
    }) : (tensor<i32>) -> tensor<?xf32>
    return %res : tensor<?xf32>
  }
}
""")
    def f(x, y):
      return xla.call_module([x, y], version=version,
                             module=module,
                             Tout=[np.float32],
                             Sout=[(None,)],
                             platforms=['CPU', 'TPU'],
                             dim_args_spec=['0.0'])

    expected_value = x + (y if self.testing_platform() == 'CPU' else -y)
    if self.testing_platform() in ['CPU', 'TPU']:
      self._assertOpOutputMatchesExpected(f, (x, y), (expected_value,))

  def test_platforms_errors(self):
    """Error reporting for the platforms attribute."""
    x = np.float32(0.)

    module_str = """
module @jit_f.0 {
  func.func public @main(%arg_platform_idx: tensor<i32>, %arg0: tensor<f32>) -> tensor<f32> {
    return %arg0 : tensor<f32>
  }
}
"""
    module, version = serialize(module_str)
    platforms = []
    def f(x):
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[np.float32],
                             Sout=[()],
                             platforms=platforms)

    # With empty platforms, there should be no platform_index argument
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'Incorrect number of arguments passed to XlaCallModule: 1. '
        'The module takes 2 arguments of which 0 platform index arguments '
        'and 0 dimension arguments.'):
      self._assertOpOutputMatchesExpected(f, (x,), (x,))

    # Same with a single platform
    platforms = ['CPU']
    if self.testing_platform() == 'CPU':
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          'Incorrect number of arguments passed to XlaCallModule: 1. '
          'The module takes 2 arguments of which 0 platform index arguments '
          'and 0 dimension arguments.'):
        self._assertOpOutputMatchesExpected(f, (x,), (x,))

    platforms = ['RANDOM_PLATFORM_1', 'RANDOM_PLATFORM_2']
    with self.assertRaisesRegex(
        errors.NotFoundError,
        'The current platform .* is not among the platforms'):
      self._assertOpOutputMatchesExpected(f, (x,), (x,))

    platforms = ['CPU', 'CUDA', 'ROCM']
    if self.testing_platform() not in platforms:
      with self.assertRaisesRegex(
          errors.NotFoundError,
          'The current platform .* is not among the platforms'):
        self._assertOpOutputMatchesExpected(f, (x,), (x,))
    else:
      self._assertOpOutputMatchesExpected(f, (x,), (x,))

    # The module cannot have i64 %arg_platform_idx
    module, version = serialize(module_str.replace('i32', 'i64'))
    platforms = ['CPU', 'CUDA', 'ROCM', 'TPU']
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'Module argument at index 0 should be a 0-dimensional '
        '32-bit integer-tensor platform index argument .* has type '
        'tensor<i64>'):
      self._assertOpOutputMatchesExpected(f, (x,), (x,))

    # A module without the platform index argument
    module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: tensor<i32>) -> tensor<i32> {
    return %arg0 : tensor<i32>
  }
}
""")
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'The module should have 1 platform index arguments and 0 dimension '
        'arguments, but it has only 1 total arguments'):
      self._assertOpOutputMatchesExpected(f, (x,), (x,))

  def test_dynamic_iota(self):
    x = np.ones((3, 5), dtype=np.int32)
    res = np.arange(x.shape[0], dtype=np.int32)

    def f(x):  # x: f32[b, 5]
      # return np.arange(x.shape[0], dtype=np.int32)
      module, version = serialize("""
module @jit_fun.1 {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?x5xi32>) -> tensor<?xi32> {
    %0 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %1 = "stablehlo.dynamic_iota"(%0) {iota_dimension = 0 : i64} : (tensor<1xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }
}
""")
      return xla.call_module([x,], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[(None,)],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_build_graph_with_any_platform(self):
    """We can construct the tf.Graph on all platforms."""
    x = np.float32(0.)

    module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg_platform_idx: tensor<i32>, %arg0: tensor<f32>) -> tensor<f32> {
    return %arg0 : tensor<f32>
  }
}
""")
    platforms = ['TPU']  # the module is compileable only on TPU
    def f(x):
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[np.float32],
                             Sout=[()],
                             platforms=platforms)
    tf_graph = def_function.function(f).get_concrete_function(x).graph
    self.assertIn('XlaCallModule', str(tf_graph.as_graph_def()))

  def test_dynamic_reshape(self):
    x = np.ones((4, 3), dtype=np.float32)
    res = x.reshape((-1,))

    def f(x):  # x: f32[b, 3]
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?x3xf32>) -> tensor<?xf32> {
    %0 = stablehlo.constant dense<3> : tensor<i32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.dynamic_reshape %arg1, %2 : (tensor<?x3xf32>, tensor<1xi32>) -> tensor<?xf32>
    return %3 : tensor<?xf32>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[(None,)],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_dynamic_gather(self):
    x = np.ones((3, 4), dtype=np.float32)
    res = np.ones((3, 2), dtype=np.float32)

    def f(x):  # x: f32[b, 4]
      module, version = serialize("""
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
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[(None, 2)],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_real_dynamic_slice(self):
    x = np.ones((3, 4), dtype=np.float32)
    res = x[-1, :]

    def f(x):  # x: f32[b, 4]
      module, version = serialize("""
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
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[x.dtype],
                             Sout=[(4,)],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_dynamic_update_slice(self):
    x = np.ones((3, 4), dtype=np.float32)
    idx = np.int32(-2)
    res = x   # The update should be a nop

    def f(x, idx):  # x: f32[b, 4]  idx: i32
      module, version = serialize("""
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
""")
      return xla.call_module([x, idx], version=version,
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
      module, version = serialize("""
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
""")
      return xla.call_module([x, y], version=version,
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
      module, version = serialize("""
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
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[res.shape],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_reduce_broadcast(self):
    x = np.broadcast_to(np.arange(3, dtype=np.float32).reshape(3, 1), (3, 5))
    res = np.arange(3, dtype=np.float32).reshape(3, 1) * 5

    def f(x):  # x: f32[b, 5]
      module, version = serialize("""
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
""")
      return xla.call_module([x,], version=version,
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
      module, version = serialize("""
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
""")
      return xla.call_module([x,], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[()],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_identity(self):
    x = np.ones((5,), dtype=np.float32)
    res = x

    def f(x):  # x: f32[b]
      module, version = serialize("""
module @jit_fun_3 {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    return %arg1 : tensor<?xf32>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[()],
                             dim_args_spec=['0.0'])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_while(self):
    """A while loop with carryied dynamic shapes."""
    x = np.ones((5,), dtype=np.float32)
    # Compute the result in Pyton first
    res0 = np.copy(x)
    for _ in range(5):
      res0 += np.arange(x.shape[0], dtype=np.float32)
    res1 = np.int64(5)

    def f(x):  # x: f32[b]
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<?xf32>) -> (tensor<?xf32>, tensor<i64>) {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1:2 = "stablehlo.while"(%arg1, %0) ({
    ^bb0(%arg2: tensor<?xf32>, %arg3: tensor<i64>):
      %2 = stablehlo.constant dense<5> : tensor<i64>
      %3 = stablehlo.compare  LT, %arg3, %2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }, {
    ^bb0(%arg2: tensor<?xf32>, %arg3: tensor<i64>):
      %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
      %3 = stablehlo.dynamic_iota %2, dim = 0 : (tensor<1xi32>) -> tensor<?xf32>
      %4 = stablehlo.add %arg2, %3 : tensor<?xf32>
      %5 = stablehlo.constant dense<1> : tensor<i64>
      %6 = stablehlo.add %arg3, %5 : tensor<i64>
      stablehlo.return %4, %6 : tensor<?xf32>, tensor<i64>
    }) : (tensor<?xf32>, tensor<i64>) -> (tensor<?xf32>, tensor<i64>)
    return %1#0, %1#1 : tensor<?xf32>, tensor<i64>
  }
}
""")
      return xla.call_module([x,], version=version,
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

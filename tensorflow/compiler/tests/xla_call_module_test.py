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

import os
import re
from typing import Optional, Sequence
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.mlir.stablehlo import stablehlo
from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


def serialize(module_str: str) -> tuple[str, int]:
  target = stablehlo.get_minimum_version()
  byte_str = stablehlo.serialize_portable_artifact_str(module_str, target)
  return byte_str, xla.call_module_maximum_supported_version()


class XlaCallModuleOpTest(xla_test.XLATestCase, parameterized.TestCase):

  def _assertOpOutputMatchesExpected(self,
                                     op,
                                     args,
                                     expected,
                                     equality_fn=None):
    """Asserts op(*args) == expected."""
    with self.test_scope():
      tf_func = def_function.function(op, autograph=False, jit_compile=True)
      result = tf_func(*args)

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
                             module=module, Tout=[x.dtype], Sout=[x.shape],
                             platforms=[self.testing_platform()])

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

  def test_basic_with_token_v8(self):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def f(x):
      # sin(cos(x))
      module, _ = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: !stablehlo.token, %arg1: tensor<3xf32>) -> (!stablehlo.token, tensor<3xf32>) {
    %0 = stablehlo.cosine %arg1 : tensor<3xf32>
    %1 = stablehlo.sine %0 : tensor<3xf32>
    return %arg0, %1 : !stablehlo.token, tensor<3xf32>
  }
}
""")
      return xla.call_module(
          [x],
          version=8,  # Version 8 uses only one prefix token
          module=module,
          Tout=[x.dtype],
          Sout=[x.shape],
          has_token_input_output=True,  # Version 8 cares about this
          platforms=[self.testing_platform()],
      )

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

  def test_basic_with_multiple_tokens(self):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def f(x):
      # sin(cos(x))
      module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: !stablehlo.token {jax.token = true}, %arg1: !stablehlo.token {jax.token = true}, %arg2: tensor<3xf32>) -> (!stablehlo.token, !stablehlo.token, tensor<3xf32>) {
    %0 = stablehlo.cosine %arg2 : tensor<3xf32>
    %1 = stablehlo.sine %0 : tensor<3xf32>
    return %arg0, %arg1, %1 : !stablehlo.token, !stablehlo.token, tensor<3xf32>
  }
}
""")
      return xla.call_module(
          [x],
          version=version,
          module=module,
          Tout=[x.dtype],
          Sout=[x.shape],
          platforms=[self.testing_platform()],
      )

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

  def test_basic_with_tokens_preceeded_by_other_args(self):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def f(x):
      # sin(cos(x))
      module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: tensor<i32>, %arg1: !stablehlo.token {jax.token = true}, %arg2: !stablehlo.token {jax.token = true}, %arg3: tensor<3xf32>) -> (!stablehlo.token, !stablehlo.token, tensor<3xf32>) {
    %0 = stablehlo.cosine %arg3 : tensor<3xf32>
    %1 = stablehlo.sine %0 : tensor<3xf32>
    return %arg1, %arg2, %1 : !stablehlo.token, !stablehlo.token, tensor<3xf32>
  }
}
""")
      return xla.call_module(
          [np.int32(0), x],
          version=version,
          module=module,
          Tout=[x.dtype],
          Sout=[x.shape],
          platforms=[self.testing_platform()],
      )

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
                             Sout=[res.shape],
                             platforms=[self.testing_platform()],)

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
                             Sout=[x.shape, y.shape],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x, y), (np.sin(x), np.cos(y)))

  # TODO(b/305813026): asan test failure for the i64 test variant.
  @parameterized.named_parameters(
      dict(testcase_name='_' + dim_var_type,
           dim_var_type=dim_var_type)
      for dim_var_type in ('i32',)
  )
  def test_poly_basic(self, *, dim_var_type: str):
    x = np.arange(6, dtype=np.float32).reshape((2, 3))

    def f(x):  # x: f32[2, b]
      # (sin(x), x.shape[1])
      module, version = serialize(f"""
module @jit_f.0 attributes {{jax.uses_shape_polymorphism = true}} {{
  func.func public @main(%arg1: tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<{dim_var_type}>) {{
    %arg0_new_i32 = "stablehlo.get_dimension_size"(%arg1) {{dimension = 1 : i64}} : (tensor<2x?xf32>) -> tensor<i32>
    %arg0_new = stablehlo.convert %arg0_new_i32 : (tensor<i32>) -> tensor<{dim_var_type}>
    %0, %1 = call @dyn_main(%arg0_new, %arg1) : (tensor<{dim_var_type}>, tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<{dim_var_type}>)
    return %0, %1 : tensor<2x?xf32>, tensor<{dim_var_type}>
  }}
  func.func private @dyn_main(%arg0: tensor<{dim_var_type}> {{jax.global_constant = "b"}}, %arg1: tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<{dim_var_type}>) {{
    %0 = stablehlo.sine %arg1 : tensor<2x?xf32>
    return %0, %arg0 : tensor<2x?xf32>, tensor<{dim_var_type}>
  }}
}}
""")
      return xla.call_module([x],
                             module=module, version=version,
                             Tout=[x.dtype, np.int32],
                             Sout=[(None, 3), ()],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(x), x.shape[1]))

  def test_wrong_actual_args_errors(self):
    x = np.arange(6, dtype=np.float32).reshape((3, 2))
    y = np.arange(6, dtype=np.int32).reshape((2, 3))

    # x: f32[a, 2], return x
    module, version = serialize("""
module @jit_f.0 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg0: tensor<?x2xf32>, %arg1: tensor<*xi32>) -> tensor<?x2xf32> {
    return %arg0 : tensor<?x2xf32>
  }
}
""")

    def f(x, y):
      return xla.call_module(
          [x, y],
          module=module,
          version=version,
          Tout=[x.dtype],
          Sout=[(None, 2)],
          platforms=[self.testing_platform()],
      )

    self._assertOpOutputMatchesExpected(f, (x, y), (x,))

    x_bad_etype = x.astype(np.int32)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        re.escape(
            'invalid refinement for argument 0, refinement element types must'
            ' match in tensor<?x2xf32> -> tensor<3x2xi32>'
        ),
    ):
      self._assertOpOutputMatchesExpected(f, (x_bad_etype, y), (x_bad_etype,))

    y_bad_etype = y.astype(np.float32)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        re.escape(
            'invalid refinement for argument 1, refinement element types must'
            ' match in tensor<*xi32> -> tensor<2x3xf32>'
        ),
    ):
      self._assertOpOutputMatchesExpected(f, (x, y_bad_etype), (x,))

    x_bad_shape = np.arange(15, dtype=np.float32).reshape(5, 3)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        re.escape(
            'invalid refinement for argument 0, refinement dimension sizes must'
            ' match for static dimensions in tensor<?x2xf32> -> tensor<5x3xf32>'
        ),
    ):
      self._assertOpOutputMatchesExpected(f, (x_bad_shape, y), (x_bad_shape,))

  @parameterized.named_parameters(
      dict(testcase_name='_' + platform_idx_type,
           platform_idx_type=platform_idx_type)
      for platform_idx_type in ('i32', 'i64')
  )
  def test_platforms_basic(self, *, platform_idx_type: str):
    x = np.float32(0.)

    #  returns x + 2. on CPU, x + 3. on GPU (CUDA or ROCM) and x + 4. on TPU
    module, version = serialize(f"""
module @jit_f.0 {{
  func.func public @main(%arg_platform_idx: tensor<{platform_idx_type}> {{jax.global_constant = "_platform_index"}}, %arg0: tensor<f32>) -> tensor<f32> {{
    %0 = stablehlo.convert %arg_platform_idx : (tensor<{platform_idx_type}>) -> tensor<i32>
    %to_add = "stablehlo.case"(%0) ({{
      %cpu_val = stablehlo.constant dense<2.> : tensor<f32>
      stablehlo.return %cpu_val : tensor<f32>
    }}, {{
      %gpu_val = stablehlo.constant dense<3.> : tensor<f32>
      stablehlo.return %gpu_val : tensor<f32>
    }}, {{
      %tpu_val = stablehlo.constant dense<4.> : tensor<f32>
      stablehlo.return %tpu_val : tensor<f32>
    }}) : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.add %arg0, %to_add : tensor<f32>
    return %1 : tensor<f32>
  }}
}}
""")

    platforms = ['CPU', 'CUDA', 'ROCM', 'TPU']
    def f(x):
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[np.float32],
                             Sout=[()],
                             platforms=platforms)

    expected_value = (
        x + dict(CPU=2.0, CUDA=3.0, ROCM=3.0, TPU=4.0)[self.testing_platform()]
    )
    self._assertOpOutputMatchesExpected(f, (x,), (expected_value,))

  def test_platforms_unknown_custom_call(self):
    # One of the platform branches ("ROCM") has custom call unknown to other
    # platforms.
    if self.testing_platform() == 'ROCM':
      raise unittest.SkipTest('Not intended for ROCM')
    x = np.float32(0.)

    #  returns x + 2. on CPU, x + 3. on GPU, and x + 4. on TPU
    module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<f32>) -> tensor<f32> {
    %to_add = "stablehlo.case"(%arg_platform_idx) ({
      %cpu_val = stablehlo.constant dense<2.> : tensor<f32>
      stablehlo.return %cpu_val : tensor<f32>
    }, {
      %gpu_val = stablehlo.constant dense<3.> : tensor<f32>
      stablehlo.return %gpu_val : tensor<f32>
    }, {
      %tpu_val = stablehlo.constant dense<4.> : tensor<f32>
      stablehlo.return %tpu_val : tensor<f32>
    }, {
      %rocm_val = stablehlo.custom_call @non_existent_target(%arg0) : (tensor<f32>) -> tensor<f32>
      stablehlo.return %rocm_val : tensor<f32>
    }) : (tensor<i32>) -> tensor<f32>
    %0 = stablehlo.add %arg0, %to_add : tensor<f32>
    return %0 : tensor<f32>
  }
}
""")

    platforms = ['CPU', 'CUDA', 'TPU', 'ROCM']
    def f(x):
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[np.float32],
                             Sout=[()],
                             platforms=platforms)

    expected_value = (
        x + dict(CPU=2.0, CUDA=3.0, TPU=4.0)[self.testing_platform()]
    )
    self._assertOpOutputMatchesExpected(f, (x,), (expected_value,))

  def test_platforms_and_poly(self):
    x = np.arange(6, dtype=np.float32)
    #  returns x + 2. on CPU, x + 3. on GPU (CUDA or ROCM) and x + 4. on TPU

    module, version = serialize("""
module @jit_f_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<?xf32>) -> (tensor<?xf32>) {
    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
    %5 = call @_wrapped_jax_export_main(%arg_platform_idx, %0, %arg0) : (tensor<i32>, tensor<i32>, tensor<?xf32>) -> tensor<?xf32>
    return %5 : tensor<?xf32>
  }

  func.func private @_wrapped_jax_export_main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> (tensor<?xf32>) {
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
    %1 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.dynamic_broadcast_in_dim %to_add, %1, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %4 = stablehlo.add %3, %arg1 : tensor<?xf32>
    return %4 : tensor<?xf32>
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

    expected_value = (
        x + dict(CPU=2.0, CUDA=3.0, ROCM=3.0, TPU=4.0)[self.testing_platform()]
    )
    self._assertOpOutputMatchesExpected(f, (x,), (expected_value,))

  def test_platforms_and_poly_and_tokens(self):
    x = np.arange(6, dtype=np.float32)
    #  returns x + 2. on CPU, x + 3. on GPU (CUDA or ROCM) and x + 4. on TPU

    module, version = serialize("""
module @jit_f_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg_tok: !stablehlo.token {jax.token = true}, %arg0: tensor<?xf32>) -> (!stablehlo.token, tensor<?xf32>) {
    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
    %5:2 = call @_wrapped_jax_export_main(%arg_platform_idx, %0, %arg_tok, %arg0) : (tensor<i32>, tensor<i32>, !stablehlo.token, tensor<?xf32>) -> (!stablehlo.token, tensor<?xf32>)
    return %5#0, %5#1 : !stablehlo.token, tensor<?xf32>
  }

  func.func private @_wrapped_jax_export_main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<i32> {jax.global_constant = "b"}, %arg_tok: !stablehlo.token {jax.token = true}, %arg1: tensor<?xf32>) -> (!stablehlo.token, tensor<?xf32>) {
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
    %1 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.dynamic_broadcast_in_dim %to_add, %1, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %4 = stablehlo.add %3, %arg1 : tensor<?xf32>
    return %arg_tok, %4 : !stablehlo.token, tensor<?xf32>
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

    expected_value = (
        x + dict(CPU=2.0, CUDA=3.0, ROCM=3.0, TPU=4.0)[self.testing_platform()]
    )
    self._assertOpOutputMatchesExpected(f, (x,), (expected_value,))

  # A module used for testing errors related to use of "platforms".
  platforms_errors_module_str = """
  module @jit_f.0 {
    func.func public @main(%arg_platform_idx: tensor<i32>, %arg0: tensor<f32>) -> tensor<f32> {
      return %arg0 : tensor<f32>
    }
  }
"""

  def platforms_errors_helper(
      self,
      *,
      module_str: str,
      platforms: Sequence[str] = ('CPU', 'CUDA', 'ROCM', 'TPU'),
      disabled_checks: Sequence[str] = (),
      expected_error: Optional[Exception] = None,
      expected_error_message: str = '',
  ):
    module, version = serialize(module_str)
    x = np.float32(0.0)

    def f(x):
      return xla.call_module(
          [x],
          version=version,
          module=module,
          Tout=[np.float32],
          Sout=[()],
          platforms=platforms,
          disabled_checks=disabled_checks,
      )

    if expected_error is None:
      self._assertOpOutputMatchesExpected(f, (x,), (x,))
    else:
      with self.assertRaisesRegex(expected_error, expected_error_message):
        self._assertOpOutputMatchesExpected(f, (x,), (x,))

  def platforms_errors_singleton_platform(self):
    # With singleton `platforms`, there should be no platform_index argument
    self.platforms_errors_helper(
        module_str=self.platforms_errors_module_str,
        platforms=(self.testing_platform(),),
        expected_error=errors.InvalidArgumentError,
        expected_error_message=(
            'Incorrect number of arguments passed to XlaCallModule = 1. The'
            ' module main function takes 2 arguments of which 0 platform index'
            ' arguments, 0 dimension arguments and 0 token arguments.'
        ),
    )

  def platforms_errors_no_platform_index_arg(self):
    module_str = self.platforms_errors_module_str.replace(
        '%arg_platform_idx: tensor<i32>, %arg0: tensor<f32>', ''
    )
    self.platforms_errors_helper(
        module_str=module_str,
        expected_error=errors.InvalidArgumentError,
        expected_error_message=(
            'The module should have a platform index argument but it has no '
            'arguments'
        ),
    )

  def platforms_errors_platform_index_i16(self):
    module_str = self.platforms_errors_module_str.replace('i32', 'i16')
    self.platforms_errors_helper(
        module_str=module_str,
        expected_error=errors.InvalidArgumentError,
        expected_error_message=(
            'Module argument at index 0 should be a 0-dimensional '
            '32-bit or 64-bit integer-tensor platform index argument '
            '.* has type tensor<i16>'
        ),
    )

  def platforms_errors_platform_index_non_scalar(self):
    module_str = self.platforms_errors_module_str.replace(
        'tensor<i32>', 'tensor<1xi32>'
    )
    self.platforms_errors_helper(
        module_str=module_str,
        expected_error=errors.InvalidArgumentError,
        expected_error_message=(
            'Module argument at index 0 should be a 0-dimensional '
            '32-bit integer-tensor platform index argument .* has type '
            'tensor<1xi32>'
        ),
    )

  def platforms_errors_platform_index_unranked(self):
    module_str = self.platforms_errors_module_str.replace(
        'tensor<i32>', 'tensor<*xi32>'
    )
    self.platforms_errors_helper(
        module_str=module_str,
        expected_error=errors.InvalidArgumentError,
        expected_error_message=(
            'Module argument at index 0 should be a 0-dimensional '
            '32-bit integer-tensor platform index argument'
        ),
    )

  def platforms_errors_different_from_current(self):
    platform_check_disabled_by_flags = (
        '--tf_xla_call_module_disabled_checks=platform'
        in os.getenv('TF_XLA_FLAGS', '')
    )
    self.platforms_errors_helper(
        module_str=self.platforms_errors_module_str,
        platforms=['RANDOM_PLATFORM_1', 'RANDOM_PLATFORM_2'],
        expected_error=(
            None if platform_check_disabled_by_flags else errors.NotFoundError
        ),
        expected_error_message='current platform .* is not among the platforms'
    )

  def platforms_errors_dissabled_check(self):
    self.platforms_errors_helper(
        module_str=self.platforms_errors_module_str,
        platforms=('RANDOM_PLATFORM_1', 'RANDOM_PLATFORM_2'),
        disabled_checks=(xla.call_module_disable_check_platform(),),
        expected_error=None,
        expected_error_message='current platform .* is not among the platforms'
    )

  def platforms_errors_empty(self):
    self.platforms_errors_helper(
        module_str=self.platforms_errors_module_str,
        platforms=[],
        disabled_checks=[xla.call_module_disable_check_platform()],
        expected_error=None,
        expected_error_message='current platform .* is not among the platforms'
    )

  def test_shape_assertion_success(self):
    x = np.ones((3, 5), dtype=np.int32)
    res = np.int32(x.shape[0])

    def f(x):  # x: f32[b, 5] and b = 3
      # return x.shape[0]
      module, version = serialize("""
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x5xi32>) -> tensor<i32> {
    %b = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x5xi32>) -> tensor<i32>
    %3 = stablehlo.constant dense<3> : tensor<i32>
    %ok = stablehlo.compare  EQ, %b, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%ok) {
      error_message = "The error message",
      has_side_effect = true
    } : (tensor<i1>) -> ()
    return %b : tensor<i32>
  }

}
""")
      return xla.call_module([x,], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[res.shape],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_shape_assertion_failure(self):
    x = np.ones((3, 5), dtype=np.int32)
    res = np.int32(x.shape[0])

    def f(x):  # x: f32[b, 5] and b = 3, with a constraint b == 4.
      # return x.shape[0]
      module, version = serialize("""
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x5xi32>) -> tensor<i32> {
    %b = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x5xi32>) -> tensor<i32>
    %4 = stablehlo.constant dense<4> : tensor<i32>
    %5 = stablehlo.constant dense<5> : tensor<i32>
    %11 = stablehlo.constant dense<11> : tensor<i32>
    %ok = stablehlo.compare  EQ, %b, %4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%ok, %b, %4, %5, %4, %5, %4, %5, %4, %5, %4, %5, %11) {
      error_message = "Expecting {0} == {1}. Extra {2,=5}, {3}, {{0}, {4}, {5}, {6}, {7}, {11}.",
      has_side_effect = true
    } : (tensor<i1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> ()
    return %b : tensor<i32>
  }
}
""")
      return xla.call_module([x,], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[res.shape],
                             platforms=[self.testing_platform()],)

    # This test runs as part of two targets, with and without
    # disabling shape_assertions.
    disabled_shape_assertions_check = (
        '--tf_xla_call_module_disabled_checks=shape_assertions'
        in os.getenv('TF_XLA_FLAGS', ''))
    if disabled_shape_assertions_check:
      # No error even though the constraint is false.
      self._assertOpOutputMatchesExpected(f, (x,), (res,))
    else:
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          re.escape('Expecting 3 == 4. Extra   5  , 4, {0}, 5, 4, 5, 4, 11.')):
        self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_invalid_shape_assertion(self):
    arg_i1 = np.bool_(True)
    arg_i32 = np.int32(2)
    res = arg_i32

    # This test runs as part of two targets, with and without
    # disabling shape_assertions.
    disabled_shape_assertions_check = (
        '--tf_xla_call_module_disabled_checks=shape_assertions'
        in os.getenv('TF_XLA_FLAGS', ''))
    if disabled_shape_assertions_check:
      self.skipTest('Test is N/A when shape_assertions are disabled')

    subtest_count = 1
    def one_subtest(error_msg: str, module_str: str):
      def f(*args):
        module, version = serialize(module_str)
        return xla.call_module(
            list(args),
            version=version,
            module=module,
            Tout=[res.dtype],
            Sout=[res.shape],
            platforms=[self.testing_platform()],
        )

      nonlocal subtest_count
      subtest_count += 1
      with self.subTest(count=subtest_count, error_msg=error_msg):
        with self.assertRaisesRegex(errors.InvalidArgumentError, error_msg):
          self._assertOpOutputMatchesExpected(f, (arg_i1, arg_i32), (res,))

    one_subtest(
        'expects assert_what .* to be a constant of type tensor<i1>',
        """
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {
    %ok = stablehlo.constant dense<0> : tensor<i32>
    stablehlo.custom_call @shape_assertion(%ok) {
      error_message = "Some error",
      has_side_effect = true
    } : (tensor<i32>) -> ()
    return %arg_i32 : tensor<i32>
  }
}
""",
    )

    one_subtest(
        'expects static assert_what',
        """
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {
    stablehlo.custom_call @shape_assertion(%arg_i1) {
      error_message = "Some error",
      has_side_effect = true
    } : (tensor<i1>) -> ()
    return %arg_i32 : tensor<i32>
  }
}
""",
    )

    one_subtest(
        '`shape_assertion` custom calls must set `has_side_effect = true`.',
        """
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {
    %ok = stablehlo.constant dense<false> : tensor<i1>
    stablehlo.custom_call @shape_assertion(%ok) {
      error_message = "Some error",
      has_side_effect = false
    } : (tensor<i1>) -> ()
    return %arg_i32 : tensor<i32>
  }
}
""",
    )

    one_subtest(
        'expects error_message .* Found specifier {0}',
        """
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {
    %ok = stablehlo.constant dense<false> : tensor<i1>
    stablehlo.custom_call @shape_assertion(%ok) {
      error_message = "Some error {0}",
      has_side_effect = true
    } : (tensor<i1>) -> ()
    return %arg_i32 : tensor<i32>
  }
}
""",
    )

    one_subtest(
        'expects static error_message_input',
        """
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {
    %ok = stablehlo.constant dense<false> : tensor<i1>
    stablehlo.custom_call @shape_assertion(%ok, %arg_i32) {
      error_message = "Some error {0}",
      has_side_effect = true
    } : (tensor<i1>, tensor<i32>) -> ()
    return %arg_i32 : tensor<i32>
  }
}
""",
    )

    one_subtest(
        'expects error_message_input .* to be a constant of type tensor<i32>',
        """
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {
    %ok = stablehlo.constant dense<false> : tensor<i1>
    %c = stablehlo.constant dense<2.0> : tensor<f32>
    stablehlo.custom_call @shape_assertion(%ok, %c) {
      error_message = "Some error {0}",
      has_side_effect = true
    } : (tensor<i1>, tensor<f32>) -> ()
    return %arg_i32 : tensor<i32>
  }
}
""",
    )

  def test_dynamic_iota(self):
    x = np.ones((3, 5), dtype=np.int32)
    res = np.arange(x.shape[0], dtype=np.int32)

    def f(x):  # x: f32[b, 5]
      # return np.arange(x.shape[0], dtype=np.int32)
      module, version = serialize("""
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x5xi32>) -> tensor<?xi32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x5xi32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x5xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x5xi32>) -> tensor<?xi32> {
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
                             platforms=[self.testing_platform()],)

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
    platforms = ['TPU']  # the module is compilable only on TPU
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
module @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x3xf32>) -> tensor<?xf32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x3xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x3xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x3xf32>) -> tensor<?xf32> {
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
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_dynamic_gather(self):
    x = np.ones((3, 4), dtype=np.float32)
    res = np.ones((3, 2), dtype=np.float32)

    def f(x):  # x: f32[b, 4]
      module, version = serialize("""
module @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x4xf32>) -> tensor<?x2xf32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x4xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x4xf32>) -> tensor<?x2xf32>
    return %0 : tensor<?x2xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x4xf32>) -> tensor<?x2xf32> {
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
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_real_dynamic_slice(self):
    x = np.ones((3, 4), dtype=np.float32)
    res = x[-1, :]

    def f(x):  # x: f32[b, 4]
      module, version = serialize("""
module @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x4xf32>) -> tensor<4xf32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x4xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x4xf32>) -> tensor<4xf32> {
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
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_dynamic_update_slice(self):
    x = np.ones((3, 4), dtype=np.float32)
    idx = np.int32(-2)
    res = x   # The update should be a nop

    def f(x, idx):  # x: f32[b, 4]  idx: i32
      module, version = serialize("""
module @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x4xf32>, %arg2: tensor<i32>) -> tensor<?x4xf32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x4xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1, %arg2) : (tensor<i32>, tensor<?x4xf32>, tensor<i32>) -> tensor<?x4xf32>
    return %0 : tensor<?x4xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x4xf32>, %arg2: tensor<i32>) -> tensor<?x4xf32> {
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
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x, idx), (res,))

  def test_dynamic_broadcast_in_dim(self):
    x = np.ones((3, 4), dtype=np.float32)
    y = np.ones((2, 3, 4), dtype=np.float32)
    res = (np.broadcast_to(x, y.shape), x + y)

    def f(x, y):  # x: f32[b, 4]  y: f32[2, b, 4]
      # return (np.broadcast_to(x, y.shape), x + y)
      module, version = serialize("""
module @jit_fun.0 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x4xf32>, %arg2: tensor<2x?x4xf32>) -> (tensor<2x?x4xf32>, tensor<2x?x4xf32>) {
    %arg0_new = "stablehlo.get_dimension_size"(%arg2) {dimension = 1 : i64} : (tensor<2x?x4xf32>) -> tensor<i32>
    %0, %1 = call @dyn_main(%arg0_new, %arg1, %arg2) : (tensor<i32>, tensor<?x4xf32>, tensor<2x?x4xf32>) -> (tensor<2x?x4xf32>, tensor<2x?x4xf32>)
    return %0, %1 : tensor<2x?x4xf32>, tensor<2x?x4xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x4xf32>, %arg2: tensor<2x?x4xf32>) -> (tensor<2x?x4xf32>, tensor<2x?x4xf32>) {
    %0 = stablehlo.constant dense<2> : tensor<1xi32>
    %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<4> : tensor<1xi32>
    %4 = "stablehlo.concatenate"(%0, %2, %3) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = "stablehlo.dynamic_broadcast_in_dim"(%arg1, %4) {broadcast_dimensions = array<i64: 1, 2>} : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<2x?x4xf32>
    %6 = stablehlo.add %5, %arg2 : (tensor<2x?x4xf32>, tensor<2x?x4xf32>) -> tensor<2x?x4xf32>
    return %5, %6 : tensor<2x?x4xf32>, tensor<2x?x4xf32>
  }
}
""")
      return xla.call_module([x, y], version=version,
                             module=module,
                             Tout=[res[0].dtype, res[1].dtype],
                             Sout=[(2, None, 4), (2, None, 4)],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x, y), res)

  @unittest.skip('TODO(necula): test is flaky')
  def test_reduce(self):
    x = np.arange(5, dtype=np.int32)
    res = np.sum(x) * x.shape[0]

    def f(x):  # x: i32[b]
      module, version = serialize("""
module @jit_fun attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?xi32>) -> tensor<i32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg2) {dimension = 0 : i64} : (tensor<?xi32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xi32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xi32>) -> tensor<i32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [0] : (tensor<?xi32>, tensor<i32>) -> tensor<i32>
     reducer(%arg2: tensor<i32>, %arg3: tensor<i32>)  {
      %4 = stablehlo.add %arg2, %arg3 : tensor<i32>
      "stablehlo.return"(%4) : (tensor<i32>) -> ()
    }
    %2 = stablehlo.multiply %1, %arg0 : tensor<i32>
    return %2 : tensor<i32>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[res.shape],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_reduce_broadcast(self):
    x = np.broadcast_to(np.arange(3, dtype=np.float32).reshape(3, 1), (3, 5))
    res = np.arange(3, dtype=np.float32).reshape(3, 1) * 5

    def f(x):  # x: f32[b, 5]
      module, version = serialize("""
module @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x5xf32>) -> tensor<?x1xf32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x5xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x5xf32>) -> tensor<?x1xf32>
    return %0 : tensor<?x1xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x5xf32>) -> tensor<?x1xf32> {
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
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_call(self):
    """A chain of calls."""
    x = np.ones((5,), dtype=np.float32)
    res = np.arange(x.shape[0], dtype=np.int32)

    def f(x):  # x: f32[b]
      module, version = serialize("""
module @jit_fun_3 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?xf32>) -> tensor<?xi32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> tensor<?xi32> {
    %0 = call @f(%arg0, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
  func.func private @f(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> tensor<?xi32> {
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
                             platforms=[self.testing_platform()])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_identity(self):
    x = np.ones((5,), dtype=np.float32)
    res = x

    def f(x):  # x: f32[b]
      module, version = serialize("""
module @jit_fun_3 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?xf32>) -> tensor<?xf32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    return %arg1 : tensor<?xf32>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[()],
                             platforms=[self.testing_platform()])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_while(self):
    """A while loop with carryied dynamic shapes."""
    x = np.ones((5,), dtype=np.float32)
    # Compute the result in Python first
    res0 = np.copy(x)
    for _ in range(5):
      res0 += np.arange(x.shape[0], dtype=np.float32)
    res1 = np.int64(5)

    def f(x):  # x: f32[b]
      module, version = serialize("""
module @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?xf32>) -> (tensor<?xf32>, tensor<i64>) {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?xf32>) -> tensor<i32>
    %0, %1 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> (tensor<?xf32>, tensor<i64>)
    return %0, %1 : tensor<?xf32>, tensor<i64>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> (tensor<?xf32>, tensor<i64>) {
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
                             platforms=[self.testing_platform()])

    self._assertOpOutputMatchesExpected(f, (x,), (res0, res1))

  def test_skip_shape_refinement(self):
    # We skipped the shape refinement, but there are dynamic shapes.
    x = np.ones((5,), dtype=np.float32)
    res = x

    module_attrs = ''  # attribute is missing
    def f(x):  # x: f32[b]
      module, version = serialize(f"""
module @jit_fun_3 {module_attrs} {{
  func.func public @main(%arg1: tensor<?xf32>) -> tensor<?xf32> {{
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {{dimension = 0 : i64}} : (tensor<?xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }}
  func.func private @dyn_main(%arg0: tensor<i32> {{jax.global_constant = "b"}}, %arg1: tensor<?xf32>) -> tensor<?xf32> {{
    return %arg1 : tensor<?xf32>
  }}
}}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[()],
                             platforms=[self.testing_platform()])

    module_attrs = ''  # attribute is missing
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Module has dynamic shapes'):
      self._assertOpOutputMatchesExpected(f, (x,), (res,))

    module_attrs = 'attributes {jax.uses_shape_polymorphism = false}'
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Module has dynamic shapes'):
      self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_uses_shape_polymorphism_before_version_8(self):
    x = np.ones((5,), dtype=np.float32)
    res = x

    def f(x):  # x: f32[b]
      # No `uses_shape_polymorphism` attribute, but it default for version 7
      version = 7
      module, _ = serialize("""
module @jit_fun_3 {
  func.func public @main(%arg1: tensor<?xf32>) -> tensor<?xf32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    return %arg1 : tensor<?xf32>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[()],
                             platforms=[self.testing_platform()])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_tf_call_function(self):
    """A TensorFlow function call inside StableHLO."""
    x = np.int32(2)
    y = np.int32(3)
    res = x + y

    @function.Defun(dtypes.int32, dtypes.int32)
    def foo(x, y):
      return x + y

    def f(x, y):
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {
      tf.backend_config = {called_index = 0}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res.dtype],
          Sout=[res.shape],
          platforms=[self.testing_platform()],
          function_list=(foo,),
      )

    self._assertOpOutputMatchesExpected(f, (x, y), (res,))

  def test_tf_call_function_multiple_funcs(self):
    """Multiple TensorFlow function calls inside StableHLO."""
    x = np.int32(2)
    y = np.int32(3)
    res = (x + y) + (x + y)

    @function.Defun(dtypes.int32, dtypes.int32)
    def foo(x, y):
      return x + y

    @function.Defun(dtypes.int32, dtypes.int32)
    def bar(x, y):
      return foo(x, y)

    def f(x, y):
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {
      tf.backend_config = {called_index = 0}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {
      tf.backend_config = {called_index = 1}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = stablehlo.custom_call @tf.call_tf_function(%0, %1) {
      tf.backend_config = {called_index = 1}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %2 : tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res.dtype],
          Sout=[res.shape],
          platforms=[self.testing_platform()],
          function_list=(foo, bar),
      )

    self._assertOpOutputMatchesExpected(f, (x, y), (res,))

  def test_shape_polymorphic_tf_call_function(self):
    """A TensorFlow function call inside StableHLO."""
    x = np.full((2,), 2, dtype=np.int32)
    y = np.full((2,), 3, dtype=np.int32)
    res = x + y

    @function.Defun(dtypes.int32, dtypes.int32)
    def foo(x, y):
      return x + y

    def f(x, y):
      module, version = serialize("""
module @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xi32>) -> tensor<i32>
    %1 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1, %0) {
      tf.backend_config = {called_index = 0},
      indices_of_shape_operands = dense<[2]> : tensor<1xi64>
    } : (tensor<?xi32>, tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res.dtype],
          Sout=[res.shape],
          platforms=[self.testing_platform()],
          function_list=(foo,),
      )

    self._assertOpOutputMatchesExpected(f, (x, y), (res,))

  def test_tf_call_function_with_token(self):
    """A TensorFlow function call inside StableHLO."""
    x = np.int32(2)
    y = np.int32(3)
    res = x + y

    @function.Defun(dtypes.int32, dtypes.int32)
    def foo(x, y):
      return x + y

    def f(x, y):
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func public @main(%arg0: !stablehlo.token, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (!stablehlo.token, tensor<i32>) {
    %0:2 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1, %arg2) {
      tf.backend_config = {called_index = 0, has_token_input_output = true}
    } : (!stablehlo.token, tensor<i32>, tensor<i32>) -> (!stablehlo.token, tensor<i32>)
    return %0#0, %0#1 : !stablehlo.token, tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res.dtype],
          Sout=[res.shape],
          platforms=[self.testing_platform()],
          function_list=(foo,),
      )

    self._assertOpOutputMatchesExpected(f, (x, y), (res,))

  def test_tf_call_function_nested(self):
    """Nested XlaCallModule inside TensorFlow function calls."""
    x = np.int32(2)
    y = np.int32(3)
    res = x + y

    @function.Defun(dtypes.int32, dtypes.int32)
    def add(x, y):
      return x + y

    @function.Defun(dtypes.int32, dtypes.int32)
    def nested_xla_call(x, y):
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {
      tf.backend_config = {called_index = 0}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res.dtype],
          Sout=[res.shape],
          platforms=[self.testing_platform()],
          function_list=(add,),
      )

    @function.Defun(dtypes.int32, dtypes.int32)
    def call(x, y):
      return nested_xla_call(x, y)

    def f(x, y):
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {
      tf.backend_config = {called_index = 0}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res.dtype],
          Sout=[res.shape],
          platforms=[self.testing_platform()],
          function_list=(call,),
      )

    self._assertOpOutputMatchesExpected(f, (x, y), (res,))

  def test_tf_call_function_nested_func_renaming(self):
    """Multiple custom calls with identically named private functions."""
    x = np.int32(2)
    y = np.int32(3)
    res0 = x + y
    res1 = x - y

    # Verify that multiple inner TF function calls with the same private
    # functions are properly renamed during StableHLO import. This test case is
    # carefully constructed such that one outer XlaCallModule op has two custom
    # calls, each of which has the same private "@call" function with different
    # body. This is to catch bugs in the func renaming logic.

    @function.Defun(dtypes.int32, dtypes.int32)
    def add(x, y):
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func private @call(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }

  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = func.call @call(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res0.dtype],
          Sout=[res0.shape],
          platforms=[self.testing_platform()],
      )

    @function.Defun(dtypes.int32, dtypes.int32)
    def subtract(x, y):
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func private @call(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }

  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = func.call @call(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res1.dtype],
          Sout=[res1.shape],
          platforms=[self.testing_platform()],
      )

    def f(x, y):
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {
      tf.backend_config = {called_index = 0}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {
      tf.backend_config = {called_index = 1}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0, %1 : tensor<i32>, tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res0.dtype, res1.dtype],
          Sout=[res0.shape, res1.shape],
          platforms=[self.testing_platform()],
          function_list=(add, subtract),
      )

    self._assertOpOutputMatchesExpected(f, (x, y), (res0, res1))

  def test_op_backward_compatibility(self):
    """Test for ensuring XlaCallModuleOp backward compatibility."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

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
      # Create the raw XlaCallModule op directly instead of calling
      # `xla.call_module`, which handles default values for unpresent
      # attributes.
      return gen_xla_ops.xla_call_module(
          [x],
          version=version,
          module=module,
          Tout=[x.dtype],
          Sout=[x.shape],
          platforms=[self.testing_platform()],
      )

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

  def test_op_backward_incompatibility(self):
    """Test for ensuring XlaCallModuleOp with invalid bytecode."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def f(x):
      # Use an invalid MLIR string that will fail to parse when loading the
      # call module op, emulating a backward incompatibility.
      corrupted_module = 'stablehlo.invalid_op'
      return gen_xla_ops.xla_call_module(
          [x],
          version=xla.call_module_maximum_supported_version(),
          module=corrupted_module,
          Tout=[x.dtype],
          Sout=[x.shape],
          platforms=[self.testing_platform()],
      )

    # Expect any error message to be included after `:`
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'Cannot deserialize computation: .+',
    ):
      f(x)


if __name__ == '__main__':
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(log_device_placement=True)
  )
  googletest.main()

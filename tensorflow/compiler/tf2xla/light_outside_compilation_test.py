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

from tensorflow.compiler.mlir.python.mlir_wrapper import filecheck_wrapper as fw
from tensorflow.compiler.tf2xla import test_ops_for_light_outside_compilation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class LightOutsideCompilationTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super().setUp()
    if not test_util.is_gpu_available():
      self.skipTest('Light outside compilation only works for GPUs now')

  def assertFilecheck(self, actual, expected):
    """Assert that FileCheck runs successfully."""
    self.assertTrue(fw.check(actual, expected))

  def test_static_tf_op(self):
    """Test operations with static shapes."""

    @def_function.function(jit_compile=True)
    def compiled_f(x):
      return test_ops_for_light_outside_compilation.test_static_tf(x)

    with context.device('/gpu:0'):
      z = random_ops.random_normal([2, 2])

      self.assertFilecheck(
          compiled_f.experimental_get_compiler_ir(z)('hlo'), r"""
          CHECK: f32[2,2]{1,0} custom-call(f32[2,2]{1,0} [[v:.*]]), custom_call_target="GenericTfCallbackGPU", api_version=API_VERSION_STATUS_RETURNING
          CHECK: \"name\":\"TestStaticTf\"
          """)

      self.assertAllClose(compiled_f(z), z)

  def test_unranked_output_error(self):
    """Test that we error out for unranked dynamic shape."""

    @def_function.function(jit_compile=True)
    def compiled_f():
      return test_ops_for_light_outside_compilation.dynamic_unranked()

    with context.device('/gpu:0'):

      with self.assertRaisesRegex(ValueError, 'Output 0 has unknown rank'):
        compiled_f.experimental_get_compiler_ir()()

  def test_dynamic_output_multidim(self):
    """Test that we correctly handle multi-dimensional dynamic output."""

    @def_function.function(jit_compile=True)
    def compiled_f(shape):
      return test_ops_for_light_outside_compilation.dynamic_multidim(shape)

    with context.device('/gpu:0'):

      # Rank is hardcoded to 5.
      shape = [3, 4, 5, 4, 3]
      hlo = compiled_f.experimental_get_compiler_ir(shape)('hlo_no_metadata')
      out = compiled_f(shape)

      self.assertFilecheck(
          hlo, r"""
          CHECK: f32[<=20,<=20,<=20,<=20,<=20]{4,3,2,1,0} custom-call(), custom_call_target="GenericTfCallbackGPU"
          CHECK: \"name\":\"DynamicMultidim\"
          """)
      self.assertAllClose(out, array_ops.ones(shape))

  def test_dynamic_output_tf_op(self):
    """Test that dynamic output is sliced properly to the size known at runtime."""

    @def_function.function(jit_compile=True)
    def compiled_f(x):
      return test_ops_for_light_outside_compilation.test_dynamic_tf(
          x, max_size=5)

    with context.device('/gpu:0'):
      z = random_ops.random_normal([10])
      out = compiled_f(z)
      hlo = compiled_f.experimental_get_compiler_ir(z)('hlo_no_metadata')

      self.assertFilecheck(
          hlo, r"""
          CHECK: f32[<=5]{0} custom-call(f32[10]{0} [[v:.*]]), custom_call_target="GenericTfCallbackGPU"
          CHECK: \"name\":\"TestDynamicTf\"
          """)
      self.assertAllClose(out, z[:2])
      self.assertEqual(len(out), 2)

  def test_dynamic_input(self):
    """Test dynamic input => returns bad status at runtime."""

    @def_function.function(jit_compile=True)
    def compiled_f(x):
      x = array_ops.unique(x).y
      return test_ops_for_light_outside_compilation.test_dynamic_tf(
          x, max_size=5)

    with context.device('/gpu:0'):
      z = random_ops.random_normal([10])

      with self.assertRaisesRegex(ValueError,
                                  'Input dynamic dimensions are not supported'):
        compiled_f.experimental_get_compiler_ir(z)()

  def test_multi_output_tf_op(self):
    """Test light outside compilation for mulitple outputs."""

    @def_function.function(jit_compile=True)
    def compiled_f(x):
      return test_ops_for_light_outside_compilation.test_static_multiple_output_tf(
          x)

    with context.device('/gpu:0'):
      z = random_ops.random_normal([2, 2])
      hlo = compiled_f.experimental_get_compiler_ir(z)('hlo_no_metadata')

      self.assertFilecheck(
          hlo, r"""
          CHECK: custom_call_target="GenericTfCallbackGPU", api_version=API_VERSION_STATUS_RETURNING
          CHECK: \"name\":\"TestStaticMultipleOutputTf\"
          """)
      self.assertAllClose(compiled_f(z)[0], z)
      self.assertAllClose(compiled_f(z)[1], z)

  def test_must_be_constant_tf_op(self):
    """Test operations with must-be-constant input."""

    @def_function.function(jit_compile=True)
    def compiled_f(x, y):
      return test_ops_for_light_outside_compilation.test_tf_must_be_constant(
          x, constant_to_add=y)

    with context.device('/gpu:0'):

      z = random_ops.random_normal([10])
      hlo = compiled_f.experimental_get_compiler_ir(z, 5)('hlo_no_metadata')

      self.assertFilecheck(
          hlo, r"""
          CHECK: custom-call(f32[10]{0} [[v:.*]]), custom_call_target="GenericTfCallbackGPU"
          CHECK: \"name\":\"TestTfMustBeConstant\"
          """)

      expected_output = [j + 5 for j in z]
      self.assertAllClose(compiled_f(z, 5), expected_output)


if __name__ == '__main__':
  ops.enable_eager_execution()
  googletest.main()

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

  def test_static_tf_op(self):
    """Test operations with static shapes."""

    @def_function.function(jit_compile=True)
    def compiled_f(x):
      return test_ops_for_light_outside_compilation.test_static_tf(x)

    with context.device('/gpu:0'):
      z = random_ops.random_normal([2, 2])
      hlo = compiled_f.experimental_get_compiler_ir(z)('hlo_no_metadata')

      self.assertIn(
          r'f32[2,2]{1,0} custom-call(f32[2,2]{1,0} %reshape.2), custom_call_target="GenericTfCallbackGPU", api_version=API_VERSION_STATUS_RETURNING, backend_config="{\"op\":{\"name\":\"TestStaticTf\",\"op\":\"TestStaticTf\",\"input\":[\"x\"],\"device\":\"\",\"attr\":{}},\"inputs\":[{\"buffer_description\":{\"shape\":{\"dim\":[{\"size\":\"2\",\"name\":\"\"},{\"size\":\"2\",\"name\":\"\"}],\"unknown_rank\":false},\"type\":\"DT_FLOAT\"}}],\"outputs\":[{\"buffer_description\":{\"shape\":{\"dim\":[{\"size\":\"2\",\"name\":\"\"},{\"size\":\"2\",\"name\":\"\"}],\"unknown_rank\":false},\"type\":\"DT_FLOAT\"}}]}',
          hlo)

      self.assertAllClose(compiled_f(z), z)

  def test_dynamic_output_tf_op(self):
    """Test dynamic output => returns bad status at runtime."""

    @def_function.function(jit_compile=True)
    def compiled_f(x):
      return test_ops_for_light_outside_compilation.test_dynamic_tf(
          x, max_size=5)

    with context.device('/gpu:0'):
      z = random_ops.random_normal([10])

      with self.assertRaisesRegex(ValueError, 'Found unknown dimension'):
        compiled_f.experimental_get_compiler_ir(z)()

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

      self.assertIn(
          r'custom_call_target="GenericTfCallbackGPU", api_version=API_VERSION_STATUS_RETURNING, backend_config="{\"op\":{\"name\":\"TestStaticMultipleOutputTf\",\"op\":\"TestStaticMultipleOutputTf\",\"input\":[\"x\"],\"device\":\"\",\"attr\":{}},\"inputs\":[{\"buffer_description\":{\"shape\":{\"dim\":[{\"size\":\"2\",\"name\":\"\"},{\"size\":\"2\",\"name\":\"\"}],\"unknown_rank\":false},\"type\":\"DT_FLOAT\"}}],\"outputs\":[{\"buffer_description\":{\"shape\":{\"dim\":[{\"size\":\"2\",\"name\":\"\"},{\"size\":\"2\",\"name\":\"\"}],\"unknown_rank\":false},\"type\":\"DT_FLOAT\"}},{\"buffer_description\":{\"shape\":{\"dim\":[{\"size\":\"2\",\"name\":\"\"},{\"size\":\"2\",\"name\":\"\"}],\"unknown_rank\":false},\"type\":\"DT_FLOAT\"}}]}',
          hlo)
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

      self.assertIn(
          r'custom-call(f32[10]{0} %reshape.2), custom_call_target="GenericTfCallbackGPU", api_version=API_VERSION_STATUS_RETURNING, backend_config="{\"op\":{\"name\":\"TestTfMustBeConstant\",\"op\":\"TestTfMustBeConstant\",\"input\":[\"x\",\"TestTfMustBeConstant/constant_to_add\"],\"device\":\"\",\"attr\":{}},\"inputs\":[{\"buffer_description\":{\"shape\":{\"dim\":[{\"size\":\"10\",\"name\":\"\"}],\"unknown_rank\":false},\"type\":\"DT_FLOAT\"}},{\"buffer_description\":{\"shape\":{\"dim\":[],\"unknown_rank\":false},\"type\":\"DT_INT32\"},\"value\":{\"dtype\":\"DT_INT32\",\"tensor_shape\":{\"dim\":[],\"unknown_rank\":false},\"version_number\":0,\"tensor_content\":\"BQAAAA==\",\"half_val\":[],\"float_val\":[],\"double_val\":[],\"int_val\":[],\"string_val\":[],\"scomplex_val\":[],\"int64_val\":[],\"bool_val\":[],\"dcomplex_val\":[],\"resource_handle_val\":[],\"variant_val\":[],\"uint32_val\":[],\"uint64_val\":[]}}],\"outputs\":[{\"buffer_description\":{\"shape\":{\"dim\":[{\"size\":\"10\",\"name\":\"\"}],\"unknown_rank\":false},\"type\":\"DT_FLOAT\"}}]}',
          hlo)

      expected_output = [j + 5 for j in z]
      self.assertAllClose(compiled_f(z, 5), expected_output)


if __name__ == '__main__':
  ops.enable_eager_execution()
  googletest.main()

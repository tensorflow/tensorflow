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

from absl.testing import parameterized

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class ContextCrossPlatformTpuTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      [(f'_{stage}', stage) for stage in ['hlo', 'hlo_serialized']]
  )
  def testGetCompilerIrOnGpuPlatform(self, stage):
    @def_function.function(jit_compile=True)
    def test_func(x):
      return 2.0 * x

    a = constant_op.constant(1.0)
    result = test_func.experimental_get_compiler_ir(a)(
        stage=stage, platform_name='GPU'
    )
    self.assertNotEmpty(result)

  @parameterized.named_parameters([
      (f'_{stage}', stage)
      for stage in [
          'optimized_hlo',
          'optimized_hlo_serialized',
          'optimized_hlo_proto_serialized',
          'optimized_hlo_dot',
      ]
  ])
  def testGetCompilerIrOnGpuPlatformOptimizedHlo(self, stage):
    @def_function.function(jit_compile=True)
    def test_func(x):
      return 2.0 * x

    a = constant_op.constant(1.0)
    with self.assertRaisesRegex(
        ValueError,
        'GetCompilerIr with requested stage is not supported on this device',
    ):
      _ = test_func.experimental_get_compiler_ir(a)(
          stage=stage, platform_name='GPU'
      )


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()

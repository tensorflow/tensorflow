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
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ContextCrossPlatformGpuTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      [(f'_{stage}', stage) for stage in ['hlo', 'hlo_serialized']]
  )
  def testGetCompilerIrOnTpuPlatform(self, stage):
    @def_function.function(jit_compile=True)
    def test_func(x):
      return 2 * x

    a = array_ops.ones((1000, 1000))  # 4 * 1000 * 1000 in bytes
    result = test_func.experimental_get_compiler_ir(a)(
        stage=stage, platform_name='TPU'
    )
    self.assertNotEmpty(result)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()

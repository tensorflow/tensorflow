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
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ContextTpuPlatformTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      [(f'_{stage}', stage) for stage in ['hlo', 'hlo_serialized']]
  )
  def testGetCompilerIrWithVariable(self, stage):
    with ops.device('TPU:0'):
      v = variables.Variable(1.0)

    @def_function.function(jit_compile=True)
    def test_func(x):
      return x * v

    a = constant_op.constant(1.0)
    result = test_func.experimental_get_compiler_ir(a)(
        stage=stage, platform_name='TPU'
    )
    self.assertNotEmpty(result)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()

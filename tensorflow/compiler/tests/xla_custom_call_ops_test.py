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
"""Tests for XLA custom call op wrapper."""

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class XlaCustomCallOpTest(xla_test.XLATestCase):

  def testXlaCustomCallOp(self):
    with ops.device('device:{}:0'.format(self.device)):

      def f(x, y):
        return xla.custom_call(
            args=(x, y),
            target_name='my_call',
            dtype=dtypes.int32,
            shape=(3, 4, 5),
            backend_config='my_backend_config')

      compiled_f = def_function.function(f, jit_compile=True)

      x = random_ops.random_normal([1, 2, 3], dtype=dtypes.float32)
      y = random_ops.random_normal([], dtype=dtypes.float32)
      hlo = compiled_f.experimental_get_compiler_ir(x, y)(stage='hlo')
      self.assertIn('s32[3,4,5]{2,1,0} custom-call(f32[1,2,3]{2,1,0}', hlo)
      self.assertIn('custom_call_target="my_call"', hlo)
      self.assertIn('backend_config="my_backend_config"', hlo)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()

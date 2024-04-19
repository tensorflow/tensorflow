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
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
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

  def testXlaCustomCallOpDoesntExist(self):
    with ops.device('device:{}:0'.format(self.device)):

      def f():
        return xla.custom_call(
            args=(1, 2),
            target_name='my_non_existing_call_target',
            dtype=dtypes.int32,
            shape=(),
            backend_config='my_backend_config',
        )

      with self.assertRaises(errors_impl.InvalidArgumentError):
        compiled_f = def_function.function(f, jit_compile=True)
        compiled_f()

  def testXlaCustomCallV2Op(self):
    with ops.device('device:{}:0'.format(self.device)):

      def f(x, y):
        return xla.custom_call_v2(
            'my_call',
            (x, y),
            (
                tensor_spec.TensorSpec((2, 3), dtypes.int32),
                tensor_spec.TensorSpec((5,), dtypes.float32),
            ),
            has_side_effect=True,
            backend_config='my_backend_config',
        )

      compiled_f = def_function.function(f, jit_compile=True)

      x = random_ops.random_normal([7, 11], dtype=dtypes.float32)
      y = random_ops.random_normal([13, 17, 19], dtype=dtypes.float32)
      hlo = compiled_f.experimental_get_compiler_ir(x, y)(stage='hlo')
      self.assertContainsInOrder([
          '= (s32[2,3]{1,0}, f32[5]{0}) custom-call(',
          'f32[7,11]{1,0}',
          'f32[13,17,19]{2,1,0}',
          'custom_call_target="my_call"',
          'custom_call_has_side_effect=true',
          'api_version=API_VERSION_STATUS_RETURNING_UNIFIED',
          'backend_config="my_backend_config"',
      ], hlo)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()

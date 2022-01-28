# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for compilation that involves constant arguments."""

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class ConstArgTest(xla_test.XLATestCase):
  # Pass constant value to XlaDynamicSlice's size parameter that must be found
  # by xla::ValueInference. Most often, constants are passed to op kernels
  # using XlaExpression with kind kConstant. To require value inference, this
  # model obfuscates the constant using operations `>=` and `where_v2`.

  def testValueInference(self):
    with self.session() as session:
      with self.test_scope():
        a = array_ops.placeholder(dtypes.int32, [], name="a")
        size = array_ops.reshape(array_ops.where_v2(a >= 0, 1, 0), [1])
        output = xla.dynamic_slice([11, 12, 13], [0], size)
      result = session.run(output, {a: 1})
      expected = [11]
      self.assertEqual(result, expected)


if __name__ == "__main__":
  googletest.main()

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for slicing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class ReshapeTest(xla_test.XLATestCase, parameterized.TestCase):

  @parameterized.named_parameters(('32_bit_index', dtypes.int32),
                                  ('64_bit_index', dtypes.int64))
  def testBasic(self, index_dtype):
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[2, 3])
        with self.test_scope():
          shape = constant_op.constant([3, 2], dtype=index_dtype)
          o = array_ops.reshape(i, shape)
        params = {
            i: [[1, 2, 3], [4, 5, 6]],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[1, 2], [3, 4], [5, 6]], result)


if __name__ == '__main__':
  googletest.main()

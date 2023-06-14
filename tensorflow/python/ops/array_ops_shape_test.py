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
"""Tests for shape op int64 output."""

from tensorflow.core.config import flags
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ArrayOpShapeSizeTest(test.TestCase):

  def testShapeInt64Flag(self):
    # The tf_shape_default_int64 flag should be set when this test runs
    self.assertTrue(flags.config().tf_shape_default_int64.value())
    s1 = array_ops.shape_v2(array_ops.zeros([1, 2]))
    self.assertEqual(s1.dtype, dtypes.int64)

  def testShapeInt64FlagTf1(self):
    # The tf_shape_default_int64 flag should be set when this test runs
    self.assertTrue(flags.config().tf_shape_default_int64.value())
    s1 = array_ops.shape(array_ops.zeros([1, 2]))
    self.assertEqual(s1.dtype, dtypes.int64)

  def testSizeInt64Flag(self):
    # The tf_shape_default_int64 flag should be set when this test runs
    self.assertTrue(flags.config().tf_shape_default_int64.value())
    s1 = array_ops.size_v2(array_ops.zeros([1, 2]))
    self.assertEqual(s1.dtype, dtypes.int64)

  def testSizeInt64FlagTf1(self):
    # The tf_shape_default_int64 flag should be set when this test runs
    self.assertTrue(flags.config().tf_shape_default_int64.value())
    s1 = array_ops.size(array_ops.zeros([1, 2]))
    self.assertEqual(s1.dtype, dtypes.int64)


if __name__ == "__main__":
  test.main()

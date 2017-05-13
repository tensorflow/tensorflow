# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for shape_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.signal.python.ops import shape_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class FramesTest(test.TestCase):

  def test_mapping_of_indices_without_padding(self):
    with self.test_session():
      tensor = constant_op.constant(np.arange(9152), dtypes.int32)
      tensor = array_ops.expand_dims(tensor, 0)

      result = shape_ops.frames(tensor, 512, 180)
      result = result.eval()

      expected = np.tile(np.arange(512), (49, 1))
      expected += np.tile(np.arange(49) * 180, (512, 1)).T

      expected = np.expand_dims(expected, axis=0)
      expected = np.array(expected, dtype=np.int32)

      self.assertAllEqual(expected, result)

  def test_mapping_of_indices_with_padding(self):
    with self.test_session():
      tensor = constant_op.constant(np.arange(10000), dtypes.int32)
      tensor = array_ops.expand_dims(tensor, 0)

      result = shape_ops.frames(tensor, 512, 192)
      result = result.eval()

      expected = np.tile(np.arange(512), (51, 1))
      expected += np.tile(np.arange(51) * 192, (512, 1)).T

      expected[expected >= 10000] = 0

      expected = np.expand_dims(expected, axis=0)
      expected = np.array(expected, dtype=np.int32)

      self.assertAllEqual(expected, result)


if __name__ == "__main__":
  test.main()

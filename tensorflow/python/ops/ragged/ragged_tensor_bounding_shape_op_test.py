# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ragged.bounding_shape."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_test_util
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorBoundingShapeOp(ragged_test_util.RaggedTensorTestCase):

  def testDocStringExample(self):
    # This is the example from ragged.bounding_shape.__doc__.
    rt = ragged_factory_ops.constant([[1, 2, 3, 4], [5], [], [6, 7, 8, 9],
                                      [10]])
    self.assertRaggedEqual(rt.bounding_shape(), [5, 4])

  def test2DRaggedTensorWithOneRaggedDimension(self):
    values = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    rt1 = ragged_tensor.RaggedTensor.from_row_splits(values, [0, 2, 5, 6, 6, 7])
    rt2 = ragged_tensor.RaggedTensor.from_row_splits(values, [0, 7])
    rt3 = ragged_tensor.RaggedTensor.from_row_splits(values, [0, 0, 7, 7])
    self.assertRaggedEqual(rt1.bounding_shape(), [5, 3])
    self.assertRaggedEqual(rt2.bounding_shape(), [1, 7])
    self.assertRaggedEqual(rt3.bounding_shape(), [3, 7])

  def test3DRaggedTensorWithOneRaggedDimension(self):
    values = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]
    rt1 = ragged_tensor.RaggedTensor.from_row_splits(values, [0, 2, 5, 6, 6, 7])
    rt2 = ragged_tensor.RaggedTensor.from_row_splits(values, [0, 7])
    rt3 = ragged_tensor.RaggedTensor.from_row_splits(values, [0, 0, 7, 7])
    self.assertRaggedEqual(rt1.bounding_shape(), [5, 3, 2])
    self.assertRaggedEqual(rt2.bounding_shape(), [1, 7, 2])
    self.assertRaggedEqual(rt3.bounding_shape(), [3, 7, 2])

  def testExplicitAxisOptimizations(self):
    rt = ragged_tensor.RaggedTensor.from_row_splits(b'a b c d e f g'.split(),
                                                    [0, 2, 5, 6, 6, 7])
    self.assertRaggedEqual(rt.bounding_shape(0), 5)
    self.assertRaggedEqual(rt.bounding_shape(1), 3)
    self.assertRaggedEqual(rt.bounding_shape([1, 0]), [3, 5])


if __name__ == '__main__':
  googletest.main()

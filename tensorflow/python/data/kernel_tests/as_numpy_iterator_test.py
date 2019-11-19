# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.Dataset.numpy()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import test


class AsNumpyIteratorTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.eager_only_combinations())
  def testBasic(self):
    ds = dataset_ops.Dataset.range(3)
    self.assertEqual([0, 1, 2], list(ds.as_numpy_iterator()))

  @combinations.generate(test_base.eager_only_combinations())
  def testNestedStructure(self):
    point = collections.namedtuple('Point', ['x', 'y'])
    ds = dataset_ops.Dataset.from_tensor_slices({
        'a': ([1, 2], [3, 4]),
        'b': [5, 6],
        'c': point([7, 8], [9, 10])
    })
    self.assertEqual([{
        'a': (1, 3),
        'b': 5,
        'c': point(7, 9)
    }, {
        'a': (2, 4),
        'b': 6,
        'c': point(8, 10)
    }], list(ds.as_numpy_iterator()))

  @combinations.generate(test_base.graph_only_combinations())
  def testNonEager(self):
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaises(RuntimeError):
      ds.as_numpy_iterator()

  def checkInvalidElement(self, element):
    ds = dataset_ops.Dataset.from_tensors(element)
    with self.assertRaisesRegex(TypeError,
                                '.*does not support datasets containing.*'):
      ds.as_numpy_iterator()

  @combinations.generate(test_base.eager_only_combinations())
  def testInvalidElements(self):
    self.checkInvalidElement(sparse_tensor.SparseTensorValue([[0]], [0], [1]))

  @combinations.generate(test_base.eager_only_combinations())
  def testRaggedElement(self):
    self.checkInvalidElement(
        ragged_tensor_value.RaggedTensorValue(
            np.array([0, 1, 2]), np.array([0, 1, 3], dtype=np.int64)))

  @combinations.generate(test_base.eager_only_combinations())
  def testDatasetElement(self):
    self.checkInvalidElement(dataset_ops.Dataset.range(3))

  @combinations.generate(test_base.eager_only_combinations())
  def testNestedNonTensorElement(self):
    tuple_elem = (constant_op.constant([1, 2, 3]), dataset_ops.Dataset.range(3))
    self.checkInvalidElement(tuple_elem)


if __name__ == '__main__':
  test.main()

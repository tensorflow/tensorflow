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
"""Tests for `tf.data.experimental.unbatch()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


@test_util.run_all_in_graph_and_eager_modes
class UnbatchTest(test_base.DatasetTestBase, parameterized.TestCase):

  def testUnbatchWithUnknownRankInput(self):
    dataset = dataset_ops.Dataset.from_tensors([0, 1, 2,
                                                3]).apply(batching.unbatch())
    self.assertDatasetProduces(dataset, range(4))

  def testUnbatchScalarDataset(self):
    data = tuple([math_ops.range(10) for _ in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    expected_types = (dtypes.int32,) * 3
    data = data.batch(2)
    self.assertEqual(expected_types, dataset_ops.get_legacy_output_types(data))
    data = data.apply(batching.unbatch())
    self.assertEqual(expected_types, dataset_ops.get_legacy_output_types(data))

    self.assertDatasetProduces(data, [(i,) * 3 for i in range(10)])

  def testUnbatchDatasetWithStrings(self):
    data = tuple([math_ops.range(10) for _ in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    data = data.map(lambda x, y, z: (x, string_ops.as_string(y), z))
    expected_types = (dtypes.int32, dtypes.string, dtypes.int32)
    data = data.batch(2)
    self.assertEqual(expected_types, dataset_ops.get_legacy_output_types(data))
    data = data.apply(batching.unbatch())
    self.assertEqual(expected_types, dataset_ops.get_legacy_output_types(data))

    self.assertDatasetProduces(
        data, [(i, compat.as_bytes(str(i)), i) for i in range(10)])

  def testUnbatchDatasetWithSparseTensor(self):
    st = sparse_tensor.SparseTensorValue(
        indices=[[i, i] for i in range(10)],
        values=list(range(10)),
        dense_shape=[10, 10])
    data = dataset_ops.Dataset.from_tensors(st)
    data = data.apply(batching.unbatch())
    data = data.batch(5)
    data = data.apply(batching.unbatch())
    expected_output = [
        sparse_tensor.SparseTensorValue([[i]], [i], [10]) for i in range(10)
    ]
    self.assertDatasetProduces(data, expected_output=expected_output)

  def testUnbatchDatasetWithDenseAndSparseTensor(self):
    st = sparse_tensor.SparseTensorValue(
        indices=[[i, i] for i in range(10)],
        values=list(range(10)),
        dense_shape=[10, 10])
    data = dataset_ops.Dataset.from_tensors((list(range(10)), st))
    data = data.apply(batching.unbatch())
    data = data.batch(5)
    data = data.apply(batching.unbatch())
    expected_output = [(i, sparse_tensor.SparseTensorValue([[i]], [i], [10]))
                       for i in range(10)]
    self.assertDatasetProduces(data, expected_output=expected_output)

  def testUnbatchSingleElementTupleDataset(self):
    data = tuple([(math_ops.range(10),) for _ in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    expected_types = ((dtypes.int32,),) * 3
    data = data.batch(2)
    self.assertEqual(expected_types, dataset_ops.get_legacy_output_types(data))
    data = data.apply(batching.unbatch())
    self.assertEqual(expected_types, dataset_ops.get_legacy_output_types(data))

    self.assertDatasetProduces(data, [((i,),) * 3 for i in range(10)])

  def testUnbatchMultiElementTupleDataset(self):
    data = tuple([(math_ops.range(10 * i, 10 * i + 10),
                   array_ops.fill([10], "hi")) for i in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    expected_types = ((dtypes.int32, dtypes.string),) * 3
    data = data.batch(2)
    self.assertAllEqual(expected_types,
                        dataset_ops.get_legacy_output_types(data))
    data = data.apply(batching.unbatch())
    self.assertAllEqual(expected_types,
                        dataset_ops.get_legacy_output_types(data))

    self.assertDatasetProduces(
        data,
        [((i, b"hi"), (10 + i, b"hi"), (20 + i, b"hi")) for i in range(10)])

  def testUnbatchEmpty(self):
    data = dataset_ops.Dataset.from_tensors(
        (constant_op.constant([]), constant_op.constant([], shape=[0, 4]),
         constant_op.constant([], shape=[0, 4, 0])))
    data = data.apply(batching.unbatch())
    self.assertDatasetProduces(data, [])

  def testUnbatchStaticShapeMismatch(self):
    data = dataset_ops.Dataset.from_tensors((np.arange(7), np.arange(8),
                                             np.arange(9)))
    with self.assertRaises(ValueError):
      data.apply(batching.unbatch())

  # Note: dynamic shape mismatch is graph specific test.
  @test_util.run_deprecated_v1
  def testSkipEagerUnbatchDynamicShapeMismatch(self):
    ph1 = array_ops.placeholder(dtypes.int32, shape=[None])
    ph2 = array_ops.placeholder(dtypes.int32, shape=None)
    data = dataset_ops.Dataset.from_tensors((ph1, ph2))
    data = data.apply(batching.unbatch())
    iterator = dataset_ops.make_initializable_iterator(data)
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      # Mismatch in the 0th dimension.
      sess.run(
          iterator.initializer,
          feed_dict={
              ph1: np.arange(7).astype(np.int32),
              ph2: np.arange(8).astype(np.int32)
          })
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(next_element)

      # No 0th dimension (i.e. scalar value) for one component.
      sess.run(
          iterator.initializer,
          feed_dict={
              ph1: np.arange(7).astype(np.int32),
              ph2: 7
          })
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(next_element)


if __name__ == "__main__":
  test.main()

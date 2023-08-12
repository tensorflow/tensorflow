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
import collections
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import test
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops


class AsNumpyIteratorTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.eager_only_combinations())
  def testBasic(self):
    ds = dataset_ops.Dataset.range(3)
    self.assertEqual([0, 1, 2], list(ds.as_numpy_iterator()))

  @combinations.generate(test_base.eager_only_combinations())
  def testImmutable(self):
    ds = dataset_ops.Dataset.from_tensors([1, 2, 3])
    arr = next(ds.as_numpy_iterator())
    with self.assertRaisesRegex(ValueError,
                                'assignment destination is read-only'):
      arr[0] = 0

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

  def _testInvalidElement(self, element):
    ds = dataset_ops.Dataset.from_tensors(element)
    with self.assertRaisesRegex(TypeError,
                                'is not supported for datasets that'):
      ds.as_numpy_iterator()

  @combinations.generate(test_base.eager_only_combinations())
  def testSparseElement(self):
    st = sparse_tensor.SparseTensor(
        indices=[(0, 0), (1, 1), (2, 2)], values=[1, 2, 3], dense_shape=(3, 3))
    ds = dataset_ops.Dataset.from_tensor_slices(st)
    dt = sparse_ops.sparse_tensor_to_dense(st)
    self.assertAllEqual(list(ds.as_numpy_iterator()), dt.numpy())

  @combinations.generate(test_base.eager_only_combinations())
  def testRaggedElement(self):
    lst = [[1, 2], [3], [4, 5, 6]]
    rt = ragged_factory_ops.constant([lst])
    # This dataset consists of exactly one ragged tensor.
    ds = dataset_ops.Dataset.from_tensor_slices(rt)
    expected = np.array([
        np.array([1, 2], dtype=np.int32),
        np.array([3], dtype=np.int32),
        np.array([4, 5, 6], dtype=np.int32)
    ], dtype=object)
    for actual in ds.as_numpy_iterator():
      self.assertEqual(len(actual), len(expected))
      for actual_arr, expected_arr in zip(actual, expected):
        self.assertTrue(np.array_equal(actual_arr, expected_arr),
                        f'{actual_arr} != {expected_arr}')

  @combinations.generate(test_base.eager_only_combinations())
  def testDatasetElement(self):
    self._testInvalidElement(dataset_ops.Dataset.range(3))

  @combinations.generate(test_base.eager_only_combinations())
  def testNestedNonTensorElement(self):
    tuple_elem = (constant_op.constant([1, 2, 3]), dataset_ops.Dataset.range(3))
    self._testInvalidElement(tuple_elem)

  @combinations.generate(test_base.eager_only_combinations())
  def testNoneElement(self):
    ds = dataset_ops.Dataset.from_tensors((2, None))
    self.assertDatasetProduces(ds, [(2, None)])

  @combinations.generate(combinations.times(
      test_base.eager_only_combinations(),
      combinations.combine(enable_async_ckpt=[True, False])
  ))
  def testCompatibleWithCheckpoint(self, enable_async_ckpt):
    ds = dataset_ops.Dataset.range(10)
    iterator = ds.as_numpy_iterator()
    ckpt = trackable_utils.Checkpoint(iterator=iterator)
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    for _ in range(5):
      next(iterator)

    prefix = os.path.join(self.get_temp_dir(), 'ckpt')
    save_path = ckpt.save(prefix, options=ckpt_options)
    self.assertEqual(5, next(iterator))
    self.assertEqual(6, next(iterator))
    restore_iter = ds.as_numpy_iterator()
    restore_ckpt = trackable_utils.Checkpoint(iterator=restore_iter)
    if enable_async_ckpt:
      ckpt.sync()  # Otherwise save may not finish yet
    restore_ckpt.restore(save_path)
    self.assertEqual(5, next(restore_iter))


if __name__ == '__main__':
  test.main()

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
"""Tests for the private `_RebatchDataset` transformation."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import image_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


class BatchSizesForWorkerTest(test_base.DatasetTestBase,
                              parameterized.TestCase):

  def _test(self, global_batch_size, num_workers, num_replicas_per_worker,
            is_batch_size_static):
    """Test that all constraints are met for given parameters."""
    if not is_batch_size_static:
      # Adding a constant value here prevents downstream computation from
      # statically deriving the value of global batch size when running
      # in graph mode.
      global_batch_size += constant_op.constant(0, dtypes.int64)

    batch_sizes_list = []
    for i in range(num_workers):
      batch_sizes_list.append(
          self.evaluate(
              distribute.batch_sizes_for_worker(global_batch_size, num_workers,
                                                num_replicas_per_worker, i)))
    for batch_sizes in batch_sizes_list:
      # Constraint (A): for any worker, len(batch_sizes) == W * R
      self.assertLen(batch_sizes, num_workers * num_replicas_per_worker)
      # Constraint (B): for any worker, sum(batch_sizes) == G
      self.assertAllEqual(np.sum(batch_sizes), global_batch_size)

    # Each per-worker batch is split into num_workers global steps
    for step_index in range(num_workers):
      actual_global_batch = 0
      offset = step_index * num_replicas_per_worker
      for batch_sizes in batch_sizes_list:
        actual_global_batch += np.sum(batch_sizes[offset:offset +
                                                  num_replicas_per_worker])
      # Constraint (C): for any step, batch size across all workers add up to G.
      self.assertAllEqual(
          global_batch_size,
          actual_global_batch,
      )

    # Constraint (D): Batch size of any two replicas differs by at most one
    self.assertLessEqual(np.max(batch_sizes_list) - np.min(batch_sizes_list), 1)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(is_batch_size_static=[True, False])))
  def testBasic(self, is_batch_size_static):
    # Manually verify basic test case.
    global_batch_size = 8
    num_workers = 2
    num_replicas_per_worker = 2
    for worker_index in range(4):
      batch_sizes = distribute.batch_sizes_for_worker(global_batch_size,
                                                      num_workers,
                                                      num_replicas_per_worker,
                                                      worker_index)
      self.assertAllEqual([2, 2, 2, 2],
                          tensor_util.constant_value(batch_sizes))
    self._test(global_batch_size, num_workers, num_replicas_per_worker,
               is_batch_size_static)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(is_batch_size_static=[True, False])))
  def testBatchSizeIndivisibleByNumWorkers(self, is_batch_size_static):
    global_batch_size = 4
    num_workers = 3
    num_replicas_per_worker = 1

    def get_batch_sizes_for_worker(worker_index):
      return tensor_util.constant_value(
          distribute.batch_sizes_for_worker(global_batch_size, num_workers,
                                            num_replicas_per_worker,
                                            worker_index))

    # Manually verify this test case.
    self.assertAllEqual([2, 1, 1], get_batch_sizes_for_worker(0))
    self.assertAllEqual([1, 1, 2], get_batch_sizes_for_worker(1))
    self.assertAllEqual([1, 2, 1], get_batch_sizes_for_worker(2))
    self._test(global_batch_size, num_workers, num_replicas_per_worker,
               is_batch_size_static)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(is_batch_size_static=[True, False])))
  def testBatchSizeIndivisibleByNumReplicas(self, is_batch_size_static):
    self._test(
        global_batch_size=4,
        num_workers=1,
        num_replicas_per_worker=5,
        is_batch_size_static=is_batch_size_static)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(is_batch_size_static=[True, False])))
  def testBatchSizeSmallerThanNumReplicas(self, is_batch_size_static):
    self._test(
        global_batch_size=4,
        num_workers=2,
        num_replicas_per_worker=5,
        is_batch_size_static=is_batch_size_static)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(is_batch_size_static=[True, False])))
  def testBatchSizeSmallerThanNumWorkers(self, is_batch_size_static):
    self._test(
        global_batch_size=4,
        num_workers=5,
        num_replicas_per_worker=1,
        is_batch_size_static=is_batch_size_static)


def _flat_shapes(dataset):
  return [
      ts.as_list()
      for ts in nest.flatten(dataset_ops.get_legacy_output_shapes(dataset))
  ]


class RebatchDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  ##############################################################################
  # The following tests exercise our static computation of output_shapes.
  ##############################################################################

  @combinations.generate(test_base.default_test_combinations())
  def testShapeInferenceNotAllBatchSizesEqual(self):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[2, 1, 1])
    expected_shapes = [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testShapeInferenceInputBatchDimDivisible(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[2, 2], drop_remainder=drop_remainder)
    expected_shapes = [[2]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testShapeInferenceInputBatchDimUnknown(self):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=False)
    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[2, 2], drop_remainder=False)
    expected_shapes = [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testShapeInferenceInputBatchDimUnknownWithDropRemainder(self):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=False)
    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[2, 2], drop_remainder=True)
    expected_shapes = [[2]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testShapeInferenceInputBatchDimIndivisible(self):
    dataset = dataset_ops.Dataset.range(10).batch(5, drop_remainder=True)
    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[2, 2], drop_remainder=False)
    expected_shapes = [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testShapeInferenceInputBatchDimIndivisibleWithDropRemainder(self):
    dataset = dataset_ops.Dataset.range(10).batch(5, drop_remainder=True)
    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[2, 2], drop_remainder=True)
    expected_shapes = [[2]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

  ##############################################################################
  # The following tests check _RebatchDataset's output.
  ##############################################################################
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testBasic(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[2, 2], drop_remainder=drop_remainder)

    expected_shapes = [[2]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    expected_output = [[0, 1], [2, 3], [4, 5], [6, 7]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testPartialBatch(self):
    dataset = dataset_ops.Dataset.range(5).batch(4, drop_remainder=False)
    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[2, 2], drop_remainder=False)

    expected_shapes = [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))
    expected_output = [[0, 1], [2, 3], [4]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testPartialBatchWithDropRemainder(self):
    dataset = dataset_ops.Dataset.range(5).batch(4, drop_remainder=False)
    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[2, 2], drop_remainder=True)

    expected_shapes = [[2]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))
    expected_output = [[0, 1], [2, 3]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testBatchSizeGreaterThanOriginal(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(12).batch(
        4, drop_remainder=False)
    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[6], drop_remainder=drop_remainder)

    expected_output = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testEmptySplits(self, drop_remainder):
    # It's possible for splits to be empty if the batch size is smaller than
    # the number of replicas. Here, we use an example with batch_size == 4
    # and num_replicas == 5.
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[1, 1, 1, 1, 0], drop_remainder=drop_remainder)

    expected_shapes = [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    expected_output = [[0], [1], [2], [3], [], [4], [5], [6], [7], []]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testEmptyFirstSplits(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[0, 1], drop_remainder=drop_remainder)

    expected_shapes = [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    # We have an extra element at the end because if the desired batch size is
    # zero, then we never read any inputs from the input_dataset at all, so we
    # will keep producting empty outputs until we reach a non zero desired batch
    # size split.
    expected_output = [[], [0], [], [1], [], [2], [], [3],
                       [], [4], [], [5], [], [6], [], [7], []]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testEmptyLastSplits(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(8).batch(4, drop_remainder=True)
    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[1, 0], drop_remainder=drop_remainder)

    expected_shapes = [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    expected_output = [[0], [], [1], [], [2], [], [3], [],
                       [4], [], [5], [], [6], [], [7], []]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testScalarBatchSizeInput(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(8).batch(
        4, drop_remainder=True)
    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=2, drop_remainder=drop_remainder)

    expected_shapes = [[2]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    expected_output = [[0, 1], [2, 3], [4, 5], [6, 7]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testMultipleBatches(self):
    dataset = dataset_ops.Dataset.range(16).batch(
        2, drop_remainder=True).batch(
            4, drop_remainder=True)
    self.assertEqual([[4, 2]], _flat_shapes(dataset))

    rebatched_dataset = distribute._RebatchDataset(dataset, [2, 2])
    self.assertEqual([[2, 2]], _flat_shapes(rebatched_dataset))
    # Each element is a list of 2 elements where each element is a list of 2.
    expected_output = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]],
                       [[12, 13], [14, 15]]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testNestedDictionaryOutput(self):
    dataset = dataset_ops.Dataset.range(8).map(
        lambda x: {"a": x, "b": {"c": x + 1}}).batch(4, drop_remainder=True)
    rebatched_dataset = distribute._RebatchDataset(dataset, [2, 2])
    self.assertEqual([[2], [2]], _flat_shapes(rebatched_dataset))

    expected_output = [{"a": [0, 1], "b": {"c": [1, 2]}},
                       {"a": [2, 3], "b": {"c": [3, 4]}},
                       {"a": [4, 5], "b": {"c": [5, 6]}},
                       {"a": [6, 7], "b": {"c": [7, 8]}}]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testRaggedDataset(self, drop_remainder):
    # Set up a dataset that produces ragged tensors with a static batch size.
    dataset = dataset_ops.Dataset.from_tensor_slices(
        ragged_tensor.RaggedTensor.from_row_lengths(
            list(range(10)), [1, 2, 3, 4]))
    # The map changes the internal representation of the ragged tensor.
    # This test will fail if we don't normalize the tensor representation.
    dataset = dataset.batch(4, drop_remainder=True).map(lambda x: x)

    rebatched_dataset = distribute._RebatchDataset(
        dataset, batch_sizes=[2, 2])

    expected_output = [
        ragged_tensor.RaggedTensor.from_row_lengths(list(range(3)), [1, 2]),
        ragged_tensor.RaggedTensor.from_row_lengths(list(range(3, 10)),
                                                    [3, 4]),
    ]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testNoneDataset(self):
    # Some datasets, e.g. datasets with None tensors, have components without
    # output shapes. Test that this doesn't break rebatching shape inference
    # logic.
    dataset = dataset_ops.Dataset.range(4)
    dataset = dataset.map(lambda x: (x, None))
    dataset = dataset.batch(4, drop_remainder=True)
    _ = distribute._RebatchDataset(dataset, batch_sizes=[2, 2])


class LegacyRebatchDatasetTest(test_base.DatasetTestBase,
                               parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testBasic(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(8).batch(
        4, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._LegacyRebatchDataset(
        dataset, num_replicas=2)

    expected_shapes = [[2]] if drop_remainder else [[None]]
    self.assertEqual(expected_shapes, _flat_shapes(rebatched_dataset))

    expected_output = [[0, 1], [2, 3], [4, 5], [6, 7]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testCanHandleUnknownRank(self):
    dataset = dataset_ops.Dataset.from_tensors("xxx")
    # decode_image results in a tensor of completely unknown shape (i.e. unknown
    # rank)
    dataset = dataset.map(image_ops.decode_image)
    self.assertEqual([tensor_shape.TensorShape(None)],
                     nest.flatten(
                         dataset_ops.get_legacy_output_shapes(dataset)))
    rebatched_dataset = distribute._LegacyRebatchDataset(
        dataset, num_replicas=4)
    # Note that we are just testing the dataset shapes, not the actual output.
    self.assertEqual(
        [tensor_shape.TensorShape(None)],
        nest.flatten(dataset_ops.get_legacy_output_shapes(rebatched_dataset)))

  @combinations.generate(test_base.default_test_combinations())
  def testCanHandleUnknownDims(self):
    dataset = dataset_ops.Dataset.range(1000)
    dataset = dataset.batch(10, drop_remainder=False)
    dataset = dataset.batch(10, drop_remainder=False)
    self.assertEqual([[None, None]], _flat_shapes(dataset))
    rebatched_dataset = distribute._LegacyRebatchDataset(
        dataset, num_replicas=4)
    # Note that we are just testing the dataset shapes, not the actual output.
    self.assertEqual([[None, None]], _flat_shapes(rebatched_dataset))

  @combinations.generate(test_base.default_test_combinations())
  def testScalarInputError(self):
    dataset = dataset_ops.Dataset.range(1024)
    distribute._LegacyRebatchDataset(dataset.batch(4), num_replicas=4)
    with self.assertRaises(ValueError):
      distribute._LegacyRebatchDataset(dataset, num_replicas=4)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testBatchNotDivisibleByNumReplicas(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(8).batch(
        4, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._LegacyRebatchDataset(
        dataset, num_replicas=3)
    self.assertEqual([[None]], _flat_shapes(rebatched_dataset))
    # This rebatches into sub-batches of size 2, since ceil(4 / 3) = 2. However,
    # this means that only the first 2 replicas will get data.
    expected_output = [[0, 1], [2, 3], [], [4, 5], [6, 7], []]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testTupleOutput(self):
    dataset = dataset_ops.Dataset.range(1024).map(lambda x: (x, x)).batch(32)
    rebatched_dataset = distribute._LegacyRebatchDataset(
        dataset, num_replicas=4)
    expected_output = [([k for k in range(i, i + 8)],  # pylint: disable=g-complex-comprehension
                        [k for k in range(i, i + 8)])
                       for i in range(0, 1024, 8)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testNestedDictionaryOutput(self):
    dataset = dataset_ops.Dataset.range(8).map(
        lambda x: {"a": x, "b": {"c": x + 1}}).batch(4)
    rebatched_dataset = distribute._LegacyRebatchDataset(
        dataset, num_replicas=2)
    expected_output = [{"a": [0, 1], "b": {"c": [1, 2]}},
                       {"a": [2, 3], "b": {"c": [3, 4]}},
                       {"a": [4, 5], "b": {"c": [5, 6]}},
                       {"a": [6, 7], "b": {"c": [7, 8]}}]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testFinalPartialBatch(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(10).batch(
        4, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._LegacyRebatchDataset(
        dataset, num_replicas=2)
    self.assertEqual([[2] if drop_remainder else [None]],
                     _flat_shapes(rebatched_dataset))
    if drop_remainder:
      expected_output = [[0, 1], [2, 3], [4, 5], [6, 7]]
    else:
      expected_output = [[0, 1], [2, 3], [4, 5], [6, 7], [8], [9]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testFinalPartialBatchAfterRebatch(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(9).batch(
        4, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._LegacyRebatchDataset(
        dataset, num_replicas=2)
    self.assertEqual([[2] if drop_remainder else [None]],
                     _flat_shapes(rebatched_dataset))
    if drop_remainder:
      expected_output = [[0, 1], [2, 3], [4, 5], [6, 7]]
    else:
      expected_output = [[0, 1], [2, 3], [4, 5], [6, 7], [8], []]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testMultipleBatches(self):
    dataset = dataset_ops.Dataset.range(16).batch(2).batch(4)
    self.assertEqual([[None, None]], _flat_shapes(dataset))

    # Each element is a list of 4 elements where each element is a list of 2.
    expected_output = [[[0, 1], [2, 3], [4, 5], [6, 7]],
                       [[8, 9], [10, 11], [12, 13], [14, 15]]]
    self.assertDatasetProduces(dataset, expected_output)

    rebatched_dataset = distribute._LegacyRebatchDataset(dataset, 2)
    self.assertEqual([[None, None]], _flat_shapes(rebatched_dataset))
    # Each element is a list of 2 elements where each element is a list of 2.
    expected_output = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]],
                       [[12, 13], [14, 15]]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testRaggedTensorDataset(self):
    # Set up a dataset that produces ragged tensors with a static batch size.
    row_lengths = np.random.randint(8, size=128)
    values = np.random.normal(size=np.sum(row_lengths)).astype(np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices(
        ragged_tensor.RaggedTensor.from_row_lengths(values, row_lengths))
    dataset = dataset.batch(32, drop_remainder=True)

    # The map changes the internal representation of the ragged tensor.
    # This test will fail if we don't normalize the tensor representation.
    dataset = dataset.map(lambda x: x)

    dataset = distribute._LegacyRebatchDataset(dataset, num_replicas=8)
    # After rebatching, batch size is now 4.
    expected_output = []
    value_index = 0
    for batch_row_lengths in row_lengths.reshape((-1, 4)):
      num_values = np.sum(batch_row_lengths)
      expected_output.append(
          ragged_tensor.RaggedTensor.from_row_lengths(
              values[value_index:(value_index + num_values)],
              batch_row_lengths))
      value_index += num_values
    self.assertDatasetProduces(dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testNoneDataset(self):
    # Some datasets, e.g. datasets with None tensors, have components without
    # output shapes. Test that this doesn't break rebatching shape inference
    # logic.
    dataset = dataset_ops.Dataset.range(4)
    dataset = dataset.map(lambda x: (x, None))
    dataset = dataset.batch(4, drop_remainder=True)
    _ = distribute._LegacyRebatchDataset(dataset, num_replicas=2)


class ComputeBatchSizeTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testComputeBatchSizeKnown(self):
    # When drop_remainder=True, batch size can be inferred from the type spec.
    dataset = dataset_ops.Dataset.range(32).batch(4, drop_remainder=True)
    dataset = dataset_ops.Dataset.zip((dataset, dataset))
    batch_size = distribute.compute_batch_size(dataset)
    self.assertEqual(4, self.evaluate(batch_size))

  @combinations.generate(test_base.default_test_combinations())
  def testComputeBatchSizeKnownAndMismatched(self):
    # Return -1 when different components have different batch sizes.
    dataset = dataset_ops.Dataset.range(32)
    dataset = dataset_ops.Dataset.zip((dataset.batch(4, drop_remainder=True),
                                       dataset.batch(8, drop_remainder=True)))
    batch_size = distribute.compute_batch_size(dataset)
    self.assertEqual(-1, self.evaluate(batch_size))

  @combinations.generate(test_base.default_test_combinations())
  def testComputeBatchSizeUnknown(self):
    dataset = dataset_ops.Dataset.range(32).batch(4)
    batch_size = distribute.compute_batch_size(dataset)
    self.assertEqual(4, self.evaluate(batch_size))

  @combinations.generate(test_base.default_test_combinations())
  def testComputeBatchSizeWithPassthrough(self):
    dataset = dataset_ops.Dataset.range(32).batch(4)
    dataset = dataset.take(5)
    batch_size = distribute.compute_batch_size(dataset)
    self.assertEqual(4, self.evaluate(batch_size))

  @combinations.generate(test_base.default_test_combinations())
  def testComputeBatchSizeWithPassthroughInvalid(self):
    dataset = dataset_ops.Dataset.range(32).batch(4)
    dataset = dataset.map(lambda x: x + 1)
    batch_size = distribute.compute_batch_size(dataset)
    self.assertEqual(-1, self.evaluate(batch_size))

  @combinations.generate(test_base.default_test_combinations())
  def testComputeBatchSizeWithZip(self):
    dataset = dataset_ops.Dataset.range(32).batch(4)
    dataset = dataset_ops.Dataset.zip((dataset, dataset))
    batch_size = distribute.compute_batch_size(dataset)
    self.assertEqual(4, self.evaluate(batch_size))

  @combinations.generate(test_base.default_test_combinations())
  def testComputeBatchSizeWithZipMismatched(self):
    dataset = dataset_ops.Dataset.range(32)
    dataset = dataset_ops.Dataset.zip((dataset.batch(4), dataset.batch(8)))
    batch_size = distribute.compute_batch_size(dataset)
    self.assertEqual(-1, self.evaluate(batch_size))

  @combinations.generate(test_base.default_test_combinations())
  def testNoneDataset(self):
    # Some datasets, e.g. datasets with None tensors, have components without
    # output shapes. Test that this doesn't break computing batch size logic.
    dataset = dataset_ops.Dataset.range(4)
    dataset = dataset.map(lambda x: (x, None))
    dataset = dataset.batch(4, drop_remainder=True)
    batch_size = distribute.compute_batch_size(dataset)
    self.assertEqual(4, self.evaluate(batch_size))


class LegacyRebatchDatasetCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):

    def build_dataset(num_elements, batch_size):
      return distribute._LegacyRebatchDataset(
          dataset_ops.Dataset.range(num_elements).batch(
              4 * batch_size, drop_remainder=True),
          num_replicas=4)

    verify_fn(self, lambda: build_dataset(64, 8), num_outputs=8)


class RebatchDatasetCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                   parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def test(self, verify_fn):

    def build_dataset(num_elements, batch_size):
      return distribute._RebatchDataset(
          dataset_ops.Dataset.range(num_elements).batch(
              2 * batch_size, drop_remainder=True),
          batch_sizes=[batch_size, batch_size])

    verify_fn(self, lambda: build_dataset(64, 8), num_outputs=8)


if __name__ == "__main__":
  test.main()

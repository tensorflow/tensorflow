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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.experimental.ops import scan_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def _flat_shapes(dataset):
  return nest.flatten(dataset_ops.get_legacy_output_shapes(dataset))


@parameterized.named_parameters(("WithDropRemainder", True),
                                ("WithoutDropRemainder", False))
@test_util.run_all_in_graph_and_eager_modes
class RebatchDatasetTest(test_base.DatasetTestBase):

  def testBasic(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(1024).batch(
        32, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    self.assertEqual(
        [[32 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(dataset)])
    self.assertEqual(
        [[8 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])

    expected_output = [[k for k in range(i, i + 8)] for i in range(0, 1024, 8)]  # pylint: disable=g-complex-comprehension
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testScalarInputError(self, _):
    dataset = dataset_ops.Dataset.range(1024)
    with self.assertRaisesRegexp(ValueError, "at least one dimension"):
      distribute._RebatchDataset(dataset, num_workers=4)

  def testNotDivisible(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(1024).batch(
        32, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=5)
    expected_output = [[k for k in range(i, i + 7)] for i in range(0, 1022, 7)]  # pylint: disable=g-complex-comprehension
    if not drop_remainder:
      expected_output.append([1022, 1023])
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testTupleOutput(self, drop_remainder):
    dataset = (
        dataset_ops.Dataset.range(1024).map(lambda x: (x, x)).batch(
            32, drop_remainder=drop_remainder))
    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    expected_output = [([k for k in range(i, i + 8)],  # pylint: disable=g-complex-comprehension
                        [k for k in range(i, i + 8)])
                       for i in range(0, 1024, 8)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testNestedDictionaryOutput(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(1024).map(
        lambda x: {"a": x, "b": {"c": x}}).batch(
            32, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    expected_output = [{"a": [k for k in range(i, i + 8)],  # pylint: disable=g-complex-comprehension
                        "b": {"c": [k for k in range(i, i + 8)]}}
                       for i in range(0, 1024, 8)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testFinalPartialBatchOriginal(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(1032).batch(
        32, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    self.assertEqual(
        [[32 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(dataset)])
    self.assertEqual(
        [[8 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])

    expected_output = [[k for k in range(i, i + 8)] for i in range(0, 1032, 8)]  # pylint: disable=g-complex-comprehension
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testFinalPartialBatchAfterRebatch(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(34).batch(
        32, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    self.assertEqual(
        [[32 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(dataset)])
    self.assertEqual(
        [[8 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])

    expected_output = [[k for k in range(i, i + 8)] for i in range(0, 32, 8)]  # pylint: disable=g-complex-comprehension
    if not drop_remainder:
      expected_output += [[32, 33]]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testMultipleBatches(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(128).batch(
        4, drop_remainder=drop_remainder)
    dataset = dataset.batch(8, drop_remainder=drop_remainder)
    self.assertEqual(
        [[8, 4]] if drop_remainder else [[None, None]],
        [ts.as_list() for ts in _flat_shapes(dataset)])
    # Each element is a list of 8 elements where each element is a list of 4.
    expected_output = [[[j, j + 1, j + 2, j + 3]  # pylint: disable=g-complex-comprehension
                        for j in range(i, i + 32, 4)]  # generates 8 elements
                       for i in range(0, 128, 32)]
    self.assertDatasetProduces(dataset, expected_output)

    rebatched_dataset = distribute._RebatchDataset(dataset, 4)
    self.assertEqual(
        [[2, 4]] if drop_remainder else [[None, None]],
        [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    # Each element is a list of 2 elements where each element is a list of 4.
    expected_output = [[[j, j + 1, j + 2, j + 3]  # pylint: disable=g-complex-comprehension
                        for j in range(i, i + 8, 4)]  # generates 2 elements
                       for i in range(0, 128, 8)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testMapAndBatch(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(1024).apply(
        batching.map_and_batch(
            math_ops.square, 32, drop_remainder=drop_remainder))
    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    self.assertEqual(
        [[32 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(dataset)])
    self.assertEqual(
        [[8 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = [[k**2 for k in range(i, i + 8)]  # pylint: disable=g-complex-comprehension
                       for i in range(0, 1024, 8)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testMapAndBatchWithCapturedInput(self, drop_remainder):
    captured_t = variables.Variable(42)
    dataset = dataset_ops.Dataset.range(1024).apply(
        batching.map_and_batch(
            lambda x: captured_t, 32, drop_remainder=drop_remainder))
    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    self.assertEqual([[32 if drop_remainder else None]],
                     [ts.as_list() for ts in _flat_shapes(dataset)])
    self.assertEqual([[8 if drop_remainder else None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = [[42 for _ in range(i, i + 8)]  # pylint: disable=g-complex-comprehension
                       for i in range(0, 1024, 8)]
    self.evaluate(variables.global_variables_initializer())
    self.assertDatasetProduces(
        rebatched_dataset, expected_output, requires_initialization=True)

  def testPaddedBatch(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(128).batch(4).padded_batch(
        8, padded_shapes=[5], drop_remainder=drop_remainder)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    self.assertEqual(
        [[8, 5]] if drop_remainder else [[None, 5]],
        [ts.as_list() for ts in _flat_shapes(dataset)])
    # Each element is a list of 8 elements in which each element is a list of 5
    # elements, first four are numbers and the last one is a padded zero.
    expected_output = [[[j, j + 1, j + 2, j + 3, 0]  # pylint: disable=g-complex-comprehension
                        for j in range(i, i + 32, 4)]  # generates 8 elements
                       for i in range(0, 128, 32)]
    self.assertDatasetProduces(dataset, expected_output)
    self.assertEqual(
        [[2, 5]] if drop_remainder else [[None, 5]],
        [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    # Each element is a list of 2 elements in which each element is a list of 5
    # elements, first four are numbers and the last one is a padded zero.
    expected_output = [[[j, j + 1, j + 2, j + 3, 0]  # pylint: disable=g-complex-comprehension
                        for j in range(i, i + 8, 4)]  # generates 2 elements
                       for i in range(0, 128, 8)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testConcatenate(self, drop_remainder):
    dataset1 = dataset_ops.Dataset.range(64).batch(
        8, drop_remainder=drop_remainder)
    dataset2 = dataset_ops.Dataset.range(32).batch(
        8, drop_remainder=drop_remainder)
    dataset = dataset1.concatenate(dataset2)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    self.assertEqual(
        [[8 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(dataset)])
    self.assertEqual(
        [[2 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = ([[i, i + 1] for i in range(0, 64, 2)] +
                       [[i, i + 1] for i in range(0, 32, 2)])
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testConcatenateDifferentShapes(self, drop_remainder):
    dataset1 = dataset_ops.Dataset.range(64).batch(
        16, drop_remainder=drop_remainder)
    dataset2 = dataset_ops.Dataset.range(32).batch(
        8, drop_remainder=drop_remainder)
    dataset = dataset1.concatenate(dataset2)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    self.assertEqual(
        [[None]], [ts.as_list() for ts in _flat_shapes(dataset)])
    self.assertEqual(
        [[None]],
        [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = ([[i, i + 1, i + 2, i + 3] for i in range(0, 64, 4)] +
                       [[i, i + 1] for i in range(0, 32, 2)])
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testZip(self, drop_remainder):
    dataset1 = dataset_ops.Dataset.range(64).batch(
        8, drop_remainder=drop_remainder)
    dataset2 = dataset_ops.Dataset.range(32).batch(
        8, drop_remainder=drop_remainder)
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    self.assertEqual(
        [[8], [8]] if drop_remainder else [[None], [None]],
        [ts.as_list() for ts in _flat_shapes(dataset)])
    self.assertEqual(
        [[2], [2]] if drop_remainder else [[None], [None]],
        [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = [([i, i + 1], [i, i + 1]) for i in range(0, 32, 2)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testZipDifferentShapes(self, drop_remainder):
    dataset1 = dataset_ops.Dataset.range(64).batch(
        16, drop_remainder=drop_remainder)
    dataset2 = dataset_ops.Dataset.range(32).batch(
        8, drop_remainder=drop_remainder)
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    self.assertEqual(
        [[16], [8]] if drop_remainder else [[None], [None]],
        [ts.as_list() for ts in _flat_shapes(dataset)])
    self.assertEqual(
        [[4], [2]] if drop_remainder else [[None], [None]],
        [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = [([2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3], [i, i + 1])
                       for i in range(0, 32, 2)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testUnsupportedTransformError(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(1024).batch(
        32, drop_remainder=drop_remainder).apply(
            scan_ops.scan([0], lambda _, a: ([0], a)))
    with self.assertRaises(errors.InvalidArgumentError):
      rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
      next_element = self.getNext(rebatched_dataset)
      self.evaluate(next_element())

  def testFlatMapBatching(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(
        2).flat_map(lambda _: dataset_ops.Dataset.range(32).batch(  # pylint: disable=g-long-lambda
            32, drop_remainder=drop_remainder))
    self.assertEqual(
        [[32 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(dataset)])
    # Two elements where each element is range(32)
    expected_output = [[k for k in range(32)] for _ in range(2)]  # pylint: disable=g-complex-comprehension
    self.assertDatasetProduces(dataset, expected_output)

    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    self.assertEqual(
        [[8 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    # Two elements where each element is a list of 4 elements where each element
    # is a list of 8.
    expected_output = [[k for k in range(i, i + 8)]  # pylint: disable=g-complex-comprehension
                       for _ in range(2)
                       for i in range(0, 32, 8)]  # generates 4 elements
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testInterleaveBatching(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(
        2).interleave(lambda _: dataset_ops.Dataset.range(32).batch(  # pylint: disable=g-long-lambda
            32, drop_remainder=drop_remainder), cycle_length=2)
    self.assertEqual(
        [[32 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(dataset)])
    # Two elements where each element is range(32)
    expected_output = [[k for k in range(32)] for _ in range(2)]  # pylint: disable=g-complex-comprehension
    self.assertDatasetProduces(dataset, expected_output)

    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    self.assertEqual(
        [[8 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    # List of 4 elements where each element is a list of 8 numbering from 0 to
    # 31 repeated twice.
    expected_output = [[k for k in range(i, i + 8)]  # pylint: disable=g-complex-comprehension
                       for i in range(0, 32, 8)  # generates 4 elements
                       for _ in range(2)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testParallelInterleaveBatching(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(
        2).interleave(lambda _: dataset_ops.Dataset.range(32).batch(  # pylint: disable=g-long-lambda
            32, drop_remainder=drop_remainder), cycle_length=2,
                      num_parallel_calls=2)
    self.assertEqual(
        [[32 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(dataset)])
    # Two elements where each element is range(32)
    expected_output = [[k for k in range(32)] for _ in range(2)]  # pylint: disable=g-complex-comprehension
    self.assertDatasetProduces(dataset, expected_output)

    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=4)
    self.assertEqual(
        [[8 if drop_remainder else None]],
        [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    # List of 4 elements where each element is a list of 8 numbering from 0 to
    # 31 repeated twice in collated fashion i.e [0...8], [0...8] etc.
    expected_output = [[k for k in range(i, i + 8)]  # pylint: disable=g-complex-comprehension
                       for i in range(0, 32, 8)  # generates 4 elements
                       for _ in range(2)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testGroupByWindowStaticBatch(self, drop_remainder):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [[array_ops.constant(i, dtype=dtypes.int64)] * 3 for i in range(40)])
    reduce_fn = lambda bucket_id, ds: ds.batch(  # pylint: disable=g-long-lambda
        batch_size=10, drop_remainder=drop_remainder)
    dataset = dataset.apply(
        grouping.group_by_window(
            key_func=lambda x: x[0] % 4, reduce_func=reduce_fn, window_size=10))
    rebatched_dataset = distribute._RebatchDataset(dataset, num_workers=2)

    self.assertEqual([[5, 3] if drop_remainder else [None, 3]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    # pylint: disable=g-complex-comprehension
    expected_output = [[[j + i * 4 + k * 20] * 3
                        for i in range(5)]
                       for j in range(4)
                       for k in range(2)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testGroupByWindowDynamicBatch(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(40).map(lambda x: x % 2)
    reduce_fn = lambda bucket_id, ds: ds.batch(  # pylint: disable=g-long-lambda
        batch_size=(bucket_id + 1) * 5, drop_remainder=drop_remainder)
    dataset = dataset.apply(
        grouping.group_by_window(
            key_func=lambda x: x, reduce_func=reduce_fn, window_size=10))
    dataset = distribute._RebatchDataset(dataset, num_workers=2)

    self.assertEqual([[None]],
                     [ts.as_list() for ts in _flat_shapes(dataset)])
    pairs = [(3, 0), (3, 0), (3, 0)]
    if not drop_remainder:
      pairs.extend([(1, 0)])
    pairs.extend([(5, 1), (5, 1)])
    pairs = pairs * 2
    expected_output = [[value] * batch_size for batch_size, value in pairs]
    self.assertDatasetProduces(dataset, expected_output)


if __name__ == "__main__":
  test.main()

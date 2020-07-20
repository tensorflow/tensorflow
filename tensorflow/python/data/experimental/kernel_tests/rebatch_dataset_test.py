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
import numpy as np

from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import image_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


def _flat_shapes(dataset):
  return nest.flatten(dataset_ops.get_legacy_output_shapes(dataset))


class RebatchDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testBasic(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(1024).batch(
        32, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[8] if drop_remainder else [None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])

    expected_output = [[k for k in range(i, i + 8)] for i in range(0, 1024, 8)]  # pylint: disable=g-complex-comprehension
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testCanHandleUnknownRank(self):
    dataset = dataset_ops.Dataset.from_tensors("xxx")
    # decode_image results in a tensor of completely unknown shape (i.e. unknown
    # rank)
    dataset = dataset.map(image_ops.decode_image)
    self.assertEqual([tensor_shape.TensorShape(None)], _flat_shapes(dataset))
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    # Note that we are just testing the dataset shapes, not the actual output.
    self.assertEqual([tensor_shape.TensorShape(None)],
                     _flat_shapes(rebatched_dataset))

  @combinations.generate(test_base.default_test_combinations())
  def testCanHandleUnknownDims(self):
    dataset = dataset_ops.Dataset.range(1000)
    dataset = dataset.batch(10, drop_remainder=False)
    dataset = dataset.batch(10, drop_remainder=False)
    self.assertEqual([[None, None]],
                     [ts.as_list() for ts in _flat_shapes(dataset)])
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    # Note that we are just testing the dataset shapes, not the actual output.
    self.assertEqual([[None, None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])

  @combinations.generate(test_base.default_test_combinations())
  def testScalarInputError(self):
    dataset = dataset_ops.Dataset.range(1024)
    distribute._RebatchDataset(dataset.batch(4), num_replicas=4)
    with self.assertRaisesRegex(ValueError, ("You can fix the issue "
                                             "by adding the `batch`")):
      distribute._RebatchDataset(dataset, num_replicas=4)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testBatchNotDivisibleByNumReplicas(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(1024).batch(
        32, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=5)
    self.assertEqual([[None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = []
    i = 0
    for _ in range(32):  # number of steps
      # first four minibatches have seven elements
      for _ in range(4):
        expected_output.append([k for k in range(i, i + 7)])
        i += 7
      # last minibatch has four elements
      expected_output.append([k for k in range(i, i + 4)])
      i += 4
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testBatchSizeNotDivisibleByNumReplicas2(self):
    dataset = dataset_ops.Dataset.range(32).batch(16, drop_remainder=True)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=5)
    # This will rebatch into sub-batches of size 4, since
    # ceil(16 / 5) = 4. However, that means only the first 4 replicas will get
    # data.
    expected_output = [[k for k in range(i, i + 4)] for i in range(0, 16, 4)]
    expected_output.extend([[]])  # Last replica gets an empty batch
    expected_output.extend(
        [[k for k in range(i, i + 4)] for i in range(16, 32, 4)])
    expected_output.extend([[]])  # Last replica gets an empty batch
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testTupleOutput(self):
    dataset = dataset_ops.Dataset.range(1024).map(lambda x: (x, x)).batch(32)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    expected_output = [([k for k in range(i, i + 8)],  # pylint: disable=g-complex-comprehension
                        [k for k in range(i, i + 8)])
                       for i in range(0, 1024, 8)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testNestedDictionaryOutput(self):
    dataset = dataset_ops.Dataset.range(1024).map(
        lambda x: {"a": x, "b": {"c": x}}).batch(32)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    expected_output = [{"a": [k for k in range(i, i + 8)],  # pylint: disable=g-complex-comprehension
                        "b": {"c": [k for k in range(i, i + 8)]}}
                       for i in range(0, 1024, 8)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testFinalPartialBatch(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(1032).batch(
        32, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[8] if drop_remainder else [None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])

    # if drop_remainder, the final partial batch is dropped, even though it
    # makes up a complete minibatch.
    expected_output = [[k for k in range(i, i + 8)] for i in range(0, 1024, 8)]  # pylint: disable=g-complex-comprehension
    if not drop_remainder:
      # The last partial batch of size 8 is split over 4 replicas
      expected_output.extend(
          [[k for k in range(i, i + 2)] for i in range(1024, 1032, 2)])
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testFinalPartialBatchAfterRebatch(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(34).batch(
        32, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[8] if drop_remainder else [None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])

    expected_output = [[k for k in range(i, i + 8)] for i in range(0, 32, 8)]  # pylint: disable=g-complex-comprehension
    if not drop_remainder:
      # The last partial batch of size 2 is split over 4 replicas
      expected_output += [[32], [33], [], []]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testMultipleBatches(self):
    dataset = dataset_ops.Dataset.range(128).batch(4).batch(8)
    self.assertEqual([[None, None]],
                     [ts.as_list() for ts in _flat_shapes(dataset)])

    # Each element is a list of 8 elements where each element is a list of 4.
    expected_output = [[[j, j + 1, j + 2, j + 3]  # pylint: disable=g-complex-comprehension
                        for j in range(i, i + 32, 4)]  # generates 8 elements
                       for i in range(0, 128, 32)]
    self.assertDatasetProduces(dataset, expected_output)

    rebatched_dataset = distribute._RebatchDataset(dataset, 4)
    self.assertEqual([[None, None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    # Each element is a list of 2 elements where each element is a list of 4.
    expected_output = [[[j, j + 1, j + 2, j + 3]  # pylint: disable=g-complex-comprehension
                        for j in range(i, i + 8, 4)]  # generates 2 elements
                       for i in range(0, 128, 8)]
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

    dataset = distribute._RebatchDataset(dataset, num_replicas=8)
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


if __name__ == "__main__":
  test.main()

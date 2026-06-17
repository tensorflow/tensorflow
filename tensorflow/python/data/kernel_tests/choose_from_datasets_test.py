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
"""Tests for `tf.data.Dataset.choose_from_dataset()`."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import test


class ChooseFromDatasetsTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testChooseFromDatasets(self):
    words = [b"foo", b"bar", b"baz"]
    datasets = [dataset_ops.Dataset.from_tensors(w).repeat() for w in words]
    choice_array = np.random.randint(3, size=(15,), dtype=np.int64)
    choice_dataset = dataset_ops.Dataset.from_tensor_slices(choice_array)
    dataset = dataset_ops.Dataset.choose_from_datasets(datasets, choice_dataset)
    next_element = self.getNext(dataset)
    for i in choice_array:
      self.assertEqual(words[i], self.evaluate(next_element()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testChooseFromDatasetsStoppingOnEmptyDataset(self):
    datasets = [
        dataset_ops.Dataset.from_tensors(b"foo").repeat(2),
        dataset_ops.Dataset.from_tensors(b"bar").repeat(),
        dataset_ops.Dataset.from_tensors(b"baz").repeat(),
    ]
    choice_array = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    choice_dataset = dataset_ops.Dataset.from_tensor_slices(choice_array)
    dataset = dataset_ops.Dataset.choose_from_datasets(
        datasets, choice_dataset, stop_on_empty_dataset=True)
    self.assertDatasetProduces(dataset, [b"foo", b"foo"])

  @combinations.generate(test_base.default_test_combinations())
  def testChooseFromDatasetsSkippingEmptyDatasets(self):
    datasets = [
        dataset_ops.Dataset.from_tensors(b"foo").repeat(2),
        dataset_ops.Dataset.from_tensors(b"bar").repeat(),
        dataset_ops.Dataset.from_tensors(b"baz").repeat(),
    ]
    choice_array = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    choice_dataset = dataset_ops.Dataset.from_tensor_slices(choice_array)
    dataset = dataset_ops.Dataset.choose_from_datasets(
        datasets, choice_dataset, stop_on_empty_dataset=False)
    # Chooses 2 elements from the first dataset while the selector specifies 3.
    self.assertDatasetProduces(
        dataset,
        [b"foo", b"foo", b"bar", b"bar", b"bar", b"baz", b"baz", b"baz"])

  @combinations.generate(test_base.default_test_combinations())
  def testChooseFromDatasetsChoiceDatasetIsEmpty(self):
    datasets = [
        dataset_ops.Dataset.from_tensors(b"foo").repeat(),
        dataset_ops.Dataset.from_tensors(b"bar").repeat(),
        dataset_ops.Dataset.from_tensors(b"baz").repeat(),
    ]
    dataset = dataset_ops.Dataset.choose_from_datasets(
        datasets,
        choice_dataset=dataset_ops.Dataset.range(0),
        stop_on_empty_dataset=False)
    self.assertDatasetProduces(dataset, [])

  @combinations.generate(test_base.default_test_combinations())
  def testChooseFromDatasetsNested(self):
    ds1 = dataset_ops.Dataset.range(10).window(2)
    ds2 = dataset_ops.Dataset.range(10, 20).window(2)
    choice_dataset = dataset_ops.Dataset.range(2).repeat(5)
    ds = dataset_ops.Dataset.choose_from_datasets([ds1, ds2], choice_dataset)
    ds = ds.flat_map(lambda x: x)
    expected = []
    for i in range(5):
      for j in range(2):
        expected.extend([10*j + 2*i, 10*j + 2*i + 1])
    self.assertDatasetProduces(ds, expected)

  @combinations.generate(test_base.default_test_combinations())
  def testErrors(self):
    with self.assertRaisesRegex(TypeError, "tf.int64"):
      dataset_ops.Dataset.choose_from_datasets(
          [
              dataset_ops.Dataset.from_tensors(0),
              dataset_ops.Dataset.from_tensors(1)
          ],
          choice_dataset=dataset_ops.Dataset.from_tensors(1.0))

    with self.assertRaisesRegex(TypeError, "scalar"):
      dataset_ops.Dataset.choose_from_datasets(
          [
              dataset_ops.Dataset.from_tensors(0),
              dataset_ops.Dataset.from_tensors(1)
          ],
          choice_dataset=dataset_ops.Dataset.from_tensors([1.0]))

    with self.assertRaisesRegex(errors.InvalidArgumentError, "out of range"):
      dataset = dataset_ops.Dataset.choose_from_datasets(
          [dataset_ops.Dataset.from_tensors(0)],
          choice_dataset=dataset_ops.Dataset.from_tensors(
              constant_op.constant(1, dtype=dtypes.int64)))
      next_element = self.getNext(dataset)
      self.evaluate(next_element())

    with self.assertRaisesRegex(
        ValueError, r"Invalid `datasets`. `datasets` should not be empty."):
      dataset_ops.Dataset.choose_from_datasets(
          datasets=[], choice_dataset=dataset_ops.Dataset.from_tensors(1.0))

    with self.assertRaisesRegex(
        TypeError, r"`choice_dataset` should be a `tf.data.Dataset`"):
      datasets = [dataset_ops.Dataset.range(42)]
      dataset_ops.Dataset.choose_from_datasets(datasets, choice_dataset=None)


class ChooseFromDatasetsCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                       parameterized.TestCase):

  def _build_dataset(self,
                     num_datasets,
                     num_elements_per_dataset,
                     options=None):
    datasets = [
        dataset_ops.Dataset.range(num_elements_per_dataset)
        for _ in range(num_datasets)
    ]
    indices = []
    for i in range(num_datasets):
      indices = indices + ([i] * num_elements_per_dataset)
    shuffled_indices = stateless_random_ops.stateless_shuffle(
        np.int64(indices), seed=[1, 2])
    choice_dataset = dataset_ops.Dataset.from_tensor_slices(shuffled_indices)
    dataset = dataset_ops.Dataset.choose_from_datasets(datasets, choice_dataset)
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def test(self, verify_fn, symbolic_checkpoint):
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self, lambda: self._build_dataset(5, 20, options), num_outputs=100)

if __name__ == "__main__":
  test.main()

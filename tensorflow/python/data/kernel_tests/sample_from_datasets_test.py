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
"""Tests for `tf.data.Dataset.sample_from_dataset()`."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import compat as tf_compat
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test


def _weights_type_combinations():
  return combinations.combine(weights_type=["list", "tensor", "dataset"])


def _get_weights_of_type(weights_list, weights_type):
  if weights_type == "list":
    return weights_list
  if weights_type == "tensor":
    return ops.convert_to_tensor(weights_list, name="weights")
  return dataset_ops.Dataset.from_tensors(weights_list).repeat()


class SampleFromDatasetsTest(test_base.DatasetTestBase, parameterized.TestCase):

  def _normalize(self, vec):
    return vec / vec.sum()

  def _chi2(self, expected, actual):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    diff = actual - expected
    chi2 = np.sum(diff * diff / expected, axis=0)
    return chi2

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _weights_type_combinations()))
  def testSampleFromDatasets(self, weights_type):
    random_seed.set_random_seed(1619)
    num_samples = 5000
    rand_probs = self._normalize(np.random.random_sample((5,)))

    # Use chi-squared test to assert that the observed distribution matches the
    # expected distribution. Based on the implementation in
    # "third_party/tensorflow/python/kernel_tests/multinomial_op_test.py".
    for probs in [[.85, .05, .1], rand_probs, [1.]]:
      weights = _get_weights_of_type(np.asarray(probs), weights_type)
      classes = len(probs)

      # Create a dataset that samples each integer in `[0, num_datasets)`
      # with probability given by `weights[i]`.
      dataset = dataset_ops.Dataset.sample_from_datasets([
          dataset_ops.Dataset.from_tensors(i).repeat() for i in range(classes)
      ], weights)
      dataset = dataset.take(num_samples)

      next_element = self.getNext(dataset, requires_initialization=True)
      freqs = np.zeros([classes])
      for _ in range(num_samples):
        freqs[self.evaluate(next_element())] += 1
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element())

      self.assertLess(self._chi2(probs, freqs / num_samples), 1e-2)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _weights_type_combinations()))
  def testSampleFromDatasetsStoppingOnEmptyDataset(self, weights_type):
    # Sampling stops when the first dataset is exhausted.
    weights = _get_weights_of_type(np.asarray([.5, .1, .4]), weights_type)
    datasets = [
        dataset_ops.Dataset.from_tensors(np.int64(-1)),
        dataset_ops.Dataset.from_tensors(np.int64(1)).repeat(),
        dataset_ops.Dataset.range(10).repeat()
    ]
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=True)

    samples_list = self.getIteratorOutput(self.getNext(
        sample_dataset, requires_initialization=True))
    self.assertEqual(samples_list.count(-1), 1)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _weights_type_combinations()))
  def testSampleFromDatasetsSkippingEmptyDataset(self, weights_type):
    # Sampling skips the first dataset after it becomes empty.
    weights = _get_weights_of_type(np.asarray([.5, .1, .4]), weights_type)
    datasets = [
        dataset_ops.Dataset.from_tensors(np.int64(-1)),
        dataset_ops.Dataset.from_tensors(np.int64(1)).repeat(),
        dataset_ops.Dataset.range(10).repeat()
    ]
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=False).take(100)

    samples_list = self.getIteratorOutput(self.getNext(
        sample_dataset, requires_initialization=True))
    self.assertLen(samples_list, 100)
    self.assertEqual(samples_list.count(-1), 1)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _weights_type_combinations()))
  def testSampleFromDatasetsWithZeroWeight(self, weights_type):
    # Sampling stops when the second dataset is exhausted.
    weights = _get_weights_of_type(np.asarray([0., 1.]), weights_type)
    datasets = [
        dataset_ops.Dataset.from_tensors(-1).repeat(2),
        dataset_ops.Dataset.from_tensors(1).repeat(2)
    ]
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=True)
    self.assertDatasetProduces(sample_dataset, [1, 1],
                               requires_initialization=True)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _weights_type_combinations()))
  def testSampleFromEmptyDataset(self, weights_type):
    weights = _get_weights_of_type(np.asarray([1., 0.]), weights_type)
    datasets = [
        dataset_ops.Dataset.range(0),
        dataset_ops.Dataset.range(1).repeat()
    ]
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=True)
    self.assertDatasetProduces(sample_dataset, [],
                               requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testSampleFromDatasetsSkippingDatasetsWithZeroWeight(self):
    # Sampling skips the first dataset.
    weights = np.asarray([0., 1.])
    datasets = [
        dataset_ops.Dataset.from_tensors(-1).repeat(),
        dataset_ops.Dataset.from_tensors(1)
    ]
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=False)
    self.assertDatasetProduces(sample_dataset, [1])

  @combinations.generate(test_base.default_test_combinations())
  def testSampleFromDatasetsAllWeightsAreZero(self):
    # Sampling skips both datasets.
    weights = np.asarray([0., 0.])
    datasets = [
        dataset_ops.Dataset.from_tensors(-1).repeat(),
        dataset_ops.Dataset.from_tensors(1).repeat()
    ]
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=False)
    self.assertDatasetProduces(sample_dataset, [])

  @combinations.generate(test_base.default_test_combinations())
  def testSampleFromDatasetsCardinality(self):
    ds1 = dataset_ops.Dataset.from_tensors([1.0]).repeat()
    ds2 = dataset_ops.Dataset.from_tensors([2.0]).repeat()
    ds = dataset_ops.Dataset.sample_from_datasets([ds1, ds2])
    self.assertEqual(self.evaluate(ds.cardinality()), dataset_ops.INFINITE)

  @combinations.generate(test_base.default_test_combinations())
  def testSampleFromDatasetsNested(self):
    ds1 = dataset_ops.Dataset.range(10).window(2)
    ds2 = dataset_ops.Dataset.range(10, 20).window(2)
    ds = dataset_ops.Dataset.sample_from_datasets([ds1, ds2],
                                                  weights=[0.3, 0.7])
    ds = ds.flat_map(lambda x: x)
    next_element = self.getNext(ds, requires_initialization=True)
    self.evaluate(next_element())

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(rerandomize=[None, True, False])))
  def testSampleFromDatasetsRerandomizeEachIterationEpochs(self, rerandomize):
    if rerandomize is not None and not tf_compat.forward_compatible(
        2022, 12, 17):
      self.skipTest(
          "target functionality not available due to forward compatibility")
    dataset1 = dataset_ops.Dataset.range(0, 10)
    dataset2 = dataset_ops.Dataset.range(100, 110)
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        [dataset1, dataset2],
        seed=42,
        weights=[0.5, 0.5],
        stop_on_empty_dataset=True,
        rerandomize_each_iteration=rerandomize)
    first_epoch = self.getDatasetOutput(sample_dataset)
    second_epoch = self.getDatasetOutput(sample_dataset)

    if rerandomize:
      self.assertNotEqual(first_epoch, second_epoch)
    else:
      self.assertEqual(first_epoch, second_epoch)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(rerandomize=[None, True, False])))
  def testSampleFromDatasetsRerandomizeRepeatEpochs(self, rerandomize):
    if rerandomize is not None and not tf_compat.forward_compatible(
        2022, 12, 17):
      self.skipTest(
          "target functionality not available due to forward compatibility")
    dataset1 = dataset_ops.Dataset.range(0, 10)
    dataset2 = dataset_ops.Dataset.range(100, 110)
    sample_dataset = dataset_ops.Dataset.sample_from_datasets(
        [dataset1, dataset2],
        seed=42,
        weights=[0.5, 0.5],
        stop_on_empty_dataset=True,
        rerandomize_each_iteration=rerandomize)
    sample_dataset = sample_dataset.repeat(2)
    epochs = self.getDatasetOutput(sample_dataset, requires_initialization=True)
    first_epoch = epochs[:len(epochs) // 2]
    second_epoch = epochs[len(epochs) // 2:]

    if rerandomize:
      self.assertNotEqual(first_epoch, second_epoch)
    else:
      self.assertEqual(first_epoch, second_epoch)

  @combinations.generate(
      combinations.times(test_base.v2_eager_only_combinations(),
                         combinations.combine(rerandomize=[None, True, False])))
  def testSampleFromDatasetsRerandomizeInsideFunction(self, rerandomize):
    if rerandomize is not None and not tf_compat.forward_compatible(
        2022, 12, 17):
      self.skipTest(
          "target functionality not available due to forward compatibility")
    @def_function.function
    def make_dataset():
      dataset1 = dataset_ops.Dataset.range(0, 10)
      dataset2 = dataset_ops.Dataset.range(100, 110)
      sample_dataset = dataset_ops.Dataset.sample_from_datasets(
          [dataset1, dataset2],
          seed=42,
          weights=[0.5, 0.5],
          stop_on_empty_dataset=True,
          rerandomize_each_iteration=rerandomize)
      return sample_dataset

    sample_dataset = make_dataset()
    first_epoch = self.getDatasetOutput(sample_dataset)
    second_epoch = self.getDatasetOutput(sample_dataset)

    if rerandomize:
      self.assertNotEqual(first_epoch, second_epoch)
    else:
      self.assertEqual(first_epoch, second_epoch)

  @combinations.generate(test_base.default_test_combinations())
  def testErrors(self):
    with self.assertRaisesRegex(ValueError, r"should have the same length"):
      dataset_ops.Dataset.sample_from_datasets(
          [dataset_ops.Dataset.range(10),
           dataset_ops.Dataset.range(20)],
          weights=[0.25, 0.25, 0.25, 0.25])

    with self.assertRaisesRegex(TypeError, "`tf.float32` or `tf.float64`"):
      dataset_ops.Dataset.sample_from_datasets(
          [dataset_ops.Dataset.range(10),
           dataset_ops.Dataset.range(20)],
          weights=[1, 1])

    with self.assertRaisesRegex(TypeError, "must have compatible"):
      dataset_ops.Dataset.sample_from_datasets([
          dataset_ops.Dataset.from_tensors(0),
          dataset_ops.Dataset.from_tensors(0.0)
      ])

    with self.assertRaisesRegex(
        ValueError, r"Invalid `datasets`. `datasets` should not be empty."):
      dataset_ops.Dataset.sample_from_datasets(datasets=[], weights=[])


class SampleFromDatasetsCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                       parameterized.TestCase):

  def _build_dataset(self, probs, num_samples, options=None):
    datasets = [
        dataset_ops.Dataset.from_tensors(i).repeat(None)
        for i in range(len(probs))
    ]
    dataset = dataset_ops.Dataset.sample_from_datasets(
        datasets, probs, seed=1813)
    dataset = dataset.take(num_samples)
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
        self,
        lambda: self._build_dataset([0.5, 0.5], 100, options),
        num_outputs=100)


if __name__ == "__main__":
  test.main()

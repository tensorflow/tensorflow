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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test


class DirectedInterleaveDatasetTest(test_base.DatasetTestBase,
                                    parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testBasic(self):
    selector_dataset = dataset_ops.Dataset.range(10).repeat(100)
    input_datasets = [
        dataset_ops.Dataset.from_tensors(i).repeat(100) for i in range(10)
    ]
    dataset = interleave_ops._DirectedInterleaveDataset(selector_dataset,
                                                        input_datasets)
    next_element = self.getNext(dataset)

    for _ in range(100):
      for i in range(10):
        self.assertEqual(i, self.evaluate(next_element()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

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
                         combinations.combine(weights_as_dataset=[False, True]))
  )
  def testSampleFromDatasets(self, weights_as_dataset):
    random_seed.set_random_seed(1619)
    num_samples = 5000
    rand_probs = self._normalize(np.random.random_sample((5,)))

    # Use chi-squared test to assert that the observed distribution matches the
    # expected distribution. Based on the implementation in
    # "third_party/tensorflow/python/kernel_tests/multinomial_op_test.py".
    for probs in [[.85, .05, .1], rand_probs, [1.]]:
      weights = np.asarray(probs)
      if weights_as_dataset:
        weights = dataset_ops.Dataset.from_tensors(weights).repeat()
      classes = len(probs)

      # Create a dataset that samples each integer in `[0, num_datasets)`
      # with probability given by `weights[i]`.
      dataset = interleave_ops.sample_from_datasets([
          dataset_ops.Dataset.from_tensors(i).repeat()
          for i in range(classes)
      ], weights)
      dataset = dataset.take(num_samples)

      next_element = self.getNext(dataset)
      freqs = np.zeros([classes])
      for _ in range(num_samples):
        freqs[self.evaluate(next_element())] += 1
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element())

      self.assertLess(self._chi2(probs, freqs / num_samples), 1e-2)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(weights_as_dataset=[False, True]))
  )
  def testSampleFromDatasetsStoppingOnEmptyDataset(self, weights_as_dataset):
    weights = np.asarray([.5, .1, .4])
    if weights_as_dataset:
      weights = dataset_ops.Dataset.from_tensors(weights).repeat()

    # Sampling stops when the first dataset is exhausted.
    datasets = [
        dataset_ops.Dataset.from_tensors(np.int64(-1)),
        dataset_ops.Dataset.from_tensors(np.int64(1)).repeat(),
        dataset_ops.Dataset.range(10).repeat()
    ]
    sample_dataset = interleave_ops.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=True)

    samples_list = self.getIteratorOutput(self.getNext(sample_dataset))
    self.assertEqual(samples_list.count(-1), 1)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(weights_as_dataset=[False, True]))
  )
  def testSampleFromDatasetsSkippingEmptyDataset(self, weights_as_dataset):
    weights = np.asarray([.5, .1, .4])
    if weights_as_dataset:
      weights = dataset_ops.Dataset.from_tensors(weights).repeat()

    # Sampling skips the first dataset after it becomes empty.
    datasets = [
        dataset_ops.Dataset.from_tensors(np.int64(-1)),
        dataset_ops.Dataset.from_tensors(np.int64(1)).repeat(),
        dataset_ops.Dataset.range(10).repeat()
    ]
    sample_dataset = interleave_ops.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=False).take(100)

    samples_list = self.getIteratorOutput(self.getNext(sample_dataset))
    self.assertLen(samples_list, 100)
    self.assertEqual(samples_list.count(-1), 1)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(weights_as_dataset=[False, True]))
  )
  def testSampleFromDatasetsWithZeroWeight(self, weights_as_dataset):
    weights = np.asarray([0., 1.])
    if weights_as_dataset:
      weights = dataset_ops.Dataset.from_tensors(weights).repeat()

    # Sampling stops when the second dataset is exhausted.
    datasets = [
        dataset_ops.Dataset.from_tensors(-1).repeat(2),
        dataset_ops.Dataset.from_tensors(1).repeat(2)
    ]
    sample_dataset = interleave_ops.sample_from_datasets(
        datasets, weights=weights, stop_on_empty_dataset=True)

    samples_list = self.getIteratorOutput(self.getNext(sample_dataset))
    self.assertEqual(samples_list, [1, 1])

  @combinations.generate(test_base.default_test_combinations())
  def testSampleFromDatasetsCardinality(self):
    ds1 = dataset_ops.Dataset.from_tensors([1.0]).repeat()
    ds2 = dataset_ops.Dataset.from_tensors([2.0]).repeat()
    ds = interleave_ops.sample_from_datasets([ds1, ds2])
    self.assertEqual(self.evaluate(ds.cardinality()), dataset_ops.INFINITE)

  @combinations.generate(test_base.default_test_combinations())
  def testChooseFromDatasets(self):
    words = [b"foo", b"bar", b"baz"]
    datasets = [dataset_ops.Dataset.from_tensors(w).repeat() for w in words]
    choice_array = np.random.randint(3, size=(15,), dtype=np.int64)
    choice_dataset = dataset_ops.Dataset.from_tensor_slices(choice_array)
    dataset = interleave_ops.choose_from_datasets(datasets, choice_dataset)
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
    dataset = interleave_ops.choose_from_datasets(
        datasets, choice_dataset, stop_on_empty_dataset=True)
    data_list = self.getIteratorOutput(self.getNext(dataset))
    self.assertEqual(data_list, [b"foo", b"foo"])

  @combinations.generate(test_base.default_test_combinations())
  def testChooseFromDatasetsSkippingEmptyDatasets(self):
    datasets = [
        dataset_ops.Dataset.from_tensors(b"foo").repeat(2),
        dataset_ops.Dataset.from_tensors(b"bar").repeat(),
        dataset_ops.Dataset.from_tensors(b"baz").repeat(),
    ]
    choice_array = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    choice_dataset = dataset_ops.Dataset.from_tensor_slices(choice_array)
    dataset = interleave_ops.choose_from_datasets(
        datasets, choice_dataset, stop_on_empty_dataset=False)
    data_list = self.getIteratorOutput(self.getNext(dataset))
    # Chooses 2 elements from the first dataset while the selector specifies 3.
    self.assertEqual(
        data_list,
        [b"foo", b"foo", b"bar", b"bar", b"bar", b"baz", b"baz", b"baz"])

  @combinations.generate(test_base.default_test_combinations())
  def testErrors(self):
    with self.assertRaisesRegex(ValueError,
                                r"vector of length `len\(datasets\)`"):
      interleave_ops.sample_from_datasets(
          [dataset_ops.Dataset.range(10),
           dataset_ops.Dataset.range(20)],
          weights=[0.25, 0.25, 0.25, 0.25])

    with self.assertRaisesRegex(TypeError, "`tf.float32` or `tf.float64`"):
      interleave_ops.sample_from_datasets(
          [dataset_ops.Dataset.range(10),
           dataset_ops.Dataset.range(20)],
          weights=[1, 1])

    with self.assertRaisesRegex(TypeError, "must have the same type"):
      interleave_ops.sample_from_datasets([
          dataset_ops.Dataset.from_tensors(0),
          dataset_ops.Dataset.from_tensors(0.0)
      ])

    with self.assertRaisesRegex(TypeError, "tf.int64"):
      interleave_ops.choose_from_datasets([
          dataset_ops.Dataset.from_tensors(0),
          dataset_ops.Dataset.from_tensors(1)
      ], choice_dataset=dataset_ops.Dataset.from_tensors(1.0))

    with self.assertRaisesRegex(TypeError, "scalar"):
      interleave_ops.choose_from_datasets([
          dataset_ops.Dataset.from_tensors(0),
          dataset_ops.Dataset.from_tensors(1)
      ], choice_dataset=dataset_ops.Dataset.from_tensors([1.0]))

    with self.assertRaisesRegex(errors.InvalidArgumentError, "out of range"):
      dataset = interleave_ops.choose_from_datasets(
          [dataset_ops.Dataset.from_tensors(0)],
          choice_dataset=dataset_ops.Dataset.from_tensors(
              constant_op.constant(1, dtype=dtypes.int64)))
      next_element = self.getNext(dataset)
      self.evaluate(next_element())


class SampleFromDatasetsCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                       parameterized.TestCase):

  def _build_dataset(self, probs, num_samples):
    dataset = interleave_ops.sample_from_datasets([
        dataset_ops.Dataset.from_tensors(i).repeat(None)
        for i in range(len(probs))
    ],
                                                  probs,
                                                  seed=1813)
    return dataset.take(num_samples)

  @combinations.generate(test_base.default_test_combinations())
  def testCheckpointCore(self):
    self.run_core_tests(lambda: self._build_dataset([0.5, 0.5], 100), 100)


if __name__ == "__main__":
  test.main()

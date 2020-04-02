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
"""Tests for `tf.data.Dataset.cache()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from os import path
import shutil
import tempfile

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class FileCacheTest(test_base.DatasetTestBase, parameterized.TestCase):

  def setUp(self):
    super(FileCacheTest, self).setUp()
    self.tmp_dir = tempfile.mkdtemp()
    self.cache_prefix = path.join(self.tmp_dir, "cache")

  def tearDown(self):
    if self.tmp_dir:
      shutil.rmtree(self.tmp_dir, ignore_errors=True)
    super(FileCacheTest, self).tearDown()

  @combinations.generate(test_base.default_test_combinations())
  def testCacheDatasetPassthrough(self):
    components = (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
                  np.array([9.0, 10.0, 11.0, 12.0]))

    def dataset_fn(count=5, filename=None):
      repeat_dataset = (
          dataset_ops.Dataset.from_tensor_slices(components).repeat(count))
      if filename:
        return repeat_dataset.cache(filename)
      else:
        return repeat_dataset

    self.assertEqual(
        tuple([c.shape[1:] for c in components]),
        dataset_ops.get_legacy_output_shapes(dataset_fn()))

    get_next = self.getNext(dataset_fn())

    # First run without caching to collect the "ground truth".
    elements = []
    for _ in range(20):
      elements.append(self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

    # Assert that the cached dataset has the same elements as the
    # "ground truth".
    get_next = self.getNext(dataset_fn(filename=self.cache_prefix))
    cached_elements = []
    for _ in range(20):
      cached_elements.append(self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    self.assertAllEqual(elements, cached_elements)

    # Re-initialize with an empty upstream (to throw errors.OutOfRangeError
    # if we didn't use the cache).
    get_next = self.getNext(dataset_fn(count=0, filename=self.cache_prefix))
    replayed_elements = []
    for _ in range(20):
      replayed_elements.append(self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    self.assertEqual(cached_elements, replayed_elements)

    # Re-initialize with an empty upstream and a missing cache file (should
    # throw errors.OutOfRangeError immediately).
    get_next = self.getNext(
        dataset_fn(count=0, filename=self.cache_prefix + "nonsense"))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testConcurrentWriters(self):
    components = (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
                  np.array([9.0, 10.0, 11.0, 12.0]))

    cache_dataset1 = (
        dataset_ops.Dataset.from_tensor_slices(components).cache(
            self.cache_prefix))
    cache_dataset2 = (
        dataset_ops.Dataset.from_tensor_slices(components).cache(
            self.cache_prefix))

    get_next1 = self.getNext(cache_dataset1)
    get_next2 = self.getNext(cache_dataset2)

    self.evaluate(get_next1())  # this should succeed

    with self.assertRaises(errors.AlreadyExistsError):
      self.evaluate(get_next2())

    self.evaluate(get_next1())  # this should continue to succeed

  @combinations.generate(test_base.default_test_combinations())
  def testConcurrentReaders(self):
    components = (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
                  np.array([9.0, 10.0, 11.0, 12.0]))

    cache_dataset1 = (
        dataset_ops.Dataset.from_tensor_slices(components).cache(
            self.cache_prefix))
    cache_dataset2 = (
        dataset_ops.Dataset.from_tensor_slices(components).cache(
            self.cache_prefix))

    get_next1 = self.getNext(cache_dataset1)
    get_next2 = self.getNext(cache_dataset2)

    elements = []
    for _ in range(4):
      elements.append(self.evaluate(get_next1()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next1())

    # Re-initialize
    get_next1 = self.getNext(cache_dataset1, requires_initialization=True)
    get_next2 = self.getNext(cache_dataset2, requires_initialization=True)

    # Reading concurrently should succeed.
    elements_itr1 = []
    elements_itr2 = []
    elements_itr2.append(self.evaluate(get_next2()))
    elements_itr1.append(self.evaluate(get_next1()))
    elements_itr2.append(self.evaluate(get_next2()))
    elements_itr1.append(self.evaluate(get_next1()))
    # Intentionally reversing the order
    elements_itr1.append(self.evaluate(get_next1()))
    elements_itr2.append(self.evaluate(get_next2()))
    elements_itr1.append(self.evaluate(get_next1()))
    elements_itr2.append(self.evaluate(get_next2()))

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next2())

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next1())

    self.assertAllEqual(elements, elements_itr1)
    self.assertAllEqual(elements, elements_itr2)

  @combinations.generate(test_base.default_test_combinations())
  def testReadingPastEndOfSequence(self):
    dataset = dataset_ops.Dataset.range(10).cache(self.cache_prefix)
    dataset = dataset.map(lambda a: a).batch(4).repeat(2)
    expected_output = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]] * 2
    self.assertDatasetProduces(dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testCleaningUpCacheFiles(self):

    def do_test(i):
      dataset = dataset_ops.Dataset.range(10).cache(self.cache_prefix)
      get_next = self.getNext(dataset)
      for _ in range(i):
        try:
          self.evaluate(get_next())
        except errors.OutOfRangeError:
          break

    if not context.executing_eagerly():
      self.skipTest(
          "Test requires eager mode for iterators to be deconstructed")

    for i in [0, 3, 10, 12, 15]:
      do_test(i)


class MemoryCacheTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testCacheDatasetPassthrough(self):
    with ops.device("cpu:0"):
      repeat_count = variables.Variable(constant_op.constant(10, dtypes.int64))
      dataset = dataset_ops.Dataset.range(3).flat_map(
          lambda x: dataset_ops.Dataset.from_tensors(x).repeat(repeat_count))

      cached_dataset = dataset.cache().repeat(2)
      uncached_dataset = dataset.repeat(2)

      self.evaluate(repeat_count.initializer)
      # Needs to be initializable to capture the variable.
      cached_next = self.getNext(cached_dataset, requires_initialization=True)
      uncached_next = self.getNext(
          uncached_dataset, requires_initialization=True)
      for i in range(3):
        for _ in range(10):
          self.assertEqual(self.evaluate(cached_next()), i)
          self.assertEqual(self.evaluate(uncached_next()), i)

      self.evaluate(repeat_count.assign(0))

      # The uncached iterator should now be empty.
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(uncached_next())

      # The cached iterator replays from cache.
      for i in range(3):
        for _ in range(10):
          self.assertEqual(self.evaluate(cached_next()), i)

      # The cached iterator should now be empty.
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(cached_next())

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyCacheReading(self):
    components = (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
                  np.array([9.0, 10.0, 11.0, 12.0]))

    repeat_dataset = (
        dataset_ops.Dataset.from_tensor_slices(components).repeat(0))
    cache_dataset = repeat_dataset.cache()

    # Create initialization ops for iterators without and with
    # caching, respectively.
    self.assertDatasetProduces(cache_dataset, expected_output=[])

  @combinations.generate(test_base.default_test_combinations())
  def testConcurrentReaders(self):

    dataset_fn = lambda: dataset_ops.Dataset.range(5).cache()
    d1 = dataset_fn().map(lambda x: x + 1)
    d2 = dataset_fn().map(lambda x: x + 6)

    get_next1 = self.getNext(d1)

    self.assertEqual(1, self.evaluate(get_next1()))
    self.assertEqual(2, self.evaluate(get_next1()))
    self.assertEqual(3, self.evaluate(get_next1()))

    get_next2 = self.getNext(d2)

    self.assertEqual(6, self.evaluate(get_next2()))
    self.assertEqual(7, self.evaluate(get_next2()))
    self.assertEqual(4, self.evaluate(get_next1()))  # interleave execution
    self.assertEqual([8, 5],
                     [self.evaluate(get_next2()),
                      self.evaluate(get_next1())])
    self.assertEqual(9, self.evaluate(get_next2()))
    self.assertEqual(10, self.evaluate(get_next2()))

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next2())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next1())

  @combinations.generate(test_base.default_test_combinations())
  def testCacheTakeRepeat(self):
    dataset = dataset_ops.Dataset.range(10).cache().take(5).repeat(2)

    expected_output = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testCacheRepeatEpochs(self):
    counter = variables.Variable(0)
    self.evaluate(counter.initializer)

    def increment_fn(x):
      counter.assign_add(1)
      return x

    dataset = dataset_ops.Dataset.range(10).map(increment_fn).cache().repeat(2)
    get_next = self.getNext(dataset, requires_initialization=True)

    # first epoch
    for i in range(10):
      self.assertEqual(i, self.evaluate(counter))
      self.assertEqual(i, self.evaluate(get_next()))
    # second epoch
    for i in range(10):
      self.assertEqual(10, self.evaluate(counter))
      self.assertEqual(i, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(combinations.combine(tf_api_version=2, mode="eager"))
  def testCacheIterationEpochs(self):
    counter = variables.Variable(0)
    self.evaluate(counter.initializer)

    def increment_fn(x):
      counter.assign_add(1)
      return x

    dataset = dataset_ops.Dataset.range(10).map(increment_fn).cache()

    # first epoch
    i = 0
    for elem in dataset:
      self.assertEqual(i, self.evaluate(elem))
      i += 1
      self.assertEqual(i, self.evaluate(counter))

    # second epoch
    i = 0
    for elem in dataset:
      self.assertEqual(10, self.evaluate(counter))
      self.assertEqual(i, self.evaluate(elem))
      i += 1

  @combinations.generate(combinations.combine(tf_api_version=2, mode="eager"))
  def testCacheV2ResourceCapture(self):

    def make_dataset():
      ids = dataset_ops.Dataset.range(10)
      ids = ids.cache()

      def interleave_fn(dataset, _):
        return dataset

      dataset = dataset_ops.Dataset.range(1)
      dataset = dataset.interleave(functools.partial(interleave_fn, ids))
      return dataset

    results = []
    for elem in make_dataset():
      results.append(elem.numpy())

    self.assertAllEqual(results, range(10))

  @combinations.generate(combinations.combine(tf_api_version=2, mode="eager"))
  def testCacheV2ConcurrentIterators(self):

    dataset = dataset_ops.Dataset.range(10).cache()

    it1 = iter(dataset)
    it2 = iter(dataset)

    for i in range(10):
      self.assertEqual(next(it1), i)
      self.assertEqual(next(it2), i)

  @combinations.generate(combinations.combine(tf_api_version=2, mode="eager"))
  def testCacheKnownCardinality(self):

    # Check that a dataset which produces random permutation of range(10) ends
    # up being cached when we read all of its element but do not reach EOF.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.shuffle(10, reshuffle_each_iteration=True).cache()

    it = iter(dataset)

    results = []
    for _ in range(10):
      results.append(next(it))

    it = iter(dataset)
    for i in range(10):
      self.assertEqual(next(it), results[i])


if __name__ == "__main__":
  test.main()

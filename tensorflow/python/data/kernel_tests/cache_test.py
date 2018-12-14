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

from os import path
import shutil
import tempfile

import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class FileCacheTest(test_base.DatasetTestBase):

  def setUp(self):
    self.tmp_dir = tempfile.mkdtemp()
    self.cache_prefix = path.join(self.tmp_dir, "cache")

  def tearDown(self):
    if self.tmp_dir:
      shutil.rmtree(self.tmp_dir, ignore_errors=True)

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
        dataset_fn().output_shapes)

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
    get_next1 = self.getNext(cache_dataset1)
    get_next2 = self.getNext(cache_dataset2)

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


@test_util.run_all_in_graph_and_eager_modes
class MemoryCacheTest(test_base.DatasetTestBase):

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

  def testEmptyCacheReading(self):
    components = (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
                  np.array([9.0, 10.0, 11.0, 12.0]))

    repeat_dataset = (
        dataset_ops.Dataset.from_tensor_slices(components).repeat(0))
    cache_dataset = repeat_dataset.cache()

    # Create initialization ops for iterators without and with
    # caching, respectively.
    self.assertDatasetProduces(cache_dataset, expected_output=[])

  def testConcurrentReaders(self):

    dataset = dataset_ops.Dataset.range(5).cache()
    d1 = dataset.map(lambda x: x + 1)
    d2 = dataset.map(lambda x: x + 6)

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

  def testCacheTakeRepeat(self):
    dataset = dataset_ops.Dataset.range(10).cache().take(5).repeat(2)

    expected_output = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    self.assertDatasetProduces(dataset, expected_output=expected_output)


if __name__ == "__main__":
  test.main()

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

import collections

import numpy as np

from tensorflow.contrib.data.python.kernel_tests import dataset_serialization_test_base
from tensorflow.contrib.data.python.ops import dataset_ops as contrib_dataset_ops
from tensorflow.contrib.data.python.ops import shuffle_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ShuffleDatasetTest(test.TestCase):

  def testShuffleDataset(self):
    components = (
        np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
        np.array([9.0, 10.0, 11.0, 12.0])
    )
    count_placeholder = array_ops.placeholder_with_default(
        constant_op.constant(5, dtypes.int64), shape=[])
    buffer_size_placeholder = array_ops.placeholder(dtypes.int64, shape=[])
    seed_placeholder = array_ops.placeholder(dtypes.int64, shape=[])

    repeat_dataset = (
        contrib_dataset_ops.Dataset.from_tensor_slices(components)
        .repeat(count_placeholder))

    shuffle_dataset = repeat_dataset.shuffle(buffer_size_placeholder,
                                             seed_placeholder)

    self.assertEqual(tuple([c.shape[1:] for c in components]),
                     shuffle_dataset.output_shapes)

    # Create initialization ops for iterators without and with
    # shuffling, respectively.
    iterator = iterator_ops.Iterator.from_structure(
        shuffle_dataset.output_types, shuffle_dataset.output_shapes)
    init_fifo_op = iterator.make_initializer(repeat_dataset)
    init_shuffle_op = iterator.make_initializer(shuffle_dataset)

    get_next = iterator.get_next()

    with self.test_session() as sess:
      # First run without shuffling to collect the "ground truth".
      sess.run(init_fifo_op)
      unshuffled_elements = []
      for _ in range(20):
        unshuffled_elements.append(sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Assert that the shuffled dataset has the same elements as the
      # "ground truth".
      sess.run(
          init_shuffle_op,
          feed_dict={buffer_size_placeholder: 100,
                     seed_placeholder: 37})
      shuffled_elements = []
      for _ in range(20):
        shuffled_elements.append(sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)
      self.assertAllEqual(
          sorted(unshuffled_elements), sorted(shuffled_elements))

      # Assert that shuffling twice with the same seeds gives the same sequence.
      sess.run(
          init_shuffle_op,
          feed_dict={buffer_size_placeholder: 100,
                     seed_placeholder: 37})
      reshuffled_elements_same_seed = []
      for _ in range(20):
        reshuffled_elements_same_seed.append(sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)
      self.assertEqual(shuffled_elements, reshuffled_elements_same_seed)

      # Assert that shuffling twice with a different seed gives a different
      # permutation of the same elements.
      sess.run(
          init_shuffle_op,
          feed_dict={buffer_size_placeholder: 100,
                     seed_placeholder: 1037})
      reshuffled_elements_different_seed = []
      for _ in range(20):
        reshuffled_elements_different_seed.append(sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)
      self.assertNotEqual(shuffled_elements, reshuffled_elements_different_seed)
      self.assertAllEqual(
          sorted(shuffled_elements), sorted(reshuffled_elements_different_seed))

      # Assert that the shuffled dataset has the same elements as the
      # "ground truth" when the buffer size is smaller than the input
      # dataset.
      sess.run(
          init_shuffle_op,
          feed_dict={buffer_size_placeholder: 2,
                     seed_placeholder: 37})
      reshuffled_elements_small_buffer = []
      for _ in range(20):
        reshuffled_elements_small_buffer.append(sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)
      self.assertAllEqual(
          sorted(unshuffled_elements), sorted(reshuffled_elements_small_buffer))

      # Test the case of shuffling an empty dataset.
      sess.run(init_shuffle_op, feed_dict={buffer_size_placeholder: 2,
                                           seed_placeholder: 37,
                                           count_placeholder: 0})
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testDefaultArguments(self):
    components = [0, 1, 2, 3, 4]
    iterator = (
        contrib_dataset_ops.Dataset.from_tensor_slices(components).shuffle(5)
        .repeat().make_one_shot_iterator())

    get_next = iterator.get_next()

    with self.test_session() as sess:
      counts = collections.defaultdict(lambda: 0)
      for _ in range(10):
        for _ in range(5):
          counts[sess.run(get_next)] += 1

    for i in range(5):
      self.assertEqual(10, counts[i])


class ShuffleDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_shuffle_dataset(
      self,
      range_limit=10,
      num_repeats=5,
      buffer_size=5,
      seed=None,
      reshuffle_each_iteration=None,
  ):
    return dataset_ops.Dataset.range(range_limit).shuffle(
        buffer_size,
        seed=seed,
        reshuffle_each_iteration=reshuffle_each_iteration).repeat(num_repeats)

  def testShuffleCore(self):

    seed = 55
    range_limit = 10
    num_repeats = 5
    num_outputs = range_limit * num_repeats
    buffer_sizes = [1, 3, 8, 10, 25, 50]
    reshuffle_each_iteration = False
    # pylint: disable=cell-var-from-loop
    # pylint: disable=g-long-lambda
    for buffer_size in buffer_sizes:
      self.run_core_tests(
          lambda: self._build_shuffle_dataset(
              range_limit=range_limit,
              num_repeats=num_repeats,
              buffer_size=buffer_size,
              seed=seed,
              reshuffle_each_iteration=reshuffle_each_iteration),
          lambda: self._build_shuffle_dataset(
              range_limit=range_limit,
              num_repeats=num_repeats,
              buffer_size=buffer_size,
              seed=10,
              reshuffle_each_iteration=reshuffle_each_iteration),
          num_outputs)
    # pylint: enable=cell-var-from-loop
    # pylint: enable=g-long-lambda


class ShuffleAndRepeatTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_ds(self, seed, count=5, num_elements=20):
    return dataset_ops.Dataset.range(num_elements).apply(
        shuffle_ops.shuffle_and_repeat(buffer_size=5, count=count, seed=seed))

  def testCorrectOutput(self):
    output = self.gen_outputs(lambda: self._build_ds(10), [], 100)
    self.assertSequenceEqual(
        sorted(output), sorted(
            np.array([range(20) for _ in range(5)]).flatten()))
    for i in range(5):
      self.assertSequenceEqual(sorted(output[i * 20:(i + 1) * 20]), range(20))

  def testReshuffling(self):
    # Check that the output orders of different epochs are indeed different.
    output = self.gen_outputs(lambda: self._build_ds(10), [], 100)
    for i in range(4):
      epoch1 = output[i * 20:(i + 1) * 20]
      epoch2 = output[(i + 1) * 20:(i + 2) * 20]
      self.assertNotEqual(epoch1, epoch2)

  def testSameOrderForSameSeeds(self):
    output1 = self.gen_outputs(lambda: self._build_ds(10), [], 100)
    output2 = self.gen_outputs(lambda: self._build_ds(10), [], 100)
    self.assertEqual(output1, output2)

  def testDifferentOrderForDifferentSeeds(self):
    output1 = self.gen_outputs(lambda: self._build_ds(10), [], 100)
    output2 = self.gen_outputs(lambda: self._build_ds(20), [], 100)
    self.assertNotEqual(output1, output2)
    self.assertEqual(sorted(output1), sorted(output2))

  def testCountNone(self):
    output1 = self.gen_outputs(
        lambda: self._build_ds(10, count=None), [], 100, verify_exhausted=False)
    output2 = self.gen_outputs(
        lambda: self._build_ds(20, count=None), [], 100, verify_exhausted=False)
    self.assertNotEqual(output1, output2)
    self.assertEqual(sorted(output1), sorted(output2))

  def testCountMinusOne(self):
    output1 = self.gen_outputs(
        lambda: self._build_ds(10, count=-1), [], 100, verify_exhausted=False)
    output2 = self.gen_outputs(
        lambda: self._build_ds(20, count=-1), [], 100, verify_exhausted=False)
    self.assertNotEqual(output1, output2)
    self.assertEqual(sorted(output1), sorted(output2))

  def testInfiniteOutputs(self):
    # Asserting the iterator is exhausted after producing 100 items should fail.
    with self.assertRaises(AssertionError):
      self.gen_outputs(lambda: self._build_ds(10, count=None), [], 100)
    with self.assertRaises(AssertionError):
      self.gen_outputs(lambda: self._build_ds(10, count=-1), [], 100)

  def testInfiniteEmpty(self):
    with self.assertRaises(errors.OutOfRangeError):
      self.gen_outputs(lambda: self._build_ds(10, count=None, num_elements=0),
                       [], 100)
    with self.assertRaises(errors.OutOfRangeError):
      self.gen_outputs(lambda: self._build_ds(10, count=-1, num_elements=0), [],
                       100)

  def testLargeBufferSize(self):
    with ops.Graph().as_default() as g:
      ds = dataset_ops.Dataset.range(20).apply(
          shuffle_ops.shuffle_and_repeat(buffer_size=21))
      get_next_op = ds.make_one_shot_iterator().get_next()
      with self.test_session(graph=g) as sess:
        sess.run(get_next_op)


class ShuffleAndRepeatSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_ds(self, seed):
    return dataset_ops.Dataset.range(20).apply(
        shuffle_ops.shuffle_and_repeat(buffer_size=5, count=5, seed=seed))

  def testCore(self):
    self.run_core_tests(lambda: self._build_ds(10), lambda: self._build_ds(20),
                        100)


if __name__ == "__main__":
  test.main()

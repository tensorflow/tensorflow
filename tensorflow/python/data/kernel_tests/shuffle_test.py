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
"""Tests for `tf.data.Dataset.shuffle()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class ShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):

  def testShuffleDataset(self):
    components = (
        np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]),
        np.array([9.0, 10.0, 11.0, 12.0])
    )

    def dataset_fn(count=5, buffer_size=None, seed=0):
      repeat_dataset = (
          dataset_ops.Dataset.from_tensor_slices(components).repeat(count))
      if buffer_size:
        shuffle_dataset = repeat_dataset.shuffle(buffer_size, seed)

        self.assertEqual(
            tuple([c.shape[1:] for c in components]),
            shuffle_dataset.output_shapes)
        return shuffle_dataset
      else:
        return repeat_dataset

    # First run without shuffling to collect the "ground truth".
    get_next = self.getNext(dataset_fn())
    unshuffled_elements = []
    for _ in range(20):
      unshuffled_elements.append(self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

    # Assert that the shuffled dataset has the same elements as the
    # "ground truth".
    get_next = self.getNext(dataset_fn(buffer_size=100, seed=37))
    shuffled_elements = []
    for _ in range(20):
      shuffled_elements.append(self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    self.assertAllEqual(sorted(unshuffled_elements), sorted(shuffled_elements))

    # Assert that shuffling twice with the same seeds gives the same sequence.
    get_next = self.getNext(dataset_fn(buffer_size=100, seed=37))
    reshuffled_elements_same_seed = []
    for _ in range(20):
      reshuffled_elements_same_seed.append(self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    self.assertEqual(shuffled_elements, reshuffled_elements_same_seed)

    # Assert that shuffling twice with a different seed gives a different
    # permutation of the same elements.
    get_next = self.getNext(dataset_fn(buffer_size=100, seed=137))
    reshuffled_elements_different_seed = []
    for _ in range(20):
      reshuffled_elements_different_seed.append(self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    self.assertNotEqual(shuffled_elements, reshuffled_elements_different_seed)
    self.assertAllEqual(
        sorted(shuffled_elements), sorted(reshuffled_elements_different_seed))

    # Assert that the shuffled dataset has the same elements as the
    # "ground truth" when the buffer size is smaller than the input
    # dataset.
    get_next = self.getNext(dataset_fn(buffer_size=2, seed=37))
    reshuffled_elements_small_buffer = []
    for _ in range(20):
      reshuffled_elements_small_buffer.append(self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    self.assertAllEqual(
        sorted(unshuffled_elements), sorted(reshuffled_elements_small_buffer))

    # Test the case of shuffling an empty dataset.
    get_next = self.getNext(dataset_fn(count=0, buffer_size=100, seed=37))

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @test_util.run_deprecated_v1
  def testSkipEagerSeedZero(self):
    """Test for same behavior when the seed is a Python or Tensor zero."""
    iterator = dataset_ops.make_one_shot_iterator(
        dataset_ops.Dataset.range(10).shuffle(10, seed=0))
    get_next = iterator.get_next()

    elems = []
    with self.cached_session() as sess:
      for _ in range(10):
        elems.append(sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

    seed_placeholder = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.make_initializable_iterator(
        dataset_ops.Dataset.range(10).shuffle(10, seed=seed_placeholder))
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(iterator.initializer, feed_dict={seed_placeholder: 0})
      for elem in elems:
        self.assertEqual(elem, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testDefaultArguments(self):
    components = [0, 1, 2, 3, 4]
    dataset = dataset_ops.Dataset.from_tensor_slices(components).shuffle(
        5).repeat()
    get_next = self.getNext(dataset)
    counts = collections.defaultdict(lambda: 0)
    for _ in range(10):
      for _ in range(5):
        counts[self.evaluate(get_next())] += 1

    for i in range(5):
      self.assertEqual(10, counts[i])

  def testShuffleNoReshuffleEachIteration(self):
    dataset = dataset_ops.Dataset.range(10).shuffle(
        10, reshuffle_each_iteration=False).batch(10).repeat(3)
    next_element = self.getNext(dataset)

    initial_permutation = self.evaluate(next_element())
    self.assertAllEqual(initial_permutation, self.evaluate(next_element()))
    self.assertAllEqual(initial_permutation, self.evaluate(next_element()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  def testShuffleReshuffleEachIteration(self):
    dataset = dataset_ops.Dataset.range(10).shuffle(
        10, seed=3, reshuffle_each_iteration=True).batch(10).repeat(3)
    next_element = self.getNext(dataset)

    initial_permutation = list(self.evaluate(next_element()))
    for _ in range(2):
      next_permutation = list(self.evaluate(next_element()))
      self.assertNotEqual(initial_permutation, next_permutation)
      self.assertAllEqual(sorted(initial_permutation), sorted(next_permutation))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @parameterized.named_parameters(
      ("ReshuffleGraphLevelSeed", True, 38, None),
      ("ReshuffleOpLevelSeed", True, None, 42),
      ("ReshuffleGraphAndOpLevelSeed", True, 38, 42),
      ("NoReshuffleGraphLevelSeed", False, 38, None),
      ("NoReshuffleOpLevelSeed", False, None, 42),
      ("NoReshuffleGraphAndOpLevelSeed", False, 38, 42),
  )
  def testSkipEagerShuffleSeed(self, reshuffle, graph_level_seed,
                               op_level_seed):
    results = []
    for _ in range(2):
      with ops.Graph().as_default() as g:
        random_seed.set_random_seed(graph_level_seed)
        dataset = dataset_ops.Dataset.range(10).shuffle(
            10, seed=op_level_seed, reshuffle_each_iteration=reshuffle).repeat(
                3)
        iterator = dataset_ops.make_one_shot_iterator(dataset)
        next_element = iterator.get_next()

        run_results = []
        with self.session(graph=g) as sess:
          for _ in range(30):
            run_results.append(sess.run(next_element))
          with self.assertRaises(errors.OutOfRangeError):
            sess.run(next_element)
        results.append(run_results)

    self.assertAllEqual(results[0], results[1])

  # TODO(b/117581999): fails for eager mode with result[0] equal to result[1],
  # debug.
  @parameterized.named_parameters(
      ("ReshuffleOneShot", True, False),
      ("ReshuffleInitializable", True, True),
      ("NoReshuffleOneShot", False, False),
      ("NoReshuffleInitializable", False, True),
  )
  def testSkipEagerMultipleIterators(self, reshuffle, initializable):
    with ops.Graph().as_default() as g:
      dataset = dataset_ops.Dataset.range(100).shuffle(
          10, reshuffle_each_iteration=reshuffle).repeat(3)

      if initializable:
        iterators = [dataset_ops.make_initializable_iterator(dataset)
                     for _ in range(2)]
      else:
        iterators = [dataset_ops.make_one_shot_iterator(dataset)
                     for _ in range(2)]

      results = []
      with self.session(graph=g) as sess:
        for iterator in iterators:
          if initializable:
            sess.run(iterator.initializer)
          next_element = iterator.get_next()
          run_results = []
          for _ in range(300):
            run_results.append(sess.run(next_element))
          with self.assertRaises(errors.OutOfRangeError):
            sess.run(next_element)

          results.append(run_results)

        self.assertNotEqual(results[0], results[1])


if __name__ == "__main__":
  test.main()

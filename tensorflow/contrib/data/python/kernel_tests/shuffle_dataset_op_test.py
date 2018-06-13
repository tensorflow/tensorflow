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

import numpy as np

from tensorflow.contrib.data.python.kernel_tests import dataset_serialization_test_base
from tensorflow.contrib.data.python.ops import iterator_ops as contrib_iterator_ops
from tensorflow.contrib.data.python.ops import shuffle_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib


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
    # pylint: disable=cell-var-from-loop
    # pylint: disable=g-long-lambda
    for reshuffle_each_iteration in [True, False]:
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

  def testNonDeterministicSeeding(self):

    range_limit = 10
    num_repeats = 5
    num_outputs = range_limit * num_repeats
    buffer_sizes = [1, 3, 8, 10, 25, 50]
    for reshuffle_each_iteration in [True, False]:
      for buffer_size in buffer_sizes:

        def ds_fn():
          # pylint: disable=cell-var-from-loop
          return self._build_shuffle_dataset(
              range_limit=range_limit,
              num_repeats=num_repeats,
              buffer_size=buffer_size,
              seed=None,  # Iterator seeds are generated non-deterministically.
              reshuffle_each_iteration=reshuffle_each_iteration)
          # pylint: enable=cell-var-from-loop

        # We checkpoint the initial state of the Dataset so that we can restore
        # the seeds in the next run. Since the seeding is non-deterministic
        # the dataset gets initialized with different seeds each time.
        expected = self.gen_outputs(
            ds_fn,
            break_points=[0],
            num_outputs=num_outputs,
            ckpt_saved=False,
            verify_exhausted=False,
            save_checkpoint_at_end=False)
        actual = self.gen_outputs(
            ds_fn,
            break_points=self.gen_break_points(num_outputs),
            num_outputs=num_outputs,
            ckpt_saved=True,
            verify_exhausted=False)
        self.match(expected, actual)

  def testMultipleIterators(self):
    range_limit = 10
    num_repeats = 5
    num_outputs = range_limit * num_repeats
    buffer_sizes = [1, 3, 8, 10, 25, 50]

    for reshuffle_each_iteration in [True, False]:
      for buffer_size in buffer_sizes:

        def ds_fn():
          # pylint: disable=cell-var-from-loop
          return self._build_shuffle_dataset(
              range_limit=range_limit,
              num_repeats=num_repeats,
              buffer_size=buffer_size,
              seed=None,  # Iterator seeds are generated non-deterministically.
              reshuffle_each_iteration=reshuffle_each_iteration)
          # pylint: enable=cell-var-from-loop

        with ops.Graph().as_default() as g:
          ds = ds_fn()
          iterators = [ds.make_one_shot_iterator(), ds.make_one_shot_iterator()]
          get_next_ops = [it.get_next() for it in iterators]
          saveables = [
              contrib_iterator_ops.make_saveable_from_iterator(it)
              for it in iterators
          ]
          for saveable in saveables:
            ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
          saver = saver_lib.Saver(allow_empty=True)
          with self.test_session(graph=g) as sess:
            self._save(sess, saver)
            expected = [sess.run(get_next_ops) for _ in range(num_outputs)]
            self._restore(saver, sess)
            actual = [sess.run(get_next_ops) for _ in range(num_outputs)]
            self.match(expected, actual)


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

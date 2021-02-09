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
"""Tests for the ShuffleDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import iterator_ops as contrib_iterator_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib


class ShuffleDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

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

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              reshuffle_each_iteration=[True, False],
              buffer_size=[1, 3, 5, 8, 10])))
  def testShuffleCore(self, reshuffle_each_iteration, buffer_size):
    seed = 55
    range_limit = 5
    num_repeats = 2
    num_outputs = range_limit * num_repeats
    # pylint: disable=g-long-lambda
    self.run_core_tests(
        lambda: self._build_shuffle_dataset(
            range_limit=range_limit,
            num_repeats=num_repeats,
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration), num_outputs)

  @combinations.generate(
      combinations.combine(
          tf_api_version=1,
          mode=["graph"],
          reshuffle_each_iteration=[True, False],
          buffer_size=[1, 3, 5, 8, 10]))
  def testMultipleIterators(self, reshuffle_each_iteration, buffer_size):
    range_limit = 5
    num_repeats = 2
    num_outputs = range_limit * num_repeats

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
      with self.session(graph=g) as sess:
        self._save(sess, saver)
        expected = [self.evaluate(get_next_ops) for _ in range(num_outputs)]
        self._restore(saver, sess)
        actual = [self.evaluate(get_next_ops) for _ in range(num_outputs)]
        self.match(expected, actual)


if __name__ == "__main__":
  test.main()

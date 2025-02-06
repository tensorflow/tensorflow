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
import collections
import functools
import sys

from absl.testing import parameterized
import numpy as np
from tensorflow.python import pywrap_sanitizers
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.experimental.ops import iterator_ops as contrib_iterator_ops
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib


def make_variable_size_dataset(per_epoch_data):
  repeat_counter = [0]

  def gen():
    for each in per_epoch_data[repeat_counter[0]]:
      yield each
    repeat_counter[0] += 1

  return dataset_ops.Dataset.from_generator(gen, dtypes.int64)


class ShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testBasic(self):
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
            dataset_ops.get_legacy_output_shapes(shuffle_dataset))
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
    get_next = self.getNext(dataset_fn(
        buffer_size=100, seed=constant_op.constant(137, dtype=dtypes.int64)))
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

  @combinations.generate(combinations.combine(tf_api_version=1, mode="graph"))
  def testSeedZero(self):
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

  @combinations.generate(test_base.default_test_combinations())
  def testDefaultArguments(self):
    components = [0, 1, 2, 3, 4]
    dataset = (
        dataset_ops.Dataset.from_tensor_slices(components).shuffle(5).repeat()
    )
    get_next = self.getNext(dataset)
    counts = collections.defaultdict(lambda: 0)
    for _ in range(10):
      for _ in range(5):
        counts[self.evaluate(get_next())] += 1

    for i in range(5):
      self.assertEqual(10, counts[i])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              dataset_range=[100],
              buffer_size=[None, 10, 200],
              seed=[None, 42],
              use_tensor_input=[True, False])))
  def testTensorInput(self, dataset_range, buffer_size, seed, use_tensor_input):
    dataset = dataset_ops.Dataset.range(dataset_range)
    unshuffled_output = self.getDatasetOutput(dataset)

    if buffer_size:
      buffer_size = (
          constant_op.constant(buffer_size, dtype=dtypes.int64)
          if use_tensor_input else buffer_size)
    else:
      buffer_size = dataset.cardinality()
    seed = (constant_op.constant(seed, dtype=dtypes.int64)
            if seed and use_tensor_input else seed)

    shuffled_dataset = dataset.shuffle(buffer_size, seed=seed)
    shuffled_output = self.getDatasetOutput(shuffled_dataset)
    self.assertEqual(unshuffled_output, list(range(dataset_range)))
    self.assertCountEqual(shuffled_output, unshuffled_output)
    self.assertNotEqual(shuffled_output, unshuffled_output)

  @combinations.generate(test_base.default_test_combinations())
  def testUnknownCardinality(self):
    components = [0, 1, 2, 3, 4]
    dataset = dataset_ops.Dataset.from_tensor_slices(components).shuffle(
        dataset_ops.UNKNOWN
    )
    get_next = self.getNext(dataset)
    counts = collections.defaultdict(lambda: 0)
    for _ in range(1):
      for _ in range(5):
        counts[self.evaluate(get_next())] += 1

    for i in range(5):
      self.assertEqual(1, counts[i])

  @combinations.generate(test_base.default_test_combinations())
  def testUnknownCardinalityWithRepeatedShuffle(self):
    components = [0, 1, 2, 3, 4]
    dataset = (
        dataset_ops.Dataset.from_tensor_slices(components)
        .shuffle(dataset_ops.UNKNOWN)
        .repeat()
    )
    get_next = self.getNext(dataset)
    counts = collections.defaultdict(lambda: 0)
    for _ in range(10):
      for _ in range(5):
        counts[self.evaluate(get_next())] += 1

    for i in range(5):
      self.assertEqual(10, counts[i])

  @combinations.generate(test_base.default_test_combinations())
  def testUnknownCardinalityWithIncreasingBufferSize(self):
    epoch_1 = list(range(5))
    epoch_2 = list(range(10, 17))
    epoch_3 = list(range(20, 28))

    ds = make_variable_size_dataset([epoch_1, epoch_2, epoch_3])
    ds = ds.shuffle(dataset_ops.UNKNOWN).repeat(3)

    expected = epoch_1 + epoch_2 + epoch_3
    self.assertDatasetProduces(ds, expected, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testUnknownCardinalityWithVariableBufferSize(self):
    epoch_1 = list(range(5))
    epoch_2 = list(range(10, 13))
    epoch_3 = list(range(20, 27))

    ds = make_variable_size_dataset([epoch_1, epoch_2, epoch_3])
    ds = ds.shuffle(dataset_ops.UNKNOWN).repeat(3)

    expected = epoch_1 + epoch_2 + epoch_3
    self.assertDatasetProduces(ds, expected, assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testInputInitializations(self):
    num_rounds = 3
    def compute_orders(dataset):
      orders = []
      for _ in range(num_rounds):
        orders.append(self.getDatasetOutput(dataset))
      return orders

    dataset = dataset_ops.Dataset.range(10).shuffle(10, seed=1)
    first_orders = compute_orders(dataset)
    dataset = dataset_ops.Dataset.range(10)

    # Adding shuffle(1) should not change the order.
    dataset = dataset_ops.Dataset.range(10).shuffle(10, seed=1).shuffle(1)
    second_orders = compute_orders(dataset)
    self.assertEqual(first_orders, second_orders)

  @combinations.generate(
      combinations.times(
          test_base.graph_only_combinations(),
          combinations.combine(reshuffle=[True, False]),
          combinations.combine(graph_seed=38, op_seed=None) +
          combinations.combine(graph_seed=None, op_seed=42) +
          combinations.combine(graph_seed=38, op_seed=42)))
  def testShuffleSeed(self, reshuffle, graph_seed, op_seed):
    results = []
    for _ in range(2):
      with ops.Graph().as_default() as g:
        random_seed.set_random_seed(graph_seed)
        dataset = dataset_ops.Dataset.range(10).shuffle(
            10, seed=op_seed, reshuffle_each_iteration=reshuffle).repeat(3)
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

  # TODO(b/117581999): enable this test for eager-mode.
  @combinations.generate(
      combinations.times(
          test_base.graph_only_combinations(),
          combinations.combine(
              reshuffle=[True, False], initializable=[True, False])))
  def testMultipleIterators(self, reshuffle, initializable):
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

  @combinations.generate(test_base.default_test_combinations())
  def testShuffleManyEmptyEpochs(self):
    sizes = [0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0]
    sizes_iter = iter(sizes)
    def gen():
      for i in range(next(sizes_iter)):
        yield i

    dataset = dataset_ops.Dataset.from_generator(
        gen, output_signature=tensor_spec.TensorSpec((), dtypes.int64))
    dataset = dataset.shuffle(10).repeat(len(sizes)).take(3)
    self.assertDatasetProduces(dataset, [0, 0, 1], assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testShuffleInfiniteRepeatNonemptyFollowedByEmpty(self):
    sizes = [1, 0, 2, 10]
    sizes_iter = iter(sizes)
    def gen():
      for i in range(next(sizes_iter)):
        yield i

    dataset = dataset_ops.Dataset.from_generator(
        gen, output_signature=tensor_spec.TensorSpec((), dtypes.int64))
    dataset = dataset.shuffle(10).repeat().take(3)
    self.assertDatasetProduces(dataset, [0, 0, 1], assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(reshuffle=[True, False], seed=[None, 42])))
  def testReshuffleRepeatEpochs(self, reshuffle, seed):
    dataset = dataset_ops.Dataset.range(10).shuffle(
        10, seed=seed, reshuffle_each_iteration=reshuffle).repeat(2)
    next_element = self.getNext(dataset)

    first_epoch = []
    for _ in range(10):
      first_epoch.append(self.evaluate(next_element()))

    second_epoch = []
    for _ in range(10):
      second_epoch.append(self.evaluate(next_element()))

    self.assertEqual(first_epoch == second_epoch, not reshuffle)

  @combinations.generate(
      combinations.times(
          combinations.combine(tf_api_version=2, mode="eager"),
          combinations.combine(reshuffle=[True, False], seed=[None, 42])))
  def testReshuffleIterationEpochs(self, reshuffle, seed):
    # TensorFlow unit tests set the global graph seed. We unset it here so that
    # we can control determinism via the `seed` parameter.
    random_seed.set_random_seed(None)
    dataset = dataset_ops.Dataset.range(10).shuffle(
        10, seed=seed, reshuffle_each_iteration=reshuffle)

    first_epoch = self.getDatasetOutput(dataset)
    second_epoch = self.getDatasetOutput(dataset)

    self.assertEqual(first_epoch == second_epoch, not reshuffle)

  @combinations.generate(combinations.combine(tf_api_version=2, mode="eager"))
  def testShuffleV2ResourceCapture(self):

    def make_dataset():
      ids = dataset_ops.Dataset.range(10)
      ids = ids.shuffle(1)

      def interleave_fn(dataset, _):
        return dataset

      dataset = dataset_ops.Dataset.range(1)
      dataset = dataset.interleave(functools.partial(interleave_fn, ids))
      return dataset

    results = []
    for elem in make_dataset():
      results.append(elem.numpy())

    self.assertAllEqual(results, range(10))

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(reshuffle=[True, False], seed=[None, 42])))
  def testReshuffleSeparateTransformations(self, reshuffle, seed):
    dataset = dataset_ops.Dataset.range(10)

    first_epoch = []
    for elem in dataset.shuffle(
        10, seed=seed, reshuffle_each_iteration=reshuffle):
      first_epoch.append(elem.numpy())

    second_epoch = []
    for elem in dataset.shuffle(
        10, seed=seed, reshuffle_each_iteration=reshuffle):
      second_epoch.append(elem.numpy())

    self.assertEqual(first_epoch != second_epoch, seed is None)

  @combinations.generate(combinations.combine(tf_api_version=2, mode="eager"))
  def testShuffleV2InFunction(self):
    counter_var = variables.Variable(0)

    @def_function.function
    def consume():
      ds = dataset_ops.Dataset.range(10)
      ds = ds.shuffle(1)
      for _ in ds:
        counter_var.assign(counter_var + 1)

    consume()
    self.assertAllEqual(self.evaluate(counter_var), 10)

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyDataset(self):
    dataset = dataset_ops.Dataset.from_tensors(1)

    def map_fn(x):
      with ops.control_dependencies([check_ops.assert_equal(x, 0)]):
        return x

    dataset = dataset.map(map_fn)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=10).repeat()

    get_next = self.getNext(dataset)

    # First time around, we get an error for the failed assertion.
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())

    # Second time around, we get an EOF because the cached dataset is empty.
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(reshuffle=[True, False])))
  def testDontRerandomizeOnReplicate(self, reshuffle):
    random_seed.set_random_seed(None)
    # Since the seed generator configuration is preserved across serialization
    # of the dataset, each instantiation of the shuffle dataset
    # should preserve the shuffle order if reshuffle=False. To preserve the
    # shuffle order, the original dataset must be kept alive, since if the
    # original dataset was destroyed, its seeds would also be destroyed.
    num_elements = 100
    dataset_1 = dataset_ops.Dataset.range(num_elements)
    dataset_2 = dataset_1.shuffle(
        num_elements, reshuffle_each_iteration=reshuffle)

    shuffle_1 = self.getDatasetOutput(dataset_2)
    dataset_3 = self.graphRoundTrip(dataset_2, allow_stateful=True)
    shuffle_2 = self.getDatasetOutput(dataset_3)

    self.assertCountEqual(shuffle_1, shuffle_2)
    if reshuffle:
      self.assertNotEqual(shuffle_1, shuffle_2)

  @combinations.generate(test_base.eager_only_combinations())
  def testCheckpointLargeBuffer(self):
    if (pywrap_sanitizers.is_asan_enabled() or
        pywrap_sanitizers.is_tsan_enabled() or
        pywrap_sanitizers.is_msan_enabled()):
      self.skipTest("Skip to avoid OOM when using sanitizers.")
    if sys.platform == "darwin":
      self.skipTest("Skip to avoid memory issues on mac.")
    dataset = dataset_ops.Dataset.range(12).batch(2)
    dataset = dataset.map(
        # Create tensors of size 512M.
        lambda seed: stateless_random_ops.stateless_random_uniform(
            (128, 1024, 1024), seed, dtype=dtypes.float32
        )
    )
    dataset = dataset.shuffle(buffer_size=6)
    iterator = iter(dataset)
    next(iterator)  # Request an element to fill the shuffle buffer
    ckpt = trackable_utils.Checkpoint(iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, self.get_temp_dir(), max_to_keep=1)
    manager.save()
    del dataset
    del iterator
    manager.restore_or_initialize()

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.from_tensors(42).shuffle(1, name="shuffle")
    self.assertDatasetProduces(dataset, [42])


class ShuffleCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                            parameterized.TestCase):

  def _build_shuffle_dataset(
      self,
      range_limit=10,
      num_repeats=5,
      buffer_size=5,
      seed=None,
      reshuffle_each_iteration=None,
      symbolic_checkpoint=None,
  ):
    dataset = (
        dataset_ops.Dataset.range(range_limit)
        .shuffle(
            buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
        )
        .repeat(num_repeats)
    )

    if symbolic_checkpoint:
      options = options_lib.Options()
      options.experimental_symbolic_checkpoint = symbolic_checkpoint
      dataset = dataset.with_options(options)

    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(
              symbolic_checkpoint=[True, False],
              reshuffle_each_iteration=[True, False],
              buffer_size=[1, 3, 5, 8, 10, dataset_ops.UNKNOWN],
          ),
      )
  )
  def test(
      self,
      verify_fn,
      symbolic_checkpoint,
      reshuffle_each_iteration,
      buffer_size,
  ):
    seed = 55
    range_limit = 5
    num_repeats = 2
    num_outputs = range_limit * num_repeats
    # pylint: disable=g-long-lambda
    verify_fn(
        self,
        lambda: self._build_shuffle_dataset(
            range_limit=range_limit,
            num_repeats=num_repeats,
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
            symbolic_checkpoint=symbolic_checkpoint,
        ),
        num_outputs,
    )

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


class ShuffleRandomAccessTest(test_base.DatasetTestBase,
                              parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidIndex(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3
                                                     ]).shuffle(buffer_size=100)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, -1))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 4))

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyDataset(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        []).shuffle(buffer_size=100)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 0))

  @combinations.generate(test_base.eager_only_combinations())
  def testBasicWithoutSeedEager(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    shuffled_dataset = dataset.shuffle(buffer_size=100)

    dataset_array = []
    shuffled_dataset_array = []

    for i in range(5):
      shuffled_dataset_array.append(
          self.evaluate(random_access.at(shuffled_dataset, i)))
      dataset_array.append(self.evaluate(random_access.at(dataset, i)))
    self.assertAllEqual(sorted(dataset_array), sorted(shuffled_dataset_array))

  @combinations.generate(test_base.default_test_combinations())
  def testSameSeedReturnsSameSequence(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    shuffled_dataset = dataset.shuffle(buffer_size=100, seed=5)
    shuffled_dataset_2 = dataset.shuffle(buffer_size=100, seed=5)

    shuffled_dataset_array = []
    shuffled_dataset_array_2 = []

    for i in range(5):
      shuffled_dataset_array.append(
          self.evaluate(random_access.at(shuffled_dataset, i)))
      shuffled_dataset_array_2.append(
          self.evaluate(random_access.at(shuffled_dataset_2, i)))
    self.assertAllEqual(shuffled_dataset_array, shuffled_dataset_array_2)

  @combinations.generate(test_base.eager_only_combinations())
  def testDifferentSeedDifferentSequence(self):
    components = list(range(1000))
    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    shuffled_dataset = dataset.shuffle(buffer_size=1000, seed=124)
    shuffled_dataset_2 = dataset.shuffle(buffer_size=1000, seed=51)

    shuffled_dataset_array = []
    shuffled_dataset_array_2 = []

    for i in range(1000):
      shuffled_dataset_array.append(
          self.evaluate(random_access.at(shuffled_dataset, i)))
      shuffled_dataset_array_2.append(
          self.evaluate(random_access.at(shuffled_dataset_2, i)))
    self.assertNotEqual(shuffled_dataset_array, shuffled_dataset_array_2)
    self.assertAllEqual(
        sorted(shuffled_dataset_array), sorted(shuffled_dataset_array_2))

if __name__ == "__main__":
  test.main()

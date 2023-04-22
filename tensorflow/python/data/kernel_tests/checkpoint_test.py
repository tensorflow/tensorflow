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
"""Tests for checkpointing tf.data iterators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import scan_ops
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.tracking import util as trackable_utils


# TODO(jsimsa): Add missing test combinations.
class CheckpointTest(test_base.DatasetTestBase, parameterized.TestCase):

  def tearDown(self):
    prefix = self._iterator_checkpoint_prefix()
    pattern = prefix + "*"
    files = gfile.Glob(pattern)
    map(gfile.Remove, files)
    super(CheckpointTest, self).tearDown()

  def _iterator_checkpoint_prefix(self):
    return os.path.join(self.get_temp_dir(), "iterator")

  def _save_op(self, iterator_resource):
    iterator_state_variant = gen_dataset_ops.serialize_iterator(
        iterator_resource)
    save_op = io_ops.write_file(
        self._iterator_checkpoint_prefix(),
        parsing_ops.serialize_tensor(iterator_state_variant))
    return save_op

  def _restore_op(self, iterator_resource):
    iterator_state_variant = parsing_ops.parse_tensor(
        io_ops.read_file(self._iterator_checkpoint_prefix()), dtypes.variant)
    restore_op = gen_dataset_ops.deserialize_iterator(iterator_resource,
                                                      iterator_state_variant)
    return restore_op

  @combinations.generate(test_base.graph_only_combinations())
  def testSaveRestore(self):

    def _build_graph(start, stop):
      iterator = dataset_ops.make_initializable_iterator(
          dataset_ops.Dataset.range(start, stop))
      init_op = iterator.initializer
      get_next = iterator.get_next()
      save_op = self._save_op(iterator._iterator_resource)
      restore_op = self._restore_op(iterator._iterator_resource)
      return init_op, get_next, save_op, restore_op

    # Saving and restoring in different sessions.
    start = 2
    stop = 10
    break_point = 5
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, _ = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

    # Saving and restoring in same session.
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  @combinations.generate(test_base.graph_only_combinations())
  def testInitThenRestore(self):
    # Note: Calling init_op before restore_op is redundant. This test just makes
    # sure we do not fail if restore is called on an already initialized
    # iterator resource.

    def _build_graph(start, stop):
      dataset = dataset_ops.Dataset.range(start, stop)
      iterator = dataset_ops.make_initializable_iterator(dataset)
      init_op = iterator.initializer
      get_next = iterator.get_next()
      save_op = self._save_op(iterator._iterator_resource)
      restore_op = self._restore_op(iterator._iterator_resource)
      return init_op, get_next, save_op, restore_op

    # Saving and restoring in different sessions.
    start = 2
    stop = 10
    break_point = 5
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, _ = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  @combinations.generate(test_base.graph_only_combinations())
  def testMultipleSaves(self):

    def _build_graph(start, stop):
      iterator = dataset_ops.make_initializable_iterator(
          dataset_ops.Dataset.range(start, stop))
      init_op = iterator.initializer
      get_next = iterator.get_next()
      save_op = self._save_op(iterator._iterator_resource)
      restore_op = self._restore_op(iterator._iterator_resource)
      return init_op, get_next, save_op, restore_op

    start = 2
    stop = 10
    break_point1 = 5
    break_point2 = 7

    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, _ = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point1):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point1, break_point2):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    break_point2 = 7
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(start, stop)
      with self.session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point2, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  @combinations.generate(test_base.graph_only_combinations())
  def testSaveRestoreWithRepeat(self):

    def _build_graph(start, stop, num_epochs):
      iterator = dataset_ops.make_initializable_iterator(
          dataset_ops.Dataset.range(start, stop).repeat(num_epochs))
      init_op = iterator.initializer
      get_next = iterator.get_next()
      save_op = self._save_op(iterator._iterator_resource)
      restore_op = self._restore_op(iterator._iterator_resource)
      return init_op, get_next, save_op, restore_op

    start = 2
    stop = 10
    num_epochs = 5
    break_range = 5
    break_epoch = 3
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(
          start, stop, num_epochs)
      with self.session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        # Note: There is no checkpoint saved currently so a NotFoundError is
        # raised.
        with self.assertRaises(errors.NotFoundError):
          sess.run(init_op)
          sess.run(restore_op)
        for _ in range(break_epoch - 1):
          for i in range(start, stop):
            self.assertEqual(i, sess.run(get_next))
        for i in range(start, break_range):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop, num_epochs)
      with self.session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_range, stop):
          self.assertEqual(i, sess.run(get_next))
        for _ in range(break_epoch, num_epochs):
          for i in range(start, stop):
            self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  @combinations.generate(test_base.graph_only_combinations())
  def testSaveRestoreExhaustedIterator(self):

    def _build_graph(start, stop, num_epochs):
      iterator = dataset_ops.make_initializable_iterator(
          dataset_ops.Dataset.range(start, stop).repeat(num_epochs))
      init_op = iterator.initializer
      get_next = iterator.get_next()
      save_op = self._save_op(iterator._iterator_resource)
      restore_op = self._restore_op(iterator._iterator_resource)
      return init_op, get_next, save_op, restore_op

    start = 2
    stop = 10
    num_epochs = 5
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(
          start, stop, num_epochs)
      with self.session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        # Note: There is no checkpoint saved currently so a NotFoundError is
        # raised.
        with self.assertRaises(errors.NotFoundError):
          sess.run(init_op)
          sess.run(restore_op)
        for _ in range(num_epochs):
          for i in range(start, stop):
            self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop, num_epochs)
      with self.session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveRestoreOneShotIterator(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6]).map(
        math_ops.square).batch(2)
    iterator = iter(dataset)
    get_next = iterator.get_next
    checkpoint = trackable_utils.Checkpoint(iterator=iterator)
    self.assertAllEqual([1, 4], get_next())
    save_path = checkpoint.save(checkpoint_prefix)
    self.assertAllEqual([9, 16], get_next())
    self.assertAllEqual([25, 36], get_next())
    checkpoint.restore(save_path).run_restore_ops()
    self.assertAllEqual([9, 16], get_next())
    self.assertAllEqual([25, 36], get_next())
    with self.assertRaises(errors.OutOfRangeError):
      get_next()

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveRestoreMultipleIterator(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    dataset = dataset.map(math_ops.square).batch(2)
    iterator_1 = iter(dataset)
    get_next_1 = iterator_1.get_next
    iterator_2 = iter(dataset)
    get_next_2 = iterator_2.get_next
    dataset_2 = dataset_ops.Dataset.range(10)
    iterator_3 = iter(dataset_2)
    get_next_3 = iterator_3.get_next
    checkpoint = trackable_utils.Checkpoint(
        iterator_1=iterator_1, iterator_2=iterator_2, iterator_3=iterator_3)
    self.assertAllEqual([1, 4], get_next_1())
    self.assertAllEqual(0, get_next_3())
    self.assertAllEqual(1, get_next_3())
    self.assertAllEqual(2, get_next_3())
    save_path = checkpoint.save(checkpoint_prefix)
    self.assertAllEqual([1, 4], get_next_2())
    self.assertAllEqual([9, 16], get_next_2())
    self.assertAllEqual(3, get_next_3())
    checkpoint.restore(save_path).run_restore_ops()
    self.assertAllEqual([9, 16], get_next_1())
    self.assertAllEqual([1, 4], get_next_2())
    self.assertAllEqual(3, get_next_3())

  @combinations.generate(test_base.eager_only_combinations())
  def testRestoreExhaustedIterator(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.range(3)
    iterator = iter(dataset)
    get_next = iterator.get_next
    checkpoint = trackable_utils.Checkpoint(iterator=iterator)
    self.assertAllEqual(0, get_next())
    self.assertAllEqual(1, get_next())
    save_path = checkpoint.save(checkpoint_prefix)
    self.assertAllEqual(2, get_next())
    checkpoint.restore(save_path).run_restore_ops()
    self.assertAllEqual(2, get_next())
    save_path = checkpoint.save(checkpoint_prefix)
    checkpoint.restore(save_path).run_restore_ops()
    with self.assertRaises(errors.OutOfRangeError):
      get_next()

  @combinations.generate(test_base.eager_only_combinations())
  def testRestoreInReconstructedIteratorInitializable(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.range(10)
    iterator = iter(dataset)
    get_next = iterator.get_next
    checkpoint = trackable_utils.Checkpoint(iterator=iterator)
    for i in range(5):
      checkpoint.restore(
          checkpoint_management.latest_checkpoint(
              checkpoint_directory)).initialize_or_restore()
      for j in range(2):
        self.assertEqual(i * 2 + j, self.evaluate(get_next()))
      checkpoint.save(file_prefix=checkpoint_prefix)

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveRestoreReshuffleDataset(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.shuffle(10, reshuffle_each_iteration=True)
    iterator = iter(dataset)
    ckpt = trackable_utils.Checkpoint(
        step=variables.Variable(0), iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, self.get_temp_dir(), max_to_keep=3)

    iter1 = [next(iterator).numpy() for _ in range(5)]

    manager.save()
    iter2 = [next(iterator).numpy() for _ in range(5)]

    ckpt.restore(manager.latest_checkpoint)
    iter3 = [next(iterator).numpy() for _ in range(5)]

    self.assertNotEqual(iter1, iter2)
    self.assertCountEqual(iter2, iter3)

  @combinations.generate(test_base.eager_only_combinations())
  def testSaveRestoreModifiedDataset(self):
    ckpt_dir = self.get_temp_dir()
    dataset = dataset_ops.Dataset.range(10)
    iterator = iter(dataset)
    ckpt = trackable_utils.Checkpoint(iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, ckpt_dir, max_to_keep=3)

    for _ in range(5):
      next(iterator)
    manager.save()

    # Define a different dataset and try to restore into its iterator.
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
    iterator = iter(dataset)
    ckpt = trackable_utils.Checkpoint(iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, ckpt_dir, max_to_keep=3)
    with self.assertRaisesRegex(
        errors.NotFoundError,
        "Make sure the dataset definition has not changed"):
      ckpt.restore(manager.latest_checkpoint)

  def _assertNotCheckpointable(self, dataset):
    iterator = iter(dataset)
    ckpt = trackable_utils.Checkpoint(
        step=variables.Variable(0), iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, self.get_temp_dir(), max_to_keep=3)
    with self.assertRaises(errors.FailedPreconditionError):
      manager.save()

  @staticmethod
  def _statefulInt64Func(_):
    return random_ops.random_uniform((), 0, 1, dtypes.int64)

  @staticmethod
  def _statefulBoolFunc(_):
    return random_ops.random_uniform((), 0, 1, dtypes.int64) < 1

  @staticmethod
  def _statefulDatasetFunc(_):
    x = random_ops.random_uniform((), 0, 1, dtypes.int64)
    return dataset_ops.Dataset.range(x)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulFilterNotCheckpointable(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.filter(self._statefulBoolFunc)
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulFlatMapNotCheckpointable(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.flat_map(self._statefulDatasetFunc)
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulInterleaveNotCheckpointable(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.interleave(self._statefulDatasetFunc)
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulMapNotCheckpointable(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.map(self._statefulBoolFunc)
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulParallelInterleaveNotCheckpointable(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.interleave(
        self._statefulDatasetFunc, num_parallel_calls=2)
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulParallelMapNotCheckpointable(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.map(self._statefulBoolFunc, num_parallel_calls=2)
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulGroupByReducerNotCheckpointable(self):
    stateful_key_func = self._statefulInt64Func
    key_func = lambda _: math_ops.cast(0, dtypes.int64)
    stateful_init_func = self._statefulBoolFunc
    init_func = lambda x: True
    stateful_reduce_func = lambda _, x: self._statefulBoolFunc(x)
    reduce_func = lambda _, x: True
    stateful_finalize_func = self._statefulBoolFunc
    finalize_func = lambda x: True

    test_cases = [
        (stateful_key_func, init_func, reduce_func, finalize_func),
        (key_func, stateful_init_func, reduce_func, finalize_func),
        (key_func, init_func, stateful_reduce_func, finalize_func),
        (key_func, init_func, reduce_func, stateful_finalize_func),
    ]
    for key_func, init_func, reduce_func, finalize_func in test_cases:
      dataset = dataset_ops.Dataset.range(10)
      reducer = grouping.Reducer(init_func, reduce_func, finalize_func)
      dataset = dataset.apply(grouping.group_by_reducer(key_func, reducer))
      self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulGroupByWindowNotCheckpointable(self):
    stateful_key_func = self._statefulInt64Func
    key_func = lambda _: math_ops.cast(0, dtypes.int64)
    stateful_reduce_func = lambda _, x: self._statefulDatasetFunc(x)
    reduce_func = lambda _, x: x
    stateful_window_func = self._statefulInt64Func
    window_func = lambda x: math_ops.cast(0, dtypes.int64)

    test_cases = [
        (stateful_key_func, reduce_func, window_func),
        (key_func, stateful_reduce_func, window_func),
        (key_func, reduce_func, stateful_window_func),
    ]
    for key_func_fn, reduce_func_fn, window_func in test_cases:
      dataset = dataset_ops.Dataset.range(10)
      dataset = dataset.apply(
          grouping.group_by_window(
              key_func_fn, reduce_func_fn, window_size_func=window_func))
      self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulMapAndBatchNotCheckpointable(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.map(self._statefulBoolFunc)
    dataset = dataset.batch(2)
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulExperimentalParallelInterleaveNotCheckpointable(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(
        interleave_ops.parallel_interleave(self._statefulDatasetFunc, 2))
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulScanNotCheckpointable(self):
    dataset = dataset_ops.Dataset.range(10)

    def stateful_scan(state, element):
      return state, self._statefulBoolFunc(element)

    dataset = dataset.apply(scan_ops.scan(0, stateful_scan))
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulTakeWhileNotCheckpointable(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(take_while_ops.take_while(self._statefulBoolFunc))
    self._assertNotCheckpointable(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulExternalPolicy(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.range(4)

    def fn(x):
      return x * x

    dataset = dataset.map(
        lambda x: script_ops.eager_py_func(fn, [x], dtypes.int64))

    options = dataset_ops.Options()
    options.experimental_external_state_policy = (
        distribute_options.ExternalStatePolicy.WARN)
    dataset = dataset.with_options(options)

    iterator = iter(dataset)
    get_next = iterator.get_next
    checkpoint = trackable_utils.Checkpoint(iterator=iterator)
    self.assertEqual(0, get_next().numpy())
    self.assertEqual(1, get_next().numpy())
    save_path = checkpoint.save(checkpoint_prefix)
    self.assertEqual(4, get_next().numpy())
    self.assertEqual(9, get_next().numpy())
    checkpoint.restore(save_path).run_restore_ops()
    self.assertEqual(4, get_next().numpy())
    self.assertEqual(9, get_next().numpy())
    with self.assertRaises(errors.OutOfRangeError):
      get_next()

if __name__ == "__main__":
  test.main()

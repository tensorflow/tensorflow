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
"""Test RangeDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.contrib.data.python.ops import enumerate_ops
from tensorflow.contrib.data.python.ops import iterator_ops as contrib_iterator_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib


class RangeDatasetTest(test.TestCase):

  def tearDown(self):
    # Remove all checkpoint files.
    prefix = self._iterator_checkpoint_prefix()
    pattern = prefix + "*"
    files = gfile.Glob(pattern)
    map(gfile.Remove, files)

  def testStop(self):
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(stop).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={stop: 5})
      for i in range(5):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStartStop(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start,
                                         stop).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 2, stop: 5})
      for i in range(2, 5):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStartStopStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 2, stop: 10, step: 2})
      for i in range(2, 10, 2):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testZeroStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer

    with self.test_session() as sess:
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(init_op, feed_dict={start: 2, stop: 10, step: 0})

  def testNegativeStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 2, stop: 10, step: -1})
      # This for loop is a no-op but will ensure that the implementation is
      # consistent with range if it ever changes.
      for i in range(2, 10, -1):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStopLessThanStart(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start,
                                         stop).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 10, stop: 2})
      # This for loop is a no-op but will ensure that the implementation is
      # consistent with range if it ever changes.
      for i in range(10, 2):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStopLessThanStartWithPositiveStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 10, stop: 2, step: 2})
      # This for loop is a no-op but will ensure that the implementation is
      # consistent with range if it ever changes.
      for i in range(10, 2, 2):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testStopLessThanStartWithNegativeStep(self):
    start = array_ops.placeholder(dtypes.int64, shape=[])
    stop = array_ops.placeholder(dtypes.int64, shape=[])
    step = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = dataset_ops.Dataset.range(start, stop,
                                         step).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op, feed_dict={start: 10, stop: 2, step: -1})
      for i in range(10, 2, -1):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testEnumerateDataset(self):
    components = (["a", "b"], [1, 2], [37.0, 38])
    start = constant_op.constant(20, dtype=dtypes.int64)

    iterator = (dataset_ops.Dataset.from_tensor_slices(components).apply(
        enumerate_ops.enumerate_dataset(start)).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual(dtypes.int64, get_next[0].dtype)
    self.assertEqual((), get_next[0].shape)
    self.assertEqual([tensor_shape.TensorShape([])] * 3,
                     [t.shape for t in get_next[1]])

    with self.test_session() as sess:
      sess.run(init_op)
      self.assertEqual((20, (b"a", 1, 37.0)), sess.run(get_next))
      self.assertEqual((21, (b"b", 2, 38.0)), sess.run(get_next))

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

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

  def testSaveRestore(self):

    def _build_graph(start, stop):
      iterator = dataset_ops.Dataset.range(start,
                                           stop).make_initializable_iterator()
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
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

    # Saving and restoring in same session.
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testSaveRestoreUsingSaverFromMetaGraph(self):

    def _build_graph(start, stop):
      iterator = dataset_ops.Dataset.range(start,
                                           stop).make_initializable_iterator()
      init_op = iterator.initializer
      get_next = iterator.get_next()
      ops.add_to_collection("iterator_ops", init_op)
      ops.add_to_collection("iterator_ops", get_next)
      saveable_obj = contrib_iterator_ops.make_saveable_from_iterator(iterator)
      # Add the SaveableObject to the `SAVEABLE_OBJECTS` collection
      # so that it can be automatically picked up by the Saver.
      ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable_obj)
      saver = saver_lib.Saver()
      return init_op, get_next, saver

    start = 2
    stop = 10
    break_point = 5
    path = self._iterator_checkpoint_prefix()
    meta_filename = path + ".meta"

    # Execute input pipeline for a few steps and save iterator state.
    with ops.Graph().as_default() as g:
      init_op, get_next, saver = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        saver.save(sess, path)

    # Build the saver from the MetaGraph using import_meta_graph and
    # check that the iterator state is restored.
    with ops.Graph().as_default() as g:
      saver = saver_lib.import_meta_graph(meta_filename)
      init_op, get_next = ops.get_collection("iterator_ops")
      with self.test_session(graph=g) as sess:
        saver.restore(sess, saver_lib.latest_checkpoint(self.get_temp_dir()))
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testSaveRestoreUsingBuiltSaver(self):

    def _build_graph(start, stop):
      iterator = dataset_ops.Dataset.range(start,
                                           stop).make_initializable_iterator()
      init_op = iterator.initializer
      get_next = iterator.get_next()
      ops.add_to_collection("iterator_ops", init_op)
      ops.add_to_collection("iterator_ops", get_next)
      # Add the SaveableObject to the `SAVEABLE_OBJECTS` collection
      # so that it can be automatically picked up by the Saver.
      saveable_obj = contrib_iterator_ops.make_saveable_from_iterator(iterator)
      ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable_obj)
      saver = saver_lib.Saver()
      return init_op, get_next, saver

    start = 2
    stop = 10
    stop_new = 15
    break_point = 5
    path = self._iterator_checkpoint_prefix()

    # Execute input pipeline for a few steps and save iterator state.
    with ops.Graph().as_default() as g:
      init_op, get_next, saver = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        saver.save(sess, path)

    # Manually build a modified Graph and Saver instead of importing
    # MetaGraph and verify that original iterator state gets restored.
    with ops.Graph().as_default() as g:
      init_op, get_next, saver = _build_graph(start, stop_new)
      with self.test_session(graph=g) as sess:
        saver.restore(sess, saver_lib.latest_checkpoint(self.get_temp_dir()))
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testSaveRestoreUsingSaverThenInit(self):

    def _build_graph(start, stop):
      iterator = dataset_ops.Dataset.range(start,
                                           stop).make_initializable_iterator()
      init_op = iterator.initializer
      get_next = iterator.get_next()
      ops.add_to_collection("iterator_ops", init_op)
      ops.add_to_collection("iterator_ops", get_next)
      # Add the SaveableObject to the `SAVEABLE_OBJECTS` collection
      # so that it can be automatically picked up by the Saver.
      saveable_obj = contrib_iterator_ops.make_saveable_from_iterator(iterator)
      ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable_obj)
      saver = saver_lib.Saver()
      return init_op, get_next, saver

    start = 2
    stop = 10
    stop_new = 15
    break_point = 5
    path = self._iterator_checkpoint_prefix()

    # Execute input pipeline for a few steps and save iterator state.
    with ops.Graph().as_default() as g:
      init_op, get_next, saver = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        saver.save(sess, path)

    # Restore iterator state call and then call init_op for the iterator and
    # verify that the new iterator hides the restored iterator.
    with ops.Graph().as_default() as g:
      init_op, get_next, saver = _build_graph(start, stop_new)
      with self.test_session(graph=g) as sess:
        saver.restore(sess, saver_lib.latest_checkpoint(self.get_temp_dir()))
        sess.run(init_op)
        for i in range(start, stop_new):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testRestoreWithoutBuildingDatasetGraph(self):

    def _build_graph(start, stop, num_epochs):
      dataset = dataset_ops.Dataset.range(start, stop).repeat(num_epochs)
      iterator = dataset.make_initializable_iterator()
      init_op = iterator.initializer
      get_next = iterator.get_next()
      save_op = self._save_op(iterator._iterator_resource)
      restore_op = self._restore_op(iterator._iterator_resource)
      return init_op, get_next, save_op, restore_op

    # Saving and restoring in different sessions.
    start = 2
    stop = 10
    num_epochs = 5
    break_point = 5
    break_epoch = 3
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, _ = _build_graph(start, stop, num_epochs)
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for _ in range(break_epoch):
          for i in range(start, stop):
            self.assertEqual(i, sess.run(get_next))
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      # Create an empty IteratorResource and restore the Iterator into it.
      output_types = dtypes.int64
      output_shapes = tensor_shape.scalar()
      iterator = iterator_ops.Iterator.from_structure(output_types,
                                                      output_shapes)
      restore_op = self._restore_op(iterator._iterator_resource)
      get_next = iterator.get_next()
      with self.test_session(graph=g) as sess:
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        for _ in range(break_epoch + 1, num_epochs):
          for i in range(start, stop):
            self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testRestoreInModifiedGraph(self):

    def _build_graph(start, stop):
      dataset = dataset_ops.Dataset.range(start, stop)
      iterator = dataset.make_initializable_iterator()
      init_op = iterator.initializer
      get_next = iterator.get_next()
      save_op = self._save_op(iterator._iterator_resource)
      restore_op = self._restore_op(iterator._iterator_resource)
      return init_op, get_next, save_op, restore_op

    # Saving and restoring in different sessions.
    start = 2
    stop = 10
    stop_1 = 8
    break_point = 5
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, _ = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      # Intentionally build a graph with a different value for stop to make sure
      # the original dataset graph is actually getting loaded.
      init_op, get_next, _, restore_op = _build_graph(start, stop_1)
      with self.test_session(graph=g) as sess:
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testInitThenRestore(self):
    # Note: Calling init_op before restore_op is redundant. This test just makes
    # sure we do not fail if restore is called on an already initialized
    # iterator resource.

    def _build_graph(start, stop):
      dataset = dataset_ops.Dataset.range(start, stop)
      iterator = dataset.make_initializable_iterator()
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
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(init_op)
        sess.run(restore_op)
        for i in range(break_point, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testMultipleSaves(self):

    def _build_graph(start, stop):
      iterator = dataset_ops.Dataset.range(start,
                                           stop).make_initializable_iterator()
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
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        for i in range(start, break_point1):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(restore_op)
        for i in range(break_point1, break_point2):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    break_point2 = 7
    with ops.Graph().as_default() as g:
      init_op, get_next, save_op, restore_op = _build_graph(start, stop)
      with self.test_session(graph=g) as sess:
        sess.run(restore_op)
        for i in range(break_point2, stop):
          self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testSaveRestoreWithRepeat(self):

    def _build_graph(start, stop, num_epochs):
      iterator = dataset_ops.Dataset.range(
          start, stop).repeat(num_epochs).make_initializable_iterator()
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
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        # Note: There is no checkpoint saved currently so a NotFoundError is
        # raised.
        with self.assertRaises(errors.NotFoundError):
          sess.run(restore_op)
        for _ in range(break_epoch - 1):
          for i in range(start, stop):
            self.assertEqual(i, sess.run(get_next))
        for i in range(start, break_range):
          self.assertEqual(i, sess.run(get_next))
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop, num_epochs)
      with self.test_session(graph=g) as sess:
        sess.run(restore_op)
        for i in range(break_range, stop):
          self.assertEqual(i, sess.run(get_next))
        for _ in range(break_epoch, num_epochs):
          for i in range(start, stop):
            self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testSaveRestoreExhaustedIterator(self):

    def _build_graph(start, stop, num_epochs):
      iterator = dataset_ops.Dataset.range(
          start, stop).repeat(num_epochs).make_initializable_iterator()
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
      with self.test_session(graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(init_op)
        # Note: There is no checkpoint saved currently so a NotFoundError is
        # raised.
        with self.assertRaises(errors.NotFoundError):
          sess.run(restore_op)
        for _ in range(num_epochs):
          for i in range(start, stop):
            self.assertEqual(i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)
        sess.run(save_op)

    with ops.Graph().as_default() as g:
      init_op, get_next, _, restore_op = _build_graph(start, stop, num_epochs)
      with self.test_session(graph=g) as sess:
        sess.run(restore_op)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)


if __name__ == "__main__":
  test.main()

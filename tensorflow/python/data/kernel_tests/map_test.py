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
"""Tests for `tf.data.Dataset.map()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import threading
import warnings

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


def _make_coordinated_sloppy_dataset(num_elements, num_parallel_calls):
  """Produces a dataset iterator and events to control the order of elements.

  Args:
    num_elements: the number of input elements
    num_parallel_calls: the degree of map parallelism

  Returns:
    A dataset iterator (represented as `get_next` op) and events that can be
    used to control the order of output elements.
  """

  # Set up threading events used to sequence when items are produced that
  # are subsequently interleaved. These events allow us to deterministically
  # simulate slowdowns and force sloppiness.
  coordination_events = {i: threading.Event() for i in range(num_elements)}

  def map_py_fn(x):
    coordination_events[x].wait()
    coordination_events[x].clear()
    return x * x

  def map_fn(x):
    return script_ops.py_func(map_py_fn, [x], x.dtype)

  options = dataset_ops.Options()
  options.experimental_deterministic = False
  dataset = dataset_ops.Dataset.range(num_elements).map(
      map_fn, num_parallel_calls).with_options(options)
  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()
  return next_element, coordination_events


class MapDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  def _buildMapDataset(self, components, count):
    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)
    return (dataset_ops.Dataset.from_tensor_slices(components).map(_map_fn)
            .repeat(count))

  def testMapDataset(self):
    """Test an dataset that maps a TF function across its input elements."""
    # The pipeline is TensorSliceDataset -> MapDataset(square_3) ->
    # RepeatDataset(count).
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))
    count = array_ops.placeholder(dtypes.int64, shape=[])

    dataset = self._buildMapDataset(components, count)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([c.shape[1:] for c in components],
                     [t.shape for t in get_next])

    with self.cached_session() as sess:
      # Test single-threaded access to the iterator.
      sess.run(init_op, feed_dict={count: 14})
      for _ in range(14):
        for i in range(7):
          result = sess.run(get_next)
          for component, result_component in zip(components, result):
            self.assertAllEqual(component[i]**2, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Test multi-threaded access to the same iterator.
      sess.run(init_op, feed_dict={count: 18})
      results = []
      def iterator_thread():
        while True:
          try:
            results.append(sess.run(get_next))
          except errors.OutOfRangeError:
            return
      threads = [self.checkedThread(target=iterator_thread) for _ in range(8)]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      # `results` will contain the same elements components**2
      # repeated 18 times, but in a non-deterministic order. Sort the
      # results, and assert that each element of components**2 is
      # produced 18 times.
      results.sort(key=lambda x: x[0])
      for i in range(7):
        for j in range(18):
          for component, result_component in zip(components,
                                                 results[i * 18 + j]):
            self.assertAllEqual(component[i]**2, result_component)

  def _buildParallelMapDataset(self, components, count, num_parallel_calls,
                               output_buffer_size):
    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)
    return (dataset_ops.Dataset.from_tensor_slices(components)
            .map(_map_fn, num_parallel_calls=num_parallel_calls)
            .prefetch(output_buffer_size)
            .repeat(count))

  def testParallelMapDataset(self):
    """Test an dataset that maps a TF function across its input elements."""
    # The pipeline is TensorSliceDataset -> ParallelMapDataset(square_3) ->
    # RepeatDataset(count).
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))
    count = array_ops.placeholder(dtypes.int64, shape=[])
    num_parallel_calls = array_ops.placeholder(dtypes.int32, shape=[])
    output_buffer_size = array_ops.placeholder(dtypes.int64, shape=[])

    dataset = self._buildParallelMapDataset(
        components, count, num_parallel_calls, output_buffer_size)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([c.shape[1:] for c in components],
                     [t.shape for t in get_next])

    with self.cached_session() as sess:

      def do_test(num_parallel_calls_val, output_buffer_size_val):
        # Test single-threaded access to the iterator.
        sess.run(init_op, feed_dict={
            count: 14,
            num_parallel_calls: num_parallel_calls_val,
            output_buffer_size: output_buffer_size_val})
        for _ in range(14):
          for i in range(7):
            result = sess.run(get_next)
            for component, result_component in zip(components, result):
              self.assertAllEqual(component[i]**2, result_component)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

        # Test multi-threaded access to the same iterator.
        sess.run(init_op, feed_dict={
            count: 18,
            num_parallel_calls: num_parallel_calls_val,
            output_buffer_size: output_buffer_size_val})
        results = []
        def iterator_thread():
          while True:
            try:
              results.append(sess.run(get_next))
            except errors.OutOfRangeError:
              return
        threads = [self.checkedThread(target=iterator_thread)
                   for _ in range(64)]
        for t in threads:
          t.start()
        for t in threads:
          t.join()

        # `results` will contain the same elements components**2
        # repeated 18 times, but in a non-deterministic order. Sort the
        # results, and assert that each element of components**2 is
        # produced 18 times.
        results.sort(key=lambda x: x[0])
        for i in range(7):
          for j in range(18):
            for component, result_component in zip(components,
                                                   results[i * 18 + j]):
              self.assertAllEqual(component[i]**2, result_component)

      for num_parallel_calls_val, output_buffer_size_val in [
          (1, 1), (1, 2), (2, 2), (2, 4), (8, 8), (8, 16)]:
        do_test(num_parallel_calls_val, output_buffer_size_val)

  def testImplicitDisposeParallelMapDataset(self):
    # Tests whether a parallel map dataset will be cleaned up correctly when
    # the pipeline does not run it until exhaustion.
    # The pipeline is TensorSliceDataset -> MapDataset(square_3) ->
    # RepeatDataset(1000).
    components = (np.arange(1000),
                  np.array([[1, 2, 3]]) * np.arange(1000)[:, np.newaxis],
                  np.array(37.0) * np.arange(1000))

    dataset = self._buildParallelMapDataset(components, 1000, 100, 100)
    # NOTE(mrry): Also test that the prefetching thread is cancelled correctly.
    dataset = dataset.prefetch(100)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(3):
        sess.run(get_next)

  def testParallelMapUnspecifiedOutputSize(self):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = (dataset_ops.Dataset.from_tensor_slices(components)
               .map(lambda x: array_ops.check_numerics(x, "message"),
                    num_parallel_calls=2))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(3):
        sess.run(get_next)

  def testParallelMapError(self):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = (dataset_ops.Dataset.from_tensor_slices(components)
               .map(lambda x: array_ops.check_numerics(x, "message"),
                    num_parallel_calls=2))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(3):
        sess.run(get_next)
      # The 4th element is NaN, so `array_ops.check_numerics()` should fail.
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(get_next)
      sess.run(get_next)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testPrefetchError(self):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = (dataset_ops.Dataset.from_tensor_slices(components)
               .map(lambda x: array_ops.check_numerics(x, "message"))
               .prefetch(2))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(3):
        sess.run(get_next)
      # The 4th element is NaN, so `array_ops.check_numerics()` should fail.
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(get_next)
      sess.run(get_next)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testCaptureIterator(self):

    def _build_ds(iterator):

      def _map_fn(x):
        get_next = iterator.get_next()
        return x * get_next

      return dataset_ops.Dataset.range(10).map(_map_fn)

    def _build_graph():
      captured_iterator = dataset_ops.Dataset.range(
          10).make_initializable_iterator()
      ds = _build_ds(captured_iterator)
      iterator = ds.make_initializable_iterator()
      init_op = iterator.initializer
      get_next = iterator.get_next()
      return captured_iterator.initializer, init_op, get_next

    with ops.Graph().as_default() as g:
      captured_init_op, init_op, get_next = _build_graph()
      with self.session(graph=g) as sess:
        sess.run(captured_init_op)
        sess.run(init_op)
        for i in range(10):
          self.assertEqual(i * i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testCaptureHashTable(self):
    # NOTE(mrry): We must use the V2 variants of `HashTable`
    # etc. because these produce a `tf.resource`-typed output that is
    # compatible with the in-graph function implementation.
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.HashTable(
        lookup_ops.KeyValueTensorInitializer(keys, values), default_val)

    input_sentences = dataset_ops.Dataset.from_tensor_slices(
        ["brain brain tank salad surgery", "surgery brain"])

    iterator = (input_sentences
                .map(lambda x: string_ops.string_split([x]).values)
                .map(table.lookup)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(table.initializer)
      sess.run(init_op)
      sess.run(get_next)
      sess.run(get_next)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testCaptureQueue(self):
    elements = np.random.randint(100, size=[200])
    queue = data_flow_ops.FIFOQueue(200, dtypes.int64, shapes=[])
    enqueue_op = queue.enqueue_many(elements)
    close_op = queue.close()
    iterator = (dataset_ops.Dataset.from_tensors(0).repeat(-1)
                .map(lambda _: queue.dequeue()).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(enqueue_op)
      sess.run(close_op)
      sess.run(init_op)
      for element in elements:
        self.assertEqual(element, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testCaptureSameResourceMultipleTimes(self):
    elements = np.random.randint(100, size=[200])
    queue = data_flow_ops.FIFOQueue(
        200, dtypes.int64, shapes=[], shared_name="shared_queue")
    queue_2 = data_flow_ops.FIFOQueue(
        200, dtypes.int64, shapes=[], shared_name="shared_queue")

    enqueue_op = queue.enqueue_many(elements)
    close_op = queue.close()

    iterator = (dataset_ops.Dataset.from_tensors(0).repeat(-1)
                .map(lambda _: (queue.dequeue(), queue_2.dequeue()))
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(enqueue_op)
      sess.run(close_op)
      sess.run(init_op)
      for i in range(100):
        self.assertEqual(sorted([elements[i * 2], elements[i * 2 + 1]]),
                         sorted(sess.run(get_next)))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testCaptureVariable(self):
    counter_var = variable_scope.get_variable(
        "counter", (), dtypes.int32, use_resource=True)
    iterator = (dataset_ops.Dataset.from_tensors(0).repeat(10)
                .map(lambda _: counter_var.assign_add(1))
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(counter_var.initializer)
      sess.run(init_op)
      for i in range(10):
        self.assertEqual(i, sess.run(counter_var))
        self.assertEqual(i + 1, sess.run(get_next))
      self.assertEqual(10, sess.run(counter_var))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)
      self.assertEqual(10, sess.run(counter_var))

  def testCaptureUninitializedVariableError(self):
    counter_var = variable_scope.get_variable(
        "counter", (), dtypes.int32, use_resource=True)
    iterator = (dataset_ops.Dataset.from_tensors(0).repeat(10)
                .map(lambda _: counter_var.assign_add(1))
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      with self.assertRaises(errors.NotFoundError):
        sess.run(get_next)

  def testSeededStatefulOperatorIsProperlyStateful(self):
    iterator = (dataset_ops.Dataset.from_tensors(0).repeat(10)
                .map(lambda _: random_ops.random_uniform((), seed=11)).batch(2)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      random_values = []
      with self.assertRaises(errors.OutOfRangeError):
        while True:
          random_values.extend(sess.run(get_next))
      self.assertEqual(10, len(random_values))
      self.assertGreater(np.abs(np.diff(random_values)).max(), 1e-6)
      sess.run(init_op)
      random_values_2 = []
      with self.assertRaises(errors.OutOfRangeError):
        while True:
          random_values_2.extend(sess.run(get_next))

      # Randomness is repeatable given same seed
      self.assertAllClose(random_values, random_values_2)

  def testStatefulMapKeepsStateAcrossIterators(self):
    iterator = (dataset_ops.Dataset.from_tensors(0).repeat(10)
                .map(lambda _: random_ops.random_uniform((), seed=11))
                .repeat(1000)
                .batch(10)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      random_values = sess.run(get_next)

      # Assert that one of the next 99 batches yielded by the iterator is
      # different from the first.
      i = 0
      while i < 99:
        if np.any(random_values != sess.run(get_next)):
          break
        i += 1
      self.assertLess(i, 99)

  def testStatefulOperationInShortCircuit(self):
    counter_var = variable_scope.get_variable(
        "counter", (), dtypes.int32, use_resource=True)

    def increment_fn(x):
      counter_var.assign_add(1)
      return x

    iterator = (dataset_ops.Dataset.range(10)
                .map(increment_fn)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(counter_var.initializer)
      sess.run(init_op)
      for i in range(10):
        self.assertEqual(i, sess.run(counter_var))
        self.assertEqual(i, sess.run(get_next))
      self.assertEqual(10, sess.run(counter_var))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)
      self.assertEqual(10, sess.run(counter_var))

  def testMapDict(self):
    iterator = (dataset_ops.Dataset.range(10)
                .map(lambda x: {"foo": x * 2, "bar": x ** 2})
                .map(lambda d: d["foo"] + d["bar"])
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(10):
        self.assertEqual(i * 2 + i**2, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testMapNamedtuple(self, count=10):
    # construct dataset of tuples
    labels = dataset_ops.Dataset.range(count)
    images = labels.map(lambda l: -l)
    dataset_tuple = dataset_ops.Dataset.zip((labels, images))

    # convert dataset of tuples to dataset of namedtuples
    example = namedtuple("Example", ["label", "image"])
    dataset_namedtuple = dataset_tuple.map(example)

    def preprocess_tuple(label, image):
      image = 2 * image
      return label, image

    def preprocess_namedtuple(example):
      return example._replace(image=2 * example.image)

    # preprocess both datasets
    dataset_tuple = dataset_tuple.map(preprocess_tuple)
    dataset_namedtuple = dataset_namedtuple.map(preprocess_namedtuple)

    next_tuple = dataset_tuple.make_one_shot_iterator().get_next()
    next_namedtuple = dataset_namedtuple.make_one_shot_iterator().get_next()

    # make sure both datasets contain the same data
    with self.cached_session() as sess:
      for i in range(count):
        tuple_, namedtuple_ = sess.run([next_tuple, next_namedtuple])
        self.assertEqual(tuple_, namedtuple_)
        self.assertEqual(tuple_, (i, -2 * i))

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_namedtuple)

  def testUseStepContainerInMap(self):
    row = np.arange(6)
    iterator = (
        dataset_ops.Dataset.from_tensors(row)
        .map(lambda elems: functional_ops.map_fn(lambda x: x * x, elems))
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      self.assertAllEqual(row**2, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testCaseAndCondInMap(self):

    def control_map_fn(x, y):

      def multiply():
        return x * 2

      def divide():
        return x // 2

      def defaults_two():
        return control_flow_ops.cond(
            math_ops.equal(math_ops.mod(x, 2), 0),
            multiply,
            divide,
            name="cond_mult")

      pred_fn_pairs = {
          math_ops.logical_or(math_ops.equal(y, 2), math_ops.equal(y, 3)):
              defaults_two,
      }

      return control_flow_ops.case(
          pred_fn_pairs, default=multiply, exclusive=True)

    def build_dataset(row, num):
      iterator = (
          dataset_ops.Dataset.from_tensor_slices(row).map(
              lambda x: control_map_fn(x, num)).make_initializable_iterator())
      init_op = iterator.initializer
      get_next = iterator.get_next()
      return init_op, get_next

    with self.cached_session() as sess:
      row = np.arange(6)
      for num in [2, 3, 4]:
        init_op, get_next = build_dataset(row, num)
        sess.run(init_op)
        for i in range(6):
          self.assertEqual(
              (i // 2 if i % 2 else i * 2) if (num == 2 or num == 3) else i * 2,
              sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testCaseInWhileInMap(self):

    def control_map_fn(x, y):

      def multiply():
        return x * 2

      def divide():
        return x // 2

      pred_fn_pairs = {
          math_ops.logical_or(math_ops.equal(y, 2), math_ops.equal(y, 3)):
              divide,
      }

      return control_flow_ops.case(
          pred_fn_pairs, default=multiply, exclusive=True)

    def build_dataset(row, num):
      # pylint: disable=g-long-lambda
      iterator = (
          dataset_ops.Dataset.from_tensors(row).map(
              lambda elems: functional_ops.map_fn(lambda x:
                                                  control_map_fn(x, num), elems)
              ).make_initializable_iterator())
      init_op = iterator.initializer
      get_next = iterator.get_next()
      return init_op, get_next

    with self.cached_session() as sess:
      row = np.arange(6)
      for num in [2, 3, 4]:
        init_op, get_next = build_dataset(row, num)
        sess.run(init_op)
        self.assertAllEqual(
            [x // 2 if (num == 2 or num == 3) else x * 2 for x in row],
            sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testCaseAndCondInWhileInMap(self):

    def control_map_fn(x, y):

      def multiply():
        return x * 2

      def divide():
        return x // 2

      def defaults_two():
        return control_flow_ops.cond(
            math_ops.equal(math_ops.mod(x, 2), 0),
            multiply,
            divide,
            name="cond_mult")

      pred_fn_pairs = {
          math_ops.logical_or(math_ops.equal(y, 2), math_ops.equal(y, 3)):
              defaults_two,
      }

      return control_flow_ops.case(
          pred_fn_pairs, default=multiply, exclusive=True)

    row = np.arange(6)
    num = 2
    # pylint: disable=g-long-lambda
    iterator = (
        dataset_ops.Dataset.from_tensors(row).map(
            lambda elems: functional_ops.map_fn(lambda x:
                                                control_map_fn(x, num), elems)
            ).make_initializable_iterator())
    # pylint: enable=g-long-lambda
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      self.assertAllEqual([(x // 2 if x % 2 else x * 2) if
                           (num == 2 or num == 3) else x * 2 for x in row],
                          sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testPrefetch(self):
    # We will use this event to test that `_map_py_func()` has been
    # invoked a certain number of times (6 times, to be exact) after
    # consuming fewer elements from the iterator.
    ev = threading.Event()

    set_event_during_invocation = 5

    def _map_py_func(x):
      if x == set_event_during_invocation:
        ev.set()
      return x * x

    def _map_fn(x):
      return script_ops.py_func(_map_py_func, [x], x.dtype)

    buffer_size_placeholder = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = (
        dataset_ops.Dataset.range(100)
        .map(_map_fn)
        .prefetch(buffer_size_placeholder)
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      # Simple test that prefetch yields the expected values in the
      # expected order.
      for buffer_size in [1, 10, 100, 1000]:
        sess.run(init_op, feed_dict={buffer_size_placeholder: buffer_size})
        for i in range(100):
          self.assertEqual(i * i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

      # We can indirectly observe that varying the buffer size has the
      # intended effect by observing when `ev` is set (on the 6th
      # invocation of `_map_py_func()`).
      # NOTE(mrry): We do not test with `buffer_size ==
      # set_event_during_invocation`, because we must consume at least
      # one element to start the prefetching.
      for buffer_size in range(1, set_event_during_invocation):
        event_will_be_set_after_consuming = (
            set_event_during_invocation - buffer_size + 1)

        ev.clear()
        sess.run(init_op, feed_dict={buffer_size_placeholder: buffer_size})
        for i in range(event_will_be_set_after_consuming):
          self.assertFalse(ev.is_set())
          self.assertEqual(i * i, sess.run(get_next))
        ev.wait()
        for i in range(event_will_be_set_after_consuming, 100):
          self.assertEqual(i * i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testReturnList(self):
    iterator = (dataset_ops.Dataset.range(10)
                .map(lambda x: [x, constant_op.constant(37.0)])
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(10):
        self.assertEqual((i, 37.0), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testMultiOutputPyFunc(self):
    # The `tf.py_func()` op returns a list of tensors for its outputs.
    def _map_fn(x_tensor):
      def _map_py_func(x):
        return x, np.array(37.0, dtype=np.float64)
      return script_ops.py_func(
          _map_py_func, [x_tensor], [dtypes.int64, dtypes.float64])

    iterator = (dataset_ops.Dataset.range(10)
                .map(_map_fn)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(10):
        self.assertEqual((i, 37.0), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testSparse(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    iterator = (dataset_ops.Dataset.range(10)
                .map(_sparse)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(10):
        actual = sess.run(get_next)
        self.assertIsInstance(actual, sparse_tensor.SparseTensorValue)
        self.assertSparseValuesEqual(actual, _sparse(i))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testSparseChain(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    def _check(i):
      self.assertTrue(sparse_tensor.is_sparse(i))
      return sparse_ops.sparse_concat(0, [i, i])

    iterator = (
        dataset_ops.Dataset.range(10).map(_sparse).map(_check)
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(10):
        actual = sess.run(get_next)
        self.assertIsInstance(actual, sparse_tensor.SparseTensorValue)
        self.assertSparseValuesEqual(actual, _check(_sparse(i)).eval())
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testParallelMapOutOfRangeError(self):
    def raising_py_func(i):
      if i == 100:
        raise StopIteration()
      else:
        return i

    iterator = (
        dataset_ops.Dataset.range(105)
        .map(lambda x: script_ops.py_func(raising_py_func, [x], dtypes.int64),
             num_parallel_calls=2)
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(100):
        self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testConstantOutput(self):
    iterator = (
        dataset_ops.Dataset.range(10).map(lambda x: [x, "hello", 10])
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(10):
        self.assertEqual((i, b"hello", 10), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testWarnOnLookupTable(self):
    def collecting_function(x):
      _ = lookup_ops.HashTable(
          lookup_ops.KeyValueTensorInitializer([], []), 0.0, name="t1")
      return x

    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
      _ = dataset_ops.Dataset.range(10).map(collecting_function)
    # NOTE(mrry): Python 3 prints other warnings in addition to the one we are
    # testing, so we search for the expected warning.
    self.assertGreaterEqual(len(w), 1)
    found_warning = False
    for warning in w:
      if ("Creating lookup tables inside a function passed to Dataset.map() is "
          "not supported." in str(warning)):
        found_warning = True
        break
    self.assertTrue(found_warning)

  def testNestedDatasetMap(self):
    # TODO(b/110122868): When iterators can yield a `tf.data.Dataset`, remove
    # the `get_single_element()` call.
    dataset = dataset_ops.Dataset.from_tensors([1.0, 2.0, 3.0]).map(
        dataset_ops.Dataset.from_tensor_slices).map(
            lambda ds: ds.batch(3)).flat_map(lambda x: x)

    self.assertDatasetProduces(dataset, [[1.0, 2.0, 3.0]])

  def testReturnValueError(self):
    dataset = dataset_ops.Dataset.from_tensors([1.0, 2.0, 3.0])
    with self.assertRaisesRegexp(
        TypeError, r"Unsupported return value from function passed to "
        r"Dataset.map\(\): None."):
      _ = dataset.map(lambda x: None)

  def testBrokenFunctionErrorOnInitialization(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([1.0, 2.0, 3.0])

    def broken_function(_):
      """A function deliberately designed to fail on instantiation."""
      value = []
      tensor_value = attr_value_pb2.AttrValue()
      tensor_value.tensor.CopyFrom(
          tensor_util.make_tensor_proto(
              value, dtype=dtypes.float32, shape=[0], verify_shape=False))
      dtype_value = attr_value_pb2.AttrValue(type=dtypes.int32.as_datatype_enum)

      # Create a "Const" op with a `tf.float32` value and a `tf.int32` type
      # attr.
      const_tensor = ops.get_default_graph().create_op(
          "Const", [], [dtypes.int32],
          attrs={
              "value": tensor_value,
              "dtype": dtype_value
          },
          name="BrokenConst").outputs[0]
      return const_tensor

    dataset = dataset.map(broken_function)
    iterator = dataset.make_initializable_iterator()

    with self.cached_session() as sess:
      with self.assertRaisesRegexp(errors.InvalidArgumentError, "BrokenConst"):
        sess.run(iterator.initializer)

# pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      ("Map", lambda dataset, func:
       dataset_ops.MapDataset(dataset, func, use_inter_op_parallelism=False)),
      ("ParallelMap", lambda dataset, func:
       dataset_ops.ParallelMapDataset(dataset, func, num_parallel_calls=1,
                                      use_inter_op_parallelism=False)),
  )
  def testNoInterOpParallelism(self, make_dataset_fn):
    dataset = dataset_ops.Dataset.from_tensors(0)

    def _get_tid():
      return np.int64(threading.current_thread().ident)

    def _map_fn(_):
      tids = []
      for _ in range(10):
        tids.append(script_ops.py_func(_get_tid, [], dtypes.int64))
      return tids

    dataset = make_dataset_fn(dataset, _map_fn)
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      tids = sess.run(get_next)
      self.assertTrue(all(tids[0] == tid for tid in tids))
# pylint: enable=g-long-lambda

  @parameterized.named_parameters(
      ("SequentialIdentity", None, lambda x: x, None),
      ("SequentialReplicate", None, lambda x: (x, x), None),
      ("SequentialSwap", (None, None), lambda x, y: (y, x), None),
      ("SequentialProject", (None, None), lambda x, y: x, None),
      ("ParallelIdentity", None, lambda x: x, 10),
      ("ParallelReplicate", None, lambda x: (x, x), 10),
      ("ParallelSwap", (None, None), lambda x, y: (y, x), 10),
      ("ParallelProject", (None, None), lambda x, y: x, 10),
  )
  def testShortCircuit(self, structure, map_fn, num_parallel_calls):
    dataset = self.structuredDataset(structure).repeat().map(
        map_fn, num_parallel_calls=num_parallel_calls)
    get_next = dataset.make_one_shot_iterator().get_next()

    with self.cached_session() as sess:
      if isinstance(structure, tuple):
        expected = map_fn(*sess.run(self.structuredElement(structure)))
      else:
        expected = map_fn(sess.run(self.structuredElement(structure)))
      self.assertEqual(expected, sess.run(get_next))

  @parameterized.named_parameters(
      ("Sequential", None),
      ("Parallel", 10),
  )
  def testShortCircuitCapturedInput(self, num_parallel_calls):
    captured_t = array_ops.placeholder(dtypes.int64, shape=[])
    dataset = self.structuredDataset(None).repeat().map(
        lambda x: captured_t, num_parallel_calls=num_parallel_calls)
    iterator = dataset.make_initializable_iterator()
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(iterator.initializer, feed_dict={captured_t: 42})
      self.assertEqual(42, sess.run(get_next))

  @parameterized.named_parameters(
      ("1", 1, 1),
      ("2", 10, 1),
      ("3", 10, 10),
      ("4", 100, 1),
      ("5", 100, 10),
      ("6", 100, 100),
  )
  def testSloppyInterleaveInOrder(self, num_elements, num_parallel_calls):
    get_next, coordination_events = _make_coordinated_sloppy_dataset(
        num_elements, num_parallel_calls)
    config = config_pb2.ConfigProto(
        inter_op_parallelism_threads=num_parallel_calls + 1,
        use_per_session_threads=True)
    with self.cached_session(config=config) as sess:
      for i in range(num_elements):
        coordination_events[i].set()
        self.assertEqual(i * i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  @parameterized.named_parameters(
      ("1", 10, 10),
      ("2", 100, 10),
      ("3", 100, 100),
  )
  def testSloppyInterleaveOutOfOrder(self, num_elements, num_parallel_calls):
    get_next, coordination_events = _make_coordinated_sloppy_dataset(
        num_elements, num_parallel_calls)
    config = config_pb2.ConfigProto(
        inter_op_parallelism_threads=num_parallel_calls + 1,
        use_per_session_threads=True)
    with self.cached_session(config=config) as sess:
      elements = [x for x in range(num_elements)]
      for i in [1, 4, 7]:
        elements[i], elements[i + 1] = elements[i + 1], elements[i]

      for element in elements:
        coordination_events[element].set()
        self.assertEqual(element * element, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


if __name__ == "__main__":
  test.main()

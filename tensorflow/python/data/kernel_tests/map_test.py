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
import collections
import functools
import threading
import time
import warnings

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_sanitizers
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test

try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None


def _test_combinations_with_mode_v1(mode):

  def new_map_fn(dataset, *args, **kwargs):
    return dataset.map(*args, **kwargs)

  def legacy_map_fn(dataset, *args, **kwargs):
    return dataset.map_with_legacy_function(*args, **kwargs)

  new_map_combinations = combinations.combine(
      tf_api_version=1,
      mode=mode,
      apply_map=combinations.NamedObject("map_fn", new_map_fn))

  legacy_map_combinations = combinations.combine(
      tf_api_version=1,
      mode=mode,
      apply_map=combinations.NamedObject("legacy_map_fn", legacy_map_fn))

  return new_map_combinations + legacy_map_combinations


def _test_combinations_with_mode_v2(mode):

  def new_map_fn(dataset, *args, **kwargs):
    return dataset.map(*args, **kwargs)

  return combinations.combine(
      tf_api_version=2,
      mode=mode,
      apply_map=combinations.NamedObject("map_fn", new_map_fn))


def _test_combinations_with_mode(mode):
  return _test_combinations_with_mode_v1(
      mode) + _test_combinations_with_mode_v2(mode)


def _test_combinations():
  return _test_combinations_with_mode("eager") + _test_combinations_with_mode(
      "graph")


def _short_circuit_test_cases():
  cases = [
      ("Identity", None, lambda x: x),
      ("Replicate", None, lambda x: (x, x)),
      ("Swap", (None, None), lambda x, y: (y, x)),
      ("Project", (None, None), lambda x, y: x)
  ]

  def reduce_fn(x, y):
    name, structure, fn = y
    return x + combinations.combine(
        structure=structure, fn=combinations.NamedObject(name, fn))

  return functools.reduce(reduce_fn, cases, [])


class Foo(object):
  """Dummy class used for invalid return value tests."""

  def __init__(self):
    pass


class MapTest(test_base.DatasetTestBase, parameterized.TestCase):

  def _map_dataset_factory(self, components, apply_map, count):

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = apply_map(dataset, _map_fn).repeat(count)
    self.assertEqual(
        [c.shape[1:] for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    return dataset

  @combinations.generate(_test_combinations())
  def testMapDataset(self, apply_map):
    """Test an dataset that maps a TF function across its input elements."""
    # The pipeline is TensorSliceDataset -> MapDataset(square_3) ->
    # RepeatDataset(count).
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))

    # Test single-threaded access to the iterator.
    get_next = self.getNext(
        self._map_dataset_factory(components, apply_map, count=14))
    for _ in range(14):
      for i in range(7):
        result = self.evaluate(get_next())
        for component, result_component in zip(components, result):
          self.assertAllEqual(component[i]**2, result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # TODO(b/117581999): add eager coverage
  @combinations.generate(_test_combinations_with_mode("graph"))
  def testMapDatasetMultiThreaded(self, apply_map):
    # Test multi-threaded access to the same iterator.
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))
    get_next = self.getNext(
        self._map_dataset_factory(components, apply_map, count=18))
    results = []
    with self.cached_session() as sess:
      def iterator_thread():
        while True:
          try:
            results.append(sess.run(get_next()))
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

  def _parallel_map_dataset_factory(self, components, apply_map, count,
                                    num_parallel_calls, buffer_size):

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = apply_map(dataset, _map_fn, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(buffer_size).repeat(count)

    self.assertEqual(
        [c.shape[1:] for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    return dataset

  @combinations.generate(
      combinations.times(
          _test_combinations(),
          combinations.combine(num_parallel_calls=1, buffer_size=1) +
          combinations.combine(num_parallel_calls=1, buffer_size=2) +
          combinations.combine(num_parallel_calls=2, buffer_size=2) +
          combinations.combine(num_parallel_calls=2, buffer_size=4) +
          combinations.combine(num_parallel_calls=8, buffer_size=8) +
          combinations.combine(num_parallel_calls=8, buffer_size=16)))
  def testParallelMapDataset(self, apply_map, num_parallel_calls, buffer_size):
    """Test an dataset that maps a TF function across its input elements."""

    # The pipeline is TensorSliceDataset -> ParallelMapDataset(square_3) ->
    # RepeatDataset(count).
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))
    # Test single-threaded access to the iterator.
    get_next = self.getNext(
        self._parallel_map_dataset_factory(components, apply_map, 14,
                                           num_parallel_calls, buffer_size))
    for _ in range(14):
      for i in range(7):
        result = self.evaluate(get_next())
        for component, result_component in zip(components, result):
          self.assertAllEqual(component[i]**2, result_component)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # TODO(b/117581999): add eager coverage
  @combinations.generate(
      combinations.times(
          _test_combinations_with_mode("graph"),
          combinations.combine(num_parallel_calls=1, buffer_size=1) +
          combinations.combine(num_parallel_calls=1, buffer_size=2) +
          combinations.combine(num_parallel_calls=2, buffer_size=2) +
          combinations.combine(num_parallel_calls=2, buffer_size=4) +
          combinations.combine(num_parallel_calls=8, buffer_size=8) +
          combinations.combine(num_parallel_calls=8, buffer_size=16)))
  def testParallelMapDatasetMultiThreaded(self, apply_map, num_parallel_calls,
                                          buffer_size):

    # Test multi-threaded access to the same iterator.
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))
    get_next = self.getNext(
        self._parallel_map_dataset_factory(components, apply_map, 18,
                                           num_parallel_calls, buffer_size))
    results = []
    with self.cached_session() as sess:

      def iterator_thread():
        while True:
          try:
            results.append(sess.run(get_next()))
          except errors.OutOfRangeError:
            return

      threads = [self.checkedThread(target=iterator_thread) for _ in range(64)]
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

  @combinations.generate(_test_combinations())
  def testImplicitDisposeParallelMapDataset(self, apply_map):
    # Tests whether a parallel map dataset will be cleaned up correctly when
    # the pipeline does not run it until exhaustion.
    # The pipeline is TensorSliceDataset -> MapDataset(square_3) ->
    # RepeatDataset(1000).
    components = (np.arange(1000),
                  np.array([[1, 2, 3]]) * np.arange(1000)[:, np.newaxis],
                  np.array(37.0) * np.arange(1000))

    dataset = self._parallel_map_dataset_factory(components, apply_map, 1000,
                                                 100, 100)
    # NOTE(mrry): Also test that the prefetching thread is cancelled correctly.
    dataset = dataset.prefetch(100)
    get_next = self.getNext(dataset)

    for _ in range(3):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testParallelMapUnspecifiedOutputSize(self, apply_map):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = apply_map(
        dataset,
        lambda x: array_ops.check_numerics(x, "message"),
        num_parallel_calls=2)
    get_next = self.getNext(dataset)

    for _ in range(3):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testParallelMapError(self, apply_map):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = apply_map(
        dataset,
        lambda x: array_ops.check_numerics(x, "message"),
        num_parallel_calls=2)
    get_next = self.getNext(dataset)

    for _ in range(3):
      self.evaluate(get_next())
    # The 4th element is NaN, so `array_ops.check_numerics()` should fail.
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())
    self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testPrefetchError(self, apply_map):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = apply_map(
        dataset, lambda x: array_ops.check_numerics(x, "message")).prefetch(2)
    get_next = self.getNext(dataset)

    for _ in range(3):
      self.evaluate(get_next())
    # The 4th element is NaN, so `array_ops.check_numerics()` should fail.
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())
    self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testCaptureIterator(self, apply_map):

    def _build_ds(iterator):

      def _map_fn(x):
        get_next = iterator.get_next()
        return x * get_next

      return apply_map(dataset_ops.Dataset.range(10), _map_fn)

    def _build_graph():
      if context.executing_eagerly():
        captured_iterator = iter(dataset_ops.Dataset.range(10))
      else:
        captured_iterator = dataset_ops.make_initializable_iterator(
            dataset_ops.Dataset.range(10))
      ds = _build_ds(captured_iterator)
      return captured_iterator, ds

    captured_iter, ds = _build_graph()
    if not context.executing_eagerly():
      self.evaluate(captured_iter.initializer)
    get_next = self.getNext(ds, requires_initialization=True)
    for i in range(10):
      self.assertEqual(i * i, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testCaptureHashTable(self, apply_map):
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

    dataset = apply_map(input_sentences,
                        lambda x: string_ops.string_split([x]).values)
    dataset = apply_map(dataset, table.lookup)

    get_next = self.getNext(dataset, requires_initialization=True)

    self.evaluate(table.initializer)
    self.evaluate(get_next())
    self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # TODO(b/123904513)
  @combinations.generate(_test_combinations_with_mode_v1("graph"))
  def testCaptureQueue(self, apply_map):
    elements = np.random.randint(100, size=[200])
    queue = data_flow_ops.FIFOQueue(200, dtypes.int64, shapes=[])
    enqueue_op = queue.enqueue_many(elements)
    close_op = queue.close()
    dataset = dataset_ops.Dataset.from_tensors(0).repeat(-1)
    dataset = apply_map(dataset, lambda _: queue.dequeue())

    get_next = self.getNext(dataset, requires_initialization=True)
    self.evaluate(enqueue_op)
    self.evaluate(close_op)

    for element in elements:
      self.assertEqual(element, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  # TODO(b/117581999): Possible deadlock in eager mode, debug.
  @combinations.generate(_test_combinations_with_mode_v1("graph"))
  def testCaptureSameResourceMultipleTimes(self, apply_map):
    elements = np.random.randint(100, size=[200])
    queue = data_flow_ops.FIFOQueue(
        200, dtypes.int64, shapes=[], shared_name="shared_queue")
    queue_2 = data_flow_ops.FIFOQueue(
        200, dtypes.int64, shapes=[], shared_name="shared_queue")

    enqueue_op = queue.enqueue_many(elements)
    close_op = queue.close()

    dataset = dataset_ops.Dataset.from_tensors(0).repeat(-1)
    dataset = apply_map(dataset, lambda _: (queue.dequeue(), queue_2.dequeue()))

    self.evaluate(enqueue_op)
    self.evaluate(close_op)
    get_next = self.getNext(dataset, requires_initialization=True)
    for i in range(100):
      self.assertCountEqual([elements[i * 2], elements[i * 2 + 1]],
                            self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testSeededStatefulOperatorIsProperlyStateful(self, apply_map):
    dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)
    fn = lambda _: random_ops.random_uniform((), seed=11)
    dataset = apply_map(dataset, fn).batch(2)

    get_next = self.getNext(dataset, requires_initialization=True)
    random_values = []
    with self.assertRaises(errors.OutOfRangeError):
      while True:
        random_values.extend(self.evaluate(get_next()))
    self.assertLen(random_values, 10)
    self.assertGreater(np.abs(np.diff(random_values)).max(), 1e-6)

    get_next = self.getNext(dataset, requires_initialization=True)
    random_values_2 = []
    with self.assertRaises(errors.OutOfRangeError):
      while True:
        random_values_2.extend(self.evaluate(get_next()))

    # Randomness is repeatable given same seed
    self.assertAllClose(random_values, random_values_2)

  @combinations.generate(_test_combinations())
  def testStatefulMapKeepsStateAcrossIterators(self, apply_map):
    dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)
    fn = lambda _: random_ops.random_uniform((), seed=11)
    dataset = apply_map(dataset, fn).repeat(1000).batch(10)

    get_next = self.getNext(dataset)
    random_values = self.evaluate(get_next())

    # Assert that one of the next 99 batches yielded by the iterator is
    # different from the first.
    i = 0
    while i < 99:
      if np.any(random_values != self.evaluate(get_next())):
        break
      i += 1
    self.assertLess(i, 99)

  @combinations.generate(_test_combinations())
  def testStatefulOperationInShortCircuit(self, apply_map):
    counter_var = variable_scope.get_variable(
        "counter", (), dtypes.int32, use_resource=True)

    def increment_fn(x):
      counter_var.assign_add(1)
      return x

    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, increment_fn)

    get_next = self.getNext(dataset, requires_initialization=True)

    self.evaluate(counter_var.initializer)
    for i in range(10):
      self.assertEqual(i, self.evaluate(counter_var))
      self.assertEqual(i, self.evaluate(get_next()))
    self.assertEqual(10, self.evaluate(counter_var))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    self.assertEqual(10, self.evaluate(counter_var))

  @combinations.generate(_test_combinations())
  def testMapDict(self, apply_map):
    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, lambda x: {"foo": x * 2, "bar": x**2})
    dataset = apply_map(dataset, lambda d: d["foo"] + d["bar"])
    self.assertDatasetProduces(
        dataset, expected_output=[i * 2 + i**2 for i in range(10)])

  @combinations.generate(_test_combinations())
  def testMapNamedtuple(self, apply_map):
    # construct dataset of tuples
    labels = dataset_ops.Dataset.range(10)
    images = apply_map(labels, lambda l: -l)
    dataset_tuple = dataset_ops.Dataset.zip((labels, images))

    # convert dataset of tuples to dataset of namedtuples
    example = collections.namedtuple("Example", ["label", "image"])
    dataset_namedtuple = apply_map(dataset_tuple, example)

    def preprocess_tuple(label, image):
      image = 2 * image
      return label, image

    def preprocess_namedtuple(example):
      return example._replace(image=2 * example.image)

    # preprocess both datasets
    dataset_tuple = apply_map(dataset_tuple, preprocess_tuple)
    dataset_namedtuple = apply_map(dataset_namedtuple, preprocess_namedtuple)

    next_tuple = self.getNext(dataset_tuple)
    next_namedtuple = self.getNext(dataset_namedtuple)

    # make sure both datasets contain the same data
    for i in range(10):
      tuple_, namedtuple_ = self.evaluate([next_tuple(), next_namedtuple()])
      self.assertEqual(tuple_, namedtuple_)
      self.assertEqual(tuple_, (i, -2 * i))

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_namedtuple())

  @combinations.generate(_test_combinations())
  def testMapAttrs(self, apply_map):
    if attr is None:
      self.skipTest("attr module is not available.")

    # construct dataset of tuples
    labels = dataset_ops.Dataset.range(10)
    images = apply_map(labels, lambda l: -l)
    dataset = dataset_ops.Dataset.zip((labels, images))

    @attr.s(cmp=True)
    class Example(object):
      label = attr.ib()
      image = attr.ib()

    dataset = apply_map(dataset, Example)

    def preprocess(example):
      example.image = 2 * example.image
      return example

    dataset = apply_map(dataset, preprocess)
    get_next = self.getNext(dataset)

    for i in range(10):
      data = self.evaluate(get_next())
      self.assertEqual(data, Example(i, -2 * i))

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testUseStepContainerInMap(self, apply_map):
    row = np.arange(6)
    dataset = dataset_ops.Dataset.from_tensors(row)
    dataset = apply_map(dataset,
                        lambda elems: map_fn.map_fn(lambda x: x * x, elems))
    self.assertDatasetProduces(dataset, expected_output=[row**2])

  @combinations.generate(_test_combinations())
  def testCaseAndCondInMap(self, apply_map):

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

      pred_fn_pairs = [
          (math_ops.logical_or(math_ops.equal(y, 2),
                               math_ops.equal(y, 3)), defaults_two),
      ]

      return control_flow_ops.case(
          pred_fn_pairs, default=multiply, exclusive=True)

    def build_dataset(row, num):
      dataset = dataset_ops.Dataset.from_tensor_slices(row)
      return apply_map(dataset, lambda x: control_map_fn(x, num))

    row = np.arange(6)
    for num in [2, 3, 4]:
      get_next = self.getNext(build_dataset(row, num))
      for i in range(6):
        self.assertEqual(
            (i // 2 if i % 2 else i * 2) if (num == 2 or num == 3) else i * 2,
            self.evaluate(get_next()))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testCaseInWhileInMap(self, apply_map):

    def control_map_fn(x, y):

      def multiply():
        return x * 2

      def divide():
        return x // 2

      pred_fn_pairs = [
          (math_ops.logical_or(math_ops.equal(y, 2),
                               math_ops.equal(y, 3)), divide),
      ]

      return control_flow_ops.case(
          pred_fn_pairs, default=multiply, exclusive=True)

    def build_dataset(row, num):
      dataset = dataset_ops.Dataset.from_tensors(row)
      return apply_map(
          dataset,
          lambda elems: map_fn.map_fn(lambda x: control_map_fn(x, num), elems))

    row = np.arange(6)
    for num in [2, 3, 4]:
      get_next = self.getNext(build_dataset(row, num))
      self.assertAllEqual(
          [x // 2 if (num == 2 or num == 3) else x * 2 for x in row],
          self.evaluate(get_next()))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testCaseAndCondInWhileInMap(self, apply_map):

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

      pred_fn_pairs = [
          (math_ops.logical_or(math_ops.equal(y, 2),
                               math_ops.equal(y, 3)), defaults_two),
      ]

      return control_flow_ops.case(
          pred_fn_pairs, default=multiply, exclusive=True)

    row = np.arange(6)
    num = 2
    dataset = dataset_ops.Dataset.from_tensors(row)
    dataset = apply_map(
        dataset,
        lambda elems: map_fn.map_fn(lambda x: control_map_fn(x, num), elems))
    get_next = self.getNext(dataset)

    self.assertAllEqual([(x // 2 if x % 2 else x * 2) if
                         (num == 2 or num == 3) else x * 2 for x in row],
                        self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testNestedListMapDataset(self, apply_map):
    dataset = dataset_ops.Dataset.from_tensors([0, 1, 2]).repeat(10)
    dataset = apply_map(dataset, lambda a: ([a[1], a[0] + a[2]], a[1]))
    expected_output = [(np.array([1, 2]), 1)] * 10
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(
      combinations.times(_test_combinations(),
                         combinations.combine(buffer_size=[1, 2, 3, 4])))
  def testPrefetch(self, apply_map, buffer_size):
    # We will use this event to test that `_map_py_func()` has been invoked a
    # certain number of times (6 times, to be exact) after consuming fewer
    # elements from the iterator.
    ev = threading.Event()

    set_event_during_invocation = 5

    def _map_py_func(x):
      if x == set_event_during_invocation:
        ev.set()
      return x * x

    def _map_fn(x):
      return script_ops.py_func(_map_py_func, [x], x.dtype)

    # We can indirectly observe that varying the buffer size has the intended
    # effect by observing when `ev` is set (on the 6th invocation of
    # `_map_py_func()`).
    # NOTE(mrry): We do not test with `buffer_size ==
    # set_event_during_invocation`, because we must consume at least one element
    # to start the prefetching.
    dataset = dataset_ops.Dataset.range(100)
    dataset = apply_map(dataset, _map_fn).prefetch(buffer_size)
    get_next = self.getNext(dataset)

    event_will_be_set_after_consuming = (
        set_event_during_invocation - buffer_size + 1)

    ev.clear()
    for i in range(event_will_be_set_after_consuming):
      self.assertFalse(ev.is_set())
      self.assertEqual(i * i, self.evaluate(get_next()))
    ev.wait()
    for i in range(event_will_be_set_after_consuming, 100):
      self.assertEqual(i * i, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testReturnList(self, apply_map):
    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, lambda x: [x, constant_op.constant(37.0)])
    self.assertDatasetProduces(
        dataset, expected_output=[(i, 37.0) for i in range(10)])

  @combinations.generate(_test_combinations())
  def testMultiOutputPyFunc(self, apply_map):
    # The `tf.py_func()` op returns a list of tensors for its outputs.
    def _map_fn(x_tensor):
      def _map_py_func(x):
        return x, np.array(37.0, dtype=np.float64)
      return script_ops.py_func(
          _map_py_func, [x_tensor], [dtypes.int64, dtypes.float64])

    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, _map_fn)
    self.assertDatasetProduces(
        dataset, expected_output=[(i, 37.0) for i in range(10)])

  @combinations.generate(_test_combinations())
  def testSparse(self, apply_map):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, _sparse)
    self.assertDatasetProduces(
        dataset, expected_output=[_sparse(i) for i in range(10)])

  @combinations.generate(_test_combinations())
  def testSparseChain(self, apply_map):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    def _check(i):
      self.assertTrue(sparse_tensor.is_sparse(i))
      return sparse_ops.sparse_concat(0, [i, i])

    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, _sparse)
    dataset = apply_map(dataset, _check)

    self.assertDatasetProduces(
        dataset,
        expected_output=[self.evaluate(_check(_sparse(i))) for i in range(10)])

  @combinations.generate(_test_combinations_with_mode("eager"))
  def testSparseMapShapeInference(self, apply_map):
    row_lengths = np.random.randint(0, 4, size=128)
    values = np.ones(np.sum(row_lengths))
    sparse = ragged_tensor.RaggedTensor.from_row_lengths(
        values, row_lengths).to_sparse()
    dataset = dataset_ops.Dataset.from_tensor_slices(sparse)
    dataset = dataset.batch(32, drop_remainder=True)
    dataset = apply_map(dataset, lambda x: x)
    self.assertEqual((32, 3), dataset.element_spec.shape)

  @combinations.generate(_test_combinations_with_mode("eager"))
  def testSparseMapShapeInferencePartial(self, apply_map):
    row_lengths = np.random.randint(0, 4, size=128)
    values = np.ones(np.sum(row_lengths))
    sparse = ragged_tensor.RaggedTensor.from_row_lengths(
        values, row_lengths).to_sparse()
    dataset = dataset_ops.Dataset.from_tensor_slices(sparse)
    dataset = dataset.batch(32, drop_remainder=False)
    dataset = apply_map(dataset, lambda x: x)
    self.assertEqual([None, 3], dataset.element_spec.shape.as_list())

  @combinations.generate(_test_combinations())
  def testTensorArray(self, apply_map):

    def _tensor_array(i):
      i = math_ops.cast(i, dtypes.int32)
      return (
          tensor_array_ops.TensorArray(dtypes.int32, element_shape=(), size=i)
          .unstack(math_ops.range(i, dtype=dtypes.int32)))

    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, _tensor_array)
    self.assertDatasetProduces(
        dataset, expected_output=[list(range(i)) for i in range(10)])

  @combinations.generate(_test_combinations())
  def testTensorArrayChain(self, apply_map):

    def _tensor_array(i):
      i = math_ops.cast(i, dtypes.int32)
      return (
          tensor_array_ops.TensorArray(dtypes.int32, element_shape=(), size=i)
          .unstack(math_ops.range(i, dtype=dtypes.int32)))

    def _check(x):
      self.assertIsInstance(x, tensor_array_ops.TensorArray)
      return x.identity()

    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, _tensor_array)
    dataset = apply_map(dataset, _check)

    self.assertDatasetProduces(
        dataset,
        expected_output=[list(range(i)) for i in range(10)])

  @combinations.generate(_test_combinations())
  def testRagged(self, apply_map):

    def _ragged(i):
      return ragged_tensor.RaggedTensor.from_tensor(i * [[1]])

    dataset = dataset_ops.Dataset.range(5)
    dataset = apply_map(dataset, _ragged)
    self.assertDatasetProduces(
        dataset,
        expected_output=[ragged_factory_ops.constant([[i]]) for i in range(5)])

  @combinations.generate(_test_combinations())
  def testRaggedChain(self, apply_map):

    def _ragged(i):
      return ragged_tensor.RaggedTensor.from_tensor(i * [[1]])

    def _concat(i):
      self.assertTrue(ragged_tensor.is_ragged(i))
      return ragged_concat_ops.concat([i, i], 0)

    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, _ragged)
    dataset = apply_map(dataset, _concat)

    self.assertDatasetProduces(
        dataset,
        expected_output=[
            self.evaluate(_concat(ragged_factory_ops.constant([[i]])))
            for i in range(10)
        ])

  # TODO(b/123904513)
  @combinations.generate(_test_combinations_with_mode_v1("graph"))
  def testParallelMapOutOfRangeError(self, apply_map):

    def raising_py_func(i):
      if i == 100:
        raise StopIteration()
      else:
        return i

    dataset = dataset_ops.Dataset.range(105)
    dataset = apply_map(
        dataset,
        lambda x: script_ops.py_func(raising_py_func, [x], dtypes.int64),
        num_parallel_calls=2)
    get_next = self.getNext(dataset)
    for i in range(100):
      self.assertEqual(i, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testConstantOutput(self, apply_map):
    dataset = dataset_ops.Dataset.range(10)
    dataset = apply_map(dataset, lambda x: [x, "hello", 10])
    self.assertDatasetProduces(dataset, [(i, b"hello", 10) for i in range(10)])

  @combinations.generate(test_base.graph_only_combinations())
  def testWarnOnSeedFromOuterGraph(self):
    with ops.Graph().as_default() as g:
      g.seed = 10
      warnings.simplefilter("always")

      def _check_warning(caught_warnings, expected_result):
        found_warning = False
        for warning in caught_warnings:
          if ("Explicitly set the seed in the function if this is not the "
              "intended behavior" in str(warning)):
            found_warning = True
            break
        self.assertEqual(found_warning, expected_result)

      # map_fun doesn't use seed, so no warning is generated.
      with warnings.catch_warnings(record=True) as w:
        _ = dataset_ops.Dataset.range(10).map(math_ops.square)
      _check_warning(w, False)

      def random_func(x):
        x = math_ops.add(x, 1)
        random_ops.random_shuffle([x, math_ops.square(x)])
        return x

      with warnings.catch_warnings(record=True) as w:
        _ = dataset_ops.Dataset.range(10).map(random_func)
      _check_warning(w, True)

      def random_func_seeded(x):
        ops.get_default_graph().seed = None
        random_ops.random_shuffle(x)
        return x

      with warnings.catch_warnings(record=True) as w:
        _ = dataset_ops.Dataset.range(10).batch(2).map(random_func_seeded)
      _check_warning(w, False)

      with warnings.catch_warnings(record=True) as w:
        _ = dataset_ops.Dataset.range(10).batch(2).map(
            lambda x: random_ops.random_shuffle(x, seed=37))
      _check_warning(w, False)

  @combinations.generate(_test_combinations())
  def testNestedDatasetMap(self, apply_map):
    dataset = dataset_ops.Dataset.from_tensors([1.0, 2.0, 3.0])
    dataset = apply_map(dataset, dataset_ops.Dataset.from_tensor_slices)
    dataset = apply_map(dataset, lambda ds: ds.batch(3)).flat_map(lambda x: x)

    self.assertDatasetProduces(dataset, expected_output=[[1.0, 2.0, 3.0]])

  @combinations.generate(_test_combinations())
  def testReturnValueError(self, apply_map):
    dataset = dataset_ops.Dataset.from_tensors([1.0, 2.0, 3.0])
    with self.assertRaisesRegex(
        TypeError, r"Unsupported return value from function passed to "
        r"Dataset.map\(\)"):
      _ = apply_map(dataset, lambda x: Foo)

  @combinations.generate(test_base.default_test_combinations())
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

      # Create a "Const" op with a `tf.float32` value and a `tf.int32` type.
      const_tensor = ops.get_default_graph().create_op(
          "Const", [], [dtypes.int32],
          attrs={
              "value": tensor_value,
              "dtype": dtype_value
          },
          name="BrokenConst").outputs[0]
      return const_tensor

    dataset = dataset.map(broken_function)
    self.assertDatasetProduces(
        dataset, expected_error=(errors.InvalidArgumentError, "Type mismatch"))

  @combinations.generate(
      combinations.times(
          _test_combinations_with_mode("graph"),
          combinations.combine(num_parallel_calls=[None, 12])))
  def testNoInterOpParallelism(self, apply_map, num_parallel_calls):
    dataset = dataset_ops.Dataset.from_tensors(0)

    def _get_tid():
      return np.int64(threading.current_thread().ident)

    def _map_fn(_):
      tids = []
      for _ in range(10):
        tids.append(script_ops.py_func(_get_tid, [], dtypes.int64))
      return tids

    dataset = apply_map(dataset, _map_fn)
    dataset._variant_tensor.op._set_attr("use_inter_op_parallelism",
                                         attr_value_pb2.AttrValue(b=False))
    get_next = self.getNext(dataset)

    tids = self.evaluate(get_next())
    self.assertTrue(all(tids[0] == tid for tid in tids))

  @combinations.generate(
      combinations.times(_test_combinations(), _short_circuit_test_cases(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testShortCircuit(self, apply_map, structure, fn, num_parallel_calls):
    dataset = self.structuredDataset(structure).repeat()
    dataset = apply_map(dataset, fn, num_parallel_calls=num_parallel_calls)
    get_next = self.getNext(dataset)

    if isinstance(structure, tuple):
      expected = fn(*self.evaluate(self.structuredElement(structure)))
    else:
      expected = fn(self.evaluate(self.structuredElement(structure)))
    self.assertEqual(expected, self.evaluate(get_next()))

  @combinations.generate(
      combinations.times(_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 12])))
  def testShortCircuitCapturedInput(self, apply_map, num_parallel_calls):
    captured_t = variables.Variable(42)
    dataset = self.structuredDataset(None).repeat()
    dataset = apply_map(
        dataset, lambda x: captured_t, num_parallel_calls=num_parallel_calls)
    self.evaluate(variables.global_variables_initializer())
    get_next = self.getNext(dataset, requires_initialization=True)

    self.assertEqual(42, self.evaluate(get_next()))

  @combinations.generate(
      combinations.combine(
          tf_api_version=2,
          mode=["eager", "graph"],
          num_parallel_calls=[None, 12]))
  def testPreserveCardinality(self, num_parallel_calls):

    def py_fn(_):
      raise StopIteration()

    dataset = dataset_ops.Dataset.from_tensors(0).map(
        lambda x: script_ops.py_func(py_fn, [x], dtypes.int64),
        num_parallel_calls=num_parallel_calls)
    get_next = self.getNext(dataset)
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations_with_mode("graph"))
  def testCollectionCopy(self, apply_map):
    w = variable_scope.get_variable("w", [])
    self.assertIn(w, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))

    def func(x):
      self.assertIn(w, ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
      return x

    dataset = dataset_ops.Dataset.from_tensors(constant_op.constant(1.0))
    _ = apply_map(dataset, func)

  @combinations.generate(
      combinations.times(
          _test_combinations_with_mode_v1("graph"),
          combinations.combine(num_parallel_calls=[None, 12])))
  def testMapCancellation(self, apply_map, num_parallel_calls):
    # Checks that a cancellation of is threaded through to map transformation.
    queue = data_flow_ops.FIFOQueue(10, dtypes.int32, ())

    def fn(_):
      return queue.dequeue()

    dataset = dataset_ops.Dataset.range(1)
    dataset = apply_map(dataset, fn, num_parallel_calls=num_parallel_calls)
    get_next = self.getNext(dataset, requires_initialization=True)

    with self.cached_session() as sess:
      thread = self.checkedThread(self.assert_op_cancelled, args=(get_next(),))
      thread.start()
      time.sleep(0.2)
      sess.close()
      thread.join()


  # TODO(b/126553094): map doesnt work with variable defined inside function in
  # eager mode, possible Graph tensors leak out of the function building context
  # from function graph in eager mode as variables are created in init_scope.
  @combinations.generate(test_base.graph_only_combinations())
  def testCreateVariableInsideFunctionWithGetter(self):

    def func(_):
      with variable_scope.variable_scope(
          "variable", reuse=variable_scope.AUTO_REUSE):
        counter_var = variable_scope.get_variable(
            "counter", (), dtypes.int32, use_resource=True)
      return counter_var.assign_add(1)

    dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)

    if hasattr(dataset, "map_with_legacy_function"):
      # NOTE: In the legacy function, resource is captured by value.
      with self.assertRaisesWithPredicateMatch(
          AttributeError, "'Tensor' object has no attribute 'assign_add'"):
        dataset.map_with_legacy_function(func)

    dataset = dataset.map(func)
    self.evaluate(variables.global_variables_initializer())

    get_next = self.getNext(dataset, requires_initialization=True)

    for i in range(10):
      self.assertEqual(i + 1, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(_test_combinations())
  def testCaptureVariable(self, apply_map):
    counter_var = variable_scope.get_variable(
        "counter", (), dtypes.int32, use_resource=True)
    dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)
    dataset = apply_map(dataset, lambda _: counter_var.assign_add(1))
    get_next = self.getNext(dataset, requires_initialization=True)

    self.evaluate(counter_var.initializer)

    for i in range(10):
      self.assertEqual(i, self.evaluate(counter_var))
      self.assertEqual(i + 1, self.evaluate(get_next()))
    self.assertEqual(10, self.evaluate(counter_var))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    self.assertEqual(10, self.evaluate(counter_var))

  @combinations.generate(_test_combinations_with_mode_v1("graph"))
  def testCaptureUninitializedVariableError(self, apply_map):
    counter_var = variable_scope.get_variable(
        "counter", (), dtypes.int32, use_resource=True)
    dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)
    dataset = apply_map(dataset, lambda _: counter_var.assign_add(1))

    get_next = self.getNext(dataset, requires_initialization=True)
    with self.assertRaises(errors.NotFoundError):
      self.evaluate(get_next())

  # TODO(b/121264236): add eager mode coverage when we have multi-device setup.
  @combinations.generate(_test_combinations_with_mode_v1("graph"))
  def testCaptureConstantsWithConflictingDevices(self, apply_map):
    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.cached_session(config=config):
      with ops.device("/device:CPU:0"):
        a = constant_op.constant(3.0)
      with ops.device("/device:CPU:1"):
        b = constant_op.constant(5.0)

      def func(_):
        return math_ops.add(a, b)

      dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)
      dataset = apply_map(dataset, func)
      expected_output = [8.0] * 10
      self.assertDatasetProduces(dataset, expected_output=expected_output)

  # TODO(b/121264236): add eager mode coverage when we have multi-device setup.
  @combinations.generate(_test_combinations_with_mode_v1("graph"))
  def testReferenceVariablesWithMultipleDevices(self, apply_map):
    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.cached_session(config=config):

      def func(_):
        with ops.device("/device:CPU:0"):
          a = variables.VariableV1(3.0)
        with ops.device("/device:CPU:1"):
          b = variables.VariableV1(5.0)
        return math_ops.add(a, b)

      # NOTE: Use the legacy function implementation as eager function will
      # convert RefVariables to ResourceVariables.
      dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)
      dataset = apply_map(dataset, func)
      self.evaluate(variables.global_variables_initializer())
      expected_output = [8.0] * 10
      self.assertDatasetProduces(
          dataset,
          expected_output=expected_output,
          requires_initialization=True)

  # TODO(b/121264236): add eager mode coverage when we have multi-device setup.
  @combinations.generate(_test_combinations_with_mode_v1("graph"))
  def testResourceVariablesWithMultipleDevices(self, apply_map):
    config = config_pb2.ConfigProto(device_count={"CPU": 3})

    def func(_):
      with variable_scope.variable_scope(
          "variable", reuse=variable_scope.AUTO_REUSE):
        with ops.device("/device:CPU:0"):
          a_var = variable_scope.get_variable(
              "a", (), dtypes.int32, use_resource=True)
          a_var = math_ops.add(a_var, 1)
        with ops.device("/device:CPU:1"):
          b_var = variable_scope.get_variable(
              "b", (), dtypes.int32, use_resource=True)
      return math_ops.add(a_var, b_var)

    g = ops.Graph()
    with self.session(config=config, graph=g):
      dataset = dataset_ops.Dataset.from_tensors(0).repeat(10)
      dataset = apply_map(dataset, func)
      self.evaluate(variables.global_variables_initializer())
      expected_output = [1] * 10
      self.assertDatasetProduces(
          dataset,
          expected_output=expected_output,
          requires_initialization=True)

  @combinations.generate(
      combinations.times(
          _test_combinations(),
          combinations.combine(
              local_determinism=[None, True, False],
              global_determinism=[True, False])))
  def testDeterminismConfiguration(self, apply_map, local_determinism,
                                   global_determinism):
    expect_determinism = local_determinism or (local_determinism is None and
                                               global_determinism)
    elements = list(range(1000))

    def dataset_fn(delay_ms):

      def sleep(x):
        time.sleep(delay_ms / 1000)
        return x

      def map_function(x):
        if math_ops.equal(x, 0):
          return script_ops.py_func(sleep, [x], x.dtype)
        else:
          return x

      dataset = dataset_ops.Dataset.from_tensor_slices(elements)
      dataset = apply_map(
          dataset,
          map_function,
          num_parallel_calls=2,
          deterministic=local_determinism)
      opts = options_lib.Options()
      opts.deterministic = global_determinism
      dataset = dataset.with_options(opts)
      return dataset

    self.checkDeterminism(
        dataset_fn, expect_determinism, expected_elements=elements)

  @combinations.generate(_test_combinations())
  def testNoneComponent(self, apply_map):
    dataset = dataset_ops.Dataset.from_tensors((42, None))

    def map_function(x, y):
      if y is None:
        return x / 2
      return x

    dataset = apply_map(dataset, map_function)
    self.assertDatasetProduces(dataset, expected_output=[21])

  @combinations.generate(test_base.eager_only_combinations())
  def testCheckpointLargeBuffer(self):
    if pywrap_sanitizers.is_tsan_enabled():
      self.skipTest("Creating a large buffer causes OOM when using tsan.")
    # Tensor of size 512M
    dataset = dataset_ops.Dataset.from_tensors(
        array_ops.ones((128, 1024, 1024), dtype=dtypes.float32))
    dataset = dataset.repeat()
    # Set parallelism to 5 to exceed the 2GB protobuf limit
    dataset = dataset.map(lambda x: x * 2, num_parallel_calls=5)
    iterator = iter(dataset)
    next(iterator)  # request an element to fill the parallel map buffer
    ckpt = trackable_utils.Checkpoint(iterator=iterator)
    manager = checkpoint_management.CheckpointManager(
        ckpt, self.get_temp_dir(), max_to_keep=1)
    manager.save()

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 1])))
  def testName(self, num_parallel_calls):
    dataset = dataset_ops.Dataset.from_tensors(21).map(
        lambda x: x * 2, num_parallel_calls=num_parallel_calls, name="map")
    self.assertDatasetProduces(dataset, [42])


class MapCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                        parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 2])))
  def testCore(self, verify_fn, num_parallel_calls):

    tensor_slice_len = 7
    num_epochs = 2
    multiplier = 37.0

    def _build_ds():

      components = (np.arange(tensor_slice_len), np.array([[1, 2, 3]]) *
                    np.arange(tensor_slice_len)[:, np.newaxis],
                    np.array(multiplier) * np.arange(tensor_slice_len))

      def _map_fn(x, y, z):
        return math_ops.square(x), math_ops.square(y), math_ops.square(z)

      return (dataset_ops.Dataset.from_tensor_slices(components).map(
          _map_fn, num_parallel_calls=num_parallel_calls).repeat(num_epochs))

    verify_fn(self, _build_ds, tensor_slice_len * num_epochs)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 2])))
  def testSaveStatefulFunction(self, num_parallel_calls):

    def _build_ds():

      def _map_fn(x):
        return random_ops.random_uniform(
            (), 0, 10, dtype=dtypes.int32) * math_ops.cast(x, dtypes.int32)

      return dataset_ops.Dataset.range(100).map(
          _map_fn, num_parallel_calls=num_parallel_calls)

    self.verify_error_on_save(_build_ds, 15, errors.FailedPreconditionError)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 2])))
  def testCaptureVariableInMapFn(self, num_parallel_calls):

    def _build_ds():
      counter_var = variable_scope.get_variable(
          "counter", (), dtypes.int32, use_resource=True)
      return (dataset_ops.Dataset.from_tensors(0).repeat(10).map(
          lambda _: counter_var.assign_add(1),
          num_parallel_calls=num_parallel_calls))

    self.verify_error_on_save(_build_ds, 15, errors.FailedPreconditionError)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 2])))
  def testCaptureConstantInMapFn(self, verify_fn, num_parallel_calls):
    num_outputs = 10

    def _build_ds():
      constant_var = constant_op.constant(5)
      return (dataset_ops.Dataset.from_tensors(0).repeat(10).map(
          lambda x: x + constant_var, num_parallel_calls=num_parallel_calls))

    verify_fn(self, _build_ds, num_outputs)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 2])))
  def testCaptureDefunInMapFn(self, verify_fn, num_parallel_calls):
    num_outputs = 10

    def _build_ds():

      @function.Defun(dtypes.int64)
      def defun_fn(x):
        return constant_op.constant(1000) + math_ops.cast(x, dtypes.int32)

      return dataset_ops.Dataset.range(num_outputs).map(
          defun_fn, num_parallel_calls=num_parallel_calls)

    verify_fn(self, _build_ds, num_outputs)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 2])))
  def testBuildDefunInMapFn(self, verify_fn, num_parallel_calls):
    num_outputs = 10

    def _build_ds():

      @function.Defun(dtypes.int64)
      def defun_fn(x):

        @function.Defun(dtypes.int32)
        def defun_fn_deep(x):
          return constant_op.constant(1000) + math_ops.cast(x, dtypes.int32)

        return constant_op.constant(11000) + defun_fn_deep(
            math_ops.cast(x, dtypes.int32))

      return dataset_ops.Dataset.range(num_outputs).map(
          defun_fn, num_parallel_calls=num_parallel_calls)

    verify_fn(self, _build_ds, num_outputs)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations(),
                         combinations.combine(num_parallel_calls=[None, 2])))
  def testSparse(self, verify_fn, num_parallel_calls):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    def _build_ds(num_outputs):
      return dataset_ops.Dataset.range(num_outputs).map(
          _sparse, num_parallel_calls=num_parallel_calls)

    num_outputs = 10
    verify_fn(self, lambda: _build_ds(num_outputs), num_outputs=num_outputs)


class MapRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.v2_only_combinations(),
                         combinations.combine(index=[-1, 4, 5])))
  def testInvalidIndex(self, index):
    dataset = dataset_ops.Dataset.from_tensor_slices([-1, 0, 1,
                                                      2]).map(lambda x: x * 2)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(
      combinations.times(test_base.v2_only_combinations(),
                         combinations.combine(index=[-1, 0])))
  def testEmptyDataset(self, index):
    dataset = dataset_ops.Dataset.from_tensor_slices([]).map(lambda x: x // 2)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, index=index))

  @combinations.generate(combinations.times(test_base.v2_only_combinations()))
  def testBasic(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0, 1, 2, 3, 4,
                                                      5]).map(lambda x: x * 3)
    for i in range(5):
      self.assertEqual(self.evaluate(random_access.at(dataset, index=i)), i * 3)

  @combinations.generate(
      combinations.times(
          test_base.v2_only_combinations(),
          combinations.combine(
              elements=[0, 10, 20, 40], num_parallel_calls=[None, 2])))
  def testMultipleCombinations(self, elements, num_parallel_calls):
    dataset = dataset_ops.Dataset.range(elements).map(
        lambda x: x // 2, num_parallel_calls=num_parallel_calls)
    for i in range(elements):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)), i // 2)

  @combinations.generate(
      combinations.times(
          test_base.v2_only_combinations(),
          combinations.combine(
              elements=[0, 10, 20, 40], num_parallel_calls=[None, 2])))
  def testMapFnInFunction(self, elements, num_parallel_calls):

    @def_function.function
    def _map_fn(x):
      return math_ops.square(x)

    dataset = dataset_ops.Dataset.range(elements).map(
        _map_fn, num_parallel_calls=num_parallel_calls)
    for i in range(elements):
      self.assertEqual(
          self.evaluate(random_access.at(dataset, index=i)),
          self.evaluate(math_ops.square(i)))


if __name__ == "__main__":
  test.main()

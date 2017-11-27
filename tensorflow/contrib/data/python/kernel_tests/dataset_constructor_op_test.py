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

import threading

import numpy as np

from tensorflow.contrib.data.python.kernel_tests import dataset_serialization_test_base
from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


class DatasetConstructorTest(test.TestCase):

  def testTensorDataset(self):
    """Test an dataset that represents a single tuple of tensors."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))

    iterator = (dataset_ops.Dataset.from_tensors(components)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([c.shape for c in components],
                     [t.shape for t in get_next])

    with self.test_session() as sess:
      sess.run(init_op)
      results = sess.run(get_next)
      for component, result_component in zip(components, results):
        self.assertAllEqual(component, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testTensorSliceDataset(self):
    """Test an dataset that represents the slices from a tuple of tensors."""
    components = (
        np.tile(np.array([[1], [2], [3], [4]]), 20), np.tile(
            np.array([[12], [13], [14], [15]]), 22),
        np.array([37.0, 38.0, 39.0, 40.0])
    )

    iterator = (dataset_ops.Dataset.from_tensor_slices(components)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([c.shape[1:] for c in components],
                     [t.shape for t in get_next])

    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(4):
        results = sess.run(get_next)
        for component, result_component in zip(components, results):
          self.assertAllEqual(component[i], result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testTensorSliceDatasetWithDict(self):
    components = {"foo": [1, 2, 3], "bar": [[4.0], [5.0], [6.0]]}
    iterator = (dataset_ops.Dataset.from_tensor_slices(components)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual(dtypes.int32, iterator.output_types["foo"])
    self.assertEqual(dtypes.float32, iterator.output_types["bar"])
    self.assertEqual((), iterator.output_shapes["foo"])
    self.assertEqual((1,), iterator.output_shapes["bar"])

    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(3):
        results = sess.run(get_next)
        self.assertEqual(components["foo"][i], results["foo"])
        self.assertEqual(components["bar"][i], results["bar"])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testSparseTensorSliceDataset(self):
    """Test a dataset based on slices of a `tf.SparseTensor`."""
    st = array_ops.sparse_placeholder(dtypes.float64)
    iterator = (dataset_ops.Dataset.from_sparse_tensor_slices(st)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = sparse_tensor.SparseTensor(*iterator.get_next())

    with self.test_session() as sess:
      slices = [[1., 2., 3.], [1.], [1.], [1., 2.], [], [1., 2.], [], [], []]

      # Test with sparse tensor in the appropriate order.
      indices = np.array(
          [[i, j] for i in range(len(slices)) for j in range(len(slices[i]))])
      values = np.array([val for s in slices for val in s])
      dense_shape = np.array([len(slices), max(len(s) for s in slices) + 1])
      sparse_feed = sparse_tensor.SparseTensorValue(indices, values,
                                                    dense_shape)
      sess.run(init_op, feed_dict={st: sparse_feed})
      for i, s in enumerate(slices):
        results = sess.run(get_next)
        self.assertAllEqual(s, results.values)
        expected_indices = np.array(
            [[j] for j in range(len(slices[i]))]).reshape([-1, 1])
        self.assertAllEqual(expected_indices, results.indices)
        self.assertAllEqual(dense_shape[1:], results.dense_shape)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Test with sparse tensor in the reverse order, which is not
      # currently supported.
      reverse_order_indices = indices[::-1, :]
      reverse_order_values = values[::-1]
      sparse_feed = sparse_tensor.SparseTensorValue(
          reverse_order_indices, reverse_order_values, dense_shape)
      with self.assertRaises(errors.UnimplementedError):
        sess.run(init_op, feed_dict={st: sparse_feed})

      # Test with an empty sparse tensor.
      empty_indices = np.empty((0, 4), dtype=np.int64)
      empty_values = np.empty((0,), dtype=np.float64)
      empty_dense_shape = [0, 4, 37, 9]
      sparse_feed = sparse_tensor.SparseTensorValue(empty_indices, empty_values,
                                                    empty_dense_shape)
      sess.run(init_op, feed_dict={st: sparse_feed})
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # pylint: disable=g-long-lambda,unnecessary-lambda
  def testNestedStructure(self):
    components = (np.array([1, 2, 3]), (np.array([4., 5.]), np.array([6., 7.])),
                  np.array([8, 9, 10]))

    dataset = dataset_ops.Dataset.from_tensors(components)
    self.assertEquals((dtypes.int64, (dtypes.float64, dtypes.float64),
                       dtypes.int64), dataset.output_types)
    self.assertEquals(([3], ([2], [2]), [3]), dataset.output_shapes)

    dataset = dataset.shuffle(10, 10)
    self.assertEquals((dtypes.int64, (dtypes.float64, dtypes.float64),
                       dtypes.int64), dataset.output_types)
    self.assertEquals(([3], ([2], [2]), [3]), dataset.output_shapes)

    dataset = dataset.repeat(-1)
    self.assertEquals((dtypes.int64, (dtypes.float64, dtypes.float64),
                       dtypes.int64), dataset.output_types)
    self.assertEquals(([3], ([2], [2]), [3]), dataset.output_shapes)

    dataset = dataset.filter(lambda x, y, z: True)
    self.assertEquals((dtypes.int64, (dtypes.float64, dtypes.float64),
                       dtypes.int64), dataset.output_types)
    self.assertEquals(([3], ([2], [2]), [3]), dataset.output_shapes)

    dataset = dataset.take(5)
    self.assertEquals((dtypes.int64, (dtypes.float64, dtypes.float64),
                       dtypes.int64), dataset.output_types)
    self.assertEquals(([3], ([2], [2]), [3]), dataset.output_shapes)

    dataset = dataset.map(lambda x, y, z: ((x, z), (y[0], y[1])))
    self.assertEquals(((dtypes.int64, dtypes.int64),
                       (dtypes.float64, dtypes.float64)), dataset.output_types)
    self.assertEquals((([3], [3]), ([2], [2])), dataset.output_shapes)

    dataset = dataset.flat_map(
        lambda x, y: dataset_ops.Dataset.from_tensors(((x[0], x[1]),
                                                       (y[0], y[1])))
    )
    self.assertEquals(((dtypes.int64, dtypes.int64),
                       (dtypes.float64, dtypes.float64)), dataset.output_types)
    self.assertEquals((([3], [3]), ([2], [2])), dataset.output_shapes)

    dataset = dataset.batch(32)
    self.assertEquals(((dtypes.int64, dtypes.int64),
                       (dtypes.float64, dtypes.float64)), dataset.output_types)
    self.assertEquals((([None, 3], [None, 3]), ([None, 2], [None, 2])),
                      nest.pack_sequence_as(dataset.output_shapes, [
                          s.as_list()
                          for s in nest.flatten(dataset.output_shapes)
                      ]))

    iterator = dataset.make_one_shot_iterator()
    (w, x), (y, z) = iterator.get_next()
    self.assertEquals(dtypes.int64, w.dtype)
    self.assertEquals(dtypes.int64, x.dtype)
    self.assertEquals(dtypes.float64, y.dtype)
    self.assertEquals(dtypes.float64, z.dtype)
    self.assertEquals([None, 3], w.shape.as_list())
    self.assertEquals([None, 3], x.shape.as_list())
    self.assertEquals([None, 2], y.shape.as_list())
    self.assertEquals([None, 2], z.shape.as_list())

    iterator = dataset.make_initializable_iterator()
    (w, x), (y, z) = iterator.get_next()
    self.assertEquals(dtypes.int64, w.dtype)
    self.assertEquals(dtypes.int64, x.dtype)
    self.assertEquals(dtypes.float64, y.dtype)
    self.assertEquals(dtypes.float64, z.dtype)
    self.assertEquals([None, 3], w.shape.as_list())
    self.assertEquals([None, 3], x.shape.as_list())
    self.assertEquals([None, 2], y.shape.as_list())
    self.assertEquals([None, 2], z.shape.as_list())

    # Define a separate set of components with matching leading
    # dimension for the from-slices constructor.
    components_for_slices = (np.array([1, 2, 3]), (np.array(
        [4., 5., 6.]), np.array([7., 8., 9.])), np.array([10, 11, 12]))

    dataset = dataset_ops.Dataset.from_tensor_slices(components_for_slices)
    self.assertEquals((dtypes.int64, (dtypes.float64, dtypes.float64),
                       dtypes.int64), dataset.output_types)
    self.assertEquals(([], ([], []), []), dataset.output_shapes)

  def testNestedDict(self):
    components = {"a": {"aa": 1, "ab": [2.0, 2.0]}, "b": [3, 3, 3]}
    dataset = dataset_ops.Dataset.from_tensors(components)
    self.assertEquals(dtypes.int32, dataset.output_types["a"]["aa"])
    self.assertEquals(dtypes.float32, dataset.output_types["a"]["ab"])
    self.assertEquals(dtypes.int32, dataset.output_types["b"])
    self.assertEquals([], dataset.output_shapes["a"]["aa"])
    self.assertEquals([2], dataset.output_shapes["a"]["ab"])
    self.assertEquals([3], dataset.output_shapes["b"])

  def testNonSequenceNestedStructure(self):
    components = np.array([1, 2, 3])

    dataset = dataset_ops.Dataset.from_tensors(components)
    self.assertEquals(dtypes.int64, dataset.output_types)
    self.assertEquals([3], dataset.output_shapes)

    dataset = dataset.filter(
        lambda x: math_ops.reduce_all(math_ops.equal(x, components)))
    self.assertEquals(dtypes.int64, dataset.output_types)
    self.assertEquals([3], dataset.output_shapes)

    dataset = dataset.map(lambda x: array_ops.stack([x, x]))
    self.assertEquals(dtypes.int64, dataset.output_types)
    self.assertEquals([2, 3], dataset.output_shapes)

    dataset = dataset.flat_map(
        lambda x: dataset_ops.Dataset.from_tensor_slices(x))
    self.assertEquals(dtypes.int64, dataset.output_types)
    self.assertEquals([3], dataset.output_shapes)

    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()
    self.assertEquals(dtypes.int64, get_next.dtype)
    self.assertEquals([3], get_next.shape)

  def _testFromGenerator(self, generator, elem_sequence, num_repeats):
    iterator = (
        dataset_ops.Dataset.from_generator(generator, output_types=dtypes.int64)
        .repeat(num_repeats)
        .prefetch(5)
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      for _ in range(2):  # Run twice to test reinitialization.
        sess.run(init_op)
        for _ in range(num_repeats):
          for elem in elem_sequence:
            self.assertAllEqual(elem, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def _testFromGeneratorOneShot(self, generator, elem_sequence, num_repeats):
    iterator = (
        dataset_ops.Dataset.from_generator(generator, output_types=dtypes.int64)
        .repeat(num_repeats)
        .prefetch(5)
        .make_one_shot_iterator())
    get_next = iterator.get_next()

    with self.test_session() as sess:
      for _ in range(num_repeats):
        for elem in elem_sequence:
          self.assertAllEqual(elem, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testFromGeneratorUsingFunction(self):
    def generator():
      for i in range(1, 100):
        yield [i] * i
    elem_sequence = list(generator())
    self._testFromGenerator(generator, elem_sequence, 1)
    self._testFromGenerator(generator, elem_sequence, 5)
    self._testFromGeneratorOneShot(generator, elem_sequence, 1)
    self._testFromGeneratorOneShot(generator, elem_sequence, 5)

  def testFromGeneratorUsingList(self):
    generator = lambda: [[i] * i for i in range(1, 100)]
    elem_sequence = list(generator())
    self._testFromGenerator(generator, elem_sequence, 1)
    self._testFromGenerator(generator, elem_sequence, 5)

  def testFromGeneratorUsingNdarray(self):
    generator = lambda: np.arange(100, dtype=np.int64)
    elem_sequence = list(generator())
    self._testFromGenerator(generator, elem_sequence, 1)
    self._testFromGenerator(generator, elem_sequence, 5)

  def testFromGeneratorUsingGeneratorExpression(self):
    # NOTE(mrry): Generator *expressions* are not repeatable (or in
    # general reusable), because they eagerly evaluate the `for`
    # expression as `iter(range(1, 100))` and discard the means of
    # reconstructing `range(1, 100)`. Wrapping the generator
    # expression in a `lambda` makes it repeatable.
    generator = lambda: ([i] * i for i in range(1, 100))
    elem_sequence = list(generator())
    self._testFromGenerator(generator, elem_sequence, 1)
    self._testFromGenerator(generator, elem_sequence, 5)

  def testFromMultipleConcurrentGenerators(self):
    num_inner_repeats = 5
    num_outer_repeats = 100

    def generator():
      for i in range(1, 10):
        yield ([i] * i, [i, i ** 2, i ** 3])
    input_list = list(generator())

    # The interleave transformation is essentially a flat map that
    # draws from multiple input datasets concurrently (in a cyclic
    # fashion). By placing `Datsaet.from_generator()` inside an
    # interleave, we test its behavior when multiple iterators are
    # active at the same time; by additionally prefetching inside the
    # interleave, we create the possibility of parallel (modulo GIL)
    # invocations to several iterators created by the same dataset.
    def interleave_fn(_):
      return (dataset_ops.Dataset.from_generator(
          generator, output_types=(dtypes.int64, dtypes.int64),
          output_shapes=([None], [3]))
              .repeat(num_inner_repeats).prefetch(5))

    iterator = (
        dataset_ops.Dataset.range(num_outer_repeats)
        .interleave(interleave_fn, cycle_length=10,
                    block_length=len(input_list))
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for _ in range(num_inner_repeats * num_outer_repeats):
        for elem in input_list:
          val0, val1 = sess.run(get_next)
          self.assertAllEqual(elem[0], val0)
          self.assertAllEqual(elem[1], val1)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testFromGeneratorsRunningInParallel(self):
    num_parallel_iterators = 3

    # Define shared state that multiple iterator instances will access to
    # demonstrate their concurrent activity.
    lock = threading.Lock()
    condition = threading.Condition(lock)
    next_ticket = [0]  # GUARDED_BY(lock)

    def generator():
      # NOTE(mrry): We yield one element before the barrier, because
      # the current implementation of `Dataset.interleave()` must
      # fetch one element from each incoming dataset to start the
      # prefetching.
      yield 0

      # Define a barrier that `num_parallel_iterators` iterators must enter
      # before any can proceed. Demonstrates that multiple iterators may be
      # active at the same time.
      condition.acquire()
      ticket = next_ticket[0]
      next_ticket[0] += 1
      if ticket == num_parallel_iterators - 1:
        # The last iterator to join the barrier notifies the others.
        condition.notify_all()
      else:
        # Wait until the last iterator enters the barrier.
        while next_ticket[0] < num_parallel_iterators:
          condition.wait()
      condition.release()

      yield 1

    # As in `testFromMultipleConcurrentGenerators()`, we use a combination of
    # `Dataset.interleave()` and `Dataset.prefetch()` to cause multiple
    # iterators to be active concurrently.
    def interleave_fn(_):
      return dataset_ops.Dataset.from_generator(
          generator, output_types=dtypes.int64, output_shapes=[]).prefetch(2)

    iterator = (
        dataset_ops.Dataset.range(num_parallel_iterators)
        .interleave(
            interleave_fn, cycle_length=num_parallel_iterators, block_length=1)
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for elem in [0, 1]:
        for _ in range(num_parallel_iterators):
          self.assertAllEqual(elem, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testFromGeneratorImplicitConversion(self):
    def generator():
      yield [1]
      yield [2]
      yield [3]

    for dtype in [dtypes.int8, dtypes.int32, dtypes.int64]:
      iterator = (dataset_ops.Dataset.from_generator(
          generator, output_types=dtype, output_shapes=[1])
                  .make_initializable_iterator())
      init_op = iterator.initializer
      get_next = iterator.get_next()

      self.assertEqual(dtype, get_next.dtype)

      with self.test_session() as sess:
        sess.run(init_op)
        for expected in [[1], [2], [3]]:
          next_val = sess.run(get_next)
          self.assertEqual(dtype.as_numpy_dtype, next_val.dtype)
          self.assertAllEqual(expected, next_val)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testFromGeneratorTypeError(self):
    def generator():
      yield np.array([1, 2, 3], dtype=np.int64)
      yield np.array([4, 5, 6], dtype=np.int64)
      yield "ERROR"
      yield np.array([7, 8, 9], dtype=np.int64)

    iterator = (dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int64, output_shapes=[3])
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      self.assertAllEqual([1, 2, 3], sess.run(get_next))
      self.assertAllEqual([4, 5, 6], sess.run(get_next))
      with self.assertRaisesOpError(r"invalid literal for long\(\)"):
        sess.run(get_next)
      self.assertAllEqual([7, 8, 9], sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testFromGeneratorShapeError(self):
    def generator():
      yield np.array([1, 2, 3], dtype=np.int64)
      yield np.array([4, 5, 6], dtype=np.int64)
      yield np.array([7, 8, 9, 10], dtype=np.int64)
      yield np.array([11, 12, 13], dtype=np.int64)

    iterator = (dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int64, output_shapes=[3])
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      self.assertAllEqual([1, 2, 3], sess.run(get_next))
      self.assertAllEqual([4, 5, 6], sess.run(get_next))
      with self.assertRaisesOpError(r"element of shape \(3,\) was expected"):
        sess.run(get_next)
      self.assertAllEqual([11, 12, 13], sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testSplitPipelineFailsWithPlacementError(self):
    with session.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:

      dataset = dataset_ops.Dataset.from_tensors(0)

      # Define a pipeline that attempts to use variables on two
      # different devices.
      #
      # Initialize the variables before creating to iterator, to avoid the
      # placement algorithm overriding the DT_RESOURCE colocation constraints.
      with ops.device("/cpu:0"):
        var_0 = resource_variable_ops.ResourceVariable(initial_value=0)
        dataset = dataset.map(lambda x: x + var_0.read_value())
      sess.run(var_0.initializer)

      with ops.device("/cpu:1"):
        var_1 = resource_variable_ops.ResourceVariable(initial_value=0)
        dataset = dataset.map(lambda x: x + var_1.read_value())
      sess.run(var_1.initializer)

      iterator = dataset.make_initializable_iterator()

      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          "Trying to access resource located in device"):
        sess.run(iterator.initializer)

  def testRestructureDataset(self):
    components = (array_ops.placeholder(dtypes.int32),
                  (array_ops.placeholder(dtypes.int32, shape=[None]),
                   array_ops.placeholder(dtypes.int32, shape=[20, 30])))
    dataset = dataset_ops.Dataset.from_tensors(components)

    i32 = dtypes.int32

    test_cases = [((i32, i32, i32), None),
                  (((i32, i32), i32), None),
                  ((i32, i32, i32), (None, None, None)),
                  ((i32, i32, i32), ([17], [17], [20, 30]))]

    for new_types, new_shape_lists in test_cases:
      # pylint: disable=protected-access
      new = batching._RestructuredDataset(dataset, new_types, new_shape_lists)
      # pylint: enable=protected-access
      self.assertEqual(new_types, new.output_types)
      if new_shape_lists is not None:
        for expected_shape_list, shape in zip(
            nest.flatten(new_shape_lists), nest.flatten(new.output_shapes)):
          if expected_shape_list is None:
            self.assertIs(None, shape.ndims)
          else:
            self.assertEqual(expected_shape_list, shape.as_list())

    fail_cases = [((i32, dtypes.int64, i32), None),
                  ((i32, i32, i32, i32), None),
                  ((i32, i32, i32), ((None, None), None)),
                  ((i32, i32, i32), (None, None, None, None)),
                  ((i32, i32, i32), (None, [None], [21, 30]))]

    for new_types, new_shape_lists in fail_cases:
      with self.assertRaises(ValueError):
        # pylint: disable=protected-access
        new = batching._RestructuredDataset(dataset, new_types, new_shape_lists)
        # pylint: enable=protected-access


class DatasetConstructorSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_tensor_dataset(self, variable_array):
    components = (variable_array, np.array([1, 2, 3]), np.array(37.0))

    return dataset_ops.Dataset.from_tensors(components)

  def testFromTensorsCore(self):
    # Equal length components
    arr = np.array(1)
    num_outputs = 1
    diff_arr = np.array(2)
    self.run_core_tests(lambda: self._build_tensor_dataset(arr),
                        lambda: self._build_tensor_dataset(diff_arr),
                        num_outputs)

  def _build_tensor_slices_dataset(self, components):
    return dataset_ops.Dataset.from_tensor_slices(components)

  def testFromTensorSlicesCore(self):
    # Equal length components
    components = (np.tile(np.array([[1], [2], [3], [4]]), 20),
                  np.tile(np.array([[12], [13], [14], [15]]), 22),
                  np.array([37.0, 38.0, 39.0, 40.0]))

    diff_comp = (np.tile(np.array([[1], [2], [3], [4]]), 20),
                 np.tile(np.array([[5], [6], [7], [8]]), 22),
                 np.array([1.0, 2.0, 3.0, 4.0]))

    dict_components = {"foo": [1, 2, 3], "bar": [[4.0], [5.0], [6.0]]}

    self.run_core_tests(lambda: self._build_tensor_slices_dataset(components),
                        lambda: self._build_tensor_slices_dataset(diff_comp), 4)
    self.run_core_tests(
        lambda: self._build_tensor_slices_dataset(dict_components), None, 3)

  def _build_sparse_tensor_slice_dataset(self, slices):
    indices = np.array(
        [[i, j] for i in range(len(slices)) for j in range(len(slices[i]))],
        dtype=np.int64)
    values = np.array([val for s in slices for val in s], dtype=np.float64)
    dense_shape = np.array(
        [len(slices), max(len(s) for s in slices) + 1], dtype=np.int64)
    sparse_components = sparse_tensor.SparseTensor(indices, values, dense_shape)
    return dataset_ops.Dataset.from_sparse_tensor_slices(sparse_components)

  def testFromSparseTensorSlicesCore(self):
    slices = [[1., 2., 3.], [1.], [1.], [1., 2.], [], [1., 2.], [], [], []]
    diff_slices = [[1., 2.], [2.], [2., 3., 4.], [], [], []]

    self.run_core_tests(
        lambda: self._build_sparse_tensor_slice_dataset(slices),
        lambda: self._build_sparse_tensor_slice_dataset(diff_slices),
        9,
        sparse_tensors=True)


if __name__ == "__main__":
  test.main()

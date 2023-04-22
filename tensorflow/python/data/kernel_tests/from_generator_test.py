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
"""Tests for tf.data.Dataset.from_generator()."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


class FromGeneratorTest(test_base.DatasetTestBase, parameterized.TestCase):

  def _testFromGenerator(self, generator, elem_sequence, num_repeats,
                         requires_initialization):
    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int64).repeat(num_repeats).prefetch(5)
    self.assertDatasetProduces(
        dataset,
        elem_sequence * num_repeats,
        requires_initialization=requires_initialization,
        num_test_iterations=2)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_repeats=[1, 2], requires_initialization=[True, False])))
  def testFromGeneratorUsingFn(self, num_repeats, requires_initialization):

    def generator():
      for i in range(1, 100):
        yield [i] * i

    elem_sequence = list(generator())
    self._testFromGenerator(
        generator,
        elem_sequence,
        num_repeats=num_repeats,
        requires_initialization=requires_initialization)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_repeats=[1, 2], requires_initialization=[True, False])))
  def testFromGeneratorUsingList(self, num_repeats, requires_initialization):
    generator = lambda: [[i] * i for i in range(1, 100)]
    elem_sequence = list(generator())
    self._testFromGenerator(
        generator,
        elem_sequence,
        num_repeats=num_repeats,
        requires_initialization=requires_initialization)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_repeats=[1, 2], requires_initialization=[True, False])))
  def testFromGeneratorUsingNdarray(self, num_repeats, requires_initialization):
    generator = lambda: np.arange(100, dtype=np.int64)
    elem_sequence = list(generator())
    self._testFromGenerator(
        generator,
        elem_sequence,
        num_repeats=num_repeats,
        requires_initialization=requires_initialization)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_repeats=[1, 2], requires_initialization=[True, False])))
  def testFromGeneratorUsingGeneratorExpression(self, num_repeats,
                                                requires_initialization):
    # NOTE(mrry): Generator *expressions* are not repeatable (or in general
    # reusable), because they eagerly evaluate the `for` expression as
    # `iter(range(1, 100))` and discard the means of reconstructing
    # `range(1, 100)`. Wrapping the generator expression in a `lambda` makes
    # it repeatable.
    generator = lambda: ([i] * i for i in range(1, 100))
    elem_sequence = list(generator())
    self._testFromGenerator(
        generator,
        elem_sequence,
        num_repeats=num_repeats,
        requires_initialization=requires_initialization)

  @combinations.generate(test_base.default_test_combinations())
  def testFromMultipleConcurrentGenerators(self):
    num_inner_repeats = 5
    num_outer_repeats = 20

    def generator():
      for i in range(1, 10):
        yield ([i] * i, [i, i ** 2, i ** 3])
    input_list = list(generator())

    # The interleave transformation is essentially a flat map that draws from
    # multiple input datasets concurrently (in a cyclic fashion). By placing
    # `Dataset.from_generator()` inside an interleave, we test its behavior when
    # multiple iterators are active at the same time; by additionally
    # prefetching inside the interleave, we create the possibility of concurrent
    # invocations to several iterators created by the same dataset.
    def interleave_fn(_):
      return (dataset_ops.Dataset.from_generator(
          generator, output_types=(dtypes.int64, dtypes.int64),
          output_shapes=([None], [3]))
              .repeat(num_inner_repeats).prefetch(5))

    dataset = dataset_ops.Dataset.range(num_outer_repeats).interleave(
        interleave_fn, cycle_length=10, block_length=len(input_list))
    get_next = self.getNext(dataset)
    for _ in range(num_inner_repeats * num_outer_repeats):
      for elem in input_list:
        val0, val1 = self.evaluate(get_next())
        self.assertAllEqual(elem[0], val0)
        self.assertAllEqual(elem[1], val1)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  def DISABLED_testFromGeneratorsRunningInParallel(self):
    self.skipTest("b/67868766")
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

    dataset = dataset_ops.Dataset.range(num_parallel_iterators).interleave(
        interleave_fn, cycle_length=num_parallel_iterators, block_length=1)
    get_next = self.getNext(dataset)

    for elem in [0, 1]:
      for _ in range(num_parallel_iterators):
        self.assertAllEqual(elem, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorImplicitConversion(self):
    def generator():
      yield [1]
      yield [2]
      yield [3]

    for dtype in [dtypes.int8, dtypes.int32, dtypes.int64]:
      dataset = dataset_ops.Dataset.from_generator(
          generator, output_types=dtype, output_shapes=[1])
      get_next = self.getNext(dataset)

      for expected in [[1], [2], [3]]:
        next_val = self.evaluate(get_next())
        self.assertEqual(dtype.as_numpy_dtype, next_val.dtype)
        self.assertAllEqual(expected, next_val)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorString(self):
    def generator():
      yield "foo"
      yield b"bar"
      yield u"baz"

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.string, output_shapes=[])
    self.assertDatasetProduces(
        dataset, expected_output=[b"foo", b"bar", b"baz"])

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorDatastructures(self):
    # Tests multiple datastructures.
    def generator():
      yield {"a": "foo", "b": [1, 2], "c": (9,)}
      yield {"a": "bar", "b": [3], "c": (7, 6)}
      yield {"a": "baz", "b": [5, 6], "c": (5, 4)}

    dataset = dataset_ops.Dataset.from_generator(
        generator,
        output_types={"a": dtypes.string, "b": dtypes.int32, "c": dtypes.int32},
        output_shapes={"a": [], "b": [None], "c": [None]})
    self.assertDatasetProduces(
        dataset,
        expected_output=[{"a": b"foo", "b": [1, 2], "c": [9]},
                         {"a": b"bar", "b": [3], "c": [7, 6]},
                         {"a": b"baz", "b": [5, 6], "c": [5, 4]}])

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorTypeError(self):
    def generator():
      yield np.array([1, 2, 3], dtype=np.int64)
      yield np.array([4, 5, 6], dtype=np.int64)
      yield "ERROR"
      yield np.array([7, 8, 9], dtype=np.int64)

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int64, output_shapes=[3])

    get_next = self.getNext(dataset)

    self.assertAllEqual([1, 2, 3], self.evaluate(get_next()))
    self.assertAllEqual([4, 5, 6], self.evaluate(get_next()))
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())
    self.assertAllEqual([7, 8, 9], self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorShapeError(self):
    def generator():
      yield np.array([1, 2, 3], dtype=np.int64)
      yield np.array([4, 5, 6], dtype=np.int64)
      yield np.array([7, 8, 9, 10], dtype=np.int64)
      yield np.array([11, 12, 13], dtype=np.int64)

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int64, output_shapes=[3])
    get_next = self.getNext(dataset)

    self.assertAllEqual([1, 2, 3], self.evaluate(get_next()))
    self.assertAllEqual([4, 5, 6], self.evaluate(get_next()))
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())
    self.assertAllEqual([11, 12, 13], self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorStructureError(self):
    def generator():
      yield 1, 2
      yield 3, 4
      yield 5
      yield 6, 7, 8
      yield 9, 10

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=(dtypes.int64, dtypes.int64))
    get_next = self.getNext(dataset)

    self.assertEqual((1, 2), self.evaluate(get_next()))
    self.assertEqual((3, 4), self.evaluate(get_next()))
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())
    self.assertEqual((9, 10), self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorHeterogeneous(self):
    def generator():
      yield 1
      yield [2, 3]

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int64)
    self.assertDatasetProduces(dataset, expected_output=[1, [2, 3]])

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorStopShort(self):

    def generator():
      yield 0
      yield 1
      yield 2

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int64)
    get_next = self.getNext(dataset)
    self.assertAllEqual(0, self.evaluate(get_next()))
    self.assertAllEqual(1, self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorDestructorCalled(self):
    # Use an `Event` to signal that the generator has been deleted.
    event = threading.Event()

    class GeneratorWrapper(object):

      def __iter__(self):
        return self

      def next(self):
        return self.__next__()

      def __next__(self):
        return 42

      def __del__(self):
        event.set()

    dataset = dataset_ops.Dataset.from_generator(
        GeneratorWrapper, output_types=dtypes.int64).take(2)
    get_next = self.getNext(dataset)

    self.assertAllEqual(42, self.evaluate(get_next()))
    self.assertAllEqual(42, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    # Test that `GeneratorWrapper` object is destroyed when the
    # iterator terminates (and the generator iterator is deleted).
    self.assertTrue(event.is_set())

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorWithArgs(self):

    def flat_map_fn(elem):

      def generator_with_arg(n):
        for _ in range(n):
          yield np.array(n, dtype=np.int64)

      return dataset_ops.Dataset.from_generator(
          generator_with_arg, output_types=dtypes.int64, output_shapes=(),
          args=(elem,))

    dataset = dataset_ops.Dataset.range(5).flat_map(flat_map_fn)
    self.assertDatasetProduces(
        dataset, expected_output=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorWithTwoArgs(self):

    def flat_map_fn(elem, message):

      def generator_with_arg(n, msg):
        for i in range(n):
          yield i, msg

      return dataset_ops.Dataset.from_generator(
          generator_with_arg, output_types=(dtypes.int64, dtypes.string),
          output_shapes=((), ()), args=(elem, message))

    dataset = dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.range(5),
         dataset_ops.Dataset.from_tensors("Hi!").repeat(None)
        )).flat_map(flat_map_fn)

    self.assertDatasetProduces(
        dataset,
        expected_output=[(0, b"Hi!"), (0, b"Hi!"), (1, b"Hi!"), (0, b"Hi!"),
                         (1, b"Hi!"), (2, b"Hi!"), (0, b"Hi!"), (1, b"Hi!"),
                         (2, b"Hi!"), (3, b"Hi!")])

  @combinations.generate(test_base.default_test_combinations())
  def testGeneratorDatasetFinalizeFunctionCalled(self):
    # NOTE(mrry): This test tests the internal `_GeneratorDataset`,
    # which affords more control over what the finalize function can do than
    # the `Dataset.from_generator()` wrapper.

    # Use an `Event` to signal that the generator has been deleted.
    event = threading.Event()

    def finalize_fn(_):
      def finalize_py_func():
        event.set()
        return 0
      return script_ops.py_func(finalize_py_func, [], [dtypes.int64],
                                stateful=True)

    dummy = constant_op.constant(37)

    dataset = dataset_ops._GeneratorDataset(
        dummy, lambda x: x, lambda x: x, finalize_fn,
        tensor_spec.TensorSpec((), dtypes.int32))

    dataset = dataset.take(2)
    get_next = self.getNext(dataset)

    self.assertAllEqual(37, self.evaluate(get_next()))
    self.assertAllEqual(37, self.evaluate(get_next()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testSharedName(self):

    def generator():
      for _ in range(10):
        yield [20]

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=(dtypes.int64))
    get_next = self.getNext(
        dataset, requires_initialization=True, shared_name="shared_dataset")

    self.assertAllEqual([20], self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorRaggedTensor(self):

    def generator():
      yield ragged_factory_ops.constant([[1, 2], [3]])

    dataset = dataset_ops.Dataset.from_generator(
        generator,
        output_signature=ragged_tensor.RaggedTensorSpec(
            shape=(2, None), dtype=dtypes.int32))
    get_next = self.getNext(dataset)

    ret = get_next()

    self.assertIsInstance(ret, ragged_tensor.RaggedTensor)
    self.assertAllEqual([[1, 2], [3]], ret)

  @combinations.generate(test_base.default_test_combinations())
  def testFromGeneratorSparseTensor(self):

    def generator():
      yield sparse_tensor.SparseTensor(
          indices=[[0, 0], [1, 2]],
          values=constant_op.constant([1, 2], dtype=dtypes.int64),
          dense_shape=[3, 4])

    dataset = dataset_ops.Dataset.from_generator(
        generator,
        output_signature=sparse_tensor.SparseTensorSpec([3, 4], dtypes.int64))

    get_next = self.getNext(dataset)

    ret = get_next()

    self.assertIsInstance(ret, sparse_tensor.SparseTensor)
    self.assertAllEqual([[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]],
                        sparse_ops.sparse_tensor_to_dense(ret))

  @combinations.generate(test_base.default_test_combinations())
  def testTypeIsListError(self):

    def generator():
      for _ in range(10):
        yield [20]

    with self.assertRaisesRegex(
        TypeError, r"Cannot convert value \[tf.int64\] to a TensorFlow DType"):
      dataset_ops.Dataset.from_generator(
          generator, output_types=[dtypes.int64])

  @combinations.generate(test_base.default_test_combinations())
  def testDimensionIsListError(self):

    def generator():
      for _ in range(10):
        yield [20]

    with self.assertRaisesRegex(TypeError,
                                r"Dimension value must be integer or None"):
      dataset_ops.Dataset.from_generator(
          generator, output_types=(dtypes.int64), output_shapes=[[1]])


if __name__ == "__main__":
  test.main()

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

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


class DatasetConstructorTest(test.TestCase):

  def _testFromGenerator(self, generator, elem_sequence, num_repeats,
                         output_types=None):
    if output_types is None:
      output_types = dtypes.int64
    iterator = (
        dataset_ops.Dataset.from_generator(generator, output_types=output_types)
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
    self._testFromGenerator(generator, elem_sequence, 1, output_types=np.int64)
    self._testFromGenerator(generator, elem_sequence, 5, output_types=np.int64)

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

  # TODO(b/67868766): Reenable this when the source of flakiness is discovered.
  def _testFromGeneratorsRunningInParallel(self):
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

  def testFromGeneratorString(self):
    def generator():
      yield "foo"
      yield b"bar"
      yield u"baz"

    iterator = (dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.string, output_shapes=[])
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for expected in [b"foo", b"bar", b"baz"]:
        next_val = sess.run(get_next)
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
      # NOTE(mrry): Type name in message differs between Python 2 (`long`) and
      # 3 (`int`).
      with self.assertRaisesOpError(r"invalid literal for"):
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

  def testFromGeneratorHeterogeneous(self):
    def generator():
      yield 1
      yield [2, 3]

    iterator = (
        dataset_ops.Dataset.from_generator(
            generator, output_types=dtypes.int64).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      self.assertAllEqual(1, sess.run(get_next))
      self.assertAllEqual([2, 3], sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testFromGeneratorStopShort(self):

    def generator():
      yield 0
      yield 1
      yield 2

    iterator = (
        dataset_ops.Dataset.from_generator(
            generator, output_types=dtypes.int64).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      self.assertAllEqual(0, sess.run(get_next))
      self.assertAllEqual(1, sess.run(get_next))

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

    iterator = dataset_ops.Dataset.from_generator(
        GeneratorWrapper,
        output_types=dtypes.int64).take(2).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with session.Session() as sess:
      sess.run(init_op)
      self.assertAllEqual(42, sess.run(get_next))
      self.assertAllEqual(42, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)
      # Test that `GeneratorWrapper` object is destroyed when the
      # iterator terminates (and the generator iterator is deleted).
      self.assertTrue(event.is_set())

  def testFromGeneratorWithArgs(self):

    def flat_map_fn(elem):

      def generator_with_arg(n):
        for _ in range(n):
          yield np.array(n, dtype=np.int64)

      return dataset_ops.Dataset.from_generator(
          generator_with_arg, output_types=dtypes.int64, output_shapes=(),
          args=(elem,))

    iterator = (dataset_ops.Dataset
                .range(5)
                .flat_map(flat_map_fn)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      expected = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
      for x in expected:
        self.assertEqual(x, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testFromGeneratorWithTwoArgs(self):

    def flat_map_fn(elem, message):

      def generator_with_arg(n, msg):
        for i in range(n):
          yield i, msg

      return dataset_ops.Dataset.from_generator(
          generator_with_arg, output_types=(dtypes.int64, dtypes.string),
          output_shapes=((), ()), args=(elem, message))

    iterator = (
        dataset_ops.Dataset.zip(
            (dataset_ops.Dataset.range(5),
             dataset_ops.Dataset.from_tensors("Hi!").repeat(None)))
        .flat_map(flat_map_fn)
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      expected = [(0, b"Hi!"),
                  (0, b"Hi!"), (1, b"Hi!"),
                  (0, b"Hi!"), (1, b"Hi!"), (2, b"Hi!"),
                  (0, b"Hi!"), (1, b"Hi!"), (2, b"Hi!"), (3, b"Hi!")]
      for x in expected:
        self.assertEqual(x, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

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
    iterator = (dataset_ops._GeneratorDataset(dummy, lambda x: x,
                                              lambda x: x, finalize_fn)
                .take(2)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      self.assertAllEqual(37, sess.run(get_next))
      self.assertAllEqual(37, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)
        self.assertTrue(event.is_set())


if __name__ == "__main__":
  test.main()

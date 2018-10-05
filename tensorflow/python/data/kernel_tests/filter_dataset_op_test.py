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

import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class FilterDatasetTest(test_base.DatasetTestBase):

  def testFilterDataset(self):
    components = (
        np.arange(7, dtype=np.int64),
        np.array([[1, 2, 3]], dtype=np.int64) * np.arange(
            7, dtype=np.int64)[:, np.newaxis],
        np.array(37.0, dtype=np.float64) * np.arange(7)
    )
    count = array_ops.placeholder(dtypes.int64, shape=[])
    modulus = array_ops.placeholder(dtypes.int64)

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    iterator = (
        dataset_ops.Dataset.from_tensor_slices(components).map(_map_fn)
        .repeat(count)
        .filter(lambda x, _y, _z: math_ops.equal(math_ops.mod(x, modulus), 0))
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([c.shape[1:] for c in components],
                     [t.shape for t in get_next])

    with self.cached_session() as sess:
      # Test that we can dynamically feed a different modulus value for each
      # iterator.
      def do_test(count_val, modulus_val):
        sess.run(init_op, feed_dict={count: count_val, modulus: modulus_val})
        for _ in range(count_val):
          for i in [x for x in range(7) if x**2 % modulus_val == 0]:
            result = sess.run(get_next)
            for component, result_component in zip(components, result):
              self.assertAllEqual(component[i]**2, result_component)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

      do_test(14, 2)
      do_test(4, 18)

      # Test an empty dataset.
      do_test(0, 1)

  def testFilterRange(self):
    dataset = dataset_ops.Dataset.range(100).filter(
        lambda x: math_ops.not_equal(math_ops.mod(x, 3), 2))
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      self.assertEqual(0, sess.run(get_next))
      self.assertEqual(1, sess.run(get_next))
      self.assertEqual(3, sess.run(get_next))

  def testFilterDict(self):
    iterator = (dataset_ops.Dataset.range(10)
                .map(lambda x: {"foo": x * 2, "bar": x ** 2})
                .filter(lambda d: math_ops.equal(d["bar"] % 2, 0))
                .map(lambda d: d["foo"] + d["bar"])
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(10):
        if (i ** 2) % 2 == 0:
          self.assertEqual(i * 2 + i ** 2, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testUseStepContainerInFilter(self):
    input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

    # Define a predicate that returns true for the first element of
    # the sequence and not the second, and uses `tf.map_fn()`.
    def _predicate(xs):
      squared_xs = functional_ops.map_fn(lambda x: x * x, xs)
      summed = math_ops.reduce_sum(squared_xs)
      return math_ops.equal(summed, 1 + 4 + 9)

    iterator = (
        dataset_ops.Dataset.from_tensor_slices([[1, 2, 3], [4, 5, 6]])
        .filter(_predicate)
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      self.assertAllEqual(input_data[0], sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testSparse(self):

    def _map_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1])), i

    def _filter_fn(_, i):
      return math_ops.equal(i % 2, 0)

    iterator = (
        dataset_ops.Dataset.range(10).map(_map_fn).filter(_filter_fn).map(
            lambda x, i: x).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(5):
        actual = sess.run(get_next)
        self.assertTrue(isinstance(actual, sparse_tensor.SparseTensorValue))
        self.assertSparseValuesEqual(actual, _map_fn(i * 2)[0])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testReturnComponent(self):
    iterator = (
        dataset_ops.Dataset.zip(
            (dataset_ops.Dataset.range(10),
             dataset_ops.Dataset.from_tensors(True).repeat(None)))
        .filter(lambda x, y: y).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(10):
        self.assertEqual((i, True), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testParallelFilters(self):
    dataset = dataset_ops.Dataset.range(10).filter(
        lambda x: math_ops.equal(x % 2, 0))
    iterators = [dataset.make_one_shot_iterator() for _ in range(10)]
    next_elements = [iterator.get_next() for iterator in iterators]
    with self.cached_session() as sess:
      self.assertEqual([0 for _ in range(10)], sess.run(next_elements))


class FilterDatasetBenchmark(test.Benchmark):

  def _benchmark(self, predicate, name):
    with ops.Graph().as_default():
      dataset = (
          dataset_ops.Dataset.from_tensors(True).repeat(None).filter(predicate))
      iterator = dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

      with session.Session() as sess:
        for _ in range(5):
          sess.run(next_element.op)
        deltas = []
        for _ in range(100):
          start = time.time()
          for _ in range(100):
            sess.run(next_element.op)
          end = time.time()
          deltas.append(end - start)

        median_wall_time = np.median(deltas) / 100
        print("Filter dataset using %s. Median wall time: %f" %
              (name, median_wall_time))
        self.report_benchmark(
            iters=100,
            wall_time=median_wall_time,
            name="benchmark_filter_dataset_%s" % name)

  def benchmarkSimpleFunction(self):
    self._benchmark(array_ops.identity, "simple_function")

  def benchmarkReturnComponentOptimization(self):
    self._benchmark(lambda x: x, "return_component")


if __name__ == "__main__":
  test.main()

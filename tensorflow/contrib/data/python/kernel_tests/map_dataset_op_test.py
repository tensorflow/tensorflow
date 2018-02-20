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

import os

import numpy as np

from tensorflow.contrib.data.python.kernel_tests import dataset_serialization_test_base
from tensorflow.contrib.data.python.ops import error_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class MapDatasetTest(test.TestCase):

  def testMapIgnoreError(self):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(components)
        .map(lambda x: array_ops.check_numerics(x, "message")).apply(
            error_ops.ignore_errors()))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for x in [1., 2., 3., 5.]:
        self.assertEqual(x, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testParallelMapIgnoreError(self):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(components).map(
            lambda x: array_ops.check_numerics(x, "message"),
            num_parallel_calls=2).prefetch(2).apply(error_ops.ignore_errors()))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for x in [1., 2., 3., 5.]:
        self.assertEqual(x, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testReadFileIgnoreError(self):
    def write_string_to_file(value, filename):
      with open(filename, "w") as f:
        f.write(value)
    filenames = [os.path.join(self.get_temp_dir(), "file_%d.txt" % i)
                 for i in range(5)]
    for filename in filenames:
      write_string_to_file(filename, filename)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(filenames).map(
            io_ops.read_file, num_parallel_calls=2).prefetch(2).apply(
                error_ops.ignore_errors()))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      # All of the files are present.
      sess.run(init_op)
      for filename in filenames:
        self.assertEqual(compat.as_bytes(filename), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Delete one of the files.
      os.remove(filenames[0])

      # Attempting to read filenames[0] will fail, but ignore_errors()
      # will catch the error.
      sess.run(init_op)
      for filename in filenames[1:]:
        self.assertEqual(compat.as_bytes(filename), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testCaptureResourceInMapFn(self):

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
      with self.test_session(graph=g) as sess:
        sess.run(captured_init_op)
        sess.run(init_op)
        for i in range(10):
          self.assertEquals(i * i, sess.run(get_next))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)


class MapDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def setUp(self):
    self._tensor_slice_len = 7
    self._num_epochs = 14
    self._num_outputs = self._tensor_slice_len * self._num_epochs

  def _build_ds(self, multiplier=37.0):
    components = (np.arange(self._tensor_slice_len), np.array([[1, 2, 3]]) *
                  np.arange(self._tensor_slice_len)[:, np.newaxis],
                  np.array(multiplier) * np.arange(self._tensor_slice_len))

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    return (
        dataset_ops.Dataset.from_tensor_slices(components).map(_map_fn)
        .repeat(self._num_epochs))

  def testSaveRestoreCore(self):
    self.run_core_tests(
        self._build_ds,
        lambda: self._build_ds(multiplier=15.0),
        self._num_outputs)

  def testSaveStatefulFunction(self):

    def _build_ds():

      def _map_fn(x):
        return random_ops.random_uniform(
            (), 0, 10, dtype=dtypes.int32) * math_ops.to_int32(x)

      return dataset_ops.Dataset.range(100).map(_map_fn)

    self.verify_error_on_save(_build_ds, 15, errors.InvalidArgumentError)

  def testCaptureVariableInMapFn(self):

    def _build_ds():
      counter_var = variable_scope.get_variable(
          "counter", (), dtypes.int32, use_resource=True)
      return (dataset_ops.Dataset.from_tensors(0).repeat(10).map(
          lambda _: counter_var.assign_add(1)))

    self.verify_error_on_save(_build_ds, 15, errors.InvalidArgumentError)

  def testCaptureConstantInMapFn(self):

    def _build_ds():
      constant_var = constant_op.constant(5)
      return (dataset_ops.Dataset.from_tensors(0).repeat(10).map(
          lambda x: x + constant_var))

    self.run_core_tests(_build_ds, None, 10)

  def testCaptureDefunInMapFn(self):
    num_outputs = 100

    def _build_ds():

      @function.Defun(dtypes.int64)
      def defun_fn(x):
        return constant_op.constant(1000) + math_ops.to_int32(x)

      return dataset_ops.Dataset.range(num_outputs).map(defun_fn)

    self.run_core_tests(_build_ds, None, num_outputs)

  def testBuildDefunInMapFn(self):
    num_outputs = 100

    def _build_ds():

      @function.Defun(dtypes.int64)
      def defun_fn(x):

        @function.Defun(dtypes.int32)
        def defun_fn_deep(x):
          return constant_op.constant(1000) + math_ops.to_int32(x)

        return constant_op.constant(11000) + defun_fn_deep(math_ops.to_int32(x))

      return dataset_ops.Dataset.range(num_outputs).map(defun_fn)

    self.run_core_tests(_build_ds, None, num_outputs)

  def testSparseCore(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    def _build_ds(num_outputs):
      return dataset_ops.Dataset.range(num_outputs).map(_sparse)

    num_outputs = 10
    self.run_core_tests(lambda: _build_ds(num_outputs),
                        lambda: _build_ds(int(num_outputs / 2)), num_outputs)


class ParallelMapDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def setUp(self):
    self._tensor_slice_len = 7
    self._num_epochs = 1
    self._num_outputs = self._tensor_slice_len * self._num_epochs

  def _build_ds(self, multiplier=37.0):
    components = (np.arange(self._tensor_slice_len), np.array([[1, 2, 3]]) *
                  np.arange(self._tensor_slice_len)[:, np.newaxis],
                  np.array(multiplier) * np.arange(self._tensor_slice_len))

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    return (dataset_ops.Dataset.from_tensor_slices(components).map(
        _map_fn, num_parallel_calls=3).repeat(self._num_epochs))

  def _build_ds_with_prefetch(self, multiplier=37.0):
    components = (np.arange(self._tensor_slice_len), np.array([[1, 2, 3]]) *
                  np.arange(self._tensor_slice_len)[:, np.newaxis],
                  np.array(multiplier) * np.arange(self._tensor_slice_len))

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    return (dataset_ops.Dataset.from_tensor_slices(components).map(
        _map_fn, num_parallel_calls=3).repeat(self._num_epochs).prefetch(5))

  def testSaveRestoreCore(self):
    for ds_fn in [self._build_ds, self._build_ds_with_prefetch]:
      self.run_core_tests(
          ds_fn,
          lambda: ds_fn(multiplier=15.0),
          self._num_outputs)

  def testSaveStatefulFunction(self):

    def _build_ds():

      def _map_fn(x):
        return random_ops.random_uniform(
            (), 0, 10, dtype=dtypes.int32) * math_ops.to_int32(x)

      return dataset_ops.Dataset.range(100).map(
          _map_fn, num_parallel_calls=2).prefetch(2)

    self.verify_error_on_save(_build_ds, 15, errors.InvalidArgumentError)

  def testCaptureVariableInMapFn(self):

    def _build_ds():
      counter_var = variable_scope.get_variable(
          "counter", (), dtypes.int32, use_resource=True)
      return (dataset_ops.Dataset.from_tensors(0).repeat(10).map(
          lambda _: counter_var.assign_add(1),
          num_parallel_calls=2).prefetch(2))

    self.verify_error_on_save(_build_ds, 15, errors.InvalidArgumentError)

  def testCaptureConstantInMapFn(self):

    def _build_ds():
      constant_var = constant_op.constant(5)
      return (dataset_ops.Dataset.from_tensors(0).repeat(10).map(
          lambda x: x + constant_var, num_parallel_calls=2).prefetch(2))

    self.run_core_tests(_build_ds, None, 10)

  def testCaptureDefunInMapFn(self):
    num_outputs = 100

    def _build_ds():

      @function.Defun(dtypes.int64)
      def defun_fn(x):
        return constant_op.constant(1000) + math_ops.to_int32(x)

      return dataset_ops.Dataset.range(num_outputs).map(
          defun_fn, num_parallel_calls=2).prefetch(2)

    self.run_core_tests(_build_ds, None, num_outputs)

  def testBuildDefunInMapFn(self):
    num_outputs = 100

    def _build_ds():

      @function.Defun(dtypes.int64)
      def defun_fn(x):

        @function.Defun(dtypes.int32)
        def defun_fn_deep(x):
          return constant_op.constant(1000) + math_ops.to_int32(x)

        return constant_op.constant(11000) + defun_fn_deep(math_ops.to_int32(x))

      return dataset_ops.Dataset.range(num_outputs).map(
          defun_fn, num_parallel_calls=2).prefetch(2)

    self.run_core_tests(_build_ds, None, num_outputs)


class IgnoreErrorsSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_ds(self, components):
    return dataset_ops.Dataset.from_tensor_slices(components).map(
        lambda x: array_ops.check_numerics(x, "message")).apply(
            error_ops.ignore_errors())

  def testIgnoreErrorsCore(self):
    components = np.array([1., 2., 3., np.nan, 5.]).astype(np.float32)
    diff_components = np.array([1., 2., 3., np.nan]).astype(np.float32)
    num_outputs = 4
    self.run_core_tests(lambda: self._build_ds(components),
                        lambda: self._build_ds(diff_components), num_outputs)


if __name__ == "__main__":
  test.main()

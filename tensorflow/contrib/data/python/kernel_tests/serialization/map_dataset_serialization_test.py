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
"""Tests for the MapDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.data.python.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


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


if __name__ == "__main__":
  test.main()

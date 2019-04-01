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
"""Tests for the ParallelMapDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


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
          lambda: ds_fn(multiplier=15.0),  # pylint: disable=cell-var-from-loop
          self._num_outputs)

  def testSaveStatefulFunction(self):

    def _build_ds():

      def _map_fn(x):
        return random_ops.random_uniform(
            (), 0, 10, dtype=dtypes.int32) * math_ops.cast(x, dtypes.int32)

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
        return constant_op.constant(1000) + math_ops.cast(x, dtypes.int32)

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
          return constant_op.constant(1000) + math_ops.cast(x, dtypes.int32)

        return constant_op.constant(11000) + defun_fn_deep(
            math_ops.cast(x, dtypes.int32))

      return dataset_ops.Dataset.range(num_outputs).map(
          defun_fn, num_parallel_calls=2).prefetch(2)

    self.run_core_tests(_build_ds, None, num_outputs)


if __name__ == "__main__":
  test.main()

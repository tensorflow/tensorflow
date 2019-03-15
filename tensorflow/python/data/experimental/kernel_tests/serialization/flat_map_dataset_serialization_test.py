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
"""Tests for the FlatMapDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


class FlatMapDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def testCore(self):
    # Complicated way of saying range(start, start+25).
    def build_ds(start):

      def map_fn(x):
        return dataset_ops.Dataset.range(x, x + 5)

      return dataset_ops.Dataset.range(start, start + 5 * 5, 5).flat_map(map_fn)

    self.run_core_tests(lambda: build_ds(0), lambda: build_ds(10), 25)

  def testMapThenFlatMap(self):

    def build_ds():

      def flat_map_fn(_):

        def map_fn(y):
          return 10 * math_ops.cast(y, dtypes.int32)

        return dataset_ops.Dataset.range(100).map(map_fn)

      return dataset_ops.Dataset.range(5).flat_map(flat_map_fn)

    self.run_core_tests(build_ds, None, 500)

  def testCaptureDefunInMapFn(self):

    def build_ds():

      def map_fn(x):

        @function.Defun(dtypes.int64)
        def defun_fn(x):
          return constant_op.constant(1000) + math_ops.cast(x, dtypes.int32)

        return dataset_ops.Dataset.from_tensor_slices([defun_fn(x)])

      return dataset_ops.Dataset.range(100).flat_map(map_fn)

    self.run_core_tests(build_ds, None, 100)

  def testDisallowVariableCapture(self):

    def build_ds():
      test_var = variable_scope.get_variable(
          name="test_var", shape=(), use_resource=True)
      return dataset_ops.Dataset.range(5).flat_map(
          lambda _: dataset_ops.Dataset.from_tensor_slices([test_var]))

    self.verify_error_on_save(build_ds, 5, errors.InvalidArgumentError)

  def testDisallowCapturingStatefulOps(self):

    def build_ds():

      def flat_map_fn(_):

        def map_fn(x):
          return random_ops.random_uniform(
              (), 0, 10, dtype=dtypes.int32) * math_ops.cast(x, dtypes.int32)

        return dataset_ops.Dataset.range(100).map(map_fn)

      return dataset_ops.Dataset.range(5).flat_map(flat_map_fn)

    self.verify_error_on_save(build_ds, 500, errors.InvalidArgumentError)

  def testSparseCore(self):

    def _map_fn(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0, 0], [1, 1]], values=(i * [1, -1]), dense_shape=[2, 2])

    def _flat_map_fn(x):
      return dataset_ops.Dataset.from_tensor_slices(
          sparse_ops.sparse_to_dense(x.indices, x.dense_shape, x.values))

    def _build_ds():
      return dataset_ops.Dataset.range(10).map(_map_fn).flat_map(_flat_map_fn)

    self.run_core_tests(_build_ds, None, 20)


if __name__ == "__main__":
  test.main()

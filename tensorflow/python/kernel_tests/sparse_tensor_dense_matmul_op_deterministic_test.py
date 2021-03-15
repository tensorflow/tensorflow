# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# ========================================================================
"""Functional tests for deterministic SparseTensorDenseMatMul."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests import sparse_tensor_dense_matmul_op_base
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class SparseTensorDenseMatmulDeterministicTest(
    sparse_tensor_dense_matmul_op_base.SparseTensorDenseMatMulTestBase):

  def _gen_data(self, m, k, n, nnz, row_occupied_rate, data_type):
    random.seed(123)
    np.random.seed(123)

    occupied_rows = random.sample(range(m), int(m * row_occupied_rate))
    sparse_input_dense_shape = [m, k]
    dense_input_shape = (k, n)
    indices = []

    for _ in range(nnz):
      row = random.choice(occupied_rows)
      col = random.randint(0, k-1)
      indices.append([row, col])

    sparse_values = sparse_tensor_dense_matmul_op_base._maybe_complex(
        np.random.normal(size=len(indices)).astype(data_type))

    dense_values = sparse_tensor_dense_matmul_op_base._maybe_complex(
        np.random.normal(size=dense_input_shape).astype(data_type))

    sparse_input = sparse_tensor.SparseTensor(
        indices, sparse_values, sparse_input_dense_shape)

    dense_input = constant_op.constant(dense_values)

    return sparse_input, dense_input

  @test_util.run_cuda_only
  @test_util.run_in_graph_and_eager_modes
  def testDeterministicSparseDenseMatmul(self):
    types_list = (np.float16, np.float32, np.float64,
                  np.complex64, np.complex128)
    unimplemented_types = (np.float64, np.complex128)

    for data_type in types_list:
      sparse_input, dense_input = self._gen_data(
          m=2430, k=615, n=857, nnz=(1<<16)+243, row_occupied_rate=0.02,
          data_type=data_type)

      repeat_count = 5
      with self.session(force_gpu=True):

        if data_type in unimplemented_types:
          with self.assertRaisesRegex(
              errors.UnimplementedError,
              "No deterministic GPU implementation of *"):
            result_ = sparse_ops.sparse_tensor_dense_matmul(
                sparse_input, dense_input)
            self.evaluate(result_)
        else:
          result_a = sparse_ops.sparse_tensor_dense_matmul(sparse_input,
                                                           dense_input)
          for _ in range(repeat_count):
            result_b = sparse_ops.sparse_tensor_dense_matmul(sparse_input,
                                                             dense_input)
            self.assertAllEqual(result_a, result_b)


if __name__ == "__main__":
  # Note that the effect of setting the following environment variable to
  # 'true' is not tested. Unless we can find a simpler pattern for testing these
  # environment variables, it would require this file to be made into a base
  # and then two more test files to be created.
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  test.main()

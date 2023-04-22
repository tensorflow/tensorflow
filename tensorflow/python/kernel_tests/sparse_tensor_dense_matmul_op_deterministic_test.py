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
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


def _gen_data(m, k, n, nnz, row_occupied_rate, data_type, seed):
  """Generate valid input data for tf.sparse.sparse_dense_matmul

  returns sparse matrix a (type SparseTensor), dense matrix b (type Tensor)

  Parameters:
    m: row count of dense version of matrix a / row count of output matrix
    k: col count of dense version of matrix a / row count of matrix b
    n: col could of matrix b / col count of output matrix
    nnz: number of non-zero elements in matrix a
    row_occupied_rate: prob that row in a has one or more non-zero element
  """
  random.seed(seed)
  np.random.seed(seed)
  occupied_rows = random.sample(range(m), int(m * row_occupied_rate))
  sparse_input_dense_shape = [m, k]
  dense_input_shape = (k, n)
  indices = []
  for _ in range(nnz):
    row = random.choice(occupied_rows)
    col = random.randint(0, k - 1)
    indices.append([row, col])

  def maybe_complex(x):
    if x.dtype.kind == "c":  # complex
      return (x + 1j * x) / 2
    return x

  sparse_values = maybe_complex(
      np.random.normal(size=len(indices)).astype(data_type))
  dense_values = maybe_complex(
      np.random.normal(size=dense_input_shape).astype(data_type))
  sparse_input = sparse_tensor.SparseTensor(indices, sparse_values,
                                            sparse_input_dense_shape)
  dense_input = constant_op.constant(dense_values)
  return sparse_input, dense_input


class SparseTensorDenseMatmulOpDeterminismExceptionsTest(test.TestCase):
  """Test d9m-unimplemented exceptions from SparseTensorDenseMatmulOp.

  Test that tf.errors.UnimplementedError is thrown, as appropriate, by the
  GPU-specific code-paths through SparseTensorDenseMatmulOp when deterministic
  ops are enabled.

  This test assumes that sparse_tensor_dense_matmul_op_test.py runs equivalent
  test cases when deterministic ops are not enabled and will therefore detect
  erroneous exception throwing in those cases.
  """

  @test_util.run_gpu_only
  @test_util.run_in_graph_and_eager_modes
  def testExceptionThrowing(self):
    with self.session(), test_util.force_gpu():
      for data_type in [
          np.float16, np.float32, np.float64, np.complex64, np.complex128
      ]:
        sparse_input, dense_input = _gen_data(
            m=5,
            k=10,
            n=7,
            nnz=20,
            row_occupied_rate=0.9,
            data_type=data_type,
            seed=456)
        with self.assertRaisesRegex(
            errors.UnimplementedError,
            "A deterministic GPU implementation of SparseTensorDenseMatmulOp" +
            " is not currently available."):
          result = sparse_ops.sparse_tensor_dense_matmul(
              sparse_input, dense_input)
          self.evaluate(result)


class SparseTensorDenseMatmulOpDeterministicTest(test.TestCase):
  """Test that SparseTensorDenseMatul operates reproducibly (on CPU only)."""

  @test_util.run_in_graph_and_eager_modes
  def testForward(self):
    for data_type in [
        np.float16, np.float32, np.float64, np.complex64, np.complex128
    ]:  # skipping int32 and bfloat16
      sparse_input, dense_input = _gen_data(
          m=2430,
          k=615,
          n=857,
          nnz=(1 << 16) + 243,
          row_occupied_rate=0.02,
          data_type=data_type,
          seed=123)
      with self.session(), test_util.force_cpu():
        result_a = sparse_ops.sparse_tensor_dense_matmul(
            sparse_input, dense_input)
        for _ in range(5):
          result_b = sparse_ops.sparse_tensor_dense_matmul(
              sparse_input, dense_input)
          self.assertAllEqual(result_a, result_b)


if __name__ == "__main__":
  # Note that the effect of setting the following environment variable to
  # 'true' is not tested. Unless we can find a simpler pattern for testing these
  # environment variables, it would require this file to be made into a base
  # and then two more test files to be created.
  os.environ["TF_DETERMINISTIC_OPS"] = "1"

  test.main()

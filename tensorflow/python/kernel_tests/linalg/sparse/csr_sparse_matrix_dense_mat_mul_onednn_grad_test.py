# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TF_ENABLE_ONEDNN_SPMM=1."""

import os

from tensorflow.python.platform import test
from tensorflow.python.kernel_tests.linalg.sparse import csr_sparse_matrix_dense_mat_mul_grad_test_base

CSRSparseMatrixDenseMatMulGradTest = \
  csr_sparse_matrix_dense_mat_mul_grad_test_base.\
  CSRSparseMatrixDenseMatMulGradTest

if __name__ == '__main__':
  os.environ['TF_ENABLE_ONEDNN_SPMM'] = '1'
  csr_sparse_matrix_dense_mat_mul_grad_test_base.\
    define_csr_sparse_matrix_dense_mat_mul_grad_tests(
      CSRSparseMatrixDenseMatMulGradTest)
  test.main()

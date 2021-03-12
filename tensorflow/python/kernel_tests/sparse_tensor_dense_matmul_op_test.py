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
# ==============================================================================
"""Functional tests for SparseTensorDenseMatMul."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from tensorflow.python.kernel_tests import sparse_tensor_dense_matmul_op_base
from tensorflow.python.platform import test


SparseTensorDenseMatMulTest = \
  sparse_tensor_dense_matmul_op_base.SparseTensorDenseMatMulTestBase


def main(_):
  print("DenseDense MatMul (w/ Sparse Flag) vs. SparseTensorDense MatMul")
  print("Matrix sizes:")
  print("  A sparse [m, k] with % nonzero values between 1% and 80%")
  print("  B dense [k, n]")
  print("")
  print("% nnz \t n \t gpu \t m \t k \t dt(dense) \t dt(sparse) "
        "\t dt(sparse)/dt(dense)")

  for thresh in (0.99, 0.8, 0.5, 0.2):
    for n in (50, 100):
      for use_gpu in (True, False):
        for m in (100, 1000):
          for k in (100, 1000):
            sparse_tensor_dense_vs_dense_matmul_benchmark(
                thresh, m, k, n, False, False, use_gpu=use_gpu)

  # Enable for large scale benchmarks, these ones take a long time to run.
  #
  # for use_gpu in (True, False):
  #   sparse_tensor_dense_vs_dense_matmul_benchmark(
  #       thresh=0.99, m=1000000, k=1000, n=100, adjoint_a=False,
  #       adjoint_b=False, use_gpu=use_gpu, skip_dense=True)


if __name__ == "__main__":
  if "--benchmarks" in sys.argv:
    sys.argv.remove("--benchmarks")
    app.run()
  else:
    test.main()

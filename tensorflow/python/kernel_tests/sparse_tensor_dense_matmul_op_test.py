# Copyright 2015-2021 The TensorFlow Authors. All Rights Reserved.
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

import sys

from tensorflow.python.kernel_tests import sparse_tensor_dense_matmul_op_base
from tensorflow.python.platform import test


SparseTensorDenseMatMulTest = \
  sparse_tensor_dense_matmul_op_base.SparseTensorDenseMatMulTestBase


def sparse_tensor_dense_vs_dense_matmul_benchmark(thresh,
                                                  m,
                                                  k,
                                                  n,
                                                  adjoint_a,
                                                  adjoint_b,
                                                  use_gpu,
                                                  skip_dense=False):
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # Configurable for benchmarking:
  # config.intra_op_parallelism_threads = 100
  # config.gpu_options.per_process_gpu_memory_fraction = 0.3

  np.random.seed([6, 117])  # Reproducibility
  x = np.random.rand(m, k).astype(np.float32)
  x[x < thresh] = 0
  y = np.random.randn(k, n).astype(np.float32)
  if adjoint_a:
    x = x.T
  if adjoint_b:
    y = y.T

  def _timer(sess, ops_fn, iterations):
    # Warm in
    sess.run(ops_fn(10, sess))

    # Timing run
    start = time.time()
    sess.run(ops_fn(iterations, sess))
    end = time.time()

    return (end - start) / (1.0 * iterations)  # Average runtime per iteration

  # Using regular matmul, marking one of the matrices as dense.
  if skip_dense:
    delta_dense = float("nan")
  else:
    with session.Session(config=config, graph=ops.Graph()) as sess:
      if not use_gpu:
        with ops.device("/cpu:0"):
          x_t = constant_op.constant(x)
          y_t = constant_op.constant(y)
          ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_dense(
              x_t, y_t, adjoint_a, adjoint_b)
      else:
        with ops.device("/device:GPU:0"):
          x_t = constant_op.constant(x)
          y_t = constant_op.constant(y)
          ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_dense(
              x_t, y_t, adjoint_a, adjoint_b)
      delta_dense = _timer(sess, ops_fn, 200)

  # Using sparse_tensor_dense_matmul.
  with session.Session("", config=config, graph=ops.Graph()) as sess:
    if not use_gpu:
      with ops.device("/cpu:0"):
        x_ind = constant_op.constant(np.vstack(np.where(x)).astype(np.int64).T)
        x_val = constant_op.constant(x[np.where(x)])
        x_shape = constant_op.constant(np.array(x.shape).astype(np.int64))
        y_t = constant_op.constant(y)
        ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_sparse(
            x_ind, x_val, x_shape, y_t, adjoint_a, adjoint_b)
    else:
      with ops.device("/device:GPU:0"):
        x_ind = constant_op.constant(np.vstack(np.where(x)).astype(np.int64).T)
        x_val = constant_op.constant(x[np.where(x)])
        x_shape = constant_op.constant(np.array(x.shape).astype(np.int64))
        y_t = constant_op.constant(y)
        ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_sparse(
            x_ind, x_val, x_shape, y_t, adjoint_a, adjoint_b)
    delta_sparse = _timer(sess, ops_fn, 200)

  print("%g \t %d \t %s \t %d \t %d \t %g \t %g \t %g" %
        (1 - thresh, n, use_gpu, m, k, delta_dense, delta_sparse,
         delta_sparse / delta_dense))

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

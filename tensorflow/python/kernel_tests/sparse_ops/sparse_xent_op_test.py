# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SparseSoftmaxCrossEntropyWithLogits op."""

import sys
import time

from absl import app
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.sparse_ops import sparse_xent_op_test_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.nn_grad import _SparseSoftmaxCrossEntropyWithLogitsGrad  # pylint: disable=unused-import
from tensorflow.python.platform import test


SparseXentOpTest = sparse_xent_op_test_base.SparseXentOpTestBase


def _sparse_vs_dense_xent_benchmark_dense(labels, logits):
  labels = array_ops.identity(labels)
  logits = array_ops.identity(logits)
  with ops_lib.device("/cpu:0"):  # Sparse-to-dense must be on CPU
    batch_size = array_ops.shape(logits)[0]
    num_entries = array_ops.shape(logits)[1]
    length = batch_size * num_entries
    labels += num_entries * math_ops.range(batch_size)
    target = sparse_ops.sparse_to_dense(labels,
                                        array_ops.stack([length]), 1.0, 0.0)
  target = array_ops.reshape(target, array_ops.stack([-1, num_entries]))
  crossent = nn_ops.softmax_cross_entropy_with_logits(
      labels=target, logits=logits, name="SequenceLoss/CrossEntropy")
  crossent_sum = math_ops.reduce_sum(crossent)
  grads = gradients_impl.gradients([crossent_sum], [logits])[0]

  return (crossent_sum, grads)


def _sparse_vs_dense_xent_benchmark_sparse(labels, logits):
  # Using sparse_softmax_cross_entropy_with_logits
  labels = labels.astype(np.int64)
  labels = array_ops.identity(labels)
  logits = array_ops.identity(logits)
  crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name="SequenceLoss/CrossEntropy")
  crossent_sum = math_ops.reduce_sum(crossent)
  grads = gradients_impl.gradients([crossent_sum], [logits])[0]

  return (crossent_sum, grads)


def sparse_vs_dense_xent_benchmark(batch_size, num_entries, use_gpu):
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.3
  labels = np.random.randint(num_entries, size=batch_size).astype(np.int32)
  logits = np.random.randn(batch_size, num_entries).astype(np.float32)

  def _timer(sess, ops):
    # Warm in
    for _ in range(20):
      sess.run(ops)

    # Timing run
    start = time.time()
    for _ in range(20):
      sess.run(ops)
    end = time.time()

    return (end - start) / 20.0  # Average runtime per iteration

  # Using sparse_to_dense and softmax_cross_entropy_with_logits
  with session.Session(config=config) as sess:
    if not use_gpu:
      with ops_lib.device("/cpu:0"):
        ops = _sparse_vs_dense_xent_benchmark_dense(labels, logits)
    else:
      ops = _sparse_vs_dense_xent_benchmark_dense(labels, logits)
    delta_dense = _timer(sess, ops)

  # Using sparse_softmax_cross_entropy_with_logits
  with session.Session(config=config) as sess:
    if not use_gpu:
      with test_util.device("/cpu:0"):
        ops = _sparse_vs_dense_xent_benchmark_sparse(labels, logits)
    else:
      ops = _sparse_vs_dense_xent_benchmark_sparse(labels, logits)
    delta_sparse = _timer(sess, ops)

  print("%d \t %d \t %s \t %f \t %f \t %f" % (batch_size, num_entries, use_gpu,
                                              delta_dense, delta_sparse,
                                              delta_sparse / delta_dense))


def main(_):
  print("Sparse Xent vs. SparseToDense + Xent")
  print("batch \t depth \t gpu \t dt(dense) \t dt(sparse) "
        "\t dt(sparse)/dt(dense)")
  for use_gpu in (False, True):
    for batch_size in (32, 64, 128):
      for num_entries in (100, 1000, 10000):
        sparse_vs_dense_xent_benchmark(batch_size, num_entries, use_gpu)
    sparse_vs_dense_xent_benchmark(32, 100000, use_gpu)
    sparse_vs_dense_xent_benchmark(8, 1000000, use_gpu)


if __name__ == "__main__":
  if "--benchmarks" in sys.argv:
    sys.argv.remove("--benchmarks")
    app.run()  # pylint: disable=no-value-for-parameter
  else:
    test.main()

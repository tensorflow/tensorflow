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
"""Benchmark for masked_matmul_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-todo, g-import-not-at-top
import time

from tensorflow.contrib.factorization.python.ops import gen_factorization_ops
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class MaskedmatmulBenchmark(test.Benchmark):
  """Benchmark masked_matmul_ops."""

  def _make_sparse_mask(self, mask_shape, nnz, sort=False):
    """Creates a sparse tensor to be used as a mask in masked_matmul.

    Args:
      mask_shape: int list, the shape of the mask.
      nnz: int, the number of non-zero elements in the mask.
      sort: boolean, whether to sort the indices of the mask (in lexicographic
        order).
    Returns:
      A sparse tensor, with nnz indices, drawn uniformly at random.
    """
    num_rows = mask_shape[0]
    num_cols = mask_shape[1]
    row_idx = random_ops.random_uniform(
        [nnz], minval=0, maxval=num_rows, dtype=dtypes.int64)
    col_idx = random_ops.random_uniform(
        [nnz], minval=0, maxval=num_cols, dtype=dtypes.int64)
    indices = array_ops.stack([row_idx, col_idx], axis=1)
    values = array_ops.ones([nnz])
    unordered_mask = sparse_tensor.SparseTensor(indices, values, mask_shape)
    return sparse_ops.sparse_reorder(unordered_mask) if sort else unordered_mask

  def _run_graph(self, a_shape, b_shape, nnz, num_iters, sort=False,
                 transpose_a=False, transpose_b=False):
    """Run the graph and return its average execution time.

    Args:
      a_shape: int list, the shape of the a matrix.
      b_shape: int list, the shape of the b matrix.
      nnz: int, the number of non-zero elements in the mask.
      num_iters: int, the number of iterations to run (the output is the average
        execution time, over num_iters).
      sort: Boolean, whether to sort the indices in the mask.
      transpose_a: boolean, whether to transpose the a matrix.
      transpose_b: boolean, whether to transpose the b matrix.

    Returns:
      The average duration of the masked_matmul op in seconds.
    """
    graph = ops.Graph()

    with graph.as_default(), session_lib.Session(graph=graph) as session:
      mask_shape = [a_shape[0], b_shape[1]]
      a_shape = a_shape if not transpose_a else [a_shape[1], a_shape[0]]
      b_shape = b_shape if not transpose_b else [b_shape[1], b_shape[0]]
      a_var = variables.Variable(random_ops.random_normal(a_shape))
      b_var = variables.Variable(random_ops.random_normal(b_shape))
      mask_indices_ph = array_ops.placeholder(dtypes.int64, shape=[nnz, 2])
      a_ph = array_ops.placeholder(dtypes.float32, shape=a_shape)
      b_ph = array_ops.placeholder(dtypes.float32, shape=b_shape)
      mask = self._make_sparse_mask(mask_shape, nnz, sort)
      masked_prod = gen_factorization_ops.masked_matmul(
          a_ph, b_ph, mask_indices_ph, transpose_a, transpose_b)
      with ops.control_dependencies([masked_prod]):
        result = control_flow_ops.no_op()

      variables.global_variables_initializer().run()
      avg_wall_time = 0
      for _ in range(num_iters):
        a, b, mask_indices = session.run([a_var, b_var, mask.indices])
        feed_dict = {
            mask_indices_ph: mask_indices,
            a_ph: a,
            b_ph: b
        }
        start_time = time.time()
        session.run(result, feed_dict=feed_dict)
        avg_wall_time += (time.time() - start_time)/num_iters

      bench_name = (
          "cpu nnz:{nnz} a_shape:{a_shape} b_shape:{b_shape} tr_a:{tr_a} "
          "tr_b:{tr_b} sort:{sort}"
      ).format(
          nnz=nnz,
          a_shape=a_shape,
          b_shape=b_shape,
          tr_a=int(transpose_a),
          tr_b=int(transpose_b),
          sort=int(sort)
      )
      print(bench_name + " - %f secs" % avg_wall_time)
      name = bench_name.replace(", ", "_").replace(":", "_").replace(" ", "_")
      self.report_benchmark(
          name=name,
          iters=num_iters,
          wall_time=avg_wall_time)

    return avg_wall_time

  # TODO(walidk): compare benchmarks to using existing tf ops.
  def benchmark_matmul(self):
    num_iters = 10
    nnz = 100000
    for transpose_a in [False, True]:
      for transpose_b in [False, True]:
        for dim in [200, 400, 800]:
          for sort in [False, True]:
            a_shape = [10000, dim]
            b_shape = [dim, 10000]
            self._run_graph(a_shape, b_shape, nnz, num_iters, sort, transpose_a,
                            transpose_b)


if __name__ == "__main__":
  test.main()

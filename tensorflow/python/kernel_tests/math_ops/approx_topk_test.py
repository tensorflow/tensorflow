# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.math.approx_max_k and tf.math.approx_min_k."""

import itertools
from absl.testing import parameterized

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager.def_function import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class ApproxTopkTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def setUp(self):
    test_util.TensorFlowTestCase.setUp(self)
    self._rng = np.random.default_rng(42)

  def compute_recall(self, result_neighbors, ground_truth_neighbors):
    """Computes the recall of an approximate nearest neighbor search.

    Args:
      result_neighbors: int32 numpy array of the shape [num_queries,
        neighbors_per_query] where the values are the indices of the dataset.
      ground_truth_neighbors: int32 numpy array of with shape [num_queries,
        ground_truth_neighbors_per_query] where the values are the indices of
        the dataset.

    Returns:
      The recall.
    """
    self.assertLen(result_neighbors.shape, 2)
    self.assertLen(ground_truth_neighbors.shape, 2)
    self.assertEqual(result_neighbors.shape[0], ground_truth_neighbors.shape[0])
    gt_sets = [set(np.asarray(x)) for x in ground_truth_neighbors]

    def hits_per_q(q, nn_per_q):
      return len(list(x for x in nn_per_q if x.item() in gt_sets[q]))

    hits = sum(
        hits_per_q(q, nn_per_q) for q, nn_per_q in enumerate(result_neighbors))
    return hits / ground_truth_neighbors.size

  @parameterized.parameters(
      itertools.product(
          [dtypes.bfloat16, dtypes.float16, dtypes.float32],
          [1, 10],  # k
          [100, 500],  # row_size
          [1, 10, 128],  # num_rows
          [True, False],  # aggregate_to_topk
      ))
  def test_non_fused_max_k(self, dtype, k, row_size, num_rows,
                           aggregate_to_topk):
    row = np.arange(row_size, dtype=np.float32)
    db = np.stack(list(self._rng.permutation(row) for _ in range(num_rows)))
    db_op = constant_op.constant(db, dtype=dtype)
    # Must jit-compile to access the xla kernel.
    @function(jit_compile=True)
    def ann(db, k):
      return nn_ops.approx_max_k(db, k, aggregate_to_topk=aggregate_to_topk)

    _, idx = self.evaluate(ann(db_op, k))
    gt = np.argsort(-db)[:, :k]
    ann_recall = self.compute_recall(idx, gt)
    self.assertGreaterEqual(ann_recall, 0.95)

  @parameterized.parameters(
      itertools.product(
          [dtypes.bfloat16, dtypes.float16, dtypes.float32],
          [1, 10],  # k
          [100, 500],  # row_size
          [1, 10, 128],  # num_rows
          [True, False],  # aggregate_to_topk
      ))
  def test_non_fused_min_k(self, dtype, k, row_size, num_rows,
                           aggregate_to_topk):
    # Use the new rng api
    row = np.arange(row_size, dtype=np.float32)
    db = np.stack(list(self._rng.permutation(row) for _ in range(num_rows)))
    db_op = constant_op.constant(db, dtype=dtype)
    # Must jit-compile to access the xla kernel.
    @function(jit_compile=True)
    def ann(db, k=10):
      return nn_ops.approx_min_k(db, k, aggregate_to_topk=aggregate_to_topk)

    _, idx = self.evaluate(ann(db_op, k))
    gt = np.argsort(db)[:, :k]
    ann_recall = self.compute_recall(idx, gt)
    self.assertGreaterEqual(ann_recall, 0.95)

  @parameterized.parameters(
      itertools.product(
          [dtypes.float32],  # Use float32 for numerical stability.
          [1, 10],  # k
          [100, 500],  # db_size
          [1, 10, 128],  # qy_size
          [2, 32],  # feature dim
      ))
  # MIPS = Maximal Inner Product Search
  def test_mips(self, dtype, k, db_size, qy_size, feature_dim):
    qy = self._rng.random([qy_size, feature_dim])
    db = self._rng.random([db_size, feature_dim])
    qy_op = constant_op.constant(qy, dtype=dtype)
    db_op = constant_op.constant(db, dtype=dtype)
    # Must jit-compile to access the xla kernel.
    @function(jit_compile=True)
    def ann(qy, db, k):
      scores = math_ops.matmul(qy, db, transpose_b=True)
      return nn_ops.approx_max_k(scores, k)

    _, idx = self.evaluate(ann(qy_op, db_op, k))
    scores = self.evaluate(-math_ops.matmul(qy_op, db_op, transpose_b=True))
    gt = np.argsort(scores)[:, :k]
    ann_recall = self.compute_recall(idx, gt)
    self.assertGreaterEqual(ann_recall, 0.95)

  @parameterized.parameters(
      itertools.product(
          [dtypes.float32],  # Use float32 for numerical stability.
          [1, 10],  # k
          [100, 500],  # db_size
          [1, 10, 128],  # qy_size
          [2, 32],  # feature dim
      ))
  # L2ANN = Approximate Nearest Neighbor search in the L2 metric space
  def test_l2ann(self, dtype, k, db_size, qy_size, feature_dim):
    qy = self._rng.random([qy_size, feature_dim])
    db = self._rng.random([db_size, feature_dim])
    db_half_norm_sq = np.linalg.norm(db, axis=1)**2 / 2
    qy_op = constant_op.constant(qy, dtype=dtype)
    db_op = constant_op.constant(db, dtype=dtype)
    db_half_norm_sq_op = constant_op.constant(db_half_norm_sq, dtype=dtype)
    # Must jit-compile to access the xla kernel.
    @function(jit_compile=True)
    def ann(qy, db, db_half_norm_sq, k):
      scores = db_half_norm_sq - math_ops.matmul(qy, db, transpose_b=True)
      return nn_ops.approx_min_k(scores, k)

    _, idx = self.evaluate(ann(qy_op, db_op, db_half_norm_sq_op, k))
    scores = self.evaluate(db_half_norm_sq_op -
                           math_ops.matmul(qy_op, db_op, transpose_b=True))
    gt = np.argsort(scores)[:, :k]
    ann_recall = self.compute_recall(idx, gt)
    self.assertGreaterEqual(ann_recall, 0.95)

  def test_highdim(self):
    db = self._rng.random([2, 10, 200, 3], dtype=np.float32)
    k = 5

    @function(jit_compile=True)
    def ann(db, k):
      return nn_ops.approx_min_k(db, k=k, reduction_dimension=2)

    _, idx = self.evaluate(ann(db, k))
    gt = np.argsort(db, axis=2)[:, :, :k, :]
    flat_idx = np.reshape(np.transpose(idx, [0, 1, 3, 2]), [2 * 10 * 3, k])
    flat_gt = np.reshape(np.transpose(gt, [0, 1, 3, 2]), [2 * 10 * 3, k])
    ann_recall = self.compute_recall(flat_idx, flat_gt)
    self.assertGreaterEqual(ann_recall, 0.95)

  @parameterized.parameters(
      itertools.product(
          [dtypes.bfloat16, dtypes.float16, dtypes.float32],
          [1, 10],  # k
          [100, 500],  # row_size
          [1, 10, 128],  # num_rows
      ))
  def test_gradients(self, dtype, k, row_size, num_rows):
    row = np.arange(row_size, dtype=np.float32)
    db = np.stack(list(self._rng.permutation(row) for _ in range(num_rows)))
    db_op = constant_op.constant(db, dtype=dtype)
    out_grads = self._rng.random([num_rows, k])
    out_grads_op = constant_op.constant(out_grads, dtype=dtype)

    # Must jit-compile to access the xla kernel.
    @function(jit_compile=True)
    def ann_with_grads(db, out_grads):
      with backprop.GradientTape() as tape:
        tape.watch(db)
        val, idx = nn_ops.approx_max_k(db, k)
      result_in_grads = tape.gradient(val, db, out_grads)
      lifted_k_idx = array_ops.reshape(idx, [num_rows, k, 1])
      iota_idx = array_ops.broadcast_to(
          array_ops.reshape(math_ops.range(num_rows), [num_rows, 1, 1]),
          [num_rows, k, 1])
      lifted_idx = array_ops.concat([iota_idx, lifted_k_idx], axis=2)
      k_idx_s = array_ops.reshape(lifted_idx, [num_rows * k, 2])
      k_gra_s = array_ops.reshape(out_grads, [num_rows * k])
      expected_in_grads = array_ops.scatter_nd(k_idx_s, k_gra_s,
                                               [num_rows, row_size])
      return [expected_in_grads, result_in_grads]

    expected_in_grads, result_in_grads = self.evaluate(
        ann_with_grads(db_op, out_grads_op))
    self.assertAllClose(expected_in_grads, result_in_grads)

  def test_invalid_input(self):
    @function(jit_compile=True)
    def fuzz_jit():
      return nn_ops.approx_max_k(
          [
              183.39395141601562,
              62.6842041015625,
              83.8385238647461,
              204.36642456054688,
          ],
          4774,
          reduction_dimension=0x8282828,
          recall_target=135.9822179933652,
          reduction_input_size_override=6154,
          aggregate_to_topk=True,
      )

    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      fuzz_jit()

  def test_b272094281(self):
    @function(jit_compile=True)
    def fuzz_jit():
      return nn_ops.approx_max_k(
          [],
          9223372036854775807,
          reduction_dimension=-4294967297 + 0x41,
          reduction_input_size_override=-9223372036854775807,
          aggregate_to_topk=False,
      )

    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      fuzz_jit()


if __name__ == '__main__':
  test.main()

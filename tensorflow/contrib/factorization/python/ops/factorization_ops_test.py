# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for factorization_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.factorization.python.ops import factorization_ops
from tensorflow.contrib.factorization.python.ops import factorization_ops_test_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

INPUT_MATRIX = factorization_ops_test_utils.INPUT_MATRIX
np_matrix_to_tf_sparse = factorization_ops_test_utils.np_matrix_to_tf_sparse


class WalsModelTest(test.TestCase):

  def sparse_input(self):
    return np_matrix_to_tf_sparse(INPUT_MATRIX)

  def count_rows(self, sp_input):
    return math_ops.cast(
        array_ops.shape(array_ops.unique(sp_input.indices[:, 0])[0])[0],
        dtypes.float32)

  def count_cols(self, sp_input):
    return math_ops.cast(
        array_ops.shape(array_ops.unique(sp_input.indices[:, 1])[0])[0],
        dtypes.float32)

  def calculate_loss_from_wals_model(self, wals_model, sp_inputs):
    current_rows = embedding_ops.embedding_lookup(
        wals_model.row_factors,
        math_ops.range(wals_model._input_rows),
        partition_strategy="div")
    current_cols = embedding_ops.embedding_lookup(
        wals_model.col_factors,
        math_ops.range(wals_model._input_cols),
        partition_strategy="div")
    row_wts = embedding_ops.embedding_lookup(
        wals_model._row_weights,
        math_ops.range(wals_model._input_rows),
        partition_strategy="div")
    col_wts = embedding_ops.embedding_lookup(
        wals_model._col_weights,
        math_ops.range(wals_model._input_cols),
        partition_strategy="div")
    return factorization_ops_test_utils.calculate_loss(
        sp_inputs, current_rows, current_cols, wals_model._regularization,
        wals_model._unobserved_weight, row_wts, col_wts)

  def setUp(self):
    self.col_init = [
        # shard 0
        [
            [-0.36444709, -0.39077035, -0.32528427],  # pyformat line break
            [1.19056475, 0.07231052, 2.11834812],
            [0.93468881, -0.71099287, 1.91826844]
        ],
        # shard 1
        [[1.18160152, 1.52490723, -0.50015002],
         [1.82574749, -0.57515913, -1.32810032]],
        # shard 2
        [[-0.15515432, -0.84675711, 0.13097958],
         [-0.9246484, 0.69117504, 1.2036494]]
    ]

    self.row_wts = [[0.1, 0.2, 0.3], [0.4, 0.5]]
    self.col_wts = [[0.1, 0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]

    # Values of factor shards after running one iteration of row and column
    # updates.
    self._row_factors_0 = [
        [0.097689, -0.219293, -0.020780],  # pyformat line break
        [0.50842, 0.64626, 0.22364],
        [0.401159, -0.046558, -0.192854]
    ]
    self._row_factors_1 = [[1.20597, -0.48025, 0.35582],
                           [1.5564, 1.2528, 1.0528]]
    self._col_factors_0 = [
        [2.4725, -1.2950, -1.9980],  # pyformat line break
        [0.44625, 1.50771, 1.27118],
        [1.39801, -2.10134, 0.73572]
    ]
    self._col_factors_1 = [[3.36509, -0.66595, -3.51208],
                           [0.57191, 1.59407, 1.33020]]
    self._col_factors_2 = [[3.3459, -1.3341, -3.3008],
                           [0.57366, 1.83729, 1.26798]]

  def _run_test_sum_weights(self, test_rows):
    # test_rows: True to test row weights, False to test column weights.

    num_rows = 5
    num_cols = 5
    unobserved_weight = 0.1
    row_weights = [[8., 18., 28., 38., 48.]]
    col_weights = [[90., 91., 92., 93., 94.]]
    sparse_indices = [[0, 1], [2, 3], [4, 1]]
    sparse_values = [666., 777., 888.]

    unobserved = unobserved_weight * num_rows * num_cols
    observed = 8. * 91. + 28. * 93. + 48. * 91.
    # sparse_indices has three unique rows and two unique columns
    observed *= num_rows / 3. if test_rows else num_cols / 2.
    want_weight_sum = unobserved + observed

    with ops.Graph().as_default(), self.test_session() as sess:
      wals_model = factorization_ops.WALSModel(
          input_rows=num_rows,
          input_cols=num_cols,
          n_components=5,
          unobserved_weight=unobserved_weight,
          row_weights=row_weights,
          col_weights=col_weights,
          use_factors_weights_cache=False)

      wals_model.initialize_op.run()
      wals_model.worker_init.run()

      update_factors = (wals_model.update_row_factors
                        if test_rows else wals_model.update_col_factors)

      (_, _, _, _, sum_weights) = update_factors(
          sp_input=sparse_tensor.SparseTensor(
              indices=sparse_indices,
              values=sparse_values,
              dense_shape=[num_rows, num_cols]),
          transpose_input=False)

      got_weight_sum = sess.run(sum_weights)

      self.assertNear(
          got_weight_sum,
          want_weight_sum,
          err=.001,
          msg="got weight sum [{}], want weight sum [{}]".format(
              got_weight_sum, want_weight_sum))

  def _run_test_process_input(self,
                              use_factors_weights_cache,
                              compute_loss=False):
    with ops.Graph().as_default(), self.test_session() as sess:
      self._wals_inputs = self.sparse_input()
      sp_feeder = array_ops.sparse_placeholder(dtypes.float32)
      num_rows = 5
      num_cols = 7
      factor_dim = 3
      wals_model = factorization_ops.WALSModel(
          num_rows,
          num_cols,
          factor_dim,
          num_row_shards=2,
          num_col_shards=3,
          regularization=0.01,
          unobserved_weight=0.1,
          col_init=self.col_init,
          row_weights=self.row_wts,
          col_weights=self.col_wts,
          use_factors_weights_cache=use_factors_weights_cache)

      wals_model.initialize_op.run()
      wals_model.worker_init.run()

      # Split input into multiple sparse tensors with scattered rows. Note that
      # this split can be different than the factor sharding and the inputs can
      # consist of non-consecutive rows. Each row needs to include all non-zero
      # elements in that row.
      sp_r0 = np_matrix_to_tf_sparse(INPUT_MATRIX, [0, 2]).eval()
      sp_r1 = np_matrix_to_tf_sparse(INPUT_MATRIX, [1, 4], shuffle=True).eval()
      sp_r2 = np_matrix_to_tf_sparse(INPUT_MATRIX, [3], shuffle=True).eval()
      input_scattered_rows = [sp_r0, sp_r1, sp_r2]

      # Test updating row factors.
      # Here we feed in scattered rows of the input.
      wals_model.row_update_prep_gramian_op.run()
      wals_model.initialize_row_update_op.run()
      (_, process_input_op, unregularized_loss, regularization,
       _) = wals_model.update_row_factors(
           sp_input=sp_feeder, transpose_input=False)
      factor_loss = unregularized_loss + regularization
      for inp in input_scattered_rows:
        feed_dict = {sp_feeder: inp}
        process_input_op.run(feed_dict=feed_dict)
      row_factors = [x.eval() for x in wals_model.row_factors]

      self.assertAllClose(row_factors[0], self._row_factors_0, atol=1e-3)
      self.assertAllClose(row_factors[1], self._row_factors_1, atol=1e-3)

      # Test row projection.
      # Using the specified projection weights for the 2 row feature vectors.
      # This is expected to reproduce the same row factors in the model as the
      # weights and feature vectors are identical to that used in model
      # training.
      projected_rows = wals_model.project_row_factors(
          sp_input=sp_feeder,
          transpose_input=False,
          projection_weights=[0.2, 0.5])
      # Don't specify the projection weight, so 1.0 will be used. The feature
      # weights will be those specified in model.
      projected_rows_no_weights = wals_model.project_row_factors(
          sp_input=sp_feeder, transpose_input=False)
      feed_dict = {
          sp_feeder:
              np_matrix_to_tf_sparse(INPUT_MATRIX, [1, 4], shuffle=False)
              .eval()
      }
      self.assertAllClose(
          projected_rows.eval(feed_dict=feed_dict),
          [self._row_factors_0[1], self._row_factors_1[1]],
          atol=1e-3)
      self.assertAllClose(
          projected_rows_no_weights.eval(feed_dict=feed_dict),
          [[0.569082, 0.715088, 0.31777], [1.915879, 1.992677, 1.109057]],
          atol=1e-3)

      if compute_loss:
        # Test loss computation after the row update
        loss = sum(
            sess.run(
                factor_loss * self.count_rows(inp) / num_rows,
                feed_dict={sp_feeder: inp}) for inp in input_scattered_rows)
        true_loss = self.calculate_loss_from_wals_model(wals_model,
                                                        self._wals_inputs)
        self.assertNear(
            loss,
            true_loss,
            err=.001,
            msg="After row update, computed loss [{}] does not match"
            " true loss [{}]".format(loss, true_loss))

      # Split input into multiple sparse tensors with scattered columns. Note
      # that here the elements in the sparse tensors are not ordered and also
      # do not need to consist of consecutive columns. However, each column
      # needs to include all non-zero elements in that column.
      sp_c0 = np_matrix_to_tf_sparse(INPUT_MATRIX, col_slices=[2, 0]).eval()
      sp_c1 = np_matrix_to_tf_sparse(
          INPUT_MATRIX, col_slices=[5, 3, 1], shuffle=True).eval()
      sp_c2 = np_matrix_to_tf_sparse(INPUT_MATRIX, col_slices=[4, 6]).eval()
      sp_c3 = np_matrix_to_tf_sparse(
          INPUT_MATRIX, col_slices=[3, 6], shuffle=True).eval()

      input_scattered_cols = [sp_c0, sp_c1, sp_c2, sp_c3]
      input_scattered_cols_non_duplicate = [sp_c0, sp_c1, sp_c2]

      # Test updating column factors.
      # Here we feed in scattered columns of the input.
      wals_model.col_update_prep_gramian_op.run()
      wals_model.initialize_col_update_op.run()
      (_, process_input_op, unregularized_loss, regularization,
       _) = wals_model.update_col_factors(
           sp_input=sp_feeder, transpose_input=False)
      factor_loss = unregularized_loss + regularization
      for inp in input_scattered_cols:
        feed_dict = {sp_feeder: inp}
        process_input_op.run(feed_dict=feed_dict)
      col_factors = [x.eval() for x in wals_model.col_factors]

      self.assertAllClose(col_factors[0], self._col_factors_0, atol=1e-3)
      self.assertAllClose(col_factors[1], self._col_factors_1, atol=1e-3)
      self.assertAllClose(col_factors[2], self._col_factors_2, atol=1e-3)

      # Test column projection.
      # Using the specified projection weights for the 3 column feature vectors.
      # This is expected to reproduce the same column factors in the model as
      # the weights and feature vectors are identical to that used in model
      # training.
      projected_cols = wals_model.project_col_factors(
          sp_input=sp_feeder,
          transpose_input=False,
          projection_weights=[0.6, 0.4, 0.2])
      # Don't specify the projection weight, so 1.0 will be used. The feature
      # weights will be those specified in model.
      projected_cols_no_weights = wals_model.project_col_factors(
          sp_input=sp_feeder, transpose_input=False)
      feed_dict = {
          sp_feeder:
              np_matrix_to_tf_sparse(
                  INPUT_MATRIX, col_slices=[5, 3, 1], shuffle=False).eval()
      }
      self.assertAllClose(
          projected_cols.eval(feed_dict=feed_dict), [
              self._col_factors_2[0], self._col_factors_1[0],
              self._col_factors_0[1]
          ],
          atol=1e-3)
      self.assertAllClose(
          projected_cols_no_weights.eval(feed_dict=feed_dict),
          [[3.471045, -1.250835, -3.598917], [3.585139, -0.487476, -3.852232],
           [0.346433, 1.360644, 1.677121]],
          atol=1e-3)

      if compute_loss:
        # Test loss computation after the column update.
        loss = sum(
            sess.run(
                factor_loss * self.count_cols(inp) / num_cols,
                feed_dict={sp_feeder: inp})
            for inp in input_scattered_cols_non_duplicate)
        true_loss = self.calculate_loss_from_wals_model(wals_model,
                                                        self._wals_inputs)
        self.assertNear(
            loss,
            true_loss,
            err=.001,
            msg="After col update, computed loss [{}] does not match"
            " true loss [{}]".format(loss, true_loss))

  def _run_test_process_input_transposed(self,
                                         use_factors_weights_cache,
                                         compute_loss=False):
    with ops.Graph().as_default(), self.test_session() as sess:
      self._wals_inputs = self.sparse_input()
      sp_feeder = array_ops.sparse_placeholder(dtypes.float32)
      num_rows = 5
      num_cols = 7
      factor_dim = 3
      wals_model = factorization_ops.WALSModel(
          num_rows,
          num_cols,
          factor_dim,
          num_row_shards=2,
          num_col_shards=3,
          regularization=0.01,
          unobserved_weight=0.1,
          col_init=self.col_init,
          row_weights=self.row_wts,
          col_weights=self.col_wts,
          use_factors_weights_cache=use_factors_weights_cache)

      wals_model.initialize_op.run()
      wals_model.worker_init.run()

      # Split input into multiple SparseTensors with scattered rows.
      # Here the inputs are transposed. But the same constraints as described in
      # the previous non-transposed test case apply to these inputs (before they
      # are transposed).
      sp_r0_t = np_matrix_to_tf_sparse(
          INPUT_MATRIX, [0, 3], transpose=True).eval()
      sp_r1_t = np_matrix_to_tf_sparse(
          INPUT_MATRIX, [4, 1], shuffle=True, transpose=True).eval()
      sp_r2_t = np_matrix_to_tf_sparse(INPUT_MATRIX, [2], transpose=True).eval()
      sp_r3_t = sp_r1_t
      input_scattered_rows = [sp_r0_t, sp_r1_t, sp_r2_t, sp_r3_t]
      input_scattered_rows_non_duplicate = [sp_r0_t, sp_r1_t, sp_r2_t]
      # Test updating row factors.
      # Here we feed in scattered rows of the input.
      # Note that the needed suffix of placeholder are in the order of test
      # case name lexicographical order and then in the line order of where
      # they appear.
      wals_model.row_update_prep_gramian_op.run()
      wals_model.initialize_row_update_op.run()
      (_, process_input_op, unregularized_loss, regularization,
       _) = wals_model.update_row_factors(
           sp_input=sp_feeder, transpose_input=True)
      factor_loss = unregularized_loss + regularization
      for inp in input_scattered_rows:
        feed_dict = {sp_feeder: inp}
        process_input_op.run(feed_dict=feed_dict)
      row_factors = [x.eval() for x in wals_model.row_factors]

      self.assertAllClose(row_factors[0], self._row_factors_0, atol=1e-3)
      self.assertAllClose(row_factors[1], self._row_factors_1, atol=1e-3)

      # Test row projection.
      # Using the specified projection weights for the 2 row feature vectors.
      # This is expected to reproduce the same row factors in the model as the
      # weights and feature vectors are identical to that used in model
      # training.
      projected_rows = wals_model.project_row_factors(
          sp_input=sp_feeder,
          transpose_input=True,
          projection_weights=[0.5, 0.2])
      # Don't specify the projection weight, so 1.0 will be used. The feature
      # weights will be those specified in model.
      projected_rows_no_weights = wals_model.project_row_factors(
          sp_input=sp_feeder, transpose_input=True)
      feed_dict = {
          sp_feeder:
              np_matrix_to_tf_sparse(
                  INPUT_MATRIX, [4, 1], shuffle=False, transpose=True).eval()
      }
      self.assertAllClose(
          projected_rows.eval(feed_dict=feed_dict),
          [self._row_factors_1[1], self._row_factors_0[1]],
          atol=1e-3)
      self.assertAllClose(
          projected_rows_no_weights.eval(feed_dict=feed_dict),
          [[1.915879, 1.992677, 1.109057], [0.569082, 0.715088, 0.31777]],
          atol=1e-3)

      if compute_loss:
        # Test loss computation after the row update
        loss = sum(
            sess.run(
                factor_loss * self.count_cols(inp) / num_rows,
                feed_dict={sp_feeder: inp})
            for inp in input_scattered_rows_non_duplicate)
        true_loss = self.calculate_loss_from_wals_model(wals_model,
                                                        self._wals_inputs)
        self.assertNear(
            loss,
            true_loss,
            err=.001,
            msg="After row update, computed loss [{}] does not match"
            " true loss [{}]".format(loss, true_loss))

      # Split input into multiple SparseTensors with scattered columns.
      # Here the inputs are transposed. But the same constraints as described in
      # the previous non-transposed test case apply to these inputs (before they
      # are transposed).
      sp_c0_t = np_matrix_to_tf_sparse(
          INPUT_MATRIX, col_slices=[0, 1], transpose=True).eval()
      sp_c1_t = np_matrix_to_tf_sparse(
          INPUT_MATRIX, col_slices=[4, 2], transpose=True).eval()
      sp_c2_t = np_matrix_to_tf_sparse(
          INPUT_MATRIX, col_slices=[5], transpose=True, shuffle=True).eval()
      sp_c3_t = np_matrix_to_tf_sparse(
          INPUT_MATRIX, col_slices=[3, 6], transpose=True).eval()

      sp_c4_t = sp_c2_t
      input_scattered_cols = [sp_c0_t, sp_c1_t, sp_c2_t, sp_c3_t, sp_c4_t]
      input_scattered_cols_non_duplicate = [sp_c0_t, sp_c1_t, sp_c2_t, sp_c3_t]

      # Test updating column factors.
      # Here we feed in scattered columns of the input.
      wals_model.col_update_prep_gramian_op.run()
      wals_model.initialize_col_update_op.run()
      (_, process_input_op, unregularized_loss, regularization,
       _) = wals_model.update_col_factors(
           sp_input=sp_feeder, transpose_input=True)
      factor_loss = unregularized_loss + regularization
      for inp in input_scattered_cols:
        feed_dict = {sp_feeder: inp}
        process_input_op.run(feed_dict=feed_dict)
      col_factors = [x.eval() for x in wals_model.col_factors]

      self.assertAllClose(col_factors[0], self._col_factors_0, atol=1e-3)
      self.assertAllClose(col_factors[1], self._col_factors_1, atol=1e-3)
      self.assertAllClose(col_factors[2], self._col_factors_2, atol=1e-3)

      # Test column projection.
      # Using the specified projection weights for the 2 column feature vectors.
      # This is expected to reproduce the same column factors in the model as
      # the weights and feature vectors are identical to that used in model
      # training.
      projected_cols = wals_model.project_col_factors(
          sp_input=sp_feeder,
          transpose_input=True,
          projection_weights=[0.4, 0.7])
      # Don't specify the projection weight, so 1.0 will be used. The feature
      # weights will be those specified in model.
      projected_cols_no_weights = wals_model.project_col_factors(
          sp_input=sp_feeder, transpose_input=True)
      feed_dict = {sp_feeder: sp_c3_t}
      self.assertAllClose(
          projected_cols.eval(feed_dict=feed_dict),
          [self._col_factors_1[0], self._col_factors_2[1]],
          atol=1e-3)
      self.assertAllClose(
          projected_cols_no_weights.eval(feed_dict=feed_dict),
          [[3.585139, -0.487476, -3.852232], [0.557937, 1.813907, 1.331171]],
          atol=1e-3)
      if compute_loss:
        # Test loss computation after the col update
        loss = sum(
            sess.run(
                factor_loss * self.count_rows(inp) / num_cols,
                feed_dict={sp_feeder: inp})
            for inp in input_scattered_cols_non_duplicate)
        true_loss = self.calculate_loss_from_wals_model(wals_model,
                                                        self._wals_inputs)
        self.assertNear(
            loss,
            true_loss,
            err=.001,
            msg="After col update, computed loss [{}] does not match"
            " true loss [{}]".format(loss, true_loss))

  # Note that when row_weights and col_weights are 0, WALS gives identical
  # results as ALS (Alternating Least Squares). However our implementation does
  # not handle the case of zero weights differently. Instead, when row_weights
  # and col_weights are set to None, we interpret that as the ALS case, and
  # trigger the more efficient ALS updates.
  # Here we test that those two give identical results.
  def _run_test_als(self, use_factors_weights_cache):
    with ops.Graph().as_default(), self.test_session():
      self._wals_inputs = self.sparse_input()
      col_init = np.random.rand(7, 3)
      als_model = factorization_ops.WALSModel(
          5,
          7,
          3,
          col_init=col_init,
          row_weights=None,
          col_weights=None,
          use_factors_weights_cache=use_factors_weights_cache)

      als_model.initialize_op.run()
      als_model.worker_init.run()
      als_model.row_update_prep_gramian_op.run()
      als_model.initialize_row_update_op.run()
      process_input_op = als_model.update_row_factors(self._wals_inputs)[1]
      process_input_op.run()
      row_factors1 = [x.eval() for x in als_model.row_factors]
      # Testing row projection. Projection weight doesn't matter in this case
      # since the model is ALS special case.
      als_projected_row_factors1 = als_model.project_row_factors(
          self._wals_inputs).eval()

      wals_model = factorization_ops.WALSModel(
          5,
          7,
          3,
          col_init=col_init,
          row_weights=0,
          col_weights=0,
          use_factors_weights_cache=use_factors_weights_cache)
      wals_model.initialize_op.run()
      wals_model.worker_init.run()
      wals_model.row_update_prep_gramian_op.run()
      wals_model.initialize_row_update_op.run()
      process_input_op = wals_model.update_row_factors(self._wals_inputs)[1]
      process_input_op.run()
      row_factors2 = [x.eval() for x in wals_model.row_factors]

      for r1, r2 in zip(row_factors1, row_factors2):
        self.assertAllClose(r1, r2, atol=1e-3)
      self.assertAllClose(
          als_projected_row_factors1,
          [row for shard in row_factors2 for row in shard],
          atol=1e-3)

      # Here we test partial column updates.
      sp_c = np_matrix_to_tf_sparse(
          INPUT_MATRIX, col_slices=[2, 0], shuffle=True).eval()

      sp_feeder = array_ops.sparse_placeholder(dtypes.float32)
      feed_dict = {sp_feeder: sp_c}
      als_model.col_update_prep_gramian_op.run()
      als_model.initialize_col_update_op.run()
      process_input_op = als_model.update_col_factors(sp_input=sp_feeder)[1]
      process_input_op.run(feed_dict=feed_dict)
      col_factors1 = [x.eval() for x in als_model.col_factors]
      # Testing column projection. Projection weight doesn't matter in this case
      # since the model is ALS special case.
      als_projected_col_factors1 = als_model.project_col_factors(
          np_matrix_to_tf_sparse(
              INPUT_MATRIX, col_slices=[2, 0], shuffle=False)).eval()

      feed_dict = {sp_feeder: sp_c}
      wals_model.col_update_prep_gramian_op.run()
      wals_model.initialize_col_update_op.run()
      process_input_op = wals_model.update_col_factors(sp_input=sp_feeder)[1]
      process_input_op.run(feed_dict=feed_dict)
      col_factors2 = [x.eval() for x in wals_model.col_factors]

      for c1, c2 in zip(col_factors1, col_factors2):
        self.assertAllClose(c1, c2, rtol=5e-3, atol=1e-2)
      self.assertAllClose(
          als_projected_col_factors1, [col_factors2[0][2], col_factors2[0][0]],
          atol=1e-2)

  def _run_test_als_transposed(self, use_factors_weights_cache):
    with ops.Graph().as_default(), self.test_session():
      self._wals_inputs = self.sparse_input()
      col_init = np.random.rand(7, 3)
      als_model = factorization_ops.WALSModel(
          5,
          7,
          3,
          col_init=col_init,
          row_weights=None,
          col_weights=None,
          use_factors_weights_cache=use_factors_weights_cache)

      als_model.initialize_op.run()
      als_model.worker_init.run()

      wals_model = factorization_ops.WALSModel(
          5,
          7,
          3,
          col_init=col_init,
          row_weights=[0] * 5,
          col_weights=[0] * 7,
          use_factors_weights_cache=use_factors_weights_cache)
      wals_model.initialize_op.run()
      wals_model.worker_init.run()
      sp_feeder = array_ops.sparse_placeholder(dtypes.float32)
      # Here test partial row update with identical inputs but with transposed
      # input for als.
      sp_r_t = np_matrix_to_tf_sparse(
          INPUT_MATRIX, [3, 1], transpose=True).eval()
      sp_r = np_matrix_to_tf_sparse(INPUT_MATRIX, [3, 1]).eval()

      feed_dict = {sp_feeder: sp_r_t}
      als_model.row_update_prep_gramian_op.run()
      als_model.initialize_row_update_op.run()
      process_input_op = als_model.update_row_factors(
          sp_input=sp_feeder, transpose_input=True)[1]
      process_input_op.run(feed_dict=feed_dict)
      # Only updated row 1 and row 3, so only compare these rows since others
      # have randomly initialized values.
      row_factors1 = [
          als_model.row_factors[0].eval()[1], als_model.row_factors[0].eval()[3]
      ]
      # Testing row projection. Projection weight doesn't matter in this case
      # since the model is ALS special case. Note that the ordering of the
      # returned results will be preserved as the input feature vectors
      # ordering.
      als_projected_row_factors1 = als_model.project_row_factors(
          sp_input=sp_feeder, transpose_input=True).eval(feed_dict=feed_dict)

      feed_dict = {sp_feeder: sp_r}
      wals_model.row_update_prep_gramian_op.run()
      wals_model.initialize_row_update_op.run()
      process_input_op = wals_model.update_row_factors(sp_input=sp_feeder)[1]
      process_input_op.run(feed_dict=feed_dict)
      # Only updated row 1 and row 3, so only compare these rows since others
      # have randomly initialized values.
      row_factors2 = [
          wals_model.row_factors[0].eval()[1],
          wals_model.row_factors[0].eval()[3]
      ]
      for r1, r2 in zip(row_factors1, row_factors2):
        self.assertAllClose(r1, r2, atol=1e-3)
      # Note that the ordering of the returned projection results is preserved
      # as the input feature vectors ordering.
      self.assertAllClose(
          als_projected_row_factors1, [row_factors2[1], row_factors2[0]],
          atol=1e-3)

  def simple_train(self, model, inp, num_iterations):
    """Helper function to train model on inp for num_iterations."""
    row_update_op = model.update_row_factors(sp_input=inp)[1]
    col_update_op = model.update_col_factors(sp_input=inp)[1]

    model.initialize_op.run()
    model.worker_init.run()
    for _ in xrange(num_iterations):
      model.row_update_prep_gramian_op.run()
      model.initialize_row_update_op.run()
      row_update_op.run()
      model.col_update_prep_gramian_op.run()
      model.initialize_col_update_op.run()
      col_update_op.run()

  # Trains an ALS model for a low-rank matrix and make sure the product of
  # factors is close to the original input.
  def _run_test_train_full_low_rank_als(self, use_factors_weights_cache):
    rows = 15
    cols = 11
    dims = 3
    with ops.Graph().as_default(), self.test_session():
      data = np.dot(np.random.rand(rows, 3), np.random.rand(
          3, cols)).astype(np.float32) / 3.0
      indices = [[i, j] for i in xrange(rows) for j in xrange(cols)]
      values = data.reshape(-1)
      inp = sparse_tensor.SparseTensor(indices, values, [rows, cols])
      model = factorization_ops.WALSModel(
          rows,
          cols,
          dims,
          regularization=1e-5,
          row_weights=None,
          col_weights=None,
          use_factors_weights_cache=use_factors_weights_cache)
      self.simple_train(model, inp, 25)
      row_factor = model.row_factors[0].eval()
      col_factor = model.col_factors[0].eval()
      self.assertAllClose(
          data,
          np.dot(row_factor, np.transpose(col_factor)),
          rtol=0.01,
          atol=0.01)

  # Trains a WALS model for a low-rank matrix and make sure the product of
  # factors is close to the original input.
  def _run_test_train_full_low_rank_wals(self, use_factors_weights_cache):
    rows = 15
    cols = 11
    dims = 3

    with ops.Graph().as_default(), self.test_session():
      data = np.dot(np.random.rand(rows, 3), np.random.rand(
          3, cols)).astype(np.float32) / 3.0
      indices = [[i, j] for i in xrange(rows) for j in xrange(cols)]
      values = data.reshape(-1)
      inp = sparse_tensor.SparseTensor(indices, values, [rows, cols])
      model = factorization_ops.WALSModel(
          rows,
          cols,
          dims,
          regularization=1e-5,
          row_weights=0,
          col_weights=[0] * cols,
          use_factors_weights_cache=use_factors_weights_cache)
      self.simple_train(model, inp, 25)
      row_factor = model.row_factors[0].eval()
      col_factor = model.col_factors[0].eval()
      self.assertAllClose(
          data,
          np.dot(row_factor, np.transpose(col_factor)),
          rtol=0.01,
          atol=0.01)

  # Trains a WALS model for a partially observed low-rank matrix and makes
  # sure the product of factors is reasonably close to the original input.
  def _run_test_train_matrix_completion_wals(self, use_factors_weights_cache):
    rows = 11
    cols = 9
    dims = 4

    def keep_index(x):
      return not (x[0] + x[1]) % 4

    with ops.Graph().as_default(), self.test_session():
      row_wts = 0.1 + np.random.rand(rows)
      col_wts = 0.1 + np.random.rand(cols)
      data = np.dot(np.random.rand(rows, 3), np.random.rand(
          3, cols)).astype(np.float32) / 3.0
      indices = np.array(
          list(
              filter(keep_index,
                     [[i, j] for i in xrange(rows) for j in xrange(cols)])))
      values = data[indices[:, 0], indices[:, 1]]
      inp = sparse_tensor.SparseTensor(indices, values, [rows, cols])
      model = factorization_ops.WALSModel(
          rows,
          cols,
          dims,
          unobserved_weight=0.01,
          regularization=0.001,
          row_weights=row_wts,
          col_weights=col_wts,
          use_factors_weights_cache=use_factors_weights_cache)
      self.simple_train(model, inp, 25)
      row_factor = model.row_factors[0].eval()
      col_factor = model.col_factors[0].eval()
      out = np.dot(row_factor, np.transpose(col_factor))
      for i in xrange(rows):
        for j in xrange(cols):
          if keep_index([i, j]):
            self.assertNear(
                data[i][j], out[i][j], err=0.4, msg="%d, %d" % (i, j))
          else:
            self.assertNear(0, out[i][j], err=0.5, msg="%d, %d" % (i, j))

  def test_process_input_with_cache(self):
    self._run_test_process_input(True)

  def test_process_input_without_cache(self):
    self._run_test_process_input(False)

  def test_process_input_transposed_with_cache(self):
    self._run_test_process_input_transposed(True)

  def test_process_input_transposed_without_cache(self):
    self._run_test_process_input_transposed(False)

  def test_als_with_cache(self):
    self._run_test_als(True)

  def test_als_without_cache(self):
    self._run_test_als(False)

  def test_als_transposed_with_cache(self):
    self._run_test_als_transposed(True)

  def test_als_transposed_without_cache(self):
    self._run_test_als_transposed(False)

  def test_train_full_low_rank_wals_with_cache(self):
    self._run_test_train_full_low_rank_wals(True)

  def test_train_full_low_rank_wals_without_cache(self):
    self._run_test_train_full_low_rank_wals(False)

  def test_train_matrix_completion_wals_with_cache(self):
    self._run_test_train_matrix_completion_wals(True)

  def test_train_matrix_completion_wals_without_cache(self):
    self._run_test_train_matrix_completion_wals(False)

  def test_loss_transposed_with_cache(self):
    self._run_test_process_input_transposed(True, compute_loss=True)

  def test_loss_transposed_without_cache(self):
    self._run_test_process_input_transposed(False, compute_loss=True)

  def test_loss_with_cache(self):
    self._run_test_process_input(True, compute_loss=True)

  def test_loss_without_cache(self):
    self._run_test_process_input(False, compute_loss=True)

  def test_sum_row_weights(self):
    self._run_test_sum_weights(True)

  def test_sum_col_weights(self):
    self._run_test_sum_weights(False)


if __name__ == "__main__":
  test.main()

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

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.factorization.python.ops import factorization_ops

INPUT_MATRIX = np.array(
    [[0.1, 0.0, 0.2, 0.0, 0.4, 0.5, 0.0],
     [0.0, 1.1, 0.0, 1.3, 1.4, 0.0, 1.6],
     [2.0, 0.0, 0.0, 2.3, 0.0, 2.5, 0.0],
     [3.0, 0.0, 3.2, 3.3, 0.0, 3.5, 0.0],
     [0.0, 4.1, 0.0, 0.0, 4.4, 0.0, 4.6]]).astype(np.float32)


def np_matrix_to_tf_sparse(np_matrix, row_slices=None,
                           col_slices=None, transpose=False,
                           shuffle=False):
  """Simple util to slice non-zero np matrix elements as tf.SparseTensor."""
  indices = np.nonzero(np_matrix)

  # Only allow slices of whole rows or whole columns.
  assert not (row_slices is not None and col_slices is not None)

  if row_slices is not None:
    selected_ind = np.concatenate(
        [np.where(indices[0] == r)[0] for r in row_slices], 0)
    indices = (indices[0][selected_ind], indices[1][selected_ind])

  if col_slices is not None:
    selected_ind = np.concatenate(
        [np.where(indices[1] == c)[0] for c in col_slices], 0)
    indices = (indices[0][selected_ind], indices[1][selected_ind])

  if shuffle:
    shuffled_ind = [x for x in range(len(indices[0]))]
    random.shuffle(shuffled_ind)
    indices = (indices[0][shuffled_ind], indices[1][shuffled_ind])

  ind = (np.concatenate(
      (np.expand_dims(indices[1], 1),
       np.expand_dims(indices[0], 1)), 1).astype(np.int64) if transpose else
         np.concatenate((np.expand_dims(indices[0], 1),
                         np.expand_dims(indices[1], 1)), 1).astype(np.int64))
  val = np_matrix[indices].astype(np.float32)
  shape = (np.array(
      [max(indices[1]) + 1, max(indices[0]) + 1]).astype(np.int64) if transpose
           else np.array(
               [max(indices[0]) + 1, max(indices[1]) + 1]).astype(np.int64))
  return tf.SparseTensor(ind, val, shape)


def sparse_input():
  return np_matrix_to_tf_sparse(INPUT_MATRIX)


class WalsModelTest(tf.test.TestCase):

  def setUp(self):
    self.col_init = [
        # shard 0
        [[-0.36444709, -0.39077035, -0.32528427],
         [1.19056475, 0.07231052, 2.11834812],
         [0.93468881, -0.71099287, 1.91826844]],
        # shard 1
        [[1.18160152, 1.52490723, -0.50015002],
         [1.82574749, -0.57515913, -1.32810032]],
        # shard 2
        [[-0.15515432, -0.84675711, 0.13097958],
         [-0.9246484, 0.69117504, 1.2036494]]
    ]

    self.row_wts = [[0.1, 0.2, 0.3], [0.4, 0.5]]
    self.col_wts = [[0.1, 0.2, 0.3],
                    [0.4, 0.5],
                    [0.6, 0.7]]
    self._wals_inputs = sparse_input()

    # Values of factor shards after running one iteration of row and column
    # updates.
    self._row_factors_0 = [[0.097689, -0.219293, -0.020780],
                           [0.50842, 0.64626, 0.22364],
                           [0.401159, -0.046558, -0.192854]]
    self._row_factors_1 = [[1.20597, -0.48025, 0.35582],
                           [1.5564, 1.2528, 1.0528]]
    self._col_factors_0 = [[2.4725, -1.2950, -1.9980],
                           [0.44625, 1.50771, 1.27118],
                           [1.39801, -2.10134, 0.73572]]
    self._col_factors_1 = [[3.36509, -0.66595, -3.51208],
                           [0.57191, 1.59407, 1.33020]]
    self._col_factors_2 = [[3.3459, -1.3341, -3.3008],
                           [0.57366, 1.83729, 1.26798]]

  def test_process_input(self):
    with self.test_session():
      sp_feeder = tf.sparse_placeholder(tf.float32)
      wals_model = factorization_ops.WALSModel(5, 7, 3,
                                               num_row_shards=2,
                                               num_col_shards=3,
                                               regularization=0.01,
                                               unobserved_weight=0.1,
                                               col_init=self.col_init,
                                               row_weights=self.row_wts,
                                               col_weights=self.col_wts)

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
      wals_model.initialize_row_update_op.run()
      process_input_op = wals_model.update_row_factors(sp_input=sp_feeder,
                                                       transpose_input=False)[1]
      for inp in input_scattered_rows:
        feed_dict = {sp_feeder: inp}
        process_input_op.run(feed_dict=feed_dict)
      row_factors = [x.eval() for x in wals_model.row_factors]

      self.assertAllClose(row_factors[0], self._row_factors_0, atol=1e-3)
      self.assertAllClose(row_factors[1], self._row_factors_1, atol=1e-3)

      # Split input into multiple sparse tensors with scattered columns. Note
      # that here the elements in the sparse tensors are not ordered and also
      # do not need to consist of consecutive columns. However, each column
      # needs to include all non-zero elements in that column.
      sp_c0 = np_matrix_to_tf_sparse(INPUT_MATRIX, col_slices=[2, 0]).eval()
      sp_c1 = np_matrix_to_tf_sparse(INPUT_MATRIX, col_slices=[5, 3, 1],
                                     shuffle=True).eval()
      sp_c2 = np_matrix_to_tf_sparse(INPUT_MATRIX, col_slices=[4, 6]).eval()
      sp_c3 = np_matrix_to_tf_sparse(INPUT_MATRIX, col_slices=[3, 6],
                                     shuffle=True).eval()

      input_scattered_cols = [sp_c0, sp_c1, sp_c2, sp_c3]

      # Test updating column factors.
      # Here we feed in scattered columns of the input.
      wals_model.initialize_col_update_op.run()
      process_input_op = wals_model.update_col_factors(sp_input=sp_feeder,
                                                       transpose_input=False)[1]
      for inp in input_scattered_cols:
        feed_dict = {sp_feeder: inp}
        process_input_op.run(feed_dict=feed_dict)
      col_factors = [x.eval() for x in wals_model.col_factors]

      self.assertAllClose(col_factors[0], self._col_factors_0, atol=1e-3)
      self.assertAllClose(col_factors[1], self._col_factors_1, atol=1e-3)
      self.assertAllClose(col_factors[2], self._col_factors_2, atol=1e-3)

  def test_process_input_transposed(self):
    with self.test_session():
      sp_feeder = tf.sparse_placeholder(tf.float32)
      wals_model = factorization_ops.WALSModel(5, 7, 3,
                                               num_row_shards=2,
                                               num_col_shards=3,
                                               regularization=0.01,
                                               unobserved_weight=0.1,
                                               col_init=self.col_init,
                                               row_weights=self.row_wts,
                                               col_weights=self.col_wts)

      wals_model.initialize_op.run()
      wals_model.worker_init.run()

      # Split input into multiple SparseTensors with scattered rows.
      # Here the inputs are transposed. But the same constraints as described in
      # the previous non-transposed test case apply to these inputs (before they
      # are transposed).
      sp_r0_t = np_matrix_to_tf_sparse(INPUT_MATRIX, [0, 3],
                                       transpose=True).eval()
      sp_r1_t = np_matrix_to_tf_sparse(INPUT_MATRIX, [4, 1],
                                       shuffle=True, transpose=True).eval()
      sp_r2_t = np_matrix_to_tf_sparse(INPUT_MATRIX, [2], transpose=True).eval()
      sp_r3_t = sp_r1_t
      input_scattered_rows = [sp_r0_t, sp_r1_t, sp_r2_t, sp_r3_t]

      # Test updating row factors.
      # Here we feed in scattered rows of the input.
      # Note that the needed suffix of placeholder are in the order of test
      # case name lexicographical order and then in the line order of where
      # they appear.
      wals_model.initialize_row_update_op.run()
      process_input_op = wals_model.update_row_factors(sp_input=sp_feeder,
                                                       transpose_input=True)[1]
      for inp in input_scattered_rows:
        feed_dict = {sp_feeder: inp}
        process_input_op.run(feed_dict=feed_dict)
      row_factors = [x.eval() for x in wals_model.row_factors]

      self.assertAllClose(row_factors[0], self._row_factors_0, atol=1e-3)
      self.assertAllClose(row_factors[1], self._row_factors_1, atol=1e-3)

      # Split input into multiple SparseTensors with scattered columns.
      # Here the inputs are transposed. But the same constraints as described in
      # the previous non-transposed test case apply to these inputs (before they
      # are transposed).
      sp_c0_t = np_matrix_to_tf_sparse(INPUT_MATRIX, col_slices=[0, 1],
                                       transpose=True).eval()
      sp_c1_t = np_matrix_to_tf_sparse(INPUT_MATRIX, col_slices=[4, 2],
                                       transpose=True).eval()
      sp_c2_t = np_matrix_to_tf_sparse(INPUT_MATRIX, col_slices=[5],
                                       transpose=True, shuffle=True).eval()
      sp_c3_t = np_matrix_to_tf_sparse(INPUT_MATRIX, col_slices=[3, 6],
                                       transpose=True).eval()

      sp_c4_t = sp_c2_t
      input_scattered_cols = [sp_c0_t, sp_c1_t, sp_c2_t, sp_c3_t,
                              sp_c4_t]

      # Test updating column factors.
      # Here we feed in scattered columns of the input.
      wals_model.initialize_col_update_op.run()
      process_input_op = wals_model.update_col_factors(sp_input=sp_feeder,
                                                       transpose_input=True)[1]
      for inp in input_scattered_cols:
        feed_dict = {sp_feeder: inp}
        process_input_op.run(feed_dict=feed_dict)
      col_factors = [x.eval() for x in wals_model.col_factors]

      self.assertAllClose(col_factors[0], self._col_factors_0, atol=1e-3)
      self.assertAllClose(col_factors[1], self._col_factors_1, atol=1e-3)
      self.assertAllClose(col_factors[2], self._col_factors_2, atol=1e-3)

  # Note that when row_weights and col_weights are 0, WALS gives dentical
  # results as ALS (Alternating Least Squares). However our implementation does
  # not handle the case of zero weights differently. Instead, when row_weights
  # and col_weights are set to None, we interpret that as the ALS case, and
  # trigger the more efficient ALS updates.
  # Here we test that those two give identical results.
  def test_als(self):
    with self.test_session():
      col_init = np.random.rand(7, 3)
      als_model = factorization_ops.WALSModel(5, 7, 3,
                                              col_init=col_init,
                                              row_weights=None,
                                              col_weights=None)

      als_model.initialize_op.run()
      als_model.worker_init.run()
      als_model.initialize_row_update_op.run()
      process_input_op = als_model.update_row_factors(self._wals_inputs)[1]
      process_input_op.run()
      row_factors1 = [x.eval() for x in als_model.row_factors]

      wals_model = factorization_ops.WALSModel(5, 7, 3,
                                               col_init=col_init,
                                               row_weights=[0] * 5,
                                               col_weights=[0] * 7)
      wals_model.initialize_op.run()
      wals_model.worker_init.run()
      wals_model.initialize_row_update_op.run()
      process_input_op = wals_model.update_row_factors(self._wals_inputs)[1]
      process_input_op.run()
      row_factors2 = [x.eval() for x in wals_model.row_factors]

      for r1, r2 in zip(row_factors1, row_factors2):
        self.assertAllClose(r1, r2, atol=1e-3)

      # Here we test partial column updates.
      sp_c = np_matrix_to_tf_sparse(INPUT_MATRIX, col_slices=[2, 0],
                                    shuffle=True).eval()

      sp_feeder = tf.sparse_placeholder(tf.float32)
      feed_dict = {sp_feeder: sp_c}
      als_model.initialize_col_update_op.run()
      process_input_op = als_model.update_col_factors(sp_input=sp_feeder)[1]
      process_input_op.run(feed_dict=feed_dict)
      col_factors1 = [x.eval() for x in als_model.col_factors]

      feed_dict = {sp_feeder: sp_c}
      wals_model.initialize_col_update_op.run()
      process_input_op = wals_model.update_col_factors(sp_input=sp_feeder)[1]
      process_input_op.run(feed_dict=feed_dict)
      col_factors2 = [x.eval() for x in wals_model.col_factors]

      for c1, c2 in zip(col_factors1, col_factors2):
        self.assertAllClose(c1, c2, rtol=5e-3, atol=1e-2)

  def test_als_transposed(self):
    with self.test_session():
      col_init = np.random.rand(7, 3)
      als_model = factorization_ops.WALSModel(5, 7, 3,
                                              col_init=col_init,
                                              row_weights=None,
                                              col_weights=None)

      als_model.initialize_op.run()
      als_model.worker_init.run()

      wals_model = factorization_ops.WALSModel(5, 7, 3,
                                               col_init=col_init,
                                               row_weights=[0] * 5,
                                               col_weights=[0] * 7)
      wals_model.initialize_op.run()
      wals_model.worker_init.run()
      sp_feeder = tf.sparse_placeholder(tf.float32)
      # Here test partial row update with identical inputs but with transposed
      # input for als.
      sp_r_t = np_matrix_to_tf_sparse(INPUT_MATRIX, [3, 1],
                                      transpose=True).eval()
      sp_r = np_matrix_to_tf_sparse(INPUT_MATRIX, [3, 1]).eval()

      feed_dict = {sp_feeder: sp_r_t}
      als_model.initialize_row_update_op.run()
      process_input_op = als_model.update_row_factors(sp_input=sp_feeder,
                                                      transpose_input=True)[1]
      process_input_op.run(feed_dict=feed_dict)
      # Only updated row 1 and row 3, so only compare these rows since others
      # have randomly initialized values.
      row_factors1 = [als_model.row_factors[0].eval()[1],
                      als_model.row_factors[0].eval()[3]]

      feed_dict = {sp_feeder: sp_r}
      wals_model.initialize_row_update_op.run()
      process_input_op = wals_model.update_row_factors(sp_input=sp_feeder)[1]
      process_input_op.run(feed_dict=feed_dict)
      # Only updated row 1 and row 3, so only compare these rows since others
      # have randomly initialized values.
      row_factors2 = [wals_model.row_factors[0].eval()[1],
                      wals_model.row_factors[0].eval()[3]]
      for r1, r2 in zip(row_factors1, row_factors2):
        self.assertAllClose(r1, r2, atol=1e-3)

  def simple_train(self,
                   model,
                   inp,
                   num_iterations):
    """Helper function to train model on inp for num_iterations."""
    row_update_op = model.update_row_factors(sp_input=inp)[1]
    col_update_op = model.update_col_factors(sp_input=inp)[1]

    model.initialize_op.run()
    model.worker_init.run()
    for _ in xrange(num_iterations):
      model.initialize_row_update_op.run()
      row_update_op.run()
      model.initialize_col_update_op.run()
      col_update_op.run()

  # Trains an ALS model for a low-rank matrix and make sure the product of
  # factors is close to the original input.
  def test_train_full_low_rank_als(self):
    rows = 15
    cols = 11
    dims = 3
    with self.test_session():
      data = np.dot(np.random.rand(rows, 3),
                    np.random.rand(3, cols)).astype(np.float32) / 3.0
      indices = [[i, j] for i in xrange(rows) for j in xrange(cols)]
      values = data.reshape(-1)
      inp = tf.SparseTensor(indices, values, [rows, cols])
      model = factorization_ops.WALSModel(rows, cols, dims,
                                          regularization=1e-5,
                                          row_weights=None,
                                          col_weights=None)
      self.simple_train(model, inp, 25)
      row_factor = model.row_factors[0].eval()
      col_factor = model.col_factors[0].eval()
      self.assertAllClose(data,
                          np.dot(row_factor, np.transpose(col_factor)),
                          rtol=0.01, atol=0.01)

  # Trains a WALS model for a low-rank matrix and make sure the product of
  # factors is close to the original input.
  def test_train_full_low_rank_wals(self):
    rows = 15
    cols = 11
    dims = 3

    with self.test_session():
      data = np.dot(np.random.rand(rows, 3),
                    np.random.rand(3, cols)).astype(np.float32) / 3.0
      indices = [[i, j] for i in xrange(rows) for j in xrange(cols)]
      values = data.reshape(-1)
      inp = tf.SparseTensor(indices, values, [rows, cols])
      model = factorization_ops.WALSModel(rows, cols, dims,
                                          regularization=1e-5,
                                          row_weights=[0] * rows,
                                          col_weights=[0] * cols)
      self.simple_train(model, inp, 25)
      row_factor = model.row_factors[0].eval()
      col_factor = model.col_factors[0].eval()
      self.assertAllClose(data,
                          np.dot(row_factor, np.transpose(col_factor)),
                          rtol=0.01, atol=0.01)

  # Trains a WALS model for a partially observed low-rank matrix and makes
  # sure the product of factors is reasonably close to the original input.
  def test_train_matrix_completion_wals(self):
    rows = 11
    cols = 9
    dims = 4
    def keep_index(x):
      return not (x[0] + x[1]) % 4

    with self.test_session():
      row_wts = 0.1 + np.random.rand(rows)
      col_wts = 0.1 + np.random.rand(cols)
      data = np.dot(np.random.rand(rows, 3),
                    np.random.rand(3, cols)).astype(np.float32) / 3.0
      indices = np.array(
          list(filter(keep_index,
                      [[i, j] for i in xrange(rows) for j in xrange(cols)])))
      values = data[indices[:, 0], indices[:, 1]]
      inp = tf.SparseTensor(indices, values, [rows, cols])
      model = factorization_ops.WALSModel(rows, cols, dims,
                                          unobserved_weight=0.01,
                                          regularization=0.001,
                                          row_weights=row_wts,
                                          col_weights=col_wts)
      self.simple_train(model, inp, 25)
      row_factor = model.row_factors[0].eval()
      col_factor = model.col_factors[0].eval()
      out = np.dot(row_factor, np.transpose(col_factor))
      for i in xrange(rows):
        for j in xrange(cols):
          if keep_index([i, j]):
            self.assertNear(data[i][j], out[i][j],
                            err=0.4, msg="%d, %d" % (i, j))
          else:
            self.assertNear(0, out[i][j], err=0.5, msg="%d, %d" % (i, j))


if __name__ == "__main__":
  tf.test.main()

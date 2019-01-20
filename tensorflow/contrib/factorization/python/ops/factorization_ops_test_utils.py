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
"""Test utils for factorization_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops


INPUT_MATRIX = np.array(
    [[0.1, 0.0, 0.2, 0.0, 0.4, 0.5, 0.0],
     [0.0, 1.1, 0.0, 1.3, 1.4, 0.0, 1.6],
     [2.0, 0.0, 0.0, 2.3, 0.0, 2.5, 0.0],
     [3.0, 0.0, 3.2, 3.3, 0.0, 3.5, 0.0],
     [0.0, 4.1, 0.0, 0.0, 4.4, 0.0, 4.6]]).astype(np.float32)


def remove_empty_rows_columns(np_matrix):
  """Simple util to remove empty rows and columns of a matrix.

  Args:
    np_matrix: A numpy array.
  Returns:
    A tuple consisting of:
    mat: A numpy matrix obtained by removing empty rows and columns from
      np_matrix.
    nz_row_ids: A numpy array of the ids of non-empty rows, such that
      nz_row_ids[i] is the old row index corresponding to new index i.
    nz_col_ids: A numpy array of the ids of non-empty columns, such that
      nz_col_ids[j] is the old column index corresponding to new index j.
  """
  nz_row_ids = np.where(np.sum(np_matrix, axis=1) != 0)[0]
  nz_col_ids = np.where(np.sum(np_matrix, axis=0) != 0)[0]
  mat = np_matrix[np.ix_(nz_row_ids, nz_col_ids)]
  return mat, nz_row_ids, nz_col_ids


def np_matrix_to_tf_sparse(np_matrix,
                           row_slices=None,
                           col_slices=None,
                           transpose=False,
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

  ind = (np.concatenate((np.expand_dims(indices[1], 1),
                         np.expand_dims(indices[0], 1)), 1).astype(np.int64) if
         transpose else np.concatenate((np.expand_dims(indices[0], 1),
                                        np.expand_dims(indices[1], 1)),
                                       1).astype(np.int64))
  val = np_matrix[indices].astype(np.float32)
  shape = (np.array([max(indices[1]) + 1, max(indices[0]) + 1]).astype(np.int64)
           if transpose else np.array(
               [max(indices[0]) + 1, max(indices[1]) + 1]).astype(np.int64))
  return sparse_tensor.SparseTensor(ind, val, shape)


def calculate_loss(input_mat, row_factors, col_factors, regularization=None,
                   w0=1., row_weights=None, col_weights=None):
  """Calculates the loss of a given factorization.

  Using a non distributed method, different than the one implemented in the
  WALS model. The weight of an observed entry (i, j) (i.e. such that
  input_mat[i, j] is non zero) is (w0 + row_weights[i]col_weights[j]).

  Args:
    input_mat: The input matrix, a SparseTensor of rank 2.
    row_factors: The row factors, a dense Tensor of rank 2.
    col_factors: The col factors, a dense Tensor of rank 2.
    regularization: the regularization coefficient, a scalar.
    w0: the weight of unobserved entries. A scalar.
    row_weights: A dense tensor of rank 1.
    col_weights: A dense tensor of rank 1.

  Returns:
    The total loss.
  """
  wr = (array_ops.expand_dims(row_weights, 1) if row_weights is not None
        else constant_op.constant(1.))
  wc = (array_ops.expand_dims(col_weights, 0) if col_weights is not None
        else constant_op.constant(1.))
  reg = (regularization if regularization is not None
         else constant_op.constant(0.))

  row_indices, col_indices = array_ops.split(input_mat.indices,
                                             axis=1,
                                             num_or_size_splits=2)
  gathered_row_factors = array_ops.gather(row_factors, row_indices)
  gathered_col_factors = array_ops.gather(col_factors, col_indices)
  sp_approx_vals = array_ops.squeeze(math_ops.matmul(
      gathered_row_factors, gathered_col_factors, adjoint_b=True))
  sp_approx = sparse_tensor.SparseTensor(
      indices=input_mat.indices,
      values=sp_approx_vals,
      dense_shape=input_mat.dense_shape)

  sp_approx_sq = math_ops.square(sp_approx)
  row_norm = math_ops.reduce_sum(math_ops.square(row_factors))
  col_norm = math_ops.reduce_sum(math_ops.square(col_factors))
  row_col_norm = math_ops.reduce_sum(math_ops.square(math_ops.matmul(
      row_factors, col_factors, transpose_b=True)))

  resid = sparse_ops.sparse_add(input_mat, sp_approx * (-1))
  resid_sq = math_ops.square(resid)
  loss = w0 * (
      sparse_ops.sparse_reduce_sum(resid_sq) -
      sparse_ops.sparse_reduce_sum(sp_approx_sq)
      )
  loss += (sparse_ops.sparse_reduce_sum(wr * (resid_sq * wc)) +
           w0 * row_col_norm + reg * (row_norm + col_norm))
  return loss.eval()

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

"""Ops for matrix factorization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# pylint: disable=wildcard-import,undefined-variable
from tensorflow.contrib.factorization.python.ops.gen_factorization_ops import *
from tensorflow.python.framework import ops
from tensorflow.python.framework.load_library import load_op_library
from tensorflow.python.platform import resource_loader

_factorization_ops = load_op_library(resource_loader.get_path_to_datafile(
    "_factorization_ops.so"))
assert _factorization_ops, "Could not load _factorization_ops.so"


class WALSModel(object):
  r"""A model for Weighted Alternating Least Squares matrix factorization.

  It minimizes the following loss function over U, V:
   \\( ||W \odot (A - U V^T) ||_F^2 + \lambda (||U||_F^2 + ||V||_F^2) )\\
    where,
    A: input matrix,
    W: weight matrix,
    U, V: row_factors and column_factors matrices,
    \\(\lambda)\\: regularization.
  Also we assume that W is of the following special form:
  \\( W_{ij} = W_0 + R_i * C_j )\\  if \\(A_{ij} \ne 0)\\,
  \\(W_{ij} = W_0)\\ otherwise.
  where,
  \\(W_0)\\: unobserved_weight,
  \\(R_i)\\: row_weights,
  \\(C_j)\\: col_weights.

  Note that the current implementation assumes that row_factors and col_factors
  can individually fit into the memory of each worker.
  """

  def __init__(self,
               input_rows,
               input_cols,
               n_components,
               unobserved_weight=0.1,
               regularization=None,
               row_init="random",
               col_init="random",
               num_row_shards=1,
               num_col_shards=1,
               row_weights=None,
               col_weights=None):
    """Creates model for WALS matrix factorization.

    Args:
      input_rows: total number of rows for input matrix.
      input_cols: total number of cols for input matrix.
      n_components: number of dimensions to use for the factors.
      unobserved_weight: weight given to unobserved entries of matrix.
      regularization: weight of L2 regularization term. If None, no
        regularization is done.
      row_init: initializer for row factor. Can be a tensor or numpy constant.
        If set to "random", the value is initialized randomly.
      col_init: initializer for column factor. See row_init for details.
      num_row_shards: number of shards to use for row factors.
      num_col_shards: number of shards to use for column factors.
      row_weights: If not None, along with col_weights, used to compute the
        weight of an observed entry. w_ij = unobserved_weight + row_weights[i] *
        col_weights[j]. If None, then w_ij = unobserved_weight, which simplifies
        to ALS.
      col_weights: See row_weights
    """
    self._input_rows = input_rows
    self._input_cols = input_cols
    self._num_row_shards = num_row_shards
    self._num_col_shards = num_col_shards
    self._n_components = n_components
    self._unobserved_weight = unobserved_weight
    self._regularization = (tf.diag(tf.constant(regularization,
                                                shape=[self._n_components],
                                                dtype=tf.float32))
                            if regularization is not None else None)
    assert (row_weights is None) == (col_weights is None)
    self._row_weights = WALSModel._create_weights(row_weights,
                                                  self._input_rows,
                                                  self._num_row_shards,
                                                  "row_weights")
    self._col_weights = WALSModel._create_weights(col_weights,
                                                  self._input_cols,
                                                  self._num_col_shards,
                                                  "col_weights")
    self._row_factors = self._create_factors(self._input_rows,
                                             self._n_components,
                                             self._num_row_shards,
                                             row_init,
                                             "row_factors")
    self._col_factors = self._create_factors(self._input_cols,
                                             self._n_components,
                                             self._num_col_shards,
                                             col_init,
                                             "col_factors")
    self._create_transient_vars()

  @property
  def row_factors(self):
    """Returns a list of tensors corresponding to row factor shards."""
    return self._row_factors

  @property
  def col_factors(self):
    """Returns a list of tensors corresponding to column factor shards."""
    return self._col_factors

  @property
  def initialize_op(self):
    """Returns an op for initializing tensorflow variables."""
    all_vars = self._row_factors + self._col_factors
    if self._row_weights is not None:
      assert self._col_weights is not None
      all_vars.extend(self._row_weights + self._col_weights)
    return tf.initialize_variables(all_vars)

  @classmethod
  def _shard_sizes(cls, dims, num_shards):
    """Helper function to split dims values into num_shards."""
    shard_size, residual = divmod(dims, num_shards)
    return [shard_size + 1] * residual + [shard_size] * (num_shards - residual)

  @classmethod
  def _create_factors(cls, rows, cols, num_shards, init, name):
    """Helper function to create row and column factors."""
    if callable(init):
      init = init()
    if isinstance(init, list):
      assert len(init) == num_shards
    elif isinstance(init, str) and init == "random":
      pass
    elif num_shards == 1:
      init = [init]
    sharded_matrix = []
    sizes = cls._shard_sizes(rows, num_shards)
    assert len(sizes) == num_shards

    def make_initializer(i, size):
      def initializer():
        if init == "random":
          return tf.random_normal([size, cols])
        else:
          return init[i]
      return initializer

    for i, size in enumerate(sizes):
      var_name = "%s_shard_%d" % (name, i)
      var_init = make_initializer(i, size)
      sharded_matrix.append(tf.Variable(
          var_init,
          dtype=tf.float32,
          name=var_name))

    return sharded_matrix

  @staticmethod
  def _create_weights(wt_init, num_wts, num_shards, name):
    """Helper functions to create sharded weight vector.

    Args:
      wt_init: init value for the weight. If None, weights are not created.
      num_wts: total size of all the weight shards
      num_shards: number of shards for the weights
      name: name for the new Variables.

    Returns:
      A list of weight shard Tensors.
    """
    if wt_init is None:
      return None
    if num_shards == 1 and len(wt_init) == num_wts:
      wt_init = [wt_init]
    assert len(wt_init) == num_shards
    return [tf.Variable(wt_init[i],
                        dtype=tf.float32,
                        name="%s_shard_%d" % (name, i))
            for i in xrange(num_shards)]

  @staticmethod
  def _transient_var(name):
    """Helper function to create a Variable."""
    return tf.Variable(1.0,
                       trainable=False,
                       collections=[tf.GraphKeys.LOCAL_VARIABLES],
                       validate_shape=False,
                       name=name)

  def _cached_copy(self, var, name):
    """Helper function to create a worker cached copy of a Variable.

    Args:
      var: Variable or list of Variable to cache. If a list, the items are
        concatenated along dimension 0 to get the cached entry.
      name: name of cached variable.

    Returns:
      Tuple consisting of following three entries:
      cache: the new transient Variable.
      cache_init: op to initialize the Variable
      cache_reset: op to reset the Variable to some default value
    """
    if var is None:
      return None, None, None
    else:
      cache = WALSModel._transient_var(name)
      with ops.colocate_with(cache):
        if isinstance(var, list):
          assert var
          if len(var) == 1:
            var = var[0]
          else:
            var = tf.concat(0, var)

      cache_init = tf.assign(cache, var, validate_shape=False)
      cache_reset = tf.assign(cache, 1.0, validate_shape=False)
      return cache, cache_init, cache_reset

  def _create_transient_vars(self):
    """Creates local cache of row and column factors and weights.

    Note that currently the caching strategy is as follows:
    When initiating a row update, column factors are cached while row factors
    cache is reset.  Similarly when initiating a column update, row factors are
    cached while cached column factors are flushed.
    Column and row weights are always cached. If memory becomes a bottleneck,
    they could be similarly flushed.
    """
    (self._row_factors_cache,
     row_factors_cache_init,
     row_factors_cache_reset) = self._cached_copy(self._row_factors,
                                                  "row_factors_cache")
    (self._col_factors_cache,
     col_factors_cache_init,
     col_factors_cache_reset) = self._cached_copy(self._col_factors,
                                                  "col_factors_cache")
    (self._row_wt_cache,
     row_wt_cache_init,
     _) = self._cached_copy(self._row_weights, "row_wt_cache")
    (self._col_wt_cache,
     col_wt_cache_init,
     _) = self._cached_copy(self._col_weights, "col_wt_cache")

    if self._row_wt_cache is not None:
      assert self._col_wt_cache is not None
      self._worker_init = tf.group(row_wt_cache_init,
                                   col_wt_cache_init,
                                   name="worker_init")
    else:
      self._worker_init = tf.no_op(name="worker_init")

    self._row_updates_init = tf.group(col_factors_cache_init,
                                      row_factors_cache_reset)
    self._col_updates_init = tf.group(row_factors_cache_init,
                                      col_factors_cache_reset)

  @property
  def worker_init(self):
    """Op to initialize worker state once before starting any updates."""
    return self._worker_init

  @property
  def initialize_row_update_op(self):
    """Op to initialize worker state before starting row updates."""
    return self._row_updates_init

  @property
  def initialize_col_update_op(self):
    """Op to initialize worker state before starting column updates."""
    return self._col_updates_init

  @staticmethod
  def _get_sharding_func(size, num_shards):
    """Create sharding function for scatter update."""
    def func(ids):
      if num_shards == 1:
        return None, ids
      else:
        ids_per_shard = size // num_shards
        extras = size % num_shards
        assignments = tf.maximum(ids // (ids_per_shard + 1),
                                 (ids - extras) // ids_per_shard)
        new_ids = tf.select(assignments < extras,
                            ids % (ids_per_shard + 1),
                            (ids - extras) % ids_per_shard)
        return assignments, new_ids
    return func

  @classmethod
  def scatter_update(cls, factor, indices, values, sharding_func):
    """Helper function for doing sharded scatter update."""
    assert isinstance(factor, list)
    if len(factor) == 1:
      with ops.colocate_with(factor[0]):
        # TODO(agarwal): assign instead of scatter update for full batch update.
        return tf.scatter_update(factor[0], indices, values).op
    else:
      num_shards = len(factor)
      assignments, new_ids = sharding_func(indices)
      assert assignments is not None
      assignments = tf.cast(assignments, tf.int32)
      sharded_ids = tf.dynamic_partition(new_ids, assignments, num_shards)
      sharded_values = tf.dynamic_partition(values, assignments, num_shards)
      updates = []
      for i in xrange(num_shards):
        updates.append(tf.scatter_update(factor[i],
                                         sharded_ids[i],
                                         sharded_values[i]))
      return tf.group(*updates)

  def update_row_factors(self, sp_input=None, transpose_input=False):
    """Updates the row factors.

    Args:
      sp_input: A SparseTensor representing a subset of rows of the full input
       in any order. Please note that this SparseTensor must retain the
       indexing as the original input.
      transpose_input: If true, logically transposes the input.

    Returns:
      A tuple consisting of the following two elements:
      new_values: New values for the row factors.
      update_op: An op that assigns the newly computed values to the row
        factors.
    """
    return self._process_input_helper(True, sp_input=sp_input,
                                      transpose_input=transpose_input)

  def update_col_factors(self, sp_input=None, transpose_input=False):
    """Updates the column factors.

    Args:
      sp_input: A SparseTensor representing a subset of columns of the full
        input. Please refer to comments for update_row_factors for
        restrictions.
      transpose_input: If true, logically transposes the input.

    Returns:
      A tuple consisting of the following two elements:
      new_values: New values for the column factors.
      update_op: An op that assigns the newly computed values to the column
        factors.
    """
    return self._process_input_helper(False, sp_input=sp_input,
                                      transpose_input=transpose_input)

  def _process_input_helper(self, update_row_factors,
                            sp_input=None, transpose_input=False):
    """Creates the graph for processing a sparse slice of input.

    Args:
      update_row_factors: if True, update the row_factors, else update the
        column factors.
      sp_input: Please refer to comments for update_row_factors and
        update_col_factors.
      transpose_input: If true, logically transpose the input.

    Returns:
      A tuple consisting of the following two elements:
      new_values: New values for the row/column factors.
      update_op: An op that assigns the newly computed values to the row/column
        factors.
    """
    assert isinstance(sp_input, ops.SparseTensor)

    if update_row_factors:
      left = self._row_factors
      right = self._col_factors_cache
      row_weights = self._row_wt_cache
      col_weights = self._col_wt_cache
      sharding_func = WALSModel._get_sharding_func(self._input_rows,
                                                   self._num_row_shards)
      right_length = self._input_cols
    else:
      left = self._col_factors
      right = self._row_factors_cache
      row_weights = self._col_wt_cache
      col_weights = self._row_wt_cache
      sharding_func = WALSModel._get_sharding_func(self._input_cols,
                                                   self._num_col_shards)
      right_length = self._input_rows
      transpose_input = not transpose_input

    # Note that the row indices of sp_input are based on the original full input
    # Here we reindex the rows and give them contiguous ids starting at 0.
    # We use tf.unique to achieve this reindexing. Note that this is done so
    # that the downstream kernel can assume that the input is "dense" along the
    # row dimension.
    row_ids, col_ids = tf.split(1, 2, sp_input.indices)

    if transpose_input:
      update_indices, all_ids = tf.unique(col_ids[:, 0])
      col_ids = tf.expand_dims(tf.cast(all_ids, tf.int64), 1)
    else:
      update_indices, all_ids = tf.unique(row_ids[:, 0])
      row_ids = tf.expand_dims(tf.cast(all_ids, tf.int64), 1)

    num_rows = tf.cast(tf.shape(update_indices)[0], tf.int64)
    row_shape = tf.constant([right_length], tf.int64)
    col_shape = [num_rows]

    new_sp_indices = tf.concat(1, [row_ids, col_ids])
    new_sp_shape = (tf.concat(0, [row_shape, col_shape]) if transpose_input
                    else tf.concat(0, [col_shape, row_shape]))
    new_sp_input = tf.SparseTensor(indices=new_sp_indices,
                                   values=sp_input.values, shape=new_sp_shape)

    # Compute lhs and rhs of the normal equations
    total_lhs = (self._unobserved_weight *
                 tf.matmul(right, right, transpose_a=True))
    if self._regularization is not None:
      total_lhs += self._regularization
    if self._row_weights is None:
      # Special case of ALS. Use a much simpler update rule.
      total_rhs = (self._unobserved_weight *
                   tf.sparse_tensor_dense_matmul(new_sp_input, right,
                                                 adjoint_a=transpose_input))
      # TODO(rmlarsen): handle transposing in tf.matrix_solve instead of
      # transposing explicitly.
      # TODO(rmlarsen): multi-thread tf.matrix_solve.
      new_left_values = tf.transpose(tf.matrix_solve(total_lhs,
                                                     tf.transpose(total_rhs)))
    else:
      row_weights_slice = tf.gather(row_weights, update_indices)
      partial_lhs, total_rhs = wals_compute_partial_lhs_and_rhs(
          right,
          col_weights,
          self._unobserved_weight,
          row_weights_slice,
          new_sp_input.indices,
          new_sp_input.values,
          num_rows,
          transpose_input,
          name="wals_compute_partial_lhs_rhs")
      total_lhs = tf.expand_dims(total_lhs, 0) + partial_lhs
      total_rhs = tf.expand_dims(total_rhs, -1)
      new_left_values = tf.squeeze(tf.batch_matrix_solve(total_lhs, total_rhs),
                                   [2])

    return (new_left_values,
            self.scatter_update(left,
                                update_indices,
                                new_left_values,
                                sharding_func))

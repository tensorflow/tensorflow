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

import collections
import numbers

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# pylint: disable=wildcard-import,undefined-variable
from tensorflow.contrib.factorization.python.ops.gen_factorization_ops import *
# pylint: enable=wildcard-import
from tensorflow.contrib.util import loader
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.platform import resource_loader

_factorization_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile("_factorization_ops.so"))


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

  Note that the current implementation supports two operation modes: The default
  mode is for the condition where row_factors and col_factors can individually
  fit into the memory of each worker and these will be cached. When this
  condition can't be met, setting use_factors_weights_cache to False allows the
  larger problem sizes with slight performance penalty as this will avoid
  creating the worker caches and instead the relevant weight and factor values
  are looked up from parameter servers at each step.

  A typical usage example (pseudocode):

    with tf.Graph().as_default():
      # Set up the model object.
      model = tf.contrib.factorization.WALSModel(....)

      # To be run only once as part of session initialization. In distributed
      # training setting, this should only be run by the chief trainer and all
      # other trainers should block until this is done.
      model_init_op = model.initialize_op

      # To be run once per worker after session is available, prior to
      # the gramian_prep_ops for row(column) can be run.
      worker_init_op = model.worker_init

      # To be run once per interation sweep before the row(column) update
      # initialize ops can be run. Note that in the distributed training
      # situations, this should only be run by the chief trainer. All other
      # trainers need to block until this is done.
      row_update_gramian_prep_op = model.row_update_prep_gramian_op
      col_update_gramian_prep_op = model.col_update_prep_gramian_op

      # To be run once per worker per iteration sweep. Must be run before
      # any actual update ops can be run.
      init_row_update_op = model.initialize_row_update_op
      init_col_update_op = model.initialize_col_update_op

      # Ops to upate row(column). This can either take the entire sparse tensor
      # or slices of sparse tensor. For distributed trainer, each trainer
      # handles just part of the matrix.
      row_update_op = model.update_row_factors(
           sp_input=matrix_slices_from_queue_for_worker_shard)[1]
      col_update_op = model.update_col_factors(
           sp_input=transposed_matrix_slices_from_queue_for_worker_shard,
           transpose_input=True)[1]

      ...

      # model_init_op is passed to Supervisor. Chief trainer runs it. Other
      # trainers wait.
      sv = tf.Supervisor(is_chief=is_chief,
                         ...,
                         init_op=tf.group(..., model_init_op, ...), ...)
      ...

      with sv.managed_session(...) as sess:
        # All workers/trainers run it after session becomes available.
        worker_init_op.run(session=sess)

        ...

        while i in iterations:

          # All trainers need to sync up here.
          while not_all_ready:
            wait

          # Row update sweep.
          if is_chief:
            row_update_gramian_prep_op.run(session=sess)
          else:
            wait_for_chief

          # All workers run upate initialization.
          init_row_update_op.run(session=sess)

          # Go through the matrix.
          reset_matrix_slices_queue_for_worker_shard
          while_matrix_slices:
            row_update_op.run(session=sess)

          # All trainers need to sync up here.
          while not_all_ready:
            wait

          # Column update sweep.
          if is_chief:
            col_update_gramian_prep_op.run(session=sess)
          else:
            wait_for_chief

          # All workers run upate initialization.
          init_col_update_op.run(session=sess)

          # Go through the matrix.
          reset_transposed_matrix_slices_queue_for_worker_shard
          while_transposed_matrix_slices:
            col_update_op.run(session=sess)
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
               row_weights=1,
               col_weights=1,
               use_factors_weights_cache=True):
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
      row_weights: Must be in one of the following three formats: None, a list
        of lists of non-negative real numbers (or equivalent iterables) or a
        single non-negative real number.
        - When set to None, w_ij = unobserved_weight, which simplifies to ALS.
        Note that col_weights must also be set to "None" in this case.
        - If it is a list of lists of non-negative real numbers, it needs to be
        in the form of [[w_0, w_1, ...], [w_k, ... ], [...]], with the number of
        inner lists matching the number of row factor shards and the elements in
        each inner list are the weights for the rows of the corresponding row
        factor shard. In this case,  w_ij = unonbserved_weight +
                                            row_weights[i] * col_weights[j].
        - If this is a single non-negative real number, this value is used for
        all row weights and w_ij = unobserved_weight + row_weights *
                                   col_weights[j].
        Note that it is allowed to have row_weights as a list while col_weights
        a single number or vice versa.
      col_weights: See row_weights.
      use_factors_weights_cache: When True, the factors and weights will be
        cached on the workers before the updates start. Defaults to True.
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
    self._use_factors_weights_cache = use_factors_weights_cache
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
    self._row_gramian = self._create_gramian(self._n_components,
                                             "row_gramian")
    self._col_gramian = self._create_gramian(self._n_components,
                                             "col_gramian")
    self._row_update_prep_gramian = self._prepare_gramian(self._col_factors,
                                                          self._col_gramian)
    self._col_update_prep_gramian = self._prepare_gramian(self._row_factors,
                                                          self._row_gramian)
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
  def row_weights(self):
    """Returns a list of tensors corresponding to row weight shards."""
    return self._row_weights

  @property
  def col_weights(self):
    """Returns a list of tensors corresponding to col weight shards."""
    return self._col_weights

  @property
  def initialize_op(self):
    """Returns an op for initializing tensorflow variables."""
    all_vars = self._row_factors + self._col_factors
    all_vars.extend([self._row_gramian, self._col_gramian])
    if self._row_weights is not None:
      assert self._col_weights is not None
      all_vars.extend(self._row_weights + self._col_weights)
    return tf.variables_initializer(all_vars)

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

  @classmethod
  def _create_weights(cls, wt_init, num_wts, num_shards, name):
    """Helper function to create sharded weight vector.

    Args:
      wt_init: init value for the weight. If None, weights are not created. This
        can be one of the None, a list of non-negative real numbers or a single
        non-negative real number (or equivalent iterables).
      num_wts: total size of all the weight shards
      num_shards: number of shards for the weights
      name: name for the new Variables.

    Returns:
      A list of weight shard Tensors.

    Raises:
      ValueError: If wt_init is not the right format.
    """

    if wt_init is None:
      return None

    init_mode = "list"
    if isinstance(wt_init, collections.Iterable):
      if num_shards == 1 and len(wt_init) == num_wts:
        wt_init = [wt_init]
      assert len(wt_init) == num_shards
    elif isinstance(wt_init, numbers.Real) and wt_init >= 0:
      init_mode = "scalar"
    else:
      raise ValueError(
          "Invalid weight initialization argument. Must be one of these: "
          "None, a real non-negative real number, or a list of lists of "
          "non-negative real numbers (or equivalent iterables) corresponding "
          "to sharded factors.")

    sizes = cls._shard_sizes(num_wts, num_shards)
    assert len(sizes) == num_shards

    def make_wt_initializer(i, size):
      def initializer():
        if init_mode == "scalar":
          return wt_init * tf.ones([size])
        else:
          return wt_init[i]
      return initializer

    sharded_weight = []
    for i, size in enumerate(sizes):
      var_name = "%s_shard_%d" % (name, i)
      var_init = make_wt_initializer(i, size)
      sharded_weight.append(tf.Variable(
          var_init,
          dtype=tf.float32,
          name=var_name))

    return sharded_weight

  @staticmethod
  def _create_gramian(n_components, name):
    """Helper function to create the gramian variable.

    Args:
      n_components: number of dimensions of the factors from which the gramian
        will be calculated.
      name: name for the new Variables.

    Returns:
      A gramian Tensor with shape of [n_components, n_components].
    """
    return tf.Variable(tf.zeros([n_components, n_components]),
                       dtype=tf.float32,
                       name=name)

  @staticmethod
  def _transient_var(name):
    """Helper function to create a Variable."""
    return tf.Variable(1.0,
                       trainable=False,
                       collections=[tf.GraphKeys.LOCAL_VARIABLES],
                       validate_shape=False,
                       name=name)

  def _prepare_gramian(self, factors, gramian):
    """Helper function to create ops to prepare/calculate gramian.

    Args:
      factors: Variable or list of Variable representing (sharded) factors.
        Used to compute the updated corresponding gramian value.
      gramian: Variable storing the gramian calculated from the factors.

    Returns:
      A op that updates the gramian with the calcuated value from the factors.
    """
    partial_gramians = []
    for f in factors:
      with ops.colocate_with(f):
        partial_gramians.append(tf.matmul(f, f, transpose_a=True))

    with ops.colocate_with(gramian):
      prep_gramian = tf.assign(gramian, tf.add_n(partial_gramians)).op

    return prep_gramian

  def _cached_copy(self, var, name, pass_through=False):
    """Helper function to create a worker cached copy of a Variable.

    This assigns the var (either a single Variable or a list of Variables) to
    local transient cache Variable(s). Note that if var is a list of Variables,
    the assignment is done sequentially to minimize the memory overheads.
    Also note that if pass_through is set to True, this does not create new
    Variables but simply return the input back.

    Args:
      var: A Variable or a list of Variables to cache.
      name: name of cached Variable.
      pass_through: when set to True, this simply pass through the var back
        through identity operator and does not actually creates a cache.

    Returns:
      Tuple consisting of following three entries:
      cache: the new transient Variable or list of transient Variables
        corresponding one-to-one with var.
      cache_init: op to initialize the Variable or the list of Variables.
      cache_reset: op to reset the Variable or the list of Variables to some
        default value.
    """
    if var is None:
      return None, None, None
    elif pass_through:
      cache = var
      cache_init = tf.no_op()
      cache_reset = tf.no_op()
    elif isinstance(var, tf.Variable):
      cache = WALSModel._transient_var(name=name)
      with ops.colocate_with(cache):
        cache_init = tf.assign(cache, var, validate_shape=False)
        cache_reset = tf.assign(cache, 1.0, validate_shape=False)
    else:
      assert isinstance(var, list)
      assert var
      cache = [WALSModel._transient_var(name='%s_shard_%d' % (name, i))
               for i in xrange(len(var))]
      reset_ops = []
      for i, c in enumerate(cache):
        with ops.colocate_with(c):
          if i == 0:
            cache_init = tf.assign(c, var[i], validate_shape=False)
          else:
            with ops.control_dependencies([cache_init]):
              cache_init = tf.assign(c, var[i], validate_shape=False)
          reset_ops.append(tf.assign(c, 1.0, validate_shape=False))
      cache_reset = tf.group(*reset_ops)

    return cache, cache_init, cache_reset

  def _create_transient_vars(self):
    """Creates local cache of factors, weights and gramian for rows and columns.

    Note that currently the caching strategy is as follows:
    When initiating a row(column) update, the column(row) gramian is computed
    and cached while the row gramian is reset; optionally, column(row) factors
    and weights are cached and row(column) factors and weights are reset when
    use_factors_weights_cache is True.
    """

    (self._row_factors_cache,
     row_factors_cache_init,
     row_factors_cache_reset) = self._cached_copy(
         self._row_factors,
         "row_factors_cache",
         pass_through=not self._use_factors_weights_cache)
    (self._col_factors_cache,
     col_factors_cache_init,
     col_factors_cache_reset) = self._cached_copy(
         self._col_factors,
         "col_factors_cache",
         pass_through=not self._use_factors_weights_cache)
    (self._row_wt_cache,
     row_wt_cache_init,
     _) = self._cached_copy(self._row_weights,
                            "row_wt_cache",
                            pass_through=not self._use_factors_weights_cache)

    (self._col_wt_cache,
     col_wt_cache_init,
     _) = self._cached_copy(self._col_weights,
                            "col_wt_cache",
                            pass_through=not self._use_factors_weights_cache)

    (self._row_gramian_cache,
     row_gramian_cache_init,
     row_gramian_cache_reset) = self._cached_copy(self._row_gramian,
                                                  "row_gramian_cache",
                                                  pass_through=False)
    (self._col_gramian_cache,
     col_gramian_cache_init,
     col_gramian_cache_reset) = self._cached_copy(self._col_gramian,
                                                  "col_gramian_cache",
                                                  pass_through=False)

    self._row_updates_init = tf.group(col_factors_cache_init,
                                      row_factors_cache_reset,
                                      col_gramian_cache_init,
                                      row_gramian_cache_reset)
    self._col_updates_init = tf.group(row_factors_cache_init,
                                      col_factors_cache_reset,
                                      row_gramian_cache_init,
                                      col_gramian_cache_reset)

    if self._row_wt_cache is not None:
      assert self._col_wt_cache is not None
      self._worker_init = tf.group(row_wt_cache_init,
                                   col_wt_cache_init,
                                   name="worker_init")
    else:
      self._worker_init = tf.no_op(name="worker_init")

  @property
  def worker_init(self):
    """Op to initialize worker state once before starting any updates."""
    return self._worker_init

  @property
  def row_update_prep_gramian_op(self):
    """Op to form the gramian before starting row updates.

    Must be run before initialize_row_update_op and should only be run by one
    trainer (usually the chief) when doing distributed training.
    """
    return self._row_update_prep_gramian

  @property
  def col_update_prep_gramian_op(self):
    """Op to form the gramian before starting col updates.

    Must be run before initialize_col_update_op and should only be run by one
    trainer (usually the chief) when doing distributed training.
    """
    return self._col_update_prep_gramian


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
        new_ids = tf.where(assignments < extras, ids % (ids_per_shard + 1),
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
      transpose_input: If true, the input will be logically transposed and the
        rows corresponding to the transposed input are updated.

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
      transpose_input: If true, the input will be logically transposed and the
        columns corresponding to the transposed input are updated.

    Returns:
      A tuple consisting of the following two elements:
      new_values: New values for the column factors.
      update_op: An op that assigns the newly computed values to the column
        factors.
    """
    return self._process_input_helper(False, sp_input=sp_input,
                                      transpose_input=transpose_input)

  def project_row_factors(self, sp_input=None, transpose_input=False,
                          projection_weights=None):
    """Projects the row factors.

    This computes the row embedding u_i for an observed row a_i by solving
    one iteration of the update equations.

    Args:
      sp_input: A SparseTensor representing a set of rows. Please note that the
        column indices of this SparseTensor must match the model column feature
        indexing while the row indices are ignored. The returned results will be
        in the same ordering as the input rows.
      transpose_input: If true, the input will be logically transposed and the
        rows corresponding to the transposed input are projected.
      projection_weights: The row weights to be used for the projection. If None
        then 1.0 is used. This can be either a scaler or a rank-1 tensor with
        the number of elements matching the number of rows to be projected.
        Note that the column weights will be determined by the underlying WALS
        model.

    Returns:
      Projected row factors.
    """
    if projection_weights is None:
      projection_weights = 1
    return self._process_input_helper(True, sp_input=sp_input,
                                      transpose_input=transpose_input,
                                      row_weights=projection_weights)[0]

  def project_col_factors(self, sp_input=None, transpose_input=False,
                          projection_weights=None):
    """Projects the column factors.

    This computes the column embedding v_j for an observed column a_j by solving
    one iteration of the update equations.

    Args:
      sp_input: A SparseTensor representing a set of columns. Please note that
        the row indices of this SparseTensor must match the model row feature
        indexing while the column indices are ignored. The returned results will
        be in the same ordering as the input columns.
      transpose_input: If true, the input will be logically transposed and the
        columns corresponding to the transposed input are projected.
      projection_weights: The column weights to be used for the projection. If
        None then 1.0 is used. This can be either a scaler or a rank-1 tensor
        with the number of elements matching the number of columns to be
        projected. Note that the row weights will be determined by the
        underlying WALS model.

    Returns:
      Projected column factors.
    """
    if projection_weights is None:
      projection_weights = 1
    return self._process_input_helper(False, sp_input=sp_input,
                                      transpose_input=transpose_input,
                                      row_weights=projection_weights)[0]

  def _process_input_helper(self, update_row_factors,
                            sp_input=None, transpose_input=False,
                            row_weights=None):
    """Creates the graph for processing a sparse slice of input.

    Args:
      update_row_factors: if True, update or project the row_factors, else
        update or project the column factors.
      sp_input: Please refer to comments for update_row_factors,
        update_col_factors, project_row_factors, and project_col_factors for
        restrictions.
      transpose_input: If True, the input is logically transposed and then the
        corresponding rows/columns of the transposed input are updated.
      row_weights: If not None, this is the row/column weights to be used for
        the update or projection. If None, use the corresponding weights from
        the model. Note that the feature (column/row) weights will be
        determined by the model. When not None, it can either be a scalar or
        a rank-1 tensor with the same number of elements as the number of rows
        of columns to be updated/projected.

    Returns:
      A tuple consisting of the following two elements:
      new_values: New values for the row/column factors.
      update_op: An op that assigns the newly computed values to the row/column
        factors.
    """
    assert isinstance(sp_input, tf.SparseTensor)

    if update_row_factors:
      left = self._row_factors
      right_factors = self._col_factors_cache
      row_wt = self._row_wt_cache
      col_wt = self._col_wt_cache
      sharding_func = WALSModel._get_sharding_func(self._input_rows,
                                                   self._num_row_shards)
      gramian = self._col_gramian_cache
    else:
      left = self._col_factors
      right_factors = self._row_factors_cache
      row_wt = self._col_wt_cache
      col_wt = self._row_wt_cache
      sharding_func = WALSModel._get_sharding_func(self._input_cols,
                                                   self._num_col_shards)
      gramian = self._row_gramian_cache
      transpose_input = not transpose_input

    # Note that the row indices of sp_input are based on the original full input
    # Here we reindex the rows and give them contiguous ids starting at 0.
    # We use tf.unique to achieve this reindexing. Note that this is done so
    # that the downstream kernel can assume that the input is "dense" along the
    # row dimension.
    row_ids, col_ids = tf.split(
        value=sp_input.indices, num_or_size_splits=2, axis=1)
    update_row_indices, all_row_ids = tf.unique(row_ids[:, 0])
    update_col_indices, all_col_ids = tf.unique(col_ids[:, 0])
    col_ids = tf.expand_dims(tf.cast(all_col_ids, tf.int64), 1)
    row_ids = tf.expand_dims(tf.cast(all_row_ids, tf.int64), 1)

    if transpose_input:
      update_indices = update_col_indices
      row_shape = [tf.cast(tf.shape(update_row_indices)[0], tf.int64)]
      gather_indices = update_row_indices
    else:
      update_indices = update_row_indices
      row_shape = [tf.cast(tf.shape(update_col_indices)[0], tf.int64)]
      gather_indices = update_col_indices

    num_rows = tf.cast(tf.shape(update_indices)[0], tf.int64)
    col_shape = [num_rows]
    right = embedding_ops.embedding_lookup(right_factors, gather_indices,
                                           partition_strategy='div')
    new_sp_indices = tf.concat_v2([row_ids, col_ids], 1)
    new_sp_shape = (tf.concat_v2([row_shape, col_shape], 0) if transpose_input
                    else tf.concat_v2([col_shape, row_shape], 0))
    new_sp_input = tf.SparseTensor(indices=new_sp_indices,
                                   values=sp_input.values,
                                   dense_shape=new_sp_shape)

    # Compute lhs and rhs of the normal equations
    total_lhs = (self._unobserved_weight * gramian)
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
      if row_weights is None:
        # TODO(yifanchen): Add special handling for single shard without using
        # embedding_lookup and perform benchmarks for those cases. Same for
        # col_weights lookup below.
        row_weights_slice = embedding_ops.embedding_lookup(
            row_wt, update_indices, partition_strategy='div')
      else:
        with ops.control_dependencies(
            [tf.assert_less_equal(tf.rank(row_weights), 1)]):
          row_weights_slice = tf.cond(tf.equal(tf.rank(row_weights), 0),
                                      lambda: (tf.ones([tf.shape(
                                          update_indices)[0]]) * row_weights),
                                      lambda: tf.cast(row_weights, tf.float32))

      col_weights = embedding_ops.embedding_lookup(
          col_wt, gather_indices, partition_strategy='div')
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
      new_left_values = tf.squeeze(tf.matrix_solve(total_lhs, total_rhs), [2])

    return (new_left_values, self.scatter_update(left,
                                                 update_indices,
                                                 new_left_values,
                                                 sharding_func))

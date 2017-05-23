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
"""Weighted Alternating Least Squares (WALS) on the tf.learn API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.factorization.python.ops import factorization_ops
from tensorflow.contrib.framework.python.ops import variables as framework_variables
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook


class _SweepHook(session_run_hook.SessionRunHook):
  """Keeps track of row/col sweeps, and runs prep ops before each sweep."""

  def __init__(self,
               is_row_sweep_var,
               train_op,
               num_rows,
               num_cols,
               processed_row_indices,
               processed_col_indices,
               row_prep_ops,
               col_prep_ops,
               cache_init_ops):
    """Initializes SweepHook.

    Args:
      is_row_sweep_var: A Boolean tf.Variable, determines whether we are
        currently doing a row or column sweep. It is updated by the hook.
      train_op: An op. All the ops created by the hook will have
        control_dependencies on train_op.
      num_rows: int, the total number of rows to be processed.
      num_cols: int, the total number of columns to be processed.
      processed_row_indices: A Tensor of type int64. The indices of the input
        rows that are processed during the current sweep. All elements of
        processed_row_indices must be in [0, num_rows).
      processed_col_indices: A Tensor of type int64. The indices of the input
        columns that are processed during the current sweep. All elements of
        processed_col_indices must be in [0, num_cols).
      row_prep_ops: list of ops, to be run before the beginning of each row
        sweep, in the given order.
      col_prep_ops: list of ops, to be run before the beginning of each column
        sweep, in the given order.
      cache_init_ops: list of ops, to be run once before training, in the given
        order. These are typically local initialization ops (such as cache
        initialization).
    """
    # TODO(walidk): Provide a counter for the number of completed sweeps.
    self._num_rows = num_rows
    self._num_cols = num_cols
    self._row_prep_ops = row_prep_ops
    self._col_prep_ops = col_prep_ops
    self._cache_init_ops = cache_init_ops
    self._is_row_sweep_var = is_row_sweep_var
    # Boolean variable that determines whether the cache_init_ops have been run.
    self._is_initialized = False
    # Boolean variable that is set to True when a sweep is completed.
    # Used to run the prep_ops at the beginning of a sweep, in before_run().
    self._is_sweep_done = False
    # Ops to run jointly with train_op, responsible for updating
    # _is_row_sweep_var and incrementing the global_step counter. They have
    # control_dependencies on train_op.
    self._fetches = self._create_switch_ops(processed_row_indices,
                                            processed_col_indices,
                                            train_op)

  def _create_switch_ops(self,
                         processed_row_indices,
                         processed_col_indices,
                         train_op):
    """Creates ops to update is_row_sweep_var and to increment global_step.

    Creates two boolean tensors processed_rows and processed_cols, which keep
    track of which rows/cols have been processed during the current sweep.
    Returns ops that should be run after each row / col update.
      - When is_row_sweep_var is True, it sets
        processed_rows[processed_row_indices] to True.
      - When is_row_sweep_var is False, it sets
        processed_cols[processed_col_indices] to True .
    When all rows or all cols have been processed, negates is_row_sweep_var and
    resets processed_rows and processed_cols to False.
    All of the ops created by this function have control_dependencies on
    train_op.

    Args:
      processed_row_indices: A Tensor. The indices of the input rows that are
        processed during the current sweep.
      processed_col_indices: A Tensor. The indices of the input columns that
        are processed during the current sweep.
      train_op: An op. All the ops created by this function have
        control_dependencies on train_op.
    Returns:
      A list consisting of:
        is_sweep_done: A Boolean tensor, determines whether the sweep is done,
          i.e. all rows (during a row sweep) or all columns (during a column
          sweep) have been processed.
        switch_ops: An op that updates is_row_sweep_var when is_sweep_done is
          True. Has control_dependencies on train_op.
        global_step_incr_op: An op that increments the global_step counter. Has
          control_dependenciens on switch_ops.
    """
    processed_rows_init = array_ops.fill(dims=[self._num_rows], value=False)
    with ops.colocate_with(processed_rows_init):
      processed_rows = variable_scope.variable(
          processed_rows_init,
          collections=[ops.GraphKeys.GLOBAL_VARIABLES],
          trainable=False,
          name="sweep_hook_processed_rows")
    processed_cols_init = array_ops.fill(dims=[self._num_cols], value=False)
    with ops.colocate_with(processed_cols_init):
      processed_cols = variable_scope.variable(
          processed_cols_init,
          collections=[ops.GraphKeys.GLOBAL_VARIABLES],
          trainable=False,
          name="sweep_hook_processed_cols")
    # After running the train_op, update processed_rows or processed_cols
    # tensors, depending on whether we are currently doing a row or a col sweep
    with ops.control_dependencies([train_op]):
      def get_row_update_op():
        with ops.colocate_with(processed_rows):
          return state_ops.scatter_update(
              processed_rows, processed_row_indices,
              array_ops.ones_like(processed_row_indices, dtype=dtypes.bool))

      def get_col_update_op():
        with ops.colocate_with(processed_cols):
          return state_ops.scatter_update(
              processed_cols, processed_col_indices,
              array_ops.ones_like(processed_col_indices, dtype=dtypes.bool))

      update_processed_op = control_flow_ops.cond(
          self._is_row_sweep_var, get_row_update_op, get_col_update_op)

      # After update_processed_op, check whether we have completed a sweep.
      # If this is the case, flip the is_row_sweep_var and reset processed_rows
      # and processed_cols tensors.
      with ops.control_dependencies([update_processed_op]):
        def get_switch_op():
          return state_ops.assign(
              self._is_row_sweep_var,
              gen_math_ops.logical_not(self._is_row_sweep_var)).op

        def get_reset_op():
          return control_flow_ops.group(
              state_ops.assign(processed_rows, processed_rows_init).op,
              state_ops.assign(processed_cols, processed_cols_init).op)

        is_sweep_done = control_flow_ops.cond(
            self._is_row_sweep_var,
            lambda: math_ops.reduce_all(processed_rows),
            lambda: math_ops.reduce_all(processed_cols),
            name="sweep_hook_is_sweep_done")
        switch_op = control_flow_ops.cond(
            is_sweep_done, get_switch_op, control_flow_ops.no_op,
            name="sweep_hook_switch_op")
        reset_op = control_flow_ops.cond(
            is_sweep_done, get_reset_op, control_flow_ops.no_op,
            name="sweep_hook_reset_op")
        switch_ops = control_flow_ops.group(switch_op, reset_op,
                                            name="sweep_hook_switch_ops")

        # Op to increment the global step
        global_step = framework_variables.get_global_step()
        with ops.control_dependencies([switch_ops]):
          if global_step is not None:
            global_step_incr_op = state_ops.assign_add(
                global_step, 1, name="global_step_incr").op
          else:
            global_step_incr_op = control_flow_ops.no_op(
                name="global_step_incr")

    return [is_sweep_done, switch_ops, global_step_incr_op]

  def begin(self):
    pass

  def before_run(self, run_context):
    """Runs the appropriate prep ops, and requests running update ops."""
    # Run the appropriate cache_init and prep ops
    sess = run_context.session
    if not self._is_initialized:
      logging.info("SweepHook running cache init ops.")
      for init_op in self._cache_init_ops:
        sess.run(init_op)

    if self._is_sweep_done or not self._is_initialized:
      logging.info("SweepHook running sweep prep ops.")
      row_sweep = sess.run(self._is_row_sweep_var)
      prep_ops = self._row_prep_ops if row_sweep else self._col_prep_ops
      for prep_op in prep_ops:
        sess.run(prep_op)

    self._is_initialized = True

    # Request running the switch_ops and the global_step_incr_op
    logging.info("Partial fit starting.")
    return session_run_hook.SessionRunArgs(fetches=self._fetches)

  def after_run(self, run_context, run_values):
    self._is_sweep_done = run_values.results[0]
    logging.info("Partial fit done.")


def _wals_factorization_model_function(features, labels, mode, params):
  """Model function for the WALSFactorization estimator.

  Args:
    features: Dictionary of features. See WALSMatrixFactorization.
    labels: Must be None.
    mode: A model_fn.ModeKeys object.
    params: Dictionary of parameters containing arguments passed to the
      WALSMatrixFactorization constructor.

  Returns:
    A ModelFnOps object.
  """
  assert labels is None
  use_factors_weights_cache = (
      params["use_factors_weights_cache_for_training"]
      and mode == model_fn.ModeKeys.TRAIN)
  use_gramian_cache = (
      params["use_gramian_cache_for_training"]
      and mode == model_fn.ModeKeys.TRAIN)
  model = factorization_ops.WALSModel(
      params["num_rows"],
      params["num_cols"],
      params["embedding_dimension"],
      unobserved_weight=params["unobserved_weight"],
      regularization=params["regularization_coeff"],
      row_init=params["row_init"],
      col_init=params["col_init"],
      num_row_shards=params["num_row_shards"],
      num_col_shards=params["num_col_shards"],
      row_weights=params["row_weights"],
      col_weights=params["col_weights"],
      use_factors_weights_cache=use_factors_weights_cache,
      use_gramian_cache=use_gramian_cache)

  # Get input rows and cols. We either update rows or columns depending on
  # the value of row_sweep, which is maintained using a session hook
  input_rows = features[WALSMatrixFactorization.INPUT_ROWS]
  input_cols = features[WALSMatrixFactorization.INPUT_COLS]
  input_row_indices, _ = array_ops.unique(input_rows.indices[:, 0])
  input_col_indices, _ = array_ops.unique(input_cols.indices[:, 0])

  # Train ops, controlled using the SweepHook
  # We need to run the following ops:
  # Before a row sweep:
  #   row_update_prep_gramian_op
  #   initialize_row_update_op
  # During a row sweep:
  #   update_row_factors_op
  # Before a col sweep:
  #   col_update_prep_gramian_op
  #   initialize_col_update_op
  # During a col sweep:
  #   update_col_factors_op

  is_row_sweep_var = variable_scope.variable(
      True, "is_row_sweep",
      collections=[ops.GraphKeys.GLOBAL_VARIABLES])
  # The row sweep is determined by is_row_sweep_var (controlled by the
  # sweep_hook) in TRAIN mode, and manually in EVAL mode.
  is_row_sweep = (features[WALSMatrixFactorization.PROJECT_ROW]
                  if mode == model_fn.ModeKeys.EVAL else is_row_sweep_var)

  def update_row_factors():
    return model.update_row_factors(sp_input=input_rows, transpose_input=False)
  def update_col_factors():
    return model.update_col_factors(sp_input=input_cols, transpose_input=True)
  _, train_op, loss = control_flow_ops.cond(
      is_row_sweep, update_row_factors, update_col_factors)

  row_prep_ops = [model.row_update_prep_gramian_op,
                  model.initialize_row_update_op]
  col_prep_ops = [model.col_update_prep_gramian_op,
                  model.initialize_col_update_op]
  cache_init_ops = [model.worker_init]

  sweep_hook = _SweepHook(
      is_row_sweep_var,
      train_op,
      params["num_rows"],
      params["num_cols"],
      input_row_indices,
      input_col_indices,
      row_prep_ops,
      col_prep_ops,
      cache_init_ops,
  )

  # Prediction ops (only return predictions in INFER mode)
  predictions = {}
  if mode == model_fn.ModeKeys.INFER:
    project_row = features[WALSMatrixFactorization.PROJECT_ROW]
    projection_weights = features.get(
        WALSMatrixFactorization.PROJECTION_WEIGHTS)
    def get_row_projection():
      return model.project_row_factors(
          sp_input=input_rows,
          projection_weights=projection_weights,
          transpose_input=False)
    def get_col_projection():
      return model.project_col_factors(
          sp_input=input_cols,
          projection_weights=projection_weights,
          transpose_input=True)

    predictions[WALSMatrixFactorization.PROJECTION_RESULT] = (
        control_flow_ops.cond(
            project_row, get_row_projection, get_col_projection))

  return model_fn.ModelFnOps(
      mode=mode,
      predictions=predictions,
      loss=loss,
      eval_metric_ops={},
      train_op=train_op,
      training_hooks=[sweep_hook])


class WALSMatrixFactorization(estimator.Estimator):
  """An Estimator for Weighted Matrix Factorization, using the WALS method.

  WALS (Weighted Alternating Least Squares) is an algorithm for weighted matrix
  factorization. It computes a low-rank approximation of a given sparse (n x m)
  matrix A, by a product of two matrices, U * V^T, where U is a (n x k) matrix
  and V is a (m x k) matrix. Here k is the rank of the approximation, also
  called the embedding dimension. We refer to U as the row factors, and V as the
  column factors.
  See tensorflow/contrib/factorization/g3doc/wals.md for the precise problem
  formulation.

  The training proceeds in sweeps: during a row_sweep, we fix V and solve for U.
  During a column sweep, we fix U and solve for V. Each one of these problems is
  an unconstrained quadratic minimization problem and can be solved exactly (it
  can also be solved in mini-batches, since the solution decouples nicely).
  The alternating between sweeps is achieved by using a hook during training,
  which is responsible for keeping track of the sweeps and running preparation
  ops at the beginning of each sweep. It also updates the global_step variable,
  which keeps track of the number of batches processed since the beginning of
  training.
  The current implementation assumes that the training is run on a single
  machine, and will fail if config.num_worker_replicas is not equal to one.
  Training is done by calling self.fit(input_fn=input_fn), where input_fn
  provides two tensors: one for rows of the input matrix, and one for rows of
  the transposed input matrix (i.e. columns of the original matrix). Note that
  during a row sweep, only row batches are processed (ignoring column batches)
  and vice-versa.
  Also note that every row (respectively every column) of the input matrix
  must be processed at least once for the sweep to be considered complete. In
  particular, training will not make progress if input_fn does not generate some
  rows.

  For prediction, given a new set of input rows A' (e.g. new rows of the A
  matrix), we compute a corresponding set of row factors U', such that U' * V^T
  is a good approximation of A'. We call this operation a row projection. A
  similar operation is defined for columns.
  Projection is done by calling self.get_projections(input_fn=input_fn), where
  input_fn satisfies the constraints given below.

  The input functions must satisfy the following constraints: Calling input_fn
  must return a tuple (features, labels) where labels is None, and features is
  a dict containing the following keys:
  TRAIN:
    - WALSMatrixFactorization.INPUT_ROWS: float32 SparseTensor (matrix).
      Rows of the input matrix to process (or to project).
    - WALSMatrixFactorization.INPUT_COLS: float32 SparseTensor (matrix).
      Columns of the input matrix to process (or to project), transposed.
  INFER:
    - WALSMatrixFactorization.INPUT_ROWS: float32 SparseTensor (matrix).
      Rows to project.
    - WALSMatrixFactorization.INPUT_COLS: float32 SparseTensor (matrix).
      Columns to project.
    - WALSMatrixFactorization.PROJECT_ROW: Boolean Tensor. Whether to project
      the rows or columns.
    - WALSMatrixFactorization.PROJECTION_WEIGHTS (Optional): float32 Tensor
      (vector). The weights to use in the projection.
  EVAL:
    - WALSMatrixFactorization.INPUT_ROWS: float32 SparseTensor (matrix).
      Rows to project.
    - WALSMatrixFactorization.INPUT_COLS: float32 SparseTensor (matrix).
      Columns to project.
    - WALSMatrixFactorization.PROJECT_ROW: Boolean Tensor. Whether to project
      the rows or columns.
  """
  # Keys to be used in model_fn
  # Features keys
  INPUT_ROWS = "input_rows"
  INPUT_COLS = "input_cols"
  PROJECT_ROW = "project_row"
  PROJECTION_WEIGHTS = "projection_weights"
  # Predictions key
  PROJECTION_RESULT = "projection"

  def __init__(self,
               num_rows,
               num_cols,
               embedding_dimension,
               unobserved_weight=0.1,
               regularization_coeff=None,
               row_init="random",
               col_init="random",
               num_row_shards=1,
               num_col_shards=1,
               row_weights=1,
               col_weights=1,
               use_factors_weights_cache_for_training=True,
               use_gramian_cache_for_training=True,
               model_dir=None,
               config=None):
    """Creates a model for matrix factorization using the WALS method.

    Args:
      num_rows: Total number of rows for input matrix.
      num_cols: Total number of cols for input matrix.
      embedding_dimension: Dimension to use for the factors.
      unobserved_weight: Weight of the unobserved entries of matrix.
      regularization_coeff: Weight of the L2 regularization term. Defaults to
        None, in which case the problem is not regularized.
      row_init: Initializer for row factor. Must be either:
        - A tensor: The row factor matrix is initialized to this tensor,
        - A numpy constant,
        - "random": The rows are initialized using a normal distribution.
      col_init: Initializer for column factor. See row_init.
      num_row_shards: Number of shards to use for the row factors.
      num_col_shards: Number of shards to use for the column factors.
      row_weights: Must be in one of the following three formats:
        - None: In this case, the weight of every entry is the unobserved_weight
          and the problem simplifies to ALS. Note that, in this case,
          col_weights must also be set to "None".
        - List of lists of non-negative scalars, of the form
          [[w_0, w_1, ...], [w_k, ... ], [...]],
          where the number of inner lists equal to the number of row factor
          shards and the elements in each inner list are the weights for the
          rows of that shard. In this case,
          w_ij = unonbserved_weight + row_weights[i] * col_weights[j].
        - A non-negative scalar: This value is used for all row weights.
          Note that it is allowed to have row_weights as a list and col_weights
          as a scalar, or vice-versa.
      col_weights: See row_weights.
      use_factors_weights_cache_for_training: Boolean, whether the factors and
        weights will be cached on the workers before the updates start, during
        training. Defaults to True.
        Note that caching is disabled during prediction.
      use_gramian_cache_for_training: Boolean, whether the Gramians will be
        cached on the workers before the updates start, during training.
        Defaults to True. Note that caching is disabled during prediction.
      model_dir: The directory to save the model results and log files.
      config: A Configuration object. See Estimator.

    Raises:
      ValueError: If config.num_worker_replicas is strictly greater than one.
        The current implementation only supports running on a single worker.
    """
    # TODO(walidk): Support power-law based weight computation.
    # TODO(walidk): Add factor lookup by indices, with caching.
    # TODO(walidk): Support caching during prediction.
    # TODO(walidk): Provide input pipelines that handle missing rows.

    params = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "embedding_dimension": embedding_dimension,
        "unobserved_weight": unobserved_weight,
        "regularization_coeff": regularization_coeff,
        "row_init": row_init,
        "col_init": col_init,
        "num_row_shards": num_row_shards,
        "num_col_shards": num_col_shards,
        "row_weights": row_weights,
        "col_weights": col_weights,
        "use_factors_weights_cache_for_training":
            use_factors_weights_cache_for_training,
        "use_gramian_cache_for_training": use_gramian_cache_for_training
    }
    self._row_factors_names = ["row_factors_shard_%d" % i
                               for i in range(num_row_shards)]
    self._col_factors_names = ["col_factors_shard_%d" % i
                               for i in range(num_col_shards)]

    super(WALSMatrixFactorization, self).__init__(
        model_fn=_wals_factorization_model_function,
        params=params,
        model_dir=model_dir,
        config=config)

    if self._config is not None and self._config.num_worker_replicas > 1:
      raise ValueError("WALSMatrixFactorization must be run on a single worker "
                       "replica.")

  def get_row_factors(self):
    """Returns the row factors of the model, loading them from checkpoint.

    Should only be run after training.

    Returns:
      A list of the row factors of the model.
    """
    return [self.get_variable_value(name) for name in self._row_factors_names]

  def get_col_factors(self):
    """Returns the column factors of the model, loading them from checkpoint.

    Should only be run after training.

    Returns:
      A list of the column factors of the model.
    """
    return [self.get_variable_value(name) for name in self._col_factors_names]

  def get_projections(self, input_fn):
    """Computes the projections of the rows or columns given in input_fn.

    Runs predict() with the given input_fn, and returns the results. Should only
    be run after training.

    Args:
      input_fn: Input function which specifies the rows or columns to project.
    Returns:
      A generator of the projected factors.
    """
    return (result[WALSMatrixFactorization.PROJECTION_RESULT]
            for result in self.predict(input_fn=input_fn))

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
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util


class _SweepHook(session_run_hook.SessionRunHook):
  """Keeps track of row/col sweeps, and runs prep ops before each sweep."""

  def __init__(self, is_row_sweep_var, is_sweep_done_var, init_op,
               row_prep_ops, col_prep_ops, row_train_op, col_train_op,
               switch_op):
    """Initializes SweepHook.

    Args:
      is_row_sweep_var: A Boolean tf.Variable, determines whether we are
        currently doing a row or column sweep. It is updated by the hook.
      is_sweep_done_var: A Boolean tf.Variable, determines whether we are
        starting a new sweep (this is used to determine when to run the prep ops
        below).
      init_op: op to be run once before training. This is typically a local
        initialization op (such as cache initialization).
      row_prep_ops: A list of TensorFlow ops, to be run before the beginning of
        each row sweep (and during initialization), in the given order.
      col_prep_ops: A list of TensorFlow ops, to be run before the beginning of
        each column sweep (and during initialization), in the given order.
      row_train_op: A TensorFlow op to be run during row sweeps.
      col_train_op: A TensorFlow op to be run during column sweeps.
      switch_op: A TensorFlow op to be run before each sweep.
    """
    self._is_row_sweep_var = is_row_sweep_var
    self._is_sweep_done_var = is_sweep_done_var
    self._init_op = init_op
    self._row_prep_ops = row_prep_ops
    self._col_prep_ops = col_prep_ops
    self._row_train_op = row_train_op
    self._col_train_op = col_train_op
    self._switch_op = switch_op
    # Boolean variable that determines whether the init_op has been run.
    self._is_initialized = False

  def before_run(self, run_context):
    """Runs the appropriate prep ops, and requests running update ops."""
    sess = run_context.session
    is_sweep_done = sess.run(self._is_sweep_done_var)
    if not self._is_initialized:
      logging.info("SweepHook running init op.")
      sess.run(self._init_op)
    if is_sweep_done:
      logging.info("SweepHook starting the next sweep.")
      sess.run(self._switch_op)
    is_row_sweep = sess.run(self._is_row_sweep_var)
    if is_sweep_done or not self._is_initialized:
      logging.info("SweepHook running prep ops for the {} sweep.".format(
          "row" if is_row_sweep else "col"))
      prep_ops = self._row_prep_ops if is_row_sweep else self._col_prep_ops
      for prep_op in prep_ops:
        sess.run(prep_op)
    self._is_initialized = True
    logging.info("Next fit step starting.")
    return session_run_hook.SessionRunArgs(
        fetches=[self._row_train_op if is_row_sweep else self._col_train_op])


class _IncrementGlobalStepHook(session_run_hook.SessionRunHook):
  """Hook that increments the global step."""

  def __init__(self):
    global_step = training_util.get_global_step()
    if global_step:
      self._global_step_incr_op = state_ops.assign_add(
          global_step, 1, name="global_step_incr").op
    else:
      self._global_step_incr_op = None

  def before_run(self, run_context):
    if self._global_step_incr_op:
      run_context.session.run(self._global_step_incr_op)


class _StopAtSweepHook(session_run_hook.SessionRunHook):
  """Hook that requests stop at a given sweep."""

  def __init__(self, last_sweep):
    """Initializes a `StopAtSweepHook`.

    This hook requests stop at a given sweep. Relies on the tensor named
    COMPLETED_SWEEPS in the default graph.

    Args:
      last_sweep: Integer, number of the last sweep to run.
    """
    self._last_sweep = last_sweep

  def begin(self):
    try:
      self._completed_sweeps_var = ops.get_default_graph().get_tensor_by_name(
          WALSMatrixFactorization.COMPLETED_SWEEPS + ":0")
    except KeyError:
      raise RuntimeError(WALSMatrixFactorization.COMPLETED_SWEEPS +
                         " counter should be created to use StopAtSweepHook.")

  def before_run(self, run_context):
    return session_run_hook.SessionRunArgs(self._completed_sweeps_var)

  def after_run(self, run_context, run_values):
    completed_sweeps = run_values.results
    if completed_sweeps >= self._last_sweep:
      run_context.request_stop()


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

  Raises:
    ValueError: If `mode` is not recognized.
  """
  assert labels is None
  use_factors_weights_cache = (params["use_factors_weights_cache_for_training"]
                               and mode == model_fn.ModeKeys.TRAIN)
  use_gramian_cache = (params["use_gramian_cache_for_training"] and
                       mode == model_fn.ModeKeys.TRAIN)
  max_sweeps = params["max_sweeps"]
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
  # the value of row_sweep, which is maintained using a session hook.
  input_rows = features[WALSMatrixFactorization.INPUT_ROWS]
  input_cols = features[WALSMatrixFactorization.INPUT_COLS]

  # TRAIN mode:
  if mode == model_fn.ModeKeys.TRAIN:
    # Training consists of the following ops (controlled using a SweepHook).
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
        True,
        trainable=False,
        name="is_row_sweep",
        collections=[ops.GraphKeys.GLOBAL_VARIABLES])
    is_sweep_done_var = variable_scope.variable(
        False,
        trainable=False,
        name="is_sweep_done",
        collections=[ops.GraphKeys.GLOBAL_VARIABLES])
    completed_sweeps_var = variable_scope.variable(
        0,
        trainable=False,
        name=WALSMatrixFactorization.COMPLETED_SWEEPS,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES])
    loss_var = variable_scope.variable(
        0.,
        trainable=False,
        name=WALSMatrixFactorization.LOSS,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES])
    # The root weighted squared error =
    #   \\(\sqrt( \sum_{i,j} w_ij * (a_ij - r_ij)^2 / \sum_{i,j} w_ij )\\)
    rwse_var = variable_scope.variable(
        0.,
        trainable=False,
        name=WALSMatrixFactorization.RWSE,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES])

    summary.scalar("loss", loss_var)
    summary.scalar("root_weighted_squared_error", rwse_var)
    summary.scalar("completed_sweeps", completed_sweeps_var)

    def create_axis_ops(sp_input, num_items, update_fn, axis_name):
      """Creates book-keeping and training ops for a given axis.

      Args:
        sp_input: A SparseTensor corresponding to the row or column batch.
        num_items: An integer, the total number of items of this axis.
        update_fn: A function that takes one argument (`sp_input`), and that
        returns a tuple of
          * new_factors: A float Tensor of the factor values after update.
          * update_op: a TensorFlow op which updates the factors.
          * loss: A float Tensor, the unregularized loss.
          * reg_loss: A float Tensor, the regularization loss.
          * sum_weights: A float Tensor, the sum of factor weights.
        axis_name: A string that specifies the name of the axis.

      Returns:
        A tuple consisting of:
          * reset_processed_items_op: A TensorFlow op, to be run before the
            beginning of any sweep. It marks all items as not-processed.
          * axis_train_op: A Tensorflow op, to be run during this axis' sweeps.
      """
      processed_items_init = array_ops.fill(dims=[num_items], value=False)
      with ops.colocate_with(processed_items_init):
        processed_items = variable_scope.variable(
            processed_items_init,
            collections=[ops.GraphKeys.GLOBAL_VARIABLES],
            trainable=False,
            name="processed_" + axis_name)
      _, update_op, loss, reg, sum_weights = update_fn(sp_input)
      input_indices = sp_input.indices[:, 0]
      with ops.control_dependencies([
          update_op,
          state_ops.assign(loss_var, loss + reg),
          state_ops.assign(rwse_var, math_ops.sqrt(loss / sum_weights))]):
        with ops.colocate_with(processed_items):
          update_processed_items = state_ops.scatter_update(
              processed_items,
              input_indices,
              array_ops.ones_like(input_indices, dtype=dtypes.bool),
              name="update_processed_{}_indices".format(axis_name))
        with ops.control_dependencies([update_processed_items]):
          is_sweep_done = math_ops.reduce_all(processed_items)
          axis_train_op = control_flow_ops.group(
              state_ops.assign(is_sweep_done_var, is_sweep_done),
              state_ops.assign_add(
                  completed_sweeps_var,
                  math_ops.cast(is_sweep_done, dtypes.int32)),
              name="{}_sweep_train_op".format(axis_name))
      return processed_items.initializer, axis_train_op

    reset_processed_rows_op, row_train_op = create_axis_ops(
        input_rows,
        params["num_rows"],
        lambda x: model.update_row_factors(sp_input=x, transpose_input=False),
        "rows")
    reset_processed_cols_op, col_train_op = create_axis_ops(
        input_cols,
        params["num_cols"],
        lambda x: model.update_col_factors(sp_input=x, transpose_input=True),
        "cols")
    switch_op = control_flow_ops.group(
        state_ops.assign(
            is_row_sweep_var, math_ops.logical_not(is_row_sweep_var)),
        reset_processed_rows_op,
        reset_processed_cols_op,
        name="sweep_switch_op")
    row_prep_ops = [
        model.row_update_prep_gramian_op, model.initialize_row_update_op]
    col_prep_ops = [
        model.col_update_prep_gramian_op, model.initialize_col_update_op]
    init_op = model.worker_init
    sweep_hook = _SweepHook(
        is_row_sweep_var, is_sweep_done_var, init_op,
        row_prep_ops, col_prep_ops, row_train_op, col_train_op, switch_op)
    global_step_hook = _IncrementGlobalStepHook()
    training_hooks = [sweep_hook, global_step_hook]
    if max_sweeps is not None:
      training_hooks.append(_StopAtSweepHook(max_sweeps))

    return model_fn.ModelFnOps(
        mode=model_fn.ModeKeys.TRAIN,
        predictions={},
        loss=loss_var,
        eval_metric_ops={},
        train_op=control_flow_ops.no_op(),
        training_hooks=training_hooks)

  # INFER mode
  elif mode == model_fn.ModeKeys.INFER:
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

    predictions = {
        WALSMatrixFactorization.PROJECTION_RESULT: control_flow_ops.cond(
            features[WALSMatrixFactorization.PROJECT_ROW],
            get_row_projection,
            get_col_projection)
    }

    return model_fn.ModelFnOps(
        mode=model_fn.ModeKeys.INFER,
        predictions=predictions,
        loss=None,
        eval_metric_ops={},
        train_op=control_flow_ops.no_op(),
        training_hooks=[])

  # EVAL mode
  elif mode == model_fn.ModeKeys.EVAL:
    def get_row_loss():
      _, _, loss, reg, _ = model.update_row_factors(
          sp_input=input_rows, transpose_input=False)
      return loss + reg
    def get_col_loss():
      _, _, loss, reg, _ = model.update_col_factors(
          sp_input=input_cols, transpose_input=True)
      return loss + reg
    loss = control_flow_ops.cond(
        features[WALSMatrixFactorization.PROJECT_ROW],
        get_row_loss,
        get_col_loss)
    return model_fn.ModelFnOps(
        mode=model_fn.ModeKeys.EVAL,
        predictions={},
        loss=loss,
        eval_metric_ops={},
        train_op=control_flow_ops.no_op(),
        training_hooks=[])

  else:
    raise ValueError("mode=%s is not recognized." % str(mode))


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
  # Name of the completed_sweeps variable
  COMPLETED_SWEEPS = "completed_sweeps"
  # Name of the loss variable
  LOSS = "WALS_loss"
  # Name of the Root Weighted Squared Error variable
  RWSE = "WALS_RWSE"

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
               max_sweeps=None,
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
          \\([[w_0, w_1, ...], [w_k, ... ], [...]]\\),
          where the number of inner lists equal to the number of row factor
          shards and the elements in each inner list are the weights for the
          rows of that shard. In this case,
          \\(w_ij = unonbserved_weight + row_weights[i] * col_weights[j]\\).
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
      max_sweeps: integer, optional. Specifies the number of sweeps for which
        to train the model, where a sweep is defined as a full update of all the
        row factors (resp. column factors).
        If `steps` or `max_steps` is also specified in model.fit(), training
        stops when either of the steps condition or sweeps condition is met.
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
        "num_rows":
            num_rows,
        "num_cols":
            num_cols,
        "embedding_dimension":
            embedding_dimension,
        "unobserved_weight":
            unobserved_weight,
        "regularization_coeff":
            regularization_coeff,
        "row_init":
            row_init,
        "col_init":
            col_init,
        "num_row_shards":
            num_row_shards,
        "num_col_shards":
            num_col_shards,
        "row_weights":
            row_weights,
        "col_weights":
            col_weights,
        "max_sweeps":
            max_sweeps,
        "use_factors_weights_cache_for_training":
            use_factors_weights_cache_for_training,
        "use_gramian_cache_for_training":
            use_gramian_cache_for_training
    }
    self._row_factors_names = [
        "row_factors_shard_%d" % i for i in range(num_row_shards)
    ]
    self._col_factors_names = [
        "col_factors_shard_%d" % i for i in range(num_col_shards)
    ]

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

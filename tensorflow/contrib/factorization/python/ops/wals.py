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

from tensorflow.contrib.framework.python.ops import variables as framework_variables
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
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
      processed_rows = variables.Variable(
          processed_rows_init,
          collections=[ops.GraphKeys.GLOBAL_VARIABLES],
          trainable=False,
          name="sweep_hook_processed_rows")
    processed_cols_init = array_ops.fill(dims=[self._num_cols], value=False)
    with ops.colocate_with(processed_cols_init):
      processed_cols = variables.Variable(
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


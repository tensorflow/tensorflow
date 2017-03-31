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
"""Tests for WALSMatrixFactorization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.factorization.python.ops import wals as wals_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import session_run_hook


class SweepHookTest(test.TestCase):

  def setUp(self):
    self._num_rows = 5
    self._num_cols = 7
    self._train_op = control_flow_ops.no_op()
    self._row_prep_done = variables.Variable(False)
    self._col_prep_done = variables.Variable(False)
    self._init_done = variables.Variable(False)
    self._row_prep_ops = [state_ops.assign(self._row_prep_done, True)]
    self._col_prep_ops = [state_ops.assign(self._col_prep_done, True)]
    self._init_ops = [state_ops.assign(self._init_done, True)]
    self._input_row_indices_ph = array_ops.placeholder(dtypes.int64)
    self._input_col_indices_ph = array_ops.placeholder(dtypes.int64)

  def run_hook_with_indices(self, sweep_hook, row_indices, col_indices):
    with self.test_session() as sess:
      # Before run
      run_context = session_run_hook.SessionRunContext(
          original_args=None, session=sess)
      sess_run_args = sweep_hook.before_run(run_context)
      feed_dict = {
          self._input_row_indices_ph: row_indices,
          self._input_col_indices_ph: col_indices
      }
      # Run
      run_results = sess.run(sess_run_args.fetches, feed_dict=feed_dict)
      run_values = session_run_hook.SessionRunValues(
          results=run_results, options=None, run_metadata=None)
      # After run
      sweep_hook.after_run(run_context, run_values)

  def test_row_sweep(self):
    with self.test_session() as sess:
      is_row_sweep_var = variables.Variable(True)
      sweep_hook = wals_lib._SweepHook(
          is_row_sweep_var,
          self._train_op,
          self._num_rows,
          self._num_cols,
          self._input_row_indices_ph,
          self._input_col_indices_ph,
          self._row_prep_ops,
          self._col_prep_ops,
          self._init_ops)

      # Initialize variables
      sess.run([variables.global_variables_initializer()])
      # Row sweep
      self.run_hook_with_indices(sweep_hook, [], [])
      self.assertTrue(sess.run(self._init_done),
                      msg='init ops not run by the sweep_hook')
      self.assertTrue(sess.run(self._row_prep_done),
                      msg='row_prep not run by the sweep_hook')
      self.run_hook_with_indices(sweep_hook, [0, 1, 2], [])
      self.assertTrue(sess.run(is_row_sweep_var),
                      msg='Row sweep is not complete but is_row_sweep is '
                      'False.')
      self.run_hook_with_indices(sweep_hook, [3, 4], [0, 1, 2, 3, 4, 5, 6])
      self.assertFalse(sess.run(is_row_sweep_var),
                       msg='Row sweep is complete but is_row_sweep is True.')
      self.assertTrue(sweep_hook._is_sweep_done,
                      msg='Sweep is complete but is_sweep_done is False.')

  def test_col_sweep(self):
    with self.test_session() as sess:
      is_row_sweep_var = variables.Variable(False)
      sweep_hook = wals_lib._SweepHook(
          is_row_sweep_var,
          self._train_op,
          self._num_rows,
          self._num_cols,
          self._input_row_indices_ph,
          self._input_col_indices_ph,
          self._row_prep_ops,
          self._col_prep_ops,
          self._init_ops)

      # Initialize variables
      sess.run([variables.global_variables_initializer()])
      # Col sweep
      self.run_hook_with_indices(sweep_hook, [], [])
      self.assertTrue(sess.run(self._col_prep_done),
                      msg='col_prep not run by the sweep_hook')
      self.run_hook_with_indices(sweep_hook, [], [0, 1, 2, 3, 4])
      self.assertFalse(sess.run(is_row_sweep_var),
                       msg='Col sweep is not complete but is_row_sweep is '
                       'True.')
      self.assertFalse(sweep_hook._is_sweep_done,
                       msg='Sweep is not complete but is_sweep_done is True.')
      self.run_hook_with_indices(sweep_hook, [], [4, 5, 6])
      self.assertTrue(sess.run(is_row_sweep_var),
                      msg='Col sweep is complete but is_row_sweep is False')
      self.assertTrue(sweep_hook._is_sweep_done,
                      msg='Sweep is complete but is_sweep_done is False.')


if __name__ == '__main__':
  test.main()

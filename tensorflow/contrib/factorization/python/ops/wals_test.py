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

import itertools
import json
import numpy as np

from tensorflow.contrib.factorization.python.ops import factorization_ops_test_utils
from tensorflow.contrib.factorization.python.ops import wals as wals_lib
from tensorflow.contrib.learn.python.learn import run_config
from tensorflow.contrib.learn.python.learn.estimators import run_config as run_config_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import session_run_hook


class WALSMatrixFactorizationTest(test.TestCase):
  INPUT_MATRIX = factorization_ops_test_utils.INPUT_MATRIX

  def np_array_to_sparse(self, np_array):
    """Transforms an np.array to a tf.SparseTensor."""
    return factorization_ops_test_utils.np_matrix_to_tf_sparse(np_array)

  def calculate_loss(self):
    """Calculates the loss of the current (trained) model."""
    current_rows = embedding_ops.embedding_lookup(
        self._model.get_row_factors(), math_ops.range(self._num_rows),
        partition_strategy='div')
    current_cols = embedding_ops.embedding_lookup(
        self._model.get_col_factors(), math_ops.range(self._num_cols),
        partition_strategy='div')
    row_wts = embedding_ops.embedding_lookup(
        self._row_weights, math_ops.range(self._num_rows),
        partition_strategy='div')
    col_wts = embedding_ops.embedding_lookup(
        self._col_weights, math_ops.range(self._num_cols),
        partition_strategy='div')
    sp_inputs = self.np_array_to_sparse(self.INPUT_MATRIX)
    return factorization_ops_test_utils.calculate_loss(
        sp_inputs, current_rows, current_cols, self._regularization_coeff,
        self._unobserved_weight, row_wts, col_wts)

  # TODO(walidk): Replace with input_reader_utils functions once open sourced.
  def remap_sparse_tensor_rows(self, sp_x, row_ids, shape):
    """Remaps the row ids of a tf.SparseTensor."""
    old_row_ids, old_col_ids = array_ops.split(
        value=sp_x.indices, num_or_size_splits=2, axis=1)
    new_row_ids = array_ops.gather(row_ids, old_row_ids)
    new_indices = array_ops.concat([new_row_ids, old_col_ids], 1)
    return sparse_tensor.SparseTensor(
        indices=new_indices, values=sp_x.values, dense_shape=shape)

  # TODO(walidk): Add an option to randomize inputs.
  def input_fn(self, np_matrix, batch_size, project_row=None,
               projection_weights=None, col_ids=None):
    """Returns an input_fn that selects row and col batches from np_matrix."""
    def extract_features(row_batch, col_batch, shape):
      row_ids = row_batch[0]
      col_ids = col_batch[0]
      rows = self.remap_sparse_tensor_rows(row_batch[1], row_ids, shape)
      cols = self.remap_sparse_tensor_rows(col_batch[1], col_ids, shape)
      features = {
          wals_lib.WALSMatrixFactorization.INPUT_ROWS: rows,
          wals_lib.WALSMatrixFactorization.INPUT_COLS: cols,
      }
      return features

    def _fn():
      num_rows = np.shape(np_matrix)[0]
      num_cols = np.shape(np_matrix)[1]
      row_ids = math_ops.range(num_rows, dtype=dtypes.int64)
      col_ids = math_ops.range(num_cols, dtype=dtypes.int64)
      sp_mat = self.np_array_to_sparse(np_matrix)
      sp_mat_t = sparse_ops.sparse_transpose(sp_mat)
      row_batch = input_lib.batch(
          [row_ids, sp_mat],
          batch_size=min(batch_size, num_rows),
          capacity=10,
          enqueue_many=True)
      col_batch = input_lib.batch(
          [col_ids, sp_mat_t],
          batch_size=min(batch_size, num_cols),
          capacity=10,
          enqueue_many=True)

      features = extract_features(row_batch, col_batch, sp_mat.dense_shape)
      if projection_weights is not None:
        weights_batch = input_lib.batch(
            projection_weights,
            batch_size=batch_size,
            capacity=10,
            enqueue_many=True)
        features[wals_lib.WALSMatrixFactorization.PROJECTION_WEIGHTS] = (
            weights_batch)
      if project_row is not None:
        features[wals_lib.WALSMatrixFactorization.PROJECT_ROW] = (
            constant_op.constant(project_row))

      labels = None
      return features, labels

    return _fn

  @property
  def row_steps(self):
    return np.ceil(self._num_rows / self.batch_size)

  @property
  def col_steps(self):
    return np.ceil(self._num_cols / self.batch_size)

  @property
  def batch_size(self):
    return 2

  @property
  def use_cache(self):
    return False

  def setUp(self):
    self._num_rows = 5
    self._num_cols = 7
    self._embedding_dimension = 3
    self._unobserved_weight = 0.1
    self._num_row_shards = 2
    self._num_col_shards = 3
    self._regularization_coeff = 0.01
    self._col_init = [
        # Shard 0.
        [[-0.36444709, -0.39077035, -0.32528427],
         [1.19056475, 0.07231052, 2.11834812],
         [0.93468881, -0.71099287, 1.91826844]],
        # Shard 1.
        [[1.18160152, 1.52490723, -0.50015002],
         [1.82574749, -0.57515913, -1.32810032]],
        # Shard 2.
        [[-0.15515432, -0.84675711, 0.13097958],
         [-0.9246484, 0.69117504, 1.2036494]],
    ]
    self._row_weights = [[0.1, 0.2, 0.3], [0.4, 0.5]]
    self._col_weights = [[0.1, 0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]

    # Values of row and column factors after running one iteration or factor
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
    self._model = wals_lib.WALSMatrixFactorization(
        self._num_rows,
        self._num_cols,
        self._embedding_dimension,
        self._unobserved_weight,
        col_init=self._col_init,
        regularization_coeff=self._regularization_coeff,
        num_row_shards=self._num_row_shards,
        num_col_shards=self._num_col_shards,
        row_weights=self._row_weights,
        col_weights=self._col_weights,
        use_factors_weights_cache_for_training=self.use_cache,
        use_gramian_cache_for_training=self.use_cache)

  def test_fit(self):
    # Row sweep.
    input_fn = self.input_fn(np_matrix=self.INPUT_MATRIX,
                             batch_size=self.batch_size)
    self._model.fit(input_fn=input_fn, steps=self.row_steps)
    row_factors = self._model.get_row_factors()
    self.assertAllClose(row_factors[0], self._row_factors_0, atol=1e-3)
    self.assertAllClose(row_factors[1], self._row_factors_1, atol=1e-3)

    # Col sweep.
    # Running fit a second time will resume training from the checkpoint.
    input_fn = self.input_fn(np_matrix=self.INPUT_MATRIX,
                             batch_size=self.batch_size)
    self._model.fit(input_fn=input_fn, steps=self.col_steps)
    col_factors = self._model.get_col_factors()
    self.assertAllClose(col_factors[0], self._col_factors_0, atol=1e-3)
    self.assertAllClose(col_factors[1], self._col_factors_1, atol=1e-3)
    self.assertAllClose(col_factors[2], self._col_factors_2, atol=1e-3)

  def test_predict(self):
    input_fn = self.input_fn(np_matrix=self.INPUT_MATRIX,
                             batch_size=self.batch_size)
    # Project rows 1 and 4 from the input matrix.
    proj_input_fn = self.input_fn(
        np_matrix=self.INPUT_MATRIX[[1, 4], :],
        batch_size=2,
        project_row=True,
        projection_weights=[[0.2, 0.5]])

    self._model.fit(input_fn=input_fn, steps=self.row_steps)
    projections = self._model.get_projections(proj_input_fn)
    projected_rows = list(itertools.islice(projections, 2))

    self.assertAllClose(
        projected_rows,
        [self._row_factors_0[1], self._row_factors_1[1]],
        atol=1e-3)

    # Project columns 5, 3, 1 from the input matrix.
    proj_input_fn = self.input_fn(
        np_matrix=self.INPUT_MATRIX[:, [5, 3, 1]],
        batch_size=3,
        project_row=False,
        projection_weights=[[0.6, 0.4, 0.2]])

    self._model.fit(input_fn=input_fn, steps=self.col_steps)
    projections = self._model.get_projections(proj_input_fn)
    projected_cols = list(itertools.islice(projections, 3))
    self.assertAllClose(
        projected_cols,
        [self._col_factors_2[0], self._col_factors_1[0],
         self._col_factors_0[1]],
        atol=1e-3)

  def test_eval(self):
    # Do a row sweep then evaluate the model on row inputs.
    # The evaluate function returns the loss of the projected rows, but since
    # projection is idempotent, the eval loss must match the model loss.
    input_fn = self.input_fn(np_matrix=self.INPUT_MATRIX,
                             batch_size=self.batch_size)
    self._model.fit(input_fn=input_fn, steps=self.row_steps)
    eval_input_fn_row = self.input_fn(np_matrix=self.INPUT_MATRIX, batch_size=1,
                                      project_row=True)
    loss = self._model.evaluate(
        input_fn=eval_input_fn_row, steps=self._num_rows)['loss']

    with self.test_session():
      true_loss = self.calculate_loss()

    self.assertNear(
        loss, true_loss, err=.001,
        msg="""After row update, eval loss = {}, does not match the true
        loss = {}.""".format(loss, true_loss))

    # Do a col sweep then evaluate the model on col inputs.
    self._model.fit(input_fn=input_fn, steps=self.col_steps)
    eval_input_fn_col = self.input_fn(np_matrix=self.INPUT_MATRIX, batch_size=1,
                                      project_row=False)
    loss = self._model.evaluate(
        input_fn=eval_input_fn_col, steps=self._num_cols)['loss']

    with self.test_session():
      true_loss = self.calculate_loss()

    self.assertNear(
        loss, true_loss, err=.001,
        msg="""After row update, eval loss = {}, does not match the true
        loss = {}.""".format(loss, true_loss))


class WALSMatrixFactorizationTestCached(WALSMatrixFactorizationTest):

  @property
  def use_cache(self):
    return True


class WALSMatrixFactorizationTestFullBatch(WALSMatrixFactorizationTest):

  @property
  def batch_size(self):
    return 100


class WALSMatrixFactorizationUnsupportedTest(test.TestCase):

  def setUp(self):
    pass

  def testDistributedWALSUnsupported(self):
    tf_config = {
        'cluster': {
            run_config_lib.TaskType.PS: ['host1:1', 'host2:2'],
            run_config_lib.TaskType.WORKER: ['host3:3', 'host4:4']
        },
        'task': {
            'type': run_config_lib.TaskType.WORKER,
            'index': 1
        }
    }
    with test.mock.patch.dict('os.environ',
                              {'TF_CONFIG': json.dumps(tf_config)}):
      config = run_config.RunConfig()
    self.assertEqual(config.num_worker_replicas, 2)
    with self.assertRaises(ValueError):
      self._model = wals_lib.WALSMatrixFactorization(1, 1, 1, config=config)


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
      # Before run.
      run_context = session_run_hook.SessionRunContext(
          original_args=None, session=sess)
      sess_run_args = sweep_hook.before_run(run_context)
      feed_dict = {
          self._input_row_indices_ph: row_indices,
          self._input_col_indices_ph: col_indices
      }
      # Run.
      run_results = sess.run(sess_run_args.fetches, feed_dict=feed_dict)
      run_values = session_run_hook.SessionRunValues(
          results=run_results, options=None, run_metadata=None)
      # After run.
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

      # Initialize variables.
      sess.run([variables.global_variables_initializer()])
      # Row sweep.
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

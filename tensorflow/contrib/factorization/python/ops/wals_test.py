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
from tensorflow.contrib.learn.python.learn.estimators import model_fn
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
from tensorflow.python.training import monitored_session


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

  # TODO(walidk): Add an option to shuffle inputs.
  def input_fn(self, np_matrix, batch_size, mode,
               project_row=None, projection_weights=None,
               remove_empty_rows_columns=False):
    """Returns an input_fn that selects row and col batches from np_matrix.

    This simple utility creates an input function from a numpy_array. The
    following transformations are performed:
    * The empty rows and columns in np_matrix are removed (if
      remove_empty_rows_columns is true)
    * np_matrix is converted to a SparseTensor.
    * The rows of the sparse matrix (and the rows of its transpose) are batched.
    * A features dictionary is created, which contains the row / column batches.

    In TRAIN mode, one only needs to specify the np_matrix and the batch_size.
    In INFER and EVAL modes, one must also provide project_row, a boolean which
    specifies whether we are projecting rows or columns.

    Args:
      np_matrix: A numpy array. The input matrix to use.
      batch_size: Integer.
      mode: Can be one of model_fn.ModeKeys.{TRAIN, INFER, EVAL}.
      project_row: A boolean. Used in INFER and EVAL modes. Specifies whether
        to project rows or columns.
      projection_weights: A float numpy array. Used in INFER mode. Specifies
        the weights to use in the projection (the weights are optional, and
        default to 1.).
      remove_empty_rows_columns: A boolean. When true, this will remove empty
        rows and columns in the np_matrix. Note that this will result in
        modifying the indices of the input matrix. The mapping from new indices
        to old indices is returned in the form of two numpy arrays.

    Returns:
      A tuple consisting of:
      _fn: A callable. Calling _fn returns a features dict.
      nz_row_ids: A numpy array of the ids of non-empty rows, such that
        nz_row_ids[i] is the old row index corresponding to new index i.
      nz_col_ids: A numpy array of the ids of non-empty columns, such that
        nz_col_ids[j] is the old column index corresponding to new index j.
    """
    if remove_empty_rows_columns:
      np_matrix, nz_row_ids, nz_col_ids = (
          factorization_ops_test_utils.remove_empty_rows_columns(np_matrix))
    else:
      nz_row_ids = np.arange(np.shape(np_matrix)[0])
      nz_col_ids = np.arange(np.shape(np_matrix)[1])

    def extract_features(row_batch, col_batch, num_rows, num_cols):
      row_ids = row_batch[0]
      col_ids = col_batch[0]
      rows = self.remap_sparse_tensor_rows(
          row_batch[1], row_ids, shape=[num_rows, num_cols])
      cols = self.remap_sparse_tensor_rows(
          col_batch[1], col_ids, shape=[num_cols, num_rows])
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

      features = extract_features(row_batch, col_batch, num_rows, num_cols)

      if mode == model_fn.ModeKeys.INFER or mode == model_fn.ModeKeys.EVAL:
        self.assertTrue(
            project_row is not None,
            msg='project_row must be specified in INFER or EVAL mode.')
        features[wals_lib.WALSMatrixFactorization.PROJECT_ROW] = (
            constant_op.constant(project_row))

      if mode == model_fn.ModeKeys.INFER and projection_weights is not None:
        weights_batch = input_lib.batch(
            projection_weights,
            batch_size=batch_size,
            capacity=10,
            enqueue_many=True)
        features[wals_lib.WALSMatrixFactorization.PROJECTION_WEIGHTS] = (
            weights_batch)

      labels = None
      return features, labels

    return _fn, nz_row_ids, nz_col_ids

  @property
  def input_matrix(self):
    return self.INPUT_MATRIX

  @property
  def row_steps(self):
    return np.ceil(self._num_rows / self.batch_size)

  @property
  def col_steps(self):
    return np.ceil(self._num_cols / self.batch_size)

  @property
  def batch_size(self):
    return 5

  @property
  def use_cache(self):
    return False

  @property
  def max_sweeps(self):
    return None

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
        max_sweeps=self.max_sweeps,
        use_factors_weights_cache_for_training=self.use_cache,
        use_gramian_cache_for_training=self.use_cache)

  def test_fit(self):
    # Row sweep.
    input_fn = self.input_fn(np_matrix=self.input_matrix,
                             batch_size=self.batch_size,
                             mode=model_fn.ModeKeys.TRAIN,
                             remove_empty_rows_columns=True)[0]
    self._model.fit(input_fn=input_fn, steps=self.row_steps)
    row_factors = self._model.get_row_factors()
    self.assertAllClose(row_factors[0], self._row_factors_0, atol=1e-3)
    self.assertAllClose(row_factors[1], self._row_factors_1, atol=1e-3)

    # Col sweep.
    # Running fit a second time will resume training from the checkpoint.
    input_fn = self.input_fn(np_matrix=self.input_matrix,
                             batch_size=self.batch_size,
                             mode=model_fn.ModeKeys.TRAIN,
                             remove_empty_rows_columns=True)[0]
    self._model.fit(input_fn=input_fn, steps=self.col_steps)
    col_factors = self._model.get_col_factors()
    self.assertAllClose(col_factors[0], self._col_factors_0, atol=1e-3)
    self.assertAllClose(col_factors[1], self._col_factors_1, atol=1e-3)
    self.assertAllClose(col_factors[2], self._col_factors_2, atol=1e-3)

  def test_predict(self):
    input_fn = self.input_fn(np_matrix=self.input_matrix,
                             batch_size=self.batch_size,
                             mode=model_fn.ModeKeys.TRAIN,
                             remove_empty_rows_columns=True,
                            )[0]
    # Project rows 1 and 4 from the input matrix.
    proj_input_fn = self.input_fn(
        np_matrix=self.INPUT_MATRIX[[1, 4], :],
        batch_size=2,
        mode=model_fn.ModeKeys.INFER,
        project_row=True,
        projection_weights=[[0.2, 0.5]])[0]

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
        mode=model_fn.ModeKeys.INFER,
        project_row=False,
        projection_weights=[[0.6, 0.4, 0.2]])[0]

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
    input_fn = self.input_fn(np_matrix=self.input_matrix,
                             batch_size=self.batch_size,
                             mode=model_fn.ModeKeys.TRAIN,
                             remove_empty_rows_columns=True,
                            )[0]
    self._model.fit(input_fn=input_fn, steps=self.row_steps)
    eval_input_fn_row = self.input_fn(np_matrix=self.input_matrix,
                                      batch_size=1,
                                      mode=model_fn.ModeKeys.EVAL,
                                      project_row=True,
                                      remove_empty_rows_columns=True)[0]
    loss = self._model.evaluate(
        input_fn=eval_input_fn_row, steps=self._num_rows)['loss']

    with self.cached_session():
      true_loss = self.calculate_loss()

    self.assertNear(
        loss, true_loss, err=.001,
        msg="""After row update, eval loss = {}, does not match the true
        loss = {}.""".format(loss, true_loss))

    # Do a col sweep then evaluate the model on col inputs.
    self._model.fit(input_fn=input_fn, steps=self.col_steps)
    eval_input_fn_col = self.input_fn(np_matrix=self.input_matrix,
                                      batch_size=1,
                                      mode=model_fn.ModeKeys.EVAL,
                                      project_row=False,
                                      remove_empty_rows_columns=True)[0]
    loss = self._model.evaluate(
        input_fn=eval_input_fn_col, steps=self._num_cols)['loss']

    with self.cached_session():
      true_loss = self.calculate_loss()

    self.assertNear(
        loss, true_loss, err=.001,
        msg="""After col update, eval loss = {}, does not match the true
        loss = {}.""".format(loss, true_loss))


class WALSMatrixFactorizationTestSweeps(WALSMatrixFactorizationTest):

  @property
  def max_sweeps(self):
    return 2

  # We set the column steps to None so that we rely only on max_sweeps to stop
  # training.
  @property
  def col_steps(self):
    return None


class WALSMatrixFactorizationTestCached(WALSMatrixFactorizationTest):

  @property
  def use_cache(self):
    return True


class WALSMatrixFactorizaiontTestPaddedInput(WALSMatrixFactorizationTest):
  PADDED_INPUT_MATRIX = np.pad(
      WALSMatrixFactorizationTest.INPUT_MATRIX,
      [(1, 0), (1, 0)], mode='constant')

  @property
  def input_matrix(self):
    return self.PADDED_INPUT_MATRIX


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

  def test_sweeps(self):
    is_row_sweep_var = variables.VariableV1(True)
    is_sweep_done_var = variables.VariableV1(False)
    init_done = variables.VariableV1(False)
    row_prep_done = variables.VariableV1(False)
    col_prep_done = variables.VariableV1(False)
    row_train_done = variables.VariableV1(False)
    col_train_done = variables.VariableV1(False)

    init_op = state_ops.assign(init_done, True)
    row_prep_op = state_ops.assign(row_prep_done, True)
    col_prep_op = state_ops.assign(col_prep_done, True)
    row_train_op = state_ops.assign(row_train_done, True)
    col_train_op = state_ops.assign(col_train_done, True)
    train_op = control_flow_ops.no_op()
    switch_op = control_flow_ops.group(
        state_ops.assign(is_sweep_done_var, False),
        state_ops.assign(is_row_sweep_var,
                         math_ops.logical_not(is_row_sweep_var)))
    mark_sweep_done = state_ops.assign(is_sweep_done_var, True)

    with self.cached_session() as sess:
      sweep_hook = wals_lib._SweepHook(
          is_row_sweep_var,
          is_sweep_done_var,
          init_op,
          [row_prep_op],
          [col_prep_op],
          row_train_op,
          col_train_op,
          switch_op)
      mon_sess = monitored_session._HookedSession(sess, [sweep_hook])
      sess.run([variables.global_variables_initializer()])

      # Row sweep.
      mon_sess.run(train_op)
      self.assertTrue(sess.run(init_done),
                      msg='init op not run by the Sweephook')
      self.assertTrue(sess.run(row_prep_done),
                      msg='row_prep_op not run by the SweepHook')
      self.assertTrue(sess.run(row_train_done),
                      msg='row_train_op not run by the SweepHook')
      self.assertTrue(
          sess.run(is_row_sweep_var),
          msg='Row sweep is not complete but is_row_sweep_var is False.')
      # Col sweep.
      mon_sess.run(mark_sweep_done)
      mon_sess.run(train_op)
      self.assertTrue(sess.run(col_prep_done),
                      msg='col_prep_op not run by the SweepHook')
      self.assertTrue(sess.run(col_train_done),
                      msg='col_train_op not run by the SweepHook')
      self.assertFalse(
          sess.run(is_row_sweep_var),
          msg='Col sweep is not complete but is_row_sweep_var is True.')
      # Row sweep.
      mon_sess.run(mark_sweep_done)
      mon_sess.run(train_op)
      self.assertTrue(
          sess.run(is_row_sweep_var),
          msg='Col sweep is complete but is_row_sweep_var is False.')


class StopAtSweepHookTest(test.TestCase):

  def test_stop(self):
    hook = wals_lib._StopAtSweepHook(last_sweep=10)
    completed_sweeps = variables.VariableV1(
        8, name=wals_lib.WALSMatrixFactorization.COMPLETED_SWEEPS)
    train_op = state_ops.assign_add(completed_sweeps, 1)
    hook.begin()

    with self.cached_session() as sess:
      sess.run([variables.global_variables_initializer()])
      mon_sess = monitored_session._HookedSession(sess, [hook])
      mon_sess.run(train_op)
      # completed_sweeps is 9 after running train_op.
      self.assertFalse(mon_sess.should_stop())
      mon_sess.run(train_op)
      # completed_sweeps is 10 after running train_op.
      self.assertTrue(mon_sess.should_stop())


if __name__ == '__main__':
  test.main()

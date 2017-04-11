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
"""Tests for learn.estimators.dynamic_rnn_estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import numpy as np

from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.layers.python.layers import target_column as target_column_lib
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import dynamic_rnn_estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.learn.python.learn.estimators import rnn_common
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class IdentityRNNCell(rnn.RNNCell):

  def __init__(self, state_size, output_size):
    self._state_size = state_size
    self._output_size = output_size

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, inputs, state):
    return array_ops.identity(inputs), array_ops.ones(
        [array_ops.shape(inputs)[0], self.state_size])


class MockTargetColumn(object):

  def __init__(self, num_label_columns=None):
    self._num_label_columns = num_label_columns

  def get_eval_ops(self, features, activations, labels, metrics):
    raise NotImplementedError(
        'MockTargetColumn.get_eval_ops called unexpectedly.')

  def logits_to_predictions(self, flattened_activations, proba=False):
    raise NotImplementedError(
        'MockTargetColumn.logits_to_predictions called unexpectedly.')

  def loss(self, activations, labels, features):
    raise NotImplementedError('MockTargetColumn.loss called unexpectedly.')

  @property
  def num_label_columns(self):
    if self._num_label_columns is None:
      raise ValueError('MockTargetColumn.num_label_columns has not been set.')
    return self._num_label_columns

  def set_num_label_columns(self, n):
    self._num_label_columns = n


def sequence_length_mask(values, lengths):
  masked = values
  for i, length in enumerate(lengths):
    masked[i, length:, :] = np.zeros_like(masked[i, length:, :])
  return masked


class DynamicRnnEstimatorTest(test.TestCase):

  NUM_RNN_CELL_UNITS = 8
  NUM_LABEL_COLUMNS = 6
  INPUTS_COLUMN = feature_column.real_valued_column(
      'inputs', dimension=NUM_LABEL_COLUMNS)

  def setUp(self):
    super(DynamicRnnEstimatorTest, self).setUp()
    self.rnn_cell = core_rnn_cell_impl.BasicRNNCell(self.NUM_RNN_CELL_UNITS)
    self.mock_target_column = MockTargetColumn(
        num_label_columns=self.NUM_LABEL_COLUMNS)

    location = feature_column.sparse_column_with_keys(
        'location', keys=['west_side', 'east_side', 'nyc'])
    location_onehot = feature_column.one_hot_column(location)
    self.context_feature_columns = [location_onehot]

    wire_cast = feature_column.sparse_column_with_keys(
        'wire_cast', ['marlo', 'omar', 'stringer'])
    wire_cast_embedded = feature_column.embedding_column(wire_cast, dimension=8)
    measurements = feature_column.real_valued_column(
        'measurements', dimension=2)
    self.sequence_feature_columns = [measurements, wire_cast_embedded]

  def GetColumnsToTensors(self):
    """Get columns_to_tensors matching setUp(), in the current default graph."""
    return {
        'location':
            sparse_tensor.SparseTensor(
                indices=[[0, 0], [1, 0], [2, 0]],
                values=['west_side', 'west_side', 'nyc'],
                dense_shape=[3, 1]),
        'wire_cast':
            sparse_tensor.SparseTensor(
                indices=[[0, 0, 0], [0, 1, 0],
                         [1, 0, 0], [1, 1, 0], [1, 1, 1],
                         [2, 0, 0]],
                values=[b'marlo', b'stringer',
                        b'omar', b'stringer', b'marlo',
                        b'marlo'],
                dense_shape=[3, 2, 2]),
        'measurements':
            random_ops.random_uniform(
                [3, 2, 2], seed=4711)
    }

  def GetClassificationTargetsOrNone(self, mode):
    """Get targets matching setUp() and mode, in the current default graph."""
    return (random_ops.random_uniform(
        [3, 2, 1], 0, 2, dtype=dtypes.int64, seed=1412) if
            mode != model_fn_lib.ModeKeys.INFER else None)

  def testBuildSequenceInputInput(self):
    sequence_input = dynamic_rnn_estimator.build_sequence_input(
        self.GetColumnsToTensors(), self.sequence_feature_columns,
        self.context_feature_columns)
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(data_flow_ops.tables_initializer())
      sequence_input_val = sess.run(sequence_input)
    expected_shape = np.array([
        3,  # expected batch size
        2,  # padded sequence length
        3 + 8 + 2  # location keys + embedding dim + measurement dimension
    ])
    self.assertAllEqual(expected_shape, sequence_input_val.shape)

  def testConstructRNN(self):
    initial_state = None
    sequence_input = dynamic_rnn_estimator.build_sequence_input(
        self.GetColumnsToTensors(), self.sequence_feature_columns,
        self.context_feature_columns)
    activations_t, final_state_t = dynamic_rnn_estimator.construct_rnn(
        initial_state, sequence_input, self.rnn_cell,
        self.mock_target_column.num_label_columns)

    # Obtain values of activations and final state.
    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(data_flow_ops.tables_initializer())
      activations, final_state = sess.run([activations_t, final_state_t])

    expected_activations_shape = np.array([3, 2, self.NUM_LABEL_COLUMNS])
    self.assertAllEqual(expected_activations_shape, activations.shape)
    expected_state_shape = np.array([3, self.NUM_RNN_CELL_UNITS])
    self.assertAllEqual(expected_state_shape, final_state.shape)

  def testGetOutputAlternatives(self):
    test_cases = (
        (rnn_common.PredictionType.SINGLE_VALUE,
         constants.ProblemType.CLASSIFICATION,
         {prediction_key.PredictionKey.CLASSES: True,
          prediction_key.PredictionKey.PROBABILITIES: True,
          dynamic_rnn_estimator._get_state_name(0): True},
         {'dynamic_rnn_output':
          (constants.ProblemType.CLASSIFICATION,
           {prediction_key.PredictionKey.CLASSES: True,
            prediction_key.PredictionKey.PROBABILITIES: True})}),

        (rnn_common.PredictionType.SINGLE_VALUE,
         constants.ProblemType.LINEAR_REGRESSION,
         {prediction_key.PredictionKey.SCORES: True,
          dynamic_rnn_estimator._get_state_name(0): True,
          dynamic_rnn_estimator._get_state_name(1): True},
         {'dynamic_rnn_output':
          (constants.ProblemType.LINEAR_REGRESSION,
           {prediction_key.PredictionKey.SCORES: True})}),

        (rnn_common.PredictionType.MULTIPLE_VALUE,
         constants.ProblemType.CLASSIFICATION,
         {prediction_key.PredictionKey.CLASSES: True,
          prediction_key.PredictionKey.PROBABILITIES: True,
          dynamic_rnn_estimator._get_state_name(0): True},
         None))

    for pred_type, prob_type, pred_dict, expected_alternatives in test_cases:
      actual_alternatives = dynamic_rnn_estimator._get_output_alternatives(
          pred_type, prob_type, pred_dict)
      self.assertEqual(expected_alternatives, actual_alternatives)

  # testGetDynamicRnnModelFn{Train,Eval,Infer}() test which fields
  # of ModelFnOps are set depending on mode.
  def testGetDynamicRnnModelFnTrain(self):
    model_fn_ops = self._GetModelFnOpsForMode(model_fn_lib.ModeKeys.TRAIN)
    self.assertIsNotNone(model_fn_ops.predictions)
    self.assertIsNotNone(model_fn_ops.loss)
    self.assertIsNotNone(model_fn_ops.train_op)
    # None may get normalized to {}; we accept neither.
    self.assertNotEqual(len(model_fn_ops.eval_metric_ops), 0)

  def testGetDynamicRnnModelFnEval(self):
    model_fn_ops = self._GetModelFnOpsForMode(model_fn_lib.ModeKeys.EVAL)
    self.assertIsNotNone(model_fn_ops.predictions)
    self.assertIsNotNone(model_fn_ops.loss)
    self.assertIsNone(model_fn_ops.train_op)
    # None may get normalized to {}; we accept neither.
    self.assertNotEqual(len(model_fn_ops.eval_metric_ops), 0)

  def testGetDynamicRnnModelFnInfer(self):
    model_fn_ops = self._GetModelFnOpsForMode(model_fn_lib.ModeKeys.INFER)
    self.assertIsNotNone(model_fn_ops.predictions)
    self.assertIsNone(model_fn_ops.loss)
    self.assertIsNone(model_fn_ops.train_op)
    # None may get normalized to {}; we accept both.
    self.assertFalse(model_fn_ops.eval_metric_ops)

  def _GetModelFnOpsForMode(self, mode):
    """Helper for testGetDynamicRnnModelFn{Train,Eval,Infer}()."""
    model_fn = dynamic_rnn_estimator._get_dynamic_rnn_model_fn(
        cell_type='basic_rnn',
        num_units=[10],
        target_column=target_column_lib.multi_class_target(n_classes=2),
        # Only CLASSIFICATION yields eval metrics to test for.
        problem_type=constants.ProblemType.CLASSIFICATION,
        prediction_type=rnn_common.PredictionType.MULTIPLE_VALUE,
        optimizer='SGD',
        sequence_feature_columns=self.sequence_feature_columns,
        context_feature_columns=self.context_feature_columns,
        learning_rate=0.1)
    labels = self.GetClassificationTargetsOrNone(mode)
    model_fn_ops = model_fn(
        features=self.GetColumnsToTensors(), labels=labels, mode=mode)
    return model_fn_ops

  def testExport(self):
    input_feature_key = 'magic_input_feature_key'

    def get_input_fn(mode):

      def input_fn():
        features = self.GetColumnsToTensors()
        if mode == model_fn_lib.ModeKeys.INFER:
          input_examples = array_ops.placeholder(dtypes.string)
          features[input_feature_key] = input_examples
          # Real code would now parse features out of input_examples,
          # but this test can just stick to the constants above.
        return features, self.GetClassificationTargetsOrNone(mode)

      return input_fn

    model_dir = tempfile.mkdtemp()

    def estimator_fn():
      return dynamic_rnn_estimator.DynamicRnnEstimator(
          problem_type=constants.ProblemType.CLASSIFICATION,
          prediction_type=rnn_common.PredictionType.MULTIPLE_VALUE,
          num_classes=2,
          num_units=self.NUM_RNN_CELL_UNITS,
          sequence_feature_columns=self.sequence_feature_columns,
          context_feature_columns=self.context_feature_columns,
          predict_probabilities=True,
          model_dir=model_dir)

    # Train a bit to create an exportable checkpoint.
    estimator_fn().fit(input_fn=get_input_fn(model_fn_lib.ModeKeys.TRAIN),
                       steps=100)
    # Now export, but from a fresh estimator instance, like you would
    # in an export binary. That means .export() has to work without
    # .fit() being called on the same object.
    export_dir = tempfile.mkdtemp()
    print('Exporting to', export_dir)
    estimator_fn().export(
        export_dir,
        input_fn=get_input_fn(model_fn_lib.ModeKeys.INFER),
        use_deprecated_input_fn=False,
        input_feature_key=input_feature_key)

  def testStateTupleDictConversion(self):
    """Test `state_tuple_to_dict` and `dict_to_state_tuple`."""
    cell_sizes = [5, 3, 7]
    # A MultiRNNCell of LSTMCells is both a common choice and an interesting
    # test case, because it has two levels of nesting, with an inner class that
    # is not a plain tuple.
    cell = core_rnn_cell_impl.MultiRNNCell(
        [core_rnn_cell_impl.LSTMCell(i) for i in cell_sizes])
    state_dict = {
        dynamic_rnn_estimator._get_state_name(i):
        array_ops.expand_dims(math_ops.range(cell_size), 0)
        for i, cell_size in enumerate([5, 5, 3, 3, 7, 7])
    }
    expected_state = (core_rnn_cell_impl.LSTMStateTuple(
        np.reshape(np.arange(5), [1, -1]), np.reshape(np.arange(5), [1, -1])),
                      core_rnn_cell_impl.LSTMStateTuple(
                          np.reshape(np.arange(3), [1, -1]),
                          np.reshape(np.arange(3), [1, -1])),
                      core_rnn_cell_impl.LSTMStateTuple(
                          np.reshape(np.arange(7), [1, -1]),
                          np.reshape(np.arange(7), [1, -1])))
    actual_state = dynamic_rnn_estimator.dict_to_state_tuple(state_dict, cell)
    flattened_state = dynamic_rnn_estimator.state_tuple_to_dict(actual_state)

    with self.test_session() as sess:
      (state_dict_val, actual_state_val, flattened_state_val) = sess.run(
          [state_dict, actual_state, flattened_state])

    def _recursive_assert_equal(x, y):
      self.assertEqual(type(x), type(y))
      if isinstance(x, (list, tuple)):
        self.assertEqual(len(x), len(y))
        for i, _ in enumerate(x):
          _recursive_assert_equal(x[i], y[i])
      elif isinstance(x, np.ndarray):
        np.testing.assert_array_equal(x, y)
      else:
        self.fail('Unexpected type: {}'.format(type(x)))

    for k in state_dict_val.keys():
      np.testing.assert_array_almost_equal(
          state_dict_val[k],
          flattened_state_val[k],
          err_msg='Wrong value for state component {}.'.format(k))
    _recursive_assert_equal(expected_state, actual_state_val)

  def testMultiRNNState(self):
    """Test that state flattening/reconstruction works for `MultiRNNCell`."""
    batch_size = 11
    sequence_length = 16
    train_steps = 5
    cell_sizes = [4, 8, 7]
    learning_rate = 0.1

    def get_shift_input_fn(batch_size, sequence_length, seed=None):

      def input_fn():
        random_sequence = random_ops.random_uniform(
            [batch_size, sequence_length + 1],
            0,
            2,
            dtype=dtypes.int32,
            seed=seed)
        labels = array_ops.slice(random_sequence, [0, 0],
                                 [batch_size, sequence_length])
        inputs = array_ops.expand_dims(
            math_ops.to_float(
                array_ops.slice(random_sequence, [0, 1],
                                [batch_size, sequence_length])), 2)
        input_dict = {
            dynamic_rnn_estimator._get_state_name(i): random_ops.random_uniform(
                [batch_size, cell_size], seed=((i + 1) * seed))
            for i, cell_size in enumerate([4, 4, 8, 8, 7, 7])
        }
        input_dict['inputs'] = inputs
        return input_dict, labels

      return input_fn

    seq_columns = [feature_column.real_valued_column('inputs', dimension=1)]
    config = run_config.RunConfig(tf_random_seed=21212)
    cell_type = 'lstm'
    sequence_estimator = dynamic_rnn_estimator.DynamicRnnEstimator(
        problem_type=constants.ProblemType.CLASSIFICATION,
        prediction_type=rnn_common.PredictionType.MULTIPLE_VALUE,
        num_classes=2,
        num_units=cell_sizes,
        sequence_feature_columns=seq_columns,
        cell_type=cell_type,
        learning_rate=learning_rate,
        config=config,
        predict_probabilities=True)

    train_input_fn = get_shift_input_fn(batch_size, sequence_length, seed=12321)
    eval_input_fn = get_shift_input_fn(batch_size, sequence_length, seed=32123)

    sequence_estimator.fit(input_fn=train_input_fn, steps=train_steps)

    prediction_dict = sequence_estimator.predict(
        input_fn=eval_input_fn, as_iterable=False)
    for i, state_size in enumerate([4, 4, 8, 8, 7, 7]):
      state_piece = prediction_dict[dynamic_rnn_estimator._get_state_name(i)]
      self.assertListEqual(list(state_piece.shape), [batch_size, state_size])

  def testLegacyConstructor(self):
    """Exercise legacy constructor function."""
    num_units = 16
    num_layers = 6
    output_keep_prob = 0.9
    input_keep_prob = 0.7
    batch_size = 11
    learning_rate = 0.1
    train_sequence_length = 21
    train_steps = 121

    def get_input_fn(batch_size, sequence_length, state_dict, starting_step=0):

      def input_fn():
        sequence = constant_op.constant(
            [[(starting_step + i + j) % 2 for j in range(sequence_length + 1)]
             for i in range(batch_size)],
            dtype=dtypes.int32)
        labels = array_ops.slice(sequence, [0, 0],
                                 [batch_size, sequence_length])
        inputs = array_ops.expand_dims(
            math_ops.to_float(
                array_ops.slice(sequence, [0, 1], [batch_size, sequence_length
                                                  ])), 2)
        input_dict = state_dict
        input_dict['inputs'] = inputs
        return input_dict, labels

      return input_fn

    seq_columns = [feature_column.real_valued_column('inputs', dimension=1)]
    config = run_config.RunConfig(tf_random_seed=21212)

    model_dir = tempfile.mkdtemp()
    sequence_estimator = dynamic_rnn_estimator.multi_value_rnn_classifier(
        num_classes=2,
        num_units=num_units,
        num_rnn_layers=num_layers,
        input_keep_probability=input_keep_prob,
        output_keep_probability=output_keep_prob,
        sequence_feature_columns=seq_columns,
        learning_rate=learning_rate,
        config=config,
        model_dir=model_dir)

    train_input_fn = get_input_fn(
        batch_size, train_sequence_length, state_dict={})

    sequence_estimator.fit(input_fn=train_input_fn, steps=train_steps)

  def testMultipleRuns(self):
    """Tests resuming training by feeding state."""
    cell_sizes = [4, 7]
    batch_size = 11
    learning_rate = 0.1
    train_sequence_length = 21
    train_steps = 121
    dropout_keep_probabilities = [0.5, 0.5, 0.5]
    prediction_steps = [3, 2, 5, 11, 6]

    def get_input_fn(batch_size, sequence_length, state_dict, starting_step=0):

      def input_fn():
        sequence = constant_op.constant(
            [[(starting_step + i + j) % 2 for j in range(sequence_length + 1)]
             for i in range(batch_size)],
            dtype=dtypes.int32)
        labels = array_ops.slice(sequence, [0, 0],
                                 [batch_size, sequence_length])
        inputs = array_ops.expand_dims(
            math_ops.to_float(
                array_ops.slice(sequence, [0, 1], [batch_size, sequence_length
                                                  ])), 2)
        input_dict = state_dict
        input_dict['inputs'] = inputs
        return input_dict, labels

      return input_fn

    seq_columns = [feature_column.real_valued_column('inputs', dimension=1)]
    config = run_config.RunConfig(tf_random_seed=21212)

    model_dir = tempfile.mkdtemp()
    sequence_estimator = dynamic_rnn_estimator.DynamicRnnEstimator(
        problem_type=constants.ProblemType.CLASSIFICATION,
        prediction_type=rnn_common.PredictionType.MULTIPLE_VALUE,
        num_classes=2,
        sequence_feature_columns=seq_columns,
        num_units=cell_sizes,
        cell_type='lstm',
        dropout_keep_probabilities=dropout_keep_probabilities,
        learning_rate=learning_rate,
        config=config,
        model_dir=model_dir)

    train_input_fn = get_input_fn(
        batch_size, train_sequence_length, state_dict={})

    sequence_estimator.fit(input_fn=train_input_fn, steps=train_steps)

    def incremental_predict(estimator, increments):
      """Run `estimator.predict` for `i` steps for `i` in `increments`."""
      step = 0
      incremental_state_dict = {}
      for increment in increments:
        input_fn = get_input_fn(
            batch_size,
            increment,
            state_dict=incremental_state_dict,
            starting_step=step)
        prediction_dict = estimator.predict(
            input_fn=input_fn, as_iterable=False)
        step += increment
        incremental_state_dict = {
            k: v
            for (k, v) in prediction_dict.items()
            if k.startswith(rnn_common.RNNKeys.STATE_PREFIX)
        }
      return prediction_dict

    pred_all_at_once = incremental_predict(sequence_estimator,
                                           [sum(prediction_steps)])
    pred_step_by_step = incremental_predict(sequence_estimator,
                                            prediction_steps)

    # Check that the last `prediction_steps[-1]` steps give the same
    # predictions.
    np.testing.assert_array_equal(
        pred_all_at_once[prediction_key.PredictionKey.CLASSES]
        [:, -1 * prediction_steps[-1]:],
        pred_step_by_step[prediction_key.PredictionKey.CLASSES],
        err_msg='Mismatch on last {} predictions.'.format(prediction_steps[-1]))
    # Check that final states are identical.
    for k, v in pred_all_at_once.items():
      if k.startswith(rnn_common.RNNKeys.STATE_PREFIX):
        np.testing.assert_array_equal(
            v, pred_step_by_step[k], err_msg='Mismatch on state {}.'.format(k))


# TODO(jamieas): move all tests below to a benchmark test.
class DynamicRNNEstimatorLearningTest(test.TestCase):
  """Learning tests for dynamic RNN Estimators."""

  def testLearnSineFunction(self):
    """Tests learning a sine function."""
    batch_size = 8
    sequence_length = 64
    train_steps = 200
    eval_steps = 20
    cell_size = [4]
    learning_rate = 0.1
    loss_threshold = 0.02

    def get_sin_input_fn(batch_size, sequence_length, increment, seed=None):

      def _sin_fn(x):
        ranger = math_ops.linspace(
            array_ops.reshape(x[0], []), (sequence_length - 1) * increment,
            sequence_length + 1)
        return math_ops.sin(ranger)

      def input_fn():
        starts = random_ops.random_uniform(
            [batch_size], maxval=(2 * np.pi), seed=seed)
        sin_curves = functional_ops.map_fn(
            _sin_fn, (starts,), dtype=dtypes.float32)
        inputs = array_ops.expand_dims(
            array_ops.slice(sin_curves, [0, 0], [batch_size, sequence_length]),
            2)
        labels = array_ops.slice(sin_curves, [0, 1],
                                 [batch_size, sequence_length])
        return {'inputs': inputs}, labels

      return input_fn

    seq_columns = [
        feature_column.real_valued_column(
            'inputs', dimension=cell_size[0])
    ]
    config = run_config.RunConfig(tf_random_seed=1234)
    sequence_estimator = dynamic_rnn_estimator.DynamicRnnEstimator(
        problem_type=constants.ProblemType.LINEAR_REGRESSION,
        prediction_type=rnn_common.PredictionType.MULTIPLE_VALUE,
        num_units=cell_size,
        sequence_feature_columns=seq_columns,
        learning_rate=learning_rate,
        dropout_keep_probabilities=[0.9, 0.9],
        config=config)

    train_input_fn = get_sin_input_fn(
        batch_size, sequence_length, np.pi / 32, seed=1234)
    eval_input_fn = get_sin_input_fn(
        batch_size, sequence_length, np.pi / 32, seed=4321)

    sequence_estimator.fit(input_fn=train_input_fn, steps=train_steps)
    loss = sequence_estimator.evaluate(
        input_fn=eval_input_fn, steps=eval_steps)['loss']
    self.assertLess(loss, loss_threshold,
                    'Loss should be less than {}; got {}'.format(loss_threshold,
                                                                 loss))

  def testLearnShiftByOne(self):
    """Tests that learning a 'shift-by-one' example.

    Each label sequence consists of the input sequence 'shifted' by one place.
    The RNN must learn to 'remember' the previous input.
    """
    batch_size = 16
    sequence_length = 32
    train_steps = 200
    eval_steps = 20
    cell_size = 4
    learning_rate = 0.3
    accuracy_threshold = 0.9

    def get_shift_input_fn(batch_size, sequence_length, seed=None):

      def input_fn():
        random_sequence = random_ops.random_uniform(
            [batch_size, sequence_length + 1],
            0,
            2,
            dtype=dtypes.int32,
            seed=seed)
        labels = array_ops.slice(random_sequence, [0, 0],
                                 [batch_size, sequence_length])
        inputs = array_ops.expand_dims(
            math_ops.to_float(
                array_ops.slice(random_sequence, [0, 1],
                                [batch_size, sequence_length])), 2)
        return {'inputs': inputs}, labels

      return input_fn

    seq_columns = [
        feature_column.real_valued_column(
            'inputs', dimension=cell_size)
    ]
    config = run_config.RunConfig(tf_random_seed=21212)
    sequence_estimator = dynamic_rnn_estimator.DynamicRnnEstimator(
        problem_type=constants.ProblemType.CLASSIFICATION,
        prediction_type=rnn_common.PredictionType.MULTIPLE_VALUE,
        num_classes=2,
        num_units=cell_size,
        sequence_feature_columns=seq_columns,
        learning_rate=learning_rate,
        config=config,
        predict_probabilities=True)

    train_input_fn = get_shift_input_fn(batch_size, sequence_length, seed=12321)
    eval_input_fn = get_shift_input_fn(batch_size, sequence_length, seed=32123)

    sequence_estimator.fit(input_fn=train_input_fn, steps=train_steps)

    evaluation = sequence_estimator.evaluate(
        input_fn=eval_input_fn, steps=eval_steps)
    accuracy = evaluation['accuracy']
    self.assertGreater(accuracy, accuracy_threshold,
                       'Accuracy should be higher than {}; got {}'.format(
                           accuracy_threshold, accuracy))

    # Testing `predict` when `predict_probabilities=True`.
    prediction_dict = sequence_estimator.predict(
        input_fn=eval_input_fn, as_iterable=False)
    self.assertListEqual(
        sorted(list(prediction_dict.keys())),
        sorted([
            prediction_key.PredictionKey.CLASSES,
            prediction_key.PredictionKey.PROBABILITIES,
            dynamic_rnn_estimator._get_state_name(0)
        ]))
    predictions = prediction_dict[prediction_key.PredictionKey.CLASSES]
    probabilities = prediction_dict[
        prediction_key.PredictionKey.PROBABILITIES]
    self.assertListEqual(list(predictions.shape), [batch_size, sequence_length])
    self.assertListEqual(
        list(probabilities.shape), [batch_size, sequence_length, 2])

  def testLearnMean(self):
    """Test learning to calculate a mean."""
    batch_size = 16
    sequence_length = 3
    train_steps = 200
    eval_steps = 20
    cell_type = 'basic_rnn'
    cell_size = 8
    optimizer_type = 'Momentum'
    learning_rate = 0.1
    momentum = 0.9
    loss_threshold = 0.1

    def get_mean_input_fn(batch_size, sequence_length, seed=None):

      def input_fn():
        # Create examples by choosing 'centers' and adding uniform noise.
        centers = math_ops.matmul(
            random_ops.random_uniform(
                [batch_size, 1], -0.75, 0.75, dtype=dtypes.float32, seed=seed),
            array_ops.ones([1, sequence_length]))
        noise = random_ops.random_uniform(
            [batch_size, sequence_length],
            -0.25,
            0.25,
            dtype=dtypes.float32,
            seed=seed)
        sequences = centers + noise

        inputs = array_ops.expand_dims(sequences, 2)
        labels = math_ops.reduce_mean(sequences, reduction_indices=[1])
        return {'inputs': inputs}, labels

      return input_fn

    seq_columns = [
        feature_column.real_valued_column(
            'inputs', dimension=cell_size)
    ]
    config = run_config.RunConfig(tf_random_seed=6)
    sequence_estimator = dynamic_rnn_estimator.DynamicRnnEstimator(
        problem_type=constants.ProblemType.LINEAR_REGRESSION,
        prediction_type=rnn_common.PredictionType.SINGLE_VALUE,
        num_units=cell_size,
        sequence_feature_columns=seq_columns,
        cell_type=cell_type,
        optimizer=optimizer_type,
        learning_rate=learning_rate,
        momentum=momentum,
        config=config)

    train_input_fn = get_mean_input_fn(batch_size, sequence_length, 121)
    eval_input_fn = get_mean_input_fn(batch_size, sequence_length, 212)

    sequence_estimator.fit(input_fn=train_input_fn, steps=train_steps)
    evaluation = sequence_estimator.evaluate(
        input_fn=eval_input_fn, steps=eval_steps)
    loss = evaluation['loss']
    self.assertLess(loss, loss_threshold,
                    'Loss should be less than {}; got {}'.format(loss_threshold,
                                                                 loss))

  def testLearnMajority(self):
    """Test learning the 'majority' function."""
    batch_size = 16
    sequence_length = 7
    train_steps = 200
    eval_steps = 20
    cell_type = 'lstm'
    cell_size = 4
    optimizer_type = 'Momentum'
    learning_rate = 2.0
    momentum = 0.9
    accuracy_threshold = 0.9

    def get_majority_input_fn(batch_size, sequence_length, seed=None):
      random_seed.set_random_seed(seed)

      def input_fn():
        random_sequence = random_ops.random_uniform(
            [batch_size, sequence_length], 0, 2, dtype=dtypes.int32, seed=seed)
        inputs = array_ops.expand_dims(math_ops.to_float(random_sequence), 2)
        labels = math_ops.to_int32(
            array_ops.squeeze(
                math_ops.reduce_sum(
                    inputs, reduction_indices=[1]) > (sequence_length / 2.0)))
        return {'inputs': inputs}, labels

      return input_fn

    seq_columns = [
        feature_column.real_valued_column(
            'inputs', dimension=cell_size)
    ]
    config = run_config.RunConfig(tf_random_seed=77)
    sequence_estimator = dynamic_rnn_estimator.DynamicRnnEstimator(
        problem_type=constants.ProblemType.CLASSIFICATION,
        prediction_type=rnn_common.PredictionType.SINGLE_VALUE,
        num_classes=2,
        num_units=cell_size,
        sequence_feature_columns=seq_columns,
        cell_type=cell_type,
        optimizer=optimizer_type,
        learning_rate=learning_rate,
        momentum=momentum,
        config=config,
        predict_probabilities=True)

    train_input_fn = get_majority_input_fn(batch_size, sequence_length, 1111)
    eval_input_fn = get_majority_input_fn(batch_size, sequence_length, 2222)

    sequence_estimator.fit(input_fn=train_input_fn, steps=train_steps)
    evaluation = sequence_estimator.evaluate(
        input_fn=eval_input_fn, steps=eval_steps)
    accuracy = evaluation['accuracy']
    self.assertGreater(accuracy, accuracy_threshold,
                       'Accuracy should be higher than {}; got {}'.format(
                           accuracy_threshold, accuracy))

    # Testing `predict` when `predict_probabilities=True`.
    prediction_dict = sequence_estimator.predict(
        input_fn=eval_input_fn, as_iterable=False)
    self.assertListEqual(
        sorted(list(prediction_dict.keys())),
        sorted([
            prediction_key.PredictionKey.CLASSES,
            prediction_key.PredictionKey.PROBABILITIES,
            dynamic_rnn_estimator._get_state_name(0),
            dynamic_rnn_estimator._get_state_name(1)
        ]))
    predictions = prediction_dict[prediction_key.PredictionKey.CLASSES]
    probabilities = prediction_dict[
        prediction_key.PredictionKey.PROBABILITIES]
    self.assertListEqual(list(predictions.shape), [batch_size])
    self.assertListEqual(list(probabilities.shape), [batch_size, 2])


if __name__ == '__main__':
  test.main()

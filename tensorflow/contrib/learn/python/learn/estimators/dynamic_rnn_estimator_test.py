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

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import dynamic_rnn_estimator


class IdentityRNNCell(tf.nn.rnn_cell.RNNCell):

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
    return tf.identity(inputs), tf.identity(state)


class MockTargetColumn(object):

  def __init__(self):
    self._num_label_columns = None

  def get_eval_ops(self, features, logits, targets, metrics):
    raise NotImplementedError(
        'MockTargetColumn.get_eval_ops called unexpectedly.')

  def logits_to_predictions(self, flattened_logits, proba=False):
    raise NotImplementedError(
        'MockTargetColumn.logits_to_predictions called unexpectedly.')

  def loss(self, logits, targets, features):
    raise NotImplementedError('MockTargetColumn.loss called unexpectedly.')

  @property
  def num_label_columns(self):
    if self._num_label_columns is None:
      raise ValueError('MockTargetColumn.num_label_columns has not been set.')
    return self._num_label_columns

  def set_num_label_columns(self, n):
    self._num_label_columns = n


class MockOptimizer(object):

  def compute_gradients(self, loss, var_list):
    raise NotImplementedError(
        'MockOptimizer.compute_gradients called unexpectedly.')

  def apply_gradients(self, processed_gradients, global_step):
    raise NotImplementedError(
        'MockOptimizer.apply_gradients called unexpectedly.')


def sequence_length_mask(values, lengths):
  masked = values
  for i, length in enumerate(lengths):
    masked[i, length:, :] = np.zeros_like(masked[i, length:, :])
  return masked


class DynamicRnnEstimatorTest(tf.test.TestCase):

  CELL_STATE_SIZE = 8
  CELL_OUTPUT_SIZE = 6

  def setUp(self):
    self._rnn_cell = IdentityRNNCell(self.CELL_STATE_SIZE,
                                     self.CELL_OUTPUT_SIZE)
    self._mock_target_column = MockTargetColumn()
    self._rnn_estimator = dynamic_rnn_estimator._MultiValueRNNEstimator(
        cell=self._rnn_cell,
        target_column=self._mock_target_column,
        optimizer=tf.train.GradientDescentOptimizer(0.1))

  def testConstructRNN(self):
    """Test `DynamicRNNEstimator._construct_rnn`."""
    batch_size = 4
    padded_length = 6
    num_classes = 4

    # Set up mocks
    self._mock_target_column.set_num_label_columns(num_classes)
    np.random.seed(111)
    mock_linear_layer_output = np.random.rand(
        batch_size, padded_length, num_classes)

    # Create features
    inputs = np.random.rand(batch_size, padded_length, self.CELL_OUTPUT_SIZE)
    sequence_length = np.random.randint(0, padded_length + 1, batch_size)
    features = {'inputs': tf.constant(
        inputs, dtype=tf.float32),
                'sequence_length': tf.constant(
                    sequence_length, dtype=tf.int32)}

    # Map feature to logits with mocked linear layer.
    with tf.test.mock.patch.object(dynamic_rnn_estimator,
                                   'layers') as mock_layers:
      mock_layers.fully_connected.return_value = tf.constant(
          mock_linear_layer_output, dtype=tf.float32)
      logits_t, final_state_t = self._rnn_estimator._construct_rnn(
          features)
      _, fully_connected_kwargs = mock_layers.fully_connected.call_args
      linear_layer_inputs_t = fully_connected_kwargs['inputs']
      linear_layer_output_dim = fully_connected_kwargs['num_outputs']

    # Obtain values of linear layer input, logits and final state.
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      linear_layer_inputs, logits, final_state = sess.run(
          [linear_layer_inputs_t, logits_t, final_state_t])

    np.testing.assert_equal(num_classes, linear_layer_output_dim)
    np.testing.assert_almost_equal(inputs, linear_layer_inputs)
    np.testing.assert_almost_equal(mock_linear_layer_output, logits)
    np.testing.assert_almost_equal(
        np.zeros([batch_size, self._rnn_cell.state_size], dtype=float),
        final_state)


class MultiValueRNNEstimatorTest(tf.test.TestCase):
  """Tests for `_MultiValueRNNEstimator` class."""
  CELL_STATE_SIZE = 8
  CELL_OUTPUT_SIZE = 6

  def setUp(self):
    self._rnn_cell = IdentityRNNCell(self.CELL_STATE_SIZE,
                                     self.CELL_OUTPUT_SIZE)
    self._mock_target_column = MockTargetColumn()
    self._seq_estimator = dynamic_rnn_estimator._MultiValueRNNEstimator(
        cell=self._rnn_cell,
        target_column=self._mock_target_column,
        optimizer=tf.train.GradientDescentOptimizer(0.1))

  def testPaddingMask(self):
    """Test `_padding_mask`."""
    batch_size = 16
    padded_length = 32
    np.random.seed(1234)
    sequence_lengths = np.random.randint(0, padded_length + 1, batch_size)

    padding_mask_t = dynamic_rnn_estimator._padding_mask(
        tf.constant(sequence_lengths, dtype=tf.int32),
        tf.constant(padded_length, dtype=tf.int32))

    with tf.Session() as sess:
      padding_mask = sess.run(padding_mask_t)

    for i in range(batch_size):
      actual_mask = padding_mask[i]
      expected_mask = np.concatenate(
          [np.ones(sequence_lengths[i]),
           np.zeros(padded_length - sequence_lengths[i])],
          axis=0)
      np.testing.assert_equal(actual_mask, expected_mask,
                              'Mismatch on row {}. Got {}; expected {}.'.format(
                                  i, actual_mask, expected_mask))

  def testMaskLogitsAndTargets(self):
    """Test `_mask_logits_and_targets`."""
    batch_size = 4
    padded_length = 6
    num_classes = 4
    np.random.seed(1234)
    sequence_length = np.random.randint(0, padded_length + 1, batch_size)
    logits = np.random.rand(batch_size, padded_length, num_classes)
    targets = np.random.randint(0, num_classes, [batch_size, padded_length])
    (logits_masked_t,
     targets_masked_t) = dynamic_rnn_estimator._mask_logits_and_targets(
         tf.constant(
             logits, dtype=tf.float32),
         tf.constant(
             targets, dtype=tf.int32),
         tf.constant(
             sequence_length, dtype=tf.int32))

    with tf.Session() as sess:
      logits_masked, targets_masked = sess.run(
          [logits_masked_t, targets_masked_t])

    expected_logits_shape = [sum(sequence_length), num_classes]
    np.testing.assert_equal(expected_logits_shape, logits_masked.shape,
                            'Wrong logits shape. Expected {}; got {}.'.format(
                                expected_logits_shape, logits_masked.shape))

    expected_targets_shape = [sum(sequence_length)]
    np.testing.assert_equal(expected_targets_shape, targets_masked.shape,
                            'Wrong targets shape. Expected {}; got {}.'.format(
                                expected_targets_shape, targets_masked.shape))
    masked_index = 0
    for i in range(batch_size):
      for j in range(sequence_length[i]):
        actual_logits = logits_masked[masked_index]
        expected_logits = logits[i, j, :]
        np.testing.assert_almost_equal(
            expected_logits,
            actual_logits,
            err_msg='Unexpected logit value at index [{}, {}, :].'
            '  Expected {}; got {}.'.format(i, j, expected_logits,
                                            actual_logits))

        actual_targets = targets_masked[masked_index]
        expected_targets = targets[i, j]
        np.testing.assert_almost_equal(
            expected_targets,
            actual_targets,
            err_msg='Unexpected logit value at index [{}, {}].'
            ' Expected {}; got {}.'.format(i, j, expected_targets,
                                           actual_targets))
        masked_index += 1

  def testLogitsToPredictions(self):
    """Test `DynamicRNNEstimator._logits_to_predictions`."""
    batch_size = 8
    sequence_length = 16
    num_classes = 3

    np.random.seed(10101)
    logits = np.random.rand(batch_size, sequence_length, num_classes)
    flattened_logits = np.reshape(logits, [-1, num_classes])
    flattened_argmax = np.argmax(flattened_logits, axis=1)
    expected_predictions = np.argmax(logits, axis=2)

    with tf.test.mock.patch.object(self._mock_target_column,
                                   'logits_to_predictions',
                                   return_value=flattened_argmax,
                                   autospec=True) as mock_logits_to_predictions:
      predictions_t = self._seq_estimator._logits_to_predictions(
          None, tf.constant(logits, dtype=tf.float32))
      (target_column_input_logits_t,), _ = mock_logits_to_predictions.call_args

    with tf.Session() as sess:
      target_column_input_logits, predictions = sess.run(
          [target_column_input_logits_t, predictions_t])

    np.testing.assert_almost_equal(flattened_logits, target_column_input_logits)
    np.testing.assert_equal(expected_predictions, predictions)

  def testLearnSineFunction(self):
    """Tests that `_MultiValueRNNEstimator` can learn a sine function."""
    batch_size = 8
    sequence_length = 64
    train_steps = 200
    eval_steps = 20
    cell_size = 4
    learning_rate = 0.1
    loss_threshold = 0.02

    def get_sin_input_fn(batch_size, sequence_length, increment, seed=None):
      def _sin_fn(x):
        ranger = tf.linspace(
            tf.reshape(x[0], []),
            (sequence_length - 1) * increment, sequence_length + 1)
        return tf.sin(ranger)

      def input_fn():
        starts = tf.random_uniform([batch_size], maxval=(2 * np.pi), seed=seed)
        sin_curves = tf.map_fn(_sin_fn, (starts,), dtype=tf.float32)
        inputs = tf.expand_dims(
            tf.slice(sin_curves, [0, 0], [batch_size, sequence_length]), 2)
        labels = tf.slice(sin_curves, [0, 1], [batch_size, sequence_length])
        return {'inputs': inputs}, labels

      return input_fn

    config = tf.contrib.learn.RunConfig(tf_random_seed=1234)
    sequence_estimator = dynamic_rnn_estimator.multi_value_rnn_regressor(
        num_units=cell_size, learning_rate=learning_rate, config=config)

    train_input_fn = get_sin_input_fn(
        batch_size, sequence_length, np.pi / 32, seed=1234)
    eval_input_fn = get_sin_input_fn(
        batch_size, sequence_length, np.pi / 32, seed=4321)

    sequence_estimator.fit(input_fn=train_input_fn, steps=train_steps)
    loss = sequence_estimator.evaluate(
        input_fn=eval_input_fn, steps=eval_steps)['loss']
    self.assertLess(loss, loss_threshold,
                    'Loss should be less than {}; got {}'.format(
                        loss_threshold, loss))

  def testLearnShiftByOne(self):
    """Tests that `_MultiValueRNNEstimator` can learn a 'shift-by-one' example.

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
        random_sequence = tf.random_uniform(
            [batch_size, sequence_length + 1], 0, 2, dtype=tf.int32, seed=seed)
        labels = tf.slice(
            random_sequence, [0, 0], [batch_size, sequence_length])
        inputs = tf.expand_dims(
            tf.to_float(tf.slice(
                random_sequence, [0, 1], [batch_size, sequence_length])), 2)
        return {'inputs': inputs}, labels
      return input_fn

    config = tf.contrib.learn.RunConfig(tf_random_seed=21212)
    sequence_estimator = dynamic_rnn_estimator.multi_value_rnn_classifier(
        num_classes=2,
        num_units=cell_size,
        learning_rate=learning_rate,
        config=config)

    train_input_fn = get_shift_input_fn(batch_size, sequence_length, seed=12321)
    eval_input_fn = get_shift_input_fn(batch_size, sequence_length, seed=32123)

    sequence_estimator.fit(input_fn=train_input_fn, steps=train_steps)
    evaluation = sequence_estimator.evaluate(
        input_fn=eval_input_fn, steps=eval_steps)
    accuracy = evaluation['accuracy']
    self.assertGreater(accuracy, accuracy_threshold,
                       'Accuracy should be higher than {}; got {}'.format(
                           accuracy_threshold, accuracy))


class SingleValueRNNEstimatorTest(tf.test.TestCase):

  def testSelectLastLogits(self):
    """Test `_select_last_logits`."""
    batch_size = 4
    padded_length = 6
    num_classes = 4
    np.random.seed(4444)
    sequence_length = np.random.randint(0, padded_length + 1, batch_size)
    logits = np.random.rand(batch_size, padded_length, num_classes)
    last_logits_t = dynamic_rnn_estimator._select_last_logits(
        tf.constant(logits, dtype=tf.float32),
        tf.constant(sequence_length, dtype=tf.int32))

    with tf.Session() as sess:
      last_logits = sess.run(last_logits_t)

    expected_logits_shape = [batch_size, num_classes]
    np.testing.assert_equal(expected_logits_shape, last_logits.shape,
                            'Wrong logits shape. Expected {}; got {}.'.format(
                                expected_logits_shape, last_logits.shape))

    for i in range(batch_size):
      actual_logits = last_logits[i, :]
      expected_logits = logits[i, sequence_length[i] - 1, :]
      np.testing.assert_almost_equal(
          expected_logits,
          actual_logits,
          err_msg='Unexpected logit value at index [{}, :].'
          '  Expected {}; got {}.'.format(i, expected_logits,
                                          actual_logits))

  def testLearnMean(self):
    """Test that `_SequenceRegressor` can learn to calculate a mean."""
    batch_size = 16
    sequence_length = 3
    train_steps = 200
    eval_steps = 20
    cell_type = 'basic_rnn'
    cell_size = 8
    optimizer_type = 'Momentum'
    learning_rate = 0.5
    momentum = 0.9
    loss_threshold = 0.1

    def get_mean_input_fn(batch_size, sequence_length, seed=None):
      def input_fn():
        # Create examples by choosing 'centers' and adding uniform noise.
        centers = tf.matmul(
            tf.random_uniform(
                [batch_size, 1], -0.75, 0.75, dtype=tf.float32, seed=seed),
            tf.ones([1, sequence_length]))
        noise = tf.random_uniform(
            [batch_size, sequence_length],
            -0.25,
            0.25,
            dtype=tf.float32,
            seed=seed)
        sequences = centers + noise

        inputs = tf.expand_dims(sequences, 2)
        labels = tf.reduce_mean(sequences, reduction_indices=[1])
        return {'inputs': inputs}, labels
      return input_fn

    config = tf.contrib.learn.RunConfig(tf_random_seed=6)
    sequence_regressor = dynamic_rnn_estimator.single_value_rnn_regressor(
        num_units=cell_size,
        cell_type=cell_type,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        momentum=momentum,
        config=config)

    train_input_fn = get_mean_input_fn(batch_size, sequence_length, 121)
    eval_input_fn = get_mean_input_fn(batch_size, sequence_length, 212)

    sequence_regressor.fit(input_fn=train_input_fn, steps=train_steps)
    evaluation = sequence_regressor.evaluate(
        input_fn=eval_input_fn, steps=eval_steps)
    loss = evaluation['loss']
    self.assertLess(loss, loss_threshold,
                    'Loss should be less than {}; got {}'.format(
                        loss_threshold, loss))

  def testLearnMajority(self):
    """Test that `_SequenceClassifier` can learn the 'majority' function."""
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
      tf.set_random_seed(seed)
      def input_fn():
        random_sequence = tf.random_uniform(
            [batch_size, sequence_length], 0, 2, dtype=tf.int32, seed=seed)
        inputs = tf.expand_dims(tf.to_float(random_sequence), 2)
        labels = tf.to_int32(
            tf.squeeze(
                tf.reduce_sum(
                    inputs, reduction_indices=[1]) > (sequence_length / 2.0)))
        return {'inputs': inputs}, labels
      return input_fn

    config = tf.contrib.learn.RunConfig(tf_random_seed=77)
    sequence_classifier = dynamic_rnn_estimator.single_value_rnn_classifier(
        num_classes=2,
        num_units=cell_size,
        cell_type=cell_type,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        momentum=momentum,
        config=config)

    train_input_fn = get_majority_input_fn(batch_size, sequence_length, 1111)
    eval_input_fn = get_majority_input_fn(batch_size, sequence_length, 2222)

    sequence_classifier.fit(input_fn=train_input_fn, steps=train_steps)
    evaluation = sequence_classifier.evaluate(
        input_fn=eval_input_fn, steps=eval_steps)
    accuracy = evaluation['accuracy']
    self.assertGreater(accuracy, accuracy_threshold,
                       'Accuracy should be higher than {}; got {}'.format(
                           accuracy_threshold, accuracy))


if __name__ == '__main__':
  tf.test.main()

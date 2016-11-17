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
from tensorflow.python.ops import rnn_cell


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

  NUM_RNN_CELL_UNITS = 8
  NUM_LABEL_COLUMNS = 6
  INPUTS_COLUMN = tf.contrib.layers.real_valued_column(
      'inputs', dimension=NUM_LABEL_COLUMNS)

  def setUp(self):
    self.rnn_cell = rnn_cell.BasicRNNCell(self.NUM_RNN_CELL_UNITS)
    self.mock_target_column = MockTargetColumn(
        num_label_columns=self.NUM_LABEL_COLUMNS)

    location = tf.contrib.layers.sparse_column_with_keys(
        'location', keys=['west_side', 'east_side', 'nyc'])
    location_onehot = tf.contrib.layers.one_hot_column(location)
    self.context_feature_columns = [location_onehot]

    wire_cast = tf.contrib.layers.sparse_column_with_keys(
        'wire_cast', ['marlo', 'omar', 'stringer'])
    wire_cast_embedded = tf.contrib.layers.embedding_column(
        wire_cast, dimension=8)
    measurements = tf.contrib.layers.real_valued_column(
        'measurements', dimension=2)
    self.sequence_feature_columns = [measurements, wire_cast_embedded]

    self.columns_to_tensors = {
        'location': tf.SparseTensor(
            indices=[[0, 0], [1, 0], [2, 0]],
            values=['west_side', 'west_side', 'nyc'],
            shape=[3, 1]),
        'wire_cast': tf.SparseTensor(
            indices=[[0, 0, 0], [0, 1, 0],
                     [1, 0, 0], [1, 1, 0], [1, 1, 1],
                     [2, 0, 0]],
            values=[b'marlo', b'stringer',
                    b'omar', b'stringer', b'marlo',
                    b'marlo'],
            shape=[3, 2, 2]),
        'measurements': tf.random_uniform([3, 2, 2])}

  def testBuildSequenceInputInput(self):
    sequence_input = dynamic_rnn_estimator.build_sequence_input(
        self.columns_to_tensors,
        self.sequence_feature_columns,
        self.context_feature_columns)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.initialize_all_tables())
      sequence_input_val = sess.run(sequence_input)
    expected_shape = np.array([
        3,         # expected batch size
        2,         # padded sequence length
        3 + 8 + 2  # location keys + embedding dim + measurement dimension
    ])
    self.assertAllEqual(expected_shape, sequence_input_val.shape)

  def testConstructRNN(self):
    initial_state = None
    sequence_input = dynamic_rnn_estimator.build_sequence_input(
        self.columns_to_tensors,
        self.sequence_feature_columns,
        self.context_feature_columns)
    activations_t, final_state_t = dynamic_rnn_estimator.construct_rnn(
        initial_state,
        sequence_input,
        self.rnn_cell,
        self.mock_target_column.num_label_columns)

    # Obtain values of activations and final state.
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.initialize_all_tables())
      activations, final_state = sess.run([activations_t, final_state_t])

    expected_activations_shape = np.array([3, 2, self.NUM_LABEL_COLUMNS])
    self.assertAllEqual(expected_activations_shape, activations.shape)
    expected_state_shape = np.array([3, self.NUM_RNN_CELL_UNITS])
    self.assertAllEqual(expected_state_shape, final_state.shape)

  def testPaddingMask(self):
    """Test `padding_mask`."""
    batch_size = 16
    padded_length = 32
    np.random.seed(1234)
    sequence_lengths = np.random.randint(0, padded_length + 1, batch_size)

    padding_mask_t = dynamic_rnn_estimator.padding_mask(
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

  def testMaskActivationsAndLabels(self):
    """Test `mask_activations_and_labels`."""
    batch_size = 4
    padded_length = 6
    num_classes = 4
    np.random.seed(1234)
    sequence_length = np.random.randint(0, padded_length + 1, batch_size)
    activations = np.random.rand(batch_size, padded_length, num_classes)
    labels = np.random.randint(0, num_classes, [batch_size, padded_length])
    (activations_masked_t,
     labels_masked_t) = dynamic_rnn_estimator.mask_activations_and_labels(
         tf.constant(
             activations, dtype=tf.float32),
         tf.constant(
             labels, dtype=tf.int32),
         tf.constant(
             sequence_length, dtype=tf.int32))

    with tf.Session() as sess:
      activations_masked, labels_masked = sess.run(
          [activations_masked_t, labels_masked_t])

    expected_activations_shape = [sum(sequence_length), num_classes]
    np.testing.assert_equal(
        expected_activations_shape, activations_masked.shape,
        'Wrong activations shape. Expected {}; got {}.'.format(
            expected_activations_shape, activations_masked.shape))

    expected_labels_shape = [sum(sequence_length)]
    np.testing.assert_equal(expected_labels_shape, labels_masked.shape,
                            'Wrong labels shape. Expected {}; got {}.'.format(
                                expected_labels_shape, labels_masked.shape))
    masked_index = 0
    for i in range(batch_size):
      for j in range(sequence_length[i]):
        actual_activations = activations_masked[masked_index]
        expected_activations = activations[i, j, :]
        np.testing.assert_almost_equal(
            expected_activations,
            actual_activations,
            err_msg='Unexpected logit value at index [{}, {}, :].'
            '  Expected {}; got {}.'.format(i, j, expected_activations,
                                            actual_activations))

        actual_labels = labels_masked[masked_index]
        expected_labels = labels[i, j]
        np.testing.assert_almost_equal(
            expected_labels,
            actual_labels,
            err_msg='Unexpected logit value at index [{}, {}].'
            ' Expected {}; got {}.'.format(i, j, expected_labels,
                                           actual_labels))
        masked_index += 1

  def testSelectLastActivations(self):
    """Test `select_last_activations`."""
    batch_size = 4
    padded_length = 6
    num_classes = 4
    np.random.seed(4444)
    sequence_length = np.random.randint(0, padded_length + 1, batch_size)
    activations = np.random.rand(batch_size, padded_length, num_classes)
    last_activations_t = dynamic_rnn_estimator.select_last_activations(
        tf.constant(activations, dtype=tf.float32),
        tf.constant(sequence_length, dtype=tf.int32))

    with tf.Session() as sess:
      last_activations = sess.run(last_activations_t)

    expected_activations_shape = [batch_size, num_classes]
    np.testing.assert_equal(
        expected_activations_shape, last_activations.shape,
        'Wrong activations shape. Expected {}; got {}.'.format(
            expected_activations_shape, last_activations.shape))

    for i in range(batch_size):
      actual_activations = last_activations[i, :]
      expected_activations = activations[i, sequence_length[i] - 1, :]
      np.testing.assert_almost_equal(
          expected_activations,
          actual_activations,
          err_msg='Unexpected logit value at index [{}, :].'
          '  Expected {}; got {}.'.format(i, expected_activations,
                                          actual_activations))

# TODO(jamieas): move all tests below to a benchmark test.
class DynamicRNNEstimatorLearningTest(tf.test.TestCase):
  """Learning tests for dymanic RNN Estimators."""

  def testLearnSineFunction(self):
    """Tests learning a sine function."""
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

    seq_columns = [tf.contrib.layers.real_valued_column(
        'inputs', dimension=cell_size)]
    config = tf.contrib.learn.RunConfig(tf_random_seed=1234)
    sequence_estimator = dynamic_rnn_estimator.multi_value_rnn_regressor(
        num_units=cell_size,
        sequence_feature_columns=seq_columns,
        learning_rate=learning_rate,
        input_keep_probability=0.9,
        output_keep_probability=0.9,
        config=config)

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
        random_sequence = tf.random_uniform(
            [batch_size, sequence_length + 1], 0, 2, dtype=tf.int32, seed=seed)
        labels = tf.slice(
            random_sequence, [0, 0], [batch_size, sequence_length])
        inputs = tf.expand_dims(
            tf.to_float(tf.slice(
                random_sequence, [0, 1], [batch_size, sequence_length])), 2)
        return {'inputs': inputs}, labels
      return input_fn

    seq_columns = [tf.contrib.layers.real_valued_column(
        'inputs', dimension=cell_size)]
    config = tf.contrib.learn.RunConfig(tf_random_seed=21212)
    sequence_estimator = dynamic_rnn_estimator.multi_value_rnn_classifier(
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
        sorted([dynamic_rnn_estimator.RNNKeys.PREDICTIONS_KEY,
                dynamic_rnn_estimator.RNNKeys.PROBABILITIES_KEY,
                dynamic_rnn_estimator.RNNKeys.FINAL_STATE_KEY]))
    predictions = prediction_dict[dynamic_rnn_estimator.RNNKeys.PREDICTIONS_KEY]
    probabilities = prediction_dict[
        dynamic_rnn_estimator.RNNKeys.PROBABILITIES_KEY]
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

    seq_columns = [tf.contrib.layers.real_valued_column(
        'inputs', dimension=cell_size)]
    config = tf.contrib.learn.RunConfig(tf_random_seed=6)
    sequence_regressor = dynamic_rnn_estimator.single_value_rnn_regressor(
        num_units=cell_size,
        sequence_feature_columns=seq_columns,
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

    seq_columns = [tf.contrib.layers.real_valued_column(
        'inputs', dimension=cell_size)]
    config = tf.contrib.learn.RunConfig(tf_random_seed=77)
    sequence_classifier = dynamic_rnn_estimator.single_value_rnn_classifier(
        num_classes=2,
        num_units=cell_size,
        sequence_feature_columns=seq_columns,
        cell_type=cell_type,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        momentum=momentum,
        config=config,
        predict_probabilities=True)

    train_input_fn = get_majority_input_fn(batch_size, sequence_length, 1111)
    eval_input_fn = get_majority_input_fn(batch_size, sequence_length, 2222)

    sequence_classifier.fit(input_fn=train_input_fn, steps=train_steps)
    evaluation = sequence_classifier.evaluate(
        input_fn=eval_input_fn, steps=eval_steps)
    accuracy = evaluation['accuracy']
    self.assertGreater(accuracy, accuracy_threshold,
                       'Accuracy should be higher than {}; got {}'.format(
                           accuracy_threshold, accuracy))

    # Testing `predict` when `predict_probabilities=True`.
    prediction_dict = sequence_classifier.predict(
        input_fn=eval_input_fn, as_iterable=False)
    self.assertListEqual(
        sorted(list(prediction_dict.keys())),
        sorted([dynamic_rnn_estimator.RNNKeys.PREDICTIONS_KEY,
                dynamic_rnn_estimator.RNNKeys.PROBABILITIES_KEY,
                dynamic_rnn_estimator.RNNKeys.FINAL_STATE_KEY]))
    predictions = prediction_dict[dynamic_rnn_estimator.RNNKeys.PREDICTIONS_KEY]
    probabilities = prediction_dict[
        dynamic_rnn_estimator.RNNKeys.PROBABILITIES_KEY]
    self.assertListEqual(list(predictions.shape), [batch_size])
    self.assertListEqual(list(probabilities.shape), [batch_size, 2])

if __name__ == '__main__':
  tf.test.main()

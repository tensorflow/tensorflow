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
"""Tests for learn.estimators.state_saving_rnn_estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import numpy as np

from tensorflow.contrib import lookup
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.layers.python.layers import target_column as target_column_lib
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.learn.python.learn.estimators import rnn_common
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.estimators import state_saving_rnn_estimator as ssre
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class PrepareInputsForRnnTest(test.TestCase):

  def _test_prepare_inputs_for_rnn(self, sequence_features, context_features,
                                   sequence_feature_columns, num_unroll,
                                   expected):
    features_by_time = ssre._prepare_inputs_for_rnn(sequence_features,
                                                    context_features,
                                                    sequence_feature_columns,
                                                    num_unroll)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(lookup_ops.tables_initializer())
      features_val = sess.run(features_by_time)
      self.assertAllEqual(expected, features_val)

  def testPrepareInputsForRnnBatchSize1(self):
    num_unroll = 3

    expected = [
        np.array([[11., 31., 5., 7.]]), np.array([[12., 32., 5., 7.]]),
        np.array([[13., 33., 5., 7.]])
    ]

    sequence_features = {
        'seq_feature0': constant_op.constant([[11., 12., 13.]]),
        'seq_feature1': constant_op.constant([[31., 32., 33.]])
    }

    sequence_feature_columns = [
        feature_column.real_valued_column(
            'seq_feature0', dimension=1),
        feature_column.real_valued_column(
            'seq_feature1', dimension=1),
    ]

    context_features = {
        'ctx_feature0': constant_op.constant([[5.]]),
        'ctx_feature1': constant_op.constant([[7.]])
    }
    self._test_prepare_inputs_for_rnn(sequence_features, context_features,
                                      sequence_feature_columns, num_unroll,
                                      expected)

  def testPrepareInputsForRnnBatchSize2(self):

    num_unroll = 3

    expected = [
        np.array([[11., 31., 5., 7.], [21., 41., 6., 8.]]),
        np.array([[12., 32., 5., 7.], [22., 42., 6., 8.]]),
        np.array([[13., 33., 5., 7.], [23., 43., 6., 8.]])
    ]

    sequence_features = {
        'seq_feature0':
            constant_op.constant([[11., 12., 13.], [21., 22., 23.]]),
        'seq_feature1':
            constant_op.constant([[31., 32., 33.], [41., 42., 43.]])
    }

    sequence_feature_columns = [
        feature_column.real_valued_column(
            'seq_feature0', dimension=1),
        feature_column.real_valued_column(
            'seq_feature1', dimension=1),
    ]

    context_features = {
        'ctx_feature0': constant_op.constant([[5.], [6.]]),
        'ctx_feature1': constant_op.constant([[7.], [8.]])
    }

    self._test_prepare_inputs_for_rnn(sequence_features, context_features,
                                      sequence_feature_columns, num_unroll,
                                      expected)

  def testPrepareInputsForRnnNoContext(self):
    num_unroll = 3

    expected = [
        np.array([[11., 31.], [21., 41.]]), np.array([[12., 32.], [22., 42.]]),
        np.array([[13., 33.], [23., 43.]])
    ]

    sequence_features = {
        'seq_feature0':
            constant_op.constant([[11., 12., 13.], [21., 22., 23.]]),
        'seq_feature1':
            constant_op.constant([[31., 32., 33.], [41., 42., 43.]])
    }

    sequence_feature_columns = [
        feature_column.real_valued_column(
            'seq_feature0', dimension=1),
        feature_column.real_valued_column(
            'seq_feature1', dimension=1),
    ]

    context_features = None

    self._test_prepare_inputs_for_rnn(sequence_features, context_features,
                                      sequence_feature_columns, num_unroll,
                                      expected)

  def testPrepareInputsForRnnSparse(self):
    num_unroll = 2
    embedding_dimension = 8

    expected = [
        np.array([[1., 1., 1., 1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1., 1., 1., 1.]]),
        np.array([[1., 1., 1., 1., 1., 1., 1., 1.],
                  [2., 2., 2., 2., 2., 2., 2., 2.],
                  [1., 1., 1., 1., 1., 1., 1., 1.]])
    ]

    sequence_features = {
        'wire_cast':
            sparse_tensor.SparseTensor(
                indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1],
                         [2, 0, 0], [2, 1, 1]],
                values=[
                    b'marlo', b'stringer', b'omar', b'stringer', b'marlo',
                    b'marlo', b'omar'
                ],
                dense_shape=[3, 2, 2])
    }

    wire_cast = feature_column.sparse_column_with_keys(
        'wire_cast', ['marlo', 'omar', 'stringer'])
    sequence_feature_columns = [
        feature_column.embedding_column(
            wire_cast,
            dimension=embedding_dimension,
            combiner='sum',
            initializer=init_ops.ones_initializer())
    ]

    context_features = None

    self._test_prepare_inputs_for_rnn(sequence_features, context_features,
                                      sequence_feature_columns, num_unroll,
                                      expected)

  def testPrepareInputsForRnnSparseAndDense(self):
    num_unroll = 2
    embedding_dimension = 8
    dense_dimension = 2

    expected = [
        np.array([[1., 1., 1., 1., 1., 1., 1., 1., 111., 112.],
                  [1., 1., 1., 1., 1., 1., 1., 1., 211., 212.],
                  [1., 1., 1., 1., 1., 1., 1., 1., 311., 312.]]),
        np.array([[1., 1., 1., 1., 1., 1., 1., 1., 121., 122.],
                  [2., 2., 2., 2., 2., 2., 2., 2., 221., 222.],
                  [1., 1., 1., 1., 1., 1., 1., 1., 321., 322.]])
    ]

    sequence_features = {
        'wire_cast':
            sparse_tensor.SparseTensor(
                indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1],
                         [2, 0, 0], [2, 1, 1]],
                values=[
                    b'marlo', b'stringer', b'omar', b'stringer', b'marlo',
                    b'marlo', b'omar'
                ],
                dense_shape=[3, 2, 2]),
        'seq_feature0':
            constant_op.constant([[[111., 112.], [121., 122.]],
                                  [[211., 212.], [221., 222.]],
                                  [[311., 312.], [321., 322.]]])
    }

    wire_cast = feature_column.sparse_column_with_keys(
        'wire_cast', ['marlo', 'omar', 'stringer'])
    wire_cast_embedded = feature_column.embedding_column(
        wire_cast,
        dimension=embedding_dimension,
        combiner='sum',
        initializer=init_ops.ones_initializer())
    seq_feature0_column = feature_column.real_valued_column(
        'seq_feature0', dimension=dense_dimension)

    sequence_feature_columns = [seq_feature0_column, wire_cast_embedded]

    context_features = None

    self._test_prepare_inputs_for_rnn(sequence_features, context_features,
                                      sequence_feature_columns, num_unroll,
                                      expected)


class StateSavingRnnEstimatorTest(test.TestCase):

  def testPrepareFeaturesForSQSS(self):
    mode = model_fn_lib.ModeKeys.TRAIN
    seq_feature_name = 'seq_feature'
    sparse_seq_feature_name = 'wire_cast'
    ctx_feature_name = 'ctx_feature'
    sequence_length = 4
    embedding_dimension = 8

    features = {
        sparse_seq_feature_name:
            sparse_tensor.SparseTensor(
                indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1],
                         [2, 0, 0], [2, 1, 1]],
                values=[
                    b'marlo', b'stringer', b'omar', b'stringer', b'marlo',
                    b'marlo', b'omar'
                ],
                dense_shape=[3, 2, 2]),
        seq_feature_name:
            constant_op.constant(
                1.0, shape=[sequence_length]),
        ctx_feature_name:
            constant_op.constant(2.0)
    }

    labels = constant_op.constant(5.0, shape=[sequence_length])

    wire_cast = feature_column.sparse_column_with_keys(
        'wire_cast', ['marlo', 'omar', 'stringer'])
    sequence_feature_columns = [
        feature_column.real_valued_column(
            seq_feature_name, dimension=1), feature_column.embedding_column(
                wire_cast,
                dimension=embedding_dimension,
                initializer=init_ops.ones_initializer())
    ]

    context_feature_columns = [
        feature_column.real_valued_column(
            ctx_feature_name, dimension=1)
    ]

    expected_sequence = {
        rnn_common.RNNKeys.LABELS_KEY:
            np.array([5., 5., 5., 5.]),
        seq_feature_name:
            np.array([1., 1., 1., 1.]),
        sparse_seq_feature_name:
            sparse_tensor.SparseTensor(
                indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1],
                         [2, 0, 0], [2, 1, 1]],
                values=[
                    b'marlo', b'stringer', b'omar', b'stringer', b'marlo',
                    b'marlo', b'omar'
                ],
                dense_shape=[3, 2, 2]),
    }

    expected_context = {ctx_feature_name: 2.}

    sequence, context = ssre._prepare_features_for_sqss(
        features, labels, mode, sequence_feature_columns,
        context_feature_columns)

    def assert_equal(expected, got):
      self.assertEqual(sorted(expected), sorted(got))
      for k, v in expected.items():
        if isinstance(v, sparse_tensor.SparseTensor):
          self.assertAllEqual(v.values.eval(), got[k].values)
          self.assertAllEqual(v.indices.eval(), got[k].indices)
          self.assertAllEqual(v.dense_shape.eval(), got[k].dense_shape)
        else:
          self.assertAllEqual(v, got[k])

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(lookup_ops.tables_initializer())
      actual_sequence, actual_context = sess.run(
          [sequence, context])
      assert_equal(expected_sequence, actual_sequence)
      assert_equal(expected_context, actual_context)

  def _getModelFnOpsForMode(self, mode):
    """Helper for testGetRnnModelFn{Train,Eval,Infer}()."""
    num_units = [4]
    seq_columns = [
        feature_column.real_valued_column(
            'inputs', dimension=1)
    ]
    features = {
        'inputs': constant_op.constant([1., 2., 3.]),
    }
    labels = constant_op.constant([1., 0., 1.])
    model_fn = ssre._get_rnn_model_fn(
        cell_type='basic_rnn',
        target_column=target_column_lib.multi_class_target(n_classes=2),
        optimizer='SGD',
        num_unroll=2,
        num_units=num_units,
        num_threads=1,
        queue_capacity=10,
        batch_size=1,
        # Only CLASSIFICATION yields eval metrics to test for.
        problem_type=constants.ProblemType.CLASSIFICATION,
        sequence_feature_columns=seq_columns,
        context_feature_columns=None,
        learning_rate=0.1)
    model_fn_ops = model_fn(features=features, labels=labels, mode=mode)
    return model_fn_ops

  # testGetRnnModelFn{Train,Eval,Infer}() test which fields
  # of ModelFnOps are set depending on mode.
  def testGetRnnModelFnTrain(self):
    model_fn_ops = self._getModelFnOpsForMode(model_fn_lib.ModeKeys.TRAIN)
    self.assertIsNotNone(model_fn_ops.predictions)
    self.assertIsNotNone(model_fn_ops.loss)
    self.assertIsNotNone(model_fn_ops.train_op)
    # None may get normalized to {}; we accept neither.
    self.assertNotEqual(len(model_fn_ops.eval_metric_ops), 0)

  def testGetRnnModelFnEval(self):
    model_fn_ops = self._getModelFnOpsForMode(model_fn_lib.ModeKeys.EVAL)
    self.assertIsNotNone(model_fn_ops.predictions)
    self.assertIsNotNone(model_fn_ops.loss)
    self.assertIsNone(model_fn_ops.train_op)
    # None may get normalized to {}; we accept neither.
    self.assertNotEqual(len(model_fn_ops.eval_metric_ops), 0)

  def testGetRnnModelFnInfer(self):
    model_fn_ops = self._getModelFnOpsForMode(model_fn_lib.ModeKeys.INFER)
    self.assertIsNotNone(model_fn_ops.predictions)
    self.assertIsNone(model_fn_ops.loss)
    self.assertIsNone(model_fn_ops.train_op)
    # None may get normalized to {}; we accept both.
    self.assertFalse(model_fn_ops.eval_metric_ops)

  def testExport(self):
    input_feature_key = 'magic_input_feature_key'
    batch_size = 8
    num_units = [4]
    sequence_length = 10
    num_unroll = 2
    num_classes = 2

    seq_columns = [
        feature_column.real_valued_column(
            'inputs', dimension=4)
    ]

    def get_input_fn(mode, seed):

      def input_fn():
        features = {}
        random_sequence = random_ops.random_uniform(
            [sequence_length + 1], 0, 2, dtype=dtypes.int32, seed=seed)
        labels = array_ops.slice(random_sequence, [0], [sequence_length])
        inputs = math_ops.to_float(
            array_ops.slice(random_sequence, [1], [sequence_length]))
        features = {'inputs': inputs}

        if mode == model_fn_lib.ModeKeys.INFER:
          input_examples = array_ops.placeholder(dtypes.string)
          features[input_feature_key] = input_examples
          labels = None
        return features, labels

      return input_fn

    model_dir = tempfile.mkdtemp()

    def estimator_fn():
      return ssre.StateSavingRnnEstimator(
          constants.ProblemType.CLASSIFICATION,
          num_units=num_units,
          num_unroll=num_unroll,
          batch_size=batch_size,
          sequence_feature_columns=seq_columns,
          num_classes=num_classes,
          predict_probabilities=True,
          model_dir=model_dir,
          queue_capacity=2 + batch_size,
          seed=1234)

    # Train a bit to create an exportable checkpoint.
    estimator_fn().fit(input_fn=get_input_fn(
        model_fn_lib.ModeKeys.TRAIN, seed=1234),
                       steps=100)
    # Now export, but from a fresh estimator instance, like you would
    # in an export binary. That means .export() has to work without
    # .fit() being called on the same object.
    export_dir = tempfile.mkdtemp()
    print('Exporting to', export_dir)
    estimator_fn().export(
        export_dir,
        input_fn=get_input_fn(
            model_fn_lib.ModeKeys.INFER, seed=4321),
        use_deprecated_input_fn=False,
        input_feature_key=input_feature_key)


# Smoke tests to ensure deprecated constructor functions still work.
class LegacyConstructorTest(test.TestCase):

  def _get_input_fn(self,
                    sequence_length,
                    seed=None):
    def input_fn():
      random_sequence = random_ops.random_uniform(
          [sequence_length + 1], 0, 2, dtype=dtypes.int32, seed=seed)
      labels = array_ops.slice(random_sequence, [0], [sequence_length])
      inputs = math_ops.to_float(
          array_ops.slice(random_sequence, [1], [sequence_length]))
      return {'inputs': inputs}, labels
    return input_fn


# TODO(jtbates): move all tests below to a benchmark test.
class StateSavingRNNEstimatorLearningTest(test.TestCase):
  """Learning tests for state saving RNN Estimators."""

  def testLearnSineFunction(self):
    """Tests learning a sine function."""
    batch_size = 8
    num_unroll = 5
    sequence_length = 64
    train_steps = 250
    eval_steps = 20
    num_rnn_layers = 1
    num_units = [4] * num_rnn_layers
    learning_rate = 0.3
    loss_threshold = 0.035

    def get_sin_input_fn(sequence_length, increment, seed=None):

      def input_fn():
        start = random_ops.random_uniform(
            (), minval=0, maxval=(np.pi * 2.0), dtype=dtypes.float32, seed=seed)
        sin_curves = math_ops.sin(
            math_ops.linspace(start, (sequence_length - 1) * increment,
                              sequence_length + 1))
        inputs = array_ops.slice(sin_curves, [0], [sequence_length])
        labels = array_ops.slice(sin_curves, [1], [sequence_length])
        return {'inputs': inputs}, labels

      return input_fn

    seq_columns = [
        feature_column.real_valued_column(
            'inputs', dimension=1)
    ]
    config = run_config.RunConfig(tf_random_seed=1234)
    dropout_keep_probabilities = [0.9] * (num_rnn_layers + 1)
    sequence_estimator = ssre.StateSavingRnnEstimator(
        constants.ProblemType.LINEAR_REGRESSION,
        num_units=num_units,
        cell_type='lstm',
        num_unroll=num_unroll,
        batch_size=batch_size,
        sequence_feature_columns=seq_columns,
        learning_rate=learning_rate,
        dropout_keep_probabilities=dropout_keep_probabilities,
        config=config,
        queue_capacity=2 * batch_size,
        seed=1234)

    train_input_fn = get_sin_input_fn(sequence_length, np.pi / 32, seed=1234)
    eval_input_fn = get_sin_input_fn(sequence_length, np.pi / 32, seed=4321)

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
    num_classes = 2
    num_unroll = 32
    sequence_length = 32
    train_steps = 300
    eval_steps = 20
    num_units = [4]
    learning_rate = 0.5
    accuracy_threshold = 0.9

    def get_shift_input_fn(sequence_length, seed=None):

      def input_fn():
        random_sequence = random_ops.random_uniform(
            [sequence_length + 1], 0, 2, dtype=dtypes.int32, seed=seed)
        labels = array_ops.slice(random_sequence, [0], [sequence_length])
        inputs = math_ops.to_float(
            array_ops.slice(random_sequence, [1], [sequence_length]))
        return {'inputs': inputs}, labels

      return input_fn

    seq_columns = [
        feature_column.real_valued_column(
            'inputs', dimension=1)
    ]
    config = run_config.RunConfig(tf_random_seed=21212)
    sequence_estimator = ssre.StateSavingRnnEstimator(
        constants.ProblemType.CLASSIFICATION,
        num_units=num_units,
        cell_type='lstm',
        num_unroll=num_unroll,
        batch_size=batch_size,
        sequence_feature_columns=seq_columns,
        num_classes=num_classes,
        learning_rate=learning_rate,
        config=config,
        predict_probabilities=True,
        queue_capacity=2 + batch_size,
        seed=1234)

    train_input_fn = get_shift_input_fn(sequence_length, seed=12321)
    eval_input_fn = get_shift_input_fn(sequence_length, seed=32123)

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
            prediction_key.PredictionKey.PROBABILITIES, ssre._get_state_name(0)
        ]))
    predictions = prediction_dict[prediction_key.PredictionKey.CLASSES]
    probabilities = prediction_dict[prediction_key.PredictionKey.PROBABILITIES]
    self.assertListEqual(list(predictions.shape), [batch_size, sequence_length])
    self.assertListEqual(
        list(probabilities.shape), [batch_size, sequence_length, 2])

  def testLearnLyrics(self):
    lyrics = 'if I go there will be trouble and if I stay it will be double'
    lyrics_list = lyrics.split()
    sequence_length = len(lyrics_list)
    vocab = set(lyrics_list)
    batch_size = 16
    num_classes = len(vocab)
    num_unroll = 7  # not a divisor of sequence_length
    train_steps = 350
    eval_steps = 30
    num_units = [4]
    learning_rate = 0.4
    accuracy_threshold = 0.65

    def get_lyrics_input_fn(seed):

      def input_fn():
        start = random_ops.random_uniform(
            (), minval=0, maxval=sequence_length, dtype=dtypes.int32, seed=seed)
        # Concatenate lyrics_list so inputs and labels wrap when start > 0.
        lyrics_list_concat = lyrics_list + lyrics_list
        inputs_dense = array_ops.slice(lyrics_list_concat, [start],
                                       [sequence_length])
        indices = array_ops.constant(
            [[i, 0] for i in range(sequence_length)], dtype=dtypes.int64)
        dense_shape = [sequence_length, 1]
        inputs = sparse_tensor.SparseTensor(
            indices=indices, values=inputs_dense, dense_shape=dense_shape)
        table = lookup.string_to_index_table_from_tensor(
            mapping=list(vocab), default_value=-1, name='lookup')
        labels = table.lookup(
            array_ops.slice(lyrics_list_concat, [start + 1], [sequence_length]))
        return {'lyrics': inputs}, labels

      return input_fn

    sequence_feature_columns = [
        feature_column.embedding_column(
            feature_column.sparse_column_with_keys('lyrics', vocab),
            dimension=8)
    ]
    config = run_config.RunConfig(tf_random_seed=21212)
    sequence_estimator = ssre.StateSavingRnnEstimator(
        constants.ProblemType.CLASSIFICATION,
        num_units=num_units,
        cell_type='basic_rnn',
        num_unroll=num_unroll,
        batch_size=batch_size,
        sequence_feature_columns=sequence_feature_columns,
        num_classes=num_classes,
        learning_rate=learning_rate,
        config=config,
        predict_probabilities=True,
        queue_capacity=2 + batch_size,
        seed=1234)

    train_input_fn = get_lyrics_input_fn(seed=12321)
    eval_input_fn = get_lyrics_input_fn(seed=32123)

    sequence_estimator.fit(input_fn=train_input_fn, steps=train_steps)

    evaluation = sequence_estimator.evaluate(
        input_fn=eval_input_fn, steps=eval_steps)
    accuracy = evaluation['accuracy']
    self.assertGreater(accuracy, accuracy_threshold,
                       'Accuracy should be higher than {}; got {}'.format(
                           accuracy_threshold, accuracy))


if __name__ == '__main__':
  test.main()

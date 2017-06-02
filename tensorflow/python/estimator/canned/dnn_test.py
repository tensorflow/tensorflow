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
"""Tests for dnn.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import numpy as np
import six

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import dnn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.estimator.inputs import pandas_io
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary import summary as summary_lib
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import monitored_session
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

try:
  # pylint: disable=g-import-not-at-top
  import pandas as pd
  HAS_PANDAS = True
except IOError:
  # Pandas writes a temporary file during import. If it fails, don't use pandas.
  HAS_PANDAS = False
except ImportError:
  HAS_PANDAS = False

# Names of variables created by model.
_LEARNING_RATE_NAME = 'dnn/regression_head/dnn/learning_rate'
_HIDDEN_WEIGHTS_NAME_PATTERN = 'dnn/hiddenlayer_%d/kernel'
_HIDDEN_BIASES_NAME_PATTERN = 'dnn/hiddenlayer_%d/bias'
_LOGITS_WEIGHTS_NAME = 'dnn/logits/kernel'
_LOGITS_BIASES_NAME = 'dnn/logits/bias'


def _create_checkpoint(weights_and_biases, global_step, model_dir):
  """Create checkpoint file with provided model weights.

  Args:
    weights_and_biases: Iterable of tuples of weight and bias values.
    global_step: Initial global step to save in checkpoint.
    model_dir: Directory into which checkpoint is saved.
  """
  weights, biases = zip(*weights_and_biases)
  model_weights = {}

  # Hidden layer weights.
  for i in range(0, len(weights) - 1):
    model_weights[_HIDDEN_WEIGHTS_NAME_PATTERN % i] = weights[i]
    model_weights[_HIDDEN_BIASES_NAME_PATTERN % i] = biases[i]

  # Output layer weights.
  model_weights[_LOGITS_WEIGHTS_NAME] = weights[-1]
  model_weights[_LOGITS_BIASES_NAME] = biases[-1]

  with ops.Graph().as_default():
    # Create model variables.
    for k, v in six.iteritems(model_weights):
      variables_lib.Variable(v, name=k, dtype=dtypes.float32)

    # Create non-model variables.
    global_step_var = training_util.create_global_step()

    # Initialize vars and save checkpoint.
    with tf_session.Session() as sess:
      variables_lib.global_variables_initializer().run()
      global_step_var.assign(global_step).eval()
      saver.Saver().save(sess, os.path.join(model_dir, 'model.ckpt'))


def _mock_head(
    testcase, hidden_units, logits_dimension, expected_logits):
  """Returns a mock head that validates logits values and variable names."""
  hidden_weights_names = [
      (_HIDDEN_WEIGHTS_NAME_PATTERN + '/part_0:0') % i
      for i in range(len(hidden_units))]
  hidden_biases_names = [
      (_HIDDEN_BIASES_NAME_PATTERN + '/part_0:0') % i
      for i in range(len(hidden_units))]
  expected_var_names = (
      hidden_weights_names + hidden_biases_names +
      [_LOGITS_WEIGHTS_NAME + '/part_0:0', _LOGITS_BIASES_NAME + '/part_0:0'])

  def _create_estimator_spec(features, mode, logits, labels, train_op_fn):
    del features, labels  # Not used.
    trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    testcase.assertItemsEqual(
        expected_var_names,
        [var.name for var in trainable_vars])
    loss = constant_op.constant(1.)
    assert_logits = _assert_close(
        expected_logits, logits, message='Failed for mode={}. '.format(mode))
    with ops.control_dependencies([assert_logits]):
      if mode == model_fn.ModeKeys.TRAIN:
        return model_fn.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op_fn(loss))
      elif mode == model_fn.ModeKeys.EVAL:
        return model_fn.EstimatorSpec(
            mode=mode,
            loss=array_ops.identity(loss))
      elif mode == model_fn.ModeKeys.PREDICT:
        return model_fn.EstimatorSpec(
            mode=mode,
            predictions={'logits': array_ops.identity(logits)})
      else:
        testcase.fail('Invalid mode: {}'.format(mode))

  mock_head = test.mock.NonCallableMagicMock(spec=head_lib._Head)
  mock_head.logits_dimension = logits_dimension
  mock_head.create_estimator_spec = test.mock.MagicMock(
      wraps=_create_estimator_spec)

  return mock_head


class DNNModelFnTest(test.TestCase):
  """Tests that _dnn_model_fn passes expected logits to mock head."""

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _test_logits(
      self, mode, hidden_units, logits_dimension, inputs, expected_logits):
    """Tests that the expected logits are passed to mock head."""
    with ops.Graph().as_default():
      training_util.create_global_step()
      head = _mock_head(
          self,
          hidden_units=hidden_units,
          logits_dimension=logits_dimension,
          expected_logits=expected_logits)
      estimator_spec = dnn._dnn_model_fn(
          features={'age': constant_op.constant(inputs)},
          labels=constant_op.constant([[1]]),
          mode=mode,
          head=head,
          hidden_units=hidden_units,
          feature_columns=[
              feature_column.numeric_column('age',
                                            shape=np.array(inputs).shape[1:])],
          optimizer=_mock_optimizer(self, hidden_units))
      with monitored_session.MonitoredTrainingSession(
          checkpoint_dir=self._model_dir) as sess:
        if mode == model_fn.ModeKeys.TRAIN:
          sess.run(estimator_spec.train_op)
        elif mode == model_fn.ModeKeys.EVAL:
          sess.run(estimator_spec.loss)
        elif mode == model_fn.ModeKeys.PREDICT:
          sess.run(estimator_spec.predictions)
        else:
          self.fail('Invalid mode: {}'.format(mode))

  def test_one_dim_logits(self):
    """Tests one-dimensional logits.

    input_layer = [[10]]
    hidden_layer_0 = [[relu(0.6*10 +0.1), relu(0.5*10 -0.1)]] = [[6.1, 4.9]]
    hidden_layer_1 = [[relu(1*6.1 -0.8*4.9 +0.2), relu(0.8*6.1 -1*4.9 -0.1)]]
                   = [[relu(2.38), relu(-0.12)]] = [[2.38, 0]]
    logits = [[-1*2.38 +1*0 +0.3]] = [[-2.08]]
    """
    base_global_step = 100
    _create_checkpoint((
        ([[.6, .5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1.], [1.]], [.3]),
    ), base_global_step, self._model_dir)

    for mode in [model_fn.ModeKeys.TRAIN,
                 model_fn.ModeKeys.EVAL,
                 model_fn.ModeKeys.PREDICT]:
      self._test_logits(
          mode,
          hidden_units=(2, 2),
          logits_dimension=1,
          inputs=[[10.]],
          expected_logits=[[-2.08]])

  def test_multi_dim_logits(self):
    """Tests multi-dimensional logits.

    input_layer = [[10]]
    hidden_layer_0 = [[relu(0.6*10 +0.1), relu(0.5*10 -0.1)]] = [[6.1, 4.9]]
    hidden_layer_1 = [[relu(1*6.1 -0.8*4.9 +0.2), relu(0.8*6.1 -1*4.9 -0.1)]]
                   = [[relu(2.38), relu(-0.12)]] = [[2.38, 0]]
    logits = [[-1*2.38 +0.3, 1*2.38 -0.3, 0.5*2.38]]
           = [[-2.08, 2.08, 1.19]]
    """
    base_global_step = 100
    _create_checkpoint((
        ([[.6, .5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),
    ), base_global_step, self._model_dir)

    for mode in [model_fn.ModeKeys.TRAIN,
                 model_fn.ModeKeys.EVAL,
                 model_fn.ModeKeys.PREDICT]:
      self._test_logits(
          mode,
          hidden_units=(2, 2),
          logits_dimension=3,
          inputs=[[10.]],
          expected_logits=[[-2.08, 2.08, 1.19]])

  def test_multi_example_multi_dim_logits(self):
    """Tests multiple examples and multi-dimensional logits.

    input_layer = [[10], [5]]
    hidden_layer_0 = [[relu(0.6*10 +0.1), relu(0.5*10 -0.1)],
                      [relu(0.6*5 +0.1), relu(0.5*5 -0.1)]]
                   = [[6.1, 4.9], [3.1, 2.4]]
    hidden_layer_1 = [[relu(1*6.1 -0.8*4.9 +0.2), relu(0.8*6.1 -1*4.9 -0.1)],
                      [relu(1*3.1 -0.8*2.4 +0.2), relu(0.8*3.1 -1*2.4 -0.1)]]
                   = [[2.38, 0], [1.38, 0]]
    logits = [[-1*2.38 +0.3, 1*2.38 -0.3, 0.5*2.38],
              [-1*1.38 +0.3, 1*1.38 -0.3, 0.5*1.38]]
           = [[-2.08, 2.08, 1.19], [-1.08, 1.08, 0.69]]
    """
    base_global_step = 100
    _create_checkpoint((
        ([[.6, .5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),
    ), base_global_step, self._model_dir)

    for mode in [model_fn.ModeKeys.TRAIN,
                 model_fn.ModeKeys.EVAL,
                 model_fn.ModeKeys.PREDICT]:
      self._test_logits(
          mode,
          hidden_units=(2, 2),
          logits_dimension=3,
          inputs=[[10.], [5.]],
          expected_logits=[[-2.08, 2.08, 1.19], [-1.08, 1.08, .69]])

  def test_multi_dim_input_one_dim_logits(self):
    """Tests multi-dimensional inputs and one-dimensional logits.

    input_layer = [[10, 8]]
    hidden_layer_0 = [[relu(0.6*10 -0.6*8 +0.1), relu(0.5*10 -0.5*8 -0.1)]]
                   = [[1.3, 0.9]]
    hidden_layer_1 = [[relu(1*1.3 -0.8*0.9 + 0.2), relu(0.8*1.3 -1*0.9 -0.2)]]
                   = [[0.78, relu(-0.06)]] = [[0.78, 0]]
    logits = [[-1*0.78 +1*0 +0.3]] = [[-0.48]]
    """
    base_global_step = 100
    _create_checkpoint((
        ([[.6, .5], [-.6, -.5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1.], [1.]], [.3]),
    ), base_global_step, self._model_dir)

    for mode in [model_fn.ModeKeys.TRAIN,
                 model_fn.ModeKeys.EVAL,
                 model_fn.ModeKeys.PREDICT]:
      self._test_logits(
          mode,
          hidden_units=(2, 2),
          logits_dimension=1,
          inputs=[[10., 8.]],
          expected_logits=[[-0.48]])

  def test_multi_dim_input_multi_dim_logits(self):
    """Tests multi-dimensional inputs and multi-dimensional logits.

    input_layer = [[10, 8]]
    hidden_layer_0 = [[relu(0.6*10 -0.6*8 +0.1), relu(0.5*10 -0.5*8 -0.1)]]
                   = [[1.3, 0.9]]
    hidden_layer_1 = [[relu(1*1.3 -0.8*0.9 + 0.2), relu(0.8*1.3 -1*0.9 -0.2)]]
                   = [[0.78, relu(-0.06)]] = [[0.78, 0]]
    logits = [[-1*0.78 + 0.3, 1*0.78 -0.3, 0.5*0.78]] = [[-0.48, 0.48, 0.39]]
    """
    base_global_step = 100
    _create_checkpoint((
        ([[.6, .5], [-.6, -.5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),
    ), base_global_step, self._model_dir)

    for mode in [model_fn.ModeKeys.TRAIN,
                 model_fn.ModeKeys.EVAL,
                 model_fn.ModeKeys.PREDICT]:
      self._test_logits(
          mode,
          hidden_units=(2, 2),
          logits_dimension=3,
          inputs=[[10., 8.]],
          expected_logits=[[-0.48, 0.48, 0.39]])

  def test_multi_feature_column_multi_dim_logits(self):
    """Tests multiple feature columns and multi-dimensional logits.

    All numbers are the same as test_multi_dim_input_multi_dim_logits. The only
    difference is that the input consists of two 1D feature columns, instead of
    one 2D feature column.
    """
    base_global_step = 100
    _create_checkpoint((
        ([[.6, .5], [-.6, -.5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),
    ), base_global_step, self._model_dir)
    hidden_units = (2, 2)
    logits_dimension = 3
    inputs = ([[10.]], [[8.]])
    expected_logits = [[-0.48, 0.48, 0.39]]

    for mode in [model_fn.ModeKeys.TRAIN,
                 model_fn.ModeKeys.EVAL,
                 model_fn.ModeKeys.PREDICT]:
      with ops.Graph().as_default():
        training_util.create_global_step()
        head = _mock_head(
            self,
            hidden_units=hidden_units,
            logits_dimension=logits_dimension,
            expected_logits=expected_logits)
        estimator_spec = dnn._dnn_model_fn(
            features={'age': constant_op.constant(inputs[0]),
                      'height': constant_op.constant(inputs[1])},
            labels=constant_op.constant([[1]]),
            mode=mode,
            head=head,
            hidden_units=hidden_units,
            feature_columns=[
                feature_column.numeric_column('age'),
                feature_column.numeric_column('height')],
            optimizer=_mock_optimizer(self, hidden_units))
        with monitored_session.MonitoredTrainingSession(
            checkpoint_dir=self._model_dir) as sess:
          if mode == model_fn.ModeKeys.TRAIN:
            sess.run(estimator_spec.train_op)
          elif mode == model_fn.ModeKeys.EVAL:
            sess.run(estimator_spec.loss)
          elif mode == model_fn.ModeKeys.PREDICT:
            sess.run(estimator_spec.predictions)
          else:
            self.fail('Invalid mode: {}'.format(mode))


class DNNRegressorEvaluateTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def test_one_dim(self):
    """Asserts evaluation metrics for one-dimensional input and logits."""
    # Create checkpoint: num_inputs=1, hidden_units=(2, 2), num_outputs=1.
    global_step = 100
    _create_checkpoint((
        ([[.6, .5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1.], [1.]], [.3]),
    ), global_step, self._model_dir)

    # Create DNNRegressor and evaluate.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=[feature_column.numeric_column('age')],
        model_dir=self._model_dir)
    def _input_fn():
      return {'age': [[10.]]}, [[1.]]
    # Uses identical numbers as DNNModelTest.test_one_dim_logits.
    # See that test for calculation of logits.
    # logits = [[-2.08]] => predictions = [-2.08].
    # loss = (1+2.08)^2 = 9.4864
    expected_loss = 9.4864
    self.assertAllClose({
        metric_keys.MetricKeys.LOSS: expected_loss,
        metric_keys.MetricKeys.LOSS_MEAN: expected_loss,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_regressor.evaluate(input_fn=_input_fn, steps=1))

  def test_multi_dim(self):
    """Asserts evaluation metrics for multi-dimensional input and logits."""
    # Create checkpoint: num_inputs=2, hidden_units=(2, 2), num_outputs=3.
    global_step = 100
    _create_checkpoint((
        ([[.6, .5], [-.6, -.5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),
    ), global_step, self._model_dir)
    label_dimension = 3

    # Create DNNRegressor and evaluate.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=[feature_column.numeric_column('age', shape=[2])],
        label_dimension=label_dimension,
        model_dir=self._model_dir)
    def _input_fn():
      return {'age': [[10., 8.]]}, [[1., -1., 0.5]]
    # Uses identical numbers as
    # DNNModelFnTest.test_multi_dim_input_multi_dim_logits.
    # See that test for calculation of logits.
    # logits = [[-0.48, 0.48, 0.39]]
    # loss = (1+0.48)^2 + (-1-0.48)^2 + (0.5-0.39)^2 = 4.3929
    expected_loss = 4.3929
    self.assertAllClose({
        metric_keys.MetricKeys.LOSS: expected_loss,
        metric_keys.MetricKeys.LOSS_MEAN: expected_loss / label_dimension,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_regressor.evaluate(input_fn=_input_fn, steps=1))


class DNNClassifierEvaluateTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def test_one_dim(self):
    """Asserts evaluation metrics for one-dimensional input and logits."""
    global_step = 100
    _create_checkpoint((
        ([[.6, .5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1.], [1.]], [.3]),
    ), global_step, self._model_dir)

    dnn_classifier = dnn.DNNClassifier(
        hidden_units=(2, 2),
        feature_columns=[feature_column.numeric_column('age')],
        model_dir=self._model_dir)
    def _input_fn():
      # batch_size = 2, one false label, and one true.
      return {'age': [[10.], [10.]]}, [[1], [0]]
    # Uses identical numbers as DNNModelTest.test_one_dim_logits.
    # See that test for calculation of logits.
    # logits = [[-2.08], [-2.08]] =>
    # logistic = 1/(1 + exp(-logits)) = [[0.11105597], [0.11105597]]
    # loss = -1. * log(0.111) -1. * log(0.889) = 2.31544200
    expected_loss = 2.31544200
    self.assertAllClose({
        metric_keys.MetricKeys.LOSS: expected_loss,
        metric_keys.MetricKeys.LOSS_MEAN: expected_loss / 2.,
        metric_keys.MetricKeys.ACCURACY: 0.5,
        metric_keys.MetricKeys.PREDICTION_MEAN: 0.11105597,
        metric_keys.MetricKeys.LABEL_MEAN: 0.5,
        metric_keys.MetricKeys.ACCURACY_BASELINE: 0.5,
        # There is no good way to calculate AUC for only two data points. But
        # that is what the algorithm returns.
        metric_keys.MetricKeys.AUC: 0.5,
        metric_keys.MetricKeys.AUC_PR: 0.75,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_classifier.evaluate(input_fn=_input_fn, steps=1))

  def test_multi_dim(self):
    """Asserts evaluation metrics for multi-dimensional input and logits."""
    global_step = 100
    _create_checkpoint((
        ([[.6, .5], [-.6, -.5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),
    ), global_step, self._model_dir)
    n_classes = 3

    dnn_classifier = dnn.DNNClassifier(
        hidden_units=(2, 2),
        feature_columns=[feature_column.numeric_column('age', shape=[2])],
        n_classes=n_classes,
        model_dir=self._model_dir)
    def _input_fn():
      # batch_size = 2, one false label, and one true.
      return {'age': [[10., 8.], [10., 8.]]}, [[1], [0]]
    # Uses identical numbers as
    # DNNModelFnTest.test_multi_dim_input_multi_dim_logits.
    # See that test for calculation of logits.
    # logits = [[-0.48, 0.48, 0.39], [-0.48, 0.48, 0.39]]
    # probabilities = exp(logits)/sum(exp(logits))
    #               = [[0.16670536, 0.43538380, 0.39791084],
    #                  [0.16670536, 0.43538380, 0.39791084]]
    # loss = -log(0.43538380) - log(0.16670536)
    expected_loss = 2.62305466
    self.assertAllClose({
        metric_keys.MetricKeys.LOSS: expected_loss,
        metric_keys.MetricKeys.LOSS_MEAN: expected_loss / 2,
        metric_keys.MetricKeys.ACCURACY: 0.5,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_classifier.evaluate(input_fn=_input_fn, steps=1))


class DNNRegressorPredictTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def test_one_dim(self):
    """Asserts predictions for one-dimensional input and logits."""
    # Create checkpoint: num_inputs=1, hidden_units=(2, 2), num_outputs=1.
    _create_checkpoint((
        ([[.6, .5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1.], [1.]], [.3]),
    ), global_step=0, model_dir=self._model_dir)

    # Create DNNRegressor and predict.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=(feature_column.numeric_column('x'),),
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        x={'x': np.array([[10.]])}, batch_size=1, shuffle=False)
    # Uses identical numbers as DNNModelTest.test_one_dim_logits.
    # See that test for calculation of logits.
    # logits = [[-2.08]] => predictions = [-2.08].
    self.assertAllClose({
        prediction_keys.PredictionKeys.PREDICTIONS: [-2.08],
    }, next(dnn_regressor.predict(input_fn=input_fn)))

  def test_multi_dim(self):
    """Asserts predictions for multi-dimensional input and logits."""
    # Create checkpoint: num_inputs=2, hidden_units=(2, 2), num_outputs=3.
    _create_checkpoint((
        ([[.6, .5], [-.6, -.5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),
    ), 100, self._model_dir)

    # Create DNNRegressor and predict.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=(feature_column.numeric_column('x', shape=(2,)),),
        label_dimension=3,
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        # Inputs shape is (batch_size, num_inputs).
        x={'x': np.array([[10., 8.]])},
        batch_size=1,
        shuffle=False)
    # Uses identical numbers as
    # DNNModelFnTest.test_multi_dim_input_multi_dim_logits.
    # See that test for calculation of logits.
    # logits = [[-0.48, 0.48, 0.39]] => predictions = [-0.48, 0.48, 0.39]
    self.assertAllClose({
        prediction_keys.PredictionKeys.PREDICTIONS: [-0.48, 0.48, 0.39],
    }, next(dnn_regressor.predict(input_fn=input_fn)))


class DNNClassifierPredictTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def test_one_dim(self):
    """Asserts predictions for one-dimensional input and logits."""
    _create_checkpoint((
        ([[.6, .5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1.], [1.]], [.3]),
    ), global_step=0, model_dir=self._model_dir)

    dnn_classifier = dnn.DNNClassifier(
        hidden_units=(2, 2),
        feature_columns=(feature_column.numeric_column('x'),),
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        x={'x': np.array([[10.]])}, batch_size=1, shuffle=False)
    # Uses identical numbers as DNNModelTest.test_one_dim_logits.
    # See that test for calculation of logits.
    # logits = [-2.08] =>
    # logistic = exp(-2.08)/(1 + exp(-2.08)) = 0.11105597
    # probabilities = [1-logistic, logistic] = [0.88894403, 0.11105597]
    # class_ids = argmax(probabilities) = [0]
    self.assertAllClose({
        prediction_keys.PredictionKeys.LOGITS: [-2.08],
        prediction_keys.PredictionKeys.LOGISTIC: [0.11105597],
        prediction_keys.PredictionKeys.PROBABILITIES: [0.88894403, 0.11105597],
        prediction_keys.PredictionKeys.CLASS_IDS: [0],
    }, next(dnn_classifier.predict(input_fn=input_fn)))

  def test_multi_dim(self):
    """Asserts predictions for multi-dimensional input and logits."""
    _create_checkpoint((
        ([[.6, .5], [-.6, -.5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),
    ), global_step=0, model_dir=self._model_dir)

    dnn_classifier = dnn.DNNClassifier(
        hidden_units=(2, 2),
        feature_columns=(feature_column.numeric_column('x', shape=(2,)),),
        n_classes=3,
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        # Inputs shape is (batch_size, num_inputs).
        x={'x': np.array([[10., 8.]])},
        batch_size=1,
        shuffle=False)
    # Uses identical numbers as
    # DNNModelFnTest.test_multi_dim_input_multi_dim_logits.
    # See that test for calculation of logits.
    # logits = [-0.48, 0.48, 0.39] =>
    # probabilities[i] = exp(logits[i]) / sum_j exp(logits[j]) =>
    # probabilities = [0.16670536, 0.43538380, 0.39791084]
    # class_ids = argmax(probabilities) = [1]
    predictions = next(dnn_classifier.predict(input_fn=input_fn))
    self.assertItemsEqual(
        [prediction_keys.PredictionKeys.LOGITS,
         prediction_keys.PredictionKeys.PROBABILITIES,
         prediction_keys.PredictionKeys.CLASS_IDS,
         prediction_keys.PredictionKeys.CLASSES],
        six.iterkeys(predictions))
    self.assertAllClose(
        [-0.48, 0.48, 0.39], predictions[prediction_keys.PredictionKeys.LOGITS])
    self.assertAllClose(
        [0.16670536, 0.43538380, 0.39791084],
        predictions[prediction_keys.PredictionKeys.PROBABILITIES])
    self.assertAllEqual(
        [1], predictions[prediction_keys.PredictionKeys.CLASS_IDS])
    self.assertAllEqual(
        [b'1'], predictions[prediction_keys.PredictionKeys.CLASSES])


def _queue_parsed_features(feature_map):
  tensors_to_enqueue = []
  keys = []
  for key, tensor in six.iteritems(feature_map):
    keys.append(key)
    tensors_to_enqueue.append(tensor)
  queue_dtypes = [x.dtype for x in tensors_to_enqueue]
  input_queue = data_flow_ops.FIFOQueue(capacity=100, dtypes=queue_dtypes)
  queue_runner.add_queue_runner(
      queue_runner.QueueRunner(
          input_queue,
          [input_queue.enqueue(tensors_to_enqueue)]))
  dequeued_tensors = input_queue.dequeue()
  return {keys[i]: dequeued_tensors[i] for i in range(len(dequeued_tensors))}


class DNNRegressorIntegrationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(
      self, train_input_fn, eval_input_fn, predict_input_fn, input_dimension,
      label_dimension, batch_size):
    feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))]
    est = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=feature_columns,
        label_dimension=label_dimension,
        model_dir=self._model_dir)

    # TRAIN
    num_steps = 10
    est.train(train_input_fn, steps=num_steps)

    # EVALUTE
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(num_steps, scores[ops.GraphKeys.GLOBAL_STEP])
    self.assertIn('loss', six.iterkeys(scores))

    # PREDICT
    predictions = np.array([
        x[prediction_keys.PredictionKeys.PREDICTIONS]
        for x in est.predict(predict_input_fn)
    ])
    self.assertAllEqual((batch_size, label_dimension), predictions.shape)

    # EXPORT
    feature_spec = feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir = est.export_savedmodel(tempfile.mkdtemp(),
                                       serving_input_receiver_fn)
    self.assertTrue(gfile.Exists(export_dir))

  def test_numpy_input_fn(self):
    """Tests complete flow with numpy_input_fn."""
    label_dimension = 2
    batch_size = 10
    data = np.linspace(0., 2., batch_size * label_dimension, dtype=np.float32)
    data = data.reshape(batch_size, label_dimension)
    # learn y = x
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        batch_size=batch_size,
        shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=label_dimension,
        label_dimension=label_dimension,
        batch_size=batch_size)

  def test_pandas_input_fn(self):
    """Tests complete flow with pandas_input_fn."""
    if not HAS_PANDAS:
      return
    label_dimension = 1
    batch_size = 10
    data = np.linspace(0., 2., batch_size, dtype=np.float32)
    x = pd.DataFrame({'x': data})
    y = pd.Series(data)
    train_input_fn = pandas_io.pandas_input_fn(
        x=x,
        y=y,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = pandas_io.pandas_input_fn(
        x=x,
        y=y,
        batch_size=batch_size,
        shuffle=False)
    predict_input_fn = pandas_io.pandas_input_fn(
        x=x,
        batch_size=batch_size,
        shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=label_dimension,
        label_dimension=label_dimension,
        batch_size=batch_size)

  def test_input_fn_from_parse_example(self):
    """Tests complete flow with input_fn constructed from parse_example."""
    label_dimension = 2
    batch_size = 10
    data = np.linspace(0., 2., batch_size * label_dimension, dtype=np.float32)
    data = data.reshape(batch_size, label_dimension)

    serialized_examples = []
    for datum in data:
      example = example_pb2.Example(features=feature_pb2.Features(
          feature={
              'x': feature_pb2.Feature(
                  float_list=feature_pb2.FloatList(value=datum)),
              'y': feature_pb2.Feature(
                  float_list=feature_pb2.FloatList(value=datum)),
          }))
      serialized_examples.append(example.SerializeToString())

    feature_spec = {
        'x': parsing_ops.FixedLenFeature([label_dimension], dtypes.float32),
        'y': parsing_ops.FixedLenFeature([label_dimension], dtypes.float32),
    }
    def _train_input_fn():
      feature_map = parsing_ops.parse_example(serialized_examples, feature_spec)
      features = _queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels
    def _eval_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      features = _queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels
    def _predict_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      features = _queue_parsed_features(feature_map)
      features.pop('y')
      return features, None

    self._test_complete_flow(
        train_input_fn=_train_input_fn,
        eval_input_fn=_eval_input_fn,
        predict_input_fn=_predict_input_fn,
        input_dimension=label_dimension,
        label_dimension=label_dimension,
        batch_size=batch_size)


class DNNClassifierIntegrationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(
      self, train_input_fn, eval_input_fn, predict_input_fn, input_dimension,
      n_classes, batch_size):
    feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))]
    est = dnn.DNNClassifier(
        hidden_units=(2, 2),
        feature_columns=feature_columns,
        n_classes=n_classes,
        model_dir=self._model_dir)

    # TRAIN
    num_steps = 10
    est.train(train_input_fn, steps=num_steps)

    # EVALUTE
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(num_steps, scores[ops.GraphKeys.GLOBAL_STEP])
    self.assertIn('loss', six.iterkeys(scores))

    # PREDICT
    predicted_proba = np.array([
        x[prediction_keys.PredictionKeys.PROBABILITIES]
        for x in est.predict(predict_input_fn)
    ])
    self.assertAllEqual((batch_size, n_classes), predicted_proba.shape)

    # EXPORT
    feature_spec = feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir = est.export_savedmodel(tempfile.mkdtemp(),
                                       serving_input_receiver_fn)
    self.assertTrue(gfile.Exists(export_dir))

  def test_numpy_input_fn(self):
    """Tests complete flow with numpy_input_fn."""
    n_classes = 2
    input_dimension = 2
    batch_size = 10
    data = np.linspace(0., 2., batch_size * input_dimension, dtype=np.float32)
    x_data = data.reshape(batch_size, input_dimension)
    y_data = np.reshape(data[:batch_size], (batch_size, 1))
    # learn y = x
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': x_data},
        y=y_data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': x_data},
        y=y_data,
        batch_size=batch_size,
        shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': x_data},
        batch_size=batch_size,
        shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=input_dimension,
        n_classes=n_classes,
        batch_size=batch_size)

  def test_pandas_input_fn(self):
    """Tests complete flow with pandas_input_fn."""
    if not HAS_PANDAS:
      return
    input_dimension = 1
    n_classes = 2
    batch_size = 10
    data = np.linspace(0., 2., batch_size, dtype=np.float32)
    x = pd.DataFrame({'x': data})
    y = pd.Series(data)
    train_input_fn = pandas_io.pandas_input_fn(
        x=x,
        y=y,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = pandas_io.pandas_input_fn(
        x=x,
        y=y,
        batch_size=batch_size,
        shuffle=False)
    predict_input_fn = pandas_io.pandas_input_fn(
        x=x,
        batch_size=batch_size,
        shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=input_dimension,
        n_classes=n_classes,
        batch_size=batch_size)

  def test_input_fn_from_parse_example(self):
    """Tests complete flow with input_fn constructed from parse_example."""
    input_dimension = 2
    n_classes = 2
    batch_size = 10
    data = np.linspace(0., 2., batch_size * input_dimension, dtype=np.float32)
    data = data.reshape(batch_size, input_dimension)

    serialized_examples = []
    for datum in data:
      example = example_pb2.Example(features=feature_pb2.Features(
          feature={
              'x': feature_pb2.Feature(
                  float_list=feature_pb2.FloatList(value=datum)),
              'y': feature_pb2.Feature(
                  float_list=feature_pb2.FloatList(value=datum[:1])),
          }))
      serialized_examples.append(example.SerializeToString())

    feature_spec = {
        'x': parsing_ops.FixedLenFeature([input_dimension], dtypes.float32),
        'y': parsing_ops.FixedLenFeature([1], dtypes.float32),
    }
    def _train_input_fn():
      feature_map = parsing_ops.parse_example(serialized_examples, feature_spec)
      features = _queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels
    def _eval_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      features = _queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels
    def _predict_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      features = _queue_parsed_features(feature_map)
      features.pop('y')
      return features, None

    self._test_complete_flow(
        train_input_fn=_train_input_fn,
        eval_input_fn=_eval_input_fn,
        predict_input_fn=_predict_input_fn,
        input_dimension=input_dimension,
        n_classes=n_classes,
        batch_size=batch_size)


def _full_var_name(var_name):
  return '%s/part_0:0' % var_name


def _assert_close(
    expected, actual, rtol=1e-04, message='', name='assert_close'):
  with ops.name_scope(name, 'assert_close', (expected, actual, rtol)) as scope:
    expected = ops.convert_to_tensor(expected, name='expected')
    actual = ops.convert_to_tensor(actual, name='actual')
    rdiff = math_ops.abs((expected - actual) / expected, 'diff')
    rtol = ops.convert_to_tensor(rtol, name='rtol')
    return check_ops.assert_less(
        rdiff,
        rtol,
        data=(
            message,
            'Condition expected =~ actual did not hold element-wise:'
            'expected = ', expected,
            'actual = ', actual,
            'rdiff = ', rdiff,
            'rtol = ', rtol,
        ),
        summarize=expected.get_shape().num_elements(),
        name=scope)


class _SummaryHook(session_run_hook.SessionRunHook):
  """Saves summaries every N steps."""

  def __init__(self):
    self._summaries = []

  def begin(self):
    self._summary_op = summary_lib.merge_all()

  def before_run(self, run_context):
    return session_run_hook.SessionRunArgs({'summary': self._summary_op})

  def after_run(self, run_context, run_values):
    s = summary_pb2.Summary()
    s.ParseFromString(run_values.results['summary'])
    self._summaries.append(s)

  def summaries(self):
    return tuple(self._summaries)


def _assert_checkpoint(
    testcase, global_step, input_units, hidden_units, output_units, model_dir):
  """Asserts checkpoint contains expected variables with proper shapes.

  Args:
    testcase: A TestCase instance.
    global_step: Expected global step value.
    input_units: The dimension of input layer.
    hidden_units: Iterable of integer sizes for the hidden layers.
    output_units: The dimension of output layer (logits).
    model_dir: The model directory.
  """
  shapes = {
      name: shape
      for (name, shape) in checkpoint_utils.list_variables(model_dir)
  }

  # Global step.
  testcase.assertEqual([], shapes[ops.GraphKeys.GLOBAL_STEP])
  testcase.assertEqual(
      global_step,
      checkpoint_utils.load_variable(
          model_dir, ops.GraphKeys.GLOBAL_STEP))

  # Hidden layer weights.
  prev_layer_units = input_units
  for i in range(len(hidden_units)):
    layer_units = hidden_units[i]
    testcase.assertAllEqual((prev_layer_units, layer_units),
                            shapes[_HIDDEN_WEIGHTS_NAME_PATTERN % i])
    testcase.assertAllEqual((layer_units,),
                            shapes[_HIDDEN_BIASES_NAME_PATTERN % i])
    prev_layer_units = layer_units

  # Output layer weights.
  testcase.assertAllEqual((prev_layer_units, output_units),
                          shapes[_LOGITS_WEIGHTS_NAME])
  testcase.assertAllEqual((output_units,), shapes[_LOGITS_BIASES_NAME])


def _mock_optimizer(testcase, hidden_units, expected_loss=None):
  """Creates a mock optimizer to test the train method.

  Args:
    testcase: A TestCase instance.
    hidden_units: Iterable of integer sizes for the hidden layers.
    expected_loss: If given, will assert the loss value.

  Returns:
    A mock Optimizer.
  """
  hidden_weights_names = [
      (_HIDDEN_WEIGHTS_NAME_PATTERN + '/part_0:0') % i
      for i in range(len(hidden_units))]
  hidden_biases_names = [
      (_HIDDEN_BIASES_NAME_PATTERN + '/part_0:0') % i
      for i in range(len(hidden_units))]
  expected_var_names = (
      hidden_weights_names + hidden_biases_names +
      [_LOGITS_WEIGHTS_NAME + '/part_0:0', _LOGITS_BIASES_NAME + '/part_0:0'])

  def _minimize(loss, global_step):
    trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    testcase.assertItemsEqual(
        expected_var_names,
        [var.name for var in trainable_vars])

    # Verify loss. We can't check the value directly, so we add an assert op.
    testcase.assertEquals(0, loss.shape.ndims)
    if expected_loss is None:
      return state_ops.assign_add(global_step, 1).op
    assert_loss = _assert_close(
        math_ops.to_float(expected_loss, name='expected'), loss,
        name='assert_loss')
    with ops.control_dependencies((assert_loss,)):
      return state_ops.assign_add(global_step, 1).op

  mock_optimizer = test.mock.NonCallableMagicMock(
      spec=optimizer.Optimizer,
      wraps=optimizer.Optimizer(use_locking=False, name='my_optimizer'))
  mock_optimizer.minimize = test.mock.MagicMock(wraps=_minimize)

  return mock_optimizer


def _assert_simple_summary(testcase, expected_values, actual_summary):
  """Assert summary the specified simple values.

  Args:
    testcase: A TestCase instance.
    expected_values: Dict of expected tags and simple values.
    actual_summary: `summary_pb2.Summary`.
  """
  testcase.assertAllClose(expected_values, {
      v.tag: v.simple_value
      for v in actual_summary.value if (v.tag in expected_values)
  })


class DNNRegressorTrainTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def test_from_scratch_with_default_optimizer(self):
    hidden_units = (2, 2)
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=hidden_units,
        feature_columns=(feature_column.numeric_column('age'),),
        model_dir=self._model_dir)

    # Train for a few steps, then validate final checkpoint.
    num_steps = 5
    dnn_regressor.train(
        input_fn=lambda: ({'age': ((1,),)}, ((10,),)), steps=num_steps)
    _assert_checkpoint(
        self, num_steps, input_units=1, hidden_units=hidden_units,
        output_units=1, model_dir=self._model_dir)

  def test_from_scratch(self):
    hidden_units = (2, 2)
    mock_optimizer = _mock_optimizer(self, hidden_units=hidden_units)
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=hidden_units,
        feature_columns=(feature_column.numeric_column('age'),),
        optimizer=mock_optimizer,
        model_dir=self._model_dir)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, then validate optimizer, summaries, and
    # checkpoint.
    num_steps = 5
    summary_hook = _SummaryHook()
    dnn_regressor.train(
        input_fn=lambda: ({'age': ((1,),)}, ((5.,),)), steps=num_steps,
        hooks=(summary_hook,))
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    _assert_checkpoint(
        self, num_steps, input_units=1, hidden_units=hidden_units,
        output_units=1, model_dir=self._model_dir)
    summaries = summary_hook.summaries()
    self.assertEqual(num_steps, len(summaries))
    for summary in summaries:
      summary_keys = [v.tag for v in summary.value]
      self.assertIn(metric_keys.MetricKeys.LOSS, summary_keys)
      self.assertIn(metric_keys.MetricKeys.LOSS_MEAN, summary_keys)

  def test_one_dim(self):
    """Asserts train loss for one-dimensional input and logits."""
    base_global_step = 100
    hidden_units = (2, 2)
    _create_checkpoint((
        ([[.6, .5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1.], [1.]], [.3]),
    ), base_global_step, self._model_dir)

    # Uses identical numbers as DNNModelFnTest.test_one_dim_logits.
    # See that test for calculation of logits.
    # logits = [-2.08] => predictions = [-2.08]
    # loss = (1 + 2.08)^2 = 9.4864
    expected_loss = 9.4864
    mock_optimizer = _mock_optimizer(
        self, hidden_units=hidden_units, expected_loss=expected_loss)
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=hidden_units,
        feature_columns=(feature_column.numeric_column('age'),),
        optimizer=mock_optimizer,
        model_dir=self._model_dir)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, then validate optimizer, summaries, and
    # checkpoint.
    num_steps = 5
    summary_hook = _SummaryHook()
    dnn_regressor.train(
        input_fn=lambda: ({'age': [[10.]]}, [[1.]]), steps=num_steps,
        hooks=(summary_hook,))
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    summaries = summary_hook.summaries()
    self.assertEqual(num_steps, len(summaries))
    for summary in summaries:
      _assert_simple_summary(
          self,
          {
              metric_keys.MetricKeys.LOSS_MEAN: expected_loss,
              'dnn/dnn/hiddenlayer_0_fraction_of_zero_values': 0.,
              'dnn/dnn/hiddenlayer_1_fraction_of_zero_values': 0.5,
              'dnn/dnn/logits_fraction_of_zero_values': 0.,
              metric_keys.MetricKeys.LOSS: expected_loss,
          },
          summary)
    _assert_checkpoint(
        self, base_global_step + num_steps, input_units=1,
        hidden_units=hidden_units, output_units=1, model_dir=self._model_dir)

  def test_multi_dim(self):
    """Asserts train loss for multi-dimensional input and logits."""
    base_global_step = 100
    hidden_units = (2, 2)
    _create_checkpoint((
        ([[.6, .5], [-.6, -.5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),
    ), base_global_step, self._model_dir)
    input_dimension = 2
    label_dimension = 3

    # Uses identical numbers as
    # DNNModelFnTest.test_multi_dim_input_multi_dim_logits.
    # See that test for calculation of logits.
    # logits = [[-0.48, 0.48, 0.39]]
    # loss = (1+0.48)^2 + (-1-0.48)^2 + (0.5-0.39)^2 = 4.3929
    expected_loss = 4.3929
    mock_optimizer = _mock_optimizer(
        self, hidden_units=hidden_units, expected_loss=expected_loss)
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=hidden_units,
        feature_columns=[
            feature_column.numeric_column('age', shape=[input_dimension])],
        label_dimension=label_dimension,
        optimizer=mock_optimizer,
        model_dir=self._model_dir)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, then validate optimizer, summaries, and
    # checkpoint.
    num_steps = 5
    summary_hook = _SummaryHook()
    dnn_regressor.train(
        input_fn=lambda: ({'age': [[10., 8.]]}, [[1., -1., 0.5]]),
        steps=num_steps,
        hooks=(summary_hook,))
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    summaries = summary_hook.summaries()
    self.assertEqual(num_steps, len(summaries))
    for summary in summaries:
      _assert_simple_summary(
          self,
          {
              metric_keys.MetricKeys.LOSS_MEAN: expected_loss / label_dimension,
              'dnn/dnn/hiddenlayer_0_fraction_of_zero_values': 0.,
              'dnn/dnn/hiddenlayer_1_fraction_of_zero_values': 0.5,
              'dnn/dnn/logits_fraction_of_zero_values': 0.,
              metric_keys.MetricKeys.LOSS: expected_loss,
          },
          summary)
    _assert_checkpoint(
        self, base_global_step + num_steps, input_units=input_dimension,
        hidden_units=hidden_units, output_units=label_dimension,
        model_dir=self._model_dir)


class DNNClassifierTrainTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def test_from_scratch_with_default_optimizer_binary(self):
    hidden_units = (2, 2)
    dnn_classifier = dnn.DNNClassifier(
        hidden_units=hidden_units,
        feature_columns=(feature_column.numeric_column('age'),),
        model_dir=self._model_dir)

    # Train for a few steps, then validate final checkpoint.
    num_steps = 5
    dnn_classifier.train(
        input_fn=lambda: ({'age': [[10.]]}, [[1]]), steps=num_steps)
    _assert_checkpoint(
        self, num_steps, input_units=1, hidden_units=hidden_units,
        output_units=1, model_dir=self._model_dir)

  def test_from_scratch_with_default_optimizer_multi_class(self):
    hidden_units = (2, 2)
    n_classes = 3
    dnn_classifier = dnn.DNNClassifier(
        hidden_units=hidden_units,
        feature_columns=(feature_column.numeric_column('age'),),
        n_classes=n_classes,
        model_dir=self._model_dir)

    # Train for a few steps, then validate final checkpoint.
    num_steps = 5
    dnn_classifier.train(
        input_fn=lambda: ({'age': [[10.]]}, [[2]]), steps=num_steps)
    _assert_checkpoint(
        self, num_steps, input_units=1, hidden_units=hidden_units,
        output_units=n_classes, model_dir=self._model_dir)

  def test_from_scratch_validate_summary(self):
    hidden_units = (2, 2)
    mock_optimizer = _mock_optimizer(self, hidden_units=hidden_units)
    dnn_classifier = dnn.DNNClassifier(
        hidden_units=hidden_units,
        feature_columns=(feature_column.numeric_column('age'),),
        optimizer=mock_optimizer,
        model_dir=self._model_dir)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, then validate optimizer, summaries, and
    # checkpoint.
    num_steps = 5
    summary_hook = _SummaryHook()
    dnn_classifier.train(
        input_fn=lambda: ({'age': [[10.]]}, [[1]]), steps=num_steps,
        hooks=(summary_hook,))
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    _assert_checkpoint(
        self, num_steps, input_units=1, hidden_units=hidden_units,
        output_units=1, model_dir=self._model_dir)
    summaries = summary_hook.summaries()
    self.assertEqual(num_steps, len(summaries))
    for summary in summaries:
      summary_keys = [v.tag for v in summary.value]
      self.assertIn(metric_keys.MetricKeys.LOSS, summary_keys)
      self.assertIn(metric_keys.MetricKeys.LOSS_MEAN, summary_keys)

  def test_binary_classification(self):
    base_global_step = 100
    hidden_units = (2, 2)
    _create_checkpoint((
        ([[.6, .5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1.], [1.]], [.3]),
    ), base_global_step, self._model_dir)

    # Uses identical numbers as DNNModelFnTest.test_one_dim_logits.
    # See that test for calculation of logits.
    # logits = [-2.08] => probabilities = [0.889, 0.111]
    # loss = -1. * log(0.111) = 2.19772100
    expected_loss = 2.19772100
    mock_optimizer = _mock_optimizer(
        self, hidden_units=hidden_units, expected_loss=expected_loss)
    dnn_classifier = dnn.DNNClassifier(
        hidden_units=hidden_units,
        feature_columns=(feature_column.numeric_column('age'),),
        optimizer=mock_optimizer,
        model_dir=self._model_dir)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, then validate optimizer, summaries, and
    # checkpoint.
    num_steps = 5
    summary_hook = _SummaryHook()
    dnn_classifier.train(
        input_fn=lambda: ({'age': [[10.]]}, [[1]]), steps=num_steps,
        hooks=(summary_hook,))
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    summaries = summary_hook.summaries()
    self.assertEqual(num_steps, len(summaries))
    for summary in summaries:
      _assert_simple_summary(
          self,
          {
              metric_keys.MetricKeys.LOSS_MEAN: expected_loss,
              'dnn/dnn/hiddenlayer_0_fraction_of_zero_values': 0.,
              'dnn/dnn/hiddenlayer_1_fraction_of_zero_values': .5,
              'dnn/dnn/logits_fraction_of_zero_values': 0.,
              metric_keys.MetricKeys.LOSS: expected_loss,
          },
          summary)
    _assert_checkpoint(
        self, base_global_step + num_steps, input_units=1,
        hidden_units=hidden_units, output_units=1, model_dir=self._model_dir)

  def test_multi_class(self):
    n_classes = 3
    base_global_step = 100
    hidden_units = (2, 2)
    _create_checkpoint((
        ([[.6, .5]], [.1, -.1]),
        ([[1., .8], [-.8, -1.]], [.2, -.2]),
        ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),
    ), base_global_step, self._model_dir)

    # Uses identical numbers as DNNModelFnTest.test_multi_dim_logits.
    # See that test for calculation of logits.
    # logits = [-2.08, 2.08, 1.19] => probabilities = [0.0109, 0.7011, 0.2879]
    # loss = -1. * log(0.7011) = 0.35505795
    expected_loss = 0.35505795
    mock_optimizer = _mock_optimizer(
        self, hidden_units=hidden_units, expected_loss=expected_loss)
    dnn_classifier = dnn.DNNClassifier(
        n_classes=n_classes,
        hidden_units=hidden_units,
        feature_columns=(feature_column.numeric_column('age'),),
        optimizer=mock_optimizer,
        model_dir=self._model_dir)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, then validate optimizer, summaries, and
    # checkpoint.
    num_steps = 5
    summary_hook = _SummaryHook()
    dnn_classifier.train(
        input_fn=lambda: ({'age': [[10.]]}, [[1]]), steps=num_steps,
        hooks=(summary_hook,))
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    summaries = summary_hook.summaries()
    self.assertEqual(num_steps, len(summaries))
    for summary in summaries:
      _assert_simple_summary(
          self,
          {
              metric_keys.MetricKeys.LOSS_MEAN: expected_loss,
              'dnn/dnn/hiddenlayer_0_fraction_of_zero_values': 0.,
              'dnn/dnn/hiddenlayer_1_fraction_of_zero_values': .5,
              'dnn/dnn/logits_fraction_of_zero_values': 0.,
              metric_keys.MetricKeys.LOSS: expected_loss,
          },
          summary)
    _assert_checkpoint(
        self, base_global_step + num_steps, input_units=1,
        hidden_units=hidden_units, output_units=n_classes,
        model_dir=self._model_dir)


if __name__ == '__main__':
  test.main()

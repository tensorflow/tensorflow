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
"""Utils to be used in testing DNN estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import numpy as np
import six

from tensorflow.python.client import session as tf_session
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.training import monitored_session
from tensorflow.python.training import optimizer
from tensorflow.python.training import saver
from tensorflow.python.training import training_util

# pylint rules which are disabled by default for test files.
# pylint: disable=invalid-name,protected-access

# Names of variables created by model.
LEARNING_RATE_NAME = 'dnn/regression_head/dnn/learning_rate'
HIDDEN_WEIGHTS_NAME_PATTERN = 'dnn/hiddenlayer_%d/kernel'
HIDDEN_BIASES_NAME_PATTERN = 'dnn/hiddenlayer_%d/bias'
LOGITS_WEIGHTS_NAME = 'dnn/logits/kernel'
LOGITS_BIASES_NAME = 'dnn/logits/bias'


def assert_close(expected, actual, rtol=1e-04, message='', name='assert_close'):
  with ops.name_scope(name, 'assert_close', (expected, actual, rtol)) as scope:
    expected = ops.convert_to_tensor(expected, name='expected')
    actual = ops.convert_to_tensor(actual, name='actual')
    rdiff = math_ops.abs((expected - actual) / expected, 'diff')
    rtol = ops.convert_to_tensor(rtol, name='rtol')
    return check_ops.assert_less(
        rdiff,
        rtol,
        data=(message, 'Condition expected =~ actual did not hold element-wise:'
              'expected = ', expected, 'actual = ', actual, 'rdiff = ', rdiff,
              'rtol = ', rtol,),
        summarize=expected.get_shape().num_elements(),
        name=scope)


def create_checkpoint(weights_and_biases, global_step, model_dir):
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
    model_weights[HIDDEN_WEIGHTS_NAME_PATTERN % i] = weights[i]
    model_weights[HIDDEN_BIASES_NAME_PATTERN % i] = biases[i]

  # Output layer weights.
  model_weights[LOGITS_WEIGHTS_NAME] = weights[-1]
  model_weights[LOGITS_BIASES_NAME] = biases[-1]

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


def mock_head(testcase, hidden_units, logits_dimension, expected_logits):
  """Returns a mock head that validates logits values and variable names."""
  hidden_weights_names = [(HIDDEN_WEIGHTS_NAME_PATTERN + '/part_0:0') % i
                          for i in range(len(hidden_units))]
  hidden_biases_names = [(HIDDEN_BIASES_NAME_PATTERN + '/part_0:0') % i
                         for i in range(len(hidden_units))]
  expected_var_names = (
      hidden_weights_names + hidden_biases_names +
      [LOGITS_WEIGHTS_NAME + '/part_0:0', LOGITS_BIASES_NAME + '/part_0:0'])

  def _create_estimator_spec(features, mode, logits, labels, train_op_fn):
    del features, labels  # Not used.
    trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    testcase.assertItemsEqual(expected_var_names,
                              [var.name for var in trainable_vars])
    loss = constant_op.constant(1.)
    assert_logits = assert_close(
        expected_logits, logits, message='Failed for mode={}. '.format(mode))
    with ops.control_dependencies([assert_logits]):
      if mode == model_fn.ModeKeys.TRAIN:
        return model_fn.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op_fn(loss))
      elif mode == model_fn.ModeKeys.EVAL:
        return model_fn.EstimatorSpec(mode=mode, loss=array_ops.identity(loss))
      elif mode == model_fn.ModeKeys.PREDICT:
        return model_fn.EstimatorSpec(
            mode=mode, predictions={'logits': array_ops.identity(logits)})
      else:
        testcase.fail('Invalid mode: {}'.format(mode))

  head = test.mock.NonCallableMagicMock(spec=head_lib._Head)
  head.logits_dimension = logits_dimension
  head.create_estimator_spec = test.mock.MagicMock(wraps=_create_estimator_spec)

  return head


def mock_optimizer(testcase, hidden_units, expected_loss=None):
  """Creates a mock optimizer to test the train method.

  Args:
    testcase: A TestCase instance.
    hidden_units: Iterable of integer sizes for the hidden layers.
    expected_loss: If given, will assert the loss value.

  Returns:
    A mock Optimizer.
  """
  hidden_weights_names = [(HIDDEN_WEIGHTS_NAME_PATTERN + '/part_0:0') % i
                          for i in range(len(hidden_units))]
  hidden_biases_names = [(HIDDEN_BIASES_NAME_PATTERN + '/part_0:0') % i
                         for i in range(len(hidden_units))]
  expected_var_names = (
      hidden_weights_names + hidden_biases_names +
      [LOGITS_WEIGHTS_NAME + '/part_0:0', LOGITS_BIASES_NAME + '/part_0:0'])

  def _minimize(loss, global_step=None, var_list=None):
    """Mock of optimizer.minimize."""
    trainable_vars = var_list or ops.get_collection(
        ops.GraphKeys.TRAINABLE_VARIABLES)
    testcase.assertItemsEqual(expected_var_names,
                              [var.name for var in trainable_vars])

    # Verify loss. We can't check the value directly, so we add an assert op.
    testcase.assertEquals(0, loss.shape.ndims)
    if expected_loss is None:
      if global_step is not None:
        return state_ops.assign_add(global_step, 1).op
      return control_flow_ops.no_op()
    assert_loss = assert_close(
        math_ops.to_float(expected_loss, name='expected'),
        loss,
        name='assert_loss')
    with ops.control_dependencies((assert_loss,)):
      if global_step is not None:
        return state_ops.assign_add(global_step, 1).op
      return control_flow_ops.no_op()

  optimizer_mock = test.mock.NonCallableMagicMock(
      spec=optimizer.Optimizer,
      wraps=optimizer.Optimizer(use_locking=False, name='my_optimizer'))
  optimizer_mock.minimize = test.mock.MagicMock(wraps=_minimize)

  return optimizer_mock


class BaseDNNModelFnTest(object):
  """Tests that _dnn_model_fn passes expected logits to mock head."""

  def __init__(self, dnn_model_fn):
    self._dnn_model_fn = dnn_model_fn

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _test_logits(self, mode, hidden_units, logits_dimension, inputs,
                   expected_logits):
    """Tests that the expected logits are passed to mock head."""
    with ops.Graph().as_default():
      training_util.create_global_step()
      head = mock_head(
          self,
          hidden_units=hidden_units,
          logits_dimension=logits_dimension,
          expected_logits=expected_logits)
      estimator_spec = self._dnn_model_fn(
          features={'age': constant_op.constant(inputs)},
          labels=constant_op.constant([[1]]),
          mode=mode,
          head=head,
          hidden_units=hidden_units,
          feature_columns=[
              feature_column.numeric_column(
                  'age', shape=np.array(inputs).shape[1:])
          ],
          optimizer=mock_optimizer(self, hidden_units))
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
    create_checkpoint(
        (([[.6, .5]], [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
         ([[-1.], [1.]], [.3]),), base_global_step, self._model_dir)

    for mode in [
        model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
        model_fn.ModeKeys.PREDICT
    ]:
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
    create_checkpoint((([[.6, .5]], [.1, -.1]), ([[1., .8], [-.8, -1.]],
                                                 [.2, -.2]),
                       ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),),
                      base_global_step, self._model_dir)

    for mode in [
        model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
        model_fn.ModeKeys.PREDICT
    ]:
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
    create_checkpoint((([[.6, .5]], [.1, -.1]), ([[1., .8], [-.8, -1.]],
                                                 [.2, -.2]),
                       ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),),
                      base_global_step, self._model_dir)

    for mode in [
        model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
        model_fn.ModeKeys.PREDICT
    ]:
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
    create_checkpoint((([[.6, .5], [-.6, -.5]],
                        [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
                       ([[-1.], [1.]], [.3]),), base_global_step,
                      self._model_dir)

    for mode in [
        model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
        model_fn.ModeKeys.PREDICT
    ]:
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
    create_checkpoint((([[.6, .5], [-.6, -.5]],
                        [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
                       ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),),
                      base_global_step, self._model_dir)

    for mode in [
        model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
        model_fn.ModeKeys.PREDICT
    ]:
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
    create_checkpoint((([[.6, .5], [-.6, -.5]],
                        [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
                       ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),),
                      base_global_step, self._model_dir)
    hidden_units = (2, 2)
    logits_dimension = 3
    inputs = ([[10.]], [[8.]])
    expected_logits = [[-0.48, 0.48, 0.39]]

    for mode in [
        model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
        model_fn.ModeKeys.PREDICT
    ]:
      with ops.Graph().as_default():
        training_util.create_global_step()
        head = mock_head(
            self,
            hidden_units=hidden_units,
            logits_dimension=logits_dimension,
            expected_logits=expected_logits)
        estimator_spec = self._dnn_model_fn(
            features={
                'age': constant_op.constant(inputs[0]),
                'height': constant_op.constant(inputs[1])
            },
            labels=constant_op.constant([[1]]),
            mode=mode,
            head=head,
            hidden_units=hidden_units,
            feature_columns=[
                feature_column.numeric_column('age'),
                feature_column.numeric_column('height')
            ],
            optimizer=mock_optimizer(self, hidden_units))
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


# pylint: enable=invalid-name,protected-access

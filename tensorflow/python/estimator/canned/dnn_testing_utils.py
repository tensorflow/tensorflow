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

from tensorflow.core.framework import summary_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.summary import summary as summary_lib
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import monitored_session
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.training import saver
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

# pylint rules which are disabled by default for test files.
# pylint: disable=invalid-name,protected-access,missing-docstring

# Names of variables created by model.
LEARNING_RATE_NAME = 'dnn/regression_head/dnn/learning_rate'
HIDDEN_WEIGHTS_NAME_PATTERN = 'dnn/hiddenlayer_%d/kernel'
HIDDEN_BIASES_NAME_PATTERN = 'dnn/hiddenlayer_%d/bias'
BATCH_NORM_BETA_NAME_PATTERN = 'dnn/hiddenlayer_%d/batchnorm_%d/beta'
BATCH_NORM_GAMMA_NAME_PATTERN = 'dnn/hiddenlayer_%d/batchnorm_%d/gamma'
BATCH_NORM_MEAN_NAME_PATTERN = 'dnn/hiddenlayer_%d/batchnorm_%d/moving_mean'
BATCH_NORM_VARIANCE_NAME_PATTERN = (
    'dnn/hiddenlayer_%d/batchnorm_%d/moving_variance')
LOGITS_WEIGHTS_NAME = 'dnn/logits/kernel'
LOGITS_BIASES_NAME = 'dnn/logits/bias'
OCCUPATION_EMBEDDING_NAME = ('dnn/input_from_feature_columns/input_layer/'
                             'occupation_embedding/embedding_weights')
CITY_EMBEDDING_NAME = ('dnn/input_from_feature_columns/input_layer/'
                       'city_embedding/embedding_weights')


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


def create_checkpoint(weights_and_biases,
                      global_step,
                      model_dir,
                      batch_norm_vars=None):
  """Create checkpoint file with provided model weights.

  Args:
    weights_and_biases: Iterable of tuples of weight and bias values.
    global_step: Initial global step to save in checkpoint.
    model_dir: Directory into which checkpoint is saved.
    batch_norm_vars: Variables used for batch normalization.
  """
  weights, biases = zip(*weights_and_biases)
  if batch_norm_vars:
    assert len(batch_norm_vars) == len(weights_and_biases) - 1
    (bn_betas, bn_gammas, bn_means, bn_variances) = zip(*batch_norm_vars)
  model_weights = {}

  # Hidden layer weights.
  for i in range(0, len(weights) - 1):
    model_weights[HIDDEN_WEIGHTS_NAME_PATTERN % i] = weights[i]
    model_weights[HIDDEN_BIASES_NAME_PATTERN % i] = biases[i]
    if batch_norm_vars:
      model_weights[BATCH_NORM_BETA_NAME_PATTERN % (i, i)] = bn_betas[i]
      model_weights[BATCH_NORM_GAMMA_NAME_PATTERN % (i, i)] = bn_gammas[i]
      model_weights[BATCH_NORM_MEAN_NAME_PATTERN % (i, i)] = bn_means[i]
      model_weights[BATCH_NORM_VARIANCE_NAME_PATTERN % (i, i)] = bn_variances[i]

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

  def _create_tpu_estimator_spec(
      features, mode, logits, labels, train_op_fn=None, optimizer=None):
    del features, labels  # Not used.
    trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    testcase.assertItemsEqual(expected_var_names,
                              [var.name for var in trainable_vars])
    loss = constant_op.constant(1.)
    assert_logits = assert_close(
        expected_logits, logits, message='Failed for mode={}. '.format(mode))
    with ops.control_dependencies([assert_logits]):
      if mode == model_fn.ModeKeys.TRAIN:
        if train_op_fn is not None:
          train_op = train_op_fn(loss)
        elif optimizer is not None:
          train_op = optimizer.minimize(loss, global_step=None)
        return model_fn._TPUEstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)
      elif mode == model_fn.ModeKeys.EVAL:
        return model_fn._TPUEstimatorSpec(
            mode=mode, loss=array_ops.identity(loss))
      elif mode == model_fn.ModeKeys.PREDICT:
        return model_fn._TPUEstimatorSpec(
            mode=mode, predictions={'logits': array_ops.identity(logits)})
      else:
        testcase.fail('Invalid mode: {}'.format(mode))

  def _create_estimator_spec(
      features, mode, logits, labels, train_op_fn=None, optimizer=None):
    tpu_spec = _create_tpu_estimator_spec(
        features, mode, logits, labels, train_op_fn, optimizer)
    return tpu_spec.as_estimator_spec()

  head = test.mock.NonCallableMagicMock(spec=head_lib._Head)
  head.logits_dimension = logits_dimension
  head._create_tpu_estimator_spec = test.mock.MagicMock(
      wraps=_create_tpu_estimator_spec)
  head.create_estimator_spec = test.mock.MagicMock(
      wraps=_create_estimator_spec)

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
      spec=optimizer_lib.Optimizer,
      wraps=optimizer_lib.Optimizer(use_locking=False, name='my_optimizer'))
  optimizer_mock.minimize = test.mock.MagicMock(wraps=_minimize)

  return optimizer_mock


class BaseDNNModelFnTest(object):
  """Tests that _dnn_model_fn passes expected logits to mock head."""

  def __init__(self, dnn_model_fn, fc_impl=feature_column):
    self._dnn_model_fn = dnn_model_fn
    self._fc_impl = fc_impl

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
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
              self._fc_impl.numeric_column(
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
                self._fc_impl.numeric_column('age'),
                self._fc_impl.numeric_column('height')
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

  def test_features_tensor_raises_value_error(self):
    """Tests that passing a Tensor for features raises a ValueError."""
    hidden_units = (2, 2)
    logits_dimension = 3
    inputs = ([[10.]], [[8.]])
    expected_logits = [[0, 0, 0]]

    with ops.Graph().as_default():
      training_util.create_global_step()
      head = mock_head(
          self,
          hidden_units=hidden_units,
          logits_dimension=logits_dimension,
          expected_logits=expected_logits)
      with self.assertRaisesRegexp(ValueError, 'features should be a dict'):
        self._dnn_model_fn(
            features=constant_op.constant(inputs),
            labels=constant_op.constant([[1]]),
            mode=model_fn.ModeKeys.TRAIN,
            head=head,
            hidden_units=hidden_units,
            feature_columns=[
                self._fc_impl.numeric_column(
                    'age', shape=np.array(inputs).shape[1:])
            ],
            optimizer=mock_optimizer(self, hidden_units))


class BaseDNNLogitFnTest(object):
  """Tests correctness of logits calculated from _dnn_logit_fn_builder."""

  def __init__(self, dnn_logit_fn_builder, fc_impl=feature_column):
    self._dnn_logit_fn_builder = dnn_logit_fn_builder
    self._fc_impl = fc_impl

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_logits(self,
                   mode,
                   hidden_units,
                   logits_dimension,
                   inputs,
                   expected_logits,
                   batch_norm=False):
    """Tests that the expected logits are calculated."""
    with ops.Graph().as_default():
      # Global step needed for MonitoredSession, which is in turn used to
      # explicitly set variable weights through a checkpoint.
      training_util.create_global_step()
      # Use a variable scope here with 'dnn', emulating the dnn model_fn, so
      # the checkpoint naming is shared.
      with variable_scope.variable_scope('dnn'):
        input_layer_partitioner = (
            partitioned_variables.min_max_variable_partitioner(
                max_partitions=0, min_slice_size=64 << 20))
        logit_fn = self._dnn_logit_fn_builder(
            units=logits_dimension,
            hidden_units=hidden_units,
            feature_columns=[
                self._fc_impl.numeric_column(
                    'age', shape=np.array(inputs).shape[1:])
            ],
            activation_fn=nn.relu,
            dropout=None,
            input_layer_partitioner=input_layer_partitioner,
            batch_norm=batch_norm)
        logits = logit_fn(
            features={'age': constant_op.constant(inputs)}, mode=mode)
        with monitored_session.MonitoredTrainingSession(
            checkpoint_dir=self._model_dir) as sess:
          self.assertAllClose(expected_logits, sess.run(logits))

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

  def test_one_dim_logits_with_batch_norm(self):
    """Tests one-dimensional logits.

    input_layer = [[10]]
    hidden_layer_0 = [[relu(0.6*10 +1), relu(0.5*10 -1)]] = [[7, 4]]
    hidden_layer_0 = [[relu(0.6*20 +1), relu(0.5*20 -1)]] = [[13, 9]]

    batch_norm_0, training (epsilon = 0.001):
      mean1 = 1/2*(7+13) = 10,
      variance1 = 1/2*(3^2+3^2) = 9
      x11 = (7-10)/sqrt(9+0.001) = -0.999944449,
      x21 = (13-10)/sqrt(9+0.001) = 0.999944449,

      mean2 = 1/2*(4+9) = 6.5,
      variance2 = 1/2*(2.5^2+.2.5^2) = 6.25
      x12 = (4-6.5)/sqrt(6.25+0.001) = -0.99992001,
      x22 = (9-6.5)/sqrt(6.25+0.001) = 0.99992001,

    logits = [[-1*(-0.999944449) + 2*(-0.99992001) + 0.3],
              [-1*0.999944449 + 2*0.99992001 + 0.3]]
           = [[-0.699895571],[1.299895571]]

    batch_norm_0, not training (epsilon = 0.001):
      moving_mean1 = 0, moving_variance1 = 1
      x11 = (7-0)/sqrt(1+0.001) = 6.996502623,
      x21 = (13-0)/sqrt(1+0.001) = 12.993504871,
      moving_mean2 = 0, moving_variance2 = 1
      x12 = (4-0)/sqrt(1+0.001) = 3.998001499,
      x22 = (9-0)/sqrt(1+0.001) = 8.995503372,

    logits = [[-1*6.996502623 + 2*3.998001499 + 0.3],
              [-1*12.993504871 + 2*8.995503372 + 0.3]]
           = [[1.299500375],[5.297501873]]
    """
    base_global_step = 100
    create_checkpoint(
        (
            ([[.6, .5]], [1., -1.]),
            ([[-1.], [2.]], [.3]),
        ),
        base_global_step,
        self._model_dir,
        batch_norm_vars=([[0, 0],  # beta.
                          [1, 1],  # gamma.
                          [0, 0],  # moving mean.
                          [1, 1],  # moving variance.
                         ],))
    self._test_logits(
        model_fn.ModeKeys.TRAIN,
        hidden_units=[2],
        logits_dimension=1,
        inputs=[[10.], [20.]],
        expected_logits=[[-0.699895571], [1.299895571]],
        batch_norm=True)
    for mode in [model_fn.ModeKeys.EVAL, model_fn.ModeKeys.PREDICT]:
      self._test_logits(
          mode,
          hidden_units=[2],
          logits_dimension=1,
          inputs=[[10.], [20.]],
          expected_logits=[[1.299500375], [5.297501873]],
          batch_norm=True)

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
        # Global step needed for MonitoredSession, which is in turn used to
        # explicitly set variable weights through a checkpoint.
        training_util.create_global_step()
        # Use a variable scope here with 'dnn', emulating the dnn model_fn, so
        # the checkpoint naming is shared.
        with variable_scope.variable_scope('dnn'):
          input_layer_partitioner = (
              partitioned_variables.min_max_variable_partitioner(
                  max_partitions=0, min_slice_size=64 << 20))
          logit_fn = self._dnn_logit_fn_builder(
              units=logits_dimension,
              hidden_units=hidden_units,
              feature_columns=[
                  self._fc_impl.numeric_column('age'),
                  self._fc_impl.numeric_column('height')
              ],
              activation_fn=nn.relu,
              dropout=None,
              input_layer_partitioner=input_layer_partitioner,
              batch_norm=False)
          logits = logit_fn(
              features={
                  'age': constant_op.constant(inputs[0]),
                  'height': constant_op.constant(inputs[1])
              },
              mode=mode)
          with monitored_session.MonitoredTrainingSession(
              checkpoint_dir=self._model_dir) as sess:
            self.assertAllClose(expected_logits, sess.run(logits))


class BaseDNNWarmStartingTest(object):

  def __init__(self,
               _dnn_classifier_fn,
               _dnn_regressor_fn,
               fc_impl=feature_column):
    self._dnn_classifier_fn = _dnn_classifier_fn
    self._dnn_regressor_fn = _dnn_regressor_fn
    self._fc_impl = fc_impl

  def setUp(self):
    # Create a directory to save our old checkpoint and vocabularies to.
    self._ckpt_and_vocab_dir = tempfile.mkdtemp()

    # Make a dummy input_fn.
    def _input_fn():
      features = {
          'city': [['Palo Alto'], ['Mountain View']],
          'locality': [['Palo Alto'], ['Mountain View']],
          'occupation': [['doctor'], ['consultant']]
      }
      return features, [0, 1]

    self._input_fn = _input_fn

  def tearDown(self):
    # Clean up checkpoint / vocab dir.
    writer_cache.FileWriterCache.clear()
    shutil.rmtree(self._ckpt_and_vocab_dir)

  def assertAllNotClose(self, t1, t2):
    """Helper assert for arrays."""
    sum_of_abs_diff = 0.0
    for x, y in zip(t1, t2):
      try:
        for a, b in zip(x, y):
          sum_of_abs_diff += abs(b - a)
      except TypeError:
        sum_of_abs_diff += abs(y - x)
    self.assertGreater(sum_of_abs_diff, 0)

  def test_classifier_basic_warm_starting(self):
    """Tests correctness of DNNClassifier default warm-start."""
    city = self._fc_impl.embedding_column(
        self._fc_impl.categorical_column_with_vocabulary_list(
            'city', vocabulary_list=['Mountain View', 'Palo Alto']),
        dimension=5)

    # Create a DNNClassifier and train to save a checkpoint.
    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=[256, 128],
        feature_columns=[city],
        model_dir=self._ckpt_and_vocab_dir,
        n_classes=4,
        optimizer='SGD')
    dnn_classifier.train(input_fn=self._input_fn, max_steps=1)

    # Create a second DNNClassifier, warm-started from the first.  Use a
    # learning_rate = 0.0 optimizer to check values (use SGD so we don't have
    # accumulator values that change).
    warm_started_dnn_classifier = self._dnn_classifier_fn(
        hidden_units=[256, 128],
        feature_columns=[city],
        n_classes=4,
        optimizer=gradient_descent.GradientDescentOptimizer(learning_rate=0.0),
        warm_start_from=dnn_classifier.model_dir)

    warm_started_dnn_classifier.train(input_fn=self._input_fn, max_steps=1)
    for variable_name in warm_started_dnn_classifier.get_variable_names():
      self.assertAllClose(
          dnn_classifier.get_variable_value(variable_name),
          warm_started_dnn_classifier.get_variable_value(variable_name))

  def test_regressor_basic_warm_starting(self):
    """Tests correctness of DNNRegressor default warm-start."""
    city = self._fc_impl.embedding_column(
        self._fc_impl.categorical_column_with_vocabulary_list(
            'city', vocabulary_list=['Mountain View', 'Palo Alto']),
        dimension=5)

    # Create a DNNRegressor and train to save a checkpoint.
    dnn_regressor = self._dnn_regressor_fn(
        hidden_units=[256, 128],
        feature_columns=[city],
        model_dir=self._ckpt_and_vocab_dir,
        optimizer='SGD')
    dnn_regressor.train(input_fn=self._input_fn, max_steps=1)

    # Create a second DNNRegressor, warm-started from the first.  Use a
    # learning_rate = 0.0 optimizer to check values (use SGD so we don't have
    # accumulator values that change).
    warm_started_dnn_regressor = self._dnn_regressor_fn(
        hidden_units=[256, 128],
        feature_columns=[city],
        optimizer=gradient_descent.GradientDescentOptimizer(learning_rate=0.0),
        warm_start_from=dnn_regressor.model_dir)

    warm_started_dnn_regressor.train(input_fn=self._input_fn, max_steps=1)
    for variable_name in warm_started_dnn_regressor.get_variable_names():
      self.assertAllClose(
          dnn_regressor.get_variable_value(variable_name),
          warm_started_dnn_regressor.get_variable_value(variable_name))

  def test_warm_starting_selective_variables(self):
    """Tests selecting variables to warm-start."""
    city = self._fc_impl.embedding_column(
        self._fc_impl.categorical_column_with_vocabulary_list(
            'city', vocabulary_list=['Mountain View', 'Palo Alto']),
        dimension=5)

    # Create a DNNClassifier and train to save a checkpoint.
    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=[256, 128],
        feature_columns=[city],
        model_dir=self._ckpt_and_vocab_dir,
        n_classes=4,
        optimizer='SGD')
    dnn_classifier.train(input_fn=self._input_fn, max_steps=1)

    # Create a second DNNClassifier, warm-started from the first.  Use a
    # learning_rate = 0.0 optimizer to check values (use SGD so we don't have
    # accumulator values that change).
    warm_started_dnn_classifier = self._dnn_classifier_fn(
        hidden_units=[256, 128],
        feature_columns=[city],
        n_classes=4,
        optimizer=gradient_descent.GradientDescentOptimizer(learning_rate=0.0),
        # The provided regular expression will only warm-start the city
        # embedding, not the kernels and biases of the hidden weights.
        warm_start_from=estimator.WarmStartSettings(
            ckpt_to_initialize_from=dnn_classifier.model_dir,
            vars_to_warm_start='.*(city).*'))

    warm_started_dnn_classifier.train(input_fn=self._input_fn, max_steps=1)
    for variable_name in warm_started_dnn_classifier.get_variable_names():
      if 'city' in variable_name:
        self.assertAllClose(
            dnn_classifier.get_variable_value(variable_name),
            warm_started_dnn_classifier.get_variable_value(variable_name))
      elif 'bias' in variable_name:
        # Hidden layer biases are zero-initialized.
        bias_values = warm_started_dnn_classifier.get_variable_value(
            variable_name)
        self.assertAllClose(np.zeros_like(bias_values), bias_values)
      elif 'kernel' in variable_name:
        # We can't override the glorot uniform initializer used for the kernels
        # in the dense layers, so just make sure we're not getting the same
        # values from the old checkpoint.
        self.assertAllNotClose(
            dnn_classifier.get_variable_value(variable_name),
            warm_started_dnn_classifier.get_variable_value(variable_name))

  def test_warm_starting_with_vocab_remapping_and_partitioning(self):
    """Tests warm-starting with vocab remapping and partitioning."""
    vocab_list = ['doctor', 'lawyer', 'consultant']
    vocab_file = os.path.join(self._ckpt_and_vocab_dir, 'occupation_vocab')
    with open(vocab_file, 'w') as f:
      f.write('\n'.join(vocab_list))
    occupation = self._fc_impl.embedding_column(
        self._fc_impl.categorical_column_with_vocabulary_file(
            'occupation',
            vocabulary_file=vocab_file,
            vocabulary_size=len(vocab_list)),
        dimension=2)

    # Create a DNNClassifier and train to save a checkpoint.
    partitioner = partitioned_variables.fixed_size_partitioner(num_shards=2)
    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=[256, 128],
        feature_columns=[occupation],
        model_dir=self._ckpt_and_vocab_dir,
        n_classes=4,
        optimizer='SGD',
        input_layer_partitioner=partitioner)
    dnn_classifier.train(input_fn=self._input_fn, max_steps=1)

    # Create a second DNNClassifier, warm-started from the first.  Use a
    # learning_rate = 0.0 optimizer to check values (use SGD so we don't have
    # accumulator values that change).  Use a new FeatureColumn with a
    # different vocabulary for occupation.
    new_vocab_list = ['doctor', 'consultant', 'engineer']
    new_vocab_file = os.path.join(self._ckpt_and_vocab_dir,
                                  'new_occupation_vocab')
    with open(new_vocab_file, 'w') as f:
      f.write('\n'.join(new_vocab_list))
    new_occupation = self._fc_impl.embedding_column(
        self._fc_impl.categorical_column_with_vocabulary_file(
            'occupation',
            vocabulary_file=new_vocab_file,
            vocabulary_size=len(new_vocab_list)),
        dimension=2)
    # We can create our VocabInfo object from the new and old occupation
    # FeatureColumn's.
    occupation_vocab_info = estimator.VocabInfo(
        new_vocab=new_occupation.categorical_column.vocabulary_file,
        new_vocab_size=new_occupation.categorical_column.vocabulary_size,
        num_oov_buckets=new_occupation.categorical_column.num_oov_buckets,
        old_vocab=occupation.categorical_column.vocabulary_file,
        old_vocab_size=occupation.categorical_column.vocabulary_size,
        # Can't use constant_initializer with load_and_remap.  In practice,
        # use a truncated normal initializer.
        backup_initializer=init_ops.random_uniform_initializer(
            minval=0.39, maxval=0.39))
    warm_started_dnn_classifier = self._dnn_classifier_fn(
        hidden_units=[256, 128],
        feature_columns=[occupation],
        n_classes=4,
        optimizer=gradient_descent.GradientDescentOptimizer(learning_rate=0.0),
        warm_start_from=estimator.WarmStartSettings(
            ckpt_to_initialize_from=dnn_classifier.model_dir,
            var_name_to_vocab_info={
                OCCUPATION_EMBEDDING_NAME: occupation_vocab_info
            },
            # Explicitly providing None here will only warm-start variables
            # referenced in var_name_to_vocab_info (no hidden weights will be
            # warmstarted).
            vars_to_warm_start=None),
        input_layer_partitioner=partitioner)

    warm_started_dnn_classifier.train(input_fn=self._input_fn, max_steps=1)
    # 'doctor' was ID-0 and still ID-0.
    self.assertAllClose(
        dnn_classifier.get_variable_value(OCCUPATION_EMBEDDING_NAME)[0, :],
        warm_started_dnn_classifier.get_variable_value(
            OCCUPATION_EMBEDDING_NAME)[0, :])
    # 'consultant' was ID-2 and now ID-1.
    self.assertAllClose(
        dnn_classifier.get_variable_value(OCCUPATION_EMBEDDING_NAME)[2, :],
        warm_started_dnn_classifier.get_variable_value(
            OCCUPATION_EMBEDDING_NAME)[1, :])
    # 'engineer' is a new entry and should be initialized with the
    # backup_initializer in VocabInfo.
    self.assertAllClose([0.39] * 2,
                        warm_started_dnn_classifier.get_variable_value(
                            OCCUPATION_EMBEDDING_NAME)[2, :])
    for variable_name in warm_started_dnn_classifier.get_variable_names():
      if 'bias' in variable_name:
        # Hidden layer biases are zero-initialized.
        bias_values = warm_started_dnn_classifier.get_variable_value(
            variable_name)
        self.assertAllClose(np.zeros_like(bias_values), bias_values)
      elif 'kernel' in variable_name:
        # We can't override the glorot uniform initializer used for the kernels
        # in the dense layers, so just make sure we're not getting the same
        # values from the old checkpoint.
        self.assertAllNotClose(
            dnn_classifier.get_variable_value(variable_name),
            warm_started_dnn_classifier.get_variable_value(variable_name))

  def test_warm_starting_with_naming_change(self):
    """Tests warm-starting with a Tensor name remapping."""
    locality = self._fc_impl.embedding_column(
        self._fc_impl.categorical_column_with_vocabulary_list(
            'locality', vocabulary_list=['Mountain View', 'Palo Alto']),
        dimension=5)

    # Create a DNNClassifier and train to save a checkpoint.
    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=[256, 128],
        feature_columns=[locality],
        model_dir=self._ckpt_and_vocab_dir,
        n_classes=4,
        optimizer='SGD')
    dnn_classifier.train(input_fn=self._input_fn, max_steps=1)

    # Create a second DNNClassifier, warm-started from the first.  Use a
    # learning_rate = 0.0 optimizer to check values (use SGD so we don't have
    # accumulator values that change).
    city = self._fc_impl.embedding_column(
        self._fc_impl.categorical_column_with_vocabulary_list(
            'city', vocabulary_list=['Mountain View', 'Palo Alto']),
        dimension=5)
    warm_started_dnn_classifier = self._dnn_classifier_fn(
        hidden_units=[256, 128],
        feature_columns=[city],
        n_classes=4,
        optimizer=gradient_descent.GradientDescentOptimizer(learning_rate=0.0),
        # The 'city' variable correspond to the 'locality' variable in the
        # previous model.
        warm_start_from=estimator.WarmStartSettings(
            ckpt_to_initialize_from=dnn_classifier.model_dir,
            var_name_to_prev_var_name={
                CITY_EMBEDDING_NAME:
                    CITY_EMBEDDING_NAME.replace('city', 'locality')
            }))

    warm_started_dnn_classifier.train(input_fn=self._input_fn, max_steps=1)
    for variable_name in warm_started_dnn_classifier.get_variable_names():
      if 'city' in variable_name:
        self.assertAllClose(
            dnn_classifier.get_variable_value(
                CITY_EMBEDDING_NAME.replace('city', 'locality')),
            warm_started_dnn_classifier.get_variable_value(CITY_EMBEDDING_NAME))
      else:
        self.assertAllClose(
            dnn_classifier.get_variable_value(variable_name),
            warm_started_dnn_classifier.get_variable_value(variable_name))


class BaseDNNClassifierEvaluateTest(object):

  def __init__(self, dnn_classifier_fn, fc_impl=feature_column):
    self._dnn_classifier_fn = dnn_classifier_fn
    self._fc_impl = fc_impl

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def test_one_dim(self):
    """Asserts evaluation metrics for one-dimensional input and logits."""
    global_step = 100
    create_checkpoint(
        (([[.6, .5]], [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
         ([[-1.], [1.]], [.3]),), global_step, self._model_dir)

    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=(2, 2),
        feature_columns=[self._fc_impl.numeric_column('age')],
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
        metric_keys.MetricKeys.PRECISION: 0.0,
        metric_keys.MetricKeys.RECALL: 0.0,
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
    create_checkpoint(
        (([[.6, .5], [-.6, -.5]], [.1, -.1]), ([[1., .8], [-.8, -1.]],
                                               [.2, -.2]),
         ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3,
                                           .0]),), global_step, self._model_dir)
    n_classes = 3

    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=(2, 2),
        feature_columns=[self._fc_impl.numeric_column('age', shape=[2])],
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

  def test_float_labels(self):
    """Asserts evaluation metrics for float labels in binary classification."""
    global_step = 100
    create_checkpoint(
        (([[.6, .5]], [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
         ([[-1.], [1.]], [.3]),), global_step, self._model_dir)

    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=(2, 2),
        feature_columns=[self._fc_impl.numeric_column('age')],
        model_dir=self._model_dir)
    def _input_fn():
      # batch_size = 2, one false label, and one true.
      return {'age': [[10.], [10.]]}, [[0.8], [0.4]]
    # Uses identical numbers as DNNModelTest.test_one_dim_logits.
    # See that test for calculation of logits.
    # logits = [[-2.08], [-2.08]] =>
    # logistic = 1/(1 + exp(-logits)) = [[0.11105597], [0.11105597]]
    # loss = -0.8 * log(0.111) -0.2 * log(0.889)
    #        -0.4 * log(0.111) -0.6 * log(0.889) = 2.7314420
    metrics = dnn_classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertAlmostEqual(2.7314420, metrics[metric_keys.MetricKeys.LOSS])

  def test_multi_dim_weights(self):
    """Tests evaluation with weights."""
    # Uses same checkpoint with test_multi_dims
    global_step = 100
    create_checkpoint((([[.6, .5], [-.6, -.5]],
                        [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
                       ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),),
                      global_step, self._model_dir)
    n_classes = 3

    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=(2, 2),
        feature_columns=[self._fc_impl.numeric_column('age', shape=[2])],
        n_classes=n_classes,
        weight_column='w',
        model_dir=self._model_dir)

    def _input_fn():
      # batch_size = 2, one false label, and one true.
      return {'age': [[10., 8.], [10., 8.]], 'w': [[10.], [100.]]}, [[1], [0]]

    # Uses identical numbers as test_multi_dims
    # See that test for calculation of logits.
    # loss = -log(0.43538380)*10 - log(0.16670536)*100
    expected_loss = 187.468007
    metrics = dnn_classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertAlmostEqual(
        expected_loss, metrics[metric_keys.MetricKeys.LOSS], places=3)


class BaseDNNRegressorEvaluateTest(object):

  def __init__(self, dnn_regressor_fn, fc_impl=feature_column):
    self._dnn_regressor_fn = dnn_regressor_fn
    self._fc_impl = fc_impl

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def test_one_dim(self):
    """Asserts evaluation metrics for one-dimensional input and logits."""
    # Create checkpoint: num_inputs=1, hidden_units=(2, 2), num_outputs=1.
    global_step = 100
    create_checkpoint(
        (([[.6, .5]], [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
         ([[-1.], [1.]], [.3]),), global_step, self._model_dir)

    dnn_regressor = self._dnn_regressor_fn(
        hidden_units=(2, 2),
        feature_columns=[self._fc_impl.numeric_column('age')],
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
        metric_keys.MetricKeys.PREDICTION_MEAN: -2.08,
        metric_keys.MetricKeys.LABEL_MEAN: 1.0,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_regressor.evaluate(input_fn=_input_fn, steps=1))

  def test_multi_dim(self):
    """Asserts evaluation metrics for multi-dimensional input and logits."""
    # Create checkpoint: num_inputs=2, hidden_units=(2, 2), num_outputs=3.
    global_step = 100
    create_checkpoint(
        (([[.6, .5], [-.6, -.5]], [.1, -.1]), ([[1., .8], [-.8, -1.]],
                                               [.2, -.2]),
         ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3,
                                           .0]),), global_step, self._model_dir)
    label_dimension = 3

    dnn_regressor = self._dnn_regressor_fn(
        hidden_units=(2, 2),
        feature_columns=[self._fc_impl.numeric_column('age', shape=[2])],
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
        metric_keys.MetricKeys.PREDICTION_MEAN: 0.39 / 3.0,
        metric_keys.MetricKeys.LABEL_MEAN: 0.5 / 3.0,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_regressor.evaluate(input_fn=_input_fn, steps=1))

  def test_multi_dim_weights(self):
    """Asserts evaluation metrics for multi-dimensional input and logits."""
    # same checkpoint with test_multi_dim.
    global_step = 100
    create_checkpoint((([[.6, .5], [-.6, -.5]],
                        [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
                       ([[-1., 1., .5], [-1., 1., .5]], [.3, -.3, .0]),),
                      global_step, self._model_dir)
    label_dimension = 3

    dnn_regressor = self._dnn_regressor_fn(
        hidden_units=(2, 2),
        feature_columns=[self._fc_impl.numeric_column('age', shape=[2])],
        label_dimension=label_dimension,
        weight_column='w',
        model_dir=self._model_dir)

    def _input_fn():
      return {'age': [[10., 8.]], 'w': [10.]}, [[1., -1., 0.5]]

    # Uses identical numbers as test_multi_dim.
    # See that test for calculation of logits.
    # loss = 4.3929*10
    expected_loss = 43.929
    metrics = dnn_regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertAlmostEqual(
        expected_loss, metrics[metric_keys.MetricKeys.LOSS], places=3)


class BaseDNNClassifierPredictTest(object):

  def __init__(self, dnn_classifier_fn, fc_impl=feature_column):
    self._dnn_classifier_fn = dnn_classifier_fn
    self._fc_impl = fc_impl

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_one_dim(self, label_vocabulary, label_output_fn):
    """Asserts predictions for one-dimensional input and logits."""
    create_checkpoint(
        (([[.6, .5]], [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
         ([[-1.], [1.]], [.3]),),
        global_step=0,
        model_dir=self._model_dir)

    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=(2, 2),
        label_vocabulary=label_vocabulary,
        feature_columns=(self._fc_impl.numeric_column('x'),),
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        x={'x': np.array([[10.]])}, batch_size=1, shuffle=False)
    # Uses identical numbers as DNNModelTest.test_one_dim_logits.
    # See that test for calculation of logits.
    # logits = [-2.08] =>
    # logistic = exp(-2.08)/(1 + exp(-2.08)) = 0.11105597
    # probabilities = [1-logistic, logistic] = [0.88894403, 0.11105597]
    # class_ids = argmax(probabilities) = [0]
    predictions = next(dnn_classifier.predict(input_fn=input_fn))
    self.assertAllClose([-2.08],
                        predictions[prediction_keys.PredictionKeys.LOGITS])
    self.assertAllClose([0.11105597],
                        predictions[prediction_keys.PredictionKeys.LOGISTIC])
    self.assertAllClose(
        [0.88894403,
         0.11105597], predictions[prediction_keys.PredictionKeys.PROBABILITIES])
    self.assertAllClose([0],
                        predictions[prediction_keys.PredictionKeys.CLASS_IDS])
    self.assertAllEqual([label_output_fn(0)],
                        predictions[prediction_keys.PredictionKeys.CLASSES])

  def test_one_dim_without_label_vocabulary(self):
    self._test_one_dim(label_vocabulary=None,
                       label_output_fn=lambda x: ('%s' % x).encode())

  def test_one_dim_with_label_vocabulary(self):
    n_classes = 2
    self._test_one_dim(
        label_vocabulary=['class_vocab_{}'.format(i) for i in range(n_classes)],
        label_output_fn=lambda x: ('class_vocab_%s' % x).encode())

  def _test_multi_dim_with_3_classes(self, label_vocabulary, label_output_fn):
    """Asserts predictions for multi-dimensional input and logits."""
    create_checkpoint(
        (([[.6, .5], [-.6, -.5]], [.1, -.1]),
         ([[1., .8], [-.8, -1.]], [.2, -.2]), ([[-1., 1., .5], [-1., 1., .5]],
                                               [.3, -.3, .0]),),
        global_step=0,
        model_dir=self._model_dir)

    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=(2, 2),
        feature_columns=(self._fc_impl.numeric_column('x', shape=(2,)),),
        label_vocabulary=label_vocabulary,
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
        [label_output_fn(1)],
        predictions[prediction_keys.PredictionKeys.CLASSES])

  def test_multi_dim_with_3_classes_but_no_label_vocab(self):
    self._test_multi_dim_with_3_classes(
        label_vocabulary=None,
        label_output_fn=lambda x: ('%s' % x).encode())

  def test_multi_dim_with_3_classes_and_label_vocab(self):
    n_classes = 3
    self._test_multi_dim_with_3_classes(
        label_vocabulary=['class_vocab_{}'.format(i) for i in range(n_classes)],
        label_output_fn=lambda x: ('class_vocab_%s' % x).encode())


class BaseDNNRegressorPredictTest(object):

  def __init__(self, dnn_regressor_fn, fc_impl=feature_column):
    self._dnn_regressor_fn = dnn_regressor_fn
    self._fc_impl = fc_impl

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def test_one_dim(self):
    """Asserts predictions for one-dimensional input and logits."""
    # Create checkpoint: num_inputs=1, hidden_units=(2, 2), num_outputs=1.
    create_checkpoint(
        (([[.6, .5]], [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
         ([[-1.], [1.]], [.3]),),
        global_step=0,
        model_dir=self._model_dir)

    dnn_regressor = self._dnn_regressor_fn(
        hidden_units=(2, 2),
        feature_columns=(self._fc_impl.numeric_column('x'),),
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
    create_checkpoint(
        (([[.6, .5], [-.6, -.5]], [.1, -.1]),
         ([[1., .8], [-.8, -1.]], [.2, -.2]), ([[-1., 1., .5], [-1., 1., .5]],
                                               [.3, -.3,
                                                .0]),), 100, self._model_dir)

    dnn_regressor = self._dnn_regressor_fn(
        hidden_units=(2, 2),
        feature_columns=(self._fc_impl.numeric_column('x', shape=(2,)),),
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
    testcase.assertAllEqual(
        (prev_layer_units, layer_units),
        shapes[HIDDEN_WEIGHTS_NAME_PATTERN % i])
    testcase.assertAllEqual(
        (layer_units,),
        shapes[HIDDEN_BIASES_NAME_PATTERN % i])
    prev_layer_units = layer_units

  # Output layer weights.
  testcase.assertAllEqual((prev_layer_units, output_units),
                          shapes[LOGITS_WEIGHTS_NAME])
  testcase.assertAllEqual((output_units,),
                          shapes[LOGITS_BIASES_NAME])


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


class BaseDNNClassifierTrainTest(object):

  def __init__(self, dnn_classifier_fn, fc_impl=feature_column):
    self._dnn_classifier_fn = dnn_classifier_fn
    self._fc_impl = fc_impl

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def test_from_scratch_with_default_optimizer_binary(self):
    hidden_units = (2, 2)
    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=hidden_units,
        feature_columns=(self._fc_impl.numeric_column('age'),),
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
    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=hidden_units,
        feature_columns=(self._fc_impl.numeric_column('age'),),
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
    opt = mock_optimizer(
        self, hidden_units=hidden_units)
    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=hidden_units,
        feature_columns=(self._fc_impl.numeric_column('age'),),
        optimizer=opt,
        model_dir=self._model_dir)
    self.assertEqual(0, opt.minimize.call_count)

    # Train for a few steps, then validate optimizer, summaries, and
    # checkpoint.
    num_steps = 5
    summary_hook = _SummaryHook()
    dnn_classifier.train(
        input_fn=lambda: ({'age': [[10.]]}, [[1]]), steps=num_steps,
        hooks=(summary_hook,))
    self.assertEqual(1, opt.minimize.call_count)
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
    create_checkpoint(
        (([[.6, .5]], [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
         ([[-1.], [1.]], [.3]),), base_global_step, self._model_dir)

    # Uses identical numbers as DNNModelFnTest.test_one_dim_logits.
    # See that test for calculation of logits.
    # logits = [-2.08] => probabilities = [0.889, 0.111]
    # loss = -1. * log(0.111) = 2.19772100
    expected_loss = 2.19772100
    opt = mock_optimizer(
        self, hidden_units=hidden_units, expected_loss=expected_loss)
    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=hidden_units,
        feature_columns=(self._fc_impl.numeric_column('age'),),
        optimizer=opt,
        model_dir=self._model_dir)
    self.assertEqual(0, opt.minimize.call_count)

    # Train for a few steps, then validate optimizer, summaries, and
    # checkpoint.
    num_steps = 5
    summary_hook = _SummaryHook()
    dnn_classifier.train(
        input_fn=lambda: ({'age': [[10.]]}, [[1]]), steps=num_steps,
        hooks=(summary_hook,))
    self.assertEqual(1, opt.minimize.call_count)
    summaries = summary_hook.summaries()
    self.assertEqual(num_steps, len(summaries))
    for summary in summaries:
      _assert_simple_summary(
          self,
          {
              metric_keys.MetricKeys.LOSS_MEAN: expected_loss,
              'dnn/dnn/hiddenlayer_0/fraction_of_zero_values': 0.,
              'dnn/dnn/hiddenlayer_1/fraction_of_zero_values': .5,
              'dnn/dnn/logits/fraction_of_zero_values': 0.,
              metric_keys.MetricKeys.LOSS: expected_loss,
          },
          summary)
    _assert_checkpoint(
        self, base_global_step + num_steps, input_units=1,
        hidden_units=hidden_units, output_units=1, model_dir=self._model_dir)

  def test_binary_classification_float_labels(self):
    base_global_step = 100
    hidden_units = (2, 2)
    create_checkpoint(
        (([[.6, .5]], [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
         ([[-1.], [1.]], [.3]),), base_global_step, self._model_dir)

    # Uses identical numbers as DNNModelFnTest.test_one_dim_logits.
    # See that test for calculation of logits.
    # logits = [-2.08] => probabilities = [0.889, 0.111]
    # loss = -0.8 * log(0.111) -0.2 * log(0.889) = 1.7817210
    expected_loss = 1.7817210
    opt = mock_optimizer(
        self, hidden_units=hidden_units, expected_loss=expected_loss)
    dnn_classifier = self._dnn_classifier_fn(
        hidden_units=hidden_units,
        feature_columns=(self._fc_impl.numeric_column('age'),),
        optimizer=opt,
        model_dir=self._model_dir)
    self.assertEqual(0, opt.minimize.call_count)

    # Train for a few steps, then validate optimizer, summaries, and
    # checkpoint.
    num_steps = 5
    dnn_classifier.train(
        input_fn=lambda: ({'age': [[10.]]}, [[0.8]]), steps=num_steps)
    self.assertEqual(1, opt.minimize.call_count)

  def test_multi_class(self):
    n_classes = 3
    base_global_step = 100
    hidden_units = (2, 2)
    create_checkpoint(
        (([[.6, .5]], [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
         ([[-1., 1., .5], [-1., 1., .5]],
          [.3, -.3, .0]),), base_global_step, self._model_dir)

    # Uses identical numbers as DNNModelFnTest.test_multi_dim_logits.
    # See that test for calculation of logits.
    # logits = [-2.08, 2.08, 1.19] => probabilities = [0.0109, 0.7011, 0.2879]
    # loss = -1. * log(0.7011) = 0.35505795
    expected_loss = 0.35505795
    opt = mock_optimizer(
        self, hidden_units=hidden_units, expected_loss=expected_loss)
    dnn_classifier = self._dnn_classifier_fn(
        n_classes=n_classes,
        hidden_units=hidden_units,
        feature_columns=(self._fc_impl.numeric_column('age'),),
        optimizer=opt,
        model_dir=self._model_dir)
    self.assertEqual(0, opt.minimize.call_count)

    # Train for a few steps, then validate optimizer, summaries, and
    # checkpoint.
    num_steps = 5
    summary_hook = _SummaryHook()
    dnn_classifier.train(
        input_fn=lambda: ({'age': [[10.]]}, [[1]]), steps=num_steps,
        hooks=(summary_hook,))
    self.assertEqual(1, opt.minimize.call_count)
    summaries = summary_hook.summaries()
    self.assertEqual(num_steps, len(summaries))
    for summary in summaries:
      _assert_simple_summary(
          self,
          {
              metric_keys.MetricKeys.LOSS_MEAN: expected_loss,
              'dnn/dnn/hiddenlayer_0/fraction_of_zero_values': 0.,
              'dnn/dnn/hiddenlayer_1/fraction_of_zero_values': .5,
              'dnn/dnn/logits/fraction_of_zero_values': 0.,
              metric_keys.MetricKeys.LOSS: expected_loss,
          },
          summary)
    _assert_checkpoint(
        self, base_global_step + num_steps, input_units=1,
        hidden_units=hidden_units, output_units=n_classes,
        model_dir=self._model_dir)


class BaseDNNRegressorTrainTest(object):

  def __init__(self, dnn_regressor_fn, fc_impl=feature_column):
    self._dnn_regressor_fn = dnn_regressor_fn
    self._fc_impl = fc_impl

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def test_from_scratch_with_default_optimizer(self):
    hidden_units = (2, 2)
    dnn_regressor = self._dnn_regressor_fn(
        hidden_units=hidden_units,
        feature_columns=(self._fc_impl.numeric_column('age'),),
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
    opt = mock_optimizer(self, hidden_units=hidden_units)
    dnn_regressor = self._dnn_regressor_fn(
        hidden_units=hidden_units,
        feature_columns=(self._fc_impl.numeric_column('age'),),
        optimizer=opt,
        model_dir=self._model_dir)
    self.assertEqual(0, opt.minimize.call_count)

    # Train for a few steps, then validate optimizer, summaries, and
    # checkpoint.
    num_steps = 5
    summary_hook = _SummaryHook()
    dnn_regressor.train(
        input_fn=lambda: ({'age': ((1,),)}, ((5.,),)), steps=num_steps,
        hooks=(summary_hook,))
    self.assertEqual(1, opt.minimize.call_count)
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
    create_checkpoint(
        (([[.6, .5]], [.1, -.1]), ([[1., .8], [-.8, -1.]], [.2, -.2]),
         ([[-1.], [1.]], [.3]),), base_global_step, self._model_dir)

    # Uses identical numbers as DNNModelFnTest.test_one_dim_logits.
    # See that test for calculation of logits.
    # logits = [-2.08] => predictions = [-2.08]
    # loss = (1 + 2.08)^2 = 9.4864
    expected_loss = 9.4864
    opt = mock_optimizer(
        self, hidden_units=hidden_units, expected_loss=expected_loss)
    dnn_regressor = self._dnn_regressor_fn(
        hidden_units=hidden_units,
        feature_columns=(self._fc_impl.numeric_column('age'),),
        optimizer=opt,
        model_dir=self._model_dir)
    self.assertEqual(0, opt.minimize.call_count)

    # Train for a few steps, then validate optimizer, summaries, and
    # checkpoint.
    num_steps = 5
    summary_hook = _SummaryHook()
    dnn_regressor.train(
        input_fn=lambda: ({'age': [[10.]]}, [[1.]]), steps=num_steps,
        hooks=(summary_hook,))
    self.assertEqual(1, opt.minimize.call_count)
    summaries = summary_hook.summaries()
    self.assertEqual(num_steps, len(summaries))
    for summary in summaries:
      _assert_simple_summary(
          self,
          {
              metric_keys.MetricKeys.LOSS_MEAN: expected_loss,
              'dnn/dnn/hiddenlayer_0/fraction_of_zero_values': 0.,
              'dnn/dnn/hiddenlayer_1/fraction_of_zero_values': 0.5,
              'dnn/dnn/logits/fraction_of_zero_values': 0.,
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
    create_checkpoint(
        (([[.6, .5], [-.6, -.5]], [.1, -.1]), ([[1., .8], [-.8, -1.]],
                                               [.2, -.2]),
         ([[-1., 1., .5], [-1., 1., .5]],
          [.3, -.3, .0]),), base_global_step, self._model_dir)
    input_dimension = 2
    label_dimension = 3

    # Uses identical numbers as
    # DNNModelFnTest.test_multi_dim_input_multi_dim_logits.
    # See that test for calculation of logits.
    # logits = [[-0.48, 0.48, 0.39]]
    # loss = (1+0.48)^2 + (-1-0.48)^2 + (0.5-0.39)^2 = 4.3929
    expected_loss = 4.3929
    opt = mock_optimizer(
        self, hidden_units=hidden_units, expected_loss=expected_loss)
    dnn_regressor = self._dnn_regressor_fn(
        hidden_units=hidden_units,
        feature_columns=[
            self._fc_impl.numeric_column('age', shape=[input_dimension])
        ],
        label_dimension=label_dimension,
        optimizer=opt,
        model_dir=self._model_dir)
    self.assertEqual(0, opt.minimize.call_count)

    # Train for a few steps, then validate optimizer, summaries, and
    # checkpoint.
    num_steps = 5
    summary_hook = _SummaryHook()
    dnn_regressor.train(
        input_fn=lambda: ({'age': [[10., 8.]]}, [[1., -1., 0.5]]),
        steps=num_steps,
        hooks=(summary_hook,))
    self.assertEqual(1, opt.minimize.call_count)
    summaries = summary_hook.summaries()
    self.assertEqual(num_steps, len(summaries))
    for summary in summaries:
      _assert_simple_summary(
          self,
          {
              metric_keys.MetricKeys.LOSS_MEAN: expected_loss / label_dimension,
              'dnn/dnn/hiddenlayer_0/fraction_of_zero_values': 0.,
              'dnn/dnn/hiddenlayer_1/fraction_of_zero_values': 0.5,
              'dnn/dnn/logits/fraction_of_zero_values': 0.,
              metric_keys.MetricKeys.LOSS: expected_loss,
          },
          summary)
    _assert_checkpoint(
        self, base_global_step + num_steps, input_units=input_dimension,
        hidden_units=hidden_units, output_units=label_dimension,
        model_dir=self._model_dir)

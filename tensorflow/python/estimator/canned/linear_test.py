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
"""Tests for linear.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import shutil
import tempfile

import numpy as np

from tensorflow.python.estimator.canned import linear
from tensorflow.python.estimator.canned import linear_testing_utils
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import optimizer


def _linear_regressor_fn(*args, **kwargs):
  return linear.LinearRegressor(*args, **kwargs)


class LinearRegressorPartitionerTest(
    linear_testing_utils.BaseLinearRegressorPartitionerTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorPartitionerTest.__init__(
        self, _linear_regressor_fn)


class LinearRegressorEvaluationTest(
    linear_testing_utils.BaseLinearRegressorEvaluationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorEvaluationTest.__init__(
        self, _linear_regressor_fn)


class LinearRegressorPredictTest(
    linear_testing_utils.BaseLinearRegressorPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorPredictTest.__init__(
        self, _linear_regressor_fn)


class LinearRegressorIntegrationTest(
    linear_testing_utils.BaseLinearRegressorIntegrationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorIntegrationTest.__init__(
        self, _linear_regressor_fn)


class LinearRegressorTrainingTest(
    linear_testing_utils.BaseLinearRegressorTrainingTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorTrainingTest.__init__(
        self, _linear_regressor_fn)


class _BaseLinearClassiferTrainingTest(object):

  def __init__(self, n_classes):
    self._n_classes = n_classes
    self._logits_dimensions = (
        self._n_classes if self._n_classes > 2 else 1)

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _mock_optimizer(self, expected_loss=None):
    expected_var_names = [
        '%s/part_0:0' % linear_testing_utils.AGE_WEIGHT_NAME,
        '%s/part_0:0' % linear_testing_utils.BIAS_NAME
    ]

    def _minimize(loss, global_step):
      trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertItemsEqual(
          expected_var_names,
          [var.name for var in trainable_vars])

      # Verify loss. We can't check the value directly, so we add an assert op.
      self.assertEquals(0, loss.shape.ndims)
      if expected_loss is None:
        return state_ops.assign_add(global_step, 1).op
      assert_loss = linear_testing_utils.assert_close(
          math_ops.to_float(expected_loss, name='expected'),
          loss,
          name='assert_loss')
      with ops.control_dependencies((assert_loss,)):
        return state_ops.assign_add(global_step, 1).op

    mock_optimizer = test.mock.NonCallableMock(
        spec=optimizer.Optimizer,
        wraps=optimizer.Optimizer(use_locking=False, name='my_optimizer'))
    mock_optimizer.minimize = test.mock.MagicMock(wraps=_minimize)

    # NOTE: Estimator.params performs a deepcopy, which wreaks havoc with mocks.
    # So, return mock_optimizer itself for deepcopy.
    mock_optimizer.__deepcopy__ = lambda _: mock_optimizer
    return mock_optimizer

  def _assert_checkpoint(
      self, expected_global_step, expected_age_weight=None, expected_bias=None):
    logits_dimension = self._logits_dimensions

    shapes = {
        name: shape for (name, shape) in
        checkpoint_utils.list_variables(self._model_dir)
    }

    self.assertEqual([], shapes[ops.GraphKeys.GLOBAL_STEP])
    self.assertEqual(
        expected_global_step,
        checkpoint_utils.load_variable(
            self._model_dir, ops.GraphKeys.GLOBAL_STEP))

    self.assertEqual([1, logits_dimension],
                     shapes[linear_testing_utils.AGE_WEIGHT_NAME])
    if expected_age_weight is not None:
      self.assertAllEqual(expected_age_weight,
                          checkpoint_utils.load_variable(
                              self._model_dir,
                              linear_testing_utils.AGE_WEIGHT_NAME))

    self.assertEqual([logits_dimension], shapes[linear_testing_utils.BIAS_NAME])
    if expected_bias is not None:
      self.assertAllEqual(expected_bias,
                          checkpoint_utils.load_variable(
                              self._model_dir, linear_testing_utils.BIAS_NAME))

  def testFromScratchWithDefaultOptimizer(self):
    n_classes = self._n_classes
    label = 0
    age = 17
    est = linear.LinearClassifier(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        n_classes=n_classes,
        model_dir=self._model_dir)

    # Train for a few steps, and validate final checkpoint.
    num_steps = 10
    est.train(
        input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
    self._assert_checkpoint(num_steps)

  def testTrainWithTwoDimsLabel(self):
    n_classes = self._n_classes
    batch_size = 20

    est = linear.LinearClassifier(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        n_classes=n_classes,
        model_dir=self._model_dir)
    data_rank_1 = np.array([0, 1])
    data_rank_2 = np.array([[0], [1]])
    self.assertEqual((2,), data_rank_1.shape)
    self.assertEqual((2, 1), data_rank_2.shape)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'age': data_rank_1},
        y=data_rank_2,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    est.train(train_input_fn, steps=200)
    self._assert_checkpoint(200)

  def testTrainWithOneDimLabel(self):
    n_classes = self._n_classes
    batch_size = 20

    est = linear.LinearClassifier(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        n_classes=n_classes,
        model_dir=self._model_dir)
    data_rank_1 = np.array([0, 1])
    self.assertEqual((2,), data_rank_1.shape)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'age': data_rank_1},
        y=data_rank_1,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    est.train(train_input_fn, steps=200)
    self._assert_checkpoint(200)

  def testTrainWithTwoDimsWeight(self):
    n_classes = self._n_classes
    batch_size = 20

    est = linear.LinearClassifier(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        weight_feature_key='w',
        n_classes=n_classes,
        model_dir=self._model_dir)
    data_rank_1 = np.array([0, 1])
    data_rank_2 = np.array([[0], [1]])
    self.assertEqual((2,), data_rank_1.shape)
    self.assertEqual((2, 1), data_rank_2.shape)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'age': data_rank_1, 'w': data_rank_2}, y=data_rank_1,
        batch_size=batch_size, num_epochs=None,
        shuffle=True)
    est.train(train_input_fn, steps=200)
    self._assert_checkpoint(200)

  def testTrainWithOneDimWeight(self):
    n_classes = self._n_classes
    batch_size = 20

    est = linear.LinearClassifier(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        weight_feature_key='w',
        n_classes=n_classes,
        model_dir=self._model_dir)
    data_rank_1 = np.array([0, 1])
    self.assertEqual((2,), data_rank_1.shape)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'age': data_rank_1, 'w': data_rank_1}, y=data_rank_1,
        batch_size=batch_size, num_epochs=None,
        shuffle=True)
    est.train(train_input_fn, steps=200)
    self._assert_checkpoint(200)

  def testFromScratch(self):
    n_classes = self._n_classes
    label = 1
    age = 17
    # For binary classifer:
    #   loss = sigmoid_cross_entropy(logits, label) where logits=0 (weights are
    #   all zero initially) and label = 1 so,
    #      loss = 1 * -log ( sigmoid(logits) ) = 0.69315
    # For multi class classifer:
    #   loss = cross_entropy(logits, label) where logits are all 0s (weights are
    #   all zero initially) and label = 1 so,
    #      loss = 1 * -log ( 1.0 / n_classes )
    # For this particular test case, as logits are same, the formular
    # 1 * -log ( 1.0 / n_classes ) covers both binary and multi class cases.
    mock_optimizer = self._mock_optimizer(
        expected_loss=-1 * math.log(1.0/n_classes))

    est = linear.LinearClassifier(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        n_classes=n_classes,
        optimizer=mock_optimizer,
        model_dir=self._model_dir)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    est.train(
        input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assert_checkpoint(
        expected_global_step=num_steps,
        expected_age_weight=[[0.]] if n_classes == 2 else [[0.] * n_classes],
        expected_bias=[0.] if n_classes == 2 else [.0] * n_classes)

  def testFromCheckpoint(self):
    # Create initial checkpoint.
    n_classes = self._n_classes
    label = 1
    age = 17
    # For binary case, the expected weight has shape (1,1). For multi class
    # case, the shape is (1, n_classes). In order to test the weights, set
    # weights as 2.0 * range(n_classes).
    age_weight = [[2.0]] if n_classes == 2 else (
        np.reshape(2.0 * np.array(list(range(n_classes)), dtype=np.float32),
                   (1, n_classes)))
    bias = [-35.0] if n_classes == 2 else [-35.0] * n_classes
    initial_global_step = 100
    with ops.Graph().as_default():
      variables.Variable(age_weight, name=linear_testing_utils.AGE_WEIGHT_NAME)
      variables.Variable(bias, name=linear_testing_utils.BIAS_NAME)
      variables.Variable(
          initial_global_step, name=ops.GraphKeys.GLOBAL_STEP,
          dtype=dtypes.int64)
      linear_testing_utils.save_variables_to_ckpt(self._model_dir)

    # For binary classifer:
    #   logits = age * age_weight + bias = 17 * 2. - 35. = -1.
    #   loss = sigmoid_cross_entropy(logits, label)
    #   so, loss = 1 * -log ( sigmoid(-1) ) = 1.3133
    # For multi class classifer:
    #   loss = cross_entropy(logits, label)
    #   where logits = 17 * age_weight + bias and label = 1
    #   so, loss = 1 * -log ( soft_max(logits)[1] )
    if n_classes == 2:
      expected_loss = 1.3133
    else:
      logits = age_weight * age + bias
      logits_exp = np.exp(logits)
      softmax = logits_exp / logits_exp.sum()
      expected_loss = -1 * math.log(softmax[0, label])

    mock_optimizer = self._mock_optimizer(expected_loss=expected_loss)

    est = linear.LinearClassifier(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        n_classes=n_classes,
        optimizer=mock_optimizer,
        model_dir=self._model_dir)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    est.train(
        input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assert_checkpoint(
        expected_global_step=initial_global_step + num_steps,
        expected_age_weight=age_weight,
        expected_bias=bias)

  def testFromCheckpointMultiBatch(self):
    # Create initial checkpoint.
    n_classes = self._n_classes
    label = [1, 0]
    age = [17, 18.5]
    # For binary case, the expected weight has shape (1,1). For multi class
    # case, the shape is (1, n_classes). In order to test the weights, set
    # weights as 2.0 * range(n_classes).
    age_weight = [[2.0]] if n_classes == 2 else (
        np.reshape(2.0 * np.array(list(range(n_classes)), dtype=np.float32),
                   (1, n_classes)))
    bias = [-35.0] if n_classes == 2 else [-35.0] * n_classes
    initial_global_step = 100
    with ops.Graph().as_default():
      variables.Variable(age_weight, name=linear_testing_utils.AGE_WEIGHT_NAME)
      variables.Variable(bias, name=linear_testing_utils.BIAS_NAME)
      variables.Variable(
          initial_global_step, name=ops.GraphKeys.GLOBAL_STEP,
          dtype=dtypes.int64)
      linear_testing_utils.save_variables_to_ckpt(self._model_dir)

    # For binary classifer:
    #   logits = age * age_weight + bias
    #   logits[0] = 17 * 2. - 35. = -1.
    #   logits[1] = 18.5 * 2. - 35. = 2.
    #   loss = sigmoid_cross_entropy(logits, label)
    #   so, loss[0] = 1 * -log ( sigmoid(-1) ) = 1.3133
    #       loss[1] = (1 - 0) * -log ( 1- sigmoid(2) ) = 2.1269
    # For multi class classifer:
    #   loss = cross_entropy(logits, label)
    #   where logits = [17, 18.5] * age_weight + bias and label = [1, 0]
    #   so, loss = 1 * -log ( soft_max(logits)[label] )
    if n_classes == 2:
      expected_loss = (1.3133 + 2.1269)
    else:
      logits = age_weight * np.reshape(age, (2, 1)) + bias
      logits_exp = np.exp(logits)
      softmax_row_0 = logits_exp[0] / logits_exp[0].sum()
      softmax_row_1 = logits_exp[1] / logits_exp[1].sum()
      expected_loss_0 = -1 * math.log(softmax_row_0[label[0]])
      expected_loss_1 = -1 * math.log(softmax_row_1[label[1]])
      expected_loss = expected_loss_0 + expected_loss_1

    mock_optimizer = self._mock_optimizer(expected_loss=expected_loss)

    est = linear.LinearClassifier(
        feature_columns=(feature_column_lib.numeric_column('age'),),
        n_classes=n_classes,
        optimizer=mock_optimizer,
        model_dir=self._model_dir)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    est.train(
        input_fn=lambda: ({'age': (age)}, (label)),
        steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assert_checkpoint(
        expected_global_step=initial_global_step + num_steps,
        expected_age_weight=age_weight,
        expected_bias=bias)


class LinearClassiferWithBinaryClassesTrainingTest(
    _BaseLinearClassiferTrainingTest, test.TestCase):

  def __init__(self, methodName='runTest'):
    test.TestCase.__init__(self, methodName)
    _BaseLinearClassiferTrainingTest.__init__(self, n_classes=2)


class LinearClassiferWithMultiClassesTrainingTest(
    _BaseLinearClassiferTrainingTest, test.TestCase):

  def __init__(self, methodName='runTest'):
    test.TestCase.__init__(self, methodName)
    _BaseLinearClassiferTrainingTest.__init__(self, n_classes=4)


if __name__ == '__main__':
  test.main()

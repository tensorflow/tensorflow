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
"""Tests for baseline.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import numpy as np
import six

from tensorflow.contrib.estimator.python.estimator import baseline
from tensorflow.contrib.estimator.python.estimator import head as head_lib
from tensorflow.python.client import session as tf_session
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import optimizer
from tensorflow.python.training import saver

# Names of variables created by model.
BIAS_NAME = 'baseline/bias'


def assert_close(expected, actual, rtol=1e-04, name='assert_close'):
  with ops.name_scope(name, 'assert_close', (expected, actual, rtol)) as scope:
    expected = ops.convert_to_tensor(expected, name='expected')
    actual = ops.convert_to_tensor(actual, name='actual')
    rdiff = math_ops.abs(expected - actual, 'diff') / math_ops.abs(expected)
    rtol = ops.convert_to_tensor(rtol, name='rtol')
    return check_ops.assert_less(
        rdiff,
        rtol,
        data=('Condition expected =~ actual did not hold element-wise:'
              'expected = ', expected, 'actual = ', actual, 'rdiff = ', rdiff,
              'rtol = ', rtol,),
        name=scope)


def save_variables_to_ckpt(model_dir):
  init_all_op = [variables.global_variables_initializer()]
  with tf_session.Session() as sess:
    sess.run(init_all_op)
    saver.Saver().save(sess, os.path.join(model_dir, 'model.ckpt'))


def _baseline_estimator_fn(
    weight_column=None, label_dimension=1, *args, **kwargs):
  """Returns a BaselineEstimator that uses regression_head."""
  return baseline.BaselineEstimator(
      head=head_lib.regression_head(
          weight_column=weight_column, label_dimension=label_dimension,
          # Tests in core (from which this test inherits) test the sum loss.
          loss_reduction=losses.Reduction.SUM),
      *args, **kwargs)


class BaselineEstimatorEvaluationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def test_evaluation_batch(self):
    """Tests evaluation for batch_size==2."""
    with ops.Graph().as_default():
      variables.Variable([13.0], name=BIAS_NAME)
      variables.Variable(
          100, name=ops.GraphKeys.GLOBAL_STEP, dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    baseline_estimator = _baseline_estimator_fn(model_dir=self._model_dir)
    eval_metrics = baseline_estimator.evaluate(
        input_fn=lambda: ({'age': ((1,), (1,))}, ((10.,), (10.,))), steps=1)

    # Logit is bias = 13, while label is 10.
    # Loss per example is 3**2 = 9.
    # Training loss is the sum over batch = 9 + 9 = 18
    # Average loss is the average over batch = 9
    self.assertDictEqual({
        metric_keys.MetricKeys.LOSS: 18.,
        metric_keys.MetricKeys.LOSS_MEAN: 9.,
        ops.GraphKeys.GLOBAL_STEP: 100
    }, eval_metrics)

  def test_evaluation_weights(self):
    """Tests evaluation with weights."""
    with ops.Graph().as_default():
      variables.Variable([13.0], name=BIAS_NAME)
      variables.Variable(
          100, name=ops.GraphKeys.GLOBAL_STEP, dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    def _input_fn():
      features = {'age': ((1,), (1,)), 'weights': ((1.,), (2.,))}
      labels = ((10.,), (10.,))
      return features, labels

    baseline_estimator = _baseline_estimator_fn(
        weight_column='weights',
        model_dir=self._model_dir)
    eval_metrics = baseline_estimator.evaluate(input_fn=_input_fn, steps=1)

    # Logit is bias = 13, while label is 10.
    # Loss per example is 3**2 = 9.
    # Training loss is the weighted sum over batch = 9 + 2*9 = 27
    # average loss is the weighted average = 9 + 2*9 / (1 + 2) = 9
    self.assertDictEqual({
        metric_keys.MetricKeys.LOSS: 27.,
        metric_keys.MetricKeys.LOSS_MEAN: 9.,
        ops.GraphKeys.GLOBAL_STEP: 100
    }, eval_metrics)

  def test_evaluation_for_multi_dimensions(self):
    label_dim = 2
    with ops.Graph().as_default():
      variables.Variable([46.0, 58.0], name=BIAS_NAME)
      variables.Variable(100, name='global_step', dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    baseline_estimator = _baseline_estimator_fn(
        label_dimension=label_dim,
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        x={
            'age': np.array([[2., 4., 5.]]),
        },
        y=np.array([[46., 58.]]),
        batch_size=1,
        num_epochs=None,
        shuffle=False)
    eval_metrics = baseline_estimator.evaluate(input_fn=input_fn, steps=1)

    self.assertItemsEqual(
        (metric_keys.MetricKeys.LOSS, metric_keys.MetricKeys.LOSS_MEAN,
         ops.GraphKeys.GLOBAL_STEP), eval_metrics.keys())

    # Logit is bias which is [46, 58]
    self.assertAlmostEqual(0, eval_metrics[metric_keys.MetricKeys.LOSS])


class BaselineEstimatorPredictTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def test_1d(self):
    """Tests predict when all variables are one-dimensional."""
    with ops.Graph().as_default():
      variables.Variable([.2], name=BIAS_NAME)
      variables.Variable(100, name='global_step', dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    baseline_estimator = _baseline_estimator_fn(model_dir=self._model_dir)

    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': np.array([[2.]])},
        y=None,
        batch_size=1,
        num_epochs=1,
        shuffle=False)
    predictions = baseline_estimator.predict(input_fn=predict_input_fn)
    predicted_scores = list([x['predictions'] for x in predictions])
    # x * weight + bias = 2. * 10. + .2 = 20.2
    self.assertAllClose([[.2]], predicted_scores)

  def testMultiDim(self):
    """Tests predict when all variables are multi-dimenstional."""
    batch_size = 2
    label_dimension = 3
    with ops.Graph().as_default():
      variables.Variable(  # shape=[label_dimension]
          [.2, .4, .6], name=BIAS_NAME)
      variables.Variable(100, name='global_step', dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    baseline_estimator = _baseline_estimator_fn(
        label_dimension=label_dimension,
        model_dir=self._model_dir)

    predict_input_fn = numpy_io.numpy_input_fn(
        # x shape=[batch_size, x_dim]
        x={'x': np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])},
        y=None,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)
    predictions = baseline_estimator.predict(input_fn=predict_input_fn)
    predicted_scores = list([x['predictions'] for x in predictions])
    # score = bias, shape=[batch_size, label_dimension]
    self.assertAllClose([[0.2, 0.4, 0.6], [0.2, 0.4, 0.6]],
                        predicted_scores)


class BaselineEstimatorIntegrationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(self, train_input_fn, eval_input_fn, predict_input_fn,
                          input_dimension, label_dimension, prediction_length):
    feature_columns = [
        feature_column_lib.numeric_column('x', shape=(input_dimension,))
    ]
    est = _baseline_estimator_fn(
        label_dimension=label_dimension,
        model_dir=self._model_dir)

    # TRAIN
    # learn y = x
    est.train(train_input_fn, steps=200)

    # EVALUTE
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(200, scores[ops.GraphKeys.GLOBAL_STEP])
    self.assertIn(metric_keys.MetricKeys.LOSS, six.iterkeys(scores))

    # PREDICT
    predictions = np.array(
        [x['predictions'] for x in est.predict(predict_input_fn)])
    self.assertAllEqual((prediction_length, label_dimension), predictions.shape)

    # EXPORT
    feature_spec = feature_column_lib.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir = est.export_savedmodel(tempfile.mkdtemp(),
                                       serving_input_receiver_fn)
    self.assertTrue(gfile.Exists(export_dir))

  def test_numpy_input_fn(self):
    """Tests complete flow with numpy_input_fn."""
    label_dimension = 2
    input_dimension = label_dimension
    batch_size = 10
    prediction_length = batch_size
    data = np.linspace(0., 2., batch_size * label_dimension, dtype=np.float32)
    data = data.reshape(batch_size, label_dimension)

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
        num_epochs=1,
        shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=None,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=input_dimension,
        label_dimension=label_dimension,
        prediction_length=prediction_length)


class BaselineEstimatorTrainingTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _mock_optimizer(self, expected_loss=None):
    expected_var_names = [
        '%s:0' % BIAS_NAME
    ]

    def _minimize(loss, global_step=None, var_list=None):
      trainable_vars = var_list or ops.get_collection(
          ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertItemsEqual(expected_var_names,
                            [var.name for var in trainable_vars])

      # Verify loss. We can't check the value directly, so we add an assert op.
      self.assertEquals(0, loss.shape.ndims)
      if expected_loss is None:
        if global_step is not None:
          return distribute_lib.increment_var(global_step)
        return control_flow_ops.no_op()
      assert_loss = assert_close(
          math_ops.to_float(expected_loss, name='expected'),
          loss,
          name='assert_loss')
      with ops.control_dependencies((assert_loss,)):
        if global_step is not None:
          return distribute_lib.increment_var(global_step)
        return control_flow_ops.no_op()

    mock_optimizer = test.mock.NonCallableMock(
        spec=optimizer.Optimizer,
        wraps=optimizer.Optimizer(use_locking=False, name='my_optimizer'))
    mock_optimizer.minimize = test.mock.MagicMock(wraps=_minimize)

    # NOTE: Estimator.params performs a deepcopy, which wreaks havoc with mocks.
    # So, return mock_optimizer itself for deepcopy.
    mock_optimizer.__deepcopy__ = lambda _: mock_optimizer
    return mock_optimizer

  def _assert_checkpoint(self,
                         label_dimension,
                         expected_global_step,
                         expected_bias=None):
    shapes = {
        name: shape
        for (name, shape) in checkpoint_utils.list_variables(self._model_dir)
    }

    self.assertEqual([], shapes[ops.GraphKeys.GLOBAL_STEP])
    self.assertEqual(expected_global_step,
                     checkpoint_utils.load_variable(self._model_dir,
                                                    ops.GraphKeys.GLOBAL_STEP))

    self.assertEqual([label_dimension], shapes[BIAS_NAME])
    if expected_bias is not None:
      self.assertEqual(expected_bias,
                       checkpoint_utils.load_variable(self._model_dir,
                                                      BIAS_NAME))

  def testFromScratch(self):
    # Create BaselineRegressor.
    label = 5.
    age = 17
    # loss = (logits - label)^2 = (0 - 5.)^2 = 25.
    mock_optimizer = self._mock_optimizer(expected_loss=25.)
    baseline_estimator = _baseline_estimator_fn(
        model_dir=self._model_dir,
        optimizer=mock_optimizer)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    baseline_estimator.train(
        input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assert_checkpoint(
        label_dimension=1,
        expected_global_step=num_steps,
        expected_bias=[0.])

  def testFromCheckpoint(self):
    # Create initial checkpoint.
    bias = 7.0
    initial_global_step = 100
    with ops.Graph().as_default():
      variables.Variable([bias], name=BIAS_NAME)
      variables.Variable(
          initial_global_step,
          name=ops.GraphKeys.GLOBAL_STEP,
          dtype=dtypes.int64)
      save_variables_to_ckpt(self._model_dir)

    # logits = bias = 6.
    # loss = (logits - label)^2 = (7 - 5)^2 = 4
    mock_optimizer = self._mock_optimizer(expected_loss=4.)
    baseline_estimator = _baseline_estimator_fn(
        model_dir=self._model_dir,
        optimizer=mock_optimizer)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, and validate optimizer and final checkpoint.
    num_steps = 10
    baseline_estimator.train(
        input_fn=lambda: ({'age': ((17,),)}, ((5.,),)), steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assert_checkpoint(
        label_dimension=1,
        expected_global_step=initial_global_step + num_steps,
        expected_bias=[bias])


if __name__ == '__main__':
  test.main()

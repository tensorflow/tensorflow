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

from tensorflow.core.framework import summary_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.estimator.canned import dnn
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary import summary as summary_lib
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import optimizer
from tensorflow.python.training import saver
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

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
    # TODO(ptucker): We shouldn't have this in the checkpoint for constant LRs.
    # Learning rate.
    variables_lib.Variable(.5, name=_LEARNING_RATE_NAME, dtype=dtypes.float32)

    # Initialize vars and save checkpoint.
    with tf_session.Session() as sess:
      variables_lib.global_variables_initializer().run()
      global_step_var.assign(global_step).eval()
      saver.Saver().save(sess, os.path.join(model_dir, 'model.ckpt'))


class DNNRegressorEvaluateTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def test_simple(self):
    # Create checkpoint: num_inputs=1, hidden_units=(2, 2), num_outputs=1.
    global_step = 100
    _create_checkpoint((
        (((1., 2.),), (3., 4.)),
        (((5., 6.), (7., 8.),), (9., 10.)),
        (((11.,), (12.,),), (13.,))
    ), global_step, self._model_dir)

    # Create DNNRegressor and evaluate.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=(feature_column.numeric_column('age'),),
        model_dir=self._model_dir)
    def _input_fn():
      return {'age': ((1,),)}, ((10.,),)
    # TODO(ptucker): Point to tool for calculating a neural net output?
    # prediction = 1778
    # loss = (10-1778)^2 = 3125824
    expected_loss = 3125824
    self.assertAllClose({
        metric_keys.MetricKeys.LOSS: expected_loss,
        metric_keys.MetricKeys.LOSS_MEAN: expected_loss,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_regressor.evaluate(input_fn=_input_fn, steps=1))

  def test_weighted(self):
    # Create checkpoint: num_inputs=1, hidden_units=(2, 2), num_outputs=1.
    global_step = 100
    _create_checkpoint((
        (((1., 2.),), (3., 4.)),
        (((5., 6.), (7., 8.),), (9., 10.)),
        (((11.,), (12.,),), (13.,))
    ), global_step, self._model_dir)

    # Create DNNRegressor and evaluate.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=(feature_column.numeric_column('age'),),
        model_dir=self._model_dir,
        weight_feature_key='label_weight')
    def _input_fn():
      return {'age': ((1,),), 'label_weight': ((1.5,),)}, ((10.,),)
    self.assertAllClose({
        # TODO(ptucker): Point to tool for calculating a neural net output?
        # prediction = 1778
        # loss = 1.5*((10-1778)^2) = 4688736
        metric_keys.MetricKeys.LOSS: 4688736,
        # average_loss = loss / 1.5 = 3125824
        metric_keys.MetricKeys.LOSS_MEAN: 3125824,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_regressor.evaluate(input_fn=_input_fn, steps=1))

  def test_multi_example(self):
    # Create initial checkpoint, 1 input, 2x2 hidden dims, 1 outputs.
    global_step = 100
    _create_checkpoint((
        (((1., 2.),), (3., 4.)),
        (((5., 6.), (7., 8.),), (9., 10.)),
        (((11.,), (12.,),), (13.,))
    ), global_step, self._model_dir)

    # Create DNNRegressor and evaluate.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=(feature_column.numeric_column('age'),),
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        x={'age': np.array(((1,), (2,), (3,)))},
        y=np.array(((10,), (9,), (8,))),
        batch_size=3,
        shuffle=False)
    self.assertAllClose({
        # TODO(ptucker): Point to tool for calculating a neural net output?
        # predictions = 1778, 2251, 2724
        # loss = ((10-1778)^2 + (9-2251)^2 + (8-2724)^2) = 15529044
        metric_keys.MetricKeys.LOSS: 15529044.,
        # average_loss = loss / 3 = 5176348
        metric_keys.MetricKeys.LOSS_MEAN: 5176348.,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_regressor.evaluate(input_fn=input_fn, steps=1))

  def test_multi_batch(self):
    # Create checkpoint: num_inputs=1, hidden_units=(2, 2), num_outputs=1.
    global_step = 100
    _create_checkpoint((
        (((1., 2.),), (3., 4.)),
        (((5., 6.), (7., 8.),), (9., 10.)),
        (((11.,), (12.,),), (13.,))
    ), global_step, self._model_dir)

    # Create DNNRegressor and evaluate.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=(feature_column.numeric_column('age'),),
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        x={'age': np.array(((1,), (2,), (3,)))},
        y=np.array(((10,), (9,), (8,))),
        batch_size=1,
        shuffle=False)
    # TODO(ptucker): Point to tool for calculating a neural net output?
    # predictions = 1778, 2251, 2724
    # loss = ((10-1778)^2 + (9-2251)^2 + (8-2724)^2) / 3 = 5176348
    expected_loss = 5176348.
    self.assertAllClose({
        metric_keys.MetricKeys.LOSS: expected_loss,
        metric_keys.MetricKeys.LOSS_MEAN: expected_loss,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_regressor.evaluate(input_fn=input_fn, steps=3))

  def test_weighted_multi_example(self):
    # Create checkpoint: num_inputs=4, hidden_units=(2, 2), num_outputs=3.
    global_step = 100
    _create_checkpoint((
        (((1., 2.), (3., 4.), (5., 6.), (7., 8.),), (9., 8.)),
        (((7., 6.), (5., 4.),), (3., 2.)),
        (((1., 2., 3.), (4., 5., 6.),), (7., 8., 9.)),
    ), global_step, self._model_dir)

    # Create batched input.
    input_fn = numpy_io.numpy_input_fn(
        x={
            # Dimensions are (batch_size, feature_column.dimension).
            'x': np.array((
                (15., 0., 1.5, 135.2),
                (45., 45000., 1.8, 158.8),
                (21., 33000., 1.7, 207.1),
                (60., 10000., 1.6, 90.2)
            )),
            # TODO(ptucker): Add test for different weight shapes when we fix
            # head._compute_weighted_loss (currently it requires weights to be
            # same shape as labels & logits).
            'label_weight': np.array((
                (1., 1., 0.),
                (.5, 1., .1),
                (.5, 0., .9),
                (0., 0., 0.),
            ))
        },
        # Label shapes is (batch_size, num_outputs).
        y=np.array((
            (5., 2., 2.),
            (-2., 1., -4.),
            (-1., -1., -1.),
            (-4., 3., 9.),
        )),
        batch_size=4,
        shuffle=False)

    # Create DNNRegressor and evaluate.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=(
            # Dimension is number of inputs.
            feature_column.numeric_column(
                'x', dtype=dtypes.int32, shape=(4,)),
        ),
        model_dir=self._model_dir,
        label_dimension=3,
        weight_feature_key='label_weight')
    self.assertAllClose({
        # TODO(ptucker): Point to tool for calculating a neural net output?
        # predictions = [
        #   [  54033.5    76909.6    99785.7]
        #   [8030393.8 11433082.4 14835771.0]
        #   [5923209.2  8433014.8 10942820.4]
        #   [1810021.6  2576969.6  3343917.6]
        # ]
        # loss = sum(label_weights*(labels-predictions)^2) = 3.10290850204e+14
        metric_keys.MetricKeys.LOSS: 3.10290850204e+14,
        # average_loss = loss / sum(label_weights) = 3.10290850204e+14 / 5.
        #              = 6.205817e+13
        metric_keys.MetricKeys.LOSS_MEAN: 6.205817e+13,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_regressor.evaluate(input_fn=input_fn, steps=1))

  def test_weighted_multi_example_multi_column(self):
    # Create checkpoint: num_inputs=4, hidden_units=(2, 2), num_outputs=3.
    global_step = 100
    _create_checkpoint((
        (((1., 2.), (3., 4.), (5., 6.), (7., 8.),), (9., 8.)),
        (((7., 6.), (5., 4.),), (3., 2.)),
        (((1., 2., 3.), (4., 5., 6.),), (7., 8., 9.)),
    ), global_step, self._model_dir)

    # Create batched input.
    input_fn = numpy_io.numpy_input_fn(
        x={
            # Dimensions are (batch_size, feature_column.dimension).
            'x': np.array((
                (15., 0.),
                (45., 45000.),
                (21., 33000.),
                (60., 10000.)
            )),
            'y': np.array((
                (1.5, 135.2),
                (1.8, 158.8),
                (1.7, 207.1),
                (1.6, 90.2)
            )),
            # TODO(ptucker): Add test for different weight shapes when we fix
            # head._compute_weighted_loss (currently it requires weights to be
            # same shape as labels & logits).
            'label_weight': np.array((
                (1., 1., 0.),
                (.5, 1., .1),
                (.5, 0., .9),
                (0., 0., 0.),
            ))
        },
        # Label shapes is (batch_size, num_outputs).
        y=np.array((
            (5., 2., 2.),
            (-2., 1., -4.),
            (-1., -1., -1.),
            (-4., 3., 9.),
        )),
        batch_size=4,
        shuffle=False)

    # Create DNNRegressor and evaluate.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=(
            # Dimensions add up to 4 (number of inputs).
            feature_column.numeric_column(
                'x', dtype=dtypes.int32, shape=(2,)),
            feature_column.numeric_column(
                'y', dtype=dtypes.float32, shape=(2,)),
        ),
        model_dir=self._model_dir,
        label_dimension=3,
        weight_feature_key='label_weight')
    self.assertAllClose({
        # TODO(ptucker): Point to tool for calculating a neural net output?
        # predictions = [
        #   [  54033.5    76909.6    99785.7]
        #   [8030393.8 11433082.4 14835771.0]
        #   [5923209.2  8433014.8 10942820.4]
        #   [1810021.6  2576969.6  3343917.6]
        # ]
        # loss = sum(label_weights*(labels-predictions)^2) = 3.10290850204e+14
        metric_keys.MetricKeys.LOSS: 3.10290850204e+14,
        # average_loss = loss / sum(label_weights) = 3.10290850204e+14 / 5.
        #              = 6.205817e+13
        metric_keys.MetricKeys.LOSS_MEAN: 6.205817e+13,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_regressor.evaluate(input_fn=input_fn, steps=1))

  def test_weighted_multi_batch(self):
    # Create checkpoint: num_inputs=4, hidden_units=(2, 2), num_outputs=3.
    global_step = 100
    _create_checkpoint((
        (((1., 2.), (3., 4.), (5., 6.), (7., 8.),), (9., 8.)),
        (((7., 6.), (5., 4.),), (3., 2.)),
        (((1., 2., 3.), (4., 5., 6.),), (7., 8., 9.)),
    ), global_step, self._model_dir)

    # Create batched input.
    input_fn = numpy_io.numpy_input_fn(
        x={
            # Dimensions are (batch_size, feature_column.dimension).
            'x': np.array((
                (15., 0., 1.5, 135.2),
                (45., 45000., 1.8, 158.8),
                (21., 33000., 1.7, 207.1),
                (60., 10000., 1.6, 90.2)
            )),
            # TODO(ptucker): Add test for different weight shapes when we fix
            # head._compute_weighted_loss (currently it requires weights to be
            # same shape as labels & logits).
            'label_weights': np.array((
                (1., 1., 0.),
                (.5, 1., .1),
                (.5, 0., .9),
                (0., 0., 0.),
            ))
        },
        # Label shapes is (batch_size, num_outputs).
        y=np.array((
            (5., 2., 2.),
            (-2., 1., -4.),
            (-1., -1., -1.),
            (-4., 3., 9.),
        )),
        batch_size=1,
        shuffle=False)

    # Create DNNRegressor and evaluate.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=(
            # Dimension is number of inputs.
            feature_column.numeric_column(
                'x', dtype=dtypes.int32, shape=(4,)),
        ),
        model_dir=self._model_dir,
        label_dimension=3,
        weight_feature_key='label_weights')
    self.assertAllClose({
        # TODO(ptucker): Point to tool for calculating a neural net output?
        # predictions = [
        #   [  54033.5    76909.6    99785.7]
        #   [8030393.8 11433082.4 14835771.0]
        #   [5923209.2  8433014.8 10942820.4]
        #   [1810021.6  2576969.6  3343917.6]
        # ]
        # losses = label_weights*(labels-predictions)^2 = [
        #  [  2.91907881e+09   5.91477894e+09                0]
        #  [  3.22436284e+13   1.30715350e+14   2.20100220e+13]
        #  [  1.75422095e+13                0   1.07770806e+14]
        #  [               0                0                0]
        # ]
        # total_loss = sum(losses) = 3.10290850204e+14
        # loss = total_loss / 4 = 7.7572712551e+13
        metric_keys.MetricKeys.LOSS: 7.7572712551e+13,
        # average_loss = total_loss / sum(label_weights) = 6.20581700408e+13
        metric_keys.MetricKeys.LOSS_MEAN: 6.20581700408e+13,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_regressor.evaluate(input_fn=input_fn, steps=4))

  def test_multi_dim(self):
    # Create checkpoint: num_inputs=3, hidden_units=(2, 2), num_outputs=2.
    global_step = 100
    _create_checkpoint((
        (((1., 2.), (3., 4.), (5., 6.),), (7., 8.)),
        (((9., 8.), (7., 6.),), (5., 4.)),
        (((3., 2.), (1., 2.),), (3., 4.)),
    ), global_step, self._model_dir)

    # Create DNNRegressor and evaluate.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=(feature_column.numeric_column('x', shape=(3,)),),
        label_dimension=2,
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        x={'x': np.array(((2., 4., 5.),))},
        y=np.array(((46., 58.),)),
        batch_size=1,
        shuffle=False)
    self.assertAllClose({
        # TODO(ptucker): Point to tool for calculating a neural net output?
        # predictions = 3198, 3094
        # loss = ((46-3198)^2 + (58-3094)^2) = 19152400
        metric_keys.MetricKeys.LOSS: 19152400,
        # average_loss = loss / 2 = 9576200
        metric_keys.MetricKeys.LOSS_MEAN: 9576200,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_regressor.evaluate(input_fn=input_fn, steps=1))

  def test_multi_feature_column(self):
    # Create checkpoint: num_inputs=2, hidden_units=(2, 2), num_outputs=1.
    global_step = 100
    _create_checkpoint((
        (((1., 2.), (3., 4.),), (5., 6.)),
        (((7., 8.), (9., 8.),), (7., 6.)),
        (((5.,), (4.,),), (3.,))
    ), global_step, self._model_dir)

    # Create DNNRegressor and evaluate.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=(feature_column.numeric_column('age'),
                         feature_column.numeric_column('height')),
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        x={'age': np.array(((20,), (40,))), 'height': np.array(((4,), (8,)))},
        y=np.array(((213.,), (421.,))),
        batch_size=2,
        shuffle=False)
    self.assertAllClose({
        # TODO(ptucker): Point to tool for calculating a neural net output?
        # predictions = 7315, 13771
        # loss = ((213-7315)^2 + (421-13771)^2) / 2 = 228660896
        metric_keys.MetricKeys.LOSS: 228660896.,
        # average_loss = loss / 2 = 114330452
        metric_keys.MetricKeys.LOSS_MEAN: 114330452.,
        ops.GraphKeys.GLOBAL_STEP: global_step
    }, dnn_regressor.evaluate(input_fn=input_fn, steps=1))


class DNNRegressorPredictTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def test_1d(self):
    """Tests predict when all variables are one-dimensional."""
    # Create checkpoint: num_inputs=1, hidden_units=(2, 2), num_outputs=1.
    _create_checkpoint((
        (((1., 2.),), (3., 4.)),
        (((5., 6.), (7., 8.),), (9., 10.)),
        (((11.,), (12.,),), (13.,))
    ), global_step=0, model_dir=self._model_dir)

    # Create DNNRegressor and predict.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=(feature_column.numeric_column('x'),),
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        x={'x': np.array(((1.,),))}, batch_size=1, shuffle=False)
    # TODO(ptucker): Point to tool for calculating a neural net output?
    # prediction = 1778
    self.assertAllClose({
        prediction_keys.PredictionKeys.PREDICTIONS: (1778.,)
    }, next(dnn_regressor.predict(input_fn=input_fn)))

  def test_multi_dim(self):
    """Tests predict when all variables are multi-dimenstional."""
    # Create checkpoint: num_inputs=4, hidden_units=(2, 2), num_outputs=3.
    _create_checkpoint((
        (((1., 2.), (3., 4.), (5., 6.), (7., 8.),), (9., 8.)),
        (((7., 6.), (5., 4.),), (3., 2.)),
        (((1., 2., 3.), (4., 5., 6.),), (7., 8., 9.)),
    ), 100, self._model_dir)

    # Create DNNRegressor and predict.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=(feature_column.numeric_column('x', shape=(4,)),),
        label_dimension=3,
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        # Inputs shape is (batch_size, num_inputs).
        x={'x': np.array(((1., 2., 3., 4.), (5., 6., 7., 8.)))},
        batch_size=2,
        shuffle=False)
    # Output shape=(batch_size, num_outputs).
    self.assertAllClose((
        # TODO(ptucker): Point to tool for calculating a neural net output?
        (3275., 4660., 6045.),
        (6939., 9876., 12813.)
    ), tuple([
        x[prediction_keys.PredictionKeys.PREDICTIONS]
        for x in dnn_regressor.predict(input_fn=input_fn)
    ]), rtol=1e-04)

  def test_two_feature_columns(self):
    """Tests predict with two feature columns."""
    # Create checkpoint: num_inputs=2, hidden_units=(2, 2), num_outputs=1.
    _create_checkpoint((
        (((1., 2.), (3., 4.),), (5., 6.)),
        (((7., 8.), (9., 8.),), (7., 6.)),
        (((5.,), (4.,),), (3.,))
    ), 100, self._model_dir)

    # Create DNNRegressor and predict.
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=(feature_column.numeric_column('x'),
                         feature_column.numeric_column('y')),
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        x={'x': np.array((20.,)), 'y': np.array((4.,))},
        batch_size=1,
        shuffle=False)
    self.assertAllClose({
        # TODO(ptucker): Point to tool for calculating a neural net output?
        # predictions = 7315
        prediction_keys.PredictionKeys.PREDICTIONS: (7315,)
    }, next(dnn_regressor.predict(input_fn=input_fn)))


class DNNRegressorIntegrationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def test_complete_flow(self):
    label_dimension = 2
    batch_size = 10
    feature_columns = [feature_column.numeric_column('x', shape=(2,))]
    est = dnn.DNNRegressor(
        hidden_units=(2, 2),
        feature_columns=feature_columns,
        label_dimension=label_dimension,
        model_dir=self._model_dir)
    data = np.linspace(0., 2., batch_size * label_dimension, dtype=np.float32)
    data = data.reshape(batch_size, label_dimension)

    # TRAIN
    # learn y = x
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    num_steps = 200
    est.train(train_input_fn, steps=num_steps)

    # EVALUTE
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        shuffle=False)
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(num_steps, scores[ops.GraphKeys.GLOBAL_STEP])
    self.assertIn('loss', six.iterkeys(scores))

    # PREDICT
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        batch_size=batch_size,
        shuffle=False)
    predictions = np.array([
        x[prediction_keys.PredictionKeys.PREDICTIONS]
        for x in est.predict(predict_input_fn)
    ])
    self.assertAllEqual((batch_size, label_dimension), predictions.shape)
    # TODO(ptucker): Deterministic test for predicted values?

    # EXPORT
    feature_spec = feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir = est.export_savedmodel(tempfile.mkdtemp(),
                                       serving_input_receiver_fn)
    self.assertTrue(gfile.Exists(export_dir))


def _full_var_name(var_name):
  return '%s/part_0:0' % var_name


def _assert_close(expected, actual, rtol=1e-04, name='assert_close'):
  with ops.name_scope(name, 'assert_close', (expected, actual, rtol)) as scope:
    expected = ops.convert_to_tensor(expected, name='expected')
    actual = ops.convert_to_tensor(actual, name='actual')
    rdiff = math_ops.abs(expected - actual, 'diff') / expected
    rtol = ops.convert_to_tensor(rtol, name='rtol')
    return check_ops.assert_less(
        rdiff,
        rtol,
        data=(
            'Condition expected =~ actual did not hold element-wise:'
            'expected = ', expected,
            'actual = ', actual,
            'rdiff = ', rdiff,
            'rtol = ', rtol,
        ),
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

  def test_simple(self):
    base_global_step = 100
    hidden_units = (2, 2)
    _create_checkpoint((
        (((1., 2.),), (3., 4.)),
        (((5., 6.), (7., 8.),), (9., 10.)),
        (((11.,), (12.,),), (13.,))
    ), base_global_step, self._model_dir)

    # Create DNNRegressor with mock optimizer.
    # TODO(ptucker): Point to tool for calculating a neural net output?
    # prediction = 1778
    # loss = (10-1778)^2 = 3125824
    expected_loss = 3125824.
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
        input_fn=lambda: ({'age': ((1,),)}, ((10.,),)), steps=num_steps,
        hooks=(summary_hook,))
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    summaries = summary_hook.summaries()
    self.assertEqual(num_steps, len(summaries))
    for summary in summaries:
      _assert_simple_summary(
          self,
          {
              metric_keys.MetricKeys.LOSS_MEAN: expected_loss,
              'dnn/dnn/hiddenlayer_0_activation': 0.,
              'dnn/dnn/hiddenlayer_0_fraction_of_zero_values': 0.,
              'dnn/dnn/hiddenlayer_1_activation': 0.,
              'dnn/dnn/hiddenlayer_1_fraction_of_zero_values': 0.,
              'dnn/dnn/logits_activation': 0.,
              'dnn/dnn/logits_fraction_of_zero_values': 0.,
              metric_keys.MetricKeys.LOSS: expected_loss,
          },
          summary)
    _assert_checkpoint(
        self, base_global_step + num_steps, input_units=1,
        hidden_units=hidden_units, output_units=1, model_dir=self._model_dir)

  def test_activation_fn(self):
    base_global_step = 100
    hidden_units = (2, 2)
    _create_checkpoint((
        (((1., 2.),), (3., 4.)),
        (((5., 6.), (7., 8.),), (9., 10.)),
        (((11.,), (12.,),), (13.,))
    ), base_global_step, self._model_dir)

    # Create DNNRegressor with mock optimizer.
    # TODO(ptucker): Point to tool for calculating a neural net output?
    # prediction = 36
    # loss = (10-36)^2 = 676
    expected_loss = 676.
    mock_optimizer = _mock_optimizer(
        self, hidden_units=hidden_units, expected_loss=expected_loss)
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=hidden_units,
        feature_columns=(feature_column.numeric_column('age'),),
        optimizer=mock_optimizer,
        model_dir=self._model_dir,
        activation_fn=nn.tanh)
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Train for a few steps, then validate optimizer, summaries, and
    # checkpoint.
    num_steps = 5
    summary_hook = _SummaryHook()
    dnn_regressor.train(
        input_fn=lambda: ({'age': ((1,),)}, ((10.,),)), steps=num_steps,
        hooks=(summary_hook,))
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    summaries = summary_hook.summaries()
    self.assertEqual(num_steps, len(summaries))
    for summary in summaries:
      _assert_simple_summary(
          self,
          {
              metric_keys.MetricKeys.LOSS: expected_loss,
              metric_keys.MetricKeys.LOSS_MEAN: expected_loss,
              'dnn/dnn/hiddenlayer_0_activation': 0.,
              'dnn/dnn/hiddenlayer_0_fraction_of_zero_values': 0.,
              'dnn/dnn/hiddenlayer_1_activation': 0.,
              'dnn/dnn/hiddenlayer_1_fraction_of_zero_values': 0.,
              'dnn/dnn/logits_activation': 0.,
              'dnn/dnn/logits_fraction_of_zero_values': 0.,
          },
          summary)
    _assert_checkpoint(
        self, base_global_step + num_steps, input_units=1,
        hidden_units=hidden_units, output_units=1, model_dir=self._model_dir)

  def test_weighted_multi_example_multi_column(self):
    hidden_units = (2, 2)
    base_global_step = 100
    _create_checkpoint((
        (((1., 2.), (3., 4.), (5., 6.), (7., 8.),), (9., 8.)),
        (((7., 6.), (5., 4.),), (3., 2.)),
        (((1., 2., 3.), (4., 5., 6.),), (7., 8., 9.)),
    ), base_global_step, self._model_dir)

    # Create DNNRegressor with mock optimizer.
    # TODO(ptucker): Point to tool for calculating a neural net output?
    # predictions = [
    #   [  54033.5    76909.6    99785.7]
    #   [8030393.8 11433082.4 14835771.0]
    #   [5923209.2  8433014.8 10942820.4]
    #   [1810021.6  2576969.6  3343917.6]
    # ]
    # loss = sum(label_weights*(labels-predictions)^2) = 3.10290850204e+14
    expected_loss = 3.10290850204e+14
    mock_optimizer = _mock_optimizer(
        self, hidden_units=hidden_units, expected_loss=expected_loss)
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=hidden_units,
        feature_columns=(
            # Dimensions add up to 4 (number of inputs).
            feature_column.numeric_column(
                'x', dtype=dtypes.int32, shape=(2,)),
            feature_column.numeric_column(
                'y', dtype=dtypes.float32, shape=(2,)),
        ),
        optimizer=mock_optimizer,
        model_dir=self._model_dir,
        label_dimension=3,
        weight_feature_key='label_weights')
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Create batched inputs.
    input_fn = numpy_io.numpy_input_fn(
        # NOTE: feature columns are concatenated in alphabetic order of keys.
        x={
            # Inputs shapes are (batch_size, feature_column.dimension).
            'x': np.array((
                (15., 0.),
                (45., 45000.),
                (21., 33000.),
                (60., 10000.)
            )),
            'y': np.array((
                (1.5, 135.2),
                (1.8, 158.8),
                (1.7, 207.1),
                (1.6, 90.2)
            )),
            # TODO(ptucker): Add test for different weight shapes when we fix
            # head._compute_weighted_loss (currently it requires weights to be
            # same shape as labels & logits).
            'label_weights': np.array((
                (1., 1., 0.),
                (.5, 1., .1),
                (.5, 0., .9),
                (0., 0., 0.),
            ))
        },
        # Labels shapes is (batch_size, num_outputs).
        y=np.array((
            (5., 2., 2.),
            (-2., 1., -4.),
            (-1., -1., -1.),
            (-4., 3., 9.),
        )),
        batch_size=4,
        num_epochs=None,
        shuffle=False)

    # Train for 1 step, then validate optimizer, summaries, and checkpoint.
    summary_hook = _SummaryHook()
    dnn_regressor.train(input_fn=input_fn, steps=1, hooks=(summary_hook,))
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    summaries = summary_hook.summaries()
    self.assertEqual(1, len(summaries))
    _assert_simple_summary(
        self,
        {
            metric_keys.MetricKeys.LOSS: expected_loss,
            # average_loss = loss / sum(label_weights) = 3.10290850204e+14 / 5.
            #              = 6.205817e+13
            metric_keys.MetricKeys.LOSS_MEAN: 6.205817e+13,
            'dnn/dnn/hiddenlayer_0_activation': 0.,
            'dnn/dnn/hiddenlayer_0_fraction_of_zero_values': 0.,
            'dnn/dnn/hiddenlayer_1_activation': 0.,
            'dnn/dnn/hiddenlayer_1_fraction_of_zero_values': 0.,
            'dnn/dnn/logits_activation': 0.,
            'dnn/dnn/logits_fraction_of_zero_values': 0.,
        },
        summaries[0])
    _assert_checkpoint(
        self,
        base_global_step + 1,
        input_units=4,  # Sum of feature column dimensions.
        hidden_units=hidden_units,
        output_units=3,  # = label_dimension
        model_dir=self._model_dir)

    # Train for 3 steps - we should still get the same loss since we're not
    # updating weights.
    dnn_regressor.train(input_fn=input_fn, steps=3)
    self.assertEqual(2, mock_optimizer.minimize.call_count)
    _assert_checkpoint(
        self,
        base_global_step + 4,
        input_units=4,  # Sum of feature column dimensions.
        hidden_units=hidden_units,
        output_units=3,  # = label_dimension
        model_dir=self._model_dir)

  def test_weighted_multi_batch(self):
    hidden_units = (2, 2)
    base_global_step = 100
    _create_checkpoint((
        (((1., 2.), (3., 4.), (5., 6.), (7., 8.),), (9., 8.)),
        (((7., 6.), (5., 4.),), (3., 2.)),
        (((1., 2., 3.), (4., 5., 6.),), (7., 8., 9.)),
    ), base_global_step, self._model_dir)

    mock_optimizer = _mock_optimizer(self, hidden_units=hidden_units)
    dnn_regressor = dnn.DNNRegressor(
        hidden_units=hidden_units,
        feature_columns=(
            # Dimension is number of inputs.
            feature_column.numeric_column(
                'x', dtype=dtypes.int32, shape=(4,)),
        ),
        optimizer=mock_optimizer,
        model_dir=self._model_dir,
        label_dimension=3,
        weight_feature_key='label_weights')
    self.assertEqual(0, mock_optimizer.minimize.call_count)

    # Create batched input.
    input_fn = numpy_io.numpy_input_fn(
        x={
            # Inputs shape is (batch_size, feature_column.dimension).
            'x': np.array((
                (15., 0., 1.5, 135.2),
                (45., 45000., 1.8, 158.8),
                (21., 33000., 1.7, 207.1),
                (60., 10000., 1.6, 90.2)
            )),
            # TODO(ptucker): Add test for different weight shapes when we fix
            # head._compute_weighted_loss (currently it requires weights to be
            # same shape as labels & logits).
            'label_weights': np.array((
                (1., 1., 0.),
                (.5, 1., .1),
                (.5, 0., .9),
                (0., 0., 0.),
            ))
        },
        # Labels shapes is (batch_size, num_outputs).
        y=np.array((
            (5., 2., 2.),
            (-2., 1., -4.),
            (-1., -1., -1.),
            (-4., 3., 9.),
        )),
        batch_size=1,
        shuffle=False)

    # Train for 1 step, then validate optimizer, summaries, and checkpoint.
    num_steps = 4
    summary_hook = _SummaryHook()
    dnn_regressor.train(
        input_fn=input_fn, steps=num_steps, hooks=(summary_hook,))
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    summaries = summary_hook.summaries()
    self.assertEqual(num_steps, len(summaries))
    # TODO(ptucker): Point to tool for calculating a neural net output?
    # predictions = [
    #   [  54033.5    76909.6    99785.7]
    #   [8030393.8 11433082.4 14835771.0]
    #   [5923209.2  8433014.8 10942820.4]
    #   [1810021.6  2576969.6  3343917.6]
    # ]
    # losses = label_weights*(labels-predictions)^2 = [
    #   [2.91907881e+09 5.91477894e+09              0]
    #   [3.22436284e+13 1.30715350e+14 2.20100220e+13]
    #   [1.75422095e+13              0 1.07770806e+14]
    #   [             0              0              0]
    # ]
    # step_losses = [sum(losses[i]) for i in 0...3]
    #             = [8833857750, 1.84969e+14, 1.2531302e+14, 0]
    expected_step_losses = (8833857750, 1.84969e+14, 1.2531302e+14, 0)
    # step_average_losses = [
    #     step_losses[i] / sum(label_weights[i]) for i in 0...3
    # ] = [4416928875, 1.1560563e+14, 8.95093e+13, 0]
    expected_step_average_losses = (4416928875, 1.1560563e+14, 8.95093e+13, 0)
    for i in range(len(summaries)):
      _assert_simple_summary(
          self,
          {
              metric_keys.MetricKeys.LOSS: expected_step_losses[i],
              metric_keys.MetricKeys.LOSS_MEAN: expected_step_average_losses[i],
              'dnn/dnn/hiddenlayer_0_activation': 0.,
              'dnn/dnn/hiddenlayer_0_fraction_of_zero_values': 0.,
              'dnn/dnn/hiddenlayer_1_activation': 0.,
              'dnn/dnn/hiddenlayer_1_fraction_of_zero_values': 0.,
              'dnn/dnn/logits_activation': 0.,
              'dnn/dnn/logits_fraction_of_zero_values': 0.,
          },
          summaries[i])
    _assert_checkpoint(
        self,
        base_global_step + num_steps,
        input_units=4,  # Sum of feature column dimensions.
        hidden_units=hidden_units,
        output_units=3,  # = label_dimension
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

    # Create DNNClassifier with mock optimizer.
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

    # Create DNNClassifier with mock optimizer.
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

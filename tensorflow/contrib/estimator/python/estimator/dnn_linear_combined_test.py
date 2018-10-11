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
"""Tests for dnn_linear_combined.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

import numpy as np
import six

from tensorflow.contrib.estimator.python.estimator import dnn_linear_combined
from tensorflow.contrib.estimator.python.estimator import head as head_lib
from tensorflow.python.estimator.canned import dnn_testing_utils
from tensorflow.python.estimator.canned import linear_testing_utils
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache


def _dnn_only_estimator_fn(
    hidden_units,
    feature_columns,
    model_dir=None,
    label_dimension=1,
    weight_column=None,
    optimizer='Adagrad',
    activation_fn=nn.relu,
    dropout=None,
    input_layer_partitioner=None,
    config=None):
  return dnn_linear_combined.DNNLinearCombinedEstimator(
      head=head_lib.regression_head(
          weight_column=weight_column, label_dimension=label_dimension,
          # Tests in core (from which this test inherits) test the sum loss.
          loss_reduction=losses.Reduction.SUM),
      model_dir=model_dir,
      dnn_feature_columns=feature_columns,
      dnn_optimizer=optimizer,
      dnn_hidden_units=hidden_units,
      dnn_activation_fn=activation_fn,
      dnn_dropout=dropout,
      input_layer_partitioner=input_layer_partitioner,
      config=config)


class DNNOnlyEstimatorEvaluateTest(
    dnn_testing_utils.BaseDNNRegressorEvaluateTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNRegressorEvaluateTest.__init__(
        self, _dnn_only_estimator_fn)


class DNNOnlyEstimatorPredictTest(
    dnn_testing_utils.BaseDNNRegressorPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNRegressorPredictTest.__init__(
        self, _dnn_only_estimator_fn)


class DNNOnlyEstimatorTrainTest(
    dnn_testing_utils.BaseDNNRegressorTrainTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNRegressorTrainTest.__init__(
        self, _dnn_only_estimator_fn)


def _linear_only_estimator_fn(
    feature_columns,
    model_dir=None,
    label_dimension=1,
    weight_column=None,
    optimizer='Ftrl',
    config=None,
    partitioner=None,
    sparse_combiner='sum'):
  return dnn_linear_combined.DNNLinearCombinedEstimator(
      head=head_lib.regression_head(
          weight_column=weight_column, label_dimension=label_dimension,
          # Tests in core (from which this test inherits) test the sum loss.
          loss_reduction=losses.Reduction.SUM),
      model_dir=model_dir,
      linear_feature_columns=feature_columns,
      linear_optimizer=optimizer,
      input_layer_partitioner=partitioner,
      config=config,
      linear_sparse_combiner=sparse_combiner)


class LinearOnlyEstimatorEvaluateTest(
    linear_testing_utils.BaseLinearRegressorEvaluationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorEvaluationTest.__init__(
        self, _linear_only_estimator_fn)


class LinearOnlyEstimatorPredictTest(
    linear_testing_utils.BaseLinearRegressorPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorPredictTest.__init__(
        self, _linear_only_estimator_fn)


class LinearOnlyEstimatorTrainTest(
    linear_testing_utils.BaseLinearRegressorTrainingTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorTrainingTest.__init__(
        self, _linear_only_estimator_fn)


class DNNLinearCombinedEstimatorIntegrationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(
      self, train_input_fn, eval_input_fn, predict_input_fn, input_dimension,
      label_dimension, batch_size):
    linear_feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))]
    dnn_feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))]
    feature_columns = linear_feature_columns + dnn_feature_columns
    est = dnn_linear_combined.DNNLinearCombinedEstimator(
        head=head_lib.regression_head(label_dimension=label_dimension),
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        dnn_hidden_units=(2, 2),
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


if __name__ == '__main__':
  test.main()

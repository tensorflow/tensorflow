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

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.estimator.canned import dnn_linear_combined
from tensorflow.python.estimator.canned import dnn_testing_utils
from tensorflow.python.estimator.canned import linear_testing_utils
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.estimator.inputs import pandas_io
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import input as input_lib

try:
  # pylint: disable=g-import-not-at-top
  import pandas as pd
  HAS_PANDAS = True
except IOError:
  # Pandas writes a temporary file during import. If it fails, don't use pandas.
  HAS_PANDAS = False
except ImportError:
  HAS_PANDAS = False


class DNNOnlyModelFnTest(dnn_testing_utils.BaseDNNModelFnTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNModelFnTest.__init__(self, self._dnn_only_model_fn)

  def _dnn_only_model_fn(
      self,
      features,
      labels,
      mode,
      head,
      hidden_units,
      feature_columns,
      optimizer='Adagrad',
      activation_fn=nn.relu,
      dropout=None,  # pylint: disable=redefined-outer-name
      input_layer_partitioner=None,
      config=None):
    return dnn_linear_combined._dnn_linear_combined_model_fn(
        features=features,
        labels=labels,
        mode=mode,
        head=head,
        linear_feature_columns=[],
        dnn_hidden_units=hidden_units,
        dnn_feature_columns=feature_columns,
        dnn_optimizer=optimizer,
        dnn_activation_fn=activation_fn,
        dnn_dropout=dropout,
        input_layer_partitioner=input_layer_partitioner,
        config=config)


# A function to mimic linear-regressor init reuse same tests.
def _linear_regressor_fn(feature_columns,
                         model_dir=None,
                         label_dimension=1,
                         weight_feature_key=None,
                         optimizer='Ftrl',
                         config=None,
                         partitioner=None):
  return dnn_linear_combined.DNNLinearCombinedRegressor(
      model_dir=model_dir,
      linear_feature_columns=feature_columns,
      linear_optimizer=optimizer,
      label_dimension=label_dimension,
      weight_feature_key=weight_feature_key,
      input_layer_partitioner=partitioner,
      config=config)


class LinearOnlyRegressorPartitionerTest(
    linear_testing_utils.BaseLinearRegressorPartitionerTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorPartitionerTest.__init__(
        self, _linear_regressor_fn)


class LinearOnlyRegressorEvaluationTest(
    linear_testing_utils.BaseLinearRegressorEvaluationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorEvaluationTest.__init__(
        self, _linear_regressor_fn)


class LinearOnlyRegressorPredictTest(
    linear_testing_utils.BaseLinearRegressorPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorPredictTest.__init__(
        self, _linear_regressor_fn)


class LinearOnlyRegressorIntegrationTest(
    linear_testing_utils.BaseLinearRegressorIntegrationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorIntegrationTest.__init__(
        self, _linear_regressor_fn)


class LinearOnlyRegressorTrainingTest(
    linear_testing_utils.BaseLinearRegressorTrainingTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorTrainingTest.__init__(
        self, _linear_regressor_fn)


class DNNLinearCombinedRegressorIntegrationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(
      self, train_input_fn, eval_input_fn, predict_input_fn, input_dimension,
      label_dimension, batch_size):
    linear_feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))]
    dnn_feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))]
    feature_columns = linear_feature_columns + dnn_feature_columns
    est = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=linear_feature_columns,
        dnn_hidden_units=(2, 2),
        dnn_feature_columns=dnn_feature_columns,
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
      features = linear_testing_utils.queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels
    def _eval_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      features = linear_testing_utils.queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels
    def _predict_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      features = linear_testing_utils.queue_parsed_features(feature_map)
      features.pop('y')
      return features, None

    self._test_complete_flow(
        train_input_fn=_train_input_fn,
        eval_input_fn=_eval_input_fn,
        predict_input_fn=_predict_input_fn,
        input_dimension=label_dimension,
        label_dimension=label_dimension,
        batch_size=batch_size)


class DNNLinearCombinedClassifierIntegrationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(
      self, train_input_fn, eval_input_fn, predict_input_fn, input_dimension,
      n_classes, batch_size):
    linear_feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))]
    dnn_feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))]
    feature_columns = linear_feature_columns + dnn_feature_columns
    est = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=linear_feature_columns,
        dnn_hidden_units=(2, 2),
        dnn_feature_columns=dnn_feature_columns,
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
      features = linear_testing_utils.queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels
    def _eval_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      features = linear_testing_utils.queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels
    def _predict_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      features = linear_testing_utils.queue_parsed_features(feature_map)
      features.pop('y')
      return features, None

    self._test_complete_flow(
        train_input_fn=_train_input_fn,
        eval_input_fn=_eval_input_fn,
        predict_input_fn=_predict_input_fn,
        input_dimension=input_dimension,
        n_classes=n_classes,
        batch_size=batch_size)


if __name__ == '__main__':
  test.main()

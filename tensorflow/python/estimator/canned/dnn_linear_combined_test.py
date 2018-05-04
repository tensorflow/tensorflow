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
from tensorflow.python.estimator import estimator
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
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import optimizer as optimizer_lib


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

  def _dnn_only_model_fn(self,
                         features,
                         labels,
                         mode,
                         head,
                         hidden_units,
                         feature_columns,
                         optimizer='Adagrad',
                         activation_fn=nn.relu,
                         dropout=None,
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
                         weight_column=None,
                         optimizer='Ftrl',
                         config=None,
                         partitioner=None):
  return dnn_linear_combined.DNNLinearCombinedRegressor(
      model_dir=model_dir,
      linear_feature_columns=feature_columns,
      linear_optimizer=optimizer,
      label_dimension=label_dimension,
      weight_column=weight_column,
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


def _linear_classifier_fn(feature_columns,
                          model_dir=None,
                          n_classes=2,
                          weight_column=None,
                          label_vocabulary=None,
                          optimizer='Ftrl',
                          config=None,
                          partitioner=None):
  return dnn_linear_combined.DNNLinearCombinedClassifier(
      model_dir=model_dir,
      linear_feature_columns=feature_columns,
      linear_optimizer=optimizer,
      n_classes=n_classes,
      weight_column=weight_column,
      label_vocabulary=label_vocabulary,
      input_layer_partitioner=partitioner,
      config=config)


class LinearOnlyClassifierTrainingTest(
    linear_testing_utils.BaseLinearClassifierTrainingTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearClassifierTrainingTest.__init__(
        self, linear_classifier_fn=_linear_classifier_fn)


class LinearOnlyClassifierClassesEvaluationTest(
    linear_testing_utils.BaseLinearClassifierEvaluationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearClassifierEvaluationTest.__init__(
        self, linear_classifier_fn=_linear_classifier_fn)


class LinearOnlyClassifierPredictTest(
    linear_testing_utils.BaseLinearClassifierPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearClassifierPredictTest.__init__(
        self, linear_classifier_fn=_linear_classifier_fn)


class LinearOnlyClassifierIntegrationTest(
    linear_testing_utils.BaseLinearClassifierIntegrationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearClassifierIntegrationTest.__init__(
        self, linear_classifier_fn=_linear_classifier_fn)


class DNNLinearCombinedRegressorIntegrationTest(test.TestCase):

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


# A function to mimic dnn-classifier init reuse same tests.
def _dnn_classifier_fn(hidden_units,
                       feature_columns,
                       model_dir=None,
                       n_classes=2,
                       weight_column=None,
                       label_vocabulary=None,
                       optimizer='Adagrad',
                       config=None,
                       input_layer_partitioner=None):
  return dnn_linear_combined.DNNLinearCombinedClassifier(
      model_dir=model_dir,
      dnn_hidden_units=hidden_units,
      dnn_feature_columns=feature_columns,
      dnn_optimizer=optimizer,
      n_classes=n_classes,
      weight_column=weight_column,
      label_vocabulary=label_vocabulary,
      input_layer_partitioner=input_layer_partitioner,
      config=config)


class DNNOnlyClassifierEvaluateTest(
    dnn_testing_utils.BaseDNNClassifierEvaluateTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNClassifierEvaluateTest.__init__(
        self, _dnn_classifier_fn)


class DNNOnlyClassifierPredictTest(
    dnn_testing_utils.BaseDNNClassifierPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNClassifierPredictTest.__init__(
        self, _dnn_classifier_fn)


class DNNOnlyClassifierTrainTest(
    dnn_testing_utils.BaseDNNClassifierTrainTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNClassifierTrainTest.__init__(
        self, _dnn_classifier_fn)


# A function to mimic dnn-regressor init reuse same tests.
def _dnn_regressor_fn(hidden_units,
                      feature_columns,
                      model_dir=None,
                      label_dimension=1,
                      weight_column=None,
                      optimizer='Adagrad',
                      config=None,
                      input_layer_partitioner=None):
  return dnn_linear_combined.DNNLinearCombinedRegressor(
      model_dir=model_dir,
      dnn_hidden_units=hidden_units,
      dnn_feature_columns=feature_columns,
      dnn_optimizer=optimizer,
      label_dimension=label_dimension,
      weight_column=weight_column,
      input_layer_partitioner=input_layer_partitioner,
      config=config)


class DNNOnlyRegressorEvaluateTest(
    dnn_testing_utils.BaseDNNRegressorEvaluateTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNRegressorEvaluateTest.__init__(
        self, _dnn_regressor_fn)


class DNNOnlyRegressorPredictTest(
    dnn_testing_utils.BaseDNNRegressorPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNRegressorPredictTest.__init__(
        self, _dnn_regressor_fn)


class DNNOnlyRegressorTrainTest(
    dnn_testing_utils.BaseDNNRegressorTrainTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNRegressorTrainTest.__init__(
        self, _dnn_regressor_fn)


class DNNLinearCombinedClassifierIntegrationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _as_label(self, data_in_float):
    return np.rint(data_in_float).astype(np.int64)

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
    n_classes = 3
    input_dimension = 2
    batch_size = 10
    data = np.linspace(
        0., n_classes - 1., batch_size * input_dimension, dtype=np.float32)
    x_data = data.reshape(batch_size, input_dimension)
    y_data = self._as_label(np.reshape(data[:batch_size], (batch_size, 1)))
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
    data = np.linspace(0., n_classes - 1., batch_size, dtype=np.float32)
    x = pd.DataFrame({'x': data})
    y = pd.Series(self._as_label(data))
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
    n_classes = 3
    batch_size = 10
    data = np.linspace(0., n_classes-1., batch_size * input_dimension,
                       dtype=np.float32)
    data = data.reshape(batch_size, input_dimension)

    serialized_examples = []
    for datum in data:
      example = example_pb2.Example(features=feature_pb2.Features(
          feature={
              'x':
                  feature_pb2.Feature(float_list=feature_pb2.FloatList(
                      value=datum)),
              'y':
                  feature_pb2.Feature(int64_list=feature_pb2.Int64List(
                      value=self._as_label(datum[:1]))),
          }))
      serialized_examples.append(example.SerializeToString())

    feature_spec = {
        'x': parsing_ops.FixedLenFeature([input_dimension], dtypes.float32),
        'y': parsing_ops.FixedLenFeature([1], dtypes.int64),
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


class DNNLinearCombinedTests(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _mock_optimizer(self, real_optimizer, var_name_prefix):
    """Verifies global_step is None and var_names start with given prefix."""

    def _minimize(loss, global_step=None, var_list=None):
      self.assertIsNone(global_step)
      trainable_vars = var_list or ops.get_collection(
          ops.GraphKeys.TRAINABLE_VARIABLES)
      var_names = [var.name for var in trainable_vars]
      self.assertTrue(
          all([name.startswith(var_name_prefix) for name in var_names]))
      # var is used to check this op called by training.
      with ops.name_scope(''):
        var = variables_lib.Variable(0., name=(var_name_prefix + '_called'))
      with ops.control_dependencies([var.assign(100.)]):
        return real_optimizer.minimize(loss, global_step, var_list)

    optimizer_mock = test.mock.NonCallableMagicMock(
        spec=optimizer_lib.Optimizer, wraps=real_optimizer)
    optimizer_mock.minimize = test.mock.MagicMock(wraps=_minimize)

    return optimizer_mock

  def test_train_op_calls_both_dnn_and_linear(self):
    opt = gradient_descent.GradientDescentOptimizer(1.)
    x_column = feature_column.numeric_column('x')
    input_fn = numpy_io.numpy_input_fn(
        x={'x': np.array([[0.], [1.]])},
        y=np.array([[0.], [1.]]),
        batch_size=1,
        shuffle=False)
    est = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[x_column],
        # verifies linear_optimizer is used only for linear part.
        linear_optimizer=self._mock_optimizer(opt, 'linear'),
        dnn_hidden_units=(2, 2),
        dnn_feature_columns=[x_column],
        # verifies dnn_optimizer is used only for linear part.
        dnn_optimizer=self._mock_optimizer(opt, 'dnn'),
        model_dir=self._model_dir)
    est.train(input_fn, steps=1)
    # verifies train_op fires linear minimize op
    self.assertEqual(100.,
                     checkpoint_utils.load_variable(
                         self._model_dir, 'linear_called'))
    # verifies train_op fires dnn minimize op
    self.assertEqual(100.,
                     checkpoint_utils.load_variable(
                         self._model_dir, 'dnn_called'))

  def test_dnn_and_linear_logits_are_added(self):
    with ops.Graph().as_default():
      variables_lib.Variable([[1.0]], name='linear/linear_model/x/weights')
      variables_lib.Variable([2.0], name='linear/linear_model/bias_weights')
      variables_lib.Variable([[3.0]], name='dnn/hiddenlayer_0/kernel')
      variables_lib.Variable([4.0], name='dnn/hiddenlayer_0/bias')
      variables_lib.Variable([[5.0]], name='dnn/logits/kernel')
      variables_lib.Variable([6.0], name='dnn/logits/bias')
      variables_lib.Variable(1, name='global_step', dtype=dtypes.int64)
      linear_testing_utils.save_variables_to_ckpt(self._model_dir)

    x_column = feature_column.numeric_column('x')
    est = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[x_column],
        dnn_hidden_units=[1],
        dnn_feature_columns=[x_column],
        model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(
        x={'x': np.array([[10.]])}, batch_size=1, shuffle=False)
    # linear logits = 10*1 + 2 = 12
    # dnn logits = (10*3 + 4)*5 + 6 = 176
    # logits = dnn + linear = 176 + 12 = 188
    self.assertAllClose(
        {
            prediction_keys.PredictionKeys.PREDICTIONS: [188.],
        },
        next(est.predict(input_fn=input_fn)))


class DNNLinearCombinedWarmStartingTest(test.TestCase):

  def setUp(self):
    # Create a directory to save our old checkpoint and vocabularies to.
    self._ckpt_and_vocab_dir = tempfile.mkdtemp()

    # Make a dummy input_fn.
    def _input_fn():
      features = {
          'age': [[23.], [31.]],
          'city': [['Palo Alto'], ['Mountain View']],
      }
      return features, [0, 1]

    self._input_fn = _input_fn

  def tearDown(self):
    # Clean up checkpoint / vocab dir.
    writer_cache.FileWriterCache.clear()
    shutil.rmtree(self._ckpt_and_vocab_dir)

  def test_classifier_basic_warm_starting(self):
    """Tests correctness of DNNLinearCombinedClassifier default warm-start."""
    age = feature_column.numeric_column('age')
    city = feature_column.embedding_column(
        feature_column.categorical_column_with_vocabulary_list(
            'city', vocabulary_list=['Mountain View', 'Palo Alto']),
        dimension=5)

    # Create a DNNLinearCombinedClassifier and train to save a checkpoint.
    dnn_lc_classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[age],
        dnn_feature_columns=[city],
        dnn_hidden_units=[256, 128],
        model_dir=self._ckpt_and_vocab_dir,
        n_classes=4,
        linear_optimizer='SGD',
        dnn_optimizer='SGD')
    dnn_lc_classifier.train(input_fn=self._input_fn, max_steps=1)

    # Create a second DNNLinearCombinedClassifier, warm-started from the first.
    # Use a learning_rate = 0.0 optimizer to check values (use SGD so we don't
    # have accumulator values that change).
    warm_started_dnn_lc_classifier = (
        dnn_linear_combined.DNNLinearCombinedClassifier(
            linear_feature_columns=[age],
            dnn_feature_columns=[city],
            dnn_hidden_units=[256, 128],
            n_classes=4,
            linear_optimizer=gradient_descent.GradientDescentOptimizer(
                learning_rate=0.0),
            dnn_optimizer=gradient_descent.GradientDescentOptimizer(
                learning_rate=0.0),
            warm_start_from=dnn_lc_classifier.model_dir))

    warm_started_dnn_lc_classifier.train(input_fn=self._input_fn, max_steps=1)
    for variable_name in warm_started_dnn_lc_classifier.get_variable_names():
      self.assertAllClose(
          dnn_lc_classifier.get_variable_value(variable_name),
          warm_started_dnn_lc_classifier.get_variable_value(variable_name))

  def test_regressor_basic_warm_starting(self):
    """Tests correctness of DNNLinearCombinedRegressor default warm-start."""
    age = feature_column.numeric_column('age')
    city = feature_column.embedding_column(
        feature_column.categorical_column_with_vocabulary_list(
            'city', vocabulary_list=['Mountain View', 'Palo Alto']),
        dimension=5)

    # Create a DNNLinearCombinedRegressor and train to save a checkpoint.
    dnn_lc_regressor = dnn_linear_combined.DNNLinearCombinedRegressor(
        linear_feature_columns=[age],
        dnn_feature_columns=[city],
        dnn_hidden_units=[256, 128],
        model_dir=self._ckpt_and_vocab_dir,
        linear_optimizer='SGD',
        dnn_optimizer='SGD')
    dnn_lc_regressor.train(input_fn=self._input_fn, max_steps=1)

    # Create a second DNNLinearCombinedRegressor, warm-started from the first.
    # Use a learning_rate = 0.0 optimizer to check values (use SGD so we don't
    # have accumulator values that change).
    warm_started_dnn_lc_regressor = (
        dnn_linear_combined.DNNLinearCombinedRegressor(
            linear_feature_columns=[age],
            dnn_feature_columns=[city],
            dnn_hidden_units=[256, 128],
            linear_optimizer=gradient_descent.GradientDescentOptimizer(
                learning_rate=0.0),
            dnn_optimizer=gradient_descent.GradientDescentOptimizer(
                learning_rate=0.0),
            warm_start_from=dnn_lc_regressor.model_dir))

    warm_started_dnn_lc_regressor.train(input_fn=self._input_fn, max_steps=1)
    for variable_name in warm_started_dnn_lc_regressor.get_variable_names():
      self.assertAllClose(
          dnn_lc_regressor.get_variable_value(variable_name),
          warm_started_dnn_lc_regressor.get_variable_value(variable_name))

  def test_warm_starting_selective_variables(self):
    """Tests selecting variables to warm-start."""
    age = feature_column.numeric_column('age')
    city = feature_column.embedding_column(
        feature_column.categorical_column_with_vocabulary_list(
            'city', vocabulary_list=['Mountain View', 'Palo Alto']),
        dimension=5)

    # Create a DNNLinearCombinedClassifier and train to save a checkpoint.
    dnn_lc_classifier = dnn_linear_combined.DNNLinearCombinedClassifier(
        linear_feature_columns=[age],
        dnn_feature_columns=[city],
        dnn_hidden_units=[256, 128],
        model_dir=self._ckpt_and_vocab_dir,
        n_classes=4,
        linear_optimizer='SGD',
        dnn_optimizer='SGD')
    dnn_lc_classifier.train(input_fn=self._input_fn, max_steps=1)

    # Create a second DNNLinearCombinedClassifier, warm-started from the first.
    # Use a learning_rate = 0.0 optimizer to check values (use SGD so we don't
    # have accumulator values that change).
    warm_started_dnn_lc_classifier = (
        dnn_linear_combined.DNNLinearCombinedClassifier(
            linear_feature_columns=[age],
            dnn_feature_columns=[city],
            dnn_hidden_units=[256, 128],
            n_classes=4,
            linear_optimizer=gradient_descent.GradientDescentOptimizer(
                learning_rate=0.0),
            dnn_optimizer=gradient_descent.GradientDescentOptimizer(
                learning_rate=0.0),
            # The provided regular expression will only warm-start the deep
            # portion of the model.
            warm_start_from=estimator.WarmStartSettings(
                ckpt_to_initialize_from=dnn_lc_classifier.model_dir,
                vars_to_warm_start='.*(dnn).*')))

    warm_started_dnn_lc_classifier.train(input_fn=self._input_fn, max_steps=1)
    for variable_name in warm_started_dnn_lc_classifier.get_variable_names():
      if 'dnn' in variable_name:
        self.assertAllClose(
            dnn_lc_classifier.get_variable_value(variable_name),
            warm_started_dnn_lc_classifier.get_variable_value(variable_name))
      elif 'linear' in variable_name:
        linear_values = warm_started_dnn_lc_classifier.get_variable_value(
            variable_name)
        # Since they're not warm-started, the linear weights will be
        # zero-initialized.
        self.assertAllClose(np.zeros_like(linear_values), linear_values)


if __name__ == '__main__':
  test.main()

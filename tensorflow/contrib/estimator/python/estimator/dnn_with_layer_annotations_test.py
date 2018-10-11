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
"""Tests for dnn_with_layer_annotations.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

import numpy as np
import six

from tensorflow.contrib.estimator.python.estimator import dnn_with_layer_annotations
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.canned import dnn
from tensorflow.python.estimator.canned import dnn_testing_utils
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.estimator.inputs import pandas_io
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import queue_runner

try:
  # pylint: disable=g-import-not-at-top
  import pandas as pd
  HAS_PANDAS = True
except IOError:
  # Pandas writes a temporary file during import. If it fails, don't use pandas.
  HAS_PANDAS = False
except ImportError:
  HAS_PANDAS = False


def _dnn_classifier_fn(*args, **kwargs):
  return dnn_with_layer_annotations.DNNClassifierWithLayerAnnotations(
      *args, **kwargs)


class DNNWarmStartingTest(dnn_testing_utils.BaseDNNWarmStartingTest,
                          test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNWarmStartingTest.__init__(self, _dnn_classifier_fn,
                                                       _dnn_regressor_fn)


class DNNWithLayerAnnotationsClassifierEvaluateTest(
    dnn_testing_utils.BaseDNNClassifierEvaluateTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNClassifierEvaluateTest.__init__(
        self, _dnn_classifier_fn)


class DNNClassifierWithLayerAnnotationsPredictTest(
    dnn_testing_utils.BaseDNNClassifierPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNClassifierPredictTest.__init__(
        self, _dnn_classifier_fn)


class DNNClassifierWithLayerAnnotationsTrainTest(
    dnn_testing_utils.BaseDNNClassifierTrainTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNClassifierTrainTest.__init__(
        self, _dnn_classifier_fn)


def _dnn_regressor_fn(*args, **kwargs):
  return dnn_with_layer_annotations.DNNRegressorWithLayerAnnotations(
      *args, **kwargs)


class DNNWithLayerAnnotationsTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def _getLayerAnnotationCollection(self, graph, collection_name):
    keys = graph.get_collection(
        dnn_with_layer_annotations.LayerAnnotationsCollectionNames.keys(
            collection_name))
    values = graph.get_collection(
        dnn_with_layer_annotations.LayerAnnotationsCollectionNames.values(
            collection_name))
    if len(keys) != len(values):
      raise ValueError('keys and values should have same length. lengths were: '
                       '%d and %d, and elements were %s and %s' %
                       (len(keys), len(values), keys, values))
    return dict(zip(keys, values))

  def _testAnnotationsPresentForEstimator(self, estimator_class):
    feature_columns = [
        feature_column.numeric_column('x', shape=(1,)),
        feature_column.embedding_column(
            feature_column.categorical_column_with_vocabulary_list(
                'y', vocabulary_list=['a', 'b', 'c']),
            dimension=3)
    ]
    estimator = estimator_class(
        hidden_units=(2, 2),
        feature_columns=feature_columns,
        model_dir=self._model_dir)
    model_fn = estimator.model_fn

    graph = ops.Graph()
    with graph.as_default():
      model_fn({
          'x': array_ops.constant([1.0]),
          'y': array_ops.constant(['a'])
      }, {},
               model_fn_lib.ModeKeys.PREDICT,
               config=None)

      unprocessed_features = self._getLayerAnnotationCollection(
          graph, dnn_with_layer_annotations.LayerAnnotationsCollectionNames
          .UNPROCESSED_FEATURES)
      processed_features = self._getLayerAnnotationCollection(
          graph, dnn_with_layer_annotations.LayerAnnotationsCollectionNames
          .PROCESSED_FEATURES)
      feature_columns = graph.get_collection(
          dnn_with_layer_annotations.LayerAnnotationsCollectionNames
          .FEATURE_COLUMNS)

      self.assertItemsEqual(unprocessed_features.keys(), ['x', 'y'])
      self.assertEqual(2, len(processed_features.keys()))
      self.assertEqual(2, len(feature_columns))

  def testAnnotationsPresentForClassifier(self):
    self._testAnnotationsPresentForEstimator(
        dnn_with_layer_annotations.DNNClassifierWithLayerAnnotations)

  def testAnnotationsPresentForRegressor(self):
    self._testAnnotationsPresentForEstimator(
        dnn_with_layer_annotations.DNNRegressorWithLayerAnnotations)

  def _testCheckpointCompatibleWithNonAnnotatedEstimator(
      self, train_input_fn, predict_input_fn, non_annotated_class,
      annotated_class, prediction_key, estimator_args):
    input_dimension = 2
    feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))
    ]
    estimator = non_annotated_class(
        model_dir=self._model_dir,
        hidden_units=(2, 2),
        feature_columns=feature_columns,
        **estimator_args)

    estimator.train(train_input_fn, steps=10)

    predictions = np.array(
        [x[prediction_key] for x in estimator.predict(predict_input_fn)])

    annotated_estimator = annotated_class(
        model_dir=self._model_dir,
        hidden_units=(2, 2),
        feature_columns=feature_columns,
        warm_start_from=self._model_dir,
        **estimator_args)

    annotated_predictions = np.array([
        x[prediction_key] for x in annotated_estimator.predict(predict_input_fn)
    ])

    self.assertAllEqual(predictions.shape, annotated_predictions.shape)
    for i, (a, b) in enumerate(
        zip(predictions.flatten(), annotated_predictions.flatten())):
      self.assertAlmostEqual(a, b, msg='index=%d' % i)

  def testCheckpointCompatibleForClassifier(self):
    n_classes = 2
    input_dimension = 2
    batch_size = 10
    data = np.linspace(
        0., n_classes - 1., batch_size * input_dimension, dtype=np.float32)
    x_data = data.reshape(batch_size, input_dimension)
    y_data = np.reshape(
        np.rint(data[:batch_size]).astype(np.int64), (batch_size, 1))
    # learn y = x
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': x_data},
        y=y_data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': x_data}, batch_size=batch_size, shuffle=False)

    self._testCheckpointCompatibleWithNonAnnotatedEstimator(
        train_input_fn,
        predict_input_fn,
        dnn.DNNClassifier,
        dnn_with_layer_annotations.DNNClassifierWithLayerAnnotations,
        prediction_key=prediction_keys.PredictionKeys.PROBABILITIES,
        estimator_args={'n_classes': n_classes})

  def testCheckpointCompatibleForRegressor(self):
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
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, batch_size=batch_size, shuffle=False)

    self._testCheckpointCompatibleWithNonAnnotatedEstimator(
        train_input_fn,
        predict_input_fn,
        dnn.DNNRegressor,
        dnn_with_layer_annotations.DNNRegressorWithLayerAnnotations,
        prediction_key=prediction_keys.PredictionKeys.PREDICTIONS,
        estimator_args={'label_dimension': label_dimension})


class DNNRegressorWithLayerAnnotationsEvaluateTest(
    dnn_testing_utils.BaseDNNRegressorEvaluateTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNRegressorEvaluateTest.__init__(
        self, _dnn_regressor_fn)


class DNNRegressorWithLayerAnnotationsPredictTest(
    dnn_testing_utils.BaseDNNRegressorPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNRegressorPredictTest.__init__(
        self, _dnn_regressor_fn)


class DNNRegressorWithLayerAnnotationsTrainTest(
    dnn_testing_utils.BaseDNNRegressorTrainTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNRegressorTrainTest.__init__(
        self, _dnn_regressor_fn)


def _queue_parsed_features(feature_map):
  tensors_to_enqueue = []
  keys = []
  for key, tensor in six.iteritems(feature_map):
    keys.append(key)
    tensors_to_enqueue.append(tensor)
  queue_dtypes = [x.dtype for x in tensors_to_enqueue]
  input_queue = data_flow_ops.FIFOQueue(capacity=100, dtypes=queue_dtypes)
  queue_runner.add_queue_runner(
      queue_runner.QueueRunner(input_queue,
                               [input_queue.enqueue(tensors_to_enqueue)]))
  dequeued_tensors = input_queue.dequeue()
  return {keys[i]: dequeued_tensors[i] for i in range(len(dequeued_tensors))}


class DNNRegressorWithLayerAnnotationsIntegrationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(self, train_input_fn, eval_input_fn, predict_input_fn,
                          input_dimension, label_dimension, batch_size):
    feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))
    ]
    est = dnn_with_layer_annotations.DNNRegressorWithLayerAnnotations(
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
        x={'x': data}, y=data, batch_size=batch_size, shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, batch_size=batch_size, shuffle=False)

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
        x=x, y=y, batch_size=batch_size, num_epochs=None, shuffle=True)
    eval_input_fn = pandas_io.pandas_input_fn(
        x=x, y=y, batch_size=batch_size, shuffle=False)
    predict_input_fn = pandas_io.pandas_input_fn(
        x=x, batch_size=batch_size, shuffle=False)

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
      example = example_pb2.Example(
          features=feature_pb2.Features(
              feature={
                  'x':
                      feature_pb2.Feature(
                          float_list=feature_pb2.FloatList(value=datum)),
                  'y':
                      feature_pb2.Feature(
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


class DNNClassifierWithLayerAnnotationsIntegrationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _as_label(self, data_in_float):
    return np.rint(data_in_float).astype(np.int64)

  def _test_complete_flow(self, train_input_fn, eval_input_fn, predict_input_fn,
                          input_dimension, n_classes, batch_size):
    feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))
    ]
    est = dnn_with_layer_annotations.DNNClassifierWithLayerAnnotations(
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
    n_classes = 3
    input_dimension = 2
    batch_size = 10
    data = np.linspace(
        0., n_classes - 1., batch_size * input_dimension, dtype=np.float32)
    x_data = data.reshape(batch_size, input_dimension)
    y_data = np.reshape(self._as_label(data[:batch_size]), (batch_size, 1))
    # learn y = x
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': x_data},
        y=y_data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': x_data}, y=y_data, batch_size=batch_size, shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': x_data}, batch_size=batch_size, shuffle=False)

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
    n_classes = 3
    batch_size = 10
    data = np.linspace(0., n_classes - 1., batch_size, dtype=np.float32)
    x = pd.DataFrame({'x': data})
    y = pd.Series(self._as_label(data))
    train_input_fn = pandas_io.pandas_input_fn(
        x=x, y=y, batch_size=batch_size, num_epochs=None, shuffle=True)
    eval_input_fn = pandas_io.pandas_input_fn(
        x=x, y=y, batch_size=batch_size, shuffle=False)
    predict_input_fn = pandas_io.pandas_input_fn(
        x=x, batch_size=batch_size, shuffle=False)

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
    data = np.linspace(
        0., n_classes - 1., batch_size * input_dimension, dtype=np.float32)
    data = data.reshape(batch_size, input_dimension)

    serialized_examples = []
    for datum in data:
      example = example_pb2.Example(
          features=feature_pb2.Features(
              feature={
                  'x':
                      feature_pb2.Feature(
                          float_list=feature_pb2.FloatList(value=datum)),
                  'y':
                      feature_pb2.Feature(
                          int64_list=feature_pb2.Int64List(
                              value=self._as_label(datum[:1]))),
              }))
      serialized_examples.append(example.SerializeToString())

    feature_spec = {
        'x': parsing_ops.FixedLenFeature([input_dimension], dtypes.float32),
        'y': parsing_ops.FixedLenFeature([1], dtypes.int64),
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


if __name__ == '__main__':
  test.main()

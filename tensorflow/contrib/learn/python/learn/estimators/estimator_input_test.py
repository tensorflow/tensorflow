# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Estimator input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import sys
import tempfile

# pylint: disable=g-bad-todo
# TODO(#6568): Remove this hack that makes dlopen() not crash.
# pylint: enable=g-bad-todo
# pylint: disable=g-import-not-at-top
if hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn import models
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.metrics.python.ops import metric_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import queue_runner_impl


_BOSTON_INPUT_DIM = 13
_IRIS_INPUT_DIM = 4


def boston_input_fn(num_epochs=None):
  boston = base.load_boston()
  features = input_lib.limit_epochs(
      array_ops.reshape(
          constant_op.constant(boston.data), [-1, _BOSTON_INPUT_DIM]),
      num_epochs=num_epochs)
  labels = array_ops.reshape(constant_op.constant(boston.target), [-1, 1])
  return features, labels


def boston_input_fn_with_queue(num_epochs=None):
  features, labels = boston_input_fn(num_epochs=num_epochs)

  # Create a minimal queue runner.
  fake_queue = data_flow_ops.FIFOQueue(30, dtypes.int32)
  queue_runner = queue_runner_impl.QueueRunner(fake_queue,
                                               [constant_op.constant(0)])
  queue_runner_impl.add_queue_runner(queue_runner)

  return features, labels


def iris_input_fn():
  iris = base.load_iris()
  features = array_ops.reshape(
      constant_op.constant(iris.data), [-1, _IRIS_INPUT_DIM])
  labels = array_ops.reshape(constant_op.constant(iris.target), [-1])
  return features, labels


def iris_input_fn_labels_dict():
  iris = base.load_iris()
  features = array_ops.reshape(
      constant_op.constant(iris.data), [-1, _IRIS_INPUT_DIM])
  labels = {
      'labels': array_ops.reshape(constant_op.constant(iris.target), [-1])
  }
  return features, labels


def boston_eval_fn():
  boston = base.load_boston()
  n_examples = len(boston.target)
  features = array_ops.reshape(
      constant_op.constant(boston.data), [n_examples, _BOSTON_INPUT_DIM])
  labels = array_ops.reshape(
      constant_op.constant(boston.target), [n_examples, 1])
  return array_ops.concat([features, features], 0), array_ops.concat(
      [labels, labels], 0)


def extract(data, key):
  if isinstance(data, dict):
    assert key in data
    return data[key]
  else:
    return data


def linear_model_params_fn(features, labels, mode, params):
  features = extract(features, 'input')
  labels = extract(labels, 'labels')

  assert mode in (model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
                  model_fn.ModeKeys.INFER)
  prediction, loss = (models.linear_regression_zero_init(features, labels))
  train_op = optimizers.optimize_loss(
      loss,
      variables.get_global_step(),
      optimizer='Adagrad',
      learning_rate=params['learning_rate'])
  return prediction, loss, train_op


def linear_model_fn(features, labels, mode):
  features = extract(features, 'input')
  labels = extract(labels, 'labels')
  assert mode in (model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
                  model_fn.ModeKeys.INFER)
  if isinstance(features, dict):
    (_, features), = features.items()
  prediction, loss = (models.linear_regression_zero_init(features, labels))
  train_op = optimizers.optimize_loss(
      loss, variables.get_global_step(), optimizer='Adagrad', learning_rate=0.1)
  return prediction, loss, train_op


def linear_model_fn_with_model_fn_ops(features, labels, mode):
  """Same as linear_model_fn, but returns `ModelFnOps`."""
  assert mode in (model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
                  model_fn.ModeKeys.INFER)
  prediction, loss = (models.linear_regression_zero_init(features, labels))
  train_op = optimizers.optimize_loss(
      loss, variables.get_global_step(), optimizer='Adagrad', learning_rate=0.1)
  return model_fn.ModelFnOps(
      mode=mode, predictions=prediction, loss=loss, train_op=train_op)


def logistic_model_no_mode_fn(features, labels):
  features = extract(features, 'input')
  labels = extract(labels, 'labels')
  labels = array_ops.one_hot(labels, 3, 1, 0)
  prediction, loss = (models.logistic_regression_zero_init(features, labels))
  train_op = optimizers.optimize_loss(
      loss, variables.get_global_step(), optimizer='Adagrad', learning_rate=0.1)
  return {
      'class': math_ops.argmax(prediction, 1),
      'prob': prediction
  }, loss, train_op


VOCAB_FILE_CONTENT = 'emerson\nlake\npalmer\n'
EXTRA_FILE_CONTENT = 'kermit\npiggy\nralph\n'


class EstimatorInputTest(test.TestCase):

  def testContinueTrainingDictionaryInput(self):
    boston = base.load_boston()
    output_dir = tempfile.mkdtemp()
    est = estimator.Estimator(model_fn=linear_model_fn, model_dir=output_dir)
    boston_input = {'input': boston.data}
    float64_target = {'labels': boston.target.astype(np.float64)}
    est.fit(x=boston_input, y=float64_target, steps=50)
    scores = est.evaluate(
        x=boston_input,
        y=float64_target,
        metrics={'MSE': metric_ops.streaming_mean_squared_error})
    del est
    # Create another estimator object with the same output dir.
    est2 = estimator.Estimator(model_fn=linear_model_fn, model_dir=output_dir)

    # Check we can evaluate and predict.
    scores2 = est2.evaluate(
        x=boston_input,
        y=float64_target,
        metrics={'MSE': metric_ops.streaming_mean_squared_error})
    self.assertAllClose(scores2['MSE'], scores['MSE'])
    predictions = np.array(list(est2.predict(x=boston_input)))
    other_score = _sklearn.mean_squared_error(predictions,
                                              float64_target['labels'])
    self.assertAllClose(other_score, scores['MSE'])

  def testBostonAll(self):
    boston = base.load_boston()
    est = estimator.SKCompat(estimator.Estimator(model_fn=linear_model_fn))
    float64_labels = boston.target.astype(np.float64)
    est.fit(x=boston.data, y=float64_labels, steps=100)
    scores = est.score(
        x=boston.data,
        y=float64_labels,
        metrics={'MSE': metric_ops.streaming_mean_squared_error})
    predictions = np.array(list(est.predict(x=boston.data)))
    other_score = _sklearn.mean_squared_error(predictions, boston.target)
    self.assertAllClose(scores['MSE'], other_score)
    self.assertTrue('global_step' in scores)
    self.assertEqual(100, scores['global_step'])

  def testBostonAllDictionaryInput(self):
    boston = base.load_boston()
    est = estimator.Estimator(model_fn=linear_model_fn)
    boston_input = {'input': boston.data}
    float64_target = {'labels': boston.target.astype(np.float64)}
    est.fit(x=boston_input, y=float64_target, steps=100)
    scores = est.evaluate(
        x=boston_input,
        y=float64_target,
        metrics={'MSE': metric_ops.streaming_mean_squared_error})
    predictions = np.array(list(est.predict(x=boston_input)))
    other_score = _sklearn.mean_squared_error(predictions, boston.target)
    self.assertAllClose(other_score, scores['MSE'])
    self.assertTrue('global_step' in scores)
    self.assertEqual(scores['global_step'], 100)

  def testIrisAll(self):
    iris = base.load_iris()
    est = estimator.SKCompat(
        estimator.Estimator(model_fn=logistic_model_no_mode_fn))
    est.fit(iris.data, iris.target, steps=100)
    scores = est.score(
        x=iris.data,
        y=iris.target,
        metrics={('accuracy', 'class'): metric_ops.streaming_accuracy})
    predictions = est.predict(x=iris.data)
    predictions_class = est.predict(x=iris.data, outputs=['class'])['class']
    self.assertEqual(predictions['prob'].shape[0], iris.target.shape[0])
    self.assertAllClose(predictions['class'], predictions_class)
    self.assertAllClose(
        predictions['class'], np.argmax(
            predictions['prob'], axis=1))
    other_score = _sklearn.accuracy_score(iris.target, predictions['class'])
    self.assertAllClose(scores['accuracy'], other_score)
    self.assertTrue('global_step' in scores)
    self.assertEqual(100, scores['global_step'])

  def testIrisAllDictionaryInput(self):
    iris = base.load_iris()
    est = estimator.Estimator(model_fn=logistic_model_no_mode_fn)
    iris_data = {'input': iris.data}
    iris_target = {'labels': iris.target}
    est.fit(iris_data, iris_target, steps=100)
    scores = est.evaluate(
        x=iris_data,
        y=iris_target,
        metrics={('accuracy', 'class'): metric_ops.streaming_accuracy})
    predictions = list(est.predict(x=iris_data))
    predictions_class = list(est.predict(x=iris_data, outputs=['class']))
    self.assertEqual(len(predictions), iris.target.shape[0])
    classes_batch = np.array([p['class'] for p in predictions])
    self.assertAllClose(classes_batch,
                        np.array([p['class'] for p in predictions_class]))
    self.assertAllClose(
        classes_batch,
        np.argmax(
            np.array([p['prob'] for p in predictions]), axis=1))
    other_score = _sklearn.accuracy_score(iris.target, classes_batch)
    self.assertAllClose(other_score, scores['accuracy'])
    self.assertTrue('global_step' in scores)
    self.assertEqual(scores['global_step'], 100)

  def testIrisInputFn(self):
    iris = base.load_iris()
    est = estimator.Estimator(model_fn=logistic_model_no_mode_fn)
    est.fit(input_fn=iris_input_fn, steps=100)
    _ = est.evaluate(input_fn=iris_input_fn, steps=1)
    predictions = list(est.predict(x=iris.data))
    self.assertEqual(len(predictions), iris.target.shape[0])

  def testIrisInputFnLabelsDict(self):
    iris = base.load_iris()
    est = estimator.Estimator(model_fn=logistic_model_no_mode_fn)
    est.fit(input_fn=iris_input_fn_labels_dict, steps=100)
    _ = est.evaluate(
        input_fn=iris_input_fn_labels_dict,
        steps=1,
        metrics={
            'accuracy':
                metric_spec.MetricSpec(
                    metric_fn=metric_ops.streaming_accuracy,
                    prediction_key='class',
                    label_key='labels')
        })
    predictions = list(est.predict(x=iris.data))
    self.assertEqual(len(predictions), iris.target.shape[0])

  def testTrainInputFn(self):
    est = estimator.Estimator(model_fn=linear_model_fn)
    est.fit(input_fn=boston_input_fn, steps=1)
    _ = est.evaluate(input_fn=boston_eval_fn, steps=1)

  def testPredictInputFn(self):
    est = estimator.Estimator(model_fn=linear_model_fn)
    boston = base.load_boston()
    est.fit(input_fn=boston_input_fn, steps=1)
    input_fn = functools.partial(boston_input_fn, num_epochs=1)
    output = list(est.predict(input_fn=input_fn))
    self.assertEqual(len(output), boston.target.shape[0])

  def testPredictInputFnWithQueue(self):
    est = estimator.Estimator(model_fn=linear_model_fn)
    boston = base.load_boston()
    est.fit(input_fn=boston_input_fn, steps=1)
    input_fn = functools.partial(boston_input_fn_with_queue, num_epochs=2)
    output = list(est.predict(input_fn=input_fn))
    self.assertEqual(len(output), boston.target.shape[0] * 2)

  def testPredictConstInputFn(self):
    est = estimator.Estimator(model_fn=linear_model_fn)
    boston = base.load_boston()
    est.fit(input_fn=boston_input_fn, steps=1)

    def input_fn():
      features = array_ops.reshape(
          constant_op.constant(boston.data), [-1, _BOSTON_INPUT_DIM])
      labels = array_ops.reshape(constant_op.constant(boston.target), [-1, 1])
      return features, labels

    output = list(est.predict(input_fn=input_fn))
    self.assertEqual(len(output), boston.target.shape[0])


if __name__ == '__main__':
  test.main()

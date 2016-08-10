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

"""Tests for Classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import _sklearn


def iris_input_fn(num_epochs=None):
  iris = tf.contrib.learn.datasets.load_iris()
  features = tf.reshape(tf.constant(iris.data), [-1, 4])
  if num_epochs:
    features = tf.train.limit_epochs(features, num_epochs=num_epochs)
  target = tf.reshape(tf.constant(iris.target), [-1])
  return features, target


def logistic_model_fn(features, target, unused_mode):
  target = tf.one_hot(target, 3, 1, 0)
  prediction, loss = tf.contrib.learn.models.logistic_regression_zero_init(
      features, target)
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=0.1)
  return prediction, loss, train_op


class ClassifierTest(tf.test.TestCase):

  def testIrisAll(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Classifier(model_fn=logistic_model_fn, n_classes=3)
    est.fit(iris.data, iris.target, steps=100)
    scores = est.evaluate(x=iris.data, y=iris.target)
    predictions = est.predict(x=iris.data)
    predictions_proba = est.predict_proba(x=iris.data)
    self.assertEqual(predictions.shape[0], iris.target.shape[0])
    self.assertAllEqual(predictions, np.argmax(predictions_proba, axis=1))
    other_score = _sklearn.accuracy_score(iris.target, predictions)
    self.assertAllClose(other_score, scores['accuracy'])

  def testIrisPredictAsIterable(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Classifier(model_fn=logistic_model_fn, n_classes=3)
    est.fit(iris.data, iris.target, steps=100)
    scores = est.evaluate(x=iris.data, y=iris.target)
    predictions = list(est.predict(x=iris.data, as_iterable=True))
    predictions_proba = list(est.predict_proba(x=iris.data, as_iterable=True))
    self.assertEqual(len(predictions), iris.target.shape[0])
    self.assertAllEqual(predictions, np.argmax(predictions_proba, axis=1))
    other_score = _sklearn.accuracy_score(iris.target, predictions)
    self.assertAllClose(other_score, scores['accuracy'])

  def testIrisInputFn(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Classifier(model_fn=logistic_model_fn, n_classes=3)
    est.fit(input_fn=iris_input_fn, steps=100)
    est.evaluate(input_fn=iris_input_fn, steps=1)
    predictions = est.predict(input_fn=iris_input_fn)
    self.assertEqual(predictions.shape[0], iris.target.shape[0])

  def testIrisPredictInputFnAsIterable(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Classifier(model_fn=logistic_model_fn, n_classes=3)
    est.fit(input_fn=iris_input_fn, steps=100)
    est.evaluate(input_fn=iris_input_fn, steps=1)
    predict_input_fn = functools.partial(iris_input_fn, num_epochs=1)
    predictions = list(est.predict(input_fn=predict_input_fn, as_iterable=True))
    self.assertEqual(len(predictions), iris.target.shape[0])


if __name__ == '__main__':
  tf.test.main()

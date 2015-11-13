#  Copyright 2015 Google Inc. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import random

import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from skflow.trainer import TensorFlowTrainer
from skflow.ops import mean_squared_error_regressor, softmax_classifier


class LinearRegression(object):
    """Linear Regression TensorFlow model."""

    def __init__(self, input_shape, graph):
        with graph.as_default():
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.inp = tf.placeholder(tf.float32, [None, input_shape], name="input")
            self.out = tf.placeholder(tf.float32, [None], name="output")
            self.weights = tf.get_variable("weights", [input_shape, 1])
            self.bias = tf.get_variable("bias", [1])
            self.predictions, self.loss = mean_squared_error_regressor(
                self.inp, self.out, self.weights, self.bias)


class LogisticRegression(object):
    """Logistic Regression TensorFlow model."""

    def __init__(self, n_classes, input_shape, graph):
        with graph.as_default():
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.inp = tf.placeholder(tf.float32, [None, input_shape], name="input")
            self.out = tf.placeholder(tf.float32, [None, n_classes], name="output")
            self.weights = tf.get_variable("weights", [input_shape, n_classes])
            self.bias = tf.get_variable("bias", [n_classes])
            self.predictions, self.loss = softmax_classifier(
                self.inp, self.out, self.weights, self.bias)

class TensorFlowEstimator(BaseEstimator):
  """Base class for all TensorFlow estimators.
  
  Parameters:
      tf_master: TensorFlow master. Empty string is default for local.
      batch_size: Mini batch size.
      steps: Number of steps to run over data.
      optimizer: Optimizer name (or class), for example "SGD", "Adam",
                 "Adagrad".
      learning_rate: Learning rate for optimizer.
  """

  def __init__(self, n_classes, tf_master="", batch_size=32, steps=50, optimizer="SGD",
               learning_rate=0.1):
    self.n_classes = n_classes
    self.tf_master = tf_master
    self.batch_size = batch_size
    self.steps = steps
    self.optimizer = optimizer
    self.learning_rate = learning_rate

  def _get_feed_dict_fn(self, model, X, y):

    def _feed_dict():
      inp = np.zeros([self.batch_size, X.shape[1]])
      if self.n_classes > 1:
        out = np.zeros([self.batch_size, self.n_classes])
      else:
        out = np.zeros([self.batch_size])
      for i in xrange(self.batch_size):
        sample = random.randint(0, X.shape[0] - 1)
        inp[i, :] = X[sample, :]
        if self.n_classes > 1:
          out[i, y[sample]] = 1.0
        else:
          out[i] = y[sample]
      return {model.inp.name: inp, model.out.name: out,}

    return _feed_dict

  def fit(self, X, y):
    with tf.Graph().as_default() as graph:
      if self.n_classes > 1:
        self._model = LogisticRegression(self.n_classes, X.shape[1], graph)
      else:
        self._model = LinearRegression(X.shape[1], graph)
      self._trainer = TensorFlowTrainer(self._model, self.optimizer, self.learning_rate)
      self._session = tf.Session(self.tf_master)
      self._trainer.initialize(self._session)
      self._trainer.train(self._session,
                          self._get_feed_dict_fn(self._model, X, y), self.steps)

  def predict(self, X):
    pred = self._session.run(self._model.predictions,
                             feed_dict={
                                 self._model.inp.name: X
                             })
    if self.n_classes < 2:
      return pred
    return pred.argmax(axis=1)


class TensorFlowRegressor(TensorFlowEstimator, RegressorMixin):
  pass


class TensorFlowClassifier(TensorFlowEstimator, ClassifierMixin):
  pass


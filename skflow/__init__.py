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
from sklearn.utils import check_array

from skflow.trainer import TensorFlowTrainer
from skflow.models import LinearRegression, LogisticRegression


class DataFeeder(object):
    """Data feeder is an example class to sample data for TF trainer.

    Parameters:
        X: feature matrix of shape [n_samples, n_features].
        y: target vector, either floats for regression or class id for
            classification.
        n_classes: number of classes, 0 and 1 are considered regression.
        batch_size: mini batch size to accumulate.
    """

    def __init__(self, X, y, n_classes, batch_size):
        self.X = check_array(X, dtype=np.float32)
        self.y = check_array(y, ensure_2d=False, dtype=None)
        self.n_classes = n_classes
        self.batch_size = batch_size

    def get_feed_dict_fn(self, input_placeholder, output_placeholder):
        """Returns a function, that will sample data and provide it to given
        placeholders.

        Args:
            input_placeholder: tf.Placeholder for input features mini batch.
            output_placeholder: tf.Placeholder for output targets.
        Returns:
            A function that when called samples a random subset of batch size
            from X and y.
        """
        def _feed_dict_fn():
            inp = np.zeros([self.batch_size, self.X.shape[1]])
            if self.n_classes > 1:
                out = np.zeros([self.batch_size, self.n_classes])
            else:
                out = np.zeros([self.batch_size])
            for i in xrange(self.batch_size):
                sample = random.randint(0, self.X.shape[0] - 1)
                inp[i, :] = self.X[sample, :]
                if self.n_classes > 1:
                    out[i, self.y[sample]] = 1.0
                else:
                    out[i] = self.y[sample]
            return {input_placeholder.name: inp, output_placeholder.name: out}
        return _feed_dict_fn


class TensorFlowEstimator(BaseEstimator):
    """Base class for all TensorFlow estimators.
  
    Parameters:
        tf_master: TensorFlow master. Empty string is default for local.
        batch_size: Mini batch size.
        steps: Number of steps to run over data.
        optimizer: Optimizer name (or class), for example "SGD", "Adam",
                   "Adagrad".
        learning_rate: Learning rate for optimizer.
        tf_random_seed: Random seed for TensorFlow initializers.
            Setting this value, allows consistency between reruns.
    """

    def __init__(self, n_classes, tf_master="", batch_size=32, steps=50, optimizer="SGD",
                 learning_rate=0.1, tf_random_seed=42):
        self.n_classes = n_classes
        self.tf_master = tf_master
        self.batch_size = batch_size
        self.steps = steps
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.tf_random_seed = tf_random_seed

    def fit(self, X, y):
        with tf.Graph().as_default() as graph:
            tf.set_random_seed(self.tf_random_seed)
            if self.n_classes > 1:
                self._model = LogisticRegression(self.n_classes, X.shape[1], graph)
            else:
                self._model = LinearRegression(X.shape[1], graph)
            self._data_feeder = DataFeeder(X, y, self.n_classes, self.batch_size)
            self._trainer = TensorFlowTrainer(self._model, self.optimizer, self.learning_rate)
            self._session = tf.Session(self.tf_master)
            self._trainer.initialize(self._session)
            self._trainer.train(self._session,
                                self._data_feeder.get_feed_dict_fn(self._model.inp,
                                                             self._model.out), self.steps)

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


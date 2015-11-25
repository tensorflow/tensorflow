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

import collections
import random

import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_array

from skflow.trainer import TensorFlowTrainer
from skflow import models, data_feeder


class TensorFlowEstimator(BaseEstimator):
    """Base class for all TensorFlow estimators.
  
    Parameters:
        model_fn: Model function, that takes input X, y tensors and outputs
                  prediction and loss tensors.
        n_classes: Number of classes in the target.
        tf_master: TensorFlow master. Empty string is default for local.
        batch_size: Mini batch size.
        steps: Number of steps to run over data.
        optimizer: Optimizer name (or class), for example "SGD", "Adam",
                   "Adagrad".
        learning_rate: Learning rate for optimizer.
        tf_random_seed: Random seed for TensorFlow initializers.
            Setting this value, allows consistency between reruns.
    """

    def __init__(self, model_fn, n_classes, tf_master="", batch_size=32, steps=50, optimizer="SGD",
                 learning_rate=0.1, tf_random_seed=42):
        self.n_classes = n_classes
        self.tf_master = tf_master
        self.batch_size = batch_size
        self.steps = steps
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.tf_random_seed = tf_random_seed
        self.model_fn = model_fn

    def fit(self, X, y):
        with tf.Graph().as_default() as graph:
            tf.set_random_seed(self.tf_random_seed)
            self._global_step = tf.Variable(0, name="global_step", trainable=False)

            # Create data feeder, to sample inputs from dataset.
            # If X and y are iterators, use StreamingDataFeeder.
            if hasattr(X, 'next'):
                assert hasattr(y, 'next')
                self._data_feeder = data_feeder.StreamingDataFeeder(X, y,
                    self.n_classes, self.batch_size)
            else:
                self._data_feeder = data_feeder.DataFeeder(X, y,
                    self.n_classes, self.batch_size)

            # Setting up input and output placeholders.
            input_shape = [None] + self._data_feeder.input_shape[1:]
            output_shape = [None] + self._data_feeder.output_shape[1:]
            self._inp = tf.placeholder(tf.float32, input_shape,
                                       name="input")
            self._out = tf.placeholder(tf.float32,  output_shape,
                name="output")

            # Create model's graph.
            self._model_predictions, self._model_loss = self.model_fn(self._inp, self._out)

            # Create trainer and augment graph with gradients and optimizer.
            self._trainer = TensorFlowTrainer(self._model_loss,
                self._global_step, self.optimizer, self.learning_rate)
            self._session = tf.Session(self.tf_master)

            # Initialize and train model.
            self._trainer.initialize(self._session)
            self._trainer.train(self._session,
                                self._data_feeder.get_feed_dict_fn(self._inp,
                                                             self._out), self.steps)

    def predict(self, X):
        pred = self._session.run(self._model_predictions,
                                 feed_dict={
                                     self._inp.name: X
                                 })
        if self.n_classes < 2:
            return pred
        return pred.argmax(axis=1)

    def predict_proba(self, X):
        pred = self._session.run(self._model.predictions,
                                 feed_dict={
                                     self._model.inp.name: X
                                 })
        if self.n_classes < 2:
            return pred
        return pred


class TensorFlowLinearRegressor(TensorFlowEstimator, RegressorMixin):
    """TensorFlow Linear Regression model."""
  
    def __init__(self, n_classes=0, tf_master="", batch_size=32, steps=50, optimizer="SGD",
                 learning_rate=0.1, tf_random_seed=42):
        super(TensorFlowLinearRegressor, self).__init__(
            model_fn=models.linear_regression, n_classes=n_classes,
            tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed)


class TensorFlowLinearClassifier(TensorFlowEstimator, ClassifierMixin):
    """TensorFlow Linear Classifier model."""
   
    def __init__(self, n_classes, tf_master="", batch_size=32, steps=50, optimizer="SGD",
                 learning_rate=0.1, tf_random_seed=42):
        super(TensorFlowLinearClassifier, self).__init__(
            model_fn=models.logistic_regression, n_classes=n_classes,
            tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed)


TensorFlowRegressor = TensorFlowLinearRegressor
TensorFlowClassifier = TensorFlowLinearClassifier


class TensorFlowDNNClassifier(TensorFlowEstimator, ClassifierMixin):
    """TensorFlow DNN Classifier model.
    
    Parameters:
        hidden_units: List of hidden units per layer.
        n_classes: Number of classes in the target.
        tf_master: TensorFlow master. Empty string is default for local.
        batch_size: Mini batch size.
        steps: Number of steps to run over data.
        optimizer: Optimizer name (or class), for example "SGD", "Adam",
                   "Adagrad".
        learning_rate: Learning rate for optimizer.
        tf_random_seed: Random seed for TensorFlow initializers.
            Setting this value, allows consistency between reruns.
    """
    
    def __init__(self, hidden_units, n_classes, tf_master="", batch_size=32, 
                 steps=50, optimizer="SGD", learning_rate=0.1, tf_random_seed=42):
        model_fn = models.get_dnn_model(hidden_units,
                                        models.logistic_regression)
        super(TensorFlowDNNClassifier, self).__init__(
            model_fn=model_fn, 
            n_classes=n_classes, tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed)


class TensorFlowDNNRegressor(TensorFlowEstimator, ClassifierMixin):
    """TensorFlow DNN Regressor model.
    
    Parameters:
        hidden_units: List of hidden units per layer.
        tf_master: TensorFlow master. Empty string is default for local.
        batch_size: Mini batch size.
        steps: Number of steps to run over data.
        optimizer: Optimizer name (or class), for example "SGD", "Adam",
                   "Adagrad".
        learning_rate: Learning rate for optimizer.
        tf_random_seed: Random seed for TensorFlow initializers.
            Setting this value, allows consistency between reruns.
    """
    
    def __init__(self, hidden_units, n_classes=0, tf_master="", batch_size=32, 
                 steps=50, optimizer="SGD", learning_rate=0.1, tf_random_seed=42):
        model_fn = models.get_dnn_model(hidden_units,
                                        models.linear_regression)
        super(TensorFlowDNNRegressor, self).__init__(
            model_fn=model_fn, 
            n_classes=n_classes, tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed)


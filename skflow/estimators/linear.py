"""Linear Estimators."""
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

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from skflow.estimators.base import TensorFlowEstimator
from skflow import models


class TensorFlowLinearRegressor(TensorFlowEstimator, RegressorMixin):
    """TensorFlow Linear Regression model."""

    def __init__(self, n_classes=0, tf_master="", batch_size=32, steps=50, optimizer="SGD",
                 learning_rate=0.1, tf_random_seed=42, continue_training=False,
                 verbose=1):
        super(TensorFlowLinearRegressor, self).__init__(
            model_fn=models.linear_regression, n_classes=n_classes,
            tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed,
            continue_training=continue_training,
            verbose=verbose)


class TensorFlowLinearClassifier(TensorFlowEstimator, ClassifierMixin):
    """TensorFlow Linear Classifier model."""

    def __init__(self, n_classes, tf_master="", batch_size=32, steps=50, optimizer="SGD",
                 learning_rate=0.1, tf_random_seed=42, continue_training=False,
                 verbose=1):
        super(TensorFlowLinearClassifier, self).__init__(
            model_fn=models.logistic_regression, n_classes=n_classes,
            tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed,
            continue_training=continue_training,
            verbose=verbose)


TensorFlowRegressor = TensorFlowLinearRegressor
TensorFlowClassifier = TensorFlowLinearClassifier

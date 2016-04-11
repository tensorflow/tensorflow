"""Linear Estimators."""
#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators.base import TensorFlowEstimator
from tensorflow.contrib.learn.python.learn import models


class TensorFlowLinearRegressor(TensorFlowEstimator, _sklearn.RegressorMixin):
    """TensorFlow Linear Regression model."""

    def __init__(self, n_classes=0, batch_size=32, steps=200, optimizer="Adagrad",
                 learning_rate=0.1, clip_gradients=5.0, continue_training=False,
                 config=None, verbose=1):

        super(TensorFlowLinearRegressor, self).__init__(
            model_fn=models.linear_regression_zero_init, n_classes=n_classes,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, clip_gradients=clip_gradients,
            continue_training=continue_training, config=config,
            verbose=verbose)

    @property
    def weights_(self):
        """Returns weights of the linear regression."""
        return self.get_tensor_value('linear_regression/weights:0')

    @property
    def bias_(self):
        """Returns bias of the linear regression."""
        return self.get_tensor_value('linear_regression/bias:0')


class TensorFlowLinearClassifier(TensorFlowEstimator, _sklearn.ClassifierMixin):
    """TensorFlow Linear Classifier model."""

    def __init__(self, n_classes, batch_size=32, steps=200, optimizer="Adagrad",
                 learning_rate=0.1, class_weight=None, clip_gradients=5.0,
                 continue_training=False, config=None,
                 verbose=1):

        super(TensorFlowLinearClassifier, self).__init__(
            model_fn=models.logistic_regression_zero_init, n_classes=n_classes,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, class_weight=class_weight,
            clip_gradients=clip_gradients,
            continue_training=continue_training, config=config,
            verbose=verbose)

    @property
    def weights_(self):
        """Returns weights of the linear classifier."""
        return self.get_tensor_value('logistic_regression/weights:0')

    @property
    def bias_(self):
        """Returns weights of the linear classifier."""
        return self.get_tensor_value('logistic_regression/bias:0')


TensorFlowRegressor = TensorFlowLinearRegressor
TensorFlowClassifier = TensorFlowLinearClassifier

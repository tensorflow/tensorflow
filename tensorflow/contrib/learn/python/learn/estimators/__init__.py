"""Scikit Flow Estimators."""
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

from tensorflow.contrib.learn.python.learn.estimators.autoencoder import TensorFlowDNNAutoencoder
from tensorflow.contrib.learn.python.learn.estimators.base import TensorFlowEstimator, TensorFlowBaseTransformer
from tensorflow.contrib.learn.python.learn.estimators.dnn import TensorFlowDNNClassifier
from tensorflow.contrib.learn.python.learn.estimators.dnn import TensorFlowDNNRegressor
from tensorflow.contrib.learn.python.learn.estimators.estimator import BaseEstimator
from tensorflow.contrib.learn.python.learn.estimators.estimator import Estimator
from tensorflow.contrib.learn.python.learn.estimators.estimator import ModeKeys
from tensorflow.contrib.learn.python.learn.estimators.linear import TensorFlowClassifier
from tensorflow.contrib.learn.python.learn.estimators.linear import TensorFlowLinearClassifier
from tensorflow.contrib.learn.python.learn.estimators.linear import TensorFlowLinearRegressor
from tensorflow.contrib.learn.python.learn.estimators.linear import TensorFlowRegressor
from tensorflow.contrib.learn.python.learn.estimators.rnn import TensorFlowRNNClassifier
from tensorflow.contrib.learn.python.learn.estimators.rnn import TensorFlowRNNRegressor
from tensorflow.contrib.learn.python.learn.estimators.run_config import RunConfig

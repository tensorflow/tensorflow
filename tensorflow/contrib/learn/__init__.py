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

# TODO(ptucker,ipolosukhin): Improve descriptions.
"""High level API for learning with TensorFlow.

## Estimators

Train and evaluate TensorFlow models.

@@BaseEstimator
@@Estimator
@@ModeKeys
@@TensorFlowClassifier
@@DNNClassifier
@@DNNRegressor
@@TensorFlowDNNClassifier
@@TensorFlowDNNRegressor
@@TensorFlowEstimator
@@LinearClassifier
@@LinearRegressor
@@TensorFlowLinearClassifier
@@TensorFlowLinearRegressor
@@TensorFlowRNNClassifier
@@TensorFlowRNNRegressor
@@TensorFlowRegressor

## Graph actions

Perform various training, evaluation, and inference actions on a graph.

@@NanLossDuringTrainingError
@@RunConfig
@@evaluate
@@infer
@@run_feeds
@@run_n
@@train

## Input processing

Queue and read batched input data.

@@extract_dask_data
@@extract_dask_labels
@@extract_pandas_data
@@extract_pandas_labels
@@extract_pandas_matrix
@@read_batch_examples
@@read_batch_features
@@read_batch_record_features

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import
from tensorflow.contrib.learn.python.learn import *
from tensorflow.python.util.all_util import make_all

__all__ = make_all(__name__)
__all__.append('datasets')

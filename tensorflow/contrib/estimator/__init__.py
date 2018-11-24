# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""estimator python module.

Importing from tensorflow.python.estimator
is unsupported and will soon break!
"""

# pylint: disable=unused-import,g-bad-import-order,g-import-not-at-top,wildcard-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Importing from tensorflow.python.estimator
# is unsupported and will soon break!

from tensorflow_estimator.contrib import estimator

# Fixes remove_undocumented not working as intended.
#
# Problem is that when the below import happens (for first time,
# Python only imports things once), Python sets attribute named
# 'python' to this package. If this first import happens
# after the call to remove_undocumented, then the 'python'
# attribute won't be removed.
import tensorflow.contrib.estimator.python

# Include attrs that start with single underscore.
_HAS_DYNAMIC_ATTRIBUTES = True
estimator.__all__ = [s for s in dir(estimator) if not s.startswith('__')]

from tensorflow_estimator.contrib.estimator import *
from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'add_metrics',
    'binary_classification_head',
    'clip_gradients_by_norm',
    'forward_features',
    'InMemoryEvaluatorHook',
    'make_stop_at_checkpoint_step_hook',
    'logistic_regression_head',
    'multi_class_head',
    'multi_head',
    'multi_label_head',
    'poisson_regression_head',
    'regression_head',
    'LinearEstimator',
    'boosted_trees_classifier_train_in_memory',
    'boosted_trees_regressor_train_in_memory',
    'call_logit_fn',
    'dnn_logit_fn_builder',
    'linear_logit_fn_builder',
    'replicate_model_fn',
    'TowerOptimizer',
    'RNNClassifier',
    'RNNEstimator',
    'export_saved_model_for_mode',
    'export_all_saved_models',
    'make_early_stopping_hook',
    'read_eval_metrics',
    'stop_if_lower_hook',
    'stop_if_higher_hook',
    'stop_if_no_increase_hook',
    'stop_if_no_decrease_hook',
    'build_raw_supervised_input_receiver_fn',
    'build_supervised_input_receiver_fn_from_input_fn',
    'SavedModelEstimator',
    'DNNClassifierWithLayerAnnotations',
    'DNNRegressorWithLayerAnnotations',
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)

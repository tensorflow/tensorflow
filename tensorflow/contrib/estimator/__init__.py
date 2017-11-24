# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Experimental utilities re:tf.estimator.*."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,wildcard-import
from tensorflow.contrib.estimator.python.estimator.dnn import *
from tensorflow.contrib.estimator.python.estimator.dnn_linear_combined import *
from tensorflow.contrib.estimator.python.estimator.extenders import *
from tensorflow.contrib.estimator.python.estimator.head import *
from tensorflow.contrib.estimator.python.estimator.linear import *
from tensorflow.contrib.estimator.python.estimator.logit_fns import *
from tensorflow.contrib.estimator.python.estimator.multi_head import *

from tensorflow.python.util.all_util import remove_undocumented
# pylint: enable=unused-import,line-too-long,wildcard-import

_allowed_symbols = [
    'add_metrics',
    'binary_classification_head',
    'clip_gradients_by_norm',
    'forward_features',
    'multi_class_head',
    'multi_head',
    'multi_label_head',
    'regression_head',
    'DNNEstimator',
    'DNNLinearCombinedEstimator',
    'LinearEstimator',
    'call_logit_fn',
    'dnn_logit_fn_builder',
    'linear_logit_fn_builder',
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)

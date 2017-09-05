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
"""Estimator: High level tools for working with models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,wildcard-import
from tensorflow.python.estimator.canned.dnn import DNNClassifier
from tensorflow.python.estimator.canned.dnn import DNNRegressor
from tensorflow.python.estimator.canned.dnn_linear_combined import DNNLinearCombinedClassifier
from tensorflow.python.estimator.canned.dnn_linear_combined import DNNLinearCombinedRegressor
from tensorflow.python.estimator.canned.linear import LinearClassifier
from tensorflow.python.estimator.canned.linear import LinearRegressor
from tensorflow.python.estimator.canned.parsing_utils import classifier_parse_example_spec
from tensorflow.python.estimator.canned.parsing_utils import regressor_parse_example_spec
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.export import export_lib as export
from tensorflow.python.estimator.inputs import inputs
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.python.estimator.run_config import RunConfig

from tensorflow.python.util.all_util import remove_undocumented
# pylint: enable=unused-import,line-too-long,wildcard-import

_allowed_symbols = [
    'DNNClassifier',
    'DNNRegressor',
    'DNNLinearCombinedClassifier',
    'DNNLinearCombinedRegressor',
    'LinearClassifier',
    'LinearRegressor',
    'classifier_parse_example_spec',
    'regressor_parse_example_spec',
    'inputs',
    'export',
    'Estimator',
    'EstimatorSpec',
    'ModeKeys',
    'RunConfig',
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)

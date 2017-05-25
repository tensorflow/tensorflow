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

# pylint: disable=unused-import,line-too-long
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.export import export_lib as export
from tensorflow.python.estimator.inputs import inputs
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.python.estimator.run_config import RunConfig

from tensorflow.python.util.all_util import remove_undocumented
# pylint: enable=unused-import,line-too-long

_allowed_symbols = [
    'inputs',
    'export',
    'Estimator',
    'EstimatorSpec',
    'ModeKeys',
    'RunConfig',
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)

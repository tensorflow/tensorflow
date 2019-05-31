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
"""Distribution-aware version of Optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.contrib.optimizer_v2.adadelta import AdadeltaOptimizer
from tensorflow.contrib.optimizer_v2.adagrad import AdagradOptimizer
from tensorflow.contrib.optimizer_v2.adam import AdamOptimizer
from tensorflow.contrib.optimizer_v2.gradient_descent import GradientDescentOptimizer
from tensorflow.contrib.optimizer_v2.momentum import MomentumOptimizer
from tensorflow.contrib.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.contrib.optimizer_v2.rmsprop import RMSPropOptimizer

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'AdadeltaOptimizer',
    'AdagradOptimizer',
    'AdamOptimizer',
    'GradientDescentOptimizer',
    'MomentumOptimizer',
    'OptimizerV2',
    'RMSPropOptimizer',
]

remove_undocumented(__name__, _allowed_symbols)

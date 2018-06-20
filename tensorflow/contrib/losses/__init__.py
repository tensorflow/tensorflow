# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Ops for building neural network losses.

See @{$python/contrib.losses}.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.losses.python import metric_learning
# pylint: disable=wildcard-import
from tensorflow.contrib.losses.python.losses import *
# pylint: enable=wildcard-import
from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'absolute_difference',
    'add_loss',
    'hinge_loss',
    'compute_weighted_loss',
    'cosine_distance',
    'get_losses',
    'get_regularization_losses',
    'get_total_loss',
    'log_loss',
    'mean_pairwise_squared_error',
    'mean_squared_error',
    'sigmoid_cross_entropy',
    'softmax_cross_entropy',
    'sparse_softmax_cross_entropy',
    'metric_learning'
]
remove_undocumented(__name__, _allowed_symbols)

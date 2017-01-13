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
"""Loss operations for use in neural networks.

Note: All the losses are added to the `GraphKeys.LOSSES` collection by default.

@@absolute_difference
@@compute_weighted_loss
@@cosine_distance
@@hinge_loss
@@log_loss
@@mean_pairwise_squared_error
@@mean_squared_error
@@sigmoid_cross_entropy
@@softmax_cross_entropy
@@sparse_softmax_cross_entropy

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from tensorflow.python.ops.losses import util
# pylint: disable=wildcard-import
from tensorflow.python.ops.losses.losses_impl import *
from tensorflow.python.ops.losses.util import *
# pylint: enable=wildcard-import
from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = []

remove_undocumented(__name__, _allowed_symbols,
                    [sys.modules[__name__], util])

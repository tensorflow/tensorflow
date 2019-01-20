# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# mixed_precisiond under the License is mixed_precisiond on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Library for mixed precision training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.mixed_precision.python.loss_scale_manager import *
from tensorflow.contrib.mixed_precision.python.loss_scale_optimizer import *

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    "LossScaleManager",
    "FixedLossScaleManager",
    "ExponentialUpdateLossScaleManager",
    "LossScaleOptimizer",
]

remove_undocumented(__name__, _allowed_symbols)

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
"""Model pruning implementation in tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.contrib.model_pruning.python.layers.layers import masked_conv2d
from tensorflow.contrib.model_pruning.python.layers.layers import masked_convolution
from tensorflow.contrib.model_pruning.python.layers.layers import masked_fully_connected
from tensorflow.contrib.model_pruning.python.layers.rnn_cells import MaskedBasicLSTMCell
from tensorflow.contrib.model_pruning.python.layers.rnn_cells import MaskedLSTMCell
from tensorflow.contrib.model_pruning.python.learning import train
from tensorflow.contrib.model_pruning.python.pruning import apply_mask
from tensorflow.contrib.model_pruning.python.pruning import get_masked_weights
from tensorflow.contrib.model_pruning.python.pruning import get_masks
from tensorflow.contrib.model_pruning.python.pruning import get_pruning_hparams
from tensorflow.contrib.model_pruning.python.pruning import get_thresholds
from tensorflow.contrib.model_pruning.python.pruning import get_weight_sparsity
from tensorflow.contrib.model_pruning.python.pruning import get_weights
from tensorflow.contrib.model_pruning.python.pruning import Pruning
# pylint: enable=unused-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'masked_convolution', 'masked_conv2d', 'masked_fully_connected',
    'MaskedBasicLSTMCell', 'MaskedLSTMCell', 'train', 'apply_mask',
    'get_masked_weights', 'get_masks', 'get_pruning_hparams', 'get_thresholds',
    'get_weights', 'get_weight_sparsity', 'Pruning'
]

remove_undocumented(__name__, _allowed_symbols)

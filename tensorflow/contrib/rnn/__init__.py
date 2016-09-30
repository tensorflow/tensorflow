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
"""Additional RNN operations and cells.

## This package provides additional contributed RNNCells.

### Block RNNCells
@@LSTMBlockCell
@@GRUBlockCell

### Fused RNNCells
@@FusedRNNCell
@@FusedRNNCellAdaptor
@@LSTMBlockFusedCell

### LSTM-like cells
@@CoupledInputForgetGateLSTMCell
@@TimeFreqLSTMCell
@@GridLSTMCell

### RNNCell wrappers
@@AttentionCellWrapper
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import, line-too-long
from tensorflow.contrib.rnn.python.ops.fused_rnn_cell import *
from tensorflow.contrib.rnn.python.ops.gru_ops import *
from tensorflow.contrib.rnn.python.ops.lstm_ops import *
from tensorflow.contrib.rnn.python.ops.rnn_cell import *

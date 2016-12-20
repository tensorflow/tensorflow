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
"""Module for constructing RNN Cells and additional RNN operations.

## Base interface for all RNN Cells

@@RNNCell

## RNN Cells for use with TensorFlow's core RNN methods

@@BasicRNNCell
@@BasicLSTMCell
@@GRUCell
@@LSTMCell

## Classes storing split `RNNCell` state

@@LSTMStateTuple

## RNN Cell wrappers (RNNCells that wrap other RNNCells)

@@MultiRNNCell
@@DropoutWrapper
@@EmbeddingWrapper
@@InputProjectionWrapper
@@OutputProjectionWrapper

### Block RNNCells
@@LSTMBlockCell
@@GRUBlockCell

### Fused RNNCells
@@FusedRNNCell
@@FusedRNNCellAdaptor
@@TimeReversedFusedRNN
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
from tensorflow.contrib.rnn.python.ops.rnn import *
from tensorflow.contrib.rnn.python.ops.rnn_cell import *
# pylint: enable=unused-import,wildcard-import,line-too-long

# Provides the links to core rnn and rnn_cell. Implementation will be moved in
# to this package instead of links as tracked in b/33235120.
from tensorflow.python.ops.rnn import bidirectional_rnn as static_bidirectional_rnn
from tensorflow.python.ops.rnn import rnn as static_rnn
from tensorflow.python.ops.rnn import state_saving_rnn as static_state_saving_rnn
from tensorflow.python.ops.rnn_cell import BasicLSTMCell
from tensorflow.python.ops.rnn_cell import BasicRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops.rnn_cell import EmbeddingWrapper
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import InputProjectionWrapper
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import OutputProjectionWrapper
from tensorflow.python.ops.rnn_cell import RNNCell

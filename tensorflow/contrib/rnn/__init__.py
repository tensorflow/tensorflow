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
@@LayerNormBasicLSTMCell

## Classes storing split `RNNCell` state

@@LSTMStateTuple

## RNN Cell wrappers (RNNCells that wrap other RNNCells)

@@MultiRNNCell
@@LSTMBlockWrapper
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


## Recurrent Neural Networks

TensorFlow provides a number of methods for constructing Recurrent Neural
Networks.

@@static_rnn
@@static_state_saving_rnn
@@static_bidirectional_rnn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.rnn.python.ops.core_rnn import static_bidirectional_rnn
from tensorflow.contrib.rnn.python.ops.core_rnn import static_rnn
from tensorflow.contrib.rnn.python.ops.core_rnn import static_state_saving_rnn

from tensorflow.contrib.rnn.python.ops.core_rnn_cell import BasicLSTMCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import BasicRNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import DropoutWrapper
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import EmbeddingWrapper
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import GRUCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import InputProjectionWrapper
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMStateTuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import MultiRNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import OutputProjectionWrapper
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell

# pylint: disable=unused-import,wildcard-import,line-too-long
from tensorflow.contrib.rnn.python.ops.fused_rnn_cell import *
from tensorflow.contrib.rnn.python.ops.gru_ops import *
from tensorflow.contrib.rnn.python.ops.lstm_ops import *
from tensorflow.contrib.rnn.python.ops.rnn import *
from tensorflow.contrib.rnn.python.ops.rnn_cell import *
# pylint: enable=unused-import,wildcard-import,line-too-long

from tensorflow.python.util.all_util import remove_undocumented
remove_undocumented(__name__, ['core_rnn_cell'])

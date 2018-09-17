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
"""RNN Cells and additional RNN operations.

See [Contrib RNN](https://tensorflow.org/api_guides/python/contrib.rnn) guide.

<!--From core-->
@@RNNCell
@@LayerRNNCell
@@BasicRNNCell
@@BasicLSTMCell
@@GRUCell
@@LSTMCell
@@LSTMStateTuple
@@DropoutWrapper
@@MultiRNNCell
@@DeviceWrapper
@@ResidualWrapper

<!--Used to be in core, but kept in contrib.-->
@@EmbeddingWrapper
@@InputProjectionWrapper
@@OutputProjectionWrapper

<!--Created in contrib, eventual plans to move to core.-->
@@LayerNormBasicLSTMCell
@@LSTMBlockWrapper
@@LSTMBlockCell
@@GRUBlockCell
@@GRUBlockCellV2
@@FusedRNNCell
@@FusedRNNCellAdaptor
@@TimeReversedFusedRNN
@@LSTMBlockFusedCell
@@CoupledInputForgetGateLSTMCell
@@TimeFreqLSTMCell
@@GridLSTMCell
@@BidirectionalGridLSTMCell
@@NASCell
@@UGRNNCell
@@IntersectionRNNCell
@@PhasedLSTMCell
@@ConvLSTMCell
@@Conv1DLSTMCell
@@Conv2DLSTMCell
@@Conv3DLSTMCell
@@HighwayWrapper
@@GLSTMCell
@@SRUCell
@@IndRNNCell
@@IndyGRUCell
@@IndyLSTMCell

<!--RNNCell wrappers-->
@@AttentionCellWrapper
@@CompiledWrapper

<!--RNN functions-->
@@static_rnn
@@static_state_saving_rnn
@@static_bidirectional_rnn
@@stack_bidirectional_dynamic_rnn
@@stack_bidirectional_rnn

<!--RNN utilities-->
@@transpose_batch_time
@@best_effort_input_batch_size
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import,line-too-long
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import EmbeddingWrapper
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import InputProjectionWrapper
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import OutputProjectionWrapper

from tensorflow.contrib.rnn.python.ops.fused_rnn_cell import *
from tensorflow.contrib.rnn.python.ops.gru_ops import *
from tensorflow.contrib.rnn.python.ops.lstm_ops import *
from tensorflow.contrib.rnn.python.ops.rnn import *
from tensorflow.contrib.rnn.python.ops.rnn_cell import *

from tensorflow.python.ops.rnn import _best_effort_input_batch_size as best_effort_input_batch_size
from tensorflow.python.ops.rnn import _transpose_batch_time as transpose_batch_time
from tensorflow.python.ops.rnn import static_bidirectional_rnn
from tensorflow.python.ops.rnn import static_rnn
from tensorflow.python.ops.rnn import static_state_saving_rnn

from tensorflow.python.ops.rnn_cell import *
# pylint: enable=unused-import,wildcard-import,line-too-long

from tensorflow.python.util.all_util import remove_undocumented
remove_undocumented(__name__)

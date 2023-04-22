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
"""Module implementing RNN Cells.

This module provides a number of basic commonly used RNN cells, such as LSTM
(Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number of
operators that allow adding dropouts, projections, or embeddings for inputs.
Constructing multi-layer cells is supported by the class `MultiRNNCell`, or by
calling the `rnn` ops several times.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.layers.legacy_rnn import rnn_cell_impl

# Remove caller that rely on private symbol in future.
# pylint: disable=protected-access
_BIAS_VARIABLE_NAME = rnn_cell_impl._BIAS_VARIABLE_NAME
_WEIGHTS_VARIABLE_NAME = rnn_cell_impl._WEIGHTS_VARIABLE_NAME
_concat = rnn_cell_impl._concat
_zero_state_tensors = rnn_cell_impl._zero_state_tensors
# pylint: disable=protected-access


assert_like_rnncell = rnn_cell_impl.assert_like_rnncell
ASSERT_LIKE_RNNCELL_ERROR_REGEXP = rnn_cell_impl.ASSERT_LIKE_RNNCELL_ERROR_REGEXP  # pylint: disable=line-too-long
BasicLSTMCell = rnn_cell_impl.BasicLSTMCell
BasicRNNCell = rnn_cell_impl.BasicRNNCell
DeviceWrapper = rnn_cell_impl.DeviceWrapper
DropoutWrapper = rnn_cell_impl.DropoutWrapper
GRUCell = rnn_cell_impl.GRUCell
LayerRNNCell = rnn_cell_impl.LayerRNNCell
LSTMCell = rnn_cell_impl.LSTMCell
LSTMStateTuple = rnn_cell_impl.LSTMStateTuple
MultiRNNCell = rnn_cell_impl.MultiRNNCell
ResidualWrapper = rnn_cell_impl.ResidualWrapper
RNNCell = rnn_cell_impl.RNNCell

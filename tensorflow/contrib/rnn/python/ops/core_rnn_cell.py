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

"""Module for constructing RNN Cells.

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
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import *
# pylint: enable=wildcard-import
from tensorflow.python.util.all_util import remove_undocumented


_allowed_symbols = []

remove_undocumented(__name__, _allowed_symbols)

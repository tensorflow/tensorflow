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
"""Deprecated library for creating sequence-to-sequence models in TensorFlow.

@@attention_decoder
@@basic_rnn_seq2seq
@@embedding_attention_decoder
@@embedding_attention_seq2seq
@@embedding_rnn_decoder
@@embedding_rnn_seq2seq
@@embedding_tied_rnn_seq2seq
@@model_with_buckets
@@one2many_rnn_seq2seq
@@rnn_decoder
@@sequence_loss
@@sequence_loss_by_example
@@tied_rnn_seq2seq
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import attention_decoder
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import basic_rnn_seq2seq
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import embedding_attention_decoder
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import embedding_attention_seq2seq
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import embedding_rnn_decoder
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import embedding_rnn_seq2seq
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import embedding_tied_rnn_seq2seq
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import model_with_buckets
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import one2many_rnn_seq2seq
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import sequence_loss
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import sequence_loss_by_example
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import tied_rnn_seq2seq

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = []

remove_undocumented(__name__, _allowed_symbols)

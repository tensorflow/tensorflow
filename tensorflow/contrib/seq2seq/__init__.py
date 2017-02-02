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

"""Ops for building neural network seq2seq decoders and losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# pylint: disable=unused-import,line-too-long
from tensorflow.contrib.seq2seq.python.ops.attention_decoder_fn import attention_decoder_fn_inference
from tensorflow.contrib.seq2seq.python.ops.attention_decoder_fn import attention_decoder_fn_train
from tensorflow.contrib.seq2seq.python.ops.attention_decoder_fn import prepare_attention
from tensorflow.contrib.seq2seq.python.ops.decoder_fn import *
from tensorflow.contrib.seq2seq.python.ops.loss import *
from tensorflow.contrib.seq2seq.python.ops.seq2seq import *
# pylint: enable=unused-import,line-too-long

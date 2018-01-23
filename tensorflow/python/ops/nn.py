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
# =============================================================================

# pylint: disable=unused-import,g-bad-import-order
"""Neural network support.

See the @{$python/nn} guide.

@@relu
@@relu6
@@crelu
@@swish
@@elu
@@leaky_relu
@@selu
@@softplus
@@softsign
@@dropout
@@bias_add
@@sigmoid
@@log_sigmoid
@@tanh
@@convolution
@@conv2d
@@depthwise_conv2d
@@depthwise_conv2d_native
@@separable_conv2d
@@atrous_conv2d
@@atrous_conv2d_transpose
@@conv2d_transpose
@@conv1d
@@conv3d
@@conv3d_transpose
@@conv2d_backprop_filter
@@conv2d_backprop_input
@@conv3d_backprop_filter_v2
@@depthwise_conv2d_native_backprop_filter
@@depthwise_conv2d_native_backprop_input
@@avg_pool
@@max_pool
@@max_pool_with_argmax
@@avg_pool3d
@@max_pool3d
@@fractional_avg_pool
@@fractional_max_pool
@@pool
@@dilation2d
@@erosion2d
@@with_space_to_batch
@@l2_normalize
@@local_response_normalization
@@sufficient_statistics
@@normalize_moments
@@moments
@@weighted_moments
@@fused_batch_norm
@@batch_normalization
@@batch_norm_with_global_normalization
@@l2_loss
@@log_poisson_loss
@@sigmoid_cross_entropy_with_logits
@@softmax
@@log_softmax
@@softmax_cross_entropy_with_logits
@@softmax_cross_entropy_with_logits_v2
@@sparse_softmax_cross_entropy_with_logits
@@weighted_cross_entropy_with_logits
@@embedding_lookup
@@embedding_lookup_sparse
@@dynamic_rnn
@@bidirectional_dynamic_rnn
@@raw_rnn
@@static_rnn
@@static_state_saving_rnn
@@static_bidirectional_rnn
@@ctc_loss
@@ctc_greedy_decoder
@@ctc_beam_search_decoder
@@top_k
@@in_top_k
@@nce_loss
@@sampled_softmax_loss
@@uniform_candidate_sampler
@@log_uniform_candidate_sampler
@@learned_unigram_candidate_sampler
@@fixed_unigram_candidate_sampler
@@compute_accidental_hits
@@quantized_conv2d
@@quantized_relu_x
@@quantized_max_pool
@@quantized_avg_pool
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys as _sys

# pylint: disable=unused-import
from tensorflow.python.ops import ctc_ops as _ctc_ops
from tensorflow.python.ops import embedding_ops as _embedding_ops
from tensorflow.python.ops import nn_grad as _nn_grad
from tensorflow.python.ops import nn_ops as _nn_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
# pylint: enable=unused-import
from tensorflow.python.util.all_util import remove_undocumented

# Bring more nn-associated functionality into this package.
# go/tf-wildcard-import
# pylint: disable=wildcard-import,unused-import
from tensorflow.python.ops.ctc_ops import *
from tensorflow.python.ops.nn_impl import *
from tensorflow.python.ops.nn_ops import *
from tensorflow.python.ops.candidate_sampling_ops import *
from tensorflow.python.ops.embedding_ops import *
from tensorflow.python.ops.rnn import *
from tensorflow.python.ops import rnn_cell
# pylint: enable=wildcard-import,unused-import


# TODO(cwhipkey): sigmoid and tanh should not be exposed from tf.nn.
_allowed_symbols = [
    "zero_fraction",  # documented in training.py
    # Modules whitelisted for reference through tf.nn.
    # TODO(cwhipkey): migrate callers to use the submodule directly.
    # Symbols whitelisted for export without documentation.
    # TODO(cwhipkey): review these and move to contrib or expose through
    # documentation.
    "all_candidate_sampler",  # Excluded in gen_docs_combined.
    "lrn",  # Excluded in gen_docs_combined.
    "relu_layer",  # Excluded in gen_docs_combined.
    "xw_plus_b",  # Excluded in gen_docs_combined.
    "rnn_cell",  # rnn_cell is a submodule of tf.nn.
]

remove_undocumented(__name__, _allowed_symbols,
                    [_sys.modules[__name__], _ctc_ops, _nn_ops, _nn_grad])

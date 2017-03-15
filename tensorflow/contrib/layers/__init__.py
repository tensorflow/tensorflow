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
"""Ops for building neural network layers, regularizers, summaries, etc.

See the @{$python/contrib.layers} guide.

@@avg_pool2d
@@batch_norm
@@convolution2d
@@conv2d_in_plane
@@convolution2d_in_plane
@@conv2d_transpose
@@convolution2d_transpose
@@dropout
@@embedding_lookup_unique
@@flatten
@@fully_connected
@@layer_norm
@@linear
@@max_pool2d
@@one_hot_encoding
@@relu
@@relu6
@@repeat
@@safe_embedding_lookup_sparse
@@separable_conv2d
@@separable_convolution2d
@@softmax
@@stack
@@unit_norm
@@bow_encoder
@@embed_sequence

@@apply_regularization
@@l1_regularizer
@@l2_regularizer
@@sum_regularizer

@@xavier_initializer
@@xavier_initializer_conv2d
@@variance_scaling_initializer

@@optimize_loss

@@summarize_activation
@@summarize_tensor
@@summarize_tensors
@@summarize_collection

@@summarize_activations

@@bucketized_column
@@check_feature_columns
@@create_feature_spec_for_parsing
@@crossed_column
@@embedding_column
@@scattered_embedding_column
@@input_from_feature_columns
@@joint_weighted_sum_from_feature_columns
@@make_place_holder_tensors_for_base_features
@@multi_class_target
@@one_hot_column
@@parse_feature_columns_from_examples
@@parse_feature_columns_from_sequence_examples
@@real_valued_column
@@shared_embedding_columns
@@sparse_column_with_hash_bucket
@@sparse_column_with_integerized_feature
@@sparse_column_with_keys
@@weighted_sparse_column
@@weighted_sum_from_feature_columns
@@infer_real_valued_columns
@@sequence_input_from_feature_columns
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.layers.python.layers import *
# pylint: enable=unused-import,wildcard-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = ['bias_add',
                    'conv2d',
                    'feature_column',
                    'legacy_fully_connected',
                    'legacy_linear',
                    'legacy_relu',
                    'OPTIMIZER_CLS_NAMES',
                    'regression_target',
                    'SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY',
                    'summaries']

remove_undocumented(__name__, _allowed_symbols)

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

## Higher level ops for building neural network layers.

This package provides several ops that take care of creating variables that are
used internally in a consistent way and provide the building blocks for many
common machine learning algorithms.

@@avg_pool2d
@@batch_norm
@@convolution2d
@@convolution2d_in_plane
@@convolution2d_transpose
@@flatten
@@fully_connected
@@max_pool2d
@@one_hot_encoding
@@repeat
@@separable_convolution2d
@@stack
@@unit_norm

Aliases for fully_connected which set a default activation function are
available: `relu`, `relu6` and `linear`.

## Regularizers

Regularization can help prevent overfitting. These have the signature
`fn(weights)`. The loss is typically added to `tf.GraphKeys.REGULARIZATION_LOSS`

@@apply_regularization
@@l1_regularizer
@@l2_regularizer
@@sum_regularizer

## Initializers

Initializers are used to initialize variables with sensible values given their
size, data type, and purpose.

@@xavier_initializer
@@xavier_initializer_conv2d
@@variance_scaling_initializer

## Optimization

Optimize weights given a loss.

@@optimize_loss

## Summaries

Helper functions to summarize specific variables or ops.

@@summarize_activation
@@summarize_tensor
@@summarize_tensors
@@summarize_collection

The layers module defines convenience functions `summarize_variables`,
`summarize_weights` and `summarize_biases`, which set the `collection` argument
of `summarize_collection` to `VARIABLES`, `WEIGHTS` and `BIASES`, respectively.

@@summarize_activations
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.layers.python.layers import *
from tensorflow.python.util.all_util import make_all

__all__ = make_all(__name__)

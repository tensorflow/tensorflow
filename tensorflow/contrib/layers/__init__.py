# Copyright 2015 Google Inc. All Rights Reserved.
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

@@convolution2d
@@fully_connected

Aliases for fully_connected which set a default activation function are
available: `relu`, `relu6` and `linear`.

## Regularizers

Regularization can help prevent overfitting. These have the signature
`fn(weights)`. The loss is typically added to `tf.GraphKeys.REGULARIZATION_LOSS`

@@l1_regularizer
@@l2_regularizer

## Initializers

Initializers are used to initialize variables with sensible values given their
size, data type, and purpose.

@@xavier_initializer
@@xavier_initializer_conv2d

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

# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.layers.python.framework.tensor_util import *
from tensorflow.contrib.layers.python.layers import *

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Probabilistic neural layers.

See ${python/contrib.bayesflow.layers}.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.contrib.bayesflow.python.ops.layers_conv_variational import *
from tensorflow.contrib.bayesflow.python.ops.layers_dense_variational_impl import *
from tensorflow.contrib.bayesflow.python.ops.layers_util import *
# pylint: enable=wildcard-import
from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'Convolution1DVariational',
    'Convolution2DVariational',
    'Convolution3DVariational',
    'Conv1DVariational',
    'Conv2DVariational',
    'Conv3DVariational',
    'convolution1d_variational',
    'convolution2d_variational',
    'convolution3d_variational',
    'conv1d_variational',
    'conv2d_variational',
    'conv3d_variational',
    'DenseReparameterization',
    'DenseLocalReparameterization',
    'DenseFlipout',
    'dense_reparameterization',
    'dense_local_reparameterization',
    'dense_flipout',
    'default_loc_scale_fn',
    'default_mean_field_normal_fn',
]

remove_undocumented(__name__, _allowed_symbols)

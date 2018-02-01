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
from tensorflow.contrib.bayesflow.python.ops.layers_dense_variational import *
from tensorflow.contrib.bayesflow.python.ops.layers_util import *
# pylint: enable=wildcard-import
from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'Convolution1DReparameterization',
    'Convolution2DReparameterization',
    'Convolution3DReparameterization',
    'Convolution1DFlipout',
    'Convolution2DFlipout',
    'Convolution3DFlipout',
    'Conv1DReparameterization',
    'Conv2DReparameterization',
    'Conv3DReparameterization',
    'Conv1DFlipout',
    'Conv2DFlipout',
    'Conv3DFlipout',
    'convolution1d_reparameterization',
    'convolution2d_reparameterization',
    'convolution3d_reparameterization',
    'convolution1d_flipout',
    'convolution2d_flipout',
    'convolution3d_flipout',
    'conv1d_reparameterization',
    'conv2d_reparameterization',
    'conv3d_reparameterization',
    'conv1d_flipout',
    'conv2d_flipout',
    'conv3d_flipout',
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

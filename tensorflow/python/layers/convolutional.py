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

"""Contains the convolutional layer classes and their functional aliases.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.legacy_tf_layers import convolutional

Conv1D = convolutional.Conv1D
conv1d = convolutional.conv1d
Conv2D = convolutional.Conv2D
conv2d = convolutional.conv2d
Conv3D = convolutional.Conv3D
conv3d = convolutional.conv3d
SeparableConv1D = convolutional.SeparableConv1D
SeparableConv2D = convolutional.SeparableConv2D
separable_conv1d = convolutional.separable_conv1d
separable_conv2d = convolutional.separable_conv2d
Conv2DTranspose = convolutional.Conv2DTranspose
conv2d_transpose = convolutional.conv2d_transpose
Conv3DTranspose = convolutional.Conv3DTranspose
conv3d_transpose = convolutional.conv3d_transpose

# Aliases

Convolution1D = Conv1D
Convolution2D = Conv2D
Convolution3D = Conv3D
SeparableConvolution2D = SeparableConv2D
Convolution2DTranspose = Deconvolution2D = Deconv2D = Conv2DTranspose
Convolution3DTranspose = Deconvolution3D = Deconv3D = Conv3DTranspose
convolution1d = conv1d
convolution2d = conv2d
convolution3d = conv3d
separable_convolution2d = separable_conv2d
convolution2d_transpose = deconvolution2d = deconv2d = conv2d_transpose
convolution3d_transpose = deconvolution3d = deconv3d = conv3d_transpose

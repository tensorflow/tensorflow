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

# pylint: disable=line-too-long
"""This library provides a set of high-level neural networks layers.

@@Dense
@@Dropout
@@Flatten
@@Conv1D
@@Conv2D
@@Conv3D
@@SeparableConv1D
@@SeparableConv2D
@@Conv2DTranspose
@@Conv3DTranspose
@@AveragePooling1D
@@MaxPooling1D
@@AveragePooling2D
@@MaxPooling2D
@@AveragePooling3D
@@MaxPooling3D
@@BatchNormalization

@@Layer
@@Input
@@InputSpec

@@dense
@@dropout
@@flatten
@@conv1d
@@conv2d
@@conv3d
@@separable_conv1d
@@separable_conv2d
@@conv2d_transpose
@@conv3d_transpose
@@average_pooling1d
@@max_pooling1d
@@average_pooling2d
@@max_pooling2d
@@average_pooling3d
@@max_pooling3d
@@batch_normalization
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.all_util import remove_undocumented

# pylint: disable=g-bad-import-order,unused-import

# Base objects.
from tensorflow.python.layers.base import Layer
from tensorflow.python.layers.base import InputSpec

# Core layers.
from tensorflow.python.layers.core import Dense
from tensorflow.python.layers.core import Dropout
from tensorflow.python.layers.core import Flatten

from tensorflow.python.layers.core import dense
from tensorflow.python.layers.core import dropout
from tensorflow.python.layers.core import flatten

# Convolutional layers.
from tensorflow.python.layers.convolutional import SeparableConv1D
from tensorflow.python.layers.convolutional import SeparableConv2D
from tensorflow.python.layers.convolutional import SeparableConvolution2D
from tensorflow.python.layers.convolutional import Conv2DTranspose
from tensorflow.python.layers.convolutional import Convolution2DTranspose
from tensorflow.python.layers.convolutional import Conv3DTranspose
from tensorflow.python.layers.convolutional import Convolution3DTranspose
from tensorflow.python.layers.convolutional import Conv1D
from tensorflow.python.layers.convolutional import Convolution1D
from tensorflow.python.layers.convolutional import Conv2D
from tensorflow.python.layers.convolutional import Convolution2D
from tensorflow.python.layers.convolutional import Conv3D
from tensorflow.python.layers.convolutional import Convolution3D

from tensorflow.python.layers.convolutional import separable_conv1d
from tensorflow.python.layers.convolutional import separable_conv2d
from tensorflow.python.layers.convolutional import conv2d_transpose
from tensorflow.python.layers.convolutional import conv3d_transpose
from tensorflow.python.layers.convolutional import conv1d
from tensorflow.python.layers.convolutional import conv2d
from tensorflow.python.layers.convolutional import conv3d

# Pooling layers.
from tensorflow.python.layers.pooling import AveragePooling1D
from tensorflow.python.layers.pooling import MaxPooling1D
from tensorflow.python.layers.pooling import AveragePooling2D
from tensorflow.python.layers.pooling import MaxPooling2D
from tensorflow.python.layers.pooling import AveragePooling3D
from tensorflow.python.layers.pooling import MaxPooling3D

from tensorflow.python.layers.pooling import average_pooling1d
from tensorflow.python.layers.pooling import max_pooling1d
from tensorflow.python.layers.pooling import average_pooling2d
from tensorflow.python.layers.pooling import max_pooling2d
from tensorflow.python.layers.pooling import average_pooling3d
from tensorflow.python.layers.pooling import max_pooling3d

# Normalization layers.
from tensorflow.python.layers.normalization import BatchNormalization

from tensorflow.python.layers.normalization import batch_normalization

# pylint: enable=g-bad-import-order,unused-import

_allowed_symbols = []

remove_undocumented(__name__, _allowed_symbols)

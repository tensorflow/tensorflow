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

"""Contains the pooling layer classes and their functional aliases.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.legacy_tf_layers import pooling


AveragePooling1D = pooling.AveragePooling1D
average_pooling1d = pooling.average_pooling1d
MaxPooling1D = pooling.MaxPooling1D
max_pooling1d = pooling.max_pooling1d
AveragePooling2D = pooling.AveragePooling2D
average_pooling2d = pooling.average_pooling2d
MaxPooling2D = pooling.MaxPooling2D
max_pooling2d = pooling.max_pooling2d
AveragePooling3D = pooling.AveragePooling3D
average_pooling3d = pooling.average_pooling3d
MaxPooling3D = pooling.MaxPooling3D
max_pooling3d = pooling.max_pooling3d

# Aliases

AvgPool2D = AveragePooling2D
MaxPool2D = MaxPooling2D
max_pool2d = max_pooling2d
avg_pool2d = average_pooling2d

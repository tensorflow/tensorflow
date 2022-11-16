# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Keras layers API."""

from tensorflow.python import tf2

# Generic layers.
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer

# Advanced activations.
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.advanced_activations import PReLU
from tensorflow.python.keras.layers.advanced_activations import ELU
from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.keras.layers.advanced_activations import ThresholdedReLU
from tensorflow.python.keras.layers.advanced_activations import Softmax

# Convolution layers.
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import Conv3D
from tensorflow.python.keras.layers.convolutional import Conv1DTranspose
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose
from tensorflow.python.keras.layers.convolutional import Conv3DTranspose
from tensorflow.python.keras.layers.convolutional import SeparableConv1D
from tensorflow.python.keras.layers.convolutional import SeparableConv2D

# Convolution layer aliases.
from tensorflow.python.keras.layers.convolutional import Convolution1D
from tensorflow.python.keras.layers.convolutional import Convolution2D
from tensorflow.python.keras.layers.convolutional import Convolution3D
from tensorflow.python.keras.layers.convolutional import Convolution2DTranspose
from tensorflow.python.keras.layers.convolutional import Convolution3DTranspose
from tensorflow.python.keras.layers.convolutional import SeparableConvolution1D
from tensorflow.python.keras.layers.convolutional import SeparableConvolution2D
from tensorflow.python.keras.layers.convolutional import DepthwiseConv2D

# Image processing layers.
from tensorflow.python.keras.layers.convolutional import UpSampling1D
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.python.keras.layers.convolutional import UpSampling3D
from tensorflow.python.keras.layers.convolutional import ZeroPadding1D
from tensorflow.python.keras.layers.convolutional import ZeroPadding2D
from tensorflow.python.keras.layers.convolutional import ZeroPadding3D
from tensorflow.python.keras.layers.convolutional import Cropping1D
from tensorflow.python.keras.layers.convolutional import Cropping2D
from tensorflow.python.keras.layers.convolutional import Cropping3D

# Core layers.
from tensorflow.python.keras.layers.core import Masking
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.core import SpatialDropout1D
from tensorflow.python.keras.layers.core import SpatialDropout2D
from tensorflow.python.keras.layers.core import SpatialDropout3D
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Reshape
from tensorflow.python.keras.layers.core import Permute
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.core import RepeatVector
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.core import ActivityRegularization

# Dense Attention layers.
from tensorflow.python.keras.layers.dense_attention import AdditiveAttention
from tensorflow.python.keras.layers.dense_attention import Attention

# Embedding layers.
from tensorflow.python.keras.layers.embeddings import Embedding

# Merge layers.
from tensorflow.python.keras.layers.merge import Add
from tensorflow.python.keras.layers.merge import Subtract
from tensorflow.python.keras.layers.merge import Multiply
from tensorflow.python.keras.layers.merge import Average
from tensorflow.python.keras.layers.merge import Maximum
from tensorflow.python.keras.layers.merge import Minimum
from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.python.keras.layers.merge import Dot
from tensorflow.python.keras.layers.merge import add
from tensorflow.python.keras.layers.merge import subtract
from tensorflow.python.keras.layers.merge import multiply
from tensorflow.python.keras.layers.merge import average
from tensorflow.python.keras.layers.merge import maximum
from tensorflow.python.keras.layers.merge import minimum
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers.merge import dot

# Pooling layers.
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling3D
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling1D
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling2D
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling3D

# Pooling layer aliases.
from tensorflow.python.keras.layers.pooling import MaxPool1D
from tensorflow.python.keras.layers.pooling import MaxPool2D
from tensorflow.python.keras.layers.pooling import MaxPool3D
from tensorflow.python.keras.layers.pooling import AvgPool1D
from tensorflow.python.keras.layers.pooling import AvgPool2D
from tensorflow.python.keras.layers.pooling import AvgPool3D
from tensorflow.python.keras.layers.pooling import GlobalAvgPool1D
from tensorflow.python.keras.layers.pooling import GlobalAvgPool2D
from tensorflow.python.keras.layers.pooling import GlobalAvgPool3D
from tensorflow.python.keras.layers.pooling import GlobalMaxPool1D
from tensorflow.python.keras.layers.pooling import GlobalMaxPool2D
from tensorflow.python.keras.layers.pooling import GlobalMaxPool3D

# Recurrent layers.
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.layers.recurrent import AbstractRNNCell
from tensorflow.python.keras.layers.recurrent import StackedRNNCells
from tensorflow.python.keras.layers.recurrent import SimpleRNNCell
from tensorflow.python.keras.layers.recurrent import PeepholeLSTMCell
from tensorflow.python.keras.layers.recurrent import SimpleRNN

if tf2.enabled():
  from tensorflow.python.keras.layers.recurrent_v2 import GRU
  from tensorflow.python.keras.layers.recurrent_v2 import GRUCell
  from tensorflow.python.keras.layers.recurrent_v2 import LSTM
  from tensorflow.python.keras.layers.recurrent_v2 import LSTMCell
  from tensorflow.python.keras.layers.recurrent import GRU as GRUV1
  from tensorflow.python.keras.layers.recurrent import GRUCell as GRUCellV1
  from tensorflow.python.keras.layers.recurrent import LSTM as LSTMV1
  from tensorflow.python.keras.layers.recurrent import LSTMCell as LSTMCellV1
  GRUV2 = GRU
  GRUCellV2 = GRUCell
  LSTMV2 = LSTM
  LSTMCellV2 = LSTMCell
else:
  from tensorflow.python.keras.layers.recurrent import GRU
  from tensorflow.python.keras.layers.recurrent import GRUCell
  from tensorflow.python.keras.layers.recurrent import LSTM
  from tensorflow.python.keras.layers.recurrent import LSTMCell
  from tensorflow.python.keras.layers.recurrent_v2 import GRU as GRUV2
  from tensorflow.python.keras.layers.recurrent_v2 import GRUCell as GRUCellV2
  from tensorflow.python.keras.layers.recurrent_v2 import LSTM as LSTMV2
  from tensorflow.python.keras.layers.recurrent_v2 import LSTMCell as LSTMCellV2
  GRUV1 = GRU
  GRUCellV1 = GRUCell
  LSTMV1 = LSTM
  LSTMCellV1 = LSTMCell

# Convolutional-recurrent layers.
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2D

# CuDNN recurrent layers.
from tensorflow.python.keras.layers.cudnn_recurrent import CuDNNLSTM
from tensorflow.python.keras.layers.cudnn_recurrent import CuDNNGRU

# # RNN Cell wrappers.
from tensorflow.python.keras.layers.rnn_cell_wrapper_v2 import DeviceWrapper
from tensorflow.python.keras.layers.rnn_cell_wrapper_v2 import DropoutWrapper
from tensorflow.python.keras.layers.rnn_cell_wrapper_v2 import ResidualWrapper

# Serialization functions
from tensorflow.python.keras.layers import serialization
from tensorflow.python.keras.layers.serialization import deserialize
from tensorflow.python.keras.layers.serialization import serialize


class VersionAwareLayers(object):
  """Utility to be used internally to access layers in a V1/V2-aware fashion.

  When using layers within the Keras codebase, under the constraint that
  e.g. `layers.BatchNormalization` should be the `BatchNormalization` version
  corresponding to the current runtime (TF1 or TF2), do not simply access
  `layers.BatchNormalization` since it would ignore e.g. an early
  `compat.v2.disable_v2_behavior()` call. Instead, use an instance
  of `VersionAwareLayers` (which you can use just like the `layers` module).
  """

  def __getattr__(self, name):
    serialization.populate_deserializable_objects()
    if name in serialization.LOCAL.ALL_OBJECTS:
      return serialization.LOCAL.ALL_OBJECTS[name]
    return super(VersionAwareLayers, self).__getattr__(name)

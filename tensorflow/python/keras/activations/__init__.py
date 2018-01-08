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
"""Keras built-in activation functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Activation functions.
from tensorflow.python.keras._impl.keras.activations import elu
from tensorflow.python.keras._impl.keras.activations import hard_sigmoid
from tensorflow.python.keras._impl.keras.activations import linear
from tensorflow.python.keras._impl.keras.activations import relu
from tensorflow.python.keras._impl.keras.activations import selu
from tensorflow.python.keras._impl.keras.activations import sigmoid
from tensorflow.python.keras._impl.keras.activations import softmax
from tensorflow.python.keras._impl.keras.activations import softplus
from tensorflow.python.keras._impl.keras.activations import softsign
from tensorflow.python.keras._impl.keras.activations import tanh

# Auxiliary utils.
# pylint: disable=g-bad-import-order
from tensorflow.python.keras._impl.keras.activations import deserialize
from tensorflow.python.keras._impl.keras.activations import serialize
from tensorflow.python.keras._impl.keras.activations import get

del absolute_import
del division
del print_function

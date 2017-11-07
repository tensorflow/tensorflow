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
"""The Keras API.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras._impl.keras import activations
from tensorflow.python.keras._impl.keras import applications
from tensorflow.python.keras._impl.keras import backend
from tensorflow.python.keras._impl.keras import callbacks
from tensorflow.python.keras._impl.keras import constraints
from tensorflow.python.keras._impl.keras import datasets
from tensorflow.python.keras._impl.keras import engine
from tensorflow.python.keras._impl.keras import estimator
from tensorflow.python.keras._impl.keras import initializers
from tensorflow.python.keras._impl.keras import layers
from tensorflow.python.keras._impl.keras import losses
from tensorflow.python.keras._impl.keras import metrics
from tensorflow.python.keras._impl.keras import models
from tensorflow.python.keras._impl.keras import optimizers
from tensorflow.python.keras._impl.keras import preprocessing
from tensorflow.python.keras._impl.keras import regularizers
from tensorflow.python.keras._impl.keras import utils
from tensorflow.python.keras._impl.keras import wrappers
from tensorflow.python.keras._impl.keras.layers import Input
from tensorflow.python.keras._impl.keras.models import Model
from tensorflow.python.keras._impl.keras.models import Sequential

__version__ = '2.0.8-tf'

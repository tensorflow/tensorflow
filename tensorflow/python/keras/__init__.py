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
"""Implementation of the Keras API meant to be a high-level API for TensorFlow.

Detailed documentation and user guides are available at
[keras.io](https://keras.io).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import activations
from tensorflow.python.keras import applications
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import constraints
from tensorflow.python.keras import datasets
from tensorflow.python.keras import estimator
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import preprocessing
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import utils
from tensorflow.python.keras import wrappers
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential

from tensorflow.python.util.tf_export import tf_export

__version__ = '2.2.4-tf'

tf_export('keras.__version__').export_constant(__name__, '__version__')

del absolute_import
del division
del print_function

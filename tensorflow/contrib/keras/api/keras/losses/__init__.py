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
"""Keras built-in loss functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Loss functions.
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.losses import categorical_hinge
from tensorflow.python.keras.losses import cosine_proximity
from tensorflow.python.keras.losses import hinge
from tensorflow.python.keras.losses import kullback_leibler_divergence
from tensorflow.python.keras.losses import logcosh
from tensorflow.python.keras.losses import mean_absolute_error
from tensorflow.python.keras.losses import mean_absolute_percentage_error
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import mean_squared_logarithmic_error
from tensorflow.python.keras.losses import poisson
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.losses import squared_hinge

# Auxiliary utils.
# pylint: disable=g-bad-import-order
from tensorflow.python.keras.losses import deserialize
from tensorflow.python.keras.losses import serialize
from tensorflow.python.keras.losses import get

del absolute_import
del division
del print_function

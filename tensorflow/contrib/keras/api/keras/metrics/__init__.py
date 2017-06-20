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
"""Keras built-in metrics functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Metrics functions.
from tensorflow.contrib.keras.python.keras.metrics import binary_accuracy
from tensorflow.contrib.keras.python.keras.metrics import binary_crossentropy
from tensorflow.contrib.keras.python.keras.metrics import categorical_accuracy
from tensorflow.contrib.keras.python.keras.metrics import categorical_crossentropy
from tensorflow.contrib.keras.python.keras.metrics import cosine_proximity
from tensorflow.contrib.keras.python.keras.metrics import hinge
from tensorflow.contrib.keras.python.keras.metrics import kullback_leibler_divergence
from tensorflow.contrib.keras.python.keras.metrics import mean_absolute_error
from tensorflow.contrib.keras.python.keras.metrics import mean_absolute_percentage_error
from tensorflow.contrib.keras.python.keras.metrics import mean_squared_error
from tensorflow.contrib.keras.python.keras.metrics import mean_squared_logarithmic_error
from tensorflow.contrib.keras.python.keras.metrics import poisson
from tensorflow.contrib.keras.python.keras.metrics import sparse_categorical_crossentropy
from tensorflow.contrib.keras.python.keras.metrics import squared_hinge
from tensorflow.contrib.keras.python.keras.metrics import top_k_categorical_accuracy

# Auxiliary utils.
# pylint: disable=g-bad-import-order
from tensorflow.contrib.keras.python.keras.metrics import deserialize
from tensorflow.contrib.keras.python.keras.metrics import serialize
from tensorflow.contrib.keras.python.keras.metrics import get

del absolute_import
del division
del print_function

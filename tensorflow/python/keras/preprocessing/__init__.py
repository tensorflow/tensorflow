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
"""Provides keras data preprocessing utils to pre-process tf.data.Datasets before they are fed to the model."""
# pylint: disable=g-import-not-at-top
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(mihaimaruseac): remove the import of keras_preprocessing and injecting
# once we update to latest version of keras_preprocessing
import keras_preprocessing

from tensorflow.python.keras import backend
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from tensorflow.python.keras.preprocessing import timeseries
from tensorflow.python.keras.utils import all_utils as utils

# This exists for compatibility with prior version of keras_preprocessing.
keras_preprocessing.set_keras_submodules(backend=backend, utils=utils)

del absolute_import
del division
del print_function

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Constants for Keras SavedModel serialization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Namespace used to store all attributes added during serialization.
# e.g. the list of layers can be accessed using `loaded.keras_api.layers`, in an
# object loaded from `tf.saved_model.load()`.
KERAS_ATTR = 'keras_api'

# Keys for the serialization cache.
# Maps to the keras serialization dict {Layer --> SerializedAttributes object}
KERAS_CACHE_KEY = 'keras_serialized_attributes'

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
"""Layer serialization/deserialization functions.
"""
# pylint: disable=wildcard-import
# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.engine import Input
from tensorflow.python.keras.engine import InputLayer
from tensorflow.python.keras.layers.advanced_activations import *
from tensorflow.python.keras.layers.convolutional import *
from tensorflow.python.keras.layers.convolutional_recurrent import *
from tensorflow.python.keras.layers.core import *
from tensorflow.python.keras.layers.cudnn_recurrent import *
from tensorflow.python.keras.layers.embeddings import *
from tensorflow.python.keras.layers.local import *
from tensorflow.python.keras.layers.merge import *
from tensorflow.python.keras.layers.noise import *
from tensorflow.python.keras.layers.normalization import *
from tensorflow.python.keras.layers.pooling import *
from tensorflow.python.keras.layers.recurrent import *
from tensorflow.python.keras.layers.wrappers import *
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object


def serialize(layer):
  return {'class_name': layer.__class__.__name__, 'config': layer.get_config()}


def deserialize(config, custom_objects=None):
  """Instantiates a layer from a config dictionary.

  Arguments:
      config: dict of the form {'class_name': str, 'config': dict}
      custom_objects: dict mapping class names (or function names)
          of custom (non-Keras) objects to class/functions

  Returns:
      Layer instance (may be Model, Sequential, Layer...)
  """
  from tensorflow.python.keras import models  # pylint: disable=g-import-not-at-top
  globs = globals()  # All layers.
  globs['Model'] = models.Model
  globs['Sequential'] = models.Sequential
  return deserialize_keras_object(
      config,
      module_objects=globs,
      custom_objects=custom_objects,
      printable_module_name='layer')

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

from tensorflow.python import tf2
from tensorflow.python.keras.engine.base_layer import AddLoss
from tensorflow.python.keras.engine.base_layer import AddMetric
from tensorflow.python.keras.engine.base_layer import TensorFlowOpLayer
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers.advanced_activations import *
from tensorflow.python.keras.layers.convolutional import *
from tensorflow.python.keras.layers.convolutional_recurrent import *
from tensorflow.python.keras.layers.core import *
from tensorflow.python.keras.layers.cudnn_recurrent import *
from tensorflow.python.keras.layers.dense_attention import *
from tensorflow.python.keras.layers.embeddings import *
from tensorflow.python.keras.layers.local import *
from tensorflow.python.keras.layers.merge import *
from tensorflow.python.keras.layers.noise import *
from tensorflow.python.keras.layers.normalization import *
from tensorflow.python.keras.layers.pooling import *
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import *
from tensorflow.python.keras.layers.recurrent import *
from tensorflow.python.keras.layers.rnn_cell_wrapper_v2 import *
from tensorflow.python.keras.layers.wrappers import *
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.util.tf_export import keras_export

if tf2.enabled():
  from tensorflow.python.keras.layers.normalization_v2 import *  # pylint: disable=g-import-not-at-top
  from tensorflow.python.keras.layers.recurrent_v2 import *     # pylint: disable=g-import-not-at-top

# This deserialization table is added for backward compatibility, as in TF 1.13,
# BatchNormalizationV1 and BatchNormalizationV2 are used as class name for v1
# and v2 version of BatchNormalization, respectively. Here we explicitly convert
# them to the canonical name in the config of deserialization.
_DESERIALIZATION_TABLE = {
    'BatchNormalizationV1': 'BatchNormalization',
    'BatchNormalizationV2': 'BatchNormalization',
}


@keras_export('keras.layers.serialize')
def serialize(layer):
  return serialize_keras_object(layer)


@keras_export('keras.layers.deserialize')
def deserialize(config, custom_objects=None):
  """Instantiates a layer from a config dictionary.

  Arguments:
      config: dict of the form {'class_name': str, 'config': dict}
      custom_objects: dict mapping class names (or function names)
          of custom (non-Keras) objects to class/functions

  Returns:
      Layer instance (may be Model, Sequential, Network, Layer...)
  """
  # Prevent circular dependencies.
  from tensorflow.python.keras import models  # pylint: disable=g-import-not-at-top
  from tensorflow.python.keras.premade.linear import LinearModel  # pylint: disable=g-import-not-at-top
  from tensorflow.python.keras.premade.wide_deep import WideDeepModel  # pylint: disable=g-import-not-at-top
  from tensorflow.python.feature_column import dense_features  # pylint: disable=g-import-not-at-top
  from tensorflow.python.feature_column import sequence_feature_column as sfc  # pylint: disable=g-import-not-at-top

  globs = globals()  # All layers.
  globs['Network'] = models.Network
  globs['Model'] = models.Model
  globs['Sequential'] = models.Sequential
  globs['LinearModel'] = LinearModel
  globs['WideDeepModel'] = WideDeepModel

  # Prevent circular dependencies with FeatureColumn serialization.
  globs['DenseFeatures'] = dense_features.DenseFeatures
  globs['SequenceFeatures'] = sfc.SequenceFeatures

  layer_class_name = config['class_name']
  if layer_class_name in _DESERIALIZATION_TABLE:
    config['class_name'] = _DESERIALIZATION_TABLE[layer_class_name]

  return deserialize_keras_object(
      config,
      module_objects=globs,
      custom_objects=custom_objects,
      printable_module_name='layer')

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
"""Contains the get_layer_policy function.

This is a separate file from policy.py to avoid a circular dependency.
get_layer_policy() relies on base_layer.py, itself which relies on policy.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.engine import base_layer
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.mixed_precision.experimental.get_layer_policy')
def get_layer_policy(layer):
  """Returns the dtype policy of a layer.

  Args:
    layer: A `tf.keras.layers.Layer`.

  Returns:
    The `tf.keras.mixed_precision.experimental.Policy` of the layer.
  """
  if not isinstance(layer, base_layer.Layer):
    raise ValueError('get_policy can only be called on a layer, but got: %s'
                     % (layer,))
  return layer._dtype_policy  # pylint: disable=protected-access

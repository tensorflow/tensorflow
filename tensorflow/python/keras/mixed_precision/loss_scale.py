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
"""Contains keras-specific LossScale functionality.

This functions cannot be in the non-keras loss_scale.py file since they depend
on keras, and files outside of keras should not depend on files inside keras.
"""

from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.training.experimental import loss_scale as loss_scale_module


def serialize(loss_scale):
  return generic_utils.serialize_keras_object(loss_scale)


def deserialize(config, custom_objects=None):
  loss_scale_module_objects = {
      'FixedLossScale': loss_scale_module.FixedLossScale,
      'DynamicLossScale': loss_scale_module.DynamicLossScale,
  }

  return generic_utils.deserialize_keras_object(
      config,
      module_objects=loss_scale_module_objects,
      custom_objects=custom_objects,
      printable_module_name='loss scale'
  )


def get(identifier):
  """Get a loss scale object."""
  if isinstance(identifier, dict):
    return deserialize(identifier)

  if isinstance(identifier, (int, float)):
    return loss_scale_module.FixedLossScale(identifier)
  if identifier == 'dynamic':
    return loss_scale_module.DynamicLossScale()
  if isinstance(identifier, loss_scale_module.LossScale):
    return identifier
  elif identifier is None:
    return None
  else:
    raise ValueError('Could not interpret loss scale identifier: %s' %
                     identifier)

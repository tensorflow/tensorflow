# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities related to layer/model functionality."""

# TODO(b/110718070): Move these functions back to tensorflow/python/keras/utils
# once __init__ files no longer require all of tf.keras to be imported together.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training.tracking import object_identity


def is_layer(obj):
  """Implicit check for Layer-like objects."""
  # TODO(b/110718070): Replace with isinstance(obj, base_layer.Layer).
  return hasattr(obj, "_is_layer") and not isinstance(obj, type)


def has_weights(obj):
  """Implicit check for Layer-like objects."""
  # TODO(b/110718070): Replace with isinstance(obj, base_layer.Layer).
  has_weight = (hasattr(type(obj), "trainable_weights")
                and hasattr(type(obj), "non_trainable_weights"))

  return has_weight and not isinstance(obj, type)


def filter_empty_layer_containers(layer_list):
  """Filter out empty Layer-like containers and uniquify."""
  existing = object_identity.ObjectIdentitySet()
  to_visit = layer_list[::-1]
  filtered = []
  while to_visit:
    obj = to_visit.pop()
    if obj in existing:
      continue
    existing.add(obj)
    if is_layer(obj):
      filtered.append(obj)
    elif hasattr(obj, "layers"):
      # Trackable data structures will not show up in ".layers" lists, but
      # the layers they contain will.
      to_visit.extend(obj.layers[::-1])
  return filtered


def gather_trainable_weights(trainable, sub_layers, extra_variables):
  """Lists the trainable weights for an object with sub-layers.

  Args:
    trainable: Whether the object collecting the variables is trainable.
    sub_layers: A flat list of Layer objects owned by this object, to collect
      variables from.
    extra_variables: Any extra variables to include. Their `.trainable` property
      is used to categorize them.

  Returns:
    A list of collected trainable weights/variables.
  """
  if not trainable:
    return []
  weights = []
  for layer in sub_layers:
    weights += layer.trainable_weights
  trainable_extra_variables = [
      v for v in extra_variables if v.trainable]
  return weights + trainable_extra_variables


def gather_non_trainable_weights(trainable, sub_layers, extra_variables):
  """Lists the non-trainable weights for an object with sub-layers.

  Args:
    trainable: Whether the object collecting the variables is trainable.
    sub_layers: A flat list of Layer objects owned by this object, to collect
      variables from.
    extra_variables: Any extra variables to include. Their `.trainable` property
      is used to categorize them.

  Returns:
    A list of collected non-trainable weights/variables.
  """
  trainable_extra_variables = []
  non_trainable_extra_variables = []
  for v in extra_variables:
    if v.trainable:
      trainable_extra_variables.append(v)
    else:
      non_trainable_extra_variables.append(v)
  weights = []
  for layer in sub_layers:
    weights += layer.non_trainable_weights
  if not trainable:
    trainable_weights = []
    for layer in sub_layers:
      trainable_weights += layer.trainable_weights
    return (trainable_weights + trainable_extra_variables
            + weights + non_trainable_extra_variables)
  return weights + non_trainable_extra_variables

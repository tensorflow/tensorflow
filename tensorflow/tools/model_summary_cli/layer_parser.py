# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Layer information extraction for tf_model_summary CLI."""


def get_layer_info(layer):
  """Extract information from a single layer.

  Args:
    layer: A Keras layer instance.

  Returns:
    A dictionary containing layer information.
  """
  info = {
      'name': layer.name,
      'class_name': layer.__class__.__name__,
      'trainable': layer.trainable,
      'dtype': str(layer.dtype) if hasattr(layer, 'dtype') else 'unknown',
  }

  # Get output shape safely
  try:
    info['output_shape'] = layer.output_shape
  except AttributeError:
    info['output_shape'] = 'multiple'
  except RuntimeError:
    info['output_shape'] = '?'

  # Get parameter count safely
  try:
    if not layer.built:
      info['params'] = 0
      info['params_note'] = 'unused'
    else:
      info['params'] = layer.count_params()
      info['params_note'] = None
  except Exception:  # pylint: disable=broad-except
    info['params'] = 0
    info['params_note'] = 'error'

  return info


def get_model_info(model):
  """Extract comprehensive information from a model.

  Args:
    model: A Keras model instance.

  Returns:
    A dictionary containing model information.
  """
  from tensorflow.python.keras.utils import layer_utils

  # Determine model type
  is_sequential = model.__class__.__name__ == 'Sequential'
  is_graph_network = getattr(model, '_is_graph_network', False)

  info = {
      'name': model.name,
      'class_name': model.__class__.__name__,
      'is_sequential': is_sequential,
      'is_graph_network': is_graph_network,
      'layers': [],
  }

  # Extract layer information
  for layer in model.layers:
    info['layers'].append(get_layer_info(layer))

  # Calculate parameter totals
  try:
    trainable_weights = getattr(
        model, '_collected_trainable_weights',
        model.trainable_weights
    )
    info['trainable_params'] = layer_utils.count_params(trainable_weights)
    info['non_trainable_params'] = layer_utils.count_params(
        model.non_trainable_weights
    )
    info['total_params'] = info['trainable_params'] + info['non_trainable_params']
  except Exception:  # pylint: disable=broad-except
    info['trainable_params'] = 0
    info['non_trainable_params'] = 0
    info['total_params'] = 0

  return info

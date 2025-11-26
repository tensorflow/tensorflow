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
"""Output formatting for tf_model_summary CLI."""


def format_params(count, note=None):
  """Format parameter count with optional note.

  Args:
    count: Integer parameter count.
    note: Optional note string (e.g., 'unused').

  Returns:
    Formatted string.
  """
  if note:
    return f"{count:,} ({note})"
  return f"{count:,}"


def format_shape(shape):
  """Format a shape tuple for display.

  Args:
    shape: Shape tuple or string.

  Returns:
    Formatted string.
  """
  if isinstance(shape, str):
    return shape
  if shape is None:
    return '?'
  return str(shape)


def format_model_summary_json(model_info):
  """Format model information as JSON.

  Args:
    model_info: Dictionary from get_model_info().

  Returns:
    JSON string.
  """
  import json

  # Convert to JSON-serializable format
  output = {
      'model_name': model_info['name'],
      'model_class': model_info['class_name'],
      'total_params': model_info['total_params'],
      'trainable_params': model_info['trainable_params'],
      'non_trainable_params': model_info['non_trainable_params'],
      'layers': []
  }

  for layer in model_info['layers']:
    layer_output = {
        'name': layer['name'],
        'class': layer['class_name'],
        'output_shape': format_shape(layer['output_shape']),
        'params': layer['params'],
        'trainable': layer['trainable'],
    }
    output['layers'].append(layer_output)

  return json.dumps(output, indent=2)

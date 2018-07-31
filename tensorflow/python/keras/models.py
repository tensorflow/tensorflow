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
# pylint: disable=protected-access
"""Code for model cloning, plus model-related API entries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import saving
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils.generic_utils import has_arg


# API entries importable from `keras.models`:
Model = training.Model  # pylint: disable=invalid-name
Sequential = sequential.Sequential  # pylint: disable=invalid-name
save_model = saving.save_model
load_model = saving.load_model
model_from_config = saving.model_from_config
model_from_yaml = saving.model_from_yaml
model_from_json = saving.model_from_json


def _clone_functional_model(model, input_tensors=None):
  """Clone a functional `Model` instance.

  Model cloning is similar to calling a model on new inputs,
  except that it creates new layers (and thus new weights) instead
  of sharing the weights of the existing layers.

  Arguments:
      model: Instance of `Model`.
      input_tensors: optional list of input tensors
          to build the model upon. If not provided,
          placeholders will be created.

  Returns:
      An instance of `Model` reproducing the behavior
      of the original model, on top of new inputs tensors,
      using newly instantiated weights.

  Raises:
      ValueError: in case of invalid `model` argument value.
  """
  if not isinstance(model, Model):
    raise ValueError('Expected `model` argument '
                     'to be a `Model` instance, got ', model)
  if isinstance(model, Sequential):
    raise ValueError('Expected `model` argument '
                     'to be a functional `Model` instance, '
                     'got a `Sequential` instance instead:', model)

  layer_map = {}  # Cache for created layers.
  tensor_map = {}  # Map {reference_tensor: (corresponding_tensor, mask)}
  if input_tensors is None:
    # Create placeholders to build the model on top of.
    input_layers = []
    input_tensors = []
    for layer in model._input_layers:
      input_tensor = Input(
          batch_shape=layer._batch_input_shape,
          dtype=layer.dtype,
          sparse=layer.sparse,
          name=layer.name)
      input_tensors.append(input_tensor)
      # Cache newly created input layer.
      newly_created_input_layer = input_tensor._keras_history[0]
      layer_map[layer] = newly_created_input_layer
    for original_input_layer, cloned_input_layer in zip(model._input_layers,
                                                        input_layers):
      layer_map[original_input_layer] = cloned_input_layer
  else:
    # Make sure that all input tensors come from a Keras layer.
    # If tensor comes from an input layer: cache the input layer.
    input_tensors = generic_utils.to_list(input_tensors)
    input_tensors_ = []
    for i, x in enumerate(input_tensors):
      if not K.is_keras_tensor(x):
        name = model._input_layers[i].name
        input_tensor = Input(tensor=x, name='input_wrapper_for_' + name)
        input_tensors_.append(input_tensor)
        # Cache newly created input layer.
        original_input_layer = x._keras_history[0]
        newly_created_input_layer = input_tensor._keras_history[0]
        layer_map[original_input_layer] = newly_created_input_layer
      else:
        input_tensors_.append(x)
    input_tensors = input_tensors_

  for x, y in zip(model.inputs, input_tensors):
    tensor_map[x] = (y, None)  # tensor, mask

  # Iterated over every node in the reference model, in depth order.
  depth_keys = list(model._nodes_by_depth.keys())
  depth_keys.sort(reverse=True)
  for depth in depth_keys:
    nodes = model._nodes_by_depth[depth]
    for node in nodes:
      # Recover the corresponding layer.
      layer = node.outbound_layer

      # Get or create layer.
      if layer not in layer_map:
        # Clone layer.
        new_layer = layer.__class__.from_config(layer.get_config())
        layer_map[layer] = new_layer
        layer = new_layer
      else:
        # Reuse previously cloned layer.
        layer = layer_map[layer]
        # Don't call InputLayer multiple times.
        if isinstance(layer, InputLayer):
          continue

      # Gather inputs to call the new layer.
      referenceinput_tensors_ = node.input_tensors
      reference_output_tensors = node.output_tensors

      # If all previous input tensors are available in tensor_map,
      # then call node.inbound_layer on them.
      computed_data = []  # List of tuples (input, mask).
      for x in referenceinput_tensors_:
        if x in tensor_map:
          computed_data.append(tensor_map[x])

      if len(computed_data) == len(referenceinput_tensors_):
        # Call layer.
        if node.arguments:
          kwargs = node.arguments
        else:
          kwargs = {}
        if len(computed_data) == 1:
          computed_tensor, computed_mask = computed_data[0]
          if has_arg(layer.call, 'mask'):
            if 'mask' not in kwargs:
              kwargs['mask'] = computed_mask
          output_tensors = generic_utils.to_list(layer(computed_tensor,
                                                       **kwargs))
          output_masks = generic_utils.to_list(
              layer.compute_mask(computed_tensor, computed_mask))
          computed_tensors = [computed_tensor]
          computed_masks = [computed_mask]
        else:
          computed_tensors = [x[0] for x in computed_data]
          computed_masks = [x[1] for x in computed_data]
          if has_arg(layer.call, 'mask'):
            if 'mask' not in kwargs:
              kwargs['mask'] = computed_masks
          output_tensors = generic_utils.to_list(layer(computed_tensors,
                                                       **kwargs))
          output_masks = generic_utils.to_list(
              layer.compute_mask(computed_tensors, computed_masks))
        # Update tensor_map.
        for x, y, mask in zip(reference_output_tensors, output_tensors,
                              output_masks):
          tensor_map[x] = (y, mask)

  # Check that we did compute the model outputs,
  # then instantiate a new model from inputs and outputs.
  output_tensors = []
  for x in model.outputs:
    assert x in tensor_map, 'Could not compute output ' + str(x)
    tensor, _ = tensor_map[x]
    output_tensors.append(tensor)
  return Model(input_tensors, output_tensors, name=model.name)


def _clone_sequential_model(model, input_tensors=None):
  """Clone a `Sequential` model instance.

  Model cloning is similar to calling a model on new inputs,
  except that it creates new layers (and thus new weights) instead
  of sharing the weights of the existing layers.

  Arguments:
      model: Instance of `Sequential`.
      input_tensors: optional list of input tensors
          to build the model upon. If not provided,
          placeholders will be created.

  Returns:
      An instance of `Sequential` reproducing the behavior
      of the original model, on top of new inputs tensors,
      using newly instantiated weights.

  Raises:
      ValueError: in case of invalid `model` argument value.
  """
  if not isinstance(model, Sequential):
    raise ValueError('Expected `model` argument '
                     'to be a `Sequential` model instance, '
                     'but got:', model)

  def clone(layer):
    return layer.__class__.from_config(layer.get_config())

  layers = [clone(layer) for layer in model.layers]
  if input_tensors is None:
    return Sequential(layers=layers, name=model.name)
  else:
    if len(generic_utils.to_list(input_tensors)) != 1:
      raise ValueError('To clone a `Sequential` model, we expect '
                       ' at most one tensor '
                       'as part of `input_tensors`.')
    x = generic_utils.to_list(input_tensors)[0]
    if K.is_keras_tensor(x):
      origin_layer = x._keras_history[0]
      if isinstance(origin_layer, InputLayer):
        return Sequential(layers=[origin_layer] + layers, name=model.name)
      else:
        raise ValueError('Cannot clone a `Sequential` model on top '
                         'of a tensor that comes from a Keras layer '
                         'other than an `InputLayer`. '
                         'Use the functional API instead.')
    input_tensor = Input(tensor=x, name='input_wrapper_for_' + str(x.name))
    input_layer = input_tensor._keras_history[0]
    return Sequential(layers=[input_layer] + layers, name=model.name)


def clone_model(model, input_tensors=None):
  """Clone any `Model` instance.

  Model cloning is similar to calling a model on new inputs,
  except that it creates new layers (and thus new weights) instead
  of sharing the weights of the existing layers.

  Arguments:
      model: Instance of `Model`
          (could be a functional model or a Sequential model).
      input_tensors: optional list of input tensors
          to build the model upon. If not provided,
          placeholders will be created.

  Returns:
      An instance of `Model` reproducing the behavior
      of the original model, on top of new inputs tensors,
      using newly instantiated weights.

  Raises:
      ValueError: in case of invalid `model` argument value.
  """
  if isinstance(model, Sequential):
    return _clone_sequential_model(model, input_tensors=input_tensors)
  else:
    return _clone_functional_model(model, input_tensors=input_tensors)

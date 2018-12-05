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
"""Home of the `Sequential` model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import layers as layer_module
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.engine.network import Network
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.models.Sequential', 'keras.Sequential')
class Sequential(Model):
  """Linear stack of layers.

  Arguments:
      layers: list of layers to add to the model.

  Example:

  ```python
  # Optionally, the first layer can receive an `input_shape` argument:
  model = Sequential()
  model.add(Dense(32, input_shape=(500,)))
  # Afterwards, we do automatic shape inference:
  model.add(Dense(32))

  # This is identical to the following:
  model = Sequential()
  model.add(Dense(32, input_dim=500))

  # And to the following:
  model = Sequential()
  model.add(Dense(32, batch_input_shape=(None, 500)))

  # Note that you can also omit the `input_shape` argument:
  # In that case the model gets built the first time you call `fit` (or other
  # training and evaluation methods).
  model = Sequential()
  model.add(Dense(32))
  model.add(Dense(32))
  model.compile(optimizer=optimizer, loss=loss)
  # This builds the model for the first time:
  model.fit(x, y, batch_size=32, epochs=10)

  # Note that when using this delayed-build pattern (no input shape specified),
  # the model doesn't have any weights until the first call
  # to a training/evaluation method (since it isn't yet built):
  model = Sequential()
  model.add(Dense(32))
  model.add(Dense(32))
  model.weights  # returns []

  # Whereas if you specify the input shape, the model gets built continuously
  # as you are adding layers:
  model = Sequential()
  model.add(Dense(32, input_shape=(500,)))
  model.add(Dense(32))
  model.weights  # returns list of length 4

  When using the delayed-build pattern (no input shape specified), you can
  choose to manually build your model by calling `build(batch_input_shape)`:
  model = Sequential()
  model.add(Dense(32))
  model.add(Dense(32))
  model.build((None, 500))
  model.weights  # returns list of length 4
  ```
  """

  @checkpointable.no_automatic_dependency_tracking
  def __init__(self, layers=None, name=None):
    super(Sequential, self).__init__(name=name)
    self.supports_masking = True
    self._build_input_shape = None
    self._compute_output_and_mask_jointly = True

    # Add to the model any layers passed to the constructor.
    if layers:
      for layer in layers:
        self.add(layer)

  @property
  def layers(self):
    # Historically, `sequential.layers` only returns layers that were added
    # via `add`, and omits the auto-generated `InputLayer` that comes at the
    # bottom of the stack.
    # `CheckpointableBase` manages the `_layers` attributes and does filtering
    # over it.
    layers = super(Sequential, self).layers
    if layers and isinstance(layers[0], InputLayer):
      return layers[1:]
    return layers[:]

  @property
  def _static_graph_friendly(self):
    return all(layer._static_graph_friendly for layer in self.layers)

  @checkpointable.no_automatic_dependency_tracking
  def add(self, layer):
    """Adds a layer instance on top of the layer stack.

    Arguments:
        layer: layer instance.

    Raises:
        TypeError: If `layer` is not a layer instance.
        ValueError: In case the `layer` argument does not
            know its input shape.
        ValueError: In case the `layer` argument has
            multiple output tensors, or is already connected
            somewhere else (forbidden in `Sequential` models).
    """
    if not isinstance(layer, base_layer.Layer):
      raise TypeError('The added layer must be '
                      'an instance of class Layer. '
                      'Found: ' + str(layer))
    self.built = False
    set_inputs = False
    if not self._layers:
      if isinstance(layer, InputLayer):
        # Corner case where the user passes an InputLayer layer via `add`.
        assert len(layer._inbound_nodes[-1].output_tensors) == 1
        set_inputs = True
      else:
        batch_shape, dtype = training_utils.get_input_shape_and_dtype(layer)
        if batch_shape:
          # Instantiate an input layer.
          x = Input(
              batch_shape=batch_shape,
              dtype=dtype,
              name=layer.name + '_input')
          # This will build the current layer
          # and create the node connecting the current layer
          # to the input layer we just created.
          layer(x)
          set_inputs = True

      if set_inputs:
        # If an input layer (placeholder) is available.
        if len(layer._inbound_nodes[-1].output_tensors) != 1:
          raise ValueError('All layers in a Sequential model '
                           'should have a single output tensor. '
                           'For multi-output layers, '
                           'use the functional API.')
        self.outputs = [layer._inbound_nodes[-1].output_tensors[0]]
        self.inputs = layer_utils.get_source_inputs(self.outputs[0])

    elif self.outputs:
      # If the model is being built continuously on top of an input layer:
      # refresh its output.
      output_tensor = layer(self.outputs[0])
      if isinstance(output_tensor, list):
        raise TypeError('All layers in a Sequential model '
                        'should have a single output tensor. '
                        'For multi-output layers, '
                        'use the functional API.')
      self.outputs = [output_tensor]
    if set_inputs or self._is_graph_network:
      self._init_graph_network(self.inputs, self.outputs, name=self.name)
      self.built = True
    else:
      self._layers.append(layer)
    if self._layers:
      self._track_layers(self._layers)

  @checkpointable.no_automatic_dependency_tracking
  def pop(self):
    """Removes the last layer in the model.

    Raises:
        TypeError: if there are no layers in the model.
    """
    if not self.layers:
      raise TypeError('There are no layers in the model.')

    self._layers.pop()
    if not self.layers:
      self.outputs = None
      self.inputs = None
      self.built = False
    elif self._is_graph_network:
      self.layers[-1]._outbound_nodes = []
      self.outputs = [self.layers[-1].output]
      self._init_graph_network(self.inputs, self.outputs, name=self.name)
      self.built = True

  @base_layer.default
  def build(self, input_shape=None):
    if self._is_graph_network:
      self._init_graph_network(self.inputs, self.outputs, name=self.name)
    else:
      if input_shape is None:
        raise ValueError('You must provide an `input_shape` argument.')
      input_shape = tuple(input_shape)
      self._build_input_shape = input_shape
      super(Sequential, self).build(input_shape)
    self.built = True

  def call(self, inputs, training=None, mask=None):
    if self._is_graph_network:
      return super(Sequential, self).call(inputs, training=training, mask=mask)

    outputs, _ = self._call_and_compute_mask(
        inputs, training=training, mask=mask)
    return outputs

  def _call_and_compute_mask(self, inputs, training=None, mask=None):
    if not self.built and self._is_graph_network:
      self._init_graph_network(self.inputs, self.outputs, name=self.name)

    x = inputs
    for layer in self.layers:
      kwargs = {}
      if 'mask' in tf_inspect.getfullargspec(layer.call).args:
        kwargs['mask'] = mask
      if 'training' in tf_inspect.getfullargspec(layer.call).args:
        kwargs['training'] = training

      if isinstance(layer, Network) and layer._compute_output_and_mask_jointly:
        x, mask = layer._call_and_compute_mask(x, **kwargs)
      else:
        if not layer.built:
          # Build layer if applicable.
          with ops.name_scope(layer._name_scope()):
            layer._maybe_build(x)
          layer.built = True
        x = layer.call(x, **kwargs)
        if layer.supports_masking:
          mask = layer.compute_mask(x, mask)
        else:
          mask = None
      if not context.executing_eagerly():
        x._keras_mask = mask
    return x, mask

  def compute_output_shape(self, input_shape):
    shape = input_shape
    for layer in self.layers:
      shape = layer.compute_output_shape(shape)
    return shape

  def compute_mask(self, inputs, mask):
    _, mask = self._call_and_compute_mask(inputs, mask=mask)
    return mask

  def predict_proba(self, x, batch_size=32, verbose=0):
    """Generates class probability predictions for the input samples.

    The input samples are processed batch by batch.

    Arguments:
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.

    Returns:
        A Numpy array of probability predictions.
    """
    preds = self.predict(x, batch_size, verbose)
    if preds.min() < 0. or preds.max() > 1.:
      logging.warning('Network returning invalid probability values. '
                      'The last layer might not normalize predictions '
                      'into probabilities '
                      '(like softmax or sigmoid would).')
    return preds

  def predict_classes(self, x, batch_size=32, verbose=0):
    """Generate class predictions for the input samples.

    The input samples are processed batch by batch.

    Arguments:
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.

    Returns:
        A numpy array of class predictions.
    """
    proba = self.predict(x, batch_size=batch_size, verbose=verbose)
    if proba.shape[-1] > 1:
      return proba.argmax(axis=-1)
    else:
      return (proba > 0.5).astype('int32')

  def save(self, filepath, overwrite=True, include_optimizer=True):
    from tensorflow.python.keras.models import save_model  # pylint: disable=g-import-not-at-top
    save_model(self, filepath, overwrite, include_optimizer)

  def get_config(self):
    layer_configs = []
    for layer in self.layers:
      layer_configs.append({
          'class_name': layer.__class__.__name__,
          'config': layer.get_config()
      })
    config = {
        'name': self.name,
        'layers': copy.deepcopy(layer_configs)
    }
    if self._build_input_shape:
      config['build_input_shape'] = self._build_input_shape
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'name' in config:
      name = config['name']
      build_input_shape = config.get('build_input_shape')
      layer_configs = config['layers']
    else:
      name = None
      build_input_shape = None
      layer_configs = config
    model = cls(name=name)
    for layer_config in layer_configs:
      layer = layer_module.deserialize(layer_config,
                                       custom_objects=custom_objects)
      model.add(layer)
    if not model.inputs and build_input_shape:
      model.build(build_input_shape)
    if not model._is_graph_network:
      # Still needs to be built when passed input data.
      model.built = False
    return model

  @property
  def input_spec(self):
    if self.layers and hasattr(self.layers[0], 'input_spec'):
      return self.layers[0].input_spec
    return None

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

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import layers as layer_module
from tensorflow.python.keras._impl.keras.engine import base_layer
from tensorflow.python.keras._impl.keras.engine import network
from tensorflow.python.keras._impl.keras.engine.input_layer import Input
from tensorflow.python.keras._impl.keras.engine.input_layer import InputLayer
from tensorflow.python.keras._impl.keras.engine.training import Model
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpointable
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

  def __init__(self, layers=None, name=None):
    super(Sequential, self).__init__(name=name)

    # Add to the model any layers passed to the constructor.
    if layers:
      for layer in layers:
        self.add(layer)

  @property
  def layers(self):
    # Historically, `sequential.layers` only returns layers that were added
    # via `add`, and omits the auto-generated `InputLayer` that comes at the
    # bottom of the stack.
    if self._layers and isinstance(self._layers[0], InputLayer):
      return self._layers[1:]
    return self._layers

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
    if not isinstance(layer, (base_layer.Layer, base_layer.TFBaseLayer)):
      raise TypeError('The added layer must be '
                      'an instance of class Layer. '
                      'Found: ' + str(layer))
    self.built = False
    if not self._layers:
      set_inputs = False
      # First layer in model: check that it is an input layer.
      if not isinstance(layer, InputLayer):
        # Create an input tensor and call `layer` on the input tensor.
        # First, we need to infer the expected input shape and dtype.
        first_layer = layer
        if isinstance(layer, (Model, Sequential)):
          # We were passed a model as first layer.
          # This requires a specific way to figure out the
          # input shape and dtype.
          if not layer.layers:
            raise ValueError('Cannot add an empty model '
                             'to a `Sequential` model.')
          # In case of nested models: recover the first layer
          # of the deepest model to infer input shape and dtype.
          first_layer = layer.layers[0]
          while isinstance(first_layer, (Model, Sequential)):
            first_layer = first_layer.layers[0]
          batch_shape = first_layer._batch_input_shape
          dtype = first_layer.dtype

        if hasattr(first_layer, '_batch_input_shape'):
          batch_shape = first_layer._batch_input_shape
          dtype = first_layer.dtype
          # Instantiate the input layer.
          x = Input(
              batch_shape=batch_shape,
              dtype=dtype,
              name=layer.name + '_input')
          # This will build the current layer
          # and create the node connecting the current layer
          # to the input layer we just created.
          layer(x)
          set_inputs = True
        else:
          # The layer doesn't know about its expected shape. We will have to
          # build the model lazily on `fit`/etc.
          batch_shape = None
      else:
        # Corner case where the user passes an InputLayer layer via `add`.
        assert len(layer._inbound_nodes[-1].output_tensors) == 1
        set_inputs = True

      if set_inputs:
        if len(layer._inbound_nodes[-1].output_tensors) != 1:
          raise ValueError('All layers in a Sequential model '
                           'should have a single output tensor. '
                           'For multi-output layers, '
                           'use the functional API.')

        self.outputs = [layer._inbound_nodes[-1].output_tensors[0]]
        self.inputs = network.get_source_inputs(self.outputs[0])
    elif self.outputs:
      output_tensor = layer(self.outputs[0])
      if isinstance(output_tensor, list):
        raise TypeError('All layers in a Sequential model '
                        'should have a single output tensor. '
                        'For multi-output layers, '
                        'use the functional API.')
      self.outputs = [output_tensor]
    if self.inputs:
      self.build()
    else:
      self._layers.append(layer)
    # In implementing Checkpointable, Sequential does not track its Layers
    # normally, since they may be added and removed (in pop()). Instead, it
    # names everything on demand (gathering dependencies in
    # _checkpoint_dependencies, and looking them up in
    # _lookup_dependency). _handle_deferred_dependencies just checks whether an
    # existing checkpoint load targets this Layer, it does not create a
    # dependency on the Layer.
    self._handle_deferred_dependencies(
        name='layer-%d' % (len(self._layers) - 1), checkpointable=layer)

  @property
  def _checkpoint_dependencies(self):
    """For implementing Checkpointable. Layers which should be saved."""
    return super(Sequential, self)._checkpoint_dependencies + [
        checkpointable.CheckpointableReference(
            name='layer-%d' % layer_index, ref=layer)
        for layer_index, layer in enumerate(self._layers)]

  def _lookup_dependency(self, name):
    """For implementing Checkpointable. Looks up a Layer."""
    super_lookup = super(Sequential, self)._lookup_dependency(name=name)
    if super_lookup is not None:
      return super_lookup
    if name.startswith('layer-'):
      try:
        return self._layers[int(name[6:])]
      except IndexError:
        return None
    else:
      return None

  def pop(self):
    """Removes the last layer in the model.

    Raises:
        TypeError: if there are no layers in the model.
    """
    if not self.layers:
      raise TypeError('There are no layers in the model.')

    self._layers.pop()
    self.built = False
    if not self.layers:
      self.outputs = None
      self.inputs = None
    elif self.outputs:
      self.layers[-1]._outbound_nodes = []
      self.outputs = [self.layers[-1].output]
      self.build()

  def build(self, input_shape=None):
    if input_shape and not self.inputs:
      batch_shape = tuple(input_shape)
      dtype = K.floatx()
      x = Input(
          batch_shape=batch_shape, dtype=dtype, name=self.name + '_input')
      self.inputs = [x]
      for layer in self._layers:
        x = layer(x)
      self.outputs = [x]

    if self.inputs:
      self._init_graph_network(self.inputs, self.outputs, name=self.name)
      self.built = True

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

  def get_config(self):
    config = []
    for layer in self.layers:
      config.append({
          'class_name': layer.__class__.__name__,
          'config': layer.get_config()
      })
    return copy.deepcopy(config)

  @classmethod
  def from_config(cls, config, custom_objects=None):
    model = cls()
    for conf in config:
      layer = layer_module.deserialize(conf, custom_objects=custom_objects)
      model.add(layer)
    return model

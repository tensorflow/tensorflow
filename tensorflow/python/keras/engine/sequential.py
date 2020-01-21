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

from tensorflow.python.keras import layers as layer_module
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import layer_utils as trackable_layer_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.models.Sequential', 'keras.Sequential')
class Sequential(training.Model):
  """`Sequential` groups a linear stack of layers into a `tf.keras.Model`.

  `Sequential` provides training and inference features on this model.

  Examples:

  >>> # Optionally, the first layer can receive an `input_shape` argument:
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Dense(8, input_shape=(16,)))
  >>> # Afterwards, we do automatic shape inference:
  >>> model.add(tf.keras.layers.Dense(4))

  >>> # This is identical to the following:
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Dense(8, input_dim=16))

  >>> # And to the following:
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Dense(8, batch_input_shape=(None, 16)))

  >>> # Note that you can also omit the `input_shape` argument.
  >>> # In that case the model doesn't have any weights until the first call
  >>> # to a training/evaluation method (since it isn't yet built):
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Dense(8))
  >>> model.add(tf.keras.layers.Dense(4))
  >>> # model.weights not created yet

  >>> # Whereas if you specify the input shape, the model gets built
  >>> # continuously as you are adding layers:
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Dense(8, input_shape=(16,)))
  >>> model.add(tf.keras.layers.Dense(4))
  >>> len(model.weights)
  4

  >>> # When using the delayed-build pattern (no input shape specified), you can
  >>> # choose to manually build your model by calling
  >>> # `build(batch_input_shape)`:
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Dense(8))
  >>> model.add(tf.keras.layers.Dense(4))
  >>> model.build((None, 16))
  >>> len(model.weights)
  4

  ```python
  # Note that when using the delayed-build pattern (no input shape specified),
  # the model gets built the first time you call `fit` (or other training and
  # evaluation methods).
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(8))
  model.add(tf.keras.layers.Dense(1))
  model.compile(optimizer='sgd', loss='mse')
  # This builds the model for the first time:
  model.fit(x, y, batch_size=32, epochs=10)
  ```
  """

  @trackable.no_automatic_dependency_tracking
  def __init__(self, layers=None, name=None):
    """Creates a `Sequential` model instance.

    Args:
      layers: Optional list of layers to add to the model.
      name: Optional name for the model.
    """
    super(Sequential, self).__init__(name=name, autocast=False)
    self.supports_masking = True
    self._build_input_shape = None
    self._compute_output_and_mask_jointly = True

    self._layer_call_argspecs = {}

    # Add to the model any layers passed to the constructor.
    if layers:
      if not isinstance(layers, (list, tuple)):
        layers = [layers]
      tf_utils.assert_no_legacy_layers(layers)
      for layer in layers:
        self.add(layer)

  @property
  def layers(self):
    # Historically, `sequential.layers` only returns layers that were added
    # via `add`, and omits the auto-generated `InputLayer` that comes at the
    # bottom of the stack.
    # `Trackable` manages the `_layers` attributes and does filtering
    # over it.
    layers = super(Sequential, self).layers
    if layers and isinstance(layers[0], input_layer.InputLayer):
      return layers[1:]
    return layers[:]

  @property
  @trackable_layer_utils.cache_recursive_attribute('dynamic')
  def dynamic(self):
    return any(layer.dynamic for layer in self.layers)

  @trackable.no_automatic_dependency_tracking
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
    # If we are passed a Keras tensor created by keras.Input(), we can extract
    # the input layer from its keras history and use that without any loss of
    # generality.
    if hasattr(layer, '_keras_history'):
      origin_layer = layer._keras_history[0]
      if isinstance(origin_layer, input_layer.InputLayer):
        layer = origin_layer

    if not isinstance(layer, base_layer.Layer):
      raise TypeError('The added layer must be '
                      'an instance of class Layer. '
                      'Found: ' + str(layer))

    tf_utils.assert_no_legacy_layers([layer])

    # This allows the added layer to broadcast mutations to the current
    # layer, which is necessary to ensure cache correctness.
    layer._attribute_sentinel.add_parent(self._attribute_sentinel)

    self.built = False
    set_inputs = False
    if not self._layers:
      if isinstance(layer, input_layer.InputLayer):
        # Corner case where the user passes an InputLayer layer via `add`.
        assert len(nest.flatten(layer._inbound_nodes[-1].output_tensors)) == 1
        set_inputs = True
      else:
        batch_shape, dtype = training_utils.get_input_shape_and_dtype(layer)
        if batch_shape:
          # Instantiate an input layer.
          x = input_layer.Input(
              batch_shape=batch_shape, dtype=dtype, name=layer.name + '_input')
          # This will build the current layer
          # and create the node connecting the current layer
          # to the input layer we just created.
          layer(x)
          set_inputs = True

      if set_inputs:
        # If an input layer (placeholder) is available.
        if len(nest.flatten(layer._inbound_nodes[-1].output_tensors)) != 1:
          raise ValueError('All layers in a Sequential model '
                           'should have a single output tensor. '
                           'For multi-output layers, '
                           'use the functional API.')
        self.outputs = [
            nest.flatten(layer._inbound_nodes[-1].output_tensors)[0]
        ]
        self.inputs = layer_utils.get_source_inputs(self.outputs[0])

    elif self.outputs:
      # If the model is being built continuously on top of an input layer:
      # refresh its output.
      output_tensor = layer(self.outputs[0])
      if len(nest.flatten(output_tensor)) != 1:
        raise TypeError('All layers in a Sequential model '
                        'should have a single output tensor. '
                        'For multi-output layers, '
                        'use the functional API.')
      self.outputs = [output_tensor]

    if self.outputs:
      # True if set_inputs or self._is_graph_network or if adding a layer
      # to an already built deferred seq model.
      self.built = True

    if set_inputs or self._is_graph_network:
      self._init_graph_network(self.inputs, self.outputs, name=self.name)
    else:
      self._layers.append(layer)
      self._handle_deferred_layer_dependencies([layer])

    self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
    # Different Model types add to `._layers` in different ways, so for safety
    # we do a cache invalidation to make sure the changes are reflected.
    self._attribute_sentinel.invalidate_all()

  @trackable.no_automatic_dependency_tracking
  def pop(self):
    """Removes the last layer in the model.

    Raises:
        TypeError: if there are no layers in the model.
    """
    if not self.layers:
      raise TypeError('There are no layers in the model.')

    layer = self._layers.pop()
    self._layer_call_argspecs.pop(layer)
    self._attribute_sentinel.invalidate_all()
    if not self.layers:
      self.outputs = None
      self.inputs = None
      self.built = False
    elif self._is_graph_network:
      self.layers[-1]._outbound_nodes = []
      self.outputs = [self.layers[-1].output]
      self._init_graph_network(self.inputs, self.outputs, name=self.name)
      self.built = True

  @base_layer_utils.default
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

  def call(self, inputs, training=None, mask=None):  # pylint: disable=redefined-outer-name
    if self._is_graph_network:
      if not self.built:
        self._init_graph_network(self.inputs, self.outputs, name=self.name)
      return super(Sequential, self).call(inputs, training=training, mask=mask)

    outputs = inputs  # handle the corner case where self.layers is empty
    for layer in self.layers:
      # During each iteration, `inputs` are the inputs to `layer`, and `outputs`
      # are the outputs of `layer` applied to `inputs`. At the end of each
      # iteration `inputs` is set to `outputs` to prepare for the next layer.
      kwargs = {}
      argspec = self._layer_call_argspecs[layer].args
      if 'mask' in argspec:
        kwargs['mask'] = mask
      if 'training' in argspec:
        kwargs['training'] = training

      outputs = layer(inputs, **kwargs)

      # `outputs` will be the inputs to the next layer.
      inputs = outputs
      mask = outputs._keras_mask

    return outputs

  def compute_output_shape(self, input_shape):
    shape = input_shape
    for layer in self.layers:
      shape = layer.compute_output_shape(shape)
    return shape

  def compute_mask(self, inputs, mask):
    # TODO(omalleyt): b/123540974 This function is not really safe to call
    # by itself because it will duplicate any updates and losses in graph
    # mode by `call`ing the Layers again.
    outputs = self.call(inputs, mask=mask)
    return outputs._keras_mask

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
    layer_configs = []
    for layer in self.layers:
      layer_configs.append(generic_utils.serialize_keras_object(layer))
    # When constructed using an `InputLayer` the first non-input layer may not
    # have the shape information to reconstruct `Sequential` as a graph network.
    if (self._is_graph_network and layer_configs and
        'batch_input_shape' not in layer_configs[0]['config'] and
        isinstance(self._layers[0], input_layer.InputLayer)):
      batch_input_shape = self._layers[0]._batch_input_shape
      layer_configs[0]['config']['batch_input_shape'] = batch_input_shape

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
    return model

  @property
  def input_spec(self):
    if self.layers and hasattr(self.layers[0], 'input_spec'):
      return self.layers[0].input_spec
    return None

  @property
  def _trackable_saved_model_saver(self):
    return model_serialization.SequentialSavedModelSaver(self)

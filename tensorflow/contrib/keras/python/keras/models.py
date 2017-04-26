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
"""Home of the Sequential model, and the `save_model`/`load_model` functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import warnings

import numpy as np

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import layers as layer_module
from tensorflow.contrib.keras.python.keras import optimizers
from tensorflow.contrib.keras.python.keras.engine import topology
from tensorflow.contrib.keras.python.keras.engine.topology import Input
from tensorflow.contrib.keras.python.keras.engine.topology import Layer
from tensorflow.contrib.keras.python.keras.engine.training import Model
from tensorflow.contrib.keras.python.keras.utils.io_utils import ask_to_proceed_with_overwrite


# pylint: disable=g-import-not-at-top
try:
  import h5py
except ImportError:
  h5py = None

try:
  import yaml
except ImportError:
  yaml = None
# pylint: enable=g-import-not-at-top


def save_model(model, filepath, overwrite=True, include_optimizer=True):
  """Save a model to a HDF5 file.

  The saved model contains:
      - the model's configuration (topology)
      - the model's weights
      - the model's optimizer's state (if any)

  Thus the saved model can be reinstantiated in
  the exact same state, without any of the code
  used for model definition or training.

  Arguments:
      model: Keras model instance to be saved.
      filepath: String, path where to save the model.
      overwrite: Whether we should overwrite any existing
          model at the target location, or instead
          ask the user with a manual prompt.
      include_optimizer: If True, save optimizer's state together.

  Raises:
      ImportError: if h5py is not available.
  """

  if h5py is None:
    raise ImportError('`save_model` requires h5py.')

  def get_json_type(obj):
    """Serialize any object to a JSON-serializable structure.

    Arguments:
        obj: the object to serialize

    Returns:
        JSON-serializable structure representing `obj`.

    Raises:
        TypeError: if `obj` cannot be serialized.
    """
    # if obj is a serializable Keras class instance
    # e.g. optimizer, layer
    if hasattr(obj, 'get_config'):
      return {'class_name': obj.__class__.__name__, 'config': obj.get_config()}

    # if obj is any numpy type
    if type(obj).__module__ == np.__name__:
      return obj.item()

    # misc functions (e.g. loss function)
    if callable(obj):
      return obj.__name__

    # if obj is a python 'type'
    if type(obj).__name__ == type.__name__:
      return obj.__name__

    raise TypeError('Not JSON Serializable:', obj)

  from tensorflow.contrib.keras.python.keras import __version__ as keras_version  # pylint: disable=g-import-not-at-top

  # If file exists and should not be overwritten.
  if not overwrite and os.path.isfile(filepath):
    proceed = ask_to_proceed_with_overwrite(filepath)
    if not proceed:
      return

  f = h5py.File(filepath, 'w')
  f.attrs['keras_version'] = str(keras_version).encode('utf8')
  f.attrs['backend'] = K.backend().encode('utf8')
  f.attrs['model_config'] = json.dumps(
      {
          'class_name': model.__class__.__name__,
          'config': model.get_config()
      },
      default=get_json_type).encode('utf8')

  model_weights_group = f.create_group('model_weights')
  model_layers = model.layers
  topology.save_weights_to_hdf5_group(model_weights_group, model_layers)

  if include_optimizer and hasattr(model, 'optimizer'):
    if isinstance(model.optimizer, optimizers.TFOptimizer):
      warnings.warn(
          'TensorFlow optimizers do not '
          'make it possible to access '
          'optimizer attributes or optimizer state '
          'after instantiation. '
          'As a result, we cannot save the optimizer '
          'as part of the model save file.'
          'You will have to compile your model again after loading it. '
          'Prefer using a Keras optimizer instead '
          '(see keras.io/optimizers).')
    else:
      f.attrs['training_config'] = json.dumps(
          {
              'optimizer_config': {
                  'class_name': model.optimizer.__class__.__name__,
                  'config': model.optimizer.get_config()
              },
              'loss': model.loss,
              'metrics': model.metrics,
              'sample_weight_mode': model.sample_weight_mode,
              'loss_weights': model.loss_weights,
          },
          default=get_json_type).encode('utf8')

      # Save optimizer weights.
      symbolic_weights = getattr(model.optimizer, 'weights')
      if symbolic_weights:
        optimizer_weights_group = f.create_group('optimizer_weights')
        weight_values = K.batch_get_value(symbolic_weights)
        weight_names = []
        for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
          # Default values of symbolic_weights is /variable for theano
          if K.backend() == 'theano':
            if hasattr(w, 'name') and w.name != '/variable':
              name = str(w.name)
            else:
              name = 'param_' + str(i)
          else:
            if hasattr(w, 'name') and w.name:
              name = str(w.name)
            else:
              name = 'param_' + str(i)
          weight_names.append(name.encode('utf8'))
        optimizer_weights_group.attrs['weight_names'] = weight_names
        for name, val in zip(weight_names, weight_values):
          param_dset = optimizer_weights_group.create_dataset(
              name, val.shape, dtype=val.dtype)
          if not val.shape:
            # scalar
            param_dset[()] = val
          else:
            param_dset[:] = val
  f.flush()
  f.close()


def load_model(filepath, custom_objects=None):
  """Loads a model saved via `save_model`.

  Arguments:
      filepath: String, path to the saved model.
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.

  Returns:
      A Keras model instance. If an optimizer was found
      as part of the saved model, the model is already
      compiled. Otherwise, the model is uncompiled and
      a warning will be displayed.

  Raises:
      ImportError: if h5py is not available.
      ValueError: In case of an invalid savefile.
  """
  if h5py is None:
    raise ImportError('`load_model` requires h5py.')

  if not custom_objects:
    custom_objects = {}

  def convert_custom_objects(obj):
    """Handles custom object lookup.

    Arguments:
        obj: object, dict, or list.

    Returns:
        The same structure, where occurences
            of a custom object name have been replaced
            with the custom object.
    """
    if isinstance(obj, list):
      deserialized = []
      for value in obj:
        if value in custom_objects:
          deserialized.append(custom_objects[value])
        else:
          deserialized.append(value)
      return deserialized
    if isinstance(obj, dict):
      deserialized = {}
      for key, value in obj.items():
        deserialized[key] = []
        if isinstance(value, list):
          for element in value:
            if element in custom_objects:
              deserialized[key].append(custom_objects[element])
            else:
              deserialized[key].append(element)
        elif value in custom_objects:
          deserialized[key] = custom_objects[value]
        else:
          deserialized[key] = value
      return deserialized
    if obj in custom_objects:
      return custom_objects[obj]
    return obj

  f = h5py.File(filepath, mode='r')

  # instantiate model
  model_config = f.attrs.get('model_config')
  if model_config is None:
    raise ValueError('No model found in config file.')
  model_config = json.loads(model_config.decode('utf-8'))
  model = model_from_config(model_config, custom_objects=custom_objects)

  # set weights
  topology.load_weights_from_hdf5_group(f['model_weights'], model.layers)

  # instantiate optimizer
  training_config = f.attrs.get('training_config')
  if training_config is None:
    warnings.warn('No training configuration found in save file: '
                  'the model was *not* compiled. Compile it manually.')
    f.close()
    return model
  training_config = json.loads(training_config.decode('utf-8'))
  optimizer_config = training_config['optimizer_config']
  optimizer = optimizers.deserialize(
      optimizer_config, custom_objects=custom_objects)

  # Recover loss functions and metrics.
  loss = convert_custom_objects(training_config['loss'])
  metrics = convert_custom_objects(training_config['metrics'])
  sample_weight_mode = training_config['sample_weight_mode']
  loss_weights = training_config['loss_weights']

  # Compile model.
  model.compile(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics,
      loss_weights=loss_weights,
      sample_weight_mode=sample_weight_mode)

  # Set optimizer weights.
  if 'optimizer_weights' in f:
    # Build train function (to get weight updates).
    if isinstance(model, Sequential):
      model.model._make_train_function()
    else:
      model._make_train_function()
    optimizer_weights_group = f['optimizer_weights']
    optimizer_weight_names = [
        n.decode('utf8') for n in optimizer_weights_group.attrs['weight_names']
    ]
    optimizer_weight_values = [
        optimizer_weights_group[n] for n in optimizer_weight_names
    ]
    model.optimizer.set_weights(optimizer_weight_values)
  f.close()
  return model


def model_from_config(config, custom_objects=None):
  """Instantiates a Keras model from its config.

  Arguments:
      config: Configuration dictionary.
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.

  Returns:
      A Keras model instance (uncompiled).
  """
  if isinstance(config, list):
    raise TypeError('`model_fom_config` expects a dictionary, not a list. '
                    'Maybe you meant to use '
                    '`Sequential.from_config(config)`?')
  return layer_module.deserialize(config, custom_objects=custom_objects)


def model_from_yaml(yaml_string, custom_objects=None):
  """Parses a yaml model configuration file and returns a model instance.

  Arguments:
      yaml_string: YAML string encoding a model configuration.
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.

  Returns:
      A Keras model instance (uncompiled).

  Raises:
      ImportError: if yaml module is not found.
  """
  if yaml is None:
    raise ImportError('Requires yaml module installed.')
  config = yaml.load(yaml_string)
  return layer_module.deserialize(config, custom_objects=custom_objects)


def model_from_json(json_string, custom_objects=None):
  """Parses a JSON model configuration file and returns a model instance.

  Arguments:
      json_string: JSON string encoding a model configuration.
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.

  Returns:
      A Keras model instance (uncompiled).
  """
  config = json.loads(json_string)
  return layer_module.deserialize(config, custom_objects=custom_objects)


class Sequential(Model):
  """Linear stack of layers.

  Arguments:
      layers: list of layers to add to the model.

  # Note
      The first layer passed to a Sequential model
      should have a defined input shape. What that
      means is that it should have received an `input_shape`
      or `batch_input_shape` argument,
      or for some type of layers (recurrent, Dense...)
      an `input_dim` argument.

  Example:

      ```python
          model = Sequential()
          # first layer must have a defined input shape
          model.add(Dense(32, input_dim=500))
          # afterwards, Keras does automatic shape inference
          model.add(Dense(32))

          # also possible (equivalent to the above):
          model = Sequential()
          model.add(Dense(32, input_shape=(500,)))
          model.add(Dense(32))

          # also possible (equivalent to the above):
          model = Sequential()
          # here the batch dimension is None,
          # which means any batch size will be accepted by the model.
          model.add(Dense(32, batch_input_shape=(None, 500)))
          model.add(Dense(32))
      ```
  """

  def __init__(self, layers=None, name=None):
    self.layers = []  # Stack of layers.
    self.model = None  # Internal Model instance.
    self.inputs = []  # List of input tensors
    self.outputs = []  # List of length 1: the output tensor (unique).
    self._trainable = True
    self._initial_weights = None

    # Model attributes.
    self.inbound_nodes = []
    self.outbound_nodes = []
    self.built = False

    # Set model name.
    if not name:
      prefix = 'sequential_'
      name = prefix + str(K.get_uid(prefix))
    self.name = name

    # Add to the model any layers passed to the constructor.
    if layers:
      for layer in layers:
        self.add(layer)

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
    if not isinstance(layer, Layer):
      raise TypeError('The added layer must be '
                      'an instance of class Layer. '
                      'Found: ' + str(layer))
    if not self.outputs:
      # first layer in model: check that it is an input layer
      if not layer.inbound_nodes:
        # create an input layer
        if not hasattr(layer, 'batch_input_shape'):
          raise ValueError('The first layer in a '
                           'Sequential model must '
                           'get an `input_shape` or '
                           '`batch_input_shape` argument.')
        # Instantiate the input layer.
        x = Input(
            batch_shape=layer.batch_input_shape,
            dtype=layer.dtype,
            name=layer.name + '_input')
        # This will build the current layer
        # and create the node connecting the current layer
        # to the input layer we just created.
        layer(x)

      if len(layer.inbound_nodes) != 1:
        raise ValueError('A layer added to a Sequential model must '
                         'not already be connected somewhere else. '
                         'Model received layer ' + layer.name + ' which has ' +
                         str(len(layer.inbound_nodes)) +
                         ' pre-existing inbound connections.')

      if len(layer.inbound_nodes[0].output_tensors) != 1:
        raise ValueError('All layers in a Sequential model '
                         'should have a single output tensor. '
                         'For multi-output layers, '
                         'use the functional API.')

      self.outputs = [layer.inbound_nodes[0].output_tensors[0]]
      self.inputs = topology.get_source_inputs(self.outputs[0])

      # We create an input node, which we will keep updated
      # as we add more layers
      topology.Node(
          outbound_layer=self,
          inbound_layers=[],
          node_indices=[],
          tensor_indices=[],
          input_tensors=self.inputs,
          output_tensors=self.outputs,
          # no model-level masking for now
          input_masks=[None for _ in self.inputs],
          output_masks=[None])
    else:
      output_tensor = layer(self.outputs[0])
      if isinstance(output_tensor, list):
        raise TypeError('All layers in a Sequential model '
                        'should have a single output tensor. '
                        'For multi-output layers, '
                        'use the functional API.')
      self.outputs = [output_tensor]
      # update self.inbound_nodes
      self.inbound_nodes[0].output_tensors = self.outputs
      self.inbound_nodes[0].output_shapes = [K.int_shape(self.outputs[0])]

    self.layers.append(layer)
    self.built = False

  def pop(self):
    """Removes the last layer in the model.

    Raises:
        TypeError: if there are no layers in the model.
    """
    if not self.layers:
      raise TypeError('There are no layers in the model.')

    self.layers.pop()
    if not self.layers:
      self.outputs = []
      self.inbound_nodes = []
      self.outbound_nodes = []
    else:
      self.layers[-1].outbound_nodes = []
      self.outputs = [self.layers[-1].output]
      # update self.inbound_nodes
      self.inbound_nodes[0].output_tensors = self.outputs
      self.inbound_nodes[0].output_shapes = [K.int_shape(self.outputs[0])]
    self.built = False

  def get_layer(self, name=None, index=None):
    """Retrieve a layer that is part of the model.

    Returns a layer based on either its name (unique)
    or its index in the graph. Indices are based on
    order of horizontal graph traversal (bottom-up).

    Arguments:
        name: string, name of layer.
        index: integer, index of layer.

    Returns:
        A layer instance.
    """
    if self.model is None:
      self.build()
    return self.model.get_layer(name, index)

  def call(self, inputs, mask=None):
    if self.model is None:
      self.build()
    return self.model.call(inputs, mask)

  def build(self, input_shape=None):
    if not self.inputs or not self.outputs:
      raise TypeError('Sequential model cannot be built: model is empty.'
                      ' Add some layers first.')
    # actually create the model
    self.model = Model(self.inputs, self.outputs[0], name=self.name + '_model')
    self.model.trainable = self.trainable

    # mirror model attributes
    self.supports_masking = self.model.supports_masking
    self._output_mask_cache = self.model._output_mask_cache
    self._output_tensor_cache = self.model._output_tensor_cache
    self._output_shape_cache = self.model._output_shape_cache
    self.input_layers = self.model.input_layers
    self.input_layers_node_indices = self.model.input_layers_node_indices
    self.input_layers_tensor_indices = self.model.input_layers_tensor_indices
    self.output_layers = self.model.output_layers
    self.output_layers_node_indices = self.model.output_layers_node_indices
    self.output_layers_tensor_indices = self.model.output_layers_tensor_indices
    self.nodes_by_depth = self.model.nodes_by_depth
    self.container_nodes = self.model.container_nodes
    self.output_names = self.model.output_names
    self.input_names = self.model.input_names
    self._feed_input_names = self.model._feed_input_names
    self._feed_inputs = self.model._feed_inputs

    # Make sure child model callbacks
    # will call the parent Sequential model.
    self.model.callback_model = self

    self.built = True

  @property
  def uses_learning_phase(self):
    if self.model is None:
      self.build()
    return self.model.uses_learning_phase

  def _gather_list_attr(self, attr):
    all_attrs = []
    for layer in self.layers:
      all_attrs += getattr(layer, attr, [])
    return all_attrs

  @property
  def trainable(self):
    return self._trainable

  @trainable.setter
  def trainable(self, value):
    if self.model:
      self.model.trainable = value
    self._trainable = value

  @property
  def trainable_weights(self):
    if not self.trainable:
      return []
    return self._gather_list_attr('trainable_weights')

  @property
  def non_trainable_weights(self):
    weights = self._gather_list_attr('non_trainable_weights')
    if not self.trainable:
      trainable_weights = self._gather_list_attr('trainable_weights')
      return trainable_weights + weights
    return weights

  @property
  def updates(self):
    if self.model is None:
      self.build()
    return self.model.updates

  @property
  def state_updates(self):
    if self.model is None:
      self.build()
    return self.model.state_updates

  def get_updates_for(self, inputs):
    if self.model is None:
      self.build()
    return self.model.get_updates_for(inputs)

  @property
  def losses(self):
    if self.model is None:
      self.build()
    return self.model.losses

  def get_losses_for(self, inputs):
    if self.model is None:
      self.build()
    return self.model.get_losses_for(inputs)

  @property
  def regularizers(self):
    if self.model is None:
      self.build()
    return self.model.regularizers

  @property
  def constraints(self):
    if self.model is None:
      self.build()
    return self.model.constraints

  def get_weights(self):
    """Retrieves the weights of the model.

    Returns:
        A flat list of Numpy arrays
        (one array per model weight).
    """
    if self.model is None:
      self.build()
    return self.model.get_weights()

  def set_weights(self, weights):
    """Sets the weights of the model.

    Arguments:
        weights: Should be a list
            of Numpy arrays with shapes and types matching
            the output of `model.get_weights()`.
    """
    if self.model is None:
      self.build()
    self.model.set_weights(weights)

  def load_weights(self, filepath, by_name=False):
    if h5py is None:
      raise ImportError('`load_weights` requires h5py.')
    f = h5py.File(filepath, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
      f = f['model_weights']
    layers = self.layers
    if by_name:
      topology.load_weights_from_hdf5_group_by_name(f, layers)
    else:
      topology.load_weights_from_hdf5_group(f, layers)
    if hasattr(f, 'close'):
      f.close()

  def save_weights(self, filepath, overwrite=True):
    if h5py is None:
      raise ImportError('`save_weights` requires h5py.')
    # If file exists and should not be overwritten:
    if not overwrite and os.path.isfile(filepath):
      proceed = ask_to_proceed_with_overwrite(filepath)
      if not proceed:
        return
    layers = self.layers
    f = h5py.File(filepath, 'w')
    topology.save_weights_to_hdf5_group(f, layers)
    f.flush()
    f.close()

  def compile(self,
              optimizer,
              loss,
              metrics=None,
              sample_weight_mode=None,
              **kwargs):
    """Configures the learning process.

    Arguments:
        optimizer: str (name of optimizer) or optimizer object.
            See [optimizers](/optimizers).
        loss: str (name of objective function) or objective function.
            See [objectives](/objectives).
        metrics: list of metrics to be evaluated by the model
            during training and testing.
            Typically you will use `metrics=['accuracy']`.
            See [metrics](/metrics).
        sample_weight_mode: if you need to do timestep-wise
            sample weighting (2D weights), set this to "temporal".
            "None" defaults to sample-wise weights (1D).
        **kwargs: for Theano backend, these are passed into K.function.
            Ignored for Tensorflow backend.

    Example:
        ```python
            model = Sequential()
            model.add(Dense(32, input_shape=(500,)))
            model.add(Dense(10, activation='softmax'))
            model.compile(optimizer='rmsprop',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        ```
    """
    # create the underlying model
    self.build()
    # call compile method of Model class
    self.model.compile(
        optimizer,
        loss,
        metrics=metrics,
        sample_weight_mode=sample_weight_mode,
        **kwargs)
    self.optimizer = self.model.optimizer
    self.loss = self.model.loss
    self.loss_weights = self.model.loss_weights
    self.metrics = self.model.metrics
    self.metrics_tensors = self.model.metrics_tensors
    self.metrics_names = self.model.metrics_names
    self.sample_weight_mode = self.model.sample_weight_mode

  def fit(self,
          x,
          y,
          batch_size=32,
          epochs=10,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0):
    """Trains the model for a fixed number of epochs.

    Arguments:
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        y: labels, as a Numpy array.
        batch_size: integer. Number of samples per gradient update.
        epochs: integer, the number of epochs to train the model.
        verbose: 0 for no logging to stdout,
            1 for progress bar logging, 2 for one log line per epoch.
        callbacks: list of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
            See [callbacks](/callbacks).
        validation_split: float (0. < x < 1).
            Fraction of the data to use as held-out validation data.
        validation_data: tuple (x_val, y_val) or tuple
            (x_val, y_val, val_sample_weights) to be used as held-out
            validation data. Will override validation_split.
        shuffle: boolean or str (for 'batch').
            Whether to shuffle the samples at each epoch.
            'batch' is a special option for dealing with the
            limitations of HDF5 data; it shuffles in batch-sized chunks.
        class_weight: dictionary mapping classes to a weight value,
            used for scaling the loss function (during training only).
        sample_weight: Numpy array of weights for
            the training samples, used for scaling the loss function
            (during training only). You can either pass a flat (1D)
            Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples),
            or in the case of temporal data,
            you can pass a 2D array with shape (samples, sequence_length),
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            sample_weight_mode="temporal" in compile().
        initial_epoch: epoch at which to start training
            (useful for resuming a previous training run)

    Returns:
        A `History` object. Its `History.history` attribute is
        a record of training loss values and metrics values
        at successive epochs, as well as validation loss values
        and validation metrics values (if applicable).

    Raises:
        RuntimeError: if the model was never compiled.
    """
    if self.model is None:
      raise RuntimeError('The model needs to be compiled ' 'before being used.')
    return self.model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_split=validation_split,
        validation_data=validation_data,
        shuffle=shuffle,
        class_weight=class_weight,
        sample_weight=sample_weight,
        initial_epoch=initial_epoch)

  def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):
    """Computes the loss on some input data, batch by batch.

    Arguments:
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        y: labels, as a Numpy array.
        batch_size: integer. Number of samples per gradient update.
        verbose: verbosity mode, 0 or 1.
        sample_weight: sample weights, as a Numpy array.

    Returns:
        Scalar test loss (if the model has no metrics)
        or list of scalars (if the model computes other metrics).
        The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        RuntimeError: if the model was never compiled.
    """
    if self.model is None:
      raise RuntimeError('The model needs to be compiled ' 'before being used.')
    return self.model.evaluate(
        x,
        y,
        batch_size=batch_size,
        verbose=verbose,
        sample_weight=sample_weight)

  def predict(self, x, batch_size=32, verbose=0):
    """Generates output predictions for the input samples.

    The input samples are processed batch by batch.

    Arguments:
        x: the input data, as a Numpy array.
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.

    Returns:
        A Numpy array of predictions.
    """
    if self.model is None:
      self.build()
    return self.model.predict(x, batch_size=batch_size, verbose=verbose)

  def predict_on_batch(self, x):
    """Returns predictions for a single batch of samples.

    Arguments:
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).

    Returns:
        A Numpy array of predictions.
    """
    if self.model is None:
      self.build()
    return self.model.predict_on_batch(x)

  def train_on_batch(self, x, y, class_weight=None, sample_weight=None):
    """Single gradient update over one batch of samples.

    Arguments:
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        y: labels, as a Numpy array.
        class_weight: dictionary mapping classes to a weight value,
            used for scaling the loss function (during training only).
        sample_weight: sample weights, as a Numpy array.

    Returns:
        Scalar training loss (if the model has no metrics)
        or list of scalars (if the model computes other metrics).
        The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        RuntimeError: if the model was never compiled.
    """
    if self.model is None:
      raise RuntimeError('The model needs to be compiled ' 'before being used.')
    return self.model.train_on_batch(
        x, y, sample_weight=sample_weight, class_weight=class_weight)

  def test_on_batch(self, x, y, sample_weight=None):
    """Evaluates the model over a single batch of samples.

    Arguments:
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        y: labels, as a Numpy array.
        sample_weight: sample weights, as a Numpy array.

    Returns:
        Scalar test loss (if the model has no metrics)
        or list of scalars (if the model computes other metrics).
        The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        RuntimeError: if the model was never compiled.
    """
    if self.model is None:
      raise RuntimeError('The model needs to be compiled ' 'before being used.')
    return self.model.test_on_batch(x, y, sample_weight=sample_weight)

  def predict_proba(self, x, batch_size=32, verbose=1):
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
      warnings.warn('Network returning invalid probability values. '
                    'The last layer might not normalize predictions '
                    'into probabilities '
                    '(like softmax or sigmoid would).')
    return preds

  def predict_classes(self, x, batch_size=32, verbose=1):
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

  def fit_generator(self,
                    generator,
                    steps_per_epoch,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_data=None,
                    validation_steps=None,
                    class_weight=None,
                    max_q_size=10,
                    workers=1,
                    pickle_safe=False,
                    initial_epoch=0):
    """Fits the model on data generated batch-by-batch by a Python generator.

    The generator is run in parallel to the model, for efficiency.
    For instance, this allows you to do real-time data augmentation
    on images on CPU in parallel to training your model on GPU.

    Arguments:
        generator: A generator.
            The output of the generator must be either
            - a tuple (inputs, targets)
            - a tuple (inputs, targets, sample_weights).
            All arrays should contain the same number of samples.
            The generator is expected to loop over its data
            indefinitely. An epoch finishes when `samples_per_epoch`
            samples have been seen by the model.
        steps_per_epoch: Total number of steps (batches of samples)
            to yield from `generator` before declaring one epoch
            finished and starting the next epoch. It should typically
            be equal to the number of unique samples of your dataset
            divided by the batch size.
        epochs: Integer, total number of iterations on the data.
        verbose: Verbosity mode, 0, 1, or 2.
        callbacks: List of callbacks to be called during training.
        validation_data: This can be either
            - A generator for the validation data
            - A tuple (inputs, targets)
            - A tuple (inputs, targets, sample_weights).
        validation_steps: Only relevant if `validation_data`
            is a generator.
            Number of steps to yield from validation generator
            at the end of every epoch. It should typically
            be equal to the number of unique samples of your
            validation dataset divided by the batch size.
        class_weight: Dictionary mapping class indices to a weight
            for the class.
        max_q_size: Maximum size for the generator queue
        workers: Maximum number of processes to spin up
        pickle_safe: Ff True, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.
        initial_epoch: Epoch at which to start training
            (useful for resuming a previous training run)

    Returns:
        A `History` object.

    Raises:
        RuntimeError: if the model was never compiled.

    Example:

    ```python
        def generate_arrays_from_file(path):
            while 1:
                f = open(path)
                for line in f:
                    # create Numpy arrays of input data
                    # and labels, from each line in the file
                    x, y = process_line(line)
                    yield (x, y)
                    f.close()

        model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                            samples_per_epoch=10000, epochs=10)
    ```
    """
    if self.model is None:
      raise RuntimeError('The model needs to be compiled ' 'before being used.')
    return self.model.fit_generator(
        generator,
        steps_per_epoch,
        epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=validation_steps,
        class_weight=class_weight,
        max_q_size=max_q_size,
        workers=workers,
        pickle_safe=pickle_safe,
        initial_epoch=initial_epoch)

  def evaluate_generator(self,
                         generator,
                         steps,
                         max_q_size=10,
                         workers=1,
                         pickle_safe=False):
    """Evaluates the model on a data generator.

    The generator should return the same kind of data
    as accepted by `test_on_batch`.

    Arguments:
        generator: Generator yielding tuples (inputs, targets)
            or (inputs, targets, sample_weights)
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        max_q_size: maximum size for the generator queue
        workers: maximum number of processes to spin up
        pickle_safe: if True, use process based threading.
            Note that because this implementation
            relies on multiprocessing, you should not pass
            non picklable arguments to the generator
            as they can't be passed easily to children processes.

    Returns:
        Scalar test loss (if the model has no metrics)
        or list of scalars (if the model computes other metrics).
        The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        RuntimeError: if the model was never compiled.
    """
    if self.model is None:
      raise RuntimeError('The model needs to be compiled ' 'before being used.')
    return self.model.evaluate_generator(
        generator,
        steps,
        max_q_size=max_q_size,
        workers=workers,
        pickle_safe=pickle_safe)

  def predict_generator(self,
                        generator,
                        steps,
                        max_q_size=10,
                        workers=1,
                        pickle_safe=False,
                        verbose=0):
    """Generates predictions for the input samples from a data generator.

    The generator should return the same kind of data as accepted by
    `predict_on_batch`.

    Arguments:
        generator: generator yielding batches of input samples.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        max_q_size: maximum size for the generator queue
        workers: maximum number of processes to spin up
        pickle_safe: if True, use process based threading.
            Note that because this implementation
            relies on multiprocessing, you should not pass
            non picklable arguments to the generator
            as they can't be passed easily to children processes.
        verbose: verbosity mode, 0 or 1.

    Returns:
        A Numpy array of predictions.
    """
    if self.model is None:
      self.build()
    return self.model.predict_generator(
        generator,
        steps,
        max_q_size=max_q_size,
        workers=workers,
        pickle_safe=pickle_safe,
        verbose=verbose)

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

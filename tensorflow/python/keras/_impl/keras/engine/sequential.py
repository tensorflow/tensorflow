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
import os

from tensorflow.python.framework import ops
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import layers as layer_module
from tensorflow.python.keras._impl.keras.engine import base_layer
from tensorflow.python.keras._impl.keras.engine import network
from tensorflow.python.keras._impl.keras.engine import saving
from tensorflow.python.keras._impl.keras.engine.input_layer import Input
from tensorflow.python.keras._impl.keras.engine.input_layer import InputLayer
from tensorflow.python.keras._impl.keras.engine.training import Model
from tensorflow.python.keras._impl.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

try:
  import h5py  # pylint: disable=g-import-not-at-top
except ImportError:
  h5py = None


@tf_export('keras.models.Sequential', 'keras.Sequential')
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
    self._is_graph_network = True
    self._is_compiled = False
    self._layers = []  # Stack of layers.
    self.model = None  # Internal Model instance.
    self.inputs = []  # List of input tensors
    self.outputs = []  # List of length 1: the output tensor (unique).
    self._trainable = True
    self._initial_weights = None
    self._input_layers = []

    # Model attributes.
    self._inbound_nodes = []
    self._outbound_nodes = []
    self.built = False

    # Set model name.
    if not name:
      prefix = 'sequential_'
      name = prefix + str(K.get_uid(prefix))
    self._name = name

    # Used by Layer base class.
    self._dtype = None
    self._activity_regularizer = None

    # The following properties are not actually used by Keras;
    # they exist for compatibility with TF's variable scoping mechanism.
    self._updates = []
    self._losses = []
    self._scope = None
    self._reuse = None
    self._base_name = name
    self._graph = ops.get_default_graph()

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
    if not isinstance(layer, (base_layer.Layer, base_layer.TFBaseLayer)):
      raise TypeError('The added layer must be '
                      'an instance of class Layer. '
                      'Found: ' + str(layer))
    if not self.outputs:
      # First layer in model: check that it is an input layer.
      if not isinstance(layer, InputLayer):
        # Create an input layer.
        # First, we need to infer its expected input shape and dtype.
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
        else:
          # We were passed a regular layer, and it should
          # know about its input shape. Otherwise, that's an error.
          if not hasattr(layer, '_batch_input_shape'):
            raise ValueError('The first layer in a '
                             'Sequential model must '
                             'get an `input_shape` argument.')
          batch_shape = layer._batch_input_shape
          dtype = layer.dtype
        # Instantiate the input layer.
        x = Input(
            batch_shape=batch_shape, dtype=dtype, name=layer.name + '_input')
        # This will build the current layer
        # and create the node connecting the current layer
        # to the input layer we just created.
        layer(x)

      if len(layer._inbound_nodes[-1].output_tensors) != 1:
        raise ValueError('All layers in a Sequential model '
                         'should have a single output tensor. '
                         'For multi-output layers, '
                         'use the functional API.')

      self.outputs = [layer._inbound_nodes[-1].output_tensors[0]]
      self.inputs = network.get_source_inputs(self.outputs[0])

      # We create an input node, which we will keep updated
      # as we add more layers
      base_layer.Node(
          outbound_layer=self,
          inbound_layers=[],
          node_indices=[],
          tensor_indices=[],
          input_tensors=self.inputs,
          output_tensors=self.outputs)
    else:
      output_tensor = layer(self.outputs[0])
      if isinstance(output_tensor, list):
        raise TypeError('All layers in a Sequential model '
                        'should have a single output tensor. '
                        'For multi-output layers, '
                        'use the functional API.')
      self.outputs = [output_tensor]
      # update self._inbound_nodes
      self._inbound_nodes[0].output_tensors = self.outputs
      self._inbound_nodes[0].output_shapes = [K.int_shape(self.outputs[0])]

    self._layers.append(layer)
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
      self._inbound_nodes = []
      self._outbound_nodes = []
    else:
      self.layers[-1]._outbound_nodes = []
      self.outputs = [self.layers[-1].output]
      # update self._inbound_nodes
      self._inbound_nodes[0].output_tensors = self.outputs
      self._inbound_nodes[0].output_shapes = [K.int_shape(self.outputs[0])]
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
    if not self.built:
      self.build()
    return self.model.get_layer(name, index)

  def call(self, inputs, **kwargs):
    if not self.built:
      self.build()
    return self.model.call(inputs, **kwargs)

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
    self._input_layers = self.model._input_layers
    self._output_layers = self.model._output_layers
    self._input_coordinates = self.model._input_coordinates
    self._output_coordinates = self.model._output_coordinates
    self._nodes_by_depth = self.model._nodes_by_depth
    self._network_nodes = self.model._network_nodes
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
    if not self.built:
      self.build()
    return self.model.uses_learning_phase

  def _gather_list_attr(self, attr):
    all_attrs = []
    for layer in self.layers:
      all_attrs += getattr(layer, attr, [])
    return all_attrs

  def _make_train_function(self):
    self.model._make_train_function()

  def _make_test_function(self):
    self.model._make_test_function()

  def _make_predict_function(self):
    self.model._make_predict_function()

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
  def regularizers(self):
    if not self.built:
      self.build()
    return self.model.regularizers

  def get_weights(self):
    """Retrieves the weights of the model.

    Returns:
        A flat list of Numpy arrays
        (one array per model weight).
    """
    if not self.built:
      self.build()
    return self.model.get_weights()

  def set_weights(self, weights):
    """Sets the weights of the model.

    Arguments:
        weights: Should be a list
            of Numpy arrays with shapes and types matching
            the output of `model.get_weights()`.
    """
    if not self.built:
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
      saving.load_weights_from_hdf5_group_by_name(f, layers)
    else:
      saving.load_weights_from_hdf5_group(f, layers)
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
    saving.save_weights_to_hdf5_group(f, layers)
    f.flush()
    f.close()

  def compile(self,
              optimizer,
              loss,
              metrics=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              **kwargs):
    """Configures the model for training.

    Arguments:
        optimizer: String (name of optimizer) or optimizer object.
            See [optimizers](/optimizers).
        loss: String (name of objective function) or objective function.
            See [losses](/losses).
            If the model has multiple outputs, you can use a different loss
            on each output by passing a dictionary or a list of losses.
            The loss value that will be minimized by the model
            will then be the sum of all individual losses.
        metrics: List of metrics to be evaluated by the model
            during training and testing.
            Typically you will use `metrics=['accuracy']`.
            To specify different metrics for different outputs of a
            multi-output model, you could also pass a dictionary,
            such as `metrics={'output_a': 'accuracy'}`.
        sample_weight_mode: If you need to do timestep-wise
            sample weighting (2D weights), set this to `"temporal"`.
            `None` defaults to sample-wise weights (1D).
            If the model has multiple outputs, you can use a different
            `sample_weight_mode` on each output by passing a
            dictionary or a list of modes.
        weighted_metrics: list of metrics to be evaluated and weighted
             by `sample_weight` or `class_weight` during training and testing.
        target_tensors: By default, Keras will create a placeholder for the
            model's target, which will be fed with the target data during
            training. If instead you would like to use your own
            target tensor (in turn, Keras will not expect external
            Numpy data for these targets at training time), you
            can specify them via the `target_tensors` argument.
            It should be a single tensor
            (for a single-output `Sequential` model).
        **kwargs: These arguments are passed into `tf.Session.run`.

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
        weighted_metrics=weighted_metrics,
        target_tensors=target_tensors,
        **kwargs)
    self.optimizer = self.model.optimizer
    self.loss = self.model.loss
    self.metrics = self.model.metrics
    self.loss_weights = self.model.loss_weights
    self.sample_weight_mode = self.model.sample_weight_mode
    self.weighted_metrics = self.model.weighted_metrics
    self.targets = self.model.targets
    self.metrics_tensors = self.model.metrics_tensors
    self.metrics_names = self.model.metrics_names
    self.sample_weights = self.model.sample_weights
    self.total_loss = self.model.total_loss

  def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          **kwargs):
    """Trains the model for a fixed number of epochs.

    Arguments:
        x: Numpy array of training data.
            If the input layer in the model is named, you can also pass a
            dictionary mapping the input name to a Numpy array.
            `x` can be `None` (default) if feeding from
            TensorFlow data tensors.
        y: Numpy array of target (label) data.
            If the output layer in the model is named, you can also pass a
            dictionary mapping the output name to a Numpy array.
            `y` can be `None` (default) if feeding from
            TensorFlow data tensors.
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, it will default to 32.
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        verbose: 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
            See [callbacks](/callbacks).
        validation_split: Float between 0 and 1:
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate
            the loss and any model metrics
            on this data at the end of each epoch.
            The validation data is selected from the last samples
            in the `x` and `y` data provided, before shuffling.
        validation_data: tuple `(x_val, y_val)` or tuple
            `(x_val, y_val, val_sample_weights)` on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data.
            This will override `validation_split`.
        shuffle: Boolean (whether to shuffle the training data
            before each epoch) or str (for 'batch').
            'batch' is a special option for dealing with the
            limitations of HDF5 data; it shuffles in batch-sized chunks.
            Has no effect when `steps_per_epoch` is not `None`.
        class_weight: Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function
            (during training only).
            This can be useful to tell the model to
            "pay more attention" to samples from
            an under-represented class.
        sample_weight: Optional Numpy array of weights for
            the training samples, used for weighting the loss function
            (during training only). You can either pass a flat (1D)
            Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples),
            or in the case of temporal data,
            you can pass a 2D array with shape
            `(samples, sequence_length)`,
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            `sample_weight_mode="temporal"` in `compile()`.
        initial_epoch: Epoch at which to start training
            (useful for resuming a previous training run).
        steps_per_epoch: Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of unique samples in your dataset divided by
            the batch size, or 1 if that cannot be determined.
        validation_steps: Only relevant if `steps_per_epoch`
            is specified. Total number of steps (batches of samples)
            to validate before stopping.
        **kwargs: Used for backwards compatibility support.

    Returns:
        A `History` object. Its `History.history` attribute is
        a record of training loss values and metrics values
        at successive epochs, as well as validation loss values
        and validation metrics values (if applicable).

    Raises:
        RuntimeError: If the model was never compiled.
        ValueError: In case of mismatch between the provided input data
            and what the model expects.
    """
    if not self.built:
      raise RuntimeError('The model needs to be compiled before being used.')
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
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps)

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
    if not self.built:
      raise RuntimeError('The model needs to be compiled before being used.')
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
    if not self.built:
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
    if not self.built:
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
    if not self.built:
      raise RuntimeError('The model needs to be compiled before being used.')
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
    if not self.built:
      raise RuntimeError('The model needs to be compiled before being used.')
    return self.model.test_on_batch(x, y, sample_weight=sample_weight)

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

  def fit_generator(self,
                    generator,
                    steps_per_epoch=None,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_data=None,
                    validation_steps=None,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True,
                    initial_epoch=0,
                    **kwargs):
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
            indefinitely. An epoch finishes when `steps_per_epoch`
            batches have been seen by the model.
        steps_per_epoch: Total number of steps (batches of samples)
            to yield from `generator` before declaring one epoch
            finished and starting the next epoch. It should typically
            be equal to the number of samples of your dataset
            divided by the batch size.
            Optional for `Sequence`: if unspecified, will use
            the `len(generator)` as a number of steps.
        epochs: Integer, total number of iterations on the data.
            Note that in conjunction with initial_epoch, the parameter
            epochs is to be understood as "final epoch". The model is
            not trained for n steps given by epochs, but until the
            epoch epochs is reached.
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
            be equal to the number of samples of your
            validation dataset divided by the batch size.
            Optional for `Sequence`: if unspecified, will use
            the `len(validation_data)` as a number of steps.
        class_weight: Dictionary mapping class indices to a weight
            for the class.
        max_queue_size: Maximum size for the generator queue
        workers: Maximum number of processes to spin up
        use_multiprocessing: If True, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.
       shuffle: Whether to shuffle the order of the batches at
              the beginning of each epoch. Only used with instances
              of `Sequence` (keras.utils.Sequence).
        initial_epoch: Epoch at which to start training
            (useful for resuming a previous training run)
        **kwargs: support for legacy arguments.

    Returns:
        A `History` object.

    Raises:
        RuntimeError: if the model was never compiled.
        ValueError: In case the generator yields
            data in an invalid format.

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
                            steps_per_epoch=1000, epochs=10)
    ```
    """
    # Legacy support
    if 'max_q_size' in kwargs:
      max_queue_size = kwargs.pop('max_q_size')
      logging.warning('The argument `max_q_size` has been renamed '
                      '`max_queue_size`. Update your method calls accordingly.')
    if 'pickle_safe' in kwargs:
      use_multiprocessing = kwargs.pop('pickle_safe')
      logging.warning('The argument `pickle_safe` has been renamed '
                      '`use_multiprocessing`. '
                      'Update your method calls accordingly.')
    if kwargs:
      raise ValueError('Unrecognized keyword arguments: ' + str(kwargs))

    if not self.built:
      raise RuntimeError('The model needs to be compiled before being used.')
    return self.model.fit_generator(
        generator,
        steps_per_epoch,
        epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=validation_steps,
        class_weight=class_weight,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        shuffle=shuffle,
        initial_epoch=initial_epoch)

  def evaluate_generator(self,
                         generator,
                         steps=None,
                         max_queue_size=10,
                         workers=1,
                         use_multiprocessing=False,
                         **kwargs):
    """Evaluates the model on a data generator.

    The generator should return the same kind of data
    as accepted by `test_on_batch`.

    Arguments:
        generator: Generator yielding tuples (inputs, targets)
            or (inputs, targets, sample_weights)
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
            Optional for `Sequence`: if unspecified, will use
            the `len(generator)` as a number of steps.
        max_queue_size: maximum size for the generator queue
        workers: maximum number of processes to spin up
        use_multiprocessing: if True, use process based threading.
            Note that because this implementation
            relies on multiprocessing, you should not pass
            non picklable arguments to the generator
            as they can't be passed easily to children processes.
        **kwargs: support for legacy arguments.

    Returns:
        Scalar test loss (if the model has no metrics)
        or list of scalars (if the model computes other metrics).
        The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        RuntimeError: if the model was never compiled.
        ValueError: In case the generator yields
            data in an invalid format.
    """
    # Legacy support
    if 'max_q_size' in kwargs:
      max_queue_size = kwargs.pop('max_q_size')
      logging.warning('The argument `max_q_size` has been renamed '
                      '`max_queue_size`. Update your method calls accordingly.')
    if 'pickle_safe' in kwargs:
      use_multiprocessing = kwargs.pop('pickle_safe')
      logging.warning('The argument `pickle_safe` has been renamed '
                      '`use_multiprocessing`. '
                      'Update your method calls accordingly.')
    if kwargs:
      raise ValueError('Unrecognized keyword arguments: ' + str(kwargs))

    if not self.built:
      raise RuntimeError('The model needs to be compiled before being used.')
    return self.model.evaluate_generator(
        generator,
        steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing)

  def predict_generator(self,
                        generator,
                        steps=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        verbose=0,
                        **kwargs):
    """Generates predictions for the input samples from a data generator.

    The generator should return the same kind of data as accepted by
    `predict_on_batch`.

    Arguments:
        generator: generator yielding batches of input samples.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
            Optional for `Sequence`: if unspecified, will use
            the `len(generator)` as a number of steps.
        max_queue_size: maximum size for the generator queue
        workers: maximum number of processes to spin up
        use_multiprocessing: if True, use process based threading.
            Note that because this implementation
            relies on multiprocessing, you should not pass
            non picklable arguments to the generator
            as they can't be passed easily to children processes.
        verbose: verbosity mode, 0 or 1.
        **kwargs: support for legacy arguments.

    Returns:
        A Numpy array of predictions.

    Raises:
        ValueError: In case the generator yields
            data in an invalid format.
    """
    # Legacy support
    if 'max_q_size' in kwargs:
      max_queue_size = kwargs.pop('max_q_size')
      logging.warning('The argument `max_q_size` has been renamed '
                      '`max_queue_size`. Update your method calls accordingly.')
    if 'pickle_safe' in kwargs:
      use_multiprocessing = kwargs.pop('pickle_safe')
      logging.warning('The argument `pickle_safe` has been renamed '
                      '`use_multiprocessing`. '
                      'Update your method calls accordingly.')
    if kwargs:
      raise ValueError('Unrecognized keyword arguments: ' + str(kwargs))

    if not self.built:
      self.build()
    return self.model.predict_generator(
        generator,
        steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
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

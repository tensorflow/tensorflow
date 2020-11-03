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
# pylint: disable=protected-access
"""Functions for saving and loading a Keras Model from HDF5 format.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.saving import model_config as model_config_lib
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging


# pylint: disable=g-import-not-at-top
try:
  import h5py
  HDF5_OBJECT_HEADER_LIMIT = 64512
except ImportError:
  h5py = None
# pylint: enable=g-import-not-at-top

# TODO(b/134426265): Switch back to single-quotes to match the rest of the file
# once the issue with copybara is fixed.
# pylint:disable=g-inconsistent-quotes
sequential_lib = LazyLoader(
    "sequential_lib", globals(),
    "tensorflow.python.keras.engine.sequential")
# pylint:enable=g-inconsistent-quotes


def save_model_to_hdf5(model, filepath, overwrite=True, include_optimizer=True):
  """Saves a model to a HDF5 file.

  The saved model contains:
      - the model's configuration (topology)
      - the model's weights
      - the model's optimizer's state (if any)

  Thus the saved model can be reinstantiated in
  the exact same state, without any of the code
  used for model definition or training.

  Arguments:
      model: Keras model instance to be saved.
      filepath: One of the following:
          - String, path where to save the model
          - `h5py.File` object where to save the model
      overwrite: Whether we should overwrite any existing
          model at the target location, or instead
          ask the user with a manual prompt.
      include_optimizer: If True, save optimizer's state together.

  Raises:
      ImportError: if h5py is not available.
  """

  if h5py is None:
    raise ImportError('`save_model` requires h5py.')

  # TODO(psv) Add warning when we save models that contain non-serializable
  # entities like metrics added using `add_metric` and losses added using
  # `add_loss.`
  if len(model.weights) != len(model._undeduplicated_weights):
    logging.warning('Found duplicated `Variable`s in Model\'s `weights`. '
                    'This is usually caused by `Variable`s being shared by '
                    'Layers in the Model. These `Variable`s will be treated '
                    'as separate `Variable`s when the Model is restored. To '
                    'avoid this, please save with `save_format="tf"`.')

  if not isinstance(filepath, h5py.File):
    # If file exists and should not be overwritten.
    if not overwrite and os.path.isfile(filepath):
      proceed = ask_to_proceed_with_overwrite(filepath)
      if not proceed:
        return

    # Try creating dir if not exist
    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
      gfile.MakeDirs(dirpath)

    f = h5py.File(filepath, mode='w')
    opened_new_file = True
  else:
    f = filepath
    opened_new_file = False

  try:
    model_metadata = saving_utils.model_metadata(model, include_optimizer)
    for k, v in model_metadata.items():
      if isinstance(v, (dict, list, tuple)):
        f.attrs[k] = json.dumps(
            v, default=json_utils.get_json_type).encode('utf8')
      else:
        f.attrs[k] = v

    model_weights_group = f.create_group('model_weights')
    model_layers = model.layers
    save_weights_to_hdf5_group(model_weights_group, model_layers)

    # TODO(b/128683857): Add integration tests between tf.keras and external
    # Keras, to avoid breaking TF.js users.
    if (include_optimizer and model.optimizer and
        not isinstance(model.optimizer, optimizer_v1.TFOptimizer)):
      save_optimizer_weights_to_hdf5_group(f, model.optimizer)

    f.flush()
  finally:
    if opened_new_file:
      f.close()


def load_model_from_hdf5(filepath, custom_objects=None, compile=True):  # pylint: disable=redefined-builtin
  """Loads a model saved via `save_model_to_hdf5`.

  Arguments:
      filepath: One of the following:
          - String, path to the saved model
          - `h5py.File` object from which to load the model
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.
      compile: Boolean, whether to compile the model
          after loading.

  Returns:
      A Keras model instance. If an optimizer was found
      as part of the saved model, the model is already
      compiled. Otherwise, the model is uncompiled and
      a warning will be displayed. When `compile` is set
      to False, the compilation is omitted without any
      warning.

  Raises:
      ImportError: if h5py is not available.
      ValueError: In case of an invalid savefile.
  """
  if h5py is None:
    raise ImportError('`load_model` requires h5py.')

  if not custom_objects:
    custom_objects = {}

  opened_new_file = not isinstance(filepath, h5py.File)
  if opened_new_file:
    f = h5py.File(filepath, mode='r')
  else:
    f = filepath

  model = None
  try:
    # instantiate model
    model_config = f.attrs.get('model_config')
    if model_config is None:
      raise ValueError('No model found in config file.')
    model_config = json_utils.decode(model_config.decode('utf-8'))
    model = model_config_lib.model_from_config(model_config,
                                               custom_objects=custom_objects)

    # set weights
    load_weights_from_hdf5_group(f['model_weights'], model.layers)

    if compile:
      # instantiate optimizer
      training_config = f.attrs.get('training_config')
      if training_config is None:
        logging.warning('No training configuration found in the save file, so '
                        'the model was *not* compiled. Compile it manually.')
        return model
      training_config = json_utils.decode(training_config.decode('utf-8'))

      # Compile model.
      model.compile(**saving_utils.compile_args_from_training_config(
          training_config, custom_objects))
      saving_utils.try_build_compiled_arguments(model)

      # Set optimizer weights.
      if 'optimizer_weights' in f:
        try:
          model.optimizer._create_all_weights(model.trainable_variables)
        except (NotImplementedError, AttributeError):
          logging.warning(
              'Error when creating the weights of optimizer {}, making it '
              'impossible to restore the saved optimizer state. As a result, '
              'your model is starting with a freshly initialized optimizer.')

        optimizer_weight_values = load_optimizer_weights_from_hdf5_group(f)
        try:
          model.optimizer.set_weights(optimizer_weight_values)
        except ValueError:
          logging.warning('Error in loading the saved optimizer '
                          'state. As a result, your model is '
                          'starting with a freshly initialized '
                          'optimizer.')
  finally:
    if opened_new_file:
      f.close()
  return model


def preprocess_weights_for_loading(layer,
                                   weights,
                                   original_keras_version=None,
                                   original_backend=None):
  """Preprocess layer weights between different Keras formats.

  Converts layers weights from Keras 1 format to Keras 2 and also weights of
  CuDNN layers in Keras 2.

  Arguments:
      layer: Layer instance.
      weights: List of weights values (Numpy arrays).
      original_keras_version: Keras version for the weights, as a string.
      original_backend: Keras backend the weights were trained with,
          as a string.

  Returns:
      A list of weights values (Numpy arrays).
  """
  def convert_nested_bidirectional(weights):
    """Converts layers nested in `Bidirectional` wrapper.

    This function uses `preprocess_weights_for_loading()` for converting
    layers.

    Arguments:
        weights: List of weights values (Numpy arrays).

    Returns:
        A list of weights values (Numpy arrays).
    """
    num_weights_per_layer = len(weights) // 2
    forward_weights = preprocess_weights_for_loading(
        layer.forward_layer, weights[:num_weights_per_layer],
        original_keras_version, original_backend)
    backward_weights = preprocess_weights_for_loading(
        layer.backward_layer, weights[num_weights_per_layer:],
        original_keras_version, original_backend)
    return forward_weights + backward_weights

  def convert_nested_time_distributed(weights):
    """Converts layers nested in `TimeDistributed` wrapper.

    This function uses `preprocess_weights_for_loading()` for converting nested
    layers.

    Arguments:
        weights: List of weights values (Numpy arrays).

    Returns:
        A list of weights values (Numpy arrays).
    """
    return preprocess_weights_for_loading(
        layer.layer, weights, original_keras_version, original_backend)

  def convert_nested_model(weights):
    """Converts layers nested in `Model` or `Sequential`.

    This function uses `preprocess_weights_for_loading()` for converting nested
    layers.

    Arguments:
        weights: List of weights values (Numpy arrays).

    Returns:
        A list of weights values (Numpy arrays).
    """
    trainable_weights = weights[:len(layer.trainable_weights)]
    non_trainable_weights = weights[len(layer.trainable_weights):]

    new_trainable_weights = []
    new_non_trainable_weights = []

    for sublayer in layer.layers:
      num_trainable_weights = len(sublayer.trainable_weights)
      num_non_trainable_weights = len(sublayer.non_trainable_weights)
      if sublayer.weights:
        preprocessed = preprocess_weights_for_loading(
            layer=sublayer,
            weights=(trainable_weights[:num_trainable_weights] +
                     non_trainable_weights[:num_non_trainable_weights]),
            original_keras_version=original_keras_version,
            original_backend=original_backend)
        new_trainable_weights.extend(preprocessed[:num_trainable_weights])
        new_non_trainable_weights.extend(preprocessed[num_trainable_weights:])

        trainable_weights = trainable_weights[num_trainable_weights:]
        non_trainable_weights = non_trainable_weights[
            num_non_trainable_weights:]

    return new_trainable_weights + new_non_trainable_weights

  # Convert layers nested in Bidirectional/Model/Sequential.
  # Both transformation should be ran for both Keras 1->2 conversion
  # and for conversion of CuDNN layers.
  if layer.__class__.__name__ == 'Bidirectional':
    weights = convert_nested_bidirectional(weights)
  if layer.__class__.__name__ == 'TimeDistributed':
    weights = convert_nested_time_distributed(weights)
  elif layer.__class__.__name__ in ['Model', 'Sequential']:
    weights = convert_nested_model(weights)

  if original_keras_version == '1':
    if layer.__class__.__name__ == 'TimeDistributed':
      weights = preprocess_weights_for_loading(
          layer.layer, weights, original_keras_version, original_backend)

    if layer.__class__.__name__ == 'Conv1D':
      shape = weights[0].shape
      # Handle Keras 1.1 format
      if shape[:2] != (layer.kernel_size[0], 1) or shape[3] != layer.filters:
        # Legacy shape:
        # (filters, input_dim, filter_length, 1)
        assert shape[0] == layer.filters and shape[2:] == (layer.kernel_size[0],
                                                           1)
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
      weights[0] = weights[0][:, 0, :, :]

    if layer.__class__.__name__ == 'Conv2D':
      if layer.data_format == 'channels_first':
        # old: (filters, stack_size, kernel_rows, kernel_cols)
        # new: (kernel_rows, kernel_cols, stack_size, filters)
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

    if layer.__class__.__name__ == 'Conv2DTranspose':
      if layer.data_format == 'channels_last':
        # old: (kernel_rows, kernel_cols, stack_size, filters)
        # new: (kernel_rows, kernel_cols, filters, stack_size)
        weights[0] = np.transpose(weights[0], (0, 1, 3, 2))
      if layer.data_format == 'channels_first':
        # old: (filters, stack_size, kernel_rows, kernel_cols)
        # new: (kernel_rows, kernel_cols, filters, stack_size)
        weights[0] = np.transpose(weights[0], (2, 3, 0, 1))

    if layer.__class__.__name__ == 'Conv3D':
      if layer.data_format == 'channels_first':
        # old: (filters, stack_size, ...)
        # new: (..., stack_size, filters)
        weights[0] = np.transpose(weights[0], (2, 3, 4, 1, 0))

    if layer.__class__.__name__ == 'GRU':
      if len(weights) == 9:
        kernel = np.concatenate([weights[0], weights[3], weights[6]], axis=-1)
        recurrent_kernel = np.concatenate(
            [weights[1], weights[4], weights[7]], axis=-1)
        bias = np.concatenate([weights[2], weights[5], weights[8]], axis=-1)
        weights = [kernel, recurrent_kernel, bias]

    if layer.__class__.__name__ == 'LSTM':
      if len(weights) == 12:
        # old: i, c, f, o
        # new: i, f, c, o
        kernel = np.concatenate(
            [weights[0], weights[6], weights[3], weights[9]], axis=-1)
        recurrent_kernel = np.concatenate(
            [weights[1], weights[7], weights[4], weights[10]], axis=-1)
        bias = np.concatenate(
            [weights[2], weights[8], weights[5], weights[11]], axis=-1)
        weights = [kernel, recurrent_kernel, bias]

    if layer.__class__.__name__ == 'ConvLSTM2D':
      if len(weights) == 12:
        kernel = np.concatenate(
            [weights[0], weights[6], weights[3], weights[9]], axis=-1)
        recurrent_kernel = np.concatenate(
            [weights[1], weights[7], weights[4], weights[10]], axis=-1)
        bias = np.concatenate(
            [weights[2], weights[8], weights[5], weights[11]], axis=-1)
        if layer.data_format == 'channels_first':
          # old: (filters, stack_size, kernel_rows, kernel_cols)
          # new: (kernel_rows, kernel_cols, stack_size, filters)
          kernel = np.transpose(kernel, (2, 3, 1, 0))
          recurrent_kernel = np.transpose(recurrent_kernel, (2, 3, 1, 0))
        weights = [kernel, recurrent_kernel, bias]

  conv_layers = ['Conv1D', 'Conv2D', 'Conv3D', 'Conv2DTranspose', 'ConvLSTM2D']
  if layer.__class__.__name__ in conv_layers:
    if K.int_shape(layer.weights[0]) != weights[0].shape:
      weights[0] = np.transpose(weights[0], (3, 2, 0, 1))
      if layer.__class__.__name__ == 'ConvLSTM2D':
        weights[1] = np.transpose(weights[1], (3, 2, 0, 1))

  # convert CuDNN layers
  return _convert_rnn_weights(layer, weights)


def _convert_rnn_weights(layer, weights):
  """Converts weights for RNN layers between native and CuDNN format.

  Input kernels for each gate are transposed and converted between Fortran
  and C layout, recurrent kernels are transposed. For LSTM biases are summed/
  split in half, for GRU biases are reshaped.

  Weights can be converted in both directions between `LSTM` and`CuDNNSLTM`
  and between `CuDNNGRU` and `GRU(reset_after=True)`. Default `GRU` is not
  compatible with `CuDNNGRU`.

  For missing biases in `LSTM`/`GRU` (`use_bias=False`) no conversion is made.

  Arguments:
      layer: Target layer instance.
      weights: List of source weights values (input kernels, recurrent
          kernels, [biases]) (Numpy arrays).

  Returns:
      A list of converted weights values (Numpy arrays).

  Raises:
      ValueError: for incompatible GRU layer/weights or incompatible biases
  """

  def transform_kernels(kernels, func, n_gates):
    """Transforms kernel for each gate separately using given function.

    Arguments:
        kernels: Stacked array of kernels for individual gates.
        func: Function applied to kernel of each gate.
        n_gates: Number of gates (4 for LSTM, 3 for GRU).

    Returns:
        Stacked array of transformed kernels.
    """
    return np.hstack([func(k) for k in np.hsplit(kernels, n_gates)])

  def transpose_input(from_cudnn):
    """Makes a function that transforms input kernels from/to CuDNN format.

    It keeps the shape, but changes between the layout (Fortran/C). Eg.:

    ```
    Keras                 CuDNN
    [[0, 1, 2],  <--->  [[0, 2, 4],
     [3, 4, 5]]          [1, 3, 5]]
    ```

    It can be passed to `transform_kernels()`.

    Arguments:
        from_cudnn: `True` if source weights are in CuDNN format, `False`
            if they're in plain Keras format.

    Returns:
        Function that converts input kernel to the other format.
    """
    order = 'F' if from_cudnn else 'C'

    def transform(kernel):
      return kernel.T.reshape(kernel.shape, order=order)

    return transform

  target_class = layer.__class__.__name__

  # convert the weights between CuDNNLSTM and LSTM
  if target_class in ['LSTM', 'CuDNNLSTM'] and len(weights) == 3:
    # determine if we're loading a CuDNNLSTM layer
    # from the number of bias weights:
    # CuDNNLSTM has (units * 8) weights; while LSTM has (units * 4)
    # if there's no bias weight in the file, skip this conversion
    units = weights[1].shape[0]
    bias_shape = weights[2].shape
    n_gates = 4

    if bias_shape == (2 * units * n_gates,):
      source = 'CuDNNLSTM'
    elif bias_shape == (units * n_gates,):
      source = 'LSTM'
    else:
      raise ValueError('Invalid bias shape: ' + str(bias_shape))

    def convert_lstm_weights(weights, from_cudnn=True):
      """Converts the weights between CuDNNLSTM and LSTM.

      Arguments:
        weights: Original weights.
        from_cudnn: Indicates whether original weights are from CuDNN layer.

      Returns:
        Updated weights compatible with LSTM.
      """

      # Transpose (and reshape) input and recurrent kernels
      kernels = transform_kernels(weights[0], transpose_input(from_cudnn),
                                  n_gates)
      recurrent_kernels = transform_kernels(weights[1], lambda k: k.T, n_gates)
      if from_cudnn:
        # merge input and recurrent biases into a single set
        biases = np.sum(np.split(weights[2], 2, axis=0), axis=0)
      else:
        # Split single set of biases evenly to two sets. The way of
        # splitting doesn't matter as long as the two sets sum is kept.
        biases = np.tile(0.5 * weights[2], 2)
      return [kernels, recurrent_kernels, biases]

    if source != target_class:
      weights = convert_lstm_weights(weights, from_cudnn=source == 'CuDNNLSTM')

  # convert the weights between CuDNNGRU and GRU(reset_after=True)
  if target_class in ['GRU', 'CuDNNGRU'] and len(weights) == 3:
    # We can determine the source of the weights from the shape of the bias.
    # If there is no bias we skip the conversion since
    # CuDNNGRU always has biases.

    units = weights[1].shape[0]
    bias_shape = weights[2].shape
    n_gates = 3

    def convert_gru_weights(weights, from_cudnn=True):
      """Converts the weights between CuDNNGRU and GRU.

      Arguments:
        weights: Original weights.
        from_cudnn: Indicates whether original weights are from CuDNN layer.

      Returns:
        Updated weights compatible with GRU.
      """

      kernels = transform_kernels(weights[0], transpose_input(from_cudnn),
                                  n_gates)
      recurrent_kernels = transform_kernels(weights[1], lambda k: k.T, n_gates)
      biases = np.array(weights[2]).reshape((2, -1) if from_cudnn else -1)
      return [kernels, recurrent_kernels, biases]

    if bias_shape == (2 * units * n_gates,):
      source = 'CuDNNGRU'
    elif bias_shape == (2, units * n_gates):
      source = 'GRU(reset_after=True)'
    elif bias_shape == (units * n_gates,):
      source = 'GRU(reset_after=False)'
    else:
      raise ValueError('Invalid bias shape: ' + str(bias_shape))

    if target_class == 'CuDNNGRU':
      target = 'CuDNNGRU'
    elif layer.reset_after:
      target = 'GRU(reset_after=True)'
    else:
      target = 'GRU(reset_after=False)'

    # only convert between different types
    if source != target:
      types = (source, target)
      if 'GRU(reset_after=False)' in types:
        raise ValueError('%s is not compatible with %s' % types)
      if source == 'CuDNNGRU':
        weights = convert_gru_weights(weights, from_cudnn=True)
      elif source == 'GRU(reset_after=True)':
        weights = convert_gru_weights(weights, from_cudnn=False)

  return weights


def save_optimizer_weights_to_hdf5_group(hdf5_group, optimizer):
  """Saves optimizer weights of a optimizer to a HDF5 group.

  Arguments:
      hdf5_group: HDF5 group.
      optimizer: optimizer instance.
  """

  symbolic_weights = getattr(optimizer, 'weights')
  if symbolic_weights:
    weights_group = hdf5_group.create_group('optimizer_weights')
    weight_names = [str(w.name).encode('utf8') for w in symbolic_weights]
    save_attributes_to_hdf5_group(weights_group, 'weight_names', weight_names)
    weight_values = K.batch_get_value(symbolic_weights)
    for name, val in zip(weight_names, weight_values):
      param_dset = weights_group.create_dataset(
          name, val.shape, dtype=val.dtype)
      if not val.shape:
        # scalar
        param_dset[()] = val
      else:
        param_dset[:] = val


def load_optimizer_weights_from_hdf5_group(hdf5_group):
  """Load optimizer weights from a HDF5 group.

  Arguments:
      hdf5_group: A pointer to a HDF5 group.

  Returns:
      data: List of optimizer weight names.
  """
  weights_group = hdf5_group['optimizer_weights']
  optimizer_weight_names = load_attributes_from_hdf5_group(
      weights_group, 'weight_names')
  return [weights_group[weight_name] for weight_name in optimizer_weight_names]


def save_weights_to_hdf5_group(f, layers):
  """Saves the weights of a list of layers to a HDF5 group.

  Arguments:
      f: HDF5 group.
      layers: List of layer instances.
  """
  from tensorflow.python.keras import __version__ as keras_version  # pylint: disable=g-import-not-at-top

  save_attributes_to_hdf5_group(
      f, 'layer_names', [layer.name.encode('utf8') for layer in layers])
  f.attrs['backend'] = K.backend().encode('utf8')
  f.attrs['keras_version'] = str(keras_version).encode('utf8')

  # Sort model layers by layer name to ensure that group names are strictly
  # growing to avoid prefix issues.
  for layer in sorted(layers, key=lambda x: x.name):
    g = f.create_group(layer.name)
    weights = _legacy_weights(layer)
    weight_values = K.batch_get_value(weights)
    weight_names = [w.name.encode('utf8') for w in weights]
    save_attributes_to_hdf5_group(g, 'weight_names', weight_names)
    for name, val in zip(weight_names, weight_values):
      param_dset = g.create_dataset(name, val.shape, dtype=val.dtype)
      if not val.shape:
        # scalar
        param_dset[()] = val
      else:
        param_dset[:] = val


def load_weights_from_hdf5_group(f, layers):
  """Implements topological (order-based) weight loading.

  Arguments:
      f: A pointer to a HDF5 group.
      layers: a list of target layers.

  Raises:
      ValueError: in case of mismatch between provided layers
          and weights file.
  """
  if 'keras_version' in f.attrs:
    original_keras_version = f.attrs['keras_version'].decode('utf8')
  else:
    original_keras_version = '1'
  if 'backend' in f.attrs:
    original_backend = f.attrs['backend'].decode('utf8')
  else:
    original_backend = None

  filtered_layers = []
  for layer in layers:
    weights = _legacy_weights(layer)
    if weights:
      filtered_layers.append(layer)

  layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
  filtered_layer_names = []
  for name in layer_names:
    g = f[name]
    weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
    if weight_names:
      filtered_layer_names.append(name)
  layer_names = filtered_layer_names
  if len(layer_names) != len(filtered_layers):
    raise ValueError('You are trying to load a weight file '
                     'containing ' + str(len(layer_names)) +
                     ' layers into a model with ' + str(len(filtered_layers)) +
                     ' layers.')

  # We batch weight value assignments in a single backend call
  # which provides a speedup in TensorFlow.
  weight_value_tuples = []
  for k, name in enumerate(layer_names):
    g = f[name]
    weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
    weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
    layer = filtered_layers[k]
    symbolic_weights = _legacy_weights(layer)
    weight_values = preprocess_weights_for_loading(
        layer, weight_values, original_keras_version, original_backend)
    if len(weight_values) != len(symbolic_weights):
      raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
                       '" in the current model) was found to '
                       'correspond to layer ' + name + ' in the save file. '
                       'However the new layer ' + layer.name + ' expects ' +
                       str(len(symbolic_weights)) +
                       ' weights, but the saved weights have ' +
                       str(len(weight_values)) + ' elements.')
    weight_value_tuples += zip(symbolic_weights, weight_values)
  K.batch_set_value(weight_value_tuples)


def load_weights_from_hdf5_group_by_name(
    f, layers, skip_mismatch=False):
  """Implements name-based weight loading.

  (instead of topological weight loading).

  Layers that have no matching name are skipped.

  Arguments:
      f: A pointer to a HDF5 group.
      layers: a list of target layers.
      skip_mismatch: Boolean, whether to skip loading of layers
          where there is a mismatch in the number of weights,
          or a mismatch in the shape of the weights.

  Raises:
      ValueError: in case of mismatch between provided layers
          and weights file and skip_match=False.
  """
  if 'keras_version' in f.attrs:
    original_keras_version = f.attrs['keras_version'].decode('utf8')
  else:
    original_keras_version = '1'
  if 'backend' in f.attrs:
    original_backend = f.attrs['backend'].decode('utf8')
  else:
    original_backend = None

  # New file format.
  layer_names = load_attributes_from_hdf5_group(f, 'layer_names')

  # Reverse index of layer name to list of layers with name.
  index = {}
  for layer in layers:
    if layer.name:
      index.setdefault(layer.name, []).append(layer)

  # We batch weight value assignments in a single backend call
  # which provides a speedup in TensorFlow.
  weight_value_tuples = []
  for k, name in enumerate(layer_names):
    g = f[name]
    weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
    weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]

    for layer in index.get(name, []):
      symbolic_weights = _legacy_weights(layer)
      weight_values = preprocess_weights_for_loading(
          layer, weight_values, original_keras_version, original_backend)
      if len(weight_values) != len(symbolic_weights):
        if skip_mismatch:
          logging.warning('Skipping loading of weights for '
                          'layer {}'.format(layer.name) + ' due to mismatch '
                          'in number of weights ({} vs {}).'.format(
                              len(symbolic_weights), len(weight_values)))
          continue
        raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
                         '") expects ' + str(len(symbolic_weights)) +
                         ' weight(s), but the saved weights' + ' have ' +
                         str(len(weight_values)) + ' element(s).')
      # Set values.
      for i in range(len(weight_values)):
        if K.int_shape(symbolic_weights[i]) != weight_values[i].shape:
          if skip_mismatch:
            logging.warning('Skipping loading of weights for '
                            'layer {}'.format(layer.name) + ' due to '
                            'mismatch in shape ({} vs {}).'.format(
                                symbolic_weights[i].shape,
                                weight_values[i].shape))
            continue
          raise ValueError('Layer #' + str(k) +' (named "' + layer.name +
                           '"), weight ' + str(symbolic_weights[i]) +
                           ' has shape {}'.format(K.int_shape(
                               symbolic_weights[i])) +
                           ', but the saved weight has shape ' +
                           str(weight_values[i].shape) + '.')

        else:
          weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
  K.batch_set_value(weight_value_tuples)


def save_attributes_to_hdf5_group(group, name, data):
  """Saves attributes (data) of the specified name into the HDF5 group.

  This method deals with an inherent problem of HDF5 file which is not
  able to store data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

  Arguments:
      group: A pointer to a HDF5 group.
      name: A name of the attributes to save.
      data: Attributes data to store.

  Raises:
    RuntimeError: If any single attribute is too large to be saved.
  """
  # Check that no item in `data` is larger than `HDF5_OBJECT_HEADER_LIMIT`
  # because in that case even chunking the array would not make the saving
  # possible.
  bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]

  # Expecting this to never be true.
  if bad_attributes:
    raise RuntimeError('The following attributes cannot be saved to HDF5 '
                       'file because they are larger than %d bytes: %s' %
                       (HDF5_OBJECT_HEADER_LIMIT, ', '.join(bad_attributes)))

  data_npy = np.asarray(data)

  num_chunks = 1
  chunked_data = np.array_split(data_npy, num_chunks)

  # This will never loop forever thanks to the test above.
  while any(x.nbytes > HDF5_OBJECT_HEADER_LIMIT for x in chunked_data):
    num_chunks += 1
    chunked_data = np.array_split(data_npy, num_chunks)

  if num_chunks > 1:
    for chunk_id, chunk_data in enumerate(chunked_data):
      group.attrs['%s%d' % (name, chunk_id)] = chunk_data
  else:
    group.attrs[name] = data


def load_attributes_from_hdf5_group(group, name):
  """Loads attributes of the specified name from the HDF5 group.

  This method deals with an inherent problem
  of HDF5 file which is not able to store
  data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

  Arguments:
      group: A pointer to a HDF5 group.
      name: A name of the attributes to load.

  Returns:
      data: Attributes data.
  """
  if name in group.attrs:
    data = [n.decode('utf8') for n in group.attrs[name]]
  else:
    data = []
    chunk_id = 0
    while '%s%d' % (name, chunk_id) in group.attrs:
      data.extend(
          [n.decode('utf8') for n in group.attrs['%s%d' % (name, chunk_id)]])
      chunk_id += 1
  return data


def _legacy_weights(layer):
  """DO NOT USE.

  For legacy reason, the layer.weights was in the order of
  [self.trainable_weights + self.non_trainable_weights], and this order was
  used for preserving the weights in h5 format. The new order of layer.weights
  are the same as layer.get_weights() which is more intuitive for user. To
  keep supporting the existing saved h5 file, this method should be used to
  save/load weights. In future version, we will delete this method and
  introduce a breaking change for h5 and stay with the new order for weights.

  Args:
    layer: a `tf.keras.Model` or `tf.keras.layers.Layer` instance.

  Returns:
    A list of variables with the order of trainable_weights, followed by
      non_trainable_weights.
  """
  weights = layer.trainable_weights + layer.non_trainable_weights
  if any(not isinstance(w, variables_module.Variable) for w in weights):
    raise NotImplementedError(
        'Save or restore weights that is not an instance of `tf.Variable` is '
        'not supported in h5, use `save_format=\'tf\'` instead. Got a model '
        'or layer {} with weights {}'.format(layer.__class__.__name__, weights))
  return weights

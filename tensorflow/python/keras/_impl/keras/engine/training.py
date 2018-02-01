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
"""Training-related part of the Keras engine.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import callbacks as cbks
from tensorflow.python.keras._impl.keras import losses
from tensorflow.python.keras._impl.keras import metrics as metrics_module
from tensorflow.python.keras._impl.keras import optimizers
from tensorflow.python.keras._impl.keras.engine.topology import Network
from tensorflow.python.keras._impl.keras.utils.data_utils import GeneratorEnqueuer
from tensorflow.python.keras._impl.keras.utils.data_utils import OrderedEnqueuer
from tensorflow.python.keras._impl.keras.utils.data_utils import Sequence
from tensorflow.python.keras._impl.keras.utils.generic_utils import Progbar
from tensorflow.python.platform import tf_logging as logging

try:
  from scipy.sparse import issparse  # pylint: disable=g-import-not-at-top
except ImportError:
  issparse = None


def _standardize_input_data(data,
                            names,
                            shapes=None,
                            check_batch_axis=True,
                            exception_prefix=''):
  """Normalizes inputs and targets provided by users.

  Users may pass data as a list of arrays, dictionary of arrays,
  or as a single array. We normalize this to an ordered list of
  arrays (same order as `names`), while checking that the provided
  arrays have shapes that match the network's expectations.

  Arguments:
      data: User-provided input data (polymorphic).
      names: List of expected array names.
      shapes: Optional list of expected array shapes.
      check_batch_axis: Boolean; whether to check that
          the batch axis of the arrays matches the expected
          value found in `shapes`.
      exception_prefix: String prefix used for exception formatting.

  Returns:
      List of standardized input arrays (one array per model input).

  Raises:
      ValueError: in case of improperly formatted user-provided data.
  """
  if not names:
    if data is not None and hasattr(data, '__len__') and len(data):
      raise ValueError('Error when checking model ' + exception_prefix + ': '
                       'expected no data, but got:', data)
    return []
  if data is None:
    return [None for _ in range(len(names))]

  if isinstance(data, dict):
    try:
      data = [
          data[x].values
          if data[x].__class__.__name__ == 'DataFrame' else data[x]
          for x in names
      ]
      data = [np.expand_dims(x, 1) if x.ndim == 1 else x for x in data]
    except KeyError as e:
      raise ValueError('No data provided for "' + e.args[0] + '". Need data '
                       'for each key in: ' + str(names))
  elif isinstance(data, list):
    data = [
        x.values if x.__class__.__name__ == 'DataFrame' else x for x in data
    ]
    data = [
        np.expand_dims(x, 1) if x is not None and x.ndim == 1 else x
        for x in data
    ]
  else:
    data = data.values if data.__class__.__name__ == 'DataFrame' else data
    data = [np.expand_dims(data, 1)] if data.ndim == 1 else [data]

  if len(data) != len(names):
    if data and hasattr(data[0], 'shape'):
      raise ValueError('Error when checking model ' + exception_prefix +
                       ': the list of Numpy arrays that you are passing to '
                       'your model is not the size the model expected. '
                       'Expected to see ' + str(len(names)) + ' array(s), '
                       'but instead got the following list of ' +
                       str(len(data)) + ' arrays: ' + str(data)[:200] + '...')
    elif len(names) > 1:
      raise ValueError(
          'Error when checking model ' + exception_prefix +
          ': you are passing a list as input to your model, '
          'but the model expects a list of ' + str(len(names)) +
          ' Numpy arrays instead. The list you passed was: ' + str(data)[:200])
    elif len(data) == 1 and not hasattr(data[0], 'shape'):
      raise TypeError('Error when checking model ' + exception_prefix +
                      ': data should be a Numpy array, or list/dict of '
                      'Numpy arrays. Found: ' + str(data)[:200] + '...')
    elif len(names) == 1:
      data = [np.asarray(data)]

  # Check shapes compatibility.
  if shapes:
    for i in range(len(names)):
      if shapes[i] is not None:
        data_shape = data[i].shape
        shape = shapes[i]
        if data[i].ndim != len(shape):
          raise ValueError('Error when checking ' + exception_prefix +
                           ': expected ' + names[i] + ' to have ' +
                           str(len(shape)) + ' dimensions, but got array '
                           'with shape ' + str(data_shape))
        if not check_batch_axis:
          data_shape = data_shape[1:]
          shape = shape[1:]
        for dim, ref_dim in zip(data_shape, shape):
          if ref_dim != dim and ref_dim:
            raise ValueError(
                'Error when checking ' + exception_prefix + ': expected ' +
                names[i] + ' to have shape ' + str(shape) +
                ' but got array with shape ' + str(data_shape))
  return data


def _standardize_sample_or_class_weights(x_weight, output_names, weight_type):
  """Maps `sample_weight` or `class_weight` to model outputs.

  Arguments:
      x_weight: User-provided `sample_weight` or `class_weight` argument.
      output_names: List of output names (strings) in the model.
      weight_type: A string used purely for exception printing.

  Returns:
      A list of `sample_weight` or `class_weight` where there are exactly
          one element per model output.

  Raises:
      ValueError: In case of invalid user-provided argument.
  """
  if x_weight is None or len(x_weight) == 0:  # pylint: disable=g-explicit-length-test
    return [None for _ in output_names]
  if len(output_names) == 1:
    if isinstance(x_weight, list) and len(x_weight) == 1:
      return x_weight
    if isinstance(x_weight, dict) and output_names[0] in x_weight:
      return [x_weight[output_names[0]]]
    else:
      return [x_weight]
  if isinstance(x_weight, list):
    if len(x_weight) != len(output_names):
      raise ValueError('Provided `' + weight_type + '` was a list of ' +
                       str(len(x_weight)) + ' elements, but the model has ' +
                       str(len(output_names)) + ' outputs. '
                       'You should provide one `' + weight_type + '`'
                       'array per model output.')
    return x_weight
  if isinstance(x_weight, dict):
    x_weights = []
    for name in output_names:
      x_weights.append(x_weight.get(name))
    return x_weights
  else:
    raise TypeError(
        'The model has multiple outputs, so `' + weight_type + '` '
        'should be either a list or a dict. '
        'Provided `' + weight_type + '` type not understood: ' + str(x_weight))


def _standardize_class_weights(class_weight, output_names):
  return _standardize_sample_or_class_weights(class_weight, output_names,
                                              'class_weight')


def _standardize_sample_weights(sample_weight, output_names):
  return _standardize_sample_or_class_weights(sample_weight, output_names,
                                              'sample_weight')


def _check_array_lengths(inputs, targets, weights=None):
  """Does user input validation for numpy arrays.

  Arguments:
      inputs: list of Numpy arrays of inputs.
      targets: list of Numpy arrays of targets.
      weights: list of Numpy arrays of sample weights.

  Raises:
      ValueError: in case of incorrectly formatted data.
  """

  def set_of_lengths(x):
    # return a set with the variation between
    # different shapes, with None => 0
    if x is None:
      return {0}
    else:
      return set([0 if y is None else y.shape[0] for y in x])

  set_x = set_of_lengths(inputs)
  set_y = set_of_lengths(targets)
  set_w = set_of_lengths(weights)
  if len(set_x) > 1:
    raise ValueError('All input arrays (x) should have '
                     'the same number of samples. Got array shapes: ' +
                     str([x.shape for x in inputs]))
  if len(set_y) > 1:
    raise ValueError('All target arrays (y) should have '
                     'the same number of samples. Got array shapes: ' +
                     str([y.shape for y in targets]))
  if set_x and set_y and list(set_x)[0] != list(set_y)[0]:
    raise ValueError('Input arrays should have '
                     'the same number of samples as target arrays. '
                     'Found ' + str(list(set_x)[0]) + ' input samples '
                     'and ' + str(list(set_y)[0]) + ' target samples.')
  if len(set_w) > 1:
    raise ValueError('All sample_weight arrays should have '
                     'the same number of samples. Got array shapes: ' +
                     str([w.shape for w in weights]))
  if set_y and set_w and list(set_y)[0] != list(set_w)[0]:
    raise ValueError('Sample_weight arrays should have '
                     'the same number of samples as target arrays. Got ' +
                     str(list(set_y)[0]) + ' input samples and ' +
                     str(list(set_w)[0]) + ' target samples.')


def _check_loss_and_target_compatibility(targets, loss_fns, output_shapes):
  """Does validation on the compatibility of targets and loss functions.

  This helps prevent users from using loss functions incorrectly.

  Arguments:
      targets: list of Numpy arrays of targets.
      loss_fns: list of loss functions.
      output_shapes: list of shapes of model outputs.

  Raises:
      ValueError: if a loss function or target array
          is incompatible with an output.
  """
  key_losses = {
      losses.mean_squared_error, losses.binary_crossentropy,
      losses.categorical_crossentropy
  }
  for y, loss, shape in zip(targets, loss_fns, output_shapes):
    if loss is None:
      continue
    if loss is losses.categorical_crossentropy:
      if y.shape[-1] == 1:
        raise ValueError('You are passing a target array of shape ' + str(
            y.shape) + ' while using as loss `categorical_crossentropy`. '
                         '`categorical_crossentropy` expects '
                         'targets to be binary matrices (1s and 0s) '
                         'of shape (samples, classes). '
                         'If your targets are integer classes, '
                         'you can convert them to the expected format via:\n'
                         '```\n'
                         'from keras.utils import to_categorical\n'
                         'y_binary = to_categorical(y_int)\n'
                         '```\n'
                         '\n'
                         'Alternatively, you can use the loss function '
                         '`sparse_categorical_crossentropy` instead, '
                         'which does expect integer targets.')
    if loss in key_losses:
      for target_dim, out_dim in zip(y.shape[1:], shape[1:]):
        if out_dim is not None and target_dim != out_dim:
          raise ValueError('A target array with shape ' + str(y.shape) +
                           ' was passed for an output of shape ' + str(shape) +
                           ' while using as loss `' + loss.__name__ + '`. '
                           'This loss expects '
                           'targets to have the same shape '
                           'as the output.')


def _collect_metrics(metrics, output_names):
  """Maps metric functions to model outputs.

  Arguments:
      metrics: a list or dict of metric functions.
      output_names: a list of the names (strings) of model outputs.

  Returns:
      A list (one entry per model output) of lists of metric functions.
      For instance, if the model has 2 outputs, and for the first output
      we want to compute "binary_accuracy" and "binary_crossentropy",
      and just "binary_accuracy" for the second output,
      the list would look like:
          `[[binary_accuracy, binary_crossentropy], [binary_accuracy]]`

  Raises:
      TypeError: if an incorrect type is passed for the `metrics` argument.
  """
  if not metrics:
    return [[] for _ in output_names]
  if isinstance(metrics, list):
    # we then apply all metrics to all outputs.
    return [copy.copy(metrics) for _ in output_names]
  elif isinstance(metrics, dict):
    nested_metrics = []
    for name in output_names:
      output_metrics = metrics.get(name, [])
      if not isinstance(output_metrics, list):
        output_metrics = [output_metrics]
      nested_metrics.append(output_metrics)
    return nested_metrics
  else:
    raise TypeError('Type of `metrics` argument not understood. '
                    'Expected a list or dictionary, found: ' + str(metrics))


def _batch_shuffle(index_array, batch_size):
  """Shuffles an array in a batch-wise fashion.

  Useful for shuffling HDF5 arrays
  (where one cannot access arbitrary indices).

  Arguments:
      index_array: array of indices to be shuffled.
      batch_size: integer.

  Returns:
      The `index_array` array, shuffled in a batch-wise fashion.
  """
  batch_count = int(len(index_array) / batch_size)
  # to reshape we need to be cleanly divisible by batch size
  # we stash extra items and reappend them after shuffling
  last_batch = index_array[batch_count * batch_size:]
  index_array = index_array[:batch_count * batch_size]
  index_array = index_array.reshape((batch_count, batch_size))
  np.random.shuffle(index_array)
  index_array = index_array.flatten()
  return np.append(index_array, last_batch)


def _make_batches(size, batch_size):
  """Returns a list of batch indices (tuples of indices).

  Arguments:
      size: Integer, total size of the data to slice into batches.
      batch_size: Integer, batch size.

  Returns:
      A list of tuples of array indices.
  """
  num_batches = (size + batch_size - 1) // batch_size  # round up
  return [(i * batch_size, min(size, (i + 1) * batch_size))
          for i in range(num_batches)]


def _slice_arrays(arrays, start=None, stop=None):
  """Slice an array or list of arrays.

  This takes an array-like, or a list of
  array-likes, and outputs:
      - arrays[start:stop] if `arrays` is an array-like
      - [x[start:stop] for x in arrays] if `arrays` is a list

  Can also work on list/array of indices: `_slice_arrays(x, indices)`

  Arguments:
      arrays: Single array or list of arrays.
      start: can be an integer index (start index)
          or a list/array of indices
      stop: integer (stop index); should be None if
          `start` was a list.

  Returns:
      A slice of the array(s).
  """
  if arrays is None:
    return [None]
  elif isinstance(arrays, list):
    if hasattr(start, '__len__'):
      # hdf5 datasets only support list objects as indices
      if hasattr(start, 'shape'):
        start = start.tolist()
      return [None if x is None else x[start] for x in arrays]
    else:
      return [None if x is None else x[start:stop] for x in arrays]
  else:
    if hasattr(start, '__len__'):
      if hasattr(start, 'shape'):
        start = start.tolist()
      return arrays[start]
    elif hasattr(start, '__getitem__'):
      return arrays[start:stop]
    else:
      return [None]


def _weighted_masked_objective(fn):
  """Adds support for masking and sample-weighting to an objective function.

  It transforms an objective function `fn(y_true, y_pred)`
  into a sample-weighted, cost-masked objective function
  `fn(y_true, y_pred, weights, mask)`.

  Arguments:
      fn: The objective function to wrap,
          with signature `fn(y_true, y_pred)`.

  Returns:
      A function with signature `fn(y_true, y_pred, weights, mask)`.
  """
  if fn is None:
    return None

  def weighted(y_true, y_pred, weights, mask=None):
    """Wrapper function.

    Arguments:
        y_true: `y_true` argument of `fn`.
        y_pred: `y_pred` argument of `fn`.
        weights: Weights tensor.
        mask: Mask tensor.

    Returns:
        Scalar tensor.
    """
    # score_array has ndim >= 2
    score_array = fn(y_true, y_pred)
    if mask is not None:
      # Cast the mask to floatX to avoid float64 upcasting in theano
      mask = K.cast(mask, K.floatx())
      # mask should have the same shape as score_array
      score_array *= mask
      #  the loss per batch should be proportional
      #  to the number of unmasked samples.
      score_array /= K.mean(mask)

    # apply sample weighting
    if weights is not None:
      # reduce score_array to same ndim as weight array
      ndim = K.ndim(score_array)
      weight_ndim = K.ndim(weights)
      score_array = K.mean(score_array, axis=list(range(weight_ndim, ndim)))
      score_array *= weights
      score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
    return K.mean(score_array)

  return weighted


def _standardize_weights(y,
                         sample_weight=None,
                         class_weight=None,
                         sample_weight_mode=None):
  """Performs sample weight validation and standardization.

  Everything gets normalized to a single sample-wise (or timestep-wise)
  weight array.

  Arguments:
      y: Numpy array of model targets to be weighted.
      sample_weight: User-provided `sample_weight` argument.
      class_weight: User-provided `class_weight` argument.
      sample_weight_mode: One of `None` or `"temporal"`.
          `"temporal"` indicated that we expect 2D weight data
          that will be applied to the last 2 dimensions of
          the targets (i.e. we are weighting timesteps, not samples).

  Returns:
      A numpy array of target weights, one entry per sample to weight.

  Raises:
      ValueError: In case of invalid user-provided arguments.
  """
  if sample_weight_mode is not None:
    if sample_weight_mode != 'temporal':
      raise ValueError('"sample_weight_mode '
                       'should be None or "temporal". '
                       'Found: ' + str(sample_weight_mode))
    if len(y.shape) < 3:
      raise ValueError('Found a sample_weight array for '
                       'an input with shape ' + str(y.shape) + '. '
                       'Timestep-wise sample weighting (use of '
                       'sample_weight_mode="temporal") is restricted to '
                       'outputs that are at least 3D, i.e. that have '
                       'a time dimension.')
    if sample_weight is not None and len(sample_weight.shape) != 2:
      raise ValueError('Found a sample_weight array with shape ' +
                       str(sample_weight.shape) + '. '
                       'In order to use timestep-wise sample weighting, '
                       'you should pass a 2D sample_weight array.')
  else:
    if sample_weight is not None and len(sample_weight.shape) != 1:
      raise ValueError('Found a sample_weight array with shape ' +
                       str(sample_weight.shape) + '. '
                       'In order to use timestep-wise sample weights, '
                       'you should specify '
                       'sample_weight_mode="temporal" '
                       'in compile(). If you just mean to use '
                       'sample-wise weights, make sure your '
                       'sample_weight array is 1D.')

  if sample_weight is not None:
    if len(sample_weight.shape) > len(y.shape):
      raise ValueError(
          'Found a sample_weight with shape' + str(sample_weight.shape) + '.'
          'Expected sample_weight with rank '
          'less than or equal to ' + str(len(y.shape)))

    if y.shape[:sample_weight.ndim] != sample_weight.shape:
      raise ValueError(
          'Found a sample_weight array with shape ' + str(sample_weight.shape) +
          ' for an input with shape ' + str(y.shape) + '. '
          'sample_weight cannot be broadcast.')
    return sample_weight
  elif isinstance(class_weight, dict):
    if len(y.shape) > 2:
      raise ValueError('`class_weight` not supported for '
                       '3+ dimensional targets.')
    if y.shape[1] > 1:
      y_classes = y.argmax(axis=1)
    elif y.shape[1] == 1:
      y_classes = np.reshape(y, y.shape[0])
    else:
      y_classes = y

    weights = np.asarray(
        [class_weight[cls] for cls in y_classes if cls in class_weight])

    if len(weights) != len(y_classes):
      # subtract the sets to pick all missing classes
      existing_classes = set(y_classes)
      existing_class_weight = set(class_weight.keys())
      raise ValueError('`class_weight` must contain all classes in the data.'
                       ' The classes %s exist in the data but not in '
                       '`class_weight`.' %
                       (existing_classes - existing_class_weight))
    return weights
  else:
    if sample_weight_mode is None:
      return np.ones((y.shape[0],), dtype=K.floatx())
    else:
      return np.ones((y.shape[0], y.shape[1]), dtype=K.floatx())


class Model(Network):
  """The `Model` class adds training & evaluation routines to a `Network`.
  """

  def compile(self,
              optimizer,
              loss,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              **kwargs):
    """Configures the model for training.

    Arguments:
        optimizer: String (name of optimizer) or optimizer instance.
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
        loss_weights: Optional list or dictionary specifying scalar
            coefficients (Python floats) to weight the loss contributions
            of different model outputs.
            The loss value that will be minimized by the model
            will then be the *weighted sum* of all individual losses,
            weighted by the `loss_weights` coefficients.
            If a list, it is expected to have a 1:1 mapping
            to the model's outputs. If a tensor, it is expected to map
            output names (strings) to scalar coefficients.
        sample_weight_mode: If you need to do timestep-wise
            sample weighting (2D weights), set this to `"temporal"`.
            `None` defaults to sample-wise weights (1D).
            If the model has multiple outputs, you can use a different
            `sample_weight_mode` on each output by passing a
            dictionary or a list of modes.
        weighted_metrics: List of metrics to be evaluated and weighted
            by sample_weight or class_weight during training and testing.
        target_tensors: By default, Keras will create placeholders for the
            model's target, which will be fed with the target data during
            training. If instead you would like to use your own
            target tensors (in turn, Keras will not expect external
            Numpy data for these targets at training time), you
            can specify them via the `target_tensors` argument. It can be
            a single tensor (for a single-output model), a list of tensors,
            or a dict mapping output names to target tensors.
        **kwargs: These arguments are passed to `tf.Session.run`.

    Raises:
        ValueError: In case of invalid arguments for
            `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
    """
    loss = loss or {}
    self.optimizer = optimizers.get(optimizer)
    self.loss = loss
    self.loss_weights = loss_weights
    self.sample_weight_mode = sample_weight_mode

    # Prepare loss functions.
    if isinstance(loss, dict):
      for name in loss:
        if name not in self.output_names:
          raise ValueError(
              'Unknown entry in loss '
              'dictionary: "' + name + '". '
              'Only expected the following keys: ' + str(self.output_names))
      loss_functions = []
      for name in self.output_names:
        if name not in loss:
          logging.warning(
              'Output "' + name + '" missing from loss dictionary. '
              'We assume this was done on purpose, '
              'and we will not be expecting '
              'any data to be passed to "' + name + '" during training.')
        loss_functions.append(losses.get(loss.get(name)))
    elif isinstance(loss, list):
      if len(loss) != len(self.outputs):
        raise ValueError('When passing a list as loss, '
                         'it should have one entry per model outputs. '
                         'The model has ' + str(len(self.outputs)) +
                         ' outputs, but you passed loss=' + str(loss))
      loss_functions = [losses.get(l) for l in loss]
    else:
      loss_function = losses.get(loss)
      loss_functions = [loss_function for _ in range(len(self.outputs))]
    self.loss_functions = loss_functions
    weighted_losses = [_weighted_masked_objective(fn) for fn in loss_functions]
    skip_target_indices = []
    skip_target_weighing_indices = []
    self._feed_outputs = []
    self._feed_output_names = []
    self._feed_output_shapes = []
    self._feed_loss_fns = []
    for i in range(len(weighted_losses)):
      if weighted_losses[i] is None:
        skip_target_indices.append(i)
        skip_target_weighing_indices.append(i)

    # Prepare output masks.
    masks = self.compute_mask(self.inputs, mask=None)
    if masks is None:
      masks = [None for _ in self.outputs]
    if not isinstance(masks, list):
      masks = [masks]

    # Prepare loss weights.
    if loss_weights is None:
      loss_weights_list = [1. for _ in range(len(self.outputs))]
    elif isinstance(loss_weights, dict):
      for name in loss_weights:
        if name not in self.output_names:
          raise ValueError(
              'Unknown entry in loss_weights '
              'dictionary: "' + name + '". '
              'Only expected the following keys: ' + str(self.output_names))
      loss_weights_list = []
      for name in self.output_names:
        loss_weights_list.append(loss_weights.get(name, 1.))
    elif isinstance(loss_weights, list):
      if len(loss_weights) != len(self.outputs):
        raise ValueError(
            'When passing a list as loss_weights, '
            'it should have one entry per model output. '
            'The model has ' + str(len(self.outputs)) +
            ' outputs, but you passed loss_weights=' + str(loss_weights))
      loss_weights_list = loss_weights
    else:
      raise TypeError('Could not interpret loss_weights argument: ' +
                      str(loss_weights) + ' - expected a list of dicts.')

    # Prepare targets of model.
    self.targets = []
    self._feed_targets = []
    if target_tensors is not None:
      if isinstance(target_tensors, list):
        if len(target_tensors) != len(self.outputs):
          raise ValueError(
              'When passing a list as `target_tensors`, '
              'it should have one entry per model output. '
              'The model has ' + str(len(self.outputs)) +
              ' outputs, but you passed target_tensors=' + str(target_tensors))
      elif isinstance(target_tensors, dict):
        for name in target_tensors:
          if name not in self.output_names:
            raise ValueError(
                'Unknown entry in `target_tensors` '
                'dictionary: "' + name + '". '
                'Only expected the following keys: ' + str(self.output_names))
        tmp_target_tensors = []
        for name in self.output_names:
          tmp_target_tensors.append(target_tensors.get(name, None))
        target_tensors = tmp_target_tensors
      else:
        raise TypeError('Expected `target_tensors` to be '
                        'a list or dict, but got:', target_tensors)
    for i in range(len(self.outputs)):
      if i in skip_target_indices:
        self.targets.append(None)
      else:
        shape = self._internal_output_shapes[i]
        name = self.output_names[i]
        if target_tensors is not None:
          target = target_tensors[i]
        else:
          target = None
        if target is None or K.is_placeholder(target):
          if target is None:
            target = K.placeholder(
                ndim=len(shape),
                name=name + '_target',
                sparse=K.is_sparse(self.outputs[i]),
                dtype=K.dtype(self.outputs[i]))
          self._feed_targets.append(target)
          self._feed_outputs.append(self.outputs[i])
          self._feed_output_names.append(name)
          self._feed_output_shapes.append(shape)
          self._feed_loss_fns.append(self.loss_functions[i])
        else:
          skip_target_weighing_indices.append(i)
        self.targets.append(target)

    # Prepare sample weights.
    sample_weights = []
    sample_weight_modes = []
    if isinstance(sample_weight_mode, dict):
      for name in sample_weight_mode:
        if name not in self.output_names:
          raise ValueError(
              'Unknown entry in '
              'sample_weight_mode dictionary: "' + name + '". '
              'Only expected the following keys: ' + str(self.output_names))
      for i, name in enumerate(self.output_names):
        if i in skip_target_weighing_indices:
          weight = None
          sample_weight_modes.append(None)
        else:
          if name not in sample_weight_mode:
            raise ValueError(
                'Output "' + name + '" missing from sample_weight_modes '
                'dictionary')
          if sample_weight_mode.get(name) == 'temporal':
            weight = K.placeholder(ndim=2, name=name + '_sample_weights')
            sample_weight_modes.append('temporal')
          else:
            weight = K.placeholder(ndim=1, name=name + '_sample_weights')
            sample_weight_modes.append(None)
        sample_weights.append(weight)
    elif isinstance(sample_weight_mode, list):
      if len(sample_weight_mode) != len(self.outputs):
        raise ValueError('When passing a list as sample_weight_mode, '
                         'it should have one entry per model output. '
                         'The model has ' + str(len(self.outputs)) +
                         ' outputs, but you passed '
                         'sample_weight_mode=' + str(sample_weight_mode))
      for i in range(len(self.output_names)):
        if i in skip_target_weighing_indices:
          weight = None
          sample_weight_modes.append(None)
        else:
          mode = sample_weight_mode[i]
          name = self.output_names[i]
          if mode == 'temporal':
            weight = K.placeholder(ndim=2, name=name + '_sample_weights')
            sample_weight_modes.append('temporal')
          else:
            weight = K.placeholder(ndim=1, name=name + '_sample_weights')
            sample_weight_modes.append(None)
        sample_weights.append(weight)
    else:
      for i, name in enumerate(self.output_names):
        if i in skip_target_weighing_indices:
          sample_weight_modes.append(None)
          sample_weights.append(None)
        else:
          if sample_weight_mode == 'temporal':
            sample_weights.append(
                K.placeholder(ndim=2, name=name + '_sample_weights'))
            sample_weight_modes.append('temporal')
          else:
            sample_weights.append(
                K.placeholder(ndim=1, name=name + '_sample_weights'))
            sample_weight_modes.append(None)
    self.sample_weight_modes = sample_weight_modes
    self._feed_sample_weight_modes = []
    for i in range(len(self.outputs)):
      if i not in skip_target_weighing_indices:
        self._feed_sample_weight_modes.append(self.sample_weight_modes[i])

    # Prepare metrics.
    self.metrics = metrics
    self.weighted_metrics = weighted_metrics
    self.metrics_names = ['loss']
    self.metrics_tensors = []

    # Compute total loss.
    total_loss = None
    with K.name_scope('loss'):
      for i in range(len(self.outputs)):
        if i in skip_target_indices:
          continue
        y_true = self.targets[i]
        y_pred = self.outputs[i]
        weighted_loss = weighted_losses[i]
        sample_weight = sample_weights[i]
        mask = masks[i]
        loss_weight = loss_weights_list[i]
        with K.name_scope(self.output_names[i] + '_loss'):
          output_loss = weighted_loss(y_true, y_pred, sample_weight, mask)
        if len(self.outputs) > 1:
          self.metrics_tensors.append(output_loss)
          self.metrics_names.append(self.output_names[i] + '_loss')
        if total_loss is None:
          total_loss = loss_weight * output_loss
        else:
          total_loss += loss_weight * output_loss
      if total_loss is None:
        if not self.losses:
          raise ValueError('The model cannot be compiled '
                           'because it has no loss to optimize.')
        else:
          total_loss = 0.

      # Add regularization penalties
      # and other layer-specific losses.
      for loss_tensor in self.losses:
        total_loss += loss_tensor

    # List of same size as output_names.
    # contains tuples (metrics for output, names of metrics).
    nested_metrics = _collect_metrics(metrics, self.output_names)
    nested_weighted_metrics = _collect_metrics(weighted_metrics,
                                               self.output_names)

    def append_metric(layer_index, metric_name, metric_tensor):
      """Helper function used in loop below."""
      if len(self.output_names) > 1:
        metric_name = self.output_names[layer_index] + '_' + metric_name
      self.metrics_names.append(metric_name)
      self.metrics_tensors.append(metric_tensor)

    with K.name_scope('metrics'):
      for i in range(len(self.outputs)):
        if i in skip_target_indices:
          continue

        y_true = self.targets[i]
        y_pred = self.outputs[i]
        weights = sample_weights[i]
        output_metrics = nested_metrics[i]
        output_weighted_metrics = nested_weighted_metrics[i]

        def handle_metrics(metrics, weights=None):
          metric_name_prefix = 'weighted_' if weights is not None else ''

          for metric in metrics:
            if metric in ('accuracy', 'acc', 'crossentropy', 'ce'):
              # custom handling of accuracy/crossentropy
              # (because of class mode duality)
              output_shape = self._internal_output_shapes[i]
              if (output_shape[-1] == 1 or
                  self.loss_functions[i] == losses.binary_crossentropy):
                # case: binary accuracy/crossentropy
                if metric in ('accuracy', 'acc'):
                  acc_fn = metrics_module.binary_accuracy
                elif metric in ('crossentropy', 'ce'):
                  acc_fn = metrics_module.binary_crossentropy
              elif self.loss_functions[
                  i] == losses.sparse_categorical_crossentropy:
                # case: categorical accuracy/crossentropy with sparse targets
                if metric in ('accuracy', 'acc'):
                  acc_fn = metrics_module.sparse_categorical_accuracy
                elif metric in ('crossentropy', 'ce'):
                  acc_fn = metrics_module.sparse_categorical_crossentropy
              else:
                # case: categorical accuracy/crossentropy
                if metric in ('accuracy', 'acc'):
                  acc_fn = metrics_module.categorical_accuracy
                elif metric in ('crossentropy', 'ce'):
                  acc_fn = metrics_module.categorical_crossentropy
              if metric in ('accuracy', 'acc'):
                suffix = 'acc'
              elif metric in ('crossentropy', 'ce'):
                suffix = 'ce'
              weighted_metric_fn = _weighted_masked_objective(acc_fn)
              metric_name = metric_name_prefix + suffix
            else:
              metric_fn = metrics_module.get(metric)
              weighted_metric_fn = _weighted_masked_objective(metric_fn)
              metric_name = metric_name_prefix + metric_fn.__name__

            with K.name_scope(metric_name):
              metric_result = weighted_metric_fn(
                  y_true, y_pred, weights=weights, mask=masks[i])
            append_metric(i, metric_name, metric_result)

        handle_metrics(output_metrics)
        handle_metrics(output_weighted_metrics, weights=weights)

    # Prepare gradient updates and state updates.
    self.total_loss = total_loss
    self.sample_weights = sample_weights
    self._feed_sample_weights = []
    for i in range(len(self.sample_weights)):
      if i not in skip_target_weighing_indices:
        self._feed_sample_weights.append(sample_weights[i])

    # Functions for train, test and predict will
    # be compiled lazily when required.
    # This saves time when the user is not using all functions.
    self._function_kwargs = kwargs

    self.train_function = None
    self.test_function = None
    self.predict_function = None

    # Collected trainable weights, sorted in topological order.
    trainable_weights = self.trainable_weights
    self._collected_trainable_weights = trainable_weights

  def _check_trainable_weights_consistency(self):
    """Check trainable weights count consistency.

    This will raise a warning if `trainable_weights` and
    `_collected_trainable_weights` are inconsistent (i.e. have different
    number of parameters).
    Inconsistency will typically arise when one modifies `model.trainable`
    without calling `model.compile` again.
    """
    if not hasattr(self, '_collected_trainable_weights'):
      return

    if len(self.trainable_weights) != len(self._collected_trainable_weights):
      logging.warning(
          UserWarning(
              'Discrepancy between trainable weights and collected trainable'
              ' weights, did you set `model.trainable` without calling'
              ' `model.compile` after ?'))

  def _make_train_function(self):
    if not hasattr(self, 'train_function'):
      raise RuntimeError('You must compile your model before using it.')
    self._check_trainable_weights_consistency()
    if self.train_function is None:
      inputs = (self._feed_inputs +
                self._feed_targets +
                self._feed_sample_weights)
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs += [K.learning_phase()]

      with K.name_scope('training'):
        with K.name_scope(self.optimizer.__class__.__name__):
          training_updates = self.optimizer.get_updates(
              params=self._collected_trainable_weights, loss=self.total_loss)
        updates = self.updates + training_updates
        # Gets loss and metrics. Updates weights at each call.
        self.train_function = K.function(
            inputs, [self.total_loss] + self.metrics_tensors,
            updates=updates,
            name='train_function',
            **self._function_kwargs)

  def _make_test_function(self):
    if not hasattr(self, 'test_function'):
      raise RuntimeError('You must compile your model before using it.')
    if self.test_function is None:
      inputs = (self._feed_inputs +
                self._feed_targets +
                self._feed_sample_weights)
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs += [K.learning_phase()]
      # Return loss and metrics, no gradient updates.
      # Does update the network states.
      self.test_function = K.function(
          inputs, [self.total_loss] + self.metrics_tensors,
          updates=self.state_updates,
          name='test_function',
          **self._function_kwargs)

  def _make_predict_function(self):
    if not hasattr(self, 'predict_function'):
      self.predict_function = None
    if self.predict_function is None:
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs = self._feed_inputs + [K.learning_phase()]
      else:
        inputs = self._feed_inputs
      # Gets network outputs. Does not update weights.
      # Does update the network states.
      kwargs = getattr(self, '_function_kwargs', {})
      self.predict_function = K.function(
          inputs,
          self.outputs,
          updates=self.state_updates,
          name='predict_function',
          **kwargs)

  def _check_num_samples(self,
                         ins,
                         batch_size=None,
                         steps=None,
                         steps_name='steps'):
    """Determine the number of samples provided for training and evaluation.

    The number of samples is not defined when running with `steps`,
    in which case the number of samples is set to `None`.

    Arguments:
        ins: List of tensors to be fed to the Keras function.
        batch_size: Integer batch size or `None` if not defined.
        steps: Total number of steps (batches of samples)
            before declaring `_predict_loop` finished.
            Ignored with the default value of `None`.
        steps_name: The public API's parameter name for `steps`.

    Raises:
        ValueError: when `steps` is `None` and the attribute `ins.shape`
        does not exist. Also raises ValueError when `steps` is not `None`
        and `batch_size` is not `None` because they are mutually
        exclusive.

    Returns:
        When steps is `None`, returns the number of samples to be
        processed based on the size of the first dimension of the
        first input numpy array. When steps is not `None` and
        `batch_size` is `None`, returns `None`.

    Raises:
        ValueError: In case of invalid arguments.
    """
    if steps is not None:
      num_samples = None
      if batch_size is not None:
        raise ValueError(
            'If ' + steps_name + ' is set, the `batch_size` must be None.')
    elif ins and hasattr(ins[0], 'shape'):
      num_samples = ins[0].shape[0]
    else:
      raise ValueError(
          'Either the input data should have '
          'a defined shape, or ' + steps_name + ' should be specified.')
    return num_samples

  def _fit_loop(self,
                f,
                ins,
                out_labels=None,
                batch_size=None,
                epochs=100,
                verbose=1,
                callbacks=None,
                val_f=None,
                val_ins=None,
                shuffle=True,
                callback_metrics=None,
                initial_epoch=0,
                steps_per_epoch=None,
                validation_steps=None):
    """Abstract fit function for `f(ins)`.

    Assume that f returns a list, labeled by out_labels.

    Arguments:
        f: Keras function returning a list of tensors
        ins: List of tensors to be fed to `f`
        out_labels: List of strings, display names of
            the outputs of `f`
        batch_size: Integer batch size or None if unknown.
        epochs: Number of times to iterate over the data
        verbose: Verbosity mode, 0, 1 or 2
        callbacks: List of callbacks to be called during training
        val_f: Keras function to call for validation
        val_ins: List of tensors to be fed to `val_f`
        shuffle: Whether to shuffle the data at the beginning of each epoch
        callback_metrics: List of strings, the display names of the metrics
            passed to the callbacks. They should be the
            concatenation of list the display names of the outputs of
             `f` and the list of display names of the outputs of `f_val`.
        initial_epoch: Epoch at which to start training
            (useful for resuming a previous training run)
        steps_per_epoch: Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. Ignored with the default value of `None`.
        validation_steps: Number of steps to run validation for
            (only if doing validation from data tensors).
            Ignored with the default value of `None`.

    Returns:
        `History` object.

    Raises:
        ValueError: in case of invalid arguments.
    """
    do_validation = False
    if val_f and val_ins:
      do_validation = True
      if verbose and ins and hasattr(ins[0], 'shape') and hasattr(
          val_ins[0], 'shape'):
        print('Train on %d samples, validate on %d samples' %
              (ins[0].shape[0], val_ins[0].shape[0]))
    if validation_steps:
      do_validation = True
      if steps_per_epoch is None:
        raise ValueError('Can only use `validation_steps` '
                         'when doing step-wise '
                         'training, i.e. `steps_per_epoch` '
                         'must be set.')

    num_train_samples = self._check_num_samples(
        ins, batch_size, steps_per_epoch, 'steps_per_epoch')
    if num_train_samples is not None:
      index_array = np.arange(num_train_samples)

    self.history = cbks.History()
    callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
    if verbose:
      if steps_per_epoch is not None:
        count_mode = 'steps'
      else:
        count_mode = 'samples'
      callbacks += [cbks.ProgbarLogger(count_mode)]
    callbacks = cbks.CallbackList(callbacks)
    out_labels = out_labels or []

    # it's possible to callback a different model than self
    # (used by Sequential models)
    if hasattr(self, 'callback_model') and self.callback_model:
      callback_model = self.callback_model
    else:
      callback_model = self

    callbacks.set_model(callback_model)
    callbacks.set_params({
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps_per_epoch,
        'samples': num_train_samples,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics or [],
    })
    callbacks.on_train_begin()
    callback_model.stop_training = False
    for cbk in callbacks:
      cbk.validation_data = val_ins

    # To prevent a slowdown, we find beforehand the arrays that need conversion.
    feed = self._feed_inputs + self._feed_targets + self._feed_sample_weights
    indices_for_conversion_to_dense = []
    for i in range(len(feed)):
      if issparse is not None and issparse(ins[i]) and not K.is_sparse(feed[i]):
        indices_for_conversion_to_dense.append(i)

    for epoch in range(initial_epoch, epochs):
      callbacks.on_epoch_begin(epoch)
      epoch_logs = {}
      if steps_per_epoch is not None:
        for step_index in range(steps_per_epoch):
          batch_logs = {}
          batch_logs['batch'] = step_index
          batch_logs['size'] = 1
          callbacks.on_batch_begin(step_index, batch_logs)
          outs = f(ins)

          if not isinstance(outs, list):
            outs = [outs]
          for l, o in zip(out_labels, outs):
            batch_logs[l] = o

          callbacks.on_batch_end(step_index, batch_logs)
          if callback_model.stop_training:
            break

        if do_validation:
          val_outs = self._test_loop(
              val_f,
              val_ins,
              batch_size=batch_size,
              steps=validation_steps,
              verbose=0)
          if not isinstance(val_outs, list):
            val_outs = [val_outs]
          # Same labels assumed.
          for l, o in zip(out_labels, val_outs):
            epoch_logs['val_' + l] = o
      else:
        if shuffle == 'batch':
          index_array = _batch_shuffle(index_array, batch_size)
        elif shuffle:
          np.random.shuffle(index_array)

        batches = _make_batches(num_train_samples, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
          batch_ids = index_array[batch_start:batch_end]
          try:
            if isinstance(ins[-1], float):
              # Do not slice the training phase flag.
              ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
            else:
              ins_batch = _slice_arrays(ins, batch_ids)
          except TypeError:
            raise TypeError('TypeError while preparing batch. '
                            'If using HDF5 input data, '
                            'pass shuffle="batch".')
          batch_logs = {}
          batch_logs['batch'] = batch_index
          batch_logs['size'] = len(batch_ids)
          callbacks.on_batch_begin(batch_index, batch_logs)
          for i in indices_for_conversion_to_dense:
            ins_batch[i] = ins_batch[i].toarray()

          outs = f(ins_batch)
          if not isinstance(outs, list):
            outs = [outs]
          for l, o in zip(out_labels, outs):
            batch_logs[l] = o

          callbacks.on_batch_end(batch_index, batch_logs)
          if callback_model.stop_training:
            break

          if batch_index == len(batches) - 1:  # Last batch.
            if do_validation:
              val_outs = self._test_loop(
                  val_f, val_ins, batch_size=batch_size, verbose=0)
              if not isinstance(val_outs, list):
                val_outs = [val_outs]
              # Same labels assumed.
              for l, o in zip(out_labels, val_outs):
                epoch_logs['val_' + l] = o
      callbacks.on_epoch_end(epoch, epoch_logs)
      if callback_model.stop_training:
        break
    callbacks.on_train_end()
    return self.history

  def _predict_loop(self, f, ins, batch_size=32, verbose=0, steps=None):
    """Abstract method to loop over some data in batches.

    Arguments:
        f: Keras function returning a list of tensors.
        ins: list of tensors to be fed to `f`.
        batch_size: integer batch size.
        verbose: verbosity mode.
        steps: Total number of steps (batches of samples)
            before declaring `_predict_loop` finished.
            Ignored with the default value of `None`.

    Returns:
        Array of predictions (if the model has a single output)
        or list of arrays of predictions
        (if the model has multiple outputs).
    """
    num_samples = self._check_num_samples(ins, batch_size, steps, 'steps')
    if verbose == 1:
      if steps is not None:
        progbar = Progbar(target=steps)
      else:
        progbar = Progbar(target=num_samples)

    indices_for_conversion_to_dense = []
    for i in range(len(self._feed_inputs)):
      if (issparse is not None and issparse(ins[i]) and
          not K.is_sparse(self._feed_inputs[i])):
        indices_for_conversion_to_dense.append(i)

    if steps is not None:
      # Step-based predictions.
      # Since we do not know how many samples
      # we will see, we cannot pre-allocate
      # the returned Numpy arrays.
      # Instead, we store one array per batch seen
      # and concatenate them upon returning.
      unconcatenated_outs = []
      for step in range(steps):
        batch_outs = f(ins)
        if not isinstance(batch_outs, list):
          batch_outs = [batch_outs]
        if step == 0:
          for batch_out in batch_outs:
            unconcatenated_outs.append([])
        for i, batch_out in enumerate(batch_outs):
          unconcatenated_outs[i].append(batch_out)
        if verbose == 1:
          progbar.update(step + 1)
      if len(unconcatenated_outs) == 1:
        return np.concatenate(unconcatenated_outs[0], axis=0)
      return [
          np.concatenate(unconcatenated_outs[i], axis=0)
          for i in range(len(unconcatenated_outs))
      ]
    else:
      # Sample-based predictions.
      outs = []
      batches = _make_batches(num_samples, batch_size)
      index_array = np.arange(num_samples)
      for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        if ins and isinstance(ins[-1], float):
          # Do not slice the training phase flag.
          ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
        else:
          ins_batch = _slice_arrays(ins, batch_ids)
        for i in indices_for_conversion_to_dense:
          ins_batch[i] = ins_batch[i].toarray()

        batch_outs = f(ins_batch)
        if not isinstance(batch_outs, list):
          batch_outs = [batch_outs]
        if batch_index == 0:
          # Pre-allocate the results arrays.
          for batch_out in batch_outs:
            shape = (num_samples,) + batch_out.shape[1:]
            outs.append(np.zeros(shape, dtype=batch_out.dtype))
        for i, batch_out in enumerate(batch_outs):
          outs[i][batch_start:batch_end] = batch_out
        if verbose == 1:
          progbar.update(batch_end)
      if len(outs) == 1:
        return outs[0]
      return outs

  def _test_loop(self, f, ins, batch_size=None, verbose=0, steps=None):
    """Abstract method to loop over some data in batches.

    Arguments:
        f: Keras function returning a list of tensors.
        ins: list of tensors to be fed to `f`.
        batch_size: integer batch size or `None`.
        verbose: verbosity mode.
        steps: Total number of steps (batches of samples)
            before declaring predictions finished.
            Ignored with the default value of `None`.

    Returns:
        Scalar loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
    """
    num_samples = self._check_num_samples(ins, batch_size, steps, 'steps')
    outs = []
    if verbose == 1:
      if steps is not None:
        progbar = Progbar(target=steps)
      else:
        progbar = Progbar(target=num_samples)

    # To prevent a slowdown, we find beforehand the arrays that need conversion.
    feed = self._feed_inputs + self._feed_targets + self._feed_sample_weights
    indices_for_conversion_to_dense = []
    for i in range(len(feed)):
      if issparse is not None and issparse(ins[i]) and not K.is_sparse(feed[i]):
        indices_for_conversion_to_dense.append(i)

    if steps is not None:
      for step in range(steps):
        batch_outs = f(ins)
        if isinstance(batch_outs, list):
          if step == 0:
            for _ in enumerate(batch_outs):
              outs.append(0.)
          for i, batch_out in enumerate(batch_outs):
            outs[i] += batch_out
        else:
          if step == 0:
            outs.append(0.)
          outs[0] += batch_outs
        if verbose == 1:
          progbar.update(step + 1)
      for i in range(len(outs)):
        outs[i] /= steps
    else:
      batches = _make_batches(num_samples, batch_size)
      index_array = np.arange(num_samples)
      for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        if isinstance(ins[-1], float):
          # Do not slice the training phase flag.
          ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
        else:
          ins_batch = _slice_arrays(ins, batch_ids)
        for i in indices_for_conversion_to_dense:
          ins_batch[i] = ins_batch[i].toarray()

        batch_outs = f(ins_batch)
        if isinstance(batch_outs, list):
          if batch_index == 0:
            for batch_out in enumerate(batch_outs):
              outs.append(0.)
          for i, batch_out in enumerate(batch_outs):
            outs[i] += batch_out * len(batch_ids)
        else:
          if batch_index == 0:
            outs.append(0.)
          outs[0] += batch_outs * len(batch_ids)

        if verbose == 1:
          progbar.update(batch_end)
      for i in range(len(outs)):
        outs[i] /= num_samples
    if len(outs) == 1:
      return outs[0]
    return outs

  def _standardize_user_data(self,
                             x,
                             y,
                             sample_weight=None,
                             class_weight=None,
                             check_batch_axis=True,
                             batch_size=None):
    if not hasattr(self, 'optimizer'):
      raise RuntimeError('You must compile a model before '
                         'training/testing. '
                         'Use `model.compile(optimizer, loss)`.')

    output_shapes = []
    for output_shape, loss_fn in zip(self._feed_output_shapes,
                                     self._feed_loss_fns):
      if loss_fn is losses.sparse_categorical_crossentropy:
        output_shapes.append(output_shape[:-1] + (1,))
      elif (not hasattr(loss_fn, '__name__') or
            getattr(losses, loss_fn.__name__, None) is None):
        # If `loss_fn` is not a function (e.g. callable class)
        # or if it not in the `losses` module, then
        # it is a user-defined loss and we make no assumptions
        # about it.
        output_shapes.append(None)
      else:
        output_shapes.append(output_shape)
    x = _standardize_input_data(
        x,
        self._feed_input_names,
        self._feed_input_shapes,
        check_batch_axis=False,
        exception_prefix='input')
    y = _standardize_input_data(
        y,
        self._feed_output_names,
        output_shapes,
        check_batch_axis=False,
        exception_prefix='target')
    sample_weights = _standardize_sample_weights(sample_weight,
                                                 self._feed_output_names)
    class_weights = _standardize_class_weights(class_weight,
                                               self._feed_output_names)
    sample_weights = [
        _standardize_weights(ref, sw, cw, mode)
        for (ref, sw, cw, mode) in zip(y, sample_weights, class_weights,
                                       self._feed_sample_weight_modes)
    ]
    _check_array_lengths(x, y, sample_weights)
    _check_loss_and_target_compatibility(y, self._feed_loss_fns,
                                         self._feed_output_shapes)
    if self.stateful and batch_size:
      if x[0].shape[0] % batch_size != 0:
        raise ValueError('In a stateful network, '
                         'you should only pass inputs with '
                         'a number of samples that can be '
                         'divided by the batch size. Found: ' +
                         str(x[0].shape[0]) + ' samples')
    return x, y, sample_weights

  def _get_deduped_metrics_names(self):
    out_labels = self.metrics_names

    # Rename duplicated metrics name
    # (can happen with an output layer shared among multiple dataflows).
    deduped_out_labels = []
    for i, label in enumerate(out_labels):
      new_label = label
      if out_labels.count(label) > 1:
        dup_idx = out_labels[:i].count(label)
        new_label += '_' + str(dup_idx + 1)
      deduped_out_labels.append(new_label)
    return deduped_out_labels

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
    """Trains the model for a fixed number of epochs (iterations on a dataset).

    Arguments:
        x: Numpy array of training data (if the model has a single input),
            or list of Numpy arrays (if the model has multiple inputs).
            If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
            `x` can be `None` (default) if feeding from
            TensorFlow data tensors.
        y: Numpy array of target (label) data
            (if the model has a single output),
            or list of Numpy arrays (if the model has multiple outputs).
            If output layers in the model are named, you can also pass a
            dictionary mapping output names to Numpy arrays.
            `y` can be `None` (default) if feeding from
            TensorFlow data tensors.
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        verbose: Integer. 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
            See [callbacks](/callbacks).
        validation_split: Float between 0 and 1.
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
            `validation_data` will override `validation_split`.
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
        initial_epoch: Integer.
            Epoch at which to start training
            (useful for resuming a previous training run).
        steps_per_epoch: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of samples in your dataset divided by
            the batch size, or 1 if that cannot be determined.
        validation_steps: Only relevant if `steps_per_epoch`
            is specified. Total number of steps (batches of samples)
            to validate before stopping.
        **kwargs: Used for backwards compatibility.

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
    # Backwards compatibility
    if batch_size is None and steps_per_epoch is None:
      batch_size = 32
    # Legacy support
    if 'nb_epoch' in kwargs:
      logging.warning(
          'The `nb_epoch` argument in `fit` '
          'has been renamed `epochs`.')
      epochs = kwargs.pop('nb_epoch')
    if kwargs:
      raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))
    if x is None and y is None and steps_per_epoch is None:
      raise ValueError('If fitting from data tensors, '
                       'you should specify the `steps_per_epoch` '
                       'argument.')
    # Validate user data.
    x, y, sample_weights = self._standardize_user_data(
        x,
        y,
        sample_weight=sample_weight,
        class_weight=class_weight,
        check_batch_axis=False,
        batch_size=batch_size)
    # Prepare validation data.
    do_validation = False
    if validation_data:
      do_validation = True
      if len(validation_data) == 2:
        val_x, val_y = validation_data  # pylint: disable=unpacking-non-sequence
        val_sample_weight = None
      elif len(validation_data) == 3:
        val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
      else:
        raise ValueError(
            'When passing validation_data, '
            'it must contain 2 (x_val, y_val) '
            'or 3 (x_val, y_val, val_sample_weights) '
            'items, however it contains %d items' % len(validation_data))

      val_x, val_y, val_sample_weights = self._standardize_user_data(
          val_x,
          val_y,
          sample_weight=val_sample_weight,
          check_batch_axis=False,
          batch_size=batch_size)
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        val_ins = val_x + val_y + val_sample_weights + [0.]
      else:
        val_ins = val_x + val_y + val_sample_weights

    elif validation_split and 0. < validation_split < 1.:
      do_validation = True
      if hasattr(x[0], 'shape'):
        split_at = int(x[0].shape[0] * (1. - validation_split))
      else:
        split_at = int(len(x[0]) * (1. - validation_split))
      x, val_x = (_slice_arrays(x, 0, split_at), _slice_arrays(x, split_at))
      y, val_y = (_slice_arrays(y, 0, split_at), _slice_arrays(y, split_at))
      sample_weights, val_sample_weights = (_slice_arrays(
          sample_weights, 0, split_at), _slice_arrays(sample_weights, split_at))
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        val_ins = val_x + val_y + val_sample_weights + [0.]
      else:
        val_ins = val_x + val_y + val_sample_weights

    elif validation_steps:
      do_validation = True
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        val_ins = [0.]

    # Prepare input arrays and training function.
    if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
      ins = x + y + sample_weights + [1.]
    else:
      ins = x + y + sample_weights
    self._make_train_function()
    f = self.train_function

    # Prepare display labels.
    out_labels = self._get_deduped_metrics_names()

    if do_validation:
      self._make_test_function()
      val_f = self.test_function
      callback_metrics = copy.copy(out_labels) + [
          'val_' + n for n in out_labels
      ]
    else:
      callback_metrics = copy.copy(out_labels)
      val_f = None
      val_ins = []

    # Delegate logic to `_fit_loop`.
    return self._fit_loop(
        f,
        ins,
        out_labels=out_labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        val_f=val_f,
        val_ins=val_ins,
        shuffle=shuffle,
        callback_metrics=callback_metrics,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps)

  def evaluate(self,
               x=None,
               y=None,
               batch_size=None,
               verbose=1,
               sample_weight=None,
               steps=None):
    """Returns the loss value & metrics values for the model in test mode.

    Computation is done in batches.

    Arguments:
        x: Numpy array of test data (if the model has a single input),
            or list of Numpy arrays (if the model has multiple inputs).
            If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
            `x` can be `None` (default) if feeding from
            TensorFlow data tensors.
        y: Numpy array of target (label) data
            (if the model has a single output),
            or list of Numpy arrays (if the model has multiple outputs).
            If output layers in the model are named, you can also pass a
            dictionary mapping output names to Numpy arrays.
            `y` can be `None` (default) if feeding from
            TensorFlow data tensors.
        batch_size: Integer or `None`.
            Number of samples per evaluation step.
            If unspecified, `batch_size` will default to 32.
        verbose: 0 or 1. Verbosity mode.
            0 = silent, 1 = progress bar.
        sample_weight: Optional Numpy array of weights for
            the test samples, used for weighting the loss function.
            You can either pass a flat (1D)
            Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples),
            or in the case of temporal data,
            you can pass a 2D array with shape
            `(samples, sequence_length)`,
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            `sample_weight_mode="temporal"` in `compile()`.
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        ValueError: in case of invalid arguments.
    """
    # Backwards compatibility.
    if batch_size is None and steps is None:
      batch_size = 32
    if x is None and y is None and steps is None:
      raise ValueError('If evaluating from data tensors, '
                       'you should specify the `steps` '
                       'argument.')
    # Validate user data.
    x, y, sample_weights = self._standardize_user_data(
        x,
        y,
        sample_weight=sample_weight,
        check_batch_axis=False,
        batch_size=batch_size)
    # Prepare inputs, delegate logic to `_test_loop`.
    if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
      ins = x + y + sample_weights + [0.]
    else:
      ins = x + y + sample_weights
    self._make_test_function()
    f = self.test_function
    return self._test_loop(
        f, ins, batch_size=batch_size, verbose=verbose, steps=steps)

  def predict(self, x, batch_size=None, verbose=0, steps=None):
    """Generates output predictions for the input samples.

    Computation is done in batches.

    Arguments:
        x: The input data, as a Numpy array
            (or list of Numpy arrays if the model has multiple outputs).
        batch_size: Integer. If unspecified, it will default to 32.
        verbose: Verbosity mode, 0 or 1.
        steps: Total number of steps (batches of samples)
            before declaring the prediction round finished.
            Ignored with the default value of `None`.

    Returns:
        Numpy array(s) of predictions.

    Raises:
        ValueError: In case of mismatch between the provided
            input data and the model's expectations,
            or in case a stateful model receives a number of samples
            that is not a multiple of the batch size.
    """
    # Backwards compatibility.
    if batch_size is None and steps is None:
      batch_size = 32
    if x is None and steps is None:
      raise ValueError('If predicting from data tensors, '
                       'you should specify the `steps` '
                       'argument.')
    # Validate user data.
    x = _standardize_input_data(
        x,
        self._feed_input_names,
        self._feed_input_shapes,
        check_batch_axis=False)
    if self.stateful:
      if x[0].shape[0] > batch_size and x[0].shape[0] % batch_size != 0:
        raise ValueError('In a stateful network, '
                         'you should only pass inputs with '
                         'a number of samples that can be '
                         'divided by the batch size. Found: ' +
                         str(x[0].shape[0]) + ' samples. '
                         'Batch size: ' + str(batch_size) + '.')

    # Prepare inputs, delegate logic to `_predict_loop`.
    if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
      ins = x + [0.]
    else:
      ins = x
    self._make_predict_function()
    f = self.predict_function
    return self._predict_loop(
        f, ins, batch_size=batch_size, verbose=verbose, steps=steps)

  def train_on_batch(self, x, y, sample_weight=None, class_weight=None):
    """Runs a single gradient update on a single batch of data.

    Arguments:
        x: Numpy array of training data,
            or list of Numpy arrays if the model has multiple inputs.
            If all inputs in the model are named,
            you can also pass a dictionary
            mapping input names to Numpy arrays.
        y: Numpy array of target data,
            or list of Numpy arrays if the model has multiple outputs.
            If all outputs in the model are named,
            you can also pass a dictionary
            mapping output names to Numpy arrays.
        sample_weight: Optional array of the same length as x, containing
            weights to apply to the model's loss for each sample.
            In the case of temporal data, you can pass a 2D array
            with shape (samples, sequence_length),
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            sample_weight_mode="temporal" in compile().
        class_weight: Optional dictionary mapping
            class indices (integers) to
            a weight (float) to apply to the model's loss for the samples
            from this class during training.
            This can be useful to tell the model to "pay more attention" to
            samples from an under-represented class.

    Returns:
        Scalar training loss
        (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
    """
    x, y, sample_weights = self._standardize_user_data(
        x,
        y,
        sample_weight=sample_weight,
        class_weight=class_weight,
        check_batch_axis=True)
    if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
      ins = x + y + sample_weights + [1.]
    else:
      ins = x + y + sample_weights
    self._make_train_function()
    outputs = self.train_function(ins)
    if len(outputs) == 1:
      return outputs[0]
    return outputs

  def test_on_batch(self, x, y, sample_weight=None):
    """Test the model on a single batch of samples.

    Arguments:
        x: Numpy array of test data,
            or list of Numpy arrays if the model has multiple inputs.
            If all inputs in the model are named,
            you can also pass a dictionary
            mapping input names to Numpy arrays.
        y: Numpy array of target data,
            or list of Numpy arrays if the model has multiple outputs.
            If all outputs in the model are named,
            you can also pass a dictionary
            mapping output names to Numpy arrays.
        sample_weight: Optional array of the same length as x, containing
            weights to apply to the model's loss for each sample.
            In the case of temporal data, you can pass a 2D array
            with shape (samples, sequence_length),
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            sample_weight_mode="temporal" in compile().

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        ValueError: in case of invalid arguments.
    """
    x, y, sample_weights = self._standardize_user_data(
        x, y, sample_weight=sample_weight, check_batch_axis=True)
    if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
      ins = x + y + sample_weights + [0.]
    else:
      ins = x + y + sample_weights
    self._make_test_function()
    outputs = self.test_function(ins)
    if len(outputs) == 1:
      return outputs[0]
    return outputs

  def predict_on_batch(self, x):
    """Returns predictions for a single batch of samples.

    Arguments:
        x: Input samples, as a Numpy array.

    Returns:
        Numpy array(s) of predictions.
    """
    x = _standardize_input_data(x, self._feed_input_names,
                                self._feed_input_shapes)
    if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
      ins = x + [0.]
    else:
      ins = x
    self._make_predict_function()
    outputs = self.predict_function(ins)
    if len(outputs) == 1:
      return outputs[0]
    return outputs

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
                    initial_epoch=0):
    """Fits the model on data yielded batch-by-batch by a Python generator.

    The generator is run in parallel to the model, for efficiency.
    For instance, this allows you to do real-time data augmentation
    on images on CPU in parallel to training your model on GPU.

    The use of `keras.utils.Sequence` guarantees the ordering
    and guarantees the single use of every input per epoch when
    using `use_multiprocessing=True`.

    Arguments:
        generator: A generator or an instance of `Sequence`
          (`keras.utils.Sequence`)
            object in order to avoid duplicate data
            when using multiprocessing.
            The output of the generator must be either
            - a tuple `(inputs, targets)`
            - a tuple `(inputs, targets, sample_weights)`.
            This tuple (a single output of the generator) makes a single batch.
            Therefore, all arrays in this tuple must have the same length (equal
            to the size of this batch). Different batches may have different
              sizes.
            For example, the last batch of the epoch is commonly smaller than
              the
            others, if the size of the dataset is not divisible by the batch
              size.
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
        verbose: Verbosity mode, 0, 1, or 2.
        callbacks: List of callbacks to be called during training.
        validation_data: This can be either
            - a generator for the validation data
            - a tuple (inputs, targets)
            - a tuple (inputs, targets, sample_weights).
        validation_steps: Only relevant if `validation_data`
            is a generator. Total number of steps (batches of samples)
            to yield from `generator` before stopping.
            Optional for `Sequence`: if unspecified, will use
            the `len(validation_data)` as a number of steps.
        class_weight: Dictionary mapping class indices to a weight
            for the class.
        max_queue_size: Integer. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Maximum number of processes to spin up
            when using process based threading.
            If unspecified, `workers` will default to 1. If 0, will
            execute the generator on the main thread.
        use_multiprocessing: Boolean. If True, use process based threading.
            If unspecified, `workers` will default to False.
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

    Returns:
        A `History` object.

    Example:

    ```python
        def generate_arrays_from_file(path):
            while 1:
                f = open(path)
                for line in f:
                    # create numpy arrays of input data
                    # and labels, from each line in the file
                    x1, x2, y = process_line(line)
                    yield ({'input_1': x1, 'input_2': x2}, {'output': y})
                f.close()

        model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                            steps_per_epoch=10000, epochs=10)
    ```

    Raises:
        ValueError: In case the generator yields
            data in an invalid format.
    """
    wait_time = 0.01  # in seconds
    epoch = initial_epoch

    do_validation = bool(validation_data)
    self._make_train_function()
    if do_validation:
      self._make_test_function()

    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
      logging.warning(
          UserWarning('Using a generator with `use_multiprocessing=True`'
                      ' and multiple workers may duplicate your data.'
                      ' Please consider using the`keras.utils.Sequence'
                      ' class.'))
    if steps_per_epoch is None:
      if is_sequence:
        steps_per_epoch = len(generator)
      else:
        raise ValueError('`steps_per_epoch=None` is only valid for a'
                         ' generator based on the `keras.utils.Sequence`'
                         ' class. Please specify `steps_per_epoch` or use'
                         ' the `keras.utils.Sequence` class.')

    # python 2 has 'next', 3 has '__next__'
    # avoid any explicit version checks
    val_gen = (
        hasattr(validation_data, 'next') or
        hasattr(validation_data, '__next__') or
        isinstance(validation_data, Sequence))
    if (val_gen and not isinstance(validation_data, Sequence) and
        not validation_steps):
      raise ValueError('`validation_steps=None` is only valid for a'
                       ' generator based on the `keras.utils.Sequence`'
                       ' class. Please specify `validation_steps` or use'
                       ' the `keras.utils.Sequence` class.')

    # Prepare display labels.
    out_labels = self._get_deduped_metrics_names()
    callback_metrics = out_labels + ['val_' + n for n in out_labels]

    # prepare callbacks
    self.history = cbks.History()
    callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
    if verbose:
      callbacks += [cbks.ProgbarLogger(count_mode='steps')]
    callbacks = cbks.CallbackList(callbacks)

    # it's possible to callback a different model than self:
    if hasattr(self, 'callback_model') and self.callback_model:
      callback_model = self.callback_model
    else:
      callback_model = self
    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': epochs,
        'steps': steps_per_epoch,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    enqueuer = None
    val_enqueuer = None

    try:
      if do_validation:
        if val_gen:
          if workers > 0:
            if isinstance(validation_data, Sequence):
              val_enqueuer = OrderedEnqueuer(
                  validation_data, use_multiprocessing=use_multiprocessing)
              if validation_steps is None:
                validation_steps = len(validation_data)
            else:
              val_enqueuer = GeneratorEnqueuer(
                  validation_data,
                  use_multiprocessing=use_multiprocessing,
                  wait_time=wait_time)
            val_enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            validation_generator = val_enqueuer.get()
          else:
            validation_generator = validation_data
        else:
          if len(validation_data) == 2:
            val_x, val_y = validation_data  # pylint: disable=unpacking-non-sequence
            val_sample_weight = None
          elif len(validation_data) == 3:
            val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
          else:
            raise ValueError(
                '`validation_data` should be a tuple '
                '`(val_x, val_y, val_sample_weight)` '
                'or `(val_x, val_y)`. Found: ' + str(validation_data))
          val_x, val_y, val_sample_weights = self._standardize_user_data(
              val_x, val_y, val_sample_weight)
          val_data = val_x + val_y + val_sample_weights
          if self.uses_learning_phase and not isinstance(
              K.learning_phase(), int):
            val_data += [0.]
          for cbk in callbacks:
            cbk.validation_data = val_data

      if workers > 0:
        if is_sequence:
          enqueuer = OrderedEnqueuer(
              generator,
              use_multiprocessing=use_multiprocessing,
              shuffle=shuffle)
        else:
          enqueuer = GeneratorEnqueuer(
              generator,
              use_multiprocessing=use_multiprocessing,
              wait_time=wait_time)
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()
      else:
        output_generator = generator

      callback_model.stop_training = False
      # Construct epoch logs.
      epoch_logs = {}
      while epoch < epochs:
        callbacks.on_epoch_begin(epoch)
        steps_done = 0
        batch_index = 0
        while steps_done < steps_per_epoch:
          generator_output = next(output_generator)

          if not hasattr(generator_output, '__len__'):
            raise ValueError('Output of generator should be '
                             'a tuple `(x, y, sample_weight)` '
                             'or `(x, y)`. Found: ' + str(generator_output))

          if len(generator_output) == 2:
            x, y = generator_output
            sample_weight = None
          elif len(generator_output) == 3:
            x, y, sample_weight = generator_output
          else:
            raise ValueError('Output of generator should be '
                             'a tuple `(x, y, sample_weight)` '
                             'or `(x, y)`. Found: ' + str(generator_output))
          # build batch logs
          batch_logs = {}
          if isinstance(x, list):
            batch_size = x[0].shape[0]
          elif isinstance(x, dict):
            batch_size = list(x.values())[0].shape[0]
          else:
            batch_size = x.shape[0]
          batch_logs['batch'] = batch_index
          batch_logs['size'] = batch_size
          callbacks.on_batch_begin(batch_index, batch_logs)

          outs = self.train_on_batch(
              x, y, sample_weight=sample_weight, class_weight=class_weight)

          if not isinstance(outs, list):
            outs = [outs]
          for l, o in zip(out_labels, outs):
            batch_logs[l] = o

          callbacks.on_batch_end(batch_index, batch_logs)

          batch_index += 1
          steps_done += 1

          # Epoch finished.
          if steps_done >= steps_per_epoch and do_validation:
            if val_gen:
              val_outs = self.evaluate_generator(
                  validation_generator, validation_steps, workers=0)
            else:
              # No need for try/except because
              # data has already been validated.
              val_outs = self.evaluate(
                  val_x,
                  val_y,
                  batch_size=batch_size,
                  sample_weight=val_sample_weights,
                  verbose=0)
            if not isinstance(val_outs, list):
              val_outs = [val_outs]
            # Same labels assumed.
            for l, o in zip(out_labels, val_outs):
              epoch_logs['val_' + l] = o

          if callback_model.stop_training:
            break

        callbacks.on_epoch_end(epoch, epoch_logs)
        epoch += 1
        if callback_model.stop_training:
          break

    finally:
      try:
        if enqueuer is not None:
          enqueuer.stop()
      finally:
        if val_enqueuer is not None:
          val_enqueuer.stop()

    callbacks.on_train_end()
    return self.history

  def evaluate_generator(self,
                         generator,
                         steps=None,
                         max_queue_size=10,
                         workers=1,
                         use_multiprocessing=False):
    """Evaluates the model on a data generator.

    The generator should return the same kind of data
    as accepted by `test_on_batch`.

    Arguments:
        generator: Generator yielding tuples (inputs, targets)
            or (inputs, targets, sample_weights)
            or an instance of Sequence (keras.utils.Sequence)
            object in order to avoid duplicate data
            when using multiprocessing.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
            Optional for `Sequence`: if unspecified, will use
            the `len(generator)` as a number of steps.
        max_queue_size: maximum size for the generator queue
        workers: Integer. Maximum number of processes to spin up
            when using process based threading.
            If unspecified, `workers` will default to 1. If 0, will
            execute the generator on the main thread.
        use_multiprocessing: if True, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        ValueError: in case of invalid arguments.

    Raises:
        ValueError: In case the generator yields
            data in an invalid format.
    """
    self._make_test_function()

    steps_done = 0
    wait_time = 0.01
    all_outs = []
    batch_sizes = []
    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
      logging.warning(
          UserWarning('Using a generator with `use_multiprocessing=True`'
                      ' and multiple workers may duplicate your data.'
                      ' Please consider using the`keras.utils.Sequence'
                      ' class.'))
    if steps is None:
      if is_sequence:
        steps = len(generator)
      else:
        raise ValueError('`steps=None` is only valid for a generator'
                         ' based on the `keras.utils.Sequence` class.'
                         ' Please specify `steps` or use the'
                         ' `keras.utils.Sequence` class.')
    enqueuer = None

    try:
      if workers > 0:
        if is_sequence:
          enqueuer = OrderedEnqueuer(
              generator, use_multiprocessing=use_multiprocessing)
        else:
          enqueuer = GeneratorEnqueuer(
              generator,
              use_multiprocessing=use_multiprocessing,
              wait_time=wait_time)
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()
      else:
        output_generator = generator

      while steps_done < steps:
        generator_output = next(output_generator)
        if not hasattr(generator_output, '__len__'):
          raise ValueError('Output of generator should be a tuple '
                           '(x, y, sample_weight) '
                           'or (x, y). Found: ' + str(generator_output))
        if len(generator_output) == 2:
          x, y = generator_output
          sample_weight = None
        elif len(generator_output) == 3:
          x, y, sample_weight = generator_output
        else:
          raise ValueError('Output of generator should be a tuple '
                           '(x, y, sample_weight) '
                           'or (x, y). Found: ' + str(generator_output))
        outs = self.test_on_batch(x, y, sample_weight=sample_weight)

        if isinstance(x, list):
          batch_size = x[0].shape[0]
        elif isinstance(x, dict):
          batch_size = list(x.values())[0].shape[0]
        else:
          batch_size = x.shape[0]
        if batch_size == 0:
          raise ValueError('Received an empty batch. '
                           'Batches should at least contain one item.')
        all_outs.append(outs)

        steps_done += 1
        batch_sizes.append(batch_size)

    finally:
      if enqueuer is not None:
        enqueuer.stop()

    if not isinstance(outs, list):
      return np.average(np.asarray(all_outs), weights=batch_sizes)
    else:
      averages = []
      for i in range(len(outs)):
        averages.append(
            np.average([out[i] for out in all_outs], weights=batch_sizes))
      return averages

  def predict_generator(self,
                        generator,
                        steps=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        verbose=0):
    """Generates predictions for the input samples from a data generator.

    The generator should return the same kind of data as accepted by
    `predict_on_batch`.

    Arguments:
        generator: Generator yielding batches of input samples
            or an instance of Sequence (keras.utils.Sequence)
            object in order to avoid duplicate data
            when using multiprocessing.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
            Optional for `Sequence`: if unspecified, will use
            the `len(generator)` as a number of steps.
        max_queue_size: Maximum size for the generator queue.
        workers: Integer. Maximum number of processes to spin up
            when using process based threading.
            If unspecified, `workers` will default to 1. If 0, will
            execute the generator on the main thread.
        use_multiprocessing: If `True`, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.
        verbose: verbosity mode, 0 or 1.

    Returns:
        Numpy array(s) of predictions.

    Raises:
        ValueError: In case the generator yields
            data in an invalid format.
    """
    self._make_predict_function()

    steps_done = 0
    wait_time = 0.01
    all_outs = []
    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
      logging.warning(
          UserWarning('Using a generator with `use_multiprocessing=True`'
                      ' and multiple workers may duplicate your data.'
                      ' Please consider using the`keras.utils.Sequence'
                      ' class.'))
    if steps is None:
      if is_sequence:
        steps = len(generator)
      else:
        raise ValueError('`steps=None` is only valid for a generator'
                         ' based on the `keras.utils.Sequence` class.'
                         ' Please specify `steps` or use the'
                         ' `keras.utils.Sequence` class.')
    enqueuer = None

    try:
      if workers > 0:
        if is_sequence:
          enqueuer = OrderedEnqueuer(
              generator, use_multiprocessing=use_multiprocessing)
        else:
          enqueuer = GeneratorEnqueuer(
              generator,
              use_multiprocessing=use_multiprocessing,
              wait_time=wait_time)
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()
      else:
        output_generator = generator

      if verbose == 1:
        progbar = Progbar(target=steps)

      while steps_done < steps:
        generator_output = next(output_generator)
        if isinstance(generator_output, tuple):
          # Compatibility with the generators
          # used for training.
          if len(generator_output) == 2:
            x, _ = generator_output
          elif len(generator_output) == 3:
            x, _, _ = generator_output
          else:
            raise ValueError('Output of generator should be '
                             'a tuple `(x, y, sample_weight)` '
                             'or `(x, y)`. Found: ' + str(generator_output))
        else:
          # Assumes a generator that only
          # yields inputs (not targets and sample weights).
          x = generator_output

        outs = self.predict_on_batch(x)
        if not isinstance(outs, list):
          outs = [outs]

        if not all_outs:
          for out in outs:
            all_outs.append([])

        for i, out in enumerate(outs):
          all_outs[i].append(out)
        steps_done += 1
        if verbose == 1:
          progbar.update(steps_done)

    finally:
      if enqueuer is not None:
        enqueuer.stop()

    if len(all_outs) == 1:
      if steps_done == 1:
        return all_outs[0][0]
      else:
        return np.concatenate(all_outs[0])
    if steps_done == 1:
      return [out[0] for out in all_outs]
    else:
      return [np.concatenate(out) for out in all_outs]

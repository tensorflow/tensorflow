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
"""Training-related utilities.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from collections import OrderedDict
import copy

import numpy as np
import six

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils.losses_utils import squeeze_or_expand_dimensions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.util import nest


@six.add_metaclass(abc.ABCMeta)
class Aggregator(object):
  """Abstract base class used to aggregate batch-level outputs of a loop.

  Attributes:
    use_steps: Whether the loop is using `step` or `batch_size`.
    num_samples_or_steps: Either `batch_size*num_batches` or `steps`.
    results: What to return at the end of the aggregation loop.
  """

  def __init__(self, use_steps, num_samples_or_steps):
    self.use_steps = use_steps
    self.num_samples_or_steps = num_samples_or_steps
    self.results = []

  @abc.abstractmethod
  def create(self, batch_outs):
    """Creates the initial results from the first batch outputs.

    Arguments:
      batch_outs: A list of batch-level outputs.
    """
    NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def aggregate(self, batch_outs, batch_start=None, batch_end=None):
    """Aggregates batch-level results into total results.

    Arguments:
      batch_outs: A list of batch-level outputs.
      batch_start: The start index of this batch. Always `None` if `use_steps`
        is `True`.
      batch_end: The end index of this batch. Always `None` if `use_steps` is
        `True`.
    """
    NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def finalize(self):
    """Prepares the total results to be returned."""
    NotImplementedError('Must be implemented in subclasses.')


class MetricsAggregator(Aggregator):
  """Aggregator that calculates loss and metrics info."""

  def create(self, batch_outs):
    self.results = [0.] * len(batch_outs)

  def aggregate(self, batch_outs, batch_start=None, batch_end=None):
    # Loss.
    if self.use_steps:
      self.results[0] += batch_outs[0]
    else:
      self.results[0] += batch_outs[0] * (batch_end - batch_start)
    # Metrics (always stateful, just grab current values.)
    self.results[1:] = batch_outs[1:]

  def finalize(self):
    self.results[0] /= self.num_samples_or_steps


class OutputsAggregator(Aggregator):
  """Aggregator that concatenates outputs."""

  def create(self, batch_outs):
    if self.use_steps:
      # Cannot pre-allocate the returned NumPy arrays bc
      # batch sizes are unknown. Concatenate batches at the end.
      for _ in batch_outs:
        self.results.append([])
    else:
      # Pre-allocate NumPy arrays.
      for batch_out in batch_outs:
        shape = (self.num_samples_or_steps,) + batch_out.shape[1:]
        self.results.append(np.zeros(shape, dtype=batch_out.dtype))

  def aggregate(self, batch_outs, batch_start=None, batch_end=None):
    if self.use_steps:
      for i, batch_out in enumerate(batch_outs):
        self.results[i].append(batch_out)
    else:
      for i, batch_out in enumerate(batch_outs):
        self.results[i][batch_start:batch_end] = batch_out

  def finalize(self):
    if self.use_steps:
      self.results = [np.concatenate(result, axis=0) for result in self.results]


def get_progbar(model, count_mode):
  """Get Progbar."""
  stateful_metric_names = None
  if hasattr(model, 'metrics_names'):
    stateful_metric_names = model.metrics_names[1:]  # Exclude `loss`
  return cbks.ProgbarLogger(count_mode, stateful_metrics=stateful_metric_names)


def slice_arrays(arrays, indices, contiguous=True):
  """Slices batches out of provided arrays (workaround for eager tensors).

  Unfortunately eager tensors don't have the same slicing behavior as
  Numpy arrays (they follow the same slicing behavior as symbolic TF tensors),
  hence we cannot use `generic_utils.slice_arrays` directly
  and we have to implement this workaround based on `concat`. This has a
  performance cost.

  Arguments:
    arrays: Single array or list of arrays.
    indices: List of indices in the array that should be included in the output
      batch.
    contiguous: Boolean flag indicating whether the indices are contiguous.

  Returns:
    Slice of data (either single array or list of arrays).
  """
  converted_to_list = False
  if not isinstance(arrays, list):
    converted_to_list = True
    arrays = [arrays]
  if any(tensor_util.is_tensor(x) for x in arrays):
    if not contiguous:
      entries = [[x[i:i + 1] for i in indices] for x in arrays]
      slices = [array_ops.concat(x, axis=0) for x in entries]
    else:
      slices = [x[indices[0]:indices[-1] + 1] for x in arrays]
  else:
    slices = generic_utils.slice_arrays(arrays, indices)

  if converted_to_list:
    slices = slices[0]
  return slices


def check_num_samples(ins,
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
  if steps is not None and batch_size is not None:
    raise ValueError(
        'If ' + steps_name + ' is set, the `batch_size` must be None.')
  if check_steps_argument(ins, steps, steps_name):
    return None
  if hasattr(ins[0], 'shape'):
    return int(ins[0].shape[0])
  return None  # Edge case where ins == [static_learning_phase]


def standardize_single_array(x, expected_shape=None):
  """Expand data of shape (x,) to (x, 1), unless len(expected_shape)==1."""
  if x is None:
    return None

  if (x.shape is not None
      and len(x.shape) == 1
      and (expected_shape is None or len(expected_shape) != 1)):
    if tensor_util.is_tensor(x):
      x = array_ops.expand_dims(x, axis=1)
    else:
      x = np.expand_dims(x, 1)
  return x


def standardize_input_data(data,
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
    if (data is not None and hasattr(data, '__len__') and len(data) and
        not isinstance(data, dict)):
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
    except KeyError as e:
      raise ValueError('No data provided for "' + e.args[0] + '". Need data '
                       'for each key in: ' + str(names))
  elif isinstance(data, (list, tuple)):
    if isinstance(data[0], (list, tuple)):
      data = [np.asarray(d) for d in data]
    elif len(names) == 1 and isinstance(data[0], (float, int)):
      data = [np.asarray(data)]
    else:
      data = [
          x.values if x.__class__.__name__ == 'DataFrame' else x for x in data
      ]
  else:
    data = data.values if data.__class__.__name__ == 'DataFrame' else data
    data = [data]
  if shapes is not None:
    data = [standardize_single_array(x, shape)
            for (x, shape) in zip(data, shapes)]
  else:
    data = [standardize_single_array(x) for x in data]

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
        if tensor_util.is_tensor(data[i]):
          tensorshape = data[i].get_shape()
          if not tensorshape:
            continue
          data_shape = tuple(tensorshape.as_list())
        else:
          data_shape = data[i].shape
        shape = shapes[i]
        if len(data_shape) != len(shape):
          raise ValueError('Error when checking ' + exception_prefix +
                           ': expected ' + names[i] + ' to have ' +
                           str(len(shape)) + ' dimensions, but got array '
                           'with shape ' + str(data_shape))
        if not check_batch_axis:
          data_shape = data_shape[1:]
          shape = shape[1:]
        for dim, ref_dim in zip(data_shape, shape):
          if ref_dim != dim and ref_dim is not None and dim is not None:
            raise ValueError(
                'Error when checking ' + exception_prefix + ': expected ' +
                names[i] + ' to have shape ' + str(shape) +
                ' but got array with shape ' + str(data_shape))
  return data


def standardize_sample_or_class_weights(x_weight, output_names, weight_type):
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
  if x_weight is None or (isinstance(x_weight, list) and len(x_weight) == 0):  # pylint: disable=g-explicit-length-test
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


def standardize_class_weights(class_weight, output_names):
  return standardize_sample_or_class_weights(class_weight, output_names,
                                             'class_weight')


def standardize_sample_weights(sample_weight, output_names):
  return standardize_sample_or_class_weights(sample_weight, output_names,
                                             'sample_weight')


def check_array_lengths(inputs, targets, weights=None):
  """Does user input validation for numpy arrays.

  Arguments:
      inputs: list of Numpy arrays of inputs.
      targets: list of Numpy arrays of targets.
      weights: list of Numpy arrays of sample weights.

  Raises:
      ValueError: in case of incorrectly formatted data.
  """

  def set_of_lengths(x):
    # Returns a set with the variation between
    # different shapes, with None => 0
    if x is None:
      return {}
    else:
      return set([y.shape[0] for y in x
                  if y is not None and not tensor_util.is_tensor(y)])

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


def check_loss_and_target_compatibility(targets, loss_fns, output_shapes):
  """Does validation on the compatibility of targets and loss functions.

  This helps prevent users from using loss functions incorrectly. This check
  is purely for UX purposes.

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
    if y is None or loss is None or tensor_util.is_tensor(y):
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


def collect_per_output_metric_info(metrics,
                                   output_names,
                                   output_shapes,
                                   loss_fns,
                                   sample_weights=None):
  """Maps metric names and functions to model outputs.

  Arguments:
      metrics: a list or dict of metric functions.
      output_names: a list of the names (strings) of model outputs.
      output_shapes: a list of the shapes (strings) of model outputs.
      loss_fns: a list of the loss functions corresponding to the model outputs.
      sample_weights: a list of weights to be applied on the model outputs.

  Returns:
      A list (one entry per model output) of dicts.
      For instance, if the model has 2 outputs, and for the first output
      we want to compute "binary_accuracy" and "binary_crossentropy",
      and just "binary_accuracy" for the second output,
      the list would look like: `[
        {
          'acc': (binary_accuracy(), mean_obj_1),
          'ce': (binary_crossentropy(), mean_obj_2)
        },
        {
          'acc': (binary_accuracy(), mean_obj_3)
        }
      ]`

  Raises:
      TypeError: if an incorrect type is passed for the `metrics` argument.
  """
  if not metrics:
    return [{} for _ in output_names]
  if isinstance(metrics, list):
    # we then apply all metrics to all outputs.
    nested_metrics = [copy.copy(metrics) for _ in output_names]
  elif isinstance(metrics, dict):
    nested_metrics = []
    for name in output_names:
      output_metrics = metrics.get(name, [])
      if not isinstance(output_metrics, list):
        output_metrics = [output_metrics]
      nested_metrics.append(output_metrics)
  else:
    raise TypeError('Type of `metrics` argument not understood. '
                    'Expected a list or dictionary, found: ' + str(metrics))

  per_output_metrics = []
  for i, metrics in enumerate(nested_metrics):
    metrics_dict = OrderedDict()
    for metric in metrics:
      weighted = False if (sample_weights is None) else (
          sample_weights[i] is not None)
      metric_name = get_metric_name(metric, weighted)
      metric_fn = get_metric_function(
          metric, output_shape=output_shapes[i], loss_fn=loss_fns[i])

      # If the metric function is not stateful, we create a stateful version and
      # return both the stateless and the stateful version together. For batch
      # APIs like `train_on_batch` we will use the stateless version and for
      # other APIs like `fit` we will use the stateful version.
      is_stateful = isinstance(metric_fn,
                               base_layer.Layer) and metric_fn.stateful
      stateful_fn = metric_fn
      if not is_stateful:
        stateful_fn = metrics_module.MeanMetricWrapper(
            metric_fn, name=metric_fn.__name__)

      metrics_dict[metric_name] = (metric_fn, stateful_fn)
    per_output_metrics.append(metrics_dict)

  return per_output_metrics


def batch_shuffle(index_array, batch_size):
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


def weighted_masked_objective(fn):
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
      mask = math_ops.cast(mask, y_pred.dtype)
      # Update weights with mask.
      if weights is None:
        weights = mask
      else:
        # Update dimensions of weights to match with mask if possible.
        mask, _, weights = squeeze_or_expand_dimensions(mask, None, weights)
        weights *= mask

    # Apply sample weighting.
    if weights is not None:

      # Update dimensions of weights to match with values if possible.
      score_array, _, weights = squeeze_or_expand_dimensions(
          score_array, None, weights)
      try:
        # Broadcast weights if possible.
        weights = weights_broadcast_ops.broadcast_weights(weights, score_array)
      except ValueError:
        # Reduce values to same ndim as weight array.
        ndim = K.ndim(score_array)
        weight_ndim = K.ndim(weights)
        score_array = K.mean(score_array, axis=list(range(weight_ndim, ndim)))

      score_array = math_ops.multiply(score_array, weights)
      score_array = math_ops.reduce_sum(score_array)
      weights = math_ops.reduce_sum(weights)
      score_array = math_ops.div_no_nan(score_array, weights)
    return K.mean(score_array)

  return weighted


def standardize_weights(y,
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
  # Iterator may return sample_weight as 1-tuple
  if isinstance(sample_weight, tuple):
    sample_weight = sample_weight[0]
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

    if (not tensor_util.is_tensor(sample_weight) and
        y.shape[:sample_weight.ndim] != sample_weight.shape):
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
      y_classes = np.argmax(y, axis=1)
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
    return None


def has_symbolic_tensors(ls):
  if context.executing_eagerly():
    return False
  return has_tensors(ls)


def has_tensors(ls):
  if isinstance(ls, (list, tuple)):
    return any(tensor_util.is_tensor(v) for v in ls)
  if isinstance(ls, dict):
    return any(tensor_util.is_tensor(v) for _, v in six.iteritems(ls))
  return tensor_util.is_tensor(ls)


def get_metric_name(metric, weighted=False):
  """Returns the name corresponding to the given metric input.

  Arguments:
    metric: Metric function name or reference.
    weighted: Boolean indicating if the given metric is weighted.

  Returns:
      The metric name.
  """
  metric_name_prefix = 'weighted_' if weighted else ''
  if metric in ('accuracy', 'acc', 'crossentropy', 'ce'):
    if metric in ('accuracy', 'acc'):
      suffix = 'acc'
    elif metric in ('crossentropy', 'ce'):
      suffix = 'ce'
  else:
    metric_fn = metrics_module.get(metric)
    # Get metric name as string
    if hasattr(metric_fn, 'name'):
      suffix = metric_fn.name
    else:
      suffix = metric_fn.__name__
  metric_name = metric_name_prefix + suffix
  return metric_name


def get_metric_function(metric, output_shape=None, loss_fn=None):
  """Returns the metric function corresponding to the given metric input.

  Arguments:
      metric: Metric function name or reference.
      output_shape: The shape of the output that this metric
          will be calculated for.
      loss_fn: The loss function used.

  Returns:
      The metric function.
  """
  if metric in ['accuracy', 'acc']:
    if output_shape[-1] == 1 or loss_fn == losses.binary_crossentropy:
      return metrics_module.binary_accuracy  # case: binary accuracy
    elif loss_fn == losses.sparse_categorical_crossentropy:
      # case: categorical accuracy with sparse targets
      return metrics_module.sparse_categorical_accuracy
    return metrics_module.categorical_accuracy  # case: categorical accuracy
  elif metric in ['crossentropy', 'ce']:
    if output_shape[-1] == 1 or loss_fn == losses.binary_crossentropy:
      return metrics_module.binary_crossentropy  # case: binary cross-entropy
    elif loss_fn == losses.sparse_categorical_crossentropy:
      # case: categorical cross-entropy with sparse targets
      return metrics_module.sparse_categorical_crossentropy
    # case: categorical cross-entropy
    return metrics_module.categorical_crossentropy
  return metrics_module.get(metric)


def call_metric_function(metric_fn, y_true, y_pred, weights=None, mask=None):
  """Invokes metric function and returns the metric result tensor."""
  if mask is None:
    return metric_fn(y_true, y_pred, sample_weight=weights)

  mask = math_ops.cast(mask, y_pred.dtype)
  if weights is None:
    # Use mask as sample weight.
    return metric_fn(y_true, y_pred, sample_weight=mask)

  # Update dimensions of weights to match with mask.
  mask, _, weights = squeeze_or_expand_dimensions(mask, None, weights)
  weights *= mask
  return metric_fn(y_true, y_pred, sample_weight=weights)


def get_loss_function(loss):
  """Returns the loss function corresponding to the given loss input."""
  if loss is None or isinstance(loss, losses.Loss):
    return loss

  # TODO(psv): After we have added all V2 losses, update this function.
  if loss in ['mse', 'MSE', 'mean_squared_error']:
    return losses.MeanSquaredError()
  return losses.get(loss)


def validate_iterator_input(x, y, sample_weight, validation_split=None):
  """Validates user input arguments when a dataset iterator is passed.

  Arguments:
    x: Input data. A `tf.data` dataset iterator.
    y: Target data. It could be either Numpy array(s) or TensorFlow tensor(s).
        Expected to be `None` when `x` is a dataset iterator.
    sample_weight: An optional sample-weight array passed by the user to
        weight the importance of each sample in `x`. Expected to be `None` when
        `x` is a dataset iterator
    validation_split: Float between 0 and 1. Fraction of the training data to
        be used as validation data. Expected to be `None` when `x` is a dataset
        iterator.

  Raises:
    ValueError: if argument `y` or `sample_weight` or `validation_split` are
        provided by user.
  """
  if y is not None:
    raise ValueError('You passed a dataset or dataset iterator (%s) as '
                     'input `x` to your model. In that case, you should '
                     'not specify a target (`y`) argument, since the dataset '
                     'or dataset iterator generates both input data and '
                     'target data. '
                     'Received: %s' % (x, y))
  if sample_weight is not None:
    raise ValueError('`sample_weight` argument is not supported when input '
                     '`x` is a dataset or a dataset iterator. Instead, you'
                     'can provide sample_weight as the third element  of your'
                     'dataset, i.e. (inputs, targets, sample_weight). '
                     'Received: x=%s, sample_weight=%s' % (x, sample_weight))
  if validation_split is not None and validation_split != 0.0:
    raise ValueError(
        '`validation_split` argument is not supported when '
        'input `x` is a dataset or a dataset iterator. '
        'Received: x=%s, validation_split=%f' % (x, validation_split))


def check_generator_arguments(y=None, sample_weight=None):
  """Validates arguments passed when using a generator."""
  if y is not None:
    raise ValueError('`y` argument is not supported when data is'
                     'a generator or Sequence instance. Instead pass targets'
                     ' as the second element of the generator.')
  if sample_weight is not None:
    raise ValueError('`sample_weight` argument is not supported when data is'
                     'a generator or Sequence instance. Instead pass sample'
                     ' weights as the third element of the generator.')


def check_steps_argument(input_data, steps, steps_name):
  """Validates `steps` argument based on input data's type.

  The cases when `steps` value must be provided are when
    1. input data passed is an iterator.
    2. model was built on top of symbolic tensors, input data is not
       required and is `None`.
    3. input data passed is a symbolic tensor.

  Arguments:
      input_data: Input data. Can be Numpy array(s) or TensorFlow tensor(s) or
        tf.data.Dataset iterator or `None`.
      steps: Integer or `None`. Total number of steps (batches of samples) to
        execute.
      steps_name: The public API's parameter name for `steps`.

  Returns:
    boolean, True if `steps` argument is required, else False.

  Raises:
      ValueError: if `steps` argument is required for given input data type
        but not provided.
  """

  is_x_iterator = (
      isinstance(input_data, iterator_ops.Iterator) or
      isinstance(input_data, iterator_ops.EagerIterator))

  if (input_data is None or is_x_iterator or has_symbolic_tensors(input_data) or
      (isinstance(input_data, list) and not input_data)):
    if steps is None:
      input_type_str = 'iterators' if is_x_iterator else 'data tensors'
      raise ValueError('When using {input_type} as input to a model, you should'
                       ' specify the `{steps_name}` argument.'.format(
                           input_type=input_type_str, steps_name=steps_name))
    return True
  return False


def cast_single_tensor(x):
  if tensor_util.is_tensor(x) and x.dtype.is_floating:
    return math_ops.cast(x, dtype=K.floatx())
  return x


def cast_if_floating_dtype(x):
  """Casts the given data tensors to the default floating point type.

  Casts only if the input is already a floating point type.
  Args:
    x: tensor or list/tuple of tensors.

  Returns:
    Converted input.

  Raises:
    RuntimeError: if data isn't tensors.
  """
  if not has_tensors(x):
    raise RuntimeError(
        'Please provide tensors for casting, got: {x}'.format(x=x))

  return nest.map_structure(cast_single_tensor, x)


def get_output_sample_weight_and_mode(skip_target_weighing_indices,
                                      sample_weight_mode, output_name,
                                      output_index):
  """Returns the sample weight and weight mode for a single output."""
  if output_index in skip_target_weighing_indices:
    return None, None

  if sample_weight_mode == 'temporal':
    default_value = [[1.]]
    shape = [None, None]
    mode = 'temporal'
  else:
    default_value = [1.]
    shape = [None]
    mode = None
  if context.executing_eagerly():
    weight = None
  else:
    weight = array_ops.placeholder_with_default(
        constant_op.constant(default_value, dtype=K.floatx()),
        shape=shape,
        name=output_name + '_sample_weights')
  return weight, mode


def prepare_sample_weights(output_names, sample_weight_mode,
                           skip_target_weighing_indices):
  """Prepares sample weights for the model.

  Args:
    output_names: List of model output names.
    sample_weight_mode: sample weight mode user input passed from compile API.
    skip_target_weighing_indices: Indices of output for which sample weights
      should be skipped.

  Returns:
    A pair of list of sample weights and sample weight modes
      (one for each output).

  Raises:
    ValueError: In case of invalid `sample_weight_mode` input.
  """
  sample_weights = []
  sample_weight_modes = []
  if isinstance(sample_weight_mode, dict):
    unknown_output = set(sample_weight_mode.keys()) - set(output_names)
    if unknown_output:
      raise ValueError('Unknown entry in '
                       'sample_weight_mode dictionary: "' + unknown_output +
                       '". Only expected the following keys: ' +
                       str(output_names))
    for i, name in enumerate(output_names):
      if (i not in skip_target_weighing_indices and
          name not in sample_weight_mode):
        raise ValueError('Output missing from sample_weight_modes dictionary')
      weight, mode = get_output_sample_weight_and_mode(
          skip_target_weighing_indices, sample_weight_mode.get(name), name, i)
      sample_weights.append(weight)
      sample_weight_modes.append(mode)
  elif isinstance(sample_weight_mode, list):
    if len(sample_weight_mode) != len(output_names):
      raise ValueError('When passing a list as sample_weight_mode, '
                       'it should have one entry per model output. '
                       'The model has ' + str(len(output_names)) +
                       ' outputs, but you passed ' +
                       str(len(sample_weight_mode)) + 'sample_weight_modes')
    for i, name in enumerate(output_names):
      weight, mode = get_output_sample_weight_and_mode(
          skip_target_weighing_indices, sample_weight_mode[i], name, i)
      sample_weights.append(weight)
      sample_weight_modes.append(mode)
  else:
    for i, name in enumerate(output_names):
      weight, mode = get_output_sample_weight_and_mode(
          skip_target_weighing_indices, sample_weight_mode, name, i)
      sample_weights.append(weight)
      sample_weight_modes.append(mode)
  return sample_weights, sample_weight_modes


# TODO(rohanj): This is a hack to get around not depending on feature_column and
# create a cyclical dependency. Figure out a cleaner solution
def is_feature_layer(layer):
  """Returns whether `layer` is a FeatureLayer or not."""
  return getattr(layer, '_is_feature_layer', False)


class ModelInputs(object):
  """Encapsulates model inputs.

  Allows for transforming model inputs while keeping the same structure.
  """

  def __init__(self, inputs):
    self._inputs = inputs
    self._is_dict = isinstance(self._inputs, dict)
    self._is_single_input = not isinstance(self._inputs, (list, tuple, dict))

    self._flattened_inputs = []
    self._input_names = []

    if self._is_dict:
      for k in sorted(self._inputs.keys()):
        self._flattened_inputs.append(self._inputs[k])
        self._input_names.append(k)
    else:
      self._flattened_inputs = nest.flatten(self._inputs)
      self._input_names = [
          'input_%d' % (i + 1) for i in range(len(self._flattened_inputs))
      ]

  def get_input_names(self):
    """Returns keys to name inputs by.

    In case inputs provided were a list, tuple or single entry, we make up a
    key 'input_%d'. For dictionary case, we return a sorted list of keys.
    """
    return self._input_names

  def get_symbolic_inputs(self, return_single_as_list=False):
    """Returns inputs to be set as self.inputs for a model."""
    # TODO(karmel): There is a side-effect here where what you get
    # with as_list and as_dict depends on whether you have called this
    # method first, since it modifies in place.
    for i in range(len(self._flattened_inputs)):
      k = self._input_names[i]
      v = self._flattened_inputs[i]
      if isinstance(v, (list, float, int)):
        v = np.asarray(v)
        if v.ndim == 1:
          v = np.expand_dims(v, 1)

      if isinstance(v, (np.ndarray, ops.EagerTensor)):
        # We fix the placeholder shape except the batch size.
        # This is suboptimal, but it is the best we can do with the info
        # we have. The user should call `model._set_inputs(placeholders)`
        # to specify custom placeholders if the need arises.
        shape = (None,) + tuple(v.shape[1:])
        v = K.placeholder(shape=shape, name=k)
      elif isinstance(v, tensor_shape.TensorShape):
        shape = (None,) + tuple(v.as_list()[1:])
        v = K.placeholder(shape=shape, name=k)

      self._flattened_inputs[i] = v

    if self._is_dict:
      return dict(zip(self._input_names, self._flattened_inputs))
    if self._is_single_input and not return_single_as_list:
      return self._flattened_inputs[0]
    return self._flattened_inputs

  def as_dict(self):
    """An iterable over a dictionary version of inputs."""
    for i in range(len(self._flattened_inputs)):
      yield self._input_names[i], self._flattened_inputs[i]

  def as_list(self):
    """Returning the inputs as a list."""
    return self._flattened_inputs


# Allow use of methods not exposed to the user.
# pylint: disable=protected-access
def get_input_shape_and_dtype(layer):
  """Retrieves input shape and input dtype of layer if applicable.

  Args:
    layer: Layer (or model) instance.

  Returns:
    Tuple (input_shape, input_dtype). Both could be None if the layer
      does not have a defined input shape.

  Raises:
    ValueError: in case an empty Sequential or Graph Network is passed.
  """

  def _is_graph_model(layer):
    return ((hasattr(layer, '_is_graph_network') and layer._is_graph_network) or
            layer.__class__.__name__ == 'Sequential')

  # In case of nested models: recover the first layer
  # of the deepest model to infer input shape and dtype.
  # Subclassed Models may not have been built so can't be checked.
  while _is_graph_model(layer):
    if not layer.layers:
      raise ValueError('An empty Model cannot be used as a Layer.')
    layer = layer.layers[0]

  if hasattr(layer, '_batch_input_shape'):
    return layer._batch_input_shape, layer.dtype
  return None, None


# pylint: enable=protected-access


def get_static_batch_size(layer):
  """Gets the static batch size of a Layer.

  Arguments:
    layer: a `Layer` instance.

  Returns:
    The static batch size of a Layer.
  """
  batch_input_shape, _ = get_input_shape_and_dtype(layer)
  if batch_input_shape is not None:
    return tensor_shape.as_dimension(batch_input_shape[0]).value
  return None


def generic_output_names(outputs_list):
  return ['output_%d' % (i + 1) for i in range(len(outputs_list))]


def trace_model_call(model, input_signature=None):
  """Trace the model call to create a tf.function for exporting a Keras model.

  Args:
    model: A Keras model.
    input_signature: optional, a list of tf.TensorSpec objects specifying the
      inputs to the model.

  Returns:
    A tf.function wrapping the model's call function with input signatures set.

  Raises:
    ValueError: if input signature cannot be inferred from the model.
  """
  if input_signature is None:
    if isinstance(model.call, def_function.PolymorphicFunction):
      input_signature = model.call.input_signature

  if input_signature is None:
    try:
      inputs = model.inputs
      input_names = model.input_names
    except AttributeError:
      raise ValueError(
          'Model {} cannot be saved because the input shapes have not been '
          'set. Usually, input shapes are automatically determined from calling'
          ' .fit() or .predict(). To manually set the shapes, call '
          'model._set_inputs(inputs).'.format(model))
    input_specs = []
    for input_tensor, input_name in zip(inputs, input_names):
      input_specs.append(tensor_spec.TensorSpec(
          shape=input_tensor.shape, dtype=input_tensor.dtype,
          name=input_name))
    # The input signature of the call function is a list with one element, since
    # all tensor inputs must be passed in as the first argument.
    input_signature = [input_specs] if len(input_specs) > 1 else input_specs

  # TODO(mdan): Should the model's call be autographed by default?
  @def_function.function(input_signature=input_signature, autograph=False)
  def _wrapped_model(*args):
    """A concrete tf.function that wraps the model's call function."""
    # When given a single input, Keras models will call the model on the tensor
    # rather than a list consisting of the single tensor.
    inputs = args[0] if len(input_signature) == 1 else list(args)
    outputs_list = nest.flatten(model(inputs=inputs))
    try:
      output_names = model.output_names
    except AttributeError:
      output_names = generic_output_names(outputs_list)
    return {name: output for name, output in zip(output_names, outputs_list)}

  return _wrapped_model


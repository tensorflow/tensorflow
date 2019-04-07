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
"""Training-related utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
from collections import OrderedDict

import numpy as np
import six

from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils.losses_utils import squeeze_or_expand_dimensions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
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
    raise NotImplementedError('Must be implemented in subclasses.')

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
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def finalize(self):
    """Prepares the total results to be returned."""
    raise NotImplementedError('Must be implemented in subclasses.')


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
    if not self.results:
      raise ValueError('Empty training data.')
    self.results[0] /= self.num_samples_or_steps


class OutputsAggregator(Aggregator):
  """Aggregator that concatenates outputs."""

  # Used to indicate that the aggregator cannot allocate an object for the
  # output at this location.
  _DEFER_OUTPUT_ALLOCATION = '__defer_output_allocation__'

  def create(self, batch_outs):
    if self.use_steps:
      # Cannot pre-allocate the returned NumPy arrays bc
      # batch sizes are unknown. Concatenate batches at the end.
      for _ in batch_outs:
        self.results.append([])
    else:
      # Pre-allocate NumPy arrays.
      for batch_out in batch_outs:
        if composite_tensor_utils.is_composite_or_composite_value(batch_out):
          # If the output is not a ndarray, it will be either a composite tensor
          # or a composite tensor's Value object. In either case, we can't
          # allocate an array to hold the object - we'll handle it later.
          self.results.append(self._DEFER_OUTPUT_ALLOCATION)
        elif isinstance(batch_out, np.ndarray):
          # If the output is a ndarray, append an output array pre-allocated
          # to the expected shape of the output.
          shape = (self.num_samples_or_steps,) + batch_out.shape[1:]
          self.results.append(np.zeros(shape, dtype=batch_out.dtype))
        else:
          # This is not a ndarray, a CompositeTensor, or a CompositeTensorValue.
          # Fail fast rather than trying to concatenate it.
          raise RuntimeError('Attempted to aggregate unsupported object %s.' %
                             batch_out)

  def aggregate(self, batch_outs, batch_start=None, batch_end=None):
    if self.use_steps:
      for i, batch_out in enumerate(batch_outs):
        self.results[i].append(batch_out)
    else:
      for i, batch_out in enumerate(batch_outs):
        if composite_tensor_utils.is_composite_or_composite_value(batch_out):
          # This is the case where we're outputting some object (a
          # CompositeTensor or CompositeTensor's value) and so cannot simply
          # convert to a numpy array.
          if self.results[i] == self._DEFER_OUTPUT_ALLOCATION:
            # If there is not yet an element in this output slot, assign the
            # currently-examined batch object.
            self.results[i] = batch_out
          else:
            # If there will be multiple calls to aggregate(), create an array
            # of objects - we cannot assume we know how to combine them.
            self.results[i] = composite_tensor_utils.append_composite_tensor(
                self.results[i], batch_out)
        elif isinstance(batch_out, np.ndarray):
          # In this case, 'results' is an ndarray - we know it's been fully
          # allocated and we can just
          self.results[i][batch_start:batch_end] = batch_out
        else:
          # This is not a ndarray, a CompositeTensor, or a CompositeTensorValue.
          # Fail fast rather than trying to concatenate it.
          raise RuntimeError('Attempted to aggregate unsupported object %s.' %
                             batch_out)

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


def check_num_samples(ins, batch_size=None, steps=None, steps_name='steps'):
  """Determine the number of samples provided for training and evaluation.

  The number of samples is not defined when running with `steps`,
  in which case the number of samples is set to `None`.

  Arguments:
      ins: List of tensors to be fed to the Keras function.
      batch_size: Integer batch size or `None` if not defined.
      steps: Total number of steps (batches of samples) before declaring
        `_predict_loop` finished. Ignored with the default value of `None`.
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
  """
  if steps is not None and batch_size is not None:
    raise ValueError('If ' + steps_name +
                     ' is set, the `batch_size` must be None.')
  if check_steps_argument(ins, steps, steps_name):
    return None
  if hasattr(ins[0], 'shape'):
    return int(ins[0].shape[0])
  return None  # Edge case where ins == [static_learning_phase]


def standardize_single_array(x, expected_shape=None):
  """Expand data of shape (x,) to (x, 1), unless len(expected_shape)==1."""
  if x is None:
    return None

  if (x.shape is not None and len(x.shape) == 1 and
      (expected_shape is None or len(expected_shape) != 1)):
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
      check_batch_axis: Boolean; whether to check that the batch axis of the
        arrays matches the expected value found in `shapes`.
      exception_prefix: String prefix used for exception formatting.

  Returns:
      List of standardized input arrays (one array per model input).

  Raises:
      ValueError: in case of improperly formatted user-provided data.
  """
  if not names:
    if (data is not None and hasattr(data, '__len__') and len(data) and
        not isinstance(data, dict)):
      raise ValueError(
          'Error when checking model ' + exception_prefix + ': '
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
    data = [
        standardize_single_array(x, shape) for (x, shape) in zip(data, shapes)
    ]
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
      raise ValueError('Error when checking model ' + exception_prefix +
                       ': you are passing a list as input to your model, '
                       'but the model expects a list of ' + str(len(names)) +
                       ' Numpy arrays instead. The list you passed was: ' +
                       str(data)[:200])
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
          tensorshape = data[i].shape
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
            raise ValueError('Error when checking ' + exception_prefix +
                             ': expected ' + names[i] + ' to have shape ' +
                             str(shape) + ' but got array with shape ' +
                             str(data_shape))
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
  if isinstance(x_weight, collections.Mapping):
    generic_utils.check_for_unexpected_keys(weight_type, x_weight, output_names)
    x_weights = []
    for name in output_names:
      x_weights.append(x_weight.get(name))
    return x_weights
  else:
    raise TypeError('The model has multiple outputs, so `' + weight_type + '` '
                    'should be either a list or a dict. '
                    'Provided `' + weight_type + '` type not understood: ' +
                    str(x_weight))


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
      return set([
          y.shape[0]
          for y in x
          if y is not None and not tensor_util.is_tensor(y)
      ])

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
  key_loss_fns = {
      losses.mean_squared_error, losses.binary_crossentropy,
      losses.categorical_crossentropy
  }
  key_loss_classes = (losses.MeanSquaredError, losses.BinaryCrossentropy,
                      losses.CategoricalCrossentropy)
  for y, loss, shape in zip(targets, loss_fns, output_shapes):
    if y is None or loss is None or tensor_util.is_tensor(y):
      continue
    if losses.is_categorical_crossentropy(loss):
      if y.shape[-1] == 1:
        raise ValueError('You are passing a target array of shape ' +
                         str(y.shape) +
                         ' while using as loss `categorical_crossentropy`. '
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

    is_loss_wrapper = isinstance(loss, losses.LossFunctionWrapper)
    if (isinstance(loss, key_loss_classes) or (is_loss_wrapper and
                                               (loss.fn in key_loss_fns))):
      for target_dim, out_dim in zip(y.shape[1:], shape[1:]):
        if out_dim is not None and target_dim != out_dim:
          loss_name = loss.name
          if loss_name is None:
            loss_type = loss.fn if is_loss_wrapper else type(loss)
            loss_name = loss_type.__name__
          raise ValueError('A target array with shape ' + str(y.shape) +
                           ' was passed for an output of shape ' + str(shape) +
                           ' while using as loss `' + loss_name + '`. '
                           'This loss expects targets to have the same shape '
                           'as the output.')


def collect_per_output_metric_info(metrics,
                                   output_names,
                                   output_shapes,
                                   loss_fns,
                                   is_weighted=False):
  """Maps metric names and functions to model outputs.

  Arguments:
      metrics: a list or a list of lists or a dict of metric functions.
      output_names: a list of the names (strings) of model outputs.
      output_shapes: a list of the shapes (strings) of model outputs.
      loss_fns: a list of the loss functions corresponding to the model outputs.
      is_weighted: Boolean indicating whether the given metrics are weighted.

  Returns:
      A list (one entry per model output) of dicts.
      For instance, if the model has 2 outputs, and for the first output
      we want to compute "binary_accuracy" and "binary_crossentropy",
      and just "binary_accuracy" for the second output,
      the list would look like: `[{
          'acc': binary_accuracy(),
          'ce': binary_crossentropy(),
        }, {
          'acc': binary_accuracy(),
        }]`

  Raises:
      TypeError: if an incorrect type is passed for the `metrics` argument.
  """
  if not metrics:
    return [{} for _ in output_names]

  if isinstance(metrics, list):
    any_sub_list = any(isinstance(m, list) for m in metrics)
    if any_sub_list:
      if len(metrics) != len(output_names):
        raise ValueError('When passing a list of lists as `metrics`, '
                         'it should have one entry per model output. '
                         'The model has ' + str(len(output_names)) +
                         ' outputs, but you passed metrics=' + str(metrics))
      # User has provided a list of len = len(outputs).
      nested_metrics = [generic_utils.to_list(m) for m in metrics]
    else:
      # If it is a single list we then apply all metrics to all outputs.
      if len(output_names) > 1:
        nested_metrics = []
        for _ in output_names:
          nested_metrics.append(
              [metrics_module.clone_metric(m) for m in metrics])
      else:
        nested_metrics = [metrics]
  elif isinstance(metrics, collections.Mapping):
    generic_utils.check_for_unexpected_keys('metrics', metrics, output_names)
    nested_metrics = []
    for name in output_names:
      output_metrics = generic_utils.to_list(metrics.get(name, []))
      nested_metrics.append(output_metrics)
  else:
    raise TypeError('Type of `metrics` argument not understood. '
                    'Expected a list or dictionary, found: ' + str(metrics))

  per_output_metrics = []
  for i, metrics in enumerate(nested_metrics):
    metrics_dict = OrderedDict()
    for metric in metrics:
      metric_name = get_metric_name(metric, is_weighted)
      metric_fn = get_metric_function(
          metric, output_shape=output_shapes[i], loss_fn=loss_fns[i])

      # If the metric function is not stateful, we create a stateful version.
      if not isinstance(metric_fn, metrics_module.Metric):
        metric_fn = metrics_module.MeanMetricWrapper(
            metric_fn, name=metric_name)
      metrics_dict[metric_name] = metric_fn
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


def standardize_weights(y,
                        sample_weight=None,
                        class_weight=None,
                        sample_weight_mode=None):
  """Performs sample weight validation and standardization.

  Everything gets normalized to a single sample-wise (or timestep-wise)
  weight array. If both `sample_weight` and `class_weight` are provided,
  the weights are multiplied.

  Arguments:
      y: Numpy array of model targets to be weighted.
      sample_weight: User-provided `sample_weight` argument.
      class_weight: User-provided `class_weight` argument.
      sample_weight_mode: One of `None` or `"temporal"`. `"temporal"` indicated
        that we expect 2D weight data that will be applied to the last 2
        dimensions of the targets (i.e. we are weighting timesteps, not
        samples).

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
      raise ValueError('Found a sample_weight with shape' +
                       str(sample_weight.shape) + '.'
                       'Expected sample_weight with rank '
                       'less than or equal to ' + str(len(y.shape)))

    if (not tensor_util.is_tensor(sample_weight) and
        y.shape[:sample_weight.ndim] != sample_weight.shape):
      raise ValueError('Found a sample_weight array with shape ' +
                       str(sample_weight.shape) + ' for an input with shape ' +
                       str(y.shape) + '. '
                       'sample_weight cannot be broadcast.')

  # Class weights applied per-sample.
  class_sample_weight = None
  if isinstance(class_weight, dict):
    if len(y.shape) > 2:
      raise ValueError('`class_weight` not supported for '
                       '3+ dimensional targets.')

    if len(y.shape) == 2:
      if y.shape[1] > 1:
        y_classes = np.argmax(y, axis=1)
      elif y.shape[1] == 1:
        y_classes = np.reshape(y, y.shape[0])
    else:
      y_classes = y

    class_sample_weight = np.asarray(
        [class_weight[cls] for cls in y_classes if cls in class_weight])

    if len(class_sample_weight) != len(y_classes):
      # subtract the sets to pick all missing classes
      existing_classes = set(y_classes)
      existing_class_weight = set(class_weight.keys())
      raise ValueError(
          '`class_weight` must contain all classes in the data.'
          ' The classes %s exist in the data but not in '
          '`class_weight`.' % (existing_classes - existing_class_weight))

  if class_sample_weight is not None and sample_weight is not None:
    # Multiply weights if both are provided.
    return class_sample_weight * sample_weight
  if sample_weight is not None:
    return sample_weight
  if class_sample_weight is not None:
    return class_sample_weight
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
  if tf2.enabled():
    # We keep the string that the user has set in compile as the metric name.
    if isinstance(metric, six.string_types):
      return metric

    metric = metrics_module.get(metric)
    return metric.name if hasattr(metric, 'name') else metric.__name__
  else:
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
      output_shape: The shape of the output that this metric will be calculated
        for.
      loss_fn: The loss function used.

  Returns:
      The metric function.
  """
  if metric not in ['accuracy', 'acc', 'crossentropy', 'ce']:
    return metrics_module.get(metric)

  is_sparse_categorical_crossentropy = (
      isinstance(loss_fn, losses.SparseCategoricalCrossentropy) or
      (isinstance(loss_fn, losses.LossFunctionWrapper) and
       loss_fn.fn == losses.sparse_categorical_crossentropy))

  is_binary_crossentropy = (
      isinstance(loss_fn, losses.BinaryCrossentropy) or
      (isinstance(loss_fn, losses.LossFunctionWrapper) and
       loss_fn.fn == losses.binary_crossentropy))

  if metric in ['accuracy', 'acc']:
    if output_shape[-1] == 1 or is_binary_crossentropy:
      return metrics_module.binary_accuracy
    elif is_sparse_categorical_crossentropy:
      return metrics_module.sparse_categorical_accuracy
    # If the output_shape[-1] is not 1, then we know output is `categorical`.
    # We assume it is sparse categorical only if loss is explicitly given
    # as sparse categorical crossentropy loss.
    return metrics_module.categorical_accuracy
  else:
    if output_shape[-1] == 1 or is_binary_crossentropy:
      return metrics_module.binary_crossentropy
    elif is_sparse_categorical_crossentropy:
      return metrics_module.sparse_categorical_crossentropy
    return metrics_module.categorical_crossentropy


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

  # Deserialize loss configuration, if needed.
  if isinstance(loss, collections.Mapping):
    loss = losses.get(loss)

  # Custom callable class.
  if callable(loss) and not hasattr(loss, '__name__'):
    return loss

  # Wrap loss function with signature `(y_true, y_pred, **kwargs)`
  # in `LossFunctionWrapper` class.
  loss_fn = losses.get(loss)
  return losses.LossFunctionWrapper(loss_fn, name=loss_fn.__name__)


def validate_dataset_input(x, y, sample_weight, validation_split=None):
  """Validates user input arguments when a dataset iterator is passed.

  Arguments:
    x: Input data. A `tf.data` dataset or iterator.
    y: Target data. It could be either Numpy array(s) or TensorFlow tensor(s).
      Expected to be `None` when `x` is a dataset iterator.
    sample_weight: An optional sample-weight array passed by the user to weight
      the importance of each sample in `x`. Expected to be `None` when `x` is a
      dataset iterator
    validation_split: Float between 0 and 1. Fraction of the training data to be
      used as validation data. Expected to be `None` when `x` is a dataset
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


def check_generator_arguments(y=None, sample_weight=None,
                              validation_split=None):
  """Validates arguments passed when using a generator."""
  if y is not None:
    raise ValueError('`y` argument is not supported when data is'
                     'a generator or Sequence instance. Instead pass targets'
                     ' as the second element of the generator.')
  if sample_weight is not None:
    raise ValueError('`sample_weight` argument is not supported when data is'
                     'a generator or Sequence instance. Instead pass sample'
                     ' weights as the third element of the generator.')
  if validation_split:
    raise ValueError('If your data is in the form of a Python generator, '
                     'you cannot use `validation_split`.')


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
  # TODO(fchollet): allow datasets with steps=None if cardinality is known.
  is_x_iterator = isinstance(
      input_data, (iterator_ops.Iterator, iterator_ops.EagerIterator))
  if (input_data is None or is_x_iterator or has_symbolic_tensors(input_data) or
      (isinstance(input_data, list) and not input_data)):
    if steps is None:
      input_type_str = 'a Dataset iterator' if is_x_iterator else 'data tensors'
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
  if isinstance(sample_weight_mode, collections.Mapping):
    generic_utils.check_for_unexpected_keys('sample_weight_mode',
                                            sample_weight_mode, output_names)
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


def prepare_loss_functions(loss, output_names):
  """Converts loss to a list of loss functions.

  Arguments:
      loss: String (name of objective function), objective function or
        `tf.losses.Loss` instance. See `tf.losses`. If the model has multiple
        outputs, you can use a different loss on each output by passing a
        dictionary or a list of losses. The loss value that will be minimized by
        the model will then be the sum of all individual losses.
      output_names: List of model output names.

  Returns:
      A list of loss objective functions.

  Raises:
      ValueError: If loss is a dict with keys not in model output names,
          or if loss is a list with len not equal to model outputs.
  """
  if isinstance(loss, collections.Mapping):
    generic_utils.check_for_unexpected_keys('loss', loss, output_names)
    loss_functions = []
    for name in output_names:
      if name not in loss:
        logging.warning(
            'Output {0} missing from loss dictionary. We assume '
            'this was done on purpose. The fit and evaluate APIs will not be '
            'expecting any data to be passed to {0}.'.format(name))
      loss_functions.append(get_loss_function(loss.get(name, None)))
  elif isinstance(loss, six.string_types):
    loss_functions = [get_loss_function(loss) for _ in output_names]
  elif isinstance(loss, collections.Sequence):
    if len(loss) != len(output_names):
      raise ValueError('When passing a list as loss, it should have one entry '
                       'per model outputs. The model has {} outputs, but you '
                       'passed loss={}'.format(len(output_names), loss))
    loss_functions = nest.map_structure(get_loss_function, loss)
  else:
    loss_functions = [get_loss_function(loss) for _ in range(len(output_names))]

  return loss_functions


def prepare_loss_weights(output_names, loss_weights=None):
  """Converts loss weights to a list of loss weights.

  Arguments:
      output_names: List of model output names.
      loss_weights: Optional list or dictionary specifying scalar coefficients
        (Python floats) to weight the loss contributions of different model
        outputs. The loss value that will be minimized by the model will then be
        the *weighted sum* of all individual losses, weighted by the
          `loss_weights` coefficients. If a list, it is expected to have a 1:1
            mapping to the model's outputs. If a dict, it is expected to map
            output names (strings) to scalar coefficients.

  Returns:
      A list of loss weights of python floats.

  Raises:
      ValueError: If loss weight is a dict with key not in model output names,
          or if loss is a list with len not equal to model outputs.
  """
  if loss_weights is None:
    weights_list = [1.] * len(output_names)
  elif isinstance(loss_weights, collections.Mapping):
    generic_utils.check_for_unexpected_keys('loss_weights', loss_weights,
                                            output_names)
    weights_list = [loss_weights.get(name, 1.) for name in output_names]
  elif isinstance(loss_weights, list):
    if len(loss_weights) != len(output_names):
      raise ValueError('When passing a list as loss_weights, '
                       'it should have one entry per model output. '
                       'The model has ' + str(len(output_names)) +
                       ' outputs, but you passed loss_weights=' +
                       str(loss_weights))
    weights_list = loss_weights
  else:
    raise TypeError('Could not interpret loss_weights argument: ' +
                    str(loss_weights) + ' - expected a list of dicts.')

  return weights_list


# TODO(rohanj): This is a hack to get around not depending on feature_column and
# create a cyclical dependency. Figure out a cleaner solution
def is_feature_layer(layer):
  """Returns whether `layer` is a FeatureLayer or not."""
  return getattr(layer, '_is_feature_layer', False)


def is_eager_dataset_or_iterator(data):
  return context.executing_eagerly() and isinstance(
      data, (dataset_ops.DatasetV1, dataset_ops.DatasetV2,
             iterator_ops.EagerIterator))


# pylint: disable=protected-access
def assert_not_batched(dataset):
  """Asserts that `dataset` is not batched.

  The algorithm used by this method is sound but not complete. In other words,
  if the method fails to establish the assertion, it does not mean the dataset
  is batched.

  Example usage:
  ```python
  try:
    assert_not_batched(dataset)
    # safe to assume `dataset` it not batched here
  expect ValueError:
    # make no assumptions about `dataset`
  ```

  Args:
    dataset: The dataset to analyze.

  Raises:
    ValueError: If the method cannot establish the assertion.
  """
  if isinstance(dataset, dataset_ops.DatasetV1Adapter):
    return assert_not_batched(dataset._dataset)
  else:
    whitelisted_types = [
        dataset_ops._OptionsDataset,
        dataset_ops.ConcatenateDataset,
        dataset_ops.CacheDataset,
        dataset_ops.FilterDataset,
        dataset_ops.MapDataset,
        dataset_ops.ParallelMapDataset,
        dataset_ops.PrefetchDataset,
        dataset_ops.RangeDataset,
        dataset_ops.RepeatDataset,
        dataset_ops.ShuffleDataset,
        dataset_ops.SkipDataset,
        dataset_ops.SparseTensorSliceDataset,
        dataset_ops.TakeDataset,
        dataset_ops.TensorDataset,
        dataset_ops.TensorSliceDataset,
        dataset_ops.ZipDataset,
        readers.FixedLengthRecordDatasetV2,
        readers.TextLineDatasetV2,
        readers.TFRecordDatasetV2,
    ]
    for ty in whitelisted_types:
      if isinstance(dataset, ty):
        for input_dataset in dataset._inputs():
          assert_not_batched(input_dataset)
        return
    raise ValueError('Could not assert that dataset is not batched.')


# pylint: disable=protected-access
def assert_not_shuffled(dataset):
  """Asserts that `dataset` is not shuffled.

  The algorithm used by this method is sound but not complete. In other words,
  if the method fails to establish the assertion, it does not mean the dataset
  is shuffled.

  Example usage:
  ```python
  try:
    assert_not_shuffled(dataset)
    # safe to assume `dataset` it not shuffled here
  expect ValueError:
    # make no assumptions about `dataset`
  ```

  Args:
    dataset: The dataset to analyze.

  Raises:
    ValueError: If the method cannot establish the assertion.
  """
  if isinstance(dataset, dataset_ops.DatasetV1Adapter):
    return assert_not_shuffled(dataset._dataset)
  else:
    whitelisted_types = [
        dataset_ops._OptionsDataset,
        dataset_ops.BatchDataset,
        dataset_ops.ConcatenateDataset,
        dataset_ops.CacheDataset,
        dataset_ops.FilterDataset,
        dataset_ops.MapDataset,
        dataset_ops.PaddedBatchDataset,
        dataset_ops.ParallelMapDataset,
        dataset_ops.PrefetchDataset,
        dataset_ops.RangeDataset,
        dataset_ops.RepeatDataset,
        dataset_ops.SkipDataset,
        dataset_ops.SparseTensorSliceDataset,
        dataset_ops.TakeDataset,
        dataset_ops.TensorDataset,
        dataset_ops.TensorSliceDataset,
        dataset_ops.WindowDataset,
        dataset_ops.ZipDataset,
        readers.FixedLengthRecordDatasetV2,
        readers.TextLineDatasetV2,
        readers.TFRecordDatasetV2,
    ]
    for ty in whitelisted_types:
      if isinstance(dataset, ty):
        for input_dataset in dataset._inputs():
          assert_not_shuffled(input_dataset)
        return
    raise ValueError('Could not assert that dataset is not shuffled.')


def verify_dataset_shuffled(x):
  """Verifies that the dataset is shuffled.

  Args:
    x: Dataset passed as an input to the model.

  Raises:
    ValueError: if the dataset is not already shuffled.
  """
  assert isinstance(x, dataset_ops.DatasetV2)
  try:
    assert_not_shuffled(x)
  except ValueError:
    # Dataset may or may not be shuffled.
    return
  else:
    logging.warning('Expected a shuffled dataset but input dataset `x` is '
                    'not shuffled. Please invoke `shuffle()` on input dataset.')


def is_dataset_or_iterator(data):
  return isinstance(data, (dataset_ops.DatasetV1, dataset_ops.DatasetV2,
                           iterator_ops.EagerIterator, iterator_ops.Iterator))


def get_iterator(dataset):
  """Create and initialize an iterator from a dataset."""
  iterator = dataset_ops.make_initializable_iterator(dataset)
  initialize_iterator(iterator)
  return iterator


def initialize_iterator(iterator):
  init_op = iterator.initializer
  if not context.executing_eagerly():
    K.get_session((init_op,)).run(init_op)


def extract_tensors_from_dataset(dataset):
  """Extract a tuple of tensors `inputs, targets, sample_weight` from a dataset.

  Arguments:
    dataset: Dataset instance.

  Returns:
    Tuple of tensors `x, y, weights`. `y` and `weights` entry may be None.
  """
  iterator = get_iterator(dataset)
  inputs, targets, sample_weight = unpack_iterator_input(iterator)
  return inputs, targets, sample_weight


def unpack_iterator_input(iterator):
  """Convert a dataset iterator to a tuple of tensors `x, y, sample_weights`.

  Arguments:
    iterator: Instance of a dataset iterator.

  Returns:
    Tuple of tensors `x, y, weights`. `y` and `weights` entry may be None.
  """
  try:
    next_element = iterator.get_next()
  except errors.OutOfRangeError:
    raise RuntimeError('Your dataset iterator ran out of data; '
                       'Make sure that your dataset can generate '
                       'required number of samples.')

  if isinstance(next_element, (list, tuple)):
    if len(next_element) not in [2, 3]:
      raise ValueError(
          'Please provide model inputs as a list or tuple of 2 or 3 '
          'elements: (input, target) or (input, target, sample_weights) '
          'Received %s' % next_element)
    if len(next_element) == 2:
      x, y = next_element
      weights = None
    else:
      x, y, weights = next_element
  else:
    x = next_element
    y = None
    weights = None
  return x, y, weights


def infer_steps_for_dataset(dataset, steps, epochs=1, steps_name='steps'):
  """Infers steps_per_epoch needed to loop through a dataset.

  Arguments:
      dataset: Input data of type tf.data.Dataset.
      steps: Number of steps to draw from the dataset (may be None if unknown).
      epochs: Number of times to iterate over the dataset.
      steps_name: The string name of the steps argument, either `steps`,
        `validation_steps`, or `steps_per_epoch`. Only used for error message
        formatting.

  Returns:
    Integer or `None`. Inferred number of steps to loop through the dataset.
    `None` is returned if the size of the dataset is unknown and `steps` was
    not specified.

  Raises:
    ValueError: In case of invalid argument values.
  """
  assert isinstance(dataset, dataset_ops.DatasetV2)
  size = K.get_value(cardinality.cardinality(dataset))
  if size == cardinality.INFINITE and steps is None:
    raise ValueError('When passing an infinitely repeating dataset, you '
                     'must specify the `%s` argument.' % (steps_name,))
  if size >= 0:
    if steps is not None and steps * epochs > size:
      if epochs > 1:
        raise ValueError('The dataset you passed contains %s batches, but you '
                         'passed `epochs=%s` and `%s=%s`, which is a total of '
                         '%s steps. We cannot draw that many steps from this '
                         'dataset. We suggest to set `%s=%s`.' %
                         (size, epochs, steps_name, steps, steps * epochs,
                          steps_name, size // epochs))
      else:
        raise ValueError('The dataset you passed contains %s batches, but you '
                         'passed `%s=%s`. We cannot draw that many steps from '
                         'this dataset. We suggest to set `%s=%s`.' %
                         (size, steps_name, steps, steps_name, size))
  if steps is None:
    if size >= 0:
      return size
    return None
  return steps


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
        dtype = dtypes.as_dtype(v.dtype)
        if dtype.is_floating:
          dtype = K.floatx()
        v = K.placeholder(shape=shape, name=k, dtype=dtype)
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
    ValueError: in case an empty Sequential or Functional model is passed.
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


def convert_eager_tensors_to_numpy(structure):
  """Convert every EagerTensor in `structure` to NumPy.

  Arguments:
    structure: An arbitrary structure of elements to be converted to NumPy
      arrays.

  Returns:
    An identical structure with EagerTensors converted to NumPy arrays.
  """

  def _convert(element):
    if isinstance(element, ops.EagerTensor):
      return element.numpy()
    return element

  return nest.map_structure(_convert, structure)


def should_run_validation(validation_freq, epoch):
  """Checks if validation should be run this epoch.

  Arguments:
    validation_freq: Integer or list. If an integer, specifies how many training
      epochs to run before a new validation run is performed. If a list,
      specifies the epochs on which to run validation.
    epoch: Integer, the number of the training epoch just completed.

  Returns:
    Bool, True if validation should be run.

  Raises:
    ValueError: if `validation_freq` is an Integer and less than 1, or if
    it is neither an Integer nor a Sequence.
  """
  # `epoch` is 0-indexed internally but 1-indexed in the public API.
  one_indexed_epoch = epoch + 1

  if isinstance(validation_freq, int):
    if validation_freq < 1:
      raise ValueError('`validation_freq` can not be less than 1.')
    return one_indexed_epoch % validation_freq == 0

  if not isinstance(validation_freq, collections.Container):
    raise ValueError('`validation_freq` must be an Integer or '
                     '`collections.Container` (e.g. list, tuple, etc.)')
  return one_indexed_epoch in validation_freq

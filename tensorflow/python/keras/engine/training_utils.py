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

import copy

import numpy as np

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.ops import math_ops


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


def standardize_single_array(x):
  if x is None:
    return None
  elif tensor_util.is_tensor(x):
    return x
  elif x.ndim == 1:
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
    except KeyError as e:
      raise ValueError('No data provided for "' + e.args[0] + '". Need data '
                       'for each key in: ' + str(names))
  elif isinstance(data, (list, tuple)):
    if isinstance(data[0], list):
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


def collect_metrics(metrics, output_names):
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
      # Cast the mask to floatX to avoid float64 upcasting in theano
      mask = math_ops.cast(mask, K.floatx())
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
      score_array /= K.mean(
          math_ops.cast(math_ops.not_equal(weights, 0), K.floatx()))
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
  return tensor_util.is_tensor(ls)


def populate_metric_names(model):
  for i in range(len(model.outputs)):
    metrics = model.nested_metrics[i]
    for metric in metrics:
      base_metric_name = get_base_metric_name(metric)
      add_metric_name(model, base_metric_name, i)


def get_base_metric_name(metric, weighted=False):
  """Returns the metric name given the metric function.

  Arguments:
      metric: Metric function name or reference.
      weighted: Boolean indicating if the metric for which we are adding
          names is weighted.

  Returns:
      a metric name.
  """
  metric_name_prefix = 'weighted_' if weighted else ''
  if metric in ('accuracy', 'acc', 'crossentropy', 'ce'):
    if metric in ('accuracy', 'acc'):
      suffix = 'acc'
    elif metric in ('crossentropy', 'ce'):
      suffix = 'ce'
    metric_name = metric_name_prefix + suffix
  else:
    metric_fn = metrics_module.get(metric)
    # Get metric name as string
    if hasattr(metric_fn, 'name'):
      metric_name = metric_fn.name
    else:
      metric_name = metric_fn.__name__
    metric_name = metric_name_prefix + metric_name

  return metric_name


def add_metric_name(model, metric_name, index):
  """Makes the metric name unique and adds it to the model's metric name list.

    If there are multiple outputs for which the metrics are calculated, the
    metric names have to be made unique by appending an integer.

  Arguments:
    model: Model to which we are adding metric names.
    metric_name: Metric name that corresponds to the metric specified by the
        user. For example: 'acc'
    index: The index of the model output for which the metric name is being
        added.
  """
  if len(model.output_names) > 1:
    metric_name = '%s_%s' % (model.output_names[index], metric_name)
  j = 1
  base_metric_name = metric_name
  while metric_name in model.metrics_names:
    metric_name = '%s_%d' % (base_metric_name, j)
    j += 1
  model.metrics_names.append(metric_name)


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
                     '`x` is a dataset or a dataset iterator. '
                     'Received: x=%s, sample_weight=%s' % (x, sample_weight))
  if validation_split is not None and validation_split != 0.0:
    raise ValueError(
        '`validation_split` argument is not supported when '
        'input `x` is a dataset or a dataset iterator. '
        'Received: x=%s, validation_split=%f' % (x, validation_split))


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

  if isinstance(x, (list, tuple)):
    return [
        math_ops.cast(val, dtype=K.floatx())
        if tensor_util.is_tensor(val) and val.dtype.is_floating else val
        for val in x
    ]
  return math_ops.cast(x, dtype=K.floatx()) if x.dtype.is_floating else x

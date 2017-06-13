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
"""Keras training and evaluation routines.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import multiprocessing
import threading
import time

import numpy as np
import six

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import callbacks as cbks
from tensorflow.contrib.keras.python.keras import losses
from tensorflow.contrib.keras.python.keras import metrics as metrics_module
from tensorflow.contrib.keras.python.keras import optimizers
from tensorflow.contrib.keras.python.keras.engine.topology import Container
from tensorflow.contrib.keras.python.keras.utils.generic_utils import Progbar
from tensorflow.python.platform import tf_logging as logging


# pylint: disable=g-import-not-at-top
try:
  import queue
except ImportError:
  import Queue as queue
# pylint: enable=g-import-not-at-top


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
    return []
  if data is None:
    return [None for _ in range(len(names))]
  if isinstance(data, dict):
    arrays = []
    for name in names:
      if name not in data:
        raise ValueError('No data provided for "' + name +
                         '". Need data for each key in: ' + str(names))
      arrays.append(data[name])
  elif isinstance(data, list):
    if len(data) != len(names):
      if data and hasattr(data[0], 'shape'):
        raise ValueError(
            'Error when checking model ' + exception_prefix +
            ': the list of Numpy arrays '
            'that you are passing to your model '
            'is not the size the model expected. '
            'Expected to see ' + str(len(names)) + ' arrays but instead got '
            'the following list of ' + str(len(data)) + ' arrays: ' +
            str(data)[:200] + '...')
      else:
        if len(names) == 1:
          data = [np.asarray(data)]
        else:
          raise ValueError('Error when checking model ' + exception_prefix +
                           ': you are passing a list as '
                           'input to your model, '
                           'but the model expects '
                           'a list of ' + str(len(names)) +
                           ' Numpy arrays instead. '
                           'The list you passed was: ' + str(data)[:200])
    arrays = data
  else:
    if not hasattr(data, 'shape'):
      raise TypeError('Error when checking model ' + exception_prefix +
                      ': data should be a Numpy array, '
                      'or list/dict of Numpy arrays. '
                      'Found: ' + str(data)[:200] + '...')
    if len(names) > 1:
      # Case: model expects multiple inputs but only received
      # a single Numpy array.
      raise ValueError('The model expects ' + str(len(names)) + exception_prefix
                       + ' arrays, but only received one array. '
                       'Found: array with shape ' + str(data.shape))
    arrays = [data]

  # Make arrays at least 2D.
  for i in range(len(names)):
    array = arrays[i]
    if len(array.shape) == 1:
      array = np.expand_dims(array, 1)
      arrays[i] = array

  # Check shapes compatibility.
  if shapes:
    for i in range(len(names)):
      if shapes[i] is None:
        continue
      array = arrays[i]
      if len(array.shape) != len(shapes[i]):
        raise ValueError(
            'Error when checking ' + exception_prefix + ': expected ' + names[i]
            + ' to have ' + str(len(shapes[i])) +
            ' dimensions, but got array with shape ' + str(array.shape))
      for j, (dim, ref_dim) in enumerate(zip(array.shape, shapes[i])):
        if not j and not check_batch_axis:
          # skip the first axis
          continue
        if ref_dim:
          if ref_dim != dim:
            raise ValueError('Error when checking ' + exception_prefix +
                             ': expected ' + names[i] + ' to have shape ' +
                             str(shapes[i]) + ' but got array with shape ' +
                             str(array.shape))
  return arrays


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
    raise TypeError('The model has multiple outputs, so `' + weight_type + '` '
                    'should be either a list of a dict. '
                    'Provided `' + weight_type + '` type not understood: ' +
                    str(x_weight))


def _standardize_class_weights(class_weight, output_names):
  return _standardize_sample_or_class_weights(class_weight, output_names,
                                              'class_weight')


def _standardize_sample_weights(sample_weight, output_names):
  return _standardize_sample_or_class_weights(sample_weight, output_names,
                                              'sample_weight')


def _check_array_lengths(inputs, targets, weights):
  """Does user input validation for numpy arrays.

  Arguments:
      inputs: list of Numpy arrays of inputs.
      targets: list of Numpy arrays of targets.
      weights: list of Numpy arrays of sample weights.

  Raises:
      ValueError: in case of incorrectly formatted data.
  """
  x_lengths = [x.shape[0] for x in inputs]
  y_lengths = [y.shape[0] for y in targets]
  w_lengths = [w.shape[0] for w in weights]
  set_x = set(x_lengths)
  if len(set_x) > 1:
    raise ValueError('All input arrays (x) should have '
                     'the same number of samples. Got array shapes: ' + str(
                         [x.shape for x in inputs]))
  set_y = set(y_lengths)
  if len(set_y) > 1:
    raise ValueError('All target arrays (y) should have '
                     'the same number of samples. Got array shapes: ' + str(
                         [y.shape for y in targets]))
  set_w = set(w_lengths)
  if len(set_w) > 1:
    raise ValueError('All sample_weight arrays should have '
                     'the same number of samples. Got array shapes: ' + str(
                         [w.shape for w in weights]))
  if set_x and set_y and list(set_x)[0] != list(set_y)[0]:
    raise ValueError('Input arrays should have '
                     'the same number of samples as target arrays. '
                     'Found ' + str(list(set_x)[0]) + ' input samples '
                     'and ' + str(list(set_y)[0]) + ' target samples.')
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
      'mean_square_error', 'binary_crossentropy', 'categorical_crossentropy'
  }
  for y, loss, shape in zip(targets, loss_fns, output_shapes):
    if loss is None:
      continue
    if loss.__name__ == 'categorical_crossentropy':
      if y.shape[-1] == 1:
        raise ValueError('You are passing a target array of shape ' + str(
            y.shape) + ' while using as loss `categorical_crossentropy`. '
                         '`categorical_crossentropy` expects '
                         'targets to be binary matrices (1s and 0s) '
                         'of shape (samples, classes). '
                         'If your targets are integer classes, '
                         'you can convert them to the expected format via:\n'
                         '```\n'
                         'from keras.utils.np_utils import to_categorical\n'
                         'y_binary = to_categorical(y_int)\n'
                         '```\n'
                         '\n'
                         'Alternatively, you can use the loss function '
                         '`sparse_categorical_crossentropy` instead, '
                         'which does expect integer targets.')
    if loss.__name__ in key_losses:
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
  num_batches = int(np.ceil(size / float(batch_size)))
  return [(i * batch_size, min(size, (i + 1) * batch_size))
          for i in range(0, num_batches)]


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
  if isinstance(arrays, list):
    if hasattr(start, '__len__'):
      # hdf5 datasets only support list objects as indices
      if hasattr(start, 'shape'):
        start = start.tolist()
      return [x[start] for x in arrays]
    else:
      return [x[start:stop] for x in arrays]
  else:
    if hasattr(start, '__len__'):
      if hasattr(start, 'shape'):
        start = start.tolist()
      return arrays[start]
    else:
      return arrays[start:stop]


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
      mask = K.cast(mask, K.floatx())
      # mask should have the same shape as score_array
      score_array *= mask
      #  the loss per batch should be proportional
      #  to the number of unmasked samples.
      score_array /= K.mean(mask)

    # reduce score_array to same ndim as weight array
    ndim = K.ndim(score_array)
    weight_ndim = K.ndim(weights)
    score_array = K.mean(score_array, axis=list(range(weight_ndim, ndim)))

    # apply sample weighting
    if weights is not None:
      score_array *= weights
      score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
    return K.mean(score_array)

  return weighted


def _masked_objective(fn):
  """Adds support for masking to an objective function.

  It transforms an objective function `fn(y_true, y_pred)`
  into a cost-masked objective function
  `fn(y_true, y_pred, mask)`.

  Arguments:
      fn: The objective function to wrap,
          with signature `fn(y_true, y_pred)`.

  Returns:
      A function with signature `fn(y_true, y_pred, mask)`.
  """

  def masked(y_true, y_pred, mask=None):
    """Wrapper function.

    Arguments:
        y_true: `y_true` argument of `fn`.
        y_pred: `y_pred` argument of `fn`.
        mask: Mask tensor.

    Returns:
        Scalar tensor.
    """
    # score_array has ndim >= 2
    score_array = fn(y_true, y_pred)
    if mask is not None:
      mask = K.cast(mask, K.floatx())
      # mask should have the same shape as score_array
      score_array *= mask
      #  the loss per batch should be proportional
      #  to the number of unmasked samples.
      score_array /= K.mean(mask)

    return K.mean(score_array)

  return masked


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
      raise ValueError('Found a sample_weight with shape' +
                       str(sample_weight.shape) + '.'
                       'Expected sample_weight with rank '
                       'less than or equal to ' + str(len(y.shape)))

    if y.shape[:sample_weight.ndim] != sample_weight.shape:
      raise ValueError('Found a sample_weight array with shape ' +
                       str(sample_weight.shape) + ' for an input with shape ' +
                       str(y.shape) + '. '
                       'sample_weight cannot be broadcast.')
    return sample_weight
  elif isinstance(class_weight, dict):
    if len(y.shape) > 2:
      raise ValueError('class_weight not supported for '
                       '3+ dimensional targets.')
    if y.shape[1] > 1:
      y_classes = y.argmax(axis=1)
    elif y.shape[1] == 1:
      y_classes = np.reshape(y, y.shape[0])
    else:
      y_classes = y
    weights = np.asarray([class_weight[cls] for cls in y_classes])
    return weights
  else:
    if sample_weight_mode is None:
      return np.ones((y.shape[0],), dtype=K.floatx())
    else:
      return np.ones((y.shape[0], y.shape[1]), dtype=K.floatx())


class GeneratorEnqueuer(object):
  """Builds a queue out of a data generator.

  Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

  Arguments:
      generator: a generator function which endlessly yields data
      pickle_safe: use multiprocessing if True, otherwise threading
  """

  def __init__(self, generator, pickle_safe=False):
    self._generator = generator
    self._pickle_safe = pickle_safe
    self._threads = []
    self._stop_event = None
    self.queue = None

  def start(self, workers=1, max_q_size=10, wait_time=0.05):
    """Kicks off threads which add data from the generator into the queue.

    Arguments:
        workers: number of worker threads
        max_q_size: queue size (when full, threads could block on put())
        wait_time: time to sleep in-between calls to put()
    """

    def data_generator_task():
      while not self._stop_event.is_set():
        try:
          if self._pickle_safe or self.queue.qsize() < max_q_size:
            generator_output = next(self._generator)
            self.queue.put(generator_output)
          else:
            time.sleep(wait_time)
        except Exception:
          self._stop_event.set()
          raise

    try:
      if self._pickle_safe:
        self.queue = multiprocessing.Queue(maxsize=max_q_size)
        self._stop_event = multiprocessing.Event()
      else:
        self.queue = queue.Queue()
        self._stop_event = threading.Event()

      for _ in range(workers):
        if self._pickle_safe:
          # Reset random seed else all children processes
          # share the same seed
          np.random.seed()
          thread = multiprocessing.Process(target=data_generator_task)
          thread.daemon = True
        else:
          thread = threading.Thread(target=data_generator_task)
        self._threads.append(thread)
        thread.start()
    except:
      self.stop()
      raise

  def is_running(self):
    return self._stop_event is not None and not self._stop_event.is_set()

  def stop(self, timeout=None):
    """Stop running threads and wait for them to exit, if necessary.

    Should be called by the same thread which called start().

    Arguments:
        timeout: maximum time to wait on thread.join()
    """
    if self.is_running():
      self._stop_event.set()

    for thread in self._threads:
      if thread.is_alive():
        if self._pickle_safe:
          thread.terminate()
        else:
          thread.join(timeout)

    if self._pickle_safe:
      if self.queue is not None:
        self.queue.close()

    self._threads = []
    self._stop_event = None
    self.queue = None


class Model(Container):
  """The `Model` class adds training & evaluation routines to a `Container`.
  """

  def compile(self,
              optimizer,
              loss,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              **kwargs):
    """Configures the model for training.

    Arguments:
        optimizer: str (name of optimizer) or optimizer object.
            See [optimizers](/optimizers).
        loss: str (name of objective function) or objective function.
            See [losses](/losses).
            If the model has multiple outputs, you can use a different loss
            on each output by passing a dictionary or a list of losses.
            The loss value that will be minimized by the model
            will then be the sum of all individual losses.
        metrics: list of metrics to be evaluated by the model
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
        sample_weight_mode: if you need to do timestep-wise
            sample weighting (2D weights), set this to `"temporal"`.
            `None` defaults to sample-wise weights (1D).
            If the model has multiple outputs, you can use a different
            `sample_weight_mode` on each output by passing a
            dictionary or a list of modes.
        **kwargs: Additional arguments passed to `tf.Session.run`.

    Raises:
        ValueError: In case of invalid arguments for
            `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
        RuntimeError: If the model has no loss to optimize.
    """
    loss = loss or {}
    self.optimizer = optimizers.get(optimizer)
    self.sample_weight_mode = sample_weight_mode
    self.loss = loss
    self.loss_weights = loss_weights

    # Prepare loss functions.
    if isinstance(loss, dict):
      for name in loss:
        if name not in self.output_names:
          raise ValueError('Unknown entry in loss '
                           'dictionary: "' + name + '". '
                           'Only expected the following keys: ' +
                           str(self.output_names))
      loss_functions = []
      for name in self.output_names:
        if name not in loss:
          logging.warning(
              'Output "' + name + '" missing from loss dictionary. '
              'We assume this was done on purpose, '
              'and we will not be expecting '
              'any data to be passed to "' + name + '" during training.',
              stacklevel=2)
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
    skip_indices = []
    self._feed_outputs = []
    self._feed_output_names = []
    self._feed_output_shapes = []
    self._feed_loss_fns = []
    for i in range(len(weighted_losses)):
      if weighted_losses[i] is None:
        skip_indices.append(i)
      else:
        self._feed_outputs.append(self.outputs[i])
        self._feed_output_names.append(self.output_names[i])
        self._feed_output_shapes.append(self.internal_output_shapes[i])
        self._feed_loss_fns.append(self.loss_functions[i])

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
          raise ValueError('Unknown entry in loss_weights '
                           'dictionary: "' + name + '". '
                           'Only expected the following keys: ' +
                           str(self.output_names))
      loss_weights_list = []
      for name in self.output_names:
        loss_weights_list.append(loss_weights.get(name, 1.))
    elif isinstance(loss_weights, list):
      if len(loss_weights) != len(self.outputs):
        raise ValueError('When passing a list as loss_weights, '
                         'it should have one entry per model outputs. '
                         'The model has ' + str(len(self.outputs)) +
                         ' outputs, but you passed loss_weights=' +
                         str(loss_weights))
      loss_weights_list = loss_weights
    else:
      raise TypeError('Could not interpret loss_weights argument: ' +
                      str(loss_weights) + ' - expected a list of dicts.')

    # Prepare sample weights.
    sample_weights = []
    sample_weight_modes = []
    if isinstance(sample_weight_mode, dict):
      for name in sample_weight_mode:
        if name not in self.output_names:
          raise ValueError('Unknown entry in '
                           'sample_weight_mode dictionary: "' + name + '". '
                           'Only expected the following keys: ' +
                           str(self.output_names))
      for i, name in enumerate(self.output_names):
        if i in skip_indices:
          weight = None
          sample_weight_modes.append(None)
        else:
          if name not in sample_weight_mode:
            raise ValueError('Output "' + name +
                             '" missing from sample_weight_modes '
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
                         'it should have one entry per model outputs. '
                         'The model has ' + str(len(self.outputs)) +
                         ' outputs, but you passed '
                         'sample_weight_mode=' + str(sample_weight_mode))
      for i in range(len(self.output_names)):
        if i in skip_indices:
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
        if i in skip_indices:
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
      if i not in skip_indices:
        self._feed_sample_weight_modes.append(self.sample_weight_modes[i])

    # Prepare targets of model.
    self.targets = []
    self._feed_targets = []
    for i in range(len(self.outputs)):
      if i in skip_indices:
        self.targets.append(None)
      else:
        shape = self.internal_output_shapes[i]
        name = self.output_names[i]
        target = K.placeholder(
            ndim=len(shape),
            name=name + '_target',
            sparse=K.is_sparse(self.outputs[i]),
            dtype=K.dtype(self.outputs[i]))
        self.targets.append(target)
        self._feed_targets.append(target)

    # Prepare metrics.
    self.metrics = metrics
    self.metrics_names = ['loss']
    self.metrics_tensors = []

    # Compute total loss.
    total_loss = None
    for i in range(len(self.outputs)):
      if i in skip_indices:
        continue
      y_true = self.targets[i]
      y_pred = self.outputs[i]
      weighted_loss = weighted_losses[i]
      sample_weight = sample_weights[i]
      mask = masks[i]
      loss_weight = loss_weights_list[i]
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
        raise RuntimeError('The model cannot be compiled '
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

    def append_metric(layer_num, metric_name, metric_tensor):
      """Helper function used in loop below."""
      if len(self.output_names) > 1:
        metric_name = self.output_layers[layer_num].name + '_' + metric_name
      self.metrics_names.append(metric_name)
      self.metrics_tensors.append(metric_tensor)

    for i in range(len(self.outputs)):
      if i in skip_indices:
        continue
      y_true = self.targets[i]
      y_pred = self.outputs[i]
      output_metrics = nested_metrics[i]
      for metric in output_metrics:
        if metric == 'accuracy' or metric == 'acc':
          # custom handling of accuracy
          # (because of class mode duality)
          output_shape = self.internal_output_shapes[i]
          acc_fn = None
          if (output_shape[-1] == 1 or
              self.loss_functions[i] == losses.binary_crossentropy):
            # case: binary accuracy
            acc_fn = metrics_module.binary_accuracy
          elif self.loss_functions[i] == losses.sparse_categorical_crossentropy:
            # case: categorical accuracy with sparse targets
            acc_fn = metrics_module.sparse_categorical_accuracy
          else:
            acc_fn = metrics_module.categorical_accuracy

          masked_fn = _masked_objective(acc_fn)
          append_metric(i, 'acc', masked_fn(y_true, y_pred, mask=masks[i]))
        else:
          metric_fn = metrics_module.get(metric)
          masked_metric_fn = _masked_objective(metric_fn)
          metric_result = masked_metric_fn(y_true, y_pred, mask=masks[i])
          metric_result = {metric_fn.__name__: metric_result}
          for name, tensor in six.iteritems(metric_result):
            append_metric(i, name, tensor)

    # Prepare gradient updates and state updates.
    self.total_loss = total_loss
    self.sample_weights = sample_weights
    self._feed_sample_weights = []
    for i in range(len(self.sample_weights)):
      if i not in skip_indices:
        self._feed_sample_weights.append(sample_weights[i])

    # Functions for train, test and predict will
    # be compiled lazily when required.
    # This saves time when the user is not using all functions.
    self.train_function = None
    self.test_function = None
    self.predict_function = None
    self._function_kwargs = kwargs

    # Collected trainable weights and sort them deterministically.
    trainable_weights = self.trainable_weights
    # Sort weights by name.
    if trainable_weights:
      trainable_weights.sort(key=lambda x: x.name)
    self._collected_trainable_weights = trainable_weights

  def _make_train_function(self):
    if not hasattr(self, 'train_function'):
      raise RuntimeError('You must compile your model before using it.')
    if self.train_function is None:
      inputs = (
          self._feed_inputs + self._feed_targets + self._feed_sample_weights)
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs += [K.learning_phase()]

      training_updates = self.optimizer.get_updates(
          self._collected_trainable_weights, self.constraints, self.total_loss)
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
      inputs = (
          self._feed_inputs + self._feed_targets + self._feed_sample_weights)
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
      self._function_kwargs = {}
    if self.predict_function is None:
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        inputs = self._feed_inputs + [K.learning_phase()]
      else:
        inputs = self._feed_inputs
      # Gets network outputs. Does not update weights.
      # Does update the network states.
      self.predict_function = K.function(
          inputs,
          self.outputs,
          updates=self.state_updates,
          name='predict_function',
          **self._function_kwargs)

  def _fit_loop(self,
                f,
                ins,
                out_labels=None,
                batch_size=32,
                epochs=100,
                verbose=1,
                callbacks=None,
                val_f=None,
                val_ins=None,
                shuffle=True,
                callback_metrics=None,
                initial_epoch=0):
    """Abstract fit function for `f(ins)`.

    Assume that f returns a list, labeled by out_labels.

    Arguments:
        f: Keras function returning a list of tensors
        ins: list of tensors to be fed to `f`
        out_labels: list of strings, display names of
            the outputs of `f`
        batch_size: integer batch size
        epochs: number of times to iterate over the data
        verbose: verbosity mode, 0, 1 or 2
        callbacks: list of callbacks to be called during training
        val_f: Keras function to call for validation
        val_ins: list of tensors to be fed to `val_f`
        shuffle: whether to shuffle the data at the beginning of each epoch
        callback_metrics: list of strings, the display names of the metrics
            passed to the callbacks. They should be the
            concatenation of list the display names of the outputs of
             `f` and the list of display names of the outputs of `f_val`.
        initial_epoch: epoch at which to start training
            (useful for resuming a previous training run)

    Returns:
        `History` object.
    """
    do_validation = False
    if val_f and val_ins:
      do_validation = True
      if verbose:
        print('Train on %d samples, validate on %d samples' %
              (ins[0].shape[0], val_ins[0].shape[0]))

    if ins and hasattr(ins[0], 'shape'):
      num_train_samples = ins[0].shape[0]
    else:
      # May happen if we are running `fit` without Numpy input data,
      # i.e. if all inputs to the models are data tensors
      # instead of placeholders.
      # In that case we will run `fit` over a single batch.
      num_train_samples = batch_size
      verbose = 2
    index_array = np.arange(num_train_samples)

    self.history = cbks.History()
    callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
    if verbose:
      callbacks += [cbks.ProgbarLogger()]
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
        'samples': num_train_samples,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics or [],
    })
    callbacks.on_train_begin()
    callback_model.stop_training = False
    for cbk in callbacks:
      cbk.validation_data = val_ins

    for epoch in range(initial_epoch, epochs):
      callbacks.on_epoch_begin(epoch)
      if shuffle == 'batch':
        index_array = _batch_shuffle(index_array, batch_size)
      elif shuffle:
        np.random.shuffle(index_array)

      batches = _make_batches(num_train_samples, batch_size)
      epoch_logs = {}
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

  def _predict_loop(self, f, ins, batch_size=32, verbose=0):
    """Abstract method to loop over some data in batches.

    Arguments:
        f: Keras function returning a list of tensors.
        ins: list of tensors to be fed to `f`.
        batch_size: integer batch size.
        verbose: verbosity mode.

    Returns:
        Array of predictions (if the model has a single output)
        or list of arrays of predictions
        (if the model has multiple outputs).
    """
    if ins and hasattr(ins[0], 'shape'):
      samples = ins[0].shape[0]
    else:
      # May happen if we are running `predict` without Numpy input data,
      # i.e. if all inputs to the models are data tensors
      # instead of placeholders.
      # In that case we will run `predict` over a single batch.
      samples = batch_size
      verbose = 2
    outs = []
    if verbose == 1:
      progbar = Progbar(target=samples)
    batches = _make_batches(samples, batch_size)
    index_array = np.arange(samples)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
      batch_ids = index_array[batch_start:batch_end]
      if ins and isinstance(ins[-1], float):
        # Do not slice the training phase flag.
        ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
      else:
        ins_batch = _slice_arrays(ins, batch_ids)

      batch_outs = f(ins_batch)
      if not isinstance(batch_outs, list):
        batch_outs = [batch_outs]
      if batch_index == 0:
        for batch_out in batch_outs:
          shape = (samples,) + batch_out.shape[1:]
          outs.append(np.zeros(shape, dtype=batch_out.dtype))

      for i, batch_out in enumerate(batch_outs):
        outs[i][batch_start:batch_end] = batch_out
      if verbose == 1:
        progbar.update(batch_end)
    if len(outs) == 1:
      return outs[0]
    return outs

  def _test_loop(self, f, ins, batch_size=32, verbose=0):
    """Abstract method to loop over some data in batches.

    Arguments:
        f: Keras function returning a list of tensors.
        ins: list of tensors to be fed to `f`.
        batch_size: integer batch size.
        verbose: verbosity mode.

    Returns:
        Scalar loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
    """
    if ins and hasattr(ins[0], 'shape'):
      samples = ins[0].shape[0]
    else:
      # May happen if we are running `evaluate` without Numpy input data,
      # i.e. if all inputs to the models are data tensors
      # instead of placeholders.
      # In that case we will run `evaluate` over a single batch.
      samples = batch_size
      verbose = 2

    outs = []
    if verbose == 1:
      progbar = Progbar(target=samples)
    batches = _make_batches(samples, batch_size)
    index_array = np.arange(samples)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
      batch_ids = index_array[batch_start:batch_end]
      if isinstance(ins[-1], float):
        # Do not slice the training phase flag.
        ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
      else:
        ins_batch = _slice_arrays(ins, batch_ids)

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
      outs[i] /= samples
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
      if loss_fn.__name__ == 'sparse_categorical_crossentropy':
        output_shapes.append(output_shape[:-1] + (1,))
      elif getattr(losses, loss_fn.__name__, None) is None:
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
          batch_size=32,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0):
    """Trains the model for a fixed number of epochs (iterations on a dataset).

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
        batch_size: integer. Number of samples per gradient update.
        epochs: integer, the number of times to iterate
            over the training data arrays.
        verbose: 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = verbose, 2 = one log line per epoch.
        callbacks: list of callbacks to be called during training.
            See [callbacks](/callbacks).
        validation_split: float between 0 and 1:
            fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate
            the loss and any model metrics
            on this data at the end of each epoch.
        validation_data: data on which to evaluate
            the loss and any model metrics
            at the end of each epoch. The model will not
            be trained on this data.
            This could be a tuple (x_val, y_val)
            or a tuple (x_val, y_val, val_sample_weights).
        shuffle: boolean, whether to shuffle the training data
            before each epoch.
        class_weight: optional dictionary mapping
            class indices (integers) to
            a weight (float) to apply to the model's loss for the samples
            from this class during training.
            This can be useful to tell the model to "pay more attention" to
            samples from an under-represented class.
        sample_weight: optional array of the same length as x, containing
            weights to apply to the model's loss for each sample.
            In the case of temporal data, you can pass a 2D array
            with shape (samples, sequence_length),
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            sample_weight_mode="temporal" in compile().
        initial_epoch: epoch at which to start training
            (useful for resuming a previous training run)

    Returns:
        A `History` instance. Its `history` attribute contains
        all information collected during training.

    Raises:
        ValueError: In case of mismatch between the provided input data
            and what the model expects.
    """
    # Validate user data.
    x, y, sample_weights = self._standardize_user_data(
        x,
        y,
        sample_weight=sample_weight,
        class_weight=class_weight,
        check_batch_axis=False,
        batch_size=batch_size)
    # Prepare validation data.
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
      self._make_test_function()
      val_f = self.test_function
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        val_ins = val_x + val_y + val_sample_weights + [0.]
      else:
        val_ins = val_x + val_y + val_sample_weights

    elif validation_split and 0. < validation_split < 1.:
      do_validation = True
      split_at = int(len(x[0]) * (1. - validation_split))
      x, val_x = (_slice_arrays(x, 0, split_at), _slice_arrays(x, split_at))
      y, val_y = (_slice_arrays(y, 0, split_at), _slice_arrays(y, split_at))
      sample_weights, val_sample_weights = (_slice_arrays(
          sample_weights, 0, split_at), _slice_arrays(sample_weights, split_at))
      self._make_test_function()
      val_f = self.test_function
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        val_ins = val_x + val_y + val_sample_weights + [0.]
      else:
        val_ins = val_x + val_y + val_sample_weights
    else:
      do_validation = False
      val_f = None
      val_ins = None

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
      callback_metrics = copy.copy(out_labels) + [
          'val_' + n for n in out_labels
      ]
    else:
      callback_metrics = copy.copy(out_labels)

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
        initial_epoch=initial_epoch)

  def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):
    """Returns the loss value & metrics values for the model in test mode.

    Computation is done in batches.

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
        batch_size: integer. Number of samples per gradient update.
        verbose: verbosity mode, 0 or 1.
        sample_weight: Array of weights to weight the contribution
            of different samples to the loss and metrics.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
    """
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
    return self._test_loop(f, ins, batch_size=batch_size, verbose=verbose)

  def predict(self, x, batch_size=32, verbose=0):
    """Generates output predictions for the input samples.

    Computation is done in batches.

    Arguments:
        x: the input data, as a Numpy array
            (or list of Numpy arrays if the model has multiple outputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.

    Returns:
        Numpy array(s) of predictions.

    Raises:
        ValueError: In case of mismatch between the provided
            input data and the model's expectations,
            or in case a stateful model receives a number of samples
            that is not a multiple of the batch size.
    """
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
    return self._predict_loop(f, ins, batch_size=batch_size, verbose=verbose)

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
        sample_weight: optional array of the same length as x, containing
            weights to apply to the model's loss for each sample.
            In the case of temporal data, you can pass a 2D array
            with shape (samples, sequence_length),
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            sample_weight_mode="temporal" in compile().
        class_weight: optional dictionary mapping
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
        sample_weight: optional array of the same length as x, containing
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
    """Fits the model on data yielded batch-by-batch by a Python generator.

    The generator is run in parallel to the model, for efficiency.
    For instance, this allows you to do real-time data augmentation
    on images on CPU in parallel to training your model on GPU.

    Arguments:
        generator: a generator.
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
            be equal to the number of unique samples if your dataset
            divided by the batch size.
        epochs: integer, total number of iterations on the data.
        verbose: verbosity mode, 0, 1, or 2.
        callbacks: list of callbacks to be called during training.
        validation_data: this can be either
            - a generator for the validation data
            - a tuple (inputs, targets)
            - a tuple (inputs, targets, sample_weights).
        validation_steps: Only relevant if `validation_data`
            is a generator. Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        class_weight: dictionary mapping class indices to a weight
            for the class.
        max_q_size: maximum size for the generator queue
        workers: maximum number of processes to spin up
            when using process based threading
        pickle_safe: if True, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.
        initial_epoch: epoch at which to start training
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

    # python 2 has 'next', 3 has '__next__'
    # avoid any explicit version checks
    val_gen = (hasattr(validation_data, 'next') or
               hasattr(validation_data, '__next__'))
    if val_gen and not validation_steps:
      raise ValueError('When using a generator for validation data, '
                       'you must specify a value for '
                       '`validation_steps`.')

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

    if do_validation and not val_gen:
      if len(validation_data) == 2:
        val_x, val_y = validation_data  # pylint: disable=unpacking-non-sequence
        val_sample_weight = None
      elif len(validation_data) == 3:
        val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
      else:
        raise ValueError('validation_data should be a tuple '
                         '`(val_x, val_y, val_sample_weight)` '
                         'or `(val_x, val_y)`. Found: ' + str(validation_data))
      val_x, val_y, val_sample_weights = self._standardize_user_data(
          val_x, val_y, val_sample_weight)
      val_data = val_x + val_y + val_sample_weights
      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
        val_data += [0.]
      for cbk in callbacks:
        cbk.validation_data = val_data
    enqueuer = None

    try:
      enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
      enqueuer.start(max_q_size=max_q_size, workers=workers)

      callback_model.stop_training = False
      while epoch < epochs:
        callbacks.on_epoch_begin(epoch)
        steps_done = 0
        batch_index = 0
        while steps_done < steps_per_epoch:
          generator_output = None
          while enqueuer.is_running():
            if not enqueuer.queue.empty():
              generator_output = enqueuer.queue.get()
              break
            else:
              time.sleep(wait_time)

          if not hasattr(generator_output, '__len__'):
            raise ValueError('output of generator should be '
                             'a tuple `(x, y, sample_weight)` '
                             'or `(x, y)`. Found: ' + str(generator_output))
          if len(generator_output) == 2:
            x, y = generator_output  # pylint: disable=unpacking-non-sequence
            sample_weight = None
          elif len(generator_output) == 3:
            x, y, sample_weight = generator_output  # pylint: disable=unpacking-non-sequence
          else:
            raise ValueError('output of generator should be '
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

          # Construct epoch logs.
          epoch_logs = {}
          batch_index += 1
          steps_done += 1

          # Epoch finished.
          if steps_done >= steps_per_epoch and do_validation:
            if val_gen:
              val_outs = self.evaluate_generator(
                  validation_data,
                  validation_steps,
                  max_q_size=max_q_size,
                  workers=workers,
                  pickle_safe=pickle_safe)
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

        callbacks.on_epoch_end(epoch, epoch_logs)
        epoch += 1
        if callback_model.stop_training:
          break

    finally:
      if enqueuer is not None:
        enqueuer.stop()

    callbacks.on_train_end()
    return self.history

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
            when using process based threading
        pickle_safe: if True, use process based threading.
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
        ValueError: In case the generator yields
            data in an invalid format.
    """
    self._make_test_function()

    steps_done = 0
    wait_time = 0.01
    all_outs = []
    batch_sizes = []
    enqueuer = None

    try:
      enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
      enqueuer.start(workers=workers, max_q_size=max_q_size)

      while steps_done < steps:
        generator_output = None
        while enqueuer.is_running():
          if not enqueuer.queue.empty():
            generator_output = enqueuer.queue.get()
            break
          else:
            time.sleep(wait_time)

        if not hasattr(generator_output, '__len__'):
          raise ValueError('output of generator should be a tuple '
                           '(x, y, sample_weight) '
                           'or (x, y). Found: ' + str(generator_output))
        if len(generator_output) == 2:
          x, y = generator_output  # pylint: disable=unpacking-non-sequence
          sample_weight = None
        elif len(generator_output) == 3:
          x, y, sample_weight = generator_output  # pylint: disable=unpacking-non-sequence
        else:
          raise ValueError('output of generator should be a tuple '
                           '(x, y, sample_weight) '
                           'or (x, y). Found: ' + str(generator_output))
        outs = self.test_on_batch(x, y, sample_weight=sample_weight)

        if isinstance(x, list):
          batch_size = len(x[0])
        elif isinstance(x, dict):
          batch_size = len(list(x.values())[0])
        else:
          batch_size = len(x)
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
                        steps,
                        max_q_size=10,
                        workers=1,
                        pickle_safe=False,
                        verbose=0):
    """Generates predictions for the input samples from a data generator.

    The generator should return the same kind of data as accepted by
    `predict_on_batch`.

    Arguments:
        generator: Generator yielding batches of input samples.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        max_q_size: Maximum size for the generator queue.
        workers: Maximum number of processes to spin up
            when using process based threading
        pickle_safe: If `True`, use process based threading.
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
    enqueuer = None

    try:
      enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
      enqueuer.start(workers=workers, max_q_size=max_q_size)

      if verbose == 1:
        progbar = Progbar(target=steps)

      while steps_done < steps:
        generator_output = None
        while enqueuer.is_running():
          if not enqueuer.queue.empty():
            generator_output = enqueuer.queue.get()
            break
          else:
            time.sleep(wait_time)

        if isinstance(generator_output, tuple):
          # Compatibility with the generators
          # used for training.
          if len(generator_output) == 2:
            x, _ = generator_output  # pylint: disable=unpacking-non-sequence
          elif len(generator_output) == 3:
            x, _, _ = generator_output  # pylint: disable=unpacking-non-sequence
          else:
            raise ValueError('output of generator should be '
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
      return [out for out in all_outs]
    else:
      return [np.concatenate(out) for out in all_outs]

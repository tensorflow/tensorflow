# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Defines ways of splicing and re-arranging time series.

This file provides methods for reading, parsing, and re-arranging a time
series. The main departure from standard TensorFlow input pipelines is a focus
on "chunking" a time series, i.e. slicing it into small contiguous windows which
are then batched together for training, a form of truncated
backpropagation. This typically provides a significant speedup compared to
looping over the whole series sequentially, by exploiting data parallelism and
by reducing redundant contributions to gradients (due to redundant information
in the series itself).

A series, consisting of times (an increasing vector of integers) and values (one
or more floating point values for each time) along with any exogenous features,
is stored either in memory or on disk in various formats (e.g. "one record per
timestep" on disk, or as a dictionary of Numpy arrays in memory). The location
and format is specified by configuring a `TimeSeriesReader` object
(e.g. `NumpyReader`, `CSVReader`), which reads the data into the TensorFlow
graph. A `TimeSeriesInputFn` object (typically `RandomWindowInputFn`) then
performs windowing and batching.

Time series are passed through this pipeline as dictionaries mapping feature
names to their values. For training and evaluation, these require at minimum
`TrainEvalFeatures.TIMES` (scalar integers, one per timestep) and
`TrainEvalFeatures.VALUES` (may be either univariate or multivariate). Exogenous
features may have any shape, but are likewise associated with a timestep. Times
themselves need not be contiguous or regular (although smaller/fewer gaps are
generally better), but each timestep must have all `VALUES` and any exogenous
features (i.e. times may be missing, but given that a time is specified, every
other feature must also be specified for that step; some models may support
making exogenous updates conditional).

The expected use case of a `TimeSeriesInputFn` is that it is first configured
(for example setting a batch or window size) and passed a reader (a
`TimeSeriesReader` object). The `TimeSeriesInputFn` can then be passed as the
input_fn of an Estimator.

For example, `RandomWindowInputFn` is useful for creating batches of random
chunks of a series for training:

```
  # Read data in the default "time,value" CSV format with no header
  reader = input_pipeline.CSVReader(csv_file_name)
  # Set up windowing and batching for training
  train_input_fn = input_pipeline.RandomWindowInputFn(
      reader, batch_size=16, window_size=16)
  # Fit model parameters to data
  estimator.train(input_fn=train_input_fn, steps=150)
```

`RandomWindowInputFn` is the primary tool for training and quantitative
evaluation of time series. `WholeDatasetInputFn`, which reads a whole series
into memory, is useful for qualitative evaluation and preparing to make
predictions with `predict_continuation_input_fn`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import numpy

from tensorflow.contrib.timeseries.python.timeseries import feature_keys
from tensorflow.contrib.timeseries.python.timeseries import model_utils

from tensorflow.python.estimator import estimator_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import training
from tensorflow.python.util import nest


def predict_continuation_input_fn(
    evaluation, steps=None, times=None, exogenous_features=None):
  """An Estimator input_fn for running predict() after evaluate().

  If the call to evaluate() we are making predictions based on had a batch_size
  greater than one, predictions will start after each of these windows
  (i.e. will have the same batch dimension).

  Args:
    evaluation: The dictionary returned by `Estimator.evaluate`, with keys
      FilteringResults.STATE_TUPLE and FilteringResults.TIMES.
    steps: The number of steps to predict (scalar), starting after the
      evaluation. If `times` is specified, `steps` must not be; one is required.
    times: A [batch_size x window_size] array of integers (not a Tensor)
      indicating times to make predictions for. These times must be after the
      corresponding evaluation. If `steps` is specified, `times` must not be;
      one is required. If the batch dimension is omitted, it is assumed to be 1.
    exogenous_features: Optional dictionary. If specified, indicates exogenous
      features for the model to use while making the predictions. Values must
      have shape [batch_size x window_size x ...], where `batch_size` matches
      the batch dimension used when creating `evaluation`, and `window_size` is
      either the `steps` argument or the `window_size` of the `times` argument
      (depending on which was specified).
  Returns:
    An `input_fn` suitable for passing to the `predict` function of a time
    series `Estimator`.
  Raises:
    ValueError: If `times` or `steps` are misspecified.
  """
  if exogenous_features is None:
    exogenous_features = {}
  predict_times = model_utils.canonicalize_times_or_steps_from_output(
      times=times, steps=steps, previous_model_output=evaluation)
  features = {
      feature_keys.PredictionFeatures.STATE_TUPLE:
          evaluation[feature_keys.FilteringResults.STATE_TUPLE],
      feature_keys.PredictionFeatures.TIMES:
          predict_times
  }
  features.update(exogenous_features)
  def _predict_input_fn():
    """An input_fn for predict()."""
    # Prevents infinite iteration with a constant output in an Estimator's
    # predict().
    limited_features = {}
    for key, values in features.items():
      limited_values = nest.map_structure(
          lambda value: training.limit_epochs(value, num_epochs=1), values)
      limited_features[key] = limited_values
    return (limited_features, None)
  return _predict_input_fn


class TimeSeriesReader(object):
  """Reads from and parses a data source for a `TimeSeriesInputFn`.

  This class provides methods that read a few records (`read`) or the full data
  set at once (`read_full`), and returns them as dictionaries mapping feature
  names to feature Tensors. Please see note at the top of the file for the
  structure of these dictionaries. The output is generally chunked by a
  `TimeSeriesInputFn` before being passed to the model.
  """

  def check_dataset_size(self, minimum_dataset_size):
    """When possible, raises an error if the dataset is too small.

    This method allows TimeSeriesReaders to raise informative error messages if
    the user has selected a window size in their TimeSeriesInputFn which is
    larger than the dataset size. However, many TimeSeriesReaders will not have
    access to a dataset size, in which case they do not need to override this
    method.

    Args:
      minimum_dataset_size: The minimum number of records which should be
        contained in the dataset. Readers should attempt to raise an error when
        possible if an epoch of data contains fewer records.
    """
    pass

  @abc.abstractmethod
  def read(self):
    """Parses one or more records into a feature dictionary.

    This method is expected to be called by a `TimeSeriesInputFn` object, and is
    not for use with models directly.

    A `TimeSeriesReader` object reads multiple records at a single time for
    efficiency; the size of these batches is an implementation detail internal
    to the input pipeline. These records should generally be sequential,
    although some out-of-order records due to file wraparounds are expected and
    must be handled by callers.

    Returns:
      A dictionary mapping feature names to `Tensor` values, each with an
      arbitrary batch dimension (for efficiency) as their first dimension.
    """
    pass

  @abc.abstractmethod
  def read_full(self):
    """Return the full dataset.

    Largely for interactive use/plotting (or evaluation on small
    datasets). Generally not very efficient. Not recommended for training.

    Returns:
      Same return type as `read`, but with the full dataset rather than an
      arbitrary chunk of it. A dictionary mapping feature names to `Tensor`
      values, where the size of the first dimension of each `Tensor` is the
      number of samples in the entire dataset. These `Tensor`s should be
      constant across graph invocations, assuming that the underlying data
      remains constant. Current implementations re-read data on each graph
      invocation, although this may change in the future.
    """
    pass


class NumpyReader(TimeSeriesReader):
  """A time series parser for feeding Numpy arrays to a `TimeSeriesInputFn`.

  Avoids embedding data in the graph as constants.
  """

  def __init__(self, data, read_num_records_hint=4096):
    """Numpy array input for a `TimeSeriesInputFn`.

    Args:
      data: A dictionary mapping feature names to Numpy arrays, with two
        possible shapes (requires keys `TrainEvalFeatures.TIMES` and
        `TrainEvalFeatures.VALUES`):
          Univariate; `TIMES` and `VALUES` are both vectors of shape [series
            length]
          Multivariate; `TIMES` is a vector of shape [series length], `VALUES`
            has shape [series length x number of features].
        In any case, `VALUES` and any exogenous features must have their shapes
        prefixed by the shape of the value corresponding to the `TIMES` key.
      read_num_records_hint: The maximum number of samples to read at one time,
        for efficiency.
    """
    self._features = _canonicalize_numpy_data(
        data, require_single_batch=True)
    self._read_num_records_hint = read_num_records_hint

  def check_dataset_size(self, minimum_dataset_size):
    """Raise an error if the dataset is too small."""
    dataset_size = self._features[feature_keys.TrainEvalFeatures.TIMES].shape[1]
    if dataset_size < minimum_dataset_size:
      raise ValueError(
          ("A TimeSeriesInputFn is configured to create windows of size {}, "
           "but only {} records were available in the dataset. Either decrease "
           "the window size or provide more records.").format(
               minimum_dataset_size, dataset_size))

  def read(self):
    """Returns a large chunk of the Numpy arrays for later re-chunking."""
    # Remove the batch dimension from all features
    features = {key: numpy.squeeze(value, axis=0)
                for key, value in self._features.items()}
    return estimator_lib.inputs.numpy_input_fn(
        x=features,
        # The first dimensions of features are the series length, since we have
        # removed the batch dimension above. We now pull out
        # self._read_num_records_hint steps of this single time series to pass
        # to the TimeSeriesInputFn.
        batch_size=self._read_num_records_hint,
        num_epochs=None,
        shuffle=False)()

  def read_full(self):
    """Returns `Tensor` versions of the full Numpy arrays."""
    features = estimator_lib.inputs.numpy_input_fn(
        x=self._features,
        batch_size=1,
        num_epochs=None,
        queue_capacity=2,  # Each queue element is a full copy of the dataset
        shuffle=False)()
    # TimeSeriesInputFn expect just a batch dimension
    return {feature_name: array_ops.squeeze(feature_value, axis=0)
            for feature_name, feature_value in features.items()}


class ReaderBaseTimeSeriesParser(TimeSeriesReader):
  """Base for time series readers which wrap a `tf.ReaderBase`."""

  def __init__(self, filenames, read_num_records_hint=4096):
    """Configure the time series reader.

    Args:
      filenames: A string or list of strings indicating files to read records
        from.
      read_num_records_hint: When not reading a full dataset, indicates the
        number of records to transfer in a single chunk (for efficiency). The
        actual number transferred at one time may vary.
    """
    self._filenames = filenames
    self._read_num_records_hint = read_num_records_hint

  @abc.abstractmethod
  def _get_reader(self):
    """Get an instance of the tf.ReaderBase associated with this class."""
    pass

  @abc.abstractmethod
  def _process_records(self, lines):
    """Given string items, return a processed dictionary of Tensors.

    Args:
      lines: A 1-dimensional string Tensor, each representing a record to parse
        (source dependent, e.g. a line of a file, or a serialized protocol
        buffer).

    Returns:
      A dictionary mapping feature names to their values. The batch dimensions
      should match the length of `lines`.
    """
    pass

  def _get_filename_queue(self, epoch_limit):
    """Constructs a filename queue with an epoch limit.

    `epoch_limit` is intended as an error checking fallback to prevent a reader
    from infinitely looping in its requests for more work items if none are
    available in any file. It should be set high enough that it is never reached
    assuming at least one record exists in some file.

    Args:
      epoch_limit: The maximum number of times to read through the complete list
        of files before throwing an OutOfRangeError.
    Returns:
      A tuple of (filename_queue, epoch_limiter):
        filename_queue: A FIFOQueue with filename work items.
        epoch_limiter: The local variable used for epoch limitation. This should
          be set to zero before a reader is passed `filename_queue` in order to
          reset the epoch limiter's state.
    """
    epoch_limiter = variable_scope.variable(
        initial_value=constant_op.constant(0, dtype=dtypes.int64),
        name="epoch_limiter",
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES])
    filenames_tensor = array_ops.reshape(
        ops.convert_to_tensor(self._filenames), [-1])
    # We can't rely on epoch_limiter being initialized, since queue runners are
    # started before local variables are initialized. Instead, we ignore epoch
    # limits before variable initialization. This means that prior to variable
    # initialization, a QueueRunner may cause a reader to enter an un-checked
    # infinite loop. However, as soon as local variables are initialized, we
    # will start incrementing and checking epoch_limiter, which will interrupt
    # any in-progress loops.
    conditional_count_up_to = control_flow_ops.cond(
        state_ops.is_variable_initialized(epoch_limiter),
        lambda: epoch_limiter.count_up_to(epoch_limit),
        lambda: constant_op.constant(0, dtype=dtypes.int64))
    with ops.control_dependencies([conditional_count_up_to]):
      filenames_tensor = array_ops.identity(filenames_tensor)
    filename_queue = input_lib.string_input_producer(
        filenames_tensor, shuffle=False, capacity=1)
    return filename_queue, epoch_limiter

  def read(self):
    """Reads a chunk of data from the `tf.ReaderBase` for later re-chunking."""
    # Assuming there is at least one item to be read among all of the files in
    # self._filenames, we will not need to go through more than
    # self._read_num_records_hint epochs to get a batch of
    # self._read_num_records_hint records. Setting this limit and resetting it
    # before each reader.read_up_to call prevents infinite looping when there
    # are no records available in any of the files.
    filename_queue, epoch_limiter = self._get_filename_queue(
        epoch_limit=self._read_num_records_hint)
    reader = self._get_reader()
    epoch_reset_op = state_ops.assign(epoch_limiter, 0)
    with ops.control_dependencies([epoch_reset_op]):
      _, records = reader.read_up_to(
          filename_queue, self._read_num_records_hint)
    return self._process_records(records)

  def read_full(self):
    """Reads a full epoch of data into memory."""
    reader = self._get_reader()
    # Set a hard limit of 2 epochs through self._filenames. If there are any
    # records available, we should only end up reading the first record in the
    # second epoch before exiting the while loop and subsequently resetting the
    # epoch limit. If there are no records available in any of the files, this
    # hard limit prevents the reader.read_up_to call from looping infinitely.
    filename_queue, epoch_limiter = self._get_filename_queue(epoch_limit=2)
    epoch_reset_op = state_ops.assign(epoch_limiter, 0)
    with ops.control_dependencies([epoch_reset_op]):
      first_key, first_value = reader.read_up_to(filename_queue, 1)
    # Read until we get a duplicate key (one epoch)
    def _while_condition(
        current_key, current_value, current_index, collected_records):
      del current_value, current_index, collected_records  # unused
      return math_ops.not_equal(array_ops.squeeze(current_key, axis=0),
                                array_ops.squeeze(first_key, axis=0))

    def _while_body(
        current_key, current_value, current_index, collected_records):
      del current_key  # unused
      new_key, new_value = reader.read_up_to(filename_queue, 1)
      new_key.set_shape([1])
      new_value.set_shape([1])
      return (new_key,
              new_value,
              current_index + 1,
              collected_records.write(current_index, current_value))
    _, _, _, records_ta = control_flow_ops.while_loop(
        _while_condition,
        _while_body,
        [constant_op.constant([""]), first_value,
         0,  # current_index starting value
         tensor_array_ops.TensorArray(  # collected_records
             dtype=dtypes.string, size=0, dynamic_size=True)])
    records = records_ta.concat()
    # Reset the reader when we're done so that subsequent requests for data get
    # the dataset in the proper order.
    with ops.control_dependencies([records]):
      reader_reset_op = reader.reset()
    with ops.control_dependencies([reader_reset_op]):
      records = array_ops.identity(records)
    return self._process_records(records)


class CSVReader(ReaderBaseTimeSeriesParser):
  """Reads from a collection of CSV-formatted files."""

  def __init__(self,
               filenames,
               column_names=(feature_keys.TrainEvalFeatures.TIMES,
                             feature_keys.TrainEvalFeatures.VALUES),
               column_dtypes=None,
               skip_header_lines=None,
               read_num_records_hint=4096):
    """CSV-parsing reader for a `TimeSeriesInputFn`.

    Args:
      filenames: A filename or list of filenames to read the time series
          from. Each line must have columns corresponding to `column_names`.
      column_names: A list indicating names for each
          feature. `TrainEvalFeatures.TIMES` and `TrainEvalFeatures.VALUES` are
          required; `VALUES` may be repeated to indicate a multivariate series.
      column_dtypes: If provided, must be a list with the same length as
          `column_names`, indicating dtypes for each column. Defaults to
          `tf.int64` for `TrainEvalFeatures.TIMES` and `tf.float32` for
          everything else.
      skip_header_lines: Passed on to `tf.TextLineReader`; skips this number of
          lines at the beginning of each file.
      read_num_records_hint: When not reading a full dataset, indicates the
          number of records to parse/transfer in a single chunk (for
          efficiency). The actual number transferred at one time may be more or
          less.
    Raises:
      ValueError: If required column names are not specified, or if lengths do
        not match.
    """
    if feature_keys.TrainEvalFeatures.TIMES not in column_names:
      raise ValueError("'{}' is a required column.".format(
          feature_keys.TrainEvalFeatures.TIMES))
    if feature_keys.TrainEvalFeatures.VALUES not in column_names:
      raise ValueError("'{}' is a required column.".format(
          feature_keys.TrainEvalFeatures.VALUES))
    if column_dtypes is not None and len(column_dtypes) != len(column_names):
      raise ValueError(
          ("If specified, the length of column_dtypes must match the length of "
           "column_names (got column_dtypes={} and column_names={}).").format(
               column_dtypes, column_names))
    if sum(1 for column_name in column_names
           if column_name == feature_keys.TrainEvalFeatures.TIMES) != 1:
      raise ValueError(
          "Got more than one times column ('{}'), but exactly "
          "one is required.".format(feature_keys.TrainEvalFeatures.TIMES))
    self._column_names = column_names
    self._column_dtypes = column_dtypes
    self._skip_header_lines = skip_header_lines
    super(CSVReader, self).__init__(
        filenames=filenames, read_num_records_hint=read_num_records_hint)

  def _get_reader(self):
    return io_ops.TextLineReader(skip_header_lines=self._skip_header_lines)

  def _process_records(self, lines):
    """Parse `lines` as CSV records."""
    if self._column_dtypes is None:
      default_values = [(array_ops.zeros([], dtypes.int64),)
                        if column_name == feature_keys.TrainEvalFeatures.TIMES
                        else () for column_name in self._column_names]
    else:
      default_values = [(array_ops.zeros([], dtype),)
                        for dtype in self._column_dtypes]
    columns = parsing_ops.decode_csv(lines, default_values)
    features_lists = {}
    for column_name, value in zip(self._column_names, columns):
      features_lists.setdefault(column_name, []).append(value)
    features = {}
    for column_name, values in features_lists.items():
      if (len(values) == 1 and
          column_name != feature_keys.TrainEvalFeatures.VALUES):
        features[column_name] = values[0]
      else:
        features[column_name] = array_ops.stack(values, axis=1)
    return features


class TimeSeriesInputFn(object):
  """Base for classes which create batches of windows from a time series."""

  @abc.abstractmethod
  def create_batch(self):
    """Creates chunked Tensors from times, values, and other features.

    Suitable for use as the input_fn argument of a tf.estimator.Estimator's
    fit() or evaluate() method.

    Returns:
      A tuple of (features, targets):
        features: A dictionary with `TrainEvalFeatures.TIMES` and
          `TrainEvalFeatures.VALUES` as keys, `TIMES` having an associated value
          with shape [batch size x window length], `VALUES` with shape [batch
          size x window length x number of features]. Any other features will
          also have shapes prefixed with [batch size x window length].
        targets: Not used, but must have a value for compatibility with the
          Estimator API. That value should be None.
    """
    pass

  def __call__(self):
    # Allow a TimeSeriesInputFn to be used as an input function directly
    return self.create_batch()


class WholeDatasetInputFn(TimeSeriesInputFn):
  """Supports passing a full time series to a model for evaluation/inference.

  Note that this `TimeSeriesInputFn` is not designed for high throughput, and
  should not be used for training. It allows for sequential evaluation on a full
  dataset (with sequential in-sample predictions), which then feeds naturally
  into `predict_continuation_input_fn` for making out-of-sample
  predictions. While this is useful for plotting and interactive use,
  `RandomWindowInputFn` is better suited to training and quantitative
  evaluation.
  """
  # TODO(allenl): A SequentialWindowInputFn for getting model end state without
  # loading the whole dataset into memory (or for quantitative evaluation of
  # sequential models). Note that an Estimator using such a TimeSeriesInputFn
  # won't return in-sample predictions for the whole dataset, which means it
  # won't be terribly useful for interactive use/plotting (unless the user
  # passes in concat metrics). Also need to be careful about state saving for
  # sequential models, particularly the gaps between chunks.

  def __init__(self, time_series_reader):
    """Initialize the `TimeSeriesInputFn`.

    Args:
      time_series_reader: A TimeSeriesReader object.
    """
    self._reader = time_series_reader
    super(WholeDatasetInputFn, self).__init__()

  def create_batch(self):
    """A suitable `input_fn` for an `Estimator`'s `evaluate()`.

    Returns:
      A dictionary mapping feature names to `Tensors`, each shape
      prefixed by [1, data set size] (i.e. a batch size of 1).
    """
    features = self._reader.read_full()
    # Add a batch dimension of one to each feature.
    return ({feature_name: feature_value[None, ...]
             for feature_name, feature_value in features.items()},
            None)


class RandomWindowInputFn(TimeSeriesInputFn):
  """Wraps a `TimeSeriesReader` to create random batches of windows.

  Tensors are first collected into sequential windows (in a windowing queue
  created by `tf.train.batch`, based on the order returned from
  `time_series_reader`), then these windows are randomly batched (in a
  `RandomShuffleQueue`), the Tensors returned by `create_batch` having shapes
  prefixed by [`batch_size`, `window_size`].

  This `TimeSeriesInputFn` is useful for both training and quantitative
  evaluation (but be sure to run several epochs for sequential models such as
  `StructuralEnsembleRegressor` to completely flush stale state left over from
  training). For qualitative evaluation or when preparing for predictions, use
  `WholeDatasetInputFn`.
  """

  def __init__(
      self, time_series_reader, window_size, batch_size,
      queue_capacity_multiplier=1000, shuffle_min_after_dequeue_multiplier=2,
      discard_out_of_order=True, discard_consecutive_batches_limit=1000,
      jitter=True, num_threads=2, shuffle_seed=None):
    """Configure the RandomWindowInputFn.

    Args:
      time_series_reader: A TimeSeriesReader object.
      window_size: The number of examples to keep together sequentially. This
        controls the length of truncated backpropagation: smaller values mean
        less sequential computation, which can lead to faster training, but
        create a coarser approximation to the gradient (which would ideally be
        computed by a forward pass over the entire sequence in order).
      batch_size: The number of windows to place together in a batch. Larger
        values will lead to more stable gradients during training.
      queue_capacity_multiplier: The capacity for the queues used to create
        batches, specified as a multiple of `batch_size` (for
        RandomShuffleQueue) and `batch_size * window_size` (for the
        FIFOQueue). Controls the maximum number of windows stored. Should be
        greater than `shuffle_min_after_dequeue_multiplier`.
      shuffle_min_after_dequeue_multiplier: The minimum number of windows in the
        RandomShuffleQueue after a dequeue, which controls the amount of entropy
        introduced during batching. Specified as a multiple of `batch_size`.
      discard_out_of_order: If True, windows of data which have times which
        decrease (a higher time followed by a lower time) are discarded. If
        False, the window and associated features are instead sorted so that
        times are non-decreasing. Discarding is typically faster, as models do
        not have to deal with artificial gaps in the data. However, discarding
        does create a bias where the beginnings and endings of files are
        under-sampled.
      discard_consecutive_batches_limit: Raise an OutOfRangeError if more than
        this number of batches are discarded without a single non-discarded
        window (prevents infinite looping when the dataset is too small).
      jitter: If True, randomly discards examples between some windows in order
        to avoid deterministic chunking patterns. This is important for models
        like AR which may otherwise overfit a fixed chunking.
      num_threads: Use this number of threads for queues. Setting a value of 1
        removes one source of non-determinism (and in combination with
        shuffle_seed should provide deterministic windowing).
      shuffle_seed: A seed for window shuffling. The default value of None
        provides random behavior. With `shuffle_seed` set and
        `num_threads=1`, provides deterministic behavior.
    """
    self._reader = time_series_reader
    self._window_size = window_size
    self._reader.check_dataset_size(minimum_dataset_size=self._window_size)
    self._batch_size = batch_size
    self._queue_capacity_multiplier = queue_capacity_multiplier
    self._shuffle_min_after_dequeue_multiplier = (
        shuffle_min_after_dequeue_multiplier)
    self._discard_out_of_order = discard_out_of_order
    self._discard_limit = discard_consecutive_batches_limit
    self._jitter = jitter
    if num_threads is None:
      self._num_threads = self._batch_size
    else:
      self._num_threads = num_threads
    self._shuffle_seed = shuffle_seed
    super(RandomWindowInputFn, self).__init__()

  def create_batch(self):
    """Create queues to window and batch time series data.

    Returns:
      A dictionary of Tensors corresponding to the output of `self._reader`
      (from the `time_series_reader` constructor argument), each with shapes
      prefixed by [`batch_size`, `window_size`].
    """
    features = self._reader.read()
    if self._jitter:
      # TODO(agarwal, allenl): Figure out if more jitter is needed here.
      jitter = random_ops.random_uniform(shape=[], maxval=2, dtype=dtypes.int32)
    else:
      jitter = 0
    # To keep things efficient, we pass from the windowing batcher to the
    # batch-of-windows batcher in batches. This avoids the need for huge numbers
    # of threads, but does mean that jitter is only applied occasionally.
    # TODO(allenl): Experiment with different internal passing sizes.
    internal_passing_size = self._batch_size
    features_windowed = input_lib.batch(
        features,
        batch_size=self._window_size * internal_passing_size + jitter,
        enqueue_many=True,
        capacity=(self._queue_capacity_multiplier
                  * internal_passing_size * self._window_size),
        num_threads=self._num_threads)
    raw_features_windowed = features_windowed
    if self._jitter:
      features_windowed = {
          key: value[jitter:]
          for key, value in features_windowed.items()}
    features_windowed = {
        key: array_ops.reshape(
            value,
            array_ops.concat(
                [[internal_passing_size, self._window_size],
                 array_ops.shape(value)[1:]],
                axis=0))
        for key, value in features_windowed.items()}
    batch_and_window_shape = tensor_shape.TensorShape(
        [internal_passing_size, self._window_size])
    for key in features_windowed.keys():
      features_windowed[key].set_shape(
          batch_and_window_shape.concatenate(
              raw_features_windowed[key].get_shape()[1:]))
    # When switching files, we may end up with windows where the time is not
    # decreasing, even if times within each file are sorted (and even if those
    # files are visited in order, when looping back around to the beginning of
    # the first file). This is hard for models to deal with, so we either
    # discard such examples, creating a bias where the beginning and end of the
    # series is under-sampled, or we sort the window, creating large gaps.
    times = features_windowed[feature_keys.TrainEvalFeatures.TIMES]
    if self._discard_out_of_order:
      non_decreasing = math_ops.reduce_all(
          times[:, 1:] >= times[:, :-1], axis=1)
      # Ensure that no more than self._discard_limit complete batches are
      # discarded contiguously (resetting the count when we find a single clean
      # window). This prevents infinite looping when the dataset is smaller than
      # the window size.
      # TODO(allenl): Figure out a way to return informative errors from
      # count_up_to.
      discarded_windows_limiter = variable_scope.variable(
          initial_value=constant_op.constant(0, dtype=dtypes.int64),
          name="discarded_windows_limiter",
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES])
      def _initialized_limit_check():
        return control_flow_ops.cond(
            math_ops.reduce_any(non_decreasing),
            lambda: state_ops.assign(discarded_windows_limiter, 0),
            lambda: discarded_windows_limiter.count_up_to(self._discard_limit))
      discard_limit_op = control_flow_ops.cond(
          state_ops.is_variable_initialized(discarded_windows_limiter),
          _initialized_limit_check,
          lambda: constant_op.constant(0, dtype=dtypes.int64))
      with ops.control_dependencies([discard_limit_op]):
        non_decreasing = array_ops.identity(non_decreasing)
    else:
      _, indices_descending = nn.top_k(
          times, k=array_ops.shape(times)[-1], sorted=True)
      indices = array_ops.reverse(indices_descending, axis=[0])
      features_windowed = {
          key: array_ops.gather(params=value, indices=indices)
          for key, value in features_windowed.items()
      }
      non_decreasing = True
    features_batched = input_lib.maybe_shuffle_batch(
        features_windowed,
        num_threads=self._num_threads,
        seed=self._shuffle_seed,
        batch_size=self._batch_size,
        capacity=self._queue_capacity_multiplier * self._batch_size,
        min_after_dequeue=(self._shuffle_min_after_dequeue_multiplier *
                           self._batch_size),
        keep_input=non_decreasing,
        enqueue_many=True)
    return (features_batched, None)


def _canonicalize_numpy_data(data, require_single_batch):
  """Do basic checking and reshaping for Numpy data.

  Args:
    data: A dictionary mapping keys to Numpy arrays, with several possible
      shapes (requires keys `TrainEvalFeatures.TIMES` and
      `TrainEvalFeatures.VALUES`):
        Single example; `TIMES` is a scalar and `VALUES` is either a scalar or a
          vector of length [number of features].
        Sequence; `TIMES` is a vector of shape [series length], `VALUES` either
          has shape [series length] (univariate) or [series length x number of
          features] (multivariate).
        Batch of sequences; `TIMES` is a vector of shape [batch size x series
          length], `VALUES` has shape [batch size x series length] or [batch
          size x series length x number of features].
      In any case, `VALUES` and any exogenous features must have their shapes
      prefixed by the shape of the value corresponding to the `TIMES` key.
    require_single_batch: If True, raises an error if the provided data has a
      batch dimension > 1.
  Returns:
    A dictionary with features normalized to have shapes prefixed with [batch
    size x series length]. The sizes of dimensions which were omitted in the
    inputs are 1.
  Raises:
    ValueError: If dimensions are incorrect or do not match, or required
      features are missing.
  """
  features = {key: numpy.array(value) for key, value in data.items()}
  if (feature_keys.TrainEvalFeatures.TIMES not in features or
      feature_keys.TrainEvalFeatures.VALUES not in features):
    raise ValueError("{} and {} are required features.".format(
        feature_keys.TrainEvalFeatures.TIMES,
        feature_keys.TrainEvalFeatures.VALUES))
  times = features[feature_keys.TrainEvalFeatures.TIMES]
  for key, value in features.items():
    if value.shape[:len(times.shape)] != times.shape:
      raise ValueError(
          ("All features must have their shapes prefixed by the shape of the"
           " times feature. Got shape {} for feature '{}', but shape {} for"
           " '{}'").format(value.shape, key, times.shape,
                           feature_keys.TrainEvalFeatures.TIMES))
  if not times.shape:  # a single example
    if not features[feature_keys.TrainEvalFeatures.VALUES].shape:  # univariate
      # Add a feature dimension (with one feature)
      features[feature_keys.TrainEvalFeatures.VALUES] = features[
          feature_keys.TrainEvalFeatures.VALUES][..., None]
    elif len(features[feature_keys.TrainEvalFeatures.VALUES].shape) > 1:
      raise ValueError(
          ("Got an unexpected number of dimensions for the '{}' feature."
           " Was expecting at most 1 dimension"
           " ([number of features]) since '{}' does not "
           "have a batch or time dimension, but got shape {}").format(
               feature_keys.TrainEvalFeatures.VALUES,
               feature_keys.TrainEvalFeatures.TIMES,
               features[feature_keys.TrainEvalFeatures.VALUES].shape))
    # Add trivial batch and time dimensions for every feature
    features = {key: value[None, None, ...] for key, value in features.items()}
  if len(times.shape) == 1:  # shape [series length]
    if len(features[feature_keys.TrainEvalFeatures.VALUES]
           .shape) == 1:  # shape [series length]
      # Add a feature dimension (with one feature)
      features[feature_keys.TrainEvalFeatures.VALUES] = features[
          feature_keys.TrainEvalFeatures.VALUES][..., None]
    elif len(features[feature_keys.TrainEvalFeatures.VALUES].shape) > 2:
      raise ValueError(
          ("Got an unexpected number of dimensions for the '{}' feature."
           " Was expecting at most 2 dimensions"
           " ([series length, number of features]) since '{}' does not "
           "have a batch dimension, but got shape {}").format(
               feature_keys.TrainEvalFeatures.VALUES,
               feature_keys.TrainEvalFeatures.TIMES,
               features[feature_keys.TrainEvalFeatures.VALUES].shape))
    # Add trivial batch dimensions for every feature
    features = {key: value[None, ...] for key, value in features.items()}
  elif len(features[feature_keys.TrainEvalFeatures.TIMES]
           .shape) != 2:  # shape [batch size, series length]
    raise ValueError(
        ("Got an unexpected number of dimensions for times. Was expecting at "
         "most two ([batch size, series length]), but got shape {}.").format(
             times.shape))
  if require_single_batch:
    # We don't expect input to be already batched; batching is done later
    if features[feature_keys.TrainEvalFeatures.TIMES].shape[0] != 1:
      raise ValueError("Got batch input, was expecting unbatched input.")
  return features

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""SequenceQueueingStateSaver and wrappers.

Please see the reading data how-to for context.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numbers

import six

from tensorflow.python import summary
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.training import queue_runner


class _SequenceInputWrapper(object):
  """A wrapper object for storing sequence-related input.

  The SequenceInputWapper accepts four objects:

    length: A scalar int containing the length of the input sequence.
    key: A scalar string containing the unique key of the input sequence.
    sequences: A dict mapping labels, like `input`, to tensors
      whose initial index dimension is at least size `length`.
    context: A dict mapping labels, like `global_target`, to tensors
      that represent data across the entire example.
  """

  def __init__(self, length, key, sequences, context):
    length = ops.convert_to_tensor(length, name="length")
    key = ops.convert_to_tensor(key, name="key")
    if not isinstance(sequences, dict):
      raise TypeError("sequences must be a dict")
    if not isinstance(context, dict):
      raise TypeError("context must be a dict")
    if not sequences:
      raise ValueError("must have at least one sequence tensor")
    for k in sequences.keys():
      if not isinstance(k, six.string_types):
        raise TypeError("sequence key must be string: %s" % k)
      if ":" in k:
        raise ValueError("sequence key may not have a colon: '%s'" % k)
    for k in context.keys():
      if not isinstance(k, six.string_types):
        raise TypeError("context key must be string: %s" % k)
      if ":" in k:
        raise ValueError("context key may not have a colon: '%s'" % k)
    sequences = dict(
        (k, ops.convert_to_tensor(v, name="sequence_%s" % k))
        for k, v in sequences.items())
    context = dict(
        (k, ops.convert_to_tensor(v, name="context_%s" % k))
        for k, v in context.items())
    self._length = length
    self._key = key
    self._sequences = sequences
    self._context = context

  @property
  def length(self):
    return self._length

  @property
  def key(self):
    return self._key

  @property
  def sequences(self):
    return self._sequences

  @property
  def context(self):
    return self._context


def _check_multiple_of(value, multiple_of):
  """Checks that value `value` is a non-zero multiple of `multiple_of`.

  Args:
    value: an int32 scalar Tensor.
    multiple_of: an int or int32 scalar Tensor.

  Returns:
    new_value: an int32 scalar Tensor matching `value`, but which includes an
      assertion that `value` is a multiple of `multiple_of`.
  """
  assert isinstance(value, ops.Tensor)
  with ops.control_dependencies([
      control_flow_ops.Assert(
          math_ops.logical_and(
              math_ops.equal(math_ops.mod(value, multiple_of), 0),
              math_ops.not_equal(value, 0)),
          [string_ops.string_join(
              ["Tensor %s should be a multiple of: " % value.name,
               string_ops.as_string(multiple_of),
               ", but saw value: ",
               string_ops.as_string(value),
               ". Consider setting pad=True."])])]):
    new_value = array_ops.identity(
        value, name="multiple_of_checked")
    return new_value


def _check_rank(value, expected_rank):
  """Check the rank of Tensor `value`, via shape inference and assertions.

  Args:
    value: A Tensor, possibly with shape associated shape information.
    expected_rank: int32 scalar (optionally a `Tensor`).

  Returns:
    new_value: A Tensor matching `value`.  Accessing this tensor tests
      assertions on its rank.  If expected_rank is not a `Tensor`, then
      new_value's shape's rank has been set.

  Raises:
    ValueError: if `expected_rank` is not a `Tensor` and the rank of `value`
      is known and is not equal to `expected_rank`.
  """
  assert isinstance(value, ops.Tensor)
  with ops.control_dependencies([
      control_flow_ops.Assert(
          math_ops.equal(expected_rank, array_ops.rank(value)),
          [string_ops.string_join(
              ["Rank of tensor %s should be: " % value.name,
               string_ops.as_string(expected_rank),
               ", shape received:"]),
           array_ops.shape(value)])]):
    new_value = array_ops.identity(value, name="rank_checked")
    if isinstance(expected_rank, ops.Tensor):
      expected_rank_value = tensor_util.constant_value(expected_rank)
      if expected_rank_value is not None:
        expected_rank = int(expected_rank_value)
    if not isinstance(expected_rank, ops.Tensor):
      try:
        new_value.set_shape(new_value.get_shape().with_rank(expected_rank))
      except ValueError as e:
        raise ValueError("Rank check failed for %s: %s"
                         % (value.name, str(e)))
    return new_value


def _check_shape(value, expected_shape):
  """Check the shape of Tensor `value`, via shape inference and assertions.

  Args:
    value: A Tensor, possibly with shape associated shape information.
    expected_shape: a `TensorShape`, list of `int32`, or a vector `Tensor`.

  Returns:
    new_value: A Tensor matching `value`.  Accessing this tensor tests
      assertions on its shape.  If expected_shape is not a `Tensor`, then
      new_value's shape has been set.

  Raises:
    ValueError: if `expected_shape` is not a `Tensor` and the shape of `value`
      is known and is not equal to `expected_shape`.
  """
  assert isinstance(value, ops.Tensor)
  if isinstance(expected_shape, tensor_shape.TensorShape):
    expected_shape = expected_shape.as_list()
  if isinstance(expected_shape, ops.Tensor):
    expected_shape_value = tensor_util.constant_value(expected_shape)
    if expected_shape_value is not None:
      expected_shape = [int(d) for d in expected_shape_value]
  if isinstance(expected_shape, ops.Tensor):
    value = _check_rank(value, array_ops.size(expected_shape))
  else:
    value = _check_rank(value, len(expected_shape))
  with ops.control_dependencies([
      control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(expected_shape, array_ops.shape(
              value))), [string_ops.string_join([
                  "Shape of tensor %s should be: " % value.name,
                  string_ops.as_string(expected_shape), ", shape received: ",
                  string_ops.as_string(array_ops.shape(value))
              ])])
  ]):
    new_value = array_ops.identity(value, name="shape_checked")
    if not isinstance(expected_shape, ops.Tensor):
      try:
        new_value.set_shape(new_value.get_shape().merge_with(expected_shape))
      except ValueError as e:
        raise ValueError("Shape check failed for %s: %s"
                         % (value.name, str(e)))
    return new_value


def _check_dimensions(value, dimensions, expected_sizes, debug_prefix):
  """Check the dimensions of Tensor `value`, via shape inference and assertions.

  Args:
    value: A Tensor, with optional / partial shape associated shape information.
    dimensions: An int list, the dimensions to check.
    expected_sizes: list of mixed ints and int32 scalar tensors.
      Optionally also a vector `Tensor`.
    debug_prefix: A string, used for naming ops and printing debugging messages.

  Returns:
    new_value: A Tensor matching `value`.  Accessing this tensor tests
      assertions on its shape.  If expected_sizes is not a `Tensor`, then
      new_value's shape has been set for all `dimensions[i]` where
      `expected_sizes[i]` is not a `Tensor`.

  Raises:
    TypeError: if any of the input contains invalid types:
      if `value` is not a `Tensor`.
      if `dimensions` is not a `list` or `tuple`.
    ValueError: if input has incorrect sizes or inferred shapes do not match:
      if `dimensions` contains repeated dimensions.
      if `expected_sizes` is not a `Tensor` and its length does not match that
        `dimensions`.
      if `value`'s shape has a well-defined rank, and one of the values in
        `dimensions` is equal to or above this rank.
      if `value`'s shape is well defined for some `dimensions[i]`, and
        `expected_sizes[i]` is not a `Tensor`, and these two values do
        not match.
  """

  if not isinstance(dimensions, (list, tuple)):
    raise TypeError("dimensions must be a list or tuple")
  if len(set(dimensions)) != len(dimensions):
    raise ValueError("dimensions are not unique: %s" % dimensions)
  if not isinstance(value, ops.Tensor):
    raise TypeError("value is not a Tensor: %s" % value)
  value_shape = value.get_shape()
  if not isinstance(expected_sizes, ops.Tensor):
    if len(dimensions) != len(expected_sizes):
      raise ValueError("len(dimensions) != len(expected_sizes): %d vs. %d" % (
          len(dimensions), len(expected_sizes)))
    if value_shape.ndims is not None:
      if value_shape.ndims <= max(dimensions):
        raise ValueError(
            "%s: rank of input is not greater than max(dimensions): "
            "%d vs. %d" % (debug_prefix,
                           value.get_shape().ndims,
                           max(dimensions)))
      value_dims = value_shape.as_list()
      for d, s in zip(dimensions, expected_sizes):
        if not isinstance(s, ops.Tensor):
          value_dims[d] = s
      try:
        value.set_shape(value.get_shape().merge_with(value_dims))
      except ValueError as e:
        raise ValueError("Dimensions check failed for %s: %s"
                         % (debug_prefix, str(e)))
  with ops.control_dependencies([
      control_flow_ops.Assert(
          math_ops.equal(expected_size, array_ops.shape(value)[dimension]),
          [string_ops.string_join(
              ["Dimension %d of tensor labeled %s should be: "
               % (dimension, debug_prefix),
               string_ops.as_string(expected_size),
               ", shape received: ",
               string_ops.as_string(array_ops.shape(value))])])
      for (dimension, expected_size) in zip(dimensions, expected_sizes)]):
    new_value = array_ops.identity(value, name="dims_checked_%s" % debug_prefix)
    return new_value


def _prepare_sequence_inputs(inputs, states):
  """Convert input to tensors and validate shape information.

  Args:
    inputs: A `_SequenceInputWrapper` instance.
    states: A dictionary mapping state names to input constants or tensors.

  Returns:
    The tuple (length, key, sorted_states, sorted_sequences, sorted_context),
    where each value has been checked for valid shape, and the sorted_* dicts
    are instances of OrderedDict; with key-value pairs sorted by key.

  Raises:
    ValueError: if the shapes of inputs.context.values(), states.values(),
      or inputs.sequences.values() are not fully defined (with the exception
      of the dimension of any `Tensor` in inputs.sequences.values()).
    TypeError: if the dtype of length is not int32.
  """
  # Convert state initial values to tensors
  states = dict((k, ops.convert_to_tensor(v, name="state_%s" % k))
                for k, v in states.items())

  def _assert_fully_defined(label, dict_, ignore_first_dimension=False):
    start_dimension = 1 if ignore_first_dimension else 0
    for k, v in dict_.items():
      if not v.get_shape()[start_dimension:].is_fully_defined():
        raise ValueError(
            "Shape for %s %s is not fully defined %s: %s"
            % (label,
               k,
               "(ignoring first dimension)" if ignore_first_dimension else "",
               v.get_shape()))

  _assert_fully_defined("state", states)
  _assert_fully_defined("context", inputs.context)
  # Sequences' first dimension (time) may be variable
  _assert_fully_defined(
      "sequence", inputs.sequences, ignore_first_dimension=True)

  # Get dictionaries' dtypes ordered by name - ordering is important
  # when switching between dicts and tuples for passing to Barrier.
  def _sort_by_name(d):
    return collections.OrderedDict(
        sorted(d.items(), key=lambda k_v: k_v[0]))
  sorted_sequences = _sort_by_name(inputs.sequences)
  sorted_context = _sort_by_name(inputs.context)
  sorted_states = _sort_by_name(states)

  length = _check_rank(inputs.length, 0)
  key = _check_rank(inputs.key, 0)

  if length.dtype != dtypes.int32:
    raise TypeError("length dtype must be int32, but recieved: %s"
                    % length.dtype)
  if key.dtype != dtypes.string:
    raise TypeError("key dtype must be string, but received: %s"
                    % key.dtype)

  return (length, key, sorted_states, sorted_sequences, sorted_context)


# NextQueuedSequenceBatch works closely with
# SequenceQueueingStateSaver and requires access to its private properties
# pylint: disable=protected-access
class NextQueuedSequenceBatch(object):
  """NextQueuedSequenceBatch stores deferred SequenceQueueingStateSaver data.

  This class is instantiated by `SequenceQueueingStateSaver` and is accessible
  via its `next_batch` property.
  """

  def __init__(self, state_saver):
    self._state_saver = state_saver

  @property
  def total_length(self):
    """The lengths of the original (non-truncated) unrolled examples.

    Returns:
      An integer vector of length `batch_size`, the total lengths.
    """
    return self._state_saver._received_total_length

  @property
  def length(self):
    """The lengths of the given truncated unrolled examples.

    For initial iterations, for which `sequence * num_unroll < length`,
    this number is `num_unroll`.  For the remainder,
    this number is between `0` and `num_unroll`.

    Returns:
      An integer vector of length `batch_size`, the lengths.
    """
    return self._state_saver._received_length

  @property
  def batch_size(self):
    """The batch_size of the given batch.

    Usually, this is the batch_size requested when initializing the SQSS, but
    if allow_small_batch=True this will become smaller when inputs are
    exhausted.

    Returns:
      A scalar integer tensor, the batch_size
    """
    return self._state_saver._received_batch_size

  @property
  def insertion_index(self):
    """The insertion indices of the examples (when they were first added).

    These indices start with the value -2**63 and increase with every
    call to the prefetch op.  Each whole example gets its own insertion
    index, and this is used to prioritize the example so that its truncated
    segments appear in adjacent iterations, even if new examples are inserted
    by the prefetch op between iterations.

    Returns:
      An int64 vector of length `batch_size`, the insertion indices.
    """
    return self._state_saver._received_indices

  @property
  def key(self):
    """The key names of the given truncated unrolled examples.

    The format of the key is:

    ```python
    "%05d_of_%05d:%s" % (sequence, sequence_count, original_key)
    ```

    where `original_key` is the unique key read in by the prefetcher.

    Returns:
      A string vector of length `batch_size`, the keys.
    """
    return self._state_saver._received_keys

  @property
  def next_key(self):
    """The key names of the next (in iteration) truncated unrolled examples.

    The format of the key is:

    ```python
    "%05d_of_%05d:%s" % (sequence + 1, sequence_count, original_key)
    ```

    if `sequence + 1 < sequence_count`, otherwise:

    ```python
    "STOP:%s" % original_key
    ```

    where `original_key` is the unique key read in by the prefetcher.

    Returns:
      A string vector of length `batch_size`, the keys.
    """
    return self._state_saver._received_next_key

  @property
  def sequence(self):
    """An int32 vector, length `batch_size`: the sequence index of each entry.

    When an input is split up, the sequence values
    ```
    0, 1, ..., sequence_count - 1
    ```
    are assigned to each split.

    Returns:
      An int32 vector `Tensor`.
    """
    return self._state_saver._received_sequence

  @property
  def sequence_count(self):
    """An int32 vector, length `batch_size`: the sequence count of each entry.

    When an input is split up, the number of splits is equal to:
    `padded_length / num_unroll`.  This is the sequence_count.

    Returns:
      An int32 vector `Tensor`.
    """
    return self._state_saver._received_sequence_count

  @property
  def context(self):
    """A dict mapping keys of `input_context` to batched context.

    Returns:
      A dict mapping keys of `input_context` to tensors.
      If we had at input:

      ```python
      context["name"].get_shape() == [d1, d2, ...]
      ```

      then for this property:

      ```python
      context["name"].get_shape() == [batch_size, d1, d2, ...]
      ```

    """
    return self._state_saver._received_context

  @property
  def sequences(self):
    """A dict mapping keys of `input_sequences` to split and rebatched data.

    Returns:
      A dict mapping keys of `input_sequences` to tensors.
      If we had at input:

      ```python
      sequences["name"].get_shape() == [None, d1, d2, ...]
      ```

      where `None` meant the sequence time was dynamic, then for this property:

      ```python
      sequences["name"].get_shape() == [batch_size, num_unroll, d1, d2, ...].
      ```

    """
    return self._state_saver._received_sequences

  def state(self, state_name):
    """Returns batched state tensors.

    Args:
      state_name: string, matches a key provided in `initial_states`.

    Returns:
      A `Tensor`: a batched set of states, either initial states (if this is
      the first run of the given example), or a value as stored during
      a previous iteration via `save_state` control flow.
      Its type is the same as `initial_states["state_name"].dtype`.
      If we had at input:

      ```python
      initial_states[state_name].get_shape() == [d1, d2, ...],
      ```

      then

      ```python
      state(state_name).get_shape() == [batch_size, d1, d2, ...]
      ```

    Raises:
      KeyError: if `state_name` does not match any of the initial states
        declared in `initial_states`.
    """
    return self._state_saver._received_states[state_name]

  def save_state(self, state_name, value, name=None):
    """Returns an op to save the current batch of state `state_name`.

    Args:
      state_name: string, matches a key provided in `initial_states`.
      value: A `Tensor`.
        Its type must match that of `initial_states[state_name].dtype`.
        If we had at input:

        ```python
        initial_states[state_name].get_shape() == [d1, d2, ...]
        ```

        then the shape of `value` must match:

        ```python
        tf.shape(value) == [batch_size, d1, d2, ...]
        ```

      name: string (optional).  The name scope for newly created ops.

    Returns:
      A control flow op that stores the new state of each entry into
      the state saver.  This op must be run for every iteration that
      accesses data from the state saver (otherwise the state saver
      will never progress through its states and run out of capacity).

    Raises:
      KeyError: if `state_name` does not match any of the initial states
        declared in `initial_states`.
    """
    if state_name not in self._state_saver._received_states.keys():
      raise KeyError("state was not declared: %s" % state_name)
    default_name = "InputQueueingStateSaver_SaveState"
    with ops.name_scope(name, default_name, values=[value]):
      # Place all operations on the CPU. Barriers and queues are only
      # implemented for CPU, but all the other book-keeping operations
      # (reshape, shape, range, ...) would be placed on GPUs if available,
      # unless we explicitly tie them to CPU.
      with ops.colocate_with(self._state_saver._capacity_queue.queue_ref):
        indices_where_not_done = array_ops.reshape(array_ops.where(
            math_ops.logical_not(self._state_saver._sequence_is_done)), [-1])
        keeping_next_key = array_ops.gather(
            self._state_saver._received_next_key, indices_where_not_done)
        value = _check_shape(
            array_ops.identity(value, name="convert_%s" % state_name),
            array_ops.shape(self._state_saver._received_states[state_name]))
        keeping_state = array_ops.gather(value, indices_where_not_done)
        return self._state_saver._barrier.insert_many(
            self._state_saver._get_barrier_index("state", state_name),
            keeping_next_key, keeping_state,
            name="BarrierInsertState_%s" % state_name)
# pylint: enable=protected-access


class SequenceQueueingStateSaver(object):
  """SequenceQueueingStateSaver provides access to stateful values from input.

  This class is meant to be used instead of, e.g., a `Queue`, for splitting
  variable-length sequence inputs into segments of sequences with fixed length
  and batching them into mini-batches.  It maintains contexts and state for a
  sequence across the segments.  It can be used in conjunction with a
  `QueueRunner` (see the example below).

  The `SequenceQueueingStateSaver` (SQSS) accepts one example at a time via the
  inputs `input_length`, `input_key`, `input_sequences` (a dict),
  `input_context` (a dict), and `initial_states` (a dict).
  The sequences, values in `input_sequences`, may have variable first dimension
  (the `padded_length`), though this dimension must always be a multiple of
  `num_unroll`.  All other dimensions must be fixed and accessible via
  `get_shape` calls.  The length prior to padding can be recorded in
  `input_length`.  The context values in `input_context` must all have fixed and
  well defined dimensions.  The initial state values must all have fixed and
  well defined dimensions.

  The SQSS splits the sequences of an input example into segments of length
  `num_unroll`.  Across examples minibatches of size `batch_size` are formed.
  These minibatches contain a segment of the sequences, copy the context values,
  and maintain state, length, and key information of the original input
  examples.  In the first segment of an example the state is still the initial
  state.  It can then be updated; and updated state values are accessible in
  subsequent segments of the same example. After each segment
  `batch.save_state()` must be called which is done by the state_saving_rnn.
  Without this call, the dequeue op associated with the SQSS will not run.
  Internally, SQSS has a queue for the input examples. Its `capacity` is
  configurable.  If set smaller than `batch_size` then the dequeue op will block
  indefinitely.  A small multiple of `batch_size` is a good rule of thumb to
  prevent that queue from becoming a bottleneck and slowing down training.
  If set too large (and note that it defaults to unbounded) memory consumption
  goes up.  Moreover, when iterating over the same input examples multiple times
  reusing the same `key` the `capacity` must be smaller than the number of
  examples.

  The prefetcher, which reads one unrolled, variable-length input sequence at
  a time, is accessible via `prefetch_op`.  The underlying `Barrier` object
  is accessible via `barrier`.  Processed minibatches, as well as
  state read and write capabilities are accessible via `next_batch`.
  Specifically, `next_batch` provides access to all of the minibatched
  data, including the following, see `NextQueuedSequenceBatch` for details:

  *  `total_length`, `length`, `insertion_index`, `key`, `next_key`,
  *  `sequence` (the index each minibatch entry's time segment index),
  *  `sequence_count` (the total time segment count for each minibatch entry),
  *  `context` (a dict of the copied minibatched context values),
  *  `sequences` (a dict of the split minibatched variable-length sequences),
  *  `state` (to access the states of the current segments of these entries)
  *  `save_state` (to save the states for the next segments of these entries)

  Example usage:

  ```python
  batch_size = 32
  num_unroll = 20
  lstm_size = 8
  cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_size)
  initial_state_values = tf.zeros(cell.state_size, dtype=tf.float32)

  raw_data = get_single_input_from_input_reader()
  length, key, sequences, context = my_parser(raw_data)
  assert "input" in sequences.keys()
  assert "label" in context.keys()
  initial_states = {"lstm_state": initial_state_value}

  stateful_reader = tf.SequenceQueueingStateSaver(
      batch_size, num_unroll,
      length=length, input_key=key, input_sequences=sequences,
      input_context=context, initial_states=initial_states,
      capacity=batch_size*100)

  batch = stateful_reader.next_batch
  inputs = batch.sequences["input"]
  context_label = batch.context["label"]

  inputs_by_time = tf.split(value=inputs, num_or_size_splits=num_unroll, axis=1)
  assert len(inputs_by_time) == num_unroll

  lstm_output, _ = tf.contrib.rnn.static_state_saving_rnn(
    cell,
    inputs_by_time,
    state_saver=batch,
    state_name="lstm_state")

  # Start a prefetcher in the background
  sess = tf.Session()
  num_threads = 3
  queue_runner = tf.train.QueueRunner(
      stateful_reader, [stateful_reader.prefetch_op] * num_threads)
  tf.train.add_queue_runner(queue_runner)
  tf.train.start_queue_runners(sess=session)

  while True:
    # Step through batches, perform training or inference...
    session.run([lstm_output])
  ```

  **Note**: Usually the barrier is given to a QueueRunner as in the
      examples above.  The QueueRunner will close the barrier if the prefetch_op
      receives an OutOfRange Error from upstream input queues (i.e., reaches
      the end of the input).  If the barrier is closed no further new examples
      are added to the SQSS.  The underlying barrier might, however, still
      contain further unroll-steps of examples that have not undergone all
      iterations.  To gracefully finish all examples, the flag
      `allow_small_batch` must be set to true, which causes the SQSS to issue
      progressively smaller mini-batches with the remaining examples.
  """

  def __init__(self,
               batch_size,
               num_unroll,
               input_length,
               input_key,
               input_sequences,
               input_context,
               initial_states,
               capacity=None,
               allow_small_batch=False,
               name=None):
    """Creates the SequenceQueueingStateSaver.

    Args:
      batch_size: int or int32 scalar `Tensor`, how large minibatches should
        be when accessing the `state()` method and `context`, `sequences`, etc,
        properties.
      num_unroll: Python integer, how many time steps to unroll at a time.
        The input sequences of length `k` are then split into `k / num_unroll`
        many segments.
      input_length: An int32 scalar `Tensor`, the length of the sequence prior
        to padding.  This value may be at most `padded_length` for any given
        input (see below for the definition of `padded_length`).
        Batched and total lengths of the current iteration are made accessible
        via the `length` and `total_length` properties.  The shape of
        input_length (scalar) must be fully specified.
      input_key: A string scalar `Tensor`, the **unique** key for the given
        input.  This is used to keep track of the split minibatch elements
        of this input.  Batched keys of the current iteration are made
        accessible via the `key` property.  The shape of `input_key` (scalar)
        must be fully specified.
      input_sequences: A dict mapping string names to `Tensor` values.  The
        values must all have matching first dimension, called `padded_length`.
        The `SequenceQueueingStateSaver` will split these tensors along
        this first dimension into minibatch elements of dimension
        `num_unroll`. Batched and segmented sequences of the current iteration
        are made accessible via the `sequences` property.

        **Note**: `padded_length` may be dynamic, and may vary from input
        to input, but must always be a multiple of `num_unroll`.  The remainder
        of the shape (other than the first dimension) must be fully specified.
      input_context: A dict mapping string names to `Tensor` values.  The values
        are treated as "global" across all time splits of the given input,
        and will be copied across for all minibatch elements accordingly.
        Batched and copied context of the current iteration are made
        accessible via the `context` property.

        **Note**: All input_context values must have fully defined shapes.
      initial_states: A dict mapping string state names to multi-dimensional
        values (e.g. constants or tensors).  This input defines the set of
        states that will be kept track of during computing iterations, and
        which can be accessed via the `state` and `save_state` methods.

        **Note**: All initial_state values must have fully defined shapes.
      capacity: The max capacity of the SQSS in number of examples. Needs to be
        at least `batch_size`. Defaults to unbounded.
      allow_small_batch: If true, the SQSS will return smaller batches when
        there aren't enough input examples to fill a whole batch and the end of
        the input has been reached (i.e., the underlying barrier has been
        closed).
      name: An op name string (optional).

    Raises:
      TypeError: if any of the inputs is not an expected type.
      ValueError: if any of the input values is inconsistent, e.g. if
      not enough shape information is available from inputs to build
      the state saver.
    """

    if capacity is not None and capacity < batch_size:
      raise ValueError("capacity must be larger or equal to batch_size")
    # The barrier is ignorant of the number of actual examples, since a long
    # example that requires many iterations produces more elements in the
    # barrier than a short example. Furthermore, we don't have an upper bound
    # on the length of examples, and hence have to keep the capacity of the
    # barrier at infinite to avoid dead-lock. Instead we have to keep track of
    # the number of active examples in this class, and block the prefetch_op
    # when capacity is reached. To this end, we employ a FIFOQueue in which we
    # store one token (its value doesn't matter) for each input example, and
    # dequeue a token for each completed example. Since the capacity of this
    # queue is limited the enqueue operation will block if capacity is reached.
    self._capacity_queue = data_flow_ops.FIFOQueue(capacity=capacity,
                                                   dtypes=[dtypes.int32],
                                                   shapes=[[]])
    # Place all operations on the CPU. Barriers and queues are only implemented
    # for CPU, but all the other book-keeping operations
    # (reshape, shape, range, ...) would be placed on GPUs if available,
    # unless we explicitly tie them to CPU.
    with ops.colocate_with(self._capacity_queue.queue_ref):
      if not isinstance(initial_states, dict):
        raise TypeError("initial_states must be a dictionary")
      if not initial_states:
        raise ValueError(
            "initial_states may not be empty: at least one state variable is "
            "required to properly enqueue split sequences to run in separate "
            "iterations")
      for k in initial_states:
        if not isinstance(k, six.string_types):
          raise TypeError("state name must be a string: %s" % k)
        if ":" in k:
          raise ValueError("state name may not have a colon: '%s'" % k)

      op_vars = ([input_length, input_key]
                 + list(input_sequences.values())
                 + list(input_context.values()))
      with ops.name_scope(name, "InputQueueingStateSaver", op_vars) as scope:
        inputs = _SequenceInputWrapper(
            input_length, input_key, input_sequences, input_context)
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._name = scope

        # This step makes sure all shapes are well defined.  We can now
        # use get_shape() on any tensor in the output of this function
        # and get a fully-defined shape.
        (self._length, self._key, self._sorted_states, self._sorted_sequences,
         self._sorted_context) = _prepare_sequence_inputs(inputs,
                                                          initial_states)
        self._padded_length = array_ops.identity(
            array_ops.shape(
                six.next(six.itervalues(self._sorted_sequences)))[0],
            name="padded_length")  # The name is useful for debugging
        self._padded_length = _check_multiple_of(
            self._padded_length, self._num_unroll)

        # sequences should have length == all matching
        self._sorted_sequences = collections.OrderedDict(
            (k, _check_dimensions(v, [0], [self._padded_length],
                                  debug_prefix="sorted_sequences_%s" % k))
            for k, v in self._sorted_sequences.items())
        self._uninitialized_states = self._sorted_states

        # Once this is set, self._get_barrier_*_index are available for use.
        self._store_index_maps(
            self._sorted_sequences, self._sorted_context, self._sorted_states)

        # Make sure that the length is <= the padded_length
        with ops.control_dependencies([
            control_flow_ops.Assert(
                math_ops.less_equal(self._length, self._padded_length),
                ["Input length should be <= than length from sequences:",
                 self._length, " vs. ", self._padded_length])]):
          self._length = array_ops.identity(self._length)

        # Only create barrier; enqueu and dequeue operations happen when you
        # access prefetch_op and next_batch.
        self._create_barrier()
        self._scope = scope
      self._allow_small_batch = allow_small_batch
      self._prefetch_op = None
      self._next_batch = None

  @property
  def name(self):
    return self._name

  @property
  def barrier(self):
    return self._barrier

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def num_unroll(self):
    return self._num_unroll

  @property
  def prefetch_op(self):
    """The op used to prefetch new data into the state saver.

    Running it once enqueues one new input example into the state saver.
    The first time this gets called, it additionally creates the prefetch_op.
    Subsequent calls simply return the previously created `prefetch_op`.

    It should be run in a separate thread via e.g. a `QueueRunner`.

    Returns:
      An `Operation` that performs prefetching.
    """
    if not self._prefetch_op:
      with ops.name_scope(None), ops.name_scope(
          self._scope, values=[self._barrier.barrier_ref]):
        self._create_prefetch_op()
    return self._prefetch_op

  @property
  def next_batch(self):
    """The `NextQueuedSequenceBatch` providing access to batched output data.

    Also provides access to the `state` and `save_state` methods.
    The first time this gets called, it additionally prepares barrier reads
    and creates `NextQueuedSequenceBatch` / next_batch objects. Subsequent
    calls simply return the previously created `next_batch`.

    In order to access data in `next_batch` without blocking, the `prefetch_op`
    must have been run at least `batch_size` times (ideally in a separate
    thread, or launched via a `QueueRunner`). After processing a segment in
    `next_batch()`, `batch.save_state()` must be called which is done by the
    state_saving_rnn. Without this call, the dequeue op associated with the SQSS
    will not run.

    Returns:
      A cached `NextQueuedSequenceBatch` instance.
    """
    # This is needed to prevent errors if next_batch is called before
    # prefetch_op is created.
    if not self._prefetch_op:
      with ops.name_scope(None), ops.name_scope(
          self._scope, values=[self._barrier.barrier_ref]):
        self._create_prefetch_op()
    if not self._next_batch:
      with ops.name_scope(None), ops.name_scope(
          self._scope, values=[self._barrier.barrier_ref]):
        self._prepare_barrier_reads()
    return self._next_batch

  def close(self, cancel_pending_enqueues=False, name=None):
    """Closes the barrier and the FIFOQueue.

    This operation signals that no more segments of new sequences will be
    enqueued. New segments of already inserted sequences may still be enqueued
    and dequeued if there is a sufficient number filling a batch or
    allow_small_batch is true. Otherwise dequeue operations will fail
    immediately.

    Args:
      cancel_pending_enqueues: (Optional.) A boolean, defaulting to
        `False`. If `True`, all pending enqueues to the underlying queues will
        be cancelled, and completing already started sequences is not possible.
      name: Optional name for the op.

    Returns:
      The operation that closes the barrier and the FIFOQueue.
    """
    with ops.name_scope(name, "SQSSClose", [self._prefetch_op]) as name:
      barrier_close = self.barrier.close(
          cancel_pending_enqueues, "BarrierClose")
      fifo_queue_close = self._capacity_queue.close(
          cancel_pending_enqueues, "FIFOClose")
      return control_flow_ops.group(barrier_close, fifo_queue_close, name=name)

  def _store_index_maps(self, sequences, context, states):
    """Prepares the internal dictionaries _name_to_index and _index_to_name.

    These dictionaries are used to keep track of indices into the barrier.

    Args:
      sequences: `OrderedDict` of string, `Tensor` pairs.
      context: `OrderedDict` of string, `Tensor` pairs.
      states: `OrderedDict` of string, `Tensor` pairs.
    """
    assert isinstance(sequences, dict)
    assert isinstance(context, dict)
    assert isinstance(states, dict)
    self._name_to_index = dict((name, ix) for (ix, name) in enumerate(
        ["__length", "__total_length", "__next_key",
         "__sequence", "__sequence_count"]
        + ["__sequence__%s" % k for k in sequences.keys()]
        + ["__context__%s" % k for k in context.keys()]
        + ["__state__%s" % k for k in states.keys()]))
    self._index_to_name = [
        name for (name, _) in sorted(
            self._name_to_index.items(), key=lambda n_ix: n_ix[1])]

  def _get_barrier_length_index(self):
    return self._name_to_index["__length"]

  def _get_barrier_total_length_index(self):
    return self._name_to_index["__total_length"]

  def _get_barrier_next_key_index(self):
    return self._name_to_index["__next_key"]

  def _get_barrier_sequence_index(self):
    return self._name_to_index["__sequence"]

  def _get_barrier_sequence_count_index(self):
    return self._name_to_index["__sequence_count"]

  def _get_barrier_index(self, index_type, name):
    assert index_type in ("sequence", "context", "state")
    key = "__%s__%s" % (index_type, name)
    assert key in self._name_to_index, (
        "Requested a name not in the value type %s: %s" % (index_type, name))
    return self._name_to_index[key]

  def _create_barrier(self):
    """Create the barrier.

    This method initializes the Barrier object with the right types and shapes.
    """
    # Create the barrier
    sequence_dtypes = [v.dtype for k, v in self._sorted_sequences.items()]
    context_dtypes = [v.dtype for k, v in self._sorted_context.items()]
    state_dtypes = [v.dtype for k, v in self._sorted_states.items()]
    types = ([dtypes.int32,   # length
              dtypes.int32,   # total_length
              dtypes.string,  # next_keys
              dtypes.int32,   # sequence
              dtypes.int32]   # expanded_sequence_count
             + sequence_dtypes + context_dtypes + state_dtypes)
    sequence_shapes = [
        [self._num_unroll] + self._sorted_sequences[k].get_shape().as_list()[1:]
        for k in self._sorted_sequences.keys()]
    context_shapes = [
        self._sorted_context[k].get_shape().as_list()
        for k in self._sorted_context.keys()]
    state_shapes = [
        self._sorted_states[k].get_shape().as_list()
        for k in self._sorted_states.keys()]
    shapes = ([(),  # length
               (),  # total_length
               (),  # next_keys
               (),  # sequence
               ()]  # expanded_sequence_count
              + sequence_shapes + context_shapes + state_shapes)

    self._barrier = data_flow_ops.Barrier(types=types, shapes=shapes)

  def _create_prefetch_op(self):
    """Group insert_many ops and create prefetch_op.

    This method implements the "meat" of the logic underlying the
    `SequenceQueueingStateSaver`.  It performs dynamic reshaping of
    sequences, copying of context, and initial insertion of these values,
    as well as the key, next_key, sequence, sequence_count, and initial
    states into the barrier.
    """
    # Step 1: identify how many barrier entries to split this input
    # into, store the result as a scalar
    sequence_count = math_ops.div(self._padded_length, self._num_unroll)
    sequence_count_vec = array_ops.expand_dims(sequence_count, 0)

    # The final unrolled sequence's length is num_unroll only in
    # the case that num_unroll divides it evenly.
    ones = array_ops.ones(sequence_count_vec, dtype=dtypes.int32)
    sequence = math_ops.range(sequence_count)
    expanded_length = math_ops.maximum(
        0, self._length - self._num_unroll * sequence)
    expanded_length = math_ops.minimum(self._num_unroll, expanded_length)
    expanded_total_length = self._length * ones
    expanded_sequence_count = sequence_count * ones
    current_keys = string_ops.string_join(
        [string_ops.as_string(sequence, width=5, fill="0"),
         "_of_",
         string_ops.as_string(sequence_count, width=5, fill="0"),
         ":",
         self._key],
        name="StringJoinCurrentKeys")
    next_keys = array_ops.concat_v2(
        [
            array_ops.slice(current_keys, [1], [-1]), array_ops.expand_dims(
                string_ops.string_join(
                    ["STOP:", self._key], name="StringJoinStop"),
                0)
        ],
        0,
        name="concat_next_keys")
    reshaped_sequences = collections.OrderedDict((
        k,
        _check_dimensions(
            # Reshape sequences to sequence_count rows
            array_ops.reshape(
                v,
                array_ops.concat_v2(
                    [
                        array_ops.expand_dims(sequence_count, 0),
                        array_ops.expand_dims(self._num_unroll, 0),
                        v.get_shape().as_list()[1:]
                    ],
                    0,
                    name="concat_sequences_%s" % k),
                name="reshape_sequences_%s" % k),
            [0, 1] + list(range(2, v.get_shape().ndims + 1)),
            [sequence_count, self._num_unroll] + v.get_shape().as_list()[1:],
            debug_prefix="reshaped_sequences_%s" %
            k)) for k, v in self._sorted_sequences.items())
    expanded_context = collections.OrderedDict(
        (
            k,
            _check_dimensions(
                # Copy context to be sequence_count rows
                array_ops.tile(
                    array_ops.expand_dims(v, 0),
                    array_ops.concat_v2(
                        [
                            array_ops.expand_dims(sequence_count, 0),
                            [1] * v.get_shape().ndims
                        ],
                        0,
                        name="concat_context_%s" % k),
                    name="tile_context_%s" % k),
                [0] + list(range(1, v.get_shape().ndims + 1)),
                [sequence_count] + v.get_shape().as_list(),
                debug_prefix="expanded_context_%s" % k))
        for k, v in self._sorted_context.items())

    # Storing into the barrier, for each current_key:
    #   sequence_ix, sequence_count, next_key, length,
    #   context... (copied), sequences... (truncated)
    # Also storing into the barrier for the first key
    #   states (using initial_states).
    insert_sequence_op = self._barrier.insert_many(
        self._get_barrier_sequence_index(),
        current_keys, sequence,
        name="BarrierInsertSequence")
    insert_sequence_count_op = self._barrier.insert_many(
        self._get_barrier_sequence_count_index(),
        current_keys, expanded_sequence_count,
        name="BarrierInsertSequenceCount")
    insert_next_key_op = self._barrier.insert_many(
        self._get_barrier_next_key_index(),
        current_keys, next_keys,
        name="BarrierInsertNextKey")
    insert_length_op = self._barrier.insert_many(
        self._get_barrier_length_index(),
        current_keys, expanded_length,
        name="BarrierInsertLength")
    insert_total_length_op = self._barrier.insert_many(
        self._get_barrier_total_length_index(),
        current_keys, expanded_total_length,
        name="BarrierInsertTotalLength")
    insert_context_ops = dict(
        (name, self._barrier.insert_many(
            self._get_barrier_index("context", name),
            current_keys, value,
            name="BarrierInsertContext_%s" % name))
        for (name, value) in expanded_context.items())
    insert_sequences_ops = dict(
        (name, self._barrier.insert_many(
            self._get_barrier_index("sequence", name),
            current_keys, value,
            name="BarrierInsertSequences_%s" % name))
        for (name, value) in reshaped_sequences.items())

    # An op that blocks if we reached capacity in number of active examples.
    TOKEN_WITH_IGNORED_VALUE = 21051976  # pylint: disable=invalid-name
    insert_capacity_token_op = self._capacity_queue.enqueue((
        TOKEN_WITH_IGNORED_VALUE,))

    # Insert just the initial state.  Specifically force this to run
    # the insert sequence op *first* so that the Barrier receives
    # an insert with *all* the segments and the segments all get the same index.
    with ops.control_dependencies([insert_sequence_op,
                                   insert_capacity_token_op]):
      insert_initial_state_ops = dict(
          (name, self._barrier.insert_many(
              self._get_barrier_index("state", name),
              array_ops.stack([current_keys[0]]),
              array_ops.stack([value]),
              name="BarrierInitialInsertState_%s" % name))
          for (name, value) in self._uninitialized_states.items())

    all_inserts = (
        [insert_capacity_token_op,
         insert_sequence_op,
         insert_sequence_count_op,
         insert_next_key_op,
         insert_length_op,
         insert_total_length_op]
        + list(insert_initial_state_ops.values())
        + list(insert_context_ops.values())
        + list(insert_sequences_ops.values()))

    self._prefetch_op = control_flow_ops.group(
        *all_inserts, name="StateSaverPrefetchGroup")

  def _prepare_barrier_reads(self):
    """Creates ops for reading the barrier, as used by properties like `length`.
    """
    # Ops for reading from the barrier.  These ops must be run in a
    # different thread than the prefetcher op to avoid blocking.
    received = self._barrier.take_many(self._batch_size,
                                       self._allow_small_batch,
                                       name="BarrierTakeMany")

    self._received_indices = received[0]
    self._received_keys = received[1]
    received_values = received[2]

    self._received_sequence = received_values[
        self._get_barrier_sequence_index()]
    self._received_sequence_count = received_values[
        self._get_barrier_sequence_count_index()]
    self._received_next_key = received_values[
        self._get_barrier_next_key_index()]
    self._received_length = received_values[
        self._get_barrier_length_index()]
    self._received_total_length = received_values[
        self._get_barrier_total_length_index()]
    self._received_context = collections.OrderedDict(
        (name, received_values[self._get_barrier_index("context", name)])
        for name in self._sorted_context.keys())
    self._received_sequences = collections.OrderedDict(
        (name, received_values[self._get_barrier_index("sequence", name)])
        for name in self._sorted_sequences.keys())

    self._received_batch_size = array_ops.squeeze(array_ops.shape(
        self._received_length))

    # Which examples are we done with?
    self._sequence_is_done = (
        self._received_sequence + 1 >= self._received_sequence_count)

    # Compute the number of finished sequences and dequeue as many tokens from
    # the capacity queue.
    finished_sequences = (math_ops.reduce_sum(math_ops.cast(
        self._sequence_is_done, dtypes.int32)))
    # TODO(ebrevdo): convert to dequeue_up_to when FIFOQueue supports it.
    dequeue_op = self._capacity_queue.dequeue_many(finished_sequences)

    # Tie the dequeue_op to the received_state, such that it is definitely
    # carried out.
    with ops.control_dependencies([dequeue_op]):
      self._received_states = collections.OrderedDict((
          name, array_ops.identity(received_values[self._get_barrier_index(
              "state", name)])) for name in self._sorted_states.keys())
    self._next_batch = NextQueuedSequenceBatch(self)


def batch_sequences_with_states(input_key, input_sequences, input_context,
                                input_length, initial_states, num_unroll,
                                batch_size, num_threads=3, capacity=1000,
                                allow_small_batch=True, pad=True, name=None):
  """Creates batches of segments of sequential input.

  This method creates a `SequenceQueueingStateSaver` (SQSS) and adds it to
  the queuerunners. It returns a `NextQueuedSequenceBatch`.

  It accepts one example at a time identified by a unique `input_key`.
  `input_sequence` is a dict with values that are tensors with time as first
  dimension. This time dimension must be the same across those tensors of an
  example. It can vary across examples. Although it always has to be a multiple
  of `num_unroll`. Hence, padding may be necessary and it is turned on by
  default by `pad=True`.

  `input_length` is a Tensor scalar or an int recording the time dimension prior
  to padding. It should be between 0 and the time dimension. One reason we want
  to keep track of it is so that we can take it into consideration when
  computing the loss. If `pad=True` then `input_length` can be `None` and will
  be inferred.

  This methods segments `input_sequence` into segments of length `num_unroll`.
  It batches input sequences from `batch_size` many examples. These mini-batches
  are available through the `sequence` property of the output. Moreover, for
  each entry in the batch we can access its original `input_key` in `key` and
  its input length in `total_length`. `length` records within this segment how
  many non-padded time steps there are.

  Static features of an example that do not vary across time can be part of the
  `input_context`, a dict with Tensor values. This method copies the context for
  each segment and makes it available in the `context` of the output.

  This method can maintain and update a state for each example. It accepts some
  initial_states as a dict with Tensor values. The first mini-batch an example
  is contained has initial_states as entry of the `state`. If save_state is
  called then the next segment will have the updated entry of the `state`.
  See `NextQueuedSequenceBatch` for a complete list of properties and methods.

  Example usage:

  ```python
  batch_size = 32
  num_unroll = 20
  num_enqueue_threads = 3
  lstm_size = 8
  cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_size)

  key, sequences, context = my_parser(raw_data)
  initial_state_values = tf.zeros((state_size,), dtype=tf.float32)
  initial_states = {"lstm_state": initial_state_values}
  batch = tf.batch_sequences_with_states(
      input_key=key,
      input_sequences=sequences,
      input_context=context,
      initial_states=initial_states,
      num_unroll=num_unroll,
      batch_size=batch_size,
      num_threads=num_enqueue_threads,
      capacity=batch_size * num_enqueue_threads * 2)

  inputs = batch.sequences["input"]
  context_label = batch.context["label"]

  inputs_by_time = tf.split(value=inputs, num_or_size_splits=num_unroll, axis=1)
  assert len(inputs_by_time) == num_unroll

  lstm_output, _ = tf.contrib.rnn.static_state_saving_rnn(
    cell,
    inputs_by_time,
    state_saver=batch,
    state_name="lstm_state")

  # Start a prefetcher in the background
  sess = tf.Session()

  tf.train.start_queue_runners(sess=session)

  while True:
    # Step through batches, perform training or inference...
    session.run([lstm_output])
  ```

  Args:
    input_key: A string scalar `Tensor`, the **unique** key for the given
      input example.  This is used to keep track of the split minibatch elements
      of this input.  Batched keys of the current iteration are made
      accessible via the `key` property.  The shape of `input_key` (scalar) must
      be fully specified.
    input_sequences: A dict mapping string names to `Tensor` values.  The values
      must all have matching first dimension, called `value_length`. They may
      vary from input to input. The remainder of the shape (other than the first
      dimension) must be fully specified.
      The `SequenceQueueingStateSaver` will split these tensors along
      this first dimension into minibatch elements of dimension `num_unrolled`.
      Batched and segmented sequences of the current iteration are made
      accessible via the `sequences` property.

      **Note**: if `pad=False`, then `value_length` must always be a multiple
        of `num_unroll`.
    input_context: A dict mapping string names to `Tensor` values.  The values
      are treated as "global" across all time splits of the given input example,
      and will be copied across for all minibatch elements accordingly.
      Batched and copied context of the current iteration are made
      accessible via the `context` property.

      **Note**: All input_context values must have fully defined shapes.
    input_length: None or an int32 scalar `Tensor`, the length of the sequence
      prior to padding. If `input_length=None` and `pad=True` then the length
      will be inferred and will be equal to `value_length`. If `pad=False` then
      `input_length` cannot be `None`: `input_length` must be specified. Its
      shape of `input_length` (scalar) must be fully specified. Its value may be
      at most `value_length` for any given input (see above for the definition
      of `value_length`). Batched and total lengths of the current iteration are
      made accessible via the `length` and `total_length` properties.
    initial_states: A dict mapping string state names to multi-dimensional
      values (e.g. constants or tensors).  This input defines the set of
      states that will be kept track of during computing iterations, and
      which can be accessed via the `state` and `save_state` methods.

      **Note**: All initial_state values must have fully defined shapes.
    num_unroll: Python integer, how many time steps to unroll at a time.
      The input sequences of length k are then split into k / num_unroll many
      segments.
    batch_size: int or int32 scalar `Tensor`, how large minibatches should
      be when accessing the `state()` method and `context`, `sequences`, etc,
      properties.
    num_threads: The int number of threads enqueuing input examples into a
      queue.
    capacity: The max capacity of the queue in number of examples. Needs to be
      at least `batch_size`. Defaults to 1000. When iterating over the same
      input example multiple times reusing their keys the `capacity` must be
      smaller than the number of examples.
    allow_small_batch: If true, the queue will return smaller batches when
      there aren't enough input examples to fill a whole batch and the end of
      the input has been reached.
    pad: If `True`, `input_sequences` will be padded to multiple of
      `num_unroll`. In that case `input_length` may be `None` and is assumed to
      be the length of first dimension of values in `input_sequences`
      (i.e. `value_length`).
    name: An op name string (optional).

  Returns:
    A NextQueuedSequenceBatch with segmented and batched inputs and their
    states.

  Raises:
    TypeError: if any of the inputs is not an expected type.
    ValueError: if any of the input values is inconsistent, e.g. if
      not enough shape information is available from inputs to build
      the state saver.
  """
  tensor_list = (
      list(input_sequences.values()) + list(input_context.values()) +
      list(initial_states.values()))
  with ops.name_scope(name, "batch_sequences_with_states", tensor_list) as name:
    if pad:
      length, input_sequences = _padding(input_sequences, num_unroll)
      input_length = input_length if input_length is not None else length
    elif input_sequences:
      # Assert that value_length is a multiple of num_unroll.
      for key, value in input_sequences.items():
        value_length = array_ops.shape(value)[0]
        with ops.control_dependencies([
            control_flow_ops.Assert(
                math_ops.logical_and(
                    math_ops.equal(value_length % num_unroll, 0),
                    math_ops.not_equal(value_length, 0)),
                [string_ops.string_join(
                    ["Tensor %s first dimension should be a multiple of: "
                     % key,
                     string_ops.as_string(num_unroll),
                     ", but saw value: ",
                     string_ops.as_string(value_length),
                     ". Consider setting pad=True."])])]):
          input_sequences[key] = array_ops.identity(
              value, name="multiple_of_checked")

    # setup stateful queue reader
    stateful_reader = SequenceQueueingStateSaver(
        batch_size, num_unroll,
        input_length=input_length,
        input_key=input_key,
        input_sequences=input_sequences,
        input_context=input_context,
        initial_states=initial_states,
        capacity=capacity,
        allow_small_batch=allow_small_batch)

    barrier = stateful_reader.barrier
    summary.scalar("queue/%s/ready_segment_batches_" % barrier.name,
                   math_ops.cast(barrier.ready_size(), dtypes.float32))

    q_runner = queue_runner.QueueRunner(
        stateful_reader, [stateful_reader.prefetch_op]*num_threads,
        queue_closed_exception_types=(errors.OutOfRangeError,
                                      errors.CancelledError))
    queue_runner.add_queue_runner(q_runner)
    return stateful_reader.next_batch


def _padding(sequences, num_unroll):
  """For a dictionary of sequences, pads tensors to a multiple of `num_unroll`.

  Args:
    sequences: dictionary with `Tensor` values.
    num_unroll: int specifying to what multiple to pad sequences to.
  Returns:
    length: Scalar `Tensor` of dimension 0 of all the values in sequences.
    padded_sequence: Dictionary of sequences that are padded to a multiple of
      `num_unroll`.
  Raises:
    ValueError: If `num_unroll` not an int or sequences not a dictionary from
                string to `Tensor`.
  """
  if not isinstance(num_unroll, numbers.Integral):
    raise ValueError("Unsupported num_unroll expected int, got: %s" %
                     str(num_unroll))
  if not isinstance(sequences, dict):
    raise TypeError("Unsupported sequences expected dict, got: %s" %
                    str(sequences))
  for key, value in sequences.items():
    if not isinstance(key, six.string_types):
      raise TypeError("Unsupported sequences key expected string, got: %s" %
                      str(key))
  if not sequences:
    return 0, {}

  sequences_dict = {}
  for key, value in sequences.items():
    sequences_dict[key] = ops.convert_to_tensor(value)

  lengths = [array_ops.shape(value)[0] for value in sequences_dict.values()]
  length = lengths[0]
  all_lengths_equal = [
      control_flow_ops.Assert(
          math_ops.equal(l, length), [string_ops.string_join(
              ["All sequence lengths must match, but received lengths: ",
               string_ops.as_string(lengths)])])
      for l in lengths]

  length = control_flow_ops.with_dependencies(all_lengths_equal, length)
  unroll = array_ops.constant(num_unroll)
  padded_length = length + ((unroll - (length % unroll)) % unroll)
  padded_sequences = {}
  for key, value in sequences_dict.items():
    # 1. create shape of paddings
    # first dimension of value will be increased by num_paddings to
    # padded_length
    num_paddings = [padded_length - array_ops.shape(value)[0]]
    # the shape of the paddings that we concat with the original value will be
    # [num_paddings, tf.shape(value)[1], tf.shape(value)[2], ...,
    #  tf.shape(value)[tf.rank(value) - 1])]
    padding_shape = array_ops.concat_v2((num_paddings,
                                         array_ops.shape(value)[1:]), 0)
    # 2. fill padding shape with dummies
    dummy = array_ops.constant("" if value.dtype == dtypes.string else 0,
                               dtype=value.dtype)
    paddings = array_ops.fill(dims=padding_shape, value=dummy)
    # 3. concat values with paddings
    padded_sequences[key] = array_ops.concat_v2([value, paddings], 0)
  return length, padded_sequences

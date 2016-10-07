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

"""Input pipeline.

Please see the [reading data how-to](../../how_tos/reading_data/index.md)
for context.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import queue_runner


def match_filenames_once(pattern, name=None):
  """Save the list of files matching pattern, so it is only computed once.

  Args:
    pattern: A file pattern (glob).
    name: A name for the operations (optional).

  Returns:
    A variable that is initialized to the list of files matching pattern.
  """
  with ops.name_scope(name, "matching_filenames", [pattern]) as name:
    return variables.Variable(io_ops.matching_files(pattern), trainable=False,
                              name=name, validate_shape=False)


def limit_epochs(tensor, num_epochs=None, name=None):
  """Returns tensor `num_epochs` times and then raises an `OutOfRange` error.

  Args:
    tensor: Any `Tensor`.
    num_epochs: A positive integer (optional).  If specified, limits the number
      of steps the output tensor may be evaluated.
    name: A name for the operations (optional).

  Returns:
    tensor or `OutOfRange`.

  Raises:
    ValueError: if `num_epochs` is invalid.
  """
  if num_epochs is None:
    return tensor
  if num_epochs <= 0:
    raise ValueError("num_epochs must be > 0 not %d." % num_epochs)
  with ops.name_scope(name, "limit_epochs", [tensor]) as name:
    zero64 = constant_op.constant(0, dtype=dtypes.int64)
    epochs = variables.Variable(
        zero64, name="epochs", trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES])
    counter = epochs.count_up_to(num_epochs)
    with ops.control_dependencies([counter]):
      return array_ops.identity(tensor, name=name)


def input_producer(input_tensor, element_shape=None, num_epochs=None,
                   shuffle=True, seed=None, capacity=32, shared_name=None,
                   summary_name=None, name=None):
  """Output the rows of `input_tensor` to a queue for an input pipeline.

  Args:
    input_tensor: A tensor with the rows to produce. Must be at least
      one-dimensional. Must either have a fully-defined shape, or
      `element_shape` must be defined.
    element_shape: (Optional.) A `TensorShape` representing the shape of a
      row of `input_tensor`, if it cannot be inferred.
    num_epochs: (Optional.) An integer. If specified `input_producer` produces
      each row of `input_tensor` `num_epochs` times before generating an
      `OutOfRange` error. If not specified, `input_producer` can cycle through
      the rows of `input_tensor` an unlimited number of times.
    shuffle: (Optional.) A boolean. If true, the rows are randomly shuffled
      within each epoch.
    seed: (Optional.) An integer. The seed to use if `shuffle` is true.
    capacity: (Optional.) The capacity of the queue to be used for buffering
      the input.
    shared_name: (Optional.) If set, this queue will be shared under the given
      name across multiple sessions.
    summary_name: (Optional.) If set, a scalar summary for the current queue
      size will be generated, using this name as part of the tag.
    name: (Optional.) A name for queue.

  Returns:
    A queue with the output rows.  A `QueueRunner` for the queue is
    added to the current `QUEUE_RUNNER` collection of the current
    graph.

  Raises:
    ValueError: If the shape of the input cannot be inferred from the arguments.
  """
  with ops.name_scope(name, "input_producer", [input_tensor]):
    input_tensor = ops.convert_to_tensor(input_tensor, name="input_tensor")
    element_shape = input_tensor.get_shape()[1:].merge_with(element_shape)
    if not element_shape.is_fully_defined():
      raise ValueError("Either `input_tensor` must have a fully defined shape "
                       "or `element_shape` must be specified")

    if shuffle:
      input_tensor = random_ops.random_shuffle(input_tensor, seed=seed)

    input_tensor = limit_epochs(input_tensor, num_epochs)

    q = data_flow_ops.FIFOQueue(capacity=capacity,
                                dtypes=[input_tensor.dtype.base_dtype],
                                shapes=[element_shape],
                                shared_name=shared_name, name=name)
    enq = q.enqueue_many([input_tensor])
    queue_runner.add_queue_runner(queue_runner.QueueRunner(q, [enq]))
    if summary_name is not None:
      logging_ops.scalar_summary("queue/%s/%s" % (q.name, summary_name),
                                 math_ops.cast(q.size(), dtypes.float32) *
                                 (1. / capacity))
    return q


def string_input_producer(string_tensor, num_epochs=None, shuffle=True,
                          seed=None, capacity=32, shared_name=None, name=None):
  """Output strings (e.g. filenames) to a queue for an input pipeline.

  Args:
    string_tensor: A 1-D string tensor with the strings to produce.
    num_epochs: An integer (optional). If specified, `string_input_producer`
      produces each string from `string_tensor` `num_epochs` times before
      generating an `OutOfRange` error. If not specified,
      `string_input_producer` can cycle through the strings in `string_tensor`
      an unlimited number of times.
    shuffle: Boolean. If true, the strings are randomly shuffled within each
      epoch.
    seed: An integer (optional). Seed used if shuffle == True.
    capacity: An integer. Sets the queue capacity.
    shared_name: (optional). If set, this queue will be shared under the given
      name across multiple sessions.
    name: A name for the operations (optional).

  Returns:
    A queue with the output strings.  A `QueueRunner` for the Queue
    is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  Raises:
    ValueError: If the string_tensor is a null Python list.  At runtime,
    will fail with an assertion if string_tensor becomes a null tensor.
  """
  not_null_err = "string_input_producer requires a non-null input tensor"
  if not isinstance(string_tensor, ops.Tensor) and not string_tensor:
    raise ValueError(not_null_err)

  with ops.name_scope(name, "input_producer", [string_tensor]) as name:
    string_tensor = ops.convert_to_tensor(string_tensor, dtype=dtypes.string)
    with ops.control_dependencies([
        control_flow_ops.Assert(
            math_ops.greater(array_ops.size(string_tensor), 0),
            [not_null_err])]):
      string_tensor = array_ops.identity(string_tensor)
    return input_producer(
        input_tensor=string_tensor,
        element_shape=[],
        num_epochs=num_epochs,
        shuffle=shuffle,
        seed=seed,
        capacity=capacity,
        shared_name=shared_name,
        name=name,
        summary_name="fraction_of_%d_full" % capacity)


def range_input_producer(limit, num_epochs=None, shuffle=True, seed=None,
                         capacity=32, shared_name=None, name=None):
  """Produces the integers from 0 to limit-1 in a queue.

  Args:
    limit: An int32 scalar tensor.
    num_epochs: An integer (optional). If specified, `range_input_producer`
      produces each integer `num_epochs` times before generating an
      OutOfRange error. If not specified, `range_input_producer` can cycle
      through the integers an unlimited number of times.
    shuffle: Boolean. If true, the integers are randomly shuffled within each
      epoch.
    seed: An integer (optional). Seed used if shuffle == True.
    capacity: An integer. Sets the queue capacity.
    shared_name: (optional). If set, this queue will be shared under the given
      name across multiple sessions.
    name: A name for the operations (optional).

  Returns:
    A Queue with the output integers.  A `QueueRunner` for the Queue
    is added to the current `Graph`'s `QUEUE_RUNNER` collection.
  """
  with ops.name_scope(name, "input_producer", [limit]) as name:
    range_tensor = math_ops.range(limit)
    return input_producer(
        range_tensor, [], num_epochs, shuffle, seed, capacity,
        shared_name, name, "fraction_of_%d_full" % capacity)


def slice_input_producer(tensor_list, num_epochs=None, shuffle=True, seed=None,
                         capacity=32, shared_name=None, name=None):
  """Produces a slice of each `Tensor` in `tensor_list`.

  Implemented using a Queue -- a `QueueRunner` for the Queue
  is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  Args:
    tensor_list: A list of `Tensor` objects. Every `Tensor` in
      `tensor_list` must have the same size in the first dimension.
    num_epochs: An integer (optional). If specified, `slice_input_producer`
      produces each slice `num_epochs` times before generating
      an `OutOfRange` error. If not specified, `slice_input_producer` can cycle
      through the slices an unlimited number of times.
    shuffle: Boolean. If true, the integers are randomly shuffled within each
      epoch.
    seed: An integer (optional). Seed used if shuffle == True.
    capacity: An integer. Sets the queue capacity.
    shared_name: (optional). If set, this queue will be shared under the given
      name across multiple sessions.
    name: A name for the operations (optional).

  Returns:
    A list of tensors, one for each element of `tensor_list`.  If the tensor
    in `tensor_list` has shape `[N, a, b, .., z]`, then the corresponding output
    tensor will have shape `[a, b, ..., z]`.

  Raises:
    ValueError: if `slice_input_producer` produces nothing from `tensor_list`.
  """
  with ops.name_scope(name, "input_producer", tensor_list):
    tensor_list = ops.convert_n_to_tensor_or_indexed_slices(tensor_list)
    if not tensor_list:
      raise ValueError(
          "Expected at least one tensor in slice_input_producer().")
    range_size = array_ops.shape(tensor_list[0])[0]
    # TODO(josh11b): Add an assertion that the first dimension of
    # everything in TensorList matches. Maybe just check the inferred shapes?
    queue = range_input_producer(range_size, num_epochs=num_epochs,
                                 shuffle=shuffle, seed=seed, capacity=capacity,
                                 shared_name=shared_name)
    index = queue.dequeue()
    output = [array_ops.gather(t, index) for t in tensor_list]
    return output


# Helpers for the batching functions ------------------------------------------


def _flatten(tensor_list_list):
  return [tensor for tensor_list in tensor_list_list for tensor in tensor_list]


class _SparseMetaData(object):
  """Store information about the Tensor: Is it sparse?, dtype, and rank."""

  def __init__(self, sparse, dtype, rank):
    self._sparse = sparse
    self._dtype = dtype
    self._rank = rank

  def __eq__(self, other):
    if self.sparse != other.sparse:
      return False
    if not self.sparse:
      return True
    if self.dtype != other.dtype:
      return False
    if not self.rank.is_compatible_with(other.rank):
      return False
    return True

  def __ne__(self, other):
    return not self.__eq__(other)

  def __str__(self):
    return "[SparseMetaData(%s, %s, %s)]" % (self.sparse, self.dtype, self.rank)

  def merge_with(self, other):
    if self != other:
      raise ValueError("SparseMetaData objects are incompatible: %s vs. %s"
                       % (self, other))
    if self.sparse:
      self.rank.merge_with(other.rank)
    return self

  @property
  def dtype(self):
    return self._dtype

  @property
  def sparse(self):
    return self._sparse

  @property
  def rank(self):
    return self._rank


def _as_tensor_list(tensors):
  if isinstance(tensors, dict):
    return [tensors[k] for k in sorted(tensors)]
  else:
    return tensors


def _as_tensor_list_list(tensors_list):
  if not tensors_list:
    raise ValueError("Expected at least one set of tensors")
  if isinstance(tensors_list[0], dict):
    expected_keys = set(tensors_list[0].keys())
    for tensors in tensors_list[1:]:
      if set(tensors.keys()) != expected_keys:
        raise ValueError("All dictionaries in tensors_list must have "
                         "the same keys")
    return [_as_tensor_list(tensors) for tensors in tensors_list]
  else:
    return tensors_list


def _as_original_type(original_tensors, tensor_list):
  if isinstance(original_tensors, dict):
    if len(original_tensors) == 1:
      # tensor_list is bogusly returned as a single tensor if only one tensor
      # was enqueued.  Make it a list again.  See b/28117485.
      tensor_list = [tensor_list]
    return {k: tensor_list[i]
            for i, k in enumerate(sorted(original_tensors))}
  else:
    return tensor_list


def _serialize_sparse_tensors(tensor_list, enqueue_many):
  """Serialize SparseTensors for feeding into batch, etc."""

  def _sparse_meta_data(t):
    if not isinstance(t, ops.SparseTensor):
      return _SparseMetaData(False, None, None)
    rank = t.shape.get_shape().with_rank(1)[0]
    if enqueue_many:
      rank -= 1
    return _SparseMetaData(sparse=True, dtype=t.dtype, rank=rank)

  def _maybe_serialize(t):
    if not isinstance(t, ops.SparseTensor):
      return t
    return (sparse_ops.serialize_many_sparse(t) if enqueue_many
            else sparse_ops.serialize_sparse(t))

  serialized_list = [_maybe_serialize(t) for t in tensor_list]
  sparse_info_list = [_sparse_meta_data(t) for t in tensor_list]
  return serialized_list, sparse_info_list


def _serialize_sparse_tensors_join(tensor_list_list, enqueue_many):
  """Serialize SparseTensors for feeding into batch_join, etc."""
  (s0, sparse_info_list) = _serialize_sparse_tensors(
      tensor_list_list[0], enqueue_many)
  serialized_list_list = [s0]
  for tensor_list in tensor_list_list[1:]:
    s, sparse_info_candidate = _serialize_sparse_tensors(
        tensor_list, enqueue_many)
    if sparse_info_list != sparse_info_candidate:
      raise ValueError("Inconsistent SparseTensors list: %s vs. %s"
                       % (tensor_list_list[0], tensor_list))
    sparse_info_list = [
        info.merge_with(candidate)
        for (info, candidate) in zip(sparse_info_list, sparse_info_candidate)]
    serialized_list_list.append(s)

  return (serialized_list_list, sparse_info_list)


def _deserialize_sparse_tensors(serialized_list, sparse_info_list):
  """Deserialize SparseTensors after dequeue in batch, batch_join, etc."""
  received_sequence = isinstance(serialized_list, collections.Sequence)
  if not received_sequence:
    serialized_list = (serialized_list,)
  tensors = [
      sparse_ops.deserialize_many_sparse(s, info.dtype, (info.rank + 1).value)
      if info.sparse else s
      for (s, info)
      in zip(serialized_list, sparse_info_list)]
  return tensors if received_sequence else tensors[0]


def _validate(tensor_list):
  tensor_list = ops.convert_n_to_tensor_or_indexed_slices(tensor_list)
  if not tensor_list:
    raise ValueError("Expected at least one tensor in batch().")
  return tensor_list


def _validate_join(tensor_list_list):
  tensor_list_list = [ops.convert_n_to_tensor_or_indexed_slices(tl)
                      for tl in tensor_list_list]
  if not tensor_list_list:
    raise ValueError("Expected at least one input in batch_join().")
  return tensor_list_list


def _dtypes(tensor_list_list):
  all_types = [[t.dtype for t in tl] for tl in tensor_list_list]
  types = all_types[0]
  for other_types in all_types[1:]:
    if other_types != types:
      raise TypeError("Expected types to be consistent: %s vs. %s." %
                      (", ".join(x.name for x in types),
                       ", ".join(x.name for x in other_types)))
  return types


def _merge_shapes(shape_list, enqueue_many):
  shape_list = [tensor_shape.as_shape(s) for s in shape_list]
  if enqueue_many:
    # We want the shapes without the leading batch dimension.
    shape_list = [s.with_rank_at_least(1)[1:] for s in shape_list]
  merged_shape = shape_list[0]
  for s in shape_list[1:]:
    merged_shape.merge_with(s)
  return merged_shape.as_list()


def _shapes(tensor_list_list, shapes, enqueue_many):
  """Calculate and merge the shapes of incoming tensors.

  Args:
    tensor_list_list: List of tensor lists.
    shapes: List of shape tuples corresponding to tensors within the lists.
    enqueue_many: Boolean describing whether shapes will be enqueued as
      batches or individual entries.

  Returns:
    A list of shapes aggregating shape inference info from `tensor_list_list`,
    or returning `shapes` if it is not `None`.

  Raises:
    ValueError: If any of the inferred shapes in `tensor_list_list` lack a
      well defined rank.
  """
  if shapes is None:
    len0 = len(tensor_list_list[0])

    for tl in tensor_list_list:
      for i in xrange(len0):
        if tl[i].get_shape().ndims is None:
          raise ValueError("Cannot infer Tensor's rank: %s" % tl[i])

    shapes = [_merge_shapes(
        [tl[i].get_shape().as_list() for tl in tensor_list_list], enqueue_many)
              for i in xrange(len0)]
  return shapes


def _enqueue_join(queue, tensor_list_list, enqueue_many):
  if enqueue_many:
    enqueue_ops = [queue.enqueue_many(tl) for tl in tensor_list_list]
  else:
    enqueue_ops = [queue.enqueue(tl) for tl in tensor_list_list]
  queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))


def _enqueue(queue, tensor_list, threads, enqueue_many):
  if enqueue_many:
    enqueue_ops = [queue.enqueue_many(tensor_list)] * threads
  else:
    enqueue_ops = [queue.enqueue(tensor_list)] * threads
  queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))


def _which_queue(dynamic_pad):
  return (data_flow_ops.PaddingFIFOQueue if dynamic_pad
          else data_flow_ops.FIFOQueue)


# Batching functions ----------------------------------------------------------


def batch(tensors, batch_size, num_threads=1, capacity=32,
          enqueue_many=False, shapes=None, dynamic_pad=False,
          allow_smaller_final_batch=False, shared_name=None, name=None):
  """Creates batches of tensors in `tensors`.

  The argument `tensors` can be a list or a dictionary of tensors.
  The value returned by the function will be of the same type
  as `tensors`.

  This function is implemented using a queue. A `QueueRunner` for the
  queue is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  If `enqueue_many` is `False`, `tensors` is assumed to represent a single
  example.  An input tensor with shape `[x, y, z]` will be output as a tensor
  with shape `[batch_size, x, y, z]`.

  If `enqueue_many` is `True`, `tensors` is assumed to represent a batch of
  examples, where the first dimension is indexed by example, and all members of
  `tensors` should have the same size in the first dimension.  If an input
  tensor has shape `[*, x, y, z]`, the output will have shape `[batch_size, x,
  y, z]`.  The `capacity` argument controls the how long the prefetching is
  allowed to grow the queues.

  The returned operation is a dequeue operation and will throw
  `tf.errors.OutOfRangeError` if the input queue is exhausted. If this
  operation is feeding another input queue, its queue runner will catch
  this exception, however, if this operation is used in your main thread
  you are responsible for catching this yourself.

  *N.B.:* If `dynamic_pad` is `False`, you must ensure that either
  (i) the `shapes` argument is passed, or (ii) all of the tensors in
  `tensors` must have fully-defined shapes. `ValueError` will be
  raised if neither of these conditions holds.

  If `dynamic_pad` is `True`, it is sufficient that the *rank* of the
  tensors is known, but individual dimensions may have shape `None`.
  In this case, for each enqueue the dimensions with value `None`
  may have a variable length; upon dequeue, the output tensors will be padded
  on the right to the maximum shape of the tensors in the current minibatch.
  For numbers, this padding takes value 0.  For strings, this padding is
  the empty string.  See `PaddingFIFOQueue` for more info.

  If `allow_smaller_final_batch` is `True`, a smaller batch value than
  `batch_size` is returned when the queue is closed and there are not enough
  elements to fill the batch, otherwise the pending elements are discarded.
  In addition, all output tensors' static shapes, as accessed via the
  `get_shape` method will have a first `Dimension` value of `None`, and
  operations that depend on fixed batch_size would fail.

  Args:
    tensors: The list or dictionary of tensors to enqueue.
    batch_size: The new batch size pulled from the queue.
    num_threads: The number of threads enqueuing `tensors`.
    capacity: An integer. The maximum number of elements in the queue.
    enqueue_many: Whether each tensor in `tensors` is a single example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensors`.
    dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
      The given dimensions are padded upon dequeue so that tensors within a
      batch have the same shapes.
    allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
      batch to be smaller if there are insufficient items left in the queue.
    shared_name: (Optional). If set, this queue will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A list or dictionary of tensors with the same types as `tensors`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensors`.
  """
  tensor_list = _as_tensor_list(tensors)
  with ops.name_scope(name, "batch", tensor_list) as name:
    tensor_list = _validate(tensor_list)
    (tensor_list, sparse_info) = _serialize_sparse_tensors(
        tensor_list, enqueue_many)
    types = _dtypes([tensor_list])
    shapes = _shapes([tensor_list], shapes, enqueue_many)
    # TODO(josh11b,mrry): Switch to BatchQueue once it is written.
    queue = _which_queue(dynamic_pad)(
        capacity=capacity, dtypes=types, shapes=shapes, shared_name=shared_name)
    _enqueue(queue, tensor_list, num_threads, enqueue_many)
    logging_ops.scalar_summary(
        "queue/%s/fraction_of_%d_full" % (queue.name, capacity),
        math_ops.cast(queue.size(), dtypes.float32) * (1. / capacity))

    if allow_smaller_final_batch:
      dequeued = queue.dequeue_up_to(batch_size, name=name)
    else:
      dequeued = queue.dequeue_many(batch_size, name=name)
    dequeued = _deserialize_sparse_tensors(dequeued, sparse_info)
    return _as_original_type(tensors, dequeued)


# TODO(josh11b): Add a thread_multiplier or num_threads (that has to be
# a multiple of len(tensor_list_list)?) parameter, to address the use
# case where you want more parallelism than you can support different
# readers (either because you don't have that many files or can't
# read that many files in parallel due to the number of seeks required).
# Once this is done, batch() can be written as a call to batch_join().
def batch_join(tensors_list, batch_size, capacity=32, enqueue_many=False,
               shapes=None, dynamic_pad=False, allow_smaller_final_batch=False,
               shared_name=None, name=None):
  """Runs a list of tensors to fill a queue to create batches of examples.

  The `tensors_list` argument is a list of tuples of tensors, or a list of
  dictionaries of tensors.  Each element in the list is treated similarly
  to the `tensors` argument of `tf.train.batch()`.

  Enqueues a different list of tensors in different threads.
  Implemented using a queue -- a `QueueRunner` for the queue
  is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  `len(tensors_list)` threads will be started,
  with thread `i` enqueuing the tensors from
  `tensors_list[i]`. `tensors_list[i1][j]` must match
  `tensors_list[i2][j]` in type and shape, except in the first
  dimension if `enqueue_many` is true.

  If `enqueue_many` is `False`, each `tensors_list[i]` is assumed
  to represent a single example. An input tensor `x` will be output as a
  tensor with shape `[batch_size] + x.shape`.

  If `enqueue_many` is `True`, `tensors_list[i]` is assumed to
  represent a batch of examples, where the first dimension is indexed
  by example, and all members of `tensors_list[i]` should have the
  same size in the first dimension.  The slices of any input tensor
  `x` are treated as examples, and the output tensors will have shape
  `[batch_size] + x.shape[1:]`.

  The `capacity` argument controls the how long the prefetching is allowed to
  grow the queues.

  The returned operation is a dequeue operation and will throw
  `tf.errors.OutOfRangeError` if the input queue is exhausted. If this
  operation is feeding another input queue, its queue runner will catch
  this exception, however, if this operation is used in your main thread
  you are responsible for catching this yourself.

  *N.B.:* If `dynamic_pad` is `False`, you must ensure that either
  (i) the `shapes` argument is passed, or (ii) all of the tensors in
  `tensors_list` must have fully-defined shapes. `ValueError` will be
  raised if neither of these conditions holds.

  If `dynamic_pad` is `True`, it is sufficient that the *rank* of the
  tensors is known, but individual dimensions may have value `None`.
  In this case, for each enqueue the dimensions with value `None`
  may have a variable length; upon dequeue, the output tensors will be padded
  on the right to the maximum shape of the tensors in the current minibatch.
  For numbers, this padding takes value 0.  For strings, this padding is
  the empty string.  See `PaddingFIFOQueue` for more info.

  If `allow_smaller_final_batch` is `True`, a smaller batch value than
  `batch_size` is returned when the queue is closed and there are not enough
  elements to fill the batch, otherwise the pending elements are discarded.
  In addition, all output tensors' static shapes, as accessed via the
  `get_shape` method will have a first `Dimension` value of `None`, and
  operations that depend on fixed batch_size would fail.

  Args:
    tensors_list: A list of tuples or dictionaries of tensors to enqueue.
    batch_size: An integer. The new batch size pulled from the queue.
    capacity: An integer. The maximum number of elements in the queue.
    enqueue_many: Whether each tensor in `tensor_list_list` is a single
      example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensor_list_list[i]`.
    dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
      The given dimensions are padded upon dequeue so that tensors within a
      batch have the same shapes.
    allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
      batch to be smaller if there are insufficient items left in the queue.
    shared_name: (Optional) If set, this queue will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A list or dictionary of tensors with the same number and types as
    `tensors_list[i]`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensor_list_list`.
  """
  tensor_list_list = _as_tensor_list_list(tensors_list)
  with ops.name_scope(name, "batch_join", _flatten(tensor_list_list)) as name:
    tensor_list_list = _validate_join(tensor_list_list)
    tensor_list_list, sparse_info = _serialize_sparse_tensors_join(
        tensor_list_list, enqueue_many)
    types = _dtypes(tensor_list_list)
    shapes = _shapes(tensor_list_list, shapes, enqueue_many)
    # TODO(josh11b,mrry): Switch to BatchQueue once it is written.
    queue = _which_queue(dynamic_pad)(
        capacity=capacity, dtypes=types, shapes=shapes, shared_name=shared_name)
    _enqueue_join(queue, tensor_list_list, enqueue_many)
    logging_ops.scalar_summary(
        "queue/%s/fraction_of_%d_full" % (queue.name, capacity),
        math_ops.cast(queue.size(), dtypes.float32) * (1. / capacity))

    if allow_smaller_final_batch:
      dequeued = queue.dequeue_up_to(batch_size, name=name)
    else:
      dequeued = queue.dequeue_many(batch_size, name=name)
    dequeued = _deserialize_sparse_tensors(dequeued, sparse_info)
    # tensors_list was validated to not be empty.
    return _as_original_type(tensors_list[0], dequeued)


def shuffle_batch(tensors, batch_size, capacity, min_after_dequeue,
                  num_threads=1, seed=None, enqueue_many=False, shapes=None,
                  allow_smaller_final_batch=False, shared_name=None, name=None):
  """Creates batches by randomly shuffling tensors.

  This function adds the following to the current `Graph`:

  * A shuffling queue into which tensors from `tensors` are enqueued.
  * A `dequeue_many` operation to create batches from the queue.
  * A `QueueRunner` to `QUEUE_RUNNER` collection, to enqueue the tensors
    from `tensors`.

  If `enqueue_many` is `False`, `tensors` is assumed to represent a
  single example.  An input tensor with shape `[x, y, z]` will be output
  as a tensor with shape `[batch_size, x, y, z]`.

  If `enqueue_many` is `True`, `tensors` is assumed to represent a
  batch of examples, where the first dimension is indexed by example,
  and all members of `tensors` should have the same size in the
  first dimension.  If an input tensor has shape `[*, x, y, z]`, the
  output will have shape `[batch_size, x, y, z]`.

  The `capacity` argument controls the how long the prefetching is allowed to
  grow the queues.

  The returned operation is a dequeue operation and will throw
  `tf.errors.OutOfRangeError` if the input queue is exhausted. If this
  operation is feeding another input queue, its queue runner will catch
  this exception, however, if this operation is used in your main thread
  you are responsible for catching this yourself.

  For example:

  ```python
  # Creates batches of 32 images and 32 labels.
  image_batch, label_batch = tf.train.shuffle_batch(
        [single_image, single_label],
        batch_size=32,
        num_threads=4,
        capacity=50000,
        min_after_dequeue=10000)
  ```

  *N.B.:* You must ensure that either (i) the `shapes` argument is
  passed, or (ii) all of the tensors in `tensors` must have
  fully-defined shapes. `ValueError` will be raised if neither of
  these conditions holds.

  If `allow_smaller_final_batch` is `True`, a smaller batch value than
  `batch_size` is returned when the queue is closed and there are not enough
  elements to fill the batch, otherwise the pending elements are discarded.
  In addition, all output tensors' static shapes, as accessed via the
  `get_shape` method will have a first `Dimension` value of `None`, and
  operations that depend on fixed batch_size would fail.

  Args:
    tensors: The list or dictionary of tensors to enqueue.
    batch_size: The new batch size pulled from the queue.
    capacity: An integer. The maximum number of elements in the queue.
    min_after_dequeue: Minimum number elements in the queue after a
      dequeue, used to ensure a level of mixing of elements.
    num_threads: The number of threads enqueuing `tensor_list`.
    seed: Seed for the random shuffling within the queue.
    enqueue_many: Whether each tensor in `tensor_list` is a single example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensor_list`.
    allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
      batch to be smaller if there are insufficient items left in the queue.
    shared_name: (Optional) If set, this queue will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A list or dictionary of tensors with the types as `tensors`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensors`.
  """
  tensor_list = _as_tensor_list(tensors)
  with ops.name_scope(name, "shuffle_batch", tensor_list) as name:
    tensor_list = _validate(tensor_list)
    tensor_list, sparse_info = _serialize_sparse_tensors(
        tensor_list, enqueue_many)
    types = _dtypes([tensor_list])
    shapes = _shapes([tensor_list], shapes, enqueue_many)
    queue = data_flow_ops.RandomShuffleQueue(
        capacity=capacity, min_after_dequeue=min_after_dequeue, seed=seed,
        dtypes=types, shapes=shapes, shared_name=shared_name)
    _enqueue(queue, tensor_list, num_threads, enqueue_many)
    full = (math_ops.cast(math_ops.maximum(0, queue.size() - min_after_dequeue),
                          dtypes.float32) *
            (1. / (capacity - min_after_dequeue)))
    # Note that name contains a '/' at the end so we intentionally do not place
    # a '/' after %s below.
    summary_name = (
        "queue/%sfraction_over_%d_of_%d_full" %
        (name, min_after_dequeue, capacity - min_after_dequeue))
    logging_ops.scalar_summary(summary_name, full)

    if allow_smaller_final_batch:
      dequeued = queue.dequeue_up_to(batch_size, name=name)
    else:
      dequeued = queue.dequeue_many(batch_size, name=name)
    dequeued = _deserialize_sparse_tensors(dequeued, sparse_info)
    return _as_original_type(tensors, dequeued)


def shuffle_batch_join(tensors_list, batch_size, capacity,
                       min_after_dequeue, seed=None, enqueue_many=False,
                       shapes=None, allow_smaller_final_batch=False,
                       shared_name=None, name=None):
  """Create batches by randomly shuffling tensors.

  The `tensors_list` argument is a list of tuples of tensors, or a list of
  dictionaries of tensors.  Each element in the list is treated similarly
  to the `tensors` argument of `tf.train.shuffle_batch()`.

  This version enqueues a different list of tensors in different threads.
  It adds the following to the current `Graph`:

  * A shuffling queue into which tensors from `tensors_list` are enqueued.
  * A `dequeue_many` operation to create batches from the queue.
  * A `QueueRunner` to `QUEUE_RUNNER` collection, to enqueue the tensors
    from `tensors_list`.

  `len(tensors_list)` threads will be started, with thread `i` enqueuing
  the tensors from `tensors_list[i]`. `tensors_list[i1][j]` must match
  `tensors_list[i2][j]` in type and shape, except in the first dimension if
  `enqueue_many` is true.

  If `enqueue_many` is `False`, each `tensors_list[i]` is assumed
  to represent a single example.  An input tensor with shape `[x, y, z]`
  will be output as a tensor with shape `[batch_size, x, y, z]`.

  If `enqueue_many` is `True`, `tensors_list[i]` is assumed to
  represent a batch of examples, where the first dimension is indexed
  by example, and all members of `tensors_list[i]` should have the
  same size in the first dimension.  If an input tensor has shape `[*, x,
  y, z]`, the output will have shape `[batch_size, x, y, z]`.

  The `capacity` argument controls the how long the prefetching is allowed to
  grow the queues.

  The returned operation is a dequeue operation and will throw
  `tf.errors.OutOfRangeError` if the input queue is exhausted. If this
  operation is feeding another input queue, its queue runner will catch
  this exception, however, if this operation is used in your main thread
  you are responsible for catching this yourself.

  If `allow_smaller_final_batch` is `True`, a smaller batch value than
  `batch_size` is returned when the queue is closed and there are not enough
  elements to fill the batch, otherwise the pending elements are discarded.
  In addition, all output tensors' static shapes, as accessed via the
  `get_shape` method will have a first `Dimension` value of `None`, and
  operations that depend on fixed batch_size would fail.

  Args:
    tensors_list: A list of tuples or dictionaries of tensors to enqueue.
    batch_size: An integer. The new batch size pulled from the queue.
    capacity: An integer. The maximum number of elements in the queue.
    min_after_dequeue: Minimum number elements in the queue after a
      dequeue, used to ensure a level of mixing of elements.
    seed: Seed for the random shuffling within the queue.
    enqueue_many: Whether each tensor in `tensor_list_list` is a single
      example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensors_list[i]`.
    allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
      batch to be smaller if there are insufficient items left in the queue.
    shared_name: (optional). If set, this queue will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A list or dictionary of tensors with the same number and types as
    `tensors_list[i]`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensors_list`.
  """
  tensor_list_list = _as_tensor_list_list(tensors_list)
  with ops.name_scope(name, "shuffle_batch_join",
                      _flatten(tensor_list_list)) as name:
    tensor_list_list = _validate_join(tensor_list_list)
    tensor_list_list, sparse_info = _serialize_sparse_tensors_join(
        tensor_list_list, enqueue_many)
    types = _dtypes(tensor_list_list)
    shapes = _shapes(tensor_list_list, shapes, enqueue_many)
    queue = data_flow_ops.RandomShuffleQueue(
        capacity=capacity, min_after_dequeue=min_after_dequeue, seed=seed,
        dtypes=types, shapes=shapes, shared_name=shared_name)
    _enqueue_join(queue, tensor_list_list, enqueue_many)
    full = (math_ops.cast(math_ops.maximum(0, queue.size() - min_after_dequeue),
                          dtypes.float32) *
            (1. / (capacity - min_after_dequeue)))
    # Note that name contains a '/' at the end so we intentionally do not place
    # a '/' after %s below.
    summary_name = (
        "queue/%sfraction_over_%d_of_%d_full" %
        (name, min_after_dequeue, capacity - min_after_dequeue))
    logging_ops.scalar_summary(summary_name, full)

    if allow_smaller_final_batch:
      dequeued = queue.dequeue_up_to(batch_size, name=name)
    else:
      dequeued = queue.dequeue_many(batch_size, name=name)
    dequeued = _deserialize_sparse_tensors(dequeued, sparse_info)
    # tensors_list was validated to not be empty.
    return _as_original_type(tensors_list[0], dequeued)

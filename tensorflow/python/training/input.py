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

Please see the [reading data
how-to](https://tensorflow.org/api_guides/python/reading_data)
for context.
"""


from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.summary import summary
from tensorflow.python.training import queue_runner
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export


# pylint: disable=protected-access
_store_sparse = sparse_ops._add_sparse_to_tensors_map
_store_many_sparse = sparse_ops._add_many_sparse_to_tensors_map
_restore_sparse = sparse_ops._take_many_sparse_from_tensors_map
# pylint: enable=protected-access


@tf_export(
    "io.match_filenames_once",
    v1=["io.match_filenames_once", "train.match_filenames_once"])
@deprecation.deprecated_endpoints("train.match_filenames_once")
def match_filenames_once(pattern, name=None):
  """Save the list of files matching pattern, so it is only computed once.

  NOTE: The order of the files returned is deterministic.

  Args:
    pattern: A file pattern (glob), or 1D tensor of file patterns.
    name: A name for the operations (optional).

  Returns:
    A variable that is initialized to the list of files matching the pattern(s).
  """
  with ops.name_scope(name, "matching_filenames", [pattern]) as name:
    return vs.variable(
        name=name, initial_value=io_ops.matching_files(pattern),
        trainable=False, validate_shape=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES])


@tf_export(v1=["train.limit_epochs"])
@deprecation.deprecated(
    None, "Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.from_tensors(tensor).repeat(num_epochs)`.")
def limit_epochs(tensor, num_epochs=None, name=None):
  """Returns tensor `num_epochs` times and then raises an `OutOfRange` error.

  Note: creates local counter `epochs`. Use `local_variables_initializer()` to
  initialize local variables.

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
    epochs = vs.variable(
        zero64, name="epochs", trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES])
    counter = epochs.count_up_to(num_epochs)
    with ops.control_dependencies([counter]):
      return array_ops.identity(tensor, name=name)


@tf_export(v1=["train.input_producer"])
@deprecation.deprecated(
    None, "Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.from_tensor_slices(input_tensor).shuffle"
    "(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)`. If "
    "`shuffle=False`, omit the `.shuffle(...)`.")
def input_producer(input_tensor,
                   element_shape=None,
                   num_epochs=None,
                   shuffle=True,
                   seed=None,
                   capacity=32,
                   shared_name=None,
                   summary_name=None,
                   name=None,
                   cancel_op=None):
  """Output the rows of `input_tensor` to a queue for an input pipeline.

  Note: if `num_epochs` is not `None`, this function creates local counter
  `epochs`. Use `local_variables_initializer()` to initialize local variables.

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
    cancel_op: (Optional.) Cancel op for the queue

  Returns:
    A queue with the output rows.  A `QueueRunner` for the queue is
    added to the current `QUEUE_RUNNER` collection of the current
    graph.

  Raises:
    ValueError: If the shape of the input cannot be inferred from the arguments.
    RuntimeError: If called with eager execution enabled.

  @compatibility(eager)
  Input pipelines based on Queues are not supported when eager execution is
  enabled. Please use the `tf.data` API to ingest data under eager execution.
  @end_compatibility
  """
  if context.executing_eagerly():
    raise RuntimeError(
        "Input pipelines based on Queues are not supported when eager execution"
        " is enabled. Please use tf.data to ingest data into your model"
        " instead.")
  with ops.name_scope(name, "input_producer", [input_tensor]):
    input_tensor = ops.convert_to_tensor(input_tensor, name="input_tensor")
    element_shape = input_tensor.shape[1:].merge_with(element_shape)
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
    queue_runner.add_queue_runner(
        queue_runner.QueueRunner(
            q, [enq], cancel_op=cancel_op))
    if summary_name is not None:
      summary.scalar(summary_name,
                     math_ops.cast(q.size(), dtypes.float32) * (1. / capacity))
    return q


@tf_export(v1=["train.string_input_producer"])
@deprecation.deprecated(
    None, "Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.from_tensor_slices(string_tensor).shuffle"
    "(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)`. If "
    "`shuffle=False`, omit the `.shuffle(...)`.")
def string_input_producer(string_tensor,
                          num_epochs=None,
                          shuffle=True,
                          seed=None,
                          capacity=32,
                          shared_name=None,
                          name=None,
                          cancel_op=None):
  """Output strings (e.g. filenames) to a queue for an input pipeline.

  Note: if `num_epochs` is not `None`, this function creates local counter
  `epochs`. Use `local_variables_initializer()` to initialize local variables.

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
      name across multiple sessions. All sessions open to the device which has
      this queue will be able to access it via the shared_name. Using this in
      a distributed setting means each name will only be seen by one of the
      sessions which has access to this operation.
    name: A name for the operations (optional).
    cancel_op: Cancel op for the queue (optional).

  Returns:
    A queue with the output strings.  A `QueueRunner` for the Queue
    is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  Raises:
    ValueError: If the string_tensor is a null Python list.  At runtime,
    will fail with an assertion if string_tensor becomes a null tensor.

  @compatibility(eager)
  Input pipelines based on Queues are not supported when eager execution is
  enabled. Please use the `tf.data` API to ingest data under eager execution.
  @end_compatibility
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
        summary_name="fraction_of_%d_full" % capacity,
        cancel_op=cancel_op)


@tf_export(v1=["train.range_input_producer"])
@deprecation.deprecated(
    None, "Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.range(limit).shuffle(limit).repeat(num_epochs)`. If "
    "`shuffle=False`, omit the `.shuffle(...)`.")
def range_input_producer(limit, num_epochs=None, shuffle=True, seed=None,
                         capacity=32, shared_name=None, name=None):
  """Produces the integers from 0 to limit-1 in a queue.

  Note: if `num_epochs` is not `None`, this function creates local counter
  `epochs`. Use `local_variables_initializer()` to initialize local variables.

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

  @compatibility(eager)
  Input pipelines based on Queues are not supported when eager execution is
  enabled. Please use the `tf.data` API to ingest data under eager execution.
  @end_compatibility
  """
  with ops.name_scope(name, "input_producer", [limit]) as name:
    range_tensor = math_ops.range(limit)
    return input_producer(
        range_tensor, [], num_epochs, shuffle, seed, capacity,
        shared_name, "fraction_of_%d_full" % capacity, name)


@tf_export(v1=["train.slice_input_producer"])
@deprecation.deprecated(
    None, "Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.from_tensor_slices(tuple(tensor_list)).shuffle"
    "(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)`. If "
    "`shuffle=False`, omit the `.shuffle(...)`.")
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

  @compatibility(eager)
  Input pipelines based on Queues are not supported when eager execution is
  enabled. Please use the `tf.data` API to ingest data under eager execution.
  @end_compatibility
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
  """Store information about the Tensor: Is it sparse?, map_op, and rank."""

  def __init__(self, sparse, map_op, rank):
    """Create the metadata.

    Args:
      sparse: Python boolean.
      map_op: The `Operation` that created the `SparseTensorsMap` in question.
        This Op contains information about the underlying Map object and the
        dtype of the original data.
      rank: The statically known rank of the `SparseTensor`.
    """
    self._sparse = sparse
    self._map_op = map_op
    self._rank = tensor_shape.as_dimension(rank)

  def __eq__(self, other):
    if self.sparse != other.sparse:
      return False
    if not self.sparse:
      return True
    # If map_ops are not the same, the data source is not the same.
    if (self.map_op is not None) != (other.map_op is not None):
      return False
    if self.map_op != other.map_op:
      return False
    if not self.rank.is_compatible_with(other.rank):
      return False
    return True

  def __ne__(self, other):
    return not self.__eq__(other)

  def __str__(self):
    return "[SparseMetaData(%s, %s, %s)]" % (self.sparse, self.map_op.name,
                                             self.rank)

  def merge_with(self, other):
    if self != other:
      raise ValueError("SparseMetaData objects are incompatible: %s vs. %s"
                       % (self, other))
    if self.sparse:
      self.rank.merge_with(other.rank)
    return self

  @property
  def map_op(self):
    return self._map_op

  @property
  def sparse(self):
    return self._sparse

  @property
  def rank(self):
    return self._rank


def _as_tensor_list(tensors):
  if isinstance(tensors, dict):
    return [tensors[k] for k in sorted(tensors, key=str)]
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
            for i, k in enumerate(sorted(original_tensors, key=str))}
  else:
    return tensor_list


def _store_sparse_tensors(tensor_list, enqueue_many, keep_input,
                          shared_map_ops=None):
  """Store SparseTensors for feeding into batch, etc.

  If `shared_map_ops` is provided, the underlying `SparseTensorsMap` objects
  are reused (shared).  This argument is useful for, e.g., `batch_join`
  where multiple enqueue operations write to the same Queue component,
  and another (dequeue) thread reads from that same location and must then
  restore the associated `SparseTensor` objects.  In this case, the sparse
  restore must have a single `SparseTensorMap` from which to read out the
  handles; so a single `SparseTensorMap` must be shared for storing
  across the multiple enqueue operations.  This sharing is performed by
  calling `_store_sparse_tensors` the first time with `shared_map_ops=None`,
  and then in subsequent times with this value set to the list of `Operation`
  objects created in the first call.

  Args:
    tensor_list: List of `Tensor` and `SparseTensor` objects.
    enqueue_many: Python `Boolean`.
    keep_input: Must be a scalar bool Tensor (not a Python bool). If False,
      don't store.
    shared_map_ops: (optional) List of `Operation` objects from a previous
      call to `_store_sparse_tensors`.  If not `None`, the op types should be
      one of `AddSparseToTensorsMap` or `AddManySparseToTensorsMap` in the
      locations corresponding to `SparseTensors` in `tensor_list`.

  Returns:
    A tuple `(stored_list, sparse_info_list)` where `stored_list` is a list
    of `Tensor` objects (same length as `tensor_list`) and `sparse_info_list`
    is a list of the same length of `_SparseMetaData` objects.
  """
  maybe_shared_map_ops = shared_map_ops or [None] * len(tensor_list)

  def _sparse_meta_data(t, storing_op, map_op):
    if not isinstance(t, sparse_tensor.SparseTensor):
      return _SparseMetaData(False, None, None)
    rank = t.dense_shape.shape.with_rank(1).dims[0]
    if enqueue_many:
      rank -= 1
    # If a shared map_op was provided, use that. Otherwise use the name of
    # the operation used to store the SparseTensor.
    return _SparseMetaData(
        sparse=True, map_op=map_op or storing_op, rank=rank)

  def _maybe_store(t, shared_map_op):
    """Store Sparse tensor, if necessary."""
    if not isinstance(t, sparse_tensor.SparseTensor):
      return t
    map_op_name = shared_map_op.name if shared_map_op else None
    def _maybe_store_sparse(t, map_op_name, keep_input):
      """Conditionally store a single sparse Tensor."""
      return utils.smart_cond(
          keep_input,
          lambda: _store_sparse(t, shared_name=map_op_name),
          lambda: constant_op.constant(-1, dtypes.int64))
    def _maybe_store_many_sparse(t, map_op_name, keep_input):
      """Conditionally store multiple sparse Tensors."""
      out_tensor = utils.smart_cond(
          keep_input,
          lambda: _store_many_sparse(t, shared_name=map_op_name),
          lambda: -1 * array_ops.ones(array_ops.shape(t)[0:1], dtypes.int64))
      out_tensor.set_shape([None])  # necessary when t.ndims is unknown
      return out_tensor
    def _sparse_values_to_keep(t, keep_input):
      """Convert a per-row `keep_input` vector to a per-value one."""
      # Get the rows of every value in the sparse Tensor.
      row_values = t.indices[:, 0]
      # The value should be kept iff the row should be kept.
      return array_ops.gather(keep_input, row_values)
    if keep_input.shape.ndims == 1:
      t = sparse_ops.sparse_retain(t, _sparse_values_to_keep(t, keep_input))
      store_f = lambda t, name, _: _store_many_sparse(t, shared_name=name)
    elif enqueue_many:
      store_f = _maybe_store_many_sparse
    else:
      store_f = _maybe_store_sparse
    return store_f(t, map_op_name, keep_input)

  stored_list = [
      _maybe_store(t, shared_map_op) for t, shared_map_op
      in zip(tensor_list, maybe_shared_map_ops)]
  # Since the output of `_store{_many}_sparse is wrapped in a tf.cond `Merge`,
  # we can't just get the Op of the resulting tensor.
  def _sparse_op(stored):
    for input_tensor in stored.op.inputs:
      if input_tensor.op.type in ("AddSparseToTensorsMap",
                                  "AddManySparseToTensorsMap"):
        return input_tensor.op
    # If there was no sparse input, then the original stored Tensor wasn't
    # sparse and we can just return the original Tensor's Op.
    return stored.op
  sparse_info_list = [
      _sparse_meta_data(t, _sparse_op(stored), shared_map_op)
      for t, stored, shared_map_op
      in zip(tensor_list, stored_list, maybe_shared_map_ops)]
  # Expand dims of stored tensors by 1 for proper enqueue shape
  stored_list = [
      array_ops.expand_dims(s, [-1]) if s_info.sparse else s
      for s, s_info in zip(stored_list, sparse_info_list)]
  return stored_list, sparse_info_list


def _store_sparse_tensors_join(tensor_list_list, enqueue_many, keep_input):
  """Store SparseTensors for feeding into batch_join, etc."""
  (s0, sparse_info_list) = _store_sparse_tensors(
      tensor_list_list[0], enqueue_many, keep_input)
  stored_list_list = [s0]
  for tensor_list in tensor_list_list[1:]:
    s, sparse_info_candidate = _store_sparse_tensors(
        tensor_list, enqueue_many, keep_input,
        [st.map_op for st in sparse_info_list])
    if sparse_info_list != sparse_info_candidate:
      raise ValueError("Inconsistent SparseTensors list: %s vs. %s"
                       % (tensor_list_list[0], tensor_list))
    sparse_info_list = [
        info.merge_with(candidate)
        for (info, candidate) in zip(sparse_info_list, sparse_info_candidate)]
    stored_list_list.append(s)

  return (stored_list_list, sparse_info_list)


def _restore_sparse_tensors(stored_list, sparse_info_list):
  """Restore SparseTensors after dequeue in batch, batch_join, etc."""
  received_sequence = isinstance(stored_list, collections_abc.Sequence)
  if not received_sequence:
    stored_list = (stored_list,)
  tensors = [
      _restore_sparse(sparse_map_op=info.map_op,
                      sparse_handles=array_ops.squeeze(s, [1]),
                      rank=tensor_shape.dimension_value(info.rank + 1))
      if info.sparse else s
      for (s, info) in zip(stored_list, sparse_info_list)]
  has_st = any(isinstance(x, sparse_tensor.SparseTensor) for x in tensors)
  if has_st:
    t_values = [
        x.values if isinstance(x, sparse_tensor.SparseTensor)
        else x
        for x in tensors]
    with_deps = lambda x: control_flow_ops.with_dependencies(t_values, x)
    ensure_restore_tensors = [
        sparse_tensor.SparseTensor(indices=with_deps(x.indices),
                                   values=with_deps(x.values),
                                   dense_shape=with_deps(x.dense_shape))
        if isinstance(x, sparse_tensor.SparseTensor)
        else with_deps(x)
        for x in tensors]
  else:
    ensure_restore_tensors = tensors
  return ensure_restore_tensors if received_sequence else tensors[0]


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


def _validate_keep_input(keep_input, enqueue_many):
  """Validate `keep_input` argument to conditional batching functions."""
  keep_input = ops.convert_to_tensor(keep_input)
  if keep_input.shape.ndims is None:
    raise ValueError(
        "`keep_input` dimensions must be known at graph construction.")
  if not enqueue_many and keep_input.shape.ndims == 1:
    raise ValueError(
        "`keep_input` cannot be a vector when `enqueue_many=False`.")
  if keep_input.shape.ndims > 1:
    raise ValueError("`keep_input` must be 0 or 1 dimensions.")
  return keep_input


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
      for i in range(len0):
        if tl[i].shape.ndims is None:
          raise ValueError("Cannot infer Tensor's rank: %s" % tl[i])

    shapes = [
        _merge_shapes([tl[i].shape.as_list()
                       for tl in tensor_list_list], enqueue_many)
        for i in range(len0)
    ]
  return shapes


def _select_which_to_enqueue(tensor_list, keep_input):
  """Select which examples to enqueue based on vector `keep_input`."""
  select_i = math_ops.cast(keep_input, dtypes.int32)
  tensor_list = [
      data_flow_ops.dynamic_partition(x, select_i, num_partitions=2)[1]
      for x in tensor_list]
  return tensor_list


def _enqueue_join(queue, tensor_list_list, enqueue_many, keep_input):
  """Enqueue `tensor_list_list` in `queue`."""
  if enqueue_many:
    enqueue_fn = queue.enqueue_many
  else:
    enqueue_fn = queue.enqueue
  if keep_input.shape.ndims == 1:
    enqueue_ops = [enqueue_fn(_select_which_to_enqueue(x, keep_input))
                   for x in tensor_list_list]
  else:
    enqueue_ops = [utils.smart_cond(
        keep_input,
        lambda: enqueue_fn(tl),  # pylint:disable=cell-var-from-loop
        control_flow_ops.no_op) for tl in tensor_list_list]
  queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))


def _enqueue(queue, tensor_list, threads, enqueue_many, keep_input):
  """Enqueue `tensor_list` in `queue`."""
  if enqueue_many:
    enqueue_fn = queue.enqueue_many
  else:
    enqueue_fn = queue.enqueue
  if keep_input.shape.ndims == 1:
    enqueue_ops = [
        enqueue_fn(_select_which_to_enqueue(tensor_list, keep_input))] * threads
  else:
    enqueue_ops = [utils.smart_cond(
        keep_input,
        lambda: enqueue_fn(tensor_list),
        control_flow_ops.no_op)] * threads
  queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))


def _which_queue(dynamic_pad):
  return (data_flow_ops.PaddingFIFOQueue if dynamic_pad
          else data_flow_ops.FIFOQueue)


def _batch(tensors, batch_size, keep_input, num_threads=1, capacity=32,
           enqueue_many=False, shapes=None, dynamic_pad=False,
           allow_smaller_final_batch=False, shared_name=None,
           name=None):
  """Helper function for `batch` and `maybe_batch`."""
  if context.executing_eagerly():
    raise ValueError(
        "Input pipelines based on Queues are not supported when eager execution"
        " is enabled. Please use tf.data to ingest data into your model"
        " instead.")
  tensor_list = _as_tensor_list(tensors)
  with ops.name_scope(name, "batch", list(tensor_list) + [keep_input]) as name:
    tensor_list = _validate(tensor_list)
    keep_input = _validate_keep_input(keep_input, enqueue_many)
    (tensor_list, sparse_info) = _store_sparse_tensors(
        tensor_list, enqueue_many, keep_input)
    types = _dtypes([tensor_list])
    shapes = _shapes([tensor_list], shapes, enqueue_many)
    # TODO(josh11b,mrry): Switch to BatchQueue once it is written.
    queue = _which_queue(dynamic_pad)(
        capacity=capacity, dtypes=types, shapes=shapes, shared_name=shared_name)
    _enqueue(queue, tensor_list, num_threads, enqueue_many, keep_input)
    summary.scalar(
        "fraction_of_%d_full" % capacity,
        math_ops.cast(queue.size(), dtypes.float32) * (1. / capacity))

    if allow_smaller_final_batch:
      dequeued = queue.dequeue_up_to(batch_size, name=name)
    else:
      dequeued = queue.dequeue_many(batch_size, name=name)
    dequeued = _restore_sparse_tensors(dequeued, sparse_info)
    return _as_original_type(tensors, dequeued)


# TODO(josh11b): Add a thread_multiplier or num_threads (that has to be
# a multiple of len(tensor_list_list)?) parameter, to address the use
# case where you want more parallelism than you can support different
# readers (either because you don't have that many files or can't
# read that many files in parallel due to the number of seeks required).
# Once this is done, batch() can be written as a call to batch_join().
def _batch_join(tensors_list, batch_size, keep_input, capacity=32,
                enqueue_many=False, shapes=None, dynamic_pad=False,
                allow_smaller_final_batch=False, shared_name=None, name=None):
  """Helper function for `batch_join` and `maybe_batch_join`."""
  if context.executing_eagerly():
    raise ValueError(
        "Input pipelines based on Queues are not supported when eager execution"
        " is enabled. Please use tf.data to ingest data into your model"
        " instead.")
  tensor_list_list = _as_tensor_list_list(tensors_list)
  with ops.name_scope(name, "batch_join",
                      _flatten(tensor_list_list) + [keep_input]) as name:
    tensor_list_list = _validate_join(tensor_list_list)
    keep_input = _validate_keep_input(keep_input, enqueue_many)
    tensor_list_list, sparse_info = _store_sparse_tensors_join(
        tensor_list_list, enqueue_many, keep_input)
    types = _dtypes(tensor_list_list)
    shapes = _shapes(tensor_list_list, shapes, enqueue_many)
    # TODO(josh11b,mrry): Switch to BatchQueue once it is written.
    queue = _which_queue(dynamic_pad)(
        capacity=capacity, dtypes=types, shapes=shapes, shared_name=shared_name)
    _enqueue_join(queue, tensor_list_list, enqueue_many, keep_input)
    summary.scalar(
        "fraction_of_%d_full" % capacity,
        math_ops.cast(queue.size(), dtypes.float32) * (1. / capacity))

    if allow_smaller_final_batch:
      dequeued = queue.dequeue_up_to(batch_size, name=name)
    else:
      dequeued = queue.dequeue_many(batch_size, name=name)
    dequeued = _restore_sparse_tensors(dequeued, sparse_info)
    # tensors_list was validated to not be empty.
    return _as_original_type(tensors_list[0], dequeued)


def _shuffle_batch(tensors, batch_size, capacity, min_after_dequeue,
                   keep_input, num_threads=1, seed=None, enqueue_many=False,
                   shapes=None, allow_smaller_final_batch=False,
                   shared_name=None, name=None):
  """Helper function for `shuffle_batch` and `maybe_shuffle_batch`."""
  if context.executing_eagerly():
    raise ValueError(
        "Input pipelines based on Queues are not supported when eager execution"
        " is enabled. Please use tf.data to ingest data into your model"
        " instead.")
  tensor_list = _as_tensor_list(tensors)
  with ops.name_scope(name, "shuffle_batch",
                      list(tensor_list) + [keep_input]) as name:
    if capacity <= min_after_dequeue:
      raise ValueError("capacity %d must be bigger than min_after_dequeue %d."
                       % (capacity, min_after_dequeue))
    tensor_list = _validate(tensor_list)
    keep_input = _validate_keep_input(keep_input, enqueue_many)
    tensor_list, sparse_info = _store_sparse_tensors(
        tensor_list, enqueue_many, keep_input)
    types = _dtypes([tensor_list])
    shapes = _shapes([tensor_list], shapes, enqueue_many)
    queue = data_flow_ops.RandomShuffleQueue(
        capacity=capacity, min_after_dequeue=min_after_dequeue, seed=seed,
        dtypes=types, shapes=shapes, shared_name=shared_name)
    _enqueue(queue, tensor_list, num_threads, enqueue_many, keep_input)
    full = (math_ops.cast(
        math_ops.maximum(0, queue.size() - min_after_dequeue), dtypes.float32) *
            (1. / (capacity - min_after_dequeue)))
    # Note that name contains a '/' at the end so we intentionally do not place
    # a '/' after %s below.
    summary_name = (
        "fraction_over_%d_of_%d_full" %
        (min_after_dequeue, capacity - min_after_dequeue))
    summary.scalar(summary_name, full)

    if allow_smaller_final_batch:
      dequeued = queue.dequeue_up_to(batch_size, name=name)
    else:
      dequeued = queue.dequeue_many(batch_size, name=name)
    dequeued = _restore_sparse_tensors(dequeued, sparse_info)
    return _as_original_type(tensors, dequeued)


def _shuffle_batch_join(tensors_list, batch_size, capacity,
                        min_after_dequeue, keep_input, seed=None,
                        enqueue_many=False, shapes=None,
                        allow_smaller_final_batch=False, shared_name=None,
                        name=None):
  """Helper function for `shuffle_batch_join` and `maybe_shuffle_batch_join`."""
  if context.executing_eagerly():
    raise ValueError(
        "Input pipelines based on Queues are not supported when eager execution"
        " is enabled. Please use tf.data to ingest data into your model"
        " instead.")
  tensor_list_list = _as_tensor_list_list(tensors_list)
  with ops.name_scope(name, "shuffle_batch_join",
                      _flatten(tensor_list_list) + [keep_input]) as name:
    tensor_list_list = _validate_join(tensor_list_list)
    keep_input = _validate_keep_input(keep_input, enqueue_many)
    tensor_list_list, sparse_info = _store_sparse_tensors_join(
        tensor_list_list, enqueue_many, keep_input)
    types = _dtypes(tensor_list_list)
    shapes = _shapes(tensor_list_list, shapes, enqueue_many)
    queue = data_flow_ops.RandomShuffleQueue(
        capacity=capacity, min_after_dequeue=min_after_dequeue, seed=seed,
        dtypes=types, shapes=shapes, shared_name=shared_name)
    _enqueue_join(queue, tensor_list_list, enqueue_many, keep_input)
    full = (math_ops.cast(
        math_ops.maximum(0, queue.size() - min_after_dequeue), dtypes.float32) *
            (1. / (capacity - min_after_dequeue)))
    # Note that name contains a '/' at the end so we intentionally do not place
    # a '/' after %s below.
    summary_name = (
        "fraction_over_%d_of_%d_full" %
        (min_after_dequeue, capacity - min_after_dequeue))
    summary.scalar(summary_name, full)

    if allow_smaller_final_batch:
      dequeued = queue.dequeue_up_to(batch_size, name=name)
    else:
      dequeued = queue.dequeue_many(batch_size, name=name)
    dequeued = _restore_sparse_tensors(dequeued, sparse_info)
    # tensors_list was validated to not be empty.
    return _as_original_type(tensors_list[0], dequeued)

# Batching functions ----------------------------------------------------------


@tf_export(v1=["train.batch"])
@deprecation.deprecated(
    None, "Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.batch(batch_size)` (or `padded_batch(...)` if "
    "`dynamic_pad=True`).")
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
  `shape` property will have a first `Dimension` value of `None`, and
  operations that depend on fixed batch_size would fail.

  Args:
    tensors: The list or dictionary of tensors to enqueue.
    batch_size: The new batch size pulled from the queue.
    num_threads: The number of threads enqueuing `tensors`.  The batching will
      be nondeterministic if `num_threads > 1`.
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
    A list or dictionary of tensors with the same types as `tensors` (except if
    the input is a list of one element, then it returns a tensor, not a list).

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensors`.

  @compatibility(eager)
  Input pipelines based on Queues are not supported when eager execution is
  enabled. Please use the `tf.data` API to ingest data under eager execution.
  @end_compatibility
  """
  return _batch(
      tensors,
      batch_size,
      keep_input=True,
      num_threads=num_threads,
      capacity=capacity,
      enqueue_many=enqueue_many,
      shapes=shapes,
      dynamic_pad=dynamic_pad,
      allow_smaller_final_batch=allow_smaller_final_batch,
      shared_name=shared_name,
      name=name)


@tf_export(v1=["train.maybe_batch"])
@deprecation.deprecated(
    None, "Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.filter(...).batch(batch_size)` (or `padded_batch(...)`"
    " if `dynamic_pad=True`).")
def maybe_batch(tensors, keep_input, batch_size, num_threads=1, capacity=32,
                enqueue_many=False, shapes=None, dynamic_pad=False,
                allow_smaller_final_batch=False, shared_name=None, name=None):
  """Conditionally creates batches of tensors based on `keep_input`.

  See docstring in `batch` for more details.

  Args:
    tensors: The list or dictionary of tensors to enqueue.
    keep_input: A `bool` Tensor.  This tensor controls whether the input is
      added to the queue or not.  If it is a scalar and evaluates `True`, then
      `tensors` are all added to the queue. If it is a vector and `enqueue_many`
      is `True`, then each example is added to the queue only if the
      corresponding value in `keep_input` is `True`. This tensor essentially
      acts as a filtering mechanism.
    batch_size: The new batch size pulled from the queue.
    num_threads: The number of threads enqueuing `tensors`.  The batching will
      be nondeterministic if `num_threads > 1`.
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
  return _batch(
      tensors,
      batch_size,
      keep_input,
      num_threads=num_threads,
      capacity=capacity,
      enqueue_many=enqueue_many,
      shapes=shapes,
      dynamic_pad=dynamic_pad,
      allow_smaller_final_batch=allow_smaller_final_batch,
      shared_name=shared_name,
      name=name)


@tf_export(v1=["train.batch_join"])
@deprecation.deprecated(
    None, "Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.interleave(...).batch(batch_size)` (or "
    "`padded_batch(...)` if `dynamic_pad=True`).")
def batch_join(tensors_list, batch_size, capacity=32, enqueue_many=False,
               shapes=None, dynamic_pad=False, allow_smaller_final_batch=False,
               shared_name=None, name=None):
  """Runs a list of tensors to fill a queue to create batches of examples.

  The `tensors_list` argument is a list of tuples of tensors, or a list of
  dictionaries of tensors.  Each element in the list is treated similarly
  to the `tensors` argument of `tf.compat.v1.train.batch()`.

  WARNING: This function is nondeterministic, since it starts a separate thread
  for each tensor.

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
  `shape` property will have a first `Dimension` value of `None`, and
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

  @compatibility(eager)
  Input pipelines based on Queues are not supported when eager execution is
  enabled. Please use the `tf.data` API to ingest data under eager execution.
  @end_compatibility
  """
  return _batch_join(
      tensors_list,
      batch_size,
      keep_input=True,
      capacity=capacity,
      enqueue_many=enqueue_many,
      shapes=shapes,
      dynamic_pad=dynamic_pad,
      allow_smaller_final_batch=allow_smaller_final_batch,
      shared_name=shared_name,
      name=name)


@tf_export(v1=["train.maybe_batch_join"])
@deprecation.deprecated(
    None, "Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.interleave(...).filter(...).batch(batch_size)` (or "
    "`padded_batch(...)` if `dynamic_pad=True`).")
def maybe_batch_join(tensors_list, keep_input, batch_size, capacity=32,
                     enqueue_many=False, shapes=None, dynamic_pad=False,
                     allow_smaller_final_batch=False, shared_name=None,
                     name=None):
  """Runs a list of tensors to conditionally fill a queue to create batches.

  See docstring in `batch_join` for more details.

  Args:
    tensors_list: A list of tuples or dictionaries of tensors to enqueue.
    keep_input: A `bool` Tensor.  This tensor controls whether the input is
      added to the queue or not.  If it is a scalar and evaluates `True`, then
      `tensors` are all added to the queue. If it is a vector and `enqueue_many`
      is `True`, then each example is added to the queue only if the
      corresponding value in `keep_input` is `True`. This tensor essentially
      acts as a filtering mechanism.
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
  return _batch_join(
      tensors_list,
      batch_size,
      keep_input,
      capacity=capacity,
      enqueue_many=enqueue_many,
      shapes=shapes,
      dynamic_pad=dynamic_pad,
      allow_smaller_final_batch=allow_smaller_final_batch,
      shared_name=shared_name,
      name=name)


@tf_export(v1=["train.shuffle_batch"])
@deprecation.deprecated(
    None, "Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.shuffle(min_after_dequeue).batch(batch_size)`.")
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
  image_batch, label_batch = tf.compat.v1.train.shuffle_batch(
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
  `shape` property will have a first `Dimension` value of `None`, and
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

  @compatibility(eager)
  Input pipelines based on Queues are not supported when eager execution is
  enabled. Please use the `tf.data` API to ingest data under eager execution.
  @end_compatibility
  """
  return _shuffle_batch(
      tensors,
      batch_size,
      capacity,
      min_after_dequeue,
      keep_input=True,
      num_threads=num_threads,
      seed=seed,
      enqueue_many=enqueue_many,
      shapes=shapes,
      allow_smaller_final_batch=allow_smaller_final_batch,
      shared_name=shared_name,
      name=name)


@tf_export(v1=["train.maybe_shuffle_batch"])
@deprecation.deprecated(
    None, "Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.filter(...).shuffle(min_after_dequeue).batch(batch_size)`"
    ".")
def maybe_shuffle_batch(tensors, batch_size, capacity, min_after_dequeue,
                        keep_input, num_threads=1, seed=None,
                        enqueue_many=False, shapes=None,
                        allow_smaller_final_batch=False, shared_name=None,
                        name=None):
  """Creates batches by randomly shuffling conditionally-enqueued tensors.

  See docstring in `shuffle_batch` for more details.

  Args:
    tensors: The list or dictionary of tensors to enqueue.
    batch_size: The new batch size pulled from the queue.
    capacity: An integer. The maximum number of elements in the queue.
    min_after_dequeue: Minimum number elements in the queue after a
      dequeue, used to ensure a level of mixing of elements.
    keep_input: A `bool` Tensor.  This tensor controls whether the input is
      added to the queue or not.  If it is a scalar and evaluates `True`, then
      `tensors` are all added to the queue. If it is a vector and `enqueue_many`
      is `True`, then each example is added to the queue only if the
      corresponding value in `keep_input` is `True`. This tensor essentially
      acts as a filtering mechanism.
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

  @compatibility(eager)
  Input pipelines based on Queues are not supported when eager execution is
  enabled. Please use the `tf.data` API to ingest data under eager execution.
  @end_compatibility
  """
  return _shuffle_batch(
      tensors,
      batch_size,
      capacity,
      min_after_dequeue,
      keep_input,
      num_threads=num_threads,
      seed=seed,
      enqueue_many=enqueue_many,
      shapes=shapes,
      allow_smaller_final_batch=allow_smaller_final_batch,
      shared_name=shared_name,
      name=name)


@tf_export(v1=["train.shuffle_batch_join"])
@deprecation.deprecated(
    None, "Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.interleave(...).shuffle(min_after_dequeue).batch"
    "(batch_size)`.")
def shuffle_batch_join(tensors_list, batch_size, capacity,
                       min_after_dequeue, seed=None, enqueue_many=False,
                       shapes=None, allow_smaller_final_batch=False,
                       shared_name=None, name=None):
  """Create batches by randomly shuffling tensors.

  The `tensors_list` argument is a list of tuples of tensors, or a list of
  dictionaries of tensors.  Each element in the list is treated similarly
  to the `tensors` argument of `tf.compat.v1.train.shuffle_batch()`.

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
  `shape` property will have a first `Dimension` value of `None`, and
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

  @compatibility(eager)
  Input pipelines based on Queues are not supported when eager execution is
  enabled. Please use the `tf.data` API to ingest data under eager execution.
  @end_compatibility
  """
  return _shuffle_batch_join(
      tensors_list,
      batch_size,
      capacity,
      min_after_dequeue,
      keep_input=True,
      seed=seed,
      enqueue_many=enqueue_many,
      shapes=shapes,
      allow_smaller_final_batch=allow_smaller_final_batch,
      shared_name=shared_name,
      name=name)


@tf_export(v1=["train.maybe_shuffle_batch_join"])
@deprecation.deprecated(
    None, "Queue-based input pipelines have been replaced by `tf.data`. Use "
    "`tf.data.Dataset.interleave(...).filter(...).shuffle(min_after_dequeue)"
    ".batch(batch_size)`.")
def maybe_shuffle_batch_join(tensors_list, batch_size, capacity,
                             min_after_dequeue, keep_input, seed=None,
                             enqueue_many=False, shapes=None,
                             allow_smaller_final_batch=False, shared_name=None,
                             name=None):
  """Create batches by randomly shuffling conditionally-enqueued tensors.

  See docstring in `shuffle_batch_join` for more details.

  Args:
    tensors_list: A list of tuples or dictionaries of tensors to enqueue.
    batch_size: An integer. The new batch size pulled from the queue.
    capacity: An integer. The maximum number of elements in the queue.
    min_after_dequeue: Minimum number elements in the queue after a
      dequeue, used to ensure a level of mixing of elements.
    keep_input: A `bool` Tensor.  This tensor controls whether the input is
      added to the queue or not.  If it is a scalar and evaluates `True`, then
      `tensors` are all added to the queue. If it is a vector and `enqueue_many`
      is `True`, then each example is added to the queue only if the
      corresponding value in `keep_input` is `True`. This tensor essentially
      acts as a filtering mechanism.
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

  @compatibility(eager)
  Input pipelines based on Queues are not supported when eager execution is
  enabled. Please use the `tf.data` API to ingest data under eager execution.
  @end_compatibility
  """
  return _shuffle_batch_join(
      tensors_list,
      batch_size,
      capacity,
      min_after_dequeue,
      keep_input,
      seed=seed,
      enqueue_many=enqueue_many,
      shapes=shapes,
      allow_smaller_final_batch=allow_smaller_final_batch,
      shared_name=shared_name,
      name=name)

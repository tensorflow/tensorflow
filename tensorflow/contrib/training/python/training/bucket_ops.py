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

"""Operations for bucketing data into groups.

The classes and functions in this module are used to queue up data into
buckets conditional on side information (e.g. sequence length).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from tensorflow.python import summary
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import input as input_py
from tensorflow.python.training import queue_runner


# pylint: disable=protected-access
_as_original_type = input_py._as_original_type
_as_tensor_list = input_py._as_tensor_list
_restore_sparse_tensors = input_py._restore_sparse_tensors
_dtypes = input_py._dtypes
_store_sparse_tensors = input_py._store_sparse_tensors
_shapes = input_py._shapes
_which_queue = input_py._which_queue
# pylint: enable=protected-access


def _validate_bucket(tensor_list):
  tensor_list = ops.convert_n_to_tensor_or_indexed_slices(tensor_list)
  if not tensor_list:
    raise ValueError("Expected at least one tensor in bucket().")
  return tensor_list


def bucket(tensors,
           which_bucket,
           batch_size,
           num_buckets,
           num_threads=1,
           capacity=32,
           shapes=None,
           dynamic_pad=False,
           allow_smaller_final_batch=False,
           keep_input=None,
           shared_name=None,
           name=None):
  """Lazy bucketing of input tensors according to `which_bucket`.

  The argument `tensors` can be a list or a dictionary of tensors.
  The value returned by the function will be of the same type
  as `tensors`.

  The tensors entering this function are put into the bucket given by
  `which_bucket`.  Each bucket has its own queue.  When a bucket contains
  `batch_size` elements, this minibatch is pushed onto a top queue.  The
  tensors returned from this function are a the result of dequeueing the
  next minibatch from this top queue.

  This function is implemented using several queues. A `QueueRunner` for the
  queues is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  As the returned tensors are the result of of a dequeue operation, evaluating
  them will throw a `tf.errors.OutOfRangeError` when the input queue is
  exhausted.  If these tensors are feeding another input queue, its queue runner
  will catch this exception, however, if they are used in your main thread
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
  `batch_size` is returned when the queues are closed and there are not enough
  elements to fill the batch, otherwise the pending elements are discarded.
  In addition, all output tensors' static shapes, as accessed via the
  `get_shape()` method will have a 0th `Dimension` value of `None`, and
  operations that depend on fixed batch_size would fail.

  Args:
    tensors: The list or dictionary of tensors, representing a single element,
      to bucket.  Nested lists are not supported.
    which_bucket: An `int32` scalar Tensor taking a value in `[0, num_buckets)`.
    batch_size: The new batch size pulled from the queue
      (python int or int32 scalar).
    num_buckets: A python integer, the number of buckets.
    num_threads: An integer.  The number of threads enqueuing `tensors`.
    capacity: An integer. The maximum number of minibatches in the top queue,
      and also the maximum number of elements within each bucket.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensors`.
    dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
      The given dimensions are padded upon dequeue so that tensors within a
      batch have the same shapes.
    allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
      batches to be smaller if there are insufficient items left in the queues.
    keep_input: (Optional).  A `bool` scalar Tensor.  If provided, this tensor
      controls whether the input is added to the queue or not.  If it evaluates
      `True`, then `tensors` are added to the bucket; otherwise they are
      dropped.  This tensor essentially acts as a filtering mechanism.
      The default behavior is to assume `keep_input=True`.
    shared_name: (Optional). If set, the queues will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A tuple `(bucket, outputs)` where `bucket` is
    a `int32` scalar tensor and `outputs` is a list or
    dictionary of batched outputs corresponding to elements of `tensors`.
    Every step will receive a new bucket of outputs.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensors`.
  """
  tensor_list = _as_tensor_list(tensors)
  with ops.name_scope(name, "bucket", tensor_list) as name:
    tensor_list = _validate_bucket(tensor_list)
    (tensor_list, sparse_info) = _store_sparse_tensors(
        tensor_list, enqueue_many=False, keep_input=constant_op.constant(True))

    # Round-trip batch_size to a tensor, and possibly back
    batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int32, name="batch_size")
    static_batch_size = tensor_util.constant_value(batch_size)
    batch_size = (
        static_batch_size if static_batch_size is not None else batch_size)

    types = _dtypes([tensor_list])
    shapes = _shapes([tensor_list], shapes, enqueue_many=False)

    which_bucket = ops.convert_to_tensor(
        which_bucket, dtype=dtypes.int32, name="which_bucket")

    queue_creator = _which_queue(dynamic_pad)
    bucket_queues = []
    for i in range(num_buckets):
      shared_name_i = (
          "%s_%d" % (shared_name, i) if shared_name is not None else None)
      bucket_queues.append(
          queue_creator(capacity=capacity,
                        dtypes=types,
                        shapes=shapes,
                        shared_name=shared_name_i, name="bucket_queue_%d" % i))

    maybe_static_batch_size = (
        None if allow_smaller_final_batch else static_batch_size)

    bucket_shapes = [tensor_shape.vector(maybe_static_batch_size).concatenate(s)
                     for s in bucket_queues[0].shapes]
    # top_queue is a PaddingFIFOQueue even if the bucket queues are regular FIFO
    # queues because if we use allow_smaller_final_batch, shapes will
    # contain Nones in their first entry; as a result, a regular
    # FIFOQueue would die when being passed shapes that are not fully defined.
    top_queue = data_flow_ops.PaddingFIFOQueue(
        capacity=capacity,
        dtypes=[dtypes.int32] + types,
        shapes=[tensor_shape.scalar()] + bucket_shapes,
        shared_name=shared_name, name="top_queue")

    def enqueue_which():
      def enqueue_single(i):
        return bucket_queues[i].enqueue(tensor_list)
      enqueues = [
          control_flow_ops.cond(
              math_ops.equal(which_bucket, i),
              functools.partial(enqueue_single, i),
              control_flow_ops.no_op)
          for i in range(num_buckets)]
      return control_flow_ops.group(*enqueues, name="group_enqueues")

    if keep_input is not None:
      # TODO(ebrevdo): Expand keep_input param to core training
      # methods, and pipe through to _store_sparse_tensors; so
      # that expensive serialization is guarded by keep_input.
      maybe_enqueue = control_flow_ops.cond(
          keep_input,
          enqueue_which,
          control_flow_ops.no_op)
    else:
      maybe_enqueue = enqueue_which()

    bucket_enqueue_ops = [maybe_enqueue] * num_threads

    if allow_smaller_final_batch:
      which_dequeue = lambda q: q.dequeue_up_to
    else:
      which_dequeue = lambda q: q.dequeue_many

    enqueues_to_top = [
        top_queue.enqueue(
            [constant_op.constant(i)] +
            which_dequeue(q)(batch_size, name="read_bucket_%d" % i),
            name="enqueue_from_bucket_%d" % i)
        for i, q in enumerate(bucket_queues)]

    for i, q in enumerate(bucket_queues):
      queue_runner.add_queue_runner(queue_runner.QueueRunner(
          q, [enqueues_to_top[i]],
          queue_closed_exception_types=(
              errors.OutOfRangeError, errors.CancelledError)))
    queue_runner.add_queue_runner(queue_runner.QueueRunner(
        top_queue, bucket_enqueue_ops,
        queue_closed_exception_types=(
            errors.OutOfRangeError, errors.CancelledError)))

    for q in bucket_queues:
      summary.scalar("bucket/%s/size" % q.name,
                     math_ops.cast(top_queue.size(), dtypes.float32))
    summary.scalar("bucket/%s/fraction_of_%d_full" % (top_queue.name, capacity),
                   math_ops.cast(top_queue.size(), dtypes.float32) *
                   (1. / capacity))

    dequeued = top_queue.dequeue(name="dequeue_top")
    which_bucket_dequeued = dequeued[0]
    dequeued = dequeued[1:]
    dequeued = _restore_sparse_tensors(dequeued, sparse_info)
    return (which_bucket_dequeued, _as_original_type(tensors, dequeued))


def bucket_by_sequence_length(input_length,
                              tensors,
                              batch_size,
                              bucket_boundaries,
                              num_threads=1,
                              capacity=32,
                              shapes=None,
                              dynamic_pad=False,
                              allow_smaller_final_batch=False,
                              keep_input=None,
                              shared_name=None,
                              name=None):
  """Lazy bucketing of inputs according to their length.

  This method calls `tf.contrib.training.bucket` under the hood, after first
  subdividing the bucket boundaries into separate buckets and identifying which
  bucket the given `input_length` belongs to.  See the documentation for
  `which_bucket` for details of the other arguments.

  Args:
    input_length: `int32` scalar `Tensor`, the sequence length of tensors.
    tensors: The list or dictionary of tensors, representing a single element,
      to bucket.  Nested lists are not supported.
    batch_size: The new batch size pulled from the queue
      (python int or int32 scalar).
    bucket_boundaries: int list, increasing non-negative numbers.
      The edges of the buckets to use when bucketing tensors.  Two extra buckets
      are created, one for `input_length < bucket_boundaries[0]` and
      one for `input_length >= bucket_boundaries[-1]`.
    num_threads: An integer.  The number of threads enqueuing `tensors`.
    capacity: An integer. The maximum number of minibatches in the top queue,
      and also the maximum number of elements within each bucket.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensors`.
    dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
      The given dimensions are padded upon dequeue so that tensors within a
      batch have the same shapes.
    allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
      batches to be smaller if there are insufficient items left in the queues.
    keep_input: (Optional).  A `bool` scalar Tensor.  If provided, this tensor
      controls whether the input is added to the queue or not.  If it evaluates
      `True`, then `tensors` are added to the bucket; otherwise they are
      dropped.  This tensor essentially acts as a filtering mechanism.
      The default behavior is to assume `keep_input=True`.
    shared_name: (Optional). If set, the queues will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A tuple `(sequence_length, outputs)` where `sequence_length` is
    a 1-D `Tensor` of size `batch_size` and `outputs` is a list or dictionary
    of batched, bucketed, outputs corresponding to elements of `tensors`.

  Raises:
    TypeError: if `bucket_boundaries` is not a list of python integers.
    ValueError: if `bucket_boundaries` is empty or contains non-increasing
      values.
  """
  tensor_list = _as_tensor_list(tensors)
  if not isinstance(bucket_boundaries, (list, tuple)):
    raise TypeError(
        "bucket_boundaries must be a list or tuple, but received: %s"
        % bucket_boundaries)
  if not bucket_boundaries:
    raise ValueError("bucket_boundaries must not be empty")
  for (s, e) in zip(bucket_boundaries[:-1], bucket_boundaries[1:]):
    if not isinstance(s, int) or not isinstance(e, int):
      raise TypeError(
          "bucket boundaries must be integers, but saw: %s and %s" % (s, e))
    if s >= e:
      raise ValueError(
          "Buckets must contain sequential increasing lengths, but saw: "
          "%d before %d" % (s, e))

  with ops.name_scope(name, "bucket_by_sequence_length",
                      [input_length] + tensor_list) as name:
    input_length = ops.convert_to_tensor(
        input_length, dtype=dtypes.int32, name="input_length")
    # Bucketing conditions are:
    #   l < b[0]
    #   b[0] <= l < b[1]
    #   b[1] <= l < b[2]
    #   ...
    #   b[N-2] <= l < b[N-1]
    #   b[N-1] <= l
    # Equivalent to:
    #   [-inf, b[0], b[1], ..., b[N-1]] <= l < [b[0], b[1], ..., b[N-1], inf]
    buckets_min = [np.iinfo(np.int32).min] + list(bucket_boundaries)
    buckets_max = list(bucket_boundaries) + [np.iinfo(np.int32).max]
    conditions_c = math_ops.logical_and(
        math_ops.less_equal(buckets_min, input_length),
        math_ops.less(input_length, buckets_max))
    which_bucket = math_ops.reduce_min(array_ops.where(conditions_c))
    which_bucket = math_ops.to_int32(which_bucket)

    if shapes is not None:
      shapes = [tensor_shape.scalar()] + shapes

    _, dequeued = bucket(
        tensors=[input_length] + tensor_list,
        which_bucket=which_bucket,
        batch_size=batch_size,
        num_buckets=len(bucket_boundaries) + 1,
        num_threads=num_threads,
        capacity=capacity,
        shapes=shapes,
        dynamic_pad=dynamic_pad,
        allow_smaller_final_batch=allow_smaller_final_batch,
        keep_input=keep_input,
        shared_name=shared_name)

    return (dequeued[0], _as_original_type(tensors, dequeued[1:]))


__all__ = [
    "bucket",
    "bucket_by_sequence_length"
]

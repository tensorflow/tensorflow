"""Data Flow Operations."""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_data_flow_ops import *


def _as_type_list(dtypes):
  """Convert dtypes to a list of types."""
  assert dtypes is not None
  if not (isinstance(dtypes, list) or isinstance(dtypes, tuple)):
    # We have a single type.
    return [dtypes]
  else:
    # We have a list or tuple of types.
    return list(dtypes)


def _as_shape_list(shapes, dtypes):
  """Convert shapes to a list of tuples of int (or None)."""
  if shapes is None: return None
  if isinstance(shapes, tensor_shape.TensorShape):
    shapes = [shapes]
  if not isinstance(shapes, (tuple, list)):
    raise TypeError(
        "shapes must be a TensorShape or a list or tuple of TensorShapes.")
  if all(isinstance(shape, int) for shape in shapes):
    # We have a single shape.
    shapes = [shapes]
  shapes = [tensor_shape.as_shape(shape) for shape in shapes]
  if any(not shape.is_fully_defined() for shape in shapes):
    raise ValueError("All shapes must be fully defined.")
  return shapes


# pylint: disable=protected-access
class QueueBase(object):
  """Base class for queue implementations.

  A queue is a TensorFlow data structure that stores tensors across
  multiple steps, and exposes operations that enqueue and dequeue
  tensors.

  Each queue element is a tuple of one or more tensors, where each
  tuple component has a static dtype, and may have a static shape. The
  queue implementations support versions of enqueue and dequeue that
  handle single elements, versions that support enqueuing and
  dequeuing a batch of elements at once.

  See [`tf.FIFOQueue`](#FIFOQueue) and
  [`tf.RandomShuffleQueue`](#RandomShuffleQueue) for concrete
  implementations of this class, and instructions on how to create
  them.

  @@enqueue
  @@enqueue_many

  @@dequeue
  @@dequeue_many

  @@size

  @@close

  """

  def __init__(self, dtypes, shapes, queue_ref):
    """Constructs a queue object from a queue reference.

    Args:
      dtypes:  A list of types.  The length of dtypes must equal the number
        of tensors in each element.
      shapes: Constraints on the shapes of tensors in an element:
        A list of shape tuples or None. This list is the same length
        as dtypes.  If the shape of any tensors in the element are constrained,
        all must be; shapes can be None if the shapes should not be constrained.
      queue_ref: The queue reference, i.e. the output of the queue op.
    """
    self._dtypes = dtypes
    if shapes is not None:
      self._shapes = [tensor_shape.TensorShape(s) for s in shapes]
    else:
      self._shapes = [tensor_shape.unknown_shape() for _ in self._dtypes]
    self._queue_ref = queue_ref
    self._name = self._queue_ref.op.name.split("/")[-1]

  @staticmethod
  def from_list(index, queues):
    """Create a queue using the queue reference from `queues[index]`.

    Args:
      index: An integer scalar tensor that determines the input that gets
        selected.
      queues: A list of `QueueBase` objects.

    Returns:
      A `QueueBase` object.

    Raises:
      TypeError: when `queues` is not a list of `QueueBase` objects,
        or when the data types of `queues` are not all the same.
    """
    if ((not queues) or
        (not isinstance(queues, list)) or
        (not all([isinstance(x, QueueBase) for x in queues]))):
      raise TypeError("A list of queues expected")

    dtypes = queues[0].dtypes
    if not all([dtypes == q.dtypes for q in queues[1:]]):
      raise TypeError("Queues do not have matching component dtypes.")

    queue_refs = [x.queue_ref for x in queues]
    selected_queue = control_flow_ops.ref_select(index, queue_refs)
    # TODO(josh11b): Unify the shapes of the queues too?
    return QueueBase(dtypes=dtypes, shapes=None, queue_ref=selected_queue)

  @property
  def queue_ref(self):
    """The underlying queue reference."""
    return self._queue_ref

  @property
  def name(self):
    """The name of the underlying queue."""
    return self._queue_ref.op.name

  @property
  def dtypes(self):
    """The list of dtypes for each component of a queue element."""
    return self._dtypes

  def enqueue(self, vals, name=None):
    """Enqueues one element to this queue.

    If the queue is full when this operation executes, it will block
    until the element has been enqueued.

    Args:
      vals: The tuple of `Tensor` objects to be enqueued.
      name: A name for the operation (optional).

    Returns:
      The operation that enqueues a new tuple of tensors to the queue.
    """
    if name is None:
      name = "%s_enqueue" % self._name
    ret = gen_data_flow_ops._queue_enqueue(self._queue_ref, vals, name=name)

    # NOTE(mrry): Not using a shape function because we need access to
    # the Queue object.
    for val, shape in zip(ret.inputs[1:], self._shapes):
      val.get_shape().assert_is_compatible_with(shape)

    return ret

  def enqueue_many(self, vals, name=None):
    """Enqueues zero or elements to this queue.

    This operation slices each component tensor along the 0th dimension to
    make multiple queue elements. All of the tensors in `vals` must have the
    same size in the 0th dimension.

    If the queue is full when this operation executes, it will block
    until all of the elements have been enqueued.

    Args:
      vals: The tensor or tuple of tensors from which the queue elements
        are taken.
      name: A name for the operation (optional).

    Returns:
      The operation that enqueues a batch of tuples of tensors to the queue.
    """
    if name is None:
      name = "%s_EnqueueMany" % self._name

    ret = gen_data_flow_ops._queue_enqueue_many(
        self._queue_ref, vals, name=name)

    # NOTE(mrry): Not using a shape function because we need access to
    # the `QueueBase` object.
    batch_dim = ret.inputs[1].get_shape()[0]
    for val, shape in zip(ret.inputs[1:], self._shapes):
      batch_dim.merge_with(val.get_shape()[0])
      val.get_shape()[1:].assert_is_compatible_with(shape)

    return ret

  def dequeue(self, name=None):
    """Dequeues one element from this queue.

    If the queue is empty when this operation executes, it will block
    until there is an element to dequeue.

    Args:
      name: A name for the operation (optional).

    Returns:
      The tuple of tensors that was dequeued.
    """
    if name is None:
      name = "%s_Dequeue" % self._name
    ret = gen_data_flow_ops._queue_dequeue(
        self._queue_ref, self._dtypes, name=name)

    # NOTE(mrry): Not using a shape function because we need access to
    # the `QueueBase` object.
    op = ret[0].op
    for output, shape in zip(op.values(), self._shapes):
      output.set_shape(shape)

    return ret if len(ret) != 1 else ret[0]

  def dequeue_many(self, n, name=None):
    """Dequeues and concatenates `n` elements from this queue.

    This operation concatenates queue-element component tensors along
    the 0th dimension to make a single component tensor.  All of the
    components in the dequeued tuple will have size `n` in the 0th dimension.

    If the queue contains fewer than `n` elements when this operation
    executes, it will block until `n` elements have been dequeued.

    Args:
      n: A scalar `Tensor` containing the number of elements to dequeue.
      name: A name for the operation (optional).

    Returns:
      The tuple of concatenated tensors that was dequeued.
    """
    if name is None:
      name = "%s_DequeueMany" % self._name

    ret = gen_data_flow_ops._queue_dequeue_many(
        self._queue_ref, n, self._dtypes, name=name)

    # NOTE(mrry): Not using a shape function because we need access to
    # the Queue object.
    op = ret[0].op
    batch_dim = tensor_shape.Dimension(tensor_util.ConstantValue(op.inputs[1]))
    for output, shape in zip(op.values(), self._shapes):
      output.set_shape(tensor_shape.TensorShape([batch_dim]).concatenate(shape))

    return ret if len(ret) != 1 else ret[0]

  def close(self, cancel_pending_enqueues=False, name=None):
    """Closes this queue.

    This operation signals that no more elements will be enqueued in
    the given queue. Subsequent `enqueue` and `enqueue_many`
    operations will fail. Subsequent `dequeue` and `dequeue_many`
    operations will continue to succeed if sufficient elements remain
    in the queue. Subsequent `dequeue` and `dequeue_many` operations
    that would block will fail immediately.

    If `cancel_pending_enqueues` is `True`, all pending requests will also
    be cancelled.

    Args:
      cancel_pending_enqueues: (Optional.) A boolean, defaulting to
        `False` (described above).
      name: A name for the operation (optional).

    Returns:
      The operation that closes the queue.
    """
    if name is None:
      name = "%s_Close" % self._name
    return gen_data_flow_ops._queue_close(
        self._queue_ref, cancel_pending_enqueues=cancel_pending_enqueues,
        name=name)

  def size(self, name=None):
    """Compute the number of elements in this queue.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this queue.
    """
    if name is None:
      name = "%s_Size" % self._name
    return gen_data_flow_ops._queue_size(self._queue_ref, name=name)


class RandomShuffleQueue(QueueBase):
  """A queue implementation that dequeues elements in a random order.

  See [`tf.QueueBase`](#QueueBase) for a description of the methods on
  this class.

  @@__init__
  """

  def __init__(self, capacity, min_after_dequeue, dtypes, shapes=None,
               seed=None, shared_name=None, name="random_shuffle_queue"):
    """Create a queue that dequeues elements in a random order.

    A `RandomShuffleQueue` has bounded capacity; supports multiple
    concurrent producers and consumers; and provides exactly-once
    delivery.

    A `RandomShuffleQueue` holds a list of up to `capacity`
    elements. Each element is a fixed-length tuple of tensors whose
    dtypes are described by `dtypes`, and whose shapes are optionally
    described by the `shapes` argument.

    If the `shapes` argument is specified, each component of a queue
    element must have the respective fixed shape. If it is
    unspecified, different queue elements may have different shapes,
    but the use of `dequeue_many` is disallowed.

    The `min_after_dequeue` argument allows the caller to specify a
    minimum number of elements that will remain in the queue after a
    `dequeue` or `dequeue_many` operation completes, to ensure a
    minimum level of mixing of elements. This invariant is maintained
    by blocking those operations until sufficient elements have been
    enqueued. The `min_after_dequeue` argument is ignored after the
    queue has been closed.

    Args:
      capacity: An integer. The upper bound on the number of elements
        that may be stored in this queue.
      min_after_dequeue: An integer (described above).
      dtypes:  A list of `DType` objects. The length of `dtypes` must equal
        the number of tensors in each queue element.
      shapes: (Optional.) A list of fully-defined `TensorShape` objects,
        with the same length as `dtypes` or `None`.
      seed: A Python integer. Used to create a random seed. See
        [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
        for behavior.
      shared_name: (Optional.) If non-empty, this queue will be shared under
        the given name across multiple sessions.
      name: Optional name for the queue operation.
    """
    dtypes = _as_type_list(dtypes)
    shapes = _as_shape_list(shapes, dtypes)
    seed1, seed2 = random_seed.get_seed(seed)
    queue_ref = gen_data_flow_ops._random_shuffle_queue(
        component_types=dtypes, shapes=shapes, capacity=capacity,
        min_after_dequeue=min_after_dequeue, seed=seed1, seed2=seed2,
        shared_name=shared_name, name=name)

    super(RandomShuffleQueue, self).__init__(dtypes, shapes, queue_ref)


class FIFOQueue(QueueBase):
  """A queue implementation that dequeues elements in first-in-first out order.

  See [`tf.QueueBase`](#QueueBase) for a description of the methods on
  this class.

  @@__init__
  """

  def __init__(self, capacity, dtypes, shapes=None, shared_name=None,
               name="fifo_queue"):
    """Creates a queue that dequeues elements in a first-in first-out order.

    A `FIFOQueue` has bounded capacity; supports multiple concurrent
    producers and consumers; and provides exactly-once delivery.

    A `FIFOQueue` holds a list of up to `capacity` elements. Each
    element is a fixed-length tuple of tensors whose dtypes are
    described by `dtypes`, and whose shapes are optionally described
    by the `shapes` argument.

    If the `shapes` argument is specified, each component of a queue
    element must have the respective fixed shape. If it is
    unspecified, different queue elements may have different shapes,
    but the use of `dequeue_many` is disallowed.

    Args:
      capacity: An integer. The upper bound on the number of elements
        that may be stored in this queue.
      dtypes:  A list of `DType` objects. The length of `dtypes` must equal
        the number of tensors in each queue element.
      shapes: (Optional.) A list of fully-defined `TensorShape` objects,
        with the same length as `dtypes` or `None`.
      shared_name: (Optional.) If non-empty, this queue will be shared under
        the given name across multiple sessions.
      name: Optional name for the queue operation.
    """
    dtypes = _as_type_list(dtypes)
    shapes = _as_shape_list(shapes, dtypes)
    queue_ref = gen_data_flow_ops._fifo_queue(
        component_types=dtypes, shapes=shapes, capacity=capacity,
        shared_name=shared_name, name=name)

    super(FIFOQueue, self).__init__(dtypes, shapes, queue_ref)


# TODO(josh11b): class BatchQueue(QueueBase):


def initialize_all_tables(name="init_all_tables"):
  """Returns an Op that initializes all tables of the default graph.

  Args:
    name: Optional name for the initialization op.

  Returns:
    An Op that initializes all tables.  Note that if there are
    not tables the returned Op is a NoOp.
  """
  initializers = ops.get_collection(ops.GraphKeys.TABLE_INITIALIZERS)
  if initializers:
    return control_flow_ops.group(*initializers, name=name)
  return control_flow_ops.no_op(name=name)


ops.NoGradient("LookupTableFind")
ops.NoGradient("LookupTableSize")
ops.NoGradient("HashTable")
ops.NoGradient("InitializeTable")


ops.RegisterShape("QueueSize")(common_shapes.scalar_shape)
ops.RegisterShape("Queue")(common_shapes.scalar_shape)
ops.RegisterShape("FIFOQueue")(common_shapes.scalar_shape)
ops.RegisterShape("RandomShuffleQueue")(common_shapes.scalar_shape)


# NOTE(mrry): The following ops use higher-level information in the
# Queue class to provide shape information.
ops.RegisterShape("QueueDequeue")(common_shapes.unknown_shape)
ops.RegisterShape("QueueDequeueMany")(common_shapes.unknown_shape)
ops.RegisterShape("QueueEnqueue")(common_shapes.unknown_shape)
ops.RegisterShape("QueueEnqueueMany")(common_shapes.unknown_shape)


@ops.RegisterShape("QueueClose")
def _ScalarToVoidShape(op):
  """Shape function for ops that take a scalar and produce no outputs."""
  unused_input_shape = op.inputs[0].get_shape().merge_with(
      tensor_shape.scalar())
  return []


@ops.RegisterShape("DynamicPartition")
def _DynamicPartitionShape(op):
  """Shape function for data_flow_ops.dynamic_partition."""
  data_shape = op.inputs[0].get_shape()
  partitions_shape = op.inputs[1].get_shape()
  # If we don't know the rank of partitions, we don't know anything
  mid = partitions_shape.ndims
  if mid is None:
    result_shape = tensor_shape.unknown_shape()
  else:
    # data_shape must start with partitions_shape
    partitions_shape.assert_is_compatible_with(data_shape[:mid])
    # The partition shape is dynamic in the 0th dimension, and matches
    # data_shape in the remaining dimensions.
    result_shape = tensor_shape.TensorShape([None]).concatenate(
        data_shape[mid:])
  return [result_shape] * op.get_attr("num_partitions")


@ops.RegisterShape("DynamicStitch")
def _DynamicStitchShape(op):
  """Shape function for data_flow_ops.dynamic_stitch."""
  num_partitions = op.get_attr("N")
  indices_shapes = [t.get_shape() for t in op.inputs[0:num_partitions]]
  data_shapes = [t.get_shape() for t in op.inputs[num_partitions:]]
  output_shape = tensor_shape.unknown_shape()
  extra_shape = tensor_shape.TensorShape(None)
  for indices_shape, data_shape in zip(indices_shapes, data_shapes):
    indices_ndims = indices_shape.ndims
    if indices_ndims is not None:
      # Assert that data_shape starts with indices_shape
      indices_shape.merge_with(data_shape[:indices_ndims])
      # The rest belongs to output
      extra_shape = extra_shape.merge_with(data_shape[indices_ndims:])
  return [tensor_shape.TensorShape([None]).concatenate(extra_shape)]


@ops.RegisterShape("LookupTableFind")
def _LookupTableFindShape(op):
  """Shape function for data_flow_ops._lookup_table_find."""
  unused_table_shape = op.inputs[0].get_shape().merge_with(
      tensor_shape.scalar())
  shape_in = op.inputs[1].get_shape()
  return [shape_in]


@ops.RegisterShape("LookupTableSize")
def _LookupTableSizeShape(op):
  """Shape function for data_flow_ops._lookup_table_find."""
  unused_table_shape = op.inputs[0].get_shape().merge_with(
      tensor_shape.scalar())
  return [tensor_shape.scalar()]


@ops.RegisterShape("HashTable")
def _HashTableShape(unused_op):
  """Shape function for data_flow_ops._hash_table."""
  return [tensor_shape.scalar()]


@ops.RegisterShape("InitializeTable")
def _InitializeLookupTableShape(op):
  """Shape function for data_flow_ops._initialize_table."""
  unused_table_shape = op.inputs[0].get_shape().merge_with(
      tensor_shape.scalar())
  keys_shape = op.inputs[1].get_shape().with_rank(1)
  unused_values_shape = op.inputs[2].get_shape().merge_with(keys_shape)
  return []

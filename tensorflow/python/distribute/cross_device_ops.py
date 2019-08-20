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
"""Classes for different algorithms of reduction and broadcasting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import enum
import six

from tensorflow.python.client import device_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import kernels
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls


def check_destinations(destinations):
  """Checks whether `destinations` is not empty.

  Args:
    destinations: a `DistributedValues`, variable, or string object.

  Returns:
    Boolean which is True if `destinations` is not empty.
  """
  # Calling bool() on a ResourceVariable is not allowed.
  if isinstance(destinations,
                (resource_variable_ops.BaseResourceVariable, ops.Tensor)):
    return bool(destinations.device)
  return bool(destinations)


def validate_destinations(destinations):
  """Validates the `destination` is one of expected types."""
  if not isinstance(
      destinations,
      (
          value_lib.DistributedValues,
          resource_variable_ops.BaseResourceVariable,
          ops.Tensor,
          value_lib.AggregatingVariable,
          six.string_types,
          value_lib.TPUMirroredVariable,
          # LogicalDeviceSpec is only used internally, e.g. as a
          # broadcast destination, never supplied by a user.
          value_lib.LogicalDeviceSpec)):
    raise ValueError("destinations must be one of a `DistributedValues` object,"
                     " a tf.Variable object, or a device string.")

  if not check_destinations(destinations):
    raise ValueError("destinations can not be empty")


def reduce_non_distributed_value(reduce_op, device_map, value, destinations):
  """Reduce a non-DistributedValue `value` to `destinations`."""
  if isinstance(value, value_lib.DistributedValues):
    raise ValueError("You are passing a `DistributedValue` to "
                     "`reduce_non_distributed_value`, which is not allowed.")

  # If the same value is present on all replicas then the PerReplica value will
  # be a single value. We also handle the case when `value` is a single value
  # and equal to 0.
  # TODO:(b/138823479): handle the tensor value properly.
  if not tensor_util.is_tensor(value) and value == 0:
    return 0
  # If there is only a single value and the reduce op is MEAN,
  # that value should be on all destinations.
  if reduce_op == reduce_util.ReduceOp.MEAN:
    return value

  validate_destinations(destinations)
  # We do not support a reduce op of SUM if the value is the same across
  # all replicas. We call this as part of assign functions for MirroredVariables
  # and summing up identical values across replicas is not clearly defined.
  if device_map.num_replicas_in_graph != 1:
    raise ValueError("A non-DistributedValues value %s cannot be reduced with "
                     "the given reduce op %s." % (value, reduce_op))
  return simple_broadcast(value, destinations)


def _make_tensor_into_per_replica(input_tensor):
  """Converts a single tensor into a PerReplica object."""
  if isinstance(input_tensor, (tuple, list)):
    raise ValueError("Cannot convert `input_tensor` to a `PerReplica` object, "
                     "got %r but expected a object that is not a tuple or list."
                     % (input_tensor,))
  if isinstance(input_tensor, value_lib.PerReplica):
    return input_tensor

  try:
    device = input_tensor.device
  except AttributeError:
    raise ValueError("Cannot convert `input_tensor` to a `PerReplica` object "
                     "because it doesn't have device set.")

  device_map = value_lib.SingleDeviceMap(device)
  return value_lib.PerReplica(device_map, (input_tensor,))


def _normalize_value_destination_pairs(value_destination_pairs):
  """Converts each tensor into a PerReplica object in the input list."""
  result = []

  value_destination_pairs = list(value_destination_pairs)

  if not isinstance(value_destination_pairs, (list, tuple)):
    raise ValueError("`value_destination_pairs` should be a list or tuple")
  for pair in value_destination_pairs:
    if not isinstance(pair, tuple):
      raise ValueError(
          "Each element of `value_destination_pairs` should be a tuple.")
    if len(pair) != 2:
      raise ValueError("Each element of `value_destination_pairs` should be a "
                       "tuple of size 2.")

    per_replica = _make_tensor_into_per_replica(pair[0])
    result.append((per_replica, pair[1]))
  return result


def _validate_value_destination_pairs(value_destination_pairs):
  # TODO(yuefengz): raise exceptions instead of returning False.
  # pylint: disable=g-missing-docstring
  if not value_destination_pairs: return False
  if not isinstance(value_destination_pairs, (list, tuple)): return False
  if not all(isinstance(pair, tuple) for pair in value_destination_pairs):
    return False
  if not all(isinstance(v[0], value_lib.PerReplica)
             for v in value_destination_pairs):
    return False
  return True


# TODO(yuefengz): consider calling this function in the caller of
# CrossDeviceOps.
def get_devices_from(destinations):
  if isinstance(destinations, value_lib.DistributedValues):
    return destinations.devices
  elif isinstance(destinations, value_lib.LogicalDeviceSpec):
    return destinations.device_map.logical_to_actual_devices(
        destinations.logical_device)
  elif isinstance(destinations, six.string_types):
    return (device_util.resolve(destinations),)
  return (device_util.resolve(destinations.device),)


def get_device_map_from(destinations):
  if isinstance(destinations, (value_lib.DistributedValues,
                               value_lib.LogicalDeviceSpec)):
    return destinations.device_map, destinations.logical_device
  if isinstance(destinations, six.string_types):
    device = device_util.resolve(destinations)
  else:
    device = destinations.device
  return value_lib.SingleDeviceMap(device), 0


def _devices_match(left, right):
  return set(get_devices_from(left)) == set(get_devices_from(right))


def _all_devices_match(value_destination_pairs):
  if not all(_devices_match(v, d) for v, d in value_destination_pairs):
    return False
  if not all(_devices_match(v, value_destination_pairs[0][0])
             for v, _ in value_destination_pairs[1:]):
    return False
  return True


def simple_broadcast(value, destinations, always_mirrored=False):
  """Broadcast `value` to `destinations` using simple copies."""
  device_map, logical_device = get_device_map_from(destinations)
  devices = device_map.logical_to_actual_devices(logical_device)
  if len(devices) == 1 and not always_mirrored:
    return cross_device_utils.copy_tensor_or_indexed_slices_to_device(
        value, devices[0])
  else:
    value_updates = []
    for d in devices:
      value_updates.append(
          cross_device_utils.copy_tensor_or_indexed_slices_to_device(
              value, d))
    return value_lib.regroup(
        device_map, value_updates, wrap_class=value_lib.Mirrored)


def _simple_reduce(per_replica_value, reduce_to_device, accumulation_fn,
                   reduce_op):
  # pylint: disable=g-missing-docstring
  all_values = per_replica_value.values
  if not all_values:
    raise ValueError("`per_replica_value` must be non-empty")
  count = len(all_values)

  if (count == 1 and all_values[0].device == reduce_to_device):
    return all_values[0]

  with ops.device(reduce_to_device):
    with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
      reduced = cross_device_utils.aggregate_tensors_or_indexed_slices(
          all_values, accumulation_fn)
      if reduce_op == reduce_util.ReduceOp.MEAN:
        reduced = cross_device_utils.divide_by_n_tensors_or_indexed_slices(
            reduced, count)
      elif reduce_op != reduce_util.ReduceOp.SUM:
        raise ValueError("`reduce_op` must be Reduce.SUM or Reduce.MEAN.")
  return reduced


@tf_export("distribute.CrossDeviceOps")
class CrossDeviceOps(object):
  """Base class for cross-device reduction and broadcasting algorithms."""

  def __init__(self):
    pass

  @property
  def _num_between_graph_workers(self):
    # Returns 1 by default, the value may be overridden by sub classes.
    return 1

  def reduce(self, reduce_op, per_replica_value, destinations):
    """Reduce `per_replica_value` to `destinations`.

    It runs the reduction operation defined by `reduce_op` and put the
    result on `destinations`.

    Args:
      reduce_op: Indicates how per_replica_value will be reduced. Accepted
        values are `tf.distribute.ReduceOp.SUM`, `tf.distribute.ReduceOp.MEAN`.
      per_replica_value: a PerReplica object or a tensor with device set.
      destinations: the reduction destinations.

    Returns:
      a Mirrored object.

    Raises:
      ValueError: if per_replica_value can't be converted to a PerReplica
        object.
    """
    if not isinstance(per_replica_value, value_lib.DistributedValues):
      per_replica_value = _make_tensor_into_per_replica(per_replica_value)

    validate_destinations(destinations)

    # Shortcut if `per_replica_value` only contains one value.
    if self._num_between_graph_workers == 1 and len(
        per_replica_value.values) == 1 and _devices_match(
            per_replica_value, destinations):
      return value_lib.regroup(
          per_replica_value.device_map,
          per_replica_value.values,
          wrap_class=value_lib.Mirrored)

    return self.reduce_implementation(reduce_op, per_replica_value,
                                      destinations)

  def batch_reduce(self, reduce_op, value_destination_pairs):
    """Reduce PerReplica objects in a batch.

    Reduce each first element in `value_destination_pairs` to each second
    element which indicates the destinations.

    Args:
      reduce_op: Indicates how per_replica_value will be reduced. Accepted
        values are `tf.distribute.ReduceOp.SUM`, `tf.distribute.ReduceOp.MEAN`.
      value_destination_pairs: a list or a tuple of tuples of PerReplica objects
        (or tensors with device set if there is one device) and destinations.

    Returns:
      a list of Mirrored objects.

    Raises:
      ValueError: if `value_destination_pairs` is not a list or a tuple of
        tuples of PerReplica objects and destinations
    """
    # TODO(yuefengz): if destinations are different, split into several
    # `_batch_reduce` invocations.
    if not _validate_value_destination_pairs(value_destination_pairs):
      # If the first element of each pair is a tensor, we try to turn it into a
      # PerReplica object.
      value_destination_pairs = _normalize_value_destination_pairs(
          value_destination_pairs)

    for _, d in value_destination_pairs:
      validate_destinations(d)

    # Shortcut all PerReplica objects only contain one value.
    if self._num_between_graph_workers == 1 and _all_devices_match(
        value_destination_pairs) and len(
            value_destination_pairs[0][0].values) == 1:
      return [
          value_lib.regroup(
              v.device_map, v.values, wrap_class=value_lib.Mirrored)
          for v, _ in value_destination_pairs
      ]

    return self.batch_reduce_implementation(reduce_op, value_destination_pairs)

  def broadcast(self, tensor, destinations):
    """Broadcast the `tensor` to destinations.

    Args:
      tensor: the tensor to broadcast.
      destinations: the broadcast destinations.

    Returns:
      a Mirrored object.
    """
    validate_destinations(destinations)
    return self.broadcast_implementation(tensor, destinations)

  @doc_controls.for_subclass_implementers
  def reduce_implementation(self, reduce_op, per_replica_value, destinations):
    """The implementation of reduce of `per_replica_value` to `destinations`.

    It runs the reduction operation defined by `reduce_op` and put the
    result on `destinations`.

    Args:
      reduce_op: Indicates how per_replica_value will be reduced. Accepted
        values are `tf.distribute.ReduceOp.SUM`, `tf.distribute.ReduceOp.MEAN`.
      per_replica_value: a PerReplica object or a tensor with device set.
      destinations: the reduction destinations.

    Returns:
      a Mirrored object.

    Raises:
      ValueError: if per_replica_value can't be converted to a PerReplica
        object.
    """
    raise NotImplementedError(
        "_reduce method must be implemented in descendants.")

  @doc_controls.for_subclass_implementers
  def batch_reduce_implementation(self, reduce_op, value_destination_pairs):
    """Implementation of reduce PerReplica objects in a batch.

    Reduce each first element in `value_destination_pairs` to each second
    element which indicates the destinations.

    Args:
      reduce_op: Indicates how per_replica_value will be reduced. Accepted
        values are `tf.distribute.ReduceOp.SUM`, `tf.distribute.ReduceOp.MEAN`.
      value_destination_pairs: a list or a tuple of tuples of PerReplica objects
        (or tensors with device set if there is one device) and destinations.

    Returns:
      a list of Mirrored objects.

    Raises:
      ValueError: if `value_destination_pairs` is not a list or a tuple of
        tuples of PerReplica objects and destinations
    """
    raise NotImplementedError(
        "_batch_reduce method must be implemented in descendants.")

  @doc_controls.for_subclass_implementers
  def broadcast_implementation(self, tensor, destinations):
    """Implementation of broadcast the `tensor` to destinations.

    Args:
      tensor: the tensor to broadcast.
      destinations: the broadcast destinations.

    Returns:
      a Mirrored object.
    """
    return simple_broadcast(tensor, destinations, always_mirrored=True)


@tf_export("distribute.ReductionToOneDevice")
class ReductionToOneDevice(CrossDeviceOps):
  """Always do reduction to one device first and then do broadcasting.

    Batch reduction is done by reduction on each element one by one.
  """

  def __init__(self, reduce_to_device=None, accumulation_fn=None):
    """Constructor.

    Args:
      reduce_to_device: the intermediate device to reduce to. If None, reduce
        to the first device in `destinations` of the reduce() method.
      accumulation_fn: a function that does accumulation.  If None, then
        `tf.math.add_n` is used.
    """
    self.reduce_to_device = reduce_to_device
    self.accumulation_fn = accumulation_fn or math_ops.add_n
    super(ReductionToOneDevice, self).__init__()

  def reduce_implementation(self, reduce_op, per_replica_value, destinations):
    if check_destinations(destinations):
      devices = get_devices_from(destinations)
    else:
      devices = get_devices_from(per_replica_value)
    reduce_to_device = self.reduce_to_device or devices[0]
    logging.log_first_n(
        logging.INFO,
        "Reduce to %s then broadcast to %r." % (reduce_to_device, devices), 10)
    reduced = _simple_reduce(per_replica_value, reduce_to_device,
                             self.accumulation_fn, reduce_op)
    return self.broadcast(reduced, destinations)

  def batch_reduce_implementation(self, reduce_op, value_destination_pairs):
    return [
        self.reduce_implementation(reduce_op, t, destinations=v)
        for t, v in value_destination_pairs
    ]


def _group_value_by_device(per_replica_values):
  """Group values into sublists by their devices.

  This grouping is needed to call the all-reduce library because it expects a
  list of the following form:
    [[(grad0_gpu0, v0_gpu0), (grad1_gpu0, v1_gpu0), (grad2_gpu0, v2_gpu0) ...],
     [(grad0_gpu1, v0_gpu1), (grad1_gpu1, v1_gpu1), (grad2_gpu1, v2_gpu1) ...],
     [(grad0_gpu2, v0_gpu2), (grad1_gpu0, v1_gpu2), (grad2_gpu0, v2_gpu2) ...],
     ...
    ]

  Args:
    per_replica_values: a list of PerReplica obejcts.

  Returns:
    a list of lists, each sublist has components for its corresponding device of
      PerReplica objects, paired with a None.
  """
  destinations = per_replica_values[0].devices
  grouped = [[] for _ in range(len(destinations))]
  for per_replica_value in per_replica_values:
    # pylint: disable=protected-access
    for i, v in enumerate(per_replica_value.values):
      assert per_replica_value.devices == destinations
      grouped[i].append((v, None))
  return grouped


def _ungroup_and_make_mirrored(grouped_reduced,
                               destinations,
                               reduce_op,
                               num_between_graph_workers=1):
  """Ungroup results from all-reduce and make Mirrored objects.

  Each all-reduce result will be divided by the number of destinations before
  Mirrored objects are created if reduce_op is "mean".

  Args:
    grouped_reduced: a list of lists, each sublist has components for each
      device, paired with a None. It is the result from
      cross_device_utils.aggregate_gradients_using*.
    destinations: a value to colocate the result with.
    reduce_op: Indicates how values will be aggregated. Accepted values
      are `tf.distribute.ReduceOp.SUM`, `tf.distribute.ReduceOp.MEAN`.
    num_between_graph_workers: number of workers in the between-graph
      replication.

  Returns:
    a list of Mirrored objects.
  """
  device_map, _ = get_device_map_from(destinations)
  num_replicas = device_map.num_replicas_in_graph * num_between_graph_workers
  index = [[] for _ in range(len(grouped_reduced[0]))]
  for per_replica_reduced in grouped_reduced:
    for i, (v, _) in enumerate(per_replica_reduced):
      if reduce_op == reduce_util.ReduceOp.MEAN:
        with ops.device(v.device):
          index[i].append(v / num_replicas)
      else:
        index[i].append(v)
  return [
      value_lib.regroup(device_map, v, wrap_class=value_lib.Mirrored)
      for v in index
  ]


class _ConcatAndSplitPacker(object):
  """Concatenate and split tensors for reduction."""

  def __init__(self, num_packs=1):
    """Initialize the _ConcatAndSplitPacker object.

    Args:
      num_packs: specifies the number of split packs that will be
        formed.

    Raises:
      ValueError: if num_packs is not greater than 0.
    """
    if num_packs <= 0:
      raise ValueError("num_packs must be greater than zero.")
    self.num_packs = num_packs

  def pack(self, grouped_grads_and_vars):
    """Pack tensors."""
    self.grouped_grads_and_vars = grouped_grads_and_vars
    self.all_device_shapes = []
    self.all_device_sizes = []

    device_grad_packs = []
    for device_grads_and_vars in grouped_grads_and_vars:
      with ops.colocate_with(device_grads_and_vars[0][0]):
        # Flatten all the grads.
        flat_grads = [
            array_ops.reshape(g, [-1]) for g, _ in device_grads_and_vars
        ]
        # Remember the original shape of all the grads.
        device_shapes = [array_ops.shape(g) for g, _ in device_grads_and_vars]
        # Remember the original sizes of all the grads.
        device_sizes = [array_ops.size(g) for g, _ in device_grads_and_vars]
        # Concat all the flat grads into a big flat tensor.
        concat_grads = array_ops.concat(flat_grads, 0)

        # Split the big tensor into num_splits packs. In cases where the
        # total size is not divisible num_splits, the last pack gets
        # more elements.
        # TODO(zhengxq): it is also possible to optimize away all the concat
        # as well.
        num_splits = self.num_packs

        # The array_ops.size function will sometimes remove static shapes. So if
        # all gradient shapes are defined, we use another method to get the
        # total size.
        # TODO(yuefengz): move this logic to array_ops.size.
        if all(g.shape.is_fully_defined() for g, _ in device_grads_and_vars):
          total_grad_size = sum(
              [g.shape.num_elements() for g, _ in device_grads_and_vars])
        else:
          total_grad_size = array_ops.size(concat_grads)

        split_size = total_grad_size // num_splits
        split_size_last = total_grad_size - split_size * (num_splits - 1)
        split_sizes = [split_size] * (num_splits - 1) + [split_size_last]
        grad_packs = array_ops.split(concat_grads, split_sizes)

        # Ready to aggregate the repacked gradients, with fake variables.
        # TODO(zhengxq): It is hacky to have to use fake variables.
        # We should remove the need for variables in
        # aggregate_gradients_using*.
        device_grad_packs.append(zip(grad_packs, [None] * num_splits))
        self.all_device_shapes.append(device_shapes)
        self.all_device_sizes.append(device_sizes)

    return device_grad_packs

  def unpack(self, summed_device_grad_packs):
    """Reverse the pack."""
    aggregated_device_grads = []
    for (summed_device_grad_packs,
         device_grads_and_vars, device_shapes, device_sizes) in zip(
             summed_device_grad_packs, self.grouped_grads_and_vars,
             self.all_device_shapes, self.all_device_sizes):
      # pylint: enable=line-too-long
      # Reverse the packing operations in the previous steps. Form the
      # summed gradients back into their original shapes.
      with ops.colocate_with(summed_device_grad_packs[0][0]):
        # Form a list of the summed grad packs.
        device_grad_packs = [g for g, _ in summed_device_grad_packs]

        # Concat them back into a big flat tensor.
        device_grads_concat = array_ops.concat(device_grad_packs, 0)

        # Split the tensors back into their original sizes.
        grads_with_sizes = array_ops.split(device_grads_concat, device_sizes)

        # Reshape the tensors back into their original shapes.
        grads_with_shapes = [
            array_ops.reshape(grad, shape)
            for shape, grad in zip(device_shapes, grads_with_sizes)
        ]

        # Form the list with the original list of variables.
        summed_device_grads = [
            (g, v) for g, (_, v) in zip(grads_with_shapes,
                                        device_grads_and_vars)
        ]
        aggregated_device_grads.append(summed_device_grads)
    return aggregated_device_grads


class _AggregateSmallTensorPacker(object):
  """Concatenate small gradient tensors together for reduction."""

  def __init__(self,
               agg_small_grads_max_bytes=1048576,
               agg_small_grads_max_group=16):
    """Initialize the _AggregateSmallTensorPacker object.

    Args:
      agg_small_grads_max_bytes: largest tensor eligible for aggregation,
        in number of bytes.
      agg_small_grads_max_group: largest permitted aggregation of small
        tensors.

    Raises:
      ValueError: if `agg_small_grads_max_bytes` or `agg_small_grads_max_group`
        is not greater than 0.
    """
    if agg_small_grads_max_bytes <= 0 or agg_small_grads_max_group <= 0:
      raise ValueError("agg_small_grads_max_bytes and agg_small_grads_max_group"
                       " should both be greater than zero.")
    self.agg_small_grads_max_bytes = agg_small_grads_max_bytes
    self.agg_small_grads_max_group = agg_small_grads_max_group

  def pack(self, grouped_grads_and_vars):
    """Aggregate small tensors."""
    if (self.agg_small_grads_max_bytes > 0 and
        self.agg_small_grads_max_group > 0):
      device_grads, self.packing = cross_device_utils.pack_small_tensors(
          grouped_grads_and_vars,
          max_bytes=self.agg_small_grads_max_bytes,
          max_group=self.agg_small_grads_max_group)
    return device_grads

  def unpack(self, summed_device_grad_packs):
    """Reverse the aggregation process."""
    return cross_device_utils.unpack_small_tensors(summed_device_grad_packs,
                                                   self.packing)


def _pack_tensors(device_grads,
                  num_packs=0,
                  agg_small_grads_max_bytes=0,
                  agg_small_grads_max_group=0):
  """Pack tensors if specified."""
  if num_packs > 0:
    tensor_packer = _ConcatAndSplitPacker(num_packs)
    device_grad_packs = tensor_packer.pack(device_grads)
  elif agg_small_grads_max_bytes > 0 and agg_small_grads_max_group > 0:
    tensor_packer = _AggregateSmallTensorPacker(agg_small_grads_max_bytes,
                                                agg_small_grads_max_group)
    device_grad_packs = tensor_packer.pack(device_grads)
  else:
    tensor_packer = None
    device_grad_packs = device_grads
  return device_grad_packs, tensor_packer


def _unpack_tensors(reduced, tensor_packer=None):
  """Unpack tensors if they are packed before all-reduce."""
  if tensor_packer:
    return tensor_packer.unpack(reduced)
  return reduced


class AllReduceCrossDeviceOps(CrossDeviceOps):
  """Reduction using all-reduce."""

  def __init__(self,
               all_reduce_alg="nccl",
               num_packs=1,
               agg_small_grads_max_bytes=0,
               agg_small_grads_max_group=10):
    """All-reduce implementation of CrossDeviceOps.

    Before performing all-reduce, tensors will be repacked or aggregated for
    more efficient cross-device transportation:
      1) If `num_packs` is non-zero, pack values into
        `num_packs` splits.
      2) Otherwise, if `agg_small_grads_max_bytes` > 0 and
        `agg_small_grads_max_group` > 0, aggregate values smaller than
        `agg_small_grads_max_bytes` into groups with at most
        `agg_small_grads_max_group` values.
      3) Otherwise, no repacking or grouping will happen.

    Args:
      all_reduce_alg: the all-reduce algorithm to use, currently only "nccl" or
        "hierarchical_copy" are supported.
      num_packs: see above.
      agg_small_grads_max_bytes: see above.
      agg_small_grads_max_group: see above.
    """
    self._all_reduce_alg = all_reduce_alg
    self._num_packs = num_packs
    self._agg_small_grads_max_bytes = agg_small_grads_max_bytes
    self._agg_small_grads_max_group = agg_small_grads_max_group
    self._simple_cross_replica_ops = ReductionToOneDevice()
    super(AllReduceCrossDeviceOps, self).__init__()

  def reduce_implementation(self, reduce_op, per_replica_value, destinations):
    if _devices_match(per_replica_value, destinations):
      return self._batch_all_reduce(reduce_op, [per_replica_value])[0]
    else:
      return self._simple_cross_replica_ops.reduce(reduce_op, per_replica_value,
                                                   destinations)

  def batch_reduce_implementation(self, reduce_op, value_destination_pairs):
    if _all_devices_match(value_destination_pairs):
      return self._batch_all_reduce(reduce_op,
                                    [v[0] for v in value_destination_pairs])
    else:
      return [
          self.reduce_implementation(reduce_op, t, destinations=v)
          for t, v in value_destination_pairs
      ]

  def _batch_all_reduce(self, reduce_op, per_replica_values):
    """All-reduce algorithm in a batch."""
    dense_values, dense_indices, sparse_values, sparse_indices = (
        cross_device_utils.split_by_sparsity(per_replica_values))
    if dense_values:
      dense_results = self._do_batch_all_reduce(reduce_op, dense_values)
    else:
      dense_results = []
    if sparse_values:
      sparse_results = self._do_batch_all_reduce_sparse(reduce_op,
                                                        sparse_values)
    else:
      sparse_results = []
    return cross_device_utils.stitch_values(((dense_results, dense_indices),
                                             (sparse_results, sparse_indices)))

  def _do_batch_all_reduce(self, reduce_op, dense_values):
    """Run batch all-reduces."""
    logging.log_first_n(
        logging.INFO, "batch_all_reduce: %d all-reduces with algorithm = %s, "
        "num_packs = %d, agg_small_grads_max_bytes = %d and "
        "agg_small_grads_max_group = %d" %
        (len(dense_values), self._all_reduce_alg, self._num_packs,
         self._agg_small_grads_max_bytes, self._agg_small_grads_max_group), 10)

    destinations = dense_values[0].devices
    grouped = _group_value_by_device(dense_values)

    device_grad_packs, tensor_packer = _pack_tensors(
        grouped, self._num_packs, self._agg_small_grads_max_bytes,
        self._agg_small_grads_max_group)

    # The actual aggregation of the repacked gradients. Note that they are
    # sharded among different aggregation trees. So it is important to strike
    # the balance on num_splits.
    if self._all_reduce_alg == "nccl":
      # TODO(yuefengz): merge this into the all-reduce library.
      reduced = cross_device_utils.aggregate_gradients_using_nccl(
          device_grad_packs)
    else:
      # TODO(yuefengz): check that gpu ids in `destinations` are in ascending
      # order.
      reduced = (
          cross_device_utils.aggregate_gradients_using_hierarchical_copy(
              destinations, device_grad_packs))

    reduced = _unpack_tensors(reduced, tensor_packer)
    return _ungroup_and_make_mirrored(reduced, dense_values[0], reduce_op)

  def _do_batch_all_reduce_sparse(self, reduce_op, sparse_values):
    """Run batch all-reduce for sparse values."""
    logging.log_first_n(
        logging.WARN,
        "Efficient allreduce is not supported for %d IndexedSlices" %
        len(sparse_values), 10)
    # Use `sparse_values` as destinations to do all-reduces. It is effectively
    # an allgather under the hood but not an efficient one.
    return self._simple_cross_replica_ops.batch_reduce(
        reduce_op, zip(sparse_values, sparse_values))


# For compatibility with code using the old name of `AllReduceCrossDeviceOps`.
AllReduceCrossTowerOps = AllReduceCrossDeviceOps


AllReduceSpecTuple = collections.namedtuple("AllReduceSpecTuple",
                                            "alg shards limit")


@tf_export("distribute.NcclAllReduce")
class NcclAllReduce(AllReduceCrossDeviceOps):
  """Reduction using NCCL all-reduce."""

  def __init__(self, num_packs=1):
    """NCCL all-reduce implementation of CrossDeviceOps.

    It uses Nvidia NCCL for all-reduce. Before performing all-reduce, tensors
    will be repacked or aggregated for more efficient cross-device
    transportation.

    Args:
      num_packs: values will be packed in this many splits.  `num_packs` should
        be greater than or equals 0. When it is zero, no packing will be done.

    Raises:
      ValueError if `num_packs` is negative.
    """
    if num_packs < 0:
      raise ValueError(
          "NCCL all-reduce requires num_packs >= 0, but {} is specified".format(
              num_packs))
    super(NcclAllReduce, self).__init__(
        all_reduce_alg="nccl", num_packs=num_packs)


@tf_export("distribute.HierarchicalCopyAllReduce")
class HierarchicalCopyAllReduce(AllReduceCrossDeviceOps):
  """Reduction using hierarchical copy all-reduce.

  It reduces to one GPU along edges in some hierarchy and broadcasts back to
  each GPU along the same path. Before performing all-reduce, tensors will be
  repacked or aggregated for more efficient cross-device transportation.

  This is a reduction created for Nvidia DGX-1 which assumes GPUs connects like
  that on DGX-1 machine. If you have different GPU inter-connections, it is
  likely that it would be slower than `tf.distribute.ReductionToOneDevice`.
  """

  def __init__(self, num_packs=1):
    """Initializes the object.

    Args:
      num_packs: values will be packed in this many splits.  `num_packs` should
        be greater than or equals 0. When it is zero, no packing will be done.

    Raises:
      ValueError if `num_packs` is negative.
    """
    if num_packs < 0:
      raise ValueError(
          "HierarchicalCopy requires num_packs >= 0, but {} is specified"
          .format(num_packs))
    super(HierarchicalCopyAllReduce, self).__init__(
        all_reduce_alg="hierarchical_copy",
        num_packs=num_packs)


class MultiWorkerAllReduce(AllReduceCrossDeviceOps):
  """All-reduce algorithms for distributed TensorFlow."""

  def __init__(self,
               worker_devices,
               num_gpus_per_worker,
               all_reduce_spec=("pscpu/pscpu", 2, -1),
               num_packs=0,
               agg_small_grads_max_bytes=0,
               agg_small_grads_max_group=10):
    """Initialize the all-reduce algorithm.

    Args:
      worker_devices: a list of device strings for workers participating in
        all-reduce.
      num_gpus_per_worker: number of GPU devices per worker.
      all_reduce_spec: a tuple or a named tuple or a list of tuples specifying
        the all-reduce algorithm.
        1. The first element of a tuple is the name of the all-reduce algorithm.
        Valid algorithm names are: "nccl", "nccl/xring", "nccl/rechd",
        "nccl/pscpu", "xring", "pscpu", "psgpu", "pscpu/pscpu". Algorithms with
        a "/" are hierarchical, so two all-reduces are executed, the first one
        aggregates tensors within a worker and the second aggregates across
        workers.
        2. The second element of a tuple is the number of shards when doing
        all-reduce. Let's say its values is M, each tensor after packing will be
        split into M shards and then M parallel all-reduces would be performed
        before finally they are concatenated backed into a complete tensor.
        3. The third element is the maximum size of tensors that will be
        applicable for the algorithm specified by the first element. For
        example, if all_reduce_spec=[("nccl", 2, 1024), ("pscpu/pscpu", 2, -1)],
        tensors with size not larger than 1024 bytes will be applied a 2-shard
        "nccl" all-reduce and other tensors will be applied a 2-shard
        "pscpu/pscpu" algorithm. The third elements should be in increasing
        order across tuples and end with -1 which indicates infinity.
      num_packs: see AllReduceCrossDeviceOps.
      agg_small_grads_max_bytes: see AllReduceCrossDeviceOps.
      agg_small_grads_max_group: see AllReduceCrossDeviceOps.
    """
    self._worker_devices = worker_devices
    self._num_gpus_per_worker = num_gpus_per_worker
    super(MultiWorkerAllReduce, self).__init__(
        num_packs=num_packs,
        agg_small_grads_max_bytes=agg_small_grads_max_bytes,
        agg_small_grads_max_group=agg_small_grads_max_group)

    def validate_and_complete_spec(spec):
      """Validate and complete the all-reduce spec."""
      # TODO(yuefengz): support namedtuple.
      if not isinstance(spec, tuple):
        raise ValueError(
            "A tuple is expected for all-reduce spec: %r" % all_reduce_spec)
      if not spec or len(spec) > 3:
        raise ValueError(
            "Too many elements in the all-reduce spec tuple: %r" % spec)
      if len(spec) == 1:
        return AllReduceSpecTuple(spec[0], 1, -1)
      elif len(spec) == 2:
        return AllReduceSpecTuple(spec[0], spec[1], -1)
      else:
        return AllReduceSpecTuple(*spec)

    self._all_reduce_spec = []
    if isinstance(all_reduce_spec, six.string_types):
      self._all_reduce_spec.append(AllReduceSpecTuple(all_reduce_spec, 1, -1))
    elif isinstance(all_reduce_spec, tuple):
      self._all_reduce_spec.append(validate_and_complete_spec(all_reduce_spec))
    elif isinstance(all_reduce_spec, list):
      self._all_reduce_spec = [
          validate_and_complete_spec(spec) for spec in all_reduce_spec
      ]

  def _batch_all_reduce(self, reduce_op, per_replica_values):
    """All-reduce algorithm in a batch."""
    logging.log_first_n(
        logging.INFO,
        "Distributed batch_all_reduce: %d all-reduces with "
        "allreduce_spec = %r, num_packs = %d, agg_small_grads_max_bytes = %d, "
        "and agg_small_grads_max_group = %d" %
        (len(per_replica_values), self._all_reduce_spec, self._num_packs,
         self._agg_small_grads_max_bytes, self._agg_small_grads_max_group), 10)

    device_grads = _group_value_by_device(per_replica_values)

    # The all-reduce library requires fully defined shapes.
    # TODO(yuefengz): when tensor sharding is not needed, static shapes are not
    # required as well.
    for device_grad in device_grads:
      for grad, _ in device_grad:
        if not grad.shape.is_fully_defined():
          raise ValueError("Shape is unknown for node %r" % grad)

    remaining_grads = device_grads
    aggregated_grads = []
    for spec_tuple in self._all_reduce_spec:
      if spec_tuple.limit < 0:
        this_grads = remaining_grads
        remaining_grads = []
      else:
        (this_grads, remaining_grads) = cross_device_utils.split_grads_by_size(
            spec_tuple.limit, remaining_grads)
      if this_grads:
        device_grad_packs, tensor_packer = _pack_tensors(
            this_grads, self._num_packs, self._agg_small_grads_max_bytes,
            self._agg_small_grads_max_group)
        range_agg_grads = cross_device_utils.sum_gradients_all_reduce(
            self._worker_devices, device_grad_packs, len(self._worker_devices),
            spec_tuple.alg, spec_tuple.shards, range(self._num_gpus_per_worker))
        range_agg_grads = _unpack_tensors(range_agg_grads, tensor_packer)

        if not aggregated_grads:
          aggregated_grads = range_agg_grads
        else:
          assert len(aggregated_grads) == len(range_agg_grads)
          for i, range_agg_grad in enumerate(range_agg_grads):
            aggregated_grads[i] += range_agg_grad
    assert not remaining_grads

    return _ungroup_and_make_mirrored(aggregated_grads, per_replica_values[0],
                                      reduce_op)


@tf_export("distribute.experimental.CollectiveCommunication")
class CollectiveCommunication(enum.Enum):
  """Communication choices for CollectiveOps.

  * `AUTO`: Default to runtime's automatic choices.
  * `RING`: TensorFlow's ring algorithms for all-reduce and
    all-gather.
  * `NCCL`: Use ncclAllReduce for all-reduce, and ring algorithms for
    all-gather.
  """
  AUTO = "AUTO"
  RING = "RING"
  NCCL = "NCCL"
  # TODO(ayushd): add ncclAllGather implementation.


# TODO(yuefengz): support in-graph collective all-reduce.
class CollectiveAllReduce(CrossDeviceOps):
  """All-reduce cross device ops using collective ops.

  In the between-graph replicated training, it will still do all-reduces across
  all workers and then put results on the right destinations.
  """

  def __init__(self,
               num_workers=1,
               num_gpus_per_worker=0,
               all_reduce_merge_scope=32,
               collective_keys=None):
    """Initializes the object.

    Args:
      num_workers: number of workers in the between-graph replicated training.
      num_gpus_per_worker: number of GPUs per worker.
      all_reduce_merge_scope: size of groups into which to partition consecutive
        gradients grouped under a common 'allreduce' name scope. This is useful
        for some optimization of collective ops.
      collective_keys: an optional CollectiveKey object.
    """
    self._num_workers = num_workers
    self._num_gpus_per_worker = num_gpus_per_worker
    self._all_reduce_merge_scope = all_reduce_merge_scope
    self._collective_keys = (collective_keys or
                             cross_device_utils.CollectiveKeys())
    super(CollectiveAllReduce, self).__init__()

  @property
  def _num_between_graph_workers(self):
    return self._num_workers

  def reduce_implementation(self, reduce_op, per_replica_value, destinations):
    all_reduced = self._batch_all_reduce(reduce_op, [per_replica_value])[0]
    device_map, logical_device = get_device_map_from(destinations)
    devices = device_map.logical_to_actual_devices(logical_device)

    if (isinstance(all_reduced, value_lib.Mirrored) and
        all_reduced.device_map is device_map and
        all_reduced.logical_device == logical_device):
      return all_reduced

    # Convert `all_reduced` to a `Mirrored` object, as a simple and uniform
    # utility to access component for a particular device.
    if not isinstance(all_reduced, value_lib.Mirrored):
      all_reduced = value_lib.Mirrored(
          value_lib.SingleDeviceMap(all_reduced.device), [all_reduced])

    index = []
    with ops.control_dependencies(all_reduced.values):
      for d in devices:
        with ops.device(d):
          if d in all_reduced.devices:
            index.append(array_ops.identity(all_reduced.get(d)))
          else:
            # TODO(josh11b): Once we add support for model parallelism, get the
            # copy from the corresponding replica instead of the primary.
            index.append(array_ops.identity(all_reduced.primary))

    return value_lib.regroup(device_map, index, wrap_class=value_lib.Mirrored)

  def batch_reduce_implementation(self, reduce_op, value_destination_pairs):
    all_devices_match = _all_devices_match(value_destination_pairs)
    if all_devices_match:
      return self._batch_all_reduce(reduce_op,
                                    [v[0] for v in value_destination_pairs])
    else:
      if not all_devices_match:
        logging.log_first_n(
            logging.WARN, "Efficient batch_reduce is not supported if "
            "destinations are different.", 10)

      return [
          self.reduce_implementation(reduce_op, t, destinations=v)
          for t, v in value_destination_pairs
      ]

  def _make_gradient_chunks(self, per_replica_values, all_reduce_merge_scope):
    """Make `per_replica_values` into chunks."""
    grouped_by_device = _group_value_by_device(per_replica_values)

    grouped_by_var = list(zip(*grouped_by_device))
    # grouped_by_var is grouped by variables and takes the following format:
    # [((grad0_gpu0, v0_gpu0), (grad0_gpu1, v0_gpu1), (grad0_gpu2, v0_gpu2) ..),
    #  ((grad1_gpu0, v1_gpu0), (grad1_gpu1, v1_gpu1), (grad1_gpu0, v1_gpu2) ..),
    #  ((grad2_gpu0, v2_gpu0), (grad2_gpu1, v2_gpu1), (grad2_gpu0, v2_gpu2) ..),
    #  ...
    # ]
    chunked_gv = [
        grouped_by_var[x:x + all_reduce_merge_scope]
        for x in range(0, len(grouped_by_var), all_reduce_merge_scope)
    ]
    return chunked_gv

  def _batch_all_reduce(self, reduce_op, per_replica_values):
    """All reduce algorithm in a batch."""
    dense_values, dense_indices, sparse_values, sparse_indices = (
        cross_device_utils.split_by_sparsity(per_replica_values))
    if dense_values:
      dense_results = self._do_batch_all_reduce_dense(reduce_op, dense_values)
    else:
      dense_results = []
    if sparse_values:
      sparse_results = self._do_batch_all_reduce_sparse(reduce_op,
                                                        sparse_values)
    else:
      sparse_results = []
    return cross_device_utils.stitch_values(((dense_results, dense_indices),
                                             (sparse_results, sparse_indices)))

  def _do_batch_all_reduce_dense(self, reduce_op, per_replica_values):
    """All-reduce across all workers in a batch."""

    logging.log_first_n(
        logging.INFO, "Collective batch_all_reduce: %d all-reduces, "
        "num_workers = %d" % (len(per_replica_values), self._num_workers), 10)

    chunked_gv = self._make_gradient_chunks(per_replica_values,
                                            self._all_reduce_merge_scope)

    reduced_gv_list = []
    for chunk in chunked_gv:
      with ops.name_scope("allreduce"):
        for grad_and_vars in chunk:
          # Gradients for the same variable but from different devices.
          scaled_grads = [g for g, _ in grad_and_vars]
          collective_reduced = cross_device_utils.build_collective_reduce(
              scaled_grads, self._num_workers, self._collective_keys, "Add",
              "Id")
          result = []
          for (_, v), g in zip(grad_and_vars, collective_reduced):
            result.append([g, v])
          reduced_gv_list.append(result)

    new_device_grads = [list(x) for x in zip(*reduced_gv_list)]
    return _ungroup_and_make_mirrored(
        new_device_grads,
        per_replica_values[0],
        reduce_op,
        num_between_graph_workers=self._num_workers)

  def _do_batch_all_reduce_sparse(self, reduce_op, per_replica_values):
    """All-reduce IndexedSlices across all workers in a batch."""

    logging.log_first_n(
        logging.INFO, "Collective batch_all_reduce for IndexedSlices: "
        "%d all-reduces, num_workers = %d" %
        (len(per_replica_values), self._num_workers), 10)

    chunked_gv = self._make_gradient_chunks(per_replica_values,
                                            self._all_reduce_merge_scope)

    reduced_gv_list = []
    for chunk in chunked_gv:
      with ops.name_scope("allreduce"):
        for grad_and_vars in chunk:
          # Gradients for the same variable but from different devices.
          scaled_grads = [g for g, _ in grad_and_vars]

          values = [g.values for g in scaled_grads]
          indices = [g.indices for g in scaled_grads]
          assert len(values) == len(indices)

          # Build two separate allgathers, one for values, the other one for
          # indices.
          gathered_values = cross_device_utils.build_collective_gather(
              values, self._num_workers, self._collective_keys)
          gathered_indices = cross_device_utils.build_collective_gather(
              indices, self._num_workers, self._collective_keys)
          assert len(gathered_values) == len(gathered_indices)

          collective_reduced = []
          for i in range(len(values)):
            reduced = ops.IndexedSlices(
                gathered_values[i],
                gathered_indices[i],
                dense_shape=scaled_grads[i].dense_shape)
            collective_reduced.append(reduced)

          result = []
          for (_, v), g in zip(grad_and_vars, collective_reduced):
            result.append([g, v])
          reduced_gv_list.append(result)

    new_device_grads = [list(x) for x in zip(*reduced_gv_list)]
    return _ungroup_and_make_mirrored(
        new_device_grads,
        per_replica_values[0],
        reduce_op,
        num_between_graph_workers=self._num_workers)


def choose_the_best(devices, session_config=None):
  """Find the best CrossDeviceOps locally given a `tf.compat.v1.ConfigProto`.

  Args:
    devices: a list of devices passed to `tf.distribute.Strategy`.
    session_config: a `tf.compat.v1.ConfigProto` or `None`. If `None`, it will
      make decision based on all local devices.

  Returns:
    A subclass of `CrossDeviceOps`.
  """
  requested_devices = set([device_util.canonicalize(d) for d in devices])
  machine_devices = device_lib.list_local_devices(session_config=session_config)
  using_devices = set()
  for d in machine_devices:
    if device_util.canonicalize(d.name) in requested_devices:
      using_devices.add(d.name)

  if len(using_devices) != len(requested_devices):
    logging.warning(
        "Some requested devices in `tf.distribute.Strategy` are not visible "
        "to TensorFlow: %s", ",".join(list(requested_devices - using_devices)))
    return ReductionToOneDevice()

  if any("gpu" not in d.lower() for d in using_devices):
    logging.warning("There is non-GPU devices in `tf.distribute.Strategy`, not "
                    "using nccl allreduce.")
    return ReductionToOneDevice()

  if kernels.get_registered_kernels_for_op("NcclAllReduce"):
    return NcclAllReduce(num_packs=1)
  else:
    logging.warning("Nccl kernel is not found, not using nccl allreduce.")
    return ReductionToOneDevice()

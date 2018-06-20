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
import six

from tensorflow.contrib.distribute.python import cross_tower_utils
from tensorflow.contrib.distribute.python import values as value_lib
from tensorflow.python.client import device_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import device_util


def _validate_destinations(destinations):
  if not isinstance(destinations,
                    (value_lib.DistributedValues, six.string_types, list)):
    raise ValueError("destinations must be one of a `DistributedValues` object,"
                     " a device string, a list of device strings or None")

  if not destinations:
    raise ValueError("destinations can not be empty")


def _validate_value_destination_pairs(value_destination_pairs):
  # pylint: disable=g-missing-docstring
  if not value_destination_pairs: return False
  if not isinstance(value_destination_pairs, (list, tuple)): return False
  if not all([isinstance(pair, tuple) for pair in value_destination_pairs]):
    return False
  if not all([isinstance(v[0], value_lib.PerDevice)
              for v in value_destination_pairs]):
    return False
  return True


# TODO(yuefengz): consider calling this function in the caller of CrossTowerOps.
def _get_devices_from(destinations):
  if isinstance(destinations, value_lib.DistributedValues):
    return list(destinations.devices)
  elif isinstance(destinations, six.string_types):
    return [device_util.resolve(destinations)]
  else:
    return [device_util.resolve(destination) for destination in destinations]


def _devices_match(left, right):
  return set(_get_devices_from(left)) == set(_get_devices_from(right))


def _all_devices_match(value_destination_pairs):
  if not all([d is None or _devices_match(v, d)
              for v, d in value_destination_pairs]):
    return False
  if not all([_devices_match(v, value_destination_pairs[0][0])
              for v, _ in value_destination_pairs[1:]]):
    return False
  return True


def _simple_broadcast(value, destinations):
  index = {}
  devices = _get_devices_from(destinations)
  for d in devices:
    index[d] = cross_tower_utils.copy_tensor_or_indexed_slices_to_device(
        value, d)
  return value_lib.Mirrored(index)


def _simple_reduce(per_device_value, reduce_to_device, accumulation_fn,
                   method_string):
  # pylint: disable=g-missing-docstring
  all_values = []
  count = 0
  for v in per_device_value._index.values():  # pylint: disable=protected-access
    if isinstance(v, value_lib.MapOutput):
      v_list = v.get()
      if not v_list:
        continue
      count += len(v_list)
      # Sum within each device before aggregating across devices.
      # TODO(yuefengz): Check whether it helps to use accumulation_fn here.
      v = cross_tower_utils.aggregate_tensors_or_indexed_slices(
          v_list, math_ops.add_n)
    else:
      count += 1
    all_values.append(v)
  if not all_values:
    raise ValueError("`per_device_value` must be non-empty")

  with ops.device(reduce_to_device):
    with context.context().device_policy(context.DEVICE_PLACEMENT_SILENT):
      reduced = cross_tower_utils.aggregate_tensors_or_indexed_slices(
          all_values, accumulation_fn)
      if method_string == "mean":
        reduced = cross_tower_utils.divide_by_n_tensors_or_indexed_slices(
            reduced, count)
      elif method_string != "sum":
        raise ValueError("`method_string` must be 'sum' or 'mean'")
  return reduced


class CrossTowerOps(object):
  """Base class for cross-tower reduction and broadcasting algorithms."""

  def __init__(self):
    pass

  def reduce(self, method_string, per_device_value, destinations=None):
    """Reduce `per_device_value` to `destinations`.

    It runs the reduction operation defined by `method_string` and put the
    result on `destinations`.

    Args:
      method_string: either 'sum' or 'mean' specifying the reduction method.
      per_device_value: a PerDevice object.
      destinations: the reduction destinations.

    Returns:
      a Mirrored object.

    Raises:
      ValueError: if per_device_value is not a PerDevice object.
    """
    if not isinstance(per_device_value, value_lib.PerDevice):
      raise ValueError("`per_device_value` must be a `PerDevice` object.")
    if destinations is not None:
      _validate_destinations(destinations)
    return self._reduce(method_string, per_device_value, destinations)

  def batch_reduce(self, method_string, value_destination_pairs):
    """Reduce PerDevice objects in a batch.

    Reduce each first element in `value_destination_pairs` to each second
    element which indicates the destinations.

    Args:
      method_string: either 'sum' or 'mean' specifying the reduction method.
      value_destination_pairs: a list or a tuple of tuples of PerDevice objects
        and destinations. If a destination is None, then the destinations
        are set to match the devices of the input PerDevice object.

    Returns:
      a list of Mirrored objects.

    Raises:
      ValueError: if `value_destination_pairs` is not a list or a tuple of
        tuples of PerDevice objects and destinations
    """
    if not _validate_value_destination_pairs(value_destination_pairs):
      raise ValueError("`value_destination_pairs` must be a list or a tuple of "
                       "tuples of PerDevice objects and destinations")
    for _, d in value_destination_pairs:
      if d is not None:
        _validate_destinations(d)

    return self._batch_reduce(method_string, value_destination_pairs)

  def broadcast(self, tensor, destinations):
    """Broadcast the `tensor` to destinations.

    Args:
      tensor: the tensor to broadcast.
      destinations: the broadcast destinations.

    Returns:
      a Mirrored object.
    """
    _validate_destinations(destinations)
    return self._broadcast(tensor, destinations)

  def _reduce(self, method_string, per_device_value, destinations):
    raise NotImplementedError(
        "_reduce method must be implemented in descendants.")

  def _batch_reduce(self, method_string, value_destination_pairs):
    raise NotImplementedError(
        "_batch_reduce method must be implemented in descendants.")

  def _broadcast(self, tensor, destinations):
    return _simple_broadcast(tensor, destinations)


class ReductionToOneDeviceCrossTowerOps(CrossTowerOps):
  """Always do reduction to one device first and then do broadcasting.

    Batch reduction is done by reduction on each element one by one.
  """

  def __init__(self, reduce_to_device=None, accumulation_fn=math_ops.add_n):
    """Constructor.

    Args:
      reduce_to_device: the intermediate device to reduce to. If None, reduce
        to the first device in `destinations` of the reduce() method.
      accumulation_fn: a function that does accumulation.
    """
    self.reduce_to_device = reduce_to_device
    self.accumulation_fn = accumulation_fn
    super(ReductionToOneDeviceCrossTowerOps, self).__init__()

  def _reduce(self, method_string, per_device_value, destinations):
    devices = _get_devices_from(destinations or per_device_value)
    reduce_to_device = self.reduce_to_device or devices[0]
    reduced = _simple_reduce(per_device_value, reduce_to_device,
                             self.accumulation_fn, method_string)
    return self.broadcast(reduced, devices)

  def _batch_reduce(self, method_string, value_destination_pairs):
    return [self._reduce(method_string, t, destinations=v)
            for t, v in value_destination_pairs]


def _group_value_by_device(per_device_values):
  """Group values into sublists by their devices.

  This grouping is needed to call the all-reduce library because it expects a
  list of the following form:
    [(grad0_gpu0, v0_gpu0), (grad1_gpu0, v1_gpu0), (grad2_gpu0, v2_gpu0) ...
     (grad0_gpu1, v0_gpu1), (grad1_gpu1, v1_gpu1), (grad2_gpu1, v2_gpu1) ...
     (grad0_gpu2, v0_gpu2), (grad1_gpu0, v1_gpu2), (grad2_gpu0, v2_gpu2) ...
     ...
    ]

  Args:
    per_device_values: a list of PerDevice obejcts.

  Returns:
    a list of lists, each sublist has components for its corresponding device of
      PerDevice objects, paired with a None.
  """
  destinations = per_device_values[0].devices
  grouped = [[] for _ in range(len(destinations))]
  for per_device_value in per_device_values:
    # pylint: disable=protected-access
    for i, v in enumerate(per_device_value._index.values()):
      assert per_device_value.devices == destinations
      grouped[i].append((v, None))
  return grouped


def _ungroup_and_make_mirrored(grouped_reduced, destinations, method_string):
  """Ungroup results from all-reduce and make Mirrored objects.

  Each all-reduce result will be divided by the number of destinations before
  Mirrored objects are created if method_string is "mean".

  Args:
    grouped_reduced: a list of lists, each sublist has components for each
      device, paired with a None. It is the result from
      cross_tower_utils.aggregate_gradients_using*.
    destinations: a list of device strings for returned Mirrored objects.
    method_string: "mean" or "sum".

  Returns:
    a list of Mirrored objects.
  """
  index = [{} for _ in range(len(grouped_reduced[0]))]
  for d, per_device_reduced in enumerate(grouped_reduced):
    for i, (v, _) in enumerate(per_device_reduced):
      if method_string == "mean":
        index[i][destinations[d]] = v / len(destinations)
      else:
        index[i][destinations[d]] = v
  return [value_lib.Mirrored(v) for v in index]


class ConcatAndSplitPacker(object):
  """Concatenate and split tensors for reduction."""

  def __init__(self, num_packs=1):
    """Initialize the ConcatAndSplitPacker object.

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
    self.all_tower_shapes = []
    self.all_tower_sizes = []

    device_grad_packs = []
    for tower_grads_and_vars in grouped_grads_and_vars:
      with ops.colocate_with(tower_grads_and_vars[0][0]):
        # Flatten all the grads.
        flat_grads = [
            array_ops.reshape(g, [-1]) for g, _ in tower_grads_and_vars
        ]
        # Remember the original shape of all the grads.
        tower_shapes = [array_ops.shape(g) for g, _ in tower_grads_and_vars]
        # Remember the original sizes of all the grads.
        tower_sizes = [array_ops.size(g) for g, _ in tower_grads_and_vars]
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
        if all([g.shape.is_fully_defined() for g, _ in tower_grads_and_vars]):
          total_grad_size = sum(
              [g.shape.num_elements() for g, _ in tower_grads_and_vars])
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
        self.all_tower_shapes.append(tower_shapes)
        self.all_tower_sizes.append(tower_sizes)

    return device_grad_packs

  def unpack(self, summed_device_grad_packs):
    """Reverse the pack."""
    aggregated_device_grads = []
    for (summed_tower_grad_packs,
         tower_grads_and_vars, tower_shapes, tower_sizes) in zip(
             summed_device_grad_packs, self.grouped_grads_and_vars,
             self.all_tower_shapes, self.all_tower_sizes):
      # pylint: enable=line-too-long
      # Reverse the packing operations in the previous steps. Form the
      # summed gradients back into their original shapes.
      with ops.colocate_with(summed_tower_grad_packs[0][0]):
        # Form a list of the summed grad packs.
        device_grad_packs = [g for g, _ in summed_tower_grad_packs]

        # Concat them back into a big flat tensor.
        device_grads_concat = array_ops.concat(device_grad_packs, 0)

        # Split the tensors back into their original sizes.
        grads_with_sizes = array_ops.split(device_grads_concat, tower_sizes)

        # Reshape the tensors back into their original shapes.
        grads_with_shapes = [
            array_ops.reshape(grad, shape)
            for shape, grad in zip(tower_shapes, grads_with_sizes)
        ]

        # Form the list with the original list of variables.
        summed_tower_grads = [
            (g, v) for g, (_, v) in zip(grads_with_shapes, tower_grads_and_vars)
        ]
        aggregated_device_grads.append(summed_tower_grads)
    return aggregated_device_grads


class AggregateSmallTensorPacker(object):
  """Concatenate small gradient tensors together for reduction."""

  def __init__(self,
               agg_small_grads_max_bytes=1048576,
               agg_small_grads_max_group=16):
    """Initialize the AggregateSmallTensorPacker object.

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
      tower_grads, self.packing = cross_tower_utils.pack_small_tensors(
          grouped_grads_and_vars,
          max_bytes=self.agg_small_grads_max_bytes,
          max_group=self.agg_small_grads_max_group)
    return tower_grads

  def unpack(self, summed_device_grad_packs):
    """Reverse the aggregation process."""
    return cross_tower_utils.unpack_small_tensors(summed_device_grad_packs,
                                                  self.packing)


def _pack_tensors(device_grads,
                  num_packs=0,
                  agg_small_grads_max_bytes=0,
                  agg_small_grads_max_group=0):
  """Pack tensors if specified."""
  if num_packs > 0:
    tensor_packer = ConcatAndSplitPacker(num_packs)
    device_grad_packs = tensor_packer.pack(device_grads)
  elif agg_small_grads_max_bytes > 0 and agg_small_grads_max_group > 0:
    tensor_packer = AggregateSmallTensorPacker(agg_small_grads_max_bytes,
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


class AllReduceCrossTowerOps(CrossTowerOps):
  """Reduction using all reduce."""

  def __init__(self,
               all_reduce_alg="nccl",
               num_packs=1,
               agg_small_grads_max_bytes=0,
               agg_small_grads_max_group=10):
    """All-reduce implementation of CrossTowerOps.

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
        tensors.
    """
    self._all_reduce_alg = all_reduce_alg
    self._num_packs = num_packs
    self._agg_small_grads_max_bytes = agg_small_grads_max_bytes
    self._agg_small_grads_max_group = agg_small_grads_max_group
    super(AllReduceCrossTowerOps, self).__init__()

  def _reduce(self, method_string, per_device_value, destinations):
    contains_indexed_slices = cross_tower_utils.contains_indexed_slices(
        per_device_value)
    if ((destinations is None or _devices_match(per_device_value, destinations))
        and not context.executing_eagerly()
        and not contains_indexed_slices):
      return self._batch_all_reduce(method_string, [per_device_value])[0]
    else:
      if contains_indexed_slices:
        logging.log_first_n(
            logging.WARN,
            "Efficient allreduce is not supported for IndexedSlices.", 10)

      devices = _get_devices_from(destinations or per_device_value)
      reduce_to_device = devices[0]
      reduced = _simple_reduce(per_device_value, reduce_to_device,
                               math_ops.add_n, method_string)
      return self.broadcast(reduced, devices)

  def _batch_reduce(self, method_string, value_destination_pairs):
    all_devices_match = _all_devices_match(value_destination_pairs)
    contains_indexed_slices = cross_tower_utils.contains_indexed_slices(
        value_destination_pairs)
    if (all_devices_match and not context.executing_eagerly()
        and not contains_indexed_slices):
      return self._batch_all_reduce(method_string,
                                    [v[0] for v in value_destination_pairs])
    else:
      if not all_devices_match:
        logging.warning("Efficient batch_reduce is not supported if "
                        "destinations are different.")

      return [
          self._reduce(method_string, t, destinations=v)
          for t, v in value_destination_pairs
      ]

  def _batch_all_reduce(self, method_string, per_device_values):
    """All reduce algorithm in a batch."""
    logging.info(
        "batch_all_reduce invoked for batches size = %d with "
        "algorithm = %s, num_packs = %d, agg_small_grads_max_bytes = %d and "
        "agg_small_grads_max_group = %d", len(per_device_values),
        self._all_reduce_alg, self._num_packs, self._agg_small_grads_max_bytes,
        self._agg_small_grads_max_group)
    destinations = per_device_values[0].devices
    grouped = _group_value_by_device(per_device_values)

    device_grad_packs, self._tensor_packer = _pack_tensors(
        grouped, self._num_packs, self._agg_small_grads_max_bytes,
        self._agg_small_grads_max_group)

    # The actual aggregation of the repacked gradients. Note that they are
    # sharded among different aggregation trees. So it is important to strike
    # the balance on num_splits.
    if self._all_reduce_alg == "nccl":
      # TODO(yuefengz): merge this into the all-reduce library.
      reduced = cross_tower_utils.aggregate_gradients_using_nccl(
          device_grad_packs)
    else:
      # TODO(yuefengz): check that gpu ids in `destinations` are in ascending
      # order.
      reduced = (
          cross_tower_utils.aggregate_gradients_using_hierarchical_copy(
              destinations, device_grad_packs))

    reduced = _unpack_tensors(reduced, self._tensor_packer)
    return _ungroup_and_make_mirrored(reduced, per_device_values[0].devices,
                                      method_string)


AllReduceSpecTuple = collections.namedtuple("AllReduceSpecTuple",
                                            "alg shards limit")


class MultiWorkerAllReduce(AllReduceCrossTowerOps):
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
      num_packs: see AllReduceCrossTowerOps.
      agg_small_grads_max_bytes: see AllReduceCrossTowerOps.
      agg_small_grads_max_group: see AllReduceCrossTowerOps.
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

  def _batch_all_reduce(self, method_string, per_device_values):
    """All reduce algorithm in a batch."""
    logging.info(
        "distributed batch_all_reduce invoked for batches size = %d with "
        "allreduce_spec = %r, num_packs = %d, agg_small_grads_max_bytes = %d "
        "and agg_small_grads_max_group = %d", len(per_device_values),
        self._all_reduce_spec, self._num_packs, self._agg_small_grads_max_bytes,
        self._agg_small_grads_max_group)

    destinations = sorted(per_device_values[0].devices)
    device_grads = _group_value_by_device(per_device_values)

    # The all reduce library requires fully defined shapes.
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
        (this_grads, remaining_grads) = cross_tower_utils.split_grads_by_size(
            spec_tuple.limit, remaining_grads)
      if this_grads:
        device_grad_packs, self._tensor_packer = _pack_tensors(
            this_grads, self._num_packs, self._agg_small_grads_max_bytes,
            self._agg_small_grads_max_group)
        range_agg_grads = cross_tower_utils.sum_gradients_all_reduce(
            self._worker_devices, device_grad_packs, len(self._worker_devices),
            spec_tuple.alg, spec_tuple.shards, range(self._num_gpus_per_worker))
        range_agg_grads = _unpack_tensors(range_agg_grads, self._tensor_packer)

        if not aggregated_grads:
          aggregated_grads = range_agg_grads
        else:
          assert len(aggregated_grads) == len(range_agg_grads)
          for i in range(len(aggregated_grads)):
            aggregated_grads[i] += range_agg_grads[i]
    assert not remaining_grads

    return _ungroup_and_make_mirrored(aggregated_grads, destinations,
                                      method_string)


_dgx1_links = [[1, 2, 3, 4], [0, 2, 3, 5], [0, 1, 3, 6], [0, 1, 2, 7],
               [0, 5, 6, 7], [1, 4, 6, 7], [2, 4, 5, 7], [3, 4, 5, 6]]


def _has_dgx1_like_links(gpu_links):
  if not gpu_links:
    return False
  # TODO(yuefengz): figure out the right topology for hierarchial copy if
  # number of gpus are less than 8.
  if len(gpu_links) < 8:
    return False
  for i, (gpu_link, dgx1_link) in enumerate(zip(gpu_links, _dgx1_links)):
    if (set(gpu_link) != set(dgx1_link) and
        set(gpu_link) != set(dgx1_link + [i])):
      return False
  return True


def _choose_all_reduce_algorithm(device_links):
  if _has_dgx1_like_links(device_links):
    logging.info("Configured hierarchical_copy with num_packs=%d",
                 len(device_links))
    return AllReduceCrossTowerOps(
        "hierarchical_copy", num_packs=len(device_links))
  else:
    logging.info("Configured nccl all-reduce.")
    return AllReduceCrossTowerOps("nccl", num_packs=1)


def choose_the_best(devices, session_config=None):
  """Find the best subclass of CrossTowerOps given a tensorflow session.

  Args:
    devices: a list of devices passed for distribute strategy.
    session_config: a tensorflow session config or None. If None, it will make
      deciesion based on all local devices.

  Returns:
    a subclass of CrossTowerOps.
  """
  requested_devices = set([device_util.canonicalize(d) for d in devices])
  machine_devices = device_lib.list_local_devices(session_config=session_config)
  using_devices = []
  for d in machine_devices:
    if device_util.canonicalize(d.name) in requested_devices:
      using_devices.append(d)
    else:
      logging.info(
          "Device is available but not used by distribute strategy: %s", d.name)

  if len(using_devices) != len(requested_devices):
    logging.warning("Not all devices in distribute strategy are visible by "
                    "TensorFlow sessions.")
    return ReductionToOneDeviceCrossTowerOps()

  if any([d.device_type.lower() != "gpu" for d in using_devices]):
    logging.warning("Not all devices in DistributionStrategy are visible to "
                    "TensorFlow session.")
    return ReductionToOneDeviceCrossTowerOps()

  device_links = [[] for _ in range(len(using_devices))]
  for i, device in enumerate(using_devices):
    for link in device.locality.links.link:
      device_links[i].append(link.device_id)

  return _choose_all_reduce_algorithm(device_links)

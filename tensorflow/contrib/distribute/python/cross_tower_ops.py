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
"""Classes for different algortihms of reduction and broadcasting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib.distribute.python import cross_tower_utils
from tensorflow.contrib.distribute.python import values as value_lib
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


def _get_devices_from(destinations):
  if isinstance(destinations, value_lib.DistributedValues):
    return list(destinations.devices)
  elif isinstance(destinations, six.string_types):
    return [device_util.canonicalize(destinations)]
  else:
    return [
        device_util.canonicalize(destination) for destination in destinations
    ]


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


def _simple_broadcast(tensor, destinations):
  index = {}
  devices = _get_devices_from(destinations)
  for d in devices:
    with ops.device(d):
      index[d] = array_ops.identity(tensor)
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
      v = math_ops.add_n(v_list)
    else:
      count += 1
    all_values.append(v)
  if not all_values:
    raise ValueError("`per_device_value` must be non-empty")

  with ops.device(reduce_to_device):
    with context.context().device_policy(context.DEVICE_PLACEMENT_SILENT):
      if method_string == "sum":
        reduced = accumulation_fn(all_values)
      elif method_string == "mean":
        reduced = accumulation_fn(all_values) / count
      else:
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
        and destinations. If a destionation is None, then the destinations
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

  This grouping is needed to call the allreduce library.

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
  """Ungroup results from allreduce and make Mirrored objects.

  Each allreduce result would be divided by the number of destinations before
  Mirrored objects are created if method_string is "mean".
  """
  index = [{} for _ in range(len(grouped_reduced[0]))]
  for d, per_device_reduced in enumerate(grouped_reduced):
    for i, (v, _) in enumerate(per_device_reduced):
      if method_string == "mean":
        index[i][destinations[d]] = v / len(destinations)
      else:
        index[i][destinations[d]] = v
  return [value_lib.Mirrored(v) for v in index]


class AllReduceCrossTowerOps(CrossTowerOps):
  """Reduction using all reduce."""

  def __init__(self, all_reduce_alg="nccl", gradient_repacking=1):
    """Initialize this subclass of CrossTowerOps with allreduce.

    Gradients would be repacked for more efficient cross-device transportation.

    Args:
      all_reduce_alg: the allreduce algorithm to use, currently only "nccl" or
        "hierarchical_copy" are supported.
      gradient_repacking: If zero, no gradient repacking would be done. If
        non-zero value it specifies the number of split packs that will be
        formed.
    """
    self.all_reduce_alg = all_reduce_alg
    self.gradient_repacking = gradient_repacking
    super(AllReduceCrossTowerOps, self).__init__()

  def _reduce(self, method_string, per_device_value, destinations):
    if ((destinations is None or _devices_match(per_device_value, destinations))
        and not context.executing_eagerly()):
      return self._batch_all_reduce(method_string, [per_device_value])[0]
    else:
      devices = _get_devices_from(destinations or per_device_value)
      reduce_to_device = devices[0]
      reduced = _simple_reduce(per_device_value, reduce_to_device,
                               math_ops.add_n, method_string)
      return self.broadcast(reduced, devices)

  def _batch_reduce(self, method_string, value_destination_pairs):
    if (_all_devices_match(value_destination_pairs) and
        not context.executing_eagerly()):
      return self._batch_all_reduce(method_string,
                                    [v[0] for v in value_destination_pairs])
    else:
      if not context.executing_eagerly():
        logging.warning("Efficient batch_reduce is not supported if "
                        "destinations are different.")
      return [
          self._reduce(method_string, t, destinations=v)
          for t, v in value_destination_pairs
      ]

  def _batch_all_reduce(self, method_string, per_device_values):
    """All reduce algorithm in a batch."""
    logging.info("batch_all_reduce invoked for batches size = %d with algorithm"
                 " = %s and gradient repacking = %d", len(per_device_values),
                 self.all_reduce_alg, self.gradient_repacking)
    destinations = per_device_values[0].devices
    grouped = _group_value_by_device(per_device_values)
    if self.gradient_repacking == 0:
      if self.all_reduce_alg == "nccl":
        reduced = cross_tower_utils.aggregate_gradients_using_nccl(grouped)
      else:
        # TODO(yuefengz): check that gpu ids in `destinations` are in ascending
        # order.
        reduced = (
            cross_tower_utils.aggregate_gradients_using_hierarchical_copy(
                destinations, grouped))
    else:
      device_grad_packs = []
      all_tower_shapes = []
      all_tower_sizes = []
      for tower_grads_and_vars in grouped:
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
          # TODO(zhengxq): it is possible to optimize away the additional
          # data movement by copying along the original variable boundary.
          # TODO(zhengxq): it is also possible to optimize away all the concat
          # as well.
          num_splits = self.gradient_repacking
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
          all_tower_shapes.append(tower_shapes)
          all_tower_sizes.append(tower_sizes)

      # The actual aggregation of the repacked gradients. Note that they are
      # sharded among different aggregation trees. So it is important to
      # strike the balance on num_splits.
      if self.all_reduce_alg == "nccl":
        summed_device_grad_packs = (
            cross_tower_utils.aggregate_gradients_using_nccl(device_grad_packs))
      else:
        summed_device_grad_packs = (
            cross_tower_utils.aggregate_gradients_using_hierarchical_copy(
                destinations, device_grad_packs))

      aggregated_device_grads = []
      for (summed_tower_grad_packs, tower_grads_and_vars, tower_shapes,
           tower_sizes) in zip(summed_device_grad_packs, grouped,
                               all_tower_shapes, all_tower_sizes):
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
              (g, v)
              for g, (_, v) in zip(grads_with_shapes, tower_grads_and_vars)
          ]
          aggregated_device_grads.append(summed_tower_grads)
      reduced = aggregated_device_grads
    return _ungroup_and_make_mirrored(reduced, per_device_values[0].devices,
                                      method_string)

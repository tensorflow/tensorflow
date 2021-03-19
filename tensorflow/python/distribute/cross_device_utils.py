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
"""Utilities for cross_device_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import threading

from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging

INSTANCE_KEY_START_NUMBER = 100


def aggregate_gradients_using_nccl(replica_grads):
  """Aggregate gradients using nccl allreduce."""
  agg_all_g_and_v = []
  for single_g_and_v in zip(*replica_grads):
    single_grads = [g for g, _ in single_g_and_v]
    agg_grads = nccl_ops.all_sum(single_grads)
    agg_all_g_and_v.append(
        [(g, v) for g, (_, v) in zip(agg_grads, single_g_and_v)])

  agg_all_g_and_v = list(zip(*agg_all_g_and_v))

  return agg_all_g_and_v


def aggregate_gradients_using_hierarchical_copy(avail_devices, replica_grads):
  """Aggregate gradients using hierarchical copies.

  Args:
    avail_devices: available GPU devices.
    replica_grads: List of lists of (gradient, variable) tuples. The outer list
      is over replicas. The inner list is over individual gradients.

  Returns:
    The list of (aggregated_gradient, variable), where the gradient has been
      summed across all replicas and the variable is chosen from the first
      replica.
  """
  # This only works for DGX-1 type of machine topology
  # Device peer to peer matrix
  # DMA: 0 1 2 3 4 5 6 7
  # 0:   Y Y Y Y Y N N N
  # 1:   Y Y Y Y N Y N N
  # 2:   Y Y Y Y N N Y N
  # 3:   Y Y Y Y N N N Y
  # 4:   Y N N N Y Y Y Y
  # 5:   N Y N N Y Y Y Y
  # 6:   N N Y N Y Y Y Y
  # 7:   N N N Y Y Y Y Y
  agg_grads = []
  num_devices = len(avail_devices)
  # In the special case of DGX-1 machine topology, the two groups have equal
  # size.
  group_size = num_devices // 2
  for i, single_grads in enumerate(zip(*replica_grads)):
    group_0_main_device = i % num_devices
    group_1_main_device = (group_0_main_device + group_size) % num_devices
    if group_0_main_device < group_size:
      group_0_begin = 0
      group_1_begin = group_size
    else:
      group_0_begin = group_size
      group_1_begin = 0

    # Aggregate the first group.
    group_0_device_grads = single_grads[group_0_begin:
                                        group_0_begin + group_size]
    with ops.device(avail_devices[group_0_main_device]):
      group_0_agg_grads, _ = aggregate_single_gradient_using_copy(
          group_0_device_grads, False, False)

    # Aggregate the second group.
    group_1_device_grads = single_grads[group_1_begin:
                                        group_1_begin + group_size]
    with ops.device(avail_devices[group_1_main_device]):
      group_1_agg_grads, _ = aggregate_single_gradient_using_copy(
          group_1_device_grads, False, False)

    # Aggregate between the groups.
    with ops.device(avail_devices[group_0_main_device]):
      (agg_total_grads, _), _ = aggregate_single_gradient_using_copy(
          [group_0_agg_grads, group_1_agg_grads], False, False)

    # Broadcast the result back into the root of each group.
    with ops.device(avail_devices[group_0_main_device]):
      group_0_agg_grads_bcast = array_ops.identity(agg_total_grads)
    with ops.device(avail_devices[group_1_main_device]):
      group_1_agg_grads_bcast = array_ops.identity(agg_total_grads)

    agg_grads_bcast = []
    for j in range(len(single_grads)):
      with ops.device(avail_devices[j]):
        # Broadcast the result back to each member in the group from the root.
        if (group_0_main_device < group_size) == (j < group_size):
          src_device_grad = group_0_agg_grads_bcast
        else:
          src_device_grad = group_1_agg_grads_bcast
        agg_grads_bcast.append(array_ops.identity(src_device_grad))

    agg_grads.append(
        [(g, v) for g, (_, v) in zip(agg_grads_bcast, single_grads)])

  agg_grads = list(zip(*agg_grads))

  return agg_grads


def aggregate_single_gradient_using_copy(grad_and_vars, use_mean,
                                         check_inf_nan):
  """Calculate the average gradient for a shared variable across all replicas.

  Note that this function provides a synchronization point across all replicas.

  Args:
    grad_and_vars: A list or tuple of (gradient, variable) tuples. Each
      (gradient, variable) pair within the outer list represents the gradient
      of the variable calculated for a single replica, and the number of pairs
      equals the number of replicas.
    use_mean: if True, mean is taken, else sum of gradients is taken.
    check_inf_nan: check grads for nans and infs.

  Returns:
    The tuple ([(average_gradient, variable),], has_nan_or_inf) where the
      gradient has been averaged across all replicas. The variable is chosen
      from the first replica. The has_nan_or_inf indicates the grads has nan or
      inf.
  """
  grads = [g for g, _ in grad_and_vars]
  grad = math_ops.add_n(grads)

  if use_mean and len(grads) > 1:
    grad = array_ops.multiply(grad, 1.0 / len(grads))

  v = grad_and_vars[0][1]
  if check_inf_nan:
    has_nan_or_inf = array_ops.logical_not(
        array_ops.reduce_all(array_ops.is_finite(grads)))
    return (grad, v), has_nan_or_inf
  else:
    return (grad, v), None


# TODO(yuefengz): use random key starts to avoid reusing keys?
class CollectiveKeys(object):
  """Class that manages collective keys.

  We need to manage three different keys for collective:

  *Group key*: an integer key to identify the set of cooperative devices.
  Collective ops work under the same set of devices must using the same group
  key.

  *Instance key*: an integer key to identify the set of same counterpart of
  tensors on different devices in a device group that need to be all-reduced.

  This class is thread safe.
  """

  def __init__(self, group_key_start=1):
    """Initializes the object.

    Args:
      group_key_start: the starting integer of group key.
    """
    self._group_key = group_key_start
    self._group_key_table = {}
    self._instance_key_table = {}
    self._lock = threading.Lock()

  def get_group_key(self, devices):
    """Returns a group key for the set of devices.

    Args:
      devices: a list of canonical device strings in a collective group.

    Returns:
      int key uniquely identifying the set of device names.
    """
    key_id = hash(tuple(sorted(devices)))
    with self._lock:
      if key_id not in self._group_key_table:
        new_key = self._group_key
        self._group_key += 1
        self._group_key_table[key_id] = new_key
        self._instance_key_table[new_key] = {}
        for device in devices:
          self._instance_key_table[new_key][device] = INSTANCE_KEY_START_NUMBER
      return self._group_key_table[key_id]

  def get_instance_key(self, group_key, device):
    """Returns a new instance key for use in defining a collective op.

    You should call this once per each collective op of a collective instance.

    Args:
      group_key: the group key returned by get_group_key(). You should not
        assign the group key yourself.
      device: a canonical device string. It should be the device this collective
        op is on.

    Returns:
      a new instance key.

    Raises:
      ValueError: when the group key is invalid or the device is not in the
      group.
    """
    with self._lock:
      group = self._instance_key_table.get(group_key, None)
      if group is None:
        raise ValueError('group {} not found'.format(group_key))
      if device not in group:
        raise ValueError('{} not in group {}'.format(device, group_key))
      v = group[device]
      group[device] += 1
      return v

  def __deepcopy__(self, memo):
    # distribute_coordinator deep-copies the strategy object, so
    # CollectiveKeys needs to support deep copy as well.
    copied = CollectiveKeys()
    copied._group_key = self._group_key
    copied._group_key_table = copy.deepcopy(self._group_key_table, memo)
    copied._instance_key_table = copy.deepcopy(self._instance_key_table, memo)
    return copied


class CollectiveReplicaLauncher(object):
  """Launch collectives on one replica."""

  _prefer_unique_instance_key = True
  _prefer_ordering_token = True

  def __init__(self,
               group_key,
               group_size,
               collective_keys,
               device):
    self._group_key = group_key
    self._group_size = group_size
    self._collective_keys = collective_keys
    self._device = device
    if self._use_ordering_token():
      with ops.init_scope(), ops.device(device):
        self._ordering_token = resource_variable_ops.ResourceVariable(0.)
    else:
      self._ordering_token = None

  def _control_input(self, control_input):
    if control_input is not None and not self._use_ordering_token():
      return ops.control_dependencies([control_input])
    return ops.NullContextmanager()

  def _use_unique_instance_key(self):
    if not ops.executing_eagerly_outside_functions():
      return False
    return CollectiveReplicaLauncher._prefer_unique_instance_key

  def _use_ordering_token(self):
    # We rely on auto control dep to insert control edges between NCCL calls,
    # but for tf1 graph mode auto control dep is not used.
    if not ops.executing_eagerly_outside_functions():
      return False
    return CollectiveReplicaLauncher._prefer_ordering_token

  def _next_instance_key(self):
    """Returns the next instance key."""
    if self._use_unique_instance_key():
      # Assigning instance keys at function building time have issues since
      # different workers may retrace the function at different times. With
      # collective V2 we can use capture_call_time_value to use a placeholder as
      # the instance key and feed it at function call time. In this way we also
      # don't reuse instance keys, which allows for per-instance cancellation.
      graph = ops.get_default_graph()
      # Control flow ops don't work with capture_call_time_value, so we put the
      # capture in the function graph of that control flow op.
      while getattr(graph, 'is_control_flow_graph', False):
        graph = graph.outer_graph
      if not context.executing_eagerly() and graph.building_function:
        with graph.as_default():
          # Capture self._next_instance_key so that when building a function
          # that calls another tf.function, the instance key assignment is
          # further delayed until we actually call the function in eager. Note
          # that capture_call_time_value doesn't automatically propagate the
          # deferred capture to the outer function.
          return graph.capture_call_time_value(
              self._next_instance_key, tensor_spec.TensorSpec([], dtypes.int32))
      else:
        instance_key = self._collective_keys.get_instance_key(
            self._group_key, self._device)
        with ops.device('CPU:0'):
          return ops.convert_to_tensor(instance_key, dtype=dtypes.int32)
    else:
      return self._collective_keys.get_instance_key(self._group_key,
                                                    self._device)

  def _get_ordering_token(self, communication_hint):
    if self._use_ordering_token() and communication_hint == 'NCCL':
      return self._ordering_token.handle
    return None

  def can_order_nccl(self):
    """Whether this launcher can order NCCL operations."""
    return self._use_ordering_token()

  def all_reduce(self,
                 input_tensor,
                 control_input=None,
                 communication_hint='AUTO',
                 timeout=0):
    """All-reduce a dense tensor.

    Args:
      input_tensor: a dense tensor. It must have the same shape on all replicas.
      control_input: if not None, add control edges between control_input and
        the all-reduce.
      communication_hint: string providing hint to runtime for choosing
        collective implementation.
      timeout: a float. The timeout in seconds.

    Returns:
      The reduced tensor.
    """
    instance_key = self._next_instance_key()
    ordering_token = self._get_ordering_token(communication_hint)
    with ops.device(self._device), \
         self._control_input(control_input):
      return collective_ops.all_reduce_v2(
          input_tensor,
          self._group_size,
          self._group_key,
          instance_key,
          communication_hint=communication_hint,
          timeout=timeout,
          ordering_token=ordering_token)

  def _all_gather(self, input_tensor, communication_hint='AUTO', timeout=0):
    """All-gather a dense tensor.

    Args:
      input_tensor: a dense tensor. It must have the same shape on all replicas.
      communication_hint: string providing hint to runtime for choosing
        collective implementation.
      timeout: a float. The timeout in seconds.

    Returns:
      The reduced tensor.
    """
    instance_key = self._next_instance_key()
    ordering_token = self._get_ordering_token(communication_hint)
    with ops.device(self._device):
      return collective_ops.all_gather_v2(
          input_tensor,
          self._group_size,
          self._group_key,
          instance_key,
          communication_hint=communication_hint,
          timeout=timeout,
          ordering_token=ordering_token)

  def batch_all_reduce(self,
                       input_tensor_packs,
                       communication_hint='AUTO',
                       timeout=0):
    """Batch all-reduce dense tensors.

    This takes a list of batches of tensors. Using multiple batches have the
    benefit that it doesn't need to wait for all inputs to be ready to start the
    all-reduce.

    Args:
      input_tensor_packs: a list of lists of dense tensors.
      communication_hint: string providing hint to runtime for choosing
        collective implementation.
      timeout: a float. The timeout in seconds.

    Returns:
      A flat list of reduced tensors.
    """
    outputs = []
    for pack in input_tensor_packs:
      if context.executing_eagerly():
        # We don't batch in eager as it sometimes makes the performance worse
        # due the concat/split ops.
        for input_tensor in pack:
          outputs.append(
              self.all_reduce(input_tensor, None, communication_hint, timeout))
      else:
        # TODO(b/169168846): inserts a parallel all_gather to verify packings
        # are the same on each replica.
        with ops.device(self._device):
          flat_tensors = [array_ops.reshape(t, [-1]) for t in pack]
          shapes = [array_ops.shape(t) for t in pack]
          if communication_hint == 'NCCL' and outputs:
            control_input = outputs[-1]
          else:
            control_input = None
          reduced = self.all_reduce(
              array_ops.concat(flat_tensors, axis=0), control_input,
              communication_hint, timeout)
          num_elements = [math_ops.reduce_prod(s) for s in shapes]
          flat_outputs = array_ops.split(reduced, num_elements, axis=0)
          for shape, flat_output in zip(shapes, flat_outputs):
            outputs.append(array_ops.reshape(flat_output, shape))

    return outputs

  def all_gather(self,
                 input_tensor,
                 axis,
                 communication_hint='AUTO',
                 timeout=0):
    """All-gather a dense tensor.

    This method must be called inside a tf.function.

    Args:
      input_tensor: a dense tensor. It must have the same rank on all replicas,
        and dimensions other than `axis` need to be the same as well.
      axis: 0-D int32 Tensor. Dimension along which to gather. Must be in the
        range [0, rank(value)).
      communication_hint: string providing hint to runtime for choosing
        collective implementation. Available options are `AUTO`, `NCCL`, and
        `RING`.
      timeout: a float. The timeout in seconds.

    Returns:
      The gathered Tensor.

    Raises:
      RuntimeError: if called in eager mode.
    """
    if context.executing_eagerly():
      raise RuntimeError('all_gather in eager mode is not supported')

    with ops.device(self._device), \
         ops.control_dependencies([array_ops.identity(input_tensor)]):
      # 1. Transpose
      # E.g. Given an input_tensor with shape [2,2,5,1] and axis to gather is 3,
      # we use perm_pre=[3 0 1 2] to reshape it to [1,2,2,5], which
      # brings the 3rd dim first; afterwards we use perm_after=[1,2,3,0] to
      # place it back.
      perm_pre = array_ops.concat(
          ([axis], math_ops.range(axis),
           math_ops.range(axis + 1, array_ops.rank(input_tensor))),
          axis=0)
      input_tensor_t = array_ops.transpose(input_tensor, perm=perm_pre)
      # 2. Pad
      gathered_shape = self._all_gather(
          array_ops.expand_dims_v2(array_ops.shape_v2(input_tensor_t), axis=0),
          communication_hint,
          timeout=timeout)
      first_dims = gathered_shape[:, 0]
      full_axis_dim = math_ops.reduce_max(first_dims)
      padded_input_tensor = _pad_util(input_tensor_t, full_axis_dim)

      # 3. Gather
      gather_padded_out_tensor = self._all_gather(
          padded_input_tensor, communication_hint, timeout=timeout)
      # 4. Unpad
      split_tensors = []
      for i in range(self._group_size):
        start_pos = i * full_axis_dim
        split_tensors.append(gather_padded_out_tensor[start_pos:start_pos +
                                                      first_dims[i]])
      out_tensor_t = array_ops.concat(split_tensors, 0)

      # 5. Transpose back
      perm_after = array_ops.concat(
          (math_ops.range(1, axis + 1), [0],
           math_ops.range(axis + 1, array_ops.rank(input_tensor_t))),
          axis=0)
      return array_ops.transpose(out_tensor_t, perm=perm_after)

  def all_reduce_indexed_slices(self,
                                input_slices,
                                communication_hint='AUTO',
                                timeout=0):
    """All-reduce an IndexedSlices.

    This method must be called inside a tf.function.

    Args:
      input_slices: an IndexedSlices.
      communication_hint: string providing hint to runtime for choosing
        collective implementation.
      timeout: a float. The timeout in seconds.

    Returns:
      The reduced IndexedSlices.

    Raises:
      RuntimeError: if called in eager mode.
    """
    if context.executing_eagerly():
      raise RuntimeError(
          'all_reduce_indexed_slices in eager mode is not supported')

    # Current CollectiveAllGather implementations require input IndexedSlices to
    # have consistent length across the board, we handle the reduction of
    # IndexedSlices as follows:
    #   1. Gather the lengths of IndexedSlices from all participants.
    #   2. If they have consistent length, apply all_gather.
    #   3. Otherwise convert IndexedSlices to dense tensors and apply
    #      all_reduce.
    with ops.device(self._device):

      def all_gather():
        """Use all_gather to aggregate `IndexedSlices`."""
        all_values = self._all_gather(
            input_slices.values, communication_hint, timeout=timeout)
        # Add control dependency to order the all-gather.
        control = [all_values] if communication_hint == 'NCCL' else []
        with ops.control_dependencies(control):
          all_indices = self._all_gather(
              input_slices.indices, communication_hint, timeout=timeout)
        return ops.IndexedSlices(
            values=all_values,
            indices=all_indices,
            dense_shape=input_slices.dense_shape)

      def densify_and_all_reduce():
        """Use all_reduce to aggregate `IndexedSlices`."""
        densified = ops.convert_to_tensor(input_slices)
        reduced = self.all_reduce(
            densified, communication_hint=communication_hint, timeout=timeout)
        # We have to convert dense grad to IndexedSlice because all_reduce()
        # and all_gather() must have the same return type as required by
        # control_flow_ops.cond.
        return ops.IndexedSlices(
            values=reduced,
            indices=math_ops.range(array_ops.shape(reduced)[0]),
            dense_shape=input_slices.dense_shape)

      length = array_ops.shape(input_slices.indices)
      all_lengths = self._all_gather(
          length, communication_hint, timeout=timeout)
      return control_flow_ops.cond(
          math_ops.equal(
              math_ops.reduce_max(all_lengths),
              math_ops.reduce_min(all_lengths)), all_gather,
          densify_and_all_reduce)


def aggregate_tensors_or_indexed_slices(values, accumulation_fn=math_ops.add_n):
  """Aggregate tensors using `accumulation_fn` and IndexedSlices via concat."""
  if any(isinstance(v, ops.IndexedSlices) for v in values):
    return backprop.aggregate_indexed_slices_gradients(values)
  else:
    return accumulation_fn(values)


def divide_by_n_tensors_or_indexed_slices(value, n):
  if isinstance(value, ops.IndexedSlices):
    value = backprop.flatten_nested_indexed_slices(value)
    return ops.IndexedSlices(
        value.values / n, value.indices, value.dense_shape)
  else:
    return value / n


def copy_tensor_or_indexed_slices_to_device(value, device):
  with ops.device(device):
    if isinstance(value, ops.IndexedSlices):
      copied_values = array_ops.identity(value.values)
      copied_indices = array_ops.identity(value.indices)
      copied_shape = array_ops.identity(value.dense_shape)
      result = ops.IndexedSlices(copied_values, copied_indices, copied_shape)
    else:
      result = array_ops.identity(value)
  return result


def is_indexed_slices(value):
  if isinstance(value, ops.IndexedSlices):
    return True
  if isinstance(value, value_lib.DistributedValues):
    return all(isinstance(v, ops.IndexedSlices) for v in value.values)
  return False


def split_by_sparsity(values):
  """Split values into dense and sparse values.

  Args:
    values: a list of tensors or `PerReplica`s.

  Returns:
    Four lists:
      a list of dense values, a list of their indices in `values` and
      a list of sparse values, a list of their indices in `values`.
  """
  dense_values = []
  dense_indices = []
  sparse_values = []
  sparse_indices = []
  for i, v in enumerate(values):
    if is_indexed_slices(v):
      sparse_values.append(v)
      sparse_indices.append(i)
    else:
      dense_values.append(v)
      dense_indices.append(i)
  return dense_values, dense_indices, sparse_values, sparse_indices


def stitch_values(values_and_indices_list):
  """Stitch values together according to their indices.

  Args:
    values_and_indices_list: a list of tuples of values and indices indicating
      the values and positions in the returned list.

  Returns:
    a stitched list of values.
  """
  length = 0
  for values_and_indices in values_and_indices_list:
    length += len(values_and_indices[0])

  result = [None] * length
  for values_and_indices in values_and_indices_list:
    if values_and_indices and values_and_indices[0]:
      for v, i in zip(*values_and_indices):
        assert result[i] is None
        result[i] = v
  return result


def group_by_size(input_tensors, bytes_per_pack):
  """Groups `input_tensors` into chunks of `bytes_per_pack`.

  The method preserves the original order of `input_tensors`. The grouping is
  best effort, each pack could have more or less bytes than `bytes_per_pack`.
  It only groups values with known shape.

  Args:
    input_tensors: a list of Tensor.
    bytes_per_pack: an integer.

  Returns:
    A list of packs of Tensor. All values are grouped into one pack if
    `bytes_per_pack` is zero or any of the value has unknown shape.
  """

  if bytes_per_pack == 0:
    return [input_tensors]
  packs = []
  last_pack_size = 0
  for value in input_tensors:
    num_elements = value.shape.num_elements()
    if num_elements is None:
      # Can't pack values with unknown shape.
      logging.warning(
          'not packing values due to the unknown or inconsistent shape of %s',
          value)
      return [input_tensors]
    size = num_elements * value.dtype.size
    # Try to keep each pack as close to bytes_per_pack as possible, while each
    # pack is at least bytes_per_pack large. I.E. we err on the side of having
    # few but large packs.
    if not packs or last_pack_size > bytes_per_pack:
      packs.append([])
      last_pack_size = 0
    packs[-1].append(value)
    last_pack_size += size
  return packs


def _pad_util(input_tensor, full_axis_dim):
  """Pad the `input_tensor`'s first dimension to be `full_axis_dim`."""
  missing_axis_dim = full_axis_dim - array_ops.shape_v2(input_tensor)[0]
  tensor_rank = array_ops.rank(input_tensor)
  paddings_axis = [[0, missing_axis_dim]]
  paddings = array_ops.concat([
      paddings_axis,
      array_ops.zeros(shape=(tensor_rank - 1, 2), dtype=dtypes.int32)
  ],
                              axis=0)
  padded_input_tensor = array_ops.pad(input_tensor, paddings)
  return padded_input_tensor

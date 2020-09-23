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
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.platform import tf_logging as logging

OP_INSTANCE_KEY_START_NUMBER = 100


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

  "Graph key": an integer key that is unique key graph. This is used to support
  multiple graphs per client session. It must be non-zero and set in the
  `config` argument of each call to `session.run`.

  This class is thread safe.
  """

  def __init__(self,
               group_key_start=1,
               op_instance_key_start=OP_INSTANCE_KEY_START_NUMBER,
               variable_instance_key_start=1000000):
    """Initializes the object.

    Args:
      group_key_start: the starting integer of group key.
      op_instance_key_start: the starting integer of instance key for ops.
      variable_instance_key_start: the starting integer of instance key for
        variables.
    """
    self._group_key = group_key_start
    self._group_key_table = {}

    assert op_instance_key_start != variable_instance_key_start
    self._op_instance_key = op_instance_key_start
    self._variable_instance_key = variable_instance_key_start
    self._lock = threading.Lock()

  def get_group_key(self, devices):
    """Returns a group key for the set of devices.

    Args:
      devices: list of strings naming devices in a collective group.

    Returns:
      int key uniquely identifying the set of device names.
    """
    parsed = [pydev.DeviceSpec.from_string(d) for d in devices]
    # In the between-graph replicated training, different workers need to get
    # the same device key. So we remove the task_type and task_id from the
    # devices.
    # TODO(yuefengz): in the in-graph replicated training, we need to include
    # task_type and task_id.
    names = sorted(['%s:%d' % (d.device_type, d.device_index) for d in parsed])
    key_id = ','.join(names)
    with self._lock:
      if key_id not in self._group_key_table:
        new_key = self._group_key
        self._group_key += 1
        self._group_key_table[key_id] = new_key
      return self._group_key_table[key_id]

  def get_op_instance_key(self):
    """Returns a new instance key for use in defining a collective op."""
    with self._lock:
      v = self._op_instance_key
      self._op_instance_key += 1
      return v

  def get_variable_instance_key(self):
    """Returns a new instance key for use in creating a Variable."""
    with self._lock:
      v = self._variable_instance_key
      self._variable_instance_key += 1
      return v

  def __deepcopy__(self, memo):
    # distribute_coordinator deep-copies the strategy object, so
    # CollectiveKeys needs to support deep copy as well.
    copied = CollectiveKeys()
    copied._group_key = self._group_key
    copied._group_key_table = copy.deepcopy(self._group_key_table, memo)
    copied._op_instance_key = self._op_instance_key
    copied._variable_instance_key = self._variable_instance_key
    return copied


def build_collective_reduce(input_tensors,
                            devices,
                            group_size,
                            collective_keys,
                            reduction_op='Add',
                            unary_op='Id',
                            communication_hint='AUTO',
                            control_inputs=None,
                            executors=None,
                            timeout=None):
  """Build a subgraph that does one full all-reduce, using the collective Op.

  If called in eager mode, it's required to supply a list of async executors for
  each input Tensor.

  Args:
    input_tensors: tensors within a single worker graph that are to be reduced
      together; must be one per device.
    devices: a list of device strings to run the collective on.
    group_size: total number of devices globally that will be doing this same
      reduction.  The reduction will actually include the corresponding tensors
      at all these workers.
    collective_keys: a CollectiveKeys object.
    reduction_op: string naming the reduction op.
    unary_op: string naming the unary final op.
    communication_hint: string providing hint to runtime for choosing collective
      implementation.
    control_inputs: if not None, add control edges between control_inputs and
      (index-wise) corresponding collective_reduce tensors
    executors: a list of async executor. Required for eager execution.
    timeout: a float or None. The timeout in seconds.

  Returns:
    An array of final tensors, one per device, computed by the full reduction.

  Raises:
    ValueError: There must be at least two tensors over all the workers.
  """
  if context.executing_eagerly():
    if (not executors or len(executors) != len(input_tensors) or
        not all(e.is_async() for e in executors)):
      raise ValueError(
          'collectives requires async executors for each device in eager mode')
  if len(input_tensors) != len(devices):
    raise ValueError('collective requires one input tensor for each device, '
                     'len(input_tensors) = %d, len(devices) = %d' %
                     (len(input_tensors), len(devices)))

  if group_size < 2:
    return input_tensors
  group_key = collective_keys.get_group_key(devices)
  instance_key = collective_keys.get_op_instance_key()
  subdiv_offsets = [0]  # TODO(tucker): maybe support non-default subdiv spec

  out_tensors = []
  for idx, input_tensor in enumerate(input_tensors):
    if context.executing_eagerly():
      executor_scope = context.executor_scope(executors[idx])
    else:
      executor_scope = ops.NullContextmanager()
    with executor_scope, \
         ops.device(devices[idx]), \
         ops.control_dependencies(
             _control_input(devices, control_inputs, idx)):
      out_tensor = collective_ops.all_reduce(
          input_tensor,
          group_size,
          group_key,
          instance_key,
          reduction_op,
          unary_op,
          subdiv_offsets,
          communication_hint,
          timeout=timeout)
    out_tensors.append(out_tensor)
  return out_tensors


def build_collective_gather(input_tensors,
                            devices,
                            group_size,
                            collective_keys,
                            axis,
                            communication_hint='AUTO',
                            control_inputs=None,
                            timeout=None):
  """Build a subgraph that does one full all-gather, using the collective Op.

  This method must be called in graph mode or inside a tf.function.

  Args:
    input_tensors: tensors within a single worker graph that are to be gathered
      together; must be one per device. Input tensors cannot have rank 0.
    devices: a list of device strings to run the collective on.
    group_size: total number of devices globally that will be doing this same
      gathering. The gathering will actually include the corresponding tensors
      at all these workers.
    collective_keys: a CollectiveKeys object.
    axis: 0-D int32 Tensor. Dimension along which to gather. Must be in the
      range [0, rank(value)).
    communication_hint: string providing hint to runtime for choosing collective
      implementation. Available options are `AUTO`, `NCCL`, and `RING`.
    control_inputs: if not None, add control edges between control_inputs and
      (index-wise) corresponding collective_gather tensors
    timeout: a float or None. The timeout in seconds.

  Returns:
    An array of final tensors, one per device, computed by the full gather.
  """
  if len(input_tensors) != len(devices):
    raise ValueError(
        'collective requires one input tensor for each device, %d != %d' %
        (len(input_tensors), len(devices)))

  if group_size < 2:
    return input_tensors
  group_key = collective_keys.get_group_key(devices)
  instance_key_tensor = collective_keys.get_op_instance_key()
  instance_key_shape = collective_keys.get_op_instance_key()

  out_tensors = []
  for idx, input_tensor in enumerate(input_tensors):
    with ops.device(devices[idx]), ops.control_dependencies(
        _control_input(devices, control_inputs, idx)):
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
      gathered_shape = collective_ops.all_gather(
          array_ops.expand_dims_v2(array_ops.shape_v2(input_tensor_t), axis=0),
          group_size,
          group_key,
          instance_key_shape,
          communication_hint,
          timeout=timeout)
      first_dims = gathered_shape[:, 0]
      full_axis_dim = math_ops.reduce_max(first_dims)
      padded_input_tensor = _pad_util(input_tensor_t, full_axis_dim)

      # 3. Gather
      gather_padded_out_tensor = collective_ops.all_gather(
          padded_input_tensor,
          group_size,
          group_key,
          instance_key_tensor,
          communication_hint,
          timeout=timeout)
      # 4. Unpad
      split_tensors = []
      for i in range(first_dims.shape[0]):
        start_pos = i * full_axis_dim
        split_tensors.append(gather_padded_out_tensor[start_pos:start_pos +
                                                      first_dims[i]])
      out_tensor_t = array_ops.concat(split_tensors, 0)

      # 5. Transpose back
      perm_after = array_ops.concat(
          (math_ops.range(1, axis + 1), [0],
           math_ops.range(axis + 1, array_ops.rank(input_tensor_t))),
          axis=0)
      out_tensor = array_ops.transpose(out_tensor_t, perm=perm_after)
      out_tensors.append(out_tensor)
  return out_tensors


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


def build_collective_gather_indexed_slices(input_slices_list,
                                           devices,
                                           group_size,
                                           collective_keys,
                                           communication_hint='AUTO',
                                           control_inputs=None,
                                           timeout=None):
  """Build a subgraph that all-gathers IndexedSlices using the collective Op.

  This method must be called in graph mode or inside a tf.function.

  Args:
    input_slices_list: a list of IndexedSlices within a single worker graph that
      are to be gathered together; must be one per device.
    devices: a list of device strings to run the collective on.
    group_size: total number of devices globally that will be doing this same
      gathering. The gathering will actually include the corresponding tensors
      at all these workers.
    collective_keys: a CollectiveKeys object.
    communication_hint: string providing hint to runtime for choosing collective
      implementation.
    control_inputs: if not None, add control edges between control_inputs and
      (index-wise) corresponding collective_reduce tensors
    timeout: a float or None. The timeout in seconds.

  Returns:
    An array of final IndexedSlices, one per device, computed by the full
    gather.

  Raises:
    ValueError: if control_inputs is not None and doesn't match the length and
      devices of inputs.
  """
  assert not context.executing_eagerly(), (
      'build_collective_gather_indexed_slices can only be called in graph mode'
      ' or inside tf.function')
  if len(input_slices_list) != len(devices):
    raise ValueError(
        'collective requires one input IndexedSlice for each device, %d != %d' %
        (len(input_slices_list), len(devices)))

  if group_size < 2:
    return input_slices_list

  group_key = collective_keys.get_group_key(devices)
  gather_length_key = collective_keys.get_op_instance_key()
  gather_indices_key = collective_keys.get_op_instance_key()
  gather_values_key = collective_keys.get_op_instance_key()
  reduce_densified_key = collective_keys.get_op_instance_key()

  # Current CollectiveAllGather implementations require input IndexedSlices to
  # have consistent length across the board, we handle the reduction of
  # IndexedSlices as follows:
  #   1. Gather the lengths of IndexedSlices from all participants.
  #   2. If they have consistent length, apply all_gather.
  #   3. Otherwise convert IndexedSlices to dense tensors and apply
  #      all_reduce.
  out_slices_list = []
  for idx, input_slices in enumerate(input_slices_list):
    # pylint: disable = cell-var-from-loop
    with ops.device(devices[idx]):

      def all_gather():
        """Use all_gather to aggregate `IndexedSlices`."""
        all_values = collective_ops.all_gather(
            input_slices.values,
            group_size,
            group_key,
            gather_values_key,
            communication_hint,
            timeout=timeout)
        # Add control dependency to order the all-gather.
        control = [all_values] if communication_hint == 'NCCL' else []
        with ops.control_dependencies(control):
          all_indices = collective_ops.all_gather(
              input_slices.indices,
              group_size,
              group_key,
              gather_indices_key,
              communication_hint,
              timeout=timeout)
        return ops.IndexedSlices(
            values=all_values,
            indices=all_indices,
            dense_shape=input_slices.dense_shape)

      def densify_and_all_reduce():
        """Use all_reduce to aggregate `IndexedSlices`."""
        densified = ops.convert_to_tensor(input_slices)
        reduced = collective_ops.all_reduce(
            densified,
            group_size,
            group_key,
            reduce_densified_key,
            'Add',
            'Id', [0],
            communication_hint,
            timeout=timeout)
        # We have to convert dense grad to IndexedSlice because all_reduce()
        # and all_gather() must have the same return type as required by
        # control_flow_ops.cond.
        return ops.IndexedSlices(
            values=reduced,
            indices=math_ops.range(array_ops.shape(reduced)[0]),
            dense_shape=input_slices.dense_shape)

      length = array_ops.shape(input_slices.indices)
      with ops.control_dependencies(
          _control_input(input_slices, control_inputs, idx)):
        all_lengths = collective_ops.all_gather(
            length,
            group_size,
            group_key,
            gather_length_key,
            communication_hint,
            timeout=timeout)
      out_slices = control_flow_ops.cond(
          math_ops.equal(
              math_ops.reduce_max(all_lengths),
              math_ops.reduce_min(all_lengths)), all_gather,
          densify_and_all_reduce)
      out_slices_list.append(out_slices)
    # pylint: enable=cell-var-from-loop
  return out_slices_list


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
  assert isinstance(value, value_lib.DistributedValues)
  return all(isinstance(v, ops.IndexedSlices) for v in value.values)


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


def per_replica_num_elements(per_replica):
  """Returns the static number of elements of one replica.

  Args:
    per_replica: A PerReplica of Tensor or IndexedSlices.

  Returns:
    Number of elements. None if some replica has a different or unknown shape.
  """

  values = per_replica._values  # pylint: disable=protected-access
  s0 = values[0].shape
  for v in values:
    assert not isinstance(v, ops.IndexedSlices)
    if v.shape != s0:
      return None
  return s0.num_elements()


def pack_by_size(per_replica_list, bytes_per_pack):
  """Packs `per_replica_list` into chunks of `bytes_per_pack`.

  The method preserves the original order of `per_replica_list`. The packing is
  best effort, each pack could have more or less bytes than `bytes_per_pack`.
  It only packs values with known shape. Note that, the usage is different from
  `cross_device_ops._pack_tensors`, this function is intended to work with the
  ScopeAllocator style batching used in `CollectiveAllReduce`.

  Args:
    per_replica_list: A list of PerReplica.
    bytes_per_pack: Bytes per pack.

  Returns:
    A list of packs of PerReplica. All values are packed into one pack if
      `bytes_per_pack` is zero or any of the value has unknown shape.
  """

  if bytes_per_pack == 0:
    return [per_replica_list]
  packs = []
  last_pack_size = 0
  for value in per_replica_list:
    num_elements = per_replica_num_elements(value)
    if num_elements is None:
      # Can't pack values with unknown shape.
      logging.warning(
          'not packing values due to the unknown or inconsistent shape of %s',
          value)
      return [per_replica_list]
    size = num_elements * value._primary.dtype.size  # pylint: disable=protected-access
    # Try to keep each pack as close to bytes_per_pack as possible, while each
    # pack is at least bytes_per_pack large. I.E. we err on the side of having
    # few but large packs.
    if not packs or last_pack_size > bytes_per_pack:
      packs.append([])
      last_pack_size = 0
    packs[-1].append(value)
    last_pack_size += size
  return packs


def _control_input(devices, control_inputs, idx):
  """Returns the `idx`-th item in control_inputs to be used in ops.control_dependencies.

  This is a helper function for building collective ops.

  Args:
    devices: a list of device strings the collective run on.
    control_inputs: a list or None.
    idx: the index into `inputs` and `control_inputs`.

  Returns:
    A one item list of the `idx`-th element of `control_inputs`, or an empty
    list if `control_inputs` is None.
  """
  if control_inputs is None:
    return []
  if len(control_inputs) != len(devices):
    raise ValueError(
        'control_inputs must match the length of the devices, %s != %s' %
        (len(control_inputs), len(devices)))
  return [control_inputs[idx]]

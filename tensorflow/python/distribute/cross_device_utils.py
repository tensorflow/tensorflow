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

import collections as pycoll
import threading

from tensorflow.python.distribute import all_reduce
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops


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


def group_device_names(devices, group_size):
  """Group device names into groups of group_size.

  Args:
    devices: a list of canonical device strings.
    group_size: integer which is equal to or greater than 1.

  Returns:
    list of lists of devices, where each inner list is group_size long,
      and each device appears at least once in an inner list.  If
      len(devices) % group_size == 0 then each device will appear exactly once.

  Raises:
    ValueError: if group_size > len(devices)
  """
  num_devices = len(devices)
  if group_size > num_devices:
    raise ValueError(
        'only %d devices, but group_size=%d' % (num_devices, group_size))
  num_groups = (
      num_devices // group_size + (1 if (num_devices % group_size != 0) else 0))
  groups = [[] for i in range(num_groups)]
  for i in range(num_groups * group_size):
    groups[i % num_groups].append(devices[i % num_devices])
  return groups


def split_grads_by_size(threshold_size, device_grads):
  """Break gradients into two sets according to tensor size.

  Args:
    threshold_size: int size cutoff for small vs large tensor.
    device_grads: List of lists of (gradient, variable) tuples.  The outer
        list is over devices. The inner list is over individual gradients.

  Returns:
    small_grads: Subset of device_grads where shape is <= threshold_size
       elements.
    large_grads: Subset of device_grads where shape is > threshold_size
       elements.
  """
  small_grads = []
  large_grads = []
  for dl in device_grads:
    small_dl = []
    large_dl = []
    for (g, v) in dl:
      tensor_size = g.get_shape().num_elements()
      if tensor_size <= threshold_size:
        small_dl.append([g, v])
      else:
        large_dl.append([g, v])
    if small_dl:
      small_grads.append(small_dl)
    if large_dl:
      large_grads.append(large_dl)
  return small_grads, large_grads


# threading.Lock() and threading.local() cannot be pickled and therefore cannot
# be a field of CollectiveKeys. Right now _thread_local is not necessary to be
# an instance member of CollectiveKeys since we always create a new thread for
# each replica.
_lock = threading.Lock()
_thread_local = threading.local()


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
  """

  def __init__(self,
               group_key_start=1,
               instance_key_start=100,
               instance_key_with_id_start=10000):
    """Initializes the object.

    Args:
      group_key_start: the starting integer of group key.
      instance_key_start: the starting integer of instance key.
      instance_key_with_id_start: the starting integer of instance key that is
        recorded with an id.
    """
    self._group_key = group_key_start
    self._group_key_table = {}

    # For instance keys with ids
    self._instance_key_id_to_key_table = {}
    self._instance_key_with_id_counter = instance_key_with_id_start

    # For instance keys without ids
    self._instance_key_start = instance_key_start

  def _get_thread_local_object(self):
    # We make instance key without key ids thread local so that it will work
    # with MirroredStrategy and distribute coordinator.
    if not hasattr(_thread_local, 'instance_key'):
      _thread_local.instance_key = self._instance_key_start
    return _thread_local

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
    with _lock:
      if key_id not in self._group_key_table:
        new_key = self._group_key
        self._group_key += 1
        self._group_key_table[key_id] = new_key
    return self._group_key_table[key_id]

  def get_instance_key(self, key_id=None):
    """Returns a new instance key for use in defining a collective op.

    Args:
      key_id: optional string. If set, key will be recorded and the same key
        will be returned when the same key_id is provided. If not, an increasing
        instance key will be returned.
    """
    if key_id:
      with _lock:
        if key_id not in self._instance_key_id_to_key_table:
          self._instance_key_with_id_counter += 1
          self._instance_key_id_to_key_table[key_id] = (
              self._instance_key_with_id_counter)
      return self._instance_key_id_to_key_table[key_id]
    else:
      v = self._get_thread_local_object().instance_key
      self._get_thread_local_object().instance_key += 1
      return v


def build_collective_reduce(input_tensors,
                            num_workers,
                            collective_keys,
                            reduction_op='Add',
                            unary_op='Id'):
  """Build a subgraph that does one full all-reduce, using the collective Op.

  Args:
    input_tensors: tensors within a single worker graph that are to be reduced
      together; must be one per device.
    num_workers: total number of workers with identical independent graphs that
      will be doing this same reduction.  The reduction will actually include
      the corresponding tensors at all these workers.
    collective_keys: a CollectiveKeys object.
    reduction_op: string naming the reduction op.
    unary_op: string naming the unary final op.

  Returns:
    An array of final tensors, one per device, computed by the full reduction.

  Raises:
    ValueError: There must be at least two tensors over all the workers.
  """
  group_size = len(input_tensors) * num_workers
  if group_size < 2:
    return input_tensors
  devices = [t.device for t in input_tensors]
  num_devices = len(devices)
  group_key = collective_keys.get_group_key(devices)
  instance_key = collective_keys.get_instance_key()
  subdiv_offsets = [0]  # TODO(tucker): maybe support non-default subdiv spec

  def collective_all_reduce():
    """Call collective allreduce."""
    assert not context.executing_eagerly()
    out_tensors = []
    for d in range(num_devices):
      with ops.device(devices[d]):
        reduce_op = collective_ops.all_reduce(
            input_tensors[d], group_size, group_key, instance_key, reduction_op,
            unary_op, subdiv_offsets)
        out_tensors.append(reduce_op)
    return out_tensors

  if context.executing_eagerly():
    # Collective ops will block unless they are executed concurrently such as in
    # a graph or a defun.
    collective_all_reduce = def_function.function(collective_all_reduce)
  return collective_all_reduce()


def build_collective_gather(input_tensors, num_workers, collective_keys):
  """Build a subgraph that does one full all-gather, using the collective Op.

  Args:
    input_tensors: tensors within a single worker graph that are to be gathered
      together; must be one per device.
    num_workers: total number of workers with identical independent graphs that
      will be doing this same reduction.  The reduction will actually include
      the corresponding tensors at all these workers.
    collective_keys: a CollectiveKeys object.

  Returns:
    An array of final tensors, one per device, computed by the full gather.

  Raises:
    ValueError: There must be at least two tensors over all the workers.
  """
  group_size = len(input_tensors) * num_workers
  if group_size < 2:
    return input_tensors
  devices = [t.device for t in input_tensors]
  num_devices = len(devices)
  group_key = collective_keys.get_group_key(devices)
  instance_key = collective_keys.get_instance_key()

  def collective_all_gather():
    """Call collective allgather."""
    assert not context.executing_eagerly()
    out_tensors = []
    for d in range(num_devices):
      with ops.device(devices[d]):
        gather_op = collective_ops.all_gather(input_tensors[d], group_size,
                                              group_key, instance_key)
        out_tensors.append(gather_op)
    return out_tensors

  if context.executing_eagerly():
    # Collective ops will block unless they are executed concurrently such as in
    # a graph or a defun.
    collective_all_gather = def_function.function(collective_all_gather)
  return collective_all_gather()


def sum_grad_and_var_all_reduce(grad_and_vars,
                                num_workers,
                                alg,
                                gpu_indices,
                                aux_devices=None,
                                num_shards=1):
  """Apply all-reduce algorithm over specified gradient tensors."""
  with ops.name_scope('allreduce'):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    scaled_grads = [g for g, _ in grad_and_vars]
    if alg == 'nccl':
      summed_grads = nccl_ops.all_sum(scaled_grads)
    elif alg == 'xring':
      summed_grads = all_reduce.build_ring_all_reduce(
          scaled_grads, num_workers, num_shards, gpu_indices, math_ops.add)
    elif alg == 'nccl/xring':
      summed_grads = all_reduce.build_nccl_then_ring(scaled_grads, num_shards,
                                                     math_ops.add)
    elif alg == 'nccl/rechd':
      summed_grads = all_reduce.build_nccl_then_recursive_hd(
          scaled_grads, math_ops.add)
    elif alg == 'nccl/pscpu':
      summed_grads = all_reduce.build_nccl_then_shuffle(
          scaled_grads, aux_devices, math_ops.add, math_ops.add_n)
    elif alg == 'pscpu/pscpu':
      second_gather_devices = aux_devices[:num_shards]
      summed_grads = all_reduce.build_shuffle_then_shuffle(
          scaled_grads, aux_devices, second_gather_devices, math_ops.add_n)
    elif alg in ['pscpu', 'psgpu']:
      summed_grads = all_reduce.build_shuffle_all_reduce(
          scaled_grads, aux_devices, math_ops.add_n)
    else:
      raise ValueError('unsupported all_reduce alg: ', alg)

  result = []
  for (_, v), g in zip(grad_and_vars, summed_grads):
    result.append([g, v])
  return result


def sum_gradients_all_reduce(dev_prefixes, replica_grads, num_workers, alg,
                             num_shards, gpu_indices):
  """Apply all-reduce algorithm over specified gradient tensors.

  Args:
    dev_prefixes: list of prefix strings to use to generate PS device names.
    replica_grads: the gradients to reduce.
    num_workers: number of worker processes across entire job.
    alg: the all-reduce algorithm to apply.
    num_shards: alg-specific sharding factor.
    gpu_indices: indices of local GPUs in order usable for ring-reduce.

  Returns:
    list of reduced tensors
  """
  alg_contains_shuffle = any(n in alg for n in ['pscpu', 'psgpu'])
  is_hierarchical = '/' in alg
  if 'pscpu' in alg:
    aux_devices = [prefix + '/cpu:0' for prefix in dev_prefixes]
  elif 'psgpu' in alg:
    aux_devices = [
        prefix + '/gpu:%d' % i
        for i in range(len(gpu_indices))
        for prefix in dev_prefixes
    ]
  else:
    aux_devices = ['/job:localhost/cpu:0']
  # Auxiliary devices for hierarchical all-reduces.
  aux_device_groups = group_device_names(
      aux_devices, num_shards if alg_contains_shuffle else 1)
  group_index = 0
  reduced_gv_list = []
  for grad_and_vars in zip(*replica_grads):
    reduced_gv_list.append(
        sum_grad_and_var_all_reduce(
            grad_and_vars, num_workers, alg, gpu_indices, aux_devices
            if is_hierarchical else aux_device_groups[group_index], num_shards))
    group_index = (group_index + 1) % len(aux_device_groups)
  new_replica_grads = [list(x) for x in zip(*reduced_gv_list)]
  return new_replica_grads


def extract_ranges(index_list, range_size_limit=32):
  """Extract consecutive ranges and singles from index_list.

  Args:
    index_list: List of monotone increasing non-negative integers.
    range_size_limit: Largest size range to return.  If a larger
      consecutive range exists, it will be returned as multiple
      ranges.

  Returns:
    (ranges, singles) where ranges is a list of [first, last] pairs of
      consecutive elements in index_list, and singles is all of the
      other elements, in original order.
  """
  if not index_list:
    return [], []
  first = index_list[0]
  last = first
  ranges = []
  singles = []
  for i in index_list[1:]:
    if i == last + 1 and (last - first) <= range_size_limit:
      last = i
    else:
      if last > first:
        ranges.append([first, last])
      else:
        singles.append(first)
      first = i
      last = i
  if last > first:
    ranges.append([first, last])
  else:
    singles.append(first)
  return ranges, singles


GradPackTuple = pycoll.namedtuple('GradPackTuple', 'indices vars shapes')


def pack_range(key, packing, grad_vars, rng):
  """Form the concatenation of a specified range of gradient tensors.

  Args:
    key: Value under which to store meta-data in packing that will be used
      later to restore the grad_var list structure.
    packing: Dict holding data describing packed ranges of small tensors.
    grad_vars: List of (grad, var) pairs for one replica.
    rng: A pair of integers giving the first, last indices of a consecutive
      range of tensors to be packed.

  Returns:
    A tensor that is the concatenation of all the specified small tensors.
  """
  to_pack = grad_vars[rng[0]:rng[1] + 1]
  members = []
  variables = []
  restore_shapes = []
  with ops.name_scope('pack'):
    for g, v in to_pack:
      variables.append(v)
      restore_shapes.append(g.shape)
      with ops.device(g.device):
        members.append(array_ops.reshape(g, [-1]))
    packing[key] = GradPackTuple(
        indices=range(rng[0], rng[1] + 1),
        vars=variables,
        shapes=restore_shapes)
    with ops.device(members[0].device):
      return array_ops.concat(members, 0)


def unpack_grad_tuple(gv, gpt):
  """Unpack a previously packed collection of gradient tensors.

  Args:
    gv: A (grad, var) pair to be unpacked.
    gpt: A GradPackTuple describing the packing operation that produced gv.

  Returns:
    A list of (grad, var) pairs corresponding to the values that were
     originally packed into gv, maybe following subsequent operations like
     reduction.
  """
  elt_widths = [x.num_elements() for x in gpt.shapes]
  with ops.device(gv[0][0].device):
    with ops.name_scope('unpack'):
      splits = array_ops.split(gv[0], elt_widths)
      unpacked_gv = []
      for idx, s in enumerate(splits):
        unpacked_gv.append((array_ops.reshape(s, gpt.shapes[idx]),
                            gpt.vars[idx]))
  return unpacked_gv


def pack_small_tensors(replica_grads, max_bytes=0, max_group=0):
  """Concatenate small gradient tensors together for reduction.

  Args:
    replica_grads: List of lists of (gradient, variable) tuples.
    max_bytes: Int giving max number of bytes in a tensor that
      may be considered small.
    max_group: Int giving max number of small tensors that may be
      concatenated into one new tensor.

  Returns:
    new_replica_grads, packing where new_replica_grads is identical to
      replica_grads except that all feasible small_tensors have been removed
      from their places and concatenated into larger tensors that are
      now in the front of the list for each replica, and packing contains
      the data necessary to restore the replica_grads structure.

  Look through the first replica for gradients of the same type (float),
  and small size, that are all sequential.  For each such group,
  replace by a new tensor that is a flattened concatenation.  Note
  that the corresponding variable will be absent, which doesn't matter
  because it isn't used during all-reduce.

  Requires:
    Every gv_list in replicas must have isomorphic structure including identical
      tensor sizes and types.
  """
  small_indices = []
  large_indices = []
  for idx, (g, _) in enumerate(replica_grads[0]):
    if g.dtype == dtypes.float32 and (4 * g.shape.num_elements()) <= max_bytes:
      small_indices.append(idx)
    else:
      large_indices.append(idx)
  small_ranges, small_singles = extract_ranges(
      small_indices, range_size_limit=max_group)
  large_indices = sorted(large_indices + small_singles)
  num_gv = len(replica_grads[0])
  packing = {}
  if small_ranges:
    new_replica_grads = []
    for dev_idx, gv_list in enumerate(replica_grads):
      assert len(gv_list) == num_gv
      new_gv_list = []
      for r in small_ranges:
        key = '%d:%d' % (dev_idx, len(new_gv_list))
        new_gv_list.append((pack_range(key, packing, gv_list, r),
                            'packing_var_placeholder'))
      for i in large_indices:
        new_gv_list.append(gv_list[i])
      new_replica_grads.append(new_gv_list)
    return new_replica_grads, packing
  else:
    return replica_grads, None


def unpack_small_tensors(replica_grads, packing):
  """Undo the structure alterations to replica_grads done by pack_small_tensors.

  Args:
    replica_grads: List of List of (grad, var) tuples.
    packing: A dict generated by pack_small_tensors describing the changes
      it made to replica_grads.

  Returns:
    new_replica_grads: identical to replica_grads except that concatenations
      of small tensors have been split apart and returned to their original
      positions, paired with their original variables.
  """
  if not packing:
    return replica_grads
  new_replica_grads = []
  num_devices = len(replica_grads)
  num_packed = len(packing.keys()) // num_devices
  for dev_idx, gv_list in enumerate(replica_grads):
    gv_list = list(gv_list)
    new_gv_list = gv_list[num_packed:]
    for i in range(num_packed):
      k = '%d:%d' % (dev_idx, i)
      gpt = packing[k]
      gv = unpack_grad_tuple(gv_list[i], gpt)
      for gi, idx in enumerate(gpt.indices):
        assert idx == gpt.indices[gi]
        new_gv_list.insert(idx, gv[gi])
    new_replica_grads.append(new_gv_list)
  return new_replica_grads


def aggregate_tensors_or_indexed_slices(values, accumulation_fn=math_ops.add_n):
  """Aggregate tensors using `accumulation_fn` and IndexedSlices via concat."""
  if any(isinstance(v, ops.IndexedSlices) for v in values):
    return gradients_util._AggregateIndexedSlicesGradients(values)  # pylint: disable=protected-access
  else:
    return accumulation_fn(values)


def divide_by_n_tensors_or_indexed_slices(value, n):
  if isinstance(value, ops.IndexedSlices):
    value = gradients_util._HandleNestedIndexedSlices(value)  # pylint: disable=protected-access
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


def contains_indexed_slices(value):
  """Check whether the value is `IndexedSlices` or contains `IndexedSlices`."""
  if isinstance(value, ops.IndexedSlices):
    return True
  elif isinstance(value, (list, tuple)) and value:
    return any(contains_indexed_slices(v) for v in value)
  elif isinstance(value, value_lib.DistributedValues):
    return contains_indexed_slices(value.values)
  else:
    return False


def is_indexed_slices(value):
  if isinstance(value, ops.IndexedSlices):
    return True
  assert isinstance(value, value_lib.DistributedValues)
  return all([isinstance(v, ops.IndexedSlices) for v in value.values])


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
      the values and postions in the returned list.

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

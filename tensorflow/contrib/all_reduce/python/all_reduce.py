# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities to construct a TF subgraph implementing distributed All-Reduce."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.contrib import nccl
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def _flatten_tensors(tensors):
  """Check tensors for isomorphism and flatten.

  Args:
    tensors: list of T @{tf.Tensor} which must all have the same shape.

  Returns:
    tensors: a list of T @{tf.Tensor} which are flattened (1D) views of tensors
    shape: the original shape of each element of input tensors

  Raises:
    ValueError: tensors are empty or non-isomorphic or have unknown shape.
  """
  if not tensors:
    raise ValueError("tensors cannot be empty")
  shape = tensors[0].shape
  for tensor in tensors:
    shape = shape.merge_with(tensor.shape)
  if not shape.is_fully_defined():
    raise ValueError("Tensors must have statically known shape.")
  if len(shape) != 1:
    reshaped = []
    for t in tensors:
      with ops.colocate_with(t):
        reshaped.append(array_ops.reshape(t, [-1]))
    tensors = reshaped
  return tensors, shape


def _reshape_tensors(tensors, shape):
  """Reshape tensors flattened by _flatten_tensors.

  Args:
    tensors: list of T @{tf.Tensor} of identical length 1D tensors.
    shape: list of integers describing the desired shape.  Product of
      the elements must equal the length of each tensor.

  Returns:
    list of T @{tf.Tensor} which are the reshaped inputs.
  """
  reshaped = []
  for t in tensors:
    with ops.colocate_with(t):
      reshaped.append(array_ops.reshape(t, shape))
  return reshaped


def _padded_split(tensor, pieces):
  """Like split for 1D tensors but pads-out case where len % pieces != 0.

  Args:
    tensor: T @{tf.Tensor} that must be 1D.
    pieces: a positive integer specifying the number of pieces into which
      tensor should be split.

  Returns:
    list of T @{tf.Tensor} of length pieces, which hold the values of
      thin input tensor, in order.  The final tensor may
      be zero-padded on the end to make its size equal to those of all
      of the other tensors.

  Raises:
    ValueError: The input tensor is not 1D.
  """
  shape = tensor.shape
  if 1 != len(shape):
    raise ValueError("input tensor must be 1D")
  tensor_len = shape[0].value
  with ops.colocate_with(tensor):
    if tensor_len % pieces != 0:
      # pad to an even length
      chunk_size = 1 + tensor_len // pieces
      if pieces > tensor_len:
        # This is an edge case that should not come up in practice,
        # i.e. a different reduction algorithm would be better,
        # but we'll make it work just for completeness.
        pad_len = pieces - tensor_len
        extended_whole = array_ops.concat(
            [tensor, array_ops.zeros([pad_len], dtype=tensor.dtype)], 0)
        parts = array_ops.split(extended_whole, pieces)
        return parts, pad_len
      elif (pieces - 1) * chunk_size >= tensor_len:
        # Another edge case of limited real interest.
        pad_len = (pieces * chunk_size) % tensor_len
        extended_whole = array_ops.concat(
            [tensor, array_ops.zeros([pad_len], dtype=tensor.dtype)], 0)
        parts = array_ops.split(extended_whole, pieces)
        return parts, pad_len
      else:
        last_chunk_size = tensor_len - (pieces - 1) * chunk_size
        pad_len = chunk_size - last_chunk_size
        piece_lens = [chunk_size for _ in range(pieces - 1)] + [last_chunk_size]
        parts = array_ops.split(tensor, piece_lens)
        parts[-1] = array_ops.concat(
            [parts[-1], array_ops.zeros([pad_len], dtype=tensor.dtype)], 0)
        return parts, pad_len
    else:
      return array_ops.split(tensor, pieces), 0


def _strip_padding(tensors, pad_len):
  """Strip the suffix padding added by _padded_split.

  Args:
    tensors: list of T @{tf.Tensor} of identical length 1D tensors.
    pad_len: number of elements to be stripped from the end of each tensor.

  Returns:
    list of T @{tf.Tensor} which are the stripped inputs.

  Raises:
    ValueError: tensors must be a non-empty list of 1D tensors, and
      each must be longer than pad_len.
  """
  if not tensors:
    raise ValueError("tensors cannot be empty")
  shape = tensors[0].shape
  if len(shape) > 1:
    raise ValueError("tensors must be 1D")
  prefix_len = int(shape[0] - pad_len)
  if prefix_len < 0:
    raise ValueError("pad_len longer than tensor")
  stripped = []
  for t in tensors:
    with ops.colocate_with(t):
      stripped.append(array_ops.slice(t, [0], [prefix_len]))
  return stripped


def _ragged_split(tensor, pieces):
  """Like split for 1D tensors but allows case where len % pieces != 0.

  Args:
    tensor: T @{tf.Tensor} that must be 1D.
    pieces: a positive integer specifying the number of pieces into which
      tensor should be split.

  Returns:
    list of T @{tf.Tensor} of length pieces, which hold the values of
      the input tensor, in order.  The final tensor may be shorter
      than the others, which will all be of equal length.

  Raises:
    ValueError: input tensor must be 1D.
  """
  shape = tensor.shape
  if 1 != len(shape):
    raise ValueError("input tensor must be 1D")
  tensor_len = shape[0].value
  chunk_size = tensor_len // pieces
  with ops.colocate_with(tensor):
    if tensor_len != (pieces * chunk_size):
      # last piece will be short
      assert pieces > 1
      last_chunk_size = tensor_len - ((pieces - 1) * chunk_size)
      assert last_chunk_size > 0
      piece_lens = [chunk_size for _ in range(pieces - 1)] + [last_chunk_size]
      return array_ops.split(tensor, piece_lens)
    else:
      return array_ops.split(tensor, pieces)


def _ring_permutations(num_workers, num_subchunks, gpu_perm):
  """"Generate an array of device index arrays, one for each subchunk.

  In the basic ring reduction algorithm there are size(T)/num_devices
  data chunks and each device process one chunk per tick, i.e. sending
  one chunk and receiving one chunk.  The idea of subchunking is that
  each device processes num_subchunks smaller data regions per tick,
  and the ring rank permutation is different for each subchunk index
  so that a device is potentially sending to and receiving from
  num_subchunks different other devices at each tick.  Where multiple
  independent data channels exist between devices, this strategy
  supplies a method of using them in parallel.

  Args:
    num_workers: number of worker tasks
    num_subchunks: number of subchunks into which to divide each per-GPU chunk.
    gpu_perm: an array of integers in [0, num_gpus-1] giving the default
      ring order of GPUs at each worker.  Other permutations will be generated
      by rotating this array and splicing together per-worker instances.

  Raises:
    ValueError: the number of subchunks may not exceed the number of GPUs.

  Returns:
    pred_by_s_d: list of lists that maps (by index) from (subchunk, dev) to
        preceding device in the permutation for that subchunk.  The
        device index of GPU i at worker j is i + (j * num_gpus).
    rank_by_s_d: list of lists that maps (by index) from (subchunk, dev) to
       local rank of device d in the permutation for that subchunk.
  """
  num_gpus = len(gpu_perm)
  devices = num_workers * num_gpus
  if devices == 0:
    return [], []
  if num_subchunks > num_gpus:
    raise ValueError(
        "num_subchunks %d must be <= num_gpus %d" % (num_subchunks, num_gpus))
  rotation_interval = max(1, int(num_gpus / num_subchunks))
  perms_by_s = []
  for s in range(0, num_subchunks):
    full_order = []
    offset = s * rotation_interval
    for w in range(0, num_workers):
      default_order = [(w * num_gpus) + i for i in gpu_perm]
      dev_order = default_order[offset:] + default_order[:offset]
      full_order += dev_order
    perms_by_s.append(full_order)
  pred_by_s_d = [[-1 for d in range(0, devices)]
                 for s in range(0, num_subchunks)]
  rank_by_s_d = [[-1 for d in range(0, devices)]
                 for s in range(0, num_subchunks)]
  for s in range(0, num_subchunks):
    for d in range(0, devices):
      for t in range(0, devices):
        if d == perms_by_s[s][t]:
          rank_by_s_d[s][d] = t
          pred_by_s_d[s][d] = perms_by_s[s][(t + devices - 1) % devices]
          break
  return (pred_by_s_d, rank_by_s_d)


def build_ring_all_reduce(input_tensors, num_workers, num_subchunks,
                          gpu_perm, red_op, un_op=None):
  """Construct a subgraph performing a ring-style all-reduce of input_tensors.

  Args:
    input_tensors: a list of T @{tf.Tensor} objects, which must all
      have the same shape and type.
    num_workers: number of worker tasks spanned by input_tensors.
    num_subchunks: number of subchunks each device should process in one tick.
    gpu_perm: a list of ints giving a ring-wise rank ordering of GPUs at
      each worker.  All workers must have the same number of
      GPUs with the same rank ordering.  If NVLINK is available, this should
      be a ring order supported by NVLINK edges.
    red_op: a binary operator for elementwise reduction.
    un_op: an optional unary operator to apply to fully reduced values.

  Raises:
    ValueError: empty input_tensors or they don't all have same
    size.

  Returns:
    a list of T @{tf.Tensor} identical sum-reductions of input_tensors.
  """
  if len(input_tensors) < 2:
    raise ValueError("input_tensors must be length 2 or longer")
  input_tensors, shape = _flatten_tensors(input_tensors)
  devices = [t.device for t in input_tensors]
  (pred_by_s_d, rank_by_s_d) = _ring_permutations(
      num_workers, num_subchunks, gpu_perm)
  chunks_by_dev, pad_len = _build_ring_gather(
      input_tensors, devices,
      num_subchunks, pred_by_s_d, rank_by_s_d, red_op)
  if un_op:
    chunks_by_dev = _apply_unary_to_chunks(un_op, chunks_by_dev)
  output_tensors = _build_ring_scatter(pred_by_s_d, rank_by_s_d,
                                       chunks_by_dev)
  if pad_len > 0:
    output_tensors = _strip_padding(output_tensors, pad_len)
  if len(shape) != 1:
    output_tensors = _reshape_tensors(output_tensors, shape)
  return output_tensors


def _build_ring_gather(input_tensors, devices, num_subchunks,
                       pred_by_s_d, rank_by_s_d, red_op):
  """Construct a subgraph for the first (reduction) pass of ring all-reduce.

  Args:
    input_tensors: a list of T @{tf.Tensor} 1D input tensors of same
      shape and type.
    devices: array of device name strings
    num_subchunks: number of subchunks each device should process in one tick.
    pred_by_s_d: as produced by _ring_permutations
    rank_by_s_d: as produced by _ring_permutations
    red_op: a binary operator for elementwise reduction

  Raises:
    ValueError: tensors must all be one dimensional.

  Returns:
    list of list of T @{tf.Tensor} of (partially) reduced values where
    exactly num_subchunks chunks at each device are fully reduced.
  """
  num_devices = len(input_tensors)
  if num_devices == 0:
    return []
  if num_devices == 1:
    return input_tensors
  shape = input_tensors[0].shape
  if 1 != len(shape):
    raise ValueError("input tensors must be 1D")
  num_chunks = num_devices * num_subchunks
  num_ticks = num_devices - 1
  # Initialize chunks_by_dev with splits of the input tensors.
  chunks_by_dev = []
  split_pad_len = 0
  for d in range(0, num_devices):
    with ops.device(devices[d]):
      splits, split_pad_len = _padded_split(input_tensors[d], num_chunks)
      chunks_by_dev.append(splits)
  # Reduction phase
  for tick in range(0, num_ticks):
    # One new partial reduction for every chunk
    new_partial_reductions = [None for _ in range(0, num_chunks)]
    # Compute reductions with respect to last tick's values
    for d in range(0, num_devices):
      with ops.device(devices[d]):
        for s in range(0, num_subchunks):
          rank = rank_by_s_d[s][d]
          seg_index = (rank + num_devices - (2 + tick)) % num_devices
          pred_dev = pred_by_s_d[s][d]
          chunk_index = (seg_index * num_subchunks) + s
          new_partial_reductions[chunk_index] = red_op(
              chunks_by_dev[pred_dev][chunk_index],
              chunks_by_dev[d][chunk_index])
    # Update chunks_by_dev with the new values at the end of the tick.
    for d in range(0, num_devices):
      for s in range(0, num_subchunks):
        rank = rank_by_s_d[s][d]
        seg_index = (rank + num_devices - (2 + tick)) % num_devices
        chunk_index = (seg_index * num_subchunks) + s
        chunks_by_dev[d][chunk_index] = new_partial_reductions[chunk_index]
  return chunks_by_dev, split_pad_len


def _apply_unary_to_chunks(f, chunks_by_dev):
  """Apply a unary op to each tensor in chunks_by_dev, on same device.

  Args:
    f: a unary function over T @{tf.Tensor}.
    chunks_by_dev: list of lists of T @{tf.Tensor}.

  Returns:
    new list of lists of T @{tf.Tensor} with the same structure as
    chunks_by_dev containing the derived tensors.
  """
  output = []
  for x in chunks_by_dev:
    with ops.colocate_with(x[0]):
      output.append([f(t) for t in x])
  return output


def _build_ring_scatter(pred_by_s_d, rank_by_s_d,
                        chunks_by_dev):
  """Construct subgraph for second (scatter) pass of ring all-reduce.

  Args:
    pred_by_s_d: as produced by _ring_permutations
    rank_by_s_d: as produced by _ring_permutations
    chunks_by_dev: list of list of T @{tf.Tensor} indexed by ints
      (device, chunk)

  Raises:
    ValueError: chunks_by_dev is not well-formed

  Returns:
    list of T @{tf.Tensor} which are the fully reduced tensors, one
    at each device corresponding to the outer dimension of chunks_by_dev.
  """
  num_devices = len(chunks_by_dev)
  num_chunks = len(chunks_by_dev[0])
  if 0 != num_chunks % num_devices:
    raise ValueError(
        "Expect number of chunks per device to be divisible by num_devices")
  num_subchunks = int(num_chunks / num_devices)
  num_ticks = num_devices - 1
  for tick in range(0, num_ticks):
    passed_values = [None for _ in range(0, num_chunks)]
    for d in range(0, num_devices):
      with ops.colocate_with(chunks_by_dev[d][0]):
        for s in range(0, num_subchunks):
          rank = rank_by_s_d[s][d]
          seg_index = (rank + num_devices - (1 + tick)) % num_devices
          pred_dev = pred_by_s_d[s][d]
          chunk_index = (seg_index * num_subchunks) + s
          passed_values[chunk_index] = array_ops.identity(
              chunks_by_dev[pred_dev][chunk_index])
    for d in range(0, num_devices):
      for s in range(0, num_subchunks):
        rank = rank_by_s_d[s][d]
        seg_index = (rank + num_devices - (1 + tick)) % num_devices
        chunk_index = (seg_index * num_subchunks) + s
        chunks_by_dev[d][chunk_index] = passed_values[chunk_index]
  # Join chunks at each device.
  output = []
  for x in chunks_by_dev:
    with ops.colocate_with(x[0]):
      output.append(array_ops.concat(x, 0))
  return output


def build_recursive_hd_all_reduce(input_tensors, red_op, un_op=None):
  """Construct a subgraph for recursive halving-doubling all-reduce.

  The recursive halving-doubling algorithm is described in
  http://www.mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf

  The concept is to arrange the participating n devices in
  a linear sequence where devices exchange data pairwise
  with one other device in each round.  During the gather
  phase there are lg(n) rounds where devices exchange
  increasingly smaller sub-tensors with another device
  at increasingly greater distances, until at the top
  each device has 1/n of the fully reduced values.  During the
  scatter phase each device exchanges its fully reduced
  sub-tensor (which doubles in length at each round)
  with one other device at increasingly smaller distances
  until each device has all of the fully reduced values.

  Note: this preliminary version requires that len(input_tensors) be a
    power of 2.  TODO(tucker): relax this restriction.  Also, the
    number of elements in each tensor must be divisible by 2^h where h
    is the number of hops in each phase.  This will also be relaxed in
    the future with edge-case specific logic.

  Args:
    input_tensors: list of T @{tf.Tensor} to be elementwise reduced.
    red_op: a binary elementwise reduction Op.
    un_op: an optional unary elementwise Op to apply to reduced values.

  Returns:
    list of T @{tf.Tensor} which are the fully reduced tensors, one
    at each device of input_tensors.

  Raises:
    ValueError: num_devices not a power of 2, or tensor len not divisible
    by 2 the proper number of times.
  """
  devices = [t.device for t in input_tensors]
  input_tensors, shape = _flatten_tensors(input_tensors)
  reduced_shards = _build_recursive_hd_gather(input_tensors, devices, red_op)
  if un_op:
    reduced_shards = [un_op(t) for t in reduced_shards]
  output_tensors = _build_recursive_hd_scatter(reduced_shards, devices)
  if len(shape) != 1:
    output_tensors = _reshape_tensors(output_tensors, shape)
  return output_tensors


def _build_recursive_hd_gather(input_tensors, devices, red_op):
  """Construct the gather phase of recursive halving-doubling all-reduce.

  Args:
    input_tensors: list of T @{tf.Tensor} to be elementwise reduced.
    devices: a list of strings naming the devices hosting input_tensors,
      which will also be used to host the (partial) reduction values.
    red_op: a binary elementwise reduction Op.

  Returns:
    list of T @{tf.Tensor} which are the fully reduced tensor shards.

  Raises:
    ValueError: num_devices not a power of 2, or tensor len not divisible
    by 2 the proper number of times.
  """
  num_devices = len(devices)
  num_hops = int(math.log(num_devices, 2))
  if num_devices != (2 ** num_hops):
    raise ValueError("num_devices must be a power of 2")
  chunks = input_tensors
  for h in range(0, num_hops):
    span = 2 ** h
    group_size = span * 2
    new_chunks = [[] for _ in devices]
    for d in range(0, num_devices):
      if (d % group_size) >= (group_size / 2):
        # skip right half of a pair
        continue
      left_dev = devices[d]
      right_dev = devices[d + span]
      left_split = array_ops.split(chunks[d], 2)
      right_split = array_ops.split(chunks[d+span], 2)
      with ops.device(left_dev):
        new_chunks[d] = red_op(left_split[0], right_split[0])
      with ops.device(right_dev):
        new_chunks[d + span] = red_op(left_split[1], right_split[1])
    chunks = new_chunks
  return chunks


def _build_recursive_hd_scatter(input_tensors, devices):
  """Construct the scatter phase of recursive halving-doublng all-reduce.

  Args:
    input_tensors: list of T @{tf.Tensor} that are fully-reduced shards.
    devices: a list of strings naming the devices on which the reconstituted
      full tensors should be placed.

  Returns:
    list of T @{tf.Tensor} which are the fully reduced tensors.
  """
  num_devices = len(devices)
  num_hops = int(math.log(num_devices, 2))
  assert num_devices == (2 ** num_hops), "num_devices must be a power of 2"
  chunks = input_tensors
  for h in reversed(range(0, num_hops)):
    span = 2 ** h
    group_size = span * 2
    new_chunks = [[] for _ in devices]
    for d in range(0, num_devices):
      if (d % group_size) >= (group_size / 2):
        # skip right half of a pair
        continue
      left_idx = d
      right_idx = d + span
      left_dev = devices[left_idx]
      right_dev = devices[right_idx]
      with ops.device(left_dev):
        new_chunks[left_idx] = array_ops.concat([chunks[left_idx],
                                                 chunks[right_idx]], 0)
      with ops.device(right_dev):
        new_chunks[right_idx] = array_ops.concat([chunks[left_idx],
                                                  chunks[right_idx]], 0)
    chunks = new_chunks
  return chunks


def build_shuffle_all_reduce(input_tensors, gather_devices, red_op, un_op=None):
  """Construct a subgraph for shuffle all-reduce.

  Shuffle reduce is essentially the algorithm implemented when using
  parameter servers.  Suppose tensor length is n, there are d devices
  and g gather shards.  Each device sends a n/g length sub-tensor to
  each gather shard.  The gather shards perform a reduction across d
  fragments, then broadcast the result back to each device.  The
  devices then join the g fully reduced fragments they receive from
  the shards.  The gather shards could perform d-1 pairwise
  reductions, or one d-way reduction.  The first is better where
  reduction Op time is low compared to transmission time, the second
  better in the other case.

  Args:
    input_tensors: list of T @(tf.Tensor} values to be reduced.
    gather_devices: list of names of devices on which reduction shards
      should be placed.
    red_op: an n-array elementwise reduction Op
    un_op: optional elementwise unary Op to be applied to fully-reduced values.

  Returns:
    list of T @{tf.Tensor} which are the fully reduced tensors.
  """
  input_tensors, shape = _flatten_tensors(input_tensors)
  dst_devices = [t.device for t in input_tensors]
  reduced_shards = _build_shuffle_gather(input_tensors, gather_devices,
                                         red_op, un_op)
  output_tensors = _build_shuffle_scatter(reduced_shards, dst_devices)
  if len(shape) != 1:
    output_tensors = _reshape_tensors(output_tensors, shape)
  return output_tensors


def _build_shuffle_gather(input_tensors, gather_devices, red_op, un_op=None):
  """Construct the gather (concentrate and reduce) phase of shuffle all-reduce.

  Args:
    input_tensors: list of T @(tf.Tensor} values to be reduced.
    gather_devices: list of names of devices on which reduction shards
      should be placed.
    red_op: the binary reduction Op
    un_op: optional elementwise unary Op to be applied to fully-reduced values.

  Returns:
    list of T @{tf.Tensor} which are the fully reduced shards.

  Raises:
    ValueError: inputs not well-formed.
  """
  num_source_devices = len(input_tensors)
  num_gather_devices = len(gather_devices)
  shape = input_tensors[0].shape
  if len(shape) != 1:
    raise ValueError("input_tensors must be 1D")
  shards_by_source = []
  for d in range(0, num_source_devices):
    with ops.colocate_with(input_tensors[d]):
      shards_by_source.append(
          _ragged_split(input_tensors[d], num_gather_devices))
  reduced_shards = []
  for d in range(0, num_gather_devices):
    with ops.device(gather_devices[d]):
      values = [s[d] for s in shards_by_source]
      red_shard = red_op(values)
      if un_op:
        red_shard = un_op(red_shard)
      reduced_shards.append(red_shard)
  return reduced_shards


def _build_shuffle_scatter(reduced_shards, dst_devices):
  """Build the scatter phase of shuffle all-reduce.

  Args:
    reduced_shards:  list of T @(tf.Tensor} fully reduced shards
    dst_devices: list of names of devices at which the fully-reduced value
      should be reconstituted.

  Returns:
    list of T @{tf.Tensor} scattered tensors.
  """
  num_devices = len(dst_devices)
  out_tensors = []
  for d in range(0, num_devices):
    with ops.device(dst_devices[d]):
      out_tensors.append(array_ops.concat(reduced_shards, 0))
  return out_tensors


def _split_by_task(devices, values):
  """Partition devices and values by common task.

  Args:
    devices: list of device name strings
    values: list of T @{tf.tensor} of same length as devices.

  Returns:
    (per_task_devices, per_task_values) where both values are
    lists of lists with isomorphic structure: the outer list is
    indexed by task, and the inner list has length of the number
    of values belonging to that task.  per_task_devices contains
    the specific devices to which the values are local, and
    per_task_values contains the corresponding values.

  Raises:
    ValueError: devices must be same length as values.
  """
  num_devices = len(devices)
  if num_devices != len(values):
    raise ValueError("len(devices) must equal len(values)")
  per_task_devices = collections.OrderedDict()
  per_task_values = collections.OrderedDict()
  for d in range(num_devices):
    d_spec = device_lib.DeviceSpec.from_string(devices[d])
    if not hasattr(d_spec, "task") or d_spec.task is None:
      assert False, "failed to parse device %s" % devices[d]
    index = (d_spec.job or "localhost", d_spec.replica or 0, d_spec.task)
    if index not in per_task_devices:
      per_task_devices[index] = []
      per_task_values[index] = []
    per_task_devices[index].append(devices[d])
    per_task_values[index].append(values[d])

  return (list(per_task_devices.values()), list(per_task_values.values()))


def build_nccl_all_reduce(input_tensors, red_op, un_op=None):
  """Build a subgraph that does one full all-reduce, using NCCL.

  Args:
    input_tensors: list of T @{tf.Tensor} of same-shape and type values to
      be reduced.
    red_op: binary elementwise reduction operator.  Must be one of
      {tf.add}
    un_op: optional unary elementwise Op to apply to fully-reduce values.

  Returns:
    list of T @{tf.Tensor} of reduced values.

  Raises:
    ValueError: red_op not supported.
  """
  if red_op == math_ops.add:
    output_tensors = nccl.all_sum(input_tensors)
  else:
    raise ValueError("red_op not supported by NCCL all-reduce: ", red_op)
  if un_op:
    un_op_wrapped = []
    for t in output_tensors:
      with ops.colocate_with(t):
        un_op_wrapped.append(un_op(t))
    output_tensors = un_op_wrapped
  return output_tensors


def _build_nccl_hybrid(input_tensors, red_op, upper_level_f):
  """Construct a subgraph for NCCL hybrid all-reduce.

  Args:
    input_tensors: list of T @{tf.Tensor} of same-shape and type values to
      be reduced.
    red_op: binary elementwise reduction operator.
    upper_level_f: function for reducing one value per worker, across
      workers.

  Returns:
    list of T @{tf.Tensor} of reduced values.

  Raises:
    ValueError: inputs not well-formed.
  """
  input_tensors, shape = _flatten_tensors(input_tensors)
  devices = [t.device for t in input_tensors]
  per_worker_devices, per_worker_values = _split_by_task(devices, input_tensors)
  num_workers = len(per_worker_devices)
  up_values = [None for w in range(0, num_workers)]
  up_devices = up_values[:]
  down_values = up_values[:]
  # First stage: reduce within each worker using NCCL
  for w in range(0, num_workers):
    worker_values = build_nccl_all_reduce(per_worker_values[w], red_op)
    # NOTE: these reductions will not run to completion unless
    # every output value is used.  Since we only need one, we
    # need to put control dependencies on the rest.
    with ops.control_dependencies(worker_values):
      with ops.device(worker_values[0].device):
        up_values[w] = array_ops.identity(worker_values[0])
      up_devices[w] = per_worker_devices[w][0]
  # Second stage: Apply upper_level_f to reduce across first device at
  # each worker
  level_2_output = upper_level_f(up_values)
  # Third stage: propagate within each worker using NCCL Broadcast
  for w in range(0, num_workers):
    dst_tensors = []
    with ops.device(per_worker_devices[w][0]):
      broadcast_src = nccl.broadcast(array_ops.identity(level_2_output[w]))
    for d in per_worker_devices[w]:
      with ops.device(d):
        dst_tensors.append(array_ops.identity(broadcast_src))
    down_values[w] = dst_tensors
  output_tensors = [v for sublist in down_values for v in sublist]
  if len(shape) != 1:
    output_tensors = _reshape_tensors(output_tensors, shape)
  return output_tensors


def _reduce_non_singleton(input_tensors, red_f, un_op):
  """If input_tensors has more than one element apply red_f, else apply un_op."""
  if len(input_tensors) > 1:
    return red_f(input_tensors)
  else:
    if not un_op:
      return input_tensors
    output_tensors = []
    for t in input_tensors:
      with ops.colocate_with(t):
        output_tensors.append(un_op(t))
    return output_tensors


def build_nccl_then_ring(input_tensors, subdiv, red_op, un_op=None):
  """Construct hybrid of NCCL within workers, Ring across workers."""
  def upper_builder(y):
    return build_ring_all_reduce(y, len(y), subdiv, [0], red_op, un_op)
  def upper_level_f(x):
    return _reduce_non_singleton(x, upper_builder, un_op)
  return _build_nccl_hybrid(input_tensors, red_op, upper_level_f)


def build_nccl_then_recursive_hd(input_tensors, red_op, un_op=None):
  """Construct hybrid of NCCL within workers, Recursive-HD across workers."""
  upper_level_f = lambda x: build_recursive_hd_all_reduce(x, red_op, un_op)
  return _build_nccl_hybrid(input_tensors, red_op, upper_level_f)


def build_nccl_then_shuffle(input_tensors, gather_devices, nccl_red_op,
                            shuffle_red_op, un_op=None):
  """Construct hybrid of NCCL within workers, Shuffle across workers."""
  upper_level_f = lambda x: build_shuffle_all_reduce(x, gather_devices,
                                                     shuffle_red_op, un_op)
  return _build_nccl_hybrid(input_tensors, nccl_red_op, upper_level_f)


def _build_shuffle_hybrid(input_tensors, gather_devices, red_op, upper_level_f):
  """Construct a subgraph for Shuffle hybrid all-reduce.

  Args:
    input_tensors: list of T @{tf.Tensor} of same-shape and type values to
      be reduced.
    gather_devices: list of device names on which to host gather shards.
    red_op: binary elementwise reduction operator.
    upper_level_f: function for reducing one value per worker, across
      workers.

  Returns:
    list of T @{tf.Tensor} of reduced values.

  Raises:
    ValueError: inputs not well-formed.
  """
  input_tensors, shape = _flatten_tensors(input_tensors)
  # First stage, reduce across each worker using gather_devices.
  devices = [t.device for t in input_tensors]
  per_worker_devices, per_worker_values = _split_by_task(devices, input_tensors)
  num_workers = len(per_worker_devices)
  up_values = []
  if len(gather_devices) != num_workers:
    raise ValueError("For shuffle hybrid, gather_devices must contain one "
                     "device per worker. ")
  for w in range(0, num_workers):
    reduced_shards = _build_shuffle_gather(
        per_worker_values[w], [gather_devices[w]], red_op)
    up_values.append(reduced_shards[0])
  # Second stage, apply upper_level_f.
  level_2_output = upper_level_f(up_values)
  # Third stage, apply shuffle scatter at each worker.
  output_tensors = []
  for w in range(0, num_workers):
    output_tensors += _build_shuffle_scatter(
        [level_2_output[w]], per_worker_devices[w])
  if len(shape) != 1:
    output_tensors = _reshape_tensors(output_tensors, shape)
  return output_tensors


def build_shuffle_then_ring(input_tensors, gather_devices, subdiv,
                            red_n_op, red_op, un_op=None):
  """Construct hybrid of Shuffle within workers, Ring across workers."""
  def upper_builder(tensors):
    return build_ring_all_reduce(tensors, len(tensors), subdiv, [0],
                                 red_op, un_op)
  def upper_level_f(tensors):
    return _reduce_non_singleton(tensors, upper_builder, un_op)
  return _build_shuffle_hybrid(
      input_tensors, gather_devices, red_n_op, upper_level_f)


def build_shuffle_then_shuffle(input_tensors, first_gather_devices,
                               second_gather_devices, red_op, un_op=None):
  """Construct hybrid of Shuffle within workers, Shuffle across workers."""
  def upper_builder(tensors):
    return build_shuffle_all_reduce(tensors, second_gather_devices,
                                    red_op, un_op)
  def upper_level_f(tensors):
    return _reduce_non_singleton(tensors, upper_builder, un_op)
  return _build_shuffle_hybrid(
      input_tensors, first_gather_devices, red_op, upper_level_f)

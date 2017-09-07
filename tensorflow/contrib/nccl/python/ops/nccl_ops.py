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
"""Ops for GPU collective operations implemented using NVIDIA nccl."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.contrib.nccl.ops import gen_nccl_ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import resource_loader

_nccl_ops_so = loader.load_op_library(
    resource_loader.get_path_to_datafile('_nccl_ops.so'))


def all_sum(tensors):
  """Returns a list of tensors with the all-reduce sum across `tensors`.

  The computation is done with an all-reduce operation, so if only some of the
  returned tensors are evaluated then the computation will hang.

  Args:
    tensors: The input tensors across which to sum; must be assigned
      to GPU devices.

  Returns:
    List of tensors, each with the sum of the input tensors, where tensor i has
    the same device as `tensors[i]`.
  """
  return _apply_all_reduce('sum', tensors)


def all_prod(tensors):
  """Returns a list of tensors with the all-reduce product across `tensors`.

  The computation is done with an all-reduce operation, so if only some of the
  returned tensors are evaluated then the computation will hang.

  Args:
    tensors: The input tensors across which to multiply; must be assigned
      to GPU devices.

  Returns:
    List of tensors, each with the product of the input tensors, where tensor i
    has the same device as `tensors[i]`.
  """
  return _apply_all_reduce('prod', tensors)


def all_min(tensors):
  """Returns a list of tensors with the all-reduce min across `tensors`.

  The computation is done with an all-reduce operation, so if only some of the
  returned tensors are evaluated then the computation will hang.

  Args:
    tensors: The input tensors across which to reduce; must be assigned
      to GPU devices.

  Returns:
    List of tensors, each with the minimum of the input tensors, where tensor i
    has the same device as `tensors[i]`.
  """
  return _apply_all_reduce('min', tensors)


def all_max(tensors):
  """Returns a list of tensors with the all-reduce max across `tensors`.

  The computation is done with an all-reduce operation, so if only some of the
  returned tensors are evaluated then the computation will hang.

  Args:
    tensors: The input tensors across which to reduce; must be assigned
      to GPU devices.

  Returns:
    List of tensors, each with the maximum of the input tensors, where tensor i
    has the same device as `tensors[i]`.
  """
  return _apply_all_reduce('max', tensors)


def broadcast(src_tensor, dst_devices):
  """Returns a list of tensors on `dst_devices`, each with value `tensor`.

  The computation is done with a broadcast nccl operation, so if only some of
  the returned tensors and src_tensor are evaluated then the computation will
  hang.

  Args:
    src_tensor: The tensor to send; must be assigned to a GPU device.
    dst_devices: The GPU devices to receive the sent tensor.

  Returns:
    List of tensors, each with the value of `src_tensor`, which the device
    of tensor i is `dst_devices[i]`.
  """
  if not dst_devices:
    raise ValueError('Must pass >0 dst_devices to broadcast')
  all_devices = [src_tensor.device] + dst_devices
  shared_name = _get_shared_name()

  with ops.device(src_tensor.device):
    send = gen_nccl_ops.nccl_broadcast_send(
        input=src_tensor, num_devices=len(all_devices), shared_name=shared_name)

  shape_op = array_ops.shape(src_tensor, out_type=dtypes.int64)
  recvs = []
  for d in dst_devices:
    with ops.device(d):
      recvs.append(
          gen_nccl_ops.nccl_broadcast_recv(
              shape=shape_op,
              T=src_tensor.dtype,
              num_devices=len(all_devices),
              shared_name=shared_name))

  return send, recvs


def _apply_all_reduce(reduction_op, tensors):
  if not tensors:
    raise ValueError('Must pass >0 tensors to all reduce operations')
  shared_name = _get_shared_name()
  res = []
  for t in tensors:
    if not device.canonical_name(t.device):
      raise ValueError('Device assignment required for nccl collective ops')
    with ops.device(t.device):
      res.append(
          gen_nccl_ops.nccl_all_reduce(
              t,
              reduction=reduction_op,
              num_devices=len(tensors),
              shared_name=shared_name))
  return res


_lock = threading.Lock()
_shared_name_counter = 0


def _get_shared_name():
  global _shared_name_counter

  with _lock:
    val = _shared_name_counter
    _shared_name_counter += 1
  return 'c%s' % val

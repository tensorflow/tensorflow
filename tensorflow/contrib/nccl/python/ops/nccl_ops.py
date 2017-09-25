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
from tensorflow.python.eager import context
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


@ops.RegisterGradient('NcclAllReduce')
def _all_sum_grad(op, grad):
  """The gradients for `all_sum`.

  Args:
    op: The `all_sum` `Operation` that we are differentiating.
    grad: Gradient with respect to the output of the `all_sum` op.

  Returns:
    The gradient with respect to the output of `all_sum`.

  Raises:
    LookupError: If `reduction` is not `sum`.
  """
  if op.get_attr('reduction') != 'sum':
    raise LookupError('No gradient defined for NcclAllReduce except all_sum.')

  _check_device_assignment(grad)
  num_devices = op.get_attr('num_devices')
  shared_name = op.get_attr('shared_name') + '_grad'

  with ops.device(grad.device):
    return gen_nccl_ops.nccl_all_reduce(
        input=grad,
        reduction='sum',
        num_devices=num_devices,
        shared_name=shared_name)


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


def reduce_sum(tensors, dst_device):
  """Returns a tensor with the reduce sum across `tensors`.

  The computation is done with a reduce operation, so only one tensor is
  returned.

  Args:
    tensors: The input tensors across which to sum; must be assigned
      to GPU devices.
    dst_device: The device of the returned tensor.

  Returns:
    A tensor containing the sum of the input tensors, with the device of the
    tensor being `dst_device`.
  """
  return _apply_reduce('sum', tensors, dst_device)


def broadcast(src_tensor, dst_devices):
  """Returns a list of tensors on `dst_devices`, each with value `tensor`.

  The computation is done with a broadcast nccl operation, so if only some of
  the returned tensors and src_tensor are evaluated then the computation will
  hang.

  Args:
    src_tensor: The tensor to send; must be assigned to a GPU device.
    dst_devices: The GPU devices to receive the sent tensor.

  Returns:
    An `Operation` to send the `src_tensor`, and a list of tensors, each with
    the value of `src_tensor`, where the device of tensor i is `dst_devices[i]`.
  """
  if not dst_devices:
    raise ValueError('Must pass >0 dst_devices to broadcast')
  _check_graph_mode()
  _check_device_assignment(src_tensor)

  shape = array_ops.shape(src_tensor, out_type=dtypes.int64)
  num_devices = len(dst_devices) + 1
  shared_name = _get_shared_name()

  with ops.device(src_tensor.device):
    send = gen_nccl_ops.nccl_broadcast_send(
        input=src_tensor, num_devices=num_devices, shared_name=shared_name)

  recvs = []
  for d in dst_devices:
    with ops.device(d):
      recvs.append(
          gen_nccl_ops.nccl_broadcast_recv(
              shape=shape,
              T=src_tensor.dtype,
              num_devices=num_devices,
              shared_name=shared_name))

  return send, recvs


def _apply_all_reduce(reduction, tensors):
  """Helper function for all_* functions."""
  if not tensors:
    raise ValueError('Must pass >0 tensors to all reduce operations')
  _check_graph_mode()

  shared_name = _get_shared_name()
  res = []

  for t in tensors:
    _check_device_assignment(t)
    with ops.device(t.device):
      res.append(
          gen_nccl_ops.nccl_all_reduce(
              input=t,
              reduction=reduction,
              num_devices=len(tensors),
              shared_name=shared_name))

  return res


def _apply_reduce(reduction, tensors, dst_device):
  """Helper function for reduce_* functions."""
  if not tensors:
    raise ValueError('Must pass >0 tensors to reduce operations')
  if not dst_device:
    raise ValueError('Must pass dst_device to reduce operations')
  _check_graph_mode()

  try:
    recv_index = next(i for i, t in enumerate(tensors)
                      if t.device == dst_device)
  except StopIteration:
    raise ValueError('One of the tensors must be assigned to dst_device')
  shared_name = _get_shared_name()

  sends = []
  for t in tensors[:recv_index] + tensors[recv_index + 1:]:
    _check_device_assignment(t)
    with ops.device(t.device):
      sends.append(
          gen_nccl_ops.nccl_reduce_send(
              input=t,
              reduction=reduction,
              num_devices=len(tensors),
              shared_name=shared_name))

  with ops.device(dst_device):
    recv = gen_nccl_ops.nccl_reduce_recv(
        input=tensors[recv_index],
        reduction=reduction,
        num_devices=len(tensors),
        shared_name=shared_name)

  return recv, sends


_lock = threading.Lock()
_shared_name_counter = 0


def _get_shared_name():
  global _shared_name_counter

  with _lock:
    val = _shared_name_counter
    _shared_name_counter += 1
  return 'c%s' % val


def _check_device_assignment(tensor):
  if not device.canonical_name(tensor.device):
    raise ValueError('Device assignment required for nccl collective ops')


def _check_graph_mode():
  if context.in_eager_mode():
    raise ValueError('Nccl ops are not supported in eager mode')

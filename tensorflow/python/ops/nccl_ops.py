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

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nccl_ops


_module_lock = threading.Lock()
_shared_name_counter = 0


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
  if op.get_attr('reduction') != b'sum':
    raise LookupError('No gradient defined for NcclAllReduce except sum.')

  _check_device(grad, expected=op.device)
  num_devices = op.get_attr('num_devices')
  shared_name = op.get_attr('shared_name') + b'_grad'

  with ops.device(op.device):
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


def reduce_sum(tensors):
  """Returns a tensor with the reduce sum across `tensors`.

  The computation is done with a reduce operation, so only one tensor is
  returned.

  Args:
    tensors: The input tensors across which to sum; must be assigned
      to GPU devices.

  Returns:
    A tensor containing the sum of the input tensors.

  Raises:
    LookupError: If context is not currently using a GPU device.
  """
  return _apply_reduce('sum', tensors)


@ops.RegisterGradient('NcclReduce')
def _reduce_sum_grad(op, grad):
  """The gradients for input `Operation` of `reduce_sum`.

  Args:
    op: The `sum send` `Operation` that we are differentiating.
    grad: Gradient with respect to the output of the `reduce_sum` op.

  Returns:
    The gradient with respect to the input of `reduce_sum` op.

  Raises:
    LookupError: If the reduction attribute of op is not `sum`.
  """
  if op.get_attr('reduction') != b'sum':
    raise LookupError('No gradient defined for NcclReduce except sum.')
  _check_device(grad, expected=op.device)

  with ops.device(op.device):
    result = gen_nccl_ops.nccl_broadcast(input=grad, shape=grad.shape)

  return [result] * len(op.inputs)


def broadcast(tensor):
  """Returns a tensor that can be efficiently transferred to other devices.

  Args:
    tensor: The tensor to send; must be assigned to a GPU device.

  Returns:
    A tensor with the value of `src_tensor`, which can be used as input to
    ops on other GPU devices.
  """
  _check_device(tensor)

  with ops.device(tensor.device):
    return gen_nccl_ops.nccl_broadcast(input=tensor, shape=tensor.shape)


@ops.RegisterGradient('NcclBroadcast')
def _broadcast_grad(op, accumulated_grad):
  """The gradients for input `Operation` of `broadcast`.

  Args:
    op: The `broadcast send` `Operation` that we are differentiating.
    accumulated_grad: Accumulated gradients with respect to the output of the
      `broadcast` op.

  Returns:
    Gradients with respect to the input of `broadcast`.
  """
  # Grab inputs of accumulated_grad and replace accumulation with reduce_sum.
  grads = [t for t in accumulated_grad.op.inputs]
  for t in grads:
    _check_device(t)

  with ops.device(op.device):
    return gen_nccl_ops.nccl_reduce(input=grads, reduction='sum')


def _apply_all_reduce(reduction, tensors):
  """Helper function for all_* functions."""
  if not tensors:
    raise ValueError('Must pass >0 tensors to all reduce operations')

  shared_name = _get_shared_name()

  def _all_reduce():
    """Call nccl allreduce."""
    res = []
    for t in tensors:
      _check_device(t)
      with ops.device(t.device):
        res.append(
            gen_nccl_ops.nccl_all_reduce(
                input=t,
                reduction=reduction,
                num_devices=len(tensors),
                shared_name=shared_name))
    return res

  if context.executing_eagerly():
    # Nccl ops will block unless they are executed concurrently such as in a
    # graph or a defun.
    return def_function.function(_all_reduce)()
  else:
    return _all_reduce()


def _apply_reduce(reduction, tensors):
  """Helper function for reduce_* functions."""
  if not tensors:
    raise ValueError('Must pass >0 tensors to reduce operations')

  for t in tensors:
    _check_device(t)
  result = gen_nccl_ops.nccl_reduce(input=tensors, reduction=reduction)
  try:
    next(t for t in tensors if t.device == result.device)
  except StopIteration:
    raise ValueError('One input tensor must be assigned to current device')
  return result


def _get_shared_name():
  global _shared_name_counter

  with _module_lock:
    val = _shared_name_counter
    _shared_name_counter += 1
  return 'c%s' % val


def _check_device(tensor, expected=None):
  if not device.canonical_name(tensor.device):
    raise ValueError('Device assignment required for nccl collective ops')
  if expected and expected != tensor.device:
    raise ValueError('Expected device %s, got %s' % (expected, tensor.device))

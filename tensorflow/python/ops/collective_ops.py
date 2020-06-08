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
"""TensorFlow collective Ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import gen_collective_ops


def all_reduce(t,
               group_size,
               group_key,
               instance_key,
               merge_op,
               final_op,
               subdiv_offsets=(0,),
               communication_hint='auto',
               timeout=0):
  """Reduces tensors collectively, across devices.

  Args:
    t: the tensor to be reduced.
    group_size: the total number of tensors to be collectively reduced.
      Each must reside on a different device.  Should be a positive integer.
    group_key: an integer identifying the group of devices.
    instance_key: an integer identifying the participating group of Ops.
    merge_op: string naming the binary Op to be applied to compute each
      partial reduction.
    final_op: string naming the unary Op to be applied to each fully
      reduced value.  Can be 'Id' for no operation.
    subdiv_offsets: a list of integer offsets into the tensor at which each
      independent subdivision should begin.  Use [0] if no subdivision should
      be done.
    communication_hint: preferred collective communication.  The implementation
      may fall back to another mechanism.  Options include `auto`, `ring`, and
      `nccl`.
    timeout: If set to a non zero, set a completion timeout to detect staleness.
      If the timer goes off, a DeadlineExceededError is raised.
      The timeout value in seconds. This feature is experimental.

  Returns:
    An Op implementing the distributed reduction.

  Raises:
    ValueError: if any of the input parameter constraints are not met.
  """
  if group_size < 1:
    raise ValueError('Parameter group_size to all_reduce must be at least 1.')
  return gen_collective_ops.collective_reduce(
      t,
      group_size=group_size,
      group_key=group_key,
      instance_key=instance_key,
      merge_op=merge_op,
      final_op=final_op,
      subdiv_offsets=subdiv_offsets,
      communication_hint=communication_hint.lower(),
      timeout_seconds=timeout)


def all_gather(t,
               group_size,
               group_key,
               instance_key,
               communication_hint='auto',
               timeout=0):
  """Accumulates tensors collectively, across devices, along first dimension.

  Args:
    t: the tensor to participate in the accumulation.
    group_size: the total number of tensors to be collectively accumulated.
      Each must reside on a different device.  Should be a positive integer.
    group_key: an integer identifying the group of devices.
    instance_key: an integer identifying the participating group of Ops.
    communication_hint: preferred collective communication.  The implementation
      may fall back to another mechanism.  Options include `auto`, `ring`, and
      `nccl`.
    timeout: If set to a non zero, set a completion timeout to detect staleness.
      If the timer goes off, a DeadlineExceededError is raised.
      The timeout value in seconds. This feature is experimental.

  Returns:
    An Op implementing the distributed operation.

  Raises:
    ValueError: if any of the input parameter constraints are not met.
  """
  if group_size < 1:
    raise ValueError('Parameter group_size to all_gather must be at least 1.')
  return gen_collective_ops.collective_gather(
      t,
      shape=[0],
      group_size=group_size,
      group_key=group_key,
      instance_key=instance_key,
      communication_hint=communication_hint.lower(),
      timeout_seconds=timeout)


def broadcast_send(t,
                   shape,
                   dtype,
                   group_size,
                   group_key,
                   instance_key,
                   communication_hint='auto',
                   timeout=0):
  """Broadcasts one tensor to a group of others, across devices.

  Args:
    t: the tensor to be sent.
    shape: the shape of the tensor being sent, which must agree with t.
    dtype: the type of the tensor being sent, which must agree with t.
    group_size: one plus the number of receiving tensors, i.e. the total
      number of devices participating.  Each tensor must reside on a
      different device.
    group_key: an integer identifying the group of devices.
    instance_key: an integer identifying the participating group of Ops.
    communication_hint: preferred collective communication.  The implementation
      may fall back to another mechanism.  Options include `auto`, `ring`, and
      `nccl`.
    timeout: If set to a non zero, set a completion timeout to detect staleness.
      If the timer goes off, a DeadlineExceededError is raised.
      The timeout value in seconds. This feature is experimental.

  Returns:
    An Op implementing the distributed broadcast send.

  Raises:
    ValueError: if any of the input parameter constraints are not met.

  Note that the shape and dtype arguments appear redundant since they
  should be obtainable from t.  The are two reasons for including
  them.  First, the shape and type of tensors passed via broadcast must
  be known ahead of time in their most specific form so that the receive
  side can allocate memory for the operation and shape/type inference can
  carry forward from there.  Including the same declarations on the
  send side clarifies a commitment already made.  Secondly, having nearly
  identical use syntax for send and receive sides may simplify tool-driven
  generation of broadcast.
  """
  if group_size <= 1:
    raise ValueError(
        'Parameter group_size to broadcast_send must be at least 2.')
  if t.shape != shape:
    raise ValueError(
        'Shape of broadcast_send tensor not equal to declared shape')
  if t.dtype != dtype:
    raise ValueError(
        'Type of broadcast_send tensor not equal to declared type')
  return gen_collective_ops.collective_bcast_send(
      t,
      shape=shape,
      group_size=group_size,
      group_key=group_key,
      instance_key=instance_key,
      communication_hint=communication_hint.lower(),
      timeout_seconds=timeout)


def broadcast_recv(shape,
                   dtype,
                   group_size,
                   group_key,
                   instance_key,
                   communication_hint='auto',
                   timeout=0):
  """Receives a broadcasts tensor, across devices.

  Args:
    shape: Shape of the tensor to be received.
    dtype: Type of the tensor to be received.
    group_size: one plus the number of receiving tensors, i.e. the total
      number of devices participating.  Each tensor must reside on a
      different device.
    group_key: an integer identifying the group of devices.
    instance_key: an integer identifying the participating group of Ops.
    communication_hint: preferred collective communication.  The implementation
      may fall back to another mechanism.  Options include `auto`, `ring`, and
      `nccl`.
    timeout: If set to a non zero, set a completion timeout to detect staleness.
      If the timer goes off, a DeadlineExceededError is raised.
      The timeout value in seconds. This feature is experimental.

  Returns:
    An Op implementing the broadcast receive.

  Raises:
    ValueError: if any of the input parameter constraints are not met.
  """
  if group_size <= 1:
    raise ValueError(
        'Parameter group_size to broadcast_send must be at least 2.')
  return gen_collective_ops.collective_bcast_recv(
      shape=shape,
      T=dtype,
      group_size=group_size,
      group_key=group_key,
      instance_key=instance_key,
      communication_hint=communication_hint.lower(),
      timeout_seconds=timeout)

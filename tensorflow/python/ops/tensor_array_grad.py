# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Gradients for operators defined in tensor_array_ops.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import tensor_array_ops

# TODO(b/31222613): These ops may be differentiable, and there may be
# latent bugs here.
ops.NotDifferentiable("TensorArray")
ops.NotDifferentiable("TensorArrayGrad")
ops.NotDifferentiable("TensorArraySize")
ops.NotDifferentiable("TensorArrayClose")

ops.NotDifferentiable("TensorArrayV2")
ops.NotDifferentiable("TensorArrayGradV2")
ops.NotDifferentiable("TensorArraySizeV2")
ops.NotDifferentiable("TensorArrayCloseV2")

ops.NotDifferentiable("TensorArrayV3")
ops.NotDifferentiable("TensorArrayGradV3")
ops.NotDifferentiable("TensorArraySizeV3")
ops.NotDifferentiable("TensorArrayCloseV3")


def _GetGradSource(op_or_tensor):
  """Identify which call to tf.gradients created this gradient op or tensor.

  TensorArray gradient calls use an accumulator TensorArray object.  If
  multiple gradients are calculated and run in the same session, the multiple
  gradient nodes may accidentally flow throuth the same accumulator TensorArray.
  This double counting breaks the TensorArray gradient flow.

  The solution is to identify which gradient call this particular
  TensorArray*Grad is being called in, by looking at the input gradient
  tensor's name, and create or lookup an accumulator gradient TensorArray
  associated with this specific call.  This solves any confusion and ensures
  different gradients from the same forward graph get their own accumulators.

  This function creates the unique label associated with the tf.gradients call
  that is used to create the gradient TensorArray.

  Args:
    op_or_tensor: `Tensor` or `Operation` which is an input to a
      TensorArray*Grad call.

  Returns:
    A python string, the unique label associated with this particular
    gradients calculation.

  Raises:
    ValueError: If not called within a gradients calculation.
  """
  name_tokens = op_or_tensor.name.split("/")
  grad_pos = [i for i, x in enumerate(name_tokens) if x.startswith("gradients")]
  if not grad_pos:
    raise ValueError(
        "Expected op/tensor name to start with gradients (excluding scope)"
        ", got: %s" % op_or_tensor.name)
  return "/".join(name_tokens[:grad_pos[-1] + 1])


@ops.RegisterGradient("TensorArrayRead")
@ops.RegisterGradient("TensorArrayReadV2")
@ops.RegisterGradient("TensorArrayReadV3")
def _TensorArrayReadGrad(op, grad):
  """Gradient for TensorArrayRead.

  Args:
    op: Forward TensorArrayRead op.
    grad: Gradient `Tensor` to TensorArrayRead.

  Returns:
    A flow `Tensor`, which can be used in control dependencies to
    force the write of `grad` to the gradient `TensorArray`.
  """
  # Note: the forward flow dependency in the call to grad() is necessary for
  # the case of dynamic sized TensorArrays.  When creating the gradient
  # TensorArray, the final size of the forward array must be known.
  # For this we need to wait until it has been created by depending on
  # the input flow of the original op.
  handle = op.inputs[0]
  index = op.inputs[1]
  flow = op.inputs[2]
  dtype = op.get_attr("dtype")
  grad_source = _GetGradSource(grad)
  g = tensor_array_ops.TensorArray(
      dtype=dtype, handle=handle, flow=flow).grad(
          source=grad_source, flow=flow)
  w_g = g.write(index, grad)
  return [None, None, w_g.flow]


@ops.RegisterGradient("TensorArrayWrite")
@ops.RegisterGradient("TensorArrayWriteV2")
@ops.RegisterGradient("TensorArrayWriteV3")
def _TensorArrayWriteGrad(op, flow):
  """Gradient for TensorArrayWrite.

  Args:
    op: Forward TensorArrayWrite op.
    flow: Gradient `Tensor` flow to TensorArrayWrite.

  Returns:
    A grad `Tensor`, the gradient created in an upstream ReadGrad or PackGrad.
  """
  # handle is the output store_handle of TensorArrayReadGrad or
  # the handle output of TensorArrayWriteGrad.  we must use this one.
  handle = op.inputs[0]
  index = op.inputs[1]
  dtype = op.get_attr("T")
  grad_source = _GetGradSource(flow)
  g = tensor_array_ops.TensorArray(
      dtype=dtype, handle=handle, flow=flow).grad(
          source=grad_source, flow=flow)
  grad = g.read(index)
  return [None, None, grad, flow]


@ops.RegisterGradient("TensorArrayGather")
@ops.RegisterGradient("TensorArrayGatherV2")
@ops.RegisterGradient("TensorArrayGatherV3")
def _TensorArrayGatherGrad(op, grad):
  """Gradient for TensorArrayGather.

  Args:
    op: Forward TensorArrayGather op.
    grad: Gradient `Tensor` to TensorArrayGather.

  Returns:
    A flow `Tensor`, which can be used in control dependencies to
    force the write of `grad` to the gradient `TensorArray`.
  """
  # Note: the forward flow dependency in the call to grad() is necessary for
  # the case of dynamic sized TensorArrays.  When creating the gradient
  # TensorArray, the final size of the forward array must be known.
  # For this we need to wait until it has been created by depending on
  # the input flow of the original op.
  handle = op.inputs[0]
  indices = op.inputs[1]
  flow = op.inputs[2]
  dtype = op.get_attr("dtype")
  grad_source = _GetGradSource(grad)
  g = tensor_array_ops.TensorArray(
      dtype=dtype, handle=handle, flow=flow).grad(
          source=grad_source, flow=flow)
  u_g = g.scatter(indices, grad)
  return [None, None, u_g.flow]


@ops.RegisterGradient("TensorArrayScatter")
@ops.RegisterGradient("TensorArrayScatterV2")
@ops.RegisterGradient("TensorArrayScatterV3")
def _TensorArrayScatterGrad(op, flow):
  """Gradient for TensorArrayScatter.

  Args:
    op: Forward TensorArrayScatter op.
    flow: Gradient `Tensor` flow to TensorArrayScatter.

  Returns:
    A grad `Tensor`, the gradient created in upstream ReadGrads or PackGrad.
  """
  handle = op.inputs[0]
  indices = op.inputs[1]
  dtype = op.get_attr("T")
  grad_source = _GetGradSource(flow)
  g = tensor_array_ops.TensorArray(
      dtype=dtype, handle=handle, flow=flow).grad(
          source=grad_source, flow=flow)
  grad = g.gather(indices)
  return [None, None, grad, flow]


@ops.RegisterGradient("TensorArrayConcat")
@ops.RegisterGradient("TensorArrayConcatV2")
@ops.RegisterGradient("TensorArrayConcatV3")
def _TensorArrayConcatGrad(op, grad, unused_lengths_grad):
  """Gradient for TensorArrayConcat.

  Args:
    op: Forward TensorArrayConcat op.
    grad: Gradient `Tensor` to TensorArrayConcat.

  Returns:
    A flow `Tensor`, which can be used in control dependencies to
    force the write of `grad` to the gradient `TensorArray`.
  """
  # Note: the forward flow dependency in the call to grad() is necessary for
  # the case of dynamic sized TensorArrays.  When creating the gradient
  # TensorArray, the final size of the forward array must be known.
  # For this we need to wait until it has been created by depending on
  # the input flow of the original op.
  handle = op.inputs[0]
  flow = op.inputs[1]
  lengths = op.outputs[1]
  dtype = op.get_attr("dtype")
  grad_source = _GetGradSource(grad)
  g = tensor_array_ops.TensorArray(
      dtype=dtype, handle=handle, flow=flow).grad(
          source=grad_source, flow=flow)
  u_g = g.split(grad, lengths=lengths)
  # handle, flow_in
  return [None, u_g.flow]


@ops.RegisterGradient("TensorArraySplit")
@ops.RegisterGradient("TensorArraySplitV2")
@ops.RegisterGradient("TensorArraySplitV3")
def _TensorArraySplitGrad(op, flow):
  """Gradient for TensorArraySplit.

  Args:
    op: Forward TensorArraySplit op.
    flow: Gradient `Tensor` flow to TensorArraySplit.

  Returns:
    A grad `Tensor`, the gradient created in upstream ReadGrads or PackGrad.
  """
  handle = op.inputs[0]
  dtype = op.get_attr("T")
  grad_source = _GetGradSource(flow)
  g = tensor_array_ops.TensorArray(
      dtype=dtype, handle=handle, flow=flow).grad(
          source=grad_source, flow=flow)
  grad = g.concat()
  # handle, value, lengths, flow_in
  return [None, grad, None, flow]

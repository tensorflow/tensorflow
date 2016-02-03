# Copyright 2015 Google Inc. All Rights Reserved.
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


ops.NoGradient("TensorArray")
ops.NoGradient("TensorArrayGrad")
ops.NoGradient("TensorArraySize")
ops.NoGradient("TensorArrayClose")


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
  if not op_or_tensor.name.startswith("gradients"):
    raise ValueError(
        "Expected op/tensor name to start with gradients, got: %s"
        % op_or_tensor.name)
  return op_or_tensor.name.split("/")[0]


@ops.RegisterGradient("TensorArrayRead")
def _TensorArrayReadGrad(op, grad):
  """Gradient for TensorArrayRead.

  Args:
    op: Forward TensorArrayRead op.
    grad: Gradient `Tensor` to TensorArrayRead.

  Returns:
    A flow `Tensor`, which can be used in control dependencies to
    force the write of `grad` to the gradient `TensorArray`.
  """
  handle = op.inputs[0]
  index = op.inputs[1]
  flow = op.inputs[2]
  dtype = op.get_attr("dtype")
  grad_source = _GetGradSource(grad)
  g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle).grad(
      source=grad_source, flow=flow)
  w_g = g.write(index, grad)
  return [None, None, w_g.flow]


@ops.RegisterGradient("TensorArrayWrite")
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
  g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle).grad(
      source=grad_source, flow=flow)
  grad = g.read(index)
  return [None, None, grad, flow]


@ops.RegisterGradient("TensorArrayPack")
def _TensorArrayPackGrad(op, grad):
  """Gradient for TensorArrayPack.

  Args:
    op: Forward TensorArrayPack op.
    grad: Gradient `Tensor` to TensorArrayPack.

  Returns:
    A flow `Tensor`, which can be used in control dependencies to
    force the write of `grad` to the gradient `TensorArray`.
  """
  handle = op.inputs[0]
  flow = op.inputs[1]
  dtype = op.get_attr("dtype")
  grad_source = _GetGradSource(grad)
  g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle).grad(
      source=grad_source, flow=flow)
  u_g = g.unpack(grad)
  return [None, u_g.flow]


@ops.RegisterGradient("TensorArrayUnpack")
def _TensorArrayUnpackGrad(op, flow):
  """Gradient for TensorArrayUnpack.

  Args:
    op: Forward TensorArrayUnpack op.
    flow: Gradient `Tensor` flow to TensorArrayUnpack.

  Returns:
    A grad `Tensor`, the gradient created in upstream ReadGrads or PackGrad.
  """
  handle = op.inputs[0]
  dtype = op.get_attr("T")
  grad_source = _GetGradSource(flow)
  g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle).grad(
      source=grad_source, flow=flow)
  grad = g.pack()
  return [None, grad, flow]
# pylint: enable=protected-access

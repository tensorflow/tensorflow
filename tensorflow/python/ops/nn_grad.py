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

"""Gradients for operators defined in nn_ops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import gen_nn_ops


@ops.RegisterGradient("Conv2DBackpropInput")
def _Conv2DBackpropGrad(op, grad):
  """The derivatives for deconvolution.

  Args:
    op: the Deconvolution op.
    grad: the tensor representing the gradient w.r.t. the output

  Returns:
    the gradients w.r.t. the input and the filter
  """
  return [None,
          nn_ops.conv2d_backprop_filter(
              grad, array_ops.shape(op.inputs[1]), op.inputs[2],
              op.get_attr("strides"), op.get_attr("padding"),
              op.get_attr("use_cudnn_on_gpu"), op.get_attr("data_format")),
          nn_ops.conv2d(
              grad, op.inputs[1], op.get_attr("strides"),
              op.get_attr("padding"), op.get_attr("use_cudnn_on_gpu"),
              op.get_attr("data_format"))]


@ops.RegisterGradient("Softmax")
def _SoftmaxGrad(op, grad_softmax):
  """The derivative of the softmax nonlinearity.

  We assume that probs is of shape [batch_size * dim]
  The formula for dsoftmax / dx = (diag(softmax) - softmax * softmax').
  This matrix is diagonal minus a rank one matrix, so it is easy to implement
  as follows:

    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

  Args:
     op: the Softmax op.
     grad_softmax:  the tensor representing the gradient w.r.t. the
       softmax output.

  Returns:
     gradient w.r.t the input to the softmax

  """
  # TODO(ilyasu): assert that the tensor has two dimensions at
  # graph-construction time?  Alternatively: do different things
  # depending on the dimensionality of the input tensors.
  softmax = op.outputs[0]
  grad_x = ((grad_softmax -
             array_ops.reshape(math_ops.reduce_sum(grad_softmax * softmax, [1]),
                               [-1, 1]))
            * softmax)
  return grad_x


@ops.RegisterGradient("BiasAdd")
def _BiasAddGrad(op, received_grad):
  """Return the gradients for the 2 inputs of bias_op.

  The first input of unused_bias_op is the tensor t, and its gradient is
  just the gradient the unused_bias_op received.

  The second input of unused_bias_op is the bias vector which has one fewer
  dimension than "received_grad" (the batch dimension.)  Its gradient is the
  received gradient Summed on the batch dimension, which is the first dimension.

  Args:
    op: The BiasOp for which we need to generate gradients.
    received_grad: Tensor.  The gradients passed to the BiasOp.

  Returns:
    Two tensors, the first one for the "tensor" input of the BiasOp,
    the second one for the "bias" input of the BiasOp.
  """
  try:
    data_format = op.get_attr("data_format")
  except ValueError:
    data_format = None
  return (received_grad, gen_nn_ops.bias_add_grad(out_backprop=received_grad,
                                                  data_format=data_format))


@ops.RegisterGradient("BiasAddV1")
def _BiasAddGradV1(unused_bias_op, received_grad):
  """Return the gradients for the 2 inputs of bias_op.

  The first input of unused_bias_op is the tensor t, and its gradient is
  just the gradient the unused_bias_op received.

  The second input of unused_bias_op is the bias vector which has one fewer
  dimension than "received_grad" (the batch dimension.)  Its gradient is the
  received gradient Summed on the batch dimension, which is the first dimension.

  Args:
    unused_bias_op: The BiasOp for which we need to generate gradients.
    received_grad: Tensor.  The gradients passed to the BiasOp.

  Returns:
    Two tensors, the first one for the "tensor" input of the BiasOp,
    the second one for the "bias" input of the BiasOp.
  """
  reduction_dim_tensor = math_ops.range(array_ops.rank(received_grad) - 1)
  return (received_grad, math_ops.reduce_sum(received_grad, reduction_dim_tensor))


@ops.RegisterGradient("Relu")
def _ReluGrad(op, grad):
  return gen_nn_ops._relu_grad(grad, op.outputs[0])


@ops.RegisterGradient("Relu6")
def _Relu6Grad(op, grad):
  return gen_nn_ops._relu6_grad(grad, op.inputs[0])


@ops.RegisterGradient("Elu")
def _EluGrad(op, grad):
  return gen_nn_ops._elu_grad(grad, op.outputs[0])


@ops.RegisterGradient("Softplus")
def _SoftplusGrad(op, grad):
  return gen_nn_ops._softplus_grad(grad, op.inputs[0])


@ops.RegisterGradient("Softsign")
def _SoftsignGrad(op, grad):
  return gen_nn_ops._softsign_grad(grad, op.inputs[0])


@ops.RegisterGradient("ReluGrad")
def _ReluGradGrad(op, grad):
  x = op.inputs[1]
  return (gen_nn_ops._relu_grad(grad, x),
          array_ops.zeros(shape=array_ops.shape(x), dtype=x.dtype))


def _BroadcastMul(vec, mat):
  """Multiply after broadcasting vec to match dimensions of mat.

  Args:
    vec: A 1-D tensor of dimension [D0]
    mat: A 2-D tensor of dimension [D0, D1]

  Returns:
    A tensor of dimension [D0, D1], the result of vec * mat
  """
  # Reshape vec to [D0, 1]
  vec = array_ops.expand_dims(vec, -1)
  return vec * mat


@ops.RegisterGradient("SoftmaxCrossEntropyWithLogits")
def _SoftmaxCrossEntropyWithLogitsGrad(op, grad_0, _):
  # grad_0 is the backprop for cost, and we multiply it with the gradients
  # (which is output[1])
  # There is no gradient for the labels
  return _BroadcastMul(grad_0, op.outputs[1]), None


@ops.RegisterGradient("SparseSoftmaxCrossEntropyWithLogits")
def _SparseSoftmaxCrossEntropyWithLogitsGrad(op, grad_0, _):
  # grad_0 is the backprop for cost, and we multiply it with the gradients
  # (which is output[1])
  # There is no gradient for the labels
  return _BroadcastMul(grad_0, op.outputs[1]), None


@ops.RegisterGradient("Conv2D")
def _Conv2DGrad(op, grad):
  return [nn_ops.conv2d_backprop_input(array_ops.shape(op.inputs[0]),
                                       op.inputs[1], grad,
                                       op.get_attr("strides"),
                                       op.get_attr("padding"),
                                       op.get_attr("use_cudnn_on_gpu"),
                                       op.get_attr("data_format")),
          nn_ops.conv2d_backprop_filter(op.inputs[0],
                                        array_ops.shape(op.inputs[1]), grad,
                                        op.get_attr("strides"),
                                        op.get_attr("padding"),
                                        op.get_attr("use_cudnn_on_gpu"),
                                        op.get_attr("data_format"))]


@ops.RegisterGradient("DepthwiseConv2dNative")
def _DepthwiseConv2dNativeGrad(op, grad):
  return [
      nn_ops.depthwise_conv2d_native_backprop_input(
          array_ops.shape(op.inputs[0]), op.inputs[1], grad,
          op.get_attr("strides"), op.get_attr("padding")),
      nn_ops.depthwise_conv2d_native_backprop_filter(
          op.inputs[0], array_ops.shape(op.inputs[1]), grad,
          op.get_attr("strides"), op.get_attr("padding"))
  ]


@ops.RegisterGradient("LRN")
def _LRNGrad(op, grad):
  depth_radius = op.get_attr("depth_radius")
  bias = op.get_attr("bias")
  alpha = op.get_attr("alpha")
  beta = op.get_attr("beta")
  return [gen_nn_ops._lrn_grad(grad, op.inputs[0], op.outputs[0],
                               depth_radius, bias, alpha, beta)]


@ops.RegisterGradient("AvgPool")
def _AvgPoolGrad(op, grad):
  return gen_nn_ops._avg_pool_grad(array_ops.shape(op.inputs[0]), grad,
                                   op.get_attr("ksize"),
                                   op.get_attr("strides"),
                                   op.get_attr("padding"),
                                   data_format=op.get_attr("data_format")
                                  )


@ops.RegisterGradient("MaxPool")
def _MaxPoolGrad(op, grad):
  return gen_nn_ops._max_pool_grad(op.inputs[0], op.outputs[0], grad,
                                   op.get_attr("ksize"),
                                   op.get_attr("strides"),
                                   padding=op.get_attr("padding"),
                                   data_format=op.get_attr("data_format")
                                  )


@ops.RegisterGradient("BatchNormWithGlobalNormalization")
def _BatchNormWithGlobalNormalizationGrad(op, grad):
  """Return the gradients for the 5 inputs of BatchNormWithGlobalNormalization.

  We do not backprop anything for the mean and var intentionally as they are
  not being trained with backprop in the operation.

  Args:
    op: The BatchNormOp for which we need to generate gradients.
    grad: Tensor.  The gradients passed to the BatchNormOp.

  Returns:
    dx: Backprop for input, which is (grad * (g * rsqrt(v + epsilon)))
    dm: Backprop for mean, which is
        sum_over_rest(grad * g) * (-1 / rsqrt(v + epsilon))
    dv: Backprop for variance, which is
        sum_over_rest(grad * g * (x - m)) * (-1/2) * (v + epsilon) ^ (-3/2)
    db: Backprop for beta, which is grad reduced in all except the
        last dimension.
    dg: Backprop for gamma, which is (grad * ((x - m) * rsqrt(v + epsilon)))
  """
  dx, dm, dv, db, dg = gen_nn_ops._batch_norm_with_global_normalization_grad(
      op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[4], grad,
      op.get_attr("variance_epsilon"), op.get_attr("scale_after_normalization"))
  return dx, dm, dv, db, dg


@ops.RegisterGradient("L2Loss")
def _L2LossGrad(op, grad):
  """Return the gradients for L2Loss.

  Args:
    op: The L2LossOp for which we need to generate gradients.
    grad: Tensor containing a single number.

  Returns:
    The gradient, which is (x * grad).
  """
  return op.inputs[0] * grad


@ops.RegisterGradient("TopK")
@ops.RegisterGradient("TopKV2")
def _TopKGrad(op, grad, _):
  """Return the gradients for TopK.

  Args:
    op: The TopKOp for which we need to generate gradients.
    grad: Tensor. The gradients passed to the TopKOp.

  Returns:
    A list of two tensors, the first being the gradient w.r.t to the input and
    TopK, and the second being the gradient w.r.t. to the indices (all zero).
  """
  in_shape = array_ops.shape(op.inputs[0])
  ind_shape = array_ops.shape(op.outputs[1])

  ind_lastdim = array_ops.gather(ind_shape, array_ops.size(ind_shape) - 1)
  # Flatten indices to 2D.
  ind_2d = array_ops.reshape(op.outputs[1], array_ops.pack([-1, ind_lastdim]))

  in_lastdim = array_ops.gather(in_shape, array_ops.size(in_shape) - 1)
  outerdim = array_ops.shape(ind_2d)[0]
  # Compute linear indices (flattened to 1D).
  ind = array_ops.reshape(ind_2d + array_ops.expand_dims(
      math_ops.range(0, outerdim * in_lastdim, in_lastdim), -1), [-1])

  # Substitute grad to appropriate locations and fill the rest with zeros,
  # finally reshaping it to the original input shape.
  return [array_ops.reshape(
      sparse_ops.sparse_to_dense(ind,
                                 array_ops.reshape(
                                     math_ops.reduce_prod(in_shape), [1]),
                                 array_ops.reshape(grad, [-1]),
                                 validate_indices=False),
      in_shape), array_ops.zeros(
          [],
          dtype=dtypes.int32)]

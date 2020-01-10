"""Gradients for operators defined in nn_ops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_nn_ops


@ops.RegisterGradient("Conv2DBackpropInput")
def _DeConv2DGrad(op, grad):
  """The derivatives for deconvolution.

  Args:
    op: the Deconvolution op.
    grad: the tensor representing the gradient w.r.t. the output

  Returns:
    the gradients w.r.t. the input and the filter
  """
  return [None,
          nn_ops.conv2d_backprop_filter(grad,
                                      array_ops.shape(op.inputs[1]),
                                      op.inputs[2],
                                      op.get_attr("strides"),
                                      op.get_attr("padding")),
          nn_ops.conv2d(grad,
                        op.inputs[1],
                        op.get_attr("strides"),
                        op.get_attr("padding"))]


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
def _BiasAddGrad(unused_bias_op, received_grad):
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


def _VerifyTensor(t, name, msg):
  """Assert that the tensor does not contain any NaN's.

  Args:
    t: Tensor
    name: name
    msg: message to log
  Returns:
    Tensor, but verified
  """
  with ops.name_scope(name):
    with ops.device(t.device or ops.get_default_graph().get_default_device()):
      verify_input = array_ops.check_numerics(t, message=msg)
      out = control_flow_ops.with_dependencies([verify_input], t)
  return out


@ops.RegisterGradient("Relu")
def _ReluGrad(op, grad):
  t = _VerifyTensor(op.inputs[0], op.name, "ReluGrad input is not finite.")
  return gen_nn_ops._relu_grad(grad, t)


@ops.RegisterGradient("Relu6")
def _Relu6Grad(op, grad):
  return gen_nn_ops._relu6_grad(grad, op.inputs[0])


@ops.RegisterGradient("Softplus")
def _SoftplusGrad(op, grad):
  return gen_nn_ops._softplus_grad(grad, op.inputs[0])


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


@ops.RegisterGradient("Conv2D")
def _Conv2DGrad(op, grad):
  return [nn_ops.conv2d_backprop_input(array_ops.shape(op.inputs[0]),
                                       op.inputs[1],
                                       grad,
                                       op.get_attr("strides"),
                                       op.get_attr("padding")),
          nn_ops.conv2d_backprop_filter(op.inputs[0],
                                        array_ops.shape(op.inputs[1]),
                                        grad,
                                        op.get_attr("strides"),
                                        op.get_attr("padding"))]


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
                                   op.get_attr("padding"))


@ops.RegisterGradient("MaxPool")
def _MaxPoolGrad(op, grad):
  return gen_nn_ops._max_pool_grad(op.inputs[0], op.outputs[0], grad,
                                   op.get_attr("ksize"),
                                   op.get_attr("strides"),
                                   padding=op.get_attr("padding"))


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

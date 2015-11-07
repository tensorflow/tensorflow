"""Wrappers for primitive Neural Net (NN) Operations."""

import tensorflow.python.platform
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import types
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import gen_nn_ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *


# Aliases for some automatically-generated names.
local_response_normalization = gen_nn_ops.lrn


def deconv2d(value, filter, output_shape, strides, padding="SAME",
             name=None):
  """The transpose of `conv2d`.

  This used to be called "deconvolution", but it is actually the transpose
  (gradient) of `conv2d`, not an actual deconvolution.

  Args:
    value: A 4-D `Tensor` of type `float` and shape
      `[batch, height, width, in_channels]`.
    filter: A 4-D `Tensor` with the same type as `value` and shape
      `[height, width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `value`.
    output_shape: A 1-D `Tensor` representing the output shape of the
      deconvolution op.
    strides: A list of ints. The stride of the sliding window for each
      dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
    name: Optional name for the returned tensor.

  Returns:
    A `Tensor` with the same type as `value`.

  Raises:
    ValueError: If input/output depth does not match `filter`'s shape, or if
      padding is other than `'VALID'` or `'SAME'`.
  """
  with ops.op_scope([value, filter, output_shape], name, "DeConv2D") as name:
    value = ops.convert_to_tensor(value, name="value")
    filter = ops.convert_to_tensor(filter, name="filter")
    if not value.get_shape()[3].is_compatible_with(filter.get_shape()[3]):
      raise ValueError(
          "input channels does not match filter's input channels, "
          "{} != {}".format(value.get_shape()[3], filter.get_shape()[3]))

    output_shape_ = ops.convert_to_tensor(output_shape, name="output_shape")
    if not output_shape_.get_shape().is_compatible_with(tensor_shape.vector(4)):
      raise ValueError("output_shape must have shape (4,), got {}"
                       .format(output_shape_.get_shape()))

    if isinstance(output_shape, (list, np.ndarray)):
      # output_shape's shape should be == [4] if reached this point.
      if not filter.get_shape()[2].is_compatible_with(output_shape[3]):
        raise ValueError(
            "output_shape does not match filter's output channels, "
            "{} != {}".format(output_shape[3], filter.get_shape()[2]))

    if padding != "VALID" and padding != "SAME":
      raise ValueError("padding must be either VALID or SAME:"
                       " {}".format(padding))

    return gen_nn_ops.conv2d_backprop_input(input_sizes=output_shape_,
                                            filter=filter,
                                            out_backprop=value,
                                            strides=strides,
                                            padding=padding,
                                            name=name)

# pylint: disable=protected-access
def bias_add(value, bias, name=None):
  """Adds `bias` to `value`.

  This is (mostly) a special case of `tf.add` where `bias` is restricted to 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.
  Unlike `tf.add`, the type of `bias` is allowed to differ from `value` in the
  case where both types are quantized.

  Args:
    value: A `Tensor` with type `float`, `double`, `int64`, `int32`, `uint8`,
      `int16`, `int8`, or `complex64`.
    bias: A 1-D `Tensor` with size matching the last dimension of `value`.
      Must be the same type as `value` unless `value` is a quantized type,
      in which case a different quantized type may be used.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `value`.
  """
  with ops.op_scope([value, bias], name, "BiasAdd") as name:
    value = ops.convert_to_tensor(value, name="input")
    bias = ops.convert_to_tensor(bias, dtype=value.dtype, name="bias")
    return gen_nn_ops._bias_add(value, bias, name=name)


ops.RegisterShape("BiasAdd")(common_shapes.bias_add_shape)



def relu6(features, name=None):
  """Computes Rectified Linear 6: `min(max(features, 0), 6)`.

  Args:
    features: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
      `int16`, or `int8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `features`.
  """
  with ops.op_scope([features], name, "Relu6") as name:
    features = ops.convert_to_tensor(features, name="features")
    return gen_nn_ops._relu6(features, name=name)


def softmax_cross_entropy_with_logits(logits, labels, name=None):
  """Computes softmax cross entropy between `logits` and `labels`.

  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.

  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.

  `logits` and `labels` must have the same shape `[batch_size, num_classes]`
  and the same dtype (either `float32` or `float64`).

  Args:
    logits: Unscaled log probabilities.
    labels: Each row `labels[i]` must be a valid probability distribution.
    name: A name for the operation (optional).

  Returns:
    A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
    softmax cross entropy loss.
  """
  # The second output tensor contains the gradients.  We use it in
  # _CrossEntropyGrad() in nn_grad but not here.
  cost, unused_backprop = gen_nn_ops._softmax_cross_entropy_with_logits(
      logits, labels, name=name)
  return cost


@ops.RegisterShape("SoftmaxCrossEntropyWithLogits")
def _SoftmaxCrossEntropyWithLogitsShape(op):
  """Shape function for SoftmaxCrossEntropyWithLogits op."""
  logits_shape = op.inputs[0].get_shape()
  labels_shape = op.inputs[1].get_shape()
  input_shape = logits_shape.merge_with(labels_shape).with_rank(2)
  batch_size = input_shape[0]
  return [tensor_shape.vector(batch_size.value), input_shape]


def avg_pool(value, ksize, strides, padding, name=None):
  """Performs the average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Args:
    value: A 4-D `Tensor` of shape `[batch, height, width, channels]` and type
      `float32`, `float64`, `qint8`, `quint8`, or `qint32`.
    ksize: A list of ints that has length >= 4.
      The size of the window for each dimension of the input tensor.
    strides: A list of ints that has length >= 4.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
    name: Optional name for the operation.

  Returns:
    A `Tensor` with the same type as `value`.  The average pooled output tensor.
  """
  with ops.op_scope([value], name, "AvgPool") as name:
    value = ops.convert_to_tensor(value, name="input")
    return gen_nn_ops._avg_pool(value, ksize=ksize, strides=strides,
                                padding=padding,
                                name=name)


def max_pool(value, ksize, strides, padding, name=None):
  """Performs the max pooling on the input.

  Args:
    value: A 4-D `Tensor` with shape `[batch, height, width, channels]` and
      type `float32`, `float64`, `qint8`, `quint8`, `qint32`.
    ksize: A list of ints that has length >= 4.  The size of the window for
      each dimension of the input tensor.
    strides: A list of ints that has length >= 4.  The stride of the sliding
      window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
    name: Optional name for the operation.

  Returns:
    A `Tensor` with the same type as `value`.  The max pooled output tensor.
  """
  with ops.op_scope([value], name, "MaxPool") as name:
    value = ops.convert_to_tensor(value, name="input")
    return gen_nn_ops._max_pool(value, ksize=ksize, strides=strides,
                                padding=padding,
                                name=name)


ops.RegisterShape("Relu")(common_shapes.unchanged_shape)
ops.RegisterShape("Relu6")(common_shapes.unchanged_shape)
ops.RegisterShape("Softplus")(common_shapes.unchanged_shape)


@ops.RegisterShape("ReluGrad")
@ops.RegisterShape("Relu6Grad")
@ops.RegisterShape("SoftplusGrad")
def _BinaryElementwiseShape(op):
  """Returns same shape as both inputs to op.

  Args:
    op: Input operation.

  Returns:
    Shape of both inputs to `op`.
  """
  return [op.inputs[0].get_shape().merge_with(op.inputs[1].get_shape())]


ops.RegisterShape("L2Loss")(common_shapes.scalar_shape)


ops.RegisterShape("LRN")(common_shapes.unchanged_shape_with_rank(4))


@ops.RegisterShape("LRNGrad")
def _LRNGradShape(op):
  """Shape function for LRNGrad op."""
  in_grads_shape = op.inputs[0].get_shape().with_rank(4)
  in_image_shape = op.inputs[1].get_shape().with_rank(4)
  out_image_shape = op.inputs[2].get_shape().with_rank(4)
  return [in_grads_shape.merge_with(in_image_shape).merge_with(out_image_shape)]


ops.RegisterShape("Softmax")(
    common_shapes.unchanged_shape_with_rank(2))


@ops.RegisterShape("InTopK")
def _InTopKShape(op):
  """Shape function for InTopK op."""
  predictions_shape = op.inputs[0].get_shape().with_rank(2)
  targets_shape = op.inputs[1].get_shape().with_rank(1)
  batch_size = predictions_shape[0].merge_with(targets_shape[0])
  return [tensor_shape.vector(batch_size.value)]


@ops.RegisterShape("TopK")
def _TopKShape(op):
  """Shape function for TopK op."""
  input_shape = op.inputs[0].get_shape().with_rank(2)
  k = op.get_attr("k")
  num_rows = input_shape[0]
  num_cols = input_shape[1]
  if num_cols.value is not None and num_cols.value < k:
    raise ValueError("input must have at least k (%d) columns" % k)
  return [tensor_shape.TensorShape([num_rows, k]),
          tensor_shape.TensorShape([num_rows, k])]


@ops.RegisterShape("BatchNormWithGlobalNormalization")
def _BatchNormShape(op):
  """Shape function for BatchNormWithGlobalNormalization op."""
  input_shape = op.inputs[0].get_shape().with_rank(4)
  mean_shape = op.inputs[1].get_shape().with_rank(1)
  var_shape = op.inputs[2].get_shape().with_rank(1)
  beta_shape = op.inputs[3].get_shape().with_rank(1)
  gamma_shape = op.inputs[4].get_shape().with_rank(1)
  mean_shape[0].merge_with(input_shape[3])
  var_shape[0].merge_with(input_shape[3])
  beta_shape[0].merge_with(input_shape[3])
  gamma_shape[0].merge_with(input_shape[3])
  return [input_shape]


@ops.RegisterShape("BatchNormWithGlobalNormalizationGrad")
def _BatchNormGradShape(op):
  """Shape function for BatchNormWithGlobalNormalizationGrad op."""
  input_shape = op.inputs[0].get_shape().with_rank(4)
  mean_shape = op.inputs[1].get_shape().with_rank(1)
  var_shape = op.inputs[2].get_shape().with_rank(1)
  beta_shape = op.inputs[3].get_shape().with_rank(1)
  out_backprop_shape = op.inputs[4].get_shape().with_rank(4)
  input_shape = input_shape.merge_with(out_backprop_shape)
  vector_dim = input_shape[3]
  vector_dim = vector_dim.merge_with(mean_shape[0])
  vector_dim = vector_dim.merge_with(var_shape[0])
  vector_dim = vector_dim.merge_with(beta_shape[0])
  return [input_shape] + ([tensor_shape.vector(vector_dim)] * 4)


ops.RegisterShape("Conv2D")(common_shapes.conv2d_shape)
ops.RegisterShape("AvgPool")(common_shapes.avg_pool_shape)
ops.RegisterShape("MaxPool")(common_shapes.max_pool_shape)


@ops.RegisterShape("MaxPoolWithArgmax")
def _MaxPoolWithArgMaxShape(op):
  """Shape function for MaxPoolWithArgmax op."""
  return common_shapes.max_pool_shape(op) * 2


@ops.RegisterShape("AvgPoolGrad")
def _AvgPoolGradShape(op):
  """Shape function for the AvgPoolGrad op."""
  orig_input_shape = tensor_util.ConstantValue(op.inputs[0])
  if orig_input_shape is not None:
    return [tensor_shape.TensorShape(orig_input_shape.tolist())]
  else:
    # NOTE(mrry): We could in principle work out the shape from the
    # gradients and the attrs, but if we do not know orig_input_shape
    # statically, then we are unlikely to know the shape of the
    # gradients either.
    return [tensor_shape.unknown_shape(ndims=4)]


@ops.RegisterShape("Conv2DBackpropFilter")
def _Conv2DBackpropFilterShape(op):
  """Shape function for the Conv2DBackpropFilter op."""
  filter_shape = tensor_util.ConstantValue(op.inputs[1])
  if filter_shape is not None:
    return [tensor_shape.TensorShape(filter_shape.tolist())]
  else:
    # NOTE(mrry): We could in principle work out the shape from the
    # gradients and the attrs, but if we do not know filter_shape
    # statically, then we are unlikely to know the shape of the
    # gradients either.
    return [tensor_shape.unknown_shape(ndims=4)]


@ops.RegisterShape("Conv2DBackpropInput")
def _Conv2DBackpropInputShape(op):
  """Shape function for the Conv2DBackpropInput op."""
  input_shape = tensor_util.ConstantValue(op.inputs[0])
  if input_shape is not None:
    return [tensor_shape.TensorShape(input_shape.tolist())]
  else:
    # NOTE(mrry): We could in principle work out the shape from the
    # gradients and the attrs, but if we do not know input_shape
    # statically, then we are unlikely to know the shape of the
    # gradients either.
    return [tensor_shape.unknown_shape(ndims=4)]


@ops.RegisterShape("MaxPoolGrad")
@ops.RegisterShape("MaxPoolGradWithArgmax")
def _MaxPoolGradShape(op):
  """Shape function for the MaxPoolGrad op."""
  orig_input_shape = op.inputs[0].get_shape().with_rank(4)
  return [orig_input_shape]

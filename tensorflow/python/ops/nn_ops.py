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
"""Wrappers for primitive Neural Net (NN) Operations."""

# pylint: disable=invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import graph_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *
# pylint: enable=wildcard-import

# Aliases for some automatically-generated names.
local_response_normalization = gen_nn_ops.lrn


def atrous_conv2d(value, filters, rate, padding, name=None):
  """Atrous convolution (a.k.a. convolution with holes or dilated convolution).

  Computes a 2-D atrous convolution, also known as convolution with holes or
  dilated convolution, given 4-D `value` and `filters` tensors. If the `rate`
  parameter is equal to one, it performs regular 2-D convolution. If the `rate`
  parameter is greater than one, it performs convolution with holes, sampling
  the input values every `rate` pixels in the `height` and `width` dimensions.
  This is equivalent to convolving the input with a set of upsampled filters,
  produced by inserting `rate - 1` zeros between two consecutive values of the
  filters along the `height` and `width` dimensions, hence the name atrous
  convolution or convolution with holes (the French word trous means holes in
  English).

  More specifically:

      output[b, i, j, k] = sum_{di, dj, q} filters[di, dj, q, k] *
            value[b, i + rate * di, j + rate * dj, q]

  Atrous convolution allows us to explicitly control how densely to compute
  feature responses in fully convolutional networks. Used in conjunction with
  bilinear interpolation, it offers an alternative to `conv2d_transpose` in
  dense prediction tasks such as semantic image segmentation, optical flow
  computation, or depth estimation. It also allows us to effectively enlarge
  the field of view of filters without increasing the number of parameters or
  the amount of computation.

  For a description of atrous convolution and how it can be used for dense
  feature extraction, please see: (Semantic Image Segmentation with Deep
  Convolutional Nets and Fully Connected CRFs)[http://arxiv.org/abs/1412.7062].
  The same operation is investigated further in (Multi-Scale Context Aggregation
  by Dilated Convolutions)[http://arxiv.org/abs/1511.07122]. Previous works
  that effectively use atrous convolution in different ways are, among others,
  (OverFeat: Integrated Recognition, Localization and Detection using
  Convolutional Networks) [http://arxiv.org/abs/1312.6229] and (Fast Image
  Scanning with Deep Max-Pooling Convolutional Neural Networks)
  [http://arxiv.org/abs/1302.1700]. Atrous convolution is also closely related
  to the so-called noble identities in multi-rate signal processing.

  There are many different ways to implement atrous convolution (see the refs
  above). The implementation here reduces

      atrous_conv2d(value, filters, rate, padding=padding)

  to the following three operations:

      paddings = ...
      net = space_to_batch(value, paddings, block_size=rate)
      net = conv2d(net, filters, strides=[1, 1, 1, 1], padding="VALID")
      crops = ...
      net = batch_to_space(net, crops, block_size=rate)

  Advanced usage. Note the following optimization: A sequence of `atrous_conv2d`
  operations with identical `rate` parameters, 'SAME' `padding`, and filters
  with odd heights/ widths:

      net = atrous_conv2d(net, filters1, rate, padding="SAME")
      net = atrous_conv2d(net, filters2, rate, padding="SAME")
      ...
      net = atrous_conv2d(net, filtersK, rate, padding="SAME")

  can be equivalently performed cheaper in terms of computation and memory as:

      pad = ...  # padding so that the input dims are multiples of rate
      net = space_to_batch(net, paddings=pad, block_size=rate)
      net = conv2d(net, filters1, strides=[1, 1, 1, 1], padding="SAME")
      net = conv2d(net, filters2, strides=[1, 1, 1, 1], padding="SAME")
      ...
      net = conv2d(net, filtersK, strides=[1, 1, 1, 1], padding="SAME")
      net = batch_to_space(net, crops=pad, block_size=rate)

  because a pair of consecutive `space_to_batch` and `batch_to_space` ops with
  the same `block_size` cancel out when their respective `paddings` and `crops`
  inputs are identical.

  Args:
    value: A 4-D `Tensor` of type `float`. It needs to be in the default "NHWC"
      format. Its shape is `[batch, in_height, in_width, in_channels]`.
    filters: A 4-D `Tensor` with the same type as `value` and shape
      `[filter_height, filter_width, in_channels, out_channels]`. `filters`'
      `in_channels` dimension must match that of `value`. Atrous convolution is
      equivalent to standard convolution with upsampled filters with effective
      height `filter_height + (filter_height - 1) * (rate - 1)` and effective
      width `filter_width + (filter_width - 1) * (rate - 1)`, produced by
      inserting `rate - 1` zeros along consecutive elements across the
      `filters`' spatial dimensions.
    rate: A positive int32. The stride with which we sample input values across
      the `height` and `width` dimensions. Equivalently, the rate by which we
      upsample the filter values by inserting zeros across the `height` and
      `width` dimensions. In the literature, the same parameter is sometimes
      called `input stride` or `dilation`.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
    name: Optional name for the returned tensor.

  Returns:
    A `Tensor` with the same type as `value`.

  Raises:
    ValueError: If input/output depth does not match `filters`' shape, or if
      padding is other than `'VALID'` or `'SAME'`.
  """
  with ops.op_scope([value, filters], name, "atrous_conv2d") as name:
    value = ops.convert_to_tensor(value, name="value")
    filters = ops.convert_to_tensor(filters, name="filters")
    value_shape = value.get_shape()
    filter_shape = filters.get_shape()
    if not value_shape[3].is_compatible_with(filter_shape[2]):
      raise ValueError(
          "value's input channels does not match filters' input channels, "
          "{} != {}".format(value_shape[3], filter_shape[2]))
    if rate < 1:
      raise ValueError("rate {} cannot be less than one".format(rate))

    if rate == 1:
      value = gen_nn_ops.conv2d(input=value,
                                filter=filters,
                                strides=[1, 1, 1, 1],
                                padding=padding)
      return value

    # We have two padding contributions. The first is used for converting "SAME"
    # to "VALID". The second is required so that the height and width of the
    # zero-padded value tensor are multiples of rate.

    # Spatial dimensions of original input
    in_height = int(value_shape[1])
    in_width = int(value_shape[2])

    # Spatial dimensions of the filters and the upsampled filters in which we
    # introduce (rate - 1) zeros between consecutive filter values.
    filter_height = int(filter_shape[0])
    filter_width = int(filter_shape[1])
    filter_height_up = filter_height + (filter_height - 1) * (rate - 1)
    filter_width_up = filter_width + (filter_width - 1) * (rate - 1)

    # Padding required to reduce to "VALID" convolution
    if padding == "SAME":
      pad_height = filter_height_up - 1
      pad_width = filter_width_up - 1
    elif padding == "VALID":
      pad_height = 0
      pad_width = 0
    else:
      raise ValueError("Invalid padding")
    # When padding is "SAME" and the pad_height (pad_width) is odd, we pad more
    # to bottom (right), following the same convention as conv2d().
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # More padding so that rate divides the height and width of the input value
    in_height = in_height + pad_top + pad_bottom
    in_width = in_width + pad_left + pad_right
    pad_bottom_extra = 0 if in_height % rate == 0 else rate - in_height % rate
    pad_right_extra = 0 if in_width % rate == 0 else rate - in_width % rate

    # The paddings argument to space_to_batch includes both padding components
    space_to_batch_pad = [[pad_top, pad_bottom + pad_bottom_extra],
                          [pad_left, pad_right + pad_right_extra]]
    value = array_ops.space_to_batch(input=value,
                                     paddings=space_to_batch_pad,
                                     block_size=rate)

    value = gen_nn_ops.conv2d(input=value,
                              filter=filters,
                              strides=[1, 1, 1, 1],
                              padding="VALID",
                              name=name)

    # The crops argument to batch_to_space is just the extra padding component
    batch_to_space_crop = [[0, pad_bottom_extra], [0, pad_right_extra]]
    value = array_ops.batch_to_space(input=value,
                                     crops=batch_to_space_crop,
                                     block_size=rate)

    return value


def conv2d_transpose(value,
                     filter,
                     output_shape,
                     strides,
                     padding="SAME",
                     name=None):
  """The transpose of `conv2d`.

  This operation is sometimes called "deconvolution" after (Deconvolutional
  Networks)[http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf], but is
  actually the transpose (gradient) of `conv2d` rather than an actual
  deconvolution.

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
  with ops.op_scope([value, filter, output_shape], name,
                    "conv2d_transpose") as name:
    value = ops.convert_to_tensor(value, name="value")
    filter = ops.convert_to_tensor(filter, name="filter")
    if not value.get_shape()[3].is_compatible_with(filter.get_shape()[3]):
      raise ValueError("input channels does not match filter's input channels, "
                       "{} != {}".format(value.get_shape()[3], filter.get_shape(
                       )[3]))

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
def bias_add(value, bias, data_format=None, name=None):
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
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `value`.
  """
  with ops.op_scope([value, bias], name, "BiasAdd") as name:
    value = ops.convert_to_tensor(value, name="input")
    bias = ops.convert_to_tensor(bias, dtype=value.dtype, name="bias")
    return gen_nn_ops._bias_add(value, bias, data_format=data_format, name=name)


ops.RegisterShape("BiasAdd")(common_shapes.bias_add_shape)

ops.RegisterShape("BiasAddGrad")(common_shapes.bias_add_grad_shape)


# pylint: disable=protected-access
def bias_add_v1(value, bias, name=None):
  """Adds `bias` to `value`.

  This is a deprecated version of bias_add and will soon to be removed.

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
  with ops.op_scope([value, bias], name, "BiasAddV1") as name:
    value = ops.convert_to_tensor(value, name="input")
    bias = ops.convert_to_tensor(bias, dtype=value.dtype, name="bias")
    return gen_nn_ops._bias_add_v1(value, bias, name=name)


ops.RegisterShape("BiasAddV1")(common_shapes.bias_add_shape)

ops.RegisterShape("BiasAddGradV1")(common_shapes.bias_add_grad_shape)


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

  **NOTE:**  While the classes are mutually exclusive, their probabilities
  need not be.  All that is required is that each row of `labels` is
  a valid probability distribution.  If they are not, the computation of the
  gradient will be incorrect.

  If using exclusive `labels` (wherein one and only
  one class is true at a time), see `sparse_softmax_cross_entropy_with_logits`.

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
  # TODO(pcmurray) Raise an error when the labels do not sum to 1. Note: This
  # could break users who call this with bad labels, but disregard the bad
  # results.

  # The second output tensor contains the gradients.  We use it in
  # _CrossEntropyGrad() in nn_grad but not here.
  cost, unused_backprop = gen_nn_ops._softmax_cross_entropy_with_logits(
      logits, labels, name=name)
  return cost


def sparse_softmax_cross_entropy_with_logits(logits, labels, name=None):
  """Computes sparse softmax cross entropy between `logits` and `labels`.

  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.

  **NOTE:**  For this operation, the probability of a given label is considered
  exclusive.  That is, soft classes are not allowed, and the `labels` vector
  must provide a single specific index for the true class for each row of
  `logits` (each minibatch entry).  For soft softmax classification with
  a probability distribution for each entry, see
  `softmax_cross_entropy_with_logits`.

  **WARNING:** This op expects unscaled logits, since it performs a softmax
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.

  `logits` must have the shape `[batch_size, num_classes]`
  and dtype `float32` or `float64`.

  `labels` must have the shape `[batch_size]` and dtype `int32` or `int64`.

  Args:
    logits: Unscaled log probabilities.
    labels: Each entry `labels[i]` must be an index in `[0, num_classes)`. Other
      values will result in a loss of 0, but incorrect gradient computations.
    name: A name for the operation (optional).

  Returns:
    A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
    softmax cross entropy loss.
  """
  # TODO(pcmurray) Raise an error when the label is not an index in
  # [0, num_classes). Note: This could break users who call this with bad
  # labels, but disregard the bad results.

  # The second output tensor contains the gradients.  We use it in
  # _CrossEntropyGrad() in nn_grad but not here.
  cost, unused_backprop = gen_nn_ops._sparse_softmax_cross_entropy_with_logits(
      logits, labels, name=name)
  return cost


@ops.RegisterShape("SparseSoftmaxCrossEntropyWithLogits")
def _SparseSoftmaxCrossEntropyWithLogitsShape(op):
  """Shape function for SparseSoftmaxCrossEntropyWithLogits op."""
  logits_shape = op.inputs[0].get_shape()
  input_shape = logits_shape.with_rank(2)
  batch_size = input_shape[0]
  # labels_shape
  op.inputs[1].get_shape().merge_with(tensor_shape.vector(batch_size))
  return [tensor_shape.vector(batch_size.value), input_shape]


@ops.RegisterShape("SoftmaxCrossEntropyWithLogits")
def _SoftmaxCrossEntropyWithLogitsShape(op):
  """Shape function for SoftmaxCrossEntropyWithLogits op."""
  logits_shape = op.inputs[0].get_shape()
  labels_shape = op.inputs[1].get_shape()
  input_shape = logits_shape.merge_with(labels_shape).with_rank(2)
  batch_size = input_shape[0]
  return [tensor_shape.vector(batch_size.value), input_shape]


def avg_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
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
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    name: Optional name for the operation.

  Returns:
    A `Tensor` with the same type as `value`.  The average pooled output tensor.
  """
  with ops.op_scope([value], name, "AvgPool") as name:
    value = ops.convert_to_tensor(value, name="input")
    return gen_nn_ops._avg_pool(value,
                                ksize=ksize,
                                strides=strides,
                                padding=padding,
                                data_format=data_format,
                                name=name)


def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
  """Performs the max pooling on the input.

  Args:
    value: A 4-D `Tensor` with shape `[batch, height, width, channels]` and
      type `tf.float32`.
    ksize: A list of ints that has length >= 4.  The size of the window for
      each dimension of the input tensor.
    strides: A list of ints that has length >= 4.  The stride of the sliding
      window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    name: Optional name for the operation.

  Returns:
    A `Tensor` with type `tf.float32`.  The max pooled output tensor.
  """
  with ops.op_scope([value], name, "MaxPool") as name:
    value = ops.convert_to_tensor(value, name="input")
    return gen_nn_ops._max_pool(value,
                                ksize=ksize,
                                strides=strides,
                                padding=padding,
                                data_format=data_format,
                                name=name)


ops.RegisterShape("Relu")(common_shapes.unchanged_shape)
ops.RegisterShape("Relu6")(common_shapes.unchanged_shape)
ops.RegisterShape("Elu")(common_shapes.unchanged_shape)
ops.RegisterShape("Softplus")(common_shapes.unchanged_shape)
ops.RegisterShape("Softsign")(common_shapes.unchanged_shape)


@ops.RegisterShape("ReluGrad")
@ops.RegisterShape("Relu6Grad")
@ops.RegisterShape("EluGrad")
@ops.RegisterShape("SoftplusGrad")
@ops.RegisterShape("SoftsignGrad")
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


ops.RegisterShape("Softmax")(common_shapes.unchanged_shape_with_rank(2))

ops.RegisterShape("LogSoftmax")(common_shapes.unchanged_shape_with_rank(2))


@ops.RegisterShape("InTopK")
def _InTopKShape(op):
  """Shape function for InTopK op."""
  predictions_shape = op.inputs[0].get_shape().with_rank(2)
  targets_shape = op.inputs[1].get_shape().with_rank(1)
  batch_size = predictions_shape[0].merge_with(targets_shape[0])
  return [tensor_shape.vector(batch_size.value)]


@ops.RegisterShape("TopK")
@ops.RegisterShape("TopKV2")
def _TopKShape(op):
  """Shape function for TopK and TopKV2 ops."""
  input_shape = op.inputs[0].get_shape().with_rank_at_least(1)
  if len(op.inputs) >= 2:
    k = tensor_util.constant_value(op.inputs[1])
  else:
    k = op.get_attr("k")
  last = input_shape[-1].value
  if last is not None and k is not None and last < k:
    raise ValueError("input.shape %s must have last dimension >= k = %d" %
                     (input_shape, k))
  output_shape = input_shape[:-1].concatenate([k])
  return [output_shape, output_shape]


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
ops.RegisterShape("DepthwiseConv2dNative")(
    common_shapes.depthwise_conv2d_native_shape)
ops.RegisterShape("AvgPool")(common_shapes.avg_pool_shape)
ops.RegisterShape("MaxPool")(common_shapes.max_pool_shape)


@ops.RegisterShape("MaxPoolWithArgmax")
def _MaxPoolWithArgMaxShape(op):
  """Shape function for MaxPoolWithArgmax op."""
  return common_shapes.max_pool_shape(op) * 2


@ops.RegisterShape("AvgPoolGrad")
def _AvgPoolGradShape(op):
  """Shape function for the AvgPoolGrad op."""
  orig_input_shape = tensor_util.constant_value(op.inputs[0])
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
  filter_shape = tensor_util.constant_value(op.inputs[1])
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
  input_shape = tensor_util.constant_value(op.inputs[0])
  if input_shape is not None:
    return [tensor_shape.TensorShape(input_shape.tolist())]
  else:
    # NOTE(mrry): We could in principle work out the shape from the
    # gradients and the attrs, but if we do not know input_shape
    # statically, then we are unlikely to know the shape of the
    # gradients either.
    return [tensor_shape.unknown_shape(ndims=4)]


@ops.RegisterShape("DepthwiseConv2dNativeBackpropFilter")
def _DepthwiseConv2dNativeBackpropFilterShape(op):
  """Shape function for the DepthwiseConv2dNativeBackpropFilter op."""
  filter_shape = tensor_util.constant_value(op.inputs[1])
  if filter_shape is not None:
    return [tensor_shape.TensorShape(filter_shape.tolist())]
  else:
    return [tensor_shape.unknown_shape(ndims=4)]


@ops.RegisterShape("DepthwiseConv2dNativeBackpropInput")
def _DepthwiseConv2dNativeBackpropInputShape(op):
  """Shape function for the DepthwiseConv2dNativeBackpropInput op."""
  input_shape = tensor_util.constant_value(op.inputs[0])
  if input_shape is not None:
    return [tensor_shape.TensorShape(input_shape.tolist())]
  else:
    return [tensor_shape.unknown_shape(ndims=4)]


@ops.RegisterShape("MaxPoolGrad")
@ops.RegisterShape("MaxPoolGradWithArgmax")
def _MaxPoolGradShape(op):
  """Shape function for the MaxPoolGrad op."""
  orig_input_shape = op.inputs[0].get_shape().with_rank(4)
  return [orig_input_shape]


@ops.RegisterStatistics("Conv2D", "flops")
def _calc_conv_flops(graph, node):
  """Calculates the compute resources needed for Conv2D."""
  input_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  input_shape.assert_is_fully_defined()
  filter_shape = graph_util.tensor_shape_from_node_def_name(graph,
                                                            node.input[1])
  filter_shape.assert_is_fully_defined()
  output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  output_shape.assert_is_fully_defined()
  filter_height = int(filter_shape[0])
  filter_width = int(filter_shape[1])
  filter_in_depth = int(filter_shape[2])
  output_count = np.prod(output_shape.as_list())
  return ops.OpStats("flops", (output_count * filter_in_depth * filter_height *
                               filter_width * 2))


@ops.RegisterStatistics("Conv2D", "weight_parameters")
def _calc_conv_weight_params(graph, node):
  """Calculates the on-disk size of the weights for Conv2D."""
  input_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  input_shape.assert_is_fully_defined()
  filter_shape = graph_util.tensor_shape_from_node_def_name(graph,
                                                            node.input[1])
  filter_shape.assert_is_fully_defined()
  output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  output_shape.assert_is_fully_defined()
  filter_height = int(filter_shape[0])
  filter_width = int(filter_shape[1])
  filter_in_depth = int(filter_shape[2])
  filter_out_depth = int(filter_shape[3])
  return ops.OpStats("weight_parameters", (filter_height * filter_width *
                                           filter_in_depth * filter_out_depth))


@ops.RegisterShape("Conv3D")
def _Conv3DShape(op):
  """Shape function for Conv3D."""
  input_shape = op.inputs[0].get_shape().with_rank(5)
  filter_shape = op.inputs[1].get_shape().with_rank(5)

  batch_size = input_shape[0]
  out_channels = filter_shape[4]
  # Check that the input number of channels is compatible between
  # input data and filter size.
  input_shape[4].assert_is_compatible_with(filter_shape[3])

  stride_b, stride_p, stride_r, stride_c, stride_d = op.get_attr("strides")
  assert stride_b == 1
  assert stride_d == 1

  padding_type = op.get_attr("padding")
  out_planes, out_rows, out_cols = common_shapes.get_conv_output_size(
      input_shape[1:4], filter_shape[0:3], (stride_p, stride_r, stride_c),
      padding_type)

  return [tensor_shape.TensorShape([batch_size, out_planes, out_rows, out_cols,
                                    out_channels])]


@ops.RegisterShape("MaxPool3D")
@ops.RegisterShape("AvgPool3D")
def _Pool3DShape(op):
  """Shape function for Max/AvgPool3D."""
  input_shape = op.inputs[0].get_shape().with_rank(5)
  ksize_b, ksize_p, ksize_r, ksize_c, ksize_d = op.get_attr("ksize")
  assert ksize_b == 1
  assert ksize_d == 1

  stride_b, stride_p, stride_r, stride_c, stride_d = op.get_attr("strides")
  assert stride_b == 1
  assert stride_d == 1

  batch_size = input_shape[0]
  channels = input_shape[4]

  padding = op.get_attr("padding")
  out_planes, out_rows, out_cols = common_shapes.get_conv_output_size(
      input_shape[1:4], (ksize_p, ksize_r, ksize_c),
      (stride_p, stride_r, stride_c), padding)
  return [tensor_shape.TensorShape([batch_size, out_planes, out_rows, out_cols,
                                    channels])]


def _ShapeOrUnknown(input_shape, ndims=5):
  if input_shape == None:  # pylint:disable=g-equals-none
    return [tensor_shape.unknown_shape(ndims=ndims)]
  else:
    return [input_shape]


@ops.RegisterShape("Conv3DBackpropFilter")
def _Conv3DBackpropFilterShape(op):
  """Shape function for the Conv3DBackpropFilter op."""
  filter_shape = op.inputs[1].get_shape()
  return _ShapeOrUnknown(filter_shape)


@ops.RegisterShape("Conv3DBackpropInput")
def _Conv3DBackpropInputShape(op):
  """Shape function for the Conv3DBackpropInput op."""
  input_shape = op.inputs[0].get_shape()
  return _ShapeOrUnknown(input_shape)


@ops.RegisterShape("AvgPool3DGrad")
def _AvgPool3DGradShape(op):
  """Shape function for the AvgPool3DGrad op."""
  orig_input_shape = tensor_util.constant_value(op.inputs[0])
  if orig_input_shape != None:  # pylint:disable=g-equals-none
    return [tensor_shape.TensorShape(orig_input_shape.tolist())]
  else:
    return [tensor_shape.unknown_shape(ndims=5)]


@ops.RegisterShape("MaxPool3DGrad")
def _MaxPool3DGradShape(op):
  """Shape function for the MaxPoolGrad op."""
  orig_input_shape = op.inputs[0].get_shape().with_rank(5)
  return [orig_input_shape]


@ops.RegisterStatistics("BiasAdd", "flops")
def _calc_bias_add_flops(graph, node):
  """Calculates the computing needed for BiasAdd."""
  input_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  input_shape.assert_is_fully_defined()
  input_count = np.prod(input_shape.as_list())
  return ops.OpStats("flops", input_count)


@ops.RegisterStatistics("BiasAdd", "weight_parameters")
def _calc_bias_add_weight_params(graph, node):
  """Calculates the on-disk weight parameters for BiasAdd."""
  bias_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[1])
  bias_shape.assert_is_fully_defined()
  bias_count = np.prod(bias_shape.as_list())
  return ops.OpStats("weight_parameters", bias_count)


def xw_plus_b(x, weights, biases, name=None):  # pylint: disable=invalid-name
  """Computes matmul(x, weights) + biases.

  Args:
    x: a 2D tensor.  Dimensions typically: batch, in_units
    weights: a 2D tensor.  Dimensions typically: in_units, out_units
    biases: a 1D tensor.  Dimensions: out_units
    name: A name for the operation (optional).  If not specified
      "xw_plus_b" is used.

  Returns:
    A 2-D Tensor computing matmul(x, weights) + biases.
    Dimensions typically: batch, out_units.
  """
  with ops.op_scope([x, weights, biases], name, "xw_plus_b") as name:
    x = ops.convert_to_tensor(x, name="x")
    weights = ops.convert_to_tensor(weights, name="weights")
    biases = ops.convert_to_tensor(biases, name="biases")
    mm = math_ops.matmul(x, weights)
    return bias_add(mm, biases, name=name)


def xw_plus_b_v1(x, weights, biases, name=None):  # pylint: disable=invalid-name
  """Computes matmul(x, weights) + biases.

  This is a deprecated version of that will soon be removed.

  Args:
    x: a 2D tensor.  Dimensions typically: batch, in_units
    weights: a 2D tensor.  Dimensions typically: in_units, out_units
    biases: a 1D tensor.  Dimensions: out_units
    name: A name for the operation (optional).  If not specified
      "xw_plus_b_v1" is used.

  Returns:
    A 2-D Tensor computing matmul(x, weights) + biases.
    Dimensions typically: batch, out_units.
  """
  with ops.op_scope([x, weights, biases], name, "xw_plus_b_v1") as name:
    x = ops.convert_to_tensor(x, name="x")
    weights = ops.convert_to_tensor(weights, name="weights")
    biases = ops.convert_to_tensor(biases, name="biases")
    mm = math_ops.matmul(x, weights)
    return bias_add_v1(mm, biases, name=name)


# pylint: disable=invalid-name
def dropout(x, keep_prob, noise_shape=None, seed=None, name=None):
  """Computes dropout.

  With probability `keep_prob`, outputs the input element scaled up by
  `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
  sum is unchanged.

  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.

  Args:
    x: A tensor.
    keep_prob: A scalar `Tensor` with the same type as x. The probability
      that each element is kept.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for this operation (optional).

  Returns:
    A Tensor of the same shape of `x`.

  Raises:
    ValueError: If `keep_prob` is not in `(0, 1]`.
  """
  with ops.op_scope([x], name, "dropout") as name:
    x = ops.convert_to_tensor(x, name="x")
    if isinstance(keep_prob, float) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
    keep_prob = ops.convert_to_tensor(keep_prob,
                                      dtype=x.dtype,
                                      name="keep_prob")
    keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob
    random_tensor += random_ops.random_uniform(noise_shape,
                                               seed=seed,
                                               dtype=x.dtype)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = math_ops.floor(random_tensor)
    ret = x * math_ops.inv(keep_prob) * binary_tensor
    ret.set_shape(x.get_shape())
    return ret


def top_k(input, k=1, sorted=True, name=None):
  """Finds values and indices of the `k` largest entries for the last dimension.

  If the input is a vector (rank-1), finds the `k` largest entries in the vector
  and outputs their values and indices as vectors.  Thus `values[j]` is the
  `j`-th largest entry in `input`, and its index is `indices[j]`.

  For matrices (resp. higher rank input), computes the top `k` entries in each
  row (resp. vector along the last dimension).  Thus,

      values.shape = indices.shape = input.shape[:-1] + [k]

  If two elements are equal, the lower-index element appears first.

  Args:
    input: 1-D or higher `Tensor` with last dimension at least `k`.
    k: 0-D `int32` `Tensor`.  Number of top elements to look for along the last
      dimension (along each row for matrices).
    sorted: If true the resulting `k` elements will be sorted by the values in
      descending order.
    name: Optional name for the operation.

  Returns:
    values: The `k` largest elements along each last dimensional slice.
    indices: The indices of `values` within the last dimension of `input`.
  """
  return gen_nn_ops._top_kv2(input, k=k, sorted=sorted, name=name)

# pylint: enable=invalid-name

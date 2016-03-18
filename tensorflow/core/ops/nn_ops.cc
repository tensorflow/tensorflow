/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
namespace tensorflow {

// --------------------------------------------------------------------------

REGISTER_OP("AvgPool")
    .Input("value: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("T: {float, double}")
    .Doc(R"doc(
Performs average pooling on the input.

Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.

value: 4-D with shape `[batch, height, width, channels]`.
ksize: The size of the sliding window for each dimension of `value`.
strides: The stride of the sliding window for each dimension of `value`.
padding: The type of padding algorithm to use.
data_format: Specify the data format of the input and output data. With the
    default format "NHWC", the data is stored in the order of:
        [batch, in_height, in_width, in_channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, in_channels, in_height, in_width].
output: The average pooled output tensor.
)doc");

REGISTER_OP("AvgPoolGrad")
    .Input("orig_input_shape: int32")
    .Input("grad: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("T: {float, double}")
    .Doc(R"doc(
Computes gradients of the average pooling function.

orig_input_shape: 1-D.  Shape of the original input to `avg_pool`.
grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t.
  the output of `avg_pool`.
ksize: The size of the sliding window for each dimension of the input.
strides: The stride of the sliding window for each dimension of the input.
padding: The type of padding algorithm to use.
data_format: Specify the data format of the input and output data. With the
    default format "NHWC", the data is stored in the order of:
        [batch, in_height, in_width, in_channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, in_channels, in_height, in_width].
output: 4-D.  Gradients w.r.t. the input of `avg_pool`.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("BatchNormWithGlobalNormalization")
    .Input("t: T")
    .Input("m: T")
    .Input("v: T")
    .Input("beta: T")
    .Input("gamma: T")
    .Output("result: T")
    .Attr("T: numbertype")
    .Attr("variance_epsilon: float")
    .Attr("scale_after_normalization: bool")
    .Doc(R"doc(
Batch normalization.

This op is deprecated. Prefer `tf.nn.batch_normalization`.

t: A 4D input Tensor.
m: A 1D mean Tensor with size matching the last dimension of t.
  This is the first output from tf.nn.moments,
  or a saved moving average thereof.
v: A 1D variance Tensor with size matching the last dimension of t.
  This is the second output from tf.nn.moments,
  or a saved moving average thereof.
beta: A 1D beta Tensor with size matching the last dimension of t.
  An offset to be added to the normalized tensor.
gamma: A 1D gamma Tensor with size matching the last dimension of t.
  If "scale_after_normalization" is true, this tensor will be multiplied
  with the normalized tensor.
variance_epsilon: A small float number to avoid dividing by 0.
scale_after_normalization: A bool indicating whether the resulted tensor
  needs to be multiplied with gamma.
)doc");

REGISTER_OP("BatchNormWithGlobalNormalizationGrad")
    .Input("t: T")
    .Input("m: T")
    .Input("v: T")
    .Input("gamma: T")
    .Input("backprop: T")
    .Output("dx: T")
    .Output("dm: T")
    .Output("dv: T")
    .Output("db: T")
    .Output("dg: T")
    .Attr("T: numbertype")
    .Attr("variance_epsilon: float")
    .Attr("scale_after_normalization: bool")
    .Doc(R"doc(
Gradients for batch normalization.

This op is deprecated. See `tf.nn.batch_normalization`.

t: A 4D input Tensor.
m: A 1D mean Tensor with size matching the last dimension of t.
  This is the first output from tf.nn.moments,
  or a saved moving average thereof.
v: A 1D variance Tensor with size matching the last dimension of t.
  This is the second output from tf.nn.moments,
  or a saved moving average thereof.
gamma: A 1D gamma Tensor with size matching the last dimension of t.
  If "scale_after_normalization" is true, this Tensor will be multiplied
  with the normalized Tensor.
backprop: 4D backprop Tensor.
variance_epsilon: A small float number to avoid dividing by 0.
scale_after_normalization: A bool indicating whether the resulted tensor
  needs to be multiplied with gamma.

dx: 4D backprop tensor for input.
dm: 1D backprop tensor for mean.
dv: 1D backprop tensor for variance.
db: 1D backprop tensor for beta.
dg: 1D backprop tensor for gamma.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("BiasAdd")
    .Attr("T: numbertype")
    .Input("value: T")
    .Input("bias: T")
    .Attr(GetConvnetDataFormatAttrString())
    .Output("output: T")
    .Doc(R"doc(
Adds `bias` to `value`.

This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions.

value: Any number of dimensions.
bias: 1-D with size the last dimension of `value`.
data_format: Specify the data format of the input and output data. With the
    default format "NHWC", the bias tensor will be added to the last dimension
    of the value tensor.
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, in_channels, in_height, in_width].
    The tensor will be added to "in_channels", the third-to-the-last
        dimension.
output: Broadcasted sum of `value` and `bias`.
)doc");
// --------------------------------------------------------------------------

REGISTER_OP("BiasAddGrad")
    .Attr("T: numbertype")
    .Input("out_backprop: T")
    .Attr(GetConvnetDataFormatAttrString())
    .Output("output: T")
    .Doc(R"doc(
The backward operation for "BiasAdd" on the "bias" tensor.

It accumulates all the values from out_backprop into the feature dimension.
For NHWC data format, the feature dimension is the last. For NCHW data format,
the feature dimension is the third-to-last.

out_backprop: Any number of dimensions.
output: 1-D with size the feature dimension of `out_backprop`.
data_format: Specify the data format of the input and output data. With the
    default format "NHWC", the bias tensor will be added to the last dimension
    of the value tensor.
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, in_channels, in_height, in_width].
    The tensor will be added to "in_channels", the third-to-the-last
        dimension.
)doc");
// --------------------------------------------------------------------------

REGISTER_OP("BiasAddV1")
    .Attr("T: numbertype")
    .Input("value: T")
    .Input("bias: T")
    .Output("output: T")
    .Doc(R"doc(
Adds `bias` to `value`.

This is a deprecated version of BiasAdd and will be soon removed.

This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions.

value: Any number of dimensions.
bias: 1-D with size the last dimension of `value`.
output: Broadcasted sum of `value` and `bias`.
)doc");
// --------------------------------------------------------------------------

REGISTER_OP("Conv2D")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Doc(R"doc(
Computes a 2-D convolution given 4-D `input` and `filter` tensors.

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, out_channels]`, this op
performs the following:

1. Flattens the filter to a 2-D matrix with shape
   `[filter_height * filter_width * in_channels, output_channels]`.
2. Extracts image patches from the input tensor to form a *virtual*
   tensor of shape `[batch, out_height, out_width,
   filter_height * filter_width * in_channels]`.
3. For each patch, right-multiplies the filter matrix and the image patch
   vector.

In detail, with the default NHWC format,

    output[b, i, j, k] =
        sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                        filter[di, dj, q, k]

Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

strides: 1-D of length 4.  The stride of the sliding window for each dimension
  of `input`. Must be in the same order as the dimension specified with format.
padding: The type of padding algorithm to use.
data_format: Specify the data format of the input and output data. With the
    default format "NHWC", the data is stored in the order of:
        [batch, in_height, in_width, in_channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, in_channels, in_height, in_width].
)doc");

REGISTER_OP("Conv2DBackpropInput")
    .Input("input_sizes: int32")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Doc(R"doc(
Computes the gradients of convolution with respect to the input.

input_sizes: An integer vector representing the shape of `input`,
  where `input` is a 4-D `[batch, height, width, channels]` tensor.
filter: 4-D with shape
  `[filter_height, filter_width, in_channels, out_channels]`.
out_backprop: 4-D with shape `[batch, out_height, out_width, out_channels]`.
  Gradients w.r.t. the output of the convolution.
strides: The stride of the sliding window for each dimension of the input
  of the convolution. Must be in the same order as the dimension specified with
  format.
padding: The type of padding algorithm to use.
output: 4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
  w.r.t. the input of the convolution.
data_format: Specify the data format of the input and output data. With the
    default format "NHWC", the data is stored in the order of:
        [batch, in_height, in_width, in_channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, in_channels, in_height, in_width].
)doc");

// TODO(jeff): Instead of 'use_cudnn_for_gpu', maybe we should have a
// more general string attribute ('kernel_impl'?) that can be used to
// select among several possible implementations.
REGISTER_OP("Conv2DBackpropFilter")
    .Input("input: T")
    .Input("filter_sizes: int32")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Doc(R"doc(
Computes the gradients of convolution with respect to the filter.

input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
filter_sizes: An integer vector representing the tensor shape of `filter`,
  where `filter` is a 4-D
  `[filter_height, filter_width, in_channels, out_channels]` tensor.
out_backprop: 4-D with shape `[batch, out_height, out_width, out_channels]`.
  Gradients w.r.t. the output of the convolution.
strides: The stride of the sliding window for each dimension of the input
  of the convolution. Must be in the same order as the dimension specified with
  format.
padding: The type of padding algorithm to use.
output: 4-D with shape
  `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
  the `filter` input of the convolution.
data_format: Specify the data format of the input and output data. With the
    default format "NHWC", the data is stored in the order of:
        [batch, in_height, in_width, in_channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, in_channels, in_height, in_width].
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("DepthwiseConv2dNative")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Doc(R"doc(
Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, channel_multiplier]`, containing
`in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
a different filter to each input channel (expanding from 1 channel to
`channel_multiplier` channels for each), then concatenates the results
together. Thus, the output has `in_channels * channel_multiplier` channels.

for k in 0..in_channels-1
  for q in 0..channel_multiplier-1
    output[b, i, j, k * channel_multiplier + q] =
      sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                        filter[di, dj, k, q]

Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

strides: 1-D of length 4.  The stride of the sliding window for each dimension
  of `input`.
padding: The type of padding algorithm to use.
)doc");

REGISTER_OP("DepthwiseConv2dNativeBackpropInput")
    .Input("input_sizes: int32")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Doc(R"doc(
Computes the gradients of depthwise convolution with respect to the input.

input_sizes: An integer vector representing the shape of `input`,
  where `input` is a 4-D `[batch, height, width, channels]` tensor.
filter: 4-D with shape
  `[filter_height, filter_width, in_channels, depthwise_multiplier]`.
out_backprop: 4-D with shape `[batch, out_height, out_width, out_channels]`.
  Gradients w.r.t. the output of the convolution.
strides: The stride of the sliding window for each dimension of the input
  of the convolution.
padding: The type of padding algorithm to use.
output: 4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
  w.r.t. the input of the convolution.
)doc");

REGISTER_OP("DepthwiseConv2dNativeBackpropFilter")
    .Input("input: T")
    .Input("filter_sizes: int32")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Doc(R"doc(
Computes the gradients of depthwise convolution with respect to the filter.

input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
filter_sizes: An integer vector representing the tensor shape of `filter`,
  where `filter` is a 4-D
  `[filter_height, filter_width, in_channels, depthwise_multiplier]` tensor.
out_backprop: 4-D with shape `[batch, out_height, out_width, out_channels]`.
  Gradients w.r.t. the output of the convolution.
strides: The stride of the sliding window for each dimension of the input
  of the convolution.
padding: The type of padding algorithm to use.
output: 4-D with shape
  `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
  the `filter` input of the convolution.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("L2Loss")
    .Input("t: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Doc(R"doc(
L2 Loss.

Computes half the L2 norm of a tensor without the `sqrt`:

    output = sum(t ** 2) / 2

t: Typically 2-D, but may have any dimensions.
output: 0-D.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("LRN")
    .Input("input: float")
    .Output("output: float")
    .Attr("depth_radius: int = 5")
    .Attr("bias: float = 1.0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.5")
    .Doc(R"doc(
Local Response Normalization.

The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
dimension), and each vector is normalized independently.  Within a given vector,
each component is divided by the weighted, squared sum of inputs within
`depth_radius`.  In detail,

    sqr_sum[a, b, c, d] =
        sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
    output = input / (bias + alpha * sqr_sum) ** beta

For details, see [Krizhevsky et al., ImageNet classification with deep
convolutional neural networks (NIPS 2012)]
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

input: 4-D.
depth_radius: 0-D.  Half-width of the 1-D normalization window.
bias: An offset (usually positive to avoid dividing by 0).
alpha: A scale factor, usually positive.
beta: An exponent.
)doc");

REGISTER_OP("LRNGrad")
    .Input("input_grads: float")
    .Input("input_image: float")
    .Input("output_image: float")
    .Output("output: float")
    .Attr("depth_radius: int = 5")
    .Attr("bias: float = 1.0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.5")
    .Doc(R"doc(
Gradients for Local Response Normalization.

input_grads: 4-D with shape `[batch, height, width, channels]`.
input_image: 4-D with shape `[batch, height, width, channels]`.
output_image: 4-D with shape `[batch, height, width, channels]`.
depth_radius: A depth radius.
bias: An offset (usually > 0 to avoid dividing by 0).
alpha: A scale factor, usually positive.
beta: An exponent.
output: The gradients for LRN.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("MaxPool")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Input("input: float")
    .Output("output: float")
    .Doc(R"doc(
Performs max pooling on the input.

ksize: The size of the window for each dimension of the input tensor.
strides: The stride of the sliding window for each dimension of the
  input tensor.
padding: The type of padding algorithm to use.
data_format: Specify the data format of the input and output data. With the
    default format "NHWC", the data is stored in the order of:
        [batch, in_height, in_width, in_channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, in_channels, in_height, in_width].
input: 4-D input to pool over.
output: The max pooled output tensor.
)doc");

REGISTER_OP("MaxPoolGrad")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Input("orig_input: float")
    .Input("orig_output: float")
    .Input("grad: float")
    .Output("output: float")
    .Doc(R"doc(
Computes gradients of the maxpooling function.

ksize: The size of the window for each dimension of the input tensor.
strides: The stride of the sliding window for each dimension of the
  input tensor.
padding: The type of padding algorithm to use.
data_format: Specify the data format of the input and output data. With the
    default format "NHWC", the data is stored in the order of:
        [batch, in_height, in_width, in_channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, in_channels, in_height, in_width].
orig_input: The original input tensor.
orig_output: The original output tensor.
grad: 4-D.  Gradients w.r.t. the output of `max_pool`.
output: Gradients w.r.t. the input to `max_pool`.
)doc");

REGISTER_OP("MaxPoolWithArgmax")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr("Targmax: {int32, int64} = DT_INT64")
    .Attr(GetPaddingAttrString())
    .Input("input: float")
    .Output("output: float")
    .Output("argmax: Targmax")
    .Doc(R"doc(
Performs max pooling on the input and outputs both max values and indices.

The indices in `argmax` are flattened, so that a maximum value at position
`[b, y, x, c]` becomes flattened index
`((b * height + y) * width + x) * channels + c`.

ksize: The size of the window for each dimension of the input tensor.
strides: The stride of the sliding window for each dimension of the
  input tensor.
padding: The type of padding algorithm to use.
input: 4-D with shape `[batch, height, width, channels]`.  Input to pool over.
output: The max pooled output tensor.
argmax: 4-D.  The flattened indices of the max values chosen for each output.
)doc");

REGISTER_OP("MaxPoolGradWithArgmax")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr("Targmax: {int32, int64}")
    .Input("input: float")
    .Input("grad: float")
    .Input("argmax: Targmax")
    .Output("output: float")
    .Doc(R"doc(
Computes gradients of the maxpooling function.

ksize: The size of the window for each dimension of the input tensor.
strides: The stride of the sliding window for each dimension of the
  input tensor.
padding: The type of padding algorithm to use.
input: The original input.
grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
  output of `max_pool`.
argmax: The indices of the maximum values chosen for each output of `max_pool`.
output: Gradients w.r.t. the input of `max_pool`.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("Relu")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes rectified linear: `max(features, 0)`.
)doc");

REGISTER_OP("ReluGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes rectified linear gradients for a Relu operation.

gradients: The backpropagated gradients to the corresponding Relu operation.
features: The features passed as input to the corresponding Relu operation, OR
  the outputs of that operation (both work equivalently).
backprops: `gradients * (features > 0)`.
)doc");

REGISTER_OP("Relu6")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes rectified linear 6: `min(max(features, 0), 6)`.
)doc");

REGISTER_OP("Relu6Grad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes rectified linear 6 gradients for a Relu6 operation.

gradients: The backpropagated gradients to the corresponding Relu6 operation.
features: The features passed as input to the corresponding Relu6 operation.
backprops: The gradients:
  `gradients * features * (features > 0) * (features < 6)`.
)doc");

REGISTER_OP("Elu")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289)
)doc");

REGISTER_OP("EluGrad")
    .Input("gradients: T")
    .Input("outputs: T")
    .Output("backprops: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Computes gradients for the exponential linear (Elu) operation.

gradients: The backpropagated gradients to the corresponding Elu operation.
outputs: The outputs of the corresponding Elu operation.
backprops: The gradients: `gradients * (outputs + 1)` if outputs < 0,
`gradients` otherwise.
)doc");

REGISTER_OP("Softplus")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes softplus: `log(exp(features) + 1)`.
)doc");

REGISTER_OP("SoftplusGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes softplus gradients for a softplus operation.

gradients: The backpropagated gradients to the corresponding softplus operation.
features: The features passed as input to the corresponding softplus operation.
backprops: The gradients: `gradients / (1 + exp(-features))`.
)doc");

REGISTER_OP("Softsign")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes softsign: `features / (abs(features) + 1)`.
)doc");

REGISTER_OP("SoftsignGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes softsign gradients for a softsign operation.

gradients: The backpropagated gradients to the corresponding softsign operation.
features: The features passed as input to the corresponding softsign operation.
backprops: The gradients: `gradients / (1 + abs(-features)) ** 2`.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("Softmax")
    .Input("logits: T")
    .Output("softmax: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Computes softmax activations.

For each batch `i` and class `j` we have

    softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))

logits: 2-D with shape `[batch_size, num_classes]`.
softmax: Same shape as `logits`.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("LogSoftmax")
    .Input("logits: T")
    .Output("logsoftmax: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Computes log softmax activations.

For each batch `i` and class `j` we have

    logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))

logits: 2-D with shape `[batch_size, num_classes]`.
logsoftmax: Same shape as `logits`.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("SoftmaxCrossEntropyWithLogits")
    .Input("features: T")
    .Input("labels: T")
    .Output("loss: T")
    .Output("backprop: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Computes softmax cross entropy cost and gradients to backpropagate.

Inputs are the logits, not probabilities.

features: batch_size x num_classes matrix
labels: batch_size x num_classes matrix
  The caller must ensure that each batch of labels represents a valid
  probability distribution.
loss: Per example loss (batch_size vector).
backprop: backpropagated gradients (batch_size x num_classes matrix).
)doc");

REGISTER_OP("SparseSoftmaxCrossEntropyWithLogits")
    .Input("features: T")
    .Input("labels: Tlabels")
    .Output("loss: T")
    .Output("backprop: T")
    .Attr("T: {float, double}")
    .Attr("Tlabels: {int32, int64} = DT_INT64")
    .Doc(R"doc(
Computes softmax cross entropy cost and gradients to backpropagate.

Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
a matrix of label probabilities, but rather a single label per row
of features.  This label is considered to have probability 1.0 for the
given row.

Inputs are the logits, not probabilities.

features: batch_size x num_classes matrix
labels: batch_size vector with values in [0, num_classes).
  This is the label for the given minibatch entry.
loss: Per example loss (batch_size vector).
backprop: backpropagated gradients (batch_size x num_classes matrix).
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("InTopK")
    .Input("predictions: float")
    .Input("targets: T")
    .Output("precision: bool")
    .Attr("k: int")
    .Attr("T: {int32, int64} = DT_INT32")
    .Doc(R"doc(
Says whether the targets are in the top `K` predictions.

This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
prediction for the target class is among the top `k` predictions among
all predictions for example `i`. Note that the behavior of `InTopK` differs
from the `TopK` op in its handling of ties; if multiple classes have the
same prediction value and straddle the top-`k` boundary, all of those
classes are considered to be in the top `k`.

More formally, let

  \\(predictions_i\\) be the predictions for all classes for example `i`,
  \\(targets_i\\) be the target class for example `i`,
  \\(out_i\\) be the output for example `i`,

$$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$

predictions: A `batch_size` x `classes` tensor.
targets: A `batch_size` vector of class ids.
k: Number of top elements to look at for computing precision.
precision: Computed Precision at `k` as a `bool Tensor`.

)doc");

REGISTER_OP("TopK")
    .Input("input: T")
    .Output("values: T")
    .Output("indices: int32")
    .Attr("k: int >= 0")
    .Attr("sorted: bool = true")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Finds values and indices of the `k` largest elements for the last dimension.

If the input is a vector (rank-1), finds the `k` largest entries in the vector
and outputs their values and indices as vectors.  Thus `values[j]` is the
`j`-th largest entry in `input`, and its index is `indices[j]`.

For matrices (resp. higher rank input), computes the top `k` entries in each
row (resp. vector along the last dimension).  Thus,

    values.shape = indices.shape = input.shape[:-1] + [k]

If two elements are equal, the lower-index element appears first.

If `k` varies dynamically, use `TopKV2` below.

input: 1-D or higher with last dimension at least `k`.
k: Number of top elements to look for along the last dimension (along each
  row for matrices).
sorted: If true the resulting `k` elements will be sorted by the values in
  descending order.
values: The `k` largest elements along each last dimensional slice.
indices: The indices of `values` within the last dimension of `input`.
)doc");

REGISTER_OP("TopKV2")
    .Input("input: T")
    .Input("k: int32")
    .Output("values: T")
    .Output("indices: int32")
    .Attr("sorted: bool = true")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Finds values and indices of the `k` largest elements for the last dimension.

If the input is a vector (rank-1), finds the `k` largest entries in the vector
and outputs their values and indices as vectors.  Thus `values[j]` is the
`j`-th largest entry in `input`, and its index is `indices[j]`.

For matrices (resp. higher rank input), computes the top `k` entries in each
row (resp. vector along the last dimension).  Thus,

    values.shape = indices.shape = input.shape[:-1] + [k]

If two elements are equal, the lower-index element appears first.

This is the same as `TopK`, but takes `k` as in input rather than an attr.

input: 1-D or higher with last dimension at least `k`.
k: 0-D.  Number of top elements to look for along the last dimension (along each
  row for matrices).
sorted: If true the resulting `k` elements will be sorted by the values in
  descending order.
values: The `k` largest elements along each last dimensional slice.
indices: The indices of `values` within the last dimension of `input`.
)doc");

}  // namespace tensorflow

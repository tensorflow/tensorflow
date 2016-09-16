/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

// A shape function that uses the tensor value at <input_idx> as a shape for
// output 0. If the tensor value is not available, it uses a shape with <ndims>
// unknown dims.
Status InputTensorShapeOrUnknown(InferenceContext* c, int input_idx,
                                 int ndims) {
  ShapeHandle out;
  const Tensor* input = c->input_tensor(input_idx);
  if (input == nullptr) {
    out = c->UnknownShapeOfRank(ndims);
  } else {
    TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(input_idx, &out));
  }
  c->set_output(0, out);
  return Status::OK();
}

Status FractionalPoolShapeFn(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));

  std::vector<float> pooling_ratio;
  TF_RETURN_IF_ERROR(c->GetAttr("pooling_ratio", &pooling_ratio));
  if (pooling_ratio.size() != 4) {
    return errors::InvalidArgument(
        "pooling_ratio field must specify 4 dimensions");
  }
  std::vector<DimensionHandle> output_dims;
  for (int i = 0; i < 4; ++i) {
    DimensionHandle d = c->Dim(input, i);
    if (c->ValueKnown(d)) {
      // This must match the same logic in the kernel function in
      // core/kernels/fractional_max_pool_op.cc.
      auto val = static_cast<int64>(floor(c->Value(d) / pooling_ratio[i]));
      if (val < 0) {
        return errors::InvalidArgument("Size computed for dim ", i,
                                       " is negative: ", val);
      }
      output_dims.push_back(c->MakeDim(val));
    } else {
      output_dims.push_back(c->UnknownDim());
    }
  }

  c->set_output(0, c->MakeShape(output_dims));
  c->set_output(1, c->Vector(output_dims[1]));
  c->set_output(2, c->Vector(output_dims[2]));
  return Status::OK();
}

}  // namespace

// --------------------------------------------------------------------------

REGISTER_OP("AvgPool")
    .Input("value: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("T: {float, half, double}")
    .SetShapeFn(shape_inference::AvgPoolShape)
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
    .Attr("T: {float, half, double}")
    .SetShapeFn([](InferenceContext* c) {
      // NOTE(mrry): We could in principle work out the shape from the
      // gradients and the attrs, but if we do not know orig_input_shape
      // statically, then we are unlikely to know the shape of the
      // gradients either.
      return InputTensorShapeOrUnknown(c, 0 /* input_idx */, 4 /* ndims */);
    })
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
    .Deprecated(9, "Use tf.nn.batch_normalization()")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));

      DimensionHandle last_dim = c->Dim(input, 3);
      for (int i = 1; i < 5; ++i) {  // covers m, v, beta, gamma
        ShapeHandle vec;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &vec));
        TF_RETURN_IF_ERROR(c->Merge(last_dim, c->Dim(vec, 0), &last_dim));
      }

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 3, last_dim, &out));
      c->set_output(0, out);
      return Status::OK();
    })
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
    .Deprecated(9, "Use tf.nn.batch_normalization()")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
      TF_RETURN_IF_ERROR(
          c->Merge(input, c->input(4), &input));  // with backprop

      DimensionHandle last_dim = c->Dim(input, 3);
      for (int i = 1; i < 4; ++i) {  // covers m, v, gamma
        ShapeHandle vec;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &vec));
        TF_RETURN_IF_ERROR(c->Merge(last_dim, c->Dim(vec, 0), &last_dim));
      }

      ShapeHandle dx;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 3, last_dim, &dx));
      c->set_output(0, dx);

      ShapeHandle vector_shape = c->Vector(last_dim);
      c->set_output(1, vector_shape);
      c->set_output(2, vector_shape);
      c->set_output(3, vector_shape);
      c->set_output(4, vector_shape);
      return Status::OK();
    })
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
    .SetShapeFn(shape_inference::BiasAddShape)
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
    .SetShapeFn(shape_inference::BiasAddGradShape)
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
    .SetShapeFn(shape_inference::BiasAddShape)
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
    .Attr("T: {half, float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn(shape_inference::Conv2DShape)
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
    .Attr("T: {half, float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn([](InferenceContext* c) {
      // NOTE(mrry): We could in principle work out the shape from the
      // gradients and the attrs, but if we do not know orig_input_shape
      // statically, then we are unlikely to know the shape of the
      // gradients either.
      return InputTensorShapeOrUnknown(c, 0 /* input_idx */, 4 /* ndims */);
    })
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
    .Attr("T: {half, float, double}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn([](InferenceContext* c) {
      // NOTE(mrry): We could in principle work out the shape from the
      // gradients and the attrs, but if we do not know orig_input_shape
      // statically, then we are unlikely to know the shape of the
      // gradients either.
      return InputTensorShapeOrUnknown(c, 1 /* input_idx */, 4 /* ndims */);
    })
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

REGISTER_OP("FusedResizeAndPadConv2D")
    .Input("input: T")
    .Input("size: int32")
    .Input("paddings: int32")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {half, float, double}")
    .Attr("resize_align_corners: bool = false")
    .Attr(GetMirrorPadModeAttrString())
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Doc(R"doc(
Performs a resize and padding as a preprocess during a convolution.

It's often possible to do spatial transformations more efficiently as part of
the packing stage of a convolution, so this op allows for an optimized
implementation where these stages are fused together. This prevents the need to
write out the intermediate results as whole tensors, reducing memory pressure,
and we can get some latency gains by merging the transformation calculations.
The data_format attribute for Conv2D isn't supported by this op, and defaults to
'NHWC' order.
Internally this op uses a single per-graph scratch buffer, which means that it
will block if multiple versions are being run in parallel. This is because this
operator is primarily an optimization to minimize memory usage.

input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
  new size for the images.
paddings: A two-column matrix specifying the padding sizes. The number of
  rows must be the same as the rank of `input`.
filter: 4-D with shape
  `[filter_height, filter_width, in_channels, out_channels]`.
resize_align_corners: If true, rescale input by (new_height - 1) / (height - 1),
  which exactly aligns the 4 corners of images and resized images. If false, rescale
  by new_height / height. Treat similarly the width dimension.
strides: 1-D of length 4.  The stride of the sliding window for each dimension
   of `input`. Must be in the same order as the dimension specified with format.
padding: The type of padding algorithm to use.
 )doc");

// --------------------------------------------------------------------------

REGISTER_OP("DepthwiseConv2dNative")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .SetShapeFn(shape_inference::DepthwiseConv2DNativeShape)
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
    .SetShapeFn([](InferenceContext* c) {
      // NOTE(mrry): We could in principle work out the shape from the
      // gradients and the attrs, but if we do not know orig_input_shape
      // statically, then we are unlikely to know the shape of the
      // gradients either.
      return InputTensorShapeOrUnknown(c, 0 /* input_idx */, 4 /* ndims */);
    })
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
    .SetShapeFn([](InferenceContext* c) {
      // NOTE(mrry): We could in principle work out the shape from the
      // gradients and the attrs, but if we do not know orig_input_shape
      // statically, then we are unlikely to know the shape of the
      // gradients either.
      return InputTensorShapeOrUnknown(c, 1 /* input_idx */, 4 /* ndims */);
    })
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
REGISTER_OP("Conv3D")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .SetShapeFn(shape_inference::Conv3DShape)
    .Doc(R"doc(
Computes a 3-D convolution given 5-D `input` and `filter` tensors.

In signal processing, cross-correlation is a measure of similarity of
two waveforms as a function of a time-lag applied to one of them. This
is also known as a sliding dot product or sliding inner-product.

Our Conv3D implements a form of cross-correlation.

input: Shape `[batch, in_depth, in_height, in_width, in_channels]`.
filter: Shape `[filter_depth, filter_height, filter_width, in_channels,
  out_channels]`. `in_channels` must match between `input` and `filter`.
strides: 1-D tensor of length 5. The stride of the sliding window for each
  dimension of `input`. Must have `strides[0] = strides[4] = 1`.
padding: The type of padding algorithm to use.

)doc");

REGISTER_OP("Conv3DBackpropInput")
    .Input("input: T")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Deprecated(10, "Use Conv3DBackpropInputV2")
    .SetShapeFn([](InferenceContext* c) {
      return UnchangedShapeWithRank(c, 5);
    })
    .Doc(R"doc(
Computes the gradients of 3-D convolution with respect to the input.

input: Shape `[batch, depth, rows, cols, in_channels]`.
filter: Shape `[depth, rows, cols, in_channels, out_channels]`.
  `in_channels` must match between `input` and `filter`.
out_backprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
  out_channels]`.
strides: 1-D tensor of length 5. The stride of the sliding window for each
  dimension of `input`. Must have `strides[0] = strides[4] = 1`.
padding: The type of padding algorithm to use.

)doc");

REGISTER_OP("Conv3DBackpropFilter")
    .Input("input: T")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Deprecated(10, "Use Conv3DBackpropFilterV2")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 5, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Computes the gradients of 3-D convolution with respect to the filter.

input: Shape `[batch, depth, rows, cols, in_channels]`.
filter: Shape `[depth, rows, cols, in_channels, out_channels]`.
  `in_channels` must match between `input` and `filter`.
out_backprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
  out_channels]`.
strides: 1-D tensor of length 5. The stride of the sliding window for each
  dimension of `input`. Must have `strides[0] = strides[4] = 1`.
padding: The type of padding algorithm to use.

)doc");

REGISTER_OP("Conv3DBackpropInputV2")
    .Input("input_sizes: int32")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 5, &s));
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
Computes the gradients of 3-D convolution with respect to the input.

input_sizes: An integer vector representing the tensor shape of `input`,
   where `input` is a 5-D
   `[batch, depth, rows, cols, in_channels]` tensor.
filter: Shape `[depth, rows, cols, in_channels, out_channels]`.
  `in_channels` must match between `input` and `filter`.
out_backprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
  out_channels]`.
strides: 1-D tensor of length 5. The stride of the sliding window for each
  dimension of `input`. Must have `strides[0] = strides[4] = 1`.
padding: The type of padding algorithm to use.

)doc");

REGISTER_OP("Conv3DBackpropFilterV2")
    .Input("input: T")
    .Input("filter_sizes: int32")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 5, &s));
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
Computes the gradients of 3-D convolution with respect to the filter.

input: Shape `[batch, depth, rows, cols, in_channels]`.
filter_sizes: An integer vector representing the tensor shape of `filter`,
  where `filter` is a 5-D
  `[filter_depth, filter_height, filter_width, in_channels, out_channels]`
  tensor.
out_backprop: Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
  out_channels]`.
strides: 1-D tensor of length 5. The stride of the sliding window for each
  dimension of `input`. Must have `strides[0] = strides[4] = 1`.
padding: The type of padding algorithm to use.

)doc");

// --------------------------------------------------------------------------

REGISTER_OP("AvgPool3D")
    .Input("input: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 5")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr("T: numbertype")
    .SetShapeFn(shape_inference::Pool3DShape)
    .Doc(R"doc(
Performs 3D average pooling on the input.

ksize: 1-D tensor of length 5. The size of the window for each dimension of
  the input tensor. Must have `ksize[0] = ksize[4] = 1`.
strides: 1-D tensor of length 5. The stride of the sliding window for each
  dimension of `input`. Must have `strides[0] = strides[4] = 1`.
padding: The type of padding algorithm to use.
input: Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
output: The average pooled output tensor.
)doc");

REGISTER_OP("AvgPool3DGrad")
    .Input("orig_input_shape: int32")
    .Input("grad: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 5")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr("T: numbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 5, &s));
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
Computes gradients of average pooling function.

ksize: 1-D tensor of length 5. The size of the window for each dimension of
  the input tensor. Must have `ksize[0] = ksize[4] = 1`.
strides: 1-D tensor of length 5. The stride of the sliding window for each
  dimension of `input`. Must have `strides[0] = strides[4] = 1`.
padding: The type of padding algorithm to use.
orig_input_shape: The original input dimensions.
grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.
output: The backprop for input.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("MaxPool3D")
    .Input("input: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 5")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr("T: numbertype")
    .SetShapeFn(shape_inference::Pool3DShape)
    .Doc(R"doc(
Performs 3D max pooling on the input.

ksize: 1-D tensor of length 5. The size of the window for each dimension of
  the input tensor. Must have `ksize[0] = ksize[4] = 1`.
strides: 1-D tensor of length 5. The stride of the sliding window for each
  dimension of `input`. Must have `strides[0] = strides[4] = 1`.
padding: The type of padding algorithm to use.
input: Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
output: The max pooled output tensor.
)doc");

REGISTER_OP("MaxPool3DGrad")
    .Input("orig_input: float")
    .Input("orig_output: float")
    .Input("grad: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 5 ")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr("T: numbertype")
    .SetShapeFn([](InferenceContext* c) {
      return UnchangedShapeWithRank(c, 5);
    })
    .Doc(R"doc(
Computes gradients of max pooling function.

ksize: 1-D tensor of length 5. The size of the window for each dimension of
  the input tensor. Must have `ksize[0] = ksize[4] = 1`.
strides: 1-D tensor of length 5. The stride of the sliding window for each
  dimension of `input`. Must have `strides[0] = strides[4] = 1`.
padding: The type of padding algorithm to use.
orig_input: The original input tensor.
orig_output: The original output tensor.
grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("L2Loss")
    .Input("t: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
L2 Loss.

Computes half the L2 norm of a tensor without the `sqrt`:

    output = sum(t ** 2) / 2

t: Typically 2-D, but may have any dimensions.
output: 0-D.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("LRN")
    .Input("input: T")
    .Output("output: T")
    .Attr("depth_radius: int = 5")
    .Attr("bias: float = 1.0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.5")
    .Attr("T: {float, half} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      return UnchangedShapeWithRank(c, 4);
    })
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
    .Input("input_grads: T")
    .Input("input_image: T")
    .Input("output_image: T")
    .Output("output: T")
    .Attr("depth_radius: int = 5")
    .Attr("bias: float = 1.0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.5")
    .Attr("T: {float, half} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &s));  // input_grads
      TF_RETURN_IF_ERROR(c->Merge(s, c->input(1), &s));     // input_image
      TF_RETURN_IF_ERROR(c->Merge(s, c->input(2), &s));     // output_image
      c->set_output(0, s);
      return Status::OK();
    })
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
    .Attr("T: {float, half} = DT_FLOAT")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::MaxPoolShape)
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
    .Input("orig_input: T")
    .Input("orig_output: T")
    .Input("grad: T")
    .Output("output: T")
    .Attr("T: {float, half} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      return UnchangedShapeWithRank(c, 4);
    })
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
    .Input("input: T")
    .Output("output: T")
    .Output("argmax: Targmax")
    .Attr("T: {float, half} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MaxPoolShape(c));
      c->set_output(1, c->output(0));
      return Status::OK();
    })
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
    .Input("input: T")
    .Input("grad: T")
    .Input("argmax: Targmax")
    .Output("output: T")
    .Attr("T: {float, half} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      return UnchangedShapeWithRank(c, 4);
    })
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

REGISTER_OP("Dilation2D")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("strides: list(int) >= 4")
    .Attr("rates: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
      ShapeHandle filter_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &filter_shape));

      std::vector<int32> strides;
      TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
      if (strides.size() != 4) {
        return errors::InvalidArgument(
            "Dilation2D requires the stride attribute to contain 4 values, but "
            "got: ",
            strides.size());
      }

      std::vector<int32> rates;
      TF_RETURN_IF_ERROR(c->GetAttr("rates", &rates));
      if (rates.size() != 4) {
        return errors::InvalidArgument(
            "Dilation2D requires the rates attribute to contain 4 values, but "
            "got: ",
            rates.size());
      }

      int32 stride_rows = strides[1];
      int32 stride_cols = strides[2];

      int32 rate_rows = rates[1];
      int32 rate_cols = rates[2];

      DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
      DimensionHandle in_rows_dim = c->Dim(input_shape, 1);
      DimensionHandle in_cols_dim = c->Dim(input_shape, 2);
      DimensionHandle filter_rows_dim = c->Dim(filter_shape, 0);
      DimensionHandle filter_cols_dim = c->Dim(filter_shape, 1);
      DimensionHandle output_depth_dim = c->Dim(filter_shape, 2);

      if (!c->ValueKnown(in_rows_dim) || !c->ValueKnown(in_cols_dim) ||
          !c->ValueKnown(filter_rows_dim) || !c->ValueKnown(filter_cols_dim)) {
        ShapeHandle output_shape =
            c->MakeShape({batch_size_dim, InferenceContext::kUnknownDim,
                          InferenceContext::kUnknownDim, output_depth_dim});
        c->set_output(0, output_shape);
        return Status::OK();
      }
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(input_shape, 3), output_depth_dim, &unused));

      auto in_rows = c->Value(in_rows_dim);
      auto in_cols = c->Value(in_cols_dim);
      auto filter_rows = c->Value(filter_rows_dim);
      auto filter_cols = c->Value(filter_cols_dim);
      auto filter_rows_eff = filter_rows + (filter_rows - 1) * (rate_rows - 1);
      auto filter_cols_eff = filter_cols + (filter_cols - 1) * (rate_cols - 1);

      Padding padding;
      TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

      int64 output_rows, output_cols;
      int64 padding_before, padding_after;
      TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
          in_rows, filter_rows_eff, stride_rows, padding, &output_rows,
          &padding_before, &padding_after));
      TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
          in_cols, filter_cols_eff, stride_cols, padding, &output_cols,
          &padding_before, &padding_after));

      ShapeHandle output_shape = c->MakeShape(
          {batch_size_dim, output_rows, output_cols, output_depth_dim});
      c->set_output(0, output_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
`filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
input channel is processed independently of the others with its own structuring
function. The `output` tensor has shape
`[batch, out_height, out_width, depth]`. The spatial dimensions of the output
tensor depend on the `padding` algorithm. We currently only support the default
"NHWC" `data_format`.

In detail, the grayscale morphological 2-D dilation is the max-sum correlation
(for consistency with `conv2d`, we use unmirrored filters):

    output[b, y, x, c] =
       max_{dy, dx} input[b,
                          strides[1] * y + rates[1] * dy,
                          strides[2] * x + rates[2] * dx,
                          c] +
                    filter[dy, dx, c]

Max-pooling is a special case when the filter has size equal to the pooling
kernel size and contains all zeros.

Note on duality: The dilation of `input` by the `filter` is equal to the
negation of the erosion of `-input` by the reflected `filter`.

input: 4-D with shape `[batch, in_height, in_width, depth]`.
filter: 3-D with shape `[filter_height, filter_width, depth]`.
strides: The stride of the sliding window for each dimension of the input
 tensor. Must be: `[1, stride_height, stride_width, 1]`.
rates: The input stride for atrous morphological dilation. Must be:
 `[1, rate_height, rate_width, 1]`.
padding: The type of padding algorithm to use.
output: 4-D with shape `[batch, out_height, out_width, depth]`.
)doc");

REGISTER_OP("Dilation2DBackpropInput")
    .Input("input: T")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("in_backprop: T")
    .Attr("T: realnumbertype")
    .Attr("strides: list(int) >= 4")
    .Attr("rates: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Computes the gradient of morphological 2-D dilation with respect to the input.

input: 4-D with shape `[batch, in_height, in_width, depth]`.
filter: 3-D with shape `[filter_height, filter_width, depth]`.
out_backprop: 4-D with shape `[batch, out_height, out_width, depth]`.
in_backprop: 4-D with shape `[batch, in_height, in_width, depth]`.
strides: 1-D of length 4. The stride of the sliding window for each dimension of
  the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
rates: 1-D of length 4. The input stride for atrous morphological dilation.
  Must be: `[1, rate_height, rate_width, 1]`.
padding: The type of padding algorithm to use.
)doc");

REGISTER_OP("Dilation2DBackpropFilter")
    .Input("input: T")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("filter_backprop: T")
    .Attr("T: realnumbertype")
    .Attr("strides: list(int) >= 4")
    .Attr("rates: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
Computes the gradient of morphological 2-D dilation with respect to the filter.

input: 4-D with shape `[batch, in_height, in_width, depth]`.
filter: 3-D with shape `[filter_height, filter_width, depth]`.
out_backprop: 4-D with shape `[batch, out_height, out_width, depth]`.
filter_backprop: 3-D with shape `[filter_height, filter_width, depth]`.
strides: 1-D of length 4. The stride of the sliding window for each dimension of
  the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
rates: 1-D of length 4. The input stride for atrous morphological dilation.
  Must be: `[1, rate_height, rate_width, 1]`.
padding: The type of padding algorithm to use.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("Relu")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Computes rectified linear: `max(features, 0)`.
)doc");

REGISTER_OP("ReluGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn)
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
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Computes rectified linear 6: `min(max(features, 0), 6)`.
)doc");

REGISTER_OP("Relu6Grad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn)
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
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289)
)doc");

REGISTER_OP("EluGrad")
    .Input("gradients: T")
    .Input("outputs: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn)
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
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Computes softplus: `log(exp(features) + 1)`.
)doc");

REGISTER_OP("SoftplusGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn)
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
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Computes softsign: `features / (abs(features) + 1)`.
)doc");

REGISTER_OP("SoftsignGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn)
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
    .Attr("T: {half, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    })
    .Doc(R"doc(
Computes softmax activations.

For each batch `i` and class `j` we have

    softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))

logits: 2-D with shape `[batch_size, num_classes]`.
softmax: Same shape as `logits`.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("LogSoftmax")
    .Input("logits: T")
    .Output("logsoftmax: T")
    .Attr("T: {half, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    })
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
    .Attr("T: {half, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
      TF_RETURN_IF_ERROR(c->Merge(input, c->input(1), &input));

      DimensionHandle batch_size = c->Dim(input, 0);
      c->set_output(0, c->Vector(batch_size));
      c->set_output(1, input);
      return Status::OK();
    })
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
    .Attr("T: {half, float, double}")
    .Attr("Tlabels: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle features;
      ShapeHandle labels;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &features));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &labels));

      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(features, 0), c->Dim(labels, 0), &batch_size));
      TF_RETURN_IF_ERROR(c->ReplaceDim(features, 0, batch_size, &features));

      c->set_output(0, c->Vector(batch_size));
      c->set_output(1, features);
      return Status::OK();
    })
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
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle predictions;
      ShapeHandle targets;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &predictions));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &targets));
      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(predictions, 0), c->Dim(targets, 0), &batch_size));
      c->set_output(0, c->Vector(batch_size));
      return Status::OK();
    })
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

namespace {

Status TopKShapeFn(InferenceContext* c) {
  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input));

  // Get the k value, either from input tensor or attribute.
  DimensionHandle k_dim;
  if (c->num_inputs() >= 2) {
    TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(1, &k_dim));
  } else {
    int32 k;
    TF_RETURN_IF_ERROR(c->GetAttr("k", &k));
    if (k < 0) {
      return errors::InvalidArgument("Need k >= 0, got ", k);
    }
    k_dim = c->MakeDim(k);
  }

  DimensionHandle last_dim = c->Dim(input, -1);
  if (c->ValueKnown(last_dim) && c->ValueKnown(k_dim) &&
      c->Value(last_dim) < c->Value(k_dim)) {
    return errors::InvalidArgument("input must have last dimension >= k = ",
                                   c->Value(k_dim), " but is ",
                                   c->Value(last_dim));
  }

  // Replace last_dim with k_dim.
  ShapeHandle s;
  TF_RETURN_IF_ERROR(c->Subshape(input, 0, -1, &s));
  TF_RETURN_IF_ERROR(c->Concatenate(s, c->Vector(k_dim), &s));
  c->set_output(0, s);
  c->set_output(1, s);
  return Status::OK();
}

}  // namespace

REGISTER_OP("TopK")
    .Input("input: T")
    .Output("values: T")
    .Output("indices: int32")
    .Attr("k: int >= 0")
    .Attr("sorted: bool = true")
    .Attr("T: realnumbertype")
    .Deprecated(7, "Use TopKV2 instead")
    .SetShapeFn(TopKShapeFn)
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
    .SetShapeFn(TopKShapeFn)
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

// --------------------------------------------------------------------------

REGISTER_OP("FractionalMaxPool")
    .Input("value: T")
    .Output("output: T")
    .Output("row_pooling_sequence: int64")
    .Output("col_pooling_sequence: int64")
    .Attr("pooling_ratio: list(float) >=4")
    .Attr("pseudo_random: bool = false")
    .Attr("overlapping: bool = false")
    .Attr("deterministic: bool = false")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("T: {float, double, int32, int64}")
    .SetShapeFn(FractionalPoolShapeFn)
    .Doc(R"doc(
Performs fractional max pooling on the input.

Fractional max pooling is slightly different than regular max pooling.  In
regular max pooling, you downsize an input set by taking the maximum value of
smaller N x N subsections of the set (often 2x2), and try to reduce the set by
a factor of N, where N is an integer.  Fractional max pooling, as you might
expect from the word "fractional", means that the overall reduction ratio N
does not have to be an integer.

The sizes of the pooling regions are generated randomly but are fairly uniform.
For example, let's look at the height dimension, and the constraints on the
list of rows that will be pool boundaries.

First we define the following:

1.  input_row_length : the number of rows from the input set
2.  output_row_length : which will be smaller than the input
3.  alpha = input_row_length / output_row_length : our reduction ratio
4.  K = floor(alpha)
5.  row_pooling_sequence : this is the result list of pool boundary rows

Then, row_pooling_sequence should satisfy:

1.  a[0] = 0 : the first value of the sequence is 0
2.  a[end] = input_row_length : the last value of the sequence is the size
3.  K <= (a[i+1] - a[i]) <= K+1 : all intervals are K or K+1 size
4.  length(row_pooling_sequence) = output_row_length+1

For more details on fractional max pooling, see this paper:
[Benjamin Graham, Fractional Max-Pooling]
(http://arxiv.org/abs/1412.6071)

value: 4-D with shape `[batch, height, width, channels]`.
pooling_ratio: Pooling ratio for each dimension of `value`, currently only
  supports row and col dimension and should be >= 1.0. For example, a valid
  pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
  must be 1.0 because we don't allow pooling on batch and channels
  dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
  respectively.
pseudo_random: When set to True, generates the pooling sequence in a
  pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
  Graham, Fractional Max-Pooling] (http://arxiv.org/abs/1412.6071) for
  difference between pseudorandom and random.
overlapping: When set to True, it means when pooling, the values at the boundary
  of adjacent pooling cells are used by both cells. For example:

  `index  0  1  2  3  4`

  `value  20 5  16 3  7`

  If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
  The result would be [20, 16] for fractional max pooling.
deterministic: When set to True, a fixed pooling region will be used when
  iterating over a FractionalMaxPool node in the computation graph. Mainly used
  in unit test to make FractionalMaxPool deterministic.
seed: If either seed or seed2 are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: An second seed to avoid seed collision.
output: output tensor after fractional max pooling.
row_pooling_sequence: row pooling sequence, needed to calculate gradient.
col_pooling_sequence: column pooling sequence, needed to calculate gradient.
)doc");

REGISTER_OP("FractionalMaxPoolGrad")
    .Input("orig_input: T")
    .Input("orig_output: T")
    .Input("out_backprop: T")
    .Input("row_pooling_sequence: int64")
    .Input("col_pooling_sequence: int64")
    .Output("output: T")
    .Attr("overlapping: bool = false")
    .Attr("T: {float, double, int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRank(c, 4);
    })
    .Doc(R"doc(
Computes gradient of the FractionalMaxPool function.

orig_input: Original input for `fractional_max_pool`
orig_output: Original output for `fractional_max_pool`
out_backprop: 4-D with shape `[batch, height, width, channels]`.  Gradients
  w.r.t. the output of `fractional_max_pool`.
row_pooling_sequence: row pooling sequence, form pooling region with
  col_pooling_sequence.
col_pooling_sequence: column pooling sequence, form pooling region with
  row_pooling sequence.
overlapping: When set to True, it means when pooling, the values at the boundary
  of adjacent pooling cells are used by both cells. For example:

  `index  0  1  2  3  4`

  `value  20 5  16 3  7`

  If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
  The result would be [20, 16] for fractional max pooling.
output: 4-D.  Gradients w.r.t. the input of `fractional_max_pool`.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("FractionalAvgPool")
    .Input("value: T")
    .Output("output: T")
    .Output("row_pooling_sequence: int64")
    .Output("col_pooling_sequence: int64")
    .Attr("pooling_ratio: list(float) >=4")
    .Attr("pseudo_random: bool = false")
    .Attr("overlapping: bool = false")
    .Attr("deterministic: bool = false")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("T: {float, double, int32, int64}")
    .SetShapeFn(FractionalPoolShapeFn)
    .Doc(R"doc(
Performs fractional average pooling on the input.

Fractional average pooling is similar to Fractional max pooling in the pooling
region generation step. The only difference is that after pooling regions are
generated, a mean operation is performed instead of a max operation in each
pooling region.

value: 4-D with shape `[batch, height, width, channels]`.
pooling_ratio: Pooling ratio for each dimension of `value`, currently only
  supports row and col dimension and should be >= 1.0. For example, a valid
  pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
  must be 1.0 because we don't allow pooling on batch and channels
  dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
  respectively.
pseudo_random: When set to True, generates the pooling sequence in a
  pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
  Graham, Fractional Max-Pooling] (http://arxiv.org/abs/1412.6071) for
  difference between pseudorandom and random.
overlapping: When set to True, it means when pooling, the values at the boundary
  of adjacent pooling cells are used by both cells. For example:

  `index  0  1  2  3  4`

  `value  20 5  16 3  7`

  If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
  The result would be [41/3, 26/3] for fractional avg pooling.
deterministic: When set to True, a fixed pooling region will be used when
  iterating over a FractionalAvgPool node in the computation graph. Mainly used
  in unit test to make FractionalAvgPool deterministic.
seed: If either seed or seed2 are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: An second seed to avoid seed collision.
output: output tensor after fractional avg pooling.
row_pooling_sequence: row pooling sequence, needed to calculate gradient.
col_pooling_sequence: column pooling sequence, needed to calculate gradient.
)doc");

REGISTER_OP("FractionalAvgPoolGrad")
    .Input("orig_input_tensor_shape: int64")
    .Input("out_backprop: T")
    .Input("row_pooling_sequence: int64")
    .Input("col_pooling_sequence: int64")
    .Output("output: T")
    .Attr("overlapping: bool = false")
    .Attr("T: {float, double, int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      if (c->input_tensor(0) != nullptr) {
        ShapeHandle out;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
        c->set_output(0, out);
      } else {
        c->set_output(0, c->UnknownShapeOfRank(4));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Computes gradient of the FractionalAvgPool function.

Unlike FractionalMaxPoolGrad, we don't need to find arg_max for
FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
out_backprop to those indices that form the same pooling cell. Therefore, we
just need to know the shape of original input tensor, instead of the whole
tensor.

orig_input_tensor_shape: Original input tensor shape for `fractional_avg_pool`
out_backprop: 4-D with shape `[batch, height, width, channels]`.  Gradients
  w.r.t. the output of `fractional_avg_pool`.
row_pooling_sequence: row pooling sequence, form pooling region with
  col_pooling_sequence.
col_pooling_sequence: column pooling sequence, form pooling region with
  row_pooling sequence.
overlapping: When set to True, it means when pooling, the values at the boundary
  of adjacent pooling cells are used by both cells. For example:

  `index  0  1  2  3  4`

  `value  20 5  16 3  7`

  If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
  The result would be [41/3, 26/3] for fractional avg pooling.
output: 4-D.  Gradients w.r.t. the input of `fractional_avg_pool`.
)doc");

}  // namespace tensorflow

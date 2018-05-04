/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// This is the common header for the input and filter backprop kernels.
//
// The operation to compute Conv2D gradients.
//
// To compute the gradients for Conv2D, we need three input tensors:
//    input, filter, and backprop for output.
// And we need to compute two backprops: one for input and one for filter. We
// compute them in two different kernels.
//
// Both backprops can be computed as straightforward conv2d.
//
// Consider a case where the input is 3x3 and the filter is 2x1:
//
// INPUT = [ A  B  C ]
//         [ D  E  F ]
//         [ G  H  I ]
//
// where each "A", "B", etc is batch x in_depth
//
// FILTER = [ X  Y ]
//
// where both "X" and "Y" are in_depth x out_depth
//
// With VALID padding, the output is 3x2:
//
// OUTPUT = [ a  b ]
//          [ c  d ]
//          [ e  f ]
//
// where each "a", "b", etc is batch x out_depth
//
// So we have:
//
//   a = A * X + B * Y
//   b = B * X + C * Y
//   c = D * X + E * Y
//   d = E * X + F * Y
//   e = G * X + H * Y
//   f = H * X + I * Y
//
// So when we have backprops for the outputs (we denote them by
// a', b', ... ):
//
// The backprops for the input are:
//
//   A' = a' * X^t
//   B' = a' * Y^t + b' * X^t
//   C' = b' * Y^t
//   ...
//
// This is essentially computing a 2d conv of
//
// INPUT = [ 0  a'  b'  0 ]
//         [ 0  c'  d'  0 ]
//         [ 0  e'  f'  0 ]
// and
//
// FILTER = [ Y^t X^t ]
//
// The backprops for the filter are:
//
//   X' = A^t * a' + B^t * b' + D^t * c' + E^t * d' + G^t * e' + H^t * f'
//   Y' = B^t * a' + C^t * b' + E^t + c' + F^t * d' + H^t * e' + I^t * f'
//
// This is essentially computing a 2d conv of
//
// INPUT = [ A^t  B^t  C^t ]
//         [ D^t  E^t  F^t ]
//         [ G^t  H^t  I^t ]
//
// and
//
// FILTER = [ a'  b' ]
//          [ c'  d' ]
//          [ e'  f' ]
//
//
//////////////////////////////////////////////////////////
//
// With stride more than one, it's a bit more complicated (we will need to
// create holes to the backprop).
//
// Consider the case where
//
// INPUT = [ A B C D E ]
//         [ F G H I J ]
//         [ K L M N O ]
// and
//
// FILTER = [ X Y Z ]
//
// with stride 2.
//
// The output will be
//
// OUTPUT = [ a b ]
//          [ c d ]
//
// where:
//
//   a = A * X + B * Y + C * Z
//   b = C * X + D * Y + E * Z
//   c = K * X + L * Y + M * Z
//   d = M * X + N * Y + O * Z
//
//
// To compute the backprop for INPUT, we need to convolve
//
// INPUT = [ 0  0  a' 0  b' 0  0 ]
//         [ 0  0  0  0  0  0  0 ]
//         [ 0  0  c' 0  d' 0  0 ]
//
// (notice the holes in INPUT)
//
// and
//
// FILTER = [ Z^t  Y^t  X^t ]
//
// with stride 1.
//
// To compute the backprop for FILTER, we need to convolve

//
// INPUT = [ A^t  B^t  C^t  D^t  E^t ]
//         [ F^t  G^t  H^t  I^t  J^t ]
//         [ K^t  L^t  M^t  N^t  O^t ]
// and
//
// FILTER = [ a' 0  b' ]
//          [ 0  0  0  ]
//          [ c' 0  d' ]
//
// (notice the holes in FILTER)
//
//
// with stride 1
//
//////////////////////////////////////////////////////////
//
//
// The case for SAME padding is in fact very similar to VALID -- we just
// need to pad the input tensor a bit when computing the filter_backprop.

#ifndef TENSORFLOW_CORE_KERNELS_CONV_GRAD_OPS_H_
#define TENSORFLOW_CORE_KERNELS_CONV_GRAD_OPS_H_

#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

// Forward declaration.
class OpKernelContext;

template <typename Device, typename T>
struct LaunchConv2DBackpropInputOp {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& out_backprop, const Tensor& filter,
                  int row_dilation, int col_dilation, int row_stride,
                  int col_stride, const Padding& padding, Tensor* in_backprop,
                  TensorFormat data_format);
};

template <typename Device, typename T>
struct LaunchConv2DBackpropFilterOp {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& out_backprop, const Tensor& input,
                  int row_dilation, int col_dilation, int row_stride,
                  int col_stride, const Padding& padding,
                  Tensor* filter_backprop, TensorFormat data_format);
};

#ifdef GOOGLE_CUDA
template <typename T>
struct LaunchConv2DBackpropInputOp<Eigen::GpuDevice, T> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter, int row_dilation,
                  int col_dilation, int row_stride, int col_stride,
                  const Padding& padding, Tensor* output,
                  TensorFormat data_format);
};

template <typename T>
struct LaunchConv2DBackpropFilterOp<Eigen::GpuDevice, T> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& out_backprop, const Tensor& input,
                  int row_dilation, int col_dilation, int row_stride,
                  int col_stride, const Padding& padding,
                  Tensor* filter_backprop, TensorFormat data_format);
};
#endif  // GOOGLE_CUDA

// Information about a single spatial dimension for a convolution
// backpropagation.
struct ConvBackpropSpatialDimension {
  int64 input_size;
  int64 filter_size;
  int64 output_size;
  int64 stride;
  int64 dilation;
  int64 expanded_output_size;

  // Number of padding elements to be added before/after this dimension of
  // the input when computing Conv?DBackpropInput.
  int64 pad_before, pad_after;
};

// Computed dimensions for a backwards convolution.
struct ConvBackpropDimensions {
  // Information about each spatial dimension.
  gtl::InlinedVector<ConvBackpropSpatialDimension, 3> spatial_dims;

  // Batch size.
  int64 batch_size;

  // Input and output feature depth.
  int64 in_depth, out_depth;
};

// Common code between implementations of Conv?DBackpropInput and
// Conv?DBackpropFilter. Verifies that the dimensions all match, and computes
// sizes/padding for the spatial dimensions.
Status ConvBackpropComputeDimensions(StringPiece label, int num_spatial_dims,
                                     const TensorShape& input_shape,
                                     const TensorShape& filter_shape,
                                     const TensorShape& out_backprop_shape,
                                     const std::vector<int32>& strides,
                                     Padding padding, TensorFormat data_format,
                                     ConvBackpropDimensions* dims);

// The V2 version computes the same outputs with arbitrary dilation rate.
// TODO(b/67112639): Merge V2 versions and the original versions eventually.
Status ConvBackpropComputeDimensionsV2(
    StringPiece label, int num_spatial_dims, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& out_backprop_shape,
    const gtl::ArraySlice<int32>& dilations, const std::vector<int32>& strides,
    Padding padding, TensorFormat data_format, ConvBackpropDimensions* dims);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONV_GRAD_OPS_H_

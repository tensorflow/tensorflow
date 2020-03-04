/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_CONV_GRAD_SHAPE_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_CONV_GRAD_SHAPE_UTILS_H_

#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
// Information about a single spatial dimension for a convolution
// backpropagation.
struct ConvBackpropSpatialDimension {
  int64 input_size;
  int64 filter_size;
  int64 output_size;
  int64 stride;
  int64 dilation;

  // Output size after scaling by the stride.
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

  // Convenience access methods for spatial dimensions properties.
  int64 input_size(int dim) const { return spatial_dims[dim].input_size; }
  int64 filter_size(int dim) const { return spatial_dims[dim].filter_size; }
  int64 output_size(int dim) const { return spatial_dims[dim].output_size; }
  int64 stride(int dim) const { return spatial_dims[dim].stride; }
  int64 dilation(int dim) const { return spatial_dims[dim].dilation; }

  // Compute padding for the given spatial dimension.
  int SpatialPadding(const Padding& padding, int dim) const;
};

// Common code between implementations of Conv?DBackpropInput and
// Conv?DBackpropFilter. Verifies that the dimensions all match, and computes
// sizes/padding for the spatial dimensions. Does not support explicit padding.
Status ConvBackpropComputeDimensions(StringPiece label, int num_spatial_dims,
                                     const TensorShape& input_shape,
                                     const TensorShape& filter_shape,
                                     const TensorShape& out_backprop_shape,
                                     const std::vector<int32>& strides,
                                     Padding padding, TensorFormat data_format,
                                     ConvBackpropDimensions* dims);

// The V2 version computes the same outputs with arbitrary dilation rate and
// supports explicit padding.
// TODO(b/67112639): Merge V2 versions and the original versions eventually.
Status ConvBackpropComputeDimensionsV2(
    StringPiece label, int num_spatial_dims, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& out_backprop_shape,
    const gtl::ArraySlice<int32>& dilations, const std::vector<int32>& strides,
    Padding padding, absl::Span<const int64> explicit_paddings,
    TensorFormat data_format, ConvBackpropDimensions* dims);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONV_GRAD_SHAPE_UTILS_H_

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

#ifndef TENSORFLOW_CORE_KERNELS_CONV_GRAD_OPS_H_
#define TENSORFLOW_CORE_KERNELS_CONV_GRAD_OPS_H_

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
  int64 expanded_output_size;

  // Number of padding elements to be added before/after this dimension of
  // the input when computing Conv2DBackpropInput.
  int64 pad_before, pad_after;
};

// Computed dimensions for a Conv2D backpropagation.
struct Conv2DBackpropDimensions {
  // Information about each spatial dimension.
  ConvBackpropSpatialDimension rows, cols;

  // Batch size.
  int64 batch_size;

  // Input and output feature depth.
  int64 in_depth, out_depth;
};

// Common code between implementations of Conv2DBackpropInput and
// Conv2DBackpropFilter. Verifies that the dimensions all match, and computes
// sizes/padding for rows and columns.
Status Conv2DBackpropComputeDimensions(
    StringPiece label, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& out_backprop_shape,
    const std::vector<int32>& strides, Padding padding,
    TensorFormat data_format, Conv2DBackpropDimensions* dims);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONV_GRAD_OPS_H_

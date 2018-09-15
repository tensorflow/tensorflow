/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_CONV_GRAD_SIZE_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_CONV_GRAD_SIZE_UTIL_H_

#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Information about a single spatial dimension for a convolution gradients and
// windowed operations.
struct SpatialDimensionOutputSizeAndPadding {
  // Effective size of the operation output (potentially expanded).
  int64 output_size;
  // Number of padding elements to be added before/after this dimension of
  // the input when computing the input gradient.
  int64 pad_before;
  int64 pad_after;
};

// Verifies that the dimensions all match, and computes the size and padding of
// a spatial dimension for convolution gradient operations.
StatusOr<SpatialDimensionOutputSizeAndPadding>
ConvGradExtractAndVerifyDimension(int64 input_size, int64 filter_size,
                                  int64 output_size, int64 dilation,
                                  int64 stride, Padding padding);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_CONV_GRAD_SIZE_UTIL_H_

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_SUPPORT_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_SUPPORT_UTILS_H_

#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/tsl/platform/status.h"

namespace xla {
namespace gpu {

// Indicates whether the `compute_capability` supports an optimized integer
// implementation of the given `conv` operation vectorized to `vector_size`.
//
// This function does not guarantee that a convolution will be padded and/or
// vectorized. It only checks that it is a valid candiate for such optimization.
StatusOr<bool> CudnnSupportsOptimizedIntegerConvolution(
    const se::CudaComputeCapability& compute_capability,
    HloCustomCallInstruction& conv, int vector_size);

// Represents configuration for the reshape-transpose-reshape operations that
// are equivalent to `cudnnReorderFilterAndBias`. This is used by int8x32
// vectorized convolutions.
//
// For filter reordering the equivalent HLO is:
//   %reshape = s8[$S] reshape(%input)
//   %transpose = s8[I/32,H,W,O/8,2,8,4,4] transpose(%reshape), dimensions={$D}
//   %result = s8[O,I/32,H,W,32] reshape(%transpose)
//
// For bias reordering the HLO is similar, but the op shapes are s8[O/32,4,2,4]
// for %transpose, and s8[O/32,2,4,4] for %result.
//
// The helper functions below calculate the shape $S (transpose_shape) and
// dimensions $D (permutation) from the convolution dimensions numbers config.
// The result_shape is fixed and is present for the convenience.
struct CudnnReorderTransposeConfig {
  Shape transpose_shape;
  Shape result_shape;
  std::vector<int64_t> permutation;
};

// Create a transposition for an int8x32 convolution filter that effectively
// does the same thing as cudnnReorderFilterAndBias, but could also be constant
// folded or fused.
StatusOr<CudnnReorderTransposeConfig> CudnnInferTransposeForFilterReordering(
    const Shape& shape, const ConvolutionDimensionNumbers& dimension_numbers);

// Create a transposition for an int8x32 convolution bias that effectively
// does the same thing as cudnnReorderFilterAndBias, but could also be constant
// folded or fused.
StatusOr<CudnnReorderTransposeConfig> CudnnInferTransposeForBiasReordering(
    const Shape& shape);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_SUPPORT_UTILS_H_

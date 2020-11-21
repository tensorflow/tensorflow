/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DOT_AS_CONVOLUTION_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DOT_AS_CONVOLUTION_UTIL_H_

#include <memory>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace dot_as_convolution_util {

// Describes the dimensions of a convolution that can be interpreted as a dot
// or a normal convolution.
struct DotConvolutionDimsInfo {
  // The dimension numbers for the operands and output corresponding to a
  // logical dimension (e.g., batch, contracting, non-contracting). If an
  // operand or the output doesn't have the logical dimension, it is set to
  // -1.
  struct DimNums {
    int64 lhs;
    int64 rhs;
    int64 output;
    // The corresponding spatial dimension in the convolution's config. Set to
    // -1 if it's not mapped to a spatial dimension.
    int64 spatial_dim;
  };
  std::vector<DimNums> batch_dims;
  std::vector<DimNums> contracting_dims;
  std::vector<DimNums> lhs_non_contracting_dims;
  std::vector<DimNums> rhs_non_contracting_dims;
  std::vector<DimNums> conv_spatial_dims;
};

// Parses a convolution and returns a DotGeneralAsConvolutionDimsInfo. If it can
// be interpreted as a dot, there is no conv_spatial_dims.
DotConvolutionDimsInfo ParseConvolutionDimsInfo(const HloInstruction* conv);

// Creates sharded convolution instruction that can be interpreted as a dot.
// This is a utility for per-op partitioners.
//  - 'conv' is the original convolution instruction.
//  - 'dot_dnums' is the result of ParseDotConvolutionDimsInfo() for 'conv'.
//  - 'sharded_lhs_hlo' and 'sharded_rhs_hlo' are sharded inputs for the result
//    convolution instruction.
StatusOr<std::unique_ptr<HloInstruction>>
CreateShardedConvForDotGeneralConvolution(
    const HloInstruction& conv, const DotConvolutionDimsInfo& dot_dnums,
    HloInstruction* sharded_lhs_hlo, HloInstruction* sharded_rhs_hlo);

// Check if a spatial dim is parallel batch dimension.
// A parallel batch dimension in DotGeneral is represented as a spatial
// dimension with window size B (batch dimension size), stride B - 1, and base
// dilation B.
bool ConvSpatialDimensionIsParallel(const WindowDimension& wd, int64 lhs_size);

// Returns a DotConvolutionDimsInfo from a kDot instruction, where all
// the spatial_dim values are set to -1.
DotConvolutionDimsInfo ParseDotGeneralFromDot(const HloInstruction* dot);

}  // namespace dot_as_convolution_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DOT_AS_CONVOLUTION_UTIL_H_

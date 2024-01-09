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

#ifndef XLA_SERVICE_DOT_AS_CONVOLUTION_UTIL_H_
#define XLA_SERVICE_DOT_AS_CONVOLUTION_UTIL_H_

#include <memory>
#include <optional>
#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace dot_as_convolution_util {

// Type of Batch representation for a convolution that has a spatial dimension
// that is effectively a batch dimension. We currently have two
// representations that we detect as "batch equivalent" and this enum allows
// differentiating between the two.
enum class SpatialBatchRepresentation {
  kNone,
  kUnpaddedVersion,
  kPaddedVersion,
};

// Describes the dimensions of a convolution that can be interpreted as a dot
// or a normal convolution.
struct DotConvolutionDimsInfo {
  // The dimension numbers for the operands and output corresponding to a
  // logical dimension (e.g., batch, contracting, non-contracting). If an
  // operand or the output doesn't have the logical dimension, it is set to
  // -1.
  struct DimNums {
    int64_t lhs;
    int64_t rhs;
    int64_t output;
    // The corresponding spatial dimension in the convolution's config. Set to
    // -1 if it's not mapped to a spatial dimension.
    int64_t spatial_dim;
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
// dilation B or an alternative representation of window size B, stride B,
// padding low/high B - 1, base dilation B - 1 and window reversal
SpatialBatchRepresentation SpatialIsBatch(int64_t lhs_spatial_size,
                                          const WindowDimension& spatial_wd);
// Returns if the spatial dimension represented by 'spatial_wd' is an LHS non
// contracting dimension.
bool SpatialIsLhsNonContracting(int64_t rhs_spatial_size,
                                const WindowDimension& spatial_wd);
// Returns if the spatial dimension represented by 'spatial_wd' is an RHS non
// contracting dimension.
bool SpatialIsRhsNonContracting(int64_t lhs_spatial_size,
                                int64_t rhs_spatial_size,
                                const WindowDimension& spatial_wd);
// Returns if the spatial dimension represented by 'spatial_wd' endsup being
// equivalent to a contracting dimension.
bool SpatialIsContracting(int64_t lhs_spatial_size, int64_t rhs_spatial_size,
                          const WindowDimension& spatial_wd);
// Returns a DotConvolutionDimsInfo from a kDot instruction, where all
// the spatial_dim values are set to -1.
DotConvolutionDimsInfo ParseDotGeneralFromDot(const HloInstruction* dot);

}  // namespace dot_as_convolution_util
}  // namespace xla

#endif  // XLA_SERVICE_DOT_AS_CONVOLUTION_UTIL_H_

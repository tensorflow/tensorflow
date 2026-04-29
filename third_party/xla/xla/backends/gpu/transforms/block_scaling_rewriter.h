/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_BLOCK_SCALING_REWRITER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_BLOCK_SCALING_REWRITER_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"
#include "xla/stream_executor/dnn.h"

namespace xla::gpu {

const se::dnn::VersionInfo kCudnnSupportsBlockScaledDot(9, 10);
const se::dnn::VersionInfo kCudnnSupportsBlockScaledDotWithGlobalScale(9, 13);

// This pass converts the block quantize/dequantize operations (represented as
// custom calls) to XLA graphs or library calls, if available (e.g. cuDNN).
//
// Supported operations:
//
// 1. "__op$quantize": takes an input tensor of arbitrary size, splits it into
//    blocks along the minor dimension; for each block calculates the scaling
//    factor and adjusts the output data, so when these are multiplied together,
//    the result is roughly equal to the input (minus the rounding error).
//
//    Example: f32[8,128] -> (f8e4m3fn[8,128], f8e4m3fn[8,4])
//
// 2. "__op$dequantize": takes two tensors as the input - quantized data and
//    scaling factor (block size is implied). Multiplies them and returns the
//    result as a wider data type. This is the inverse of the quantize op.
//
//    Example: (f8e4m3fn[8,128], f8e4m3fn[8,4]) -> f32[8,128]
//
// 3. "__op$block_scaled_dot": performs the dot operation on quantized inputs.
//    The number of parameters is either 3 (if only LHS input is quantized) or 4
//    (if both inputs are quantized).
//    The contracting dimension must be the minor one for both LHS and RHS.
//    The batch dimension (if present) must be the major one.
//
//    Example: (f8e4m3fn[8,128], f8e4m3fn[16,128],
//              f8e8m0fnu[8,4], f8e8m0fnu[16,4]) -> f32[8,16]
//
//    The following HLO:
//        %res = f32[4,8,16] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
//               custom_call_target="__op$block_scaled_dot"
//    is equivalent to:
//        %lhs_dq = f32[4,8,128] custom-call(%lhs, %lhs_scale),
//                  custom_call_target="__op$dequantize"
//        %rhs_dq = f32[4,16,128] custom-call(%rhs, %rhs_scale),
//                  custom_call_target="__op$dequantize"
//        %res = f32[4,8,16] dot(%lhs_dq, %rhs_dq),
//               lhs_batch_dims={0}, lhs_contracting_dims={2},
//               rhs_batch_dims={0}, rhs_contracting_dims={2}
//
//    Note: the scale tensor may be padded; with cuDNN lowering, the underlying
//    kernel will handle this correctly, with default lowering the extra values
//    will be ignored. An explicit block size must be passed in the backend
//    config if the block scaled dimension is padded.
class BlockScalingRewriter : public OpExpanderPass {
 public:
  explicit BlockScalingRewriter(se::dnn::VersionInfo cudnn_version)
      : cudnn_version_(cudnn_version) {};

  absl::string_view name() const override { return "block-scaling-rewriter"; }

  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;

  // Custom call targets.
  static constexpr absl::string_view kQuantizeCustomCallTarget =
      "__op$quantize";
  static constexpr absl::string_view kDequantizeCustomCallTarget =
      "__op$dequantize";
  static constexpr absl::string_view kBlockScaledDotCustomCallTarget =
      "__op$block_scaled_dot";

  // Common block size constants.
  static constexpr int kBlockSizeMXFP8 = 32;
  static constexpr int kBlockSizeNVFP4 = 16;

 private:
  se::dnn::VersionInfo cudnn_version_;
};

// Helper class for building cuDNN scaled dot operations.
class CudnnScaledDotHelper {
 public:
  // Check if the scaled dot fusion is supported by cuDNN.
  static bool IsSupported(const HloScaledDotInstruction* scaled_dot);

  // Extract scale tensor swizzling from the block scaled dot fusion into
  // separate computations.
  static absl::StatusOr<HloInstruction*> AddScaleSwizzle(
      HloFusionInstruction* fusion);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_BLOCK_SCALING_REWRITER_H_

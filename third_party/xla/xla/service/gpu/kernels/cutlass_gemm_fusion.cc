/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/kernels/custom_fusion.h"
#include "xla/service/gpu/kernels/custom_fusion_pattern.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/cutlass_gemm_kernel.h"
#include "xla/shape.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// Cutlass Gemm pattern matching helpers
//===----------------------------------------------------------------------===//

static Status IsF32Gemm(const HloDotInstruction* dot) {
  const Shape& lhs = dot->operand(0)->shape();
  const Shape& rhs = dot->operand(1)->shape();
  const Shape& out = dot->shape();

  if (lhs.dimensions_size() != 2 || rhs.dimensions_size() != 2)
    return absl::InternalError("dot operands must have rank 2");

  if (lhs.element_type() != PrimitiveType::F32 ||
      rhs.element_type() != PrimitiveType::F32 ||
      out.element_type() != PrimitiveType::F32)
    return absl::InternalError("dot operations must use F32 data type");

  // Check that we do not transpose any of the operands.
  auto& dot_dims = dot->dot_dimension_numbers();

  if (dot_dims.lhs_contracting_dimensions().size() != 1 ||
      dot_dims.lhs_contracting_dimensions()[0] != 1)
    return absl::InternalError("lhs contracting dimensions must be 1");

  if (dot_dims.rhs_contracting_dimensions().size() != 1 ||
      dot_dims.rhs_contracting_dimensions()[0] != 0)
    return absl::InternalError("rhs contracting dimensions must be 0");

  return OkStatus();
}

//===----------------------------------------------------------------------===//
// CutlassGemmPattern
//===----------------------------------------------------------------------===//

class CutlassGemmPattern : public CustomFusionPattern {
 public:
  std::optional<Match> TryMatch(HloInstruction* instr) const override {
    auto* dot = DynCast<HloDotInstruction>(instr);
    if (!dot || !IsF32Gemm(dot).ok()) return std::nullopt;

    CustomFusionConfig config;
    config.set_name("cutlass_gemm");
    return Match{config, {instr}};
  }
};

//===----------------------------------------------------------------------===//
// CutlassGemmFusion
//===----------------------------------------------------------------------===//

class CutlassGemmFusion : public CustomFusion {
 public:
  StatusOr<std::vector<CustomKernel>> LoadKernels(
      const HloComputation* computation) const final {
    auto* dot = DynCast<HloDotInstruction>(computation->root_instruction());
    if (dot == nullptr)
      return absl::InternalError(
          "cutlass_gemm requires ROOT operation to be a dot");

    TF_RETURN_IF_ERROR(IsF32Gemm(dot));

    auto dtype = dot->shape().element_type();

    auto& lhs_shape = dot->operand(0)->shape();
    auto& rhs_shape = dot->operand(1)->shape();

    size_t m = lhs_shape.dimensions(0);
    size_t k = lhs_shape.dimensions(1);
    size_t n = rhs_shape.dimensions(1);

    TF_ASSIGN_OR_RETURN(auto kernel,
                        kernel::GetCutlassGemmKernel(dtype, m, n, k));
    return std::vector<CustomKernel>{std::move(kernel)};
  }
};

}  // namespace xla::gpu

XLA_REGISTER_CUSTOM_FUSION_PATTERN(::xla::gpu::CutlassGemmPattern);
XLA_REGISTER_CUSTOM_FUSION("cutlass_gemm", ::xla::gpu::CutlassGemmFusion);

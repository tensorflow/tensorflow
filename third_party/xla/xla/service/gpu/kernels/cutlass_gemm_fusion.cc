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

#include "xla/service/gpu/kernels/cutlass_gemm_fusion.h"

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
#include "xla/service/pattern_matcher.h"
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

namespace {
namespace m = match;

// Pattern for matching mixed precision GEMMs.
struct GemmWithUpcast {
  explicit GemmWithUpcast(HloDotInstruction* dot) : dot(dot) {}

  HloInstruction* dot;
  HloInstruction* lhs_upcast = nullptr;  // HLO convert instr
  HloInstruction* rhs_upcast = nullptr;  // HLO convert instr
};
}  // namespace

// Returns OK if dot instruction is a simple 2D row-major gemm.
static Status MatchRowMajorGemm(HloDotInstruction* dot) {
  if (dot->operand(0)->shape().dimensions_size() != 2 ||
      dot->operand(1)->shape().dimensions_size() != 2) {
    return absl::InternalError("operands must have rank 2");
  }

  auto& dot_dims = dot->dot_dimension_numbers();

  if (dot_dims.lhs_contracting_dimensions().size() != 1 ||
      dot_dims.lhs_contracting_dimensions()[0] != 1) {
    return absl::InternalError("lhs contracting dimensions must be 1");
  }

  if (dot_dims.rhs_contracting_dimensions().size() != 1 ||
      dot_dims.rhs_contracting_dimensions()[0] != 0) {
    return absl::InternalError("rhs contracting dimensions must be 0");
  }

  return OkStatus();
}

// Return OK if dot instruction is a simple gemm with all operands and result
// having the same data type.
static Status MatchSimpleGemm(HloDotInstruction* dot, PrimitiveType dtype) {
  TF_RETURN_IF_ERROR(MatchRowMajorGemm(dot));

  if (dot->operand(0)->shape().element_type() != dtype ||
      dot->operand(1)->shape().element_type() != dtype ||
      dot->shape().element_type() != dtype) {
    return absl::InternalError("operands and result must have the same type");
  }

  return OkStatus();
}

// Returns matched GEMM with one of the operands upcasted to the accumulator
// data type with an HLO convert instruction.
static StatusOr<GemmWithUpcast> MatchGemmWithUpcast(HloDotInstruction* dot) {
  TF_RETURN_IF_ERROR(MatchRowMajorGemm(dot));

  GemmWithUpcast matched(dot);

  // C <- convert(A) * B
  if (Match(const_cast<HloInstruction*>(dot->operand(0)),
            m::Convert(&matched.lhs_upcast, m::Op()))) {
    return matched;
  }

  // C <- A * convert(B)
  if (Match(const_cast<HloInstruction*>(dot->operand(1)),
            m::Convert(&matched.rhs_upcast, m::Op()))) {
    return matched;
  }

  return absl::InternalError("unsupported gemm with upcasing");
}

//===----------------------------------------------------------------------===//
// Cutlass Gemm Patterns
//===----------------------------------------------------------------------===//

std::optional<CustomFusionPattern::Match> CutlassGemmPattern::TryMatch(
    HloInstruction* instr) const {
  auto* dot = DynCast<HloDotInstruction>(instr);
  if (!dot) return std::nullopt;

  auto matched = MatchSimpleGemm(dot, PrimitiveType::F32);
  if (!matched.ok()) return std::nullopt;

  CustomFusionConfig config;
  config.set_name("cutlass_gemm");
  return Match{config, {instr}};
}

std::optional<CustomFusionPattern::Match>
CutlassGemmWithUpcastPattern::TryMatch(HloInstruction* instr) const {
  auto* dot = DynCast<HloDotInstruction>(instr);
  if (!dot) return std::nullopt;

  auto matched = MatchGemmWithUpcast(dot);
  if (!matched.ok()) return std::nullopt;

  // Only one operand can be upcasted.
  DCHECK(matched->lhs_upcast == nullptr || matched->rhs_upcast == nullptr);

  CustomFusionConfig config;
  config.set_name("cutlass_gemm_with_upcast");

  return matched->lhs_upcast ? Match{config, {matched->lhs_upcast, instr}}
                             : Match{config, {matched->rhs_upcast, instr}};
}

//===----------------------------------------------------------------------===//
// Cutlass Gemm Fusions
//===----------------------------------------------------------------------===//

class CutlassGemmFusion : public CustomFusion {
 public:
  StatusOr<std::vector<CustomKernel>> LoadKernels(
      const HloComputation* computation) const final {
    auto* dot = DynCast<HloDotInstruction>(computation->root_instruction());
    if (dot == nullptr) {
      return absl::InternalError(
          "cutlass_gemm requires ROOT operation to be a dot");
    }

    TF_RETURN_IF_ERROR(MatchSimpleGemm(dot, PrimitiveType::F32));

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

class CutlassGemmWithUpcastFusion : public CustomFusion {
 public:
  StatusOr<std::vector<CustomKernel>> LoadKernels(
      const HloComputation* computation) const final {
    auto* dot = DynCast<HloDotInstruction>(computation->root_instruction());
    if (dot == nullptr) {
      return absl::InternalError(
          "cutlass_gemm requires ROOT operation to be a dot");
    }

    TF_ASSIGN_OR_RETURN(auto matched, MatchGemmWithUpcast(dot));

    // We only support upcasting of rhs operand.
    if (matched.lhs_upcast != nullptr)
      return absl::InternalError("only rhs upcasting is implemented");

    auto dot_dtype = dot->shape().element_type();
    auto upcast_dtype = matched.rhs_upcast->shape().element_type();

    // We only support BF16 <- BF16 x S8 upcasted gemm.
    if (dot_dtype != PrimitiveType::BF16 || upcast_dtype != PrimitiveType::S8)
      return absl::InternalError("unsupported upcasting pattern");

    return absl::UnimplementedError("requires CUTLASS 3.3.0");
  }
};

}  // namespace xla::gpu

XLA_REGISTER_CUSTOM_FUSION_PATTERN(::xla::gpu::CutlassGemmPattern);

XLA_REGISTER_CUSTOM_FUSION("cutlass_gemm", ::xla::gpu::CutlassGemmFusion);
XLA_REGISTER_CUSTOM_FUSION("cutlass_gemm_with_upcast",
                           ::xla::gpu::CutlassGemmWithUpcastFusion);

/* Copyright 2023 The OpenXLA Authors.

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
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion_pattern.h"
#include "xla/service/gpu/kernels/cutlass_gemm.h"
#include "xla/service/gpu/kernels/cutlass_gemm_custom_kernel.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// Cutlass Gemm pattern matching helpers
//===----------------------------------------------------------------------===//

namespace {
namespace m = match;

// If custom fusion requires extra workspace at run time, ROOT instruction will
// be a tuple with second operand being a result of workspace allocation custom
// call.
struct RootWithWorkspace {
  HloInstruction* root;
  HloInstruction* workspace;
};

static RootWithWorkspace MatchRootWithWorkspace(HloInstruction* root) {
  RootWithWorkspace result;
  if (Match(root,
            m::Tuple(m::Op(&result.root),
                     m::CustomCall(&result.workspace,
                                   {CustomKernelFusionPattern::kWorkspace})))) {
    return result;
  }
  return {root, nullptr};
}

// Pattern for matching mixed precision GEMMs.
struct GemmWithUpcast {
  explicit GemmWithUpcast(HloDotInstruction* dot) : dot(dot) {}

  HloInstruction* dot;
  HloInstruction* lhs_upcast = nullptr;  // HLO convert instr
  HloInstruction* rhs_upcast = nullptr;  // HLO convert instr
};

// Pattern for matching GEMM with surrounding dynamic-slice/update-slice.
struct GemmWithDynamicSlice {
  explicit GemmWithDynamicSlice(HloDynamicUpdateSliceInstruction* update_slice)
      : update_slice(update_slice) {}

  std::vector<HloInstruction*> Instrs() {
    // Bitcast could be optional
    if (bitcast == nullptr) {
      return {dot, update_slice};
    }
    return {dot, bitcast, update_slice};
  }

  HloInstruction* dot = nullptr;
  HloInstruction* bitcast = nullptr;       // result bitcast
  HloInstruction* update_slice = nullptr;  // update result slice
};
}  // namespace

// Returns OK if dot instruction is a simple 2D row-major gemm.
static absl::Status MatchRowMajorGemm(HloDotInstruction* dot) {
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

  return absl::OkStatus();
}

// Return OK if dot instruction is a simple gemm with all operands and result
// having the same data type.
static absl::Status MatchSimpleGemm(
    HloDotInstruction* dot, absl::Span<const PrimitiveType> support_dtypes) {
  TF_RETURN_IF_ERROR(MatchRowMajorGemm(dot));

  for (PrimitiveType dtype : support_dtypes) {
    if (dot->operand(0)->shape().element_type() == dtype &&
        dot->operand(1)->shape().element_type() == dtype &&
        dot->shape().element_type() == dtype) {
      return absl::OkStatus();
    }
  }

  return absl::InternalError("unsupported operands type");
}

// Returns matched GEMM with one of the operands upcasted to the accumulator
// data type with an HLO convert instruction.
static absl::StatusOr<GemmWithUpcast> MatchGemmWithUpcast(
    HloDotInstruction* dot) {
  TF_RETURN_IF_ERROR(MatchRowMajorGemm(dot));

  GemmWithUpcast match(dot);

  // C <- convert(A) * B
  if (Match(const_cast<HloInstruction*>(dot->operand(0)),
            m::Convert(&match.lhs_upcast, m::Op()))) {
    return match;
  }

  // C <- A * convert(B)
  if (Match(const_cast<HloInstruction*>(dot->operand(1)),
            m::Convert(&match.rhs_upcast, m::Op()))) {
    return match;
  }

  return absl::InternalError("unsupported gemm with upcasing");
}

template <typename Pattern>
auto OptionalBitcast(HloInstruction** optional_bitcast, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Bitcast(optional_bitcast, pattern),
                                  std::move(pattern));
}

// Returns matched GEMM with result used to update a slice.
static absl::StatusOr<GemmWithDynamicSlice> MatchGemmWithDynamicUpdateSlice(
    HloDynamicUpdateSliceInstruction* update_slice) {
  GemmWithDynamicSlice match(update_slice);

  if (!Match(const_cast<HloInstruction*>(update_slice->update()),
             OptionalBitcast(&match.bitcast,
                             m::Dot(&match.dot, m::Op(), m::Op())))) {
    return absl::InternalError("failed to match update slice instr");
  }

  TF_RETURN_IF_ERROR(MatchRowMajorGemm(Cast<HloDotInstruction>(match.dot)));

  return match;
}

static bool AreInstructionsOnTheSameStream(
    absl::Span<const HloInstruction* const> instructions) {
  absl::flat_hash_set<int64_t> stream_set;
  for (const HloInstruction* inst : instructions) {
    auto gpu_config = inst->backend_config<GpuBackendConfig>();
    if (!gpu_config.ok()) {
      continue;
    }
    stream_set.insert(gpu_config->operation_queue_id());
    if (stream_set.size() > 1) {
      return false;
    }
  }
  return true;
};

//===----------------------------------------------------------------------===//
// Cutlass Gemm Patterns
//===----------------------------------------------------------------------===//

std::optional<CustomKernelFusionPattern::Match> CutlassGemmPattern::TryMatch(
    const se::DeviceDescription& device, HloInstruction* instr) const {
  auto* dot = DynCast<HloDotInstruction>(instr);
  if (!dot) return std::nullopt;

  auto matched = MatchSimpleGemm(dot, {PrimitiveType::F32});
  if (!matched.ok()) return std::nullopt;

  CustomFusionConfig config;
  config.set_name("cutlass_gemm");
  return Match{config, {instr}};
}

std::optional<CustomKernelFusionPattern::Match>
CutlassGemmWithDynamicUpdateSlicePattern::TryMatch(
    const se::DeviceDescription& device, HloInstruction* instr) const {
  auto* update_slice = DynCast<HloDynamicUpdateSliceInstruction>(instr);
  if (!update_slice) return std::nullopt;

  auto matched = MatchGemmWithDynamicUpdateSlice(update_slice);
  if (!matched.ok() || !AreInstructionsOnTheSameStream(matched->Instrs()))
    return std::nullopt;

  CustomFusionConfig config;
  config.set_name("cutlass_gemm_with_dynamic_update_slice");

  Match match(config, matched->Instrs());

  // Add an optional replacement for intermediate dot instruction as a
  // dynamic-slice from the fusion result.
  match.AddReplacement(matched->dot, [=](HloFusionInstruction* fusion) {
    HloComputation* parent = fusion->parent();
    auto* dus = Cast<HloDynamicUpdateSliceInstruction>(matched->update_slice);
    bool has_bitcast = matched->bitcast != nullptr;
    const Shape dus_shape =
        has_bitcast ? matched->bitcast->shape() : matched->dot->shape();
    auto* slice = parent->AddInstruction(HloInstruction::CreateDynamicSlice(
        dus_shape, fusion, dus->index_operands(), dus_shape.dimensions()));

    return parent->AddInstruction(
        HloInstruction::CreateBitcast(matched->dot->shape(), slice));
  });

  return match;
}

std::optional<CustomKernelFusionPattern::Match>
CutlassGemmWithUpcastPattern::TryMatch(const se::DeviceDescription& device,
                                       HloInstruction* instr) const {
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

class CutlassGemmFusion : public CustomKernelFusion {
 public:
  absl::StatusOr<std::vector<CustomKernel>> LoadKernels(
      const se::DeviceDescription& device,
      const HloComputation* computation) const final {
    auto* dot = DynCast<HloDotInstruction>(computation->root_instruction());
    if (dot == nullptr) {
      return absl::InternalError(
          "cutlass_gemm requires ROOT operation to be a dot");
    }

    TF_RETURN_IF_ERROR(MatchSimpleGemm(dot, {PrimitiveType::F32}));

    auto dtype = dot->shape().element_type();

    auto* lhs = Cast<HloParameterInstruction>(dot->operand(0));
    auto* rhs = Cast<HloParameterInstruction>(dot->operand(1));

    // Mapping from fusion arguments to gemm kernel arguments.
    kernel::gemm_universal::ArgsIndices indices = {
        lhs->parameter_number(), rhs->parameter_number(),
        computation->num_parameters()};

    auto& lhs_shape = lhs->shape();
    auto& rhs_shape = rhs->shape();

    size_t m = lhs_shape.dimensions(0);
    size_t k = lhs_shape.dimensions(1);
    size_t n = rhs_shape.dimensions(1);

    TF_ASSIGN_OR_RETURN(
        auto kernel,
        kernel::gemm_universal::GetCutlassGemmKernel(
            "cutlass_gemm", dtype, m, n, k, indices, /*slices=*/{}, device));
    return std::vector<CustomKernel>{std::move(kernel)};
  }
};

class CutlassGemmWithUpcastFusion : public CustomKernelFusion {
 public:
  absl::StatusOr<std::vector<CustomKernel>> LoadKernels(
      const se::DeviceDescription& device,
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

class CutlassGemmWithDynamicUpdateSliceFusion : public CustomKernelFusion {
 public:
  absl::StatusOr<std::vector<CustomKernel>> LoadKernels(
      const se::DeviceDescription& device,
      const HloComputation* computation) const final {
    auto [root, workspace] =
        MatchRootWithWorkspace(computation->root_instruction());

    auto* dus = DynCast<HloDynamicUpdateSliceInstruction>(root);
    if (dus == nullptr) {
      return absl::InternalError(
          "cutlass_gemm_with_dynamic_update_slice requires ROOT operation to "
          "be a dynamic update slice");
    }

    TF_ASSIGN_OR_RETURN(auto matched, MatchGemmWithDynamicUpdateSlice(dus));
    TF_RETURN_IF_ERROR(
        MatchSimpleGemm(Cast<HloDotInstruction>(matched.dot),
                        {PrimitiveType::F32, PrimitiveType::BF16}));

    auto dtype = matched.dot->shape().element_type();

    auto* lhs = Cast<HloParameterInstruction>(matched.dot->operand(0));
    auto* rhs = Cast<HloParameterInstruction>(matched.dot->operand(1));
    auto* out = Cast<HloParameterInstruction>(matched.update_slice->operand(0));

    // Mapping from fusion arguments to gemm kernel arguments.
    kernel::gemm_universal::ArgsIndices args_indices = {
        lhs->parameter_number(), rhs->parameter_number(),
        out->parameter_number(), /*has_workspace=*/workspace != nullptr};

    // Mapping to a buffer that holds output slice offset.
    auto* offset =
        Cast<HloParameterInstruction>(matched.update_slice->operand(2));
    kernel::gemm_universal::DynamicSliceIndices slices;
    slices.out = offset->parameter_number();

    auto& lhs_shape = lhs->shape();
    auto& rhs_shape = rhs->shape();

    size_t m = lhs_shape.dimensions(0);
    size_t k = lhs_shape.dimensions(1);
    size_t n = rhs_shape.dimensions(1);

    TF_ASSIGN_OR_RETURN(
        auto kernel, kernel::gemm_universal::GetCutlassGemmKernel(
                         "cutlass_gemm_with_dynamic_update_slice", dtype, m, n,
                         k, args_indices, slices, device));
    return std::vector<CustomKernel>{std::move(kernel)};
  }
};

}  // namespace xla::gpu

XLA_REGISTER_CUSTOM_FUSION_PATTERN(
    ::xla::gpu::CutlassGemmWithDynamicUpdateSlicePattern);

XLA_REGISTER_CUSTOM_FUSION("cutlass_gemm", ::xla::gpu::CutlassGemmFusion);
XLA_REGISTER_CUSTOM_FUSION("cutlass_gemm_with_upcast",
                           ::xla::gpu::CutlassGemmWithUpcastFusion);
XLA_REGISTER_CUSTOM_FUSION("cutlass_gemm_with_dynamic_update_slice",
                           ::xla::gpu::CutlassGemmWithDynamicUpdateSliceFusion);

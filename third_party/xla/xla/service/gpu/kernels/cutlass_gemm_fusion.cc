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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion_pattern.h"
#include "xla/service/gpu/kernels/cutlass_gemm.h"
#include "xla/service/gpu/kernels/cutlass_gemm_custom_kernel.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// Cutlass Gemm pattern matching helpers
//===----------------------------------------------------------------------===//

namespace {

// If custom fusion requires extra workspace at run time, ROOT instruction will
// be a tuple with second operand being a result of workspace allocation custom
// call.
struct RootWithWorkspace {
  HloInstruction* root;
  HloInstruction* workspace;
};

RootWithWorkspace MatchRootWithWorkspace(HloInstruction* root) {
  RootWithWorkspace result;
  if (Match(root, match::Tuple(match::Op(&result.root),
                               match::CustomCall(
                                   &result.workspace,
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

// Returns OK if dot instruction is a simple 2D row-major gemm.
absl::Status MatchRowMajorGemm(HloDotInstruction* dot) {
  if (dot->operand(0)->shape().dimensions().size() != 2 ||
      dot->operand(1)->shape().dimensions().size() != 2) {
    return absl::InternalError("operands must have rank 2");
  }

  if (dot->shape().layout().minor_to_major().back() != 0) {
    return absl::InternalError("The dot result must have row major layout.");
  }

  auto& dot_dims = dot->dot_dimension_numbers();

  if (dot_dims.lhs_contracting_dimensions().size() != 1) {
    return absl::InternalError("Lhs contracting dimensions must be of size 1.");
  }

  if (dot_dims.rhs_contracting_dimensions().size() != 1) {
    return absl::InternalError("Rhs contracting dimensions must be of size 1.");
  }

  if (dot->operand(0)->shape().layout().minor_to_major(0) !=
      dot_dims.lhs_contracting_dimensions()[0]) {
    return absl::InternalError(
        "Lhs contracting dimension should be along the minor axis (elements "
        "that are stored contiguous in memory).");
  }

  if (dot->operand(1)->shape().layout().minor_to_major(1) !=
      dot_dims.rhs_contracting_dimensions()[0]) {
    return absl::InternalError(
        "Rhs contracting dimension should be along the major axis (elements "
        "that are NOT stored contiguous in memory).");
  }

  return absl::OkStatus();
}
}  // namespace

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

// Returns matched GEMM with one or both the operands upcasted to the
// accumulator data type with an HLO convert instruction.
static absl::StatusOr<GemmWithUpcast> MatchGemmWithUpcast(
    HloDotInstruction* dot) {
  TF_RETURN_IF_ERROR(MatchRowMajorGemm(dot));

  GemmWithUpcast match(dot);

  // C <- convert(A) * convert(B)
  if (Match(const_cast<HloInstruction*>(dot->operand(0)),
            match::Convert(&match.lhs_upcast, match::Op())) &&
      Match(const_cast<HloInstruction*>(dot->operand(1)),
            match::Convert(&match.rhs_upcast, match::Op()))) {
    return match;
  }

  // C <- convert(A) * B
  if (Match(const_cast<HloInstruction*>(dot->operand(0)),
            match::Convert(&match.lhs_upcast, match::Op()))) {
    return match;
  }

  // C <- A * convert(B)
  if (Match(const_cast<HloInstruction*>(dot->operand(1)),
            match::Convert(&match.rhs_upcast, match::Op()))) {
    return match;
  }

  return absl::InternalError("unsupported gemm with upcasing");
}

template <typename Pattern>
auto OptionalBitcast(HloInstruction** optional_bitcast, Pattern pattern) {
  return match::AnyOf<HloInstruction>(match::Bitcast(optional_bitcast, pattern),
                                      std::move(pattern));
}

// Returns matched GEMM with result used to update a slice.
static absl::StatusOr<GemmWithDynamicSlice> MatchGemmWithDynamicUpdateSlice(
    HloDynamicUpdateSliceInstruction* update_slice) {
  GemmWithDynamicSlice match(update_slice);

  if (!Match(const_cast<HloInstruction*>(update_slice->update()),
             OptionalBitcast(&match.bitcast, match::Dot(&match.dot, match::Op(),
                                                        match::Op())))) {
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
  // This pattern is disabled for VOLTA. See b/380087823.
  if (std::holds_alternative<se::CudaComputeCapability>(
          device.gpu_compute_capability())) {
    if (device.cuda_compute_capability().major ==
        se::CudaComputeCapability::CudaComputeCapabilities::kVolta) {
      return std::nullopt;
    }
  }
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

namespace {
bool IsSupportedKernel(PrimitiveType lhs, PrimitiveType rhs,
                       PrimitiveType dot) {
  // List of supported kernels using {lhs_type, rhs_type, dot_type}.
  constexpr std::array<std::array<PrimitiveType, 3>, 4> kSupportedKernels = {
      {{BF16, BF16, F32}, {F32, BF16, F32}, {BF16, S8, F32}}};
  return absl::c_linear_search(kSupportedKernels,
                               std::array<PrimitiveType, 3>{lhs, rhs, dot});
}
}  // namespace

std::optional<CustomKernelFusionPattern::Match>
CutlassGemmWithUpcastPattern::TryMatch(const se::DeviceDescription& device,
                                       HloInstruction* instr) const {
  auto* dot = DynCast<HloDotInstruction>(instr);
  if (!dot) return std::nullopt;

  absl::StatusOr<GemmWithUpcast> matched = MatchGemmWithUpcast(dot);

  if (!matched.ok()) {
    VLOG(3) << "No match due to unsupported gemm with upcast: "
            << matched.status();
    return std::nullopt;
  }

  CustomFusionConfig config;
  config.set_name("cutlass_gemm_with_upcast");

  HloInstruction* lhs = matched->lhs_upcast;
  HloInstruction* rhs = matched->rhs_upcast;
  PrimitiveType dot_type = dot->shape().element_type();
  PrimitiveType lhs_type = lhs != nullptr
                               ? lhs->operand(0)->shape().element_type()
                               : dot->operand(0)->shape().element_type();
  PrimitiveType rhs_type = rhs != nullptr
                               ? rhs->operand(0)->shape().element_type()
                               : dot->operand(1)->shape().element_type();
  if (!IsSupportedKernel(lhs_type, rhs_type, dot_type)) {
    VLOG(3) << "No match due to unsupported kernel input types: "
            << PrimitiveType_Name(lhs_type) << "x"
            << PrimitiveType_Name(rhs_type) << "To"
            << PrimitiveType_Name(dot_type);
    return std::nullopt;
  }

  if (lhs != nullptr && rhs == nullptr) {
    return Match{config, {matched->lhs_upcast, instr}};
  } else if (lhs == nullptr && rhs != nullptr) {
    return Match{config, {matched->rhs_upcast, instr}};
  } else {
    return Match{config, {matched->lhs_upcast, matched->rhs_upcast, instr}};
  }
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

    PrimitiveType dot_type = dot->shape().element_type();

    auto* lhs = Cast<HloParameterInstruction>(dot->operand(0));
    auto* rhs = Cast<HloParameterInstruction>(dot->operand(1));

    // Mapping from fusion arguments to gemm kernel arguments.
    kernel::gemm_universal::ArgsIndices indices = {
        lhs->parameter_number(), rhs->parameter_number(),
        computation->num_parameters()};

    const Shape& lhs_shape = lhs->shape();
    const Shape& rhs_shape = rhs->shape();

    size_t m = lhs_shape.dimensions(0);
    size_t k = lhs_shape.dimensions(1);
    size_t n = rhs_shape.dimensions(1);

    PrimitiveType lhs_type = lhs->shape().element_type();
    PrimitiveType rhs_type = rhs->shape().element_type();

    return GetCutlassGemmKernels("cutlass_gemm", dot_type, lhs_type, rhs_type,
                                 m, n, k, indices,
                                 /*slices=*/{}, device);
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
          "cutlass_gemm_with_upcast requires ROOT operation to be a dot");
    }

    TF_ASSIGN_OR_RETURN(GemmWithUpcast matched, MatchGemmWithUpcast(dot));

    const HloParameterInstruction* lhs;
    const HloParameterInstruction* rhs;

    if (matched.lhs_upcast == nullptr && matched.rhs_upcast != nullptr) {
      lhs = Cast<HloParameterInstruction>(matched.dot->operand(0));
      rhs = Cast<HloParameterInstruction>(matched.rhs_upcast->operand(0));
    } else if (matched.lhs_upcast != nullptr && matched.rhs_upcast == nullptr) {
      lhs = Cast<HloParameterInstruction>(matched.lhs_upcast->operand(0));
      rhs = Cast<HloParameterInstruction>(matched.dot->operand(1));
    } else {
      lhs = Cast<HloParameterInstruction>(matched.lhs_upcast->operand(0));
      rhs = Cast<HloParameterInstruction>(matched.rhs_upcast->operand(0));
    }

    const Shape& lhs_shape = lhs->shape();
    const Shape& rhs_shape = rhs->shape();

    size_t m = lhs_shape.dimensions(0);
    size_t k = lhs_shape.dimensions(1);
    size_t n = rhs_shape.dimensions(1);

    PrimitiveType dot_type = dot->shape().element_type();
    PrimitiveType lhs_type = lhs_shape.element_type();
    PrimitiveType rhs_type = rhs_shape.element_type();

    // Mapping from fusion arguments to gemm kernel arguments.
    kernel::gemm_universal::ArgsIndices args_indices = {
        lhs->parameter_number(), rhs->parameter_number(),
        computation->num_parameters()};

    return GetCutlassGemmKernels("cutlass_gemm_with_upcast", dot_type, lhs_type,
                                 rhs_type, m, n, k, args_indices, /*slices=*/{},
                                 device);
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

    auto dot_type = matched.dot->shape().element_type();

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

    const Shape& lhs_shape = lhs->shape();
    const Shape& rhs_shape = rhs->shape();

    size_t m = lhs_shape.dimensions(0);
    size_t k = lhs_shape.dimensions(1);
    size_t n = rhs_shape.dimensions(1);

    PrimitiveType lhs_type = lhs->shape().element_type();
    PrimitiveType rhs_type = rhs->shape().element_type();

    return GetCutlassGemmKernels("cutlass_gemm_with_dynamic_update_slice",
                                 dot_type, lhs_type, rhs_type, m, n, k,
                                 args_indices, slices, device);
  }
};

}  // namespace xla::gpu

XLA_REGISTER_CUSTOM_FUSION_PATTERN(::xla::gpu::CutlassGemmWithUpcastPattern);
XLA_REGISTER_CUSTOM_FUSION_PATTERN(
    ::xla::gpu::CutlassGemmWithDynamicUpdateSlicePattern);

XLA_REGISTER_CUSTOM_FUSION("cutlass_gemm", ::xla::gpu::CutlassGemmFusion);
XLA_REGISTER_CUSTOM_FUSION("cutlass_gemm_with_upcast",
                           ::xla::gpu::CutlassGemmWithUpcastFusion);
XLA_REGISTER_CUSTOM_FUSION("cutlass_gemm_with_dynamic_update_slice",
                           ::xla::gpu::CutlassGemmWithDynamicUpdateSliceFusion);

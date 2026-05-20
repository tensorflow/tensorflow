/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/ragged_dot_fusion_rewriter.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

std::unique_ptr<HloComputation> CreateScalarAddComputation(PrimitiveType type) {
  auto embedded_builder = HloComputation::Builder("add");
  auto lhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(type, {}), "lhs"));
  auto rhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(type, {}), "rhs"));
  embedded_builder.AddInstruction(
      HloInstruction::CreateBinary(lhs->shape(), HloOpcode::kAdd, lhs, rhs));
  return embedded_builder.Build();
}

std::unique_ptr<HloInstruction> Zero(PrimitiveType type) {
  return HloInstruction::CreateConstant(LiteralUtil::Zero(type));
}

// Takes an array of shape [batch_dims..., num_groups] and returns an array of
// the same shape with the elements of the array along the last dimension
// now representing the cumulative sum of all elements in the input array up to
// the current group.
std::unique_ptr<HloInstruction> CreateCumulativeSum(
    HloInstruction* group_sizes) {
  int64_t batch_dims = group_sizes->shape().dimensions().size() - 1;
  int64_t num_groups = group_sizes->shape().dimensions(batch_dims);

  Window cumsum_window;
  // Add batch dimensions.
  for (int i = 0; i < batch_dims; ++i) {
    WindowDimension* dim = cumsum_window.add_dimensions();
    dim->set_size(1);
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_stride(1);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  // Add group dimension.
  WindowDimension* dim = cumsum_window.add_dimensions();
  dim->set_size(num_groups);
  dim->set_padding_low(num_groups - 1);
  dim->set_padding_high(0);
  dim->set_stride(1);
  dim->set_window_dilation(1);
  dim->set_base_dilation(1);

  auto type = group_sizes->shape().element_type();
  HloComputation* add = group_sizes->GetModule()->AddEmbeddedComputation(
      CreateScalarAddComputation(type));
  auto zero = group_sizes->parent()->AddInstruction(Zero(type));
  return HloInstruction::CreateReduceWindow(group_sizes->shape(), group_sizes,
                                            zero, cumsum_window, add);
}

absl::StatusOr<std::unique_ptr<HloInstruction>> RaggedToCuDNNFusion(
    HloRaggedDotInstruction* ragged_dot) {
  std::string fusion_name =
      absl::StrCat("ragged_dot_fusion_", ragged_dot->name());
  HloComputation::Builder builder(absl::StrCat(fusion_name, "_computation"));

  HloInstruction* fused_input =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ragged_dot->operand(0)->shape(), ragged_dot->operand(0)->name()));
  HloInstruction* fused_weight =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ragged_dot->operand(1)->shape(), ragged_dot->operand(1)->name()));

  auto computation = ragged_dot->parent();
  HloInstruction* group_sizes = ragged_dot->mutable_operand(2);
  // cuDNN accepts cumulative sum of group sizes
  HloInstruction* cumulative_sum =
      computation->AddInstruction(CreateCumulativeSum(group_sizes));
  HloInstruction* sub =
      computation->AddInstruction(HloInstruction::CreateBinary(
          cumulative_sum->shape(), HloOpcode::kSubtract, cumulative_sum,
          group_sizes));
  HloInstruction* fused_sub_cum_group_size = builder.AddInstruction(
      HloInstruction::CreateParameter(2, sub->shape(), sub->name()));

  HloInstruction* fused_ragged_dot =
      builder.AddInstruction(ragged_dot->CloneWithNewOperands(
          ragged_dot->shape(),
          {fused_input, fused_weight, fused_sub_cum_group_size}));

  HloComputation* new_computation =
      ragged_dot->GetModule()->AddComputationAndUnifyNamesAndIds(
          builder.Build(fused_ragged_dot), /*is_entry=*/false);
  std::vector<HloInstruction*> fusion_params = {
      ragged_dot->mutable_operand(0), ragged_dot->mutable_operand(1), sub};
  return HloInstruction::CreateFusion(ragged_dot->shape(),
                                      HloInstruction::FusionKind::kCustom,
                                      fusion_params, new_computation);
}

}  // namespace

absl::StatusOr<bool> RaggedDotFusionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloRaggedDotInstruction*> ragged_dots;
  for (auto* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kRaggedDot) {
        ragged_dots.push_back(Cast<HloRaggedDotInstruction>(instruction));
      }
    }
  }

  for (auto* ragged_dot : ragged_dots) {
    TF_ASSIGN_OR_RETURN(auto ragged_dot_fusion,
                        RaggedToCuDNNFusion(ragged_dot));
    gpu::GpuBackendConfig gpu_backend_config;
    gpu::FusionBackendConfig* fusion_config =
        gpu_backend_config.mutable_fusion_backend_config();
    fusion_config->set_kind(gpu::kCuDnnFusionKind);
    TF_RETURN_IF_ERROR(
        ragged_dot_fusion->set_backend_config(gpu_backend_config));
    ragged_dot_fusion->set_metadata(ragged_dot->metadata());
    TF_RETURN_IF_ERROR(ragged_dot->parent()->ReplaceWithNewInstruction(
        ragged_dot, std::move(ragged_dot_fusion)));
  }

  return !ragged_dots.empty();
}

}  // namespace gpu
}  // namespace xla

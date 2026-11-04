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

#include "xla/backends/gpu/transforms/horizontal_fusion.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/codegen/thread_block_specialized.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

// Maximum total kernel arguments (deduplicated inputs + outputs).
constexpr int64_t kMaxKernelArgs = 64;

struct FusionBatch {
  std::vector<HloInstruction*> instructions;
  absl::flat_hash_set<const HloInstruction*> unique_operands;
  int64_t total_blocks = 0;
};

bool IsZeroCostOp(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kConstant:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
      return true;
    default:
      return false;
  }
}

std::optional<LaunchDimensions> GetLaunchDimensions(
    const HloInstruction* instr, const se::DeviceDescription& device_info) {
  const auto* fusion = DynCast<HloFusionInstruction>(instr);
  if (fusion == nullptr) {
    return std::nullopt;
  }

  // Single-output native MLIR fusions only.
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom ||
      !fusion->shape().IsArray()) {
    return std::nullopt;
  }

  // Dead or side-effecting fusions are not supported.
  if (fusion->IsDead() || fusion->HasSideEffect() ||
      !fusion->output_to_operand_aliasing().empty()) {
    return std::nullopt;
  }

  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info);

  PreBufferAssignmentFusionInfo fusion_info(analysis);
  // Trivial copy and dynamic update slice fusions are not supported.
  if (fusion_info.GetCopyFusion().has_value() ||
      (IsDynamicUpdateSliceFusion(analysis.fusion_spec()) &&
       fusion_info.CanEmitDynamicUpdateSliceInPlace())) {
    return std::nullopt;
  }

  std::unique_ptr<MlirKernelEmitter> emitter =
      ThreadBlockSpecializedFusion::CreateNativeNestedEmitter(analysis);
  if (emitter == nullptr) {
    return std::nullopt;
  }
  LaunchDimensions launch_dims = emitter->launch_dimensions();

  // If a nested kernel has less than max_threads and uses a barrier, we need to
  // use `bar.sync 0, T_i` in PTX. The instruction has a requirement that T_i
  // must be a multiple of the warp size.
  //
  // Only loop emitter can have a count of threads that is not a multiple of the
  // warp size, but we also know that loop emitter never uses barriers.
  if (launch_dims.num_threads_per_block() % WarpSize(device_info) != 0 &&
      analysis.emitter_fusion_kind() !=
          HloFusionAnalysis::EmitterFusionKind::kLoop) {
    return std::nullopt;
  }

  return launch_dims;
}

bool TryAddToBatch(FusionBatch& batch, HloInstruction* candidate,
                   const HloReachabilityMap& reachability,
                   const se::DeviceDescription& device_info,
                   const HorizontalFusion::Options& options) {
  if (batch.instructions.size() >= options.max_nested_kernels) {
    return false;
  }

  if (absl::c_any_of(batch.instructions, [&](const HloInstruction* batch_inst) {
        return reachability.IsReachable(batch_inst, candidate);
      })) {
    return false;
  }

  std::optional<LaunchDimensions> candidate_dims =
      GetLaunchDimensions(candidate, device_info);
  if (!candidate_dims.has_value()) {
    return false;
  }

  const int64_t total_blocks =
      batch.total_blocks + candidate_dims->num_blocks();
  if (options.max_blocks_per_sm_factor > 0 && device_info.core_count() > 0 &&
      total_blocks >
          options.max_blocks_per_sm_factor * device_info.core_count()) {
    return false;
  }

  // Count deduplicated outer operands + outputs.
  absl::flat_hash_set<const HloInstruction*> unique_operands =
      batch.unique_operands;
  unique_operands.insert(candidate->operands().begin(),
                         candidate->operands().end());

  const int64_t num_outputs = batch.instructions.size() + 1;
  const int64_t total_kernel_args = unique_operands.size() + num_outputs;
  if (total_kernel_args > kMaxKernelArgs) {
    return false;
  }

  batch.instructions.push_back(candidate);
  batch.unique_operands = std::move(unique_operands);
  batch.total_blocks = total_blocks;
  return true;
}

absl::Status FuseCandidates(HloModule* module, HloComputation* comp,
                            absl::Span<HloInstruction* const> candidates) {
  HloComputation::Builder builder("horizontal_fusion_computation");
  builder.AddInstruction(HloInstruction::CreateTuple({}));
  HloComputation* outer_computation =
      module->AddEmbeddedComputation(builder.Build());
  HloInstruction* outer_fusion = comp->AddInstruction(
      HloInstruction::CreateFusion(ShapeUtil::MakeNil(),
                                   HloInstruction::FusionKind::kLoop,
                                   /*operands=*/{}, outer_computation),
      "horizontal_fusion");

  for (HloInstruction* nested_fusion : candidates) {
    HloComputation* comp_i = nested_fusion->fused_instructions_computation();
    ABSL_RETURN_IF_ERROR(outer_fusion->CopyAllControlDepsFrom(nested_fusion));
    ABSL_RETURN_IF_ERROR(nested_fusion->DropAllControlDeps());
    outer_fusion->FuseInstructionIntoMultiOutput(nested_fusion);
    ABSL_RETURN_IF_ERROR(comp->RemoveInstruction(nested_fusion));
    ABSL_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(comp_i));
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> HorizontalFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->has_schedule()) {
    return false;
  }

  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    if (!module->schedule().is_computation_scheduled(comp)) {
      continue;
    }

    const std::vector<HloInstruction*> sequence =
        module->schedule().sequence(comp).instructions();
    std::unique_ptr<HloReachabilityMap> reachability =
        HloReachabilityMap::Build(comp);

    std::vector<std::vector<HloInstruction*>> batches_to_fuse;
    FusionBatch current_batch;

    auto flush_batch = [&]() {
      if (current_batch.instructions.size() >= 2) {
        batches_to_fuse.push_back(std::move(current_batch.instructions));
      }
      current_batch = FusionBatch{};
    };

    for (HloInstruction* instr : sequence) {
      if (IsZeroCostOp(instr)) {
        continue;
      }

      if (!TryAddToBatch(current_batch, instr, *reachability, device_info_,
                         options_)) {
        flush_batch();
        TryAddToBatch(current_batch, instr, *reachability, device_info_,
                      options_);
      }
    }
    flush_batch();

    for (const std::vector<HloInstruction*>& batch : batches_to_fuse) {
      ABSL_RETURN_IF_ERROR(FuseCandidates(module, comp, batch));
      changed = true;
    }
  }

  if (changed) {
    ABSL_RETURN_IF_ERROR(module->schedule().Update());
  }
  return changed;
}

}  // namespace xla::gpu

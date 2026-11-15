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
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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
#include "xla/service/gpu/backend_configs.pb.h"
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
  int64_t total_outputs = 0;
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

  // Native MLIR fusions only. This is only a first filter. The call to
  // CreateNativeNestedEmitter below verifies that fusion actually can be
  // emitted with a native emitter.
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom) {
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

  // TODO(b/480040242): Consider unrolling small fusions in grid-bound
  // horizontal fusions as well.
  std::unique_ptr<MlirKernelEmitter> emitter =
      ThreadBlockSpecializedFusion::CreateNativeNestedEmitter(
          analysis, /*unroll_small_fusions=*/false);
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

  const int64_t num_outputs =
      batch.total_outputs + ShapeUtil::GetLeafCount(candidate->shape());
  if (unique_operands.size() + num_outputs > kMaxKernelArgs) {
    return false;
  }

  batch.instructions.push_back(candidate);
  batch.unique_operands = std::move(unique_operands);
  batch.total_blocks = total_blocks;
  batch.total_outputs = num_outputs;
  return true;
}

absl::Status FuseCandidates(HloModule* module, HloComputation* comp,
                            absl::Span<HloInstruction* const> candidates) {
  HloComputation::Builder builder("horizontal_fusion_computation");
  std::vector<HloInstruction*> outer_operands;
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> operand_to_param;
  std::vector<HloInstruction*> tuple_elements;

  for (HloInstruction* nested_fusion : candidates) {
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(nested_fusion->operand_count());
    for (HloInstruction* operand : nested_fusion->mutable_operands()) {
      auto [it, inserted] = operand_to_param.try_emplace(operand, nullptr);
      if (inserted) {
        it->second = builder.AddInstruction(HloInstruction::CreateParameter(
            outer_operands.size(), operand->shape(),
            absl::StrCat("param_", outer_operands.size())));
        outer_operands.push_back(operand);
      }
      new_operands.push_back(it->second);
    }

    HloInstruction* cloned_fusion =
        builder.AddInstruction(nested_fusion->CloneWithNewOperands(
            nested_fusion->shape(), new_operands, /*suffix=*/""));
    if (nested_fusion->shape().IsTuple()) {
      for (int64_t i = 0; i < nested_fusion->shape().tuple_shapes_size(); ++i) {
        tuple_elements.push_back(builder.AddInstruction(
            HloInstruction::CreateGetTupleElement(cloned_fusion, i)));
      }
    } else {
      tuple_elements.push_back(cloned_fusion);
    }
  }

  HloInstruction* root_tuple =
      builder.AddInstruction(HloInstruction::CreateTuple(tuple_elements));
  HloComputation* outer_computation =
      module->AddEmbeddedComputation(builder.Build(root_tuple));
  HloInstruction* outer_fusion = comp->AddInstruction(
      HloInstruction::CreateFusion(root_tuple->shape(),
                                   HloInstruction::FusionKind::kCustom,
                                   outer_operands, outer_computation),
      "horizontal_fusion");

  GpuBackendConfig gpu_backend_config;
  gpu_backend_config.mutable_fusion_backend_config()->set_kind(
      kThreadBlockSpecializedFusionKind);
  ABSL_RETURN_IF_ERROR(outer_fusion->set_backend_config(gpu_backend_config));

  int64_t output_idx = 0;
  for (HloInstruction* nested_fusion : candidates) {
    HloComputation* comp_i = nested_fusion->fused_instructions_computation();
    ABSL_RETURN_IF_ERROR(outer_fusion->CopyAllControlDepsFrom(nested_fusion));
    ABSL_RETURN_IF_ERROR(nested_fusion->DropAllControlDeps());
    if (nested_fusion->shape().IsTuple()) {
      // Redirect GetTupleElement users to the new outer fusion.
      for (HloInstruction* user : std::vector<HloInstruction*>(
               nested_fusion->users().begin(), nested_fusion->users().end())) {
        if (user->opcode() != HloOpcode::kGetTupleElement) {
          continue;
        }
        user->set_tuple_index(output_idx + user->tuple_index());
        ABSL_RETURN_IF_ERROR(
            nested_fusion->ReplaceUseWithDifferentShape(user, outer_fusion));
      }

      // If there are still users left, then the tuple output was used as a
      // whole. Replace it with a new tuple create with gtes of the outer
      // fusion.
      if (nested_fusion->user_count() > 0 || nested_fusion->IsRoot()) {
        int64_t num_outputs = nested_fusion->shape().tuple_shapes().size();
        std::vector<HloInstruction*> new_gtes;
        new_gtes.reserve(num_outputs);
        // It's intentional that we create new gtes here instead of reusing the
        // one created above. Potentially, gte can have scheduling predecessor
        // annotation that we don't want to inherit.
        for (int64_t i = 0; i < num_outputs; ++i) {
          new_gtes.push_back(
              comp->AddInstruction(HloInstruction::CreateGetTupleElement(
                  outer_fusion, output_idx + i)));
        }
        HloInstruction* new_tuple =
            comp->AddInstruction(HloInstruction::CreateTuple(new_gtes));
        ABSL_RETURN_IF_ERROR(nested_fusion->ReplaceAllUsesWith(new_tuple));
      }
    } else {
      HloInstruction* new_gte = comp->AddInstruction(
          HloInstruction::CreateGetTupleElement(outer_fusion, output_idx));
      ABSL_RETURN_IF_ERROR(nested_fusion->ReplaceAllUsesWith(new_gte));
    }
    output_idx += ShapeUtil::GetLeafCount(nested_fusion->shape());

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

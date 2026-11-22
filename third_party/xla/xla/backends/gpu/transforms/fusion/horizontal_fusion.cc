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

#include "xla/backends/gpu/transforms/fusion/horizontal_fusion.h"

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
#include "llvm/ADT/STLExtras.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/codegen/thread_block_specialized.h"
#include "xla/backends/gpu/codegen/triton/fusion.h"
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
#include "xla/service/tuple_util.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
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

  LaunchDimensions launch_dims;
  if (analysis.emitter_fusion_kind() ==
      HloFusionAnalysis::EmitterFusionKind::kThreadBlockSpecialized) {
    launch_dims = ThreadBlockSpecializedFusion(analysis).launch_dimensions();
  } else if (analysis.emitter_fusion_kind() ==
             HloFusionAnalysis::EmitterFusionKind::kTriton) {
    std::optional<TritonFusion::LaunchConfig> launch_config =
        TritonFusion::GetLaunchConfig(&analysis);
    if (!launch_config.has_value()) {
      return std::nullopt;
    }
    if (launch_config->block_level_parameters.is_tma_allowed ||
        launch_config->block_level_parameters.is_warp_specialization_allowed) {
      return std::nullopt;
    }
    launch_dims = launch_config->launch_dimensions;
  } else if (std::unique_ptr<MlirKernelEmitter> emitter =
                 // TODO(b/480040242): Consider unrolling small fusions in
                 // grid-bound horizontal fusions as well.
             ThreadBlockSpecializedFusion::CreateNativeNestedEmitter(
                 analysis, /*unroll_small_fusions=*/false)) {
    launch_dims = emitter->launch_dimensions();
  } else {
    return std::nullopt;
  }

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

// Unpacks `inst` inside `builder`, returning a ShapeTree whose leaves are the
// non-tuple sub-elements of `inst` (extracted via GetTupleElement).
ShapeTree<HloInstruction*> DisassembleTuple(HloComputation::Builder& builder,
                                            HloInstruction* inst) {
  ShapeTree<HloInstruction*> unpacked(&inst->shape());
  unpacked.ForEachMutableElement(
      [&](ShapeIndexView index, HloInstruction** elem) {
        if (index.empty()) {
          *elem = inst;
        } else {
          HloInstruction* parent =
              unpacked.element(index.subspan(0, index.size() - 1));
          *elem = builder.AddInstruction(
              HloInstruction::CreateGetTupleElement(parent, index.back()));
        }
      });
  return unpacked;
}

// Replaces all uses of `old_inst` with outputs from the flattened
// `outer_fusion` tuple, using `flat_indices` to map each leaf ShapeIndex of
// `old_inst` to its flat index in `outer_fusion`.
//
// For tuple-shaped instructions, direct (possibly nested) GetTupleElement users
// are redirected in-place to their corresponding flat leaf index (preserving
// control dependencies and avoiding unnecessary tuple ops after scheduling).
// Any remaining whole-tuple uses are replaced with a reconstructed tuple
// matching `old_inst->shape()`. Control dependencies on `old_inst` itself (for
// intermediate tuple-shaped GetTupleElements being removed) are propagated to
// redirected GTE users, any reconstructed tuple, and `outer_fusion` before
// being dropped.
absl::Status ReplaceUsesWithOuterFusion(
    HloComputation* comp, HloInstruction* old_inst,
    HloInstruction* outer_fusion, const ShapeTree<int64_t>& flat_indices) {
  if (!old_inst->shape().IsTuple()) {
    HloInstruction* new_gte =
        comp->AddInstruction(HloInstruction::CreateGetTupleElement(
            outer_fusion, flat_indices.element({})));
    return old_inst->ReplaceAllUsesWith(new_gte);
  }

  // Redirect GetTupleElement users to the flat outer_fusion outputs.
  for (HloInstruction* user : std::vector<HloInstruction*>(
           old_inst->users().begin(), old_inst->users().end())) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      continue;
    }
    for (HloInstruction* pred : old_inst->control_predecessors()) {
      ABSL_RETURN_IF_ERROR(pred->AddControlDependencyTo(user));
    }
    if (user->shape().IsTuple()) {
      ABSL_ASSIGN_OR_RETURN(ShapeTree<int64_t> child_indices,
                       flat_indices.SubShapeTree({user->tuple_index()}));
      ABSL_RETURN_IF_ERROR(
          ReplaceUsesWithOuterFusion(comp, user, outer_fusion, child_indices));
      ABSL_RETURN_IF_ERROR(comp->RemoveInstruction(user));
    } else {
      user->set_tuple_index(flat_indices.element({user->tuple_index()}));
      ABSL_RETURN_IF_ERROR(
          old_inst->ReplaceUseWithDifferentShape(user, outer_fusion));
    }
  }

  // If there are still users left, then the tuple output was used as a whole.
  // Reconstruct a matching tuple from fresh GTEs of the outer fusion (rather
  // than reusing existing GTEs, which may carry scheduling annotations).
  if (old_inst->user_count() > 0 || old_inst->IsRoot()) {
    ShapeTree<HloInstruction*> elements(&old_inst->shape());
    for (const auto& [index, flat_idx] : flat_indices.leaves()) {
      *elements.mutable_element(index) = comp->AddInstruction(
          HloInstruction::CreateGetTupleElement(outer_fusion, flat_idx));
    }
    HloInstruction* new_tuple =
        TupleUtil::AssembleTupleInstruction(comp, std::move(elements));
    ABSL_RETURN_IF_ERROR(new_tuple->CopyAllControlDepsFrom(old_inst));
    if (flat_indices.leaf_begin() == flat_indices.leaf_end() &&
        !old_inst->control_successors().empty()) {
      ABSL_RETURN_IF_ERROR(outer_fusion->AddControlDependencyTo(new_tuple));
    }
    ABSL_RETURN_IF_ERROR(old_inst->DropAllControlDeps());
    ABSL_RETURN_IF_ERROR(old_inst->ReplaceAllUsesWith(new_tuple));
  } else {
    for (HloInstruction* succ : old_inst->control_successors()) {
      ABSL_RETURN_IF_ERROR(outer_fusion->AddControlDependencyTo(succ));
    }
    ABSL_RETURN_IF_ERROR(old_inst->SafelyDropAllControlDependencies());
  }
  return absl::OkStatus();
}

absl::Status FuseCandidates(HloModule* module, HloComputation* comp,
                            absl::Span<HloInstruction* const> candidates) {
  HloComputation::Builder builder("horizontal_fusion_computation");
  std::vector<HloInstruction*> outer_operands;
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> operand_to_param;
  std::vector<HloInstruction*> tuple_elements;
  std::vector<ShapeTree<int64_t>> candidate_flat_indices;
  candidate_flat_indices.reserve(candidates.size());

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
    ShapeTree<HloInstruction*> unpacked =
        DisassembleTuple(builder, cloned_fusion);
    ShapeTree<int64_t> flat_indices(&cloned_fusion->shape(), -1);
    for (const auto& [index, leaf] : unpacked.leaves()) {
      *flat_indices.mutable_element(index) = tuple_elements.size();
      tuple_elements.push_back(leaf);
    }
    candidate_flat_indices.push_back(std::move(flat_indices));
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

  for (const auto& [nested_fusion, flat_indices] :
       llvm::zip_equal(candidates, candidate_flat_indices)) {
    HloComputation* comp_i = nested_fusion->fused_instructions_computation();
    ABSL_RETURN_IF_ERROR(outer_fusion->CopyAllControlDepsFrom(nested_fusion));
    ABSL_RETURN_IF_ERROR(nested_fusion->DropAllControlDeps());
    ABSL_RETURN_IF_ERROR(ReplaceUsesWithOuterFusion(comp, nested_fusion,
                                               outer_fusion, flat_indices));

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

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

#include "xla/backends/gpu/transforms/multi_phase_fusion.h"

#include <cstddef>
#include <cstdint>
#include <memory>
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
#include "xla/backends/gpu/codegen/emitters/loop.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/codegen/thread_block_specialized.h"
#include "xla/backends/gpu/transforms/fusion_wrapper.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu {
namespace {

// Tracks the ordered stages and O(1) stage membership of a candidate
// multi-phase fusion chain under construction.
struct ChainState {
  std::vector<HloInstruction*> stages;
  absl::flat_hash_set<const HloInstruction*> stage_set;
};

// Returns true if `instr` is a single-array 1D `kLoop` fusion (or unfused op
// that `FusionWrapper` would wrap as one) without side effects, aliasing, or
// control dependencies whose grid fits in one cluster.
bool IsEligibleStage(const HloInstruction* instr, FusionWrapper& fusion_wrapper,
                     const se::DeviceDescription& device_info) {
  if (!instr->shape().IsArray() || instr->IsDead() || instr->HasSideEffect() ||
      instr->HasControlDependencies()) {
    return false;
  }
  if (absl::c_any_of(instr->operands(), [](const HloInstruction* op) {
        return op->shape().IsTuple();
      })) {
    return false;
  }
  if (const auto* fusion = DynCast<HloFusionInstruction>(instr)) {
    if (fusion->fusion_kind() != HloInstruction::FusionKind::kLoop ||
        !fusion->output_to_operand_aliasing().empty()) {
      return false;
    }
  } else if (!instr->IsFusible() ||
             !fusion_wrapper.MustWrapInstruction(*instr) ||
             fusion_wrapper.ChooseFusionKind(*instr, *instr) !=
                 HloInstruction::FusionKind::kLoop) {
    return false;
  }

  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*instr, device_info);
  if (analysis.emitter_fusion_kind() !=
      HloFusionAnalysis::EmitterFusionKind::kLoop) {
    return false;
  }
  PreBufferAssignmentFusionInfo fusion_info(analysis);
  if (fusion_info.GetCopyFusion().has_value() ||
      (IsDynamicUpdateSliceFusion(analysis.fusion_spec()) &&
       fusion_info.CanEmitDynamicUpdateSliceInPlace())) {
    return false;
  }

  LaunchDimensions dims = LoopFusion(analysis).launch_dimensions();
  const se::BlockDim& blocks = dims.block_counts();
  const se::ThreadDim& threads = dims.thread_counts_per_block();
  return blocks.x > 0 &&
         blocks.x <= ThreadBlockSpecializedFusion::kMaxBlocksPerCluster &&
         blocks.y == 1 && blocks.z == 1 && threads.y == 1 && threads.z == 1;
}

// Appends `candidate` as the next phase of `chain` if it is a live eligible
// stage and introduces neither skip-phase edges nor cycles.
bool TryAppendConsumer(
    HloInstruction* candidate,
    const absl::flat_hash_map<int, HloInstruction*>& eligible_stages,
    const HloReachabilityMap& reachability, ChainState& chain) {
  if (!eligible_stages.contains(candidate->unique_id()) ||
      chain.stage_set.contains(candidate)) {
    return false;
  }

  const HloInstruction* prev = chain.stages.back();
  for (const HloInstruction* op : candidate->operands()) {
    if (op == prev) {
      continue;
    }
    // Reject skip-phase dependencies (`candidate` reading `stages[0..k-2]`) and
    // indirect paths (`stages[j] -> ... -> op -> candidate`) that would form a
    // cycle once `stages` is collapsed into a single fusion instruction.
    if (chain.stage_set.contains(op) ||
        absl::c_any_of(chain.stages, [&](const HloInstruction* stage) {
          return reachability.IsReachable(stage, op);
        })) {
      return false;
    }
  }

  chain.stages.push_back(candidate);
  chain.stage_set.insert(candidate);
  return true;
}

// Wraps `chain` into a `__phased_block_cluster` custom fusion, rewires external
// uses via `get-tuple-element`, and removes the original stage instructions.
absl::Status CreateMultiPhaseFusion(HloModule* module, HloComputation* comp,
                                    absl::Span<HloInstruction* const> chain) {
  HloComputation::Builder builder("phased_block_cluster_computation");
  std::vector<HloInstruction*> outer_operands;
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> param_map;
  for (size_t i = 0; i < chain.size(); ++i) {
    for (HloInstruction* op : chain[i]->operands()) {
      if (i > 0 && op == chain[i - 1]) {
        continue;
      }
      auto [it, inserted] = param_map.try_emplace(op, nullptr);
      if (inserted) {
        int64_t param_idx = static_cast<int64_t>(outer_operands.size());
        it->second = builder.AddInstruction(HloInstruction::CreateParameter(
            param_idx, op->shape(), absl::StrCat("param_", param_idx)));
        outer_operands.push_back(op);
      }
    }
  }

  std::vector<HloInstruction*> nested_fusions;
  nested_fusions.reserve(chain.size());
  for (size_t i = 0; i < chain.size(); ++i) {
    HloInstruction* stage = chain[i];
    auto map_operand = [&](HloInstruction* op) -> HloInstruction* {
      return (i > 0 && op == chain[i - 1]) ? nested_fusions[i - 1]
                                           : param_map.at(op);
    };

    HloComputation* stage_comp = nullptr;
    std::vector<HloInstruction*> stage_operands;
    if (auto* fusion_stage = DynCast<HloFusionInstruction>(stage)) {
      stage_operands.reserve(fusion_stage->operand_count());
      for (HloInstruction* op : fusion_stage->operands()) {
        stage_operands.push_back(map_operand(op));
      }
      stage_comp = module->AddEmbeddedComputation(
          fusion_stage->fused_instructions_computation()->Clone());
    } else {
      HloComputation::Builder stage_builder(
          absl::StrCat(stage->name(), "_computation"));
      absl::flat_hash_map<const HloInstruction*, HloInstruction*>
          inner_param_map;
      std::vector<HloInstruction*> inner_operands;
      inner_operands.reserve(stage->operand_count());
      for (HloInstruction* op : stage->operands()) {
        auto [it, inserted] = inner_param_map.try_emplace(op, nullptr);
        if (inserted) {
          int64_t param_idx = static_cast<int64_t>(stage_operands.size());
          it->second =
              stage_builder.AddInstruction(HloInstruction::CreateParameter(
                  param_idx, op->shape(), absl::StrCat("param_", param_idx)));
          stage_operands.push_back(map_operand(op));
        }
        inner_operands.push_back(it->second);
      }
      stage_builder.AddInstruction(
          stage->CloneWithNewOperands(stage->shape(), inner_operands));
      stage_comp = module->AddEmbeddedComputation(stage_builder.Build());
      stage_comp->SetExecutionThread(comp->execution_thread());
    }

    nested_fusions.push_back(
        builder.AddInstruction(HloInstruction::CreateFusion(
            stage->shape(), HloInstruction::FusionKind::kLoop, stage_operands,
            stage_comp)));
  }

  // Preserve all intermediate and final phase outputs in `ROOT tuple(...)` for
  // compatibility with bufferization, because phases exchange data via DRAM
  // buffers.
  HloInstruction* root_tuple =
      builder.AddInstruction(HloInstruction::CreateTuple(nested_fusions));
  HloComputation* outer_comp =
      module->AddEmbeddedComputation(builder.Build(root_tuple));
  outer_comp->SetExecutionThread(comp->execution_thread());
  HloInstruction* outer_fusion = comp->AddInstruction(
      HloInstruction::CreateFusion(outer_comp->root_instruction()->shape(),
                                   HloInstruction::FusionKind::kCustom,
                                   outer_operands, outer_comp),
      "phased_block_cluster");

  GpuBackendConfig gpu_config;
  gpu_config.mutable_fusion_backend_config()->set_kind(
      kPhasedBlockClusterFusionKind);
  ABSL_RETURN_IF_ERROR(outer_fusion->set_backend_config(gpu_config));

  // Rewire external uses and remove fused stages from the tail backward so each
  // stage's in-chain consumer is already detached when checking `user_count()`.
  for (int64_t i = static_cast<int64_t>(chain.size()) - 1; i >= 0; --i) {
    HloInstruction* stage = chain[i];
    if (stage->user_count() > 0 || comp->root_instruction() == stage) {
      HloInstruction* gte =
          comp->AddInstruction(HloInstruction::CreateGetTupleElement(
              stage->shape(), outer_fusion, i));
      ABSL_RETURN_IF_ERROR(stage->ReplaceAllUsesWith(gte));
    }
    ABSL_RETURN_IF_ERROR(comp->RemoveInstruction(stage));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> MultiPhaseFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  FusionWrapper fusion_wrapper(device_info_);
  bool changed = false;
  for (HloComputation* comp :
       GetFusibleComputations(*module, execution_threads)) {
    std::vector<int> candidate_ids;
    absl::flat_hash_map<int, HloInstruction*> eligible_stages;
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (IsEligibleStage(instr, fusion_wrapper, device_info_)) {
        candidate_ids.push_back(instr->unique_id());
        eligible_stages[instr->unique_id()] = instr;
      }
    }
    if (eligible_stages.size() < 2) {
      continue;
    }

    std::unique_ptr<HloReachabilityMap> reachability =
        HloReachabilityMap::Build(comp);

    for (int head_id : candidate_ids) {
      auto head_it = eligible_stages.find(head_id);
      if (head_it == eligible_stages.end()) {
        continue;
      }
      HloInstruction* head = head_it->second;

      ChainState chain{/*stages=*/{head}, /*stage_set=*/{head}};

      bool extended = true;
      while (extended) {
        extended = false;
        for (HloInstruction* user : chain.stages.back()->users()) {
          if (TryAppendConsumer(user, eligible_stages, *reachability, chain)) {
            extended = true;
            break;
          }
        }
      }

      if (chain.stages.size() < 2) {
        continue;
      }
      for (HloInstruction* stage : chain.stages) {
        eligible_stages.erase(stage->unique_id());
      }
      ABSL_RETURN_IF_ERROR(CreateMultiPhaseFusion(module, comp, chain.stages));
      changed = true;
      if (eligible_stages.size() < 2) {
        break;
      }
      reachability = HloReachabilityMap::Build(comp);
    }
  }

  if (changed) {
    ABSL_RETURN_IF_ERROR(module->RemoveUnusedComputations());
    if (module->has_schedule()) {
      ABSL_RETURN_IF_ERROR(module->schedule().Update());
    }
  }
  return changed;
}

}  // namespace xla::gpu

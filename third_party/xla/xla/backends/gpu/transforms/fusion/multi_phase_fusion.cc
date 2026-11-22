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

#include "xla/backends/gpu/transforms/fusion/multi_phase_fusion.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <queue>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SetVector.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/codegen/thread_block_specialized.h"
#include "xla/backends/gpu/transforms/fusion/fusion_wrapper.h"
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
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

// Maximum total kernel arguments (deduplicated outer operands + intermediate
// and final stage outputs) to bound compile time and stay well within PTX's
// parameter space limit.
constexpr int64_t kMaxKernelArgs = 256;

// Tracks the stages, deduplicated external operands, and root-tuple output
// count (both intermediate phase buffers and final outputs) of a candidate
// multi-phase fusion DAG under construction.
struct PhaseDag {
  llvm::SetVector<HloInstruction*> stages;
  llvm::SetVector<HloInstruction*> outer_operands;
  int64_t total_outputs = 0;
};

// Whether `shape` is an array or a non-empty flat tuple of arrays.
bool HasValidStageOutputShape(const Shape& shape) {
  return shape.IsArray() ||
         (shape.IsTuple() && !shape.tuple_shapes().empty() &&
          absl::c_all_of(shape.tuple_shapes(),
                         [](const Shape& sub) { return sub.IsArray(); }));
}

// Unwraps a `get-tuple-element` projection to its tuple producer, or returns
// `instr` unchanged.
HloInstruction* MaybeUnwrapGetTupleElement(HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kGetTupleElement
             ? instr->mutable_operand(0)
             : instr;
}

// Returns the stage producing `operand`, looking through `get-tuple-element`
// ops, if needed. Returns nullptr if no such stage exists in `stages`.
HloInstruction* FindProducerStage(
    HloInstruction* operand, const llvm::SetVector<HloInstruction*>& stages) {
  HloInstruction* prod = MaybeUnwrapGetTupleElement(operand);
  return stages.contains(prod) ? prod : nullptr;
}

// Whether `instr` is a native MLIR fusion (`kLoop`, `kReduction`,
// `kTranspose`, `kConcatenate`, or an unfused op that `FusionWrapper` would
// wrap as one) without side effects, aliasing, or control dependencies whose
// grid fits in one cluster.
bool IsEligibleStage(const HloInstruction* instr, FusionWrapper& fusion_wrapper,
                     const se::DeviceDescription& device_info) {
  if (!HasValidStageOutputShape(instr->shape()) || instr->IsDead() ||
      instr->HasSideEffect() || instr->HasControlDependencies()) {
    return false;
  }
  if (const auto* fusion = DynCast<HloFusionInstruction>(instr)) {
    if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom ||
        !fusion->output_to_operand_aliasing().empty()) {
      return false;
    }
  } else if (!instr->IsFusible() ||
             !fusion_wrapper.MustWrapInstruction(*instr)) {
    return false;
  }

  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*instr, device_info);
  PreBufferAssignmentFusionInfo fusion_info(analysis);
  if (fusion_info.GetCopyFusion().has_value() ||
      (IsDynamicUpdateSliceFusion(analysis.fusion_spec()) &&
       fusion_info.CanEmitDynamicUpdateSliceInPlace())) {
    return false;
  }

  bool unroll_small_fusions =
      device_info.cuda_compute_capability().IsAtLeastHopper();
  std::unique_ptr<MlirKernelEmitter> emitter =
      ThreadBlockSpecializedFusion::CreateNativeNestedEmitter(
          analysis, /*unroll_small_fusions=*/unroll_small_fusions);
  if (emitter == nullptr) {
    return false;
  }
  LaunchDimensions dims = emitter->launch_dimensions();
  if (dims.num_blocks() == 0 ||
      dims.num_blocks() > ThreadBlockSpecializedFusion::kMaxBlocksPerCluster) {
    return false;
  }
  // Sub-kernels with `num_threads < max_threads` that emit CTA barriers
  // (`bar.sync 0, T_i`) or full-warp shuffles require a warp-multiple thread
  // count; `LoopFusion` and `ConcatenateFusion` use neither.
  if (dims.num_threads_per_block() % WarpSize(device_info) != 0 &&
      analysis.emitter_fusion_kind() !=
          HloFusionAnalysis::EmitterFusionKind::kLoop &&
      analysis.emitter_fusion_kind() !=
          HloFusionAnalysis::EmitterFusionKind::kConcatenate) {
    return false;
  }
  return true;
}

// Adds `candidate` to `dag` if it is a new eligible stage that keeps the fused
// kernel within `kMaxKernelArgs` and introduces no external dataflow cycles
// once `dag` is collapsed into a single fusion.
bool TryAddStage(
    HloInstruction* candidate,
    const absl::flat_hash_map<int, HloInstruction*>& eligible_stages,
    const HloReachabilityMap& reachability, PhaseDag& dag) {
  if (!eligible_stages.contains(candidate->unique_id()) ||
      dag.stages.contains(candidate)) {
    return false;
  }

  // Every stage's outputs (including intermediate phase buffers) become
  // elements of the outer fusion's root tuple and thus separate PTX `.entry`
  // parameters; exit early on wide multi-output stages before scanning operands
  // and users.
  const int64_t new_outputs =
      dag.total_outputs + ShapeUtil::GetLeafCount(candidate->shape());
  if (new_outputs > kMaxKernelArgs) {
    return false;
  }

  int64_t internalized_operands = 0;
  if (candidate->shape().IsTuple()) {
    for (HloInstruction* user : candidate->users()) {
      if (user->opcode() == HloOpcode::kGetTupleElement &&
          dag.outer_operands.contains(user)) {
        ++internalized_operands;
      }
    }
  } else if (dag.outer_operands.contains(candidate)) {
    internalized_operands = 1;
  }

  llvm::SmallSetVector<HloInstruction*, 8> added_outer_operands;
  for (HloInstruction* operand : candidate->operands()) {
    if (FindProducerStage(operand, dag.stages) == nullptr &&
        !dag.outer_operands.contains(operand)) {
      added_outer_operands.insert(operand);
    }
  }
  const int64_t new_outer_operand_count =
      static_cast<int64_t>(dag.outer_operands.size()) - internalized_operands +
      static_cast<int64_t>(added_outer_operands.size());
  if (new_outer_operand_count + new_outputs > kMaxKernelArgs) {
    return false;
  }

  // Reject indirect paths through external instructions
  // (`stage -> ... -> operand -> candidate` or
  // `candidate -> ... -> operand -> stage`).
  for (HloInstruction* operand : added_outer_operands) {
    if (absl::c_any_of(dag.stages, [&](const HloInstruction* stage) {
          return reachability.IsReachable(stage, operand);
        })) {
      return false;
    }
  }
  for (HloInstruction* operand : dag.outer_operands) {
    if (MaybeUnwrapGetTupleElement(operand) != candidate &&
        reachability.IsReachable(candidate, operand)) {
      return false;
    }
  }

  if (candidate->shape().IsTuple()) {
    for (HloInstruction* user : candidate->users()) {
      if (user->opcode() == HloOpcode::kGetTupleElement) {
        dag.outer_operands.remove(user);
      }
    }
  } else {
    dag.outer_operands.remove(candidate);
  }
  dag.outer_operands.insert(added_outer_operands.begin(),
                            added_outer_operands.end());
  dag.stages.insert(candidate);
  dag.total_outputs = new_outputs;
  return true;
}

// Greedily grows a connected DAG of eligible stages around `seed` in both
// directions (producers and consumers) via BFS.
PhaseDag BuildPhaseDag(
    HloInstruction* seed,
    const absl::flat_hash_map<int, HloInstruction*>& eligible_stages,
    const HloReachabilityMap& reachability) {
  PhaseDag dag;
  std::queue<HloInstruction*> worklist;
  auto try_add_stage_and_enqueue = [&](HloInstruction* candidate) {
    if (TryAddStage(candidate, eligible_stages, reachability, dag)) {
      worklist.push(candidate);
    }
  };

  try_add_stage_and_enqueue(seed);
  while (!worklist.empty()) {
    HloInstruction* stage = worklist.front();
    worklist.pop();

    for (HloInstruction* operand : stage->operands()) {
      try_add_stage_and_enqueue(MaybeUnwrapGetTupleElement(operand));
    }

    if (stage->shape().IsTuple()) {
      for (HloInstruction* user : stage->users()) {
        if (user->opcode() == HloOpcode::kGetTupleElement) {
          for (HloInstruction* gte_user : user->users()) {
            try_add_stage_and_enqueue(gte_user);
          }
        }
      }
    } else {
      for (HloInstruction* user : stage->users()) {
        try_add_stage_and_enqueue(user);
      }
    }
  }
  return dag;
}

// Wraps `sorted_stages` (ordered topologically) into a `__phased_block_cluster`
// custom fusion, rewires external uses via `get-tuple-element`, and removes the
// original stage instructions.
absl::Status CreateMultiPhaseFusion(
    HloModule* module, HloComputation* comp,
    const std::vector<HloInstruction*>& sorted_stages, const PhaseDag& dag,
    FusionWrapper& fusion_wrapper) {
  HloComputation::Builder builder("phased_block_cluster_computation");
  std::vector<HloInstruction*> outer_operands;
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> param_map;
  for (HloInstruction* stage : sorted_stages) {
    for (HloInstruction* operand : stage->operands()) {
      // Skip operands that have a producer stage inside the dag.
      if (FindProducerStage(operand, dag.stages) != nullptr) {
        continue;
      }
      auto [it, inserted] = param_map.try_emplace(operand, nullptr);
      if (inserted) {
        int64_t param_idx = static_cast<int64_t>(outer_operands.size());
        it->second = builder.AddInstruction(HloInstruction::CreateParameter(
            param_idx, operand->shape(), absl::StrCat("param_", param_idx)));
        outer_operands.push_back(operand);
      }
    }
  }

  absl::flat_hash_map<const HloInstruction*, std::vector<HloInstruction*>>
      stage_outputs;
  stage_outputs.reserve(sorted_stages.size());
  std::vector<int64_t> stage_output_offsets(sorted_stages.size(), 0);
  std::vector<HloInstruction*> tuple_elements;
  tuple_elements.reserve(dag.total_outputs);
  for (size_t i = 0; i < sorted_stages.size(); ++i) {
    HloInstruction* stage = sorted_stages[i];
    auto map_operand = [&](HloInstruction* operand) -> HloInstruction* {
      if (const HloInstruction* prod = FindProducerStage(operand, dag.stages)) {
        const std::vector<HloInstruction*>& outputs = stage_outputs.at(prod);
        return operand == prod ? outputs[0] : outputs[operand->tuple_index()];
      }
      return param_map.at(operand);
    };

    HloInstruction* nested_fusion = nullptr;
    if (auto* fusion_stage = DynCast<HloFusionInstruction>(stage)) {
      std::vector<HloInstruction*> stage_operands;
      stage_operands.reserve(fusion_stage->operand_count());
      for (HloInstruction* operand : fusion_stage->operands()) {
        stage_operands.push_back(map_operand(operand));
      }
      nested_fusion = builder.AddInstruction(fusion_stage->CloneWithNewOperands(
          fusion_stage->shape(), stage_operands, /*suffix=*/""));
    } else {
      HloComputation::Builder stage_builder(
          absl::StrCat(stage->name(), "_computation"));
      absl::flat_hash_map<const HloInstruction*, HloInstruction*>
          inner_param_map;
      std::vector<HloInstruction*> stage_operands;
      std::vector<HloInstruction*> inner_operands;
      inner_operands.reserve(stage->operand_count());
      for (HloInstruction* operand : stage->operands()) {
        HloInstruction* mapped_operand = map_operand(operand);
        auto [it, inserted] =
            inner_param_map.try_emplace(mapped_operand, nullptr);
        if (inserted) {
          int64_t param_idx = static_cast<int64_t>(stage_operands.size());
          it->second =
              stage_builder.AddInstruction(HloInstruction::CreateParameter(
                  param_idx, mapped_operand->shape(),
                  absl::StrCat("param_", param_idx)));
          stage_operands.push_back(mapped_operand);
        }
        inner_operands.push_back(it->second);
      }
      stage_builder.AddInstruction(
          stage->CloneWithNewOperands(stage->shape(), inner_operands));
      HloComputation* stage_comp =
          module->AddEmbeddedComputation(stage_builder.Build());
      stage_comp->SetExecutionThread(comp->execution_thread());
      // `ChooseFusionKind` takes `(producer, consumer)`; pass `*stage` for both
      // when wrapping a standalone instruction, matching `FusionWrapperBase`.
      nested_fusion = builder.AddInstruction(HloInstruction::CreateFusion(
          stage->shape(), fusion_wrapper.ChooseFusionKind(*stage, *stage),
          stage_operands, stage_comp));
    }

    stage_output_offsets[i] = static_cast<int64_t>(tuple_elements.size());
    std::vector<HloInstruction*>& outputs = stage_outputs[stage];
    if (stage->shape().IsTuple()) {
      for (int64_t k = 0; k < stage->shape().tuple_shapes_size(); ++k) {
        outputs.push_back(builder.AddInstruction(
            HloInstruction::CreateGetTupleElement(nested_fusion, k)));
      }
    } else {
      outputs.push_back(nested_fusion);
    }
    tuple_elements.insert(tuple_elements.end(), outputs.begin(), outputs.end());
  }

  // Preserve all intermediate and final phase outputs as flat arrays in
  // `ROOT tuple(...)` for compatibility with bufferization and
  // `ThreadBlockSpecializedFusion`, because phases exchange data via DRAM
  // buffers.
  HloInstruction* root_tuple =
      builder.AddInstruction(HloInstruction::CreateTuple(tuple_elements));
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

  // Rewire external uses and remove fused stages in reverse topological order
  // so each stage's internal consumers are already detached when checking
  // `user_count()`.
  for (int64_t i = static_cast<int64_t>(sorted_stages.size()) - 1; i >= 0;
       --i) {
    HloInstruction* stage = sorted_stages[i];
    int64_t output_idx = stage_output_offsets[i];
    if (stage->shape().IsTuple()) {
      for (HloInstruction* user : std::vector<HloInstruction*>(
               stage->users().begin(), stage->users().end())) {
        if (user->opcode() != HloOpcode::kGetTupleElement) {
          continue;
        }
        user->set_tuple_index(output_idx + user->tuple_index());
        ABSL_RETURN_IF_ERROR(
            stage->ReplaceUseWithDifferentShape(user, outer_fusion));
      }
      if (stage->user_count() > 0 || comp->root_instruction() == stage) {
        int64_t num_outputs = stage->shape().tuple_shapes().size();
        std::vector<HloInstruction*> new_gtes;
        new_gtes.reserve(num_outputs);
        for (int64_t k = 0; k < num_outputs; ++k) {
          new_gtes.push_back(
              comp->AddInstruction(HloInstruction::CreateGetTupleElement(
                  outer_fusion, output_idx + k)));
        }
        HloInstruction* new_tuple =
            comp->AddInstruction(HloInstruction::CreateTuple(new_gtes));
        ABSL_RETURN_IF_ERROR(stage->ReplaceAllUsesWith(new_tuple));
      }
    } else if (stage->user_count() > 0 || comp->root_instruction() == stage) {
      HloInstruction* gte =
          comp->AddInstruction(HloInstruction::CreateGetTupleElement(
              stage->shape(), outer_fusion, output_idx));
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

    for (int seed_id : candidate_ids) {
      auto seed_it = eligible_stages.find(seed_id);
      if (seed_it == eligible_stages.end()) {
        continue;
      }

      PhaseDag dag =
          BuildPhaseDag(seed_it->second, eligible_stages, *reachability);
      if (dag.stages.size() < 2) {
        continue;
      }
      // Sort stages topologically to create a nested fusion.
      std::vector<HloInstruction*> sorted_stages(dag.stages.begin(),
                                                 dag.stages.end());
      absl::c_stable_sort(
          sorted_stages, [&](const HloInstruction* a, const HloInstruction* b) {
            return reachability->GetIndex(a) < reachability->GetIndex(b);
          });
      for (HloInstruction* stage : sorted_stages) {
        eligible_stages.erase(stage->unique_id());
      }
      ABSL_RETURN_IF_ERROR(CreateMultiPhaseFusion(module, comp, sorted_stages, dag,
                                             fusion_wrapper));
      changed = true;
      if (eligible_stages.size() < 2) {
        break;
      }
      // TODO(tjoerg): Update `reachability` incrementally instead of rebuilding
      // it from scratch after every formed fusion.
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

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

#include "xla/service/cpu/small_region_hoisting_pass.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla::cpu {
namespace {

// An instruction inherently needs runtime services and cannot live inside a
// single-kernel region. These are the region boundaries.
bool InstructionIsUnavailable(const HloInstruction* instr) {
  auto backend_config_or = instr->backend_config<BackendConfig>();
  if (backend_config_or.ok() &&
      !backend_config_or->outer_dimension_partitions().empty()) {
    return true;
  }
  switch (instr->opcode()) {
    case HloOpcode::kCustomCall:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kScatter:
    case HloOpcode::kSort:
    case HloOpcode::kFft:
    case HloOpcode::kPartitionId:
    case HloOpcode::kReplicaId:
      return true;
    // Legacy call emitter does not support custom fusions (library fusions).
    case HloOpcode::kFusion:
      return instr->fusion_kind() == HloInstruction::FusionKind::kCustom;
    default:
      return IsCollective(instr);
  }
}

// True if `instr` or any instruction reachable through its called computations
// is unavailable. Memoized across the pass.
bool ContainsUnavailableInstruction(
    const HloInstruction* instr,
    absl::flat_hash_map<const HloInstruction*, bool>& memo) {
  if (auto it = memo.find(instr); it != memo.end()) {
    return it->second;
  }
  if (InstructionIsUnavailable(instr)) {
    return memo[instr] = true;
  }
  for (const HloComputation* called : instr->called_computations()) {
    for (const HloInstruction* sub : called->instructions()) {
      if (ContainsUnavailableInstruction(sub, memo)) {
        return memo[instr] = true;
      }
    }
  }
  return memo[instr] = false;
}

bool IsControlFlow(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kWhile ||
         instr->opcode() == HloOpcode::kConditional;
}

// A maximal schedule-contiguous run of region-eligible instructions.
struct Region {
  std::vector<HloInstruction*> members;  // in topological order
};

}  // namespace

SmallRegionHoistingPass::SmallRegionHoistingPass(
    int64_t small_buffer_access_size, int64_t min_region_size,
    int64_t max_instruction_count, int64_t max_regions_limit)
    : small_buffer_access_size_(small_buffer_access_size),
      min_region_size_(min_region_size),
      max_instruction_count_(max_instruction_count),
      max_regions_limit_(max_regions_limit) {}

// Partitions `comp`'s schedule into maximal region-eligible runs and outlines
// each cost-model-passing run into an `xla_cpu_small_call` kCall. Returns true
// if `comp` was modified. `unavailable_memo` is shared across computations.
// Identifies cost-model-passing regions in `comp` that are eligible for
// hoisting.
static absl::StatusOr<std::vector<Region>> GetCandidateRegions(
    HloComputation* comp, int64_t small_buffer_access_size,
    int64_t min_region_size,
    absl::flat_hash_map<const HloInstruction*, bool>& unavailable_memo) {
  // Partition the computation's topological order into maximal runs of
  // region-eligible instructions, split at unavailable instructions.
  // Parameters are region inputs, not members, and do not break a run.
  std::vector<Region> regions;
  Region current;
  auto close_region = [&]() {
    if (!current.members.empty()) {
      regions.push_back(std::move(current));
      current = Region{};
    }
  };
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kParameter) {
      continue;
    }
    if (ContainsUnavailableInstruction(instr, unavailable_memo)) {
      close_region();
      continue;
    }
    current.members.push_back(instr);
  }
  close_region();

  // Split straight-line regions that exceed the size cap (design doc §2.3).
  // Split into roughly equal chunks to avoid sub-regions being discarded by the
  // minimum size floor (min_region_size).
  std::vector<Region> split_regions;
  constexpr int64_t kMaxStraightLineRegionSize = 48;
  for (auto& region : regions) {
    bool contains_control_flow = absl::c_any_of(region.members, IsControlFlow);
    if (!contains_control_flow &&
        region.members.size() > kMaxStraightLineRegionSize) {
      int64_t num_members = region.members.size();
      int64_t num_chunks = (num_members + kMaxStraightLineRegionSize - 1) /
                           kMaxStraightLineRegionSize;
      int64_t chunk_size = (num_members + num_chunks - 1) / num_chunks;
      for (size_t i = 0; i < region.members.size(); i += chunk_size) {
        size_t end = std::min(i + chunk_size, region.members.size());
        Region sub_region;
        sub_region.members.assign(region.members.begin() + i,
                                  region.members.begin() + end);
        split_regions.push_back(std::move(sub_region));
      }
    } else {
      split_regions.push_back(std::move(region));
    }
  }
  regions = std::move(split_regions);

  std::vector<Region> candidates;
  std::vector<Region> salvaged_regions;
  for (const Region& region : regions) {
    absl::flat_hash_set<const HloInstruction*> member_set(
        region.members.begin(), region.members.end());

    // Cost-model gate: a region is worth a single kernel when it is small and
    // either has enough instructions to beat per-op dispatch, or contains
    // control flow (whose dispatch cost scales with trip count regardless of
    // static instruction count).
    bool contains_control_flow = absl::c_any_of(region.members, IsControlFlow);
    if (region.members.size() < static_cast<size_t>(min_region_size) &&
        !contains_control_flow) {
      continue;
    }

    HloCostAnalysis cost_analysis(&CpuExecutable::ShapeSizeBytes);
    int64_t bytes_accessed = 0;
    for (HloInstruction* member : region.members) {
      RETURN_IF_ERROR(cost_analysis.RevisitInstruction(member));
      bytes_accessed += cost_analysis.bytes_accessed(*member);
    }
    if (bytes_accessed >= small_buffer_access_size) {
      if (contains_control_flow) {
        for (HloInstruction* member : region.members) {
          if (IsControlFlow(member)) {
            int64_t member_bytes = cost_analysis.bytes_accessed(*member);
            if (member_bytes < small_buffer_access_size) {
              Region salvaged;
              salvaged.members.push_back(member);
              salvaged_regions.push_back(std::move(salvaged));
            }
          }
        }
      }
      continue;
    }

    // Conservative correctness guard: don't hoist a region whose members carry
    // control dependencies that CROSS the region boundary — outlining would
    // drop the ordering they encode. Side-effecting ops (infeed/outfeed/
    // custom-call/collectives) are already region boundaries, so such crossing
    // control edges are rare in practice; skipping these regions is safe and
    // costs only the status quo. Control deps internal to the region
    // (member<->member) are fine: the region collapses into one kernel and that
    // ordering is subsumed by data dependencies inside it.
    auto crosses_boundary = [&](const HloInstruction* other) {
      return !member_set.contains(other);
    };
    bool has_boundary_control_dep =
        absl::c_any_of(region.members, [&](const HloInstruction* member) {
          return absl::c_any_of(member->control_predecessors(),
                                crosses_boundary) ||
                 absl::c_any_of(member->control_successors(), crosses_boundary);
        });
    if (has_boundary_control_dep) {
      continue;
    }

    std::vector<HloInstruction*> outputs;
    for (HloInstruction* member : region.members) {
      bool used_outside =
          member == comp->root_instruction() ||
          absl::c_any_of(member->users(), [&](const HloInstruction* user) {
            return !member_set.contains(user);
          });
      if (used_outside) {
        outputs.push_back(member);
      }
    }
    // A region whose only effect is its root must have at least one output.
    if (outputs.empty()) {
      continue;
    }

    candidates.push_back(region);
  }

  // Process salvaged regions.
  for (const Region& region : salvaged_regions) {
    HloInstruction* member = region.members[0];
    bool has_boundary_control_dep = !member->control_predecessors().empty() ||
                                    !member->control_successors().empty();
    if (has_boundary_control_dep) {
      continue;
    }
    if (member->user_count() == 0 && member != comp->root_instruction()) {
      continue;
    }
    candidates.push_back(region);
  }

  return candidates;
}

// Outlines a region in `comp` into a separate function, tagged
// `xla_cpu_small_call`.
static absl::Status HoistRegion(HloModule* module, HloComputation* comp,
                                const Region& region) {
  absl::flat_hash_set<const HloInstruction*> member_set(region.members.begin(),
                                                        region.members.end());

  // Liveness boundary. Inputs: values used by the region but defined outside
  // it (computation parameters or earlier instructions). Outputs: region
  // members used outside the region, or the computation root.
  std::vector<HloInstruction*> inputs;
  absl::flat_hash_set<const HloInstruction*> input_set;
  for (HloInstruction* member : region.members) {
    for (HloInstruction* operand : member->operands()) {
      if (!member_set.contains(operand) && input_set.insert(operand).second) {
        inputs.push_back(operand);
      }
    }
  }
  std::vector<HloInstruction*> outputs;
  for (HloInstruction* member : region.members) {
    bool used_outside =
        member == comp->root_instruction() ||
        absl::c_any_of(member->users(), [&](const HloInstruction* user) {
          return !member_set.contains(user);
        });
    if (used_outside) {
      outputs.push_back(member);
    }
  }
  // A region whose only effect is its root must have at least one output.
  if (outputs.empty()) {
    return absl::InternalError("Region has no outputs during hoisting");
  }

  // Build the outlined computation: a parameter per input, a clone per
  // member, rooted at the single output or a tuple of outputs.
  HloComputation::Builder builder(
      absl::StrCat(region.members.back()->name(), "_region"));
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> map;
  for (int64_t i = 0; i < inputs.size(); ++i) {
    map[inputs[i]] = builder.AddInstruction(HloInstruction::CreateParameter(
        i, inputs[i]->shape(), absl::StrCat("p", i)));
  }
  for (HloInstruction* member : region.members) {
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(member->operand_count());
    for (HloInstruction* operand : member->operands()) {
      new_operands.push_back(map.at(operand));
    }
    map[member] = builder.AddInstruction(
        member->CloneWithNewOperands(member->shape(), new_operands));
  }
  HloInstruction* outlined_root;
  Shape call_shape;
  if (outputs.size() == 1) {
    outlined_root = map.at(outputs[0]);
    call_shape = outputs[0]->shape();
  } else {
    std::vector<HloInstruction*> output_clones;
    output_clones.reserve(outputs.size());
    std::vector<Shape> output_shapes;
    output_shapes.reserve(outputs.size());
    for (HloInstruction* output : outputs) {
      output_clones.push_back(map.at(output));
      output_shapes.push_back(output->shape());
    }
    outlined_root =
        builder.AddInstruction(HloInstruction::CreateTuple(output_clones));
    call_shape = ShapeUtil::MakeTupleShape(output_shapes);
  }
  HloComputation* outlined =
      module->AddEmbeddedComputation(builder.Build(outlined_root));

  HloInstruction* call = comp->AddInstruction(
      HloInstruction::CreateCall(call_shape, inputs, outlined));
  call->add_frontend_attribute("xla_cpu_small_call", "true");

  // Redirect external uses to the call (or per-output get-tuple-elements).
  if (outputs.size() == 1) {
    RETURN_IF_ERROR(outputs[0]->ReplaceAllUsesWith(call));
  } else {
    for (int64_t i = 0; i < outputs.size(); ++i) {
      HloInstruction* gte = comp->AddInstruction(
          HloInstruction::CreateGetTupleElement(outputs[i]->shape(), call, i));
      RETURN_IF_ERROR(outputs[i]->ReplaceAllUsesWith(gte));
    }
  }

  // Remove the now-dead members in reverse topological order.
  for (auto it = region.members.rbegin(); it != region.members.rend(); ++it) {
    RETURN_IF_ERROR((*it)->SafelyDropAllControlDependencies());
    RETURN_IF_ERROR(comp->RemoveInstruction(*it));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> SmallRegionHoistingPass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Gate 1: Check total instruction count of the module.
  if (module->instruction_count() > max_instruction_count_) {
    return false;
  }

  HloComputation* entry = module->entry_computation();
  if (entry == nullptr) {
    return false;
  }

  struct Candidate {
    HloComputation* comp;
    Region region;
  };
  std::vector<Candidate> all_candidates;

  // Process the entry, then descend into control-flow bodies. We enqueue a
  // body only AFTER partitioning the computation that contains its
  // while/conditional op: if that op was swallowed into a hoisted `_region`,
  // it is gone from the computation and its body is not enqueued (we do not
  // partition the called computations of an already-hoisted region). The
  // `_region` computations we create are never enqueued because we only
  // descend through control-flow ops, never through the small_call kCall.
  std::vector<HloComputation*> worklist = {entry};
  absl::flat_hash_set<HloComputation*> seen = {entry};

  absl::flat_hash_map<const HloInstruction*, bool> unavailable_memo;
  for (size_t i = 0; i < worklist.size(); ++i) {
    HloComputation* comp = worklist[i];
    ASSIGN_OR_RETURN(std::vector<Region> candidates,
                     GetCandidateRegions(comp, small_buffer_access_size_,
                                         min_region_size_, unavailable_memo));

    // We only descend into control flow bodies of ops that are NOT inside a
    // candidate region.
    absl::flat_hash_set<const HloInstruction*> hoisted_instructions;
    for (const auto& candidate : candidates) {
      hoisted_instructions.insert(candidate.members.begin(),
                                  candidate.members.end());
    }

    for (HloInstruction* instr : comp->instructions()) {
      if (hoisted_instructions.contains(instr)) {
        continue;
      }
      auto enqueue = [&](HloComputation* c) {
        if (seen.insert(c).second) {
          worklist.push_back(c);
        }
      };
      if (instr->opcode() == HloOpcode::kWhile) {
        enqueue(instr->while_body());
        enqueue(instr->while_condition());
      } else if (instr->opcode() == HloOpcode::kConditional) {
        for (HloComputation* branch : instr->branch_computations()) {
          enqueue(branch);
        }
      }
    }

    for (auto& candidate : candidates) {
      all_candidates.push_back({comp, std::move(candidate)});
    }

    // Gate 2: Capping the total candidate count across the module.
    if (all_candidates.size() > max_regions_limit_) {
      return false;
    }
  }

  if (all_candidates.empty()) {
    return false;
  }

  for (const auto& candidate : all_candidates) {
    RETURN_IF_ERROR(HoistRegion(module, candidate.comp, candidate.region));
  }

  return true;
}

}  // namespace xla::cpu

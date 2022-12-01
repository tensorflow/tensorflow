/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.h"

#include <deque>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"

namespace xla {
namespace gpu {

namespace {


bool ShouldScheduleAsEarlyAsPossible(const HloInstruction& instr) {
  switch (instr.opcode()) {
    case HloOpcode::kAllReduceStart:
      return true;
    case HloOpcode::kCustomCall:
      return static_cast<const HloCustomCallInstruction&>(instr)
                 .custom_call_schedule() ==
             CustomCallSchedule::SCHEDULE_EARLIEST;
    default:
      return false;
  }
}

bool ShouldScheduleSuccessor(const HloInstruction& sussessor,
                             const HloPredicate& is_scheduled) {
  return ShouldScheduleAsEarlyAsPossible(sussessor) &&
         absl::c_all_of(sussessor.operands(), is_scheduled) &&
         absl::c_all_of(sussessor.control_predecessors(), is_scheduled);
}

bool ShouldScheduleAsLateAsPossible(const HloInstruction& instr) {
  switch (instr.opcode()) {
    case HloOpcode::kAllReduceDone:
      return true;
    case HloOpcode::kCustomCall:
      return static_cast<const HloCustomCallInstruction&>(instr)
                 .custom_call_schedule() == CustomCallSchedule::SCHEDULE_LATEST;
    default:
      return false;
  }
}

bool ShouldSchedulePredecessor(const HloInstruction& predecessor,
                               const HloPredicate& is_scheduled) {
  return ShouldScheduleAsLateAsPossible(predecessor) &&
         absl::c_all_of(predecessor.users(), is_scheduled) &&
         absl::c_all_of(predecessor.control_successors(), is_scheduled);
}

// Schedules certain ops as early or late as possible. This supports a
// custom-call use case, where a logical operation is lowered into two HLOs
// (e.g., PerformX and PerformXDone). We utilize this mechanism to either hide
// host latencies between the pair of the custom-calls or more accurately
// identify the def-use relationship of the two calls (typically PerformX is
// scheduled right after all of its producers have been scheduled and
// PerformXDone is scheduled right before its first consumer.)
HloInstructionSequence PostprocessorEagerInstruction(
    const HloInstructionSequence& input) {
  std::vector<HloInstruction*> earliest_scheduled;
  {
    absl::flat_hash_set<HloInstruction*> scheduled;
    auto is_scheduled = [&](const HloInstruction* instr) -> bool {
      return scheduled.contains(instr);
    };
    auto add_to_schedule = [&](HloInstruction* instr) {
      earliest_scheduled.push_back(instr);
      scheduled.insert(instr);
    };
    // Make EARLIEST instructions earlier.
    for (HloInstruction* instr : input.instructions()) {
      if (is_scheduled(instr)) {
        continue;
      }

      add_to_schedule(instr);

      // Schedule any successor that should be scheduled as early as possible if
      // all of its producers and control_predecessors have been scheduled.
      for (HloInstruction* user : instr->users()) {
        if (ShouldScheduleSuccessor(*user, is_scheduled)) {
          add_to_schedule(user);
        }
      }
      for (HloInstruction* successor : instr->control_successors()) {
        if (ShouldScheduleSuccessor(*successor, is_scheduled)) {
          add_to_schedule(successor);
        }
      }
    }
  }

  std::deque<HloInstruction*> latest_scheduled;
  {
    absl::flat_hash_set<HloInstruction*> scheduled;
    auto is_scheduled = [&](const HloInstruction* instr) -> bool {
      return scheduled.contains(instr);
    };
    auto add_to_schedule = [&](HloInstruction* instr) {
      latest_scheduled.push_front(instr);
      scheduled.insert(instr);
    };
    // Make LATEST instructions later.
    for (auto it = earliest_scheduled.rbegin(); it != earliest_scheduled.rend();
         it++) {
      if (is_scheduled(*it)) {
        continue;
      }

      add_to_schedule(*it);

      // Schedule any predecessor that should be scheduled as late as possible
      // if all of its users and control_successors have been scheduled.
      for (HloInstruction* operand : (*it)->operands()) {
        if (ShouldSchedulePredecessor(*operand, is_scheduled)) {
          add_to_schedule(operand);
        }
      }
      for (HloInstruction* predecessor : (*it)->control_predecessors()) {
        if (ShouldSchedulePredecessor(*predecessor, is_scheduled)) {
          add_to_schedule(predecessor);
        }
      }
    }
  }

  HloInstructionSequence result;
  absl::c_for_each(latest_scheduled,
                   [&](HloInstruction* i) { result.push_back(i); });
  return result;
}

std::vector<HloInstruction*> EagerClusterPostprocessorPass(
    const std::vector<HloInstruction*>& input, bool is_fwd) {
  // Mapping of HloInstructions to the index of a "constrained" root instruction
  // (i.e. one that has either the EARLIEST or LATEST modifier).
  using InstrIntMap = absl::flat_hash_map<const HloInstruction*, int32_t>;
  InstrIntMap root_map;

  // Vector of vectors, with each inner vector storing the instructions that are
  // constrained by a constrained root. These instructions will be one of the
  // 'users' or 'ctrl_succ' instructions in the forward pass, and will be one of
  // 'operands', or 'ctrl_pred' in the backward pass.
  std::vector<std::vector<HloInstruction*>> constrained_inst;

  std::vector<HloInstruction*> result;

  // Update the root index value stored in map. Instructions that are
  // present in the map are dependent on one or more constrained instructions
  // (i.e. those with either the EARLIEST or the LATEST modifier). The index
  // value stored in the map is the index of the last root instruction that is
  // encountered during the instruction traversal.
  auto root_map_update = [](InstrIntMap& map, const HloInstruction* instr,
                            int32_t cur_root) {
    // Get the previous index from the map, or zero initialize if the
    // instruction key was not in the map.
    auto& index = map[instr];
    // Store the maximum of the previous and the new value so that the
    // instruction is attached to the constrained root that will be scheduled
    // later.
    index = std::max(index, cur_root);
  };
  // Update the constrained instruction map for dependent instructions
  auto update_dep_instr = [&](InstrIntMap& map, const HloInstruction* instr,
                              int32_t root_index, bool is_fwd) -> void {
    if (is_fwd) {
      // Forward pass: users and control successors must be scheduled AFTER
      // a "root" constrained instruction that has the LATEST modifier.
      absl::c_for_each(instr->users(), [&](const HloInstruction* i) {
        root_map_update(map, i, root_index);
      });
      absl::c_for_each(instr->control_successors(),
                       [&](const HloInstruction* i) {
                         root_map_update(map, i, root_index);
                       });
    } else {
      // Backward pass: operands and control predecessors must be scheduled
      // BEFORE a "root" constrained instruction that has the EARLIEST modifier.
      absl::c_for_each(instr->operands(), [&](const HloInstruction* i) {
        root_map_update(map, i, root_index);
      });
      absl::c_for_each(instr->control_predecessors(),
                       [&](const HloInstruction* i) {
                         root_map_update(map, i, root_index);
                       });
    }
  };

  auto is_constrained_root = [](const HloInstruction* instr,
                                bool is_fwd) -> bool {
    return is_fwd ? ShouldScheduleAsLateAsPossible(*instr)
                  : ShouldScheduleAsEarlyAsPossible(*instr);
  };

  // Walk through the instructions in reverse order. (This will reverse the
  // order of the returned instructon sequence, but With two passes, the
  // result after the second pass will be in the correct order.)
  for (auto it = input.rbegin(); it != input.rend(); ++it) {
    HloInstruction* instr = *it;
    // Check for instructions that are directly constrained (via opcode or
    // custom scheduling preference).
    if (is_constrained_root(instr, is_fwd)) {
      // Update dependent instructions of the root constrained op by adding
      // them to the map.
      update_dep_instr(root_map, instr,
                       static_cast<int32_t>(constrained_inst.size()), is_fwd);
      // Add a new vector to hold instructions that are dependent on this
      // instruction.
      constrained_inst.push_back({instr});
    } else {
      // Look for the instruction in the constrained instruction map.
      auto it = root_map.find(instr);
      if (it != root_map.end()) {
        auto root_index = it->second;
        // Add the instruction to the dependent instruction list for the
        // assocated root instruction.
        constrained_inst[root_index].push_back(instr);
        // Add 'dependent' instructions to the map so that they are
        // collected when encountered in the input instruction sequence.
        update_dep_instr(root_map, instr, root_index, is_fwd);
      } else {
        // The instruction is not a root constrained instruction, and it is not
        // dependent on any constrained instructions. Add this instruction
        // directly to the instruction sequence.
        result.push_back(instr);
      }
    }
  }

  // We have passed through the original instruction sequence. "Independent"
  // instructions have been scheduled, and "dependent" or "constrained"
  // instructions have been collected into vectors associated with a
  // constrained root instruction. We now add these constrained instructions.
  for (auto& inst_vec : constrained_inst) {
    for (auto inst : inst_vec) {
      result.push_back(inst);
    }
  }

  return result;
}

HloInstructionSequence PostprocessorEagerCluster(
    const HloInstructionSequence& input) {
  std::vector<HloInstruction*> bwd_result =
      EagerClusterPostprocessorPass(input.instructions(), false);
  std::vector<HloInstruction*> fwd_result =
      EagerClusterPostprocessorPass(bwd_result, true);
  HloInstructionSequence result;
  absl::c_for_each(fwd_result, [&](HloInstruction* i) { result.push_back(i); });
  return result;
}

HloInstructionSequence PostprocessorNone(const HloInstructionSequence& input) {
  return HloInstructionSequence(input);
}

HloInstructionSequence PostprocessorToScheduleAsEarlyOrLateAsPossible(
    const HloInstructionSequence& input, HloModule* module) {
  const DebugOptions& debug_options = module->config().debug_options();
  VLOG(2) << "HLO Schedule Postprocessor: "
          << DebugOptions::MemorySchedulePostprocessor_Name(
                 debug_options.xla_gpu_mem_sched_postproc());
  switch (debug_options.xla_gpu_mem_sched_postproc()) {
    case DebugOptions::SCHED_POSTPROC_EAGER_INSTRUCTION:
    default:
      return PostprocessorEagerInstruction(input);
    case DebugOptions::SCHED_POSTPROC_EAGER_CLUSTER:
      return PostprocessorEagerCluster(input);
    case DebugOptions::SCHED_POSTPROC_NONE:
      return PostprocessorNone(input);
  }
}

}  // end namespace

StatusOr<HloSchedule> ScheduleGpuModule(const HloModule* module,
                                        int64_t pointer_size) {
  return ScheduleModule(
      module,
      [pointer_size](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), pointer_size);
      },
      ComputationSchedulerToModuleScheduler(
          DefaultMemoryScheduler,
          PostprocessorToScheduleAsEarlyOrLateAsPossible));
}

}  // namespace gpu
}  // namespace xla

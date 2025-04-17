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

#ifndef XLA_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_
#define XLA_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_

#include <cstdint>

#include "absl/status/status.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo_value.h"

namespace xla {
namespace gpu {

inline constexpr int64_t kCollectiveMemorySpaceColor = 1;
inline constexpr int64_t kTempBufferMemorySpaceColor = 2;

// Set memory space to kCollectiveMemorySpaceColor for all allocations used by
// all-reduce, all-gather, and reduce-scatter. This memory space maps to
// collective memory using ncclMemAlloc in the runtime.
inline BufferAssigner::Colorer CollectiveColorer() {
  return [](HloAliasAnalysis* alias_analysis, const HloOrdering&) {
    static const auto* const kSupportedOpcodes =
        new absl::flat_hash_set<HloOpcode>{
            HloOpcode::kAllReduce,
            HloOpcode::kAllReduceStart,
            HloOpcode::kAllReduceDone,
            HloOpcode::kAllGather,
            HloOpcode::kAllGatherStart,
            HloOpcode::kAllGatherDone,
            HloOpcode::kReduceScatter,
            HloOpcode::kCollectivePermute,
            HloOpcode::kCollectivePermuteStart,
            HloOpcode::kCollectivePermuteDone,
            HloOpcode::kAllToAll,
        };
    auto is_collective_memory_instr = [&](const HloInstruction* instr) {
      return kSupportedOpcodes->contains(instr->opcode()) ||
             // opcode or async wrapped opcode is in kSupportedOpcodes.
             ((instr->opcode() == HloOpcode::kAsyncStart ||
               instr->opcode() == HloOpcode::kAsyncDone) &&
              kSupportedOpcodes->contains(instr->async_wrapped_opcode()));
    };
    auto has_collective_memory_in_uses = [&](const HloValue* input_alias) {
      // If any use is a collective instruction, we must color the value to use
      // collective memory space.
      for (auto& use : input_alias->GetUses()) {
        if (is_collective_memory_instr(use.instruction)) {
          return true;
        }
      }
      return false;
    };
    for (HloValue* value : alias_analysis->dataflow_analysis().values()) {
      auto& buffer = alias_analysis->GetBufferContainingValue(*value);
      for (const auto& alias : buffer.values()) {
        if (is_collective_memory_instr(alias->instruction()) ||
            has_collective_memory_in_uses(alias)) {
          value->set_color(kCollectiveMemorySpaceColor);
        }
      }
      if (!value->has_color()) {
        value->set_color(0);
      }
    }
    return absl::OkStatus();
  };
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_MEMORY_SPACE_ASSIGNMENT_H_

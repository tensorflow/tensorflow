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

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_value.h"

namespace xla {
namespace gpu {

inline constexpr int64_t kCollectiveMemorySpaceColor = 1;
inline constexpr int64_t kTempBufferMemorySpaceColor = 2;

// Set memory space to kCollectiveMemorySpaceColor for all allocations used by
// all-reduce, all-gather, and reduce-scatter. This memory space maps to
// collective memory using ncclMemAlloc in the runtime.
inline BufferAssigner::Colorer CollectiveColorer(bool use_user_buffers,
                                                 bool use_nvshmem) {
  return [use_user_buffers, use_nvshmem](HloAliasAnalysis* alias_analysis,
                                         const HloOrdering&) {
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

    auto is_nvshmem_op = [](const HloInstruction* inst) {
      bool is_nvshmem_collective = false;
      if (inst->has_backend_config()) {
        auto gpu_config = inst->backend_config<GpuBackendConfig>();
        if (!gpu_config.ok()) {
          return false;
        }
        const CollectiveBackendConfig& backend_config =
            gpu_config.value().collective_backend_config();
        is_nvshmem_collective =
            backend_config.backend() == CollectiveBackendConfig::NVSHMEM;
      }
      return is_nvshmem_collective;
    };

    auto is_mosaic_gpu_nvshmem_instr = [](const HloInstruction* instr) {
      return instr->opcode() == HloOpcode::kCustomCall &&
             (instr->custom_call_target() == "mosaic_gpu" ||
              instr->custom_call_target() == "mosaic_gpu_v2") &&
             absl::StrContains(instr->raw_backend_config_string(), "nvshmem");
    };
    auto is_collective_memory_instr = [&](const HloInstruction* instr) {
      if (use_user_buffers) {
        return kSupportedOpcodes->contains(instr->opcode()) ||
               // opcode or async wrapped opcode is in kSupportedOpcodes.
               ((instr->opcode() == HloOpcode::kAsyncStart ||
                 instr->opcode() == HloOpcode::kAsyncDone) &&
                kSupportedOpcodes->contains(instr->async_wrapped_opcode()));
      }
      if (use_nvshmem) {
        return is_mosaic_gpu_nvshmem_instr(instr) || is_nvshmem_op(instr);
      }
      return false;
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
      // If the value has a layout with non-default memory space, use the memory
      // space from the layout.
      const HloPosition& defining_position = value->defining_position();
      if (defining_position.shape().has_layout()) {
        auto memory_space = defining_position.shape().layout().memory_space();
        if (memory_space != 0) {
          value->set_color(BufferValue::Color(memory_space));
          continue;
        }
      }

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

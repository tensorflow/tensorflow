/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_memory_space_assignment.h"

#include <cstdint>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_value.h"
#include "xla/status_macros.h"

namespace xla::gpu {

// NOTE: The explicit internal constructor is needed as an explicitly typed
// variable to avoid a method ambiguity error when compiling for CUDA 12.4.
const absl::NoDestructor<absl::flat_hash_set<HloOpcode>>
    kSupportedCollectiveOpcodes(absl::flat_hash_set<HloOpcode>{
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
    });

bool IsNvshmemInstruction(const HloInstruction* inst) {
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
}

bool IsCollectiveMemoryInstruction(const HloInstruction* inst) {
  return kSupportedCollectiveOpcodes->contains(inst->opcode()) ||
         // opcode or async wrapped opcode is in kSupportedCollectiveOpcodes.
         ((inst->opcode() == HloOpcode::kAsyncStart ||
           inst->opcode() == HloOpcode::kAsyncDone) &&
          kSupportedCollectiveOpcodes->contains(inst->async_wrapped_opcode()));
}

bool HasCollectiveMemoryInstruction(const HloValue& input_alias,
                                    bool require_nvshmem = false) {
  // If any use is a collective instruction, we must color the value to use
  // collective memory space.
  for (const HloUse& use : input_alias.GetUses()) {
    if (IsCollectiveMemoryInstruction(use.instruction) &&
        (!require_nvshmem || IsNvshmemInstruction(use.instruction))) {
      return true;
    }
  }
  return IsCollectiveMemoryInstruction(input_alias.instruction()) &&
         (!require_nvshmem || IsNvshmemInstruction(input_alias.instruction()));
}

bool HasMosaicInstruction(const HloValue& input_alias,
                          absl::FunctionRef<bool(HloInstruction&)> predicate) {
  for (const HloUse& use : input_alias.GetUses()) {
    if (predicate(*ABSL_DIE_IF_NULL(use.instruction))) {
      return true;
    }
  }

  return predicate(*ABSL_DIE_IF_NULL(input_alias.instruction()));
}

bool HasMosaicWithNvshmemInstruction(const HloValue& input_alias) {
  return HasMosaicInstruction(input_alias, IsMosaicWithNvshmem);
}

bool HasMosaicWithMultimemInstruction(const HloValue& input_alias) {
  return HasMosaicInstruction(input_alias, IsMosaicWithMultimem);
}

// Set memory space to MemorySpaceColor::kCollective for all allocations used by
// all-reduce, all-gather, and reduce-scatter. This memory space maps to
// collective memory using ncclMemAlloc in the runtime.
absl::Status AssignColors(bool use_collective_memory, bool use_nvshmem,
                          HloAliasAnalysis* alias_analysis) {
  for (HloValue* value : alias_analysis->dataflow_analysis().values()) {
    // If the value has a layout with non-default memory space, use the memory
    // space from the layout.
    const HloPosition& defining_position = value->defining_position();
    if (defining_position.shape().has_layout()) {
      const int64_t memory_space =
          defining_position.shape().layout().memory_space();
      if (memory_space != 0) {
        value->set_color(BufferValue::Color(memory_space));
        continue;
      }
    } else if (defining_position.shape().IsTuple()) {
      // Making sure tuples live in default memory space.
      value->set_color((int)MemorySpaceColor::kDefault);
      continue;
    }

    for (const xla::HloValue* alias :
         alias_analysis->GetBufferContainingValue(*value).values()) {
      TF_RET_CHECK(alias != nullptr);
      // TODO(479768130): Mark only buffers used with multimem instructions
      // instead of marking all buffers.
      if ((HasMosaicWithNvshmemInstruction(*alias) && use_nvshmem) ||
          HasMosaicWithMultimemInstruction(*alias)) {
        // This is a temporary solution until a separate BFC allocator will be
        // added for the symmetric memory space.
        value->set_color((int)MemorySpaceColor::kCollective);
      } else if (((use_collective_memory &&
                   HasCollectiveMemoryInstruction(*alias)) ||
                  (use_nvshmem && HasCollectiveMemoryInstruction(
                                      *alias, /*require_nvshmem=*/true)))) {
        value->set_color((int)MemorySpaceColor::kCollective);
      } else if (!value->has_color()) {
        value->set_color((int)MemorySpaceColor::kDefault);
      }
    }
  }
  return absl::OkStatus();
}

BufferAssigner::Colorer CreateColorer(const DebugOptions& option) {
  // NCCL old registered buffers.
  bool nccl_user_buffers = option.xla_gpu_enable_nccl_user_buffers();
  bool nccl_symmetric_buffers =
      option.xla_gpu_experimental_enable_nccl_symmetric_buffers();
  bool use_nvshmem = option.xla_gpu_experimental_enable_nvshmem();

  bool use_collective_memory = nccl_user_buffers || nccl_symmetric_buffers;

  return [use_collective_memory, use_nvshmem](HloAliasAnalysis* alias_analysis,
                                              const HloOrdering&) {
    return AssignColors(use_collective_memory, use_nvshmem, alias_analysis);
  };
}
}  // namespace xla::gpu

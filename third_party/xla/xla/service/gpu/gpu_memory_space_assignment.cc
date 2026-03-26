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
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/tsl/platform/status_macros.h"

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

absl::StatusOr<MemorySpaceColor> AsMemorySpaceColor(int64_t memory_space) {
  switch (memory_space) {
    case static_cast<int64_t>(MemorySpaceColor::kDefault):
    case static_cast<int64_t>(MemorySpaceColor::kCollective):
    case static_cast<int64_t>(MemorySpaceColor::kTempBuffer):
      return static_cast<MemorySpaceColor>(memory_space);
    default:
      return InvalidArgument(
          "Invalid memory space %d. "
          "Valid values are 0 (default), 1 (collective), 2 (temp).",
          memory_space);
  }
}

absl::StatusOr<std::vector<std::pair<int64_t, MemorySpaceColor>>>
ParseIndexMemorySpacePairs(absl::string_view str) {
  if (!absl::ConsumePrefix(&str, "{") || !absl::ConsumeSuffix(&str, "}")) {
    return InvalidArgument("Expected format {index:memory_space,...}, got: %s",
                           str);
  }

  std::vector<std::pair<int64_t, MemorySpaceColor>> result;
  if (str.empty()) {
    return result;
  }

  for (absl::string_view pair : absl::StrSplit(str, ',')) {
    pair = absl::StripAsciiWhitespace(pair);
    std::vector<absl::string_view> parts = absl::StrSplit(pair, ':');
    if (parts.size() != 2) {
      return InvalidArgument("Expected index:memory_space pair, got: %s", pair);
    }
    int64_t index, memory_space;
    if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(parts[0]), &index) ||
        !absl::SimpleAtoi(absl::StripAsciiWhitespace(parts[1]),
                          &memory_space)) {
      return InvalidArgument("Failed to parse integers in pair: %s", pair);
    }
    ASSIGN_OR_RETURN(MemorySpaceColor color, AsMemorySpaceColor(memory_space));
    result.emplace_back(index, color);
  }

  return result;
}

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

// Returns the memory space requested for the given custom call use, or
// MemorySpaceColor::kDefault if none is specified.
static absl::StatusOr<MemorySpaceColor> GetCustomCallOperandMemorySpace(
    const HloUse& use) {
  if (use.instruction->opcode() != HloOpcode::kCustomCall ||
      !use.operand_index.empty()) {
    return MemorySpaceColor::kDefault;
  }

  auto attr =
      use.instruction->get_frontend_attribute(kOperandsMemorySpacesAttr);
  if (!attr.has_value()) {
    return MemorySpaceColor::kDefault;
  }

  ASSIGN_OR_RETURN(auto pairs, ParseIndexMemorySpacePairs(*attr));
  for (auto [index, memory_space] : pairs) {
    if (index == use.operand_number) {
      return memory_space;
    }
  }

  return MemorySpaceColor::kDefault;
}

// Returns the memory space requested for a custom call result value, or
// MemorySpaceColor::kDefault if none is specified.
static absl::StatusOr<MemorySpaceColor> GetCustomCallResultMemorySpace(
    const HloValue& value) {
  const HloInstruction* instr = value.instruction();
  if (instr->opcode() != HloOpcode::kCustomCall) {
    return MemorySpaceColor::kDefault;
  }

  auto attr = instr->get_frontend_attribute(kResultsMemorySpacesAttr);
  if (!attr.has_value()) {
    return MemorySpaceColor::kDefault;
  }

  ASSIGN_OR_RETURN(auto pairs, ParseIndexMemorySpacePairs(*attr));
  const ShapeIndex& idx = value.defining_index();
  for (auto [index, memory_space] : pairs) {
    if (instr->shape().IsTuple() ? (idx.size() == 1 && idx[0] == index)
                                 : (idx.empty() && index == 0)) {
      return memory_space;
    }
  }

  return MemorySpaceColor::kDefault;
}

// Set memory space to MemorySpaceColor::kCollective for all allocations used by
// all-reduce, all-gather, and reduce-scatter. This memory space maps to
// collective memory using ncclMemAlloc in the runtime.
//
// Also assigns memory space colors for custom call operands and results based
// on `operands_memory_spaces` and `results_memory_spaces` frontend attributes.
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

    // Check if this value is a custom call result with a requested memory
    // space.
    ASSIGN_OR_RETURN(MemorySpaceColor result_ms,
                     GetCustomCallResultMemorySpace(*value));
    if (result_ms != MemorySpaceColor::kDefault) {
      value->set_color(static_cast<int>(result_ms));
      continue;
    }

    for (const xla::HloValue* alias :
         alias_analysis->GetBufferContainingValue(*value).values()) {
      TF_RET_CHECK(alias != nullptr);

      // Check if any use of this alias is a custom call operand with a
      // requested memory space.
      for (const HloUse& use : alias->GetUses()) {
        ASSIGN_OR_RETURN(MemorySpaceColor operand_ms,
                         GetCustomCallOperandMemorySpace(use));
        if (operand_ms != MemorySpaceColor::kDefault) {
          value->set_color(static_cast<int>(operand_ms));
        }
      }
      if (value->has_color()) {
        break;
      }

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

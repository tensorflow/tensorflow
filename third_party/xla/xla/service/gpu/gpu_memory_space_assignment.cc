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
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

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
        HloOpcode::kRaggedAllToAll,
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

// Returns true if the instruction's collectives mode requires symmetric
// (collective) memory. Device-initiated and one-sided collectives need all
// buffers registered with the collective runtime ahead of time.
bool RequiresCollectiveSymmetricMemorySpace(const HloInstruction* inst) {
  if (!inst->has_backend_config()) {
    return false;
  }
  auto gpu_config = inst->backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return false;
  }
  const auto mode = gpu_config->collective_backend_config().collectives_mode();
  return mode == DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY;
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
  // Tuple-shaped values are pointer containers and never hold data that needs
  // to live in collective memory. Only array sub-elements do.
  if (input_alias.shape().IsTuple()) {
    return false;
  }
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

bool HasSymmetricMemoryInstruction(const HloValue& input_alias) {
  // Tuple-shaped values are pointer containers and never hold data that needs
  // to live in collective memory. Only array sub-elements do.
  if (input_alias.shape().IsTuple()) {
    return false;
  }
  for (const HloUse& use : input_alias.GetUses()) {
    if (RequiresCollectiveSymmetricMemorySpace(use.instruction)) {
      return true;
    }
  }
  return RequiresCollectiveSymmetricMemorySpace(input_alias.instruction());
}

bool HasMosaicInstruction(const HloValue& input_alias,
                          absl::FunctionRef<bool(HloInstruction&)> predicate) {
  // Tuple-shaped values are pointer containers and never hold data that needs
  // to live in collective memory. Only array sub-elements do.
  if (input_alias.shape().IsTuple()) {
    return false;
  }
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

// Returns true if the value is
// 1. Used by a RaggedAllToAll as Operand(1)
// 2. RaggedAllToAll result
bool IsRaggedAllToAllCollectiveOperandOrResult(const HloValue& value) {
  if (value.shape().IsTuple()) {
    return false;
  }

  // Check if the value is DEFINED by an RA2A (the result)
  if (IsRaggedAllToAllOrAsyncDoneRaggedAllToAll(value.defining_instruction())) {
    return true;
  }

  // Check if the value is USED by an RA2A as the destination (Operand 1)
  for (const HloUse& use : value.GetUses()) {
    if (IsRaggedAllToAllOrAsyncStartRaggedAllToAll(use.instruction) &&
        use.operand_number == 1) {
      return true;
    }
  }
  return false;
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

namespace {
// Determines the memory space color for the given HLO buffer
absl::StatusOr<BufferValue::Color> DetermineBufferColor(
    const HloBuffer& buffer, bool use_collective_memory, bool use_nvshmem,
    bool is_one_shot_ra2a_with_nccl) {
  // Collect Color Candidates
  absl::InlinedVector<BufferValue::Color, 4> candidates;
  for (const HloValue* value : buffer.values()) {
    // If the value has a layout with non-default memory space, use the memory
    // space from the layout.
    const HloPosition& defining_position = value->defining_position();
    if (defining_position.shape().has_layout()) {
      const BufferValue::Color memory_space =
          defining_position.shape().layout().memory_space();
      if (memory_space != 0) {
        candidates.push_back(memory_space);
      }
    }

    // Check if this value is a custom call result with a requested memory
    // space.
    ASSIGN_OR_RETURN(MemorySpaceColor result_ms,
                     GetCustomCallResultMemorySpace(*value));
    if (result_ms != MemorySpaceColor::kDefault) {
      candidates.push_back(static_cast<BufferValue::Color>(result_ms));
    }

    // Check if any use of this alias is a custom call operand with a
    // requested memory space.
    for (const HloUse& use : value->GetUses()) {
      ASSIGN_OR_RETURN(MemorySpaceColor operand_ms,
                       GetCustomCallOperandMemorySpace(use));
      if (operand_ms != MemorySpaceColor::kDefault) {
        candidates.push_back(static_cast<BufferValue::Color>(operand_ms));
      }
    }

    // Collective/Mosaic Candidates
    // TODO(479768130): Mark only buffers used with multimem instructions
    // instead of marking all buffers.
    // TODO(508106498): We need to start to respect replica groups once
    // mosaic will support them.
    const bool is_mosaic_with_nvshmem = HasMosaicWithNvshmemInstruction(*value);
    const bool is_mosaic_with_multimem =
        HasMosaicWithMultimemInstruction(*value);

    if ((is_mosaic_with_nvshmem && use_nvshmem) || is_mosaic_with_multimem) {
      VLOG(1) << "Assigning color kCollective to value of instruction "
              << value->instruction()->ToShortString()
              << " is_mosaic_with_multimem " << is_mosaic_with_multimem
              << " is_mosaic_with_nvshmem " << is_mosaic_with_nvshmem;
      // This is a temporary solution until a separate BFC
      // allocator will be added for the symmetric memory space.
      candidates.push_back(
          static_cast<BufferValue::Color>(MemorySpaceColor::kCollective));
    } else if (is_one_shot_ra2a_with_nccl &&
               IsRaggedAllToAllCollectiveOperandOrResult(*value)) {
      // One-shot RaggedAllToAll with NCCL requires collective memory for
      // both operand 1 and the result.
      candidates.push_back(
          static_cast<BufferValue::Color>(MemorySpaceColor::kCollective));
    } else if (HasSymmetricMemoryInstruction(*value)) {
      // Device-initiated and one-sided collectives require symmetric memory.
      candidates.push_back(
          static_cast<BufferValue::Color>(MemorySpaceColor::kCollective));
    } else if (((use_collective_memory &&
                 HasCollectiveMemoryInstruction(*value)) ||
                (use_nvshmem && HasCollectiveMemoryInstruction(
                                    *value, /*require_nvshmem=*/true)))) {
      candidates.push_back(
          static_cast<BufferValue::Color>(MemorySpaceColor::kCollective));
    }
  }

  if (candidates.empty()) {
    return static_cast<BufferValue::Color>(MemorySpaceColor::kDefault);
  }
  BufferValue::Color color = candidates[0];
  // Check for color conflicts.
  for (BufferValue::Color candidate : candidates) {
    if (color != candidate) {
      return absl::InternalError(absl::StrCat(
          "Memory space conflict detected in HloBuffer ", buffer.id(),
          ". Conflicting colors: ", color, " and ", candidate));
    }
  }
  return color;
}
}  // namespace

// Assigns memory space colors to all HloBuffers in the module.
//
// Relies on DetermineBufferColor to aggregate memory space constraints from
// the HloValues in the buffer. If a valid, conflict-free color is found, it
// is uniformly applied to all HloValues within the buffer.
absl::Status AssignColors(bool use_collective_memory, bool use_nvshmem,
                          bool is_one_shot_ra2a_with_nccl,
                          HloAliasAnalysis* alias_analysis) {
  HloDataflowAnalysis& dataflow_analysis = alias_analysis->dataflow_analysis();
  for (const HloBuffer& buffer : alias_analysis->buffers()) {
    ASSIGN_OR_RETURN(
        BufferValue::Color color,
        DetermineBufferColor(buffer, use_collective_memory, use_nvshmem,
                             is_one_shot_ra2a_with_nccl));
    // Apply buffer color to all values in the buffer.
    for (const HloValue* const_value : buffer.values()) {
      HloValue& mutable_value = dataflow_analysis.GetValue(const_value->id());
      mutable_value.set_color(color);
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

  // Is one-shot RaggedAllToAll with NCCL feature is enabled.
  bool is_one_shot_ra2a_with_nccl =
      IsOneShotRaggedAllToAllWithNcclEnabled(option);

  return [use_collective_memory, use_nvshmem, is_one_shot_ra2a_with_nccl](
             HloAliasAnalysis* alias_analysis, const HloOrdering&) {
    return AssignColors(use_collective_memory, use_nvshmem,
                        is_one_shot_ra2a_with_nccl, alias_analysis);
  };
}
}  // namespace xla::gpu

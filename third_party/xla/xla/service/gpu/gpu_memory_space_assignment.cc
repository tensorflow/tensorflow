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

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
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
    case 1:
      // Legacy value for collective memory space before
      // Layout::kCollectiveMemorySpace (7) was introduced
      return MemorySpaceColor::kCollective;
    case static_cast<int64_t>(MemorySpaceColor::kDefault):
    case static_cast<int64_t>(MemorySpaceColor::kCollective):
    case static_cast<int64_t>(MemorySpaceColor::kTempBuffer):
      return static_cast<MemorySpaceColor>(memory_space);
    default:
      return InvalidArgument(
          "Invalid memory space %d. "
          "Valid values are %d (default), %d (collective), %d (temp).",
          memory_space, MemorySpaceColor::kDefault,
          MemorySpaceColor::kCollective, MemorySpaceColor::kTempBuffer);
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
    ABSL_ASSIGN_OR_RETURN(MemorySpaceColor color, AsMemorySpaceColor(memory_space));
    result.emplace_back(index, color);
  }

  return result;
}

// Only the dynamic-slice rewriter remaps these index-based annotations when
// forming a fusion. Other fusion kinds must not consume inherited annotations.
static bool SupportsMemorySpaceAnnotations(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kCustomCall ||
         GetCustomFusionConfigName(inst) == kDynamicSliceFusionConfigName;
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

bool IsNcclSymmetricOrUserBuffersEnabledForInstruction(
    const HloInstruction* inst, const DebugOptions& option) {
  if (!IsCollectiveMemoryInstruction(inst)) {
    return false;
  }
  return option.xla_gpu_enable_nccl_user_buffers() ||
         IsNcclSymmetricBuffersEnabledForCollective(inst, option);
}

// Returns the memory space requested for the given operand use, or
// MemorySpaceColor::kDefault if none is specified.
static absl::StatusOr<MemorySpaceColor> GetOperandMemorySpace(
    const HloUse& use) {
  if (!SupportsMemorySpaceAnnotations(use.instruction) ||
      !use.operand_index.empty()) {
    return MemorySpaceColor::kDefault;
  }

  auto attr =
      use.instruction->get_frontend_attribute(kOperandsMemorySpacesAttr);
  if (!attr.has_value()) {
    return MemorySpaceColor::kDefault;
  }

  ABSL_ASSIGN_OR_RETURN(auto pairs, ParseIndexMemorySpacePairs(*attr));
  for (auto [index, memory_space] : pairs) {
    if (index == use.operand_number) {
      return memory_space;
    }
  }

  return MemorySpaceColor::kDefault;
}

// Returns the memory space requested for an instruction result value, or
// MemorySpaceColor::kDefault if none is specified.
static absl::StatusOr<MemorySpaceColor> GetResultMemorySpace(
    const HloValue& value) {
  const HloInstruction* instr = value.instruction();
  if (!SupportsMemorySpaceAnnotations(instr)) {
    return MemorySpaceColor::kDefault;
  }

  auto attr = instr->get_frontend_attribute(kResultsMemorySpacesAttr);
  if (!attr.has_value()) {
    return MemorySpaceColor::kDefault;
  }

  ABSL_ASSIGN_OR_RETURN(auto pairs, ParseIndexMemorySpacePairs(*attr));
  const ShapeIndex& idx = value.defining_index();
  for (auto [index, memory_space] : pairs) {
    if (instr->shape().IsTuple() ? (idx.size() == 1 && idx[0] == index)
                                 : (idx.empty() && index == 0)) {
      return memory_space;
    }
  }

  return MemorySpaceColor::kDefault;
}

bool RequiresCollectiveInput(const HloUse& use, const DebugOptions& option) {
  const HloInstruction* user = use.instruction;
  if (ShapeUtil::GetSubshape(user->operand(use.operand_number)->shape(),
                             use.operand_index)
          .IsTuple()) {
    return false;
  }

  // Handle standard collectives under NCCL user/symmetric buffers, or
  // device-initiated/one-sided collectives requiring symmetric memory.
  if (IsNcclSymmetricOrUserBuffersEnabledForInstruction(user, option) ||
      RequiresCollectiveSymmetricMemorySpace(user)) {
    return true;
  }

  // Handle one-shot RaggedAllToAll with NCCL enabled: operand 1 (the output
  // buffer) requires collective memory.
  if (IsOneShotRaggedAllToAllWithNcclEnabled(option) &&
      IsRaggedAllToAllOrAsyncStartRaggedAllToAll(user) &&
      use.operand_number == 1) {
    return true;
  }

  // Check custom calls and dynamic-slice fusions with memory-space annotations.
  absl::StatusOr<MemorySpaceColor> operand_ms = GetOperandMemorySpace(use);
  return operand_ms.ok() && *operand_ms == MemorySpaceColor::kCollective;
}

bool RequiresCollectiveOutput(const HloValue& value,
                              const DebugOptions& option) {
  // Tuple-shaped values are pointer containers and never hold data that needs
  // to live in collective memory. Only array sub-elements do.
  if (value.shape().IsTuple()) {
    return false;
  }

  const HloInstruction* def = value.defining_instruction();

  // Handle standard collectives under NCCL user/symmetric buffers, or
  // device-initiated/one-sided collectives requiring symmetric memory.
  if (IsNcclSymmetricOrUserBuffersEnabledForInstruction(def, option) ||
      RequiresCollectiveSymmetricMemorySpace(def)) {
    return true;
  }

  // Handle one-shot RaggedAllToAll with NCCL enabled.
  if (IsOneShotRaggedAllToAllWithNcclEnabled(option) &&
      IsRaggedAllToAllOrAsyncDoneRaggedAllToAll(def)) {
    return true;
  }

  // Check custom calls and dynamic-slice fusions with memory-space annotations.
  absl::StatusOr<MemorySpaceColor> result_ms = GetResultMemorySpace(value);
  return result_ms.ok() && *result_ms == MemorySpaceColor::kCollective;
}

namespace {

bool HasCollectiveMemoryInstruction(const HloValue& input_alias,
                                    const DebugOptions& option) {
  // Tuple-shaped values are pointer containers and never hold data that needs
  // to live in collective memory. Only array sub-elements do.
  if (input_alias.shape().IsTuple()) {
    return false;
  }
  if (RequiresCollectiveOutput(input_alias, option)) {
    return true;
  }
  for (const HloUse& use : input_alias.GetUses()) {
    if (RequiresCollectiveInput(use, option)) {
      return true;
    }
  }
  return false;
}

// Determines the memory space color for the given HLO buffer
absl::StatusOr<BufferValue::Color> DetermineBufferColor(
    const HloBuffer& buffer, const DebugOptions& option) {
  // Collect Color Candidates
  absl::InlinedVector<BufferValue::Color, 4> candidates;
  for (const HloValue* value : buffer.values()) {
    // If the value has a layout with non-default memory space, use the memory
    // space from the layout.
    const HloPosition& defining_position = value->defining_position();
    if (defining_position.shape().has_layout()) {
      BufferValue::Color memory_space =
          defining_position.shape().layout().memory_space();
      if (memory_space == 1) {
        // Legacy value for collective memory space before
        // Layout::kCollectiveMemorySpace (7) was introduced.
        memory_space =
            static_cast<BufferValue::Color>(MemorySpaceColor::kCollective);
      }

      if (memory_space != 0) {
        candidates.push_back(memory_space);
      }
    }

    // Check if this value has a requested result memory space.
    ABSL_ASSIGN_OR_RETURN(MemorySpaceColor result_ms, GetResultMemorySpace(*value));
    if (result_ms != MemorySpaceColor::kDefault) {
      candidates.push_back(static_cast<BufferValue::Color>(result_ms));
    }

    // Check if any use of this alias has a requested operand memory space.
    for (const HloUse& use : value->GetUses()) {
      ABSL_ASSIGN_OR_RETURN(MemorySpaceColor operand_ms, GetOperandMemorySpace(use));
      if (operand_ms != MemorySpaceColor::kDefault) {
        candidates.push_back(static_cast<BufferValue::Color>(operand_ms));
      }
    }

    // Collective Candidates
    if (HasCollectiveMemoryInstruction(*value, option)) {
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
absl::Status AssignColors(const DebugOptions& option,
                          HloAliasAnalysis* alias_analysis) {
  HloDataflowAnalysis& dataflow_analysis = alias_analysis->dataflow_analysis();
  for (const HloBuffer& buffer : alias_analysis->buffers()) {
    ABSL_ASSIGN_OR_RETURN(BufferValue::Color color,
                     DetermineBufferColor(buffer, option));
    // Apply buffer color to all values in the buffer.
    for (const HloValue* const_value : buffer.values()) {
      HloValue& mutable_value = dataflow_analysis.GetValue(const_value->id());
      mutable_value.set_color(color);
    }
  }

  return absl::OkStatus();
}

BufferAssigner::Colorer CreateColorer(const DebugOptions& option) {
  return [&](HloAliasAnalysis* alias_analysis, const HloOrdering&) {
    return AssignColors(option, alias_analysis);
  };
}

namespace {

// nullopt means all collective types are enabled.
using CollectiveOpTypes =
    std::optional<absl::flat_hash_set<DebugOptions::CollectiveOpType>>;

CollectiveOpTypes GetEnabledCollectiveOpTypes(const DebugOptions& option) {
  const auto& filter =
      option.xla_gpu_enable_collectives_command_buffer_filter();
  if (filter.empty() ||
      absl::c_linear_search(filter, DebugOptions::ALLCOLLECTIVES)) {
    return std::nullopt;
  }
  absl::flat_hash_set<DebugOptions::CollectiveOpType> enabled;
  for (int type : filter) {
    enabled.insert(static_cast<DebugOptions::CollectiveOpType>(type));
  }
  return enabled;
}

// Collective groups are matched through the collectives they wrap.
bool IsCommandBufferCollective(const HloInstruction* instruction,
                               const CollectiveOpTypes& enabled) {
  switch (instruction->opcode()) {
    // The filter has no type for send and recv. Host transfers lower to host
    // send/recv thunks, which are never recorded.
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
      return !Cast<HloSendRecvInstruction>(instruction)->is_host_transfer();
    default:
      break;
  }
  std::optional<DebugOptions::CollectiveOpType> type =
      GetCollectiveOpType(instruction);
  return type.has_value() && (!enabled.has_value() || enabled->contains(*type));
}

bool IsCommandBufferCollectiveValue(const HloAliasAnalysis& alias_analysis,
                                    const HloValue& value,
                                    const CollectiveOpTypes& enabled) {
  if (value.shape().IsTuple() || alias_analysis.ValueLivesOut(value)) {
    return false;
  }
  if (IsCommandBufferCollective(value.instruction(), enabled)) {
    return true;
  }
  return absl::c_any_of(value.GetUses(), [&](const HloUse& use) {
    return IsCommandBufferCollective(use.instruction, enabled);
  });
}

}  // namespace

std::optional<BufferAssigner::MustNotLiveOut> CreateMustNotLiveOut(
    const DebugOptions& option) {
  // Only the VA remapping modes keep the temp allocation at a stable address.
  const DebugOptions::CommandBufferUpdateMode update_mode =
      option.xla_gpu_command_buffer_update_mode();
  // TODO(magaonka-amd): Account for ragged all-to-all, which the command
  // buffer conversion pass records as COLLECTIVES_KERNEL when it uses the
  // symmetric memory kernel.
  if (!option.xla_gpu_command_buffer_keep_collectives_out_of_outputs() ||
      (update_mode != DebugOptions::SKIP_TEMP &&
       update_mode != DebugOptions::SKIP_PROFILED) ||
      !absl::c_linear_search(option.xla_gpu_enable_command_buffer(),
                             DebugOptions::COLLECTIVES)) {
    return std::nullopt;
  }
  return [enabled = GetEnabledCollectiveOpTypes(option)](
             const HloAliasAnalysis& alias_analysis,
             const HloInstruction* instruction, const ShapeIndex& index) {
    const HloValueSet& value_set =
        alias_analysis.dataflow_analysis().GetValueSet(instruction, index);
    return absl::c_any_of(value_set.values(), [&](const HloValue* value) {
      return IsCommandBufferCollectiveValue(alias_analysis, *value, enabled);
    });
  };
}

BufferAssigner::Options CreateBufferAssignerOptions(
    const DebugOptions& option) {
  BufferAssigner::Options opts;
  opts.colorer = CreateColorer(option);
  opts.must_not_live_out = CreateMustNotLiveOut(option);

  // Allow S(0) buffers to reuse S(1) temp allocations. S(1) allocations
  // satisfy stricter alignment and symmetric-offset requirements, so they are
  // valid storage for S(0) buffers. Keep this directional: S(1) buffers must
  // not be assigned to S(0) allocations.
  opts.can_use_allocation = BufferAssigner::AllowCrossColorReuse(
      static_cast<int>(MemorySpaceColor::kDefault),
      static_cast<int>(MemorySpaceColor::kCollective));
  return opts;
}

}  // namespace xla::gpu

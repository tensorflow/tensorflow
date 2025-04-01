/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/host_offload_utils.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/memory_annotations.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/util.h"

namespace xla {
namespace host_offload_utils {

namespace {

using ::xla::memory_annotations::kMoveToDeviceCustomCallTarget;
using ::xla::memory_annotations::kMoveToHostCustomCallTarget;

bool CustomCallReusesBuffer(const HloInstruction* custom_call,
                            int64_t operand_index) {
  if (custom_call->custom_call_target() == kMoveToDeviceCustomCallTarget ||
      custom_call->custom_call_target() == kMoveToHostCustomCallTarget) {
    // Does not define a new buffer.
    return true;
  }
  // Check the custom call's output_to_operand_aliasing.
  const std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>&
      aliases = custom_call->output_operand_aliasing();
  for (const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>& alias :
       aliases) {
    int64_t alias_operand_index = alias.second.first;
    if (alias_operand_index == operand_index) {
      // This operand aliases with the output.
      return true;
    }
  }
  // By default, assume custom calls define new buffers.
  return false;
}

}  // namespace

absl::StatusOr<std::vector<InstructionAndShapeIndex>> GetSuccessors(
    const InstructionAndShapeIndex& instruction_and_shape_index) {
  std::vector<InstructionAndShapeIndex> result;
  HloInstruction* instruction = instruction_and_shape_index.instruction;
  if (instruction->IsRoot()) {
    // Successor of the root is the call instruction(s).
    std::unique_ptr<CallGraph> call_graph =
        CallGraph::Build(instruction->GetModule());
    auto callers = call_graph->GetComputationCallers(instruction->parent());
    for (HloInstruction* caller : callers) {
      result.push_back({caller, instruction_and_shape_index.shape_index});
    }
  }
  for (HloInstruction* user : instruction->users()) {
    if (user->opcode() == HloOpcode::kTuple) {
      auto operand_indices = user->OperandIndices(instruction);
      for (const auto i : operand_indices) {
        auto tmp_shape_index = instruction_and_shape_index.shape_index;
        tmp_shape_index.push_front(i);
        result.push_back({user, std::move(tmp_shape_index)});
      }
    } else if (user->opcode() == HloOpcode::kGetTupleElement) {
      ShapeIndex tmp_shape_index = instruction_and_shape_index.shape_index;
      const auto index = tmp_shape_index.front();
      if (index == user->tuple_index()) {
        // This GTE is for the buffer we're tracking.
        tmp_shape_index.pop_front();
        result.push_back({user, std::move(tmp_shape_index)});
      }
    } else if (user->opcode() == HloOpcode::kCall) {
      auto operand_indices = user->OperandIndices(instruction);
      CHECK(user->called_computations().size() == 1)
          << "Expect call to only have one called computation.";
      for (const auto i : operand_indices) {
        HloComputation* called_computation =
            user->called_computations().front();
        HloInstruction* parameter_instruction =
            called_computation->parameter_instruction(i);
        result.push_back(
            {parameter_instruction, instruction_and_shape_index.shape_index});
      }
    } else if (user->opcode() == HloOpcode::kWhile) {
      auto operand_indices = user->OperandIndices(instruction);
      HloComputation* while_body_computation = user->while_body();
      HloComputation* while_condition_computation = user->while_condition();
      for (const auto i : operand_indices) {
        HloInstruction* parameter_instruction =
            while_body_computation->parameter_instruction(i);
        result.push_back(
            {parameter_instruction, instruction_and_shape_index.shape_index});

        HloInstruction* condition_instruction =
            while_condition_computation->parameter_instruction(i);
        result.push_back(
            {condition_instruction, instruction_and_shape_index.shape_index});
      }
    } else if (user->opcode() == HloOpcode::kAsyncStart) {
      auto operand_indices = user->OperandIndices(instruction);
      CHECK(user->called_computations().size() == 1)
          << "Expect async-start to only have one called computation.";
      for (const auto i : operand_indices) {
        HloComputation* called_computation =
            user->called_computations().front();
        HloInstruction* parameter_instruction =
            called_computation->parameter_instruction(i);
        result.push_back(
            {parameter_instruction, instruction_and_shape_index.shape_index});
      }
    } else if (user->opcode() == HloOpcode::kCustomCall) {
      const auto operand_indices = user->OperandIndices(instruction);
      // TODO(b/342650757): Rather than a boolean indicating whether the
      // instruction reuses the buffer, return the shape index of the output
      // that the operand aliases with.
      bool found_one = false;
      for (const auto i : operand_indices) {
        if (CustomCallReusesBuffer(user, i)) {
          if (found_one) {
            return absl::InternalError(
                "Found multiple operands of a custom call that reuse the same "
                "output buffer.");
          }
          result.push_back({user, instruction_and_shape_index.shape_index});
          found_one = true;
        }
      }
    } else {
      result.push_back({user, instruction_and_shape_index.shape_index});
    }
  }
  return result;
}

std::vector<InstructionAndShapeIndex> GetPredecessors(
    const InstructionAndShapeIndex& instruction_and_shape_index) {
  std::vector<InstructionAndShapeIndex> result;
  HloInstruction* instruction = instruction_and_shape_index.instruction;
  if (instruction->opcode() == HloOpcode::kGetTupleElement) {
    const int64_t index = instruction->tuple_index();
    auto tmp_shape_index = instruction_and_shape_index.shape_index;
    tmp_shape_index.push_front(index);
    result.push_back({instruction->mutable_operand(0), tmp_shape_index});
  } else if (instruction->opcode() == HloOpcode::kTuple) {
    CHECK(!instruction_and_shape_index.shape_index.empty())
        << "Did not store an index before encountering a tuple.";
    auto tmp_shape_index = instruction_and_shape_index.shape_index;
    const int64_t index = tmp_shape_index.front();
    tmp_shape_index.pop_front();
    result.push_back({instruction->mutable_operand(index), tmp_shape_index});
  } else if (instruction->opcode() == HloOpcode::kCall) {
    // Predecessor of a call is its computation's root instruction.
    CHECK(instruction->called_computations().size() == 1)
        << "Expect call to only have one called computation.";
    HloComputation* called_computation =
        instruction->called_computations().front();
    result.push_back({called_computation->root_instruction(),
                      instruction_and_shape_index.shape_index});
  } else if (instruction->opcode() == HloOpcode::kParameter) {
    std::unique_ptr<CallGraph> call_graph =
        CallGraph::Build(instruction->GetModule());
    auto callers = call_graph->GetComputationCallers(instruction->parent());
    for (HloInstruction* caller : callers) {
      result.push_back(
          {caller->mutable_operand(instruction->parameter_number()),
           instruction_and_shape_index.shape_index});
    }
  } else if (instruction->opcode() == HloOpcode::kDynamicSlice) {
    result.push_back({instruction->mutable_operand(0),
                      instruction_and_shape_index.shape_index});
  } else if (instruction->opcode() == HloOpcode::kDynamicUpdateSlice) {
    result.push_back({instruction->mutable_operand(0),
                      instruction_and_shape_index.shape_index});
  } else if (instruction->opcode() == HloOpcode::kWhile) {
    HloComputation* while_body_computation = instruction->while_body();
    result.push_back({while_body_computation->root_instruction(),
                      instruction_and_shape_index.shape_index});
  } else if (instruction->opcode() == HloOpcode::kPad) {
    result.push_back({instruction->mutable_operand(0),
                      instruction_and_shape_index.shape_index});
  } else {
    CHECK(instruction->operand_count() == 1) << absl::StreamFormat(
        "Expecting instruction %s to have 1 operand, but it has %d.",
        instruction->name(), instruction->operand_count());
    result.push_back({instruction->mutable_operand(0),
                      instruction_and_shape_index.shape_index});
  }
  return result;
}

bool IsValidDuringPureMemoryOffload(const HloInstruction* instruction) {
  static constexpr std::array allowed_opcodes = {
      HloOpcode::kGetTupleElement,
      HloOpcode::kBitcast,
      HloOpcode::kTuple,
      HloOpcode::kCall,
      HloOpcode::kWhile,
      HloOpcode::kParameter,
      HloOpcode::kOptimizationBarrier,
      HloOpcode::kAsyncStart,
      HloOpcode::kAsyncDone,
      HloOpcode::kCustomCall};
  return absl::c_linear_search(allowed_opcodes, instruction->opcode());
}

bool operator==(const InstructionAndShapeIndex& lhs,
                const InstructionAndShapeIndex& rhs) {
  return lhs.instruction == rhs.instruction &&
         lhs.shape_index == rhs.shape_index;
}

std::string InstructionAndShapeIndex::ToString() const {
  return absl::StrFormat("{Instr: %s, ShapeIndex: %s}", instruction->name(),
                         shape_index.ToString());
}

bool IsHostAsyncStart(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kAsyncStart &&
         instruction->async_execution_thread() == HloInstruction::kHostThread;
}

bool IsSynchronousCopyFromOrToHost(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kCopy) {
    return false;
  }
  if (!instruction->shape().has_layout() ||
      !instruction->operand(0)->shape().has_layout()) {
    // Host offloading copies do not exist without layouts.
    return false;
  }
  const int64_t copy_memory_space =
      instruction->shape().layout().memory_space();
  const int64_t operand_memory_space =
      instruction->operand(0)->shape().layout().memory_space();
  return (copy_memory_space == Layout::kHostMemorySpace &&
          operand_memory_space != Layout::kHostMemorySpace) ||
         (copy_memory_space != Layout::kHostMemorySpace &&
          operand_memory_space == Layout::kHostMemorySpace);
}

bool ComputeTypeIsHost(const HloInstruction* hlo_instruction) {
  const auto& frontend_attributes_map =
      hlo_instruction->frontend_attributes().map();
  return (frontend_attributes_map.find(kXlaComputeTypeAttr) !=
              frontend_attributes_map.end() &&
          frontend_attributes_map.find(kXlaComputeTypeAttr)->second ==
              kXlaComputeTypeHost);
}

void SetHostComputeFrontendAttribute(HloInstruction& host_instruction) {
  FrontendAttributes frontend_attributes =
      host_instruction.frontend_attributes();
  frontend_attributes.mutable_map()->insert(
      {kXlaComputeTypeAttr, kXlaComputeTypeHost});
  host_instruction.set_frontend_attributes(frontend_attributes);
}

}  // namespace host_offload_utils
}  // namespace xla

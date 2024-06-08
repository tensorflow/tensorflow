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

#include "xla/service/host_offloader.h"

#include <array>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_value.h"
#include "xla/service/host_memory_offload_annotations.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

using ::xla::host_memory_offload_annotations::kMoveToDeviceCustomCallTarget;
using ::xla::host_memory_offload_annotations::kMoveToHostCustomCallTarget;

void SetMemorySpace(Shape* shape, int64_t memory_space_color) {
  CHECK(shape->has_layout());
  shape->mutable_layout()->set_memory_space(memory_space_color);
}

bool SetBuffersToMemorySpaceColor(
    const std::vector<InstructionAndShapeIndex>& buffers_to_set_to_host_memory,
    int64_t memory_space_color) {
  bool changed = false;
  for (const auto& instr_and_shape : buffers_to_set_to_host_memory) {
    VLOG(2) << absl::StreamFormat("Setting %s to memory space %d",
                                  instr_and_shape.ToString(),
                                  memory_space_color);
    Shape* shape = ShapeUtil::GetMutableSubshape(
        instr_and_shape.instruction->mutable_shape(),
        instr_and_shape.shape_index);
    CHECK(shape->has_layout()) << "Shape must have a layout";
    SetMemorySpace(ShapeUtil::GetMutableSubshape(
                       instr_and_shape.instruction->mutable_shape(),
                       instr_and_shape.shape_index),
                   memory_space_color);
    changed = true;
  }
  return changed;
}

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

// If an instruction's user is a call, we descend into the call first.
// Eventually, a later invocation of this function while walking the graph will
// return the call itself as a successor of the ROOT instruction of the
// computation.
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
        tmp_shape_index.push_back(i);
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
      LOG(INFO) << "Instruction " << instruction->name()
                << " feeds into async-start " << user->name();
      auto operand_indices = user->OperandIndices(instruction);
      CHECK(user->called_computations().size() == 1)
          << "Expect async-start to only have one called computation.";
      for (const auto i : operand_indices) {
        HloComputation* called_computation =
            user->called_computations().front();
        HloInstruction* parameter_instruction =
            called_computation->parameter_instruction(i);
        LOG(INFO) << "Which is used by parameter "
                  << parameter_instruction->name();
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

// If an instruction's operand is a call, return the call now. A follow up call
// of this function on that call returns the ROOT. Eventually, once the given
// instruction is a parameter, the returned predecessor will be the appropriate
// operand of the call (not the call itself, since we already returned it).
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
  } else {
    CHECK(instruction->operand_count() == 1) << absl::StreamFormat(
        "Expecting instruction %s to have 1 operand, but it has %d.",
        instruction->name(), instruction->operand_count());
    result.push_back({instruction->mutable_operand(0),
                      instruction_and_shape_index.shape_index});
  }
  return result;
}

}  // namespace

bool operator==(const InstructionAndShapeIndex& lhs,
                const InstructionAndShapeIndex& rhs) {
  return lhs.instruction == rhs.instruction &&
         lhs.shape_index == rhs.shape_index;
}

std::string InstructionAndShapeIndex::ToString() const {
  return absl::StrFormat("{Instr: %s, ShapeIndex: %s}", instruction->name(),
                         shape_index.ToString());
}

bool HostOffloader::IsValidDuringPureMemoryOffload(
    const HloInstruction* instruction) const {
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

bool HostOffloader::InstructionIsAllowedBetweenMoveToHostAndDus(
    const HloInstruction* instruction) const {
  if (instruction->opcode() == HloOpcode::kReshape) {
    return ShapeUtil::ReshapeIsBitcast(instruction->operand(0)->shape(),
                                       instruction->shape());
  }
  return instruction->opcode() == HloOpcode::kBitcast;
}

bool HostOffloader::InstructionIsAllowedBetweenDsAndMoveToDevice(
    const HloInstruction* instruction) const {
  if (instruction->opcode() == HloOpcode::kReduce) {
    // TODO(b/333902007): Remove this once trivial reduces no longer appear.
    return ShapeUtil::TrueRank(instruction->operand(0)->shape()) ==
           ShapeUtil::TrueRank(instruction->shape());
  }
  if (instruction->opcode() == HloOpcode::kReshape) {
    return ShapeUtil::ReshapeIsBitcast(instruction->operand(0)->shape(),
                                       instruction->shape());
  }
  return instruction->opcode() == HloOpcode::kBitcast ||
         instruction->opcode() == HloOpcode::kCopy;
}

absl::StatusOr<bool> HostOffloader::WalkDownHostMemoryOffloadPaths(
    const InstructionAndShapeIndex& starting_instruction_and_index,
    bool insert_copy_before) {
  bool changed = false;
  absl::flat_hash_set<HloInstruction*> mth_custom_calls_to_remove;
  absl::flat_hash_set<HloInstruction*> slices_to_dynamify;
  absl::flat_hash_set<HloInstruction*> custom_calls_to_insert_copies_before;
  std::vector<InstructionAndShapeIndex> buffers_to_set_to_host_memory;
  std::vector<HloInstruction*> dynamic_update_slices;
  HloInstruction* starting_instruction =
      starting_instruction_and_index.instruction;
  std::queue<InstructionAndShapeIndex> queue;
  queue.push(starting_instruction_and_index);
  while (!queue.empty()) {
    InstructionAndShapeIndex instruction_and_shape_index = queue.front();
    queue.pop();
    HloInstruction* instruction = instruction_and_shape_index.instruction;
    VLOG(4) << absl::StreamFormat("Visiting instruction: %s",
                                  instruction_and_shape_index.ToString());
    bool already_saved_buffer = false;
    if (instruction->opcode() == HloOpcode::kCustomCall &&
        instruction->custom_call_target() ==
            host_memory_offload_annotations::kMoveToHostCustomCallTarget) {
      // This MoveToHost custom call is a no-op; save it to remove later.
      already_visited_move_to_host_custom_calls_.insert(instruction);
      mth_custom_calls_to_remove.insert(instruction);
    } else if (instruction->opcode() == HloOpcode::kCustomCall &&
               instruction->custom_call_target() ==
                   host_memory_offload_annotations::
                       kMoveToDeviceCustomCallTarget) {
      // This MoveToDevice marks the end of this path.
      custom_calls_to_insert_copies_before.insert(instruction);
      continue;
    } else if (instruction->opcode() == HloOpcode::kDynamicUpdateSlice) {
      if (instruction == starting_instruction) {
        dynamic_update_slices.push_back(instruction);
      } else {
        // The input to this DynamicUpdateSlice is already in host memory. Save
        // this so that we don't try to create an AllocateBuffer later.
        dynamic_update_slices_already_allocated_.insert(instruction);
      }
    } else if (IsValidDuringPureMemoryOffload(instruction)) {
      if (instruction->opcode() == HloOpcode::kAsyncStart) {
        // When visiting the parameter, we already set the memory space of the
        // input of the async-start; do not set it now.
        already_saved_buffer = true;
      } else if (instruction->opcode() == HloOpcode::kAsyncDone) {
        // Also set host memory space for the output in the async-start's shape.
        HloInstruction* async_start = instruction->mutable_operand(0);
        buffers_to_set_to_host_memory.emplace_back(async_start, ShapeIndex{1});
      } else if (instruction->opcode() == HloOpcode::kParameter) {
        // When setting the memory space of a parameter, also set the memory
        // space of the call site of the computation with this parameter if that
        // caller is an async-start.
        std::unique_ptr<CallGraph> call_graph =
            CallGraph::Build(instruction->GetModule());
        std::vector<HloInstruction*> callers =
            call_graph->GetComputationCallers(instruction->parent());
        for (HloInstruction* caller : callers) {
          if (caller->opcode() == HloOpcode::kAsyncStart) {
            ShapeIndex tmp_index = instruction_and_shape_index.shape_index;
            tmp_index.push_front(instruction->parameter_number());
            tmp_index.push_front(
                0);  // Index 0 for the inputs of the async-start. The shape of
                     // async-start is ((inputs, ...), output, context).
            buffers_to_set_to_host_memory.emplace_back(caller, tmp_index);
          }
        }
      }
    } else if (instruction->opcode() == HloOpcode::kDynamicSlice) {
      TF_RETURN_IF_ERROR(
          ValidateSliceLeadsToMoveToDeviceCustomCall(instruction));
      // This DynamicSlice is the end of this path of host memory offload.
      continue;
    } else if (instruction->opcode() == HloOpcode::kSlice) {
      TF_RETURN_IF_ERROR(
          ValidateSliceLeadsToMoveToDeviceCustomCall(instruction));
      // This Slice is the end of this path of host memory offload.
      // This Slice should be a DynamicSlice to be able to work with host
      // memory.
      slices_to_dynamify.insert(instruction);
      continue;
    } else {
      // Found an instruction which is invalid during host memory offloading.
      return absl::InvalidArgumentError(
          absl::StrFormat("Tensor which is moved to host (starting from "
                          "\"%s\") is used by an instruction (\"%s\") which is "
                          "not acceptable during pure memory offload.",
                          starting_instruction->name(), instruction->name()));
    }

    if (!already_saved_buffer) {
      // Save buffer to be set to host memory.
      VLOG(5) << "Saving " << instruction_and_shape_index.ToString()
              << " to be set to host memory.";
      buffers_to_set_to_host_memory.push_back(instruction_and_shape_index);
    }

    // Check if this path ends at the output of the entry computation.
    if (instruction->IsRoot() && instruction->parent()->IsEntryComputation()) {
      const Shape& output_shape = ShapeUtil::GetSubshape(
          instruction->GetModule()->entry_computation_layout().result_shape(),
          instruction_and_shape_index.shape_index);
      CHECK(output_shape.has_layout())
          << "Expecting output shape of entry computation to have a layout.";
      if (output_shape.layout().memory_space() == kHostMemorySpaceColor) {
        VLOG(2) << absl::StreamFormat(
            "Memory offloaded starting from %s is output streamed",
            starting_instruction_and_index.ToString());
        continue;
      } else {
        return absl::InvalidArgumentError(
            absl::StrFormat("Tensor which is moved to host (starting from %s) "
                            "is returned from the entry computation but the "
                            "layout for this output is not set to host memory.",
                            starting_instruction->name()));
      }
    }
    // Push successors onto the queue to be visited.
    TF_ASSIGN_OR_RETURN(const std::vector<InstructionAndShapeIndex> successors,
                        GetSuccessors(instruction_and_shape_index));
    for (const InstructionAndShapeIndex& successor : successors) {
      queue.push(successor);
    }
  }

  // Finished walking all host memory paths. Now we'll make all the necessary
  // changes.
  const bool set_buffers_changed = SetBuffersToMemorySpaceColor(
      buffers_to_set_to_host_memory, kHostMemorySpaceColor);
  changed = changed || set_buffers_changed;

  for (HloInstruction* dus : dynamic_update_slices) {
    // Create a host AllocateBuffer instruction which this DynamicUpdateSlice
    // will update-slice into.
    TF_RETURN_IF_ERROR(CreateAllocateBufferForDynamicUpdateSlice(dus));
    changed = true;
  }

  if (insert_copy_before) {
    const auto predecessors = GetPredecessors(starting_instruction_and_index);
    CHECK_EQ(predecessors.size(), 1);
    TF_ASSIGN_OR_RETURN(bool inserted_copy,
                        InsertCopyBetween(predecessors.front(),
                                          starting_instruction_and_index));
    changed = changed || inserted_copy;
  }

  // Insert copies to move to device.
  for (HloInstruction* custom_call : custom_calls_to_insert_copies_before) {
    HloInstruction* data_to_copy = custom_call->mutable_operand(0);
    HloInstruction* copy_to_device =
        data_to_copy->parent()->AddInstruction(HloInstruction::CreateUnary(
            data_to_copy->shape(), HloOpcode::kCopy, data_to_copy));
    SetMemorySpace(copy_to_device->mutable_shape(),
                   Layout::kDefaultMemorySpace);
    VLOG(1) << absl::StreamFormat(
        "Inserted copy \"%s\" before custom call \"%s\"",
        copy_to_device->name(), custom_call->name());
    TF_RETURN_IF_ERROR(custom_call->ReplaceAllUsesWith(copy_to_device));
    changed = true;
  }

  // All host memory offloading has been completed. Remove MoveToHost custom
  // calls.
  for (HloInstruction* custom_call : mth_custom_calls_to_remove) {
    VLOG(1) << absl::StreamFormat("Removing MoveToHost custom call \"%s\"",
                                  custom_call->name());
    TF_RETURN_IF_ERROR(
        custom_call->ReplaceAllUsesWith(custom_call->mutable_operand(0)));
    TF_RETURN_IF_ERROR(custom_call->parent()->RemoveInstruction(custom_call));
    changed = true;
  }

  for (HloInstruction* slice : slices_to_dynamify) {
    TF_ASSIGN_OR_RETURN(HloInstruction * dynamic_slice, DynamifySlice(slice));
    // We've already validated this slice. Since we're changing it to a dynamic
    // slice, save the new dynamic slice so that we don't try to validate it
    // again.
    validated_slices_.insert(dynamic_slice);
    changed = true;
  }

  return changed;
}

absl::StatusOr<bool> HostOffloader::HandleInputStreaming(
    HloComputation* entry_computation) {
  bool changed = false;
  const ComputationLayout& entry_computation_layout =
      entry_computation->parent()->entry_computation_layout();

  for (int i = 0; i < entry_computation_layout.parameter_count(); ++i) {
    if (entry_computation_layout.parameter_shape(i).IsToken()) {
      LOG(WARNING) << "Token parameters are not supported for streaming.";
      continue;
    }
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        entry_computation_layout.parameter_shape(i),
        [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.has_layout() &&
              subshape.layout().memory_space() == kHostMemorySpaceColor) {
            HloInstruction* parameter_instruction =
                entry_computation->parameter_instruction(i);
            VLOG(1) << "Host parameter streamed into program with shape: "
                    << subshape.ToString(/*print_layout=*/true) << " at index "
                    << index.ToString();
            TF_ASSIGN_OR_RETURN(
                bool result,
                WalkDownHostMemoryOffloadPaths(
                    InstructionAndShapeIndex(parameter_instruction, index),
                    /*insert_copy_before=*/false));
            changed = changed || result;
          }
          return absl::OkStatus();
        }));
  }
  return changed;
}

absl::StatusOr<bool> HostOffloader::HandleMoveToHostCustomCall(
    HloInstruction* custom_call_instruction) {
  if (already_visited_move_to_host_custom_calls_.contains(
          custom_call_instruction)) {
    return false;
  }
  VLOG(1) << "Offloading " << custom_call_instruction->operand(0)->name()
          << " to host.";
  TF_ASSIGN_OR_RETURN(
      std::vector<InstructionAndShapeIndex> starting_instruction_and_shapes,
      GetStartingInstructions(custom_call_instruction));
  if (starting_instruction_and_shapes.empty()) {
    // Either:
    //  1. This custom call has no users.
    //  2. It is the root of the entry computation.
    // In the case of 1, there is nothing to do. You could argue that we should
    // still copy the data to the host, as it is side effecting. However, that
    // would be wasteful, so we won't do it. In the case of 2, we'll simply
    // insert a copy to host and replace the root instruction with that.
    if (custom_call_instruction == custom_call_instruction->GetModule()
                                       ->entry_computation()
                                       ->root_instruction()) {
      HloInstruction* data_to_copy =
          custom_call_instruction->mutable_operand(0);
      HloInstruction* copy_to_host =
          data_to_copy->parent()->AddInstruction(HloInstruction::CreateUnary(
              data_to_copy->shape(), HloOpcode::kCopy, data_to_copy));
      SetMemorySpace(copy_to_host->mutable_shape(), kHostMemorySpaceColor);
      TF_RETURN_IF_ERROR(
          custom_call_instruction->ReplaceAllUsesWith(copy_to_host));
      VLOG(2) << absl::StreamFormat(
          "Custom call \"%s\" is entry computation root. Inserted copy \"%s\" "
          "and replaced root instruction.",
          custom_call_instruction->name(), copy_to_host->name());
    }
  }

  // Walk down the graph from each starting instruction.
  for (const InstructionAndShapeIndex& starting_instruction_and_shape :
       starting_instruction_and_shapes) {
    const bool should_insert_copy_before_instruction =
        starting_instruction_and_shape.instruction->opcode() !=
        HloOpcode::kDynamicUpdateSlice;
    TF_ASSIGN_OR_RETURN(
        bool result,
        WalkDownHostMemoryOffloadPaths(starting_instruction_and_shape,
                                       should_insert_copy_before_instruction));
    (void)result;  // This function *will* change the HloModule. We don't care
                   // if WalkDownHostMemoryOffloadPaths changed it or not.
  }

  already_visited_move_to_host_custom_calls_.insert(custom_call_instruction);

  // Remove custom call.
  VLOG(2) << absl::StreamFormat("Removing MoveToHost custom call \"%s\"",
                                custom_call_instruction->name());
  TF_RETURN_IF_ERROR(custom_call_instruction->ReplaceAllUsesWith(
      custom_call_instruction->mutable_operand(0)));
  TF_RETURN_IF_ERROR(custom_call_instruction->parent()->RemoveInstruction(
      custom_call_instruction));
  return true;
}

absl::StatusOr<bool> HostOffloader::HandleMoveToDeviceCustomCall(
    HloInstruction* custom_call_instruction) {
  VLOG(2) << absl::StreamFormat("Removing MoveToDevice custom call \"%s\"",
                                custom_call_instruction->name());
  TF_RETURN_IF_ERROR(custom_call_instruction->ReplaceAllUsesWith(
      custom_call_instruction->mutable_operand(0)));
  TF_RETURN_IF_ERROR(custom_call_instruction->parent()->RemoveInstruction(
      custom_call_instruction));
  move_to_device_custom_calls_to_remove_.insert(custom_call_instruction);
  return true;
}

absl::StatusOr<bool> HostOffloader::InsertCopyBetween(
    const InstructionAndShapeIndex& before_instruction_and_index,
    const InstructionAndShapeIndex& after_instruction_and_index) {
  bool changed = false;
  HloInstruction* after_instruction = after_instruction_and_index.instruction;

  // Get a list of instructions to insert copies before. Normally, this is just
  // `after_instruction_and_index.instruction`, however, if this instruction is
  // a parameter, then we need to insert the copies before the call sites.
  std::vector<InstructionAndShapeIndex> instructions_to_insert_copies_before;
  if (after_instruction->opcode() == HloOpcode::kParameter) {
    // To insert a copy between an instruction and a parameter means we actually
    // want to insert a copy between the instruction and the call site of the
    // computation with this parameter.
    std::unique_ptr<CallGraph> call_graph =
        CallGraph::Build(after_instruction->GetModule());
    auto callers =
        call_graph->GetComputationCallers(after_instruction->parent());
    for (HloInstruction* caller : callers) {
      const auto indices =
          caller->OperandIndices(before_instruction_and_index.instruction);
      for (int64_t index : indices) {
        instructions_to_insert_copies_before.push_back(
            InstructionAndShapeIndex{caller, {index}});
      }
    }
  } else {
    // Instruction is not a parameter, replacement is straightforward.
    instructions_to_insert_copies_before.push_back(after_instruction_and_index);
  }

  // Insert a copy before each of the above instructions.
  for (const InstructionAndShapeIndex& instruction_and_index :
       instructions_to_insert_copies_before) {
    if (already_inserted_copy_before_.find(instruction_and_index) ==
        already_inserted_copy_before_.end()) {
      HloInstruction* data_to_copy = before_instruction_and_index.instruction;
      HloInstruction* copy_to_host;
      auto it = copies_created_after_.find(data_to_copy);
      if (it == copies_created_after_.end()) {
        // Don't have a copy yet; create it.
        copy_to_host =
            data_to_copy->parent()->AddInstruction(HloInstruction::CreateUnary(
                data_to_copy->shape(), HloOpcode::kCopy, data_to_copy));
        SetMemorySpace(copy_to_host->mutable_shape(), kHostMemorySpaceColor);
        copies_created_after_[data_to_copy] = copy_to_host;
      } else {
        // We already have a copy which feeds into this instruction.
        copy_to_host = it->second;
      }
      const int64_t operand_index =
          after_instruction_and_index.shape_index.empty()
              ? 0
              : after_instruction_and_index.shape_index.front();
      TF_RETURN_IF_ERROR(instruction_and_index.instruction->ReplaceOperandWith(
          operand_index, copy_to_host));
      VLOG(2) << absl::StreamFormat(
          "Inserted copy \"%s\" between \"%s\" and \"%s\"",
          copy_to_host->name(), before_instruction_and_index.ToString(),
          after_instruction_and_index.ToString());
      already_inserted_copy_before_.insert(instruction_and_index);
      changed = true;
    }
  }
  return changed;
}

absl::StatusOr<std::vector<InstructionAndShapeIndex>>
HostOffloader::GetStartingInstructions(
    HloInstruction* custom_call_instruction) {
  // We want to offload the single operand of this custom call to the host.
  // For each user, it either:
  // 1. Feeds into a DynamicUpdateSlice.
  // 2. Does "normal" memory offloading.
  std::vector<InstructionAndShapeIndex> result;
  std::queue<InstructionAndShapeIndex> queue;
  TF_ASSIGN_OR_RETURN(
      const std::vector<InstructionAndShapeIndex> successors_of_custom_call,
      GetSuccessors(InstructionAndShapeIndex(custom_call_instruction)));
  for (const InstructionAndShapeIndex& successor : successors_of_custom_call) {
    queue.push(successor);
  }
  while (!queue.empty()) {
    InstructionAndShapeIndex instruction_and_shape = queue.front();
    queue.pop();
    HloInstruction* current_instruction = instruction_and_shape.instruction;
    if (current_instruction->opcode() == HloOpcode::kDynamicUpdateSlice) {
      // Found a DynamicUpdateSlice.
      result.push_back(instruction_and_shape);
      continue;
    } else if (!InstructionIsAllowedBetweenMoveToHostAndDus(
                   current_instruction)) {
      // Found the start of "normal" memory offloading.
      result.push_back(instruction_and_shape);
      continue;
    } else {
      // Is a logical bitcast/reshape, we won't offload this yet.
    }
    TF_ASSIGN_OR_RETURN(const std::vector<InstructionAndShapeIndex> successors,
                        GetSuccessors(instruction_and_shape));
    for (const InstructionAndShapeIndex& successor : successors) {
      queue.push(successor);
    }
  }
  return result;
}

absl::Status HostOffloader::ValidateSliceLeadsToMoveToDeviceCustomCall(
    HloInstruction* slice) {
  if (validated_slices_.find(slice) != validated_slices_.end()) {
    // Already validated this one.
    return absl::OkStatus();
  }
  // Every host-to-device DynamicSlice/Slice must be followed by a MoveToDevice
  // custom call. This function verifiest that.
  CHECK(slice->opcode() == HloOpcode::kDynamicSlice ||
        slice->opcode() == HloOpcode::kSlice)
      << "This function must only be called with a slice or dynamic slice.";
  std::queue<InstructionAndShapeIndex> queue;
  TF_ASSIGN_OR_RETURN(
      const std::vector<InstructionAndShapeIndex> successors_of_slice,
      GetSuccessors(InstructionAndShapeIndex(slice)));
  for (const InstructionAndShapeIndex& successor : successors_of_slice) {
    queue.push(successor);
  }
  while (!queue.empty()) {
    InstructionAndShapeIndex instruction_and_shape = queue.front();
    queue.pop();
    HloInstruction* current_instruction = instruction_and_shape.instruction;
    if (current_instruction->opcode() == HloOpcode::kCustomCall &&
        current_instruction->custom_call_target() ==
            host_memory_offload_annotations::kMoveToDeviceCustomCallTarget) {
      // This path ended with the MoveToDevice custom call. This path is good.
      continue;
    }
    if (!InstructionIsAllowedBetweenDsAndMoveToDevice(current_instruction)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tensor which is moved to host and back to device (ending at \"%s\") "
          "has an invalid instruction (\"%s\") between DynamicSlice/Slice and "
          "the MoveToDevice custom call.",
          slice->name(), current_instruction->name()));
    }
    TF_ASSIGN_OR_RETURN(const std::vector<InstructionAndShapeIndex> successors,
                        GetSuccessors(instruction_and_shape));
    for (const InstructionAndShapeIndex& successor : successors) {
      queue.push(successor);
    }
  }
  validated_slices_.insert(slice);
  return absl::OkStatus();
}

absl::Status HostOffloader::CreateAllocateBufferForDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  if (dynamic_update_slices_already_allocated_.find(dynamic_update_slice) !=
      dynamic_update_slices_already_allocated_.end()) {
    // Already added an AllocateBuffer for this DynamicUpdateSlice.
    return absl::OkStatus();
  }
  VLOG(2) << absl::StreamFormat(
      "Creating a AllocateBuffer in host memory space for \"%s\"",
      dynamic_update_slice->name());
  // Walk the graph up. We expect to find a broadcast. Also, while walking up
  // the graph, set host memory space on everything between the AllocateBuffer
  // and the DynamicUpdateSlice.
  std::queue<InstructionAndShapeIndex> queue;
  queue.push(InstructionAndShapeIndex(dynamic_update_slice));
  bool found_broadcast = false;
  while (!queue.empty()) {
    InstructionAndShapeIndex instruction_and_shape = queue.front();
    queue.pop();
    VLOG(2) << absl::StreamFormat("Setting %s to have host memory space",
                                  instruction_and_shape.ToString());
    SetMemorySpace(ShapeUtil::GetMutableSubshape(
                       instruction_and_shape.instruction->mutable_shape(),
                       instruction_and_shape.shape_index),
                   kHostMemorySpaceColor);
    HloInstruction* instruction = instruction_and_shape.instruction;
    if (instruction->opcode() == HloOpcode::kParameter) {
      // If this is a parameter of a while_body, we also need to find the
      // matching parameter in the while_condition and set the memory spaces
      // there.
      std::unique_ptr<CallGraph> call_graph =
          CallGraph::Build(instruction->GetModule());
      const std::vector<HloInstruction*> callers =
          call_graph->GetComputationCallers(instruction->parent());
      for (HloInstruction* caller : callers) {
        if (caller->opcode() == HloOpcode::kWhile) {
          // This parameter belongs to a while.
          CHECK(caller->while_body() == instruction->parent())
              << "We assume that we're starting from the while body";
          HloComputation* while_condition_computation =
              caller->while_condition();
          CHECK(while_condition_computation->num_parameters() == 1)
              << "Expecting While to have just 1 parameter";
          HloInstruction* while_condition_parameter =
              while_condition_computation->parameter_instruction(0);
          VLOG(2) << absl::StreamFormat("Setting %s to have host memory space",
                                        while_condition_parameter->name());
          SetMemorySpace(ShapeUtil::GetMutableSubshape(
                             while_condition_parameter->mutable_shape(),
                             instruction_and_shape.shape_index),
                         kHostMemorySpaceColor);
          // Walk further down the graph and set the memory spaces of all uses
          // too. This includes verifying that no compute is done on the buffer.
          // Another, better way, to do this, is to walk down the graph starting
          // from the newly created AllocateBuffer and set everything visited as
          // host memory space.
          std::queue<InstructionAndShapeIndex> nested_queue;
          nested_queue.push(InstructionAndShapeIndex(
              while_condition_parameter, instruction_and_shape.shape_index));
          while (!nested_queue.empty()) {
            InstructionAndShapeIndex nested_instruction_and_shape =
                nested_queue.front();
            nested_queue.pop();
            if (!IsValidDuringPureMemoryOffload(
                    nested_instruction_and_shape.instruction)) {
              return absl::InvalidArgumentError(absl::StrFormat(
                  "Tensor which is moved to host is used by an invalid "
                  "instruction (\"%s\") during while condition body.",
                  nested_instruction_and_shape.instruction->name()));
            }
            SetMemorySpace(
                ShapeUtil::GetMutableSubshape(
                    nested_instruction_and_shape.instruction->mutable_shape(),
                    nested_instruction_and_shape.shape_index),
                kHostMemorySpaceColor);
            TF_ASSIGN_OR_RETURN(
                const std::vector<InstructionAndShapeIndex> successors,
                GetSuccessors(nested_instruction_and_shape));
            for (const InstructionAndShapeIndex& successor : successors) {
              nested_queue.push(successor);
            }
          }
        }
      }
    } else if (instruction->opcode() == HloOpcode::kDynamicUpdateSlice) {
      // The AllocateBuffer that we're about to create will suffice for every
      // DynamicUpdateSlice we pass through as we walk up the graph.
      dynamic_update_slices_already_allocated_.insert(instruction);
    }
    const std::vector<InstructionAndShapeIndex> predecessors =
        GetPredecessors(instruction_and_shape);
    for (const InstructionAndShapeIndex& predecessor : predecessors) {
      HloInstruction* predecessor_instruction = predecessor.instruction;
      if (predecessor_instruction->opcode() == HloOpcode::kBroadcast) {
        // Found a broadcast.
        found_broadcast = true;
        HloInstruction* broadcast_user = instruction_and_shape.instruction;
        const auto operand_indices =
            broadcast_user->OperandIndices(predecessor_instruction);
        CHECK(!operand_indices.empty())
            << "We could only have the broadcast as a predecessor if it is an "
               "operand of this instruction; something is wrong.";
        HloInstruction* allocate_buffer =
            predecessor_instruction->parent()->AddInstruction(
                HloInstruction::CreateCustomCall(
                    predecessor_instruction->shape(), {}, "AllocateBuffer"));
        VLOG(1) << absl::StreamFormat(
            "Created new AllocateBuffer instruction \"%s\"",
            allocate_buffer->ToString());
        SetMemorySpace(allocate_buffer->mutable_shape(), kHostMemorySpaceColor);
        for (int64_t index : operand_indices) {
          TF_RETURN_IF_ERROR(
              broadcast_user->ReplaceOperandWith(index, allocate_buffer));
        }
        if (predecessor_instruction->user_count() == 0) {
          // No remaining users. Remove the broadcast.
          VLOG(3) << absl::StreamFormat(
              "Broadcast \"%s\" has no remaining users; removing.",
              predecessor_instruction->name());
          TF_RETURN_IF_ERROR(
              predecessor_instruction->parent()->RemoveInstruction(
                  predecessor_instruction));
        }
      } else {
        queue.push(predecessor);
      }
    }
  }
  if (!found_broadcast) {
    return absl::InvalidArgumentError(
        absl::StrFormat("DynamicUpdateSlice \"%s\"'s first operand is not the "
                        "result of a broadcast.",
                        dynamic_update_slice->name()));
  }
  return absl::OkStatus();
}

absl::StatusOr<HloInstruction*> HostOffloader::DynamifySlice(
    HloInstruction* slice) {
  std::vector<HloInstruction*> start_constants;
  for (int64_t start : slice->slice_starts()) {
    HloInstruction* constant = slice->parent()->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(start)));
    start_constants.push_back(constant);
  }
  std::vector<int64_t> slice_sizes;
  slice_sizes.reserve(slice->slice_limits().size());
  for (int i = 0; i < slice->slice_limits().size(); ++i) {
    slice_sizes.push_back(slice->slice_limits()[i] - slice->slice_starts()[i]);
  }
  HloInstruction* new_ds =
      slice->parent()->AddInstruction(HloInstruction::CreateDynamicSlice(
          slice->shape(), slice->mutable_operand(0), start_constants,
          slice_sizes));
  TF_RETURN_IF_ERROR(slice->ReplaceAllUsesWith(new_ds));
  VLOG(2) << absl::StreamFormat(
      "Changed slice \"%s\" into dynamic slice \"%s\"", slice->name(),
      new_ds->name());
  TF_RETURN_IF_ERROR(slice->parent()->RemoveInstruction(slice));
  return new_ds;
}

absl::StatusOr<bool> HostOffloader::ApplySchedulingFix(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module));
  auto uses_parameter_buffer = [&](HloInstruction* hlo) {
    for (const HloBuffer* buffer : alias_analysis->ComputeBuffersAt(hlo)) {
      for (const HloValue* value : buffer->values()) {
        for (const HloPosition& pos : value->positions()) {
          if (absl::c_linear_search(hlo->parent()->parameter_instructions(),
                                    pos.instruction)) {
            return true;
          }
        }
      }
    }
    return false;
  };
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    if (computation == computation->parent()->entry_computation()) {
      continue;
    }
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kDynamicUpdateSlice) {
        continue;
      }
      if (instruction->shape().layout().memory_space() !=
          kHostMemorySpaceColor) {
        continue;
      }
      // Replace DynamicUpdateSlice's 1st operand with a copy in case it
      // used parameter buffer directly because in case of aliasing with loop
      // parameters control dependencies can mess with scheduling.
      HloInstruction* operand = instruction->mutable_operand(1);
      if (uses_parameter_buffer(operand)) {
        HloInstruction* copy =
            instruction->parent()->AddInstruction(HloInstruction::CreateUnary(
                operand->shape(), HloOpcode::kCopy, operand));
        VLOG(5) << "Added copy " << std::quoted(copy->name())
                << " for DynamicUpdateSlice " << instruction->name()
                << "'s 1st operand " << operand->name();
        TF_RETURN_IF_ERROR(instruction->ReplaceOperandWith(1, copy));
        changed = true;
      }
    }
  }
  return changed;
}

absl::StatusOr<bool> HostOffloader::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  TF_ASSIGN_OR_RETURN(const bool input_streaming_changed_module,
                      HandleInputStreaming(module->entry_computation()));
  changed = changed || input_streaming_changed_module;

  // Since we're modifying the graph as we iterate over it, any time we change
  // it, we need to re-run the loop.
  bool changed_in_loop;
  do {
    changed_in_loop = false;
    for (HloComputation* computation :
         module->MakeComputationPostOrder(execution_threads)) {
      for (HloInstruction* instruction :
           computation->MakeInstructionPostOrder()) {
        if (instruction->IsCustomCall(
                host_memory_offload_annotations::kMoveToHostCustomCallTarget)) {
          TF_ASSIGN_OR_RETURN(changed_in_loop,
                              HandleMoveToHostCustomCall(instruction));
          if (changed_in_loop) {
            changed = true;
            break;
          }
        }
      }
      if (changed_in_loop) {
        break;
      }
    }
  } while (changed_in_loop);

  // Remove all MoveToDevice custom calls.
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->IsCustomCall(
              host_memory_offload_annotations::kMoveToDeviceCustomCallTarget)) {
        TF_ASSIGN_OR_RETURN(bool result,
                            HandleMoveToDeviceCustomCall(instruction));
        changed = changed || result;
      }
    }
  }

  TF_ASSIGN_OR_RETURN(bool applied_scheduling_fix,
                      ApplySchedulingFix(module, execution_threads));
  changed = changed || applied_scheduling_fix;

  // Finally, run CSE to do a little cleanup.
  HloCSE cse(/*is_layout_sensitive=*/true);
  TF_ASSIGN_OR_RETURN(bool cse_changed, cse.Run(module, execution_threads));
  changed = changed || cse_changed;

  return changed;
}

}  // namespace xla

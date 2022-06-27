/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/async_op_canonicalizer.h"

namespace xla {

namespace {

struct AsyncGroup {
  std::optional<int64_t> id;
  std::vector<HloInstruction*> instructions;
};

StatusOr<bool> CreateAsyncGroups(HloModule* module,
                                 std::vector<AsyncGroup>& async_groups) {
  absl::flat_hash_map<int64_t, AsyncGroup*> async_groups_by_id;
  absl::flat_hash_map<const HloInstruction*, AsyncGroup*>
      async_groups_by_instruction;

  for (const HloComputation* computation :
       module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kAsyncStart) {
        std::optional<int64_t> group_id = instruction->async_group_id();
        // We expect that there weren't any other async-starts with the same
        // group id. Treat it as an error in case there is a collision.
        TF_RET_CHECK(!group_id.has_value() ||
                     !async_groups_by_id.contains(*group_id))
            << "The group id was taken by another group already.";
        async_groups.push_back({group_id, {instruction}});
        async_groups_by_instruction[instruction] = &async_groups.back();
        if (group_id.has_value()) {
          async_groups_by_id[*group_id] = &async_groups.back();
        }
      } else if (instruction->opcode() == HloOpcode::kAsyncUpdate ||
                 instruction->opcode() == HloOpcode::kAsyncDone) {
        // We expect the instruction group id to match the operand's id.
        TF_RET_CHECK(instruction->async_group_id() ==
                     instruction->operand(0)->async_group_id());
        // Use the operand to find the async group (not the group id) because
        // the instruction might not have a group id assigned yet.
        auto async_group_it =
            async_groups_by_instruction.find(instruction->operand(0));
        TF_RET_CHECK(async_group_it != async_groups_by_instruction.end());
        AsyncGroup* async_group = async_group_it->second;
        async_group->instructions.push_back(instruction);
        async_groups_by_instruction[instruction] = async_group;
      }
    }
  }

  // Assign ids to async groups that don't have one.
  int64_t next_id = 0;
  auto get_next_id = [&]() {
    while (async_groups_by_id.contains(next_id)) {
      ++next_id;
    }
    return next_id;
  };
  bool modified = false;
  for (AsyncGroup& async_group : async_groups) {
    if (!async_group.id.has_value()) {
      async_group.id = get_next_id();
      async_groups_by_id[*async_group.id] = &async_group;
    }
    for (HloInstruction* instruction : async_group.instructions) {
      modified |= async_group.id != instruction->async_group_id();
      instruction->set_async_group_id(async_group.id);
    }
  }

  return modified;
}

}  // namespace

StatusOr<bool> AsyncOpCanonicalizer::Run(HloModule* module) {
  XLA_VLOG_LINES(
      1, module->ToString(HloPrintOptions().set_syntax_sugar_async_ops(false)));

  std::vector<AsyncGroup> async_groups;
  TF_ASSIGN_OR_RETURN(bool modified, CreateAsyncGroups(module, async_groups));

  for (const AsyncGroup& async_group : async_groups) {
    HloComputation* computation =
        async_group.instructions[0]->async_wrapped_computation();
    for (int i = 1; i < async_group.instructions.size(); ++i) {
      HloInstruction* instruction = async_group.instructions[i];
      if (instruction->async_wrapped_computation() != computation) {
        instruction->async_wrapped_computation()->RemoveAsyncInstruction(
            instruction);
        instruction->ReplaceCalledComputations(
            [&](HloComputation*) { return computation; });
        computation->AddAsyncInstruction(instruction);
      }
    }
  }
  XLA_VLOG_LINES(
      1, module->ToString(HloPrintOptions().set_syntax_sugar_async_ops(false)));
  return modified;
}

}  // namespace xla

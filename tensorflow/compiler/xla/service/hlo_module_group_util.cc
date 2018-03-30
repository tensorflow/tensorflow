/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_module_group_util.h"

#include <algorithm>
#include <list>
#include <queue>
#include <stack>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

std::vector<HloInstruction*> HloModuleGroupUtil::GlobalPredecessors(
    HloInstruction* instruction) {
  std::vector<HloInstruction*> predecessors;

  // Adds to the unique predecessors list and also add companion instructions
  // if the given predecessor has those.
  auto add_unique_predecessor = [&](HloInstruction* predecessor) {
    if (std::find(predecessors.begin(), predecessors.end(), predecessor) !=
        predecessors.end()) {
      return;
    }
    if (!metadata_.IsCompanionInstruction(predecessor)) {
      predecessors.push_back(predecessor);
      return;
    }
    for (HloInstruction* companion : metadata_.Companions(predecessor)) {
      predecessors.push_back(companion);
    }
  };

  // If the given instruction is a companion instruction, we need to find the
  // predecessors of all of its companion instructions.
  std::vector<HloInstruction*> instruction_group;
  if (metadata_.IsCompanionInstruction(instruction)) {
    for (HloInstruction* companion : metadata_.Companions(instruction)) {
      instruction_group.push_back(companion);
    }
  } else {
    instruction_group.push_back(instruction);
  }

  for (HloInstruction* hlo : instruction_group) {
    for (HloInstruction* operand : hlo->operands()) {
      add_unique_predecessor(operand);
    }
    for (HloInstruction* control_predecessor : hlo->control_predecessors()) {
      add_unique_predecessor(control_predecessor);
    }
  }
  if (instruction->opcode() == HloOpcode::kRecvDone) {
    // Send is a remote predecessor of RecvDone.
    HloInstruction* send = metadata_.GetChannel(instruction->channel_id()).send;
    add_unique_predecessor(send);
  }
  if (instruction->opcode() == HloOpcode::kSend) {
    // Recv is a remote predecessor of Send.
    HloInstruction* recv_done =
        metadata_.GetChannel(instruction->channel_id()).recv_done;
    CHECK(recv_done->opcode() == HloOpcode::kRecvDone);
    CHECK_EQ(recv_done->operand_count(), 1);
    HloInstruction* recv = recv_done->mutable_operand(0);
    add_unique_predecessor(recv);
  }
  return predecessors;
}

std::vector<HloInstruction*> HloModuleGroupUtil::GlobalSuccessors(
    HloInstruction* instruction) {
  std::vector<HloInstruction*> successors;

  // Adds to the unique successors list and also add companion instructions
  // if the given successor has those.
  auto add_unique_successor = [&](HloInstruction* successor) {
    if (std::find(successors.begin(), successors.end(), successor) !=
        successors.end()) {
      return;
    }
    if (!metadata_.IsCompanionInstruction(successor)) {
      successors.push_back(successor);
      return;
    }
    for (HloInstruction* companion : metadata_.Companions(successor)) {
      successors.push_back(companion);
    }
  };

  // If the given instruction is a companion instruction, we need to find the
  // successors of all of its companion instructions.
  std::vector<HloInstruction*> instruction_group;
  if (metadata_.IsCompanionInstruction(instruction)) {
    for (HloInstruction* companion : metadata_.Companions(instruction)) {
      instruction_group.push_back(companion);
    }
  } else {
    instruction_group.push_back(instruction);
  }

  for (HloInstruction* hlo : instruction_group) {
    for (HloInstruction* user : hlo->users()) {
      add_unique_successor(user);
    }
    for (HloInstruction* control_successor : hlo->control_successors()) {
      add_unique_successor(control_successor);
    }
  }
  if (instruction->opcode() == HloOpcode::kRecv) {
    // Send is a remote successor of Recv.
    const HloInstruction* recv_done = instruction->users().front();
    CHECK(recv_done->opcode() == HloOpcode::kRecvDone);
    HloInstruction* send = metadata_.GetChannel(instruction->channel_id()).send;
    add_unique_successor(send);
  }
  if (instruction->opcode() == HloOpcode::kSend) {
    // RecvDone is a remote successor of Send.
    HloInstruction* recv_done =
        metadata_.GetChannel(instruction->channel_id()).recv_done;
    add_unique_successor(recv_done);
  }
  return successors;
}

std::vector<HloInstruction*> HloModuleGroupUtil::RootInstructions(
    tensorflow::gtl::ArraySlice<HloComputation*> computations) {
  std::vector<HloInstruction*> roots;
  for (HloComputation* computation : computations) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (GlobalSuccessors(instruction).empty()) {
        roots.push_back(instruction);
      }
    }
  }
  return roots;
}

Status HloModuleGroupUtil::VisitTopologicalOrder(
    VisitStates* visit_state, const VisitFunction& visit_function,
    HloInstruction* root) {
  // Stack of HLO instructions visited in DFS order.
  std::stack<HloInstruction*> stack;
  stack.push(root);

  while (!stack.empty()) {
    HloInstruction* hlo = stack.top();

    // Find the instruction group of the currently visited instruction. The
    // instruction group represents all companion instructions of the
    // current instruction, and are considered to be a single entity for the
    // purpose of the traversal (i.e., they must always be in the same visit
    // state).
    std::vector<HloInstruction*> instruction_group;
    if (metadata_.IsCompanionInstruction(hlo)) {
      for (HloInstruction* companion : metadata_.Companions(hlo)) {
        instruction_group.push_back(companion);
      }
    } else {
      instruction_group.push_back(hlo);
    }

    if ((*visit_state)[hlo] == VisitState::kVisited) {
      // All instructions in the group must be in the same state.
      for (HloInstruction* instruction : instruction_group) {
        TF_RET_CHECK((*visit_state)[instruction] == VisitState::kVisited);
      }
      stack.pop();
      continue;
    }

    if ((*visit_state)[hlo] == VisitState::kVisiting) {
      TF_RETURN_IF_ERROR(visit_function(hlo, instruction_group));

      // Set the visit state of all instructions in the group to kVisited.
      for (HloInstruction* instruction : instruction_group) {
        TF_RET_CHECK((*visit_state)[instruction] == VisitState::kVisiting);
        (*visit_state)[instruction] = VisitState::kVisited;
      }
      stack.pop();
      continue;
    }

    // Set the visit state of all instructions in the group to kVisiting.
    for (HloInstruction* instruction : instruction_group) {
      TF_RET_CHECK((*visit_state)[instruction] == VisitState::kNotVisited)
          << instruction->ToString();
      (*visit_state)[instruction] = VisitState::kVisiting;
    }

    // For each instruction in the group, visit its predecessors (operands,
    // control predecessors and remote predecessors).
    for (HloInstruction* instruction : instruction_group) {
      for (HloInstruction* predecessor : GlobalPredecessors(instruction)) {
        // Visiting a node that is already being visited implies that there is
        // a cycle. Generate an error with the list of instructions in the
        // cycle.
        if ((*visit_state)[predecessor] == VisitState::kVisiting) {
          string cyclic_instructions;
          for (const auto& state : *visit_state) {
            if (state.second == VisitState::kVisiting) {
              tensorflow::strings::StrAppend(&cyclic_instructions,
                                             state.first->ToString(), "\n");
            }
          }
          // TODO(b/64305524): Improve the error message to print out the
          // instructions in a deterministic order that forms the cycle.
          return FailedPrecondition(
              "Cross-computation cycle detected via communicating nodes. The "
              "cycle contains the node %s. The cycle is found among the "
              "following nodes. Note that the order of the nodes is arbitrary "
              "and that the list may include nodes that are not part of the "
              "cycle.\n%s",
              predecessor->ToString().c_str(), cyclic_instructions.c_str());
        }
        stack.push(predecessor);
      }
    }
  }

  return Status::OK();
}

Status HloModuleGroupUtil::VerifyComputations(
    tensorflow::gtl::ArraySlice<HloComputation*> computations) {
  auto visit_function =
      [&](HloInstruction* instruction,
          const std::vector<HloInstruction*>& instruction_group) {
        return Status::OK();
      };
  int64 instructions_count = 0;
  VisitStates visit_states;
  for (HloComputation* computation : computations) {
    // Visit all instructions, and not just from the root instruction of the
    // computation. This allows us to detect dead cycles (i.e., cycles that
    // are not reachable from the root) or to enforce an order for the
    // communication instructions that are not reachable from any roots.
    for (HloInstruction* instruction : computation->instructions()) {
      TF_RETURN_IF_ERROR(
          VisitTopologicalOrder(&visit_states, visit_function, instruction));
    }
    instructions_count += computation->instruction_count();
  }

  // Check if all instructions are visited and are in the visited state.
  TF_RET_CHECK(visit_states.size() == instructions_count);
  for (auto& state : visit_states) {
    TF_RET_CHECK(state.second == VisitState::kVisited);
  }

  return Status::OK();
}

StatusOr<std::unique_ptr<HloReachabilityMap>>
HloModuleGroupUtil::ComputeReachability(
    tensorflow::gtl::ArraySlice<HloComputation*> computations) {
  std::list<HloInstruction*> post_order;
  auto visit_function =
      [&](HloInstruction* instruction,
          const std::vector<HloInstruction*>& instruction_group) {
        post_order.insert(post_order.end(), instruction_group.begin(),
                          instruction_group.end());
        return Status::OK();
      };
  HloModuleGroupUtil::VisitStates visit_states;
  for (HloInstruction* root : RootInstructions(computations)) {
    TF_RETURN_IF_ERROR(
        VisitTopologicalOrder(&visit_states, visit_function, root));
  }
  auto reachability = absl::make_unique<HloReachabilityMap>(post_order);
  for (HloInstruction* hlo : post_order) {
    reachability->SetReachabilityToUnion(GlobalPredecessors(hlo), hlo);
  }
  return std::move(reachability);
}

void HloModuleGroupUtil::UpdateReachabilityThroughInstruction(
    HloInstruction* instruction, HloReachabilityMap* reachability_map) {
  std::queue<HloInstruction*> worklist;
  worklist.push(instruction);

  while (!worklist.empty()) {
    HloInstruction* item = worklist.front();
    worklist.pop();
    if (reachability_map->SetReachabilityToUnion(GlobalPredecessors(item),
                                                 item)) {
      for (HloInstruction* successor : GlobalSuccessors(item)) {
        worklist.push(successor);
      }
    }
  }
}

}  // namespace xla

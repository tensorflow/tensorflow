/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/slice_delaying.h"
#include <algorithm>
#include <utility>
#include <set>
#include <vector>
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

namespace {

class SliceDelayer {
 public:
  // Returns whether the instruction has been visited and computed change cost
  bool IsVisited(const HloInstruction* instruction) const;

  // Returns whether the slices are delayed successfully.
  Status MergeWithPeers(const HloInstruction* inst);

  // Elimenate dead instructions.
  void EliminateDeadInstructions();

  // Clear containers.
  void Clear();

 private:
  // Collects true operands of inst.
  StatusOr<std::vector<HloInstruction*>> GetTrueOperands(
      const HloInstruction* inst);

  // Collects true users of operands with the same opcode of inst, and update
  // visited_
  StatusOr<std::vector<HloInstruction*>> GetTrueUsers(
      const HloInstruction* inst,
      const std::vector<HloInstruction*>& operands);

  // Generate new operation instead of sliced operations, then slice the result.
  // Record the stale users and slices for prepare removing
  void GenerateNewOp(const std::vector<HloInstruction*>& operands,
      const std::vector<HloInstruction*>& users);

  std::set<const HloInstruction*> visited_;

  std::set<HloInstruction*> slices_;

  std::set<HloInstruction*> removed_;
};

// Computes the cost of implimentation of delaying slice, and returns whether
// it should be changed.
bool ShouldReplace(const std::vector<HloInstruction*>& operands,
    const std::vector<HloInstruction*>& users) {
  // operands and user have the same shape because of elementwise operation
  int64 sum = 0;
  for (HloInstruction* user : users) {
    sum += xla::ShapeUtil::ElementsIn(user->shape());
  }
  return sum >= xla::ShapeUtil::ElementsIn(operands[0]->shape());
}

}  // namespace

bool SliceDelayer::IsVisited(const HloInstruction* instruction) const {
  return std::find(visited_.begin(), visited_.end(), instruction)
      != visited_.end();
}

// =================================Before======================================
//
//       +-----operand-----+        <operands>
//       |                 |
//       v                 v
// bundled-slice     bundled-slice   <bundled-slices>
//       |                 |
//       v                 v
//      user              user      <users>
//
// ==================================After======================================
//
//            operand
//               |
//               v
//       +----new-user-----+
//       |                 |
//       v                 v
// bundled-slice     bundled-slice   <bundled-slices>
//
Status SliceDelayer::MergeWithPeers(const HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> operands,
                      GetTrueOperands(inst));

  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> users,
                      GetTrueUsers(inst, operands));

  // Change HLO graph
  GenerateNewOp(operands, users);
  return Status::OK();
}

void SliceDelayer::EliminateDeadInstructions() {
  // Remove dead users
  for (auto inst : removed_) {
    VLOG(10) << "Delete: " << inst->ToString();
    inst->parent()->RemoveInstruction(inst);
  }

  // Remove dead slices
  for (auto inst : slices_) {
    if (inst->user_count() == 0) {
      VLOG(10) << "Delete: " << inst->ToString();
      inst->parent()->RemoveInstruction(inst);
    }
  }
}

void SliceDelayer::Clear() {
  visited_.clear();
  slices_.clear();
  removed_.clear();
}

StatusOr<std::vector<HloInstruction*>> SliceDelayer::GetTrueOperands(
    const HloInstruction* inst) {
  std::vector<HloInstruction*> operands;
  // Check operand:
  // the inst's operand should be a slice of the true operand.
  // the operands-vector keeps the true-operands.
  for (HloInstruction* slice : inst->operands()) {
    if (slice->opcode() != HloOpcode::kSlice) {
      visited_.insert(inst);
      return tensorflow::errors::FailedPrecondition(
          "Operation's operand should be slice");
    }
    HloInstruction* operand = slice->mutable_operand(0);
    operands.push_back(operand);
  }

  // No operands, skip this instruction
  if (operands.empty()) {
    return tensorflow::errors::FailedPrecondition(
        "Operation has no true operands");
  }

  // Check operands:
  // true operands should have the same shape.(because of elementwise)
  const Shape shape = operands[0]->shape();
  for (const HloInstruction* operand : operands) {
    // Only support element-wise now
    if (!ShapeUtil::Equal(operand->shape(), shape)) {
      visited_.insert(inst);
      return tensorflow::errors::FailedPrecondition(
          "Operation's true operand should be the same shape");
    }
  }
  // operands should slice from the same location of true operands
  const HloInstruction* operand0 = inst->operand(0);
  for (const HloInstruction* operand : inst->operands()) {
    if (operand0->slice_starts() != operand->slice_starts() ||
        operand0->slice_limits() != operand->slice_limits() ||
        operand0->slice_strides() != operand->slice_strides()) {
      visited_.insert(inst);
      return tensorflow::errors::FailedPrecondition(
          "Operation's true operand should be the same shape");
    }
  }
  return operands;
}

StatusOr<std::vector<HloInstruction*>> SliceDelayer::GetTrueUsers(
    const HloInstruction* inst,
    const std::vector<HloInstruction*>& operands) {
  std::vector<HloInstruction*> users;
  HloInstruction* operand0 = operands[0];

  for (const HloInstruction* slice_0 : operand0->users()) {
    // skip non-slice user
    if (slice_0->opcode() != HloOpcode::kSlice) {
      continue;
    }

    for (HloInstruction* user : slice_0->users()) {
      // user should be the same operation and same operand count as inst
      // skip the visited user to avoid redundant computation
      if (IsVisited(user) || user->opcode() != inst->opcode() ||
          user->operand_count() != inst->operand_count()) {
        continue;
      }

      // user's every operand should be a slice of the true operand in order
      bool isValidUser = true;
      for (int64 j = 0; j < operands.size(); ++j) {
        const HloInstruction* slice_j = user->operand(j);
        // the slice of operands should sliced from the same location (only for
        // elementwise)
        if (slice_j->opcode() != HloOpcode::kSlice ||
            slice_j->operand(0) != operands[j] ||
            slice_0->slice_starts() != slice_j->slice_starts() ||
            slice_0->slice_limits() != slice_j->slice_limits() ||
            slice_0->slice_strides() != slice_j->slice_strides()) {
          isValidUser = false;
          break;
        }
      }
      if (!isValidUser) {
        continue;
      }

      // found the user
      users.push_back(user);
      visited_.insert(user);
    }  // end for loop slice_0->users
  }  // end for loop operand0's slice users

  // calculate the cost. If the no user found or only few small users, skip this
  // instruction.
  if (users.empty()) {
    return tensorflow::errors::FailedPrecondition(
        "No found valid users");
  } else if (!ShouldReplace(operands, users)) {
    return tensorflow::errors::FailedPrecondition(
        "No Enough elements slice");
  } else {
    return users;
  }
}

void SliceDelayer::GenerateNewOp(const std::vector<HloInstruction*>& operands,
    const std::vector<HloInstruction*>& users) {
  // generate new ops
  const Shape shape = operands[0]->shape();
  HloComputation* computation = users[0]->parent();
  auto new_op = computation->AddInstruction(
      users[0]->CloneWithNewOperands(shape, operands));
  VLOG(10) << "Add NewOp: " << new_op->ToString();

  // replace the old user and its slice operands with new operation and its
  // slice user.
  for (HloInstruction* user : users) {
    const HloInstruction* slice = user->operand(0);
    auto new_user = computation->AddInstruction(
        slice->CloneWithNewOperands(user->shape(), {new_op}));
    VLOG(10) << "Add NewSlice: " << new_user->ToString()
            << "\nReplace: " << user->ToString();
    user->ReplaceAllUsesWith(new_user);

    for (HloInstruction* slice_operand : user->operands()) {
      slices_.insert(slice_operand);
    }
    removed_.insert(user);
  }  // end for users
}

StatusOr<bool> SliceDelaying::Run(HloModule* module) {
  VLOG(3) << "Run Pass: " << name();
  VLOG(10) << "before: " << name() << "\n" << module->ToString();
  SliceDelayer slice_delayer;
  bool changed = false;

  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction :
        computation->MakeInstructionPostOrder()) {
      // Skip the visited instruction tries merge elementwise instruction with
      // its peers.
      if (!slice_delayer.IsVisited(instruction) &&
          instruction->IsElementwise() && instruction->operand_count() != 0) {
        // TODO(xinan): more other instructions
        VLOG(10) << "Merge inst: " << instruction->ToString();
        changed |= slice_delayer.MergeWithPeers(instruction).ok();
      }
    }  // end for instructions in computation
  }  // end for computations in module

  // Clears dead nodes
  slice_delayer.EliminateDeadInstructions();
  slice_delayer.Clear();
  VLOG(10) << "after: " << name() << "\n" <<  module->ToString();
  return changed;
}

}  // namespace xla

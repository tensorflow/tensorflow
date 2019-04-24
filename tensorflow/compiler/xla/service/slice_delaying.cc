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
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

namespace {

class SliceDelayer {
 public:
  // Returns whether the instruction has been visited. The visited instruction
  // has been grouped in a true user group with its peers and it should not be
  // grouped in another true user group.
  bool IsVisited(const HloInstruction* instruction) const;

  // Returns whether the slices are delayed successfully.
  StatusOr<bool> MergeWithPeers(const HloInstruction* inst);

 private:
  // Collects true operands of inst. The inst's operands should all be slices,
  // and the true operands are the operands of slices. Returns true operand
  // vector if all true operands are found, or return an empty vector.
  StatusOr<std::vector<HloInstruction*>> GetTrueOperands(
      const HloInstruction* inst);

  // Collects the true user group included inst. The slice from true operand is
  // operand-slice. The true user's operands are all operand-slices.
  // The operand-slices of a true user should be sliced from the same range of
  // the true operands. It means the operand-slices have the same slice_starts,
  // slice_limits, and slice_strides.
  // The true users in a group have the same opcode and the same true operands
  // in order. Of course, the inst is a true user, and the others in the group
  // are the inst's peers.
  // Returns an empty vector if the cost of change is more than the profit.
  // Additionally, records the user which has been visited in a true user group,
  // and skips it in later traverse.
  StatusOr<std::vector<HloInstruction*>> GetTrueUsers(
      const HloInstruction* inst,
      const std::vector<HloInstruction*>& operands);

  // Generates the new user with the whole tensor and slices its output instead
  // of the true users with sliced tensor.
  // Records the stale true users and their operand-slices to prepare removing.
  void GenerateNewOp(const std::vector<HloInstruction*>& operands,
      const std::vector<HloInstruction*>& users);

  absl::flat_hash_set<const HloInstruction*> visited_;
};

// Computes the cost of implimentation of delaying slice, and returns whether
// it should be changed.
bool ShouldReplace(const std::vector<HloInstruction*>& operands,
    const std::vector<HloInstruction*>& users) {
  int64 sum = 0;
  // Sums the total element number of the true users.
  for (HloInstruction* user : users) {
    sum += ShapeUtil::ElementsIn(user->shape());
  }
  // Operand and user have the same element number in shape because of
  // elementwise operation, so the elements in operands[0] is the same as in
  // new user if new user is generated.
  // Compares the total elements in all true users and the elements of the new
  // user with whole shape.
  return sum >= xla::ShapeUtil::ElementsIn(operands[0]->shape());
}

}  // namespace

bool SliceDelayer::IsVisited(const HloInstruction* instruction) const {
  return visited_.contains(instruction);
}

// =================================Before======================================
//
//       +--true-operand---+         <operands>
//       |                 |
//       v                 v
// operand-slice     operand-slice   <bundled-slices>
//       |                 |
//       v                 v
//   true-user         true-user     <users>
//
// ==================================After======================================
//
//          true-operand
//               |
//               v
//       +----new-user-----+
//       |                 |
//       v                 v
//   user-slice       user-slice     <bundled-slices>
//
StatusOr<bool> SliceDelayer::MergeWithPeers(const HloInstruction* inst) {
  // Collects true operands of inst.
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> operands,
                      GetTrueOperands(inst));
  if (operands.empty()) {
    return false;
  }

  // Collects true users grouped togather with inst.
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> users,
                      GetTrueUsers(inst, operands));
  if (users.empty()) {
    return false;
  }

  // Change HLO graph.
  GenerateNewOp(operands, users);
  return true;
}

StatusOr<std::vector<HloInstruction*>> SliceDelayer::GetTrueOperands(
    const HloInstruction* inst) {
  std::vector<HloInstruction*> operands;
  // Check operand:
  // The inst's operand should be a slice of the true operand.
  // The operands-vector keeps all true operands of inst.
  for (HloInstruction* operand_slice : inst->operands()) {
    if (operand_slice->opcode() != HloOpcode::kSlice) {
      // If the operand is not operand-slice, returns empty vector.
      visited_.insert(inst);
      operands.clear();
      return operands;
    }
    // The operand-slice's operand is a true operand.
    HloInstruction* operand = operand_slice->mutable_operand(0);
    operands.push_back(operand);
  }

  // No true operands found. returns empty vector.
  if (operands.empty()) {
    visited_.insert(inst);
    return operands;
  }

  // Check operands:
  // True operands should have the same shape because inst is elementwise.
  const Shape shape = operands[0]->shape();
  for (const HloInstruction* operand : operands) {
    // Only support element-wise now
    if (!ShapeUtil::Compatible(operand->shape(), shape)) {
      visited_.insert(inst);
      operands.clear();
      return operands;
    }
  }
  // Inst's operand-slices should be sliced from the same location of true
  // operands.
  const HloInstruction* operand0 = inst->operand(0);
  for (const HloInstruction* operand : inst->operands()) {
    if (operand0->slice_starts() != operand->slice_starts() ||
        operand0->slice_limits() != operand->slice_limits() ||
        operand0->slice_strides() != operand->slice_strides()) {
      visited_.insert(inst);
      operands.clear();
      return operands;
    }
  }

  // Returns all true operands of inst.
  return operands;
}

StatusOr<std::vector<HloInstruction*>> SliceDelayer::GetTrueUsers(
    const HloInstruction* inst,
    const std::vector<HloInstruction*>& operands) {
  std::vector<HloInstruction*> users;
  HloInstruction* operand0 = operands[0];

  // Traverses a true operand's all operand-slices.
  for (const HloInstruction* operand_slice0 : operand0->users()) {
    // Skips not-slices
    if (operand_slice0->opcode() != HloOpcode::kSlice) {
      continue;
    }

    // The user of operand-slices is a candidate of true user.
    for (HloInstruction* user : operand_slice0->users()) {
      // The inst's peers should be the same operation and same operand count as
      // inst. The visited user is in another group of other true users. Skips
      // them to avoid useless match.
      if (IsVisited(user) || user->opcode() != inst->opcode() ||
          user->operand_count() != inst->operand_count()) {
        continue;
      }

      // Checks operand-slices:
      bool isTrueUser = true;
      for (int64 j = 0; j < operands.size(); ++j) {
        // User's operands should be operand-slices of the true operands in
        // order.
        // The operand-slices of the same true user should be sliced from the
        // same location of true operands. (because of elementwise)
        const HloInstruction* operand_slice = user->operand(j);
        if (operand_slice->opcode() != HloOpcode::kSlice ||
            operand_slice->operand(0) != operands[j] ||
            operand_slice0->slice_starts() != operand_slice->slice_starts() ||
            operand_slice0->slice_limits() != operand_slice->slice_limits() ||
            operand_slice0->slice_strides() != operand_slice->slice_strides()) {
          isTrueUser = false;
          break;
        }
      }
      if (!isTrueUser) {
        continue;
      }

      // Found the true user.
      users.push_back(user);
      visited_.insert(user);
    }
  }

  // Calculates the costs. If cost is more than profit, returns empty vector.
  if (!ShouldReplace(operands, users)) {
    users.clear();
  }
  return users;
}

void SliceDelayer::GenerateNewOp(const std::vector<HloInstruction*>& operands,
    const std::vector<HloInstruction*>& users) {
  // Generates new user.
  const Shape shape = operands[0]->shape();
  HloComputation* computation = users[0]->parent();
  auto new_user = computation->AddInstruction(
      users[0]->CloneWithNewOperands(shape, operands));
  VLOG(10) << "Add NewUser: " << new_user->ToString();

  // Replaces the true users with new user and its user-slices.
  for (HloInstruction* user : users) {
    const HloInstruction* operand_slice = user->operand(0);
    // Generates user slices of new user.
    auto user_slice = computation->AddInstruction(
        operand_slice->CloneWithNewOperands(user->shape(), {new_user}));
    VLOG(10) << "Add NewSlice: " << user_slice->ToString()
             << " Replace: " << user->ToString();
    // Replaces true users with user slices.
    user->ReplaceAllUsesWith(user_slice);
  }
}

StatusOr<bool> SliceDelaying::Run(HloModule* module) {
  VLOG(3) << "Run Pass: " << name();
  VLOG(10) << "before: " << name() << "\n" << module->ToString();
  SliceDelayer slice_delayer;
  bool changed = false;

  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction :
        computation->MakeInstructionPostOrder()) {
      // Skips the visited instruction and tries merge elementwise instruction
      // with its peers.
      if (!slice_delayer.IsVisited(instruction) &&
          instruction->IsElementwise() && instruction->operand_count() != 0) {
        // TODO(xinan): Supports more non-elementwise instructions.
        VLOG(10) << "Merge inst: " << instruction->ToString();
        TF_ASSIGN_OR_RETURN(bool success,
                            slice_delayer.MergeWithPeers(instruction));
        changed |= success;
      }
    }
  }

  VLOG(10) << "after: " << name() << "\n" <<  module->ToString();
  return changed;
}

}  // namespace xla

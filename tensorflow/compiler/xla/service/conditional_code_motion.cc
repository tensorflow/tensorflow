/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/conditional_code_motion.h"

#include <iterator>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {

namespace conditional_opt {

class BoundaryVisitor {
 public:
  // start with an existing conditional computation.
  explicit BoundaryVisitor(HloInstruction* conditional) {
    Boundary b(Boundary::Position::kInsideBranch);
    b.mutable_operands().push_back(conditional);
    worklist_.push_back(b);
  }
  // Start with an empty work list.
  BoundaryVisitor() {}
  // Get next boundary to visit.
  Boundary PopNextBoundary() {
    CHECK(!worklist_.empty());
    Boundary b = worklist_.front();
    worklist_.pop_front();
    // if b is already visited, it must have multiple users and is already in
    // new boundaries. Skip it. Only checking the first operand of b because b
    // is expected to have at least one operand, and all the operands in b
    // must be identical instructions from different branches for b to be moved.
    while (!worklist_.empty() && ContainsKey(visited_, b.operands()[0])) {
      b = worklist_.front();
      worklist_.pop_front();
    }
    visited_.insert(b.operands()[0]);
    return b;
  }
  void AddToWorkList(const Boundary& b) {
    CHECK(!b.operands().empty());
    worklist_.push_back(b);
  }

  bool HasNextBoundary() {
    while (!worklist_.empty()) {
      Boundary b = worklist_.front();
      if (!ContainsKey(visited_, b.operands()[0])) {
        break;
      }
      worklist_.pop_front();
    }
    return !worklist_.empty();
  }

 private:
  // worklist is the deque that contains instructions to be visited.
  std::deque<Boundary> worklist_;
  absl::flat_hash_set<HloInstruction*> visited_;
};

// Returns estimation of potential reuses carried by a given pair of
// instructions.  Use different integers to classify different levels
// of reuses This is used as a placeholder only, assuming all
// instructions can be fused to enable data reuses
int64 ReusesCarriedBy(HloInstruction* op, HloInstruction* user) {
  VLOG(1) << "ConditionalCodeMotion: Add reuses carried by instr: "
          << op->ToString() << "=>" << user->ToString() << "\n";
  switch (user->opcode()) {
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kTuple:
      return 0;
    default:
      break;
  }
  switch (op->opcode()) {
      // These instructions are lightweight and easy to fuse.
    case HloOpcode::kConstant:
    case HloOpcode::kGetTupleElement:
      return 0;
    default:
      // Assume fusion will not happen anyway if user count > 1)
      if (op->user_count() > 1) {
        return 0;
      }
      return 10;
  }
}

// Compare if the instructions to be visited at each branches are identical.
bool InstructionWithinBranchIdentical(
    const std::vector<HloInstruction*>& instructions,
    bool is_layout_sensitive) {
  // Identical includes the shape of each operands are equal.
  auto eq_operand = [&](const HloInstruction* a, const HloInstruction* b) {
    bool eq_operands = is_layout_sensitive
                           ? ShapeUtil::Equal(a->shape(), b->shape())
                           : ShapeUtil::Compatible(a->shape(), b->shape());
    return eq_operands;
  };

  auto eq_computations = [](const HloComputation* a, const HloComputation* b) {
    return *a == *b;
  };

  if (instructions.empty()) {
    return false;
  }

  if (instructions[0]->IsCrossModuleAllReduce()) {
    return std::all_of(
        instructions.begin(), instructions.end(),
        [&](HloInstruction* instruction) {
          if (!instruction->IsCrossModuleAllReduce()) {
            return false;
          }
          auto old_channel_id = instruction->channel_id();
          instruction->set_channel_id(instructions[0]->channel_id());
          bool eq_instructions = instructions[0]->Identical(
              *instruction, eq_operand, eq_computations, is_layout_sensitive);
          instruction->set_channel_id(old_channel_id);
          return eq_instructions;
        });
  }

  return std::all_of(instructions.begin(), instructions.end(),
                     [&](HloInstruction* instruction) {
                       return instructions[0]->Identical(
                           *instruction, eq_operand, eq_computations,
                           is_layout_sensitive);
                     });
}

// Copy the ith instruction in boundary to outside of conditional, or do the
// opposite (for moving in).
Status CopyInOrOutOfConditional(
    Boundary& boundary, int64 dest_index, HloComputation* parent,
    absl::flat_hash_map<HloInstruction*, Boundary>& hoisted_instructions) {
  CHECK(dest_index == 0 || boundary.IsOutsideBranch());
  HloInstruction* op = boundary.operands()[0];
  absl::InlinedVector<HloInstruction*, 4> new_operands;
  for (int i = 0; i < op->operands().size(); ++i) {
    auto op_i = op->operands()[i];
    VLOG(2) << "Looking for operand:" << op_i->ToString() << "\n";
    if (ContainsKey(hoisted_instructions, op_i)) {
      auto new_op_i =
          FindOrDie(hoisted_instructions, op_i).operands()[dest_index];
      VLOG(2) << "new operand:" << new_op_i->ToString() << "\n";
      new_operands.push_back(new_op_i);
    } else {
      CHECK(op_i->opcode() == HloOpcode::kConstant);
      auto new_op_i = parent->AddInstruction(op_i->Clone());
      VLOG(2) << "new operand:" << new_op_i->ToString() << "\n";
      new_operands.push_back(new_op_i);
    }
  }
  HloInstruction* new_instruction = parent->AddInstruction(
      op->CloneWithNewOperands(op->shape(), new_operands));
  VLOG(2) << "new instruction:" << new_instruction->ToString() << "\n";
  // Maps the instruction outside of conditional to the instruction
  // inside of the conditional.
  for (HloInstruction* op : boundary.operands()) {
    Boundary b2 = ContainsKey(hoisted_instructions, op)
                      ? hoisted_instructions[op]
                      : Boundary(boundary.IsOutsideBranch()
                                     ? Boundary::Position::kInsideBranch
                                     : Boundary::Position::kOutsideBranch);
    b2.mutable_operands().push_back(new_instruction);
    hoisted_instructions[op] = b2;
  }
  return Status::OK();
}

// Identify converts to be hoisted/rematerialized out of the branch
// computations.
absl::flat_hash_set<int64> FindSpecialConverts(HloInstruction* old_root,
                                               int branch_count,
                                               HloInstruction* conditional,
                                               bool is_layout_sensitive) {
  absl::flat_hash_set<int64> kspecial_convert;
  for (int64 operand_num = 0; operand_num < old_root->operand_count();
       ++operand_num) {
    if (old_root->operand(operand_num)->opcode() != HloOpcode::kConvert) {
      continue;
    }
    bool replica = true;
    HloInstruction* kspecial_convert_candidate =
        old_root->mutable_operand(operand_num);
    // Check whether an identical candidate appears in other branches
    for (int others = 1; others < branch_count; ++others) {
      HloInstruction* others_root =
          conditional->branch_computation(others)->root_instruction();
      bool eq_shape =
          is_layout_sensitive
              ? ShapeUtil::Equal(others_root->operand(operand_num)->shape(),
                                 kspecial_convert_candidate->shape())
              : ShapeUtil::Compatible(
                    others_root->operand(operand_num)->shape(),
                    kspecial_convert_candidate->shape());
      if ((others_root->operand(operand_num)->opcode() ==
           HloOpcode::kConvert) &&
          eq_shape) {
        // Nothing to be done.
      } else {
        replica = false;
        break;
      }
    }
    if (replica) {
      kspecial_convert.insert(operand_num);
    }
  }
  return kspecial_convert;
}

// Restructuring the conditional instruction as follows:
// i.e., %result = conditional() becomes
// x = conditional()
// y.{0..n} = gte(x, {0..n})
// z = tuple(y.0, y.1, ...y.n)
// Doing so ensures that we can accommodate the possible shape-change of the
// conditional when the instructions are hoisted.
Status RestructureConditionalInstruction(HloComputation* computation,
                                         HloInstruction* conditional) {
  HloInstruction* old_root = computation->root_instruction();
  std::vector<HloInstruction*> new_operands;
  int cur_index = 0;
  for (; cur_index < ShapeUtil::TupleElementCount(conditional->shape());
       ++cur_index) {
    new_operands.push_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            ShapeUtil::GetTupleElementShape(conditional->shape(), cur_index),
            conditional, cur_index)));
  }
  HloInstruction* new_tuple =
      computation->AddInstruction(HloInstruction::CreateTuple(new_operands));
  if (old_root == conditional) {
    computation->set_root_instruction(new_tuple);
  } else {
    std::vector<HloInstruction*> new_tuple_users;
    for (auto conditional_user : conditional->users()) {
      auto is_new_gte = absl::c_find_if(
          new_operands,
          [&](HloInstruction* instr) { return instr == conditional_user; });
      if (is_new_gte == new_operands.end()) {
        new_tuple_users.push_back(conditional_user);
      }
    }
    for (auto new_tuple_user : new_tuple_users) {
      TF_RETURN_IF_ERROR(
          conditional->ReplaceUseWith(new_tuple_user, new_tuple));
    }
  }
  VLOG(2) << "computation after root restructure:\n" << computation->ToString();
  return Status::OK();
}

StatusOr<bool> ConvertSpecialMove(HloInstruction* conditional,
                                  bool is_layout_sensitive) {
  int branch_count = conditional->branch_count();
  if (branch_count <= 0) {
    return false;
  }

  HloInstruction* old_root =
      conditional->branch_computation(0)->root_instruction();
  if (old_root->opcode() != HloOpcode::kTuple) {
    return false;
  } else {
    VLOG(2) << "BEFORE :" << conditional->parent()->parent()->ToString();
    // Identify the gte using `index'.
    auto find_gte = [](const HloInstruction* conditional_result,
                       int64 index) -> HloInstruction* {
      for (HloInstruction* instr : conditional_result->users()) {
        if (instr->opcode() != HloOpcode::kGetTupleElement) {
          return nullptr;
        }
        if (instr->tuple_index() == index) {
          return instr;
        }
      }
      return nullptr;
    };

    // Captures tuple indices refering to converts to be rematerialized/hoisted.
    absl::flat_hash_set<int64> kspecial_convert = FindSpecialConverts(
        old_root, branch_count, conditional, is_layout_sensitive);

    // Exit if we cannot find any converts to be hoisted.
    if (kspecial_convert.empty()) {
      return false;
    }

    TF_RETURN_IF_ERROR(
        RestructureConditionalInstruction(conditional->parent(), conditional));

    for (int branch = 0; branch < branch_count; branch++) {
      old_root = conditional->branch_computation(branch)->root_instruction();
      absl::flat_hash_map<HloInstruction*, int64> map_inst_to_tuple_index;
      std::vector<HloInstruction*> new_operands(old_root->operand_count());
      absl::flat_hash_set<HloInstruction*> to_hoist_set;

      for (int64 operand_num = 0; operand_num < old_root->operand_count();
           ++operand_num) {
        map_inst_to_tuple_index[old_root->mutable_operand(operand_num)] =
            operand_num;
      }
      for (int64 operand_num = 0; operand_num < old_root->operand_count();
           ++operand_num) {
        HloInstruction* hoist = old_root->mutable_operand(operand_num);
        if (!kspecial_convert.contains(operand_num)) {
          new_operands[operand_num] = old_root->mutable_operand(operand_num);
          continue;
        }

        to_hoist_set.insert(hoist);
        int64 new_tuple_count = old_root->operand_count();

        // Replace the hoisted instr in the tuple with the operand/operands.
        // We will replace at least one of the operands of the hoist at the
        // tuple place; the rest will be added at the end.
        bool inplace = true;
        CHECK(!hoist->operands().empty());
        for (HloInstruction* prod : hoist->operands()) {
          if (inplace) {
            map_inst_to_tuple_index[prod] = map_inst_to_tuple_index[hoist];
            new_operands[map_inst_to_tuple_index[hoist]] = prod;
            inplace = false;
          } else {
            map_inst_to_tuple_index[prod] = new_tuple_count++;
            new_operands.push_back(prod);
          }
        }
      }

      // Create the new root instruction.
      HloComputation* cur_branch = conditional->branch_computation(branch);
      HloInstruction* new_branch_root =
          cur_branch->AddInstruction(HloInstruction::CreateTuple(new_operands));
      // The shape can vary since the operands to convert are now
      // being returned through the branches' root.
      cur_branch->set_root_instruction(new_branch_root, true /*new shape*/);
      TF_CHECK_OK(cur_branch->RemoveInstruction(old_root));

      // Only one of the branches needs to change the conditional->parent().
      if (branch != 0) {
        continue;
      }
      HloComputation* conditional_parent = conditional->parent();
      HloInstruction* newconditional =
          conditional_parent->AddInstruction(HloInstruction::CreateConditional(
              cur_branch->root_instruction()->shape(),
              conditional->mutable_operand(0),
              absl::MakeSpan(conditional->branch_computations()),
              absl::MakeSpan(conditional->operands()).subspan(1)));
      // Ensure that all the users of conditional refer to the new one.
      TF_RETURN_IF_ERROR(
          conditional->ReplaceAllUsesWithDifferentShape(newconditional));
      TF_CHECK_OK(conditional_parent->RemoveInstruction(conditional));
      conditional = newconditional;
      // Add the hoisted instructions in the parent.
      for (HloInstruction* hoist : to_hoist_set) {
        VLOG(2) << "Hoisting instruction:" << hoist->ToString();
        int64 hoist_index = map_inst_to_tuple_index[hoist];
        // Find out the gte that captured the hoisted instr result.
        HloInstruction* gte_hoist = find_gte(conditional, hoist_index);
        CHECK(gte_hoist != nullptr);
        std::vector<HloInstruction*> new_operands;
        for (HloInstruction* op : hoist->operands()) {
          HloInstruction* gte = conditional_parent->AddInstruction(
              HloInstruction::CreateGetTupleElement(
                  op->shape(), conditional, map_inst_to_tuple_index[op]));
          new_operands.push_back(gte);
        }
        HloInstruction* hoisted = conditional_parent->AddInstruction(
            hoist->CloneWithNewOperands(hoist->shape(), new_operands));
        VLOG(2) << "Hoisted instruction in parent:" << hoisted->ToString();
        TF_RETURN_IF_ERROR(gte_hoist->ReplaceAllUsesWith(hoisted));
        TF_CHECK_OK(conditional_parent->RemoveInstruction(gte_hoist));
      }
      // No need to explicitly delete a hoisted instruction since if its dead
      // then the subsequent DCE will remove it.
    }
  }
  VLOG(2) << "AFTER :" << conditional->parent()->parent()->ToString();
  return true;
}

// Hoist identical ops out of the conditional. The definition of identical
// are the shape of the operands are identical and their properties are
// identical. Will start from the root instruction of each branch and get
// the identical ops to hoist.
StatusOr<bool> ConditionalCodeMotion::MoveInstructionOut(
    HloInstruction* conditional, std::vector<Boundary>& to_move_out,
    std::vector<Boundary>& new_boundaries) {
  if (to_move_out.empty()) {
    return false;
  }
  VLOG(1) << "number of boundaries to move out:" << to_move_out.size() << "\n";
  HloComputation* conditional_parent = conditional->parent();
  // save the old users before add new conditional user instructions
  std::vector<HloInstruction*> old_conditional_users = conditional->users();
  // Maps instructions in the conditional body to instructions hoisted outside
  // the conditional that compute the same value.
  absl::flat_hash_map<HloInstruction*, Boundary> hoisted_instructions;
  // Insert GetTupleElement before the instructions whose operands might still
  // be within the conditional.
  VLOG(2) << "before opt:"
          << conditional_parent->ToString(HloPrintOptions::Fingerprint())
          << "\n";
  int64 op_index = 0;
  for (Boundary b : new_boundaries) {
    HloInstruction* op = b.operands()[0];
    CHECK(op != nullptr);
    VLOG(2) << "Mapping new boundary instr: " << op->ToString() << "\n";
    HloInstruction* gtr = conditional_parent->AddInstruction(
        HloInstruction::CreateGetTupleElement(op->shape(), conditional,
                                              op_index++));
    Boundary b2(Boundary::Position::kOutsideBranch);
    b2.mutable_operands().push_back(gtr);
    hoisted_instructions[op] = b2;
  }
  // Copy boundary instructions out of the conditional.
  // Visit the operands before its users and copy it, so that the copied
  // user will point to the correct operand.
  for (int64 i = to_move_out.size() - 1; i >= 0; i--) {
    TF_RETURN_IF_ERROR(CopyInOrOutOfConditional(
        to_move_out[i], 0, conditional_parent, hoisted_instructions));
  }
  VLOG(2) << "Done copy branch instructions out\n"
          << conditional_parent->ToString(HloPrintOptions::Fingerprint())
          << "\n";
  // Change original users of the conditional to use the correct operands.
  HloInstruction* old_root =
      conditional->branch_computation(0)->root_instruction();
  for (auto user_instr : old_conditional_users) {
    CHECK(user_instr->opcode() == HloOpcode::kGetTupleElement);
    auto tuple_opd = static_cast<HloGetTupleElementInstruction*>(user_instr);
    int64 index = tuple_opd->tuple_index();
    HloInstruction* old_opd = old_root->operands()[index];
    HloInstruction* new_opd = hoisted_instructions[old_opd].operands()[0];
    CHECK(old_opd != nullptr);
    CHECK(new_opd != nullptr);
    TF_RETURN_IF_ERROR(user_instr->ReplaceAllUsesWith(new_opd));
    TF_RETURN_IF_ERROR(conditional_parent->RemoveInstruction(user_instr));
  }
  // Create tuple element within each branch and set it as root.
  int64 branch_count = conditional->branch_count();
  for (int i = 0; i < branch_count; i++) {
    auto computation = conditional->branch_computation(i);
    std::vector<HloInstruction*> elements;
    for (auto b1 : new_boundaries) {
      HloInstruction* op = b1.operands()[i];
      VLOG(1) << "branch count=" << i << "\n";
      CHECK(op != nullptr);
      VLOG(1) << "Adding to root " << i << " with " << op->ToString() << "\n";
      elements.push_back(op);
    }
    HloInstruction* tuple =
        computation->AddInstruction(HloInstruction::CreateTuple(elements));
    computation->set_root_instruction(tuple, true);
    VLOG(2) << "computation is :" << computation->ToString() << "\n";
    // Remove hoisted instructions from the branches.
    for (auto b2 : to_move_out) {
      VLOG(2) << "Removing boundary:" << b2.ToString() << "\n";
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(b2.operands()[i]));
    }
  }
  // Change conditional instruction shape to the shape of the new root.
  HloInstruction* new_root =
      conditional->branch_computation(0)->root_instruction();
  *conditional->mutable_shape() = new_root->shape();
  //
  VLOG(2) << "done moving instructions out of branches\n"
          << conditional_parent->ToString(HloPrintOptions::Fingerprint())
          << "\n";
  return true;
}

// Hoist ops from outside of the conditional to inside the branches.
StatusOr<bool> ConditionalCodeMotion::MoveInstructionIn(
    HloInstruction* conditional, std::vector<Boundary>& to_move_in,
    std::vector<Boundary>& new_boundaries) {
  if (to_move_in.empty()) {
    return false;
  }
  VLOG(1) << "number of boundaries to move in:" << to_move_in.size() << "\n";
  HloComputation* conditional_parent = conditional->parent();
  VLOG(2) << "before opt:"
          << conditional_parent->ToString(HloPrintOptions::Fingerprint())
          << "\n";
  // Mapping instructions to be moved to their new representations.
  absl::flat_hash_map<HloInstruction*, Boundary> hoisted_instructions;
  int64 to_move_in_size = to_move_in.size();
  int64 branch_count = conditional->branch_count();
  int64 op_index = conditional->shape().tuple_shapes_size();
  // Map conditional to its old root, then create a new root instruction in each
  // branch.
  Boundary b(Boundary::Position::kInsideBranch);
  for (int i = 0; i < branch_count; i++) {
    auto computation = conditional->branch_computation(i);
    auto old_root = computation->root_instruction();
    b.mutable_operands().push_back(old_root);
    HloInstruction* new_root = nullptr;
    if (old_root->opcode() == HloOpcode::kTuple) {
      new_root = computation->AddInstruction(old_root->Clone());
    } else {
      std::vector<HloInstruction*> operands;
      if (!old_root->shape().IsTuple()) {
        operands.push_back(old_root);
      } else {
        const Shape& old_shape = old_root->shape();
        for (int64 i = 0; i < old_shape.tuple_shapes_size(); ++i) {
          auto element =
              computation->AddInstruction(HloInstruction::CreateGetTupleElement(
                  old_shape.tuple_shapes(i), old_root, i));
          operands.push_back(element);
        }
      }
      new_root =
          computation->AddInstruction(HloInstruction::CreateTuple(operands));
    }
    VLOG(2) << "setting new root: " << new_root->ToString() << "\n";
    computation->set_root_instruction(new_root);
    VLOG(2) << "new branch computation: " << computation->ToString() << "\n";
  }
  hoisted_instructions[conditional] = b;
  for (int64 i = 0; i < to_move_in_size; i++) {
    Boundary b_to_move = to_move_in[i];
    HloInstruction* op = b_to_move.operands()[0];
    CHECK(op != nullptr);
    bool to_be_used_outside = true;
    VLOG(2) << "Mapping new boundary instr: " << op->ToString() << "\n";
    if (i < to_move_in_size - 1 && op->user_count() == 1 &&
        op->users()[0] == to_move_in[i + 1].operands()[0]) {
      to_be_used_outside = false;
      VLOG(2) << "Instruction is not to be used outside the branch\n";
    }
    Boundary b(Boundary::Position::kInsideBranch);
    for (int i = 0; i < branch_count; i++) {
      auto computation = conditional->branch_computation(i);
      TF_RETURN_IF_ERROR(CopyInOrOutOfConditional(b_to_move, i, computation,
                                                  hoisted_instructions));
      VLOG(2) << "After Copying to branch: " << computation->ToString() << "\n";
      if (to_be_used_outside) {
        auto new_op = hoisted_instructions[op].operands()[i];
        auto new_root = computation->root_instruction();
        new_root->AppendOperand(new_op);
        *new_root->mutable_shape()->add_tuple_shapes() = new_op->shape();
        VLOG(2) << "Extending conditional root " << i << " : "
                << new_root->ToString() << "\n";
      }
      VLOG(2) << "After extending branch root: " << computation->ToString()
              << "\n";
    }
    if (to_be_used_outside) {
      // Modify uses of instructions outside of the conditionals
      HloInstruction* gtr = conditional_parent->AddInstruction(
          HloInstruction::CreateGetTupleElement(op->shape(), conditional,
                                                op_index++));
      TF_RETURN_IF_ERROR(op->ReplaceAllUsesWith(gtr));
      if (conditional_parent->root_instruction() == op) {
        conditional_parent->set_root_instruction(gtr);
      }
    }
  }
  VLOG(2) << "Done copying instructions inside branch: "
          << conditional->ToString(HloPrintOptions::Fingerprint()) << "\n";
  // Change conditional instruction shape to the shape of the new root.
  HloInstruction* new_root =
      conditional->branch_computation(0)->root_instruction();
  *conditional->mutable_shape() = new_root->shape();
  VLOG(2) << "Before removing instructions:" << conditional_parent->ToString()
          << "\n";
  // Remove hoisted instructions from the branches.
  for (int64 i = to_move_in_size - 1; i >= 0; i--) {
    Boundary boundary_to_move_in = to_move_in[i];
    VLOG(2) << "Removing boundary:" << boundary_to_move_in.ToString() << "\n";
    HloInstruction* op = boundary_to_move_in.operands()[0];
    for (auto user : op->users()) {
      VLOG(2) << "Has User: " << user->ToString() << "\n";
    }
    TF_RETURN_IF_ERROR(conditional_parent->RemoveInstruction(op));
  }
  VLOG(2) << "Done moving instructions inside branches\n"
          << conditional_parent->ToString(HloPrintOptions::Fingerprint())
          << "\n";
  return true;
}

// Group single chains of operands or uses of boundaries into new boundaries
class GroupConnectedBoundaries {
 private:
  std::vector<Boundary> connected_boundaries_, new_boundaries_;
  HloInstruction* conditional_;
  HloComputation* conditional_parent_;
  bool is_layout_sensitive_;
  absl::flat_hash_set<HloInstruction*> visited_;

 public:
  explicit GroupConnectedBoundaries(HloInstruction* conditional,
                                    bool is_layout_sensitive)
      : conditional_(conditional),
        conditional_parent_(conditional->parent()),
        is_layout_sensitive_(is_layout_sensitive) {}
  // Returns true if `instruction` is worth hoisting out.
  bool WorthHoisting(HloInstruction* instruction) {
    // This is needed for the "moving-in" transformation, to prevent the root
    // of the parent computation (which contains the conditional) to be moved
    // inside the conditional.
    if (instruction->opcode() == HloOpcode::kTuple &&
        instruction == conditional_parent_->root_instruction()) {
      return false;
    }
    switch (instruction->opcode()) {
      case HloOpcode::kConvert:
        // If Convert is after AllReduce, it is worth moving out AllReduce
        // out of conditional for AR/CRS combine. If Convert is after other
        // ops such as Dot or Convolutional, it is better to keep convert
        // within conditional so that convert can be fused with Dot or
        // Convolutional.
        //
        // TODO(b/154283721): figure out the scenario when convert can be
        // fused with AllReduce out of conditional.
        switch (instruction->operand(0)->opcode()) {
          case HloOpcode::kAllReduce:
          case HloOpcode::kReshape:
            return true;
          default:
            VLOG(1) << "Instruction is convert and its operand is not know to "
                       "be worth hoisting\n";
            return false;
        }
      case HloOpcode::kAllReduce:
      case HloOpcode::kAdd:
      case HloOpcode::kPower:
      case HloOpcode::kConstant:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
      case HloOpcode::kDivide:
      case HloOpcode::kTuple:
      case HloOpcode::kSqrt:
      case HloOpcode::kReshape:
      case HloOpcode::kGetTupleElement:
        return true;
      default:
        VLOG(1) << "Instruction is not known to be worth hoisting\n";
        return false;
    }
  }
  int64 ReusesBeforeBoundary(HloInstruction* user) {
    int64 reuses = 0;
    for (auto op : user->operands()) {
      // Only consider single-user cases as reuseable.
      if (ContainsKey(visited_, op) && op->user_count() == 1) {
        reuses += ReusesCarriedBy(op, user);
      } else if (op->opcode() == HloOpcode::kConditional &&
                 user->opcode() == HloOpcode::kGetTupleElement) {
        if (user->user_count() == 1) {
          reuses += ReusesCarriedBy(op, user->users()[0]);
        }
      }
    }
    VLOG(1) << "Reuses before instruction " << user->ToString() << ":" << reuses
            << "\n";
    return reuses;
  }

  int64 ReusesAfterBoundary(HloInstruction* user) {
    CHECK(user != nullptr);
    auto all_users = user->users();
    // For now, assume that if an instruction has multiple-consumers, it
    // will not be reused, as the reuse may require duplication in
    // fusion and so is expensive. If the situation changes in the future,
    // some aspects of the overall algorithm need to be redesigned to
    // accommandate the change.
    if (all_users.size() > 1) {
      return 0;
    }
    if (!all_users.empty()) {
      auto op = all_users[0];
      int64 reuses = 0;
      // Only count reuses that run through the conditional root.
      if (op == conditional_->branch_computation(0)->root_instruction()) {
        int64 index = op->operand_index(user);
        for (auto op2 : conditional_->users()) {
          // If the use is not get tuple, right now do not consider it.
          if (op2->opcode() == HloOpcode::kGetTupleElement) {
            auto tuple_opd = static_cast<HloGetTupleElementInstruction*>(op2);
            if (index == tuple_opd->tuple_index()) {
              all_users = op2->users();
              if (!all_users.empty()) {
                reuses += ReusesCarriedBy(user, all_users[0]);
                break;
              }
            }
          }
        }
      } else if (ContainsKey(visited_, op)) {
        reuses += ReusesCarriedBy(user, op);
      }
      VLOG(1) << "reuses after instruction " << user->ToString() << ":"
              << reuses << "\n";
      return reuses;
    }
    return 0;
  }

  int64 BenefitForMovingBoundaries(const std::vector<Boundary>& boundaries) {
    int64 reuses_before = 0, reuses_after = 0;
    if (boundaries.size() == 1 && boundaries[0].IsOutsideBranch()) {
      // The only boundary of moving-in is the get_tuple_element op.
      return -1;
    }
    for (Boundary b : boundaries) {
      auto op = b.operands()[0];
      if (op == conditional_->branch_computation(0)->root_instruction()) {
        continue;
      }
      reuses_before += ReusesBeforeBoundary(op);
      VLOG(1) << "Reuses before boundary so far: " << reuses_before << "\n";
      reuses_after += ReusesAfterBoundary(op);
      VLOG(1) << "Reuese after boundary so far : " << reuses_after << "\n";
    }
    if (reuses_after == 0 && reuses_before == 0) {
      return -1;
    } else if (boundaries[0].IsInsideBranch()) {
      return reuses_after - reuses_before;
    } else {
      return reuses_before - reuses_after;
    }
  }

  Boundary GetNextBoundary(const Boundary& b, int64 op_index) {
    Boundary b2(b.GetPosition());
    for (int j = 0; j < b.operands().size(); ++j) {
      HloInstruction* inst = b.operands()[j];
      CHECK(inst != nullptr);
      HloInstruction* op = (b.IsInsideBranch()) ? inst->operands()[op_index]
                                                : inst->users()[op_index];
      CHECK(op != nullptr);
      b2.mutable_operands().push_back(op);
    }
    return b2;
  }
  int64 CountNonLeafOps(const xla::HloInstruction::InstructionVector& ops) {
    int64 count = 0;
    absl::flat_hash_set<HloInstruction*> op_set;
    for (auto op : ops) {
      if (!op_set.contains(op) && op->opcode() != HloOpcode::kConstant) {
        count++;
        op_set.insert(op);
      }
    }
    return count;
  }
  // This function is reused both for moving the boundary outside or into a
  // conditional. As the result, the readability is somewhat compromised.
  // It might be nice to refactor this function to factor the outside-inside
  // considerations into separate function pointer parameters to improve
  // readability.
  void AddBoundaries(const Boundary& boundary) {
    BoundaryVisitor visitor;
    visitor.AddToWorkList(boundary);
    while (visitor.HasNextBoundary()) {
      Boundary b = visitor.PopNextBoundary();
      VLOG(1) << "visiting boundary " << b.ToString() << "\n";
      if ((b.IsOutsideBranch() || InstructionWithinBranchIdentical(
                                      b.operands(), is_layout_sensitive_)) &&
          WorthHoisting(b.operands()[0])) {
        connected_boundaries_.push_back(b);
        VLOG(1) << "boundary can be moved\n";
        int64 operand_count = (b.IsInsideBranch())
                                  ? b.operands()[0]->operand_count()
                                  : b.operands()[0]->users().size();
        for (int i = 0; i < operand_count; i++) {
          Boundary next_boundary = GetNextBoundary(b, i);
          int64 next_boundary_count =
              (next_boundary.IsInsideBranch())
                  ? next_boundary.operands()[0]->user_count()
                  : CountNonLeafOps(next_boundary.operands()[0]->operands());
          // only consider adding an exclusive producor into the same group.
          if (next_boundary_count == 1) {
            VLOG(2) << "Add operand " << i << " to visit later\n";
            visitor.AddToWorkList(next_boundary);
          } else {
            VLOG(2) << "Next boundary " << i
                    << " has multiple uses: " << next_boundary_count << "\n";
            if (!ContainsKey(visited_, next_boundary.operands()[0])) {
              visited_.insert(next_boundary.operands()[0]);
              new_boundaries_.push_back(next_boundary);
            }
          }
        }
      } else {
        VLOG(1) << "boundary cannot be moved\n";
        visited_.insert(b.operands()[0]);
        new_boundaries_.push_back(b);
      }
    }
  }
  std::vector<Boundary> BoundariesToMoveInOrOut(const Boundary& b) {
    // At the beginning of optimization, a conditional itself is added to a
    // worklist. Here the conditional is expanded into two sets of boundaries:
    // the first set contains the boundary that is inside branches and
    // contains the root of all branches; the second set of boundaries
    // contains all the users of the conditional.
    HloInstruction* inst = b.operands()[0];
    if (inst->opcode() == HloOpcode::kConditional) {
      int branch_count = inst->branch_count();
      // Add conditional roots as a new boundary to visit.
      Boundary boundary_in(Boundary::Position::kInsideBranch);
      for (int i = 0; i < branch_count; i++) {
        HloComputation* branch_computation = inst->branch_computation(i);
        HloInstruction* root_inst = branch_computation->root_instruction();
        CHECK(root_inst != nullptr);
        boundary_in.mutable_operands().push_back(root_inst);
      }
      new_boundaries_.push_back(boundary_in);
      // Add conditional users as new boundaries to visit.
      for (auto u : inst->users()) {
        Boundary boundary_in(Boundary::Position::kOutsideBranch);
        boundary_in.mutable_operands().push_back(u);
        new_boundaries_.push_back(boundary_in);
      }
    } else {
      AddBoundaries(b);
    }
    return connected_boundaries_;
  }
  void AddNewBoundaries(std::vector<Boundary>& b) {
    b.insert(b.end(), new_boundaries_.begin(), new_boundaries_.end());
  }
};

ConditionalCodeMotion::Decision ConditionalCodeMotion::ConsiderCodeMotion(
    HloInstruction* conditional, const Boundary& cur_boundary,
    std::vector<Boundary>& to_move, std::vector<Boundary>& new_boundaries) {
  GroupConnectedBoundaries connect(conditional, is_layout_sensitive_);
  auto move_in_or_out = connect.BoundariesToMoveInOrOut(cur_boundary);
  if (!move_in_or_out.empty()) {
    auto benefit = connect.BenefitForMovingBoundaries(move_in_or_out);
    VLOG(1) << "benefit of moving in or out "
            << cur_boundary.operands()[0]->ToString() << ":" << benefit << "\n";
    if (benefit >= 0) {
      new_boundaries.clear();
      connect.AddNewBoundaries(new_boundaries);
      // The whole sequence in move_in_or_out is either all moving into a
      // conditional, or all moving out of a conditional. So looking only
      // at the first entry of the sequence is sufficient to know which
      // direction the move is intended.
      to_move = move_in_or_out;
      return to_move[0].IsInsideBranch() ? Decision::kMoveOutOfBranch
                                         : Decision::kMoveIntoBranch;
    }
  } else {
    connect.AddNewBoundaries(new_boundaries);
  }
  return ConditionalCodeMotion::Decision::kNoChange;
}

StatusOr<bool> ConditionalCodeMotion::Run(HloModule* module) {
  // Gather all the conditional ops in the module ahead of time, to avoid
  // potential complications of modifying the code that affecting traversal.
  std::vector<HloInstruction*> conditional_ops;
  for (auto* comp : module->MakeComputationPostOrder()) {
    for (auto* instr : comp->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kConditional) {
        conditional_ops.push_back(instr);
      }
    }
  }

  bool changed = false;
  for (HloInstruction* conditional : conditional_ops) {
    // Boundaries to move out or to move into the branches.
    std::vector<Boundary> to_move_out, to_move_in, new_boundaries;
    // The conditional is moved into a worklist as the seed (starting point).
    // The conditional will be expanded into multiple seeds (starting points),
    // its roots and its users, when it is visited by GroupConnectedBoundaries.
    // A NO_CHANGE decision will always be returned for the conditional itself,
    // so that the other seeding boundaries can be visited in turn.
    BoundaryVisitor visitor(conditional);
    VLOG(2) << "Analyzing conditional:" << conditional->ToString() << "\n";
    ConditionalCodeMotion::Decision d = Decision::kNoChange;
    // The following loop breaks out as soon as a decision to modify the
    // conditional is reached --- irrespective of whether visitor is empty.
    while (d == Decision::kNoChange && visitor.HasNextBoundary()) {
      std::vector<Boundary> to_move, next_boundary;
      Boundary boundary = visitor.PopNextBoundary();
      VLOG(2) << "Analyzing boundary:" << boundary.ToString() << "\n";
      d = ConsiderCodeMotion(conditional, boundary, to_move, next_boundary);
      switch (d) {
        case Decision::kMoveOutOfBranch:
          VLOG(2) << "Decision is move out of branch\n";
          to_move_out.insert(to_move_out.end(), to_move.begin(), to_move.end());
          new_boundaries.insert(new_boundaries.end(), next_boundary.begin(),
                                next_boundary.end());
          break;
        case Decision::kMoveIntoBranch:
          VLOG(2) << "Decision is move into branch\n";
          to_move_in.insert(to_move_in.end(), to_move.begin(), to_move.end());
          new_boundaries.insert(new_boundaries.end(), next_boundary.begin(),
                                next_boundary.end());
          break;
        case Decision::kNoChange:
          VLOG(2) << "Decision is no change\n";
          for (const Boundary& b : next_boundary) {
            visitor.AddToWorkList(b);
          }
          break;
      }
    }
    // At most one of to_move_out or to_move_in can be non-empty, since there is
    // only one optimization decision.
    if (!to_move_out.empty()) {
      TF_ASSIGN_OR_RETURN(
          bool result,
          MoveInstructionOut(conditional, to_move_out, new_boundaries));
      VLOG(2) << "moving out result:" << result << "\n";
      changed |= result;
    } else if (!to_move_in.empty()) {
      TF_ASSIGN_OR_RETURN(
          bool result,
          MoveInstructionIn(conditional, to_move_in, new_boundaries));
      VLOG(2) << "moving in result:" << result << "\n";
      changed |= result;
    }
  }
  // handling convert rematerialization/hoisting
  if (!changed && pursue_full_conditional_code_motion_) {
    std::vector<HloInstruction*> conditional_ops;
    for (auto* comp : module->MakeComputationPostOrder()) {
      for (auto* instr : comp->MakeInstructionPostOrder()) {
        if (instr->opcode() == HloOpcode::kConditional) {
          conditional_ops.push_back(instr);
        }
      }
    }
    for (HloInstruction* conditional_op : conditional_ops) {
      TF_ASSIGN_OR_RETURN(
          bool convert_result,
          ConvertSpecialMove(conditional_op, is_layout_sensitive_));
      changed |= convert_result;
    }
  }
  if (changed) {
    HloPassPipeline subpipeline(
        "after_conditional_code_motion_after_convert_hoisting");
    subpipeline.AddPass<HloDCE>();
    subpipeline.AddPass<TupleSimplifier>();
    subpipeline.AddPass<HloDCE>();
    TF_ASSIGN_OR_RETURN(bool cleanup_changed, subpipeline.Run(module));
    changed |= cleanup_changed;
  }
  return changed;
}
}  // namespace conditional_opt

}  // namespace xla

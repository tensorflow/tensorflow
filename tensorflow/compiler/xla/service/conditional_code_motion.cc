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
#include <unordered_set>
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

// Visit the root instructions to its operands follow BFS.
// Will visit an instructions after all its users have been visited. Parameters
// are not visited.
class BoundaryVisitor {
 public:
  // start with an existing conditional computation.
  explicit BoundaryVisitor(HloInstruction* conditional) {
    Boundary b(Boundary::Position::kInsideBranch);
    b.Operands().push_back(conditional);
    worklist_.push_back(b);
  }
  // Start with an empty work list.
  BoundaryVisitor() {}
  // Get next intruction to visit.
  Boundary PopNextBoundary() {
    CHECK(!worklist_.empty());
    Boundary inst = worklist_.front();
    worklist_.pop_front();
    return inst;
  }
  void AddToWorkList(const Boundary& b) {
    CHECK_GT(b.Operands().size(), 0);
    worklist_.push_back(b);
  }

  bool HasNextBoundary() const { return !worklist_.empty(); }

 private:
  // worklist is the deque that contains instructions to be visited.
  std::deque<Boundary> worklist_;
};

// Returns estimation of potential reuses carried by a given instruction.
// Use different integers to classify different levels of reuses
// This is used as a placeholder only, assuming all instructions can be
// fused to enable data reuses
int64 ReusesCarriedBy(HloInstruction* op, HloInstruction* user) {
  VLOG(1) << "ConditionalCodeMotion: Add reuses carried by instr: "
          << op->ToString() << "=>" << user->ToString() << "\n";
  switch (user->opcode()) {
    case HloOpcode::kGetTupleElement:
      return 0;
    default:
      break;
  }
  switch (op->opcode()) {
      // These instructions are lightweight and easy to fuse.
    case HloOpcode::kConstant:
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

// Copy identical instructions within conditional outside of conditional.
Status CopyOutOfConditional(
    Boundary& boundary, HloComputation* conditional_parent,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>&
        hoisted_instructions) {
  // Insert GetTupleElement before the instructions whose operands might still
  // be within the conditional.
  HloInstruction* op = boundary.Operands()[0];
  absl::InlinedVector<HloInstruction*, 4> new_operands;
  for (int i = 0; i < op->operands().size(); ++i) {
    auto op_i = op->operands()[i];
    VLOG(2) << "Looking for operand:" << op_i->ToString() << "\n";
    CHECK(ContainsKey(hoisted_instructions, op_i));
    new_operands.push_back(FindOrDie(hoisted_instructions, op_i));
  }
  HloInstruction* new_instruction = conditional_parent->AddInstruction(
      op->CloneWithNewOperands(op->shape(), new_operands));
  // Maps the instruction outside of conditional to the instruction
  // inside of the conditional.
  for (HloInstruction* op : boundary.Operands()) {
    hoisted_instructions[op] = new_instruction;
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
      std::unordered_set<HloInstruction*> to_hoist_set;

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
  absl::flat_hash_map<HloInstruction*, HloInstruction*> hoisted_instructions;
  // Maps instructions in the conditional body to instructions hoisted outside
  // the conditional that compute the same value.
  VLOG(2) << "before opt:"
          << conditional_parent->ToString(HloPrintOptions::Fingerprint())
          << "\n";
  int64 op_index = 0;
  for (Boundary b : new_boundaries) {
    HloInstruction* op = b.Operands()[0];
    CHECK(op != nullptr);
    VLOG(2) << "Mapping new boundary instr: " << op->ToString() << "\n";
    HloInstruction* gtr = conditional_parent->AddInstruction(
        HloInstruction::CreateGetTupleElement(op->shape(), conditional,
                                              op_index++));
    hoisted_instructions[op] = gtr;
  }
  // Copy boundary instructions out of the conditional.
  // Visit the operands before its users and copy it, so that the copied
  // user will point to the correct operand.
  for (int64 i = to_move_out.size() - 1; i >= 0; i--) {
    TF_RETURN_IF_ERROR(CopyOutOfConditional(to_move_out[i], conditional_parent,
                                            hoisted_instructions));
  }
  VLOG(2) << "Done copy branch instructions out\n"
          << conditional_parent->ToString(HloPrintOptions::Fingerprint())
          << "\n";
  // Change original users of the conditional to use the correct operands.
  HloInstruction* old_root =
      conditional->branch_computation(0)->root_instruction();
  for (auto user_instr : old_conditional_users) {
    CHECK(user_instr->opcode() == HloOpcode::kGetTupleElement);
    auto tuple_opd = down_cast<HloGetTupleElementInstruction*>(user_instr);
    int64 index = tuple_opd->tuple_index();
    HloInstruction* old_opd = old_root->operands()[index];
    HloInstruction* new_opd = hoisted_instructions[old_opd];
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
      HloInstruction* op = b1.Operands()[i];
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
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(b2.Operands()[i]));
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

// Group single chains of operands or uses of boundaries into new boundaries
class GroupConnectedBoundaries {
 private:
  std::unordered_set<HloInstruction*> visited_;
  std::vector<Boundary> connected_boundaries_, new_boundaries_;
  HloInstruction* conditional_;
  bool is_layout_sensitive_;

 public:
  explicit GroupConnectedBoundaries(HloInstruction* conditional,
                                    bool is_layout_sensitive)
      : conditional_(conditional), is_layout_sensitive_(is_layout_sensitive) {}
  // Returns true if `instruction` is worth hoisting out.
  bool WorthHoisting(HloInstruction* instruction) {
    switch (instruction->opcode()) {
      case HloOpcode::kConvert:
        // If Convert is after AllReduce, it is worth moving out AllReduce out
        // of conditional for AR/CRS combine. If Convert is after other ops such
        // as Dot or Convolutional, it is better to keep convert within
        // conditional so that convert can be fused with Dot or Convolutional.
        //
        // TODO(b/154283721): figure out the scenario when convert can be fused
        // with AllReduce out of conditional.
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
  // Calculates the degree of reuses carried by a pair of conditional
  // boundaries, if b1 is inside a conditional and b2 is outside.
  int64 ReusesBeforeBoundary(HloInstruction* user) {
    int64 reuses = 0;
    for (auto op : user->operands()) {
      // Only consider single-user cases as reuseable.
      if (ContainsKey(visited_, op) && op->user_count() == 1) {
        reuses += ReusesCarriedBy(op, user);
      }
    }
    VLOG(1) << "cost to be paied after moving out" << user->ToString() << ":"
            << reuses << "\n";
    return reuses;
  }

  int64 ReusesAfterBoundary(HloInstruction* user) {
    CHECK(user != nullptr);
    auto all_users = user->users();
    // For now, assume that if an instruction has multiple-consumers, it will
    // not be reused (the reuse currently requires duplication in fusion and so
    // is expensive).
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
          CHECK(op2->opcode() == HloOpcode::kGetTupleElement);
          auto tuple_opd = down_cast<HloGetTupleElementInstruction*>(op2);
          if (index == tuple_opd->tuple_index()) {
            all_users = op2->users();
            if (!all_users.empty()) {
              reuses += ReusesCarriedBy(user, all_users[0]);
              break;
            }
          }
        }
      }
      VLOG(1) << "reuses to be gained after moving " << user->ToString() << ":"
              << reuses << "\n";
      return reuses;
    }
    return 0;
  }

  int64 BenefitForMovingBoundaries(const std::vector<Boundary>& boundaries) {
    int64 reuses_before = 0, reuses_after = 0;
    for (Boundary b : boundaries) {
      auto op = b.Operands()[0];
      if (op == conditional_->branch_computation(0)->root_instruction()) {
        continue;
      }
      reuses_before += ReusesBeforeBoundary(op);
      VLOG(1) << "Cost of moving so far: " << reuses_before << "\n";
      reuses_after += ReusesAfterBoundary(op);
      VLOG(1) << "Benefit from moving so far : " << reuses_after << "\n";
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
    CHECK(b.Operands().size() == conditional_->branch_count());
    for (int j = 0; j < b.Operands().size(); ++j) {
      HloInstruction* inst = b.Operands()[j];
      CHECK(inst != nullptr);
      HloInstruction* op = (b.IsInsideBranch()) ? inst->operands()[op_index]
                                                : inst->users()[op_index];
      CHECK(op != nullptr);
      b2.Operands().push_back(op);
    }
    return b2;
  }
  void AddBoundaries(const Boundary& boundary) {
    BoundaryVisitor visitor;
    visitor.AddToWorkList(boundary);
    while (visitor.HasNextBoundary()) {
      Boundary b = visitor.PopNextBoundary();
      // if b is already visited, it must have multiple users and is already in
      // new boundaries. Skip it.
      if (ContainsKey(visited_, b.Operands()[0])) {
        continue;
      }
      VLOG(1) << "visiting boundary " << b.ToString() << "\n";
      if ((b.Operands().size() == 1 ||
           InstructionWithinBranchIdentical(b.Operands(),
                                            is_layout_sensitive_)) &&
          WorthHoisting(b.Operands()[0])) {
        connected_boundaries_.push_back(b);
        VLOG(1) << "boundary can be moved\n";
        int64 operand_count = (b.IsInsideBranch())
                                  ? b.Operands()[0]->operand_count()
                                  : b.Operands()[0]->users().size();
        for (int i = 0; i < operand_count; i++) {
          Boundary b2 = GetNextBoundary(b, i);
          int64 b2_count = (b2.IsInsideBranch())
                               ? b2.Operands()[0]->user_count()
                               : b2.Operands()[0]->operand_count();
          // only consider adding an exclusive producor into the same group.
          if (b2_count == 1) {
            VLOG(2) << "Add operand " << i << " to visit later\n";
            visitor.AddToWorkList(b2);
          } else {
            VLOG(2) << "Operand " << i << " has multiple uses\n";
            if (!ContainsKey(visited_, b2.Operands()[0])) {
              visited_.insert(b2.Operands()[0]);
              new_boundaries_.push_back(b2);
            }
          }
        }
      } else {
        VLOG(1) << "boundary cannot be moved\n";
        visited_.insert(b.Operands()[0]);
        new_boundaries_.push_back(b);
      }
    }
  }
  std::vector<Boundary> BoundariesToMoveOut(const Boundary& b) {
    HloInstruction* inst = b.Operands()[0];
    if (inst->opcode() == HloOpcode::kConditional) {
      int branch_count = inst->branch_count();
      // Visit instructions from the root instruction to the operands using BFS.
      Boundary boundary_in(Boundary::Position::kInsideBranch);
      for (int i = 0; i < branch_count; i++) {
        HloComputation* branch_computation = inst->branch_computation(i);
        HloInstruction* root_inst = branch_computation->root_instruction();
        CHECK(root_inst != nullptr);
        boundary_in.Operands().push_back(root_inst);
      }
      AddBoundaries(boundary_in);
    }
    return connected_boundaries_;
  }
  std::vector<Boundary> BoundariesToMoveIn(const Boundary& b) {
    if (b.IsInsideBranch()) {
      return std::vector<Boundary>();
    }
    AddBoundaries(b);
    return connected_boundaries_;
  }
  std::vector<Boundary> GetNewBoundaries() { return new_boundaries_; }
};

ConditionalCodeMotion::Decision ConditionalCodeMotion::ConsiderCodeMotion(
    HloInstruction* conditional, const Boundary& cur_boundary,
    std::vector<Boundary>& to_move, std::vector<Boundary>& new_boundaries) {
  GroupConnectedBoundaries connect(conditional, is_layout_sensitive_);
  auto move_out = connect.BoundariesToMoveOut(cur_boundary);
  if (!move_out.empty()) {
    std::vector<Boundary> next_boundaries = connect.GetNewBoundaries();
    auto benefit = connect.BenefitForMovingBoundaries(move_out);
    VLOG(1) << "benefit of moving " << cur_boundary.Operands()[0]->ToString()
            << ":" << benefit << "\n";
    if (benefit >= 0) {
      new_boundaries = next_boundaries;
      to_move = move_out;
      return Decision::kMoveOutOfBranch;
    }
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
  std::vector<Boundary> to_move_out, to_move_in, new_boundaries;
  for (HloInstruction* conditional : conditional_ops) {
    BoundaryVisitor visitor(conditional);
    VLOG(2) << "Analyzing conditional:" << conditional->ToString() << "\n";
    // Boundariess to move out of and to move into the branches.
    while (visitor.HasNextBoundary()) {
      std::vector<Boundary> to_move, next_boundary;
      Boundary boundary = visitor.PopNextBoundary();
      VLOG(2) << "Analyzing boundary:" << boundary.ToString() << "\n";
      ConditionalCodeMotion::Decision d =
          ConsiderCodeMotion(conditional, boundary, to_move, next_boundary);
      switch (d) {
        case Decision::kMoveOutOfBranch:
          VLOG(2) << "Decision is move out of branch\n";
          to_move_out.insert(to_move_out.end(), to_move.begin(), to_move.end());
          break;
        case Decision::kMoveIntoBranch:
          VLOG(2) << "Decision is move into branch\n";
          to_move_in.insert(to_move_in.end(), to_move.begin(), to_move.end());
          break;
        case Decision::kNoChange:
          VLOG(2) << "Decision is no change\n";
          new_boundaries.push_back(boundary);
          break;
      }
      for (const Boundary& b : next_boundary) {
        visitor.AddToWorkList(b);
      }
    }
    TF_ASSIGN_OR_RETURN(
        bool result,
        MoveInstructionOut(conditional, to_move_out, new_boundaries));
    VLOG(2) << "moving out result:" << result << "\n";
    changed |= result;
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

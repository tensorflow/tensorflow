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

#include <algorithm>
#include <iterator>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
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
    // new boundaries. Skip it.
    while (!worklist_.empty() && ContainsKey(visited_, b)) {
      b = worklist_.front();
      worklist_.pop_front();
    }
    visited_.insert(b);
    return b;
  }
  void AddToWorkList(const Boundary& b) {
    CHECK(!b.operands().empty());
    worklist_.push_back(b);
  }

  bool HasNextBoundary() {
    while (!worklist_.empty()) {
      Boundary b = worklist_.front();
      if (!ContainsKey(visited_, b)) {
        break;
      }
      worklist_.pop_front();
    }
    return !worklist_.empty();
  }

 private:
  // worklist is the deque that contains instructions to be visited.
  std::deque<Boundary> worklist_;
  absl::flat_hash_set<Boundary> visited_;
};

template <class OpCollection>
int64_t CountNonLeafOps(const OpCollection& ops) {
  absl::flat_hash_set<HloInstruction*> op_set;
  for (auto op : ops) {
    if (!op_set.contains(op) && op->opcode() != HloOpcode::kConstant) {
      op_set.insert(op);
    }
  }
  return op_set.size();
}

// Returns estimation of potential reuses carried by a given pair of
// instructions.  Use different integers to classify different levels
// of reuses This is used as a placeholder only, assuming all
// instructions can be fused to enable data reuses
int64_t ReusesCarriedBy(HloOpcode op, HloOpcode user) {
  // Reuses in some way work like forces that pull instructions
  // towards each other. We use a number 0-10 to classify how strong the force
  // is between a pair of operations. Given a group of instructions that can be
  // moved together, if the forces inside a conditional are stronger, the group
  // will be moved incide or remain inside the conditional; otherwise, it will
  // be moved outside to or remain outside of the conditional.
  switch (user) {
    case HloOpcode::kGetTupleElement:
      return 0;
    case HloOpcode::kConvert:
      // Because convert is treated not moveable when following Dot or
      // convolution, here if op is dot or convolution, they must be separated
      // by a conditional boundary. Here we do not try to pull convert inside
      // conditionals to be together with the dot or convolution.
      switch (op) {
        case HloOpcode::kConvolution:
        case HloOpcode::kDot:
          return 0;
        default:
          break;
      }
      break;
    default:
      break;
  }
  switch (op) {
      // These instructions do not carry weight of reuse themselves.
    case HloOpcode::kParameter:
    case HloOpcode::kConstant:
    case HloOpcode::kGetTupleElement:
      return 0;
    case HloOpcode::kConditional:
      return 10;
    default:
      return -10;
  }
}

// Returns true if `op` is worth hoisting.
bool WorthHoisting(HloOpcode op, HloOpcode child_op) {
  // TOOD[b/169182921] The following cost model is rather incomplete. Will
  // need to extend to cover most of element-wise ops.
  switch (op) {
    case HloOpcode::kConvert:
      // If Convert is after AllReduce, it is worth moving out AllReduce
      // out of conditional for AR/CRS combine. If Convert is after other
      // ops such as Dot or Convolutional, it is better to keep convert
      // within conditional so that convert can be fused with Dot or
      // Convolutional.
      switch (child_op) {
        case HloOpcode::kAllReduce:
        case HloOpcode::kReshape:
        case HloOpcode::kGetTupleElement:
          return true;
        default:
          return false;
      }
    case HloOpcode::kGetTupleElement:
      switch (child_op) {
        // do not move GTE if its operand is a parameter
        case HloOpcode::kParameter:
          return false;
        default:
          return true;
      }
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAbs:
    case HloOpcode::kReduce:
    case HloOpcode::kAdd:
    case HloOpcode::kPower:
    case HloOpcode::kCopy:
    case HloOpcode::kConstant:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply:
    case HloOpcode::kDivide:
    case HloOpcode::kTuple:
    case HloOpcode::kSqrt:
    case HloOpcode::kRsqrt:
    case HloOpcode::kReshape:
    case HloOpcode::kMinimum:
    case HloOpcode::kMaximum:
      return true;
    default:
      return false;
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

// Copy the boundary out of the conditional and update hoisted_boundaries.
void CopyOutOfConditional(
    Boundary& boundary, HloInstruction* conditional,
    absl::flat_hash_map<Boundary, Boundary>& hoisted_boundaries) {
  CHECK(boundary.IsInsideBranch());
  absl::InlinedVector<HloInstruction*, 4> new_operands;
  // All of the branch operands should have the same opcode and shape, so just
  // use branch 0.
  const HloInstruction* branch0_inst = boundary.operands()[0];
  for (int i = 0; i < branch0_inst->operands().size(); ++i) {
    Boundary operand_boundary(boundary.GetPosition());
    for (HloInstruction* operand : boundary.operands()) {
      operand_boundary.mutable_operands().push_back(operand->operands()[i]);
    }
    VLOG(2) << "Looking for: " << operand_boundary.ToString();
    auto hoisted_boundaries_it = hoisted_boundaries.find(operand_boundary);
    CHECK(hoisted_boundaries_it != hoisted_boundaries.end());
    Boundary hoisted_boundary = hoisted_boundaries_it->second;
    CHECK(hoisted_boundary.IsOutsideBranch());
    CHECK_EQ(hoisted_boundary.operands().size(), 1);
    new_operands.push_back(hoisted_boundary.operands()[0]);
  }
  HloInstruction* new_instruction = conditional->parent()->AddInstruction(
      branch0_inst->CloneWithNewOperands(branch0_inst->shape(), new_operands));
  VLOG(2) << "new instruction:" << new_instruction->ToString();
  // Maps the instruction outside of conditional to the instruction
  // inside of the conditional.
  Boundary hoisted_boundary(Boundary::Position::kOutsideBranch);
  hoisted_boundary.mutable_operands().push_back(new_instruction);
  hoisted_boundaries[boundary] = hoisted_boundary;
}

// Copy the boundary into the conditional and update hoisted_boundaries.
void CopyIntoConditional(
    Boundary& boundary, HloInstruction* conditional,
    absl::flat_hash_map<Boundary, Boundary>& hoisted_boundaries) {
  CHECK(boundary.IsOutsideBranch());
  CHECK_EQ(boundary.operands().size(), 1);
  int num_branches = conditional->branch_count();
  std::vector<absl::InlinedVector<HloInstruction*, 4>> new_operands(
      num_branches);
  HloInstruction* op = boundary.operands()[0];
  for (HloInstruction* operand : op->operands()) {
    Boundary operand_boundary(boundary.GetPosition());
    operand_boundary.mutable_operands().push_back(operand);
    VLOG(2) << "Looking for: " << operand_boundary.ToString();
    auto hoisted_boundaries_it = hoisted_boundaries.find(operand_boundary);
    if (hoisted_boundaries_it != hoisted_boundaries.end()) {
      Boundary hoisted_boundary = hoisted_boundaries_it->second;
      CHECK(hoisted_boundary.IsInsideBranch());
      CHECK_EQ(hoisted_boundary.operands().size(), num_branches);
      for (int j = 0; j < num_branches; ++j) {
        new_operands[j].push_back(hoisted_boundary.operands()[j]);
      }
    } else {
      for (int j = 0; j < num_branches; ++j) {
        switch (operand->opcode()) {
          case HloOpcode::kConstant: {
            auto new_operand =
                conditional->branch_computation(j)->AddInstruction(
                    operand->Clone());
            VLOG(2) << "new instruction:" << new_operand->ToString();
            new_operands[j].push_back(new_operand);
            break;
          }
          case HloOpcode::kGetTupleElement: {
            auto gte = Cast<HloGetTupleElementInstruction>(operand);
            int64_t index = gte->tuple_index();
            HloInstruction* root =
                conditional->branch_computation(j)->root_instruction();
            CHECK(root->opcode() == HloOpcode::kTuple &&
                  index < root->operand_count())
                << root->ToString() << " " << gte->ToString();
            auto new_operand = root->mutable_operand(index);
            VLOG(2) << "new instruction:" << new_operand->ToString();
            new_operands[j].push_back(new_operand);
            break;
          }
          default:
            LOG(FATAL) << "Unexpected out-of-boundary instruction:"
                       << operand->ToString() << "\n";
        }
      }
    }
  }

  Boundary hoisted_boundary(Boundary::Position::kInsideBranch);
  for (int j = 0; j < num_branches; ++j) {
    HloInstruction* new_instruction =
        conditional->branch_computation(j)->AddInstruction(
            op->CloneWithNewOperands(op->shape(), new_operands[j]));
    VLOG(2) << "new instruction:" << new_instruction->ToString();
    hoisted_boundary.mutable_operands().push_back(new_instruction);
  }
  hoisted_boundaries[boundary] = hoisted_boundary;
}

// Identify converts to be hoisted/rematerialized out of the branch
// computations.
absl::flat_hash_set<int64_t> FindSpecialConverts(HloInstruction* old_root,
                                                 int branch_count,
                                                 HloInstruction* conditional,
                                                 bool is_layout_sensitive) {
  absl::flat_hash_set<int64_t> special_convert;

  // TODO(b/216487727): Allow hoisting converts that feed or fed by other
  // converts by addressing possible duplicates left behind in the tuple output.
  // The conditional code motion pass should handle these duplicates and hence,
  // merging these snippets of code would be one alternative.
  auto convert_invalid =
      [](const HloInstruction* convert_set_candidate) -> bool {
    bool invalid_user = absl::c_any_of(
        convert_set_candidate->users(), [](const HloInstruction* user) -> bool {
          return (user->opcode() == HloOpcode::kConvert);
        });
    bool invalid_producer =
        absl::c_any_of(convert_set_candidate->operands(),
                       [](const HloInstruction* operand) -> bool {
                         return (operand->opcode() == HloOpcode::kConvert);
                       });
    return (invalid_user || invalid_producer);
  };

  for (int64_t operand_num = 0; operand_num < old_root->operand_count();
       ++operand_num) {
    if (old_root->operand(operand_num)->opcode() != HloOpcode::kConvert) {
      continue;
    }
    bool replica = true;
    HloInstruction* special_convert_candidate =
        old_root->mutable_operand(operand_num);
    // TODO(b/216487727): Remove duplicates in tuple outputs while hoisting.
    auto repeated =
        absl::c_count_if(old_root->operands(),
                         [&](const HloInstruction* operand) -> bool {
                           return (special_convert_candidate == operand);
                         }) > 1;
    if (convert_invalid(special_convert_candidate) || repeated) {
      continue;
    }
    // Check whether an identical candidate appears in other branches
    for (int others = 1; others < branch_count; ++others) {
      HloInstruction* others_root =
          conditional->branch_computation(others)->root_instruction();
      const HloInstruction* other_convert = others_root->operand(operand_num);
      if (other_convert->opcode() != HloOpcode::kConvert ||
          convert_invalid(other_convert)) {
        replica = false;
        break;
      }
      // Do not move converts if their operands have different shapes in
      // different branches.
      bool eq_shape =
          is_layout_sensitive
              ? ShapeUtil::Equal(other_convert->shape(),
                                 special_convert_candidate->shape()) &&
                    ShapeUtil::Equal(
                        other_convert->operand(0)->shape(),
                        special_convert_candidate->operand(0)->shape())
              : ShapeUtil::Compatible(other_convert->shape(),
                                      special_convert_candidate->shape()) &&
                    ShapeUtil::Compatible(
                        other_convert->operand(0)->shape(),
                        special_convert_candidate->operand(0)->shape());
      if (!eq_shape) {
        replica = false;
        break;
      }
      auto repeated =
          absl::c_count_if(others_root->operands(),
                           [&](const HloInstruction* operand) -> bool {
                             return (special_convert_candidate == operand);
                           }) > 1;
      if (repeated) {
        replica = false;
        break;
      }
    }
    if (replica) {
      special_convert.insert(operand_num);
    }
  }
  return special_convert;
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
  return OkStatus();
}

StatusOr<bool> ConvertSpecialMove(HloInstruction* conditional,
                                  bool is_layout_sensitive) {
  int branch_count = conditional->branch_count();
  if (branch_count <= 0) {
    return false;
  }

  // Determining whether all branch roots are tuples
  for (int branch_num = 0; branch_num < branch_count; ++branch_num) {
    HloInstruction* branch_root =
        conditional->branch_computation(branch_num)->root_instruction();
    if (branch_root->opcode() != HloOpcode::kTuple) {
      return false;
    }
  }

  HloInstruction* old_root =
      conditional->branch_computation(0)->root_instruction();
  VLOG(2) << "BEFORE :" << conditional->parent()->parent()->ToString();
  // Identify the gte using `index'.
  auto find_gte = [](const HloInstruction* conditional_result,
                     int64_t index) -> HloInstruction* {
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
  absl::flat_hash_set<int64_t> special_convert = FindSpecialConverts(
      old_root, branch_count, conditional, is_layout_sensitive);

  // Exit if we cannot find any converts to be hoisted.
  if (special_convert.empty()) {
    return false;
  }

  TF_RETURN_IF_ERROR(
      RestructureConditionalInstruction(conditional->parent(), conditional));

  for (int branch = 0; branch < branch_count; branch++) {
    old_root = conditional->branch_computation(branch)->root_instruction();
    absl::flat_hash_map<HloInstruction*, int64_t> map_inst_to_tuple_index;
    std::vector<HloInstruction*> new_operands(old_root->operand_count());
    absl::flat_hash_set<HloInstruction*> to_hoist_set;

    for (int64_t operand_num = 0; operand_num < old_root->operand_count();
         ++operand_num) {
      map_inst_to_tuple_index[old_root->mutable_operand(operand_num)] =
          operand_num;
    }
    for (int64_t operand_num = 0; operand_num < old_root->operand_count();
         ++operand_num) {
      HloInstruction* hoist = old_root->mutable_operand(operand_num);
      if (!special_convert.contains(operand_num)) {
        new_operands[operand_num] = old_root->mutable_operand(operand_num);
        continue;
      }

      to_hoist_set.insert(hoist);
      int64_t new_tuple_count = old_root->operand_count();

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
      int64_t hoist_index = map_inst_to_tuple_index[hoist];
      // Find out the gte that captured the hoisted instr result.
      HloInstruction* gte_hoist = find_gte(conditional, hoist_index);
      CHECK(gte_hoist != nullptr);
      std::vector<HloInstruction*> new_operands;
      for (HloInstruction* op : hoist->operands()) {
        HloInstruction* gte = conditional_parent->AddInstruction(
            HloInstruction::CreateGetTupleElement(op->shape(), conditional,
                                                  map_inst_to_tuple_index[op]));
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
  VLOG(1) << "Modifying code--number of boundaries to move out:"
          << to_move_out.size() << "\n";
  HloComputation* conditional_parent = conditional->parent();
  // save the old users before add new conditional user instructions
  std::vector<HloInstruction*> old_conditional_users = conditional->users();
  // Maps boundaries in the conditional body to boundaries hoisted outside
  // the conditional that compute the same value.
  absl::flat_hash_map<Boundary, Boundary> hoisted_boundaries;
  // Insert GetTupleElement before the instructions whose operands might still
  // be within the conditional.
  VLOG(1) << "before opt:"
          << conditional_parent->ToString(HloPrintOptions::Fingerprint())
          << "\n";
  int64_t op_index = 0;
  for (const Boundary& b : new_boundaries) {
    HloInstruction* op = b.operands()[0];
    CHECK(op != nullptr);
    VLOG(2) << "Mapping new boundary instr: " << op->ToString() << "\n";
    HloInstruction* gtr = conditional_parent->AddInstruction(
        HloInstruction::CreateGetTupleElement(op->shape(), conditional,
                                              op_index++));
    Boundary b2(Boundary::Position::kOutsideBranch);
    b2.mutable_operands().push_back(gtr);
    hoisted_boundaries[b] = b2;
  }
  // Copy boundary instructions out of the conditional.
  // Visit the operands before its users and copy it, so that the copied
  // user will point to the correct operand.
  for (int64_t i = to_move_out.size() - 1; i >= 0; i--) {
    CopyOutOfConditional(to_move_out[i], conditional, hoisted_boundaries);
  }
  VLOG(2) << "Done copy branch instructions out\n"
          << conditional_parent->ToString(HloPrintOptions::Fingerprint())
          << "\n";
  // Change original users of the conditional to use the correct operands.
  for (auto user_instr : old_conditional_users) {
    VLOG(2) << "Checking conditional user: " << user_instr->ToString() << "\n";
    CHECK(user_instr->opcode() == HloOpcode::kGetTupleElement);
    auto tuple_opd = static_cast<HloGetTupleElementInstruction*>(user_instr);
    int64_t index = tuple_opd->tuple_index();
    Boundary old_user_boundary(Boundary::Position::kInsideBranch);
    for (const HloComputation* called_computation :
         conditional->called_computations()) {
      HloInstruction* root = called_computation->root_instruction();
      CHECK(root->operands().size() > index);
      old_user_boundary.mutable_operands().push_back(root->operands()[index]);
    }
    CHECK(ContainsKey(hoisted_boundaries, old_user_boundary));
    HloInstruction* new_opd =
        hoisted_boundaries[old_user_boundary].operands()[0];
    CHECK(new_opd != nullptr);
    VLOG(2) << "Try replace all uses of :" << old_user_boundary.ToString()
            << "\n";
    TF_RETURN_IF_ERROR(user_instr->ReplaceAllUsesWith(new_opd));
    TF_RETURN_IF_ERROR(conditional_parent->RemoveInstruction(user_instr));
  }
  VLOG(2) << "Done changing conditional users\n"
          << conditional_parent->ToString() << "\n";
  // Create tuple element within each branch and set it as root.
  int64_t branch_count = conditional->branch_count();
  for (int i = 0; i < branch_count; i++) {
    auto computation = conditional->branch_computation(i);
    std::vector<HloInstruction*> elements;
    for (const auto& b1 : new_boundaries) {
      HloInstruction* op = b1.operands()[i];
      CHECK(op != nullptr);
      VLOG(2) << "Adding to root " << i << " with " << op->ToString() << "\n";
      elements.push_back(op);
    }
    HloInstruction* tuple =
        computation->AddInstruction(HloInstruction::CreateTuple(elements));
    computation->set_root_instruction(tuple, true);
    VLOG(2) << "computation is :" << computation->ToString() << "\n";
    // Remove hoisted instructions from the branches.
    for (const auto& b2 : to_move_out) {
      auto instr_to_remove = b2.operands()[i];
      // Double check to make sure it is safe to delete the instruction.
      // Complications may arise due to some operations in the alternative
      // branches (branches 1..n) being placed into the boundaries multiple
      // times.
      if (!computation->IsMarkedAsDead(instr_to_remove) &&
          instr_to_remove->IsDead()) {
        VLOG(2) << "Removing boundary:" << b2.ToString() << "\n";
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instr_to_remove));
      }
    }
  }
  // Change conditional instruction shape to the shape of the new root.
  HloInstruction* new_root =
      conditional->branch_computation(0)->root_instruction();
  *conditional->mutable_shape() = new_root->shape();
  // Keep conditional instruction sharding consistent with the branches. Note
  // that this sharding could be lost after this pass.
  conditional->set_sharding(new_root->sharding_ptr());
  VLOG(1) << "done moving instructions out of branches\n"
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
  VLOG(1) << "Modifying code---number of boundaries to move in:"
          << to_move_in.size() << "\n";
  VLOG(1) << "before opt:"
          << conditional->parent()->ToString(HloPrintOptions::Fingerprint())
          << "\n";
  // Mapping boundaries to be moved to their new representations.
  absl::flat_hash_map<Boundary, Boundary> hoisted_boundaries;
  int64_t to_move_in_size = to_move_in.size();
  int64_t branch_count = conditional->branch_count();
  HloGetTupleElementInstruction* tuple_use =
      DynCast<HloGetTupleElementInstruction>(to_move_in[0].operands()[0]);
  // If use_index is -1, the old conditional root entry used by to_move_in
  // instructions still need to be included as an entry of the modified
  // conditional root, and the new result of the to_move_in instructions
  // need to be added as an extra entry of the modified root; otherwise, the
  // old root entry will be replaced with the new result in the modified root.
  // The entry replacement should be allowed only if tuple_use has <=1 users.
  int64_t use_index = (tuple_use != nullptr && tuple_use->user_count() == 1)
                          ? tuple_use->tuple_index()
                          : -1;
  VLOG(2) << "Tuple use index = " << use_index << "\n";
  // Number of old conditional entries still to be used outside.
  // If conditional shape is not tuple, will create a tuple and use subscript
  // 0 to save the old operand being used.
  int64_t op_index =
      conditional->shape().IsTuple()
          ? ((use_index >= 0) ? conditional->shape().tuple_shapes_size() - 1
                              : conditional->shape().tuple_shapes_size())
          : 0;
  // Use to map the tuple_use instruction to its operand;
  Boundary b_opd_use(Boundary::Position::kInsideBranch);
  Boundary b_old_root(Boundary::Position::kInsideBranch);
  // Create a new root instruction in each branch.
  for (int i = 0; i < branch_count; i++) {
    auto computation = conditional->branch_computation(i);
    auto old_root = computation->root_instruction();
    b_old_root.mutable_operands().push_back(old_root);
    std::vector<HloInstruction*> operands;
    if (old_root->opcode() == HloOpcode::kTuple) {
      // Use operands of old_root directly, so old_root can be removed later.
      for (int i = 0; i < old_root->operand_count(); ++i) {
        if (i != use_index) {
          operands.push_back(old_root->operands()[i]);
        } else {  // Map conditional use to the tuple operand.
          b_opd_use.mutable_operands().push_back(old_root->operands()[i]);
        }
      }
    } else if (old_root->shape().IsTuple()) {
      // If old_root is not a kTuple but has tuple shape, elements within the
      // tuple must be extracted first to be used by the new instructions.
      const Shape& old_shape = old_root->shape();
      for (int i = 0; i < old_shape.tuple_shapes_size(); ++i) {
        auto element =
            computation->AddInstruction(HloInstruction::CreateGetTupleElement(
                old_shape.tuple_shapes(i), old_root, i));
        if (i != use_index) {
          operands.push_back(element);
        } else {
          b_opd_use.mutable_operands().push_back(element);
        }
      }
    } else {
      // If old_root is not a tuple and does not have tuple shape, use it
      // to replace the conditional directly in the new computation.
      b_opd_use.mutable_operands().push_back(conditional);
    }

    HloInstruction* new_root =
        computation->AddInstruction(HloInstruction::CreateTuple(operands));
    VLOG(2) << "setting new root: " << new_root->ToString() << "\n";
    computation->set_root_instruction(new_root,
                                      /*accept_different_shape*/ true);
    if (old_root->opcode() == HloOpcode::kTuple) {
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(old_root));
    }
    VLOG(2) << "new branch computation: " << computation->ToString() << "\n";
  }
  // Update get tuple element index of the conditional.
  if (use_index != -1) {
    for (auto* user : conditional->users()) {
      if (user->opcode() == HloOpcode::kGetTupleElement &&
          user->tuple_index() > use_index) {
        user->set_tuple_index(user->tuple_index() - 1);
      }
    }
  }
  Boundary conditional_boundary(Boundary::Position::kOutsideBranch);
  conditional_boundary.mutable_operands().push_back(conditional);
  hoisted_boundaries[conditional_boundary] = b_old_root;
  int64_t cp_start = 0;
  if (use_index >= 0) {
    VLOG(2) << "Mapping GTE: " << tuple_use->ToString() << "\n";
    Boundary tuple_use_boundary(Boundary::Position::kOutsideBranch);
    tuple_use_boundary.mutable_operands().push_back(tuple_use);
    hoisted_boundaries[tuple_use_boundary] = b_opd_use;
  }
  cp_start = (tuple_use != nullptr) ? 1 : 0;
  for (int64_t to_move_index = cp_start; to_move_index < to_move_in_size;
       to_move_index++) {
    Boundary b_to_move = to_move_in[to_move_index];
    HloInstruction* op = b_to_move.operands()[0];
    CHECK(op != nullptr);
    bool to_be_used_outside = true;
    VLOG(2) << "Mapping new boundary instr: " << op->ToString() << "\n";
    if (to_move_index < to_move_in_size - 1 && op->user_count() == 1 &&
        op->users()[0] == to_move_in[to_move_index + 1].operands()[0]) {
      to_be_used_outside = false;
      VLOG(2) << "Instruction is not to be used outside the branch\n";
    }
    Boundary b(Boundary::Position::kInsideBranch);
    CopyIntoConditional(b_to_move, conditional, hoisted_boundaries);
    if (to_be_used_outside) {
      for (int i = 0; i < branch_count; ++i) {
        auto computation = conditional->branch_computation(i);
        auto new_op = hoisted_boundaries[b_to_move].operands()[i];
        auto new_root = computation->root_instruction();
        new_root->AppendOperand(new_op);
        *new_root->mutable_shape()->add_tuple_shapes() = new_op->shape();
        VLOG(2) << "Extending conditional root " << i << " : "
                << new_root->ToString() << "\n";
      }
      // Modify uses of instructions outside of the conditionals
      HloInstruction* gtr = conditional->parent()->AddInstruction(
          HloInstruction::CreateGetTupleElement(op->shape(), conditional,
                                                op_index++));
      TF_RETURN_IF_ERROR(op->ReplaceAllUsesWith(gtr));
      if (conditional->parent()->root_instruction() == op) {
        conditional->parent()->set_root_instruction(gtr);
      }
    }
  }
  VLOG(2) << "Done copying instructions inside branch: "
          << conditional->ToString(HloPrintOptions::Fingerprint()) << "\n";
  // Change conditional instruction shape to the shape of the new root.
  HloInstruction* new_root =
      conditional->branch_computation(0)->root_instruction();
  *conditional->mutable_shape() = new_root->shape();
  // Keep conditional instruction sharding consistent with the branches. Note
  // that this sharding could be lost after this pass.
  conditional->set_sharding(new_root->sharding_ptr());
  VLOG(2) << "Before removing instructions:"
          << conditional->parent()->ToString() << "\n";
  // Remove hoisted instructions from the branches.
  for (int64_t i = to_move_in_size - 1; i >= 0; i--) {
    Boundary boundary_to_move_in = to_move_in[i];
    HloInstruction* op = boundary_to_move_in.operands()[0];
    if (op->user_count() == 0) {
      VLOG(2) << "Removing boundary:" << boundary_to_move_in.ToString() << "\n";
      TF_RETURN_IF_ERROR(conditional->parent()->RemoveInstruction(op));
      VLOG(2) << "Done removing boundary.\n";
    }
  }

  // Reset shapes of user gtes to the new shape.
  if (use_index != -1) {
    for (auto* user : conditional->users()) {
      if (user->opcode() == HloOpcode::kGetTupleElement) {
        VLOG(2) << "Resetting shape of user: " << user->ToString() << "\n";
        *user->mutable_shape() =
            conditional->shape().tuple_shapes(user->tuple_index());
      }
    }
  }
  VLOG(1) << "Done moving instructions inside branches\n"
          << conditional->parent()->ToString(HloPrintOptions::Fingerprint())
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
  // Instructions that have been visited but are not going to be moved.
  absl::flat_hash_map<HloInstruction*, int>& visited_count_;
  // The following four lines are configurations of the cost model, which will
  // be used to determine whether to move an instruction (move_config_) and how
  // strongly preferred it is to keep a pair of ops together (reuse_config_).
  // The search_config_ is used to control how to navigate the search space of
  // the cost model in the context of auto/manual tuning. The flipped array is
  // used to save which entries in the configuration have been changed in the
  // search/tuning process.
  std::vector<std::vector<int64_t>>& move_config_;
  std::vector<std::vector<int64_t>>& reuse_config_;
  absl::Span<int64_t> search_config_vec_;
  int64_t& search_config_;
  int64_t search_subscript_;
  absl::flat_hash_map<const int64_t*, int64_t> flipped_;

  // The FlipMutation function serves to implement the search of alternative
  // cost models by deciding whether to flip a given configuration, saved in
  // the loc parameter. The non_zero parameter provides the new value to use
  // to flip a zero. The msg parameter is only used for debugging purpposes.
  int64_t FlipMutation(int64_t* loc, const int64_t non_zero,
                       const std::string& msg) {
    if (search_config_ == 0 || ContainsKey(flipped_, loc)) {
      VLOG(2) << "Configured not to search or loc is already flipped.";
      return *loc;
    }
    // The last 8 digits control when to start the first flip.
    int c = ConditionalCodeMotion::flip_start(search_config_);
    VLOG(2) << "flip start index = " << c << "\n";
    // Only flip the decision if c reaches 0.
    if (c > 0) {
      search_config_--;
      return *loc;
    }
    // The 8-16 digits control the maximum number of times to flip a config.
    auto flip_count = ConditionalCodeMotion::DecrementMaxFlip(&search_config_);
    VLOG(2) << "max flip count = " << flip_count << "\n";
    VLOG(2) << "Updating max Flipping configuration = " << search_config_
            << "\n";
    if (flip_count == 0) {
      VLOG(2) << "Maximum flip count has reached. ";
      if (search_subscript_ + 1 < search_config_vec_.size()) {
        VLOG(2) << "search_subscript_ = " << search_subscript_;
        VLOG(2) << "search config vec size = " << search_config_vec_.size();
        search_config_ = search_config_vec_[++search_subscript_];
      } else {
        return *loc;
      }
    }
    // Reload the 16-23 digits of the configuration, which controls how
    // frequently a configuration should be flipped.
    auto flip_stride = ConditionalCodeMotion::flip_stride(search_config_);
    search_config_ += flip_stride;
    VLOG(2) << "flip stride = " << flip_stride << "\n";
    VLOG(2) << "Updating Flipping Stride = " << search_config_ << "\n";

    flipped_[loc] = *loc;
    // Copy the last 8 bits back to the first 8 bits of configuration.
    switch (*loc) {
      case 0:
        *loc = non_zero;
        break;
      default:
        *loc = 0;
        break;
    }
    VLOG(2) << "Flipping decision for: " << msg << ": from " << flipped_[loc]
            << " to " << *loc << "\n";
    return *loc;
  }

  static std::vector<int64_t>& EnsureSearchConfig(
      std::vector<int64_t>& search_config) {
    if (search_config.empty()) {
      search_config.push_back(0);
    }
    return search_config;
  }

 public:
  explicit GroupConnectedBoundaries(
      HloInstruction* conditional, bool is_layout_sensitive,
      absl::flat_hash_map<HloInstruction*, int>& visited_count,
      std::vector<std::vector<int64_t>>* move_config,
      std::vector<std::vector<int64_t>>* reuse_config,
      std::vector<int64_t>& search_config)
      : conditional_(conditional),
        conditional_parent_(conditional->parent()),
        is_layout_sensitive_(is_layout_sensitive),
        visited_count_(visited_count),
        move_config_(*move_config),
        reuse_config_(*reuse_config),
        search_config_vec_(EnsureSearchConfig(search_config)),
        search_config_(search_config_vec_.front()),
        search_subscript_(0) {
    VLOG(2) << "Initializing Group Connected Boundaries\n";
  }
  // Returns estimation of potential reuses carried by a given pair of
  // instructions. Use different integers to classify different levels
  // of reuses. Assume all instructions can be fused to enable data reuses.
  int64_t ReusesCarriedBy(HloInstruction* op, HloInstruction* user) {
    std::vector<int64_t>& curconfig =
        reuse_config_[static_cast<uint32_t>(op->opcode())];
    // Flip the reuse configuration if tuning the cost model.
    // When flipping, use -10 if flipping to the default reuse model. Other
    // values can be specified if needed to fine-control the decision making.
    int64_t config =
        (search_config_ < 0)
            ? FlipMutation(&curconfig[static_cast<uint32_t>(user->opcode())],
                           -10,
                           HloOpcodeString(op->opcode()) + "->" +
                               HloOpcodeString(user->opcode()))
            : curconfig[static_cast<uint32_t>(user->opcode())];
    VLOG(2) << "ConditionalCodeMotion: Add reuses carried by instr: "
            << op->ToString() << "=>" << user->ToString() << " : " << config
            << "\n";
    if (config < 0) {
      // Assume the reuse decreases with increasing user count.
      int count1 = CountNonLeafOps(op->users());
      int count2 = CountNonLeafOps(user->operands());
      return (-config) / count1 / count2;
    }
    return config;
  }
  void clear_recently_visited() {
    for (const auto& boundary : new_boundaries_) {
      visited_count_.erase(boundary.operands()[0]);
    }
  }
  // Returns true if `instruction` is worth hoisting.
  bool WorthHoisting(HloInstruction* instruction, bool is_inside_branch) {
    // This is needed for the "moving-in" transformation, to prevent the root
    // of the parent computation (which contains the conditional) to be moved
    // inside the conditional.
    HloOpcode opcode = instruction->opcode();
    if (opcode == HloOpcode::kTuple &&
        instruction == conditional_parent_->root_instruction()) {
      return false;
    }
    // It is not safe to move collective ops from outside to inside
    // conditional branches, as it may cause synchronization problems,
    // when different layouts are assigned to different branches.
    if (DynCast<HloCollectiveInstruction>(instruction) && !is_inside_branch) {
      return false;
    }

    // It is not legal to move the parameter instructions.
    if (opcode == HloOpcode::kParameter) {
      return false;
    }

    // Use configuration given from outside (e.g., by autotuner).
    std::vector<int64_t>& curconfig =
        move_config_[static_cast<uint32_t>(opcode)];
    auto col = (curconfig.size() == 1) ? 0
               : (instruction->operand_count() > 0)
                   ? static_cast<uint32_t>(instruction->operand(0)->opcode())
                   : 0;
    VLOG(2) << "column = " << col << "\n";
    VLOG(2) << "config size = " << curconfig.size() << "\n";
    VLOG(2) << "search_config = " << search_config_ << "\n";
    CHECK(col < curconfig.size());
    uint32_t config = (search_config_ > 0)
                          ? FlipMutation(&curconfig[col], 1,
                                         "Move-" + HloOpcodeString(opcode))
                          : curconfig[col];
    VLOG(2) << "Checking instruction is worth moving: " << config << "\n";
    VLOG(2) << "after checking search_config = " << search_config_ << "\n";
    return (config != 0);
  }

  int64_t ReusesBeforeBoundary(HloInstruction* user) {
    int64_t reuses = 0;
    for (auto op : user->operands()) {
      // The operand must be an instruction that is not going to be moved (if
      // user is inside the conditional); otherwise it must be the conditional
      // itself and its user must be outside of the conditional.
      if (!ContainsKey(visited_count_, op) && op != conditional_) {
        continue;
      }
      if (auto tuple_gte = DynCast<HloGetTupleElementInstruction>(user)) {
        if (op->opcode() == HloOpcode::kConditional) {
          auto tuple = op->branch_computation(0)->root_instruction();
          if (tuple->opcode() == HloOpcode::kTuple) {
            auto index = tuple_gte->tuple_index();
            CHECK(index < tuple->operand_count());
            op = tuple->mutable_operand(index);
          }
        }
        reuses += ReusesCarriedBy(op, user->users()[0]);
      } else {
        reuses += ReusesCarriedBy(op, user);
      }
    }
    VLOG(2) << "Reuses before instruction " << user->ToString() << ":" << reuses
            << "\n";
    return reuses;
  }

  int64_t ReusesAfterBoundary(HloInstruction* user) {
    CHECK(user != nullptr);
    auto all_users = user->users();
    // For now, assume that if an instruction has multiple-consumers, it
    // will not be reused, as the reuse may require duplication in
    // fusion and so is expensive. If the situation changes in the future,
    // some aspects of the overall algorithm need to be redesigned to
    // accommandate the change.
    if (all_users.size() > 1) {
      VLOG(2) << "Having multiple users from: " << user->ToString() << "\n";
      return 0;
    }
    if (!all_users.empty()) {
      auto op = all_users[0];
      int64_t reuses = 0;
      // Only count reuses that run through the conditional root.
      if (op == conditional_->branch_computation(0)->root_instruction()) {
        int64_t index = op->operand_index(user);
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
      } else if (ContainsKey(visited_count_, op)) {
        reuses += ReusesCarriedBy(user, op);
      }
      VLOG(2) << "reuses after instruction " << user->ToString() << ":"
              << reuses << "\n";
      return reuses;
    }
    return 0;
  }

  int64_t BenefitForMovingBoundaries(const std::vector<Boundary>& boundaries,
                                     bool perform_reuse_analysis = true) {
    int64_t reuses_before = 0, reuses_after = 0;
    if (boundaries.size() == 1) {
      if (boundaries[0].IsOutsideBranch() &&
          boundaries[0].operands()[0]->opcode() ==
              HloOpcode::kGetTupleElement) {
        // The only boundary of moving-in is the get_tuple_element op.
        return -1;
      }
      if (boundaries[0].IsInsideBranch() &&
          boundaries[0].operands()[0]->opcode() == HloOpcode::kTuple) {
        // The only boundary of moving-out is the tuple op inside branches.
        return -1;
      }
    }
    // If trying alternative moving configurations, turn off reuse analysis.
    if (!perform_reuse_analysis) {
      return 1;
    }
    // For cases like :
    // branch0 {
    //   ROOT copy
    // }
    // branch1 {
    //   ...
    // }
    // cond = conditional(branch0, branch1)
    // copy = copy(cond)
    //
    // We can fold the two copies thus reducing computation.
    auto get_copy_folding_benefit = [&](HloInstruction* hlo) -> int64_t {
      if (hlo->opcode() != HloOpcode::kCopy) {
        return 0;
      }
      const HloGetTupleElementInstruction* gte =
          DynCast<HloGetTupleElementInstruction>(hlo->operand(0));
      if (gte == nullptr) {
        return 0;
      }
      const HloInstruction* conditional = gte->operand(0);
      if (conditional != conditional_) {
        return 0;
      }
      int64_t benefit = 0;
      for (auto* branch : conditional->called_computations()) {
        HloInstruction* root = branch->root_instruction();
        if (root->opcode() == HloOpcode::kTuple) {
          const auto* tuple_operand = root->operand(gte->tuple_index());
          if (tuple_operand->opcode() == HloOpcode::kCopy) {
            if (Shape::Equal()(tuple_operand->operand(0)->shape(),
                               hlo->shape())) {
              benefit += 10;
            }
          }
        }
      }
      return benefit;
    };
    for (const Boundary& b : boundaries) {
      auto op = b.operands()[0];
      if (op == conditional_->branch_computation(0)->root_instruction()) {
        continue;
      }
      VLOG(2) << "Benefit for " << op->ToString();
      reuses_before += ReusesBeforeBoundary(op);
      VLOG(2) << "Reuses before boundary so far: " << reuses_before << "\n";
      reuses_after += ReusesAfterBoundary(op);
      VLOG(2) << "Reuese after boundary so far : " << reuses_after << "\n";
    }

    int64_t copy_folding_benefit = 0;
    if (boundaries[0].IsOutsideBranch()) {
      for (const Boundary& b : boundaries) {
        auto op = b.operands()[0];
        copy_folding_benefit += get_copy_folding_benefit(op);
      }
    }
    VLOG(2) << "Copy folding benefit: " << copy_folding_benefit;

    if (reuses_after == 0 && reuses_before == 0 && copy_folding_benefit == 0) {
      return -1;
    } else if (boundaries[0].IsInsideBranch()) {
      return reuses_after - reuses_before;
    } else {
      return reuses_before - reuses_after - 1 + copy_folding_benefit;
    }
  }

  Boundary GetNextBoundary(const Boundary& b, int64_t op_index) {
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

  // Checking whether it is safe to move a boundary when visited through a
  // dependent already considered for moving.
  bool IsSafeToMoveBoundary(const Boundary& next_boundary) {
    int64_t next_boundary_count =
        (next_boundary.IsInsideBranch())
            ? next_boundary.operands()[0]->user_count()
            : CountNonLeafOps(next_boundary.operands()[0]->operands());
    if (next_boundary_count <= 1) {
      // If boundary has only a single or no dependent, safe to move.
      return true;
    } else {
      if (!ContainsKey(visited_count_, next_boundary.operands()[0])) {
        VLOG(2) << "Skip next boundary " << next_boundary.ToString() << "\n"
                << " because it has multiple dependents: "
                << next_boundary_count << "\n";
        visited_count_[next_boundary.operands()[0]] = 1;
        new_boundaries_.push_back(next_boundary);
      } else {
        auto pos = std::find(new_boundaries_.begin(), new_boundaries_.end(),
                             next_boundary);
        if (pos != new_boundaries_.end() ||
            next_boundary.operands().size() == 1) {
          int count = ++visited_count_[next_boundary.operands()[0]];
          if (count == next_boundary_count) {
            VLOG(2) << "Recovering next boundary " << next_boundary.ToString()
                    << "\n"
                    << " because all of its dependents have been visited: "
                    << next_boundary_count << "\n";
            visited_count_.erase(next_boundary.operands()[0]);
            if (pos != new_boundaries_.end()) {
              new_boundaries_.erase(pos);
            }
            return true;
          }
        } else {
          VLOG(2) << "Skip incompatible multi-dependent boundary: "
                  << next_boundary.ToString() << ":" << next_boundary_count
                  << "\n";
        }
      }
    }
    return false;
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
      VLOG(2) << "visiting boundary " << b.ToString() << "\n";
      if ((b.IsOutsideBranch() || InstructionWithinBranchIdentical(
                                      b.operands(), is_layout_sensitive_)) &&
          IsSafeToMoveBoundary(b) &&
          WorthHoisting(b.operands()[0], b.IsInsideBranch())) {
        connected_boundaries_.push_back(b);
        VLOG(2) << "boundary can be moved\n";
        int64_t operand_count = (b.IsInsideBranch())
                                    ? b.operands()[0]->operand_count()
                                    : b.operands()[0]->users().size();
        for (int i = 0; i < operand_count; i++) {
          Boundary next_boundary = GetNextBoundary(b, i);
          VLOG(2) << "Add operand/user " << i << " to visit later\n";
          visitor.AddToWorkList(next_boundary);
        }
      } else {
        VLOG(2) << "boundary cannot be moved\n";
        visited_count_[b.operands()[0]] = 1;
        new_boundaries_.push_back(b);
      }
    }
  }
  std::vector<Boundary> BoundariesToMoveInOrOut(HloInstruction* conditional,
                                                const Boundary& b) {
    // At the beginning of optimization, a conditional itself is added to a
    // worklist. Here the conditional is expanded into two sets of boundaries:
    // the first set contains the boundary that is inside branches and
    // contains the root of all branches; the second set of boundaries
    // contains all the users of the conditional.
    HloInstruction* inst = b.operands()[0];
    if (inst == conditional) {
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
    std::vector<Boundary>& to_move, std::vector<Boundary>& new_boundaries,
    absl::flat_hash_map<HloInstruction*, int>& visited_count) {
  GroupConnectedBoundaries connect(conditional, is_layout_sensitive_,
                                   visited_count, &move_config_, &reuse_config_,
                                   search_config_);
  auto move_in_or_out =
      connect.BoundariesToMoveInOrOut(conditional, cur_boundary);
  if (!move_in_or_out.empty()) {
    auto benefit = connect.BenefitForMovingBoundaries(
        move_in_or_out, search_config_map_.empty());
    VLOG(2) << "benefit of moving in or out "
            << cur_boundary.operands()[0]->ToString() << ":" << benefit << "\n";
    if (benefit >= 0) {
      new_boundaries.clear();
      connect.AddNewBoundaries(new_boundaries);
      // The whole sequence in move_in_or_out is either all moving into a
      // conditional, or all moving out of a conditional. So looking only
      // at the first entry of the sequence is sufficient to know which
      // direction the move is intended.
      to_move = move_in_or_out;
      return Decision(to_move[0].IsInsideBranch()
                          ? Decision::Direction::kMoveOutOfBranch
                          : Decision::Direction::kMoveIntoBranch,
                      benefit);
    } else {
      connect.clear_recently_visited();
    }
  } else {
    connect.AddNewBoundaries(new_boundaries);
  }
  return Decision(Decision::Direction::kNoChange, 0);
}

StatusOr<bool> ConditionalCodeMotion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "Begin a new pass of conditional code motion optimization.\n";
  // Use to support debugging of optimization, by disabling the opt after it has
  // been applied a pre-determined times (to isolate impact of transformations).
  if (!ConsumeFuel("conditional_code_motion", [&] {
        return "Skipping conditional opt after allowed limit reaching 0.\n";
      })) {
    return false;
  }
  bool changed = false;
  bool cleanup_changed = false;
  {
    HloPassPipeline subpipeline("before_conditional_code_motion");
    subpipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/is_layout_sensitive_);
    subpipeline.AddPass<HloDCE>();
    TF_ASSIGN_OR_RETURN(auto cleanup_changed_now,
                        subpipeline.Run(module, execution_threads));
    cleanup_changed |= cleanup_changed_now;
  }
  // Gather all the conditional ops in the module ahead of time, to avoid
  // potential complications of modifying the code that affecting traversal.
  std::vector<HloInstruction*> conditional_ops;
  // Track how many times each branch computation is shared.
  absl::flat_hash_map<HloComputation*, int> conditional_computations;
  for (auto* comp : module->MakeComputationPostOrder(execution_threads)) {
    for (auto* instr : comp->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kConditional) {
        int branch_count = instr->branch_count();
        for (int i = 0; i < branch_count; ++i) {
          HloComputation* branch_i = instr->branch_computation(i);
          if (ContainsKey(conditional_computations, branch_i)) {
            conditional_computations[branch_i]++;
          } else {
            conditional_computations[branch_i] = 0;
          }
        }
        if (instr->shape().IsTuple()) {
          bool can_change_tuple_shape = true;
          for (auto user : instr->users()) {
            VLOG(2) << "user is : " << user->ToString() << "\n";
            if (user->opcode() != HloOpcode::kGetTupleElement) {
              can_change_tuple_shape = false;
            }
          }
          if (can_change_tuple_shape) {
            conditional_ops.push_back(instr);
          }
        } else {
          conditional_ops.push_back(instr);
        }
      }
    }
  }

  int64_t conditional_index = 0;
  // Use to collect mappings between cloned instructions.
  HloCloneContext clone_context(module);
  for (HloInstruction* conditional : conditional_ops) {
    if (conditional_index == 0 || !search_config_map_.empty()) {
      auto config_entry = search_config_map_.find(conditional_index);
      if (config_entry != search_config_map_.end()) {
        search_config_ = (*config_entry).second;
        VLOG(2) << "config entry value extracted:" << search_config_.size();
        search_config_index_ = 0;
      }
      VLOG(2) << "Obtaining default configuration for conditional "
              << conditional_index << "\n";
      SetDefaultMoveConfig();
      VLOG(2) << "Done obtaining default configuration\n";
      conditional_index++;
    }
    int branch_count = conditional->branch_count();
    // check for shared conditional computations
    bool conditional_is_shared = false;
    for (int i = 0; i < branch_count; ++i) {
      HloComputation* branch_i = conditional->branch_computation(i);
      if (conditional_computations[branch_i] > 0) {
        conditional_is_shared = true;
        break;
      }
    }

    // Boundaries to move out or to move into the branches.
    std::vector<std::vector<Boundary>> to_move_out, to_move_in;
    std::vector<std::vector<Boundary>> new_boundaries_for_moveout;
    std::vector<std::vector<Boundary>> new_boundaries_for_movein;
    // Number of times each instruction has been visited for moving.
    absl::flat_hash_map<HloInstruction*, int> visited_count;
    int benefit_move_out = 0, benefit_move_in = 0;
    Decision::Direction final_d = Decision::Direction::kNoChange;
    // The conditional is moved into a worklist as the seed (starting point).
    // The conditional will be expanded into multiple seeds (starting points),
    // its roots and its users, when it is visited by GroupConnectedBoundaries.
    // A NO_CHANGE decision will always be returned for the conditional itself,
    // so that the other seeding boundaries can be visited in turn.
    BoundaryVisitor visitor(conditional);
    VLOG(2) << "Analyzing conditional:" << conditional->ToString() << "\n";
    // Try visit all the boundaries, collect the analysis results, and save
    // all the benefitical non-conflicting decisions. If two decisions conflict
    // with each other, save the more benefitical one.
    while (visitor.HasNextBoundary()) {
      std::vector<Boundary> to_move, next_boundary;
      Boundary boundary = visitor.PopNextBoundary();
      VLOG(2) << "Analyzing boundary:" << boundary.ToString() << "\n";
      auto d = ConsiderCodeMotion(conditional, boundary, to_move, next_boundary,
                                  visited_count);
      switch (d.GetDirection()) {
        case Decision::Direction::kMoveOutOfBranch:
          VLOG(2) << "Local Decision is move out of branch\n";
          to_move_out.push_back(to_move);
          new_boundaries_for_moveout.push_back(next_boundary);
          benefit_move_out += d.GetBenefit();
          if (benefit_move_out >= benefit_move_in) {
            final_d = Decision::Direction::kMoveOutOfBranch;
            VLOG(2) << "Current Decision is move out of branch ("
                    << to_move_out.size() << ")\n";
          } else {
            VLOG(2) << "Current Decision remains move into branch\n";
          }
          break;
        case Decision::Direction::kMoveIntoBranch:
          VLOG(2) << "Decision is move into branch\n";
          to_move_in.push_back(to_move);
          new_boundaries_for_movein.push_back(next_boundary);
          benefit_move_in += d.GetBenefit();
          if (benefit_move_out >= benefit_move_in) {
            VLOG(2) << "Current Decision remains move out of branch\n";
          } else {
            final_d = Decision::Direction::kMoveIntoBranch;
            VLOG(2) << "Current Decision is move into branch ("
                    << to_move_in.size() << ")\n";
          }
          break;
        case Decision::Direction::kNoChange:
          VLOG(2) << "Decision is no change\n";
          for (const Boundary& b : next_boundary) {
            visitor.AddToWorkList(b);
            VLOG(2) << "Adding new boundary to worklist:" << b.ToString()
                    << "\n";
          }
          break;
      }
    }
    // If modification is to be made, need to clone the shared branches.
    if (final_d != Decision::Direction::kNoChange && conditional_is_shared) {
      for (int i = 0; i < branch_count; ++i) {
        HloComputation* branch_i = conditional->branch_computation(i);
        if (conditional_computations[branch_i] > 0) {
          // Cloning is absolutely needed if the computation is shared by
          // different branches, but the cloning can be potentially avoided
          // if the sharing is only among branches of the same conditional.
          // If cloning these branches causes a problem due to space issues,
          // a fix can pass a vector of unique branches to the actual
          // transformations, as an alternative representation of the
          // conditional branches to be modified. Right now we assume the
          // overhead of cloning is minimal since later stages of the compiler
          // inline all the computations anyway.
          HloComputation* clone_i =
              conditional->parent()->parent()->AddEmbeddedComputation(
                  branch_i->Clone("clone", &clone_context));
          conditional->set_branch_computation(i, clone_i);
          conditional_computations[branch_i]--;
          // Need to translate the analysis result to generate correct result.
          auto update_boundary = [&](Boundary& boundary) {
            auto cloned_instr =
                clone_context.FindInstruction(boundary.operands()[i]);
            CHECK(cloned_instr != nullptr);
            VLOG(2) << "boundary before cloning:" << boundary.operands()[i]
                    << "\n";
            boundary.mutable_operands()[i] = cloned_instr;
            VLOG(2) << "boundary after cloning:" << boundary.operands()[i]
                    << "\n";
          };
          // Only boundaries to move out need to be updated.
          if (final_d == Decision::Direction::kMoveOutOfBranch) {
            for (int i = 0; i < to_move_out.size(); ++i) {
              std::vector<Boundary>& m = to_move_out[i];
              std::for_each(m.begin(), m.end(), update_boundary);
            }
            for (int i = 0; i < new_boundaries_for_moveout.size(); ++i) {
              std::vector<Boundary>& m = new_boundaries_for_moveout[i];
              std::for_each(m.begin(), m.end(), update_boundary);
            }
          }
        }
      }
      VLOG(2) << "Cloned branches as needed: " << conditional->ToString()
              << "\n";
    }
    // At most one of to_move_out or to_move_in can be non-empty, since there is
    // only one optimization decision.
    if (final_d == Decision::Direction::kMoveOutOfBranch) {
      CHECK(to_move_out.size() == new_boundaries_for_moveout.size());
      for (int i = 0; i < to_move_out.size(); ++i) {
        TF_ASSIGN_OR_RETURN(bool result,
                            MoveInstructionOut(conditional, to_move_out[i],
                                               new_boundaries_for_moveout[i]));
        changed |= result;
      }
      VLOG(2) << "Done moving out of branches " << to_move_out.size()
              << " times. \n";
      if (!ConsumeFuel("conditional_code_motion", [&] {
            return "Skipping conditional opt after allowed limit reaching 0.\n";
          })) {
        break;
      }
    } else if (final_d == Decision::Direction::kMoveIntoBranch) {
      CHECK(to_move_in.size() == new_boundaries_for_movein.size());
      for (int i = 0; i < to_move_in.size(); ++i) {
        TF_ASSIGN_OR_RETURN(bool result,
                            MoveInstructionIn(conditional, to_move_in[i],
                                              new_boundaries_for_movein[i]));
        changed |= result;
      }
      VLOG(2) << "Done moving into branches " << to_move_in.size()
              << " times. \n";
      if (!ConsumeFuel("conditional_code_motion", [&] {
            return "Skipping conditional opt after allowed limit reaching 0.\n";
          })) {
        break;
      }
    } else if (pursue_full_conditional_code_motion_ && !conditional_is_shared) {
      // Invoke special handling for convert rematerialization/hoisting
      // We need to make sure no sharing is present in the branches because no
      // cloning has been done by the earlier analysis.
      // TOOD[b/165848866]: extend solution to handle cloning for special move.
      TF_ASSIGN_OR_RETURN(
          bool convert_result,
          ConvertSpecialMove(conditional, is_layout_sensitive_));
      if (convert_result) {
        VLOG(2) << "Done special moving of convert\n";
        if (!ConsumeFuel("conditional_code_motion", [&] {
              return "Skipping conditional opt after allowed limit reaching "
                     "0.\n";
            })) {
          break;
        }
      }
      changed |= convert_result;
    }
  }
  if (changed) {
    HloPassPipeline subpipeline(
        "after_conditional_code_motion_after_convert_hoisting");
    VLOG(2) << "starting after motion passes: DCE\n";
    subpipeline.AddPass<HloDCE>();
    subpipeline.AddPass<TupleSimplifier>();
    subpipeline.AddPass<HloDCE>();
    TF_ASSIGN_OR_RETURN(auto cleanup_changed_now, subpipeline.Run(module));
    cleanup_changed |= cleanup_changed_now;
  }
  if (cleanup_changed) {
    VLOG(2) << "subpipeline cleanup have modified code\n";
  }
  return changed;
}

void ConditionalCodeMotion::SetDefaultMoveConfig() {
  VLOG(2) << "search_config_index = " << search_config_index_ << "\n";
  VLOG(2) << "search_config_ size = " << search_config_.size() << "\n";
  int64_t cur_search_config = (search_config_index_ < 0 ||
                               search_config_index_ >= search_config_.size())
                                  ? 0
                                  : search_config_[search_config_index_];
  enum class TuningOption {
    kDoNotTune = 0,
    kTuneTransformationDecision = 1,
    kTuneReuseModel = 2,
  };
  TuningOption tuning_option =
      (cur_search_config == 0)  ? TuningOption::kDoNotTune
      : (cur_search_config > 0) ? TuningOption::kTuneTransformationDecision
                                : TuningOption::kTuneReuseModel;

  auto row = HloOpcodeCount();
  auto col = row;
  VLOG(2) << "Start setting default configuration\n";
  reuse_config_.clear();
  move_config_.clear();
  reuse_config_.reserve(row);
  move_config_.reserve(row);
  for (int64_t opcode = 0; opcode < row; ++opcode) {
    // To save whether an instruction is preferred to be moved.
    std::vector<int64_t> reuse_vec(col, 0);
    for (uint32_t j = 0; j < col; ++j) {
      reuse_vec[j] = ReusesCarriedBy(static_cast<HloOpcode>(opcode),
                                     static_cast<HloOpcode>(j));
    }
    reuse_config_.push_back(reuse_vec);
    std::vector<int64_t> move_vec;
    switch (tuning_option) {
      case TuningOption::kTuneTransformationDecision:
        // Tuning transformation decision --- start with all yes.
        // Only a single entry is needed if we don't consider operands of an op
        // when searching/tuning transformation decisions.
        move_vec.push_back(1);
        break;
        // Tune the ReusesCarriedBy results only.
      case TuningOption::kTuneReuseModel:
      case TuningOption::kDoNotTune:
        // No tuning --- use the default configuration.
        // Use the opcode of first operand to configure default.
        move_vec.reserve(col);
        for (uint32_t j = 0; j < col; ++j) {
          move_vec.push_back(WorthHoisting(static_cast<HloOpcode>(opcode),
                                           static_cast<HloOpcode>(j)));
        }
        break;
    }
    move_config_.push_back(move_vec);
  }
}

// The search configuration is specified using a string in the format of
// 'config1;config2; ...;config_n', where each config_i is in the format of
// 'index,start,max,stride' (four integers separated by comma), which specify
// the index number of the conditional being configured, the index of the first
// transformation decision to flip for the conditional, the max number of
// decisions to flip, and how many decisions to skip in between the flips.
void ConditionalCodeMotion::ParseSearchConfiguration(
    const std::string& search_config) {
  if (search_config.empty()) {
    return;
  }
  search_config_index_ = 0;
  std::vector<std::string> configs = absl::StrSplit(search_config, ';');
  for (const std::string& config : configs) {
    std::vector<std::string> specs = absl::StrSplit(config, ',');
    CHECK_EQ(specs.size(), 4);
    int64_t condition_index;
    CHECK(absl::SimpleAtoi(specs[0], &condition_index));
    auto& cur_config_entry = search_config_map_[condition_index];
    int64_t flip_start, max_flip, flip_stride;
    CHECK(absl::SimpleAtoi(specs[1], &flip_start));
    CHECK(absl::SimpleAtoi(specs[2], &max_flip));
    CHECK(absl::SimpleAtoi(specs[3], &flip_stride));
    int64_t cur_config = MakeSearchConfig(flip_start, max_flip, flip_stride);
    cur_config_entry.push_back(cur_config);
    VLOG(2) << "Setting search config " << condition_index << "->" << cur_config
            << "\n";
  }
}

}  // namespace conditional_opt

}  // namespace xla

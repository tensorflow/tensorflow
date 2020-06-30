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

namespace {

struct ConditionalBoundary {
  ConditionalBoundary(HloInstruction* op, int64 op_index, HloInstruction* usr)
      : operand(op), operand_index(op_index), user(usr) {}
  // `operand` is one of `user`'s operand.

  // Instruction that remains in the conditional but one of its user
  // is moved out of conditonal.
  HloInstruction* operand;
  // operand_index for `operand` in the `user`.
  int64 operand_index;
  // Instruction that moved out of conditional.
  HloInstruction* user;
};

// Visit the root instructions to its operands follow BFS.
// Will visit an instructions after all its users have been visited. Parameters
// are not visited.
class BranchVisitor {
 public:
  explicit BranchVisitor(const HloComputation* branch_computation) {
    HloInstruction* root_inst = branch_computation->root_instruction();
    worklist_.push_back(root_inst);
    visited_.insert(root_inst);
    for (auto parameter_inst : branch_computation->parameter_instructions()) {
      parameter_instructions_.insert(parameter_inst);
    }
  }
  // Get next intruction to visit.
  HloInstruction* GetNextInstruction() {
    if (!worklist_.empty()) {
      HloInstruction* inst = worklist_.front();
      worklist_.pop_front();
      return inst;
    }
    return nullptr;
  }

  // Add operands of one instruction to worklist for further visit.
  void AddInstructionOperands(HloInstruction* inst) {
    int64 operand_count = inst->operand_count();
    for (int i = 0; i < operand_count; i++) {
      HloInstruction* operand = inst->mutable_operand(i);
      if (ContainsKey(visited_, operand)) {
        continue;
      }
      bool all_user_visited = std::all_of(
          operand->users().begin(), operand->users().end(),
          [&](HloInstruction* user) { return ContainsKey(visited_, user); });

      if (!all_user_visited) {
        continue;
      }
      // Do not visit parameter_instructions.
      if (ContainsKey(parameter_instructions_, operand)) {
        // Add the operand and this instruction to the boundaries.
        boundaries_.emplace_back(operand, i, inst);
        continue;
      }
      worklist_.push_back(operand);
      visited_.insert(operand);
    }
  }

  // Add instruction and its users to conditional boundaries.
  void AddInstructionToBoundary(HloInstruction* inst) {
    for (auto user : inst->users()) {
      boundaries_.emplace_back(inst, user->operand_index(inst), user);
    }
  }

  // Add instruction to the to be removed instructions set and vector.
  void AddInstructionToHoist(HloInstruction* inst) {
    instructions_to_hoist_set_.insert(inst);
    instructions_to_hoist_.emplace_back(inst);
  }

  // If visitor has next instruction to visit.
  bool HasNextInstruction() const { return !worklist_.empty(); }

  // If there is no hoist intruction.
  int64 HoistInstructionSize() { return instructions_to_hoist_.size(); }

  // Get boundaries of this branch.
  const std::vector<ConditionalBoundary>& boundaries() const {
    return boundaries_;
  }

  // Get instructions to hoist in this branch.
  const std::vector<HloInstruction*>& instructions_to_hoist() const {
    return instructions_to_hoist_;
  }

  // Get hoist instruction set in this branch.
  const std::unordered_set<HloInstruction*>& instructions_to_hoist_set() const {
    return instructions_to_hoist_set_;
  }

 private:
  // worklist is the deque that contains instructions to be visited.
  std::deque<HloInstruction*> worklist_;

  // instructions that has been visited.
  std::unordered_set<HloInstruction*> visited_;

  // parameter instructions of the branch.
  std::unordered_set<HloInstruction*> parameter_instructions_;

  // Boundaries contains the set of instructions that its operand is within
  // conditional but it can be hoist out of conditional.
  std::vector<ConditionalBoundary> boundaries_;

  // Instructions to hoist.
  std::unordered_set<HloInstruction*> instructions_to_hoist_set_;

  // Instructions to hoist, the order within this vector is BFS and
  // an instruction's order will always be after its users.
  std::vector<HloInstruction*> instructions_to_hoist_;
};

// Returns true if `instruction` is worth hoisting out.
bool WorthHoisting(HloInstruction* instruction) {
  for (const auto* operand : instruction->operands()) {
    // Only move out instructions that won't share the same operand
    // to avoid copy of the operand.
    if (operand->user_count() > 1) {
      return false;
    }
  }
  switch (instruction->opcode()) {
    case HloOpcode::kConvert:
      // If Convert is after AllReduce, it is worth moving out AllReduce out
      // of conditional for AR/CRS combine. If Convert is after other ops such
      // as Dot or Convolutional, it is better to keep convert within
      // conditional so that convert can be fused with Dot or Convolutional.
      //
      // TODO(b/154283721): figure out the scenario when convert can be fused
      // with AllReduce out of conditional.
      if (instruction->operand(0)->opcode() == HloOpcode::kAllReduce) {
        return true;
      }
      return false;
    case HloOpcode::kAllReduce:
    case HloOpcode::kAdd:
    case HloOpcode::kConstant:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply:
    case HloOpcode::kDivide:
    case HloOpcode::kTuple:
    case HloOpcode::kSqrt:
    case HloOpcode::kGetTupleElement:
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

  if (instructions[0] == nullptr) {
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

// Returns if all the visitors/branches has next instruction to visit.
bool HasNextInstruction(const std::vector<BranchVisitor>& visitors) {
  bool has_next = true;
  for (const auto& visitor : visitors) {
    has_next &= visitor.HasNextInstruction();
  }
  return has_next;
}

// Create tuple element as the new root of the branch. The tuple will contain
// the operands that can't move out of conditional but its user will be moved
// out of conditional.
HloInstruction* CreateNewRoot(
    const std::vector<ConditionalBoundary>& boundaries,
    const std::unordered_set<HloInstruction*>& instructions_to_hoist_set,
    HloComputation* computation) {
  std::vector<HloInstruction*> elements;
  elements.reserve(boundaries.size());
  for (auto boundary : boundaries) {
    if (ContainsKey(instructions_to_hoist_set, boundary.user)) {
      elements.push_back(boundary.operand);
    }
  }
  return computation->AddInstruction(HloInstruction::CreateTuple(elements));
}

// Copy identical instructions within conditional outside of conditional.
void CopyIdenticalInstructionsOutOfConditional(
    const std::vector<HloInstruction*>& instructions_to_hoist,
    HloComputation* conditional_parent,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>*
        hoisted_instructions) {
  int64 instructions_size = instructions_to_hoist.size();
  // Visit the operands before its users and copy it, so that the copied
  // user will point to the correct operand.
  for (int64 i = instructions_size - 1; i >= 0; i--) {
    HloInstruction* old_instruction = instructions_to_hoist[i];
    auto get_new_operand = [&](HloInstruction* old_operand) {
      // If the operand can't be found in `instructions_to_hoist`, this
      // operand will be in the `boundaries`, GetTupleElement instructions
      // will be added later to replace this operand.
      if (!ContainsKey(*hoisted_instructions, old_operand)) {
        return old_operand;
      }
      return FindOrDie(*hoisted_instructions, old_operand);
    };

    absl::InlinedVector<HloInstruction*, 4> new_operands;
    absl::c_transform(old_instruction->operands(),
                      std::back_inserter(new_operands), get_new_operand);

    HloInstruction* new_instruction = conditional_parent->AddInstruction(
        old_instruction->CloneWithNewOperands(old_instruction->shape(),
                                              new_operands));
    // Maps the instruction outside of conditional to the instruction
    // inside of the conditional.
    InsertOrDie(hoisted_instructions, old_instruction, new_instruction);
  }
}

// If there are instructions to hoist, the root of the conditional must be
// moved out. Change the users of the conditional to the hoisted instruction
// of the new root.
Status ChangeConditionalUsers(
    HloInstruction* conditional, HloInstruction* old_root,
    const absl::flat_hash_map<HloInstruction*, HloInstruction*>&
        hoisted_instructions) {
  HloInstruction* new_root = FindOrDie(hoisted_instructions, old_root);
  TF_RETURN_IF_ERROR(conditional->ReplaceAllUsesWith(new_root));
  return Status::OK();
}

// Insert GetTupleElement before the instructions whose operands might still
// be within the conditional.
Status CreateGetTupleElementAfterConditional(
    const std::vector<ConditionalBoundary>& boundaries,
    const std::unordered_set<HloInstruction*>& instructions_to_hoist_set,
    const absl::flat_hash_map<HloInstruction*, HloInstruction*>&
        hoisted_instructions,
    HloInstruction* conditional, HloComputation* computation) {
  int boundary_instruction_size = boundaries.size();

  // Inserts GetTupleElement before the boundary instructions.
  for (int i = 0; i < boundary_instruction_size; i++) {
    HloInstruction* gte =
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            boundaries[i].operand->shape(), conditional, i));

    HloInstruction* new_instruction =
        FindOrDie(hoisted_instructions, boundaries[i].user);
    TF_RETURN_IF_ERROR(
        new_instruction->ReplaceOperandWith(boundaries[i].operand_index, gte));
  }
  return Status::OK();
}

// Remove instructions to be hoisted out of the branch computation.
Status RemoveInstructionFromComputation(
    const std::vector<HloInstruction*>& instructions_to_hoist,
    HloComputation* branch) {
  // Will visit the instructions after its users.
  for (auto* instruction : instructions_to_hoist) {
    TF_RETURN_IF_ERROR(branch->RemoveInstruction(instruction));
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
StatusOr<bool> MergeIdenticalElements(HloInstruction* conditional,
                                      bool is_layout_sensitive) {
  VLOG(1) << " visiting conditional:" << conditional->ToString();
  int branch_count = conditional->branch_count();
  if (branch_count <= 0) {
    return false;
  }

  std::vector<BranchVisitor> visitors;
  visitors.reserve(branch_count);
  // Visit instructions from the root instruction to the operands using BFS.
  for (int i = 0; i < branch_count; i++) {
    visitors.emplace_back(BranchVisitor(conditional->branch_computation(i)));
  }

  // The instructions to be visited within each branch.
  std::vector<HloInstruction*> front_instructions(branch_count);

  while (HasNextInstruction(visitors)) {
    for (int i = 0; i < branch_count; i++) {
      front_instructions[i] = visitors[i].GetNextInstruction();
    }
    // If two instructions has the same shape, opcode and its operands has the
    // same shape, then this instruction can be moved out of conditional.
    if (WorthHoisting(front_instructions[0]) &&
        InstructionWithinBranchIdentical(front_instructions,
                                         is_layout_sensitive)) {
      for (int i = 0; i < branch_count; i++) {
        visitors[i].AddInstructionOperands(front_instructions[i]);
        visitors[i].AddInstructionToHoist(front_instructions[i]);
      }
    } else {
      for (int i = 0; i < branch_count; i++) {
        // If the ops are not identical, these ops and its users will
        // be in the boundaries` of the conditional. These ops will be stayed
        // within the conditional, but one its only user will be moved out
        // of conditional.
        visitors[i].AddInstructionToBoundary(front_instructions[i]);
      }
    }
  }

  if (visitors[0].HoistInstructionSize() < 1) {
    return false;
  }

  HloInstruction* old_root =
      conditional->branch_computation(0)->root_instruction();
  HloComputation* conditional_parent = conditional->parent();
  // Maps instructions in the conditional body to instructions hoisted outside
  // the conditional that compute the same value.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> hoisted_instructions;
  // Copy identical instructions out of the conditional.
  CopyIdenticalInstructionsOutOfConditional(visitors[0].instructions_to_hoist(),
                                            conditional_parent,
                                            &hoisted_instructions);
  // If there are instructions to hoist, the root of the conditional must be
  // moved out. Change the users of the conditional to the hoisted instruction
  // of the new root.
  TF_RETURN_IF_ERROR(
      ChangeConditionalUsers(conditional, old_root, hoisted_instructions));

  // Create tuple element within each branch and set it as root.
  for (int i = 0; i < branch_count; i++) {
    HloInstruction* tuple = CreateNewRoot(
        visitors[i].boundaries(), visitors[i].instructions_to_hoist_set(),
        conditional->branch_computation(i));
    conditional->branch_computation(i)->set_root_instruction(tuple, true);
  }
  // Changes conditional instruction shape to the shape of the new root.
  *conditional->mutable_shape() =
      conditional->branch_computation(0)->root_instruction()->shape();

  // Insert GetTupleElement before the instructions whose operands might still
  // be within the conditional.
  TF_RETURN_IF_ERROR(CreateGetTupleElementAfterConditional(
      visitors[0].boundaries(), visitors[0].instructions_to_hoist_set(),
      hoisted_instructions, conditional, conditional_parent));

  // Remove hoist instructions from the branches.
  for (int i = 0; i < branch_count; i++) {
    TF_RETURN_IF_ERROR(
        RemoveInstructionFromComputation(visitors[i].instructions_to_hoist(),
                                         conditional->branch_computation(i)));
  }
  return true;
}

}  // namespace

StatusOr<bool> ConditionalCodeMotion::Run(HloModule* module) {
  bool changed = false;

  if (pursue_full_conditional_code_motion_) {
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
          bool result,
          MergeIdenticalElements(conditional_op, is_layout_sensitive_));
      changed |= result;
    }

    if (changed) {
      HloPassPipeline subpipeline("after_conditional_code_motion");
      subpipeline.AddPass<HloDCE>();
      subpipeline.AddPass<TupleSimplifier>();
      subpipeline.AddPass<HloDCE>();
      TF_ASSIGN_OR_RETURN(bool cleanup_changed, subpipeline.Run(module));
      changed |= cleanup_changed;
    }
  }

  // handling convert rematerialization/hoisting
  {
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

}  // namespace xla

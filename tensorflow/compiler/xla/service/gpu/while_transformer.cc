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

#include "tensorflow/compiler/xla/service/gpu/while_transformer.h"

#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {

namespace {

// TODO(b/33483676) Use an expression tree to specify computations to pattern
// match for while transformations.

// ExprTree is a simple recursive data structure used to express computation
// patterns to match.
//
// Each ExprTree node is comprised of an HloOpcode, and a set of operands (each
// of type ExprTree). Operands can be added by specifying the index and HloOpcode
// of the operand.
//
// For example, the following computation:
//
//            Parameter
//               |
//   Const  GetTupleElemet
//      \   /
//       Add (root)
//
// Can be matched with the following expression tree:
//
//   ExprTree add(HloOpcode::kAdd,
//                ExprTree(HloOpcode::kConstant),
//                ExprTree(HloOpcode::kGetTupleElement,
//                         tuple_index, ExprTree(HloOpcode::kParameter)));
//
// Match the ExprTree root against an Hlo graph:
//
//   ExprTree::TaggedInstructionMap tagged_instructions;
//   TF_RETURN_IF_ERROR(add.Match(computation_->root_instruction(),
//                                &tagged_instructions));
//
// Instructions that are "tagged" with a context-specific string will
// be returned in 'tagged_instructions' for further procesing (i.e. parsing
// constants or recording the tuple_index).
//
class ExprTree {
 public:
  explicit ExprTree(HloOpcode opcode) : opcode_(opcode) {}
  ExprTree(HloOpcode opcode, const string& tag) : opcode_(opcode), tag_(tag) {}
  ExprTree(HloOpcode opcode, const ExprTree& operand0) : opcode_(opcode) {
    SetOperand(0, operand0);
  }
  ExprTree(HloOpcode opcode, int64 index0, const ExprTree& operand0)
      : opcode_(opcode) {
    SetOperand(index0, operand0);
  }
  ExprTree(HloOpcode opcode, int64 index0, const ExprTree& operand0,
           int64 index1, const ExprTree& operand1)
      : opcode_(opcode) {
    SetOperand(index0, operand0);
    SetOperand(index1, operand1);
  }
  ExprTree(HloOpcode opcode, const string& tag, const ExprTree& operand0)
      : opcode_(opcode), tag_(tag) {
    SetOperand(0, operand0);
  }
  ExprTree(HloOpcode opcode, const ExprTree& operand0, const ExprTree& operand1)
      : opcode_(opcode) {
    SetOperand(0, operand0);
    SetOperand(1, operand1);
  }

  ExprTree(const ExprTree& to_copy) {
    opcode_ = to_copy.opcode_;
    tag_ = to_copy.tag_;
    if (to_copy.fused_root_tree_ != nullptr) {
      fused_root_tree_.reset(new ExprTree(*to_copy.fused_root_tree_));
    }
    for (auto& pair : to_copy.operands_) {
      CHECK(operands_.find(pair.first) == operands_.end());
      operands_.insert(std::make_pair(
          pair.first, std::unique_ptr<ExprTree>(new ExprTree(*pair.second))));
    }
  }

  void SetFusedRoot(const ExprTree& fused_root) {
    fused_root_tree_.reset(new ExprTree(fused_root));
  }

  typedef std::unordered_map<string, const HloInstruction*>
      TaggedInstructionMap;

  // Matches 'instruction' HloOpcode against 'opcode_'.
  // Recursively matches each operand in 'operands_'.
  // Recursively matches fused instructions starting at 'fused_root_tree_'
  // if 'opcode_ == kFusion'.
  // Returns OK status, and instructions in 'tagged_instructions' for each
  // matched ExprTree node with a non-empty 'tag_'.
  // Returns error message on failure.
  Status Match(const HloInstruction* instruction,
               TaggedInstructionMap* tagged_instructions) const {
    if (opcode_ != instruction->opcode()) {
      return InvalidArgument("got opcode %s, want %s",
                             HloOpcodeString(instruction->opcode()).c_str(),
                             HloOpcodeString(opcode_).c_str());
    }

    VLOG(2) << "Matched " << HloOpcodeString(opcode_) << ": " << tag_;
    if (!tag_.empty()) {
      tagged_instructions->insert({tag_, instruction});
    }

    if (instruction->opcode() == HloOpcode::kFusion) {
      CHECK(fused_root_tree_ != nullptr);
      // Match fused instructions for this node starting a 'fused_root_tree'.
      TF_RETURN_IF_ERROR(fused_root_tree_->Match(
          instruction->fused_expression_root(), tagged_instructions));
    }

    // Match each operand in 'operands_'.
    for (auto& pair : operands_) {
      TF_RETURN_IF_ERROR(pair.second->Match(instruction->operand(pair.first),
                                            tagged_instructions));
    }
    return tensorflow::Status::OK();
  }

 private:
  void SetOperand(int64 index, const ExprTree& operand) {
    CHECK_EQ(0, operands_.count(index));
    operands_.insert(std::make_pair(index, MakeUnique<ExprTree>(operand)));
  }

  HloOpcode opcode_;
  std::unordered_map<int64, std::unique_ptr<ExprTree>> operands_;
  std::unique_ptr<ExprTree> fused_root_tree_;
  string tag_;
};

// MatcherBase is a base class that provides common functionality for
// sub-classes which match specific target sub-computations (i.e. loop
// induction variable initialization, comparison and update).
class MatcherBase {
 public:
  MatcherBase() {}
  virtual ~MatcherBase() {}

  // Attempts to match each ExprTree in 'expr_trees_'.
  // Returns OK on the first successful match, error status otherwise.
  virtual tensorflow::Status Run() {
    Status status;
    for (const ExprTree& expr_tree : expr_trees_) {
      status = MatchExprTree(expr_tree);
      if (status.ok()) {
        return status;
      }
    }
    return status;
  }

  virtual Status MatchExprTree(const ExprTree& expr_tree) = 0;

  // Returns the constant value parsed form kConstant 'instruction'.
  // Returns error status otherwise.
  Status ParseConstInteger(const HloInstruction* instruction,
                           int64* const_value) const {
    CHECK_EQ(HloOpcode::kConstant, instruction->opcode());
    PrimitiveType element_type = instruction->shape().element_type();
    if (element_type != S32 && element_type != S64) {
      return InvalidArgument("Expected constant of integral type.");
    }
    const Literal& literal = instruction->literal();
    PrimitiveType type = literal.shape().element_type();
    if (type != S32 && type != S64) {
      return InvalidArgument("Must use S32 or S64 integral types.");
    }
    if (type == S32) {
      *const_value =
          static_cast<int64>(LiteralUtil::GetFirstElement<int32>(literal));
    } else if (type == S64) {
      *const_value = LiteralUtil::GetFirstElement<int64>(literal);
    }
    return tensorflow::Status::OK();
  }

  StatusOr<const HloInstruction*> GetTaggedInstruction(
      const string& tag,
      const ExprTree::TaggedInstructionMap& tagged_instructions) {
    auto it = tagged_instructions.find(tag);
    if (it == tagged_instructions.end()) {
      return InvalidArgument("Cound not find instruction for tag: %s",
                             tag.c_str());
    }
    return it->second;
  }

 protected:
  std::vector<ExprTree> expr_trees_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatcherBase);
};

// WhileConditionComputationMatcher attempst to match a target computation
// pattern in the while condition sub-computation.
// If the target pattern is matched, two pieces of information are extracted
// from 'tagged' instructions returned by the matcher:
//
// *) 'tuple_index':
//    *) The loop induction variable tuple_index from the GetTupleElement
//       instruction of the matched computation.
//    *) Used in subsequent matching passes of while init operand and body
//       computations to select loop induction variable tuple element.
//
// *) 'loop_limit':
//    *) The integral value from Constant root operand in matched computation.
//    *) Used as the constant for the loop limit.
//
class WhileConditionComputationMatcher : public MatcherBase {
 public:
  explicit WhileConditionComputationMatcher(const HloComputation* computation)
      : computation_(computation) {
    expr_trees_.emplace_back(BuildCondExprTree());
  }

  int64 loop_limit() const { return loop_limit_; }
  int64 tuple_index() const { return tuple_index_; }

 private:
  // Builds expression tree for the following condition computation:
  //
  //     Const  Parameter
  //        \     /
  //         Fusion ------------> FusionParam FusionParam
  //                                  \          /
  //                                  GTE       /
  //                                    \      /
  //                                    LessThan (fused root)
  //
  ExprTree BuildCondExprTree() {
    // Build ExprTree for fused instructions.
    ExprTree fused_root(
        HloOpcode::kLt,
        ExprTree(HloOpcode::kGetTupleElement, "gte",
                 ExprTree(HloOpcode::kParameter, "gte.fusion_param.param0")),
        ExprTree(HloOpcode::kParameter));

    // Build top-level computation.
    ExprTree root(HloOpcode::kFusion,
                  ExprTree(HloOpcode::kConstant, "loop_limit"),
                  ExprTree(HloOpcode::kParameter, "param0"));

    root.SetFusedRoot(fused_root);
    return root;
  }

  Status MatchExprTree(const ExprTree& expr_tree) override {
    VLOG(2) << "MATCHING while condition";
    ExprTree::TaggedInstructionMap tagged_instructions;
    TF_RETURN_IF_ERROR(expr_tree.Match(computation_->root_instruction(),
                                       &tagged_instructions));

    // Get tagged GTE instruction and set 'tuple_index_'.
    TF_ASSIGN_OR_RETURN(const HloInstruction* gte,
                        GetTaggedInstruction("gte", tagged_instructions));
    tuple_index_ = gte->tuple_index();

    // Get tagged Constant instruction and parse 'loop_limit_'.
    TF_ASSIGN_OR_RETURN(
        const HloInstruction* const_hlo,
        GetTaggedInstruction("loop_limit", tagged_instructions));
    TF_RETURN_IF_ERROR(ParseConstInteger(const_hlo, &loop_limit_));

    // Get tagged "param0" instruction, and check that it matches
    // 'computation_' parameter 0.
    TF_ASSIGN_OR_RETURN(const HloInstruction* param0,
                        GetTaggedInstruction("param0", tagged_instructions));
    if (param0 != computation_->parameter_instruction(0)) {
      return InvalidArgument("Unexpected Parameter0 instruction : %s",
                             param0->name().c_str());
    }

    // Get tagged 'gte.fusion_param.param0', find its associated fusion operand,
    // and compare it to 'computation_' parameter0.
    TF_ASSIGN_OR_RETURN(
        const HloInstruction* gte_fusion_param0,
        GetTaggedInstruction("gte.fusion_param.param0", tagged_instructions));
    CHECK_EQ(HloOpcode::kParameter, gte_fusion_param0->opcode());
    CHECK(gte_fusion_param0->IsFused());
    if (gte_fusion_param0->fusion_instruction()->operand(
            gte_fusion_param0->parameter_number()) !=
        computation_->parameter_instruction(0)) {
      return InvalidArgument("Could not match fusion param: %s",
                             gte_fusion_param0->name().c_str());
    }

    return tensorflow::Status::OK();
  }

  const HloComputation* computation_;

  int64 loop_limit_ = -1;
  int64 tuple_index_ = -1;

  TF_DISALLOW_COPY_AND_ASSIGN(WhileConditionComputationMatcher);
};

// WhileInitOperandMatcher matches a target computation pattern of the
// while instructions 'init' operand, indexing the tuple at 'tuple_index'.
// On success, parses constant 'loop_start' which represents the loop induction
// variable start values, then returns OK.
// Returns error status otherwise.
class WhileInitOperandMatcher : public MatcherBase {
 public:
  WhileInitOperandMatcher(const HloInstruction* while_hlo,
                          const int64 tuple_index)
      : while_hlo_(while_hlo), tuple_index_(tuple_index) {
    expr_trees_.emplace_back(BuildInitExprTree());
  }

  int64 loop_start() const { return loop_start_; }

 private:
  // Builds expression tree for the following while init operand subcomputation:
  //
  //             Const
  //               |
  //             Copy
  //               |
  //             Tuple0
  //               |
  //             While
  //
  ExprTree BuildInitExprTree() {
    return ExprTree(
        HloOpcode::kWhile, "while",
        ExprTree(HloOpcode::kTuple, tuple_index_,
                 ExprTree(HloOpcode::kCopy,
                          ExprTree(HloOpcode::kConstant, "loop_start"))));
  }

  Status MatchExprTree(const ExprTree& expr_tree) override {
    VLOG(2) << "MATCHING while init";
    ExprTree::TaggedInstructionMap tagged_instructions;
    TF_RETURN_IF_ERROR(expr_tree.Match(while_hlo_, &tagged_instructions));

    // Get tagged while instruction check against 'while_hlo_'.
    TF_ASSIGN_OR_RETURN(const HloInstruction* while_hlo,
                        GetTaggedInstruction("while", tagged_instructions));
    if (while_hlo != while_hlo_) {
      return InvalidArgument("Expected While for instruction : %s",
                             while_hlo->name().c_str());
    }

    // Get tagged Constant instruction and parse 'loop_start_'.
    TF_ASSIGN_OR_RETURN(
        const HloInstruction* const_hlo,
        GetTaggedInstruction("loop_start", tagged_instructions));
    TF_RETURN_IF_ERROR(ParseConstInteger(const_hlo, &loop_start_));

    return tensorflow::Status::OK();
  }

  const HloInstruction* while_hlo_;
  const int64 tuple_index_;

  int64 loop_start_ = -1;

  TF_DISALLOW_COPY_AND_ASSIGN(WhileInitOperandMatcher);
};

// WhileBodyComputationMatcher matches a target computation pattern for
// the loop induction variable update. Matching proceeds from the while body
// computation root[tuple_index] to param[tuple_index], where 'tuple_index'
// If the target pattern is matched, parses a constant which represents the
// loop induction variable increment value, then returns status OK.
// Returns error status otherwise.
class WhileBodyComputationMatcher : public MatcherBase {
 public:
  WhileBodyComputationMatcher(const HloComputation* computation,
                              const int64 tuple_index)
      : computation_(computation), tuple_index_(tuple_index) {
    expr_trees_.emplace_back(BuildBodyExprTree(0, 1));
    expr_trees_.emplace_back(BuildBodyExprTree(1, 0));
  }

  int64 loop_increment() const { return loop_increment_; }

 private:
  // Builds expression tree for the following while body computation:
  //
  //
  //                               FusionParam FusionParam
  //                                     \      /
  //                  Const Param         \   GTE1
  //                     \  /              \  /
  //                    Fusion -----------> Add
  //                      |
  //                     Copy
  //                      |
  //                     Tuple0
  //
  ExprTree BuildBodyExprTree(const int64 const_index, const int64 gte_index) {
    // Build ExprTree for fused instructions.
    ExprTree gte1 =
        ExprTree(HloOpcode::kGetTupleElement, "gte",
                 ExprTree(HloOpcode::kParameter, "gte.fusion_param.param0"));
    ExprTree fused_root(HloOpcode::kAdd, const_index,
                        ExprTree(HloOpcode::kParameter), gte_index, gte1);

    // Build fusion instruction (and set fused root).
    ExprTree fusion(HloOpcode::kFusion, 0,
                    ExprTree(HloOpcode::kConstant, "loop_increment"), 1,
                    ExprTree(HloOpcode::kParameter, "param0"));
    fusion.SetFusedRoot(fused_root);

    // Build top-level computation.
    ExprTree tuple0(HloOpcode::kTuple, tuple_index_,
                    ExprTree(HloOpcode::kCopy, fusion));
    return tuple0;
  }

  Status MatchExprTree(const ExprTree& expr_tree) override {
    VLOG(2) << "MATCHING while body";
    ExprTree::TaggedInstructionMap tagged_instructions;
    TF_RETURN_IF_ERROR(expr_tree.Match(computation_->root_instruction(),
                                       &tagged_instructions));

    for (const auto& pair : tagged_instructions) {
      const auto& tag = pair.first;
      const auto& inst = pair.second;

      if (tag == "gte" && inst->tuple_index() != tuple_index_) {
        // Check that the matched GTE instruction is at the 'tuple_index' we
        // matched in the while condition computation.
        return InvalidArgument("Unexpected tuple index instruction : %s",
                               inst->name().c_str());
      } else if (tag == "loop_increment") {
        // Parse the constant which represents the loop induction variable
        // increment value.
        TF_RETURN_IF_ERROR(ParseConstInteger(inst, &loop_increment_));
      } else if (tag == "param0" &&
                 inst != computation_->parameter_instruction(0)) {
        // Check that the matched parameter == parameter 0 from 'computation_'.
        return InvalidArgument("Unexpected Parameter0 instruction : %s",
                               inst->name().c_str());
      } else if (tag == "gte.fusion_param.param0") {
        // Fusion parameter: lookup and compare with associated fusion operand.
        CHECK_EQ(HloOpcode::kParameter, inst->opcode());
        CHECK(inst->IsFused());
        if (inst->fusion_instruction()->operand(inst->parameter_number()) !=
            computation_->parameter_instruction(0)) {
          return InvalidArgument("Could not match fusion param: %s",
                                 inst->name().c_str());
        }
      }
    }
    return tensorflow::Status::OK();
  }

  const HloComputation* computation_;
  const int64 tuple_index_;

  int64 loop_increment_ = -1;

  TF_DISALLOW_COPY_AND_ASSIGN(WhileBodyComputationMatcher);
};

}  // namespace

StatusOr<std::tuple<int64, int64, int64>> CanTransformWhileToFor(
    const HloInstruction* while_hlo) {
  if (while_hlo->opcode() != HloOpcode::kWhile) {
    return InvalidArgument("Expected While instruction.");
  }

  WhileConditionComputationMatcher cond_matcher(while_hlo->while_condition());
  TF_RETURN_IF_ERROR(cond_matcher.Run());

  WhileInitOperandMatcher init_matcher(while_hlo, cond_matcher.tuple_index());
  TF_RETURN_IF_ERROR(init_matcher.Run());

  WhileBodyComputationMatcher body_matcher(while_hlo->while_body(),
                                           cond_matcher.tuple_index());
  TF_RETURN_IF_ERROR(body_matcher.Run());

  // Check for valid For loop parameters.
  if (init_matcher.loop_start() >= cond_matcher.loop_limit()) {
    return InvalidArgument("Loop start must be less than loop limit.");
  }
  if (body_matcher.loop_increment() <= 0) {
    return InvalidArgument("Loop increment must greater than zero.");
  }
  return std::make_tuple(init_matcher.loop_start(), cond_matcher.loop_limit(),
                         body_matcher.loop_increment());
}

}  // namespace gpu
}  // namespace xla

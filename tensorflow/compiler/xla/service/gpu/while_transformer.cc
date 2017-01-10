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

#include <vector>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {

namespace {

// MatcherBase is a base class that provides common functionality for
// sub-classes which match specific target sub-computations (i.e. loop
// induction variable initialization, comparison and update).
// TODO(b/33483676) Use an expression tree to specify computations to pattern
// match for while transformations.
class MatcherBase {
 public:
  enum State {
    ADD,
    PRED,
    CONST,
    COPY,
    GTE0,
    GTE1,
    PARAM,
    TUPLE0,
    TUPLE1,
    WHILE
  };

  // Initializes MatcherBase with 'computation' and initial state 'state'.
  explicit MatcherBase(const HloComputation* computation, State state)
      : computation_(computation), state_(state) {}

  // Initializes MatcherBase with 'computation', initial state 'state', and
  // value for 'tuple_index'.
  MatcherBase(const HloComputation* computation, State state,
              const int64 tuple_index)
      : computation_(computation), state_(state), tuple_index_(tuple_index) {}
  virtual ~MatcherBase() {}

  // Overridden by sub-classes to match specific target sub-computations.
  // Returns OK if target sub-computation was matched, error status otherwise.
  virtual tensorflow::Status Run() = 0;

  // Matches a Constant instruction of integral type, parses its value, and
  // stores the value in 'const_value_'.
  // Returns OK on success, error status otherwise.
  tensorflow::Status MatchConst() {
    const HloInstruction* instruction = stack_.back();
    stack_.pop_back();
    if (instruction->opcode() != HloOpcode::kConstant) {
      return InvalidArgument("Expected constant instruction.");
    }
    if (!IsSupportedIntType(instruction->shape())) {
      return InvalidArgument("Expected constant of integral type.");
    }
    const Literal& literal = instruction->literal();
    PrimitiveType type = literal.shape().element_type();
    if (type == S32) {
      const_value_ =
          static_cast<int64>(LiteralUtil::GetFirstElement<int32>(literal));
    } else if (type == S64) {
      const_value_ = LiteralUtil::GetFirstElement<int64>(literal);
    } else {
      return InvalidArgument("Must use S32 or S64 integral types.");
    }
    return tensorflow::Status::OK();
  }

  // Matches a Copy instruction.
  // Pushes its operand on the stack for subsequent processing.
  // Returns OK on success, error status otherwise.
  tensorflow::Status MatchCopy() {
    const HloInstruction* instruction = stack_.back();
    stack_.pop_back();
    if (instruction->opcode() != HloOpcode::kCopy) {
      return InvalidArgument("Expectecd Copy.");
    }
    stack_.push_back(instruction->operand(0));
    return tensorflow::Status::OK();
  }

  // Matches a GetTupleElement instruction and either parses its 'tuple_index'
  // parameter (if not initialized) or compares its 'tuple_index' with the
  // previously initialized value.
  // Pushes its operand on the stack for subsequent processing.
  // Returns OK on success, error status otherwise.
  tensorflow::Status MatchGetTupleElement() {
    const HloInstruction* instruction = stack_.back();
    stack_.pop_back();
    if (instruction->opcode() != HloOpcode::kGetTupleElement) {
      return InvalidArgument("Expected GetTupleElement instruction.");
    }
    if (!IsSupportedIntType(instruction->shape())) {
      return InvalidArgument("GetTupleElement instruction be integral type.");
    }
    if (tuple_index_ == -1) {
      tuple_index_ = instruction->tuple_index();
    } else if (tuple_index_ != instruction->tuple_index()) {
      return InvalidArgument("Invalid tuple index");
    }
    stack_.push_back(instruction->operand(0));
    return tensorflow::Status::OK();
  }

  // Matches a Parameter instruction and compares it with 'computation_'
  // parameter instruction at index 0.
  // Returns OK on success, error status otherwise.
  tensorflow::Status MatchParameter() {
    const HloInstruction* instruction = stack_.back();
    stack_.pop_back();
    if (instruction != computation_->parameter_instruction(0)) {
      return InvalidArgument("Expected Parameter instruction.");
    }
    return tensorflow::Status::OK();
  }

  // Matches a Tuple instruction.
  // Pushes operand at 'tuple_index_' on the stack for subsequent processing.
  // Returns OK on success, error status otherwise.
  tensorflow::Status MatchTuple() {
    const HloInstruction* instruction = stack_.back();
    stack_.pop_back();
    if (instruction->opcode() != HloOpcode::kTuple) {
      return InvalidArgument("Expected Tuple instruction.");
    }
    stack_.push_back(instruction->operand(tuple_index_));
    return tensorflow::Status::OK();
  }

 protected:
  const HloComputation* computation_;
  State state_;
  int64 tuple_index_ = -1;
  int64 const_value_ = -1;
  std::vector<const HloInstruction*> stack_;

 private:
  bool IsSupportedIntType(const Shape& shape) {
    return shape.element_type() == S32 || shape.element_type() == S64;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(MatcherBase);
};

// WhileConditionComputationMatcher matches one of the following two
// target While condition computations:
//
// Case 1: LessThan
//
//            PARAM
//              |
//              |
//              GTE0      CONST
//                 \        /
//                  \      /
//                    PRED
//
//
// Case 2: GreaterThan
//
//                         PARAM
//                           |
//                           |
//               CONST      GTE0
//                  \       /
//                   \     /
//                    PRED
//
// If we do not successufully match one of the two target cases, we return a
// descriptive error status.
//
// If we do successfully match one of the cases, we parse and store the
// following two pieces of information from the computation:
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
  WhileConditionComputationMatcher(const HloComputation* computation)
      : MatcherBase(computation, PRED) {
    stack_.push_back(computation_->root_instruction());
  }

  // Loop attempting to match target computation.
  tensorflow::Status Run() {
    while (!stack_.empty()) {
      switch (state_) {
        case PRED: {
          TF_RETURN_IF_ERROR(MatchPred());
          break;
        }
        case CONST: {
          TF_RETURN_IF_ERROR(MatchConst());
          state_ = GTE0;
          break;
        }
        case GTE0: {
          TF_RETURN_IF_ERROR(MatchGetTupleElement());
          state_ = PARAM;
          break;
        }
        case PARAM: {
          TF_RETURN_IF_ERROR(MatchParameter());
          break;
        }
        default:
          return InvalidArgument("Unexpected state.");
      }
    }
    return tensorflow::Status::OK();
  }

  int64 loop_limit() const { return const_value_; }
  int64 tuple_index() const { return tuple_index_; }

 private:
  tensorflow::Status MatchPred() {
    const HloInstruction* instruction = stack_.back();
    stack_.pop_back();
    // Push operands in canonical order: GetTupleElement, Constant.
    if (instruction->opcode() == HloOpcode::kLt) {
      stack_.push_back(instruction->operand(0));
      stack_.push_back(instruction->operand(1));
    } else if (instruction->opcode() == HloOpcode::kGt) {
      stack_.push_back(instruction->operand(1));
      stack_.push_back(instruction->operand(0));
    } else {
      return InvalidArgument("Condition must be LT or GT.");
    }
    state_ = CONST;
    return tensorflow::Status::OK();
  }

  TF_DISALLOW_COPY_AND_ASSIGN(WhileConditionComputationMatcher);
};

// WhileInitOperandMatcher matches one of the following two target while
// init operand sub-computations:
//
// Case 1: No copy.
//
//             CONST    // Tuple.operand(tuple_index)
//               |
//             TUPLE0   // While.operand(0)
//               |
//             WHILE
//
// Case 2: With copy.
//
//             CONST    // Tuple1.operand(tuple_index)
//               |
//             TUPLE1   // GetTupleElement.operand(0)
//               |
//             GTE0     // Copy.operand(0)
//               |
//             COPY     // Tuple0.operand(tuple_index)
//               |
//             TUPLE0   // While.operand(0)
//               |
//             While
//
class WhileInitOperandMatcher : public MatcherBase {
 public:
  WhileInitOperandMatcher(const HloInstruction* while_hlo,
                          const int64 tuple_index)
      : MatcherBase(while_hlo->parent(), WHILE, tuple_index) {
    stack_.push_back(while_hlo);
  }

  // Loop attempting to match target computation.
  tensorflow::Status Run() {
    while (!stack_.empty()) {
      switch (state_) {
        case WHILE: {
          TF_RETURN_IF_ERROR(MatchWhile());
          break;
        }
        case TUPLE0: {
          TF_RETURN_IF_ERROR(MatchTuple());
          TF_RETURN_IF_ERROR(PostMatchTuple());
          break;
        }
        case TUPLE1: {
          TF_RETURN_IF_ERROR(MatchTuple());
          state_ = CONST;
          break;
        }
        case CONST: {
          TF_RETURN_IF_ERROR(MatchConst());
          break;
        }
        case COPY: {
          TF_RETURN_IF_ERROR(MatchCopy());
          state_ = GTE0;
          break;
        }
        case GTE0: {
          TF_RETURN_IF_ERROR(MatchGetTupleElement());
          state_ = TUPLE1;
          break;
        }
        default:
          return InvalidArgument("Unexpected state.");
      }
    }
    return tensorflow::Status::OK();
  }

  int64 loop_start() const { return const_value_; }

 private:
  tensorflow::Status MatchWhile() {
    const HloInstruction* instruction = stack_.back();
    stack_.pop_back();
    if (instruction->opcode() != HloOpcode::kWhile) {
      return InvalidArgument("While init match expected while instruction.");
    }
    // Push while 'init' operand.
    stack_.push_back(instruction->operand(0));
    state_ = TUPLE0;
    return tensorflow::Status::OK();
  }

  tensorflow::Status PostMatchTuple() {
    // Transition to the next state based on matched tuple operand.
    const HloInstruction* operand = stack_.back();
    if (operand->opcode() == HloOpcode::kConstant) {
      state_ = CONST;
    } else if (operand->opcode() == HloOpcode::kCopy) {
      state_ = COPY;
    } else {
      return InvalidArgument("Expected constant or copy tuple operand.");
    }
    return tensorflow::Status::OK();
  }

  TF_DISALLOW_COPY_AND_ASSIGN(WhileInitOperandMatcher);
};

// WhileBodyComputationMatcher matches one of the following two target
// sub-computations:
//
// Case 1:
//
//                PARAM
//                  |
//          CONST  GTE1
//             \   /
//              ADD      // Tuple.operand(tuple_index).
//               |
//             TUPLE0 (root)
//
// Case 2:
//
//                PARAM
//                  |
//          CONST  GTE1
//             \   /
//              ADD      // Tuple.operand(tuple_index).
//               |
//             TUPLE1
//               |
//             GTE0
//               |
//             COPY
//               |
//             TUPLE0 (root)
//
// Note that the induction variable tuple element can have multiple users
// in the while loop body computation, but one update path.
// Matching proceeds from the root[tuple_index] to param[tuple_index].
//
class WhileBodyComputationMatcher : public MatcherBase {
 public:
  WhileBodyComputationMatcher(const HloComputation* computation,
                              const int64 tuple_index)
      : MatcherBase(computation, TUPLE0, tuple_index) {
    stack_.push_back(computation_->root_instruction());
  }

  // Loop attempting to match target computation.
  tensorflow::Status Run() {
    while (!stack_.empty()) {
      switch (state_) {
        case TUPLE0: {
          TF_RETURN_IF_ERROR(MatchTuple());
          TF_RETURN_IF_ERROR(PostMatchTuple());
          break;
        }
        case TUPLE1: {
          TF_RETURN_IF_ERROR(MatchTuple());
          state_ = ADD;
          break;
        }
        case ADD: {
          TF_RETURN_IF_ERROR(MatchAdd());
          break;
        }
        case CONST: {
          TF_RETURN_IF_ERROR(MatchConst());
          state_ = GTE1;
          break;
        }
        case COPY: {
          TF_RETURN_IF_ERROR(MatchCopy());
          state_ = GTE0;
          break;
        }
        case GTE0: {
          TF_RETURN_IF_ERROR(MatchGetTupleElement());
          state_ = TUPLE1;
          break;
        }
        case GTE1: {
          TF_RETURN_IF_ERROR(MatchGetTupleElement());
          state_ = PARAM;
          break;
        }
        case PARAM: {
          TF_RETURN_IF_ERROR(MatchParameter());
          break;
        }
        default:
          return InvalidArgument("Unexpected state.");
      }
    }
    return tensorflow::Status::OK();
  }

  int64 loop_increment() const { return const_value_; }

 private:
  tensorflow::Status MatchAdd() {
    const HloInstruction* instruction = stack_.back();
    stack_.pop_back();
    if (instruction->opcode() != HloOpcode::kAdd) {
      return InvalidArgument("Expected Add induction variable update.");
    }
    // Push in canonical order: GetTupleElement, Constant.
    if (instruction->operand(0)->opcode() == HloOpcode::kConstant &&
        instruction->operand(1)->opcode() == HloOpcode::kGetTupleElement) {
      stack_.push_back(instruction->operand(1));
      stack_.push_back(instruction->operand(0));
    } else if (instruction->operand(1)->opcode() == HloOpcode::kConstant &&
               instruction->operand(0)->opcode() ==
                   HloOpcode::kGetTupleElement) {
      stack_.push_back(instruction->operand(0));
      stack_.push_back(instruction->operand(1));
    } else {
      return InvalidArgument("Invalid types for Add operands");
    }
    state_ = CONST;
    return tensorflow::Status::OK();
  }

  tensorflow::Status PostMatchTuple() {
    // Transition to the next state based on matched tuple operand.
    const HloInstruction* operand = stack_.back();
    if (operand->opcode() == HloOpcode::kAdd) {
      state_ = ADD;
    } else if (operand->opcode() == HloOpcode::kCopy) {
      state_ = COPY;
    } else {
      return InvalidArgument("Expected add or copy tuple operand.");
    }
    return tensorflow::Status::OK();
  }

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

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

#include "tensorflow/compiler/xla/service/while_loop_analysis.h"

#include "absl/base/casts.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"

namespace xla {

using absl::nullopt;
using absl::optional;
namespace m = match;

// Finds and returns the non-constant operand in instr.
//
// CHECK-fails if instr doesn't have exactly one unique non-constant operand.
static const HloInstruction* NonConstantOperand(const HloInstruction* instr) {
  const HloInstruction* result = nullptr;
  for (const HloInstruction* operand : instr->operands()) {
    if (!operand->IsConstant()) {
      if (result != nullptr) {
        CHECK_EQ(result, operand);
      }
      result = operand;
    }
  }
  CHECK_NE(result, nullptr);
  return result;
}

// If all of instr's operands are either constants or have the form
//   get-tuple-element(gte_operand, N)
// for the same value N, returns N.  Otherwise, returns nullopt.
static optional<int64_t> GetGTEOperandIndex(const HloInstruction* instr,
                                            const HloInstruction* gte_operand) {
  VLOG(2) << "GetGTEOperandIndex(" << instr->ToString() << ", "
          << gte_operand->ToString() << ")";

  // Among the operands of `instr`, find one that is a get-tuple-element op or
  // the one that is copy fed by a get-tuple-element.
  auto gte_it =
      absl::c_find_if(instr->operands(), [](const HloInstruction* instr) {
        return (instr->opcode() == HloOpcode::kGetTupleElement) ||
               (instr->opcode() == HloOpcode::kCopy &&
                instr->operand(0)->opcode() == HloOpcode::kGetTupleElement);
      });
  if (gte_it == instr->operands().end()) {
    VLOG(2) << "instr does not have a gte operand.";
    return nullopt;
  }

  // All operands of `instr` must be either constants or of the form
  //   get-tuple-element(gte_operand, tuple_idx)
  // for the same value tuple_idx.
  int64_t tuple_idx = (*gte_it)->tuple_index();
  for (const HloInstruction* operand : instr->operands()) {
    if (!Match(operand, m::Constant()) &&
        !Match(operand,
               m::GetTupleElement(m::Op().Is(gte_operand), tuple_idx))) {
      VLOG(2)
          << "instr uses something other than a constant or gte(gte_operand, "
          << tuple_idx << "): " << operand->ToString();
      return nullopt;
    }
  }
  return tuple_idx;
}

// The below function identifies a subset of all possible auxiliary
// induction variables (AIV). Specifically, candidates are gtes, e.g.,
// gte(param0, N)
// The function checks if the loop body plumbs the AIV
// through the same tuple index at root, and that ops involving AIV
// involve constants.
//   op2 = op(constants, gte(param0, N), constants)
//   op3 = op(constants, f(op2, gte(param0, N), constants)
//   op4 = op(constants, f(op3, constants)
//   root = tuple(..., op4, ...)
// Further, the ops are restricted to basic math ops (+,-,*,/).
// Finally, loop invariant GTEs are excluded from AIVs.
// We can expand the ops category/nature of AIVs as needed.
std::vector<const HloInstruction*> GetAuxiliaryLoopInductionVars(
    const HloInstruction* while_op) {
  std::vector<const HloInstruction*> aux_ind_gte;
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_body = while_op->while_body();
  auto* while_body_param = while_body->parameter_instruction(0);
  VLOG(2) << "Aux Induction Variables for loop:" << while_op->ToShortString();
  VLOG(2) << "the parameter instr:" << while_body_param->ToShortString();
  VLOG(2) << "the parameter user count:" << while_body_param->users().size();
  if (while_body_param == nullptr) return aux_ind_gte;

  // candidates_pairs = pair<inst, inst>(
  //   operands of the root while body,
  //   GTE only operands that index into the same position in the parameter)
  // for each candidate_pair (x, y)
  //  find all paths between x and y,
  //  each paths should satisfy the above listed criterion
  //  index that x and y used is added as a aux variable index
  std::map<int64_t, const HloInstruction*> extractions;
  for (const HloInstruction* indx_instr : while_body_param->users()) {
    if (indx_instr->opcode() != HloOpcode::kGetTupleElement) {
      continue;
    }
    auto it = extractions.find(indx_instr->tuple_index());
    // if we find two extractions at the same index, we ignore such
    // a candidate
    if (it != extractions.end()) {
      it->second = nullptr;
      VLOG(2) << "two extractions at same index:" << indx_instr->ToString();
    } else {
      extractions.insert(std::make_pair(indx_instr->tuple_index(), indx_instr));
      VLOG(2) << "inserting extraction :" << indx_instr->ToString();
    }
  }
  VLOG(2) << "total extractions size:" << extractions.size() << std::endl;
  if (extractions.empty()) {
    return aux_ind_gte;
  }

  auto* while_body_root = while_body->root_instruction();
  if (while_body_root->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "While body root is not a tuple:" << while_body_root->ToString();
    return aux_ind_gte;
  }
  int64_t index = -1;
  std::map<int64_t, const HloInstruction*> insertions;
  for (const HloInstruction* operand : while_body_root->operands()) {
    index++;
    if (!operand->IsConstant()) {
      auto it = insertions.find(index);
      if (it != insertions.end()) {
        it->second = nullptr;
        VLOG(2) << "two insertions at same index:" << operand->ToString();
      } else {
        insertions.insert(std::make_pair(index, operand));
        VLOG(2) << "inserting insertions:" << operand->ToString();
      }
    }
  }
  if (insertions.empty()) {
    return aux_ind_gte;
  }

  std::map<int64_t, std::pair<const HloInstruction*, const HloInstruction*>>
      candidate_pairs;
  for (; index >= 0; --index) {
    const HloInstruction *ext, *inst;
    ext = (extractions.find(index) != extractions.end())
              ? extractions.find(index)->second
              : nullptr;
    inst = (insertions.find(index) != insertions.end())
               ? insertions.find(index)->second
               : nullptr;
    if (ext != nullptr && inst != nullptr) {
      // Filter out trivial aux, i.e., extract directly to an insert.
      if (ext != inst) {
        candidate_pairs.insert(
            std::make_pair(index, std::make_pair(ext, inst)));
      }
    }
  }
  VLOG(2) << "total candidate pairs:" << candidate_pairs.size() << std::endl;

  // Passed to ReachabilityMap to decide the type of produce-consumer edges
  // along the reachability path.
  const auto add_dependencies = [](const HloInstruction* hlo,
                                   std::vector<HloInstruction*>* inputs) {
    HloInstruction* non_const_operand = nullptr;
    int num_non_constants = 0;
    for (HloInstruction* operand : hlo->operands()) {
      if (!operand->IsConstant()) {
        num_non_constants++;
        non_const_operand = operand;
      }
    }
    if (num_non_constants == 1 &&
        (hlo->opcode() == HloOpcode::kGetTupleElement ||
         hlo->opcode() == HloOpcode::kAdd ||
         hlo->opcode() == HloOpcode::kMultiply ||
         hlo->opcode() == HloOpcode::kDivide ||
         hlo->opcode() == HloOpcode::kSubtract)) {
      inputs->push_back(non_const_operand);
    }
  };

  std::unique_ptr<HloReachabilityMap> hrm =
      HloReachabilityMap::BuildWithRestrictions(
          while_body,
          absl::FunctionRef<void(const HloInstruction* hlo,
                                 std::vector<HloInstruction*>* inputs)>(
              add_dependencies));

  for (auto candidates : candidate_pairs) {
    VLOG(2) << "are reachable?:" << (candidates.second.first)->ToString()
            << "*************" << (candidates.second.second)->ToString()
            << std::endl;
    if (hrm->IsReachable(candidates.second.first, candidates.second.second)) {
      aux_ind_gte.push_back(candidates.second.first);
      VLOG(2) << "YES";
    } else {
      VLOG(2) << "NO";
    }
  }
  VLOG(2) << "num auxiliary candidates :" << aux_ind_gte.size();
  return aux_ind_gte;
}

// Tries to get the tuple index of the induction variable of a while loop.
//
// Checks that the loop condition and body both plumb the induction variable
// through the same tuple index, and that they both apply exactly one op to the
// induction variable before  deciding whether to do another loop iteration (in
// the loop condition's case) or packing the induction variable into the result
// tuple (in the loop body's case).
//
// Specifically, checks that the loop condition has structure
//
//   root = op(constants, get-tuple-elem(param0, N), constants)
//
// and the loop body has the structure
//
//   inc = op(constants, get-tuple-elem(param0, N), constants)
//   root = tuple(..., inc, ...)  // inc is N'th operand of tuple().
//
// If so, returns N.  Otherwise, returns nullopt.
optional<int64_t> GetLoopInductionVarTupleIdx(const HloInstruction* while_op) {
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);
  VLOG(2) << "Finding induction variable for loop "
          << while_op->ToShortString();

  // The while_cond computation should have the form
  //
  //   while_cond_root =
  //       op(constants, get-tuple-elem(while_cond_param, N), constants).
  //
  // If it does, set indvar_tuple_idx to N.
  auto* while_cond = while_op->while_condition();
  auto* while_cond_root = while_cond->root_instruction();
  auto* while_cond_param = while_cond->parameter_instruction(0);
  optional<int64_t> indvar_tuple_idx =
      GetGTEOperandIndex(while_cond_root, while_cond_param);
  if (!indvar_tuple_idx) {
    VLOG(2) << "Induction variable not found in loop condition: "
            << while_cond->root_instruction()->ToString();
    return nullopt;
  }

  // The while_body computation should have the form
  //
  //   while_body_inc =
  //       op(constants, get-tuple-elem(while_body_param, N), constants)
  //   while_body_root = tuple(..., while_body_inc, ...)
  //
  // where while_body_inc is operand N of while_body_root.
  auto* while_body = while_op->while_body();
  auto* while_body_root = while_body->root_instruction();
  if (while_body_root->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "While body's root is not a tuple instruction: "
            << while_body_root->ToString();
    return nullopt;
  }

  auto* while_body_inc = while_body_root->operand(*indvar_tuple_idx);
  auto* while_body_param = while_body->parameter_instruction(0);
  optional<int64_t> while_body_indvar_tuple_idx =
      GetGTEOperandIndex(while_body_inc, while_body_param);
  if (!while_body_indvar_tuple_idx) {
    VLOG(2)
        << "Induction variable not found in while body increment instruction: "
        << while_body_inc->ToString();
    return nullopt;
  }
  if (while_body_indvar_tuple_idx != indvar_tuple_idx) {
    VLOG(2) << "Tuple index of induction variable does not match between loop "
               "condition ("
            << *indvar_tuple_idx << ") and while body ("
            << *while_body_indvar_tuple_idx << ")";
    return nullopt;
  }

  // Finally, check that the while loop's initial value is a tuple with enough
  // elements.
  auto* while_init = while_op->operand(0);
  if (while_init->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "While init expected to be a tuple: " << while_init->ToString();
    return nullopt;
  }

  VLOG(2) << "Induction variable's tuple index: " << *indvar_tuple_idx;
  return indvar_tuple_idx;
}

// Converts the given literal to a scalar int64_t, if possible.
//
// Fails if the literal is not an integral type or if the value it contains
// cannot be represented in an int64_t.
static optional<int64_t> LiteralAsScalarInt64(const Literal& l) {
  if (!ShapeUtil::IsEffectiveScalar(l.shape())) {
    VLOG(2) << "literal is not an effective scalar: " << l.ToString();
    return nullopt;
  }
  switch (l.shape().element_type()) {
    case S8:
      return l.GetFirstElement<int8_t>();
    case S16:
      return l.GetFirstElement<int16_t>();
    case S32:
      return l.GetFirstElement<int32_t>();
    case S64:
      return l.GetFirstElement<int64_t>();
    case U8:
      return l.GetFirstElement<uint8_t>();
    case U16:
      return l.GetFirstElement<uint16_t>();
    case U32:
      return l.GetFirstElement<uint32_t>();
    case U64: {
      uint64_t v = l.GetFirstElement<uint64_t>();
      if (v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        VLOG(2) << "uint64_t literal is out of range for int64_t: " << v;
        return nullopt;
      }
      return v;
    }
    default:
      VLOG(2) << "literal is of non-integral type " << l.shape().ToString();
      return nullopt;
  }
}

// Computes a + b, returning nullopt if it overflows.
optional<int64_t> CheckedAdd(int64_t a, int64_t b) {
  // Overflow occurred iff `a` and `b` have the same sign and `a + b` has a
  // different sign, see Hacker's Delignt 2nd Ed. pp 28.
  uint64_t aa = absl::bit_cast<uint64_t>(a);
  uint64_t bb = absl::bit_cast<uint64_t>(b);
  int64_t result = absl::bit_cast<int64_t>(aa + bb);
  if (a >= 0 == b >= 0 && result >= 0 != a >= 0) {
    return nullopt;
  }
  return result;
}

// Computes a - b, returning nullopt if it overflows.
optional<int64_t> CheckedSubtract(int64_t a, int64_t b) {
  uint64_t aa = absl::bit_cast<uint64_t>(a);
  uint64_t bb = absl::bit_cast<uint64_t>(b);
  int64_t result = absl::bit_cast<int64_t>(aa - bb);
  // Overflow occurred iff `a` and `b` have different signs and the sign of
  // `a - b` is the same as that of `b`, see Hacker's Delight 2nd Ed. pp 29.
  if (a >= 0 != b >= 0 && result >= 0 == b >= 0) {
    return nullopt;
  }
  return result;
}

// Check if
//  - `i` is initialized to a scalar constant K (namely, `indvar_init`),
//  - the while condition does `i < N` or `i <= N`, and
//  - the while body does `i++`.
// If so, it's trivial to compute the loop bound.
static optional<int64_t> PatternMatchLoopTripCount(HloInstruction* while_op,
                                                   int64_t indvar_tuple_idx,
                                                   const Literal& indvar_init) {
  // First, find the scalar constant K that `i` is initialized to.
  optional<int64_t> indvar_init_val = LiteralAsScalarInt64(indvar_init);
  if (!indvar_init_val) {
    VLOG(2) << "Pattern-match failed: induction variable init is not a "
               "constant scalar representable as an int64_t: "
            << indvar_init.ToString();
    return nullopt;
  }

  // Check that `i` goes as `i++` in the while body.
  //
  // TODO(jlebar): We could also handle i-- and other idioms.
  auto* while_body = while_op->while_body();
  auto* while_body_indvar_update =
      while_body->root_instruction()->operand(indvar_tuple_idx);
  auto* while_body_indvar = NonConstantOperand(while_body_indvar_update);
  if (!Match(while_body_indvar_update,
             m::AddAnyOrder(m::Op().Is(while_body_indvar),
                            m::ConstantEffectiveScalar(1)))) {
    VLOG(2) << "Pattern-match failed: induction variable does not go as i++: "
            << while_body_indvar_update->ToString();
    return nullopt;
  }

  // Check that we do op(i, N) or op(N, i) as the while condition.  Capture the
  // value N.
  auto* while_cond = while_op->while_condition();
  auto* while_cond_root = while_cond->root_instruction();
  auto* while_cond_indvar = NonConstantOperand(while_cond_root);
  HloInstruction* while_cond_bound = nullptr;
  if (!Match(while_cond_root,
             m::Op().WithBinaryOperandsAnyOrder(
                 m::Op().Is(while_cond_indvar),
                 m::ConstantEffectiveScalar(&while_cond_bound)))) {
    VLOG(2) << "Pattern-match failed: while condition is not of the form "
               "op(i, N) or op(N, i).";
    return nullopt;
  }
  // Note: If this succeeds, the constant `N` is representable as an int64_t --
  // that is, if it's an XLA U64, it fits within an int64_t.
  optional<int64_t> while_cond_bound_val =
      LiteralAsScalarInt64(while_cond_bound->literal());
  if (!while_cond_bound_val) {
    VLOG(2) << "Pattern-match failed: while condition induction variable is "
               "not a constant scalar representable as an int64_t.";
    return nullopt;
  }

  // Handle `i = K; i < N; ++i`.
  if (Match(while_cond_root,
            m::Op()
                .WithComparisonDirection(ComparisonDirection::kLt)
                .WithOperand(0, m::Op().Is(while_cond_indvar)))) {
    VLOG(2) << "Pattern-match succeeded: loop condition is i < N: "
            << while_cond_root->ToString();
    optional<int64_t> trips =
        CheckedSubtract(*while_cond_bound_val, *indvar_init_val);
    if (trips) {
      return std::max(int64_t{0}, *trips);
    } else {
      VLOG(2) << "Pattern-match failed: Trip count exceeds INT64_MAX.";
      return nullopt;
    }
  }

  // Handle `i = K; i <= N; ++i`.
  if (Match(while_cond_root,
            m::Op()
                .WithComparisonDirection(ComparisonDirection::kLe)
                .WithOperand(0, m::Op().Is(while_cond_indvar)))) {
    VLOG(2) << "Pattern-match succeeded: loop condition is i <= N: "
            << while_cond_root->ToString();
    optional<int64_t> trips =
        CheckedSubtract(*while_cond_bound_val, *indvar_init_val);
    if (!trips) {
      VLOG(2) << "Pattern-match failed: Trip count exceeds INT64_MAX";
      return nullopt;
    }
    trips = CheckedAdd(*trips, 1);
    if (!trips) {
      VLOG(2) << "Pattern-match failed: Trip count exceeds INT64_MAX";
      return nullopt;
    }
    return std::max<int64_t>(0, *trips);
  }

  VLOG(2) << "Pattern-match failed: while condition follows unknown pattern: "
          << while_cond_root->ToString();
  return nullopt;
}

optional<int64_t> ComputeWhileLoopTripCount(HloInstruction* while_op,
                                            int64_t max_brute_force_iters) {
  VLOG(2) << "Getting trip count for loop " << while_op->ToString();

  // The loop's induction variable is found at
  //
  //   get-tuple-elem(comp->parameter_instruction(0), *indvar_tuple_idx),
  //
  // where comp is while_op->while_body() or while_op->while_condition().
  optional<int64_t> indvar_tuple_idx = GetLoopInductionVarTupleIdx(while_op);
  if (!indvar_tuple_idx) {
    return nullopt;
  }

  // Now that we know the index of the induction variable, we can we can try to
  // compute how many times the loop executes.  Start by computing the induction
  // variable's initial value.
  HloEvaluator evaluator(/*max_loop_iterations=*/0);
  auto* while_init = while_op->mutable_operand(0);
  auto* indvar_init = while_init->mutable_operand(*indvar_tuple_idx);
  StatusOr<Literal> indvar_init_result = evaluator.Evaluate(indvar_init);
  if (!indvar_init_result.ok()) {
    VLOG(2) << "Couldn't evaluate induction variable init, "
            << indvar_init_result.status() << ", " << indvar_init->ToString();
    return nullopt;
  }
  Literal indvar_iter_val = std::move(indvar_init_result).ValueOrDie();

  // First, try to pattern-match.
  if (auto trip_count = PatternMatchLoopTripCount(while_op, *indvar_tuple_idx,
                                                  indvar_iter_val)) {
    return trip_count;
  }

  // If our pattern-match failed, try brute-forcing the loop trip count.
  auto* while_body = while_op->while_body();
  auto* while_body_indvar_update =
      while_body->root_instruction()->operand(*indvar_tuple_idx);
  auto* while_body_indvar = NonConstantOperand(while_body_indvar_update);

  auto* while_cond = while_op->while_condition();
  auto* while_cond_root = while_cond->root_instruction();
  auto* while_cond_indvar = NonConstantOperand(while_cond_root);

  for (int64_t trip_count = 0; trip_count != max_brute_force_iters + 1;
       ++trip_count) {
    StatusOr<Literal> result = evaluator.EvaluateWithSubstitutions(
        while_cond_root, {{while_cond_indvar, &indvar_iter_val}});
    if (!result.ok()) {
      VLOG(2) << "Couldn't evaluate while cond: " << result.status();
      return nullopt;
    }
    if (result.ValueOrDie().data<bool>() == absl::Span<const bool>{false}) {
      VLOG(2) << "Loop has static trip count of " << trip_count;
      return trip_count;
    }

    // Calculate the value of the induction variable after one iteration of the
    // loop, and check whether the while condition is true with this new value.
    StatusOr<Literal> indvar_next_result = evaluator.EvaluateWithSubstitutions(
        while_body_indvar_update, {{while_body_indvar, &indvar_iter_val}});
    if (!indvar_next_result.ok()) {
      VLOG(2) << "Couldn't evaluate induction variable update: "
              << indvar_next_result.status();
      return nullopt;
    }
    indvar_iter_val = std::move(indvar_next_result).ValueOrDie();
  }

  VLOG(2) << "Loop has unknown trip count.";
  return nullopt;
}

// If the only user of this instruction is a get-tuple-element, return that
// get-tuple-element, otherwise return null. If this runs before CSE/DCE, we may
// get a false negative if there are several copies of the same GTE, or there
// are unused GTEs, but we can live with this.
static HloInstruction* GetOnlyGTE(HloInstruction* inst) {
  if (inst->user_count() != 1) {
    return nullptr;
  }

  HloInstruction* user = inst->users().back();
  if (user->opcode() != HloOpcode::kGetTupleElement) {
    return nullptr;
  }
  return user;
}

optional<int64_t> ComputeWhileLoopTripCountUpperBound(
    HloInstruction* while_op) {
  // If we know the exact trip count, it's also the upper bound.
  auto exact_trip_count = ComputeWhileLoopTripCount(while_op);
  if (exact_trip_count) {
    VLOG(2) << "Loop has exact trip count.";
    return exact_trip_count;
  }

  // There is one more case we know how to handle. If the loop condition only
  // looks at one element of the tuple, and the loop body sets this element to a
  // constant, there are two options:
  // 1) Evaluating the condition on this constant returns true. In this case,
  // the loop either executes 0 times, or is an infinite loop, depending on the
  // init value.
  // 2) Evaluating the condition on this constant returns false. In this case,
  // the loop executes 0 or 1 times, depending on the init value. This means
  // that, regardless of the init value, the upper bound on the trip count is 1.

  // Check whether the condition depends on a single parameter, and find out
  // which.
  auto* while_cond = while_op->while_condition();
  auto* while_cond_param = while_cond->parameter_instruction(0);
  auto* cond_gte = GetOnlyGTE(while_cond_param);
  if (!cond_gte) {
    VLOG(2) << "Induction variable not found in loop condition: "
            << while_cond->root_instruction()->ToString();
    return nullopt;
  }

  // Now check whether this gets set to a constant by the while body.
  auto* while_body = while_op->while_body();
  auto* while_body_root = while_body->root_instruction();
  if (while_body_root->opcode() != HloOpcode::kTuple) {
    VLOG(3) << "While body's root is not a tuple instruction: "
            << while_body_root->ToString();
    return nullopt;
  }

  int64_t indvar_index = cond_gte->tuple_index();
  auto* while_body_indvar = while_body_root->operand(indvar_index);
  if (while_body_indvar->opcode() != HloOpcode::kConstant) {
    VLOG(3) << "While body does not set the IV to a constant: "
            << while_body_indvar->ToString();
    return nullopt;
  }

  // We have a constant. Evaluate the condition on this constant.
  HloEvaluator evaluator(/*max_loop_iterations=*/0);
  Literal fake_input = Literal::CreateFromShape(while_cond_param->shape());
  TF_CHECK_OK(fake_input.CopyFrom(while_body_indvar->literal(),
                                  /*dest_shape_index=*/{indvar_index},
                                  /*src_shape_index=*/{}));
  StatusOr<Literal> eval_result =
      evaluator.Evaluate(*while_cond, {std::move(fake_input)});

  if (!eval_result.ok()) {
    VLOG(2) << "Couldn't evaluate while loop condition.";
    return nullopt;
  }

  Literal cond_result_pred = std::move(eval_result.ValueOrDie());
  CHECK(Shape::Equal().IgnoreLayout()(cond_result_pred.shape(),
                                      ShapeUtil::MakeShape(PRED, {})));

  // Per the explanation above, if the evaluated condition returns false, the
  // loop executes at most once.
  bool cond_returns_true = cond_result_pred.GetFirstElement<bool>();
  if (!cond_returns_true) {
    VLOG(2) << "Upper bound on the trip count is 1";
    return 1;
  }

  VLOG(2) << "Loop has no known upper bound on the trip count.";
  return nullopt;
}

}  // namespace xla

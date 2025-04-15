/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/analysis/while_loop_analysis.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/constant_value.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/value_range.h"
#include "xla/shape_util.h"
#include "xla/tools/hlo_extractor.h"
#include "xla/tsl/platform/status.h"
#include "xla/xla_data.pb.h"

namespace xla {

using std::nullopt;
using std::optional;
namespace m = match;

namespace {

// Traces through a chain of copy instructions and a GTE-tuple pair until a
// non-copy or a non-GTE-tuple pair is found.
const HloInstruction* TraceThroughCopyAndGteTupleChain(
    const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kGetTupleElement &&
      instr->operand(0)->opcode() == HloOpcode::kTuple) {
    return TraceThroughCopyAndGteTupleChain(
        instr->operand(0)->operand(instr->tuple_index()));
  }
  if (instr->opcode() == HloOpcode::kCopy ||
      instr->opcode() == HloOpcode::kCopyStart ||
      instr->opcode() == HloOpcode::kCopyDone) {
    return TraceThroughCopyAndGteTupleChain(instr->operand(0));
  }

  return instr;
}
}  // namespace

// Finds and returns the non-constant operand in instr, if there is only one
// such operand.
//
// Returns nullptr if instr doesn't have exactly one unique non-constant
// operand.
static const HloInstruction* NonConstantOperand(const HloInstruction* instr) {
  const HloInstruction* result = nullptr;
  for (const HloInstruction* operand : instr->operands()) {
    if (!operand->IsConstant()) {
      if (result != nullptr && result != operand) {
        return nullptr;
      }
      result = operand;
    }
  }
  return result;
}

// If all of instr's operands are either constants or have the form
//   get-tuple-element(gte_operand, N)
// for the same value N, returns N.  Otherwise, returns nullopt.
static optional<int64_t> GetGTEOperandIndex(const HloInstruction* instr,
                                            const HloInstruction* gte_operand) {
  VLOG(2) << "GetGTEOperandIndex(" << instr->ToString()
          << ", GTE Operand: " << gte_operand->ToString() << ")";

  // All operands of `instr` must be either constants or of the form
  //   get-tuple-element(gte_operand, tuple_idx)
  // for the same value tuple_idx. We also support the case where GTE feeds a
  // copy that is then used.
  optional<int64_t> tuple_idx;
  for (const HloInstruction* operand : instr->operands()) {
    if (operand->opcode() == HloOpcode::kConstant) {
      continue;
    }
    auto possibly_gte = operand;

    if (operand->opcode() == HloOpcode::kCopy) {
      possibly_gte = operand->operand(0);
    }

    if (possibly_gte->opcode() != HloOpcode::kGetTupleElement) {
      return nullopt;
    }

    if (possibly_gte->operand(0) != gte_operand) {
      return nullopt;
    }

    int64_t operand_tuple_idx = possibly_gte->tuple_index();
    // This is the first GTE we are seeing. Set tuple_idx.
    if (!tuple_idx.has_value()) {
      tuple_idx = operand_tuple_idx;
    } else {
      if (operand_tuple_idx != tuple_idx) {
        return nullopt;
      }
    }
  }
  return tuple_idx;
}

// This function returns true if the operation is a simple scalar operation.
// While loop analysis can execute such an operation at compile time without
// incurring huge overheads.
static bool IsScalarOp(const HloInstruction* op) {
  if (IsCollective(op)) return false;
  switch (op->opcode()) {
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kCustomCall:
      return false;
    default:
      break;
  }
  for (const HloComputation* computation : op->called_computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (!IsScalarOp(instruction)) return false;
    }
  }
  return ShapeUtil::IsScalar(op->shape());
}

// If `out` is a function of a single value in the tuple `in` and has no other
// dependence, i.e. if `out=f(gte(in))`, then this function will return the
// unique get-tuple-element index for the dependence.
//
// For example, in the following HLO, this function will return `1`:
//   in = (s32[], s32[], s32[]) tuple(a,b,c)
//   gte.1 = get-tuple-element(in), index=1
//   out = fusion(gte.1), ...
// Also checks whether all ops on the path from `in` to `out` are ops with a
// scalar shape.
static std::optional<int64_t> GetUniqueGTEDependenceIndex(
    const HloInstruction* out, const HloInstruction* in) {
  // Fast path : pattern matching.
  std::optional<int64_t> tuple_idx = GetGTEOperandIndex(out, in);
  if (tuple_idx != std::nullopt) {
    return tuple_idx;
  }

  if (out->parent() != in->parent() || !in->shape().IsTuple()) {
    return std::nullopt;
  }

  // Extracts the instruction `out` as a function of the instruction `in`.
  // HloModule extracted
  // ENTRY main {
  //   in = parameter(0)
  //   //... some calculations
  //   ROOT out = ...
  // }
  std::unique_ptr<HloModule> extracted = ExtractModule(
      /*instruction=*/out, /*height=*/-1, /*extract_selector=*/
      [in](const HloInstruction* inst) -> bool { return inst != in; },
      /*replace_type_selector=*/
      [](const HloInstruction* inst) -> ReplaceType {
        return ReplaceType::kReplaceParam;
      },
      /*cross_computation=*/false, /*inline_calls_and_fusions=*/true,
      /*run_verifier=*/false);
  HloComputation* entry = extracted->entry_computation();

  // Check that the extracted module takes nothing but `in` as input. If `out`
  // does not depend on in, the extracted module will have some other shape for
  // input.
  if (entry->num_parameters() != 1 ||
      entry->parameter_instruction(0)->shape() != in->shape()) {
    return std::nullopt;
  }
  HloInstruction* param = entry->parameter_instruction(0);

  // If there are no users for the input `in`, it would mean that `out` does not
  // depend on a get-tuple-element of `in`.
  if (param->user_count() == 0) {
    return nullopt;
  }

  // If any of the users of the input `in` is not a get-tuple-element
  // instruction, then that would mean that the output does not depend uniquely
  // on a get-tuple-element of on `in`, instead it depends on some other
  // calculations on `in`.
  if (absl::c_any_of(param->users(), [](const HloInstruction* inst) -> bool {
        return inst->opcode() != HloOpcode::kGetTupleElement;
      })) {
    return std::nullopt;
  }

  // We extract the candidate index from the first user. At this point we
  // already know that the all the users are get-tuple-elements and that there
  // is atleast one user.
  int64_t candidate_index = param->users()[0]->tuple_index();

  // We check that all the users of the input instruction `in` (which we already
  // know to be get-tuple-element instructions) have the same tuple index.
  if (absl::c_any_of(param->users(),
                     [candidate_index](const HloInstruction* inst) -> bool {
                       return inst->tuple_index() != candidate_index;
                     })) {
    return std::nullopt;
  }

  if (absl::c_any_of(
          entry->instructions(), [](const HloInstruction* inst) -> bool {
            return inst->opcode() != HloOpcode::kParameter && !IsScalarOp(inst);
          })) {
    return std::nullopt;
  }

  return candidate_index;
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
      GetUniqueGTEDependenceIndex(while_cond_root, while_cond_param);
  if (!indvar_tuple_idx) {
    VLOG(2) << "Induction variable not found in loop condition: "
            << while_cond->root_instruction()->ToString();
    return nullopt;
  }

  // The while_body computation should have the form:
  //
  // Form 1:
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
  const HloInstruction* while_body_inc;
  while_body_inc = TraceThroughCopyAndGteTupleChain(
      while_body_root->operand(*indvar_tuple_idx));
  auto* while_body_param = while_body->parameter_instruction(0);
  optional<int64_t> while_body_indvar_tuple_idx =
      GetUniqueGTEDependenceIndex(while_body_inc, while_body_param);
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

optional<Range> MatchTrivialLoopRange(const HloInstruction* while_op) {
  std::optional<int64_t> indvar_tuple_idx =
      GetLoopInductionVarTupleIdx(while_op);
  if (!indvar_tuple_idx.has_value()) {
    return nullopt;
  }
  if (!while_op->operand(0)->operand(*indvar_tuple_idx)->IsConstant()) {
    return nullopt;
  }
  const Literal& indvar_init =
      while_op->operand(0)->operand(*indvar_tuple_idx)->literal();

  // First, find the scalar constant init that `i` is initialized to.
  optional<int64_t> indvar_init_val =
      LiteralUtil::LiteralAsScalarInt64(indvar_init);
  if (!indvar_init_val) {
    VLOG(2) << "Pattern-match failed: induction variable init is not a "
               "constant scalar representable as an int64_t: "
            << indvar_init.ToString();
    return nullopt;
  }

  // Check that `i` goes as `i += C` in the while body where C is a natural
  // number.
  auto* while_body = while_op->while_body();
  auto* while_body_indvar_update =
      while_body->root_instruction()->mutable_operand(indvar_tuple_idx.value());
  auto* while_body_indvar = NonConstantOperand(while_body_indvar_update);
  if (while_body_indvar == nullptr ||
      while_body_indvar !=
          hlo_query::GetUniqueGteInstruction(
              while_body->parameter_instruction(0), indvar_tuple_idx.value())) {
    VLOG(2) << "Pattern-match failed: update of induction variable is not in "
               "the form of op(gte, constant): "
            << while_body_indvar_update->ToString();
    return std::nullopt;
  }
  HloInstruction* trip_count_increase_step_instr = nullptr;
  int64_t trip_count_step = 0;
  if (!Match(while_body_indvar_update,
             m::AddAnyOrder(m::Op().Is(while_body_indvar),
                            m::Op(&trip_count_increase_step_instr)))) {
    if (trip_count_increase_step_instr == nullptr) {
      VLOG(2) << "Pattern-match failed: induction variable is not getting "
                 "updated by an add operation: "
              << while_body_indvar_update->ToString();
      return nullopt;
    }
    if (!trip_count_increase_step_instr->IsConstant() ||
        !ShapeUtil::IsEffectiveScalar(
            trip_count_increase_step_instr->shape())) {
      VLOG(2) << "Pattern-match failed: induction variable is not getting "
                 "incremented by constant: "
              << while_body_indvar_update->ToString();
      return nullopt;
    }
    if (!LiteralUtil::LiteralAsScalarInt64(
             trip_count_increase_step_instr->literal())
             .has_value()) {
      VLOG(2)
          << "Pattern-match failed: trip count step is not an integral type: "
          << trip_count_increase_step_instr->shape().ToString();
      return nullopt;
    }
    VLOG(2) << "Pattern-match for trip count step failed: "
            << trip_count_increase_step_instr->ToString();
  }

  trip_count_step = LiteralUtil::LiteralAsScalarInt64(
                        trip_count_increase_step_instr->literal())
                        .value();
  if (trip_count_step <= 0) {
    VLOG(2) << "Pattern-match failed: trip count step is not a natural number: "
            << trip_count_step;
    return nullopt;
  }
  // Check that we do op(i, N) or op(N, i) as the while condition.  Capture the
  // value N.
  auto* while_cond = while_op->while_condition();
  auto* while_cond_root = while_cond->root_instruction();
  auto* while_cond_indvar = NonConstantOperand(while_cond_root);
  if (while_cond_indvar == nullptr ||
      while_cond_indvar !=
          hlo_query::GetUniqueGteInstruction(
              while_cond->parameter_instruction(0), indvar_tuple_idx.value())) {
    VLOG(2) << "Pattern-match failed: while condition is not supported.";
    return std::nullopt;
  }
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
      LiteralUtil::LiteralAsScalarInt64(while_cond_bound->literal());
  if (!while_cond_bound_val) {
    VLOG(2) << "Pattern-match failed: while condition induction variable is "
               "not a constant scalar representable as an int64_t.";
    return nullopt;
  }

  // If the while loop condition does not support equality, then we need to
  // deduct one from the bound.
  bool while_cond_bound_supports_equality;
  if (Match(while_cond_root,
            m::Op().WithComparisonDirection(ComparisonDirection::kLt)) ||
      Match(while_cond_root,
            m::Op().WithComparisonDirection(ComparisonDirection::kGt))) {
    while_cond_bound_supports_equality = false;
  } else if (Match(while_cond_root,
                   m::Op().WithComparisonDirection(ComparisonDirection::kLe)) ||
             Match(while_cond_root,
                   m::Op().WithComparisonDirection(ComparisonDirection::kGe))) {
    while_cond_bound_supports_equality = true;
  } else {
    VLOG(2) << "Pattern-match failed: while condition comparison is not "
               "LT, GT, LE, or GE.";
    return nullopt;
  }
  if (!while_cond_bound_supports_equality) {
    while_cond_bound_val.value()--;
  }

  // We also need to round the bound down so that the difference between bound
  // and init_value is a multiple of the step size.
  // We want to round down the expression
  // (while_cond_bound_val.value() - indvar_init_val.value()) to a multiple of
  // trip_count_step by adjusting the bound value. We need to be careful not to
  // run into overflows.
  int64_t bound_value_remainder =
      while_cond_bound_val.value() % trip_count_step;
  int64_t init_value_remainder = indvar_init_val.value() % trip_count_step;
  int64_t remainder =
      (bound_value_remainder - init_value_remainder) % trip_count_step;
  if (remainder < 0) {
    remainder += trip_count_step;
  }
  while_cond_bound_val.value() -= remainder;

  const int64_t init_bitwidth =
      primitive_util::BitWidth(indvar_init.shape().element_type());
  const bool init_is_signed =
      primitive_util::IsSignedIntegralType(indvar_init.shape().element_type());

  const int64_t bound_bitwidth =
      primitive_util::BitWidth(while_cond_bound->shape().element_type());
  const bool bound_is_signed = primitive_util::IsSignedIntegralType(
      while_cond_bound->shape().element_type());

  return Range{ConstantValue::Get(indvar_init_val.value(), init_bitwidth,
                                  /*is_signed=*/init_is_signed),
               ConstantValue::Get(while_cond_bound_val.value(), bound_bitwidth,
                                  /*is_signed=*/bound_is_signed),
               ConstantValue::Get(trip_count_step, /*bitwidth=*/64,
                                  /*is_signed=*/true),
               /*is_linear=*/true};
}

optional<int64_t> MatchTrivialLoopTripCount(const HloInstruction* while_op,
                                            int64_t indvar_tuple_idx,
                                            const Literal& indvar_init) {
  // First, find the scalar constant init that `i` is initialized to.
  optional<int64_t> indvar_init_val =
      LiteralUtil::LiteralAsScalarInt64(indvar_init);
  if (!indvar_init_val) {
    VLOG(2) << "Pattern-match failed: induction variable init is not a "
               "constant scalar representable as an int64_t: "
            << indvar_init.ToString();
    return nullopt;
  }

  // Check that `i` goes as `i += k` in the while body where k is a natural
  // number.
  auto* while_body = while_op->while_body();
  auto* while_body_indvar_update =
      const_cast<HloInstruction*>(TraceThroughCopyAndGteTupleChain(
          while_body->root_instruction()->mutable_operand(indvar_tuple_idx)));
  auto* while_body_indvar = NonConstantOperand(while_body_indvar_update);
  if (while_body_indvar == nullptr ||
      TraceThroughCopyAndGteTupleChain(while_body_indvar) !=
          hlo_query::GetUniqueGteInstruction(
              while_body->parameter_instruction(0), indvar_tuple_idx)) {
    return std::nullopt;
  }
  HloInstruction* trip_count_increase_step_instr = nullptr;
  int64_t trip_count_step = 0;
  if (!Match(while_body_indvar_update,
             m::AddAnyOrder(m::Op().Is(while_body_indvar),
                            m::Constant(&trip_count_increase_step_instr)))) {
    if (trip_count_increase_step_instr == nullptr) {
      VLOG(2) << "Pattern-match failed: induction variable is not getting "
                 "updated by an add operation: "
              << while_body_indvar_update->ToString();
      return nullopt;
    }
    if (!trip_count_increase_step_instr->IsConstant() ||
        !ShapeUtil::IsEffectiveScalar(
            trip_count_increase_step_instr->shape())) {
      VLOG(2) << "Pattern-match failed: induction variable is not getting "
                 "incremented by constant: "
              << while_body_indvar_update->ToString();
      return nullopt;
    }
    if (!LiteralUtil::LiteralAsScalarInt64(
             trip_count_increase_step_instr->literal())
             .has_value()) {
      VLOG(2)
          << "Pattern-match failed: trip count step is not an integral type: "
          << trip_count_increase_step_instr->shape().ToString();
      return nullopt;
    }
    VLOG(2) << "Pattern-match for trip count step failed: "
            << trip_count_increase_step_instr->ToString();
  }

  trip_count_step = LiteralUtil::LiteralAsScalarInt64(
                        trip_count_increase_step_instr->literal())
                        .value();
  if (trip_count_step <= 0) {
    VLOG(2) << "Pattern-match failed: trip count step is not a natural number: "
            << trip_count_step;
    return nullopt;
  }
  // Check that we do op(i, N) or op(N, i) as the while condition.  Capture the
  // value N.
  auto* while_cond = while_op->while_condition();
  auto* while_cond_root = while_cond->root_instruction();
  auto* while_cond_indvar = NonConstantOperand(while_cond_root);
  if (while_cond_indvar == nullptr ||
      while_cond_indvar !=
          hlo_query::GetUniqueGteInstruction(
              while_cond->parameter_instruction(0), indvar_tuple_idx)) {
    return std::nullopt;
  }
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
      LiteralUtil::LiteralAsScalarInt64(while_cond_bound->literal());
  if (!while_cond_bound_val) {
    VLOG(2) << "Pattern-match failed: while condition induction variable is "
               "not a constant scalar representable as an int64_t.";
    return nullopt;
  }

  // Handle `i = init; i < N; i+=k`.
  if (Match(while_cond_root,
            m::Op()
                .WithComparisonDirection(ComparisonDirection::kLt)
                .WithOperand(0, m::Op().Is(while_cond_indvar)))) {
    VLOG(2) << "Pattern-match succeeded: loop condition is i < N: "
            << while_cond_root->ToString();
    optional<int64_t> trips =
        CheckedSubtract(*while_cond_bound_val, *indvar_init_val);
    if (trips) {
      const int64_t remainder = std::remainder(*trips, trip_count_step);
      const int64_t div = std::floor(*trips / trip_count_step);
      if (remainder == 0) {
        return std::max(int64_t{0}, div);
      }
      trips = CheckedAdd(div, 1);
      if (!trips) {
        VLOG(2) << "Pattern-match failed: Trip count exceeds INT64_MAX.";
        return nullopt;
      }
      if (*trips < *while_cond_bound_val) {
        return std::max(int64_t{0}, *trips);
      }
      return std::max(int64_t{0}, div);
    }
    VLOG(2) << "Pattern-match failed: Trip count exceeds INT64_MAX.";
    return nullopt;
  }

  // Handle `i = init; i <= N; i+=k`.
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
    trips = CheckedAdd(std::floor(*trips / trip_count_step), 1);
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

optional<int64_t> ComputeWhileLoopTripCount(const HloInstruction* while_op,
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
  auto* while_init = while_op->operand(0);
  auto* indvar_init = while_init->operand(*indvar_tuple_idx);
  absl::StatusOr<Literal> indvar_init_result =
      evaluator.Evaluate(TraceThroughCopyAndGteTupleChain(indvar_init));
  VLOG(2) << "indvar_init_result: " << indvar_init_result.status();
  if (!indvar_init_result.ok()) {
    VLOG(2) << "Couldn't evaluate induction variable init, "
            << indvar_init_result.status() << ", " << indvar_init->ToString();
    return nullopt;
  }
  Literal indvar_iter_val = std::move(indvar_init_result).value();

  // First, try to pattern-match.
  if (auto trip_count = MatchTrivialLoopTripCount(while_op, *indvar_tuple_idx,
                                                  indvar_iter_val)) {
    return trip_count;
  }

  // If our pattern-match failed, try brute-forcing the loop trip count.
  auto* while_body = while_op->while_body();
  auto* while_body_indvar_update =
      while_body->root_instruction()->operand(*indvar_tuple_idx);
  auto* while_body_indvar = NonConstantOperand(while_body_indvar_update);
  if (while_body_indvar == nullptr ||
      while_body_indvar !=
          hlo_query::GetUniqueGteInstruction(
              while_body->parameter_instruction(0), *indvar_tuple_idx)) {
    return std::nullopt;
  }

  auto* while_cond = while_op->while_condition();
  auto* while_cond_root = while_cond->root_instruction();
  auto* while_cond_indvar = NonConstantOperand(while_cond_root);
  if (while_cond_indvar == nullptr ||
      while_cond_indvar !=
          hlo_query::GetUniqueGteInstruction(
              while_cond->parameter_instruction(0), *indvar_tuple_idx)) {
    return std::nullopt;
  }

  for (int64_t trip_count = 0; trip_count != max_brute_force_iters + 1;
       ++trip_count) {
    absl::StatusOr<Literal> result = evaluator.Evaluate(
        while_cond_root, {}, false, {{while_cond_indvar, &indvar_iter_val}});
    if (!result.ok()) {
      VLOG(2) << "Couldn't evaluate while cond: " << result.status();
      return nullopt;
    }
    if (result.value().data<bool>() == absl::Span<const bool>{false}) {
      VLOG(2) << "Loop has static trip count of " << trip_count;
      return trip_count;
    }

    // Calculate the value of the induction variable after one iteration of the
    // loop, and check whether the while condition is true with this new value.
    absl::StatusOr<Literal> indvar_next_result =
        evaluator.Evaluate(while_body_indvar_update, {}, false,
                           {{while_body_indvar, &indvar_iter_val}});
    if (!indvar_next_result.ok()) {
      VLOG(2) << "Couldn't evaluate induction variable update: "
              << indvar_next_result.status();
      return nullopt;
    }
    indvar_iter_val = std::move(indvar_next_result).value();
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
    const HloInstruction* while_op) {
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
  // Create a new while cond computation accessing only the single parameter
  // extracted by the GTE above to avoid excessive memory allocation for the
  // evaluator.
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  auto new_param = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({cond_gte->shape()}), "temp");
  replacements[cond_gte] =
      HloInstruction::CreateGetTupleElement(new_param.get(), 0);
  replacements[while_cond_param] = std::move(new_param);
  auto new_module = std::make_unique<HloModule>("temp_mod", HloModuleConfig{});
  auto* new_computation = new_module->AddEmbeddedComputation(
      while_cond->CloneWithReplacements(&replacements));

  // We have a constant. Evaluate the condition on this constant.
  HloEvaluator evaluator(/*max_loop_iterations=*/0);
  Literal fake_input = Literal::CreateFromShape(
      new_computation->parameter_instruction(0)->shape());
  TF_CHECK_OK(fake_input.CopyFrom(while_body_indvar->literal(),
                                  /*dest_shape_index=*/{0},
                                  /*src_shape_index=*/{}));
  absl::StatusOr<Literal> eval_result =
      evaluator.Evaluate(*new_computation, {std::move(fake_input)});

  if (!eval_result.ok()) {
    VLOG(2) << "Couldn't evaluate while loop condition.";
    return nullopt;
  }

  Literal cond_result_pred = std::move(eval_result.value());
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

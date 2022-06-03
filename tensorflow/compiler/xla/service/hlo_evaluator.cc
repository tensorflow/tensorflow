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
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/internal/endian.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_matmul.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator_typed_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {

namespace {

template <typename OperandT>
StatusOr<Literal> Compare(const Shape& shape, ComparisonDirection direction,
                          LiteralSlice lhs_literal, LiteralSlice rhs_literal) {
  std::function<bool(OperandT, OperandT)> compare_op;
  switch (direction) {
    case ComparisonDirection::kEq:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el == rhs_el;
      };
      break;
    case ComparisonDirection::kNe:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el != rhs_el;
      };
      break;
    case ComparisonDirection::kGe:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el >= rhs_el;
      };
      break;
    case ComparisonDirection::kGt:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el > rhs_el;
      };
      break;
    case ComparisonDirection::kLe:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el <= rhs_el;
      };
      break;
    case ComparisonDirection::kLt:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el < rhs_el;
      };
      break;
  }

  Literal result(shape);
  TF_RETURN_IF_ERROR(
      result.Populate<bool>([&](absl::Span<const int64_t> multi_index) {
        return compare_op(lhs_literal.Get<OperandT>(multi_index),
                          rhs_literal.Get<OperandT>(multi_index));
      }));

  return std::move(result);
}

template <>
StatusOr<Literal> Compare<complex64>(const Shape& shape,
                                     ComparisonDirection direction,
                                     LiteralSlice lhs_literal,
                                     LiteralSlice rhs_literal) {
  std::function<bool(complex64, complex64)> compare_op;
  switch (direction) {
    case ComparisonDirection::kEq:
      compare_op = [](complex64 lhs_el, complex64 rhs_el) {
        return lhs_el == rhs_el;
      };
      break;
    case ComparisonDirection::kNe:
      compare_op = [](complex64 lhs_el, complex64 rhs_el) {
        return lhs_el != rhs_el;
      };
      break;
    default:
      LOG(FATAL) << "unhandled direction for conversion to Comparison: "
                 << ComparisonDirectionToString(direction);
  }

  Literal result(shape);
  TF_RETURN_IF_ERROR(
      result.Populate<bool>([&](absl::Span<const int64_t> multi_index) {
        return compare_op(lhs_literal.Get<complex64>(multi_index),
                          rhs_literal.Get<complex64>(multi_index));
      }));

  return std::move(result);
}

template <>
StatusOr<Literal> Compare<complex128>(const Shape& shape,
                                      ComparisonDirection direction,
                                      LiteralSlice lhs_literal,
                                      LiteralSlice rhs_literal) {
  std::function<bool(complex128, complex128)> compare_op;
  switch (direction) {
    case ComparisonDirection::kEq:
      compare_op = [](complex128 lhs_el, complex128 rhs_el) {
        return lhs_el == rhs_el;
      };
      break;
    case ComparisonDirection::kNe:
      compare_op = [](complex128 lhs_el, complex128 rhs_el) {
        return lhs_el != rhs_el;
      };
      break;
    default:
      LOG(FATAL) << "unhandled direction for conversion to Comparison: "
                 << ComparisonDirectionToString(direction);
  }

  Literal result(shape);
  TF_RETURN_IF_ERROR(
      result.Populate<bool>([&](absl::Span<const int64_t> multi_index) {
        return compare_op(lhs_literal.Get<complex128>(multi_index),
                          rhs_literal.Get<complex128>(multi_index));
      }));

  return std::move(result);
}

// Represents an index into the while argument tuple and / or a value.
// At least one of param_index and value has a value; both of them could have
// a value.
struct ParamIndexAndValue {
  std::optional<int64_t> param_index;
  std::optional<int64_t> value;

  bool IsValid() const { return param_index.has_value() || value.has_value(); }
};

// Represents the while loop condition comparison.
// We assume comparison is of the form: lhs comp rhs.
struct WhileCondComparison {
  ComparisonDirection comparson_direction;
  ParamIndexAndValue lhs;
  ParamIndexAndValue rhs;
};

// Represents the parsed while loop condition. The loop induction variable may
// either be used in a comparison or returned directly, i.e., NoOp. In the case
// of NoOp, it contains the parameter index and initial value of the loop
// induction variable.
using WhileCondComparisonOrNoOp =
    absl::variant<WhileCondComparison, ParamIndexAndValue>;

// Finds the while loop condition comparison by matching the loop condition root
// with known patterns.
std::optional<WhileCondComparisonOrNoOp> PatternMatchLoopCondComparison(
    HloInstruction* loop_cond_root) {
  // Base pattern #1: gte-0 comp gte-1
  if (Match(loop_cond_root,
            match::Compare()
                .WithOperand(0, match::GetTupleElement().WithOperand(
                                    0, match::Parameter().WithParameterNum(0)))
                .WithOperand(1,
                             match::GetTupleElement().WithOperand(
                                 0, match::Parameter().WithParameterNum(0))))) {
    return WhileCondComparison{
        loop_cond_root->comparison_direction(),
        {/*param_index=*/loop_cond_root->operand(0)->tuple_index()},
        {/*param_index=*/loop_cond_root->operand(1)->tuple_index()}};
  }
  // Base pattern #2: constant comp gte
  if (Match(loop_cond_root,
            match::Compare()
                .WithOperand(0, match::Constant())
                .WithOperand(1,
                             match::GetTupleElement().WithOperand(
                                 0, match::Parameter().WithParameterNum(0))))) {
    std::optional<int64_t> lhs_value =
        loop_cond_root->operand(0)->literal().GetFirstInteger();
    if (!lhs_value.has_value()) {
      return std::nullopt;
    }
    return WhileCondComparison{
        loop_cond_root->comparison_direction(),
        {/*param_index=*/std::nullopt, /*value=*/*lhs_value},
        {/*param_index=*/loop_cond_root->operand(1)->tuple_index()}};
  }
  // Base pattern #3: gte comp constant
  if (Match(loop_cond_root,
            match::Compare()
                .WithOperand(0, match::GetTupleElement().WithOperand(
                                    0, match::Parameter().WithParameterNum(0)))
                .WithOperand(1, match::Constant()))) {
    std::optional<int64_t> rhs_value =
        loop_cond_root->operand(1)->literal().GetFirstInteger();
    if (!rhs_value.has_value()) {
      return std::nullopt;
    }
    return WhileCondComparison{
        loop_cond_root->comparison_direction(),
        {/*param_index=*/loop_cond_root->operand(0)->tuple_index(),
         /*value=*/std::nullopt},
        {/*param_index=*/std::nullopt, /*value=*/*rhs_value},
    };
  }
  // Base pattern #4: gte is a boolean scalar and it was return immediately.
  if (Match(loop_cond_root, match::GetTupleElement().WithOperand(
                                0, match::Parameter().WithParameterNum(0)))) {
    if (loop_cond_root->shape().element_type() != PrimitiveType::PRED &&
        loop_cond_root->shape().rank() != 0) {
      return std::nullopt;
    }
    return ParamIndexAndValue{{/*param_index=*/loop_cond_root->tuple_index()}};
  }

  // Recursive pattern #1:
  // loop_cond_root is a GetTupleElement whose operand is a call with a single
  // parameter which takes the computation's single parameter.
  // In this case, if the called computation's root is a tuple, we can recurse
  // on that tuple's element as the new loop_cond_root.
  if (Match(loop_cond_root,
            match::GetTupleElement().WithOperand(
                0, match::Call().WithNumOperands(1).WithOperand(
                       0, match::Parameter().WithParameterNum(0))))) {
    HloInstruction* call_instruction = loop_cond_root->mutable_operand(0);
    HloComputation* to_apply = call_instruction->to_apply();
    HloInstruction* to_apply_root = to_apply->root_instruction();
    if (Match(to_apply_root, match::Tuple())) {
      return PatternMatchLoopCondComparison(
          to_apply_root->mutable_operand(loop_cond_root->tuple_index()));
    }
  }
  // Recursive pattern #2:
  // loop_cond_root is a GetTupleElement whose operand is a tuple.
  // We can recurse on the tuple's element as the new loop_cond_root.
  if (Match(loop_cond_root,
            match::GetTupleElement().WithOperand(0, match::Tuple()))) {
    HloInstruction* new_cond_root =
        loop_cond_root->mutable_operand(0)->mutable_operand(
            loop_cond_root->tuple_index());
    return PatternMatchLoopCondComparison(new_cond_root);
  }
  return std::nullopt;
}

// Tries to parse the loop body to find how the induction variable is updated
// using pattern matching.
std::optional<int64_t> PatternMatchInductionVarUpdate(
    HloInstruction* loop_body_root, int64_t tuple_index) {
  // Pattern #1: induc_var = induc_var + constant
  if (Match(loop_body_root,
            match::Tuple().WithOperand(
                tuple_index,
                match::Add()
                    .WithOperand(0, match::GetTupleElement()
                                        .WithTupleIndex(tuple_index)
                                        .WithOperand(0, match::Parameter()))
                    .WithOperand(1, match::Constant())))) {
    std::optional<int64_t> step_size = loop_body_root->operand(tuple_index)
                                           ->operand(1)
                                           ->literal()
                                           .GetFirstInteger();
    if (!step_size.has_value()) {
      return std::nullopt;
    }
    return *step_size;
  }
  // Pattern #2: induc_var = constant + induc_var
  if (Match(
          loop_body_root,
          match::Tuple().WithOperand(
              tuple_index,
              match::Add()
                  .WithOperand(0, match::Constant())
                  .WithOperand(1, match::GetTupleElement()
                                      .WithTupleIndex(tuple_index)
                                      .WithOperand(0, match::Parameter()))))) {
    std::optional<int64_t> step_size = loop_body_root->operand(tuple_index)
                                           ->operand(0)
                                           ->literal()
                                           .GetFirstInteger();
    if (!step_size.has_value()) {
      return std::nullopt;
    }
    return *step_size;
  }

  // Pattern #3: induc_var = induc_var - constant
  if (Match(loop_body_root,
            match::Tuple().WithOperand(
                tuple_index,
                match::Subtract()
                    .WithOperand(0, match::GetTupleElement()
                                        .WithTupleIndex(tuple_index)
                                        .WithOperand(0, match::Parameter()))
                    .WithOperand(1, match::Constant())))) {
    std::optional<int64_t> step_size = loop_body_root->operand(tuple_index)
                                           ->operand(1)
                                           ->literal()
                                           .GetFirstInteger();
    if (!step_size.has_value()) {
      return std::nullopt;
    }
    return -*step_size;
  }

  // Pattern #4: the induc_var is directly returned from the loop body with
  // no changes.
  if (Match(loop_body_root,
            match::Tuple().WithOperand(
                tuple_index,
                match::GetTupleElement()
                    .WithOperand(0, match::Parameter().WithParameterNum(0))
                    .WithTupleIndex(tuple_index)))) {
    return 0;
  }
  return std::nullopt;
}

std::optional<bool> PatternMatchLoopCondVarOverride(
    HloInstruction* loop_body_root, int64_t tuple_index) {
  if (Match(loop_body_root, match::Tuple()) &&
      loop_body_root->operand_count() > tuple_index) {
    HloInstruction* cond_var_override =
        loop_body_root->mutable_operand(tuple_index);
    HloEvaluator evaluator;
    StatusOr<Literal> new_cond_var = evaluator.Evaluate(
        cond_var_override, /*recursively_evaluate_nonconstant_operands=*/true);
    if (new_cond_var.ok()) {
      return new_cond_var->GetFirstElement<bool>();
    }
  }
  return std::nullopt;
}

// Repesents a value that might or might not be determined statically.
struct DynamicOrStaticValue {
  std::optional<int64_t> static_value;
  bool is_dynamic() const { return !static_value.has_value(); }
};

constexpr absl::string_view kEvalErrorDetailUrl = "EvalErrorDetailUrl";

// Use this class to represent the precise details of the error to enable
// special treatment.
enum class EvalErrorDetail : uint32_t {
  // The evaluation result depends on dynamic values such as parameters and
  // infeed. Therefore, the HLO's value cannot be statically evaluated.
  kDynamicValueDependence = 0,
};

Status MakeEvalErrorDueToParamOrInfeed() {
  Status error = tensorflow::errors::FailedPrecondition(
      "Failed to evaluate instruction since it depends on infeed or "
      "parameters to its parent computation.");
  std::string error_payload;
  error_payload.resize(sizeof(EvalErrorDetail));
  absl::little_endian::Store32(
      const_cast<char*>(error_payload.data()),
      static_cast<uint32_t>(EvalErrorDetail::kDynamicValueDependence));
  error.SetPayload(kEvalErrorDetailUrl, error_payload);
  return error;
}

std::optional<EvalErrorDetail> ParseEvalErrorDetail(const Status& error) {
  std::optional<tensorflow::StringPiece> error_detail =
      error.GetPayload(kEvalErrorDetailUrl);
  if (!error_detail.has_value() && error_detail->empty()) {
    return std::nullopt;
  }
  return static_cast<EvalErrorDetail>(
      absl::little_endian::Load32(error_detail->data()));
}

// A convenience wrapper to compute the while loop's argument's init value at
// the given tuple_index. If the init value depends on parameters to the
// while loop's parent computation or infeed, we consider the init value
// dynamic.
std::optional<DynamicOrStaticValue> EvaluateWhileLoopParamInitValue(
    HloInstruction* param_instruction, int64_t tuple_index) {
  if (param_instruction->opcode() != HloOpcode::kTuple) {
    return std::nullopt;
  }
  HloInstruction* element_instruction =
      param_instruction->mutable_operand(tuple_index);
  HloEvaluator evaluator;
  StatusOr<Literal> value = evaluator.Evaluate(
      element_instruction, /*recursively_evaluate_nonconstant_operands=*/true);
  if (value.ok()) {
    if (element_instruction->shape().element_type() == PrimitiveType::PRED) {
      return DynamicOrStaticValue{
          static_cast<int64_t>(value->GetFirstElement<bool>())};
    } else {
      return DynamicOrStaticValue{value->GetFirstInteger()};
    }
  } else {
    std::optional<EvalErrorDetail> eval_error_detail =
        ParseEvalErrorDetail(value.status());
    if (eval_error_detail.has_value() &&
        *eval_error_detail == EvalErrorDetail::kDynamicValueDependence) {
      return DynamicOrStaticValue{std::nullopt};
    }
  }
  return std::nullopt;
}

}  // namespace

std::optional<ParsedWhileLoop> PatternMatchParseWhileLoop(
    HloInstruction* while_op) {
  HloComputation* while_cond = while_op->while_condition();
  HloComputation* while_body = while_op->while_body();
  HloInstruction* while_operand = while_op->mutable_operand(0);
  // Try to parse the loop condition comparison.
  std::optional<WhileCondComparisonOrNoOp> loop_comparison_or_noop =
      PatternMatchLoopCondComparison(while_cond->root_instruction());
  if (!loop_comparison_or_noop.has_value()) {
    return std::nullopt;
  }
  if (loop_comparison_or_noop->index() == 1) {
    ParamIndexAndValue& parameter_index_and_value =
        absl::get<ParamIndexAndValue>(*loop_comparison_or_noop);
    CHECK(parameter_index_and_value.param_index.has_value());
    int64_t loop_cond_var_index = *parameter_index_and_value.param_index;
    std::optional<DynamicOrStaticValue> noop_value =
        EvaluateWhileLoopParamInitValue(while_operand, loop_cond_var_index);

    if (noop_value.has_value()) {
      if (noop_value->is_dynamic()) {
        return kParsedDynamicWhileLoop;
      } else if (*noop_value->static_value == 0) {
        return ParsedWhileLoop{
            ParsedStaticWhileLoop{/*trip_count=*/0,
                                  /*induction_var_index=*/loop_cond_var_index,
                                  /*induction_var_init_value=*/0,
                                  /*step_size=*/0,
                                  /*loop_bound=*/0}};
      }
      std::optional<bool> updated_loop_cond_var =
          PatternMatchLoopCondVarOverride(while_body->root_instruction(),
                                          loop_cond_var_index);
      if (updated_loop_cond_var.has_value()) {
        if (!*updated_loop_cond_var) {
          return ParsedWhileLoop{
              ParsedStaticWhileLoop{/*trip_count=*/1,
                                    /*induction_var_index=*/loop_cond_var_index,
                                    /*induction_var_init_value=*/0,
                                    /*step_size=*/1,
                                    /*loop_bound=*/1}};
        } else {
          // This is an infinite loop and we set trip_count to -1.
          return ParsedWhileLoop{
              ParsedStaticWhileLoop{/*trip_count=*/-1,
                                    /*induction_var_index=*/loop_cond_var_index,
                                    /*induction_var_init_value=*/0,
                                    /*step_size=*/0,
                                    /*loop_bound=*/1}};
        }
      }
    }
    return std::nullopt;
  }
  CHECK_EQ(loop_comparison_or_noop->index(), 0);
  WhileCondComparison loop_comparison =
      absl::get<WhileCondComparison>(*loop_comparison_or_noop);
  CHECK(loop_comparison.lhs.IsValid() && loop_comparison.rhs.IsValid());

  // If the while loop condition comparison's both sides take an init value
  // from the while loop's parent computation's parameter, the loop is dynamic.
  if (while_operand->opcode() == HloOpcode::kParameter) {
    if (loop_comparison.lhs.param_index.has_value() ||
        loop_comparison.rhs.param_index.has_value()) {
      return kParsedDynamicWhileLoop;
    }
  }

  // We can't handle the case when the while loop argument is not a Tuple
  // instruction.
  if (while_operand->opcode() != HloOpcode::kTuple) {
    return std::nullopt;
  }

  // If loop cond comparison LHS does not have a value defined inside the loop
  // cond computation, try to evaluate its init value inside the while loop's
  // parent computation.
  if (!loop_comparison.lhs.value.has_value()) {
    std::optional<DynamicOrStaticValue> lhs_init_value =
        EvaluateWhileLoopParamInitValue(while_operand,
                                        *loop_comparison.lhs.param_index);
    if (lhs_init_value.has_value()) {
      if (lhs_init_value->is_dynamic()) {
        return kParsedDynamicWhileLoop;
      } else {
        loop_comparison.lhs.value = *(lhs_init_value->static_value);
      }
    } else {
      return std::nullopt;
    }
  }

  // If loop cond comparison RHS does not have a value defined inside the loop
  // cond computation, try to evaluate its init value inside the while loop's
  // parent computation.
  if (!loop_comparison.rhs.value.has_value()) {
    std::optional<DynamicOrStaticValue> rhs_init_value =
        EvaluateWhileLoopParamInitValue(while_operand,
                                        *loop_comparison.rhs.param_index);
    if (rhs_init_value.has_value()) {
      if (rhs_init_value->is_dynamic()) {
        return kParsedDynamicWhileLoop;
      } else {
        loop_comparison.rhs.value = *(rhs_init_value->static_value);
      }
    } else {
      return std::nullopt;
    }
  }

  // We have either successfully evaluated the init value for both LHS and RHS
  // or have returned as dynamic loop or failure.
  CHECK(loop_comparison.lhs.value.has_value());
  CHECK(loop_comparison.rhs.value.has_value());

  if (loop_comparison.lhs.param_index.has_value()) {
    VLOG(3) << __func__ << " lhs index: " << *loop_comparison.lhs.param_index;
  }

  VLOG(3) << __func__ << " lhs bound: " << *loop_comparison.lhs.value;

  if (loop_comparison.rhs.param_index.has_value()) {
    VLOG(3) << __func__ << " rhs index: " << *loop_comparison.rhs.param_index;
  }

  VLOG(3) << __func__ << " rhs bound: " << *loop_comparison.rhs.value;

  // Check whether LHS is the loop induction var.
  std::optional<int64_t> lhs_induction_var_update;
  if (loop_comparison.lhs.param_index.has_value()) {
    lhs_induction_var_update = PatternMatchInductionVarUpdate(
        while_body->root_instruction(), *loop_comparison.lhs.param_index);
  }

  // Check whether LHS is the loop induction var.
  std::optional<int64_t> rhs_induction_var_update;
  if (loop_comparison.rhs.param_index.has_value()) {
    rhs_induction_var_update = PatternMatchInductionVarUpdate(
        while_body->root_instruction(), *loop_comparison.rhs.param_index);
  }

  // Lhs is the induction variable.
  if (lhs_induction_var_update.has_value()) {
    // We cannot handle the case when both LHS and RHS are updated inside
    // the loop body.
    if (rhs_induction_var_update.has_value() &&
        *rhs_induction_var_update != 0) {
      return std::nullopt;
    }
    if (*lhs_induction_var_update > 0 &&
        (loop_comparison.comparson_direction == Comparison::Direction::kLt ||
         loop_comparison.comparson_direction == Comparison::Direction::kLe)) {
      int64_t trip_count =
          (*loop_comparison.rhs.value - *loop_comparison.lhs.value - 1) /
              *lhs_induction_var_update +
          1;
      // Additional logic to deal with Equal comparison.
      if (loop_comparison.comparson_direction == Comparison::Direction::kLe &&
          (*loop_comparison.rhs.value - *loop_comparison.lhs.value) %
                  *lhs_induction_var_update ==
              0) {
        trip_count += 1;
      }
      return ParsedWhileLoop{ParsedStaticWhileLoop{
          /*trip_count=*/trip_count,
          /*induction_var_index=*/*loop_comparison.lhs.param_index,
          /*induction_var_init_value=*/*loop_comparison.lhs.value,
          /*step_size=*/*lhs_induction_var_update,
          /*loop_bound=*/*loop_comparison.rhs.value}};
    } else if (*lhs_induction_var_update < 0 &&
               (loop_comparison.comparson_direction ==
                    Comparison::Direction::kGt ||
                loop_comparison.comparson_direction ==
                    Comparison::Direction::kGe)) {
      int trip_count =
          (*loop_comparison.lhs.value - *loop_comparison.rhs.value - 1) /
              *lhs_induction_var_update +
          1;
      if (loop_comparison.comparson_direction == Comparison::Direction::kGe &&
          (*loop_comparison.lhs.value - *loop_comparison.rhs.value) %
                  *lhs_induction_var_update ==
              0) {
        trip_count += 1;
      }
      return ParsedWhileLoop{ParsedStaticWhileLoop{
          /*trip_count=*/trip_count,
          /*induction_var_index=*/*(loop_comparison.lhs.param_index),
          /*induction_var_init_value=*/*(loop_comparison.lhs.value),
          /*step_size=*/-*lhs_induction_var_update,
          /*loop_bound=*/*(loop_comparison.rhs.value)}};
    }
    return std::nullopt;
  }
  // Rhs is the induction variable.
  if (rhs_induction_var_update.has_value()) {
    // We cannot handle the case when both LHS and RHS are updated inside
    // the loop body.
    if (lhs_induction_var_update.has_value() &&
        *lhs_induction_var_update == 0) {
      return std::nullopt;
    }
    if (*rhs_induction_var_update > 0 &&
        (loop_comparison.comparson_direction == Comparison::Direction::kGt ||
         loop_comparison.comparson_direction == Comparison::Direction::kGe)) {
      int trip_count =
          (*loop_comparison.lhs.value - *loop_comparison.rhs.value - 1) /
              *rhs_induction_var_update +
          1;
      if (loop_comparison.comparson_direction == Comparison::Direction::kGe &&
          (*loop_comparison.lhs.value - *loop_comparison.rhs.value) %
                  *rhs_induction_var_update ==
              0) {
        trip_count += 1;
      }
      return ParsedWhileLoop{ParsedStaticWhileLoop{
          /*trip_count=*/trip_count,
          /*induction_var_index=*/*(loop_comparison.rhs.param_index),
          /*induction_var_init_value=*/*(loop_comparison.rhs.value),
          /*step_size=*/*rhs_induction_var_update,
          /*loop_bound=*/*(loop_comparison.lhs.value)}};
    } else if (*rhs_induction_var_update < 0 &&
               (loop_comparison.comparson_direction ==
                    Comparison::Direction::kLt ||
                loop_comparison.comparson_direction ==
                    Comparison::Direction::kLe)) {
      int trip_count =
          (*loop_comparison.rhs.value - *loop_comparison.lhs.value - 1) /
              *rhs_induction_var_update +
          1;
      if (loop_comparison.comparson_direction == Comparison::Direction::kLe &&
          (*loop_comparison.rhs.value - *loop_comparison.lhs.value) %
                  *rhs_induction_var_update ==
              0) {
        trip_count += 1;
      }
      return ParsedWhileLoop{ParsedStaticWhileLoop{
          /*trip_count=*/trip_count,
          /*induction_var_index=*/*(loop_comparison.rhs.param_index),
          /*induction_var_init_value=*/*(loop_comparison.rhs.value),
          /*step_size=*/-*rhs_induction_var_update,
          /*loop_bound=*/*(loop_comparison.lhs.value)}};
    }
    return std::nullopt;
  }
  return std::nullopt;
}

// Note that unsupported types by the typed visitor does not necessarily imply
// the non-typed HloEvaluator (parent evaluator) would not support them either
// in the type-agnostic handler. For e.g., HandleGetTupleElement in the parent
// type-agnostic evaluator will be able to accept Tuple primitive type, whereas
// HloEvaluatorTypedVisitor cannot.
HloEvaluator::HloEvaluator(int64_t max_loop_iterations)
    : max_loop_iterations_(max_loop_iterations) {
  typed_visitors_[PRED] =
      absl::make_unique<HloEvaluatorTypedVisitor<bool>>(this);
  typed_visitors_[U8] =
      absl::make_unique<HloEvaluatorTypedVisitor<uint8_t>>(this);
  typed_visitors_[U16] =
      absl::make_unique<HloEvaluatorTypedVisitor<uint16_t>>(this);
  typed_visitors_[U32] =
      absl::make_unique<HloEvaluatorTypedVisitor<uint32_t>>(this);
  typed_visitors_[U64] =
      absl::make_unique<HloEvaluatorTypedVisitor<uint64_t>>(this);
  typed_visitors_[S8] =
      absl::make_unique<HloEvaluatorTypedVisitor<int8_t>>(this);
  typed_visitors_[S16] =
      absl::make_unique<HloEvaluatorTypedVisitor<int16_t>>(this);
  typed_visitors_[S32] =
      absl::make_unique<HloEvaluatorTypedVisitor<int32_t>>(this);
  typed_visitors_[S64] =
      absl::make_unique<HloEvaluatorTypedVisitor<int64_t>>(this);
  typed_visitors_[F16] =
      absl::make_unique<HloEvaluatorTypedVisitor<Eigen::half, float>>(this);
  typed_visitors_[F32] =
      absl::make_unique<HloEvaluatorTypedVisitor<float>>(this);
  typed_visitors_[F64] =
      absl::make_unique<HloEvaluatorTypedVisitor<double>>(this);
  typed_visitors_[C64] =
      absl::make_unique<HloEvaluatorTypedVisitor<complex64>>(this);
  typed_visitors_[C128] =
      absl::make_unique<HloEvaluatorTypedVisitor<complex128>>(this);

  // Most of the evaluator computations we use don't support BF16 (e.g.,
  // std::ceil, std::tanh). To make evaluator work with BF16, we set all
  // elementwise computations to be done in F32 and do BF16<->F32 conversion
  // around the input and the output of the computations.
  typed_visitors_[BF16] =
      absl::make_unique<HloEvaluatorTypedVisitor<bfloat16, float>>(this);

  typed_visitors_[TUPLE] =
      absl::make_unique<FunctionVisitor>([](HloInstruction*) {
        return Unimplemented(
            "HloEvaluatorTypedVisitor: unhandled primitive type: TUPLE.");
      });
  typed_visitors_[OPAQUE_TYPE] =
      absl::make_unique<FunctionVisitor>([](HloInstruction*) {
        return Unimplemented(
            "HloEvaluatorTypedVisitor: unhandled primitive type: OPAQUE_TYPE.");
      });
  typed_visitors_[TOKEN] =
      absl::make_unique<FunctionVisitor>([](HloInstruction*) {
        return Unimplemented(
            "HloEvaluatorTypedVisitor: unhandled primitive type: TOKEN.");
      });
}

StatusOr<Literal> HloEvaluator::Evaluate(
    const HloComputation& computation,
    absl::Span<const Literal* const> arg_literals) {
  CHECK(computation.parent() != nullptr);
  XLA_VLOG_LINES(
      2, "HloEvaluator::Evaluate computation:\n" + computation.ToString());

  if (arg_literals.size() != computation.num_parameters()) {
    return InvalidArgument(
        "Expected %d argument%s, but got %d.", computation.num_parameters(),
        computation.num_parameters() == 1 ? "" : "s", arg_literals.size());
  }
  for (int64_t i = 0; i < arg_literals.size(); ++i) {
    const auto& computation_shape =
        computation.parameter_instruction(i)->shape();
    const auto& arg_shape = arg_literals[i]->shape();
    if (!Shape::Equal().MinorToMajorOnlyInLayout()(computation_shape,
                                                   arg_shape)) {
      return InvalidArgument(
          "Shape mismatch at parameter %d. Computation expected %s, but arg "
          "was %s.",
          i, ShapeUtil::HumanStringWithLayout(computation_shape),
          ShapeUtil::HumanStringWithLayout(arg_shape));
    }
  }

  evaluated_.clear();
  arg_literals_.clear();
  for (const auto& literal_ptr : arg_literals) {
    arg_literals_.push_back(&*literal_ptr);
  }

  // Re-seed RNG, either from the configuration's seed or a monotonic
  // per-evaluator seed (which prevents two evaluators from returning the same
  // random sequence).
  if (computation.parent()->config().seed()) {
    seed_ = computation.parent()->config().seed();
  } else {
    // Start global_seed at a (true) random value.
    static std::atomic<uint64_t> global_seed{std::random_device()()};
    seed_ = global_seed.fetch_add(1);
  }
  engine_.seed(seed_);

  TF_RETURN_IF_ERROR(computation.Accept(this));
  const Literal& result =
      GetEvaluatedLiteralFor(computation.root_instruction());
  if (VLOG_IS_ON(100)) {
    for (const HloInstruction* instr : computation.instructions()) {
      VLOG(100) << instr->name() << " = " << GetEvaluatedLiteralFor(instr);
    }
  }
  if (!result.IsKnown()) {
    return MakeEvalErrorDueToParamOrInfeed();
  }
  return result.Clone();
}

StatusOr<Literal> HloEvaluator::Evaluate(
    HloInstruction* instruction,
    bool recursively_evaluate_nonconstant_operands) {
  arg_literals_.clear();
  evaluated_.clear();
  auto enable_partial_evaluation_cleanup =
      absl::MakeCleanup([this] { enable_partial_evaluation_ = false; });
  enable_partial_evaluation_ = recursively_evaluate_nonconstant_operands;
  TF_RETURN_IF_ERROR(
      EvaluateInternal(instruction, /*shape_index=*/{},
                       recursively_evaluate_nonconstant_operands));
  const Literal& result = GetEvaluatedLiteralFor(instruction);
  if (!result.IsKnown()) {
    return MakeEvalErrorDueToParamOrInfeed();
  }
  return result.Clone();
}

bool HloEvaluator::TryEvaluate(HloInstruction* instruction, Literal* result,
                               bool recursively_evaluate_nonconstant_operands) {
  CHECK(result != nullptr);
  auto result_or =
      Evaluate(instruction, recursively_evaluate_nonconstant_operands);
  if (!result_or.ok()) {
    VLOG(1) << "TryEvaluate failed:" << result_or.status();
    return false;
  }

  *result = result_or.ConsumeValueOrDie();
  return true;
}

StatusOr<Literal> HloEvaluator::EvaluateWithSubstitutions(
    const HloInstruction* instruction,
    const absl::flat_hash_map<const HloInstruction*, const Literal*>&
        substitutions) {
  std::vector<std::unique_ptr<HloInstruction>> owned_operands;
  for (const HloInstruction* operand : instruction->operands()) {
    auto it = substitutions.find(operand);
    if (it == substitutions.end()) {
      owned_operands.push_back(operand->Clone());
    } else {
      owned_operands.push_back(
          HloInstruction::CreateConstant(it->second->Clone()));
    }
  }

  std::vector<HloInstruction*> operands;
  operands.reserve(owned_operands.size());
  for (auto& operand : owned_operands) {
    operands.push_back(operand.get());
  }

  std::unique_ptr<HloInstruction> cloned_instruction =
      instruction->CloneWithNewOperands(instruction->shape(), operands);
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

StatusOr<Literal> HloEvaluator::EvaluateElementwiseBinaryOp(
    HloOpcode opcode, const Literal& lhs, const Literal& rhs) {
  std::unique_ptr<HloInstruction> lhs_instr =
      HloInstruction::CreateConstant(lhs.Clone());
  std::unique_ptr<HloInstruction> rhs_instr =
      HloInstruction::CreateConstant(rhs.Clone());

  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateBinary(lhs.shape(), opcode, lhs_instr.get(),
                                   rhs_instr.get());
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

StatusOr<Literal> HloEvaluator::EvaluateElementwiseTernaryOp(
    HloOpcode opcode, const Literal& lhs, const Literal& rhs,
    const Literal& ehs) {
  std::unique_ptr<HloInstruction> lhs_instr =
      HloInstruction::CreateConstant(lhs.Clone());
  std::unique_ptr<HloInstruction> rhs_instr =
      HloInstruction::CreateConstant(rhs.Clone());
  std::unique_ptr<HloInstruction> ehs_instr =
      HloInstruction::CreateConstant(ehs.Clone());
  TF_ASSIGN_OR_RETURN(auto output_shape,
                      ShapeInference::InferTernaryOpShape(
                          opcode, lhs.shape(), rhs.shape(), ehs.shape()));
  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateTernary(output_shape, opcode, lhs_instr.get(),
                                    rhs_instr.get(), ehs_instr.get());
  return Evaluate(cloned_instruction.get());
}

StatusOr<Literal> HloEvaluator::EvaluateElementwiseCompareOp(
    ComparisonDirection direction, const Literal& lhs, const Literal& rhs) {
  std::unique_ptr<HloInstruction> lhs_instr =
      HloInstruction::CreateConstant(lhs.Clone());
  std::unique_ptr<HloInstruction> rhs_instr =
      HloInstruction::CreateConstant(rhs.Clone());

  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateCompare(
          ShapeUtil::ChangeElementType(lhs.shape(), PRED), lhs_instr.get(),
          rhs_instr.get(), direction);
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

StatusOr<Literal> HloEvaluator::EvaluateElementwiseUnaryOp(
    HloOpcode opcode, const Literal& operand) {
  std::unique_ptr<HloInstruction> operand_instr =
      HloInstruction::CreateConstant(operand.Clone());

  TF_ASSIGN_OR_RETURN(Shape inferred_shape, ShapeInference::InferUnaryOpShape(
                                                opcode, operand.shape()));
  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateUnary(inferred_shape, opcode, operand_instr.get());
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

StatusOr<Literal> HloEvaluator::EvaluateDotOp(
    const DotDimensionNumbers& dim_numbers,
    const PrecisionConfig& precision_config, const Literal& lhs,
    const Literal& rhs) {
  std::unique_ptr<HloInstruction> lhs_instr =
      HloInstruction::CreateConstant(lhs.Clone());
  std::unique_ptr<HloInstruction> rhs_instr =
      HloInstruction::CreateConstant(rhs.Clone());

  TF_ASSIGN_OR_RETURN(
      Shape dot_shape,
      ShapeInference::InferDotOpShape(lhs.shape(), rhs.shape(), dim_numbers,
                                      /*preferred_element_type=*/std::nullopt));

  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateDot(dot_shape, lhs_instr.get(), rhs_instr.get(),
                                dim_numbers, precision_config);
  return Evaluate(cloned_instruction.get());
}

Status HloEvaluator::EvaluateInternal(
    HloInstruction* instruction, const ShapeIndex& shape_index,
    bool recursively_evaluate_nonconstant_operands) {
  // Don't need to evaluate this instruction again if it has already been
  // evaluated.
  if (IsAlreadyEvaluated(instruction, shape_index)) {
    return OkStatus();
  }

  if (!recursively_evaluate_nonconstant_operands) {
    if (!hlo_query::AllOperandsAreConstants(*instruction)) {
      return tensorflow::errors::FailedPrecondition(
          "Not all operands are constants.");
    }
  } else {
    if (instruction->opcode() == HloOpcode::kGetTupleElement) {
      ShapeIndex new_shape_index = shape_index;
      new_shape_index.push_front(instruction->tuple_index());
      TF_RETURN_IF_ERROR(
          EvaluateInternal(instruction->mutable_operand(0), new_shape_index,
                           /*recursively_evaluate_nonconstant_operands=*/true));
    } else if (instruction->opcode() == HloOpcode::kTuple &&
               !shape_index.empty()) {
      ShapeIndex new_shape_index = shape_index;
      int64_t tuple_index = new_shape_index.front();
      new_shape_index.pop_front();
      TF_RETURN_IF_ERROR(EvaluateInternal(
          instruction->mutable_operand(tuple_index), new_shape_index,
          /*recursively_evaluate_nonconstant_operands=*/true));
    } else {
      for (HloInstruction* operand : instruction->operands()) {
        TF_RETURN_IF_ERROR(EvaluateInternal(
            operand, /*shape_index=*/{},
            /*recursively_evaluate_nonconstant_operands=*/true));
        // Except for the above and following cases, we do not support handling
        // unknown operands for other HLOs. So mark the result as unknown.
        if ((!GetEvaluatedLiteralFor(operand).IsKnown() &&
             instruction->opcode() != HloOpcode::kCopy &&
             instruction->opcode() != HloOpcode::kCopyStart &&
             instruction->opcode() != HloOpcode::kCopyDone &&
             instruction->opcode() != HloOpcode::kAsyncStart &&
             instruction->opcode() != HloOpcode::kAsyncUpdate &&
             instruction->opcode() != HloOpcode::kAsyncDone &&
             instruction->opcode() != HloOpcode::kWhile)) {
          evaluated_[instruction] =
              Literal::CreateFromShapeWithUnknownLeafArrays(
                  instruction->shape());
          return OkStatus();
        }
      }
    }
  }
  visitor_shape_index_ = shape_index;
  TF_RETURN_IF_ERROR(Preprocess(instruction));
  TF_RETURN_IF_ERROR(instruction->Visit(this));
  TF_RETURN_IF_ERROR(Postprocess(instruction));
  return OkStatus();
}

Status HloEvaluator::HandleBitcast(HloInstruction* bitcast) {
  const Literal& operand_literal = GetEvaluatedLiteralFor(bitcast->operand(0));
  Literal result(bitcast->shape());
  // Bitcast output is allowed to be smaller than the input if the backend-
  // specific buffer sizes for the input and output are the same. Since the HLO
  // evaluator doesn't have access to the backend-specific shape size function,
  // assume it's OK to bitcast if output <= input.
  TF_RET_CHECK(operand_literal.size_bytes() >= result.size_bytes());
  memcpy(result.untyped_data(), operand_literal.untyped_data(),
         result.size_bytes());
  evaluated_[bitcast] = std::move(result);
  return OkStatus();
}

Status HloEvaluator::HandleGetDimensionSize(
    HloInstruction* get_dimension_size) {
  HloInstruction* operand = get_dimension_size->mutable_operand(0);
  int64_t dim = get_dimension_size->dimension();
  if (dynamic_dimension_inference_ == nullptr) {
    return InvalidArgument(
        "Evaluator cannot evaluate get_dimension_size without "
        "set_dynamic_dimension_inference.");
  }
  HloInstruction* dynamic_size =
      dynamic_dimension_inference_->GetDynamicSize(operand, {}, dim);
  if (dynamic_size != nullptr) {
    evaluated_[get_dimension_size] =
        GetEvaluatedLiteralFor(dynamic_size).Clone();
    return OkStatus();
  }

  const Shape& shape = get_dimension_size->operand(0)->shape();
  Literal output(ShapeUtil::MakeShape(S32, {}));
  output.PopulateWithValue(
      static_cast<int32_t>(shape.dimensions(get_dimension_size->dimension())));
  evaluated_[get_dimension_size] = std::move(output);
  return OkStatus();
}

Status HloEvaluator::HandleSetDimensionSize(
    HloInstruction* set_dimension_size) {
  const Literal& operand_literal =
      GetEvaluatedLiteralFor(set_dimension_size->operand(0));
  Literal result(set_dimension_size->shape());
  memcpy(result.untyped_data(), operand_literal.untyped_data(),
         operand_literal.size_bytes());
  const Literal& size_literal =
      GetEvaluatedLiteralFor(set_dimension_size->operand(1));
  result.SetDynamicSize(set_dimension_size->dimension(),
                        size_literal.Get<int32_t>({}));
  evaluated_[set_dimension_size] = std::move(result);
  return OkStatus();
}

Status HloEvaluator::HandleParameter(HloInstruction* parameter) {
  if (arg_literals_.empty()) {
    if (!enable_partial_evaluation_) {
      return tensorflow::errors::FailedPrecondition(
          "Failed to evaluate instruction since its operands are unknown "
          "or undetermined and partial evaluation is not enabled.");
    }
    evaluated_[parameter] =
        Literal::CreateFromShapeWithUnknownLeafArrays(parameter->shape());
    return OkStatus();
  }

  // Nothing to do other than sanity checks. Parameters' values are stored in
  // arg_literals_.
  CHECK_LT(parameter->parameter_number(), arg_literals_.size());

#ifndef NDEBUG
  const Literal* input_literal = arg_literals_[parameter->parameter_number()];
  VLOG(2) << "Parameter evaluated to: " << input_literal->ToString();
  DCHECK(Shape::Equal().MinorToMajorOnlyInLayout()(parameter->shape(),
                                                   input_literal->shape()))
      << "parameter shape is: "
      << ShapeUtil::HumanStringWithLayout(parameter->shape())
      << ", but input literal shape is: "
      << ShapeUtil::HumanStringWithLayout(input_literal->shape());
#endif

  return OkStatus();
}

Status HloEvaluator::HandleInfeed(HloInstruction* infeed) {
  if (!enable_partial_evaluation_) {
    return tensorflow::errors::FailedPrecondition(
        "Failed to evaluate instruction since its operands are unknown "
        "or undetermined and partial evaluation is not enabled.");
  }
  evaluated_[infeed] =
      Literal::CreateFromShapeWithUnknownLeafArrays(infeed->shape());
  return OkStatus();
}

Status HloEvaluator::HandleConstant(HloInstruction*) { return OkStatus(); }

Status HloEvaluator::HandleReshape(HloInstruction* reshape) {
  TF_ASSIGN_OR_RETURN(evaluated_[reshape],
                      GetEvaluatedLiteralFor(reshape->operand(0))
                          .Reshape(reshape->shape().dimensions()));
  return OkStatus();
}

Status HloEvaluator::HandleTranspose(HloInstruction* transpose) {
  evaluated_[transpose] = GetEvaluatedLiteralFor(transpose->operand(0))
                              .Transpose(transpose->dimensions());
  return OkStatus();
}

Status HloEvaluator::HandleConcatenate(HloInstruction* concatenate) {
  absl::Span<HloInstruction* const> operands(concatenate->operands());
  // The result concatenate dimension is going to be the sum of all
  // concatenate dimensions of the operands taking part of the operation.
  const Shape& reference_shape = operands[0]->shape();
  CHECK(reference_shape.IsArray());
  const int64_t rank = reference_shape.rank();
  const int64_t concat_dim = concatenate->dimensions()[0];
  CHECK_GE(concat_dim, 0);
  CHECK_LT(concat_dim, rank);

  DimensionVector concat_dimensions(reference_shape.dimensions().begin(),
                                    reference_shape.dimensions().end());

  for (int64_t i = 1; i < operands.size(); ++i) {
    const Shape& operand_shape = operands[i]->shape();
    CHECK(operand_shape.IsArray());
    // Accumulate the concat dimension from all tensors taking part to the
    // operation.
    concat_dimensions[concat_dim] +=
        ShapeUtil::GetDimension(operand_shape, concat_dim);
  }

  auto result_literal = LiteralUtil::CreateFromDimensions(
      reference_shape.element_type(), concat_dimensions);
  DimensionVector source_indices(rank, 0);
  DimensionVector dest_indices(concat_dimensions.size(), 0);

  for (auto operand : operands) {
    const Shape& operand_shape = operand->shape();
    TF_RETURN_IF_ERROR(result_literal.CopySliceFrom(
        GetEvaluatedLiteralFor(operand), source_indices, dest_indices,
        operand_shape.dimensions()));
    dest_indices[concat_dim] +=
        ShapeUtil::GetDimension(operand_shape, concat_dim);
  }

  evaluated_[concatenate] = std::move(result_literal);
  return OkStatus();
}

Status HloEvaluator::HandleIsFinite(HloInstruction* is_finite) {
  auto operand = is_finite->operand(0);
  auto elem_ty = operand->shape().element_type();
  switch (elem_ty) {
    case PRED:
    case TUPLE:
    case OPAQUE_TYPE:
    case TOKEN:
    case S8:
    case S16:
    case S32:
    case S64:
    case U8:
    case U16:
    case U32:
    case U64:
    case C64:
    case C128:
    // Explicitly enumerate all types in this switch so that when we add a new
    // type, we'll get a compile error here.
    case PRIMITIVE_TYPE_INVALID:
    case PrimitiveType_INT_MIN_SENTINEL_DO_NOT_USE_:
    case PrimitiveType_INT_MAX_SENTINEL_DO_NOT_USE_:
      return InvalidArgument(
          "expected element type in shape to be floating point, but "
          "got: %s",
          PrimitiveType_Name(elem_ty));

    case F16: {
      auto result_or = ElementWiseUnaryOpImpl<bool, Eigen::half>(
          is_finite,
          [](Eigen::half elem_operand) {
            return std::isfinite(static_cast<float>(elem_operand));
          },
          GetEvaluatedLiteralFor(operand));
      TF_ASSIGN_OR_RETURN(evaluated_[is_finite], std::move(result_or));
      break;
    }
    case BF16: {
      auto result_or = ElementWiseUnaryOpImpl<bool, bfloat16>(
          is_finite,
          [](bfloat16 elem_operand) {
            return std::isfinite(static_cast<float>(elem_operand));
          },
          GetEvaluatedLiteralFor(operand));
      TF_ASSIGN_OR_RETURN(evaluated_[is_finite], std::move(result_or));
      break;
    }
    case F32: {
      auto result_or = ElementWiseUnaryOpImpl<bool, float>(
          is_finite,
          [](float elem_operand) { return std::isfinite(elem_operand); },
          GetEvaluatedLiteralFor(operand));
      TF_ASSIGN_OR_RETURN(evaluated_[is_finite], std::move(result_or));
      break;
    }
    case F64: {
      auto result_or = ElementWiseUnaryOpImpl<bool, double>(
          is_finite,
          [](double elem_operand) { return std::isfinite(elem_operand); },
          GetEvaluatedLiteralFor(operand));
      TF_ASSIGN_OR_RETURN(evaluated_[is_finite], std::move(result_or));
      break;
    }
  }

  return OkStatus();
}

Status HloEvaluator::HandleReal(HloInstruction* real) {
  auto operand = real->operand(0);
  switch (operand->shape().element_type()) {
    case BF16: {
      auto result_or = ElementWiseUnaryOpImpl<bfloat16, bfloat16>(
          real, [](bfloat16 elem_operand) { return elem_operand; },
          GetEvaluatedLiteralFor(operand));
      TF_ASSIGN_OR_RETURN(evaluated_[real], std::move(result_or));
      break;
    }
    case C64: {
      auto result_or = ElementWiseUnaryOpImpl<float, complex64>(
          real, [](complex64 elem_operand) { return std::real(elem_operand); },
          GetEvaluatedLiteralFor(operand));
      TF_ASSIGN_OR_RETURN(evaluated_[real], std::move(result_or));
      break;
    }
    case C128: {
      auto result_or = ElementWiseUnaryOpImpl<double, complex128>(
          real, [](complex128 elem_operand) { return std::real(elem_operand); },
          GetEvaluatedLiteralFor(operand));
      TF_ASSIGN_OR_RETURN(evaluated_[real], std::move(result_or));
      break;
    }
    case F16: {
      auto result_or = ElementWiseUnaryOpImpl<Eigen::half, Eigen::half>(
          real, [](Eigen::half elem_operand) { return elem_operand; },
          GetEvaluatedLiteralFor(operand));
      TF_ASSIGN_OR_RETURN(evaluated_[real], std::move(result_or));
      break;
    }
    case F32: {
      auto result_or = ElementWiseUnaryOpImpl<float, float>(
          real, [](float elem_operand) { return elem_operand; },
          GetEvaluatedLiteralFor(operand));
      TF_ASSIGN_OR_RETURN(evaluated_[real], std::move(result_or));
      break;
    }
    case F64: {
      auto result_or = ElementWiseUnaryOpImpl<double, double>(
          real, [](double elem_operand) { return elem_operand; },
          GetEvaluatedLiteralFor(operand));
      TF_ASSIGN_OR_RETURN(evaluated_[real], std::move(result_or));
      break;
    }
    default:
      LOG(FATAL) << "HandleReal: unknown/unhandled primitive type: "
                 << PrimitiveType_Name(operand->shape().element_type());
  }

  return OkStatus();
}

Status HloEvaluator::HandleImag(HloInstruction* imag) {
  auto operand = imag->operand(0);
  switch (operand->shape().element_type()) {
    case BF16: {
      auto result_or = ElementWiseUnaryOpImpl<bfloat16, bfloat16>(
          imag, [](bfloat16 elem_operand) { return bfloat16(0); },
          GetEvaluatedLiteralFor(imag->operand(0)));

      TF_ASSIGN_OR_RETURN(evaluated_[imag], std::move(result_or));
      break;
    }
    case C64: {
      auto result_or = ElementWiseUnaryOpImpl<float, complex64>(
          imag, [](complex64 elem_operand) { return std::imag(elem_operand); },
          GetEvaluatedLiteralFor(imag->operand(0)));

      TF_ASSIGN_OR_RETURN(evaluated_[imag], std::move(result_or));
      break;
    }
    case C128: {
      auto result_or = ElementWiseUnaryOpImpl<double, complex128>(
          imag, [](complex128 elem_operand) { return std::imag(elem_operand); },
          GetEvaluatedLiteralFor(imag->operand(0)));

      TF_ASSIGN_OR_RETURN(evaluated_[imag], std::move(result_or));
      break;
    }
    case F16: {
      auto result_or = ElementWiseUnaryOpImpl<Eigen::half, Eigen::half>(
          imag, [](Eigen::half elem_operand) { return Eigen::half(0); },
          GetEvaluatedLiteralFor(imag->operand(0)));

      TF_ASSIGN_OR_RETURN(evaluated_[imag], std::move(result_or));
      break;
    }
    case F32: {
      auto result_or = ElementWiseUnaryOpImpl<float, float>(
          imag, [](float elem_operand) { return 0; },
          GetEvaluatedLiteralFor(imag->operand(0)));

      TF_ASSIGN_OR_RETURN(evaluated_[imag], std::move(result_or));
      break;
    }
    case F64: {
      auto result_or = ElementWiseUnaryOpImpl<double, double>(
          imag, [](double elem_operand) { return 0; },
          GetEvaluatedLiteralFor(imag->operand(0)));

      TF_ASSIGN_OR_RETURN(evaluated_[imag], std::move(result_or));
      break;
    }
    default:
      LOG(FATAL) << "HandleImag: unknown/unhandled primitive type: "
                 << PrimitiveType_Name(operand->shape().element_type());
  }

  return OkStatus();
}

Status HloEvaluator::HandleComplex(HloInstruction* complex) {
  const Literal& real = GetEvaluatedLiteralFor(complex->operand(0));
  const Literal& imag = GetEvaluatedLiteralFor(complex->operand(1));
  TF_RET_CHECK(ShapeUtil::Compatible(real.shape(), imag.shape()));

  Literal result(complex->shape());
  switch (complex->shape().element_type()) {
    case C64: {
      TF_RETURN_IF_ERROR(result.Populate<complex64>(
          [&](absl::Span<const int64_t> multi_index) {
            return std::complex<float>(real.Get<float>(multi_index),
                                       imag.Get<float>(multi_index));
          }));
      break;
    }
    case C128: {
      TF_RETURN_IF_ERROR(result.Populate<complex128>(
          [&](absl::Span<const int64_t> multi_index) {
            return std::complex<double>(real.Get<double>(multi_index),
                                        imag.Get<double>(multi_index));
          }));
      break;
    }
    default:
      LOG(FATAL) << "HandleComplex: unknown/unhandled primitive type: "
                 << PrimitiveType_Name(complex->shape().element_type());
  }

  evaluated_[complex] = std::move(result);
  return OkStatus();
}

Status HloEvaluator::HandleCompare(HloInstruction* compare) {
  ComparisonDirection direction = compare->comparison_direction();
  auto lhs = compare->operand(0);
  auto rhs = compare->operand(1);
  DCHECK(ShapeUtil::SameDimensions(compare->shape(), rhs->shape()) &&
         ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

  TF_RET_CHECK(lhs->shape().element_type() == rhs->shape().element_type());

  const Literal& lhs_literal = GetEvaluatedLiteralFor(lhs);
  const Literal& rhs_literal = GetEvaluatedLiteralFor(rhs);

  // Note here we switch on the operand's type.
  switch (lhs->shape().element_type()) {
    case PRED: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<bool>(compare->shape(), direction, lhs_literal, rhs_literal));
    } break;
    case U8: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<uint8_t>(compare->shape(), direction,
                                           lhs_literal, rhs_literal));
    } break;
    case U16: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<uint16_t>(compare->shape(), direction,
                                            lhs_literal, rhs_literal));
    } break;
    case U32: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<uint32_t>(compare->shape(), direction,
                                            lhs_literal, rhs_literal));
    } break;
    case U64: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<uint64_t>(compare->shape(), direction,
                                            lhs_literal, rhs_literal));
    } break;
    case S8: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<int8_t>(compare->shape(), direction,
                                          lhs_literal, rhs_literal));
    } break;
    case S16: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<int16_t>(compare->shape(), direction,
                                           lhs_literal, rhs_literal));
    } break;
    case S32: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<int32_t>(compare->shape(), direction,
                                           lhs_literal, rhs_literal));
    } break;
    case S64: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<int64_t>(compare->shape(), direction,
                                           lhs_literal, rhs_literal));
    } break;
    case F16: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<half>(compare->shape(), direction, lhs_literal, rhs_literal));
    } break;
    case BF16: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<bfloat16>(compare->shape(), direction,
                                            lhs_literal, rhs_literal));
    } break;
    case F32: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<float>(compare->shape(), direction,
                                         lhs_literal, rhs_literal));
    } break;
    case F64: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<double>(compare->shape(), direction,
                                          lhs_literal, rhs_literal));
    } break;
    case C64: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<complex64>(compare->shape(), direction,
                                             lhs_literal, rhs_literal));
    } break;
    case C128: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<complex128>(compare->shape(), direction,
                                              lhs_literal, rhs_literal));
    } break;
    default:
      LOG(FATAL) << "HandleCompare: unknown primitive type: "
                 << PrimitiveType_Name(lhs->shape().element_type());
  }

  return OkStatus();
}

Status HloEvaluator::HandleTuple(HloInstruction* tuple) {
  std::vector<const Literal*> operand_literals;
  std::vector<Literal> operand_literal_values;
  if (!visitor_shape_index_.empty()) {
    // We only need to evaluate tuple at visitor_shape_index_. The other
    // operands might not have been evaluated, so mark the other operands as
    // undetermined.
    int64_t tuple_index = visitor_shape_index_.front();
    operand_literal_values.resize(tuple->operand_count());
    for (int operand_index = 0; operand_index < tuple->operand_count();
         ++operand_index) {
      if (operand_index == tuple_index) {
        operand_literals.push_back(
            &GetEvaluatedLiteralFor(tuple->mutable_operand(operand_index)));
      } else {
        operand_literal_values[operand_index] =
            Literal::CreateFromShapeWithUndeterminedLeafArrays(
                ShapeUtil::GetSubshape(tuple->shape(), {operand_index}));
        operand_literals.push_back(&operand_literal_values[operand_index]);
      }
    }
  } else {
    for (auto operand : tuple->operands()) {
      operand_literals.push_back(&GetEvaluatedLiteralFor(operand));
    }
  }

  if (evaluated_.contains(tuple)) {
    Literal new_result = LiteralUtil::MakeTuple(operand_literals);
    CHECK(new_result.IsDetermined(visitor_shape_index_));
    TF_RETURN_IF_ERROR(
        evaluated_[tuple].CopyFrom(new_result,
                                   /*dest_shape_index=*/visitor_shape_index_,
                                   /*src_shape_index=*/visitor_shape_index_));
  } else {
    evaluated_[tuple] = LiteralUtil::MakeTuple(operand_literals);
  }
  return OkStatus();
}

namespace {

// These helper templates convert the data type and are intended to be used only
// within the DFT implementation below. The special case is IRFFT, where the
// specialization drops imaginary parts of complex values and returns real
// numbers.
template <typename ToType, typename FromType>
struct TypeConverter {
  static inline ToType GetAs(FromType value) {
    return static_cast<ToType>(value);
  }
};

template <typename FromType>
struct TypeConverter<float, FromType> {
  static inline float GetAs(FromType value) {
    return static_cast<float>(value.real());
  }
};

// This class implements the discrete Fourier transform. All transform types
// (FFT, IFFT, RFFT, and IRFFT) are supported, as well as the arbitrary rank and
// length of each dimension of the transform, and arbitrary layouts of the input
// and output literals. The class template parameter must be a complex type, and
// all internal calculations will be performed using this type.
//
// The input literal provides input data, which must be complex64 for FFT, IFFT,
// IRFFT transforms and float for RFFT. The transform is computed over the
// innermost dimensions of the input, thus the rank of the input data must be
// same as fft_rank or larger. The input is expected to provide Ni values along
// each transform axis with one exception: for IRFFT, only (N0 / 2) + 1 values
// are needed along the X axis (the innermost index). To increase flexibility,
// this implementation can handle mismatches between the input size and
// transform lengths by either dropping extra input values or using zeroes in
// place of missing input values as necessary. If the input data has rank higher
// than the transform, the transform is applied for each valid combination of
// the higher-ranking indices.
//
// The output contains complex64 values for FFT, IFFT, RFFT, and float values
// for IRFFT. The rank of the output as well as the sizes of the dimensions
// above the rank of the transform must match those of the input. Sizes of the
// output's "fft_rank" innermost dimensions are expected to match the length of
// the transform along respective axes with one exception: for RFFT, the output
// is trimmed along the X axis to have only (N0 / 2) + 1 values. In case the
// length(s) mismatch, the FFT output is trimmed to fit into the provided output
// shape, or the output is padded with zero values appropriately.
//
// For example, 2D FFT transform of size 16x16 applied to complex64[2][15][17]
// input array will perform two transforms over the [][15][17] data in the sub
// arrays [0][][] and [1][][], dropping the values along axis X and padding axis
// Y with zeroes to create 16x16 working sets, and generating
// complex64[2][16][16] output. 3D IRFFT transform of size 64x16x16 applied to
// complex64[64][16][9] input array will use all input values and will produce
// float[64][16][16] output.
//
// The implementation of the 1D transform for lengths, that are powers of 2, is
// the Cooley-Tukey radix-2 decimation-in-time. For all other 1D transform
// lengths, a straightforward, but slow, loop nest is used. The transforms of
// higher ranks apply sets of 1D transforms along each axis. For example, the 2D
// transform is computed by applying 1D transforms to each column followed by
// applying 1D transforms to each row.
//
// In general, a transform of rank n runs in O(N0*N1*...*Nn*(N0+N1+...+Nn))
// time, where Ni is the length of the transform's i-th dimension. However, for
// dimension lengths, which are powers of 2, the run time along these dimensions
// is reduced to log(Ni) in the summation, giving the runtime of
// O(N0*N1*...*Nn*(log(N0)+log(N1)+...+log(Nn)) in the best case.
//
template <typename ComplexType>
class FftTransform {
 public:
  explicit FftTransform(HloInstruction* fft)
      : fft_type_(fft->fft_type()),
        fft_rank_(fft->fft_length().size()),
        fft_lengths_(fft->fft_length()) {
    // Make fft_lengths_[0] the minormost dimension.
    absl::c_reverse(fft_lengths_);
  }

  Status ComputeFft(HloInstruction* fft, const Literal& input_literal,
                    Literal* output_literal) {
    const Shape& input_shape = input_literal.shape();
    const Shape& output_shape = fft->shape();

    TF_RETURN_IF_ERROR(CheckParameters(input_shape, output_shape));

    const auto fft_strides = ComputeStrides(fft_lengths_);

    // Working set size.
    const int64_t fft_size = fft_strides[fft_rank_];

    if (fft_size > 0) {
      // Linearized working data set.
      std::vector<ComplexType> data(fft_size);

      // Temporary buffer allocated once and used in 1D sweeps. For dimension
      // length values that are powers of 2, the buffer should be twice as large
      // to simultaneously hold input and output in Fft1D() above.
      int64_t buffer_size = 0;
      for (auto len : fft_lengths_) {
        int64_t size =
            absl::has_single_bit(static_cast<uint64_t>(len)) ? len * 2 : len;
        buffer_size = std::max(buffer_size, size);
      }
      std::vector<ComplexType> buffer(buffer_size);

      // Sizes of each axis of input and output literals.
      const auto input_lengths = GetDimensionLengths(input_literal);
      const auto output_lengths = GetDimensionLengths(*output_literal);

      // Strides for generating linearized indices into multidimensional arrays.
      const auto input_strides = ComputeStrides(input_lengths, input_literal);
      const auto output_strides =
          ComputeStrides(output_lengths, *output_literal);

      // Visit all elements in the dimensions with ranks above the FFT rank. For
      // each such element invoke the transform. Use separate indices for the
      // input and the output to allow different layouts.
      auto base_case = [&](int64_t axis, int64_t output_index,
                           int64_t input_index, bool within_src_bounds) {
        if (axis == fft_rank_ - 1) {
          // Base case: copy the data from the input literal, apply the
          // transform, and copy the result to the output literal.
          CHECK(within_src_bounds);
          bool input_is_zero = CopyDataFromInput(
              input_literal, input_index, fft_size, fft_lengths_, fft_strides,
              input_lengths, input_strides, absl::MakeSpan(data));
          if (!input_is_zero) {
            // Make 1D sweeps along each transform axis.
            Sweep(fft_lengths_, fft_strides, absl::MakeSpan(data),
                  absl::MakeSpan(buffer));
          }
          CopyDataToOutput(absl::MakeSpan(data), output_index, fft_lengths_,
                           fft_strides, output_lengths, output_strides,
                           output_literal);
          return true;
        }
        return false;
      };
      GenerateIndices(output_lengths, output_strides, input_lengths,
                      input_strides, input_shape.rank(), 0, 0, base_case);
    }

    return OkStatus();
  }

 private:
  // Common code used by 1D implementations, which copies data from the input to
  // the contiguous buffer. Returns true if all copied values are zero.
  static bool GatherToBuffer(absl::Span<ComplexType> data, int64_t length,
                             int64_t start, int64_t stride, bool expand_input,
                             absl::Span<ComplexType> buffer) {
    CHECK_GE(buffer.size(), length);
    bool input_is_zero = true;
    const int64_t ub = expand_input ? length / 2 + 1 : length;
    CHECK_GE(data.size(), start + (ub - 1) * stride);
    for (int64_t k = 0; k < ub; k++) {
      ComplexType value = data[start + k * stride];
      input_is_zero &= value == ComplexType(0.0, 0.0);
      buffer[k] = value;
      if (expand_input) {
        // Use conjugates of the values at indices [1 ... (ub - 2)] when the
        // length is even and at indices [1 ... (ub - 1)] when the length is odd
        // to calculate missing values at indices [(length - 1) ... ub].
        if (k > 0 && k < (length - ub + 1)) {
          buffer[length - k] = std::conj(value);
        }
      }
    }
    return input_is_zero;
  }

  // Returns (conjugated, if 'inverse' is true) k-th twiddle for the given
  // length.
  static inline ComplexType Twiddle(int64_t k, int64_t length, bool inverse) {
    auto coeff = std::exp(ComplexType(0.0, -2.0 * M_PI * k / length));
    return inverse ? std::conj(coeff) : coeff;
  }

  // Straightforward implementation of 1D DFT transform of arbitrary length.
  // Uses passed-in start index and stride to gather inputs from the data vector
  // into the preallocated buffer, computes the result, and writes it back to
  // the same locations in the data vector. Runs in O(length^2) time.
  //
  // Parameters contract_output and expand_input are used to avoid unnecessary
  // calculations. When contract_output is set to true, then only (length / 2) +
  // 1 output values are computed. When expand_input is set to true, then
  // (length / 2) + 1 values from the data set are used to re-create the full
  // set of size 'length', on which the transform is then performed.
  //
  static void NaiveDft1D(int64_t length, int64_t start, int64_t stride,
                         bool inverse, bool contract_output, bool expand_input,
                         absl::Span<ComplexType> data,
                         absl::Span<ComplexType> buffer) {
    const bool input_is_zero =
        GatherToBuffer(data, length, start, stride, expand_input, buffer);

    if (!input_is_zero) {
      const int64_t ub = contract_output ? length / 2 + 1 : length;
      for (int64_t k = 0; k < ub; k++) {
        ComplexType value = ComplexType(0.0, 0.0);
        for (int n = 0; n < length; n++) {
          value += buffer[n] * Twiddle(n * k, length, inverse);
        }
        data[start + k * stride] =
            inverse ? value / ComplexType(length, 0.0) : value;
      }
    }
  }

  // Non-recursive implementation of the Cooley-Tukey radix-2 decimation in
  // time. Performs 1D FFT transform for the lengths, which are powers of 2.
  // Runs in O(length * log(length)) time. Uses the same parameters as the naive
  // implementation above, except that the preallocated buffer must be at least
  // twice as big as the length of the transform, because the buffer is used to
  // hold both input and output values for each stage of the transform.
  //
  static void Fft1D(int64_t length, int64_t start, int64_t stride, bool inverse,
                    bool contract_output, bool expand_input,
                    absl::Span<ComplexType> data,
                    absl::Span<ComplexType> buffer) {
    CHECK(absl::has_single_bit(static_cast<uint64_t>(length)));
    const bool input_is_zero =
        GatherToBuffer(data, length, start, stride, expand_input, buffer);

    if (!input_is_zero) {
      auto generate_twiddles = [](int64_t length, bool inverse) {
        std::vector<ComplexType> twiddles;
        // Need only half the twiddles.
        for (int64_t k = 0; k < length / 2; k++) {
          twiddles.push_back(Twiddle(k, length, inverse));
        }
        return twiddles;
      };

      // Indices into the parts of the buffer used for input and output values.
      int64_t in_base = length;
      int64_t out_base = 0;

      // At each stage, we "split" the input data into num_blocks, with
      // block_size values in each block.
      for (int64_t num_blocks = 1; num_blocks < length; num_blocks *= 2) {
        // Swap input and output parts of the buffer.
        std::swap(in_base, out_base);
        auto twiddles = generate_twiddles(num_blocks * 2, inverse);
        const int64_t block_size = length / num_blocks;
        const int64_t next_iteration_block_size = block_size / 2;
        for (int64_t block = 0; block < num_blocks; block++) {
          const int64_t in_offset = in_base + block * block_size;
          const int64_t out_offset =
              out_base + block * next_iteration_block_size;
          // For each (even, odd) pair of values in the block, calculate two
          // output values as even + twiddle * odd and even - twiddle * odd.
          for (int64_t pair = 0; pair < block_size / 2; pair++) {
            const ComplexType even = buffer[in_offset + pair];
            const ComplexType odd = buffer[in_offset + block_size / 2 + pair];
            const ComplexType twiddled_odd = twiddles[block] * odd;
            buffer[out_offset + pair] = even + twiddled_odd;
            buffer[out_offset + length / 2 + pair] = even - twiddled_odd;
          }
        }
      }
      // Copy computed result back to data.
      const int64_t ub = contract_output ? length / 2 + 1 : length;
      for (int64_t k = 0; k < ub; k++) {
        ComplexType value = buffer[out_base + k];
        data[start + k * stride] =
            inverse ? value / ComplexType(length, 0.0) : value;
      }
    }
  }

  // Determine, which implementation of 1D transform to use and call it.
  static void Dft1D(int64_t length, int64_t start, int64_t stride, bool inverse,
                    bool contract_output, bool expand_input,
                    absl::Span<ComplexType> data,
                    absl::Span<ComplexType> buffer) {
    if (absl::has_single_bit(static_cast<uint64_t>(length))) {
      Fft1D(length, start, stride, inverse, contract_output, expand_input, data,
            buffer);
    } else {
      NaiveDft1D(length, start, stride, inverse, contract_output, expand_input,
                 data, buffer);
    }
  }

  // Helper to reverse the order of dimension lengths in the passed-in literal.
  static std::vector<int64_t> GetDimensionLengths(const Literal& literal) {
    auto dimensions = literal.shape().dimensions();
    return std::vector<int64_t>(dimensions.rbegin(), dimensions.rend());
  }

  // Helper to compute strides for creating linear indices into multidimensional
  // data from the dimension lengths and the layout. Returns a new vector of
  // size lengths.size() + 1. The last element of the returned vector at index
  // [lengths.size()] contains the product of all dimension lengths.
  static std::vector<int64_t> ComputeStrides(
      const absl::Span<const int64_t> lengths, const Layout& layout) {
    const int64_t num_dimensions = lengths.size();

    // Make sure that the layout length matches the number of dimensions.
    CHECK_EQ(num_dimensions, layout.minor_to_major_size());

    // Calculate strides using layout-specified ordering of the dimensions and
    // place the stride for axis 0 at index 0, for axis 1 at index 1, etc.
    std::vector<int64_t> strides(num_dimensions + 1);
    int64_t stride = 1;
    for (int64_t i = 0; i < num_dimensions; i++) {
      // Reverse the ordering of the dimensions in the layout.
      const int64_t index = (num_dimensions - 1) - layout.minor_to_major(i);
      strides[index] = stride;
      stride *= lengths[index];
    }
    strides[num_dimensions] = stride;

    return strides;
  }

  // Compute strides as above using the default layout.
  static std::vector<int64_t> ComputeStrides(
      const absl::Span<const int64_t> lengths) {
    return ComputeStrides(lengths,
                          LayoutUtil::GetDefaultLayoutForRank(lengths.size()));
  }

  // Compute strides as above using the layout from the literal, if available.
  static std::vector<int64_t> ComputeStrides(
      const absl::Span<const int64_t> lengths, const Literal& literal) {
    return literal.shape().has_layout()
               ? ComputeStrides(lengths, literal.shape().layout())
               : ComputeStrides(lengths);
  }

  // Make 1D sweeps along each transform axis.
  void Sweep(const absl::Span<const int64_t> fft_lengths,
             const absl::Span<const int64_t> fft_strides,
             absl::Span<ComplexType> data, absl::Span<ComplexType> buffer) {
    const bool inverse =
        fft_type_ == FftType::IFFT || fft_type_ == FftType::IRFFT;
    const bool input_is_truncated = fft_type_ == FftType::IRFFT;
    const bool output_is_truncated = fft_type_ == FftType::RFFT;

    // Recursively visit each column of the data along the sweep_axis. Calculate
    // linearized index of that column's first element and the stride, then
    // invoke 1D transform. For RFFT, avoid calculating unused output values:
    // first, compute only (length_x / 2) + 1 values along the X axis, then
    // limit the X coordinate to [0 ... (length / 2)] during the sweeps along
    // other axes. Similarly, for IRFFT sweep along higher dimensions first,
    // while keeping the X coordinate in the [0 ... (length / 2)] range, then
    // re-create negative frequencies omitted in the input and perform the
    // full-length transform along the X axis in the last sweep.
    std::function<void(int64_t, int64_t, int64_t)> sweep =
        [&](int64_t sweep_axis, int64_t axis, int64_t start) {
          if (axis < 0) {
            // Base case: invoke 1D transform.
            const int64_t length = fft_lengths[sweep_axis];
            const int64_t stride = fft_strides[sweep_axis];
            const bool expand_input = input_is_truncated && sweep_axis == 0;
            const bool contract_oputput =
                output_is_truncated && sweep_axis == 0;
            Dft1D(length, start, stride, inverse, contract_oputput,
                  expand_input, data, buffer);
          } else if (axis == sweep_axis) {
            // Visit only the elements with coordinate 0 along the sweep axis.
            sweep(sweep_axis, axis - 1, start);
          } else {
            const int64_t length = fft_lengths[axis];
            const bool is_truncated = input_is_truncated || output_is_truncated;
            const int64_t ub =
                is_truncated && axis == 0 ? (length / 2) + 1 : length;
            for (int64_t i = 0; i < ub; i++) {
              sweep(sweep_axis, axis - 1, start + i * fft_strides[axis]);
            }
          }
        };
    if (input_is_truncated) {
      // Sweep along the X axis last for IRFFT.
      for (int64_t sweep_axis = fft_rank_ - 1; sweep_axis >= 0; sweep_axis--) {
        sweep(sweep_axis, fft_rank_ - 1, 0);
      }
    } else {
      // Sweep along the X axis first for RFFT. The order does not matter for
      // FFT and IFFT types; handle them here as well.
      for (int64_t sweep_axis = 0; sweep_axis < fft_rank_; sweep_axis++) {
        sweep(sweep_axis, fft_rank_ - 1, 0);
      }
    }
  }

  // This template generates two linearized indices, which can be used to access
  // multidimensional arrays. It uses a recursive function, which passes the
  // indices to the user-supplied callback function. The destination index is
  // always within dst_lengths[] bounds. The boolean parameter within_src_bounds
  // indicates whether the source index is within src_lengths[] bounds.
  //
  // The value returned from the callback function controls the recursion depth.
  // Returning true indicates that the base case had been hit and the recursion
  // stops. Otherwise, the recursion proceeds along the next less-major axis.
  //
  // For example, the base case when the axis value becomes negative invokes the
  // callback function for each possible index within dst_lengths[] bounds. The
  // base case when the axis value is equal to zero limits the indices to point
  // only to first elements along the minor-most dimension, allowing the
  // callback function to handle all values along the X axis.
  //
  template <typename BaseFn>
  static void GenerateIndices(const absl::Span<const int64_t> dst_lengths,
                              const absl::Span<const int64_t> dst_strides,
                              const absl::Span<const int64_t> src_lengths,
                              const absl::Span<const int64_t> src_strides,
                              int64_t rank, int64_t dst_start,
                              int64_t src_start, BaseFn&& base) {
    CHECK_EQ(dst_lengths.size() + 1, dst_strides.size());
    CHECK_GE(dst_lengths.size(), rank);
    CHECK_EQ(src_lengths.size() + 1, src_strides.size());
    CHECK_GE(src_lengths.size(), rank);

    std::function<void(int64_t, int64_t, int64_t, bool)> generate =
        [&](int64_t axis, int64_t dst_index, int64_t src_index,
            bool within_src_bounds) {
          if (!base(axis, dst_index, src_index, within_src_bounds)) {
            for (int64_t i = 0; i < dst_lengths[axis]; i++) {
              // Because the loop goes over dst_lengths[], the source index may
              // be out of src_lengths[] bounds. In this case, within_src_bounds
              // is false.
              within_src_bounds &= i < src_lengths[axis];
              generate(axis - 1, dst_index, src_index, within_src_bounds);
              dst_index += dst_strides[axis];
              src_index += src_strides[axis];
            }
          }
        };
    generate(rank - 1, dst_start, src_start, true);
  }

  // Copies the input data from a literal to a pre-allocated vector. The sizes
  // of the input and the transform do not need to match. For each axis of the
  // transform, any extra input values beyond the transform length are ignored.
  // Conversely, if the input does not contain enough elements along any axis,
  // the data is padded with zeroes.
  //
  // For IRFFT transforms, we use (length_x / 2) + 1 elements from the input,
  // where length_x is the size of the full transform along the X axis.
  //
  // The input literal may have a rank higher than the rank of the transform.
  // Passed-in input_index value points to the first element of the input
  // literal to be copied.
  //
  // Returns true if all values in the work data set are zeroes.
  //
  template <typename InputType>
  bool CopyDataFromInput(const Literal& input_literal, int64_t input_start,
                         int64_t fft_size,
                         const absl::Span<const int64_t> fft_lengths,
                         const absl::Span<const int64_t> fft_strides,
                         const absl::Span<const int64_t> input_lengths,
                         const absl::Span<const int64_t> input_strides,
                         absl::Span<ComplexType> data) {
    CHECK_GE(data.size(), fft_size);

    const bool input_is_truncated = fft_type_ == FftType::IRFFT;

    // Recursively visit each transform dimension to copy input values to the
    // working data set. The base case handles inputs along the X axis.
    bool input_is_zero = true;
    const InputType* input_data = input_literal.data<InputType>().data();
    auto base_case = [&](int64_t axis, int64_t dst_index, int64_t src_index,
                         bool within_src_bounds) {
      if (axis == 0) {
        // For IRFFT, the negative frequencies are only needed for the sweep
        // along the X axis, which is performed last. Leave this part of the
        // working set uninitialized until then.
        const int64_t length = fft_lengths[axis];
        const int64_t ub = input_is_truncated ? (length / 2) + 1 : length;
        for (int64_t i = 0; i < ub; i++) {
          ComplexType value = ComplexType(0);
          // Read input value only if the index is within bounds.
          if (within_src_bounds && i < input_lengths[axis]) {
            value = TypeConverter<ComplexType, InputType>::GetAs(
                input_data[src_index + i * input_strides[axis]]);
            input_is_zero &= value == ComplexType(0.0, 0.0);
          }
          data[dst_index + i * fft_strides[axis]] = value;
        }
        return true;
      }
      return false;
    };
    GenerateIndices(fft_lengths, fft_strides, input_lengths, input_strides,
                    fft_rank_, 0, input_start, base_case);
    return input_is_zero;
  }

  // Copies the result of the transform to the literal output. The sizes of the
  // transform and output must match.
  //
  // For RFFT transforms, we copy (length_x / 2) + 1 elements, where length_x is
  // the size of the full transform along the X axis (the most minor dimension).
  //
  // The output literal may have a rank higher than the rank of the transform.
  // Passed-in output_index value points to the first element of the output
  // literal to be filled in.
  //
  template <typename OutputType>
  void CopyDataToOutput(const absl::Span<ComplexType> data,
                        int64_t output_start,
                        const absl::Span<const int64_t> fft_lengths,
                        const absl::Span<const int64_t> fft_strides,
                        const absl::Span<const int64_t> output_lengths,
                        const absl::Span<const int64_t> output_strides,
                        Literal* output_literal) {
    const bool output_is_truncated = fft_type_ == FftType::RFFT;

    // Base case for recursive copy of the results to the output. The code
    // avoids making a recursive call for each output element by handling axis 0
    // in the loop (as opposed to making "axis < 0" to be the base case).
    OutputType* output_data = output_literal->data<OutputType>().data();
    auto base_case = [&](int64_t axis, int64_t dst_index, int64_t src_index,
                         bool within_src_bounds) {
      if (axis == 0) {
        // Drop negative frequencies for RFFT.
        const int64_t length = fft_lengths[axis];
        const int64_t ub = output_is_truncated ? (length / 2) + 1 : length;
        for (int64_t i = 0; i < output_lengths[axis]; i++) {
          OutputType value = OutputType(0);
          // Read data only if the index is within bounds.
          if (within_src_bounds && i < ub) {
            value = TypeConverter<OutputType, ComplexType>::GetAs(
                data[src_index + i * fft_strides[axis]]);
          }
          output_data[dst_index + i * output_strides[axis]] = value;
        }
        return true;
      }
      return false;
    };
    GenerateIndices(output_lengths, output_strides, fft_lengths, fft_strides,
                    fft_rank_, output_start, 0, base_case);
  }

  // Determine the type to use with the CopyDataFromInput<> template above.
  bool CopyDataFromInput(const Literal& input_literal, int64_t input_start,
                         int64_t fft_size,
                         const absl::Span<const int64_t> fft_lengths,
                         const absl::Span<const int64_t> fft_strides,
                         const absl::Span<const int64_t> input_lengths,
                         const absl::Span<const int64_t> input_strides,
                         absl::Span<ComplexType> data) {
    const bool input_is_float = fft_type_ == FftType::RFFT;
    if (input_is_float) {
      return CopyDataFromInput<float>(input_literal, input_start, fft_size,
                                      fft_lengths, fft_strides, input_lengths,
                                      input_strides, data);
    } else {
      return CopyDataFromInput<complex64>(input_literal, input_start, fft_size,
                                          fft_lengths, fft_strides,
                                          input_lengths, input_strides, data);
    }
  }

  // Determine the type to use with the CopyDataToOutput<> template above.
  void CopyDataToOutput(const absl::Span<ComplexType> data,
                        int64_t output_start,
                        const absl::Span<const int64_t> fft_lengths,
                        const absl::Span<const int64_t> fft_strides,
                        const absl::Span<const int64_t> output_lengths,
                        const absl::Span<const int64_t> output_strides,
                        Literal* output_literal) {
    const bool output_is_float = fft_type_ == FftType::IRFFT;
    if (output_is_float) {
      CopyDataToOutput<float>(data, output_start, fft_lengths, fft_strides,
                              output_lengths, output_strides, output_literal);
    } else {
      CopyDataToOutput<complex64>(data, output_start, fft_lengths, fft_strides,
                                  output_lengths, output_strides,
                                  output_literal);
    }
  }

  Status CheckParameters(const Shape& input_shape, const Shape& output_shape) {
    // Check FFT parameters.
    if (fft_rank_ <= 0) {
      return InvalidArgument("Zero or negative FFT rank.");
    }
    if (*absl::c_min_element(fft_lengths_) < 0) {
      return InvalidArgument("Negative FFT length.");
    }

    // Check input-related values.
    TF_CHECK_OK(ShapeUtil::ValidateShape(input_shape));
    if (!input_shape.IsArray()) {
      return Unimplemented("Only array input shapes are supported.");
    }
    auto input_elt_type = input_shape.element_type();
    if (fft_type_ == FftType::RFFT && input_elt_type != PrimitiveType::F32) {
      return InvalidArgument("Invalid input type: %d, must be %d (float).",
                             input_elt_type, PrimitiveType::F32);
    }
    if (fft_type_ != FftType::RFFT && input_elt_type != PrimitiveType::C64) {
      return InvalidArgument("Invalid input type: %d, must be %d (complex64).",
                             input_elt_type, PrimitiveType::C64);
    }
    const int64_t input_rank = input_shape.rank();
    if (input_rank < fft_rank_) {
      return InvalidArgument("Input shape rank is smaller than FFT rank.");
    }

    // Check output-related values.
    TF_CHECK_OK(ShapeUtil::ValidateShape(output_shape));
    if (!output_shape.IsArray()) {
      return Unimplemented("Only array output shapes are supported.");
    }
    auto output_elt_type = output_shape.element_type();
    if (fft_type_ == FftType::IRFFT && output_elt_type != PrimitiveType::F32) {
      return InvalidArgument("Invalid output type: %d, must be %d (float).",
                             output_elt_type, PrimitiveType::F32);
    }
    if (fft_type_ != FftType::IRFFT && output_elt_type != PrimitiveType::C64) {
      return InvalidArgument("Invalid output type: %d, must be %d (complex64).",
                             output_elt_type, PrimitiveType::C64);
    }
    const int64_t output_rank = output_shape.rank();
    if (output_rank < fft_rank_) {
      return InvalidArgument("Output shape rank is smaller than FFT rank.");
    }

    // Consistency of input and output parameters.
    if (input_rank != output_rank) {
      return InvalidArgument(
          "Ranks of input shape and output shape do not match.");
    }
    for (int64_t dim = 0; dim < input_rank - fft_rank_; dim++) {
      if (ShapeUtil::GetDimension(input_shape, dim) !=
          ShapeUtil::GetDimension(output_shape, dim)) {
        return InvalidArgument(
            "Higher dimension lengths of input shape and output shape do not "
            "match.");
      }
    }

    return OkStatus();
  }

 private:
  const FftType fft_type_;
  const int64_t fft_rank_;
  std::vector<int64_t> fft_lengths_;
};

}  // namespace

Status HloEvaluator::HandleFft(HloInstruction* fft) {
  const Literal& input_literal = GetEvaluatedLiteralFor(fft->operand(0));
  Literal output_literal = Literal::CreateFromShape(fft->shape());

  FftTransform<complex128> transform(fft);
  TF_RETURN_IF_ERROR(transform.ComputeFft(fft, input_literal, &output_literal));
  evaluated_[fft] = std::move(output_literal);

  return OkStatus();
}

// Returns an ShapeUtil::IndexIterationSpace that iterates over the output batch
// dimensions while keeping the rest of the output dimensions clamped to 0.
ShapeUtil::IndexIterationSpace IterationSpaceForOutputBatchIndices(
    const Shape& output_shape, const GatherDimensionNumbers& dim_numbers) {
  int64_t output_rank = output_shape.dimensions_size();
  std::vector<int64_t> index_base(output_rank, 0);
  std::vector<int64_t> index_count;
  index_count.reserve(output_rank);
  for (int64_t i = 0; i < output_rank; i++) {
    bool is_output_batch_dim =
        !absl::c_binary_search(dim_numbers.offset_dims(), i);
    index_count.push_back(is_output_batch_dim ? output_shape.dimensions(i) : 1);
  }

  return {std::move(index_base), std::move(index_count),
          std::vector<int64_t>(output_rank, 1)};
}

// Return an ShapeUtil::IndexIterationSpace that iterates over the output slice
// dimensions while keeping the rest of the output dimensions clamped to 0.
ShapeUtil::IndexIterationSpace IterationSpaceForOutputOffsetIndices(
    int64_t output_rank, absl::Span<const int64_t> slice_sizes,
    const GatherDimensionNumbers& dim_numbers) {
  std::vector<int64_t> index_base(output_rank, 0);
  std::vector<int64_t> index_count(output_rank, 1);
  int64_t slice_sizes_idx = 0;
  for (int64_t i = 0; i < output_rank; i++) {
    bool is_output_window_dim =
        absl::c_binary_search(dim_numbers.offset_dims(), i);
    if (is_output_window_dim) {
      while (absl::c_binary_search(dim_numbers.collapsed_slice_dims(),
                                   slice_sizes_idx)) {
        slice_sizes_idx++;
      }
      index_count[i] = slice_sizes[slice_sizes_idx++];
    }
  }

  return {std::move(index_base), std::move(index_count),
          std::vector<int64_t>(output_rank, 1)};
}

// This functor computes the contribution of start_indices to an input index
// corresponding to an output index.  That is, given an output index I, it picks
// out the batch indices in I and uses them to look up a starting index, G, from
// the start indices tensor, and expands G into the input space according to
// start_index_map.
class OutputBatchIndexToInputIndex {
 public:
  // The constructor does some setup work that is amortized across all
  // iterations.
  explicit OutputBatchIndexToInputIndex(
      const GatherDimensionNumbers* dim_numbers, const Shape& input_shape,
      const Shape& output_shape, const Literal* start_indices)
      : dim_numbers_(*dim_numbers), start_indices_(*start_indices) {
    for (int64_t i = 0; i < output_shape.dimensions_size(); i++) {
      output_dim_is_batch_dims_.push_back(
          !absl::c_binary_search(dim_numbers_.offset_dims(), i));
    }

    for (int64_t i = 0; i < input_shape.dimensions_size(); i++) {
      int64_t index_of_input_dim_in_index_vector =
          std::distance(dim_numbers_.start_index_map().begin(),
                        absl::c_find(dim_numbers_.start_index_map(), i));
      if (index_of_input_dim_in_index_vector ==
          dim_numbers_.start_index_map_size()) {
        input_dim_value_to_index_vector_.push_back(-1);
      } else {
        input_dim_value_to_index_vector_.push_back(
            index_of_input_dim_in_index_vector);
      }
    }

    index_vector_index_.resize(start_indices_.shape().dimensions_size());
    input_index_.resize(input_shape.dimensions_size());
    int64_t index_vector_size =
        start_indices_.shape().dimensions(dim_numbers_.index_vector_dim());
    index_vector_.resize(index_vector_size);
  }

  // Returns the contribution of start_indices to the input index corresponding
  // to output_index.  See gather_inner_loop_body.
  //
  // This is conceptually  a stateless transformation from output_index to the
  // gather input index, but:
  //
  //  - Instead of allocating memory to represent the gather input index on
  //    every invocation we reuse the same storage for the result
  //    (input_index_), mutating it in place.
  //  - Instead of allocating buffers for temporary values like
  //    index_vector_index_ and index_vector on every invocation, we reuse the
  //    same storage for all invocations.
  //
  // This returns a Span into memory owned by the class.
  StatusOr<absl::Span<const int64_t>> operator()(
      absl::Span<const int64_t> output_index) {
    PropagateOutputIndexGatherDimsToIndexVectorIndex(output_index);
    TF_RETURN_IF_ERROR(FetchIndexVector());
    PropagateIndexVectorToInputIndex();
    return absl::Span<const int64_t>(input_index_);
  }

 private:
  // Propagates the batch dimensions from the output index into
  // index_vector_index_ by mutating index_vector_index_ in place.  Does not
  // update the dim_numbers.index_vector_dim() dimension -- that's the dimension
  // we iterate over in FetchIndexVector.
  void PropagateOutputIndexGatherDimsToIndexVectorIndex(
      absl::Span<const int64_t> output_index) {
    int64_t index_vector_index_i = 0;
    for (int64_t i = 0, e = output_index.size(); i < e; i++) {
      if (!output_dim_is_batch_dims_[i]) {
        continue;
      }

      if (index_vector_index_i == dim_numbers_.index_vector_dim()) {
        index_vector_index_i++;
      }

      index_vector_index_[index_vector_index_i++] = output_index[i];
    }
  }

  // Populates index_vector_ by iterating over start_indices_ according to
  // index_vector_index_.
  Status FetchIndexVector() {
    int64_t index_vector_dim = dim_numbers_.index_vector_dim();
    for (int64_t i = 0, e = index_vector_.size(); i < e; i++) {
      index_vector_index_[index_vector_dim] = i;
      auto start_index = start_indices_.GetIntegralAsS64(index_vector_index_);
      TF_RET_CHECK(start_index.has_value());
      index_vector_[i] = *start_index;
    }
    return OkStatus();
  }

  // Populates input_index_.
  void PropagateIndexVectorToInputIndex() {
    for (int64_t i = 0, e = input_index_.size(); i < e; i++) {
      if (input_dim_value_to_index_vector_[i] != -1) {
        input_index_[i] = index_vector_[input_dim_value_to_index_vector_[i]];
      }

      // If input_dim_value_to_index_vector_[i] == -1 then input_index_[i]
      // remains 0, as set by the constructor.
    }
  }

  // input_dim_value_to_index_vector_[i] tells us how to compute dimension i of
  // the input index from the index vector.  See
  // PropagateIndexVectorToInputIndex.
  std::vector<int64_t> input_dim_value_to_index_vector_;

  // output_dim_is_batch_dims_[i] is true iff the output index i is a gather
  // dimension.
  std::vector<bool> output_dim_is_batch_dims_;

  // The buffer into which we construct an index into start_indices_ to fetch
  // the index vector.
  std::vector<int64_t> index_vector_index_;

  // The index vector fetched from start_indices_.
  std::vector<int64_t> index_vector_;

  // The result computed by this functor.  operator() returns a Span into
  // this vector.
  std::vector<int64_t> input_index_;

  const GatherDimensionNumbers& dim_numbers_;
  const Literal& start_indices_;
};

// This functor computes the contribution of the offset indices in an output
// index to an input index.  That is, given an output index I it picks out the
// output offset indices in I and expands it into an index into the input shape.
class OutputOffsetIndexToInputIndex {
 public:
  // The constructor does some setup work that is amortized across all
  // iterations.
  explicit OutputOffsetIndexToInputIndex(
      const GatherDimensionNumbers& dim_numbers, const Shape& input_shape,
      const Shape& output_shape) {
    std::vector<int64_t> window_index_to_output_index;
    int64_t output_index_count = 0;
    for (int64_t i = 0; i < output_shape.dimensions_size(); i++) {
      if (absl::c_binary_search(dim_numbers.offset_dims(), i)) {
        window_index_to_output_index.push_back(output_index_count++);
      } else {
        output_index_count++;
      }
    }

    int64_t window_dim_count = 0;
    for (int64_t i = 0; i < input_shape.dimensions_size(); i++) {
      if (absl::c_binary_search(dim_numbers.collapsed_slice_dims(), i)) {
        input_dim_value_to_output_index_.push_back(-1);
      } else {
        input_dim_value_to_output_index_.push_back(
            window_index_to_output_index[window_dim_count++]);
      }
    }

    input_index_.resize(input_shape.dimensions_size());
  }

  // Returns the contribution of the window indices to the input index
  // corresponding to output_index.  See gather_inner_loop_body.
  //
  // This is conceptually a stateless transformation from output_index to the
  // window input index, but instead of allocating memory to represent the
  // gather input index on every invocation we reuse the same storage for the
  // result (input_index_), mutating it in place.
  //
  // This returns a Span into memory owned by the class.
  StatusOr<absl::Span<const int64_t>> operator()(
      absl::Span<const int64_t> output_index) {
    PropagateOutputIndexWindowDimsToInputIndex(output_index);
    return absl::Span<const int64_t>(input_index_);
  }

  // Returns for a given 'input_dim' the corresponding output dimension index,
  // or -1 if 'input_dim' is an elided window dimension.
  int64_t input_dim_value_to_output_index(int64_t input_dim) {
    return input_dim_value_to_output_index_[input_dim];
  }

 private:
  // Propagates window dimensions from the output index to input_index_ by
  // mutating input_index_ in place.
  void PropagateOutputIndexWindowDimsToInputIndex(
      absl::Span<const int64_t> output_index) {
    for (int64_t i = 0, e = input_index_.size(); i < e; i++) {
      if (input_dim_value_to_output_index_[i] != -1) {
        input_index_[i] = output_index[input_dim_value_to_output_index_[i]];
      }

      // If input_dim_value_to_index_vector_[i] == -1 then input_index_[i]
      // remains 0, as set by the constructor.
    }
  }

  // input_dim_value_to_index_vector_[i] tells us how to compute dimension i of
  // the input index from the output index. See
  // PropagateOutputIndexWindowDimsToInputIndex.
  std::vector<int64_t> input_dim_value_to_output_index_;

  // The result computed by this functor.  operator() returns a Span into
  // this vector.
  std::vector<int64_t> input_index_;
};

// Reshapes the gather indices input to have a trailing degenerate `1` dimension
// if necessary.  Hands over the ownership of the newly created literal (if
// there is one) to `reshaped_start_indices`.
static StatusOr<std::reference_wrapper<const Literal>> ReshapedGatherIndices(
    int64_t index_vector_dim, const Literal& start_indices,
    Literal* reshaped_start_indices) {
  if (start_indices.shape().dimensions_size() != index_vector_dim) {
    return std::cref(start_indices);
  }

  std::vector<int64_t> new_shape(start_indices.shape().dimensions().begin(),
                                 start_indices.shape().dimensions().end());
  new_shape.push_back(1);
  TF_ASSIGN_OR_RETURN(*reshaped_start_indices,
                      start_indices.Reshape(new_shape));
  return std::cref(*reshaped_start_indices);
}

Status HloEvaluator::HandleGather(HloInstruction* gather) {
  Literal result = Literal::CreateFromShape(gather->shape());
  const Shape& shape = gather->shape();
  const GatherDimensionNumbers& dim_numbers =
      gather->gather_dimension_numbers();
  const Literal& operand = GetEvaluatedLiteralFor(gather->operand(0));
  Literal reshaped_start_indices;
  TF_ASSIGN_OR_RETURN(
      const Literal& start_indices,
      ReshapedGatherIndices(dim_numbers.index_vector_dim(),
                            GetEvaluatedLiteralFor(gather->operand(1)),
                            &reshaped_start_indices));

  // We iterate over the gather dimensions in the output shape in an outer loop
  // nest, and iterate over the window dimensions in the output shape in an
  // inner loop nest.

  ShapeUtil::IndexIterationSpace start_indices_iteration_space =
      IterationSpaceForOutputBatchIndices(shape, dim_numbers);
  ShapeUtil::IndexIterationSpace offset_indices_iteration_space =
      IterationSpaceForOutputOffsetIndices(
          shape.dimensions_size(), gather->gather_slice_sizes(), dim_numbers);

  // Scratch buffers that hold an index in the output shape and the
  // corresponding index in the input shape.
  std::vector<int64_t> input_index(operand.shape().dimensions_size());
  std::vector<int64_t> output_index(gather->shape().dimensions_size());
  std::vector<int64_t> input_index_clamped(operand.shape().dimensions_size());

  OutputBatchIndexToInputIndex output_batch_index_to_input_index(
      &gather->gather_dimension_numbers(), /*input_shape=*/operand.shape(),
      /*output_shape=*/shape, &start_indices);
  OutputOffsetIndexToInputIndex output_offset_index_to_input_index(
      gather->gather_dimension_numbers(), /*input_shape=*/operand.shape(),
      /*output_shape=*/shape);

  const Shape& operand_shape = operand.shape();
  if (ShapeUtil::IsZeroElementArray(operand_shape)) {
    evaluated_[gather] = std::move(result);
    return OkStatus();
  }

  auto gather_inner_loop_body =
      [&](absl::Span<const int64_t> output_window_index,
          absl::Span<const int64_t> input_gather_index,
          absl::Span<const int64_t> output_gather_index) -> StatusOr<bool> {
    TF_ASSIGN_OR_RETURN(
        absl::Span<const int64_t> input_window_index,
        output_offset_index_to_input_index(output_window_index));
    for (int i = 0, e = output_index.size(); i < e; i++) {
      output_index[i] = output_gather_index[i] + output_window_index[i];
      DCHECK_LT(output_index[i], shape.dimensions(i));
    }
    for (int i = 0, e = input_gather_index.size(); i < e; i++) {
      int64_t output_dim =
          output_offset_index_to_input_index.input_dim_value_to_output_index(i);
      // If 'output_dim' is -1, it means 'i' is an elided window dim. This means
      // we set the iteration index to 0, so for the purpose of the following
      // calculations we can consider the output dimension size to be 1.
      int64_t output_dim_size =
          output_dim == -1 ? 1 : shape.dimensions(output_dim);
      // Clamp the gather index so that the gather region fits in the operand.
      // input_index_clamped[i] = clamp(input_gather_index[i], 0,
      //                                       operand_shape.dimensions(i) -
      //                                       output_dim_size);
      input_index_clamped[i] =
          std::min(operand_shape.dimensions(i) - output_dim_size,
                   std::max(int64_t{0}, input_gather_index[i]));
    }
    for (int i = 0, e = input_index.size(); i < e; i++) {
      input_index[i] = input_index_clamped[i] + input_window_index[i];
      DCHECK_GE(input_index[i], 0);
      DCHECK_LT(input_index[i], operand_shape.dimensions(i));
    }
    TF_RETURN_IF_ERROR(
        result.CopyElementFrom(operand, input_index, output_index));
    return true;
  };

  auto gather_outer_loop_body =
      [&](absl::Span<const int64_t> output_gather_index) -> StatusOr<bool> {
    TF_ASSIGN_OR_RETURN(absl::Span<const int64_t> input_gather_index,
                        output_batch_index_to_input_index(output_gather_index));
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
        shape, offset_indices_iteration_space,
        std::bind(gather_inner_loop_body, std::placeholders::_1,
                  input_gather_index, output_gather_index)));
    return true;
  };

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
      shape, start_indices_iteration_space, gather_outer_loop_body));
  evaluated_[gather] = std::move(result);
  return OkStatus();
}

namespace {
// Reshapes the scatter indices input to have a trailing degenerate `1`
// dimension if necessary.  Hands over the ownership of the newly created
// literal (if there is one) to `reshaped_indices`.
StatusOr<std::reference_wrapper<const Literal>> ReshapedScatterIndices(
    int64_t index_vector_dim, const Literal& indices,
    Literal* reshaped_indices) {
  if (indices.shape().dimensions_size() != index_vector_dim) {
    return std::cref(indices);
  }

  std::vector<int64_t> new_shape(indices.shape().dimensions().begin(),
                                 indices.shape().dimensions().end());
  new_shape.push_back(1);
  TF_ASSIGN_OR_RETURN(*reshaped_indices, indices.Reshape(new_shape));
  return std::cref(*reshaped_indices);
}

template <bool kForUpdateWindowIndices>
ShapeUtil::IndexIterationSpace GetIterationSpaceImpl(
    absl::Span<const int64_t> updates_dims,
    const ScatterDimensionNumbers& dim_numbers) {
  int64_t updates_rank = updates_dims.size();
  std::vector<int64_t> index_base(updates_rank, 0);
  std::vector<int64_t> index_count(updates_rank, 1);
  for (int64_t i = 0; i < updates_rank; i++) {
    // Use if constexpr when we can use c++17 or above.
    if (kForUpdateWindowIndices) {
      bool is_update_window_dim =
          absl::c_binary_search(dim_numbers.update_window_dims(), i);
      if (is_update_window_dim) {
        index_count[i] = updates_dims[i];
      }
    } else {
      bool is_update_scatter_dim =
          !absl::c_binary_search(dim_numbers.update_window_dims(), i);
      if (is_update_scatter_dim) {
        index_count[i] = updates_dims[i];
      }
    }
  }
  return {std::move(index_base), std::move(index_count),
          std::vector<int64_t>(updates_rank, 1)};
}

// Returns an ShapeUtil::IndexIterationSpace that iterates over the update
// scatter dimensions while keeping the rest of the update dimensions clamped
// to 0.
ShapeUtil::IndexIterationSpace IterationSpaceForUpdateScatterIndices(
    absl::Span<const int64_t> updates_dims,
    const ScatterDimensionNumbers& dim_numbers) {
  return GetIterationSpaceImpl</*kForUpdateWindowIndices=*/false>(updates_dims,
                                                                  dim_numbers);
}

// Return an ShapeUtil::IndexIterationSpace that iterates over the update
// window dimensions while keeping the rest of the update dimensions clamped
// to 0.
ShapeUtil::IndexIterationSpace IterationSpaceForUpdateWindowIndices(
    absl::Span<const int64_t> updates_dims,
    const ScatterDimensionNumbers& dim_numbers) {
  return GetIterationSpaceImpl</*kForUpdateWindowIndices=*/true>(updates_dims,
                                                                 dim_numbers);
}

// This functor computes the contribution of scatter_indices to an input index
// corresponding to an update index.  That is, given an update index I, it
// picks out the scatter indices in I and uses them to look up a scatter
// index, S, from the scatter indices tensor, and expands S into the input
// space according to scatter_dims_to_operand_dims.
//
// This is similar to the class HloEvaluator::OutputGatherIndexToInputIndex
// that does the corresponding function for Gather.
class UpdateScatterIndexToInputIndex {
 public:
  // The constructor does some setup work that is amortized across all
  // iterations.
  explicit UpdateScatterIndexToInputIndex(
      const ScatterDimensionNumbers& dim_numbers, int64_t input_rank,
      int64_t updates_rank, const Literal* scatter_indices)
      : dim_numbers_(dim_numbers), scatter_indices_(*scatter_indices) {
    for (int64_t i = 0; i < updates_rank; i++) {
      update_dim_is_scatter_dims_.push_back(
          !absl::c_binary_search(dim_numbers_.update_window_dims(), i));
    }

    for (int64_t i = 0; i < input_rank; i++) {
      int64_t index_of_input_dim_in_index_vector =
          FindIndex(dim_numbers_.scatter_dims_to_operand_dims(), i);
      if (index_of_input_dim_in_index_vector ==
          dim_numbers_.scatter_dims_to_operand_dims_size()) {
        input_dim_value_to_index_vector_.push_back(-1);
      } else {
        input_dim_value_to_index_vector_.push_back(
            index_of_input_dim_in_index_vector);
      }
    }

    index_vector_index_.resize(scatter_indices_.shape().dimensions_size());
    input_index_.resize(input_rank);
    int64_t index_vector_size =
        scatter_indices_.shape().dimensions(dim_numbers_.index_vector_dim());
    index_vector_.resize(index_vector_size);
  }

  // Returns the contribution of scatter_indices to the input index
  // corresponding to update_index.  See scatter_inner_loop_body.
  //
  // This is conceptually  a stateless transformation from update_index to the
  // scatter input index, but:
  //
  //  - Instead of allocating memory to represent the scatter input index on
  //    every invocation we reuse the same storage for the result
  //    (input_index_), mutating it in place.
  //  - Instead of allocating buffers for temporary values like
  //    index_vector_index_ and index_vector on every invocation, we reuse the
  //    same storage for all invocations.
  //
  // This returns a Span into memory owned by the class.
  StatusOr<absl::Span<const int64_t>> operator()(
      absl::Span<const int64_t> update_index) {
    PropagateUpdateIndexScatterDimsToIndexVectorIndex(update_index);
    TF_RETURN_IF_ERROR(FetchIndexVector());
    PropagateIndexVectorToInputIndex();
    return absl::Span<const int64_t>(input_index_);
  }

 private:
  // Propagates the scatter index dimensions from the update index into
  // index_vector_index_ by mutating index_vector_index_ in place.  Does not
  // update the dim_numbers.index_vector_dim() dimension -- that's the
  // dimension we iterate over in FetchIndexVector.
  void PropagateUpdateIndexScatterDimsToIndexVectorIndex(
      absl::Span<const int64_t> update_index) {
    int64_t index_vector_index_i = 0;
    for (int64_t i = 0, e = update_index.size(); i < e; i++) {
      if (!update_dim_is_scatter_dims_[i]) {
        continue;
      }

      if (index_vector_index_i == dim_numbers_.index_vector_dim()) {
        index_vector_index_i++;
      }

      index_vector_index_[index_vector_index_i++] = update_index[i];
    }
  }

  // Populates index_vector_ by iterating over scatter_indices_ according to
  // index_vector_index_.
  Status FetchIndexVector() {
    int64_t index_vector_dim = dim_numbers_.index_vector_dim();
    for (int64_t i = 0, e = index_vector_.size(); i < e; i++) {
      index_vector_index_[index_vector_dim] = i;
      index_vector_[i] =
          *scatter_indices_.GetIntegralAsS64(index_vector_index_);
    }
    return OkStatus();
  }

  // Populates input_index_.
  void PropagateIndexVectorToInputIndex() {
    for (int64_t i = 0, e = input_index_.size(); i < e; i++) {
      if (input_dim_value_to_index_vector_[i] != -1) {
        input_index_[i] = index_vector_[input_dim_value_to_index_vector_[i]];
      }

      // If input_dim_value_to_index_vector_[i] == -1 then input_index_[i]
      // remains 0, as set by the constructor.
    }
  }

  // input_dim_value_to_index_vector_[i] tells us how to compute dimension i
  // of the input index from the index vector.  See
  // PropagateIndexVectorToInputIndex.
  std::vector<int64_t> input_dim_value_to_index_vector_;

  // update_dim_is_scatter_dims_[i] is true iff the update index i is a
  // scatter dimension.
  std::vector<bool> update_dim_is_scatter_dims_;

  // The buffer into which we construct an index into scatter_indices_ to
  // fetch the index vector.
  std::vector<int64_t> index_vector_index_;

  // The index vector fetched from scatter_indices_.
  std::vector<int64_t> index_vector_;

  // The result computed by this functor.  operator() returns a Span
  // into this vector.
  std::vector<int64_t> input_index_;

  const ScatterDimensionNumbers& dim_numbers_;
  const Literal& scatter_indices_;
};

// This functor computes the contribution of the window indices in an update
// index to an input index.  That is, given an update index I it picks out the
// update window indices in I and expands it into a window index into the
// input shape.
//
// This is similar to the class HloEvaluator::OutputWindowIndexToInputIndex
// that does the corresponding function for Gather.
class UpdateWindowIndexToInputIndex {
 public:
  // The constructor does some setup work that is amortized across all
  // iterations.
  explicit UpdateWindowIndexToInputIndex(
      const ScatterDimensionNumbers& dim_numbers, int64_t input_rank,
      int64_t update_rank) {
    std::vector<int64_t> window_index_to_update_index;
    int64_t update_index_count = 0;
    for (int64_t i = 0; i < update_rank; i++) {
      if (absl::c_binary_search(dim_numbers.update_window_dims(), i)) {
        window_index_to_update_index.push_back(update_index_count++);
      } else {
        update_index_count++;
      }
    }

    int64_t window_dim_count = 0;
    for (int64_t i = 0; i < input_rank; i++) {
      if (absl::c_binary_search(dim_numbers.inserted_window_dims(), i)) {
        input_dim_value_to_update_index_.push_back(-1);
      } else {
        input_dim_value_to_update_index_.push_back(
            window_index_to_update_index[window_dim_count++]);
      }
    }

    input_index_.resize(input_rank);
  }

  // Returns the contribution of the window indices to the input index
  // corresponding to update_index.  See scatter_inner_loop_body.
  //
  // This is conceptually a stateless transformation from update_index to the
  // window input index, but instead of allocating memory to represent the
  // scatter input index on every invocation we reuse the same storage for the
  // result (input_index_), mutating it in place.
  //
  // This returns a Span into memory owned by the class.
  StatusOr<absl::Span<const int64_t>> operator()(
      absl::Span<const int64_t> update_index) {
    PropagateUpdateIndexWindowDimsToInputIndex(update_index);
    return absl::Span<const int64_t>(input_index_);
  }

  // Returns for a given 'input_dim' the corresponding update dimension index,
  // or -1 if 'input_dim' is an elided window dimension.
  int64_t input_dim_value_to_update_index(int64_t input_dim) {
    return input_dim_value_to_update_index_[input_dim];
  }

 private:
  // Propagates window dimensions from the update index to input_index_ by
  // mutating input_index_ in place.
  void PropagateUpdateIndexWindowDimsToInputIndex(
      absl::Span<const int64_t> update_index) {
    for (int64_t i = 0, e = input_index_.size(); i < e; i++) {
      if (input_dim_value_to_update_index_[i] != -1) {
        input_index_[i] = update_index[input_dim_value_to_update_index_[i]];
      }

      // If input_dim_value_to_index_vector_[i] == -1 then input_index_[i]
      // remains 0, as set by the constructor.
    }
  }

  // input_dim_value_to_index_vector_[i] tells us how to compute dimension i
  // of the input index from the update index. See
  // PropagateUpdateIndexWindowDimsToInputIndex.
  std::vector<int64_t> input_dim_value_to_update_index_;

  // The result computed by this functor.  operator() returns a Span
  // into this vector.
  std::vector<int64_t> input_index_;
};
}  // namespace

Status HloEvaluator::HandleScatter(HloInstruction* hlo) {
  auto* scatter = DynCast<HloScatterInstruction>(hlo);
  const ScatterDimensionNumbers& dim_numbers =
      scatter->scatter_dimension_numbers();
  absl::InlinedVector<const Literal*, 1> operands;
  operands.reserve(scatter->scatter_operand_count());
  for (HloInstruction* operand_inst : scatter->scatter_operands()) {
    operands.push_back(&GetEvaluatedLiteralFor(operand_inst));
  }
  Literal reshaped_scatter_indices;
  TF_ASSIGN_OR_RETURN(
      const Literal& scatter_indices,
      ReshapedScatterIndices(dim_numbers.index_vector_dim(),
                             GetEvaluatedLiteralFor(scatter->scatter_indices()),
                             &reshaped_scatter_indices));
  absl::InlinedVector<const Literal*, 1> updates;
  updates.reserve(operands.size());
  for (HloInstruction* updates_inst : scatter->scatter_updates()) {
    updates.push_back(&GetEvaluatedLiteralFor(updates_inst));
  }
  auto updates_dims = updates[0]->shape().dimensions();
  auto operand_dims = operands[0]->shape().dimensions();

  ShapeUtil::IndexIterationSpace scatter_indices_iteration_space =
      IterationSpaceForUpdateScatterIndices(updates_dims, dim_numbers);
  ShapeUtil::IndexIterationSpace window_indices_iteration_space =
      IterationSpaceForUpdateWindowIndices(updates_dims, dim_numbers);

  std::vector<int64_t> input_index(operand_dims.size());
  std::vector<int64_t> update_index(updates_dims.size());

  UpdateScatterIndexToInputIndex update_scatter_index_to_input_index(
      scatter->scatter_dimension_numbers(),
      /*input_rank=*/operand_dims.size(), updates_dims.size(),
      &scatter_indices);
  UpdateWindowIndexToInputIndex update_window_index_to_input_index(
      scatter->scatter_dimension_numbers(),
      /*input_rank=*/operand_dims.size(), updates_dims.size());

  // Initialize the result with the operand. This makes it easier to handle
  // the updates even when the indices are repeated.
  Literal result = operands.size() > 1 ? LiteralUtil::MakeTuple(operands)
                                       : operands[0]->Clone();
  auto maybe_slice = [](MutableLiteralBase& literal, int idx) {
    if (literal.shape().IsTuple()) {
      return MutableBorrowingLiteral(&literal, {idx});
    }
    DCHECK_EQ(idx, 0);
    return MutableBorrowingLiteral(&literal);
  };

  HloEvaluator embedded_evaluator;
  auto scatter_inner_loop_body =
      [&](absl::Span<const int64_t> update_window_index,
          absl::Span<const int64_t> input_scatter_index,
          absl::Span<const int64_t> update_scatter_index) -> StatusOr<bool> {
    TF_ASSIGN_OR_RETURN(
        absl::Span<const int64_t> input_window_index,
        update_window_index_to_input_index(update_window_index));
    for (int i = 0, e = update_index.size(); i < e; i++) {
      update_index[i] = update_scatter_index[i] + update_window_index[i];
      DCHECK_LT(update_index[i], updates_dims[i]);
    }
    for (int i = 0, e = input_scatter_index.size(); i < e; i++) {
      int64_t update_dim =
          update_window_index_to_input_index.input_dim_value_to_update_index(i);
      // If 'update_dim' is -1, it means 'i' is an elided window dim. This
      // means we set the iteration index to 0, so for the purpose of the
      // following calculations we can consider the update dimension size to
      // be 1.
      int64_t update_dim_size = update_dim == -1 ? 1 : updates_dims[update_dim];
      // If any part of the update region is out-of-bounds, then do not
      // perform any update on the input.
      if ((input_scatter_index[i] < 0) ||
          (input_scatter_index[i] > operand_dims[i] - update_dim_size)) {
        return true;
      }
    }
    for (int i = 0, e = input_index.size(); i < e; i++) {
      input_index[i] = input_scatter_index[i] + input_window_index[i];
    }

    absl::InlinedVector<Literal, 2> to_apply_args;
    to_apply_args.reserve(operands.size() + updates.size());
    for (int i = 0, n = operands.size(); i < n; ++i) {
      to_apply_args.push_back(
          LiteralUtil::GetScalarLiteral(maybe_slice(result, i), input_index));
    }
    for (int i = 0, n = operands.size(); i < n; ++i) {
      to_apply_args.push_back(
          LiteralUtil::GetScalarLiteral(*updates[i], update_index));
    }
    Literal updated_result =
        embedded_evaluator.Evaluate(*scatter->to_apply(), to_apply_args)
            .ConsumeValueOrDie();
    // Clear visit states so that the we can use the evaluate again on the
    // same computation.
    embedded_evaluator.ResetVisitStates();
    for (int i = 0, n = operands.size(); i < n; ++i) {
      auto result_slice = maybe_slice(result, i);
      LiteralUtil::SetScalarLiteral(result_slice, input_index,
                                    maybe_slice(updated_result, i));
    }
    return true;
  };

  auto scatter_outer_loop_body =
      [&](absl::Span<const int64_t> update_scatter_index) -> StatusOr<bool> {
    TF_ASSIGN_OR_RETURN(
        absl::Span<const int64_t> input_scatter_index,
        update_scatter_index_to_input_index(update_scatter_index));
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
        updates[0]->shape(), window_indices_iteration_space,
        [&](absl::Span<const int64_t> update_window_index) {
          return scatter_inner_loop_body(
              update_window_index, input_scatter_index, update_scatter_index);
        }));
    return true;
  };

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
      updates[0]->shape(), scatter_indices_iteration_space,
      scatter_outer_loop_body));
  evaluated_[scatter] = std::move(result);
  return OkStatus();
}

Status HloEvaluator::HandleBroadcast(HloInstruction* broadcast) {
  const Literal& operand = GetEvaluatedLiteralFor(broadcast->operand(0));
  TF_RET_CHECK(broadcast->shape().element_type() ==
               operand.shape().element_type())
      << " broadcast from a different data type is not supported";
  TF_RET_CHECK(broadcast->dimensions().size() == operand.shape().rank())
      << "broadcast dimensions is of size: " << broadcast->dimensions().size()
      << " and rank of operand_to_broadcast is: " << operand.shape().rank();
  // Checks that operand's dimensions are the same as the broadcast's
  // dimensions along the dimensions to be broadcasted.
  for (int64_t i = 0; i < broadcast->dimensions().size(); ++i) {
    auto operand_dim_size = operand.shape().dimensions(i);
    auto broadcast_dim_size =
        broadcast->shape().dimensions(broadcast->dimensions(i));
    TF_RET_CHECK(operand_dim_size == broadcast_dim_size) << absl::StreamFormat(
        "Operand dimension %d is broadcast to output dimension %d, but the "
        "sizes of these two dims do not match (%d vs %d): %s",
        i, broadcast->dimensions(i), operand_dim_size, broadcast_dim_size,
        broadcast->ToString());
  }

  TF_ASSIGN_OR_RETURN(
      evaluated_[broadcast],
      operand.Broadcast(broadcast->shape(), broadcast->dimensions()));

  return OkStatus();
}

Status HloEvaluator::HandleAfterAll(HloInstruction* after_all) {
  evaluated_[after_all] = LiteralUtil::CreateToken();
  return OkStatus();
}

Status HloEvaluator::HandleAddDependency(HloInstruction* add_dependency) {
  // AddDedendency just forwards its zero-th operand.
  evaluated_[add_dependency] =
      GetEvaluatedLiteralFor(add_dependency->operand(0)).Clone();
  return OkStatus();
}

Status HloEvaluator::HandleGetTupleElement(HloInstruction* get_tuple_element) {
  const auto result_shape = get_tuple_element->shape();
  const int64_t index = get_tuple_element->tuple_index();

  auto operand = get_tuple_element->operand(0);
  TF_ASSIGN_OR_RETURN(
      auto inferred_return_shape,
      ShapeInference::InferGetTupleElementShape(operand->shape(), index));
  TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
      << "return shape set to: " << ShapeUtil::HumanString(result_shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);

  const Literal& operand_tuple_literal = GetEvaluatedLiteralFor(operand);

  evaluated_[get_tuple_element] =
      Literal(ShapeUtil::GetTupleElementShape(operand->shape(), index));
  return evaluated_[get_tuple_element].CopyFrom(operand_tuple_literal,
                                                /*dest_shape_index=*/{},
                                                /*src_shape_index=*/{index});
}

Status HloEvaluator::HandleCopy(HloInstruction* copy) {
  TF_RET_CHECK(ShapeUtil::Compatible(copy->shape(), copy->operand(0)->shape()));
  evaluated_[copy] = GetEvaluatedLiteralFor(copy->operand(0)).Clone();
  return OkStatus();
}

Status HloEvaluator::HandleAsyncStart(HloInstruction* async_start) {
  std::vector<const Literal*> arg_literals;
  arg_literals.reserve(async_start->operands().size());
  for (auto operand : async_start->operands()) {
    const Literal& arg_literal = GetEvaluatedLiteralFor(operand);
    arg_literals.push_back(&arg_literal);
  }

  HloEvaluator embedded_evaluator;
  embedded_evaluator.set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  TF_ASSIGN_OR_RETURN(
      Literal result,
      embedded_evaluator.Evaluate(*async_start->async_wrapped_computation(),
                                  arg_literals));

  evaluated_[async_start] = Literal(async_start->shape());
  // Copy the operand values to the index {0, i} of the output.
  for (int i = 0; i < arg_literals.size(); ++i) {
    TF_RETURN_IF_ERROR(evaluated_[async_start].CopyFrom(
        *arg_literals[i], /*dest_shape_index=*/{0, i},
        /*src_shape_index=*/{}));
  }
  // Move the output value to the index {1} of the output.
  TF_RETURN_IF_ERROR(evaluated_[async_start].MoveFrom(
      std::move(result), /*dest_shape_index=*/{1}));

  return OkStatus();
}

Status HloEvaluator::HandleAsyncUpdate(HloInstruction* async_update) {
  const Literal& operand_tuple_literal =
      GetEvaluatedLiteralFor(async_update->operand(0));
  evaluated_[async_update] = Literal(async_update->shape());
  TF_RETURN_IF_ERROR(evaluated_[async_update].CopyFrom(operand_tuple_literal,
                                                       /*dest_shape_index=*/{},
                                                       /*src_shape_index=*/{}));
  return OkStatus();
}

Status HloEvaluator::HandleAsyncDone(HloInstruction* async_done) {
  const Literal& operand_tuple_literal =
      GetEvaluatedLiteralFor(async_done->operand(0));
  evaluated_[async_done] = Literal(async_done->shape());
  TF_RETURN_IF_ERROR(evaluated_[async_done].CopyFrom(operand_tuple_literal,
                                                     /*dest_shape_index=*/{},
                                                     /*src_shape_index=*/{1}));
  return OkStatus();
}

Status HloEvaluator::HandleCopyStart(HloInstruction* copy_start) {
  if (copy_start->user_count() != 1 ||
      copy_start->users().at(0)->opcode() != HloOpcode::kCopyDone) {
    return tensorflow::errors::FailedPrecondition(
        "Cannot evaluate a kCopyStart that doesn't have a single kCopyDone "
        "user.");
  }

  // The context in index {2} is undefined, but since we can't represent
  // undefined values using a Literal, we just use 0. This should be safe though
  // since we ensure that the only user of a kCopyStart is a kCopyDone which
  // consumes the context. Also note that MakeTuple copies its arguments, so
  // this is memory-safe.
  const Literal context_literal = LiteralUtil::CreateR0<uint32_t>(0);
  evaluated_[copy_start] = LiteralUtil::MakeTuple(
      {&GetEvaluatedLiteralFor(copy_start->operand(0)),
       &GetEvaluatedLiteralFor(copy_start->operand(0)), &context_literal});
  return OkStatus();
}

Status HloEvaluator::HandleCopyDone(HloInstruction* copy_done) {
  const HloInstruction* operand = copy_done->operand(0);
  if (operand->opcode() != HloOpcode::kCopyStart) {
    return tensorflow::errors::FailedPrecondition(
        "Cannot evaluate a kCopyDone that doesn't have a kCopyStart as "
        "operand.");
  }

  const Literal& operand_tuple_literal = GetEvaluatedLiteralFor(operand);
  evaluated_[copy_done] =
      Literal(ShapeUtil::GetTupleElementShape(operand->shape(), /*index=*/0));
  TF_RETURN_IF_ERROR(evaluated_[copy_done].CopyFrom(operand_tuple_literal,
                                                    /*dest_shape_index=*/{},
                                                    /*src_shape_index=*/{0}));
  return OkStatus();
}

Status HloEvaluator::HandleCall(HloInstruction* call) {
  auto* computation = call->to_apply();
  auto operands = call->operands();

  std::vector<const Literal*> arg_literals;
  arg_literals.reserve(operands.size());
  for (auto operand : operands) {
    const Literal& arg_literal = GetEvaluatedLiteralFor(operand);
    arg_literals.push_back(&arg_literal);
  }

  std::unique_ptr<HloEvaluator> embedded_evaluator =
      CreateEmbedded(max_loop_iterations_);
  embedded_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  TF_ASSIGN_OR_RETURN(Literal result,
                      embedded_evaluator->Evaluate(*computation, arg_literals));

  evaluated_[call] = std::move(result);
  return OkStatus();
}

Status HloEvaluator::HandleFusion(HloInstruction* fusion) {
  HloModuleConfig config;
  // Attach cloned computation to an empty HLO module so the existing ones are
  // not modified.
  HloModule empty_hlo_module("EmptyModuleForFusion", config);
  HloCloneContext context(&empty_hlo_module);
  auto cloned_fused_computation =
      fusion->fused_instructions_computation()->Clone(
          /*suffix=*/"clone_with_layout", &context);
  for (auto* instruction : cloned_fused_computation->instructions()) {
    if (!LayoutUtil::HasLayout(instruction->shape())) {
      LayoutUtil::SetToDefaultLayout(instruction->mutable_shape());
    }
  }
  auto readded_computation =
      empty_hlo_module.AddEntryComputation(std::move(cloned_fused_computation));

  auto operands = fusion->operands();
  std::vector<const Literal*> arg_literals;
  arg_literals.reserve(operands.size());
  for (auto operand : operands) {
    const Literal& arg_literal = GetEvaluatedLiteralFor(operand);
    arg_literals.push_back(&arg_literal);
  }

  std::unique_ptr<HloEvaluator> embedded_evaluator =
      CreateEmbedded(max_loop_iterations_);
  embedded_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  TF_ASSIGN_OR_RETURN(Literal result, embedded_evaluator->Evaluate(
                                          *readded_computation, arg_literals));

  evaluated_[fusion] = std::move(result);
  return OkStatus();
}

Status HloEvaluator::HandleConditional(HloInstruction* conditional) {
  const auto& branch_index_literal =
      GetEvaluatedLiteralFor(conditional->operand(0));
  int branch_index;
  if (conditional->operand(0)->shape().element_type() == PRED) {
    branch_index = branch_index_literal.Get<bool>({}) ? 0 : 1;
  } else {
    branch_index = branch_index_literal.Get<int32_t>({});
    if (branch_index < 0 || branch_index >= conditional->branch_count()) {
      branch_index = conditional->branch_count() - 1;
    }
  }
  const auto& branch_computation_arg =
      GetEvaluatedLiteralFor(conditional->operand(1 + branch_index));

  std::unique_ptr<HloEvaluator> embedded_evaluator =
      CreateEmbedded(max_loop_iterations_);
  embedded_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  TF_ASSIGN_OR_RETURN(Literal result,
                      embedded_evaluator->Evaluate(
                          *conditional->branch_computation(branch_index),
                          {&branch_computation_arg}));

  evaluated_[conditional] = std::move(result);
  return OkStatus();
}

Status HloEvaluator::HandleSelect(HloInstruction* select) {
  const auto& pred = GetEvaluatedLiteralFor(select->operand(0));
  const auto& on_true = GetEvaluatedLiteralFor(select->operand(1));
  const auto& on_false = GetEvaluatedLiteralFor(select->operand(2));

  // If predicate is of scalar type, no element-wise selection would be needed.
  if (ShapeUtil::IsScalar(pred.shape())) {
    if (pred.Get<bool>({})) {
      evaluated_[select] = on_true.Clone();
    } else {
      evaluated_[select] = on_false.Clone();
    }
    return OkStatus();
  }

  return DefaultAction(select);
}

namespace {

StatusOr<Literal> CreateScalarLiteral(int64_t value,
                                      PrimitiveType element_type) {
  Literal result;
  switch (element_type) {
    case S8:
      result = LiteralUtil::CreateR0(static_cast<int8_t>(value));
      break;
    case U8:
      result = LiteralUtil::CreateR0(static_cast<uint8_t>(value));
      break;
    case S16:
      result = LiteralUtil::CreateR0(static_cast<int16_t>(value));
      break;
    case U16:
      result = LiteralUtil::CreateR0(static_cast<uint16_t>(value));
      break;
    case S32:
      result = LiteralUtil::CreateR0(static_cast<int32_t>(value));
      break;
    case U32:
      result = LiteralUtil::CreateR0(static_cast<uint32_t>(value));
      break;
    case S64:
      result = LiteralUtil::CreateR0(static_cast<int64_t>(value));
      break;
    case U64:
      result = LiteralUtil::CreateR0(static_cast<uint64_t>(value));
      break;
    default:
      return InvalidArgument("Unsupported element type.");
  }
  return result;
}

// Parses the while loop if it matches one of the known patterns. Returns the
// value of the loop induction variable after the loop execution if the loop is
// static.
StatusOr<Literal> TryParseAndEvaluateWhileInductionVar(
    HloInstruction* while_hlo) {
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(while_hlo);
  if (!parsed_while_loop.has_value() || parsed_while_loop->is_dynamic()) {
    return FailedPrecondition(
        "Cannot evaluate a while loop's induction variable since the loop "
        "does not match a known loop pattern or the loop is not static.");
  }
  int64_t induction_var_value =
      parsed_while_loop->static_while_loop->induction_var_init_value +
      parsed_while_loop->static_while_loop->trip_count *
          parsed_while_loop->static_while_loop->step_size;
  Shape result_shape = while_hlo->shape().tuple_shapes(
      parsed_while_loop->static_while_loop->induction_var_index);
  TF_ASSIGN_OR_RETURN(
      Literal result,
      CreateScalarLiteral(induction_var_value, result_shape.element_type()));
  std::vector<Literal*> while_result_element_ptrs;
  while_result_element_ptrs.reserve(while_hlo->shape().tuple_shapes_size());
  std::vector<Literal> while_result_elements(
      while_hlo->shape().tuple_shapes_size());
  for (int i = 0; i < while_hlo->shape().tuple_shapes_size(); ++i) {
    if (i == parsed_while_loop->static_while_loop->induction_var_index) {
      while_result_element_ptrs.push_back(&result);
    } else {
      const Shape& shape = while_hlo->shape().tuple_shapes(i);
      while_result_elements[i] =
          Literal::CreateFromShapeWithUnknownLeafArrays(shape);
    }
  }
  return LiteralUtil::MakeTuple(while_result_element_ptrs);
}

}  // namespace

Status HloEvaluator::HandleWhile(HloInstruction* while_hlo) {
  HloComputation* cond_comp = while_hlo->while_condition();
  HloComputation* body_comp = while_hlo->while_body();
  // Initialize the loop carried valued with the input to the While instruction.
  auto lcv = GetEvaluatedLiteralFor(while_hlo->operand(0)).Clone();
  if (!lcv.IsKnown()) {
    std::optional<ParsedWhileLoop> parsed_while_loop =
        PatternMatchParseWhileLoop(while_hlo);
    evaluated_[while_hlo] =
        Literal::CreateFromShapeWithUnknownLeafArrays(while_hlo->shape());
    if (!parsed_while_loop.has_value() || parsed_while_loop->is_dynamic() ||
        visitor_shape_index_.size() != 1 ||
        parsed_while_loop->static_while_loop->induction_var_index !=
            visitor_shape_index_[0]) {
      return OkStatus();
    }
    Shape induction_var_shape =
        ShapeUtil::GetSubshape(while_hlo->shape(), visitor_shape_index_);
    int64_t trip_count = parsed_while_loop->static_while_loop->trip_count;
    TF_ASSIGN_OR_RETURN(
        Literal induction_var_val,
        CreateScalarLiteral(trip_count, induction_var_shape.element_type()));
    TF_RETURN_IF_ERROR(evaluated_[while_hlo].CopyFrom(
        induction_var_val, /*dest_shape_index=*/visitor_shape_index_,
        /*src_shape_index=*/{}));
    return OkStatus();
  }
  bool keep_going = true;
  int64_t iteration_count = 0;
  std::unique_ptr<HloEvaluator> cond_evaluator =
      CreateEmbedded(max_loop_iterations_);
  cond_evaluator->set_dynamic_dimension_inference(dynamic_dimension_inference_);
  std::unique_ptr<HloEvaluator> loop_body_evaluator =
      CreateEmbedded(max_loop_iterations_);
  loop_body_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  while (keep_going) {
    if (max_loop_iterations_ >= 0 && iteration_count++ > max_loop_iterations_) {
      StatusOr<Literal> result =
          TryParseAndEvaluateWhileInductionVar(while_hlo);
      if (result.ok()) {
        lcv = result.ConsumeValueOrDie();
        break;
      } else {
        return InvalidArgument("Loop %s exceeded loop iteration limit (%d).",
                               while_hlo->name(), max_loop_iterations_);
      }
    }
    TF_ASSIGN_OR_RETURN(auto cond_val,
                        cond_evaluator->Evaluate(*cond_comp, {&lcv}));
    keep_going = cond_val.GetFirstElement<bool>();
    if (keep_going) {
      TF_ASSIGN_OR_RETURN(auto body_val,
                          loop_body_evaluator->Evaluate(*body_comp, {&lcv}));
      VLOG(3) << "Loop iteration result: " << body_val.ToString();
      lcv = std::move(body_val);
      cond_evaluator->ResetVisitStates();
      loop_body_evaluator->ResetVisitStates();
    }
  }
  evaluated_[while_hlo] = std::move(lcv);
  return OkStatus();
}

namespace {
template <typename NativeT>
Literal ExtractLiteralFromIndexPositions(const Literal& from,
                                         absl::Span<int64_t const> indices,
                                         bool extract_as_scalar) {
  if (extract_as_scalar) {
    return LiteralUtil::CreateR0<NativeT>(from.Get<NativeT>({indices[0]}));
  }
  // We use a InlinedVector here because we need to convert it to an
  // absl::Span later, and this would not work with std::vector<bool>.
  absl::InlinedVector<NativeT, 10> values;
  for (int64_t index : indices) {
    values.push_back(from.Get<NativeT>({index}));
  }
  return LiteralUtil::CreateR1<NativeT>(values);
}

StatusOr<Literal> ExtractFromIndexPositions(const Literal& from,
                                            absl::Span<int64_t const> indices,
                                            bool extract_as_scalar = false) {
  if (extract_as_scalar) {
    CHECK_EQ(indices.size(), 1);
  }
  PrimitiveType type = from.shape().element_type();
  switch (type) {
    case PRED: {
      return ExtractLiteralFromIndexPositions<bool>(from, indices,
                                                    extract_as_scalar);
    }
    case U8: {
      return ExtractLiteralFromIndexPositions<uint8_t>(from, indices,
                                                       extract_as_scalar);
    }
    case S8: {
      return ExtractLiteralFromIndexPositions<int8_t>(from, indices,
                                                      extract_as_scalar);
    }
    case BF16: {
      return ExtractLiteralFromIndexPositions<bfloat16>(from, indices,
                                                        extract_as_scalar);
    }
    case F16: {
      return ExtractLiteralFromIndexPositions<Eigen::half>(from, indices,
                                                           extract_as_scalar);
    }
    case U16: {
      return ExtractLiteralFromIndexPositions<uint16_t>(from, indices,
                                                        extract_as_scalar);
    }
    case S16: {
      return ExtractLiteralFromIndexPositions<int16_t>(from, indices,
                                                       extract_as_scalar);
    }
    case F32: {
      return ExtractLiteralFromIndexPositions<float>(from, indices,
                                                     extract_as_scalar);
    }
    case U32: {
      return ExtractLiteralFromIndexPositions<uint32_t>(from, indices,
                                                        extract_as_scalar);
    }
    case S32: {
      return ExtractLiteralFromIndexPositions<int32_t>(from, indices,
                                                       extract_as_scalar);
    }
    case F64: {
      return ExtractLiteralFromIndexPositions<double>(from, indices,
                                                      extract_as_scalar);
    }
    case C64: {
      return ExtractLiteralFromIndexPositions<std::complex<float>>(
          from, indices, extract_as_scalar);
    }
    case U64: {
      return ExtractLiteralFromIndexPositions<uint64_t>(from, indices,
                                                        extract_as_scalar);
    }
    case S64: {
      return ExtractLiteralFromIndexPositions<int64_t>(from, indices,
                                                       extract_as_scalar);
    }
    case C128: {
      return ExtractLiteralFromIndexPositions<std::complex<double>>(
          from, indices, extract_as_scalar);
    }
    default:
      return InvalidArgument("Unsupported type for Sort: %s",
                             PrimitiveType_Name(type));
  }
}
}  // namespace

Status HloEvaluator::HandleSort(HloInstruction* sort) {
  TF_RET_CHECK(sort->operand_count() >= 1)
      << "Expected at least 1 operand for sort";
  for (int64_t i = 1; i < sort->operand_count(); ++i) {
    TF_RET_CHECK(ShapeUtil::SameDimensions(sort->operand(0)->shape(),
                                           sort->operand(i)->shape()))
        << "All Sort operands must have the same dimensions";
  }

  if (VLOG_IS_ON(3)) {
    for (int64_t i = 0; i < sort->operand_count(); ++i) {
      VLOG(3) << "HandleSort operand " << i << " literal: "
              << GetEvaluatedLiteralFor(sort->operand(i)).ToString();
    }
  }
  Shape key_shape = sort->operand(0)->shape();
  auto rank = key_shape.rank();
  std::vector<Literal> result_literals;
  result_literals.reserve(sort->operand_count());
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    result_literals.emplace_back(sort->operand(i)->shape());
  }
  std::vector<int64_t> zero_base(rank, 0);
  std::vector<int64_t> increment(rank, 1);
  int64_t sort_dim = sort->dimensions(0);
  int64_t sort_dim_elements = key_shape.dimensions(sort_dim);
  TF_RET_CHECK(sort_dim >= 0 && sort_dim < increment.size())
      << "Unexpected out-of-bound sort dimension " << sort_dim
      << " accessing increment of size " << increment.size();
  increment[sort_dim] = sort_dim_elements;
  std::unique_ptr<HloEvaluator> embedded_evaluator =
      CreateEmbedded(max_loop_iterations_);
  // Iterate through each dimension except 'sort_dim'.
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
      key_shape, zero_base, key_shape.dimensions(), increment,
      [&](absl::Span<const int64_t> indices) -> StatusOr<bool> {
        // Extract a slice from each operand literal that corresponds to
        // exactly the row in dimension 'sort_dim'.
        std::vector<int64_t> limit_indices(indices.begin(), indices.end());
        absl::c_for_each(limit_indices, [](int64_t& index) { ++index; });
        limit_indices[sort_dim] = sort_dim_elements;
        std::vector<Literal> literals_to_sort;
        literals_to_sort.reserve(sort->operand_count());
        for (int64_t i = 0; i < sort->operand_count(); ++i) {
          TF_ASSIGN_OR_RETURN(auto literal_to_sort,
                              GetEvaluatedLiteralFor(sort->operand(i))
                                  .Slice(indices, limit_indices)
                                  .Reshape({sort_dim_elements}));
          literals_to_sort.push_back(std::move(literal_to_sort));
        }
        std::vector<int64_t> indices_to_sort(sort_dim_elements);
        std::iota(indices_to_sort.begin(), indices_to_sort.end(), 0);
        Status compare_status = OkStatus();
        auto comparator = [sort, &compare_status,
                           embedded_evaluator = embedded_evaluator.get(),
                           &literals_to_sort](int64_t a, int64_t b) {
          std::vector<Literal> literals;
          literals.reserve(2 * sort->operand_count());
          for (int64_t i = 0; i < sort->operand_count(); ++i) {
            auto lhs = ExtractFromIndexPositions(literals_to_sort[i], {a},
                                                 /*extract_as_scalar=*/true);
            if (!lhs.ok()) {
              compare_status = lhs.status();
              return false;
            }
            literals.push_back(std::move(lhs.ValueOrDie()));
            auto rhs = ExtractFromIndexPositions(literals_to_sort[i], {b},
                                                 /*extract_as_scalar=*/true);
            if (!rhs.ok()) {
              compare_status = rhs.status();
              return false;
            }
            literals.push_back(std::move(rhs.ValueOrDie()));
          }
          std::vector<const Literal*> literal_ptrs;
          absl::c_transform(literals, std::back_inserter(literal_ptrs),
                            [](const Literal& literal) { return &literal; });

          auto computed_result =
              embedded_evaluator->Evaluate(*sort->to_apply(), literal_ptrs);
          // Clear visit states so that we can use the evaluator again
          // on the same computation.
          embedded_evaluator->ResetVisitStates();
          if (!computed_result.ok()) {
            compare_status = computed_result.status();
            return false;
          }
          return computed_result.ValueOrDie().Get<bool>({});
        };
        if (Cast<HloSortInstruction>(sort)->is_stable()) {
          std::stable_sort(indices_to_sort.begin(), indices_to_sort.end(),
                           comparator);
        } else {
          std::sort(indices_to_sort.begin(), indices_to_sort.end(), comparator);
        }
        if (!compare_status.ok()) {
          return compare_status;
        }
        std::vector<int64_t> slice_dimensions(rank, 1);
        slice_dimensions[sort_dim] = sort_dim_elements;
        std::vector<int64_t> start_indices(rank, 0);
        for (int64_t i = 0; i < sort->operand_count(); ++i) {
          TF_ASSIGN_OR_RETURN(
              Literal sorted_literal,
              ExtractFromIndexPositions(literals_to_sort[i], indices_to_sort));
          TF_ASSIGN_OR_RETURN(auto sorted_literal_reshaped,
                              sorted_literal.Reshape(slice_dimensions));
          TF_RETURN_IF_ERROR(result_literals[i].CopySliceFrom(
              sorted_literal_reshaped, start_indices, indices,
              slice_dimensions));
        }
        return true;
      }));

  if (sort->operand_count() == 1) {
    evaluated_[sort] = std::move(result_literals[0]);
  } else {
    std::vector<const Literal*> literal_ptrs;
    absl::c_transform(result_literals, std::back_inserter(literal_ptrs),
                      [](const Literal& literal) { return &literal; });

    Literal result_tuple = LiteralUtil::MakeTuple(literal_ptrs);
    VLOG(3) << "HandleSort result_tuple: " << result_tuple.ToString();

    evaluated_[sort] = std::move(result_tuple);
  }
  return OkStatus();
}

static bool IsScalarAdd(HloComputation* computation) {
  HloInstruction* instruction = computation->root_instruction();
  if (instruction->opcode() == HloOpcode::kAdd &&
      computation->num_parameters() == 2) {
    const HloInstruction* lhs = instruction->operand(0);
    const HloInstruction* rhs = instruction->operand(1);
    return lhs->opcode() == HloOpcode::kParameter &&
           ShapeUtil::IsScalar(lhs->shape()) &&
           rhs->opcode() == HloOpcode::kParameter &&
           ShapeUtil::IsScalar(rhs->shape()) && lhs != rhs;
  }
  return false;
}

// Run a single step of an inner loop while running reduction, which applies
// the user-provided computation on the accumulator and the output element
// (until the reduction is completed, the output element is also used as
// an accumulator).
static StatusOr<bool> PerformReductionStep(
    bool is_tuple, absl::Span<const int64_t> input_index,
    absl::Span<const int64_t> output_index,
    absl::Span<const Literal* const> input_args, absl::Span<Literal> results,
    HloComputation* computation, HloEvaluator* embedded_evaluator) {
  int num_args = results.size();

  absl::InlinedVector<Literal, 1> arg_values;
  arg_values.reserve(num_args);
  absl::InlinedVector<Literal, 1> accumulators;
  accumulators.reserve(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    arg_values.emplace_back(
        ShapeUtil::MakeShape(input_args[i]->shape().element_type(), {}));
    accumulators.emplace_back(
        ShapeUtil::MakeShape(input_args[i]->shape().element_type(), {}));

    TF_RETURN_IF_ERROR(
        arg_values[i].CopyElementFrom(*input_args[i], input_index, {}));
    TF_RETURN_IF_ERROR(
        accumulators[i].CopyElementFrom(results[i], output_index, {}));
  }

  // Evaluate computation with specified literal operands.
  absl::InlinedVector<Literal*, 2> embedded_operands;
  for (Literal& accumulator : accumulators) {
    embedded_operands.push_back(&accumulator);
  }
  for (Literal& local_input : arg_values) {
    embedded_operands.push_back(&local_input);
  }

  TF_ASSIGN_OR_RETURN(
      Literal computed_result,
      embedded_evaluator->Evaluate(*computation, embedded_operands));

  // Clear visit states so that we can use the evaluator again on the same
  // computation.
  embedded_evaluator->ResetVisitStates();

  if (is_tuple) {
    std::vector<Literal> computed_results = computed_result.DecomposeTuple();
    for (int64_t i = 0; i < num_args; ++i) {
      TF_RETURN_IF_ERROR(
          results[i].CopyElementFrom(computed_results[i], {}, output_index));
    }
  } else {
    TF_RETURN_IF_ERROR(
        results[0].CopyElementFrom(computed_result, {}, output_index));
  }

  return true;
}

static StatusOr<bool> GenerateReduceOutputElement(
    bool is_tuple, absl::Span<const int64_t> output_index,

    absl::Span<const Literal* const> init_values,
    absl::Span<const Literal* const> input_args, absl::Span<Literal> results,

    HloComputation* function, HloEvaluator* embedded_evaluator,

    absl::Span<const int64_t> arg_dim_steps,
    absl::Span<const int64_t> arg_dim_counts,
    absl::Span<const int64_t> result_to_arg_index) {
  bool use_fast_add = ShapeUtil::ElementIsFloating(init_values[0]->shape()) &&
                      IsScalarAdd(function) && !is_tuple;

  const Shape& arg_shape = input_args[0]->shape();
  absl::Span<const int64_t> arg_dimensions = arg_shape.dimensions();
  std::vector<int64_t> base(arg_dimensions.size());
  for (int64_t i = 0; i < output_index.size(); ++i) {
    base[result_to_arg_index[i]] = output_index[i];
  }

  for (int64_t i = 0; i < results.size(); ++i) {
    TF_RETURN_IF_ERROR(
        results[i].CopyElementFrom(*init_values[i], {}, output_index));
  }

  if (use_fast_add) {
    double computed_result = *init_values[0]->GetAsDouble({});
    auto reduction_step =
        [&](absl::Span<const int64_t> input_index) -> StatusOr<bool> {
      double argument = *input_args[0]->GetAsDouble(input_index);
      computed_result += argument;
      return true;
    };
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
        arg_shape, base, arg_dim_counts, arg_dim_steps, reduction_step));
    TF_RETURN_IF_ERROR(results[0].SetFromDouble(output_index, computed_result));
    return true;
  }

  // Iterates only over reduced shape, as counts and steps are set to zero
  // for all non-reduced dimensions.
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
      arg_shape, base, arg_dim_counts, arg_dim_steps,
      [&](absl::Span<const int64_t> input_index) {
        return PerformReductionStep(is_tuple, input_index, output_index,
                                    input_args, results, function,
                                    embedded_evaluator);
      }));
  return true;
}

Status HloEvaluator::HandleReduce(HloInstruction* instr) {
  HloReduceInstruction* reduce = Cast<HloReduceInstruction>(instr);
  int64_t num_args = reduce->inputs().size();
  absl::Span<const int64_t> dimensions_to_reduce(reduce->dimensions());
  HloComputation* function = reduce->to_apply();

  absl::InlinedVector<const Shape*, 1> operand_shapes;
  for (const HloInstruction* operand : reduce->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                      ShapeInference::InferReduceShape(
                          operand_shapes, dimensions_to_reduce,
                          /*to_apply=*/function->ComputeProgramShape()));
  TF_RET_CHECK(ShapeUtil::CompatibleIgnoringFpPrecision(reduce->shape(),
                                                        inferred_return_shape))
      << "return shape is set to: " << ShapeUtil::HumanString(reduce->shape())
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);

  absl::InlinedVector<const Literal*, 1> input_args(num_args);
  absl::InlinedVector<const Literal*, 1> init_values(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    input_args[i] = &GetEvaluatedLiteralFor(reduce->inputs()[i]);
    VLOG(3) << "HandleReduce arg_literal: " << input_args[i]->ToString();
    init_values[i] = &GetEvaluatedLiteralFor(reduce->init_values()[i]);
    VLOG(3) << "HandleReduce init_literal: " << init_values[i]->ToString();
    TF_RET_CHECK(ShapeUtil::IsScalar(init_values[i]->shape()));
  }

  // All args and results have the same dimensions, so pick an arbitrary one.
  const Shape& arg_shape = input_args[0]->shape();
  const Shape& out_shape = inferred_return_shape;
  bool is_tuple = out_shape.IsTuple();
  const Shape& output_shape = inferred_return_shape.IsTuple()
                                  ? inferred_return_shape.tuple_shapes(0)
                                  : inferred_return_shape;

  absl::Span<const int64_t> arg_dimensions = arg_shape.dimensions();

  // All increments are set to 0.
  std::vector<int64_t> arg_dim_steps(arg_dimensions.size());

  // All counts are set to 0.
  std::vector<int64_t> arg_dim_counts(arg_dimensions.size());

  // Set steps and counts for reduced dimensions.
  // This avoids iterating over non-reduced dimensions, as their step
  // and count is set to zero.
  for (const int64_t dim : dimensions_to_reduce) {
    arg_dim_steps[dim] = 1;
    arg_dim_counts[dim] = arg_dimensions[dim];
  }

  // Map each dimension in the result to a dimension in arg that isn't
  // being reduced.
  std::vector<int64_t> result_to_arg_index;
  for (int64_t i = 0; i < arg_dimensions.size(); ++i) {
    if (arg_dim_steps[i] == 0) {
      result_to_arg_index.push_back(i);
    }
  }

  std::unique_ptr<HloEvaluator> embedded_evaluator =
      CreateEmbedded(max_loop_iterations_);
  absl::InlinedVector<Literal, 1> results(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    results[i] = Literal(is_tuple ? out_shape.tuple_shapes(i) : out_shape);
  }

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
      output_shape, [&](absl::Span<const int64_t> output_index) {
        return GenerateReduceOutputElement(
            is_tuple, output_index, init_values, input_args,
            absl::Span<Literal>(results), function, embedded_evaluator.get(),
            arg_dim_steps, arg_dim_counts, result_to_arg_index);
      }));

  if (is_tuple) {
    Literal tuple_result(inferred_return_shape);
    for (int64_t i = 0; i < num_args; ++i) {
      TF_CHECK_OK(tuple_result.MoveFrom(std::move(results[i]), {i}));
    }
    evaluated_[reduce] = std::move(tuple_result);
  } else {
    CHECK_EQ(results.size(), 1);
    evaluated_[reduce] = std::move(results[0]);
  }
  if (!ShapeUtil::Compatible(reduce->shape(), inferred_return_shape)) {
    TF_ASSIGN_OR_RETURN(evaluated_[reduce],
                        evaluated_[reduce].ConvertToShape(reduce->shape()));
  }
  return OkStatus();
}

Status HloEvaluator::HandleReduceWindow(HloInstruction* hlo) {
  // Here we delegate the handling to the typed visitor class, instantiated by
  // using the type of the first input of ReduceWindow. The support for the
  // variadic case inside the typed_visitor is made to not use the template
  // parameter so it doesn't really matter which type is used to instantiate it
  // here. We choose not to move the implementation for handle ReduceWindow
  // from the typed visitor to here because we need to reuse the
  // IterateThroughWindow method, which is defined and only avaiable inside the
  // typed visitor.
  if (hlo->shape().IsTuple()) {
    return hlo->Visit(
        typed_visitors_[hlo->shape().tuple_shapes(0).element_type()].get());
  } else {
    return DefaultAction(hlo);
  }
}

Status HloEvaluator::HandleCustomCall(HloInstruction* custom_call) {
  if (!custom_call_handler_) {
    // No handler is registered; this means custom-calls are not allowed.
    return DefaultAction(custom_call);
  }

  // Evaluate input operands so the handler has access to the operand data.
  std::vector<const Literal*> operands;
  operands.reserve(custom_call->operand_count());
  for (const HloInstruction* operand : custom_call->operands()) {
    operands.push_back(&GetEvaluatedLiteralFor(operand));
  }

  // Synchronously issue the handler to populate the instruction output literal.
  TF_ASSIGN_OR_RETURN(
      auto output, custom_call_handler_(custom_call, absl::MakeSpan(operands)));

  evaluated_[custom_call] = std::move(output);
  return OkStatus();
}

Status HloEvaluator::Preprocess(HloInstruction* hlo) {
  VLOG(2) << "About to visit HLO: " << hlo->ToString();
  if (!enable_partial_evaluation_) {
    for (HloInstruction* operand : hlo->mutable_operands()) {
      if (!IsAlreadyEvaluated(operand) ||
          !GetEvaluatedLiteralFor(operand).IsKnown()) {
        return tensorflow::errors::FailedPrecondition(
            "Failed to evaluate instruction since its operands are unknown "
            "or undetermined and partial evaluation is not enabled.");
      }
    }
  }
  return ShapeUtil::ValidateShape(hlo->shape());
}

Status HloEvaluator::Postprocess(HloInstruction* hlo) {
  VLOG(2) << "Finished visiting " << hlo->ToString()
          << "; evaluated value is: " << GetEvaluatedLiteralFor(hlo).ToString();
  // Out of convenience the literal may have been produced with a different
  // layout. Relayout as indicated by the HLO instruction.
  if (!Layout::Equal().MinorToMajorOnly()(
          GetEvaluatedLiteralFor(hlo).shape().layout(),
          hlo->shape().layout())) {
    evaluated_.at(hlo) = evaluated_.at(hlo).Relayout(hlo->shape());
  }
  return OkStatus();
}

namespace {
template <typename T>
std::unique_ptr<Array2D<T>> MatmulArray2DImpl(
    const Array2D<T>& lhs, const Array2D<T>& rhs,
    const std::function<void(const void* run_options_ptr, T* out, T* lhs,
                             T* rhs, int64_t m, int64_t n, int64_t k,
                             int32_t transpose_lhs, int32_t transpose_rhs)>&
        impl_fn) {
  CHECK_EQ(lhs.width(), rhs.height());
  int m = lhs.height();
  int n = rhs.width();
  int k = lhs.width();
  auto result = absl::make_unique<Array2D<T>>(m, n);
  // Because Eigen is a header-oriented library, make sure that the Eigen code
  // is the same as the code used by the CPU backend (otherwise the linker will
  // randomly pick *some* definition).
  impl_fn(
      /*run_options_ptr=*/nullptr, result->data(), rhs.data(), lhs.data(), n, m,
      k,
      /*transpose_lhs=*/0,
      /*transpose_rhs=*/0);
  return result;
}
}  // namespace

std::unique_ptr<Array2D<Eigen::half>> HloEvaluator::MatmulArray2D(
    const Array2D<Eigen::half>& lhs, const Array2D<Eigen::half>& rhs) {
  return MatmulArray2DImpl<Eigen::half>(
      lhs, rhs, __xla_cpu_runtime_EigenSingleThreadedMatMulF16);
}

std::unique_ptr<Array2D<float>> HloEvaluator::MatmulArray2D(
    const Array2D<float>& lhs, const Array2D<float>& rhs) {
  return MatmulArray2DImpl<float>(
      lhs, rhs, __xla_cpu_runtime_EigenSingleThreadedMatMulF32);
}

std::unique_ptr<Array2D<double>> HloEvaluator::MatmulArray2D(
    const Array2D<double>& lhs, const Array2D<double>& rhs) {
  return MatmulArray2DImpl<double>(
      lhs, rhs, __xla_cpu_runtime_EigenSingleThreadedMatMulF64);
}

std::unique_ptr<Array2D<std::complex<float>>> HloEvaluator::MatmulArray2D(
    const Array2D<std::complex<float>>& lhs,
    const Array2D<std::complex<float>>& rhs) {
  return MatmulArray2DImpl<std::complex<float>>(
      lhs, rhs, __xla_cpu_runtime_EigenSingleThreadedMatMulC64);
}

std::unique_ptr<Array2D<std::complex<double>>> HloEvaluator::MatmulArray2D(
    const Array2D<std::complex<double>>& lhs,
    const Array2D<std::complex<double>>& rhs) {
  return MatmulArray2DImpl<std::complex<double>>(
      lhs, rhs, __xla_cpu_runtime_EigenSingleThreadedMatMulC128);
}

std::unique_ptr<Array2D<int32_t>> HloEvaluator::MatmulArray2D(
    const Array2D<int32_t>& lhs, const Array2D<int32_t>& rhs) {
  return MatmulArray2DImpl<int32_t>(
      lhs, rhs, __xla_cpu_runtime_EigenSingleThreadedMatMulS32);
}

}  // namespace xla

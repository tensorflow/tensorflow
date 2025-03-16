/* Copyright 2017 The OpenXLA Authors.

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
#include "xla/hlo/evaluator/hlo_evaluator.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/internal/endian.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "xla/array2d.h"
#include "xla/comparison_util.h"
#include "xla/hlo/analysis/tuple_points_to_analysis.h"
#include "xla/hlo/evaluator/hlo_evaluator_typed_visitor.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/index_util.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/compilation_environments.h"
#include "xla/service/cpu/runtime_single_threaded_matmul.h"
#include "xla/service/gather_scatter_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"

namespace xla {

namespace {

using primitive_util::NativeTypeOf;

template <typename OperandT>
absl::StatusOr<Literal> Compare(const Shape& shape, Comparison comparison,
                                LiteralSlice lhs_literal,
                                LiteralSlice rhs_literal) {
  auto populate = [&](auto compare_op) -> absl::StatusOr<Literal> {
    Literal result(shape);

    // If layout is the same, we can use linear indexing into the literals.
    const Layout& lhs_layout = lhs_literal.shape().layout();
    const Layout& rhs_layout = rhs_literal.shape().layout();
    bool same_layout = LayoutUtil::Equal(lhs_layout, rhs_layout);

    if (same_layout) {
      TF_RETURN_IF_ERROR(result.PopulateLinearParallel<bool>(
          [&](int64_t linear_index, int /*thread_id*/) {
            auto lhs = lhs_literal.GetLinear<OperandT>(linear_index);
            auto rhs = rhs_literal.GetLinear<OperandT>(linear_index);
            if constexpr (is_specialized_floating_point_v<OperandT>) {
              if (comparison.IsTotalOrder()) {
                return compare_op(ToSignMagnitude(lhs), ToSignMagnitude(rhs));
              }
            }
            return compare_op(lhs, rhs);
          }));
    } else {
      TF_RETURN_IF_ERROR(result.PopulateParallel<bool>(
          [&](absl::Span<const int64_t> multi_index, int /*thread_id*/) {
            auto lhs = lhs_literal.Get<OperandT>(multi_index);
            auto rhs = rhs_literal.Get<OperandT>(multi_index);
            if constexpr (is_specialized_floating_point_v<OperandT>) {
              if (comparison.IsTotalOrder()) {
                return compare_op(ToSignMagnitude(lhs), ToSignMagnitude(rhs));
              }
            }
            return compare_op(lhs, rhs);
          }));
    }
    return std::move(result);
  };
  switch (comparison.GetDirection()) {
    case ComparisonDirection::kEq:
      return populate([](auto lhs, auto rhs) { return lhs == rhs; });
    case ComparisonDirection::kNe:
      return populate([](auto lhs, auto rhs) { return lhs != rhs; });
    case ComparisonDirection::kGe:
      if constexpr (!is_complex_v<OperandT>) {
        return populate([](auto lhs, auto rhs) { return lhs >= rhs; });
      }
      break;
    case ComparisonDirection::kGt:
      if constexpr (!is_complex_v<OperandT>) {
        return populate([](auto lhs, auto rhs) { return lhs > rhs; });
      }
      break;
    case ComparisonDirection::kLe:
      if constexpr (!is_complex_v<OperandT>) {
        return populate([](auto lhs, auto rhs) { return lhs <= rhs; });
      }
      break;
    case ComparisonDirection::kLt:
      if constexpr (!is_complex_v<OperandT>) {
        return populate([](auto lhs, auto rhs) { return lhs < rhs; });
      }
      break;
  }

  LOG(FATAL) << "unhandled direction for conversion to Comparison: "
             << comparison.ToString();
}

std::optional<bool> GetInstructionStaticValueAsBool(
    const HloInstruction* instruction) {
  HloEvaluator evaluator;
  absl::StatusOr<Literal> static_value =
      evaluator.Evaluate(instruction, /*precomputed_analyses=*/{},
                         /*recursively_evaluate_nonconstant_operands=*/true);
  if (static_value.ok()) {
    return static_value->GetFirstElement<bool>();
  }
  return std::nullopt;
}

template <PrimitiveType kType>
struct PopulateParallelImpl {
  using NativeT = NativeTypeOf<kType>;
  static absl::Status Run(
      Literal& literal,
      absl::FunctionRef<Literal(absl::Span<const int64_t>, int)>
          literal_generator) {
    return literal.PopulateParallel<NativeT>(
        [&literal_generator](absl::Span<const int64_t> output_index,
                             int thread_id) {
          return literal_generator(output_index, thread_id)
              .template Get<NativeT>({});
        });
  }
};

template <PrimitiveType kType>
struct PopulateImpl {
  using NativeT = NativeTypeOf<kType>;
  static absl::Status Run(
      Literal& literal,
      absl::FunctionRef<Literal(absl::Span<const int64_t>)> literal_generator) {
    return literal.Populate<NativeT>(
        [&literal_generator](absl::Span<const int64_t> output_index) {
          return literal_generator(output_index).template Get<NativeT>({});
        });
  }
};

// Helper function for when it has a Literal generator (typically from an
// embedded evaluator to evaluate subcomputations) but needs to extract the
// scalar value of a specific type from it to populate a Literal. Putting such
// evaluation implementations in typed visitors gives no performance benefits
// but leads to unnecessarily large code size therefore providing a delegation
// to small templated helpers just for the parts that require manipulating the
// native types to avoid templating the whole implementations.
template <template <PrimitiveType> typename Trait, typename F>
absl::Status Apply(Literal& literal, F&& literal_generator) {
  return primitive_util::PrimitiveTypeSwitch<absl::Status>(
      [&, literal_generator = std::forward<F>(literal_generator)](
          auto primitive_type_constant) -> absl::Status {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          return Trait<primitive_type_constant>::Run(
              literal, std::move(literal_generator));
        }
        LOG(FATAL) << "Unhandled primitive type "
                   << literal.shape().element_type();
      },
      literal.shape().element_type());
}

absl::Status MakeEvalErrorDueToParamOrInfeed(
    const HloInstruction& eval_instruction) {
  absl::Status error = absl::FailedPreconditionError(absl::StrCat(
      "Failed to evaluate instruction (", eval_instruction.name(),
      ") since it depends on infeed or parameters to its parent computation (",
      eval_instruction.parent()->name(), ")."));
  std::string error_payload;
  error_payload.resize(sizeof(internal::EvalErrorDetail));
  absl::little_endian::Store32(
      const_cast<char*>(error_payload.data()),
      static_cast<uint32_t>(
          internal::EvalErrorDetail::kDynamicValueDependence));
  error.SetPayload(internal::kEvalErrorDetailUrl, absl::Cord(error_payload));
  return error;
}

// Represents a value that might or might not be determined statically.
struct DynamicOrStaticInteger {
  std::optional<int64_t> static_value;
  bool is_dynamic() const { return !static_value.has_value(); }

  std::string ToString() const {
    return is_dynamic() ? std::string("DYNAMIC") : absl::StrCat(*static_value);
  }
};

std::optional<DynamicOrStaticInteger> GetInstructionValueAsInteger(
    const HloInstruction* instruction,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  HloEvaluator evaluator;
  absl::StatusOr<Literal> static_value =
      evaluator.Evaluate(instruction, precomputed_analyses,
                         /*recursively_evaluate_nonconstant_operands=*/true);
  if (static_value.ok()) {
    if (instruction->shape().element_type() == PrimitiveType::PRED) {
      return DynamicOrStaticInteger{
          static_cast<int64_t>(static_value->GetFirstElement<bool>())};
    } else {
      return DynamicOrStaticInteger{static_value->GetFirstInteger()};
    }
  }

  std::optional<internal::EvalErrorDetail> eval_error_detail =
      internal::ParseEvalErrorDetail(static_value.status());
  if (eval_error_detail.has_value() &&
      *eval_error_detail ==
          internal::EvalErrorDetail::kDynamicValueDependence) {
    return DynamicOrStaticInteger{std::nullopt};
  }
  return std::nullopt;
}

// Represents an index into the while argument tuple and / or a value.
// At least one of param_index and value has a value; both of them could have
// a value.
struct ParamIndexAndValue {
  std::optional<int64_t> param_index;
  std::optional<DynamicOrStaticInteger> value;

  bool IsValid() const { return param_index.has_value() || value.has_value(); }
  std::string ToString() const {
    return absl::StrCat(
        "param_index:",
        !param_index.has_value() ? std::string("UNKNOWN")
                                 : absl::StrCat(*param_index),
        ",", "value:",
        !value.has_value() ? std::string("UNKONWN") : value->ToString());
  }
};

std::optional<ParamIndexAndValue> TryParsingInstructionAsParameterAndInteger(
    const HloInstruction* instruction,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  // Skip copies.
  if (instruction->opcode() == HloOpcode::kCopy) {
    return TryParsingInstructionAsParameterAndInteger(instruction->operand(0),
                                                      precomputed_analyses);
  }
  if (instruction->opcode() == HloOpcode::kCopyDone) {
    return TryParsingInstructionAsParameterAndInteger(
        instruction->operand(0)->operand(1), precomputed_analyses);
  }
  ParamIndexAndValue result;
  if (Match(instruction, match::GetTupleElement().WithOperand(
                             0, match::Parameter().WithParameterNum(0)))) {
    result.param_index = instruction->tuple_index();
  }
  std::optional<DynamicOrStaticInteger> integer_value =
      GetInstructionValueAsInteger(instruction, precomputed_analyses);
  result.value = std::move(integer_value);
  if (!result.IsValid()) {
    return std::nullopt;
  }
  return std::optional<ParamIndexAndValue>(std::move(result));
}

// Represents the while loop condition comparison.
// We assume comparison is of the form: lhs comp rhs.
struct WhileCondComparison {
  ComparisonDirection comparison_direction;
  ParamIndexAndValue lhs;
  ParamIndexAndValue rhs;

  std::string ToString() const {
    return absl::StrCat("WhileCondComparison{", "LHS:{", lhs.ToString(),
                        "},RHS:{", rhs.ToString(), "}}");
  }
};

// Represents the parsed while loop condition. The loop induction variable may
// either be used in a comparison or returned directly, i.e., NoOp. In the case
// of NoOp, it contains the parameter index and initial value of the loop
// induction variable.
using WhileCondComparisonOrNoOp =
    std::variant<WhileCondComparison, ParamIndexAndValue>;

std::optional<ParamIndexAndValue> ParseComparisonOperand(
    const HloInstruction* operand,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  if (operand->opcode() == HloOpcode::kCopy ||
      operand->opcode() == HloOpcode::kCopyStart ||
      operand->opcode() == HloOpcode::kCopyDone) {
    return ParseComparisonOperand(operand->operand(0), precomputed_analyses);
  }
  std::optional<int64_t> param_index;
  if (Match(operand, match::GetTupleElement().WithOperand(
                         0, match::Parameter().WithParameterNum(0)))) {
    param_index = operand->tuple_index();
  }
  std::optional<DynamicOrStaticInteger> operand_value =
      GetInstructionValueAsInteger(operand, precomputed_analyses);
  if (!param_index.has_value() && !operand_value.has_value()) {
    return std::nullopt;
  }
  return ParamIndexAndValue{param_index, operand_value};
}

std::optional<WhileCondComparisonOrNoOp> PatternMatchLoopCondComparison(
    const HloInstruction* comparison,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  CHECK_EQ(comparison->opcode(), HloOpcode::kCompare);
  std::optional<ParamIndexAndValue> lhs =
      ParseComparisonOperand(comparison->operand(0), precomputed_analyses);
  std::optional<ParamIndexAndValue> rhs =
      ParseComparisonOperand(comparison->operand(1), precomputed_analyses);
  if (!lhs.has_value() || !rhs.has_value()) {
    return std::nullopt;
  }
  return WhileCondComparison{comparison->comparison_direction(),
                             *std::move(lhs), *std::move(rhs)};
}
// Finds the while loop condition comparison by matching the loop condition root
// with known patterns.
std::optional<WhileCondComparisonOrNoOp> PatternMatchLoopCondRoot(
    const HloInstruction* loop_cond_root,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  if (loop_cond_root->opcode() == HloOpcode::kCopy) {
    return PatternMatchLoopCondRoot(loop_cond_root->operand(0),
                                    precomputed_analyses);
  }
  if (loop_cond_root->opcode() == HloOpcode::kCopyDone) {
    return PatternMatchLoopCondRoot(loop_cond_root->operand(0)->operand(1),
                                    precomputed_analyses);
  }
  if (loop_cond_root->opcode() == HloOpcode::kCompare) {
    // Base pattern #1: gte-0 comp gte-1
    // Base pattern #2: constant comp gte
    // Base pattern #3: gte comp constant
    return PatternMatchLoopCondComparison(loop_cond_root, precomputed_analyses);
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
    const HloInstruction* call_instruction = loop_cond_root->operand(0);
    const HloComputation* to_apply = call_instruction->to_apply();
    const HloInstruction* to_apply_root = to_apply->root_instruction();
    if (Match(to_apply_root, match::Tuple())) {
      return PatternMatchLoopCondRoot(
          to_apply_root->operand(loop_cond_root->tuple_index()),
          precomputed_analyses);
    }
  }
  // Recursive pattern #2:
  // loop_cond_root is a GetTupleElement whose operand is a tuple.
  // We can recurse on the tuple's element as the new loop_cond_root.
  if (Match(loop_cond_root,
            match::GetTupleElement().WithOperand(0, match::Tuple()))) {
    const HloInstruction* new_cond_root =
        loop_cond_root->operand(0)->operand(loop_cond_root->tuple_index());
    return PatternMatchLoopCondRoot(new_cond_root, precomputed_analyses);
  }
  return std::nullopt;
}

std::optional<DynamicOrStaticInteger> PatternMatchInductionVarUpdate(
    const HloInstruction* induction_var_update, int64_t tuple_index,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  if (induction_var_update->opcode() == HloOpcode::kCopy) {
    return PatternMatchInductionVarUpdate(induction_var_update->operand(0),
                                          tuple_index, precomputed_analyses);
  }
  if (induction_var_update->opcode() == HloOpcode::kCopyDone) {
    return PatternMatchInductionVarUpdate(
        induction_var_update->operand(0)->operand(1), tuple_index,
        precomputed_analyses);
  }
  std::optional<ParamIndexAndValue> update_param_index_and_value =
      TryParsingInstructionAsParameterAndInteger(induction_var_update,
                                                 precomputed_analyses);

  if (update_param_index_and_value.has_value()) {
    if (update_param_index_and_value->param_index.has_value()) {
      if (*update_param_index_and_value->param_index == tuple_index) {
        // Pattern: the induc_var is directly returned from the loop body with
        // no changes.
        VLOG(3) << "PatternMatchInductionVarUpdate, pattern: [induc_var].";
        return DynamicOrStaticInteger{/*static_value=*/0};
      } else {
        VLOG(3)
            << "PatternMatchInductionVarUpdate, induction variable is set to "
               "another parameter value. Parsed update: "
            << update_param_index_and_value->ToString();
        return std::nullopt;
      }
    }
    if (update_param_index_and_value->value.has_value() &&
        !update_param_index_and_value->value->is_dynamic()) {
      VLOG(3) << "PatternMatchInductionVarUpdate, induction variable is set to "
                 "a constant. Parsed update: "
              << update_param_index_and_value->ToString();
      return std::nullopt;
    }
  }

  if (induction_var_update->opcode() != HloOpcode::kAdd &&
      induction_var_update->opcode() != HloOpcode::kSubtract) {
    return std::nullopt;
  }
  bool negate_update = induction_var_update->opcode() == HloOpcode::kSubtract;
  const HloInstruction* update_lhs = induction_var_update->operand(0);
  VLOG(3) << "PatternMatchInductionVarUpdate, LHS: " << update_lhs->ToString();
  std::optional<ParamIndexAndValue> update_lhs_param_index_and_value =
      TryParsingInstructionAsParameterAndInteger(update_lhs,
                                                 precomputed_analyses);

  const HloInstruction* update_rhs = induction_var_update->operand(1);
  VLOG(3) << "PatternMatchInductionVarUpdate, RHS: " << update_rhs->ToString();
  std::optional<ParamIndexAndValue> update_rhs_param_index_and_value =
      TryParsingInstructionAsParameterAndInteger(update_rhs,
                                                 precomputed_analyses);

  if (!update_lhs_param_index_and_value.has_value() ||
      !update_lhs_param_index_and_value->value.has_value() ||
      !update_rhs_param_index_and_value.has_value() ||
      !update_rhs_param_index_and_value->value.has_value()) {
    VLOG(3) << "PatternMatchInductionVarUpdate, failed to parse operands. "
               "Induction var update instruction: "
            << induction_var_update->ToString();
    return std::nullopt;
  }

  VLOG(3) << "update_lhs: " << update_lhs->ToString();
  VLOG(3) << "update_rhs: " << update_rhs->ToString();

  if (update_lhs_param_index_and_value->param_index.has_value() &&
      *update_lhs_param_index_and_value->param_index == tuple_index &&
      update_lhs_param_index_and_value->value->is_dynamic()) {
    if (update_rhs_param_index_and_value->value->is_dynamic()) {
      return update_rhs_param_index_and_value->value;
    }
    int64_t update_value =
        *update_rhs_param_index_and_value->value->static_value;
    return negate_update
               ? DynamicOrStaticInteger{/*static_value=*/-update_value}
               : DynamicOrStaticInteger{/*static_value=*/update_value};
  }

  if (update_rhs_param_index_and_value->param_index.has_value() &&
      *update_rhs_param_index_and_value->param_index == tuple_index &&
      update_rhs_param_index_and_value->value->is_dynamic() && !negate_update) {
    return update_lhs_param_index_and_value->value;
  }
  VLOG(3) << "Failed to pattern match induction variable update.";
  return std::nullopt;
}

// Tries to parse the loop body to find how the induction variable is updated
// using pattern matching.
std::optional<DynamicOrStaticInteger>
PatternMatchInductionVarUpdateFromLoopBodyRoot(
    const HloInstruction* loop_body_root, int64_t tuple_index,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  if (loop_body_root->opcode() != HloOpcode::kTuple ||
      loop_body_root->operand_count() <= tuple_index) {
    return std::nullopt;
  }
  const HloInstruction* induction_var_update =
      loop_body_root->operand(tuple_index);
  return PatternMatchInductionVarUpdate(induction_var_update, tuple_index,
                                        precomputed_analyses);
}

std::optional<bool> PatternMatchLoopCondVarOverride(
    const HloInstruction* loop_body_root, int64_t tuple_index) {
  if (!Match(loop_body_root, match::Tuple()) ||
      loop_body_root->operand_count() <= tuple_index) {
    return std::nullopt;
  }
  const HloInstruction* cond_var_override =
      loop_body_root->operand(tuple_index);
  return GetInstructionStaticValueAsBool(cond_var_override);
}

// A convenience wrapper to compute the while loop's argument's init value at
// the given tuple_index. If the init value depends on parameters to the
// while loop's parent computation or infeed, we consider the init value
// dynamic.
std::optional<DynamicOrStaticInteger> EvaluateWhileLoopParamInitValue(
    const HloInstruction* param_instruction, int64_t tuple_index) {
  if (param_instruction->opcode() != HloOpcode::kTuple) {
    return std::nullopt;
  }
  const HloInstruction* element_instruction =
      param_instruction->operand(tuple_index);
  return GetInstructionValueAsInteger(element_instruction,
                                      /*precomputed_analyses=*/{});
}

}  // namespace

namespace internal {

constexpr absl::string_view kEvalErrorDetailUrl = "EvalErrorDetailUrl";

std::optional<EvalErrorDetail> ParseEvalErrorDetail(const absl::Status& error) {
  auto error_detail = error.GetPayload(kEvalErrorDetailUrl);
  if (!error_detail.has_value() || error_detail->empty()) {
    return std::nullopt;
  }
  return static_cast<EvalErrorDetail>(
      absl::little_endian::Load32(error_detail->Flatten().data()));
}

}  // namespace internal

std::optional<ParsedWhileLoop> HandleNoopLoopCondition(
    const ParamIndexAndValue& parameter_index_and_value,
    const HloInstruction* while_operand, const HloComputation* while_body) {
  CHECK(parameter_index_and_value.param_index.has_value());
  int64_t loop_cond_var_index = *parameter_index_and_value.param_index;
  std::optional<DynamicOrStaticInteger> noop_value =
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
    std::optional<bool> updated_loop_cond_var = PatternMatchLoopCondVarOverride(
        while_body->root_instruction(), loop_cond_var_index);
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

int64_t ComputeTripCountFromComparison(int64_t init, int64_t bound,
                                       int64_t update,
                                       bool comparison_with_equal) {
  if (comparison_with_equal && init > bound) {
    return 0;
  }
  if (!comparison_with_equal && init >= bound) {
    return 0;
  }
  int64_t distance = bound - init;
  int64_t trip_count = (distance + update - 1) / update;
  CHECK_GE(trip_count, 0);
  // Additional logic to deal with equal comparison.
  if (comparison_with_equal && (bound - init) % update == 0) {
    trip_count += 1;
  }
  return trip_count;
}

std::optional<ParsedWhileLoop> HandleStaticLoopComparison(
    int64_t lhs, int64_t rhs, Comparison::Direction comparison_direction) {
  if ((comparison_direction == Comparison::Direction::kLt && lhs < rhs) ||
      (comparison_direction == Comparison::Direction::kLe && lhs <= rhs) ||
      (comparison_direction == Comparison::Direction::kGt && lhs > rhs) ||
      (comparison_direction == Comparison::Direction::kGe && lhs >= rhs) ||
      (comparison_direction == Comparison::Direction::kEq && lhs == rhs) ||
      (comparison_direction == Comparison::Direction::kNe && lhs != rhs)) {
    // This is an infinite loop and we set trip_count to -1.
    // There is no induction variable.
    return ParsedWhileLoop{ParsedStaticWhileLoop{/*trip_count=*/-1,
                                                 /*induction_var_index=*/-1,
                                                 /*induction_var_init_value=*/0,
                                                 /*step_size=*/0,
                                                 /*loop_bound=*/1}};
  }
  return ParsedWhileLoop{ParsedStaticWhileLoop{/*trip_count=*/0,
                                               /*induction_var_index=*/-1,
                                               /*induction_var_init_value=*/0,
                                               /*step_size=*/0,
                                               /*loop_bound=*/0}};
}

std::optional<ParsedWhileLoop> PatternMatchParseWhileLoop(
    const HloInstruction* while_op,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  VLOG(3) << "PatternMatchParseWhileLoop, while_op: " << while_op->name();
  const HloComputation* while_cond = while_op->while_condition();
  const HloComputation* while_body = while_op->while_body();
  const HloInstruction* while_operand = while_op->operand(0);
  // Try to parse the loop condition comparison.
  std::optional<WhileCondComparisonOrNoOp> loop_comparison_or_noop =
      PatternMatchLoopCondRoot(while_cond->root_instruction(),
                               precomputed_analyses);
  if (!loop_comparison_or_noop.has_value()) {
    return std::nullopt;
  }
  if (loop_comparison_or_noop->index() == 1) {
    return HandleNoopLoopCondition(
        std::get<ParamIndexAndValue>(*loop_comparison_or_noop), while_operand,
        while_body);
  }
  CHECK_EQ(loop_comparison_or_noop->index(), 0);
  WhileCondComparison loop_comparison =
      std::get<WhileCondComparison>(*loop_comparison_or_noop);
  CHECK(loop_comparison.lhs.IsValid() && loop_comparison.rhs.IsValid());

  // We can't handle the case when the while loop argument is not a Tuple
  // instruction.
  if (while_operand->opcode() != HloOpcode::kTuple) {
    return std::nullopt;
  }

  if (!loop_comparison.lhs.value.has_value() ||
      !loop_comparison.rhs.value.has_value()) {
    return std::nullopt;
  }

  // We have either successfully parsed the init value for both LHS and RHS
  // or have returned as failure.
  CHECK(loop_comparison.lhs.value.has_value());
  CHECK(loop_comparison.rhs.value.has_value());

  VLOG(3) << loop_comparison.ToString();

  // If both operands of the loop condition comparison have dynamic value, the
  // trip count might be dynamic or static. This is a case that our existing
  // patterns could not yet handle, so we return std::nullopt.
  if (loop_comparison.lhs.value->is_dynamic() &&
      loop_comparison.rhs.value->is_dynamic()) {
    VLOG(3) << "Both operands of the loop condition comparison are dynamic.";
    return std::nullopt;
  }
  // We would have returned if both operands are dynamic. So there is at most
  // one dynamic operand, which is potentially the loop induction variable.
  CHECK(!loop_comparison.lhs.value->is_dynamic() ||
        !loop_comparison.rhs.value->is_dynamic());

  if (!loop_comparison.lhs.value->is_dynamic() &&
      !loop_comparison.rhs.value->is_dynamic()) {
    int64_t lhs_value = *loop_comparison.lhs.value->static_value;
    int64_t rhs_value = *loop_comparison.rhs.value->static_value;
    Comparison::Direction comparison_direction =
        loop_comparison.comparison_direction;
    return HandleStaticLoopComparison(lhs_value, rhs_value,
                                      comparison_direction);
  }
  std::optional<DynamicOrStaticInteger> induction_var_init;
  std::optional<DynamicOrStaticInteger> induction_var_update;
  bool lhs_is_induction_var = true;
  if (loop_comparison.lhs.value->is_dynamic()) {
    if (loop_comparison.lhs.param_index.has_value()) {
      VLOG(3) << "Comparison LHS is induction variable.";
      induction_var_init = EvaluateWhileLoopParamInitValue(
          while_operand, *loop_comparison.lhs.param_index);
      induction_var_update = PatternMatchInductionVarUpdateFromLoopBodyRoot(
          while_body->root_instruction(), *loop_comparison.lhs.param_index,
          precomputed_analyses);
      lhs_is_induction_var = true;
    }
  } else {
    CHECK(loop_comparison.rhs.value->is_dynamic());
    if (loop_comparison.rhs.param_index.has_value()) {
      VLOG(3) << "Comparison RHS is induction variable.";
      induction_var_init = EvaluateWhileLoopParamInitValue(
          while_operand, *loop_comparison.rhs.param_index);
      induction_var_update = PatternMatchInductionVarUpdateFromLoopBodyRoot(
          while_body->root_instruction(), *loop_comparison.rhs.param_index,
          precomputed_analyses);
      lhs_is_induction_var = false;
    }
  }

  if (!induction_var_init.has_value() || !induction_var_update.has_value()) {
    return std::nullopt;
  }
  VLOG(3) << "induction_var_init: " << induction_var_init->ToString();
  VLOG(3) << "induction_var_update: " << induction_var_update->ToString();
  if (induction_var_init->is_dynamic() || induction_var_update->is_dynamic()) {
    return kParsedDynamicWhileLoop;
  }

  int64_t init_value = *induction_var_init->static_value;
  int64_t update_value = *induction_var_update->static_value;
  Comparison::Direction comparison_direction =
      loop_comparison.comparison_direction;
  ParsedWhileLoop parsed_static_while_loop = ParsedWhileLoop{
      ParsedStaticWhileLoop{/*trip_count=*/0,
                            // Unassigned.
                            /*induction_var_index=*/-1,
                            /*induction_var_init_value=*/init_value,
                            /*step_size=*/update_value,
                            // Unassigned.
                            /*loop_bound=*/-1}};
  // Lhs is the induction variable.
  if (lhs_is_induction_var) {
    CHECK(loop_comparison.rhs.value.has_value() &&
          !loop_comparison.rhs.value->is_dynamic());
    int64_t bound = *loop_comparison.rhs.value->static_value;
    parsed_static_while_loop.static_while_loop->induction_var_index =
        *loop_comparison.lhs.param_index;
    parsed_static_while_loop.static_while_loop->loop_bound = bound;
    if (update_value > 0 &&
        (comparison_direction == Comparison::Direction::kLt ||
         comparison_direction == Comparison::Direction::kLe)) {
      int64_t trip_count = ComputeTripCountFromComparison(
          init_value, bound, update_value,
          comparison_direction == Comparison::Direction::kLe);
      parsed_static_while_loop.static_while_loop->trip_count = trip_count;
      return parsed_static_while_loop;
    }
    if (update_value < 0 &&
        (comparison_direction == Comparison::Direction::kGt ||
         comparison_direction == Comparison::Direction::kGe)) {
      int64_t trip_count = ComputeTripCountFromComparison(
          bound, init_value, -update_value,
          comparison_direction == Comparison::Direction::kGe);
      parsed_static_while_loop.static_while_loop->trip_count = trip_count;
      return parsed_static_while_loop;
    }
    return std::nullopt;
  }
  // Rhs is the induction variable.
  CHECK(loop_comparison.lhs.value.has_value() &&
        !loop_comparison.lhs.value->is_dynamic());
  int64_t bound = *loop_comparison.lhs.value->static_value;
  parsed_static_while_loop.static_while_loop->induction_var_index =
      *loop_comparison.rhs.param_index;
  parsed_static_while_loop.static_while_loop->loop_bound = bound;
  if (update_value > 0 &&
      (comparison_direction == Comparison::Direction::kGt ||
       comparison_direction == Comparison::Direction::kGe)) {
    int64_t trip_count = ComputeTripCountFromComparison(
        init_value, bound, update_value,
        comparison_direction == Comparison::Direction::kGe);
    parsed_static_while_loop.static_while_loop->trip_count = trip_count;
    return parsed_static_while_loop;
  }
  if (update_value < 0 &&
      (comparison_direction == Comparison::Direction::kLt ||
       comparison_direction == Comparison::Direction::kLe)) {
    int64_t trip_count = ComputeTripCountFromComparison(
        bound, init_value, -update_value,
        comparison_direction == Comparison::Direction::kLe);
    parsed_static_while_loop.static_while_loop->trip_count = trip_count;
    return parsed_static_while_loop;
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
  for (int i = PrimitiveType_MIN; i < PrimitiveType_ARRAYSIZE; ++i) {
    if (!primitive_util::IsArrayType(PrimitiveType{i})) {
      continue;
    }
    primitive_util::PrimitiveTypeSwitch<void>(
        [&](auto primitive_type) {
          if constexpr (primitive_util::IsArrayType(primitive_type)) {
            using NativeT = primitive_util::NativeTypeOf<primitive_type>;
            if constexpr (primitive_util::IsSignedIntegralType(
                              primitive_type)) {
              typed_visitors_[primitive_type] =
                  std::make_unique<HloEvaluatorTypedVisitor<NativeT, int64_t>>(
                      this);
            } else if constexpr (primitive_util::IsUnsignedIntegralType(
                                     primitive_type)) {
              typed_visitors_[primitive_type] =
                  std::make_unique<HloEvaluatorTypedVisitor<NativeT, uint64_t>>(
                      this);
            } else if constexpr (primitive_util::IsFloatingPointType(
                                     primitive_type) &&
                                 sizeof(NativeT) < sizeof(float)) {
              // Most of the evaluator computations we use don't support BF16
              // and F8 (e.g., std::ceil, std::tanh). To make evaluator work
              // with these dtypes, we set all elementwise computations to be
              // done in F32 and do BF16<->F32 or F8<->F32 conversion around the
              // input and the output of the computations.
              typed_visitors_[primitive_type] =
                  std::make_unique<HloEvaluatorTypedVisitor<NativeT, float>>(
                      this);
            } else {
              typed_visitors_[primitive_type] =
                  std::make_unique<HloEvaluatorTypedVisitor<NativeT>>(this);
            }
          }
        },
        PrimitiveType{i});
  }

  typed_visitors_[TUPLE] =
      std::make_unique<ConstFunctionVisitor>([](const HloInstruction*) {
        return Unimplemented(
            "HloEvaluatorTypedVisitor: unhandled primitive type: TUPLE.");
      });
  typed_visitors_[OPAQUE_TYPE] =
      std::make_unique<ConstFunctionVisitor>([](const HloInstruction*) {
        return Unimplemented(
            "HloEvaluatorTypedVisitor: unhandled primitive type: OPAQUE_TYPE.");
      });
  typed_visitors_[TOKEN] =
      std::make_unique<ConstFunctionVisitor>([](const HloInstruction*) {
        return Unimplemented(
            "HloEvaluatorTypedVisitor: unhandled primitive type: TOKEN.");
      });
}

absl::StatusOr<Literal> HloEvaluator::Evaluate(
    const HloComputation& computation, absl::Span<const Literal* const> args) {
  CHECK(computation.parent() != nullptr);
  XLA_VLOG_LINES(
      2, "HloEvaluator::Evaluate computation:\n" + computation.ToString());
  OnEvaluateComputation(computation);

  if (args.size() != computation.num_parameters()) {
    return InvalidArgument(
        "Expected %d argument%s, but got %d.", computation.num_parameters(),
        computation.num_parameters() == 1 ? "" : "s", args.size());
  }
  for (int64_t i = 0; i < args.size(); ++i) {
    const auto& computation_shape =
        computation.parameter_instruction(i)->shape();
    const auto& arg_shape = args[i]->shape();
    bool ignore_layout = !computation_shape.has_layout();
    if (!Shape::Equal()
             .IgnoreLayout(ignore_layout)
             .MinorToMajorOnlyInLayout()(computation_shape, arg_shape)) {
      return InvalidArgument(
          "Shape mismatch at parameter %d. Computation expected %s, but arg "
          "was %s.",
          i, ShapeUtil::HumanStringWithLayout(computation_shape),
          ShapeUtil::HumanStringWithLayout(arg_shape));
    }
  }

  // Reset evaluation state with the argument literals.
  ScopedEvaluateState evaluate_state(&state_, args);

  call_graph_cache_.reset();
  tuple_points_to_analysis_cache_.reset();

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
  if (VLOG_IS_ON(100)) {
    for (const HloInstruction* instr : computation.instructions()) {
      VLOG(100) << instr->name() << " = " << GetEvaluatedLiteralFor(instr);
    }
  }

  Literal result = ExtractEvaluatedLiteralFor(computation.root_instruction());
  if (!result.IsKnown()) {
    return MakeEvalErrorDueToParamOrInfeed(*computation.root_instruction());
  }
  return result;
}

absl::StatusOr<Literal> HloEvaluator::Evaluate(
    const HloInstruction* instruction, PrecomputedAnalyses precomputed_analyses,
    bool recursively_evaluate_nonconstant_operands) {
  ScopedEvaluateState evaluate_state(&state_);

  call_graph_cache_.reset();
  tuple_points_to_analysis_cache_.reset();
  auto enable_partial_evaluation_cleanup =
      absl::MakeCleanup([this] { enable_partial_evaluation_ = false; });
  enable_partial_evaluation_ = recursively_evaluate_nonconstant_operands;
  TF_RETURN_IF_ERROR(
      EvaluateInternal(instruction, precomputed_analyses, /*shape_index=*/{},
                       recursively_evaluate_nonconstant_operands));
  Literal result = ExtractEvaluatedLiteralFor(instruction);
  if (!result.IsKnown()) {
    return MakeEvalErrorDueToParamOrInfeed(*instruction);
  }
  return result;
}

bool HloEvaluator::TryEvaluate(const HloInstruction* instruction,
                               Literal* result,
                               bool recursively_evaluate_nonconstant_operands) {
  CHECK(result != nullptr);
  auto result_or = Evaluate(instruction, /*precomputed_analyses=*/{},
                            recursively_evaluate_nonconstant_operands);
  if (!result_or.ok()) {
    VLOG(1) << "TryEvaluate failed:" << result_or.status();
    return false;
  }

  *result = std::move(result_or).value();
  return true;
}

absl::StatusOr<Literal> HloEvaluator::EvaluateWithSubstitutions(
    const HloInstruction* instruction,
    const absl::flat_hash_map<const HloInstruction*, const LiteralBase*>&
        substitutions,
    bool recursively_evaluate_nonconstant_operands) {
  std::vector<std::unique_ptr<HloInstruction>> owned_operands;
  for (const HloInstruction* operand : instruction->operands()) {
    auto it = substitutions.find(operand);
    if (it == substitutions.end()) {
      if (recursively_evaluate_nonconstant_operands) {
        TF_ASSIGN_OR_RETURN(Literal value,
                            EvaluateWithSubstitutions(
                                operand, substitutions,
                                recursively_evaluate_nonconstant_operands));
        owned_operands.push_back(HloInstruction::CreateConstant(value.Clone()));
      } else {
        if (!operand->IsConstant()) {
          VLOG(2) << "EvaluateWithSubstitutions called when not all operands "
                     "are constant. Consider calling it with "
                     "`recursively_evaluate_non_constant_operands` true.";
        }
        owned_operands.push_back(operand->Clone());
      }
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
  // TODO(phawkins): it's unfortunate that we need to call set_parent() here.
  // It's probably better to avoid constructing new instructions here in the
  // first place.
  cloned_instruction->set_parent(
      const_cast<HloComputation*>(instruction->parent()));
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

absl::StatusOr<Literal> HloEvaluator::EvaluateElementwiseBinaryOp(
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

absl::StatusOr<Literal> HloEvaluator::EvaluateElementwiseTernaryOp(
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

absl::StatusOr<Literal> HloEvaluator::EvaluateElementwiseCompareOp(
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

absl::StatusOr<Literal> HloEvaluator::EvaluateElementwiseUnaryOp(
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

absl::StatusOr<Literal> HloEvaluator::EvaluateDotOp(
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

absl::Status HloEvaluator::EvaluateParameterFromCallerArgument(
    const HloInstruction* parameter, const ShapeIndex& shape_index,
    PrecomputedAnalyses analyses) {
  CHECK(!state_.has_evaluated(parameter));
  const HloComputation* parent_computation = parameter->parent();
  std::vector<HloInstruction*> computation_callers =
      analyses.call_graph->GetComputationCallers(parent_computation);
  // If the parent computation has multiple callers, we cannot determine from
  // which caller the arguments are passed.
  if (computation_callers.size() != 1) {
    return tsl::errors::FailedPrecondition(
        "The computation ", parent_computation->name(), " is called by ",
        computation_callers.size(),
        " callers and thus its argument value "
        "cannot be determined statically.");
  }
  const HloInstruction* computation_caller = computation_callers[0];
  const HloInstruction* caller_operand = computation_caller->operand(0);
  if (computation_caller->opcode() != HloOpcode::kWhile &&
      computation_caller->opcode() != HloOpcode::kCall) {
    return tsl::errors::FailedPrecondition(
        "The computation ", parent_computation->name(), " is called by ",
        "instruction ", computation_caller->name(),
        ", which is not yet supported.");
  }
  if (computation_caller->opcode() == HloOpcode::kWhile) {
    HloComputation* while_body = computation_caller->while_body();
    TF_ASSIGN_OR_RETURN(
        const LogicalBuffer* logical_buffer,
        analyses.tuple_points_to->GetBufferDefinedAt(
            while_body->parameter_instruction(parameter->parameter_number()),
            shape_index));
    const TuplePointsToAnalysis::BufferAliasVector& buffer_aliases =
        analyses.tuple_points_to->GetBufferAliases(*logical_buffer);
    bool unchanged_in_return = false;
    for (const BufferAlias& buffer_alias : buffer_aliases) {
      if (buffer_alias.instruction() == while_body->root_instruction() &&
          buffer_alias.index() == shape_index) {
        unchanged_in_return = true;
      }
    }
    if (!unchanged_in_return) {
      return MakeEvalErrorDueToParamOrInfeed(*parameter);
    }
  }
  TF_RETURN_IF_ERROR(
      EvaluateInternal(caller_operand, analyses, shape_index, true));
  const Literal& caller_operand_literal =
      GetEvaluatedLiteralFor(caller_operand);
  Literal literal =
      Literal::CreateFromShapeWithUnknownLeafArrays(parameter->shape());
  TF_RETURN_IF_ERROR(literal.CopyFrom(caller_operand_literal,
                                      /*dest_shape_index=*/shape_index,
                                      /*src_shape_index=*/shape_index));
  SetEvaluatedLiteralFor(parameter, std::move(literal));
  return absl::OkStatus();
}

std::vector<int64_t> HloEvaluator::GetS64Indices(
    absl::Span<HloInstruction* const> start_indices) {
  auto get_first_s64 = [&](const Literal& index) -> int64_t {
    return primitive_util::PrimitiveTypeSwitch<int64_t>(
        [&](auto primitive_type_constant) -> int64_t {
          if constexpr (primitive_util::IsIntegralType(
                            primitive_type_constant)) {
            return static_cast<int64_t>(
                index.GetFirstElement<NativeTypeOf<primitive_type_constant>>());
          }
          LOG(FATAL) << "GetS64Indices: unhandled primitive type for "
                     << PrimitiveType_Name(index.shape().element_type());
        },
        index.shape().element_type());
  };
  std::vector<int64_t> start;
  start.reserve(start_indices.size());
  for (HloInstruction* index : start_indices) {
    start.push_back(get_first_s64(GetEvaluatedLiteralFor(index)));
  }
  return start;
}

DimensionVector HloEvaluator::MakeDimMultipliers(const Shape& shape) {
  DimensionVector v(shape.rank());
  int64_t scale = 1;
  for (auto dim : LayoutUtil::MinorToMajor(shape)) {
    v[dim] = scale;
    scale *= shape.dimensions(dim);
  }
  return v;
}

absl::Status HloEvaluator::EvaluateInternal(
    const HloInstruction* instruction, PrecomputedAnalyses precomputed_analyses,
    const ShapeIndex& shape_index,
    bool recursively_evaluate_nonconstant_operands) {
  // Don't need to evaluate this instruction again if it has already been
  // evaluated.
  if (IsAlreadyEvaluated(instruction, shape_index)) {
    return absl::OkStatus();
  }

  if (!recursively_evaluate_nonconstant_operands) {
    if (!hlo_query::AllOperandsAreConstants(*instruction)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Not all operands are constants. Instruction: ",
                       instruction->ToString()));
    }
  } else {
    if (instruction->opcode() == HloOpcode::kGetTupleElement) {
      ShapeIndex new_shape_index = shape_index;
      new_shape_index.push_front(instruction->tuple_index());
      TF_RETURN_IF_ERROR(EvaluateInternal(
          instruction->operand(0), precomputed_analyses, new_shape_index,
          /*recursively_evaluate_nonconstant_operands=*/true));
    } else if (instruction->opcode() == HloOpcode::kTuple &&
               !shape_index.empty()) {
      ShapeIndex new_shape_index = shape_index;
      int64_t tuple_index = new_shape_index.front();
      new_shape_index.pop_front();
      TF_RETURN_IF_ERROR(
          EvaluateInternal(instruction->operand(tuple_index),
                           precomputed_analyses, new_shape_index,
                           /*recursively_evaluate_nonconstant_operands=*/true));
    } else if (instruction->opcode() == HloOpcode::kParameter) {
      CallGraph* call_graph =
          (precomputed_analyses.call_graph != nullptr)
              ? precomputed_analyses.call_graph
              : std::invoke([this, instruction]() -> CallGraph* {
                  call_graph_cache_ =
                      CallGraph::Build(instruction->GetModule());
                  return call_graph_cache_.get();
                });
      TuplePointsToAnalysis* tuple_points_to_analysis =
          (precomputed_analyses.tuple_points_to != nullptr)
              ? precomputed_analyses.tuple_points_to
              : std::invoke([this, instruction]() -> TuplePointsToAnalysis* {
                  absl::StatusOr<std::unique_ptr<TuplePointsToAnalysis>>
                      tuple_points_to_analysis =
                          TuplePointsToAnalysis::Run(instruction->GetModule());
                  if (!tuple_points_to_analysis.ok()) {
                    return nullptr;
                  }
                  tuple_points_to_analysis_cache_ =
                      *std::move(tuple_points_to_analysis);
                  return tuple_points_to_analysis_cache_.get();
                });
      if (call_graph && tuple_points_to_analysis) {
        absl::Status argument_eval_status = EvaluateParameterFromCallerArgument(
            instruction, shape_index, {tuple_points_to_analysis, call_graph});
        if (!argument_eval_status.ok()) {
          VLOG(4) << "Failed to evaluate parameter " << instruction->name()
                  << " from caller. Reason: " << argument_eval_status.message();
        } else {
          VLOG(4) << "Successfully evaluated parameter: "
                  << instruction->name();
        }
      }
    } else {
      for (HloInstruction* operand : instruction->operands()) {
        TF_RETURN_IF_ERROR(EvaluateInternal(
            operand, precomputed_analyses, /*shape_index=*/{},
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
          SetEvaluatedLiteralFor(instruction,
                                 Literal::CreateFromShapeWithUnknownLeafArrays(
                                     instruction->shape()));
          return absl::OkStatus();
        }
      }
    }
  }
  visitor_shape_index_ = shape_index;
  TF_RETURN_IF_ERROR(Preprocess(instruction));
  TF_RETURN_IF_ERROR(instruction->Visit(this));
  TF_RETURN_IF_ERROR(Postprocess(instruction));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleBitcast(const HloInstruction* bitcast) {
  Shape result_shape = bitcast->shape();

  // Allow effective scalars without layouts as the result is unambiguous.
  if (!result_shape.has_layout() &&
      ShapeUtil::IsEffectiveScalar(result_shape)) {
    result_shape = LayoutUtil::GetWithDefaultLayout(result_shape);
  }

  // In general, we require a layout to evaluate a bitcast: this is the only
  // operation where indexing is physical rather than logical.
  if (!result_shape.has_layout()) {
    return InvalidArgument(
        "Evaluator cannot evaluate bitcast for non-scalar operand without "
        "assigned layout.");
  }
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(result_shape));

  Literal result(result_shape);

  // Bitcast output is allowed to be smaller than the input if the backend-
  // specific buffer sizes for the input and output are the same. Since the HLO
  // evaluator doesn't have access to the backend-specific shape size function,
  // assume it's OK to bitcast if output <= input.
  const Literal& operand_literal = GetEvaluatedLiteralFor(bitcast->operand(0));
  TF_RET_CHECK(operand_literal.size_bytes() >= result.size_bytes());
  memcpy(result.untyped_data(), operand_literal.untyped_data(),
         result.size_bytes());
  SetEvaluatedLiteralFor(bitcast, std::move(result));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleBitcastConvert(const HloInstruction* convert) {
  const HloInstruction* operand = convert->operand(0);
  TF_ASSIGN_OR_RETURN(
      Literal result,
      GetEvaluatedLiteralFor(operand).BitcastConvert(convert->shape()));

  SetEvaluatedLiteralFor(convert, std::move(result));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleGetDimensionSize(
    const HloInstruction* get_dimension_size) {
  const HloInstruction* operand = get_dimension_size->operand(0);
  int64_t dim = get_dimension_size->dimension();
  if (dynamic_dimension_inference_ == nullptr) {
    return InvalidArgument(
        "Evaluator cannot evaluate get_dimension_size without "
        "set_dynamic_dimension_inference.");
  }
  const HloInstruction* dynamic_size =
      dynamic_dimension_inference_->GetDynamicSize(operand, {}, dim);
  if (dynamic_size != nullptr) {
    SetEvaluatedLiteralFor(get_dimension_size,
                           GetEvaluatedLiteralFor(dynamic_size).Clone());
    return absl::OkStatus();
  }

  const Shape& shape = get_dimension_size->operand(0)->shape();
  Literal output(ShapeUtil::MakeShape(S32, {}));
  output.PopulateWithValue(
      static_cast<int32_t>(shape.dimensions(get_dimension_size->dimension())));
  SetEvaluatedLiteralFor(get_dimension_size, std::move(output));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleSetDimensionSize(
    const HloInstruction* set_dimension_size) {
  const Literal& operand_literal =
      GetEvaluatedLiteralFor(set_dimension_size->operand(0));
  Literal result(set_dimension_size->shape());
  memcpy(result.untyped_data(), operand_literal.untyped_data(),
         operand_literal.size_bytes());
  const Literal& size_literal =
      GetEvaluatedLiteralFor(set_dimension_size->operand(1));
  result.SetDynamicSize(set_dimension_size->dimension(),
                        size_literal.Get<int32_t>({}));
  SetEvaluatedLiteralFor(set_dimension_size, std::move(result));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleParameter(const HloInstruction* parameter) {
  if (!IsAlreadyEvaluated(parameter, visitor_shape_index_)) {
    if (!enable_partial_evaluation_) {
      return tsl::errors::FailedPrecondition(
          "Failed to evaluate instruction since its operands are unknown "
          "or undetermined and partial evaluation is not enabled.");
    }
    SetEvaluatedLiteralFor(
        parameter,
        Literal::CreateFromShapeWithUnknownLeafArrays(parameter->shape()));
    return absl::OkStatus();
  }

  if (state_.has_args()) {
    // Nothing to do other than sanity checks. Parameters' values are stored in
    // the state_.args() array.
    CHECK_LT(parameter->parameter_number(), state_.args().size());
#ifndef NDEBUG
    const Literal* input_literal = state_.arg(parameter->parameter_number());
    VLOG(2) << "Parameter evaluated to: " << input_literal->ToString();
    bool check_layout = parameter->shape().has_layout();
    DCHECK(Shape::Equal()
               .IgnoreLayout(!check_layout)
               .MinorToMajorOnlyInLayout()(parameter->shape(),
                                           input_literal->shape()))
        << "parameter shape is: "
        << ShapeUtil::HumanStringWithLayout(parameter->shape())
        << ", but input literal shape is: "
        << ShapeUtil::HumanStringWithLayout(input_literal->shape());
#endif
  }

  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleInfeed(const HloInstruction* infeed) {
  if (!enable_partial_evaluation_) {
    return tsl::errors::FailedPrecondition(
        "Failed to evaluate instruction since its operands are unknown "
        "or undetermined and partial evaluation is not enabled.");
  }
  SetEvaluatedLiteralFor(
      infeed, Literal::CreateFromShapeWithUnknownLeafArrays(infeed->shape()));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleConstant(const HloInstruction*) {
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleReshape(const HloInstruction* reshape) {
  TF_ASSIGN_OR_RETURN(Literal literal,
                      GetEvaluatedLiteralFor(reshape->operand(0))
                          .Reshape(reshape->shape().dimensions()));
  SetEvaluatedLiteralFor(reshape, std::move(literal));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleTranspose(const HloInstruction* transpose) {
  SetEvaluatedLiteralFor(transpose,
                         GetEvaluatedLiteralFor(transpose->operand(0))
                             .Transpose(transpose->dimensions()));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleConcatenate(
    const HloInstruction* concatenate) {
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

  SetEvaluatedLiteralFor(concatenate, std::move(result_literal));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleIsFinite(const HloInstruction* is_finite) {
  auto operand = is_finite->operand(0);
  auto elem_ty = operand->shape().element_type();
  return primitive_util::PrimitiveTypeSwitch<absl::Status>(
      [&](auto primitive_type_constant) -> absl::Status {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          TF_ASSIGN_OR_RETURN(
              Literal literal,
              (ElementWiseUnaryOpImpl<bool, NativeT>(
                  is_finite,
                  [](NativeT elem_operand) {
                    return Eigen::numext::isfinite(elem_operand);
                  },
                  GetEvaluatedLiteralFor(operand))));
          SetEvaluatedLiteralFor(is_finite, std::move(literal));
          return absl::OkStatus();
        }
        return InvalidArgument(
            "expected element type in shape to be floating point, but got: %s",
            PrimitiveType_Name(elem_ty));
      },
      elem_ty);
}

absl::Status HloEvaluator::HandleReal(const HloInstruction* real) {
  auto operand = real->operand(0);
  return primitive_util::PrimitiveTypeSwitch<absl::Status>(
      [&](auto primitive_type_constant) -> absl::Status {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          TF_ASSIGN_OR_RETURN(
              Literal literal,
              (ElementWiseUnaryOpImpl<NativeT, NativeT>(
                  real, [](NativeT elem_operand) { return elem_operand; },
                  GetEvaluatedLiteralFor(operand))));
          SetEvaluatedLiteralFor(real, std::move(literal));
          return absl::OkStatus();
        }
        if constexpr (primitive_util::IsComplexType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          TF_ASSIGN_OR_RETURN(
              Literal literal,
              (ElementWiseUnaryOpImpl<typename NativeT::value_type, NativeT>(
                  real,
                  [](NativeT elem_operand) { return std::real(elem_operand); },
                  GetEvaluatedLiteralFor(operand))));
          SetEvaluatedLiteralFor(real, std::move(literal));
          return absl::OkStatus();
        }
        LOG(FATAL) << "HandleReal: unknown/unhandled primitive type: "
                   << PrimitiveType_Name(operand->shape().element_type());
      },
      operand->shape().element_type());
}

absl::Status HloEvaluator::HandleImag(const HloInstruction* imag) {
  auto operand = imag->operand(0);
  return primitive_util::PrimitiveTypeSwitch<absl::Status>(
      [&](auto primitive_type_constant) -> absl::Status {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          TF_ASSIGN_OR_RETURN(
              Literal literal,
              (ElementWiseUnaryOpImpl<NativeT, NativeT>(
                  imag, [](NativeT elem_operand) { return NativeT(0); },
                  GetEvaluatedLiteralFor(operand))));
          SetEvaluatedLiteralFor(imag, std::move(literal));
          return absl::OkStatus();
        }
        if constexpr (primitive_util::IsComplexType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          TF_ASSIGN_OR_RETURN(
              Literal literal,
              (ElementWiseUnaryOpImpl<typename NativeT::value_type, NativeT>(
                  imag,
                  [](NativeT elem_operand) { return std::imag(elem_operand); },
                  GetEvaluatedLiteralFor(operand))));
          SetEvaluatedLiteralFor(imag, std::move(literal));
          return absl::OkStatus();
        }
        LOG(FATAL) << "HandleImag: unknown/unhandled primitive type: "
                   << PrimitiveType_Name(operand->shape().element_type());
      },
      operand->shape().element_type());
}

absl::Status HloEvaluator::HandleComplex(const HloInstruction* complex) {
  const Literal& real = GetEvaluatedLiteralFor(complex->operand(0));
  const Literal& imag = GetEvaluatedLiteralFor(complex->operand(1));
  TF_RET_CHECK(ShapeUtil::Compatible(real.shape(), imag.shape()));

  Literal result(complex->shape());
  return primitive_util::PrimitiveTypeSwitch<absl::Status>(
      [&](auto primitive_type_constant) -> absl::Status {
        if constexpr (primitive_util::IsComplexType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          TF_RETURN_IF_ERROR(result.Populate<NativeT>(
              [&](absl::Span<const int64_t> multi_index) {
                return NativeT(
                    real.Get<typename NativeT::value_type>(multi_index),
                    imag.Get<typename NativeT::value_type>(multi_index));
              }));
          SetEvaluatedLiteralFor(complex, std::move(result));
          return absl::OkStatus();
        }
        LOG(FATAL) << "HandleComplex: unknown/unhandled primitive type: "
                   << PrimitiveType_Name(complex->shape().element_type());
      },
      complex->shape().element_type());
}

absl::Status HloEvaluator::HandleCompare(const HloInstruction* compare) {
  ComparisonDirection direction = compare->comparison_direction();
  ComparisonOrder order = compare->comparison_order();
  auto lhs = compare->operand(0);
  auto rhs = compare->operand(1);
  DCHECK(ShapeUtil::SameDimensions(compare->shape(), rhs->shape()) &&
         ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

  TF_RET_CHECK(lhs->shape().element_type() == rhs->shape().element_type());
  auto element_type = lhs->shape().element_type();
  Comparison comparison(direction, element_type, order);

  const Literal& lhs_literal = GetEvaluatedLiteralFor(lhs);
  const Literal& rhs_literal = GetEvaluatedLiteralFor(rhs);
  // Note here we switch on the operand's type.
  return primitive_util::PrimitiveTypeSwitch<absl::Status>(
      [&](auto primitive_type_constant) -> absl::Status {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          TF_ASSIGN_OR_RETURN(Literal literal,
                              Compare<NativeT>(compare->shape(), comparison,
                                               lhs_literal, rhs_literal));
          SetEvaluatedLiteralFor(compare, std::move(literal));
          return absl::OkStatus();
        }
        LOG(FATAL) << "HandleCompare: unknown primitive type: "
                   << PrimitiveType_Name(element_type);
      },
      element_type);
}

absl::Status HloEvaluator::HandleTuple(const HloInstruction* tuple) {
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
            &GetEvaluatedLiteralFor(tuple->operand(operand_index)));
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

  // Inline part of LiteralUtil::MakeTuple() that avoids creating the leaf
  // buffers; these buffers can be extremely large.
  std::vector<const Shape*> element_shapes;
  element_shapes.reserve(operand_literals.size());
  for (const auto* element : operand_literals) {
    element_shapes.push_back(&element->shape());
  }
  Literal new_result = Literal::CreateFromShapeWithUndeterminedLeafArrays(
      ShapeUtil::MakeTupleShapeWithPtrs(element_shapes));
  for (int i = 0, end = operand_literals.size(); i < end; ++i) {
    TF_RETURN_IF_ERROR(
        new_result.CopyFrom(*operand_literals[i], /*dest_shape_index=*/{i}));
  }

  if (state_.has_evaluated(tuple)) {
    CHECK(new_result.IsDetermined(visitor_shape_index_));
    Literal literal;
    TF_RETURN_IF_ERROR(
        literal.CopyFrom(std::move(new_result),
                         /*dest_shape_index=*/visitor_shape_index_,
                         /*src_shape_index=*/visitor_shape_index_));
    SetEvaluatedLiteralFor(tuple, std::move(literal));
  } else {
    SetEvaluatedLiteralFor(tuple, std::move(new_result));
  }
  return absl::OkStatus();
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
  explicit FftTransform(const HloInstruction* fft)
      : fft_type_(fft->fft_type()),
        fft_rank_(fft->fft_length().size()),
        fft_lengths_(fft->fft_length()) {
    // Make fft_lengths_[0] the minormost dimension.
    absl::c_reverse(fft_lengths_);
  }

  absl::Status ComputeFft(const HloInstruction* fft,
                          const Literal& input_literal,
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

    return absl::OkStatus();
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
        twiddles.reserve(length / 2);
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

  absl::Status CheckParameters(const Shape& input_shape,
                               const Shape& output_shape) {
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

    return absl::OkStatus();
  }

 private:
  const FftType fft_type_;
  const int64_t fft_rank_;
  std::vector<int64_t> fft_lengths_;
};

}  // namespace

absl::Status HloEvaluator::HandleFft(const HloInstruction* fft) {
  const Literal& input_literal = GetEvaluatedLiteralFor(fft->operand(0));
  Literal output_literal = Literal::CreateFromShape(fft->shape());

  FftTransform<complex128> transform(fft);
  TF_RETURN_IF_ERROR(transform.ComputeFft(fft, input_literal, &output_literal));
  SetEvaluatedLiteralFor(fft, std::move(output_literal));

  return absl::OkStatus();
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

    absl::flat_hash_map<int64_t, int64_t> start_indices_dims_to_output_dims =
        GetStartIndicesDimToOutputDimForExplicitBatchingDims(
            dim_numbers_.start_indices_batching_dims(),
            dim_numbers_.index_vector_dim(), dim_numbers_.offset_dims(),
            start_indices_.shape().rank(), output_shape.rank());
    for (int64_t i = 0; i < dim_numbers->operand_batching_dims().size(); ++i) {
      int64_t operand_dim = dim_numbers->operand_batching_dims(i);
      int64_t start_indices_dim = dim_numbers->start_indices_batching_dims(i);
      int64_t output_dim = start_indices_dims_to_output_dims[start_indices_dim];
      explicit_batch_dims_operand_dim_to_output_dim_[operand_dim] = output_dim;
    }
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
  absl::StatusOr<absl::Span<const int64_t>> operator()(
      absl::Span<const int64_t> output_index) {
    PropagateOutputIndexGatherDimsToIndexVectorIndex(output_index);
    TF_RETURN_IF_ERROR(FetchIndexVector());
    PropagateIndexVectorToInputIndex();
    PropagateExplicitBatchDimsToInputIndex(output_index);
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
  absl::Status FetchIndexVector() {
    int64_t index_vector_dim = dim_numbers_.index_vector_dim();
    for (int64_t i = 0, e = index_vector_.size(); i < e; i++) {
      index_vector_index_[index_vector_dim] = i;
      auto start_index = start_indices_.GetIntegralAsS64(index_vector_index_);
      TF_RET_CHECK(start_index.has_value());
      index_vector_[i] = *start_index;
    }
    return absl::OkStatus();
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

  void PropagateExplicitBatchDimsToInputIndex(
      absl::Span<const int64_t> output_index) {
    for (const auto [operand_dim, output_dim] :
         explicit_batch_dims_operand_dim_to_output_dim_) {
      input_index_[operand_dim] = output_index[output_dim];
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

  absl::flat_hash_map<int64_t, int64_t>
      explicit_batch_dims_operand_dim_to_output_dim_;

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
      const GatherDimensionNumbers& dim_numbers, const Shape& input_shape) {
    CHECK(absl::c_is_sorted(dim_numbers.offset_dims()));
    int64_t window_dim_count = 0;
    for (int64_t i = 0; i < input_shape.dimensions_size(); i++) {
      if (IsCollapsedOrBatchingDim(dim_numbers.collapsed_slice_dims(),
                                   dim_numbers.operand_batching_dims(), i)) {
        input_dim_value_to_output_index_.push_back(-1);
      } else {
        input_dim_value_to_output_index_.push_back(
            dim_numbers.offset_dims()[window_dim_count++]);
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
  absl::StatusOr<absl::Span<const int64_t>> operator()(
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
static absl::StatusOr<std::reference_wrapper<const Literal>>
ReshapedGatherIndices(int64_t index_vector_dim, const Literal& start_indices,
                      Literal* reshaped_start_indices) {
  if (start_indices.shape().dimensions_size() != index_vector_dim) {
    return std::cref(start_indices);
  }

  std::vector<int64_t> new_shape(start_indices.shape().dimensions().begin(),
                                 start_indices.shape().dimensions().end());
  new_shape.push_back(1);
  if (start_indices.shape().is_dynamic()) {
    // TODO(b/243182930): If we add support for dynamic reshape, remove this
    // check and the call to ToStatic().
    TF_ASSIGN_OR_RETURN(*reshaped_start_indices,
                        start_indices.ToStatic().Reshape(new_shape));
  } else {
    TF_ASSIGN_OR_RETURN(*reshaped_start_indices,
                        start_indices.Reshape(new_shape));
  }
  return std::cref(*reshaped_start_indices);
}

absl::Status HloEvaluator::HandleGather(const HloInstruction* gather) {
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
      gather->gather_dimension_numbers(), /*input_shape=*/operand.shape());

  const Shape& operand_shape = operand.shape();
  if (ShapeUtil::IsZeroElementArray(operand_shape)) {
    SetEvaluatedLiteralFor(gather, std::move(result));
    return absl::OkStatus();
  }

  auto gather_inner_loop_body =
      [&](absl::Span<const int64_t> output_window_index,
          absl::Span<const int64_t> input_gather_index,
          absl::Span<const int64_t> output_gather_index)
      -> absl::StatusOr<bool> {
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
    result.CopyElementFrom(operand, input_index, output_index);
    return true;
  };

  auto gather_outer_loop_body =
      [&](absl::Span<const int64_t> output_gather_index)
      -> absl::StatusOr<bool> {
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
  SetEvaluatedLiteralFor(gather, std::move(result));
  return absl::OkStatus();
}

namespace {
// Reshapes the scatter indices input to have a trailing degenerate `1`
// dimension if necessary.  Hands over the ownership of the newly created
// literal (if there is one) to `reshaped_indices`.
absl::StatusOr<std::reference_wrapper<const Literal>> ReshapedScatterIndices(
    int64_t index_vector_dim, const Literal& indices,
    Literal* reshaped_indices) {
  if (indices.shape().dimensions_size() != index_vector_dim) {
    return std::cref(indices);
  }

  std::vector<int64_t> new_shape(indices.shape().dimensions().begin(),
                                 indices.shape().dimensions().end());
  new_shape.push_back(1);
  if (indices.shape().is_dynamic()) {
    // TODO(b/243182930): If we add support for dynamic reshape, remove this
    // check and the call to ToStatic().
    TF_ASSIGN_OR_RETURN(*reshaped_indices,
                        indices.ToStatic().Reshape(new_shape));
  } else {
    TF_ASSIGN_OR_RETURN(*reshaped_indices, indices.Reshape(new_shape));
  }
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

    absl::flat_hash_map<int64_t, int64_t> scatter_indices_dims_to_update_dims =
        GetStartIndicesDimToOutputDimForExplicitBatchingDims(
            dim_numbers_.scatter_indices_batching_dims(),
            dim_numbers_.index_vector_dim(), dim_numbers_.update_window_dims(),
            scatter_indices_.shape().rank(), updates_rank);
    for (int64_t i = 0; i < dim_numbers.input_batching_dims().size(); ++i) {
      int64_t input_dim = dim_numbers.input_batching_dims(i);
      int64_t scatter_indices_dim =
          dim_numbers.scatter_indices_batching_dims(i);
      int64_t update_dim =
          scatter_indices_dims_to_update_dims[scatter_indices_dim];
      explicit_batch_dims_input_dim_to_update_dim_[input_dim] = update_dim;
    }
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
  absl::StatusOr<absl::Span<const int64_t>> operator()(
      absl::Span<const int64_t> update_index) {
    PropagateUpdateIndexScatterDimsToIndexVectorIndex(update_index);
    TF_RETURN_IF_ERROR(FetchIndexVector());
    PropagateIndexVectorToInputIndex();
    PropagateExplicitBatchDimsToInputIndex(update_index);
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
  absl::Status FetchIndexVector() {
    int64_t index_vector_dim = dim_numbers_.index_vector_dim();
    for (int64_t i = 0, e = index_vector_.size(); i < e; i++) {
      index_vector_index_[index_vector_dim] = i;
      index_vector_[i] =
          *scatter_indices_.GetIntegralAsS64(index_vector_index_);
    }
    return absl::OkStatus();
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

  void PropagateExplicitBatchDimsToInputIndex(
      absl::Span<const int64_t> update_index) {
    for (const auto& [input_dim, update_dim] :
         explicit_batch_dims_input_dim_to_update_dim_) {
      input_index_[input_dim] = update_index[update_dim];
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

  absl::flat_hash_map<int64_t, int64_t>
      explicit_batch_dims_input_dim_to_update_dim_;

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
      const ScatterDimensionNumbers& dim_numbers, int64_t input_rank) {
    CHECK(absl::c_is_sorted(dim_numbers.update_window_dims()));
    int64_t window_dim_count = 0;
    for (int64_t i = 0; i < input_rank; i++) {
      if (IsCollapsedOrBatchingDim(dim_numbers.inserted_window_dims(),
                                   dim_numbers.input_batching_dims(), i)) {
        input_dim_value_to_update_index_.push_back(-1);
      } else {
        input_dim_value_to_update_index_.push_back(
            dim_numbers.update_window_dims()[window_dim_count++]);
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
  absl::StatusOr<absl::Span<const int64_t>> operator()(
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

absl::Status HloEvaluator::HandleScatter(const HloInstruction* hlo) {
  auto* scatter = DynCast<HloScatterInstruction>(hlo);
  const ScatterDimensionNumbers& dim_numbers =
      scatter->scatter_dimension_numbers();
  absl::InlinedVector<const Literal*, 1> operands;
  operands.reserve(scatter->scatter_operand_count());
  for (const HloInstruction* operand_inst : scatter->scatter_operands()) {
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
  for (const HloInstruction* updates_inst : scatter->scatter_updates()) {
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
      scatter->scatter_dimension_numbers(), /*input_rank=*/operand_dims.size());

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

  // If `scater->to_apply()` computation simply forwards one of its parameter,
  // we can avoid evaluating it, and update the result literal with a scalar
  // extracted from the update literal.
  std::optional<int32_t> forward_parameter;
  const HloInstruction* scatter_root = scatter->to_apply()->root_instruction();
  if (scatter_root->opcode() == HloOpcode::kParameter) {
    CHECK_EQ(operands.size(), 1) << "Expected a scatter with a single operand";
    forward_parameter = scatter_root->parameter_number();
  }

  HloEvaluator embedded_evaluator;
  auto scatter_inner_loop_body =
      [&](absl::Span<const int64_t> update_window_index,
          absl::Span<const int64_t> input_scatter_index,
          absl::Span<const int64_t> update_scatter_index)
      -> absl::StatusOr<bool> {
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

    // Fast path: update result with a scalar extracted from the update literal.
    if (forward_parameter.has_value()) {
      DCHECK_LT(*forward_parameter, 2);
      // Forwarding parameter 0 is a no-op.
      if (*forward_parameter == 1) {
        result.CopyElementFrom(*updates[0], update_index, input_index);
      }
      return true;
    }

    // Slow path: evaluate the scatter computation to get an update value.
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
    TF_ASSIGN_OR_RETURN(
        Literal updated_result,
        embedded_evaluator.Evaluate(*scatter->to_apply(), to_apply_args));
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
      [&](absl::Span<const int64_t> update_scatter_index)
      -> absl::StatusOr<bool> {
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
  SetEvaluatedLiteralFor(scatter, std::move(result));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleBroadcast(const HloInstruction* broadcast) {
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
      Literal literal,
      operand.Broadcast(broadcast->shape(), broadcast->dimensions()));
  SetEvaluatedLiteralFor(broadcast, std::move(literal));

  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAfterAll(const HloInstruction* after_all) {
  SetEvaluatedLiteralFor(after_all, LiteralUtil::CreateToken());
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAddDependency(
    const HloInstruction* add_dependency) {
  // AddDedendency just forwards its zero-th operand.
  SetEvaluatedLiteralFor(
      add_dependency,
      GetEvaluatedLiteralFor(add_dependency->operand(0)).Clone());
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleGetTupleElement(
    const HloInstruction* get_tuple_element) {
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

  Literal literal =
      Literal(ShapeUtil::GetTupleElementShape(operand->shape(), index));
  TF_RETURN_IF_ERROR(literal.CopyFrom(operand_tuple_literal,
                                      /*dest_shape_index=*/{},
                                      /*src_shape_index=*/{index}));
  SetEvaluatedLiteralFor(get_tuple_element, std::move(literal));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCopy(const HloInstruction* copy) {
  // If only the element type is different, try converting the literal.
  if (copy->shape().element_type() !=
      copy->operand(0)->shape().element_type()) {
    TF_ASSIGN_OR_RETURN(Literal result,
                        GetEvaluatedLiteralFor(copy->operand(0))
                            .Convert(copy->shape().element_type()));
    TF_RET_CHECK(ShapeUtil::Compatible(copy->shape(), result.shape()));
    SetEvaluatedLiteralFor(copy, std::move(result));
  } else {
    TF_RET_CHECK(
        ShapeUtil::Compatible(copy->shape(), copy->operand(0)->shape()));
    SetEvaluatedLiteralFor(copy,
                           GetEvaluatedLiteralFor(copy->operand(0)).Clone());
  }
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAsyncStart(const HloInstruction* async_start) {
  std::vector<const Literal*> arg_literals;
  arg_literals.reserve(async_start->operands().size());
  for (auto operand : async_start->operands()) {
    const Literal& arg_literal = GetEvaluatedLiteralFor(operand);
    arg_literals.push_back(&arg_literal);
  }

  std::unique_ptr<HloEvaluator> embedded_evaluator =
      CreateEmbedded(max_loop_iterations_);
  embedded_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  TF_ASSIGN_OR_RETURN(
      Literal result,
      embedded_evaluator->Evaluate(*async_start->async_wrapped_computation(),
                                   arg_literals));

  Literal literal = Literal(async_start->shape());

  // Copy the operand values to the index {0, i} of the output.
  for (int i = 0; i < arg_literals.size(); ++i) {
    TF_RETURN_IF_ERROR(literal.CopyFrom(*arg_literals[i],
                                        /*dest_shape_index=*/{0, i},
                                        /*src_shape_index=*/{}));
  }
  // Move the output value to the index {1} of the output.
  TF_RETURN_IF_ERROR(
      literal.MoveFrom(std::move(result), /*dest_shape_index=*/{1}));

  SetEvaluatedLiteralFor(async_start, std::move(literal));

  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAsyncUpdate(
    const HloInstruction* async_update) {
  const Literal& operand_tuple_literal =
      GetEvaluatedLiteralFor(async_update->operand(0));
  Literal literal = Literal(async_update->shape());
  TF_RETURN_IF_ERROR(literal.CopyFrom(operand_tuple_literal,
                                      /*dest_shape_index=*/{},
                                      /*src_shape_index=*/{}));
  SetEvaluatedLiteralFor(async_update, std::move(literal));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAsyncDone(const HloInstruction* async_done) {
  const Literal& operand_tuple_literal =
      GetEvaluatedLiteralFor(async_done->operand(0));
  Literal literal = Literal(async_done->shape());
  TF_RETURN_IF_ERROR(literal.CopyFrom(operand_tuple_literal,
                                      /*dest_shape_index=*/{},
                                      /*src_shape_index=*/{1}));
  SetEvaluatedLiteralFor(async_done, std::move(literal));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCopyStart(const HloInstruction* copy_start) {
  if (copy_start->user_count() != 1 ||
      copy_start->users().at(0)->opcode() != HloOpcode::kCopyDone) {
    return absl::FailedPreconditionError(
        absl::StrCat("Cannot evaluate a kCopyStart that doesn't have a single "
                     "kCopyDone user. Instruction: ",
                     copy_start->ToString()));
  }

  // The context in index {2} is undefined, but since we can't represent
  // undefined values using a Literal, we just use 0. This should be safe though
  // since we ensure that the only user of a kCopyStart is a kCopyDone which
  // consumes the context. Also note that MakeTuple copies its arguments, so
  // this is memory-safe.
  const Literal context_literal = LiteralUtil::CreateR0<uint32_t>(0);
  Literal literal = LiteralUtil::MakeTuple(
      {&GetEvaluatedLiteralFor(copy_start->operand(0)),
       &GetEvaluatedLiteralFor(copy_start->operand(0)), &context_literal});
  SetEvaluatedLiteralFor(copy_start, std::move(literal));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCopyDone(const HloInstruction* copy_done) {
  const HloInstruction* operand = copy_done->operand(0);
  if (operand->opcode() != HloOpcode::kCopyStart) {
    return absl::FailedPreconditionError(
        absl::StrCat("Cannot evaluate a kCopyDone that doesn't have a "
                     "kCopyStart as operand. Instruction: ",
                     copy_done->ToString()));
  }

  const Literal& operand_tuple_literal = GetEvaluatedLiteralFor(operand);
  Literal literal =
      Literal(ShapeUtil::GetTupleElementShape(operand->shape(), /*index=*/0));
  TF_RETURN_IF_ERROR(literal.CopyFrom(operand_tuple_literal,
                                      /*dest_shape_index=*/{},
                                      /*src_shape_index=*/{0}));
  SetEvaluatedLiteralFor(copy_done, std::move(literal));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCall(const HloInstruction* call) {
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

  SetEvaluatedLiteralFor(call, std::move(result));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleFusion(const HloInstruction* fusion) {
  HloModuleConfig config;
  // Attach cloned computation to an empty HLO module so the existing ones are
  // not modified.
  HloModule empty_hlo_module("EmptyModuleForFusion", config,
                             std::make_unique<CompilationEnvironments>(
                                 fusion->GetModule()->comp_envs()));
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

  SetEvaluatedLiteralFor(fusion, std::move(result));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleConditional(
    const HloInstruction* conditional) {
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

  SetEvaluatedLiteralFor(conditional, std::move(result));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleConvert(const HloInstruction* convert) {
  const HloInstruction* operand = convert->operand(0);
  TF_RET_CHECK(ShapeUtil::SameDimensions(operand->shape(), convert->shape()));
  TF_ASSIGN_OR_RETURN(Literal result, GetEvaluatedLiteralFor(operand).Convert(
                                          convert->shape().element_type()));
  SetEvaluatedLiteralFor(convert, std::move(result));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleDynamicSlice(
    const HloInstruction* dynamic_slice) {
  auto operand = dynamic_slice->operand(0);
  auto start_indices = dynamic_slice->operand(1);
  auto result_shape = dynamic_slice->shape();
  TF_ASSIGN_OR_RETURN(
      auto inferred_return_shape,
      ShapeInference::InferDynamicSliceShape(
          operand->shape(),
          Cast<HloDynamicSliceInstruction>(dynamic_slice)->index_shapes(),
          dynamic_slice->dynamic_slice_sizes()));
  TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
      << "return shape is set to: " << ShapeUtil::HumanString(result_shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);
  TF_RET_CHECK(
      primitive_util::IsIntegralType(start_indices->shape().element_type()));

  const Literal& operand_literal = GetEvaluatedLiteralFor(operand);

  std::vector<int64_t> start =
      GetS64Indices(absl::MakeConstSpan(dynamic_slice->operands()).subspan(1));

  // Clamp the start indices so the slice is in-bounds w.r.t the operand.
  for (int64_t i = 0; i < start.size(); ++i) {
    start[i] = std::min<int64_t>(
        std::max(int64_t{0}, start[i]),
        operand_literal.shape().dimensions(i) - result_shape.dimensions(i));
  }

  std::vector<int64_t> operand_index(start.size());
  Literal result(result_shape);
  const size_t element_byte_size =
      primitive_util::ByteWidth(result_shape.element_type());
  auto* operand_base = static_cast<const char*>(operand_literal.untyped_data());
  auto func = [&](void* dest, absl::Span<const int64_t> result_index) {
    for (int64_t i = 0; i < operand_index.size(); ++i) {
      CHECK_GE(result_index[i] + start[i], 0);
      operand_index[i] = result_index[i] + start[i];
    }

    auto* src = operand_base + (element_byte_size *
                                IndexUtil::MultidimensionalIndexToLinearIndex(
                                    operand_literal.shape(), operand_index));

    std::memcpy(dest, src, element_byte_size);
    return true;
  };
  TF_RETURN_IF_ERROR(result.PopulateInplace(func));
  SetEvaluatedLiteralFor(dynamic_slice, std::move(result));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleDynamicUpdateSlice(const HloInstruction* dus) {
  auto operand = dus->operand(0);
  auto update = dus->operand(1);
  auto start_indices = dus->operand(2);
  auto result_shape = dus->shape();
  TF_ASSIGN_OR_RETURN(
      auto inferred_return_shape,
      ShapeInference::InferDynamicUpdateSliceShape(
          operand->shape(), update->shape(),
          Cast<HloDynamicUpdateSliceInstruction>(dus)->index_shapes()));
  TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
      << "return shape is set to: " << ShapeUtil::HumanString(result_shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);
  TF_RET_CHECK(
      primitive_util::IsIntegralType(start_indices->shape().element_type()));
  TF_RET_CHECK(ShapeUtil::Compatible(result_shape, operand->shape()));

  const Literal& operand_literal = GetEvaluatedLiteralFor(operand);
  const Literal& update_literal = GetEvaluatedLiteralFor(update);

  auto result = operand_literal.Clone();
  const auto rank = result.shape().rank();
  std::vector<int64_t> start =
      GetS64Indices(absl::MakeConstSpan(dus->operands()).subspan(2));

  // Clamp the update start indices so the slice is in-bounds w.r.t the
  // operand.
  for (int64_t i = 0; i < rank; ++i) {
    start[i] = std::min<int64_t>(
        std::max<int64_t>(0, start[i]),
        result.shape().dimensions(i) - update_literal.shape().dimensions(i));
  }
  std::vector<int64_t> result_index(rank, 0);

  auto func = [&](absl::Span<const int64_t> update_index) {
    std::transform(update_index.begin(), update_index.end(), start.begin(),
                   result_index.begin(), std::plus<int64_t>());
    result.CopyElementFrom(update_literal, update_index, result_index);
    return true;
  };

  std::vector<int64_t> base(update_literal.shape().dimensions_size(), 0);
  std::vector<int64_t> step(update_literal.shape().dimensions_size(), 1);
  ShapeUtil::ForEachIndexNoStatus(update_literal.shape(), base,
                                  update_literal.shape().dimensions(), step,
                                  func);
  SetEvaluatedLiteralFor(dus, std::move(result));

  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleSelect(const HloInstruction* select) {
  const auto& pred = GetEvaluatedLiteralFor(select->operand(0));
  const auto& on_true = GetEvaluatedLiteralFor(select->operand(1));
  const auto& on_false = GetEvaluatedLiteralFor(select->operand(2));

  // If predicate is of scalar type, no element-wise selection would be needed.
  if (ShapeUtil::IsScalar(pred.shape())) {
    if (pred.Get<bool>({})) {
      SetEvaluatedLiteralFor(select, on_true.Clone());
    } else {
      SetEvaluatedLiteralFor(select, on_false.Clone());
    }
    return absl::OkStatus();
  }

  return DefaultAction(select);
}

namespace {

absl::StatusOr<Literal> CreateScalarLiteral(int64_t value,
                                            PrimitiveType element_type) {
  return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<Literal> {
        if constexpr (primitive_util::IsIntegralType(primitive_type_constant)) {
          return LiteralUtil::CreateR0(
              static_cast<NativeTypeOf<primitive_type_constant>>(value));
        }
        return InvalidArgument("Unsupported element type.");
      },
      element_type);
}

// Parses the while loop if it matches one of the known patterns. Returns the
// value of the loop induction variable after the loop execution if the loop is
// static.
absl::StatusOr<Literal> TryParseAndEvaluateWhileInductionVar(
    const HloInstruction* while_hlo) {
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(while_hlo, /*precomputed_analyses=*/{});
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
      while_result_element_ptrs.push_back(&while_result_elements[i]);
    }
  }
  return LiteralUtil::MakeTuple(while_result_element_ptrs);
}

}  // namespace

absl::Status HloEvaluator::HandleWhile(const HloInstruction* while_hlo) {
  const HloComputation* cond_comp = while_hlo->while_condition();
  const HloComputation* body_comp = while_hlo->while_body();
  // Initialize the loop carried valued with the input to the While instruction.
  auto lcv = GetEvaluatedLiteralFor(while_hlo->operand(0)).Clone();
  if (!lcv.IsKnown()) {
    std::optional<ParsedWhileLoop> parsed_while_loop =
        PatternMatchParseWhileLoop(while_hlo,
                                   /*precomputed_analyses=*/{});
    Literal literal =
        Literal::CreateFromShapeWithUnknownLeafArrays(while_hlo->shape());
    if (!parsed_while_loop.has_value() || parsed_while_loop->is_dynamic() ||
        visitor_shape_index_.size() != 1 ||
        parsed_while_loop->static_while_loop->induction_var_index !=
            visitor_shape_index_[0]) {
      return absl::OkStatus();
    }
    Shape induction_var_shape =
        ShapeUtil::GetSubshape(while_hlo->shape(), visitor_shape_index_);
    int64_t trip_count = parsed_while_loop->static_while_loop->trip_count;
    TF_ASSIGN_OR_RETURN(
        Literal induction_var_val,
        CreateScalarLiteral(trip_count, induction_var_shape.element_type()));
    TF_RETURN_IF_ERROR(literal.CopyFrom(
        induction_var_val, /*dest_shape_index=*/visitor_shape_index_,
        /*src_shape_index=*/{}));
    SetEvaluatedLiteralFor(while_hlo, std::move(literal));
    return absl::OkStatus();
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
      absl::StatusOr<Literal> result =
          TryParseAndEvaluateWhileInductionVar(while_hlo);
      if (result.ok()) {
        lcv = std::move(result).value();
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
  SetEvaluatedLiteralFor(while_hlo, std::move(lcv));
  return absl::OkStatus();
}

namespace {
template <typename NativeT>
Literal ExtractLiteralFromIndexPositions(const Literal& from,
                                         absl::Span<int64_t const> indices) {
  // We use a InlinedVector here because we need to convert it to an
  // absl::Span later, and this would not work with std::vector<bool>.
  absl::InlinedVector<NativeT, 10> values;
  for (int64_t index : indices) {
    values.push_back(from.Get<NativeT>({index}));
  }
  return LiteralUtil::CreateR1<NativeT>(values);
}

absl::StatusOr<Literal> ExtractFromIndexPositions(
    const Literal& from, absl::Span<int64_t const> indices) {
  PrimitiveType type = from.shape().element_type();
  return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<Literal> {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          return ExtractLiteralFromIndexPositions<
              NativeTypeOf<primitive_type_constant>>(from, indices);
        }
        return InvalidArgument("Unsupported type for Sort: %s",
                               PrimitiveType_Name(type));
      },
      type);
}

// For one particular placement of a window in a base shape (the placement is
// represented as `window_count_index`), iterates inside the window.
// Translates the window index into base index. If the base index is within
// bound, call `f` with the base index.
void IterateThroughWindow(
    const Shape& window_shape, const Window& window, const Shape& base_shape,
    const absl::Span<const int64_t> window_count_index,
    const std::function<void(absl::Span<const int64_t>)>& f) {
  const int64_t rank = base_shape.rank();
  DimensionVector window_index(rank);
  std::fill(window_index.begin(), window_index.end(), 0);
  do {
    DimensionVector base_index(rank);
    bool out_of_bound = false;
    for (int64_t i = 0; i < rank; ++i) {
      // Padding is applied to the dilated base. Say that padding is 3 and
      // dilation is 2 for some dimension. After applying base dilation and
      // padding, the dimension looks like:
      // P P P E D D E D D ... E D D E P P P
      // where E are the elements and D are the holes. So, the elements are
      // located in indices: padding + k*base_dilation for k = {0, 1, 2, ...}.
      // We are accessing elements in the transformed base at indices:
      // window_count_index * stride + window_index * window_dilation.
      // Solving for k gives us
      // (win_count_i * stride + win_i * win_dilation - pad) / base_dilation
      // When this is a natural number, we index an original element.
      // Otherwise, we index a 0 (pad or hole), and we don't need to apply
      // the callback f.
      base_index[i] = window_count_index[i] * window.dimensions(i).stride() +
                      window_index[i] * window.dimensions(i).window_dilation() -
                      window.dimensions(i).padding_low();
      if (base_index[i] % window.dimensions(i).base_dilation() != 0) {
        out_of_bound = true;
        break;
      }
      base_index[i] /= window.dimensions(i).base_dilation();
      if (base_index[i] < 0 || base_index[i] >= base_shape.dimensions(i)) {
        out_of_bound = true;
        break;
      }
    }
    if (!out_of_bound) {
      f(base_index);
    }
  } while (IndexUtil::BumpIndices(window_shape, absl::MakeSpan(window_index)));
}

template <typename Fp, typename Uint, typename ResultT>
absl::StatusOr<Literal> StochasticConvertOp(const Literal& operand_literal,
                                            const Literal& random_literal,
                                            const Shape& result_shape) {
  std::function<ResultT(Fp, Uint)> stochastic_convert_op =
      [](Fp operand, Uint random) -> ResultT {
    bool is_negative = static_cast<bool>(SignAndMagnitude(operand).first);
    if (Eigen::numext::isinf(operand)) {
      return is_negative ? std::numeric_limits<ResultT>::min()
                         : std::numeric_limits<ResultT>::max();
    }
    if (Eigen::numext::isnan(operand)) {
      return static_cast<ResultT>(0);
    }
    if (operand >= static_cast<Fp>(std::numeric_limits<ResultT>::max())) {
      return std::numeric_limits<ResultT>::max();
    }
    if (operand <= static_cast<Fp>(std::numeric_limits<ResultT>::min())) {
      return std::numeric_limits<ResultT>::min();
    }

    operand = Eigen::numext::abs(operand);

    // Gets the integral piece of the floating point input.
    auto truncated = static_cast<ResultT>(operand);

    // Removes the integral piece to obtain the fractional piece.
    Fp fractional = operand - static_cast<Fp>(truncated);
    if (fractional == Fp{0}) {
      // No rounding necessary.
      return is_negative ? -truncated : truncated;
    }

    // Compares fractional values against unsigned random values by
    // normalizing random values into [0, 1): fractional vs. (random /
    // random_max). This equals to comparing (fractional * random_max) vs.
    // random.
    auto fixed_fractional = static_cast<Uint>(std::ldexp(
        static_cast<double>(fractional), std::numeric_limits<Uint>::digits));

    // Rounds the integer output up if the fractional pieces is larger than
    // the input random number.
    if (random < fixed_fractional) {
      // This only happens when the operand is in the (min, -max) range and
      // should be rounded to min.
      if (truncated == std::numeric_limits<ResultT>::max()) {
        return std::numeric_limits<ResultT>::min();
      }
      truncated++;
    }
    return is_negative ? -truncated : truncated;
  };

  Literal result(result_shape);
  TF_RETURN_IF_ERROR(
      result.Populate<ResultT>([&](absl::Span<const int64_t> multi_index) {
        return stochastic_convert_op(operand_literal.Get<Fp>(multi_index),
                                     random_literal.Get<Uint>(multi_index));
      }));
  return std::move(result);
}

// Converts from primitive types to native types.
template <PrimitiveType operand_type, PrimitiveType random_type,
          PrimitiveType result_type>
absl::StatusOr<Literal> StochasticConvertOp(const Literal& operand_literal,
                                            const Literal& random_literal,
                                            const Shape& result_shape) {
  return StochasticConvertOp<
      typename primitive_util::PrimitiveTypeToNative<operand_type>::type,
      typename primitive_util::PrimitiveTypeToNative<random_type>::type,
      typename primitive_util::PrimitiveTypeToNative<result_type>::type>(
      operand_literal, random_literal, result_shape);
}

// Evaluates all possible paths of converting to different integers.
template <PrimitiveType operand_type, PrimitiveType random_type>
absl::StatusOr<Literal> StochasticConvertOp(const Literal& operand_literal,
                                            const Literal& random_literal,
                                            const Shape& result_shape) {
  return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<Literal> {
        if constexpr (primitive_util::IsSignedIntegralType(
                          primitive_type_constant)) {
          return StochasticConvertOp<operand_type, random_type,
                                     primitive_type_constant>(
              operand_literal, random_literal, result_shape);
        }
        // TODO(b/232442915): Enable converting big floats to small floats.
        return Unimplemented(
            "Stochastically converting from type %s to type %s is not "
            "implemented.",
            PrimitiveType_Name(operand_literal.shape().element_type()),
            PrimitiveType_Name(result_shape.element_type()));
      },
      result_shape.element_type());
}

absl::StatusOr<Literal> StochasticConvertOp(const Literal& operand_literal,
                                            const Literal& random_literal,
                                            const Shape& result_shape) {
  return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<Literal> {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          return StochasticConvertOp<
              primitive_type_constant,
              primitive_util::UnsignedIntegralTypeForBitWidth(
                  primitive_util::BitWidth(primitive_type_constant))>(
              operand_literal, random_literal, result_shape);
        }  // TODO(b/232442915): Enable converting big floats to small floats.
        return Unimplemented(
            "Stochastically converting from type %s to type %s is not "
            "implemented.",
            PrimitiveType_Name(operand_literal.shape().element_type()),
            PrimitiveType_Name(result_shape.element_type()));
      },
      operand_literal.shape().element_type());
}
}  // namespace

absl::Status HloEvaluator::HandleReverse(const HloInstruction* reverse) {
  const Shape& result_shape = reverse->shape();
  const auto reverse_dimensions = reverse->dimensions();

  auto operand = reverse->operand(0);
  TF_ASSIGN_OR_RETURN(
      auto inferred_return_shape,
      ShapeInference::InferReverseShape(operand->shape(), reverse_dimensions));

  TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
      << "return shape set to: " << ShapeUtil::HumanString(result_shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);

  const Literal& operand_literal = GetEvaluatedLiteralFor(operand);
  Literal result(result_shape);
  const size_t element_byte_size =
      primitive_util::ByteWidth(result_shape.element_type());
  auto* operand_base = static_cast<const char*>(operand_literal.untyped_data());
  TF_RETURN_IF_ERROR(result.PopulateInplaceParallel(
      [&](void* dest, absl::Span<const int64_t> out_index, int) {
        std::vector<int64_t> from_index(out_index.begin(), out_index.end());
        for (const int64_t dim : reverse_dimensions) {
          from_index[dim] = result_shape.dimensions(dim) - 1 - out_index[dim];
        }
        auto* src =
            operand_base +
            (element_byte_size * IndexUtil::MultidimensionalIndexToLinearIndex(
                                     operand_literal.shape(), from_index));
        std::memcpy(dest, src, element_byte_size);
      }));

  SetEvaluatedLiteralFor(reverse, std::move(result));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleSelectAndScatter(
    const HloInstruction* select_and_scatter) {
  auto operand = select_and_scatter->operand(0);
  auto source = select_and_scatter->operand(1);
  const Window& window = select_and_scatter->window();

  const Literal& init_literal =
      GetEvaluatedLiteralFor(select_and_scatter->operand(2));
  TF_RET_CHECK(ShapeUtil::IsScalar(init_literal.shape()));

  // Initialize result array with the init value.
  TF_ASSIGN_OR_RETURN(Literal result,
                      init_literal.Broadcast(select_and_scatter->shape(), {}));

  std::vector<int64_t> window_dimension_sizes;
  for (const auto& window_dimension : window.dimensions()) {
    window_dimension_sizes.push_back(window_dimension.size());
  }
  const Shape window_shape = ShapeUtil::MakeShape(
      operand->shape().element_type(), window_dimension_sizes);

  const HloComputation* select = select_and_scatter->select();
  const HloComputation* scatter = select_and_scatter->scatter();

  const Literal& operand_literal = GetEvaluatedLiteralFor(operand);
  const Literal& source_literal = GetEvaluatedLiteralFor(source);

  int64_t rank = operand_literal.shape().rank();

  HloEvaluator embedded_evaluator(max_loop_iterations_);
  DimensionVector source_index(rank, 0);

  do {
    // For each element in `source`, we place a window in `operand`. For each
    // window placement, we iterate inside the window twice:
    //
    // 1. Find the selected index by applying `select` function to all
    // elements. E.g., If the `select` function is GreaterEqual, the first
    // iteration through the window finds the biggest value and returns its
    // index.
    //
    // 2. Using the selected index, scatter value from `source` to result. We
    // do this by iterating through the window, and compare each index with
    // the selected index.
    std::optional<Literal> selected_val;
    std::optional<DimensionVector> selected_index;

    IterateThroughWindow(
        window_shape, window, operand_literal.shape(), source_index,
        [&](absl::Span<const int64_t> operand_index) {
          auto curr_val =
              LiteralUtil::GetScalarLiteral(operand_literal, operand_index);
          if (!selected_val.has_value()) {
            selected_val.emplace(curr_val.Clone());
            selected_index.emplace(operand_index.begin(), operand_index.end());
          }
          Literal computed_result =
              embedded_evaluator
                  .Evaluate(*select, {&selected_val.value(), &curr_val})
                  .value();
          bool selected = !computed_result.Get<bool>({});
          if (selected) {
            *selected_val = std::move(curr_val);
            selected_index.emplace(operand_index.begin(), operand_index.end());
          }
          embedded_evaluator.ResetVisitStates();
        });

    IterateThroughWindow(
        window_shape, window, operand_literal.shape(), source_index,
        [&](absl::Span<const int64_t> operand_index) {
          if (std::equal(operand_index.begin(), operand_index.end(),
                         selected_index->begin())) {
            auto source =
                LiteralUtil::GetScalarLiteral(source_literal, source_index);
            auto scattered =
                LiteralUtil::GetScalarLiteral(result, operand_index);
            Literal computed_result =
                embedded_evaluator.Evaluate(*scatter, {&source, &scattered})
                    .value();
            LiteralUtil::SetScalarLiteral(result, operand_index,
                                          computed_result);
            // Clear visit states so that the we can use the evaluator again
            // on the same computation.
            embedded_evaluator.ResetVisitStates();
          }
        });
  } while (
      IndexUtil::BumpIndices(source->shape(), absl::MakeSpan(source_index)));

  SetEvaluatedLiteralFor(select_and_scatter, std::move(result));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleSlice(const HloInstruction* slice) {
  auto operand = slice->operand(0);
  const Shape& shape = slice->shape();
  TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                      ShapeInference::InferSliceShape(
                          operand->shape(), slice->slice_starts(),
                          slice->slice_limits(), slice->slice_strides()));
  TF_RET_CHECK(ShapeUtil::Compatible(shape, inferred_return_shape))
      << "return shape set to: " << ShapeUtil::HumanString(shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);

  const int64_t rank = operand->shape().rank();
  const Literal& operand_literal = GetEvaluatedLiteralFor(operand);
  const size_t element_byte_size =
      primitive_util::ByteWidth(shape.element_type());
  auto* operand_base = static_cast<const char*>(operand_literal.untyped_data());
  auto func = [&](void* dest, absl::Span<const int64_t> out_index, int) {
    DimensionVector operand_index(rank);
    for (int64_t i = 0; i < rank; ++i) {
      operand_index[i] =
          slice->slice_starts(i) + out_index[i] * slice->slice_strides(i);
    }
    auto* src = operand_base + (element_byte_size *
                                IndexUtil::MultidimensionalIndexToLinearIndex(
                                    operand_literal.shape(), operand_index));
    std::memcpy(dest, src, element_byte_size);
  };

  Literal result(shape);
  TF_RETURN_IF_ERROR(result.PopulateInplaceParallel(func));
  SetEvaluatedLiteralFor(slice, std::move(result));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleSort(const HloInstruction* sort) {
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

  auto comparator =
      [sort](absl::Span<const Literal> literals_to_sort, int64_t a, int64_t b,
             HloEvaluator* embedded_evaluator) -> absl::StatusOr<bool> {
    absl::InlinedVector<Literal, 8> literals;
    literals.reserve(2 * sort->operand_count());
    for (int64_t i = 0; i < sort->operand_count(); ++i) {
      literals.push_back(
          LiteralUtil::GetScalarLiteral(literals_to_sort[i], {a}));
      literals.push_back(
          LiteralUtil::GetScalarLiteral(literals_to_sort[i], {b}));
    }
    absl::InlinedVector<const Literal*, 8> literal_ptrs;
    absl::c_transform(literals, std::back_inserter(literal_ptrs),
                      [](const Literal& literal) { return &literal; });

    TF_ASSIGN_OR_RETURN(
        auto computed_result,
        embedded_evaluator->Evaluate(*sort->to_apply(), literal_ptrs));
    // Clear visit states so that we can use the evaluator again
    // on the same computation.
    embedded_evaluator->ResetVisitStates();
    return computed_result.Get<bool>({});
  };
  auto less_than =
      [&comparator](absl::Span<const Literal> literals_to_sort, int64_t a,
                    int64_t b,
                    HloEvaluator* embedded_evaluator) -> absl::StatusOr<bool> {
    TF_ASSIGN_OR_RETURN(bool a_is_smaller,
                        comparator(literals_to_sort, a, b, embedded_evaluator));
#ifndef NDEBUG
    // Let's see if the comparator violates strict weak ordering.
    // N.B. This does not test transitivity.
    TF_ASSIGN_OR_RETURN(bool b_is_smaller,
                        comparator(literals_to_sort, b, a, embedded_evaluator));
    TF_RET_CHECK(!(b_is_smaller && a_is_smaller));
    TF_ASSIGN_OR_RETURN(bool b_is_reflexive,
                        comparator(literals_to_sort, b, b, embedded_evaluator));
    TF_RET_CHECK(!b_is_reflexive);
    TF_ASSIGN_OR_RETURN(bool a_is_reflexive,
                        comparator(literals_to_sort, a, a, embedded_evaluator));
    TF_RET_CHECK(!a_is_reflexive);
#endif
    return a_is_smaller;
  };
  std::function<absl::Status(absl::Span<const Literal>, absl::Span<int64_t>,
                             absl::Span<int64_t>, absl::Span<int64_t>,
                             std::vector<int64_t>&, HloEvaluator*)>
      merge = [&](absl::Span<const Literal> literals_to_sort,
                  absl::Span<int64_t> lhs, absl::Span<int64_t> rhs,
                  absl::Span<int64_t> output, std::vector<int64_t>& tmp,
                  HloEvaluator* embedded_evaluator) -> absl::Status {
    tmp.clear();
    tmp.reserve(output.size());
    // Keep picking between elements.
    while (!lhs.empty() && !rhs.empty()) {
      // If rhs < lhs, pick rhs. Otherwise, pick lhs. This should ensure
      // stability as lhs comes first in the array.
      TF_ASSIGN_OR_RETURN(bool rhs_is_smaller,
                          less_than(literals_to_sort, rhs.front(), lhs.front(),
                                    embedded_evaluator));
      if (rhs_is_smaller) {
        tmp.push_back(rhs.front());
        rhs.remove_prefix(1);
      } else {
        tmp.push_back(lhs.front());
        lhs.remove_prefix(1);
      }
    }
    // At least one of the two input arrays are now empty, we need to copy
    // the remaining elements.
    absl::c_copy(lhs, std::back_inserter(tmp));
    absl::c_copy(rhs, std::back_inserter(tmp));
    absl::c_copy(tmp, output.begin());
    return absl::OkStatus();
  };
  auto* env = tsl::Env::Default();
  const int max_parallelism = tsl::port::MaxParallelism();
  constexpr size_t kMinElementsPerThread{1024};
  const size_t useful_parallelism = std::min<size_t>(
      sort_dim_elements / kMinElementsPerThread, max_parallelism);
  const size_t work_per_thread = useful_parallelism > 1
                                     ? sort_dim_elements / useful_parallelism
                                     : std::numeric_limits<size_t>::max();
  std::function<absl::Status(absl::Span<const Literal>, absl::Span<int64_t>,
                             std::vector<int64_t>*, HloEvaluator*)>
      mergesort = [&merge, &mergesort, &less_than, this, env, work_per_thread](
                      absl::Span<const Literal> literals_to_sort,
                      absl::Span<int64_t> to_sort,
                      std::vector<int64_t>* scratch,
                      HloEvaluator* embedded_evaluator) -> absl::Status {
    // Base case: inputs with 0 or 1 elements are already sorted.
    if (to_sort.size() < 2) {
      return absl::OkStatus();
    }
    size_t halfway = to_sort.size() / 2;
    auto lhs = to_sort.subspan(/*pos=*/0, halfway);
    auto rhs = to_sort.subspan(/*pos=*/halfway);

    // Allocate an evaluator if we never got one, we will reuse an
    // allocator so long as we are not moving it between threads.
    std::unique_ptr<HloEvaluator> thread_local_embedded_evaluator;
    if (embedded_evaluator == nullptr) {
      thread_local_embedded_evaluator = CreateEmbedded(max_loop_iterations_);
      embedded_evaluator = thread_local_embedded_evaluator.get();
    }

    constexpr size_t kMinElementsForMergesort{9};
    if (to_sort.size() >= kMinElementsForMergesort) {
      std::unique_ptr<std::vector<int64_t>> thread_local_scratch;
      if (!scratch) {
        thread_local_scratch = std::make_unique<std::vector<int64_t>>();
        scratch = thread_local_scratch.get();
      }
      // Overlap sorting the LHS with the RHS if we have enough work to
      // do. The recursive call for to `mergesort(rhs)` will potentially
      // create more threads.
      absl::Status lhs_status;
      if (to_sort.size() >= work_per_thread) {
        std::unique_ptr<tsl::Thread> thread = absl::WrapUnique(env->StartThread(
            tsl::ThreadOptions(), "XLA_mergesort",
            [literals_to_sort, lhs, &mergesort, &lhs_status] {
              lhs_status = mergesort(literals_to_sort, lhs, nullptr, nullptr);
            }));
        TF_RETURN_IF_ERROR(
            mergesort(literals_to_sort, rhs, scratch, embedded_evaluator));
        // Here, `thread` will run its destructor ensuring that it is done
        // sorting `lhs`.
        thread.reset();
      } else {
        TF_RETURN_IF_ERROR(
            mergesort(literals_to_sort, rhs, scratch, embedded_evaluator));
        lhs_status =
            mergesort(literals_to_sort, lhs, scratch, embedded_evaluator);
      }
      TF_RETURN_IF_ERROR(lhs_status);
      TF_RETURN_IF_ERROR(merge(literals_to_sort, lhs, rhs, to_sort, *scratch,
                               embedded_evaluator));
    } else {
      // Do an insertion sort. Values to the left of `i` are sorted.
      // Any values larger than it in will be moved past `i`. Binary
      // search in [0, i) looking for the smallest value larger than `i`
      // which we will call `ub`. By induction, [ub, i) are all larger
      // than `i`.
      for (auto i = to_sort.begin(); i != to_sort.end(); ++i) {
        auto len = i - to_sort.begin();
        auto ub = to_sort.begin();
        auto needle = *i;
        while (len != 0) {
          auto half_len = len / 2;
          auto midpoint = ub + half_len;
          TF_ASSIGN_OR_RETURN(bool is_smaller,
                              less_than(literals_to_sort, needle, *midpoint,
                                        embedded_evaluator));
          if (is_smaller) {
            // Our needle is smaller than the midpoint, we need to shrink
            // the range by trimming the rightmost portion of it. We can't
            // exclude the midpoint value yet.
            len = half_len;
          } else {
            // Our needle is at least as big as the midpoint but we want
            // something larger, we can exclude the midpoint.
            ub = midpoint + 1;
            len -= half_len + 1;
          }
        }
        // Shift values larger than `i` to the right by 1 and insert `i`
        // in the new gap. Now the sorted range is [0, i].
        std::rotate(ub, i, i + 1);
      }
    }
    return absl::OkStatus();
  };

  // Iterate through each dimension except 'sort_dim'.
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
      key_shape, zero_base, key_shape.dimensions(), increment,
      [&](absl::Span<const int64_t> indices) -> absl::StatusOr<bool> {
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
        TF_RETURN_IF_ERROR(mergesort(literals_to_sort,
                                     absl::MakeSpan(indices_to_sort), nullptr,
                                     nullptr));
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
    SetEvaluatedLiteralFor(sort, std::move(result_literals[0]));
  } else {
    std::vector<const Literal*> literal_ptrs;
    absl::c_transform(result_literals, std::back_inserter(literal_ptrs),
                      [](const Literal& literal) { return &literal; });

    Literal result_tuple = LiteralUtil::MakeTuple(literal_ptrs);
    VLOG(3) << "HandleSort result_tuple: " << result_tuple.ToString();

    SetEvaluatedLiteralFor(sort, std::move(result_tuple));
  }
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleStochasticConvert(
    const HloInstruction* stochastic_convert) {
  const HloInstruction* operand = stochastic_convert->operand(0);
  const HloInstruction* random = stochastic_convert->operand(1);
  const Shape& result_shape = stochastic_convert->shape();
  TF_RET_CHECK(ShapeUtil::SameDimensions(operand->shape(), random->shape()));
  TF_RET_CHECK(ShapeUtil::SameDimensions(operand->shape(), result_shape));

  const Literal& operand_literal = GetEvaluatedLiteralFor(operand);
  const Literal& random_literal = GetEvaluatedLiteralFor(random);
  TF_ASSIGN_OR_RETURN(
      Literal literal,
      StochasticConvertOp(operand_literal, random_literal, result_shape));
  SetEvaluatedLiteralFor(stochastic_convert, std::move(literal));
  return absl::OkStatus();
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
static absl::StatusOr<bool> PerformReductionStep(
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

    arg_values[i].CopyElementFrom(*input_args[i], input_index, {});
    accumulators[i].CopyElementFrom(results[i], output_index, {});
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
      results[i].CopyElementFrom(computed_results[i], {}, output_index);
    }
  } else {
    results[0].CopyElementFrom(computed_result, {}, output_index);
  }

  return true;
}

static absl::StatusOr<bool> GenerateReduceOutputElement(
    bool is_tuple, bool use_fast_path, absl::Span<const int64_t> output_index,

    absl::Span<const Literal* const> init_values,
    absl::Span<const Literal* const> input_args, absl::Span<Literal> results,

    HloComputation* function, HloEvaluator* embedded_evaluator,

    absl::Span<const int64_t> arg_dim_steps,
    absl::Span<const int64_t> arg_dim_counts,
    absl::Span<const int64_t> result_to_arg_index) {
  bool use_fast_add = use_fast_path &&
                      ShapeUtil::ElementIsFloating(init_values[0]->shape()) &&
                      IsScalarAdd(function) && !is_tuple;

  const Shape& arg_shape = input_args[0]->shape();
  absl::Span<const int64_t> arg_dimensions = arg_shape.dimensions();
  std::vector<int64_t> base(arg_dimensions.size());
  for (int64_t i = 0; i < output_index.size(); ++i) {
    base[result_to_arg_index[i]] = output_index[i];
  }

  for (int64_t i = 0; i < results.size(); ++i) {
    results[i].CopyElementFrom(*init_values[i], {}, output_index);
  }

  if (use_fast_add) {
    double computed_result = *init_values[0]->GetAsDouble({});
    const Literal* input_arg0 = input_args[0];
    const Shape& shape = input_arg0->shape();
    absl::Span<const int64_t> minor_to_major = LayoutUtil::MinorToMajor(shape);

    static constexpr int kChunkSize = 512;
    int64_t linear_indices[kChunkSize];
    int n_linear_indices = 0;

    auto reduction_step = [&](absl::Span<const int64_t> input_index) -> bool {
      linear_indices[n_linear_indices++] =
          IndexUtil::MultidimensionalIndexToLinearIndex(shape, minor_to_major,
                                                        input_index);
      if (n_linear_indices == kChunkSize) {
        // Periodically compute partial sum to avoid linear_indices getting
        // large
        computed_result += *input_arg0->GetSumAsDouble(
            absl::MakeConstSpan(linear_indices, n_linear_indices));
        n_linear_indices = 0;
      }
      return true;
    };
    ShapeUtil::ForEachIndexNoStatus(arg_shape, base, arg_dim_counts,
                                    arg_dim_steps, reduction_step);
    if (n_linear_indices > 0) {
      // Add in sum over any final indices collected
      computed_result += *input_arg0->GetSumAsDouble(
          absl::MakeConstSpan(linear_indices, n_linear_indices));
    }
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

absl::Status HloEvaluator::HandleReduce(const HloInstruction* hlo) {
  const HloReduceInstruction* reduce = Cast<HloReduceInstruction>(hlo);
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

  const int num_threads = ShapeUtil::GetForEachIndexParallelThreadCount() + 1;
  std::vector<std::unique_ptr<HloEvaluator>> embedded_evaluators;
  embedded_evaluators.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    embedded_evaluators.push_back(CreateEmbedded(max_loop_iterations_));
  }

  absl::InlinedVector<Literal, 1> results(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    results[i] = Literal(is_tuple ? out_shape.tuple_shapes(i) : out_shape);
  }

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexParallelWithStatus(
      output_shape, [&](absl::Span<const int64_t> output_index, int thread_id) {
        return GenerateReduceOutputElement(
            is_tuple, use_fast_path_reduce_, output_index, init_values,
            input_args, absl::Span<Literal>(results), function,
            embedded_evaluators[thread_id + 1].get(), arg_dim_steps,
            arg_dim_counts, result_to_arg_index);
      }));

  if (is_tuple) {
    Literal tuple_result(inferred_return_shape);
    for (int64_t i = 0; i < num_args; ++i) {
      TF_CHECK_OK(tuple_result.MoveFrom(std::move(results[i]), {i}));
    }
    SetEvaluatedLiteralFor(reduce, std::move(tuple_result));
  } else {
    CHECK_EQ(results.size(), 1);
    SetEvaluatedLiteralFor(reduce, std::move(results[0]));
  }
  if (!ShapeUtil::Compatible(reduce->shape(), inferred_return_shape)) {
    TF_ASSIGN_OR_RETURN(
        Literal converted,
        GetEvaluatedLiteralFor(reduce).ConvertToShape(reduce->shape()));
    SetEvaluatedLiteralFor(reduce, std::move(converted));
  }
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleReduceWindow(const HloInstruction* hlo) {
  auto* reduce_window = Cast<HloReduceWindowInstruction>(hlo);
  const Window& window = reduce_window->window();
  HloComputation* function = reduce_window->to_apply();
  TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                      ShapeInference::InferReduceWindowShape(
                          reduce_window->input_shapes(),
                          reduce_window->init_value_shapes(), window,
                          /*to_apply_shape=*/function->ComputeProgramShape()));
  TF_RET_CHECK(
      ShapeUtil::Compatible(reduce_window->shape(), inferred_return_shape))
      << "return shape is set to: "
      << ShapeUtil::HumanStringWithLayout(reduce_window->shape())
      << " but is inferred to be: "
      << ShapeUtil::HumanStringWithLayout(inferred_return_shape);

  absl::InlinedVector<const Literal*, 2> input_literal_vec, init_literal_vec;
  auto input_arrays = reduce_window->inputs();
  auto init_values = reduce_window->init_values();
  int64_t num_args = input_arrays.size();
  for (int i = 0; i < num_args; ++i) {
    const Literal& input_literal = GetEvaluatedLiteralFor(input_arrays[i]);
    VLOG(3) << "HandleReduceWindow arg_literal: " << input_literal.ToString();
    input_literal_vec.push_back(&input_literal);
    const Literal& init_literal = GetEvaluatedLiteralFor(init_values[i]);
    VLOG(3) << "HandleReduceWindow init_literal: " << init_literal.ToString();
    TF_RET_CHECK(ShapeUtil::IsScalar(init_literal.shape()));
    init_literal_vec.push_back(&init_literal);
  }
  // Creates a Shape object from window, for iteration below.
  absl::InlinedVector<int64_t, 2> window_dimension_sizes;
  for (const auto& window_dimension : window.dimensions()) {
    window_dimension_sizes.push_back(window_dimension.size());
  }
  const Shape window_shape = ShapeUtil::MakeShape(
      input_arrays[0]->shape().element_type(), window_dimension_sizes);

  const int num_threads = ShapeUtil::GetForEachIndexParallelThreadCount() + 1;
  std::vector<std::unique_ptr<HloEvaluator>> embedded_evaluators;
  embedded_evaluators.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    embedded_evaluators.push_back(CreateEmbedded(max_loop_iterations_));
  }

  // For each resulting dimension, calculate and assign computed value.
  auto evaluate_impl = [&init_literal_vec, &window_shape, &window,
                        &input_literal_vec, &embedded_evaluators, function,
                        &inferred_return_shape](
                           absl::Span<const int64_t> output_index,
                           int thread_id) -> absl::InlinedVector<Literal, 2> {
    const int embedded_evaluator_index = thread_id + 1;
    CHECK_GE(embedded_evaluator_index, 0);
    CHECK_LT(embedded_evaluator_index, embedded_evaluators.size());
    HloEvaluator& embedded_evaluator =
        *embedded_evaluators[embedded_evaluator_index];
    absl::InlinedVector<Literal, 2> computed_result;
    computed_result.reserve(init_literal_vec.size());
    for (const auto* init : init_literal_vec) {
      computed_result.push_back(init->Clone());
    }
    IterateThroughWindow(
        window_shape, window, input_literal_vec[0]->shape(), output_index,
        [&](absl::Span<const int64_t> operand_index) -> void {
          absl::InlinedVector<const Literal*, 2> args;
          for (auto& curr_result_val : computed_result) {
            VLOG(2) << "Pushing:" << curr_result_val.ToString() << "\n";
            args.push_back(&curr_result_val);
          }
          absl::InlinedVector<Literal, 2> curr_val_literal_vec;
          curr_val_literal_vec.reserve(input_literal_vec.size());
          for (const auto* input_literal : input_literal_vec) {
            // Evaluate computation with specified literal operands.
            curr_val_literal_vec.push_back(Literal(ShapeUtil::MakeShape(
                input_literal->shape().element_type(), {})));
            curr_val_literal_vec.back().CopyElementFrom(*input_literal,
                                                        operand_index, {});
            VLOG(2) << "Pushing:" << curr_val_literal_vec.back().ToString()
                    << "\n";
            args.push_back(&curr_val_literal_vec.back());
          }
          computed_result[0] =
              embedded_evaluator.Evaluate(*function, args).value();
          VLOG(2) << "Computed result:" << computed_result[0].ToString()
                  << "\n";
          // Clear visit states so that the we can use the evaluate again
          // on the same computation.
          embedded_evaluator.ResetVisitStates();
          if (inferred_return_shape.IsTuple()) {
            auto decomposed = computed_result[0].DecomposeTuple();
            computed_result.clear();
            computed_result.reserve(decomposed.size());
            for (int i = 0; i < decomposed.size(); ++i) {
              computed_result.push_back(std::move(decomposed[i]));
            }
          }
        });
    VLOG(2) << "Final result size:" << computed_result.size() << "\n";
    for (const auto& res : computed_result) {
      VLOG(2) << res.ToString() << "\n";
    }
    return computed_result;
  };
  Literal result(inferred_return_shape);
  if (inferred_return_shape.IsTuple()) {
    absl::InlinedVector<Literal, 1> results(num_args);
    for (int64_t i = 0; i < num_args; ++i) {
      results[i] = Literal(inferred_return_shape.tuple_shapes(i));
    }
    ShapeUtil::ForEachIndexParallel(
        inferred_return_shape.tuple_shapes(0),
        [&results, &evaluate_impl](absl::Span<const int64_t> output_index,
                                   int thread_id) -> bool {
          absl::InlinedVector<Literal, 2> computed_result_vec =
              evaluate_impl(output_index, thread_id);
          for (int i = 0; i < computed_result_vec.size(); ++i) {
            // We are reading from `computed_result_vec[i]` at the top-level
            // literal index and writing to `results[i]` at `output_index`.
            // This is thread-safe because:
            //  - `results[i]` is not changing size.
            //  - `computed_result_vec[i]` is thread-local.
            //  - There is exactly one write to `results[i]` for each
            //    `output_index`.
            results[i].CopyElementFrom(computed_result_vec[i], {},
                                       output_index);
          }
          return true;
        });
    result = Literal::MoveIntoTuple(absl::MakeSpan(results));
    VLOG(2) << "Final result is:" << result.ToString() << "\n";
  } else {
    TF_RETURN_IF_ERROR(Apply<PopulateParallelImpl>(
        result, [&evaluate_impl](absl::Span<const int64_t> output_index,
                                 int thread_id) {
          return std::move(evaluate_impl(output_index, thread_id)[0]);
        }));
  }
  VLOG(2) << "Final result is:" << result.ToString() << "\n";
  SetEvaluatedLiteralFor(reduce_window, std::move(result));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleMap(const HloInstruction* map) {
  auto operands = map->operands();
  const HloComputation* computation = map->to_apply();

  Literal result(map->shape());

  HloEvaluator embedded_evaluator(max_loop_iterations_);
  TF_RETURN_IF_ERROR(
      Apply<PopulateImpl>(result, [&](absl::Span<const int64_t> multi_index) {
        std::vector<Literal> arg_literals;
        arg_literals.reserve(operands.size());

        // Construct scalar literal parameters to be passed to the map
        // computation.
        for (auto operand : operands) {
          const Literal& arg_literal = GetEvaluatedLiteralFor(operand);
          arg_literals.push_back(
              LiteralUtil::GetScalarLiteral(arg_literal, multi_index));
        }

        Literal computed_result =
            embedded_evaluator.Evaluate(*computation, arg_literals).value();
        // Clear visit states so that the we can use the evaluate again on
        // the same computation.
        embedded_evaluator.ResetVisitStates();

        return computed_result;
      }));
  SetEvaluatedLiteralFor(map, std::move(result));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCustomCall(const HloInstruction* custom_call) {
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

  SetEvaluatedLiteralFor(custom_call, std::move(output));
  return absl::OkStatus();
}

absl::Status HloEvaluator::Preprocess(const HloInstruction* hlo) {
  VLOG(3) << "About to visit HLO: " << hlo->ToString();
  if (!enable_partial_evaluation_) {
    for (const HloInstruction* operand : hlo->operands()) {
      if (!IsAlreadyEvaluated(operand) ||
          !GetEvaluatedLiteralFor(operand).IsKnown()) {
        return tsl::errors::FailedPrecondition(
            "Failed to evaluate instruction since its operands are unknown "
            "or undetermined and partial evaluation is not enabled.");
      }
    }
  }
  return ShapeUtil::ValidateShapeWithOptionalLayout(hlo->shape());
}

absl::Status HloEvaluator::Postprocess(const HloInstruction* hlo) {
  VLOG(3) << "Finished visiting " << hlo->ToString()
          << "; evaluated value is: " << GetEvaluatedLiteralFor(hlo).ToString();

  auto eq = Layout::Equal().MinorToMajorOnly();

  // Out of convenience the literal may have been produced with a different
  // layout. Relayout as indicated by the HLO instruction.
  const Shape& evaluated_shape = GetEvaluatedLiteralFor(hlo).shape();

  // If both shapes don't have a layout, we don't need to do anything.
  if (!evaluated_shape.has_layout() && !hlo->shape().has_layout()) {
    return absl::OkStatus();
  }

  // If both shapes have the same layout, we don't need to do anything.
  if (evaluated_shape.has_layout() && hlo->shape().has_layout() &&
      eq(evaluated_shape.layout(), hlo->shape().layout())) {
    return absl::OkStatus();
  }

  // At this point we know that the shapes have different layouts and we need to
  // relayout the evaluated literal.
  Shape shape = hlo->shape();

  if (shape.IsArray() && !shape.has_layout()) {
    *shape.mutable_layout() = LayoutUtil::GetDefaultLayoutForShape(shape);
  }
  if (evaluated_shape.has_layout() && shape.has_layout() &&
      !eq(evaluated_shape.layout(), shape.layout())) {
    SetEvaluatedLiteralFor(hlo, GetEvaluatedLiteralFor(hlo).Relayout(shape));
  }

  return absl::OkStatus();
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
  auto result = std::make_unique<Array2D<T>>(m, n);
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

std::unique_ptr<Array2D<uint8_t>> HloEvaluator::MatmulArray2D(
    const Array2D<uint8_t>& lhs, const Array2D<uint8_t>& rhs) {
  return MatmulArray2DImpl<uint8_t>(
      lhs, rhs, __xla_cpu_runtime_EigenSingleThreadedMatMulU8);
}

/* static */ std::unique_ptr<Array2D<float>> Array2DF8E5M2ToF32(
    const Array2D<tsl::float8_e5m2>& input) {
  auto result = std::make_unique<Array2D<float>>(input.height(), input.width());
  for (int64_t rowno = 0; rowno < input.height(); ++rowno) {
    for (int64_t colno = 0; colno < input.width(); ++colno) {
      (*result)(rowno, colno) = static_cast<float>(input(rowno, colno));
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<float>> Array2DF8E4M3FNToF32(
    const Array2D<tsl::float8_e4m3fn>& input) {
  auto result = std::make_unique<Array2D<float>>(input.height(), input.width());
  for (int64_t rowno = 0; rowno < input.height(); ++rowno) {
    for (int64_t colno = 0; colno < input.width(); ++colno) {
      (*result)(rowno, colno) = static_cast<float>(input(rowno, colno));
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<tsl::float8_e5m2>> Array2DF32ToF8E5M2(
    const Array2D<float>& input) {
  auto result = std::make_unique<Array2D<tsl::float8_e5m2>>(input.height(),
                                                            input.width());
  for (int64_t rowno = 0; rowno < input.height(); ++rowno) {
    for (int64_t colno = 0; colno < input.width(); ++colno) {
      (*result)(rowno, colno) =
          static_cast<tsl::float8_e5m2>(input(rowno, colno));
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<tsl::float8_e4m3fn>> Array2DF32ToF8E4M3FN(
    const Array2D<float>& input) {
  auto result = std::make_unique<Array2D<tsl::float8_e4m3fn>>(input.height(),
                                                              input.width());
  for (int64_t rowno = 0; rowno < input.height(); ++rowno) {
    for (int64_t colno = 0; colno < input.width(); ++colno) {
      (*result)(rowno, colno) =
          static_cast<tsl::float8_e4m3fn>(input(rowno, colno));
    }
  }
  return result;
}

static bool promote_f8_to_f32 = true;

std::unique_ptr<Array2D<tsl::float8_e5m2>> HloEvaluator::MatmulArray2D(
    const Array2D<tsl::float8_e5m2>& lhs,
    const Array2D<tsl::float8_e5m2>& rhs) {
  if (promote_f8_to_f32) {
    auto lhs_float = Array2DF8E5M2ToF32(lhs);
    auto rhs_float = Array2DF8E5M2ToF32(rhs);
    auto result = MatmulArray2D(*lhs_float, *rhs_float);
    return Array2DF32ToF8E5M2(*result);
  } else {
    return MatmulArray2DImpl<tsl::float8_e5m2>(
        lhs, rhs, __xla_cpu_runtime_EigenSingleThreadedMatMulF8E5M2);
  }
}

std::unique_ptr<Array2D<tsl::float8_e4m3fn>> HloEvaluator::MatmulArray2D(
    const Array2D<tsl::float8_e4m3fn>& lhs,
    const Array2D<tsl::float8_e4m3fn>& rhs) {
  if (promote_f8_to_f32) {
    auto lhs_float = Array2DF8E4M3FNToF32(lhs);
    auto rhs_float = Array2DF8E4M3FNToF32(rhs);
    auto result = MatmulArray2D(*lhs_float, *rhs_float);
    return Array2DF32ToF8E4M3FN(*result);
  } else {
    return MatmulArray2DImpl<tsl::float8_e4m3fn>(
        lhs, rhs, __xla_cpu_runtime_EigenSingleThreadedMatMulF8E4M3FN);
  }
}

}  // namespace xla

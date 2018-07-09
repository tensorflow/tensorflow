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
#include <cstdlib>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator_typed_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

namespace {

using tensorflow::gtl::ArraySlice;

template <typename OperandT>
StatusOr<std::unique_ptr<Literal>> Compare(const Shape& shape, HloOpcode opcode,
                                           LiteralSlice lhs_literal,
                                           LiteralSlice rhs_literal) {
  std::function<bool(OperandT, OperandT)> compare_op;
  switch (opcode) {
    case HloOpcode::kEq:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el == rhs_el;
      };
      break;
    case HloOpcode::kNe:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el != rhs_el;
      };
      break;
    case HloOpcode::kGe:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el >= rhs_el;
      };
      break;
    case HloOpcode::kGt:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el > rhs_el;
      };
      break;
    case HloOpcode::kLe:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el <= rhs_el;
      };
      break;
    case HloOpcode::kLt:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el < rhs_el;
      };
      break;
    default:
      LOG(FATAL) << "unhandled HLO opcode for conversion to Comparison: "
                 << HloOpcodeString(opcode);
  }

  auto result = MakeUnique<Literal>(shape);
  TF_RETURN_IF_ERROR(result->Populate<bool>([&](ArraySlice<int64> multi_index) {
    return compare_op(lhs_literal.Get<OperandT>(multi_index),
                      rhs_literal.Get<OperandT>(multi_index));
  }));

  return std::move(result);
}

template <>
StatusOr<std::unique_ptr<Literal>> Compare<complex64>(
    const Shape& shape, HloOpcode opcode, LiteralSlice lhs_literal,
    LiteralSlice rhs_literal) {
  std::function<bool(complex64, complex64)> compare_op;
  switch (opcode) {
    case HloOpcode::kEq:
      compare_op = [](complex64 lhs_el, complex64 rhs_el) {
        return lhs_el == rhs_el;
      };
      break;
    case HloOpcode::kNe:
      compare_op = [](complex64 lhs_el, complex64 rhs_el) {
        return lhs_el != rhs_el;
      };
      break;
    default:
      LOG(FATAL) << "unhandled HLO opcode for conversion to Comparison: "
                 << HloOpcodeString(opcode);
  }

  auto result = MakeUnique<Literal>(shape);
  TF_RETURN_IF_ERROR(result->Populate<bool>([&](ArraySlice<int64> multi_index) {
    return compare_op(lhs_literal.Get<complex64>(multi_index),
                      rhs_literal.Get<complex64>(multi_index));
  }));

  return std::move(result);
}

}  // namespace

HloEvaluator::HloEvaluator(int64 max_loop_iterations)
    : max_loop_iterations_(max_loop_iterations) {
  typed_visitors_[PRED] = MakeUnique<HloEvaluatorTypedVisitor<bool>>(this);
  typed_visitors_[U8] = MakeUnique<HloEvaluatorTypedVisitor<uint8>>(this);
  typed_visitors_[U16] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented(
        "HloEvaluator::HloEvaluatorTypedVisitor: unhandled primitive type: "
        "U16.");
  });
  typed_visitors_[U32] = MakeUnique<HloEvaluatorTypedVisitor<uint32>>(this);
  typed_visitors_[U64] = MakeUnique<HloEvaluatorTypedVisitor<uint64>>(this);
  typed_visitors_[S8] = MakeUnique<HloEvaluatorTypedVisitor<int8>>(this);
  typed_visitors_[S16] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented(
        "HloEvaluator::HloEvaluatorTypedVisitor: unhandled primitive type: "
        "S16.");
  });
  typed_visitors_[S32] = MakeUnique<HloEvaluatorTypedVisitor<int32>>(this);
  typed_visitors_[S64] = MakeUnique<HloEvaluatorTypedVisitor<int64>>(this);
  typed_visitors_[F16] =
      MakeUnique<HloEvaluatorTypedVisitor<Eigen::half, float>>(this);
  typed_visitors_[F32] = MakeUnique<HloEvaluatorTypedVisitor<float>>(this);
  typed_visitors_[F64] = MakeUnique<HloEvaluatorTypedVisitor<double>>(this);
  typed_visitors_[C64] = MakeUnique<HloEvaluatorTypedVisitor<complex64>>(this);

  // Most of the evaluator computations we use don't support BF16 (e.g.,
  // std::ceil, std::tanh). To make evaluator work with BF16, we set all
  // elementwise computations to be done in F32 and do BF16<->F32 conversion
  // around the input and the output of the computations.
  typed_visitors_[BF16] =
      MakeUnique<HloEvaluatorTypedVisitor<bfloat16, float>>(this);

  typed_visitors_[TUPLE] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented(
        "HloEvaluatorTypedVisitor: unhandled primitive type: TUPLE.");
  });
  typed_visitors_[OPAQUE] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented(
        "HloEvaluatorTypedVisitor: unhandled primitive type: OPAQUE.");
  });
}

template <typename LiteralPtr>
StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    const HloModule& module, ArraySlice<LiteralPtr> arg_literals) {
  XLA_VLOG_LINES(2, "HloEvaluator::Evaluate module:\n" + module.ToString());

  evaluated_.clear();
  arg_literals_.clear();
  for (const auto& literal_ptr : arg_literals) {
    arg_literals_.push_back(&*literal_ptr);
  }

  TF_RETURN_IF_ERROR(module.entry_computation()->Accept(this));

  return GetEvaluatedLiteralFor(module.entry_computation()->root_instruction())
      .CloneToUnique();
}

template <typename LiteralPtr>
StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    const HloComputation& computation, ArraySlice<LiteralPtr> arg_literals) {
  CHECK(computation.parent() != nullptr);
  XLA_VLOG_LINES(
      2, "HloEvaluator::Evaluate computation:\n" + computation.ToString());

  evaluated_.clear();
  arg_literals_.clear();
  for (const auto& literal_ptr : arg_literals) {
    arg_literals_.push_back(&*literal_ptr);
  }

  TF_RETURN_IF_ERROR(computation.Accept(this));
  return GetEvaluatedLiteralFor(computation.root_instruction()).CloneToUnique();
}

template <typename LiteralPtr>
StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    HloInstruction* instruction, ArraySlice<LiteralPtr> arg_literals) {
  TF_RET_CHECK(hlo_query::AllOperandsAreParametersOrConstants(*instruction));
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(instruction->shape()));

  evaluated_.clear();
  arg_literals_.clear();
  for (const auto& literal_ptr : arg_literals) {
    arg_literals_.push_back(&*literal_ptr);
  }

  // Evaluate operands of Parameter type against the input literals which
  // caches the evaluated literal results.
  for (const auto operand : instruction->operands()) {
    if (operand->opcode() == HloOpcode::kParameter) {
      const Literal* input_literal = arg_literals_[operand->parameter_number()];
      VLOG(2) << "Parameter operand evaluated to: "
              << input_literal->ToString();
      TF_RET_CHECK(ShapeUtil::Equal(operand->shape(), input_literal->shape()));

      evaluated_[operand] = input_literal->CloneToUnique();
    }
  }

  TF_RETURN_IF_ERROR(Preprocess(instruction));
  TF_RETURN_IF_ERROR(instruction->Visit(this));
  TF_RETURN_IF_ERROR(Postprocess(instruction));
  return GetEvaluatedLiteralFor(instruction).CloneToUnique();
}

StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kParameter) {
    return tensorflow::errors::FailedPrecondition(
        "Cannot evaluate a parameter.");
  }
  if (!hlo_query::AllOperandsAreConstants(*instruction)) {
    return tensorflow::errors::FailedPrecondition(
        "Not all operands are constants.");
  }
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(instruction->shape()));

  arg_literals_.clear();
  evaluated_.clear();

  TF_RETURN_IF_ERROR(Preprocess(instruction));
  TF_RETURN_IF_ERROR(instruction->Visit(this));
  TF_RETURN_IF_ERROR(Postprocess(instruction));
  return GetEvaluatedLiteralFor(instruction).CloneToUnique();
}

std::unique_ptr<Literal> HloEvaluator::TryEvaluate(
    HloInstruction* instruction) {
  auto result_or = Evaluate(instruction);
  if (!result_or.ok()) {
    VLOG(1) << "TryEvaluate failed:" << result_or.status();
    return nullptr;
  }

  return result_or.ConsumeValueOrDie();
}

StatusOr<std::unique_ptr<Literal>> HloEvaluator::EvaluateWithSubstitutions(
    const HloInstruction* instruction,
    const std::unordered_map<const HloInstruction*, const Literal*>&
        substitutions) {
  std::vector<std::unique_ptr<HloInstruction>> owned_operands;
  for (const HloInstruction* operand : instruction->operands()) {
    auto it = substitutions.find(operand);
    if (it == substitutions.end()) {
      owned_operands.push_back(operand->Clone());
    } else {
      owned_operands.push_back(
          HloInstruction::CreateConstant(it->second->CloneToUnique()));
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

StatusOr<std::unique_ptr<Literal>> HloEvaluator::EvaluateElementwiseBinaryOp(
    HloOpcode opcode, const Literal& lhs, const Literal& rhs) {
  std::unique_ptr<HloInstruction> lhs_instr =
      HloInstruction::CreateConstant(lhs.CloneToUnique());
  std::unique_ptr<HloInstruction> rhs_instr =
      HloInstruction::CreateConstant(rhs.CloneToUnique());

  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateBinary(lhs.shape(), opcode, lhs_instr.get(),
                                   rhs_instr.get());
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

StatusOr<std::unique_ptr<Literal>> HloEvaluator::EvaluateElementwiseUnaryOp(
    HloOpcode opcode, const Literal& operand) {
  std::unique_ptr<HloInstruction> operand_instr =
      HloInstruction::CreateConstant(operand.CloneToUnique());

  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateUnary(operand.shape(), opcode, operand_instr.get());
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

Status HloEvaluator::HandleParameter(HloInstruction* parameter) {
  CHECK_LT(parameter->parameter_number(), arg_literals_.size());
  const Literal* input_literal = arg_literals_[parameter->parameter_number()];
  VLOG(2) << "Parameter evaluated to: " << input_literal->ToString();
  DCHECK(ShapeUtil::Equal(parameter->shape(), input_literal->shape()))
      << "parameter shape is: " << ShapeUtil::HumanString(parameter->shape())
      << ", but input literal shape is: "
      << ShapeUtil::HumanString(input_literal->shape());

  evaluated_[parameter] = input_literal->CloneToUnique();
  return Status::OK();
}

Status HloEvaluator::HandleConstant(HloInstruction*) { return Status::OK(); }

Status HloEvaluator::HandleReshape(HloInstruction* reshape) {
  TF_ASSIGN_OR_RETURN(
      evaluated_[reshape],
      GetEvaluatedLiteralFor(reshape->operand(0))
          .Reshape(AsInt64Slice(reshape->shape().dimensions())));
  return Status::OK();
}

Status HloEvaluator::HandleTranspose(HloInstruction* transpose) {
  evaluated_[transpose] = GetEvaluatedLiteralFor(transpose->operand(0))
                              .Transpose(transpose->dimensions());
  return Status::OK();
}

Status HloEvaluator::HandleConcatenate(HloInstruction* concatenate) {
  ArraySlice<HloInstruction*> operands(concatenate->operands());
  // The result concatenate dimension is going to be the sum of all
  // concatenate dimensions of the operands taking part of the operation.
  const Shape& reference_shape = operands[0]->shape();
  CHECK(ShapeUtil::IsArray(reference_shape));
  const int64 rank = ShapeUtil::Rank(reference_shape);
  const int64 concat_dim = concatenate->dimensions()[0];
  CHECK_GE(concat_dim, 0);
  CHECK_LT(concat_dim, rank);

  DimensionVector concat_dimensions(reference_shape.dimensions().begin(),
                                    reference_shape.dimensions().end());

  for (int64 i = 1; i < operands.size(); ++i) {
    const Shape& operand_shape = operands[i]->shape();
    CHECK(ShapeUtil::IsArray(operand_shape));
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
    TF_RETURN_IF_ERROR(result_literal->CopySliceFrom(
        GetEvaluatedLiteralFor(operand), source_indices, dest_indices,
        AsInt64Slice(operand_shape.dimensions())));
    dest_indices[concat_dim] +=
        ShapeUtil::GetDimension(operand_shape, concat_dim);
  }

  evaluated_[concatenate] = std::move(result_literal);
  return Status::OK();
}

Status HloEvaluator::HandleIsFinite(HloInstruction* is_finite) {
  auto operand = is_finite->operand(0);
  if (!ShapeUtil::ElementIsFloating(operand->shape())) {
    return InvalidArgument(
        "expected element type in shape to be float for IsFinite op, got: %s",
        PrimitiveType_Name(operand->shape().element_type()).c_str());
  }

  switch (operand->shape().element_type()) {
    case F16:
      return Unimplemented("unhandled primitive type: F16.");
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
    default:
      LOG(FATAL) << "HandleIsFinite: unknown/unhandled primitive type: "
                 << PrimitiveType_Name(operand->shape().element_type());
  }

  return Status::OK();
}

Status HloEvaluator::HandleCompare(HloInstruction* compare) {
  HloOpcode opcode = compare->opcode();
  auto lhs = compare->operand(0);
  auto rhs = compare->operand(1);
  // TODO(b/35950897, b/27796129): add DCHECK back once implicit broadcast is
  // removed.
  if (!(ShapeUtil::SameDimensions(compare->shape(), rhs->shape()) &&
        ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()))) {
    return Unimplemented(
        "Implicit broadcasting is currently unsupported in HLO evaluator "
        "Shape Mismatch: %s vs %s vs %s",
        ShapeUtil::HumanString(compare->shape()).c_str(),
        ShapeUtil::HumanString(lhs->shape()).c_str(),
        ShapeUtil::HumanString(rhs->shape()).c_str());
  }

  TF_RET_CHECK(lhs->shape().element_type() == rhs->shape().element_type());

  const Literal& lhs_literal = GetEvaluatedLiteralFor(lhs);
  const Literal& rhs_literal = GetEvaluatedLiteralFor(rhs);

  // Note here we switch on the operand's type.
  switch (lhs->shape().element_type()) {
    case PRED: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<bool>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case U8: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<uint8>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case U16:
      return Unimplemented("unhandled primitive type: U16.");
    case U32: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<uint32>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case U64: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<uint64>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case S8: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<int8>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case S16:
      return Unimplemented("unhandled primitive type: S16.");
    case S32: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<int32>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case S64: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<int64>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case F16:
      return Unimplemented("unhandled primitive type: F16.");
    case BF16: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<bfloat16>(compare->shape(), opcode,
                                            lhs_literal, rhs_literal));
    } break;
    case F32: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<float>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case F64: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<double>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case C64: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<complex64>(compare->shape(), opcode,
                                             lhs_literal, rhs_literal));
    } break;
    default:
      LOG(FATAL) << "HandleCompare: unknown primitive type: "
                 << PrimitiveType_Name(lhs->shape().element_type());
  }

  return Status::OK();
}

Status HloEvaluator::HandleTuple(HloInstruction* tuple) {
  std::vector<const Literal*> operand_literals;
  for (auto operand : tuple->operands()) {
    operand_literals.push_back(&GetEvaluatedLiteralFor(operand));
  }

  evaluated_[tuple] = LiteralUtil::MakeTuple(operand_literals);
  return Status::OK();
}

// Returns an ShapeUtil::IndexIterationSpace that iterates over the output
// gather dimensions while keeping the rest of the output dimensions clamped to
// 0.
ShapeUtil::IndexIterationSpace IterationSpaceForOutputGatherIndices(
    const Shape& output_shape, const GatherDimensionNumbers& dim_numbers) {
  int64 output_rank = output_shape.dimensions_size();
  std::vector<int64> index_base(output_rank, 0);
  std::vector<int64> index_count;
  index_count.reserve(output_rank);
  for (int64 i = 0; i < output_rank; i++) {
    bool is_output_gather_dim =
        !c_binary_search(dim_numbers.output_window_dims(), i);
    index_count.push_back(is_output_gather_dim ? output_shape.dimensions(i)
                                               : 1);
  }

  return {std::move(index_base), std::move(index_count),
          std::vector<int64>(output_rank, 1)};
}

// Return an ShapeUtil::IndexIterationSpace that iterates over the output window
// dimensions while keeping the rest of the output dimensions clamped to 0.
ShapeUtil::IndexIterationSpace IterationSpaceForOutputWindowIndices(
    int64 output_rank, ArraySlice<int64> window_bounds,
    const GatherDimensionNumbers& dim_numbers) {
  std::vector<int64> index_base(output_rank, 0);
  std::vector<int64> index_count(output_rank, 1);
  int64 window_bounds_idx = 0;
  for (int64 i = 0; i < output_rank; i++) {
    bool is_output_window_dim =
        c_binary_search(dim_numbers.output_window_dims(), i);
    if (is_output_window_dim) {
      while (c_binary_search(dim_numbers.elided_window_dims(),
                             window_bounds_idx)) {
        window_bounds_idx++;
      }
      index_count[i] = window_bounds[window_bounds_idx++];
    }
  }

  return {std::move(index_base), std::move(index_count),
          std::vector<int64>(output_rank, 1)};
}

// This functor computes the contribution of gather_indices to an input index
// corresponding to an output index.  That is, given an output index I, it picks
// out the gather output indices in I and uses them to look up a gather index,
// G, from the gather indices tensor, and expands G into the input space
// according to gather_dims_to_operand_dims.
class OutputGatherIndexToInputIndex {
 public:
  // The constructor does some setup work that is amortized across all
  // iterations.
  explicit OutputGatherIndexToInputIndex(
      const GatherDimensionNumbers* dim_numbers, const Shape& input_shape,
      const Shape& output_shape, const Literal* gather_indices)
      : dim_numbers_(*dim_numbers), gather_indices_(*gather_indices) {
    for (int64 i = 0; i < output_shape.dimensions_size(); i++) {
      output_dim_is_gather_dims_.push_back(
          !c_binary_search(dim_numbers_.output_window_dims(), i));
    }

    for (int64 i = 0; i < input_shape.dimensions_size(); i++) {
      int64 index_of_input_dim_in_index_vector =
          std::distance(dim_numbers_.gather_dims_to_operand_dims().begin(),
                        c_find(dim_numbers_.gather_dims_to_operand_dims(), i));
      if (index_of_input_dim_in_index_vector ==
          dim_numbers_.gather_dims_to_operand_dims_size()) {
        input_dim_value_to_index_vector_.push_back(-1);
      } else {
        input_dim_value_to_index_vector_.push_back(
            index_of_input_dim_in_index_vector);
      }
    }

    index_vector_index_.resize(gather_indices_.shape().dimensions_size());
    input_index_.resize(input_shape.dimensions_size());
    int64 index_vector_size =
        gather_indices_.shape().dimensions(dim_numbers_.index_vector_dim());
    index_vector_.resize(index_vector_size);
  }

  // Returns the contribution of gather_indices to the input index corresponding
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
  // This returns an arrayslice into memory owned by the class.
  StatusOr<ArraySlice<int64>> operator()(ArraySlice<int64> output_index) {
    PropagateOutputIndexGatherDimsToIndexVectorIndex(output_index);
    TF_RETURN_IF_ERROR(FetchIndexVector());
    PropagateIndexVectorToInputIndex();
    return ArraySlice<int64>(input_index_);
  }

 private:
  // Propagates the gather index dimensions from the output index into
  // index_vector_index_ by mutating index_vector_index_ in place.  Does not
  // update the dim_numbers.index_vector_dim() dimension -- that's the dimension
  // we iterate over in FetchIndexVector.
  void PropagateOutputIndexGatherDimsToIndexVectorIndex(
      ArraySlice<int64> output_index) {
    int64 index_vector_index_i = 0;
    for (int64 i = 0, e = output_index.size(); i < e; i++) {
      if (!output_dim_is_gather_dims_[i]) {
        continue;
      }

      if (index_vector_index_i == dim_numbers_.index_vector_dim()) {
        index_vector_index_i++;
      }

      index_vector_index_[index_vector_index_i++] = output_index[i];
    }
  }

  // Populates index_vector_ by iterating over gather_indices_ according to
  // index_vector_index_.
  Status FetchIndexVector() {
    int64 index_vector_dim = dim_numbers_.index_vector_dim();
    for (int64 i = 0, e = index_vector_.size(); i < e; i++) {
      index_vector_index_[index_vector_dim] = i;
      TF_ASSIGN_OR_RETURN(index_vector_[i], gather_indices_.GetIntegralAsS64(
                                                index_vector_index_));
    }
    return Status::OK();
  }

  // Populates input_index_.
  void PropagateIndexVectorToInputIndex() {
    for (int64 i = 0, e = input_index_.size(); i < e; i++) {
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
  std::vector<int64> input_dim_value_to_index_vector_;

  // output_dim_is_gather_dims_[i] is true iff the output index i is a gather
  // dimension.
  std::vector<bool> output_dim_is_gather_dims_;

  // The buffer into which we construct an index into gather_indices_ to fetch
  // the index vector.
  std::vector<int64> index_vector_index_;

  // The index vector fetched from gather_indices_.
  std::vector<int64> index_vector_;

  // The result computed by this functor.  operator() returns an ArraySlice into
  // this vector.
  std::vector<int64> input_index_;

  const GatherDimensionNumbers& dim_numbers_;
  const Literal& gather_indices_;
};

// This functor computes the contribution of the window indices in an output
// index to an input index.  That is, given an output index I it picks out the
// output window indices in I and expands it into a window index into the input
// shape.
class OutputWindowIndexToInputIndex {
 public:
  // The constructor does some setup work that is amortized across all
  // iterations.
  explicit OutputWindowIndexToInputIndex(
      const GatherDimensionNumbers& dim_numbers, const Shape& input_shape,
      const Shape& output_shape) {
    std::vector<int64> window_index_to_output_index;
    int64 output_index_count = 0;
    for (int64 i = 0; i < output_shape.dimensions_size(); i++) {
      if (c_binary_search(dim_numbers.output_window_dims(), i)) {
        window_index_to_output_index.push_back(output_index_count++);
      } else {
        output_index_count++;
      }
    }

    int64 window_dim_count = 0;
    for (int64 i = 0; i < input_shape.dimensions_size(); i++) {
      if (c_binary_search(dim_numbers.elided_window_dims(), i)) {
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
  // This returns an arrayslice into memory owned by the class.
  StatusOr<ArraySlice<int64>> operator()(ArraySlice<int64> output_index) {
    PropagateOutputIndexWindowDimsToInputIndex(output_index);
    return ArraySlice<int64>(input_index_);
  }

 private:
  // Propagates window dimensions from the output index to input_index_ by
  // mutating input_index_ in place.
  void PropagateOutputIndexWindowDimsToInputIndex(
      ArraySlice<int64> output_index) {
    for (int64 i = 0, e = input_index_.size(); i < e; i++) {
      if (input_dim_value_to_output_index_[i] != -1) {
        input_index_[i] = output_index[input_dim_value_to_output_index_[i]];
      }

      // If input_dim_value_to_index_vector_[i] == -1 then input_index_[i]
      // remains 0, as set by the constructor.
    }
  }

  // input_dim_value_to_index_vector_[i] tells us how to compute dimension i of
  // the input index from the output index. See
  // PropagateOutputIndexToInputIndex.
  std::vector<int64> input_dim_value_to_output_index_;

  // The result computed by this functor.  operator() returns an ArraySlice into
  // this vector.
  std::vector<int64> input_index_;
};

// Rehapes the gather indices input to have a trailing degenerate `1` dimension
// if necessary.  Hands over the ownership of the newly created literal (if
// there is one) to `reshaped_gather_indices`.
static StatusOr<std::reference_wrapper<const Literal>> ReshapedGatherIndices(
    int64 index_vector_dim, const Literal& gather_indices,
    std::unique_ptr<Literal>* reshaped_gather_indices) {
  if (gather_indices.shape().dimensions_size() != index_vector_dim) {
    return std::cref(gather_indices);
  }

  std::vector<int64> new_shape(gather_indices.shape().dimensions().begin(),
                               gather_indices.shape().dimensions().end());
  new_shape.push_back(1);
  TF_ASSIGN_OR_RETURN(*reshaped_gather_indices,
                      gather_indices.Reshape(new_shape));
  return std::cref(**reshaped_gather_indices);
}

Status HloEvaluator::HandleGather(HloInstruction* gather) {
  std::unique_ptr<Literal> result = Literal::CreateFromShape(gather->shape());
  const Shape& shape = gather->shape();
  const GatherDimensionNumbers& dim_numbers =
      gather->gather_dimension_numbers();
  const Literal& operand = GetEvaluatedLiteralFor(gather->operand(0));
  std::unique_ptr<Literal> reshaped_gather_indices;
  TF_ASSIGN_OR_RETURN(
      const Literal& gather_indices,
      ReshapedGatherIndices(dim_numbers.index_vector_dim(),
                            GetEvaluatedLiteralFor(gather->operand(1)),
                            &reshaped_gather_indices));

  // We iterate over the gather dimensions in the output shape in an outer loop
  // nest, and iterate over the window dimensions in the output shape in an
  // inner loop nest.

  ShapeUtil::IndexIterationSpace gather_indices_iteration_space =
      IterationSpaceForOutputGatherIndices(shape, dim_numbers);
  ShapeUtil::IndexIterationSpace window_indices_iteration_space =
      IterationSpaceForOutputWindowIndices(
          shape.dimensions_size(), gather->gather_window_bounds(), dim_numbers);

  // Scratch buffers that hold an index in the output shape and the
  // corresponding index in the input shape.
  std::vector<int64> input_index(operand.shape().dimensions_size());
  std::vector<int64> output_index(gather->shape().dimensions_size());

  OutputGatherIndexToInputIndex output_gather_index_to_input_index(
      &gather->gather_dimension_numbers(), /*input_shape=*/operand.shape(),
      /*output_shape=*/shape, &gather_indices);
  OutputWindowIndexToInputIndex output_window_index_to_input_index(
      gather->gather_dimension_numbers(), /*input_shape=*/operand.shape(),
      /*output_shape=*/shape);

  const Shape& operand_shape = operand.shape();

  auto gather_inner_loop_body =
      [&](ArraySlice<int64> output_window_index,
          ArraySlice<int64> input_gather_index,
          ArraySlice<int64> output_gather_index) -> StatusOr<bool> {
    TF_ASSIGN_OR_RETURN(
        ArraySlice<int64> input_window_index,
        output_window_index_to_input_index(output_window_index));
    for (int i = 0, e = output_index.size(); i < e; i++) {
      output_index[i] = output_gather_index[i] + output_window_index[i];
      DCHECK_LT(output_index[i], shape.dimensions(i));
    }
    for (int i = 0, e = input_index.size(); i < e; i++) {
      // TODO(b/74360564): We should implement whatever out of bounds behavior
      // we decide for dynamic-slice here as well.
      input_index[i] = (input_gather_index[i] + input_window_index[i]) %
                       operand_shape.dimensions(i);
      if (input_index[i] < 0) {
        input_index[i] += operand_shape.dimensions(i);
      }
    }
    TF_RETURN_IF_ERROR(
        result->CopyElementFrom(operand, input_index, output_index));
    return true;
  };

  auto gather_outer_loop_body =
      [&](ArraySlice<int64> output_gather_index) -> StatusOr<bool> {
    TF_ASSIGN_OR_RETURN(
        ArraySlice<int64> input_gather_index,
        output_gather_index_to_input_index(output_gather_index));
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
        shape, window_indices_iteration_space,
        std::bind(gather_inner_loop_body, std::placeholders::_1,
                  input_gather_index, output_gather_index)));
    return true;
  };

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
      shape, gather_indices_iteration_space, gather_outer_loop_body));
  evaluated_[gather] = std::move(result);
  return Status::OK();
}

Status HloEvaluator::HandleBroadcast(HloInstruction* broadcast) {
  const Literal& operand = GetEvaluatedLiteralFor(broadcast->operand(0));

  TF_RET_CHECK(broadcast->dimensions().size() ==
               ShapeUtil::Rank(operand.shape()))
      << "broadcast dimensions is of size: " << broadcast->dimensions().size()
      << " and rank of operand_to_broadcast is: "
      << ShapeUtil::Rank(operand.shape());
  // Checks that operand's dimensions are the same as the broadcast's
  // dimensions along the dimensions to be broadcasted.
  for (int64 i = 0; i < broadcast->dimensions().size(); ++i) {
    TF_RET_CHECK(broadcast->shape().dimensions(broadcast->dimensions(i)) ==
                 operand.shape().dimensions(i));
  }

  TF_ASSIGN_OR_RETURN(
      evaluated_[broadcast],
      operand.Broadcast(broadcast->shape(), broadcast->dimensions()));

  return Status::OK();
}

Status HloEvaluator::HandleAfterAll(HloInstruction* token) {
  evaluated_[token] = LiteralUtil::CreateToken();
  return Status::OK();
}

Status HloEvaluator::HandleGetTupleElement(HloInstruction* get_tuple_element) {
  const auto result_shape = get_tuple_element->shape();
  const int64 index = get_tuple_element->tuple_index();

  auto operand = get_tuple_element->operand(0);
  TF_ASSIGN_OR_RETURN(
      auto inferred_return_shape,
      ShapeInference::InferGetTupleElementShape(operand->shape(), index));
  TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
      << "return shape set to: " << ShapeUtil::HumanString(result_shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);

  const Literal& operand_tuple_literal = GetEvaluatedLiteralFor(operand);

  evaluated_[get_tuple_element] = MakeUnique<Literal>(
      ShapeUtil::GetTupleElementShape(operand->shape(), index));
  return evaluated_[get_tuple_element]->CopyFrom(operand_tuple_literal,
                                                 /*dest_shape_index=*/{},
                                                 /*src_shape_index=*/{index});
}

Status HloEvaluator::HandleCopy(HloInstruction* copy) {
  TF_RET_CHECK(ShapeUtil::Compatible(copy->shape(), copy->operand(0)->shape()));

  auto result = GetEvaluatedLiteralFor(copy->operand(0)).CloneToUnique();
  evaluated_[copy] = std::move(result);
  return Status::OK();
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

  HloEvaluator embedded_evaluator;
  std::unique_ptr<Literal> result =
      embedded_evaluator.Evaluate<const Literal*>(*computation, arg_literals)
          .ConsumeValueOrDie();

  evaluated_[call] = std::move(result);
  return Status::OK();
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
    LayoutUtil::SetToDefaultLayout(instruction->mutable_shape());
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

  HloEvaluator embedded_evaluator;
  std::unique_ptr<Literal> result =
      embedded_evaluator
          .Evaluate<const Literal*>(*readded_computation, arg_literals)
          .ConsumeValueOrDie();

  evaluated_[fusion] = std::move(result);
  return Status::OK();
}

Status HloEvaluator::HandleConditional(HloInstruction* conditional) {
  const auto& pred = GetEvaluatedLiteralFor(conditional->operand(0));
  const auto& true_computation_arg =
      GetEvaluatedLiteralFor(conditional->operand(1));
  const auto& false_computation_arg =
      GetEvaluatedLiteralFor(conditional->operand(2));

  auto* true_computation = conditional->true_computation();
  auto* false_computation = conditional->false_computation();

  HloEvaluator embedded_evaluator;
  std::unique_ptr<Literal> result;
  if (pred.Get<bool>({})) {
    result = embedded_evaluator
                 .Evaluate<const Literal*>(*true_computation,
                                           {&true_computation_arg})
                 .ConsumeValueOrDie();
  } else {
    result = embedded_evaluator
                 .Evaluate<const Literal*>(*false_computation,
                                           {&false_computation_arg})
                 .ConsumeValueOrDie();
  }

  evaluated_[conditional] = std::move(result);
  return Status::OK();
}

Status HloEvaluator::HandleSelect(HloInstruction* select) {
  const auto& pred = GetEvaluatedLiteralFor(select->operand(0));
  const auto& on_true = GetEvaluatedLiteralFor(select->operand(1));
  const auto& on_false = GetEvaluatedLiteralFor(select->operand(2));

  // If predicate is of scalar type, no element-wise selection would be needed.
  if (ShapeUtil::IsScalar(pred.shape())) {
    if (pred.Get<bool>({})) {
      evaluated_[select] = on_true.CloneToUnique();
    } else {
      evaluated_[select] = on_false.CloneToUnique();
    }
    return Status::OK();
  }

  return DefaultAction(select);
}

Status HloEvaluator::HandleTupleSelect(HloInstruction* tuple_select) {
  const auto& pred = GetEvaluatedLiteralFor(tuple_select->operand(0));
  const auto& on_true = GetEvaluatedLiteralFor(tuple_select->operand(1));
  const auto& on_false = GetEvaluatedLiteralFor(tuple_select->operand(2));

  if (pred.Get<bool>({})) {
    evaluated_[tuple_select] = on_true.CloneToUnique();
  } else {
    evaluated_[tuple_select] = on_false.CloneToUnique();
  }
  return Status::OK();
}

Status HloEvaluator::HandleWhile(HloInstruction* while_hlo) {
  HloComputation* cond_comp = while_hlo->while_condition();
  HloComputation* body_comp = while_hlo->while_body();
  // Initialize the loop carried valued with the input to the While instruction.
  auto lcv = GetEvaluatedLiteralFor(while_hlo->operand(0)).CloneToUnique();
  bool keep_going = true;
  int64 iteration_count = 0;
  HloEvaluator cond_evaluator(max_loop_iterations_);
  HloEvaluator loop_body_evaluator(max_loop_iterations_);
  while (keep_going) {
    if (max_loop_iterations_ >= 0 && iteration_count++ > max_loop_iterations_) {
      return InvalidArgument("Loop %s exceeded loop iteration limit (%lld).",
                             while_hlo->name().c_str(), max_loop_iterations_);
    }
    TF_ASSIGN_OR_RETURN(auto cond_val, cond_evaluator.Evaluate<Literal*>(
                                           *cond_comp, {lcv.get()}));
    keep_going = cond_val->GetFirstElement<bool>();
    if (keep_going) {
      TF_ASSIGN_OR_RETURN(auto body_val, loop_body_evaluator.Evaluate<Literal*>(
                                             *body_comp, {lcv.get()}));
      VLOG(3) << "Loop iteration result: " << body_val->ToString();
      lcv = std::move(body_val);
      cond_evaluator.ResetVisitStates();
      loop_body_evaluator.ResetVisitStates();
    }
  }
  evaluated_[while_hlo] = std::move(lcv);
  return Status::OK();
}

// Key-value sort is a special snowflake: it's templated on two different
// element types, one for the keys, and one for the values. Jump through some
// hoops to make this work.
namespace {
template <typename KeyType, typename ValueType>
std::unique_ptr<Literal> EvaluateSortInternal(HloInstruction* sort,
                                              const Literal& keys_literal,
                                              const Literal& values_literal) {
  CHECK_EQ(sort->operand_count(), 2);
  // We need to sort and array of keys and an array of values, where the
  // sorted order of the values is determined by the keys. The simplest(?)
  // way to do this is to go to an array-of-pairs representation, sort the
  // array using the keys, and then go back to pair-of-arrays.
  VLOG(3) << "HandleSort keys_literal: " << keys_literal.ToString();
  VLOG(3) << "HandleSort values_literal: " << values_literal.ToString();
  const auto& keys_data = keys_literal.data<KeyType>();
  const auto& values_data = values_literal.data<ValueType>();
  using kv_pair = std::pair<KeyType, ValueType>;
  std::vector<kv_pair> key_value_vector;
  CHECK_EQ(keys_data.size(), values_data.size());
  key_value_vector.reserve(keys_data.size());
  for (int i = 0; i < keys_data.size(); ++i) {
    key_value_vector.push_back(std::make_pair(keys_data[i], values_data[i]));
  }
  std::sort(key_value_vector.begin(), key_value_vector.end(),
            [](const kv_pair& a, const kv_pair& b) {
              return SafeLess<KeyType>(a.first, b.first);
            });
  std::vector<KeyType> result_keys;
  std::vector<ValueType> result_values;
  for (const auto& key_value : key_value_vector) {
    result_keys.push_back(key_value.first);
    result_values.push_back(key_value.second);
  }
  auto result_keys_literal = MakeUnique<Literal>(sort->operand(0)->shape());
  result_keys_literal->PopulateR1(
      tensorflow::gtl::ArraySlice<KeyType>(result_keys));
  auto result_values_literal = MakeUnique<Literal>(sort->operand(1)->shape());
  result_values_literal->PopulateR1(
      tensorflow::gtl::ArraySlice<ValueType>(result_values));
  auto result_tuple = LiteralUtil::MakeTuple(
      {result_keys_literal.get(), result_values_literal.get()});
  VLOG(3) << "HandleSort result_tuple: " << result_tuple->ToString();
  return result_tuple;
}

template <typename KeyType>
StatusOr<std::unique_ptr<Literal>> EvaluateSortCurried(
    HloInstruction* sort, const Literal& keys_literal,
    const Literal& values_literal) {
  switch (sort->operand(1)->shape().element_type()) {
    case F32:
      return EvaluateSortInternal<KeyType, float>(sort, keys_literal,
                                                  values_literal);
    case U32:
      return EvaluateSortInternal<KeyType, uint32>(sort, keys_literal,
                                                   values_literal);
    case S32:
      return EvaluateSortInternal<KeyType, int32>(sort, keys_literal,
                                                  values_literal);
    case BF16:
      return EvaluateSortInternal<KeyType, bfloat16>(sort, keys_literal,
                                                     values_literal);
    default:
      return InvalidArgument("Unsupported type for Sort");
  }
}

StatusOr<std::unique_ptr<Literal>> EvaluateSort(HloInstruction* sort,
                                                const Literal& keys_literal,
                                                const Literal& values_literal) {
  switch (sort->operand(0)->shape().element_type()) {
    case F32:
      return EvaluateSortCurried<float>(sort, keys_literal, values_literal);
    case U32:
      return EvaluateSortCurried<uint32>(sort, keys_literal, values_literal);
    case S32:
      return EvaluateSortCurried<int32>(sort, keys_literal, values_literal);
    case BF16:
      return EvaluateSortCurried<bfloat16>(sort, keys_literal, values_literal);
    default:
      return InvalidArgument("Unsupported type for Sort");
  }
}
}  // namespace

Status HloEvaluator::HandleSort(HloInstruction* sort) {
  if (!ShapeUtil::IsTuple(sort->shape())) {
    return DefaultAction(sort);
  } else {
    auto result = EvaluateSort(sort, GetEvaluatedLiteralFor(sort->operand(0)),
                               GetEvaluatedLiteralFor(sort->operand(1)));
    if (result.ok()) {
      evaluated_[sort] = std::move(result.ValueOrDie());
      return Status::OK();
    } else {
      return result.status();
    }
  }
}

Status HloEvaluator::Preprocess(HloInstruction* hlo) {
  VLOG(2) << "About to visit HLO: " << hlo->ToString();
  return Status::OK();
}

Status HloEvaluator::Postprocess(HloInstruction* hlo) {
  VLOG(2) << "Finished visiting " << hlo->ToString()
          << "; evaluated value is: " << GetEvaluatedLiteralFor(hlo).ToString();
  return Status::OK();
}

// Explicit instantiation of templatized Evaluate* methods.
//
template StatusOr<std::unique_ptr<Literal>>
HloEvaluator::Evaluate<const Literal*>(const HloModule& module,
                                       ArraySlice<const Literal*> arg_literals);
template StatusOr<std::unique_ptr<Literal>>
HloEvaluator::Evaluate<std::unique_ptr<Literal>>(
    const HloModule& module, ArraySlice<std::unique_ptr<Literal>> arg_literals);

template StatusOr<std::unique_ptr<Literal>>
HloEvaluator::Evaluate<const Literal*>(const HloComputation& computation,
                                       ArraySlice<const Literal*> arg_literals);
template StatusOr<std::unique_ptr<Literal>>
HloEvaluator::Evaluate<std::unique_ptr<Literal>>(
    const HloComputation& computation,
    ArraySlice<std::unique_ptr<Literal>> arg_literals);

template StatusOr<std::unique_ptr<Literal>>
HloEvaluator::Evaluate<const Literal*>(HloInstruction* instruction,
                                       ArraySlice<const Literal*> arg_literals);
template StatusOr<std::unique_ptr<Literal>>
HloEvaluator::Evaluate<std::unique_ptr<Literal>>(
    HloInstruction* instruction,
    ArraySlice<std::unique_ptr<Literal>> arg_literals);

}  // namespace xla

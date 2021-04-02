/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/client/value_inference.h"

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {
Literal CreatePredLiteral(bool pred, const Shape& reference_shape) {
  if (reference_shape.IsTuple()) {
    std::vector<Literal> sub_literals;
    for (const Shape& shape : reference_shape.tuple_shapes()) {
      sub_literals.emplace_back(CreatePredLiteral(pred, shape));
    }
    return Literal::MoveIntoTuple(absl::MakeSpan(sub_literals));
  }
  PrimitiveType element_type = reference_shape.element_type();
  if (element_type == TOKEN) {
    return LiteralUtil::CreateR0(pred);
  }
  Literal literal = LiteralUtil::CreateR0(pred);
  Literal literal_broadcast =
      literal.Broadcast(ShapeUtil::ChangeElementType(reference_shape, PRED), {})
          .ValueOrDie();
  return literal_broadcast;
}

// Create a literal with garbage data. The data inside is undefined and
// shouldn't be used in any meaningful computation.
Literal CreateGarbageLiteral(const Shape& reference_shape) {
  if (reference_shape.IsTuple()) {
    std::vector<Literal> sub_literals;
    for (const Shape& shape : reference_shape.tuple_shapes()) {
      sub_literals.emplace_back(CreateGarbageLiteral(shape));
    }
    return Literal::MoveIntoTuple(absl::MakeSpan(sub_literals));
  }
  PrimitiveType element_type = reference_shape.element_type();
  if (element_type == TOKEN) {
    return LiteralUtil::CreateToken();
  }
  if (primitive_util::IsFloatingPointType(element_type)) {
    Literal literal = LiteralUtil::NanValue(element_type).ValueOrDie();
    return literal.Broadcast(reference_shape, {}).ValueOrDie();
  } else {
    Literal literal = LiteralUtil::MaxValue(element_type);
    return literal.Broadcast(reference_shape, {}).ValueOrDie();
  }
}

using GetOperand =
    std::function<StatusOr<Literal>(int64 operand_index, int64 opreand_handle)>;

// HloProtoEvaluator evaluates an hlo proto and returns a literal. The user has
// to provide operand as literals through the get_operand function.
struct HloProtoEvaluator {
  explicit HloProtoEvaluator(HloInstructionProto inst, GetOperand get_operand)
      : inst(std::move(inst)),
        get_operand(get_operand),
        module("EmptyModuleForEvaluation", HloModuleConfig()) {}

  // WithOpCode changes the called computation of the instruction being
  // evaluated.
  HloProtoEvaluator& WithComputation(
      std::unique_ptr<HloComputation> new_computation) {
    computation = new_computation.get();
    computation->ClearUniqueIdInternal();
    for (HloInstruction* inst : computation->instructions()) {
      inst->ClearUniqueIdInternal();
    }
    module.AddEmbeddedComputation(std::move(new_computation));
    return *this;
  }

  // WithOpCode changes the primitive type of the instruction being evaluated.
  HloProtoEvaluator& WithPrimitiveType(PrimitiveType new_primitive_type) {
    primitive_type = new_primitive_type;
    return *this;
  }

  // WithOpCode changes the opcode of the instruction being evaluated.
  HloProtoEvaluator& WithOpCode(HloOpcode new_opcode) {
    opcode = new_opcode;
    return *this;
  }

  StatusOr<Literal> Evaluate() {
    // Evaluate the instruction by swapping it's operands with constant
    // instructions with given literals.
    HloComputation::Builder builder("EmptyComputation");
    absl::flat_hash_map<int64, HloInstruction*> operand_map;
    for (int64 i = 0; i < inst.operand_ids_size(); ++i) {
      int64 operand_handle = inst.operand_ids(i);
      TF_ASSIGN_OR_RETURN(auto literal, get_operand(i, inst.operand_ids(i)));
      std::unique_ptr<HloInstruction> operand =
          HloInstruction::CreateConstant(std::move(literal));
      operand_map[operand_handle] = operand.get();
      builder.AddInstruction(std::move(operand));
    }

    if (primitive_type.has_value()) {
      *inst.mutable_shape() = ShapeUtil::ChangeElementType(
                                  Shape(inst.shape()), primitive_type.value())
                                  .ToProto();
    }
    if (opcode.has_value()) {
      *inst.mutable_opcode() = HloOpcodeString(opcode.value());
    }
    absl::flat_hash_map<int64, HloComputation*> computation_map;
    if (inst.called_computation_ids_size() != 0) {
      TF_RET_CHECK(inst.called_computation_ids_size() == 1 &&
                   computation != nullptr)
          << inst.DebugString();
      computation_map[inst.called_computation_ids(0)] = computation;
    }
    TF_ASSIGN_OR_RETURN(
        auto new_instruction,
        HloInstruction::CreateFromProto(inst, operand_map, computation_map));
    new_instruction->ClearUniqueIdInternal();
    builder.AddInstruction(std::move(new_instruction));
    auto computation = builder.Build();
    module.AddEntryComputation(std::move(computation));
    HloEvaluator evaluator;
    return evaluator.Evaluate(module.entry_computation()->root_instruction());
  }

  HloInstructionProto inst;
  GetOperand get_operand;
  HloModule module;
  HloComputation* computation = nullptr;
  absl::optional<PrimitiveType> primitive_type = absl::nullopt;
  absl::optional<HloOpcode> opcode = absl::nullopt;
};

}  // namespace

// Analyze a tensor's constant value, upper-bound value or lower-bound value.
StatusOr<Literal> ValueInference::AnalyzeConstantValue(
    int64 handle, ValueInferenceMode mode) {
  auto get_value = [mode, this](int64 handle) {
    switch (mode) {
      case kValue:
        return AnalyzeConstant(handle);
      case kUpperBound:
        return AnalyzeUpperBound(handle);
      case kLowerBound:
        return AnalyzeLowerBound(handle);
    }
  };
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      builder_->LookUpInstructionByHandle(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  switch (opcode) {
      // Non functional ops.
    case HloOpcode::kRng:
    case HloOpcode::kAllReduce:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kCall:
    case HloOpcode::kCustomCall:
    case HloOpcode::kWhile:
    case HloOpcode::kConditional:
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
    case HloOpcode::kParameter: {
      // The value is dynamic. We return a garbage literal here, which
      // will be masked out later.
      return CreateGarbageLiteral(Shape(root->shape()));
    }
    // Subtract and Divide use lower-bound as second operand.
    case HloOpcode::kSubtract:
    case HloOpcode::kCos:
    case HloOpcode::kSin:
    case HloOpcode::kNegate:
    case HloOpcode::kAbs:
    case HloOpcode::kDivide:
    case HloOpcode::kGetDimensionSize: {
      return InvalidArgument("AnalyzeConstantValue can't handle opcode: %s",
                             root->opcode());
    }
    case HloOpcode::kGetTupleElement: {
      int64 operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                          builder_->LookUpInstructionByHandle(operand_handle));
      TF_ASSIGN_OR_RETURN(HloOpcode operand_opcode,
                          StringToHloOpcode(operand_proto->opcode()));
      if (operand_opcode == HloOpcode::kParameter) {
        // Don't materialize the whole parameter if it's followed by a GTE.
        return CreateGarbageLiteral(Shape(root->shape()));
      }
      return HloProtoEvaluator(*root,
                               [&](int64 operand_index, int64 operand_handle) {
                                 return get_value(operand_handle);
                               })
          .WithPrimitiveType(PRED)
          .Evaluate();
    }
    case HloOpcode::kReduce:
    case HloOpcode::kScatter:
    case HloOpcode::kReduceWindow: {
      HloComputationProto computation_proto =
          builder_->embedded_[root->called_computation_ids(0)];
      TF_ASSIGN_OR_RETURN(auto computation, HloComputation::CreateFromProto(
                                                computation_proto, {}));
      return HloProtoEvaluator(*root,
                               [&](int64 operand_index, int64 operand_handle) {
                                 return get_value(operand_handle);
                               })
          .WithComputation(std::move(computation))
          .Evaluate();
    }
    default:
      return HloProtoEvaluator(*root,
                               [&](int64 operand_index, int64 operand_handle) {
                                 return get_value(operand_handle);
                               })
          .Evaluate();
  }
}

StatusOr<Literal> ValueInference::AnalyzeUpperBound(int64 handle) {
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      builder_->LookUpInstructionByHandle(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64 dimension = root->dimensions(0);
      int64 operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                          builder_->LookUpInstructionByHandle(operand_handle));
      return LiteralUtil::CreateR0<int32>(
          operand_proto->shape().dimensions(dimension));
    }
    case HloOpcode::kAbs: {
      // upper-bound(abs(operand)) = max(abs(lower-bound(operand)),
      // abs(upper-bound(operand)))
      int64 operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(auto lower_bound, AnalyzeLowerBound(operand_handle));
      TF_ASSIGN_OR_RETURN(auto upper_bound, AnalyzeUpperBound(operand_handle));
      HloEvaluator evaluator;
      TF_ASSIGN_OR_RETURN(
          auto lower_bound_abs,
          evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kAbs, lower_bound));
      TF_ASSIGN_OR_RETURN(
          auto upper_bound_abs,
          evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kAbs, upper_bound));
      return evaluator.EvaluateElementwiseBinaryOp(
          HloOpcode::kMaximum, lower_bound_abs, upper_bound_abs);
    }
    case HloOpcode::kNegate: {
      // upper-bound(negate(operand)) = negate(lower-bound(operand))
      int64 operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(auto lower_bound, AnalyzeLowerBound(operand_handle));
      HloEvaluator evaluator;
      return evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kNegate,
                                                  lower_bound);
    }
    case HloOpcode::kSubtract:
    case HloOpcode::kDivide: {
      // Lower-bound is used for second operand of subtract and divide.
      return HloProtoEvaluator(
                 *root,
                 [&](int64 operand_index,
                     int64 operand_handle) -> StatusOr<Literal> {
                   if (operand_index == 0) {
                     return AnalyzeUpperBound(operand_handle);
                   } else {
                     TF_ASSIGN_OR_RETURN(auto lower_bound,
                                         AnalyzeLowerBound(operand_handle));
                     if (opcode == HloOpcode::kSubtract ||
                         !IsValueEffectiveInteger(operand_handle)) {
                       return lower_bound;
                     }
                     // Because in many cases the lower bound of a value is
                     // integer 0, instead of throwing an divide-by-zero error
                     // at compile time, we set the bound defer the check to
                     // runtime. In those cases we use the upper-bound of
                     // first operand as a placeholder.
                     HloEvaluator evaluator;
                     auto zero =
                         LiteralUtil::Zero(lower_bound.shape().element_type());
                     zero =
                         zero.Broadcast(lower_bound.shape(), {}).ValueOrDie();
                     TF_ASSIGN_OR_RETURN(
                         auto lower_bound_is_zero,
                         evaluator.EvaluateElementwiseCompareOp(
                             ComparisonDirection::kEq, lower_bound, zero));

                     auto one =
                         LiteralUtil::One(lower_bound.shape().element_type());
                     one = one.Broadcast(lower_bound.shape(), {}).ValueOrDie();
                     auto result = evaluator.EvaluateElementwiseTernaryOp(
                         HloOpcode::kSelect, lower_bound_is_zero, one,
                         lower_bound);
                     return result;
                   }
                 })
          .Evaluate();
    }
    default:
      return AnalyzeConstantValue(handle, ValueInferenceMode::kUpperBound);
  }
}

StatusOr<Literal> ValueInference::AnalyzeLowerBound(int64 handle) {
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      builder_->LookUpInstructionByHandle(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64 dimension = root->dimensions(0);
      int64 operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                          builder_->LookUpInstructionByHandle(operand_handle));
      if (!operand_proto->shape().is_dynamic_dimension(dimension)) {
        return LiteralUtil::CreateR0<int32>(
            operand_proto->shape().dimensions(dimension));
      } else {
        return LiteralUtil::CreateR0<int32>(0);
      }
    }
    case HloOpcode::kAbs: {
      // lower-bound(abs(operand)) = min(abs(lower-bound(operand)),
      // abs(upper-bound(operand)))
      int64 operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(auto lower_bound, AnalyzeLowerBound(operand_handle));
      TF_ASSIGN_OR_RETURN(auto upper_bound, AnalyzeUpperBound(operand_handle));
      HloEvaluator evaluator;
      TF_ASSIGN_OR_RETURN(
          auto lower_bound_abs,
          evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kAbs, lower_bound));
      TF_ASSIGN_OR_RETURN(
          auto upper_bound_abs,
          evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kAbs, upper_bound));
      return evaluator.EvaluateElementwiseBinaryOp(
          HloOpcode::kMinimum, lower_bound_abs, upper_bound_abs);
    }
    case HloOpcode::kNegate: {
      // lower-bound(negate(operand)) = negate(upper-bound(operand))
      int64 operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(auto upper_bound, AnalyzeUpperBound(operand_handle));
      HloEvaluator evaluator;
      return evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kNegate,
                                                  upper_bound);
    }
    case HloOpcode::kSubtract:
    case HloOpcode::kDivide: {
      // Upper bound is used for second operand of subtract and divide.
      return HloProtoEvaluator(*root,
                               [&](int64 operand_index, int64 operand_handle) {
                                 if (operand_index == 0) {
                                   return AnalyzeLowerBound(operand_handle);
                                 } else {
                                   return AnalyzeUpperBound(operand_handle);
                                 }
                               })
          .Evaluate();
    }
    default:
      return AnalyzeConstantValue(handle, ValueInferenceMode::kLowerBound);
  }
}

StatusOr<Literal> ValueInference::AnalyzeConstant(int64 handle) {
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      builder_->LookUpInstructionByHandle(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64 dimension = root->dimensions(0);
      int64 operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                          builder_->LookUpInstructionByHandle(operand_handle));
      if (operand_proto->shape().is_dynamic_dimension(dimension)) {
        // The value is dynamic, we return garbage data here and mask them out
        // later.
        return CreateGarbageLiteral(Shape(root->shape()));
      } else {
        return LiteralUtil::CreateR0<int32>(
            operand_proto->shape().dimensions(dimension));
      }
    }
    case HloOpcode::kSubtract:
    case HloOpcode::kCos:
    case HloOpcode::kSin:
    case HloOpcode::kNegate:
    case HloOpcode::kAbs:
    case HloOpcode::kDivide: {
      return HloProtoEvaluator(*root,
                               [&](int64 operand_index, int64 operand_handle) {
                                 return AnalyzeConstant(operand_handle);
                               })
          .Evaluate();
    }
    default:
      return AnalyzeConstantValue(handle, ValueInferenceMode::kValue);
  }
}

StatusOr<Literal> ValueInference::AnalyzeIsDynamic(int64 handle,
                                                   ValueInferenceMode mode) {
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      builder_->LookUpInstructionByHandle(handle));
  // Invariant check.
  TF_RET_CHECK(root);
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64 dimension = root->dimensions(0);
      int64 operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                          builder_->LookUpInstructionByHandle(operand_handle));
      if (mode == ValueInferenceMode::kLowerBound ||
          mode == ValueInferenceMode::kUpperBound) {
        // The bound of dynamic dimension is not dynamic.
        return LiteralUtil::CreateR0<bool>(false);
      }
      // The value of dynamic dimension is dynamic.
      return LiteralUtil::CreateR0<bool>(
          operand_proto->shape().is_dynamic_dimension(dimension));
    }
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCos:
    case HloOpcode::kClz:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kRsqrt:
    case HloOpcode::kLogistic:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kConvert:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kTanh: {
      // Forward operand as they don't change if a value is dynamic or static.
      int64 operand_handle = root->operand_ids(0);
      return AnalyzeIsDynamic(operand_handle, mode);
    }
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kDivide:
    case HloOpcode::kComplex:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kCompare:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical: {
      return HloProtoEvaluator(*root,
                               [&](int64 operand_index, int64 operand_handle) {
                                 return AnalyzeIsDynamic(operand_handle, mode);
                               })
          .WithPrimitiveType(PRED)
          .WithOpCode(HloOpcode::kOr)
          .Evaluate();
    }
    case HloOpcode::kTuple:
    case HloOpcode::kTranspose:
    case HloOpcode::kSlice:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReshape:
    case HloOpcode::kPad: {
      return HloProtoEvaluator(*root,
                               [&](int64 operand_index, int64 operand_handle) {
                                 return AnalyzeIsDynamic(operand_handle, mode);
                               })
          .WithPrimitiveType(PRED)
          .Evaluate();
    }
    case HloOpcode::kGetTupleElement: {
      int64 operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                          builder_->LookUpInstructionByHandle(operand_handle));
      TF_ASSIGN_OR_RETURN(HloOpcode operand_opcode,
                          StringToHloOpcode(operand_proto->opcode()));
      if (operand_opcode == HloOpcode::kParameter) {
        // Don't materialize the whole parameter if it's followed by a GTE.
        return CreatePredLiteral(true, Shape(root->shape()));
      }
      return HloProtoEvaluator(*root,
                               [&](int64 operand_index, int64 operand_handle) {
                                 return AnalyzeIsDynamic(operand_handle, mode);
                               })
          .WithPrimitiveType(PRED)
          .Evaluate();
    }

    case HloOpcode::kReduce: {
      std::vector<std::unique_ptr<HloInstruction>> operand_storage;
      absl::flat_hash_map<int64, HloInstruction*> operand_map;
      absl::flat_hash_map<int64, HloComputation*> computation_map;

      Shape scalar_shape = ShapeUtil::MakeScalarShape(xla::PRED);
      HloComputation::Builder b("reduce_or");
      auto lhs = b.AddInstruction(
          HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
      auto rhs = b.AddInstruction(
          HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
      b.AddInstruction(
          HloInstruction::CreateBinary(scalar_shape, HloOpcode::kOr, lhs, rhs));
      auto reduce_computation = b.Build();
      return HloProtoEvaluator(*root,
                               [&](int64 operand_index, int64 operand_handle) {
                                 return AnalyzeIsDynamic(operand_handle, mode);
                               })
          .WithPrimitiveType(PRED)
          .WithComputation(std::move(reduce_computation))
          .Evaluate();
    }
    case HloOpcode::kConstant:
    case HloOpcode::kIota: {
      return CreatePredLiteral(false, Shape(root->shape()));
    }
    case HloOpcode::kParameter: {
      return CreatePredLiteral(true, Shape(root->shape()));
    }
    case HloOpcode::kSelect: {
      TF_ASSIGN_OR_RETURN(OptionaLiteral optional_selector_literal,
                          AnalyzeOptionalConstant(root->operand_ids(0),
                                                  ValueInferenceMode::kValue));
      TF_ASSIGN_OR_RETURN(LiteralSlice lhs,
                          AnalyzeIsDynamic(root->operand_ids(1), mode));
      TF_ASSIGN_OR_RETURN(LiteralSlice rhs,
                          AnalyzeIsDynamic(root->operand_ids(2), mode));

      auto result = CreatePredLiteral(true, Shape(root->shape()));

      result.MutableEachCell<bool>(
          [&](absl::Span<const int64> indices, bool value) {
            absl::optional<bool> optional_selector =
                optional_selector_literal.Get<bool>(indices);

            bool lhs_value = lhs.Get<bool>(indices);
            bool rhs_value = rhs.Get<bool>(indices);
            if (optional_selector.has_value()) {
              // Manually evaluate the selection without using Evaluator.
              if (*optional_selector) {
                return lhs_value;
              } else {
                return rhs_value;
              }
            } else {
              // Conservatively assume value is dynamic if selector is dynamic.
              return true;
            }
          });
      return result;
    }
    case HloOpcode::kGather: {
      TF_ASSIGN_OR_RETURN(OptionaLiteral optional_selector_literal,
                          AnalyzeOptionalConstant(root->operand_ids(1),
                                                  ValueInferenceMode::kValue));
      if (!optional_selector_literal.AllValid()) {
        // Conservatively assume result are dynamic.
        return CreatePredLiteral(true, Shape(root->shape()));
      }
      return HloProtoEvaluator(*root,
                               [&](int64 operand_index, int64 operand_handle) {
                                 if (operand_index == 1) {
                                   return AnalyzeConstant(operand_handle);
                                 } else {
                                   return AnalyzeIsDynamic(operand_handle,
                                                           mode);
                                 }
                               })
          .WithPrimitiveType(PRED)
          .Evaluate();
    }
    case HloOpcode::kCustomCall: {
      if (root->custom_call_target() == "SetBound") {
        if (mode == ValueInferenceMode::kLowerBound ||
            mode == ValueInferenceMode::kUpperBound) {
          return CreatePredLiteral(false, Shape(root->shape()));
        } else {
          return CreatePredLiteral(true, Shape(root->shape()));
        }
      } else {
        return InvalidArgument(
            "Dynamic inferencing on custom call %s is not supported",
            root->DebugString());
      }

      break;
    }
    default:
      return Unimplemented("Can't infer upper bound through %s: %s",
                           root->opcode(), root->DebugString());
  }
}

bool ValueInference::IsValueEffectiveInteger(int64 handle) {
  const HloInstructionProto* instr =
      builder_->LookUpInstructionByHandle(handle).ValueOrDie();
  if (primitive_util::IsIntegralType(instr->shape().element_type())) {
    return true;
  }
  // Also returns true if this is a convert that converts an integer to float.
  HloOpcode opcode = StringToHloOpcode(instr->opcode()).ValueOrDie();
  if (opcode != HloOpcode::kConvert) {
    return false;
  }
  const HloInstructionProto* parent =
      builder_->LookUpInstructionByHandle(instr->operand_ids(0)).ValueOrDie();
  if (primitive_util::IsIntegralType(parent->shape().element_type())) {
    return true;
  }
  return false;
}

StatusOr<OptionaLiteral> ValueInference::AnalyzeOptionalConstant(
    int64 handle, ValueInferenceMode mode) {
  auto analyze_constant = [this](int64 handle,
                                 ValueInferenceMode mode) -> StatusOr<Literal> {
    switch (mode) {
      case kValue:
        return AnalyzeConstant(handle);
      case kUpperBound:
        return AnalyzeUpperBound(handle);
      case kLowerBound:
        return AnalyzeLowerBound(handle);
    }
  };
  TF_ASSIGN_OR_RETURN(Literal value, analyze_constant(handle, mode));
  TF_ASSIGN_OR_RETURN(Literal mask, AnalyzeIsDynamic(handle, mode));
  return OptionaLiteral(std::move(value), std::move(mask));
}

}  // namespace xla

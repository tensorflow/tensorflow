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
#include "tensorflow/compiler/xla/literal.h"
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
  Literal literal = LiteralUtil::CreateR0(pred);
  Literal literal_broadcast =
      literal
          .Broadcast(ShapeUtil::ChangeElementType(Shape(reference_shape), PRED),
                     {})
          .ValueOrDie();
  return literal_broadcast;
}

Literal CreateZeroLiteral(const Shape& reference_shape) {
  if (reference_shape.IsTuple()) {
    std::vector<Literal> sub_literals;
    for (const Shape& shape : reference_shape.tuple_shapes()) {
      sub_literals.emplace_back(CreateZeroLiteral(shape));
    }
    return Literal::MoveIntoTuple(absl::MakeSpan(sub_literals));
  }
  Literal literal = LiteralUtil::Zero(reference_shape.element_type());
  Literal literal_broadcast =
      literal.Broadcast(reference_shape, {}).ValueOrDie();

  return literal_broadcast;
}

using GetOperand = std::function<StatusOr<LiteralSlice>(int64 operand_index,
                                                        int64 opreand_handle)>;

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
          HloInstruction::CreateConstant(literal.Clone());
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

StatusOr<Literal> ValueInference::AnalyzeConstantLiteral(int64 handle) {
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
        // The value is dynamic, but we return a 0 here as garbage data.
        return CreateZeroLiteral(Shape(root->shape()));
      } else {
        return LiteralUtil::CreateR0<int32>(
            operand_proto->shape().dimensions(dimension));
      }
    }
      // Non functional ops.
    case HloOpcode::kRng:
    case HloOpcode::kAllReduce:
      // TODO(b/33009255): Implement constant folding for cross replica sum.
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kCall:
      // TODO(b/32495713): We aren't checking the to_apply computation itself,
      // so we conservatively say that computations containing the Call op
      // cannot be constant.  We cannot set is_functional=false in other similar
      // cases since we're already relying on IsConstant to return true.
    case HloOpcode::kCustomCall:
    case HloOpcode::kWhile:
    case HloOpcode::kConditional:
      // TODO(b/32495713): We aren't checking the condition and body
      // computations themselves.
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
    case HloOpcode::kParameter: {
      // The values are dynamic, but we return 0s here as garbage data.
      return CreateZeroLiteral(Shape(root->shape()));
    }
    case HloOpcode::kGetTupleElement: {
      int64 operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                          builder_->LookUpInstructionByHandle(operand_handle));
      TF_ASSIGN_OR_RETURN(HloOpcode operand_opcode,
                          StringToHloOpcode(operand_proto->opcode()));
      if (operand_opcode == HloOpcode::kParameter) {
        // Don't materialize the whole parameter if it's followed by a GTE.
        return CreateZeroLiteral(Shape(root->shape()));
      }
      return HloProtoEvaluator(*root,
                               [&](int64 operand_index, int64 operand_handle) {
                                 return AnalyzeConstant(operand_handle);
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
                                 return AnalyzeConstant(operand_handle);
                               })
          .WithComputation(std::move(computation))
          .Evaluate();
    }
    default:
      return HloProtoEvaluator(*root,
                               [&](int64 operand_index, int64 operand_handle) {
                                 return AnalyzeConstant(operand_handle);
                               })
          .Evaluate();
  }
}

StatusOr<Literal> ValueInference::AnalyzeIsDynamicLiteral(int64 handle) {
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      builder_->LookUpInstructionByHandle(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64 dimension = root->dimensions(0);
      int64 operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                          builder_->LookUpInstructionByHandle(operand_handle));
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
      TF_ASSIGN_OR_RETURN(auto literal, AnalyzeIsDynamic(operand_handle));
      return literal.Clone();
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
                                 return AnalyzeIsDynamic(operand_handle);
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
                                 return AnalyzeIsDynamic(operand_handle);
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
                                 return AnalyzeIsDynamic(operand_handle);
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
                                 return AnalyzeIsDynamic(operand_handle);
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
      TF_ASSIGN_OR_RETURN(OptionaLiteralSlice optional_selector_literal,
                          AnalyzeOptionalConstant(root->operand_ids(0)));
      TF_ASSIGN_OR_RETURN(LiteralSlice lhs,
                          AnalyzeIsDynamic(root->operand_ids(1)));
      TF_ASSIGN_OR_RETURN(LiteralSlice rhs,
                          AnalyzeIsDynamic(root->operand_ids(2)));

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
      TF_ASSIGN_OR_RETURN(OptionaLiteralSlice optional_selector_literal,
                          AnalyzeOptionalConstant(root->operand_ids(1)));
      if (!optional_selector_literal.AllValid()) {
        // Conservatively assume result are dynamic.
        return CreatePredLiteral(true, Shape(root->shape()));
      }
      return HloProtoEvaluator(*root,
                               [&](int64 operand_index, int64 operand_handle) {
                                 if (operand_index == 1) {
                                   return AnalyzeConstant(operand_handle);
                                 } else {
                                   return AnalyzeIsDynamic(operand_handle);
                                 }
                               })
          .WithPrimitiveType(PRED)
          .Evaluate();
    }
    case HloOpcode::kCustomCall: {
      if (root->custom_call_target() == "SetBound") {
        return CreatePredLiteral(true, Shape(root->shape()));
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

StatusOr<LiteralSlice> ValueInference::AnalyzeIsDynamic(int64 handle) {
  if (is_dynamic_.contains(handle)) {
    return LiteralSlice(is_dynamic_[handle]);
  }
  TF_ASSIGN_OR_RETURN(Literal literal, AnalyzeIsDynamicLiteral(handle));
  is_dynamic_[handle] = std::move(literal);
  return LiteralSlice(is_dynamic_[handle]);
}

StatusOr<LiteralSlice> ValueInference::AnalyzeConstant(int64 handle) {
  if (constant_.contains(handle)) {
    return LiteralSlice(constant_[handle]);
  }
  TF_ASSIGN_OR_RETURN(Literal literal, AnalyzeConstantLiteral(handle));
  constant_[handle] = std::move(literal);
  return LiteralSlice(constant_[handle]);
}

StatusOr<OptionaLiteralSlice> ValueInference::AnalyzeOptionalConstant(
    int64 handle) {
  TF_ASSIGN_OR_RETURN(LiteralSlice value, AnalyzeConstant(handle));
  TF_ASSIGN_OR_RETURN(LiteralSlice mask, AnalyzeIsDynamic(handle));
  return OptionaLiteralSlice(value, mask);
}

}  // namespace xla

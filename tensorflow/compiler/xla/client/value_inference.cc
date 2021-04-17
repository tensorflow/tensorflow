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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

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



// HloProtoEvaluator evaluates an hlo proto and returns a literal. The user has
// to provide operand as literals through the get_operand function.
struct HloProtoEvaluator {
  explicit HloProtoEvaluator(HloInstructionProto inst)
      : inst(std::move(inst)),
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

  // WithOpCode changes the opcode of the instruction being evaluated.
  HloProtoEvaluator& WithOperands(absl::Span<Literal> operands) {
    this->operands = operands;
    return *this;
  }

  StatusOr<Literal> Evaluate() {
    // Evaluate the instruction by swapping it's operands with constant
    // instructions with given literals.
    HloComputation::Builder builder("EmptyComputation");
    absl::flat_hash_map<int64, HloInstruction*> operand_map;
    for (int64 i = 0; i < inst.operand_ids_size(); ++i) {
      int64 operand_handle = inst.operand_ids(i);
      std::unique_ptr<HloInstruction> operand =
          HloInstruction::CreateConstant(operands[i].Clone());
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

  HloModule module;
  absl::Span<Literal> operands;
  HloComputation* computation = nullptr;
  absl::optional<PrimitiveType> primitive_type = absl::nullopt;
  absl::optional<HloOpcode> opcode = absl::nullopt;
};

enum PostorderDFSNodeType {
  // This node is about figuring out the constant value.
  kConstantValue = 0,
  // This node is about figuring out the constant bound.
  kConstantUpperBound,
  kConstantLowerBound,
  // This node is about figuring out whether a value is dynamic.
  kValueIsDynamic,
  // This node is about figuring out whether a bound value is dynamic. It's
  // similar to kValueIsDynamic, but views shape bound as static values.
  kBoundIsDynamic,
};

// Each node in the postorder traversal tree may depend on traversing the
// values of the node's children.
struct PostorderDFSDep {
  explicit PostorderDFSDep(int64 handle, PostorderDFSNodeType type)
      : handle(handle), type(type) {}
  int64 handle;
  PostorderDFSNodeType type;
};

// This function represents the logic to visit a node once its dependencies
// (operands) are all resolved.
using Visit = std::function<StatusOr<Literal>(absl::Span<Literal>)>;
// Convenient specializations of Visit function for different operands.
using Visit0D = std::function<StatusOr<Literal>()>;
using Visit1D = std::function<StatusOr<Literal>(Literal)>;
using Visit2D = std::function<StatusOr<Literal>(Literal, Literal)>;

// A postorder dfs node can be visited once its dependency requests are all
// fulfilled.
struct PostorderDFSNode {
  PostorderDFSNode& AddDependency(int64 handle, PostorderDFSNodeType type) {
    dependencies.emplace_back(handle, type);
    return *this;
  }

  PostorderDFSNode& AddVisit(const Visit& visit) {
    this->visit = visit;
    return *this;
  }

  PostorderDFSNode& AddVisit(const Visit0D& visit) {
    this->visit = [visit](absl::Span<Literal> literals) { return visit(); };
    return *this;
  }

  PostorderDFSNode& AddVisit(const Visit1D& visit) {
    this->visit = [visit](absl::Span<Literal> literals) {
      return visit(std::move(literals[0]));
    };
    return *this;
  }

  PostorderDFSNode& AddVisit(const Visit2D& visit) {
    this->visit = [visit](absl::Span<Literal> literals) {
      return visit(std::move(literals[0]), std::move(literals[1]));
    };
    return *this;
  }

  std::vector<PostorderDFSDep> dependencies;
  Visit visit;
};

// Convert an interger handle to HloInstructionProto.
using HandleToInstruction = std::function<const HloInstructionProto*(int64)>;
using HandleToComputation = std::function<const HloComputationProto*(int64)>;

struct PostorderDFSVisitor {
  PostorderDFSVisitor(HandleToInstruction handle_to_instruction,
                        HandleToComputation handle_to_computation)
      : handle_to_instruction(handle_to_instruction),
        handle_to_computation(handle_to_computation) {}

  StatusOr<PostorderDFSNode> AnalyzeUpperBound(int64 handle);
  StatusOr<PostorderDFSNode> AnalyzeLowerBound(int64 handle);
  StatusOr<PostorderDFSNode> AnalyzeIsDynamic(int64 handle,
                                              PostorderDFSNodeType type);
  StatusOr<PostorderDFSNode> AnalyzeConstant(int64 handle);
  StatusOr<PostorderDFSNode> AnalyzeConstantValueFallback(int64 handle,
                                                  PostorderDFSNodeType type);

  StatusOr<Literal> PostOrderDFSVisit(int64 handle, PostorderDFSNodeType type);

  // Returns true if a value represented by `handle` is an integeral type or
  // just got converted from an integral type to floating point type.
  bool IsValueEffectiveInteger(int64 handle) {
    const HloInstructionProto* instr = handle_to_instruction(handle);
    if (primitive_util::IsIntegralType(instr->shape().element_type())) {
      return true;
    }
    // Also returns true if this is a convert that converts an integer to float.
    HloOpcode opcode = StringToHloOpcode(instr->opcode()).ValueOrDie();
    if (opcode != HloOpcode::kConvert) {
      return false;
    }
    const HloInstructionProto* parent =
        handle_to_instruction(instr->operand_ids(0));
    if (primitive_util::IsIntegralType(parent->shape().element_type())) {
      return true;
    }
    return false;
  }

  absl::flat_hash_map<std::pair<int64, PostorderDFSNodeType>, Literal>
      evaluated;
  HandleToInstruction handle_to_instruction;
  HandleToComputation handle_to_computation;
};

}  // namespace

// Analyze a tensor's constant value, upper-bound value or lower-bound value.
StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeConstantValueFallback(
    int64 handle, PostorderDFSNodeType type) {
  const HloInstructionProto* root = handle_to_instruction(handle);
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  PostorderDFSNode result;
  for (auto operand_id : root->operand_ids()) {
    result.AddDependency(operand_id, type);
  }
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
      return result.AddVisit([root](absl::Span<Literal>) {
        // The value is dynamic. We return a garbage literal here, which
        // will be masked out later.
        return CreateGarbageLiteral(Shape(root->shape()));
      });
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
      const HloInstructionProto* operand_proto =
          handle_to_instruction(operand_handle);
      TF_ASSIGN_OR_RETURN(HloOpcode operand_opcode,
                          StringToHloOpcode(operand_proto->opcode()));
      if (operand_opcode == HloOpcode::kParameter) {
        return PostorderDFSNode().AddVisit([root](absl::Span<Literal>) {
          // The value is dynamic. We return a garbage literal here, which
          // will be masked out later.
          return CreateGarbageLiteral(Shape(root->shape()));
        });
      }

      return result.AddVisit([root](absl::Span<Literal> operands) {
        return HloProtoEvaluator(*root)
            .WithOperands(operands)
            .Evaluate();
      });
    }
    case HloOpcode::kReduce:
    case HloOpcode::kScatter:
    case HloOpcode::kReduceWindow: {
      const HloComputationProto* computation_proto =
          handle_to_computation(root->called_computation_ids(0));
      return result.AddVisit(
          [root, computation_proto](
              absl::Span<Literal> operands) -> StatusOr<Literal> {
            TF_ASSIGN_OR_RETURN(
                auto computation,
                HloComputation::CreateFromProto(*computation_proto, {}));
            return HloProtoEvaluator(*root)
                .WithOperands(operands)
                .WithComputation(std::move(computation))
                .Evaluate();
          });
    }
    default: {
      return result.AddVisit([root](absl::Span<Literal> operands) {
        return HloProtoEvaluator(*root).WithOperands(operands).Evaluate();
      });
    }
  }
}

StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeUpperBound(
    int64 handle) {
  const HloInstructionProto* root = handle_to_instruction(handle);
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64 dimension = root->dimensions(0);
      int64 operand_handle = root->operand_ids(0);
      const HloInstructionProto* operand_proto =
          handle_to_instruction(operand_handle);
      return PostorderDFSNode().AddVisit(
          [operand_proto, dimension]() -> StatusOr<Literal> {
            return LiteralUtil::CreateR0<int32>(
                operand_proto->shape().dimensions(dimension));
          });
    }
    case HloOpcode::kAbs: {
      // upper-bound(abs(operand)) = max(abs(lower-bound(operand)),
      //                                 abs(upper-bound(operand)))
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantLowerBound)
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantUpperBound)
          .AddVisit([](Literal lower_bound,
                       Literal upper_bound) -> StatusOr<Literal> {
            HloEvaluator evaluator;
            TF_ASSIGN_OR_RETURN(auto lower_bound_abs,
                                evaluator.EvaluateElementwiseUnaryOp(
                                    HloOpcode::kAbs, lower_bound));
            TF_ASSIGN_OR_RETURN(auto upper_bound_abs,
                                evaluator.EvaluateElementwiseUnaryOp(
                                    HloOpcode::kAbs, upper_bound));
            return evaluator.EvaluateElementwiseBinaryOp(
                HloOpcode::kMaximum, lower_bound_abs, upper_bound_abs);
          });
    }
    case HloOpcode::kNegate: {
      // upper-bound(negate(operand)) = negate(lower-bound(operand))
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantLowerBound)
          .AddVisit([](Literal lower_bound) -> StatusOr<Literal> {
            HloEvaluator evaluator;
            return evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kNegate,
                                                        lower_bound);
          });
    }
    case HloOpcode::kSubtract:
    case HloOpcode::kDivide: {
      // Lower-bound is used for second operand of subtract and divide.
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantUpperBound)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kConstantLowerBound)
          .AddVisit(
              [root, opcode, this](Literal upper_bound,
                                   Literal lower_bound) -> StatusOr<Literal> {
                if (opcode == HloOpcode::kDivide &&
                    this->IsValueEffectiveInteger(root->operand_ids(1))) {
                  // Because in many cases the lower bound of a value is
                  // integer 0, instead of throwing an divide-by-zero error
                  // at compile time, we set the bound defer the check to
                  // runtime. In those cases we use the upper-bound of
                  // first operand as a placeholder.
                  HloEvaluator evaluator;
                  auto zero =
                      LiteralUtil::Zero(lower_bound.shape().element_type());
                  zero = zero.Broadcast(lower_bound.shape(), {}).ValueOrDie();
                  TF_ASSIGN_OR_RETURN(
                      auto lower_bound_is_zero,
                      evaluator.EvaluateElementwiseCompareOp(
                          ComparisonDirection::kEq, lower_bound, zero));

                  auto one =
                      LiteralUtil::One(lower_bound.shape().element_type());
                  one = one.Broadcast(lower_bound.shape(), {}).ValueOrDie();
                  TF_ASSIGN_OR_RETURN(
                      lower_bound, evaluator.EvaluateElementwiseTernaryOp(
                                       HloOpcode::kSelect, lower_bound_is_zero,
                                       one, lower_bound));
                }
                std::vector<Literal> new_operands;
                new_operands.emplace_back(std::move(upper_bound));
                new_operands.emplace_back(std::move(lower_bound));
                return HloProtoEvaluator(*root)
                    .WithOperands(absl::MakeSpan(new_operands))
                    .Evaluate();
              });
    }
    default:
      return AnalyzeConstantValueFallback(
          handle, PostorderDFSNodeType::kConstantUpperBound);
  }
}

StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeLowerBound(
    int64 handle) {
  const HloInstructionProto* root = handle_to_instruction(handle);
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64 dimension = root->dimensions(0);
      int64 operand_handle = root->operand_ids(0);
      const HloInstructionProto* operand_proto =
          handle_to_instruction(operand_handle);
      return PostorderDFSNode().AddVisit(
          [dimension, operand_proto]() -> StatusOr<Literal> {
            if (operand_proto->shape().is_dynamic_dimension(dimension)) {
              return LiteralUtil::CreateR0<int32>(0);
            } else {
              return LiteralUtil::CreateR0<int32>(
                  operand_proto->shape().dimensions(dimension));
            }
          });
    }
    case HloOpcode::kAbs: {
      // lower-bound(abs(operand)) = min(abs(lower-bound(operand)),
      // abs(upper-bound(operand)))
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantLowerBound)
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantUpperBound)
          .AddVisit([](Literal lower_bound,
                       Literal upper_bound) -> StatusOr<Literal> {
            HloEvaluator evaluator;
            TF_ASSIGN_OR_RETURN(auto lower_bound_abs,
                                evaluator.EvaluateElementwiseUnaryOp(
                                    HloOpcode::kAbs, lower_bound));
            TF_ASSIGN_OR_RETURN(auto upper_bound_abs,
                                evaluator.EvaluateElementwiseUnaryOp(
                                    HloOpcode::kAbs, upper_bound));
            return evaluator.EvaluateElementwiseBinaryOp(
                HloOpcode::kMinimum, lower_bound_abs, upper_bound_abs);
          });
    }
    case HloOpcode::kNegate: {
      // lower-bound(negate(operand)) = negate(upper-bound(operand))
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantUpperBound)
          .AddVisit([](Literal upper_bound) -> StatusOr<Literal> {
            HloEvaluator evaluator;
            return evaluator.EvaluateElementwiseUnaryOp(HloOpcode::kNegate,
                                                        upper_bound);
          });
    }
    case HloOpcode::kSubtract:
    case HloOpcode::kDivide: {
      // Upper bound is used for second operand of subtract and divide.
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantLowerBound)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kConstantUpperBound)
          .AddVisit([root](absl::Span<Literal> operands) -> StatusOr<Literal> {
            return HloProtoEvaluator(*root).WithOperands(operands).Evaluate();
          });
    }
    default:
      return AnalyzeConstantValueFallback(
          handle, PostorderDFSNodeType::kConstantLowerBound);
  }
}

StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeConstant(
    int64 handle) {
  const HloInstructionProto* root = handle_to_instruction(handle);
  HloOpcode opcode = StringToHloOpcode(root->opcode()).ValueOrDie();
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64 dimension = root->dimensions(0);
      int64 operand_handle = root->operand_ids(0);
      const HloInstructionProto* operand_proto =
          handle_to_instruction(operand_handle);
      return PostorderDFSNode().AddVisit(
          [operand_proto, dimension, root]() -> StatusOr<Literal> {
            if (operand_proto->shape().is_dynamic_dimension(dimension)) {
              // The value is dynamic, we return garbage data here and mask them
              // out later.
              return CreateGarbageLiteral(Shape(root->shape()));
            } else {
              return LiteralUtil::CreateR0<int32>(
                  operand_proto->shape().dimensions(dimension));
            }
          });
    }
    case HloOpcode::kSubtract:
    case HloOpcode::kCos:
    case HloOpcode::kSin:
    case HloOpcode::kNegate:
    case HloOpcode::kAbs:
    case HloOpcode::kDivide: {
      PostorderDFSNode result;
      for (auto operand_id : root->operand_ids()) {
        result.AddDependency(operand_id, PostorderDFSNodeType::kConstantValue);
      }
      return result.AddVisit(
          [root](absl::Span<Literal> operands) -> StatusOr<Literal> {
            return HloProtoEvaluator(*root).WithOperands(operands).Evaluate();
          });
    }
    default:
      return AnalyzeConstantValueFallback(handle,
                                          PostorderDFSNodeType::kConstantValue);
  }
}

StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeIsDynamic(
    int64 handle, PostorderDFSNodeType type) {
  const HloInstructionProto* root = handle_to_instruction(handle);
  // Invariant check.
  TF_RET_CHECK(root);
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  PostorderDFSNode result;
  for (auto operand_id : root->operand_ids()) {
    result.AddDependency(operand_id, type);
  }
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64 dimension = root->dimensions(0);
      int64 operand_handle = root->operand_ids(0);
      const HloInstructionProto* operand_proto =
          handle_to_instruction(operand_handle);
      return PostorderDFSNode().AddVisit([operand_proto, dimension,
                                          type]() -> StatusOr<Literal> {
        if (type == PostorderDFSNodeType::kBoundIsDynamic) {
          // The bound of dynamic dimension is not dynamic.
          return LiteralUtil::CreateR0<bool>(false);
        }
        // The value of dynamic dimension is dynamic.
        return LiteralUtil::CreateR0<bool>(
            operand_proto->shape().is_dynamic_dimension(dimension));
      });
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
      return result.AddVisit([](Literal operand) { return operand; });
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
      return result.AddVisit([root](absl::Span<Literal> operands) {
        return HloProtoEvaluator(*root)
            .WithOperands(operands)
            .WithPrimitiveType(PRED)
            .WithOpCode(HloOpcode::kOr)
            .Evaluate();
      });
    }
    case HloOpcode::kTuple:
    case HloOpcode::kTranspose:
    case HloOpcode::kSlice:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReshape:
    case HloOpcode::kPad: {
      return result.AddVisit([root](absl::Span<Literal> operands) {
        return HloProtoEvaluator(*root)
            .WithOperands(operands)
            .WithPrimitiveType(PRED)
            .Evaluate();
      });
    }
    case HloOpcode::kGetTupleElement: {
      int64 operand_handle = root->operand_ids(0);
      const HloInstructionProto* operand_proto =
          handle_to_instruction(operand_handle);
      TF_ASSIGN_OR_RETURN(HloOpcode operand_opcode,
                          StringToHloOpcode(operand_proto->opcode()));
      if (operand_opcode == HloOpcode::kParameter) {
        PostorderDFSNode().AddVisit([root]() -> StatusOr<Literal> {
          // Don't materialize the whole parameter if it's followed by a GTE.
          return CreatePredLiteral(true, Shape(root->shape()));
        });
      }
      return result.AddVisit([root](absl::Span<Literal> operands) {
        return HloProtoEvaluator(*root)
            .WithOperands(operands)
            .WithPrimitiveType(PRED)
            .Evaluate();
      });
    }

    case HloOpcode::kReduce: {
      return result.AddVisit([root](absl::Span<Literal> operands) {
        Shape scalar_shape = ShapeUtil::MakeScalarShape(xla::PRED);
        HloComputation::Builder b("reduce_or");
        auto lhs = b.AddInstruction(
            HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
        auto rhs = b.AddInstruction(
            HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
        b.AddInstruction(HloInstruction::CreateBinary(
            scalar_shape, HloOpcode::kOr, lhs, rhs));
        auto reduce_computation = b.Build();
        return HloProtoEvaluator(*root)
            .WithOperands(operands)
            .WithPrimitiveType(PRED)
            .WithComputation(std::move(reduce_computation))
            .Evaluate();
      });
    }
    case HloOpcode::kConstant:
    case HloOpcode::kIota: {
      return result.AddVisit(
          [root]() { return CreatePredLiteral(false, Shape(root->shape())); });
    }
    case HloOpcode::kParameter: {
      return result.AddVisit(
          [root]() { return CreatePredLiteral(true, Shape(root->shape())); });
    }
    case HloOpcode::kSelect: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantValue)
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kValueIsDynamic)
          // lhs dependency.
          .AddDependency(root->operand_ids(1), type)
          // rhs dependency.
          .AddDependency(root->operand_ids(2), type)
          .AddVisit([root](absl::Span<Literal> operands) -> StatusOr<Literal> {
            OptionalLiteral optional_selector_literal(std::move(operands[0]),
                                                      std::move(operands[1]));
            Literal lhs = std::move(operands[2]);
            Literal rhs = std::move(operands[3]);
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
                    // Conservatively assume value is dynamic if selector is
                    // dynamic.
                    return true;
                  }
                });
            return result;
          });
    }
    case HloOpcode::kGather: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0), type)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kConstantValue)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kValueIsDynamic)
          .AddVisit([root](absl::Span<Literal> operands) -> StatusOr<Literal> {
            OptionalLiteral optional_selector_literal(std::move(operands[1]),
                                                      std::move(operands[2]));

            if (!optional_selector_literal.AllValid()) {
              // Conservatively assume results are dynamic.
              return CreatePredLiteral(true, Shape(root->shape()));
            }
            std::vector<Literal> new_operands;
            new_operands.emplace_back(std::move(operands[0]));
            new_operands.emplace_back(
                optional_selector_literal.GetValue()->Clone());

            return HloProtoEvaluator(*root)
                .WithOperands(absl::MakeSpan(new_operands))
                .WithPrimitiveType(PRED)
                .Evaluate();
          });
    }
    case HloOpcode::kCustomCall: {
      if (root->custom_call_target() == "SetBound") {
        return PostorderDFSNode().AddVisit([type, root]() -> StatusOr<Literal> {
          if (type == PostorderDFSNodeType::kBoundIsDynamic) {
            return CreatePredLiteral(false, Shape(root->shape()));
          } else {
            return CreatePredLiteral(true, Shape(root->shape()));
          }
        });
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

StatusOr<Literal> PostorderDFSVisitor::PostOrderDFSVisit(
    int64 handle, PostorderDFSNodeType type) {
  enum VisitState {
    kUnvisited = 0,
    kVisiting,
    kVisited,
  };

  struct WorkItem {
    explicit WorkItem(int64 handle, PostorderDFSNodeType type, VisitState state)
        : handle(handle), type(type), state(state) {}
    int64 handle;
    PostorderDFSNodeType type;
    VisitState state;
    PostorderDFSNode node;
  };

  std::vector<WorkItem> stack;
  stack.push_back(WorkItem(handle, type, kUnvisited));
  while (!stack.empty()) {
    WorkItem& item = stack.back();
    VLOG(1) << "stack top" << handle_to_instruction(item.handle)->DebugString();
    if (item.state == kVisiting) {
      VLOG(1) << "visiting";
      // The operands are ready, visit the node itself.

      // Gather dependencies.
      std::vector<Literal> literals;
      for (const PostorderDFSDep& dep : item.node.dependencies) {
        std::pair<int64, PostorderDFSNodeType> key(dep.handle, dep.type);
        TF_RET_CHECK(evaluated.contains(key));
        literals.emplace_back(evaluated.at(key).Clone());
      }
      VLOG(1) << "start visiting";
      TF_ASSIGN_OR_RETURN(auto literal,
                          item.node.visit(absl::MakeSpan(literals)));
      VLOG(1) << "end visiting: " << literal.ToString();
      std::pair<int64, PostorderDFSNodeType> key(item.handle, item.type);
      evaluated[key] = std::move(literal);
      stack.pop_back();
      continue;
    }
    // This is the first time we see this node, we want to gather its
    // dependenceis.
    VLOG(1) << "unvisited";
    item.state = kVisiting;
    PostorderDFSNode node;
    switch (item.type) {
      case PostorderDFSNodeType::kConstantValue: {
      TF_ASSIGN_OR_RETURN(node, AnalyzeConstant(item.handle));
        break;
      }
      case PostorderDFSNodeType::kConstantLowerBound: {
      TF_ASSIGN_OR_RETURN(node, AnalyzeLowerBound(item.handle));
        break;
      }
      case PostorderDFSNodeType::kConstantUpperBound: {
        TF_ASSIGN_OR_RETURN(node, AnalyzeUpperBound(item.handle));
        break;
      }
      case PostorderDFSNodeType::kBoundIsDynamic:
      case PostorderDFSNodeType::kValueIsDynamic: {
        TF_ASSIGN_OR_RETURN(node, AnalyzeIsDynamic(item.handle, item.type));
        break;
      }
    }
    // Store the node which is needed when its dependencies are resolved.
    item.node = node;
    // Enqueue dependencies into the stack.
    for (const PostorderDFSDep& dep : node.dependencies) {
      VLOG(1) << "dep" << handle_to_instruction(dep.handle)->DebugString();
      stack.emplace_back(dep.handle, dep.type, kUnvisited);
    }
  }
  VLOG(1) << "done" << evaluated[std::make_pair(handle, type)].ToString();
  return evaluated[std::make_pair(handle, type)].Clone();
}

StatusOr<Literal> ValueInference::AnalyzeIsDynamic(XlaOp op) {
  PostorderDFSVisitor visitor(
      [&](int64 handle) {
        return builder_->LookUpInstructionByHandle(handle).ValueOrDie();
      },
      [&](int64 handle) { return &(builder_->embedded_[handle]); });
  return visitor.PostOrderDFSVisit(op.handle(),
                                   PostorderDFSNodeType::kValueIsDynamic);
}

StatusOr<OptionalLiteral> ValueInference::AnalyzeConstant(
    XlaOp op, ValueInferenceMode mode) {
  PostorderDFSVisitor visitor(
      [&](int64 handle) {
        return builder_->LookUpInstructionByHandle(handle).ValueOrDie();
      },
      [&](int64 handle) { return &(builder_->embedded_[handle]); });
  switch (mode) {
    case ValueInferenceMode::kLowerBound: {
      TF_ASSIGN_OR_RETURN(
          Literal value,
          visitor.PostOrderDFSVisit(op.handle(),
                                    PostorderDFSNodeType::kConstantLowerBound));
      TF_ASSIGN_OR_RETURN(
          Literal mask,
          visitor.PostOrderDFSVisit(op.handle(),
                                    PostorderDFSNodeType::kBoundIsDynamic));
      return OptionalLiteral(std::move(value), std::move(mask));
    }

    case ValueInferenceMode::kUpperBound: {
      TF_ASSIGN_OR_RETURN(
          Literal value,
          visitor.PostOrderDFSVisit(op.handle(),
                                    PostorderDFSNodeType::kConstantUpperBound));
      TF_ASSIGN_OR_RETURN(
          Literal mask,
          visitor.PostOrderDFSVisit(op.handle(),
                                    PostorderDFSNodeType::kBoundIsDynamic));

      return OptionalLiteral(std::move(value), std::move(mask));
    }
    case ValueInferenceMode::kValue: {
      TF_ASSIGN_OR_RETURN(
          Literal value,
          visitor.PostOrderDFSVisit(op.handle(),
                                    PostorderDFSNodeType::kConstantValue));
      TF_ASSIGN_OR_RETURN(
          Literal mask,
          visitor.PostOrderDFSVisit(op.handle(),
                                    PostorderDFSNodeType::kValueIsDynamic));
      return OptionalLiteral(std::move(value), std::move(mask));
    }
  }
}

}  // namespace xla

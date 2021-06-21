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

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"
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
  Literal literal = LiteralUtil::Zero(element_type);
  return literal.Broadcast(reference_shape, {}).ValueOrDie();
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

  // WithPrimitiveType changes the primitive type of the instruction being
  // evaluated.
  HloProtoEvaluator& WithPrimitiveType(PrimitiveType new_primitive_type) {
    primitive_type = new_primitive_type;
    return *this;
  }

  // WithOpCode changes the opcode of the instruction being evaluated.
  HloProtoEvaluator& WithOpCode(HloOpcode new_opcode) {
    opcode = new_opcode;
    return *this;
  }

  // WithOperands changes the operands of the instruction being evaluated.
  HloProtoEvaluator& WithOperands(absl::Span<Literal> operands) {
    this->operands = operands;
    return *this;
  }

  // When WithSubshape is set, the result tuple shape will be decomposed and
  // specific the literal will be returned.
  HloProtoEvaluator& WithSubshape(ShapeIndex shape_index) {
    this->shape_index = std::move(shape_index);
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
    if (shape_index.empty()) {
      return evaluator.Evaluate(module.entry_computation()->root_instruction());
    } else {
      TF_ASSIGN_OR_RETURN(
          auto result,
          evaluator.Evaluate(module.entry_computation()->root_instruction()));
      return result.SubLiteral(this->shape_index);
    }
  }

  HloInstructionProto inst;

  HloModule module;
  absl::Span<Literal> operands;
  ShapeIndex shape_index = {};
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

struct InferenceContext {
  explicit InferenceContext(ShapeIndex shape_index,
                            std::vector<int64> caller_operand_handles)
      : shape_index(std::move(shape_index)),
        caller_operand_handles(std::move(caller_operand_handles)) {}
  // `shape_index` represents the subshape that we care about in the inference.
  // It is used to avoid meterializing the whole tuple when we only care about a
  // sub tensor of it.
  ShapeIndex shape_index;

  // caller_operand_handles is a stack that helps argument forwarding. The top
  // of the stack represents the tensor to be forwarded to the
  // parameter of the inner most function. E.g.,:
  // inner_true_computation {
  //   inner_p0 = param()
  //   ...
  // }
  //
  // true_computaion {
  //   p0 = param()
  //   conditional(pred, p0, inner_true_computation,
  //                     ...)
  // }
  //
  // main {
  //   op = ..
  //   conditional(pred, op, true_computation, ...)
  // }
  //
  // In this case, when we analyze inner_true_computation, the
  // `caller_operand_handlers` will be [op, p0] -- p0 is what should be
  // forwarded to inner_p0 and op is what should be forwarded to p0. similarly,
  // when we analyze true_computation, the `caller_operand_handlers` will be
  // [op].
  std::vector<int64> caller_operand_handles;
};

// Each node in the postorder traversal tree may depend on traversing the
// values of the node's children.
struct PostorderDFSDep {
  explicit PostorderDFSDep(int64 handle, PostorderDFSNodeType type,
                           InferenceContext context, std::string annotation)
      : handle(handle),
        type(type),
        context(std::move(context)),
        annotation(std::move(annotation)) {}
  int64 handle;
  PostorderDFSNodeType type;
  InferenceContext context;
  std::string annotation;
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
struct TF_MUST_USE_RESULT PostorderDFSNode {
  PostorderDFSNode& AddDependency(int64 handle, PostorderDFSNodeType type,
                                  InferenceContext context,
                                  std::string annotation = "") {
    dependencies.emplace_back(handle, type, std::move(context),
                              std::move(annotation));
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

  StatusOr<PostorderDFSNode> AnalyzeUpperBound(int64 handle,
                                               InferenceContext context);
  StatusOr<PostorderDFSNode> AnalyzeLowerBound(int64 handle,
                                               InferenceContext context);
  StatusOr<PostorderDFSNode> AnalyzeIsDynamic(int64 handle,
                                              PostorderDFSNodeType type,
                                              InferenceContext context);
  StatusOr<PostorderDFSNode> AnalyzeConstant(int64 handle,
                                             InferenceContext context);
  StatusOr<PostorderDFSNode> AnalyzeConstantValueFallback(
      int64 handle, PostorderDFSNodeType type, InferenceContext context);

  StatusOr<Literal> PostOrderDFSVisit(int64 handle, PostorderDFSNodeType type);

  // Returns true if a value represented by `handle` is an integeral type or
  // a floating pointer type that just got converted from an integral type.
  // E.g.,:
  // 1 -> true
  // float(1) -> true
  // 1.1 -> false
  // 1.0 -> false -- We don't know the concrete value at runtime, except for its
  // type.
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

  absl::flat_hash_map<int64, Literal> evaluated;
  HandleToInstruction handle_to_instruction;
  HandleToComputation handle_to_computation;
};

}  // namespace

// Analyze a tensor's constant value, upper-bound value or lower-bound value.
StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeConstantValueFallback(
    int64 handle, PostorderDFSNodeType type, InferenceContext context) {
  const HloInstructionProto* root = handle_to_instruction(handle);
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  PostorderDFSNode result;
  // By default, the dependencies of current node are its operands.
  for (auto operand_id : root->operand_ids()) {
    InferenceContext dep_context = context;
    dep_context.shape_index = {};
    result.AddDependency(operand_id, type, dep_context);
  }
  switch (opcode) {
      // Non functional ops.
    case HloOpcode::kRng:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceScatter:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kCall:
    case HloOpcode::kCustomCall:
    case HloOpcode::kWhile:
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
    case HloOpcode::kParameter: {
      if (opcode == HloOpcode::kParameter &&
          !context.caller_operand_handles.empty()) {
        int64 caller_operand = context.caller_operand_handles.back();
        context.caller_operand_handles.pop_back();
        return result.AddDependency(caller_operand, type, context)
            .AddVisit([](Literal literal) { return literal; });
      }
      return PostorderDFSNode().AddVisit([root, context](absl::Span<Literal>) {
        // The value is dynamic. We return a garbage literal here, which
        // will be masked out later.
        return CreateGarbageLiteral(
            ShapeUtil::GetSubshape(Shape(root->shape()), context.shape_index));
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
      return InvalidArgument(
          "AnalyzeConstantValueFallback can't handle opcode: %s",
          root->opcode());
    }

    case HloOpcode::kConditional: {
      auto node = PostorderDFSNode();
      auto* conditional_proto = root;
      InferenceContext predicate_context = context;
      predicate_context.shape_index = {};
      // Add dependencies to analyze the predicate of the conditional.
      node.AddDependency(conditional_proto->operand_ids(0),
                         PostorderDFSNodeType::kConstantValue,
                         predicate_context)
          .AddDependency(conditional_proto->operand_ids(0),
                         PostorderDFSNodeType::kValueIsDynamic,
                         predicate_context);
      const int64 branch_size =
          conditional_proto->called_computation_ids_size();
      for (int64 i = 0; i < branch_size; ++i) {
        int64 branch_root =
            handle_to_computation(conditional_proto->called_computation_ids(i))
                ->root_id();
        InferenceContext branch_context = context;
        branch_context.caller_operand_handles.push_back(
            conditional_proto->operand_ids(i + 1));
        node.AddDependency(branch_root, PostorderDFSNodeType::kConstantValue,
                           branch_context);
      }
      return node.AddVisit(
          [](absl::Span<Literal> operands) -> StatusOr<Literal> {
            int64 pred_is_dynamic = operands[1].Get<bool>({});
            if (pred_is_dynamic) {
              // If predicate is dynamic, return the value of the first branch
              // -- If all branches return the same value, this is the value
              // that we want; If not, the value will be masked anyway so the
              // value inside doesn't matter.
              return std::move(operands[2]);
            } else {
              // If predicate is static, return the value of the given branch.
              int64 branch_index = 0;
              if (operands[0].shape().element_type() == PRED) {
                if (operands[0].Get<bool>({})) {
                  branch_index = 0;
                } else {
                  branch_index = 1;
                }
              } else {
                branch_index = operands[0].GetIntegralAsS64({}).value();
              }
              const int64 branch_dynamism_index = 2 + branch_index;
              return std::move(operands[branch_dynamism_index]);
            }
          });
    }
    case HloOpcode::kGetTupleElement: {
      int64 operand_handle = root->operand_ids(0);
      PostorderDFSNode result;
      context.shape_index.push_front(root->tuple_index());
      return PostorderDFSNode()
          .AddDependency(operand_handle, type, context)
          .AddVisit([](Literal operand) { return operand; });
    }
    case HloOpcode::kReduce:
    case HloOpcode::kScatter:
    case HloOpcode::kReduceWindow: {
      const HloComputationProto* computation_proto =
          handle_to_computation(root->called_computation_ids(0));
      return result.AddVisit(
          [root, computation_proto,
           context](absl::Span<Literal> operands) -> StatusOr<Literal> {
            TF_ASSIGN_OR_RETURN(
                auto computation,
                HloComputation::CreateFromProto(*computation_proto, {}));
            return HloProtoEvaluator(*root)
                .WithOperands(operands)
                .WithComputation(std::move(computation))
                .WithSubshape(context.shape_index)
                .Evaluate();
          });
    }
    default: {
      if (opcode == HloOpcode::kTuple && !context.shape_index.empty()) {
        // There could be many operands of a tuple, but only one that we are
        // interested in, represented by `tuple_operand_index`.
        int64 tuple_operand_index = context.shape_index.front();
        InferenceContext tuple_operand_context = context;
        tuple_operand_context.shape_index.pop_front();
        return PostorderDFSNode()
            .AddDependency(root->operand_ids(tuple_operand_index), type,
                           tuple_operand_context)
            .AddVisit([](Literal operand) { return operand; });
      }
      return result.AddVisit([root](absl::Span<Literal> operands) {
        return HloProtoEvaluator(*root).WithOperands(operands).Evaluate();
      });
    }
  }
}

StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeUpperBound(
    int64 handle, InferenceContext context) {
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
                         PostorderDFSNodeType::kConstantLowerBound, context)
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantUpperBound, context)
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
                         PostorderDFSNodeType::kConstantLowerBound, context)
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
                         PostorderDFSNodeType::kConstantUpperBound, context)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kConstantLowerBound, context)
          .AddVisit([root, opcode, this](
                        Literal upper_bound,
                        Literal lower_bound) -> StatusOr<Literal> {
            if (opcode == HloOpcode::kDivide &&
                this->IsValueEffectiveInteger(root->operand_ids(1))) {
              // Because in many cases the lower bound of a value is
              // integer 0, instead of throwing an divide-by-zero error
              // at compile time, we set the bound defer the check to
              // runtime. In those cases we use the upper-bound of
              // first operand as a placeholder.
              HloEvaluator evaluator;
              auto zero = LiteralUtil::Zero(lower_bound.shape().element_type());
              zero = zero.Broadcast(lower_bound.shape(), {}).ValueOrDie();
              TF_ASSIGN_OR_RETURN(
                  auto lower_bound_is_zero,
                  evaluator.EvaluateElementwiseCompareOp(
                      ComparisonDirection::kEq, lower_bound, zero));

              auto one = LiteralUtil::One(lower_bound.shape().element_type());
              one = one.Broadcast(lower_bound.shape(), {}).ValueOrDie();
              TF_ASSIGN_OR_RETURN(
                  lower_bound, evaluator.EvaluateElementwiseTernaryOp(
                                   HloOpcode::kSelect, lower_bound_is_zero, one,
                                   lower_bound));
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
          handle, PostorderDFSNodeType::kConstantUpperBound, context);
  }
}

StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeLowerBound(
    int64 handle, InferenceContext context) {
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
                         PostorderDFSNodeType::kConstantLowerBound, context)
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantUpperBound, context)
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
                         PostorderDFSNodeType::kConstantUpperBound, context)
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
                         PostorderDFSNodeType::kConstantLowerBound, context)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kConstantUpperBound, context)
          .AddVisit([root](absl::Span<Literal> operands) -> StatusOr<Literal> {
            return HloProtoEvaluator(*root).WithOperands(operands).Evaluate();
          });
    }
    default:
      return AnalyzeConstantValueFallback(
          handle, PostorderDFSNodeType::kConstantLowerBound, context);
  }
}

StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeConstant(
    int64 handle, InferenceContext context) {
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
        result.AddDependency(operand_id, PostorderDFSNodeType::kConstantValue,
                             context);
      }
      return result.AddVisit(
          [root](absl::Span<Literal> operands) -> StatusOr<Literal> {
            return HloProtoEvaluator(*root).WithOperands(operands).Evaluate();
          });
    }
    default:
      return AnalyzeConstantValueFallback(
          handle, PostorderDFSNodeType::kConstantValue, context);
  }
}

StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeIsDynamic(
    int64 handle, PostorderDFSNodeType type, InferenceContext context) {
  const HloInstructionProto* root = handle_to_instruction(handle);
  // Invariant check.
  TF_RET_CHECK(root);
  VLOG(1) << "Analyzing IsDynamic on " << root->DebugString();
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  PostorderDFSNode result;
  for (auto operand_id : root->operand_ids()) {
    InferenceContext dep_context = context;
    dep_context.shape_index = {};
    result.AddDependency(operand_id, type, dep_context);
  }
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64 dimension = root->dimensions(0);
      int64 operand_handle = root->operand_ids(0);
      const HloInstructionProto* operand_proto =
          handle_to_instruction(operand_handle);
      return PostorderDFSNode().AddVisit(
          [operand_proto, dimension, type]() -> StatusOr<Literal> {
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
    case HloOpcode::kReverse:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReshape:
    case HloOpcode::kPad: {
      if (opcode == HloOpcode::kTuple && !context.shape_index.empty()) {
        // There could be many operands of a tuple, but only one that we are
        // interested in, represented by `tuple_operand_index`.
        int64 tuple_operand_index = context.shape_index.front();
        InferenceContext tuple_operand_context = context;
        tuple_operand_context.shape_index.pop_front();
        return PostorderDFSNode()
            .AddDependency(root->operand_ids(tuple_operand_index), type,
                           tuple_operand_context)
            .AddVisit([](Literal operand) { return operand; });
      }
      return result.AddVisit([root](absl::Span<Literal> operands) {
        return HloProtoEvaluator(*root)
            .WithOperands(operands)
            .WithPrimitiveType(PRED)
            .Evaluate();
      });
    }
    case HloOpcode::kConditional: {
      auto node = PostorderDFSNode();
      auto* conditional_proto = root;
      InferenceContext predicate_context = context;
      predicate_context.shape_index = {};
      // Add dependencies to analyze the predicate of the conditional.
      node.AddDependency(conditional_proto->operand_ids(0),
                         PostorderDFSNodeType::kConstantValue,
                         predicate_context)
          .AddDependency(conditional_proto->operand_ids(0),
                         PostorderDFSNodeType::kValueIsDynamic,
                         predicate_context);
      const int64 branch_size =
          conditional_proto->called_computation_ids_size();
      for (int64 i = 0; i < branch_size; ++i) {
        int64 branch_root =
            handle_to_computation(conditional_proto->called_computation_ids(i))
                ->root_id();
        InferenceContext branch_context = context;
        branch_context.caller_operand_handles.push_back(
            conditional_proto->operand_ids(i + 1));
        node.AddDependency(branch_root, PostorderDFSNodeType::kConstantValue,
                           branch_context,
                           absl::StrFormat("branch %lld's value", i))
            .AddDependency(branch_root, PostorderDFSNodeType::kValueIsDynamic,
                           branch_context,
                           absl::StrFormat("branch %lld's dynamism", i));
      }
      // Predicate uses 2 dependencies:
      // 0: Predicate value.
      // 1: Predicate is dynamic.
      // Each branch i has 2 dependenices:
      // 2*i: Branch result value
      // 2*i + 1: Branch value is dynamic.
      return node.AddVisit(
          [root, branch_size,
           context](absl::Span<Literal> operands) -> StatusOr<Literal> {
            int64 pred_is_dynamic = operands[1].Get<bool>({});
            auto result = CreatePredLiteral(
                true, ShapeUtil::GetSubshape(Shape(root->shape()),
                                             context.shape_index));
            if (pred_is_dynamic) {
              VLOG(1) << "predict is dynamic value" << result.ToString();
              // If predicate is dynamic, the result is only static if all
              // branches are static and return the same value.
              result.MutableEachCell<bool>([&](absl::Span<const int64> indices,
                                               bool value) {
                string branch_value = operands[2].GetAsString(indices, {});
                for (int64 i = 0; i < branch_size; ++i) {
                  const int64 branch_value_index = 2 + 2 * i;
                  const int64 branch_dynamism_index = 2 + 2 * i + 1;
                  auto branch_is_dynamic =
                      operands[branch_dynamism_index].Get<bool>(indices);
                  if (branch_is_dynamic) {
                    return true;
                  }

                  if (branch_value !=
                      operands[branch_value_index].GetAsString(indices, {})) {
                    return true;
                  }
                }
                // Value of the branch is static.
                return false;
              });
              return result;
            } else {
              VLOG(1) << "predict is constant value";
              // If predicate is static, return true if given branch result
              // value is dynamic.
              int64 branch_index = 0;
              if (operands[0].shape().element_type() == PRED) {
                if (operands[0].Get<bool>({})) {
                  branch_index = 0;
                } else {
                  branch_index = 1;
                }
              } else {
                branch_index = operands[0].GetIntegralAsS64({}).value();
              }
              const int64 branch_dynamism_index = 2 + 2 * branch_index + 1;
              return std::move(operands[branch_dynamism_index]);
            }
          });
    }
    case HloOpcode::kGetTupleElement: {
      int64 operand_handle = root->operand_ids(0);
      PostorderDFSNode result;
      context.shape_index.push_front(root->tuple_index());
      return PostorderDFSNode()
          .AddDependency(operand_handle, type, context)
          .AddVisit([](Literal operand) { return operand; });
    }

    case HloOpcode::kReduce: {
      return result.AddVisit([root, context](absl::Span<Literal> operands) {
        Shape root_shape = Shape(root->shape());
        Shape scalar_shape = ShapeUtil::MakeScalarShape(xla::PRED);
        std::unique_ptr<HloComputation> reduce_or;
        if (root_shape.IsTuple()) {
          // Variadic reduce.
          HloComputation::Builder b("reduce_or");
          std::vector<HloInstruction*> results;
          results.reserve(root_shape.tuple_shapes_size());
          for (int64 i = 0; i < root_shape.tuple_shapes_size(); ++i) {
            auto lhs = b.AddInstruction(
                HloInstruction::CreateParameter(i, scalar_shape, "lhs"));
            auto rhs = b.AddInstruction(HloInstruction::CreateParameter(
                i + root_shape.tuple_shapes_size(), scalar_shape, "rhs"));
            results.push_back(b.AddInstruction(HloInstruction::CreateBinary(
                scalar_shape, HloOpcode::kOr, lhs, rhs)));
          }
          b.AddInstruction(HloInstruction::CreateTuple(results));
          reduce_or = b.Build();
        } else {
          HloComputation::Builder b("reduce_or");
          auto lhs = b.AddInstruction(
              HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
          auto rhs = b.AddInstruction(
              HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
          b.AddInstruction(HloInstruction::CreateBinary(
              scalar_shape, HloOpcode::kOr, lhs, rhs));
          reduce_or = b.Build();
        }

        return HloProtoEvaluator(*root)
            .WithOperands(operands)
            .WithPrimitiveType(PRED)
            .WithComputation(std::move(reduce_or))
            // Reduce could produce tuple shape, only fetch what we need.
            .WithSubshape(context.shape_index)
            .Evaluate();
      });
    }
    case HloOpcode::kConstant:
    case HloOpcode::kIota: {
      return result.AddVisit(
          [root]() { return CreatePredLiteral(false, Shape(root->shape())); });
    }
    case HloOpcode::kParameter: {
      if (opcode == HloOpcode::kParameter &&
          !context.caller_operand_handles.empty()) {
        int64 caller_operand = context.caller_operand_handles.back();
        context.caller_operand_handles.pop_back();
        return result.AddDependency(caller_operand, type, context)
            .AddVisit([](Literal literal) { return literal; });
      }
      return result.AddVisit([root, context]() {
        return CreatePredLiteral(
            true,
            ShapeUtil::GetSubshape(Shape(root->shape()), context.shape_index));
      });
    }
    case HloOpcode::kSelect: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantValue, context)
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kValueIsDynamic, context)
          // lhs dependency.
          .AddDependency(root->operand_ids(1), type, context)
          // rhs dependency.
          .AddDependency(root->operand_ids(2), type, context)
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
          .AddDependency(root->operand_ids(0), type, context)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kConstantValue, context)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kValueIsDynamic, context)
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
            if (root->literal().shape().element_type() == TUPLE) {
              // First literal of SetBound contains bounds, second literal
              // contains dynamism indicators.
              return Literal::CreateFromProto(
                  root->literal().tuple_literals(1));
            } else {
              return Literal::CreateFromProto(root->literal());
            }
          }
        });
      } else {
        return InvalidArgument(
            "Dynamic inferencing on custom call %s is not supported",
            root->DebugString());
      }

      break;
    }

    case HloOpcode::kCall:
    case HloOpcode::kWhile: {
      return PostorderDFSNode().AddVisit([root,
                                          context]() -> StatusOr<Literal> {
        return CreatePredLiteral(
            true,
            ShapeUtil::GetSubshape(Shape(root->shape()), context.shape_index));
      });
      break;
    }
    default:
      return Unimplemented("Can't infer dynamism through %s: %s",
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

  int64 unique_id = 0;
  struct WorkItem {
    explicit WorkItem(int64 handle, InferenceContext context,
                      PostorderDFSNodeType type, VisitState state, int64 id)
        : handle(handle),
          context(std::move(context)),
          type(type),
          state(state),
          id(id) {}
    int64 handle;  // Handle of the node in the graph.
    InferenceContext context;
    PostorderDFSNodeType type;
    VisitState state;
    Visit visit;  // The handler to call once the dependencies are resolved into
                  // literal form.
    int64 id;     // Unique id in the work queue, starting from 0.
    std::vector<int64> dependencies;
  };

  std::vector<WorkItem> stack;
  WorkItem root(handle, InferenceContext({}, {}), type, kUnvisited,
                unique_id++);
  stack.push_back(root);
  while (!stack.empty()) {
    WorkItem& item = stack.back();
    VLOG(1) << "stack top shape index: " << item.context.shape_index.ToString();
    VLOG(1) << "stack top "
            << handle_to_instruction(item.handle)->DebugString();
    if (item.state == kVisiting) {
      VLOG(1) << "visiting";
      // The dependencies are ready, visit the node itself.

      // Gather dependencies and transform them into literals.
      std::vector<Literal> literals;
      for (int64 dep_id : item.dependencies) {
        TF_RET_CHECK(evaluated.contains(dep_id));
        literals.emplace_back(evaluated.at(dep_id).Clone());
      }
      VLOG(1) << "start visiting with" << literals.size();
      TF_ASSIGN_OR_RETURN(auto literal, item.visit(absl::MakeSpan(literals)));
      VLOG(1) << "end visiting: " << literal.ToString();
      evaluated[item.id] = std::move(literal);
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
        VLOG(1) << "constant value";
        TF_ASSIGN_OR_RETURN(node, AnalyzeConstant(item.handle, item.context));
        break;
      }
      case PostorderDFSNodeType::kConstantLowerBound: {
        VLOG(1) << "constant lower bound";
        TF_ASSIGN_OR_RETURN(node, AnalyzeLowerBound(item.handle, item.context));
        break;
      }
      case PostorderDFSNodeType::kConstantUpperBound: {
        VLOG(1) << "constant upper bound";
        TF_ASSIGN_OR_RETURN(node, AnalyzeUpperBound(item.handle, item.context));
        break;
      }
      case PostorderDFSNodeType::kBoundIsDynamic:
      case PostorderDFSNodeType::kValueIsDynamic: {
        VLOG(1) << "value is dynamic";
        TF_ASSIGN_OR_RETURN(
            node, AnalyzeIsDynamic(item.handle, item.type, item.context));
        break;
      }
    }
    // Store the visit function which is needed when its dependencies are
    // resolved.
    item.visit = node.visit;

    // Dependencies of this item have id in the range of [unique_id, unique_id +
    // dependencies.size())
    for (int64 i = 0; i < node.dependencies.size(); ++i) {
      item.dependencies.push_back(unique_id + i);
    }
    // Enqueue dependencies into the stack. `item` shouldn't be accessed after
    // this point.
    for (const PostorderDFSDep& dep : node.dependencies) {
      VLOG(1) << "dep " << dep.annotation << ":"
              << handle_to_instruction(dep.handle)->DebugString();
      stack.emplace_back(dep.handle, dep.context, dep.type, kUnvisited,
                         unique_id++);
    }
  }
  VLOG(1) << "done" << evaluated[root.id].ToString();
  return evaluated[root.id].Clone();
}

StatusOr<Literal> ValueInference::AnalyzeIsDynamic(XlaOp op) {
  PostorderDFSVisitor visitor(
      [&](int64 handle) {
        return builder_->LookUpInstructionByHandle(handle).ValueOrDie();
      },
      [&](int64 handle) { return &(builder_->embedded_[handle]); });
  auto result = visitor.PostOrderDFSVisit(
      op.handle(), PostorderDFSNodeType::kValueIsDynamic);
  return result;
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

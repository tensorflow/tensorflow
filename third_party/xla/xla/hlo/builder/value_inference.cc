/* Copyright 2021 The OpenXLA Authors.

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
#include "xla/hlo/builder/value_inference.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/lib/gtl/value_or_die.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {
Literal CreatePredLiteral(bool pred, const Shape& reference_shape) {
  if (reference_shape.IsTuple()) {
    std::vector<Literal> sub_literals;
    const auto& reference_shape_tuple_shapes = reference_shape.tuple_shapes();
    sub_literals.reserve(reference_shape_tuple_shapes.size());
    for (const Shape& shape : reference_shape_tuple_shapes) {
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
          .value();
  return literal_broadcast;
}

Literal CreateS64Literal(int64_t value, const Shape& reference_shape) {
  if (reference_shape.IsTuple()) {
    std::vector<Literal> sub_literals;
    const auto& reference_shape_tuple_shapes = reference_shape.tuple_shapes();
    sub_literals.reserve(reference_shape_tuple_shapes.size());
    for (const Shape& shape : reference_shape_tuple_shapes) {
      sub_literals.emplace_back(CreateS64Literal(value, shape));
    }
    return Literal::MoveIntoTuple(absl::MakeSpan(sub_literals));
  }
  PrimitiveType element_type = reference_shape.element_type();
  if (element_type == TOKEN) {
    return LiteralUtil::CreateToken();
  }
  Literal literal = LiteralUtil::CreateR0<int64_t>(value);
  return literal
      .Broadcast(ShapeUtil::ChangeElementType(reference_shape, S64), {})
      .value();
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
  Literal literal = LiteralUtil::One(element_type);
  return literal.Broadcast(reference_shape, {}).value();
}

// HloProtoEvaluator evaluates an hlo proto and returns a literal. The user has
// to provide operand as literals through the get_operand function.
struct HloProtoEvaluator {
  explicit HloProtoEvaluator(HloEvaluator& evaluator, HloInstructionProto inst)
      : evaluator(evaluator),
        inst(std::move(inst)),
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

  absl::StatusOr<Literal> Evaluate() {
    // Evaluate the instruction by swapping it's operands with constant
    // instructions with given literals.
    HloComputation::Builder builder("EmptyComputation");
    absl::flat_hash_map<int64_t, HloInstruction*> operand_map;
    for (int64_t i = 0; i < inst.operand_ids_size(); ++i) {
      int64_t operand_handle = inst.operand_ids(i);
      std::unique_ptr<HloInstruction> operand =
          HloInstruction::CreateConstant(operands[i].Clone());
      operand_map[operand_handle] = operand.get();
      builder.AddInstruction(std::move(operand));
    }

    if (primitive_type.has_value()) {
      TF_ASSIGN_OR_RETURN(Shape shape, Shape::FromProto(inst.shape()));
      *inst.mutable_shape() =
          ShapeUtil::ChangeElementType(shape, primitive_type.value()).ToProto();
    }
    if (opcode.has_value()) {
      *inst.mutable_opcode() = std::string(HloOpcodeString(opcode.value()));
    }
    absl::flat_hash_map<int64_t, HloComputation*> computation_map;
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
    if (shape_index.empty()) {
      return evaluator.Evaluate(module.entry_computation()->root_instruction());
    } else {
      TF_ASSIGN_OR_RETURN(
          auto result,
          evaluator.Evaluate(module.entry_computation()->root_instruction()));
      return result.SubLiteral(this->shape_index);
    }
  }

  HloEvaluator& evaluator;
  HloInstructionProto inst;

  HloModule module;
  absl::Span<Literal> operands;
  ShapeIndex shape_index = {};
  HloComputation* computation = nullptr;
  std::optional<PrimitiveType> primitive_type = std::nullopt;
  std::optional<HloOpcode> opcode = std::nullopt;
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

std::string PostorderDFSNodeTypeToString(PostorderDFSNodeType type) {
  switch (type) {
    case kConstantValue:
      return "kConstantValue";
    case kConstantUpperBound:
      return "kConstantUpperBound";
    case kConstantLowerBound:
      return "kConstantLowerBound";
    case kValueIsDynamic:
      return "kValueIsDynamic";
    case kBoundIsDynamic:
      return "kBoundIsDynamic";
  }
}

struct InferenceContext {
  explicit InferenceContext(ShapeIndex shape_index,
                            std::vector<int64_t> caller_operand_handles)
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
  std::vector<int64_t> caller_operand_handles;
};

// Each node in the postorder traversal tree may depend on traversing the
// values of the node's children.
struct PostorderDFSDep {
  explicit PostorderDFSDep(int64_t handle, PostorderDFSNodeType type,
                           InferenceContext context, std::string annotation)
      : handle(handle),
        type(type),
        context(std::move(context)),
        annotation(std::move(annotation)) {}
  int64_t handle;
  PostorderDFSNodeType type;
  InferenceContext context;
  std::string annotation;
};

// This function represents the logic to visit a node once its dependencies
// (operands) are all resolved.
using Visit = std::function<absl::StatusOr<Literal>(absl::Span<Literal>)>;
// Convenient specializations of Visit function for different operands.
using Visit0D = std::function<absl::StatusOr<Literal>()>;
using Visit1D = std::function<absl::StatusOr<Literal>(Literal)>;
using Visit2D = std::function<absl::StatusOr<Literal>(Literal, Literal)>;

// A postorder dfs node can be visited once its dependency requests are all
// fulfilled.
struct [[nodiscard]] PostorderDFSNode {
  PostorderDFSNode& AddDependency(int64_t handle, PostorderDFSNodeType type,
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
using HandleToInstruction =
    std::function<absl::StatusOr<const HloInstructionProto*>(int64_t)>;
using HandleToComputation = std::function<const HloComputationProto*(int64_t)>;

struct PostorderDFSVisitor {
  PostorderDFSVisitor(HloEvaluator& evaluator,
                      HandleToInstruction handle_to_instruction,
                      HandleToComputation handle_to_computation)
      : evaluator(evaluator),
        handle_to_instruction(handle_to_instruction),
        handle_to_computation(handle_to_computation) {}

  absl::StatusOr<PostorderDFSNode> AnalyzeUpperBound(int64_t handle,
                                                     InferenceContext context);
  absl::StatusOr<PostorderDFSNode> AnalyzeLowerBound(int64_t handle,
                                                     InferenceContext context);
  absl::StatusOr<PostorderDFSNode> AnalyzeIsDynamic(int64_t handle,
                                                    PostorderDFSNodeType type,
                                                    InferenceContext context);
  absl::StatusOr<PostorderDFSNode> AnalyzeConstant(int64_t handle,
                                                   InferenceContext context);
  absl::StatusOr<PostorderDFSNode> AnalyzeConstantValueFallback(
      int64_t handle, PostorderDFSNodeType type, InferenceContext context);

  absl::StatusOr<Literal> PostOrderDFSVisit(int64_t handle,
                                            PostorderDFSNodeType type);

  // Returns true if a value represented by `handle` is an integeral type or
  // a floating pointer type that just got converted from an integral type.
  // E.g.,:
  // int(a) -> true
  // float(int(a)) -> true
  // float(a) -> false -- We don't know the concrete value of `a` at
  // compile time, except for its type.
  bool IsValueEffectiveInteger(int64_t handle) {
    // handle_to_instruction's failure status should be checked by parent.
    const HloInstructionProto* instr = handle_to_instruction(handle).value();
    if (primitive_util::IsIntegralType(instr->shape().element_type())) {
      return true;
    }
    // Also returns true if this is a convert that converts an integer to float.
    HloOpcode opcode = StringToHloOpcode(instr->opcode()).value();
    if (opcode != HloOpcode::kConvert) {
      return false;
    }
    const HloInstructionProto* parent =
        handle_to_instruction(instr->operand_ids(0)).value();
    if (primitive_util::IsIntegralType(parent->shape().element_type())) {
      return true;
    }
    return false;
  }

  // Checks the size of outputs and inputs. Returns true if any of them has size
  // beyond kLargeShapeElementLimit and the instruction needs evaluation (e.g.,
  // kGetDimensionSize or kSetDimensionSize doesn't need evaluation).
  bool IsInstructionOverLimit(const HloInstructionProto* proto,
                              const InferenceContext& context) {
    auto subshape = std::make_unique<Shape>(ShapeUtil::GetSubshape(
        tsl::gtl::ValueOrDie(Shape::FromProto(proto->shape())),
        context.shape_index));

    if (subshape->IsArray() &&
        ShapeUtil::ElementsIn(*subshape) > kLargeShapeElementLimit) {
      return true;
    }
    HloOpcode opcode = StringToHloOpcode(proto->opcode()).value();
    for (int64_t operand_id : proto->operand_ids()) {
      const HloInstructionProto* operand =
          handle_to_instruction(operand_id).value();
      auto operand_shape = std::make_unique<Shape>(operand->shape());

      if (operand_shape->IsArray() &&
          ShapeUtil::ElementsIn(*operand_shape) > kLargeShapeElementLimit &&
          opcode != HloOpcode::kGetDimensionSize &&
          opcode != HloOpcode::kSetDimensionSize) {
        return true;
      }
    }
    return false;
  }

  struct CacheKey {
    CacheKey(int64_t handle, InferenceContext context,
             PostorderDFSNodeType type)
        : handle(handle), context(context), type(type) {}
    int64_t handle;
    InferenceContext context;
    PostorderDFSNodeType type;

    template <typename H>
    friend H AbslHashValue(H h, const CacheKey& key) {
      h = H::combine(std::move(h), key.handle);
      h = H::combine(std::move(h), key.context.shape_index.ToString());
      h = H::combine(std::move(h),
                     VectorString(key.context.caller_operand_handles));
      h = H::combine(std::move(h), key.type);
      return h;
    }

    friend bool operator==(const CacheKey& lhs, const CacheKey& rhs) {
      return lhs.handle == rhs.handle &&
             lhs.context.shape_index == rhs.context.shape_index &&
             lhs.context.caller_operand_handles ==
                 rhs.context.caller_operand_handles &&
             lhs.type == rhs.type;
    }
  };

  HloEvaluator& evaluator;
  absl::flat_hash_map<CacheKey, Literal> evaluated;
  HandleToInstruction handle_to_instruction;
  HandleToComputation handle_to_computation;
  // Give up when dealing with more than 1M elements.
  static constexpr int64_t kLargeShapeElementLimit = 1000 * 1000;
};

// Returns a result representing that value is fully dynamic and can't be
// inferred. In other words, "give up" and return most conservative value.
PostorderDFSNode CreateAllDynamicResult(const Shape& shape,
                                        const PostorderDFSNodeType& type) {
  return PostorderDFSNode().AddVisit(
      [shape, type](absl::Span<Literal>) -> Literal {
        if (type == PostorderDFSNodeType::kConstantValue ||
            type == PostorderDFSNodeType::kConstantUpperBound ||
            type == PostorderDFSNodeType::kConstantLowerBound) {
          // When inferencing constant values, create garbage data, which will
          // be masked out by dynamism counterpart.
          return CreateGarbageLiteral(shape);
        } else {
          // When dynamism, return true, indicating all values are dynamic.
          return CreatePredLiteral(true, shape);
        }
      });
}

}  // namespace

// Analyze a tensor's constant value, upper-bound value or lower-bound value.
absl::StatusOr<PostorderDFSNode>
PostorderDFSVisitor::AnalyzeConstantValueFallback(int64_t handle,
                                                  PostorderDFSNodeType type,
                                                  InferenceContext context) {
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      handle_to_instruction(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  TF_ASSIGN_OR_RETURN(Shape root_shape, Shape::FromProto(root->shape()));
  Shape subshape = ShapeUtil::GetSubshape(root_shape, context.shape_index);
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
    case HloOpcode::kReduceScatter:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kCustomCall:
    case HloOpcode::kWhile:
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecvDone:
    case HloOpcode::kParameter: {
      if (opcode == HloOpcode::kParameter &&
          !context.caller_operand_handles.empty()) {
        int64_t caller_operand = context.caller_operand_handles.back();
        context.caller_operand_handles.pop_back();
        return result.AddDependency(caller_operand, type, context)
            .AddVisit([](Literal literal) { return literal; });
      }
      return CreateAllDynamicResult(subshape, type);
    }
    // Subtract and Divide use lower-bound as second operand.
    case HloOpcode::kSubtract:
    case HloOpcode::kCos:
    case HloOpcode::kSin:
    case HloOpcode::kTan:
    case HloOpcode::kNegate:
    case HloOpcode::kAbs:
    case HloOpcode::kDivide:
    case HloOpcode::kGetDimensionSize: {
      return InvalidArgument(
          "AnalyzeConstantValueFallback can't handle opcode: %s",
          root->opcode());
    }
    case HloOpcode::kCall: {
      auto node = PostorderDFSNode();
      auto* call_proto = root;
      if (call_proto->operand_ids_size() != 1) {
        // Only support single operand forwarding.
        return CreateAllDynamicResult(subshape, type);
      }
      int64_t called_root =
          handle_to_computation(call_proto->called_computation_ids(0))
              ->root_id();
      InferenceContext call_context = context;
      call_context.caller_operand_handles.push_back(call_proto->operand_ids(0));
      node.AddDependency(called_root, PostorderDFSNodeType::kConstantValue,
                         call_context, "callee's root instruction");
      return node.AddVisit([](Literal operand) -> absl::StatusOr<Literal> {
        // Forward result of callee's root to caller.
        return operand;
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
      const int64_t branch_size =
          conditional_proto->called_computation_ids_size();
      for (int64_t i = 0; i < branch_size; ++i) {
        int64_t branch_root =
            handle_to_computation(conditional_proto->called_computation_ids(i))
                ->root_id();
        InferenceContext branch_context = context;
        branch_context.caller_operand_handles.push_back(
            conditional_proto->operand_ids(i + 1));
        node.AddDependency(branch_root, PostorderDFSNodeType::kConstantValue,
                           branch_context);
      }
      return node.AddVisit(
          [](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
            int64_t pred_is_dynamic = operands[1].Get<bool>({});
            if (pred_is_dynamic) {
              // If predicate is dynamic, return the value of the first branch
              // -- If all branches return the same value, this is the value
              // that we want; If not, the value will be masked anyway so the
              // value inside doesn't matter.
              return std::move(operands[2]);
            } else {
              // If predicate is static, return the value of the given branch.
              int64_t branch_index = 0;
              if (operands[0].shape().element_type() == PRED) {
                if (operands[0].Get<bool>({})) {
                  branch_index = 0;
                } else {
                  branch_index = 1;
                }
              } else {
                branch_index = operands[0].GetIntegralAsS64({}).value();
              }
              const int64_t branch_dynamism_index = 2 + branch_index;
              return std::move(operands[branch_dynamism_index]);
            }
          });
    }
    case HloOpcode::kGetTupleElement: {
      int64_t operand_handle = root->operand_ids(0);
      PostorderDFSNode result;
      context.shape_index.push_front(root->tuple_index());
      return PostorderDFSNode()
          .AddDependency(operand_handle, type, context)
          .AddVisit([](Literal operand) { return operand; });
    }
    case HloOpcode::kReduce:
    case HloOpcode::kSort:
    case HloOpcode::kScatter:
    case HloOpcode::kReduceWindow: {
      const HloComputationProto* computation_proto =
          handle_to_computation(root->called_computation_ids(0));
      return result.AddVisit(
          [root, computation_proto, context,
           this](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
            TF_ASSIGN_OR_RETURN(
                auto computation,
                HloComputation::CreateFromProto(*computation_proto, {}));
            return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                ->WithOperands(operands)
                .WithComputation(std::move(computation))
                .WithSubshape(context.shape_index)
                .Evaluate();
          });
    }
    default: {
      if (opcode == HloOpcode::kTuple && !context.shape_index.empty()) {
        // There could be many operands of a tuple, but only one that we are
        // interested in, represented by `tuple_operand_index`.
        int64_t tuple_operand_index = context.shape_index.front();
        InferenceContext tuple_operand_context = context;
        tuple_operand_context.shape_index.pop_front();
        return PostorderDFSNode()
            .AddDependency(root->operand_ids(tuple_operand_index), type,
                           tuple_operand_context)
            .AddVisit([](Literal operand) { return operand; });
      }
      return result.AddVisit([root, this](absl::Span<Literal> operands) {
        return std::make_unique<HloProtoEvaluator>(evaluator, *root)
            ->WithOperands(operands)
            .Evaluate();
      });
    }
  }
}

absl::StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeUpperBound(
    int64_t handle, InferenceContext context) {
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      handle_to_instruction(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  TF_ASSIGN_OR_RETURN(Shape root_shape, Shape::FromProto(root->shape()));
  Shape subshape = ShapeUtil::GetSubshape(root_shape, context.shape_index);

  if (IsInstructionOverLimit(root, context)) {
    return CreateAllDynamicResult(subshape,
                                  PostorderDFSNodeType::kConstantUpperBound);
  }
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64_t dimension = root->dimensions(0);
      int64_t operand_handle = root->operand_ids(0);
      const HloInstructionProto* operand_proto =
          handle_to_instruction(operand_handle).value();
      return PostorderDFSNode().AddVisit(
          [operand_proto, dimension]() -> absl::StatusOr<Literal> {
            return LiteralUtil::CreateR0<int32_t>(
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
          .AddVisit([this](Literal lower_bound,
                           Literal upper_bound) -> absl::StatusOr<Literal> {
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
    case HloOpcode::kSort: {
      auto dfs = PostorderDFSNode();
      InferenceContext dep_context = context;
      dep_context.shape_index = {};
      if (!context.shape_index.empty()) {
        // Lazy evaluation: Only need to evaluate a subelement in a
        // variadic-sort tensor.
        dfs.AddDependency(root->operand_ids(context.shape_index[0]),
                          PostorderDFSNodeType::kConstantUpperBound,
                          dep_context);
      } else {
        for (int64_t i = 0; i < root->operand_ids_size(); ++i) {
          dfs.AddDependency(root->operand_ids(i),
                            PostorderDFSNodeType::kConstantUpperBound,
                            dep_context);
        }
      }

      return dfs.AddVisit(
          [root,
           context](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
            std::vector<Literal> results;
            results.reserve(operands.size());
            // Conservatively set each element of the tensor to the max value.
            for (int64_t i = 0; i < operands.size(); ++i) {
              auto max = LiteralUtil::MaxElement(operands[i]);
              results.emplace_back(
                  max.Broadcast(operands[i].shape(), {}).value());
            }
            TF_ASSIGN_OR_RETURN(Shape root_shape,
                                Shape::FromProto(root->shape()));
            if (ShapeUtil::GetSubshape(root_shape, context.shape_index)
                    .IsTuple()) {
              return LiteralUtil::MakeTupleOwned(std::move(results));
            } else {
              return std::move(results[0]);
            }
          });
    }
    case HloOpcode::kNegate: {
      // upper-bound(negate(operand)) = negate(lower-bound(operand))
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantLowerBound, context)
          .AddVisit([this](Literal lower_bound) -> absl::StatusOr<Literal> {
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
                        Literal lower_bound) -> absl::StatusOr<Literal> {
            if (opcode == HloOpcode::kDivide &&
                this->IsValueEffectiveInteger(root->operand_ids(1))) {
              // Because in many cases the lower bound of a value is
              // integer 0, instead of throwing an divide-by-zero error
              // at compile time, we set the bound defer the check to
              // runtime. In those cases we use the upper-bound of
              // first operand as a placeholder.
              auto zero = LiteralUtil::Zero(lower_bound.shape().element_type());
              zero = zero.Broadcast(lower_bound.shape(), {}).value();
              TF_ASSIGN_OR_RETURN(
                  auto lower_bound_is_zero,
                  evaluator.EvaluateElementwiseCompareOp(
                      ComparisonDirection::kEq, lower_bound, zero));

              auto one = LiteralUtil::One(lower_bound.shape().element_type());
              one = one.Broadcast(lower_bound.shape(), {}).value();
              TF_ASSIGN_OR_RETURN(
                  lower_bound, evaluator.EvaluateElementwiseTernaryOp(
                                   HloOpcode::kSelect, lower_bound_is_zero, one,
                                   lower_bound));
            }
            std::vector<Literal> new_operands;
            new_operands.emplace_back(std::move(upper_bound));
            new_operands.emplace_back(std::move(lower_bound));
            return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                ->WithOperands(absl::MakeSpan(new_operands))
                .Evaluate();
          });
    }
    case HloOpcode::kCustomCall: {
      if (root->custom_call_target() == "SetBound") {
        return PostorderDFSNode().AddVisit([root]() -> absl::StatusOr<Literal> {
          if (root->literal().shape().element_type() == TUPLE) {
            // First literal of SetBound contains bounds, second literal
            // contains dynamism indicators.
            return Literal::CreateFromProto(root->literal().tuple_literals(0));
          } else {
            return Literal::CreateFromProto(root->literal());
          }
        });
      } else if (root->custom_call_target() == "Sharding") {
        return PostorderDFSNode()
            .AddDependency(root->operand_ids(0),
                           PostorderDFSNodeType::kConstantUpperBound, context)
            .AddVisit([](Literal operand) { return operand; });
      }
      return InvalidArgument(
          "Upper-bound inferencing on custom call %s is not supported",
          root->DebugString());
    }
    case HloOpcode::kGather: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantUpperBound, context)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kConstantValue, context)
          .AddVisit([root, this](absl::Span<Literal> operands) {
            return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                ->WithOperands(operands)
                .Evaluate();
          });
    }
    default:
      return AnalyzeConstantValueFallback(
          handle, PostorderDFSNodeType::kConstantUpperBound, context);
  }
}

absl::StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeLowerBound(
    int64_t handle, InferenceContext context) {
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      handle_to_instruction(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  TF_ASSIGN_OR_RETURN(Shape root_shape, Shape::FromProto(root->shape()));
  Shape subshape = ShapeUtil::GetSubshape(root_shape, context.shape_index);
  if (IsInstructionOverLimit(root, context)) {
    return CreateAllDynamicResult(subshape,
                                  PostorderDFSNodeType::kConstantLowerBound);
  }
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64_t dimension = root->dimensions(0);
      int64_t operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                          handle_to_instruction(operand_handle));
      return PostorderDFSNode().AddVisit(
          [dimension, operand_proto]() -> absl::StatusOr<Literal> {
            if (operand_proto->shape().is_dynamic_dimension(dimension)) {
              return LiteralUtil::CreateR0<int32_t>(0);
            } else {
              return LiteralUtil::CreateR0<int32_t>(
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
          .AddVisit([this](Literal lower_bound,
                           Literal upper_bound) -> absl::StatusOr<Literal> {
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
          .AddVisit([this](Literal upper_bound) -> absl::StatusOr<Literal> {
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
          .AddVisit(
              [root,
               this](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
                return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                    ->WithOperands(operands)
                    .Evaluate();
              });
    }
    case HloOpcode::kGather: {
      return PostorderDFSNode()
          .AddDependency(root->operand_ids(0),
                         PostorderDFSNodeType::kConstantLowerBound, context)
          .AddDependency(root->operand_ids(1),
                         PostorderDFSNodeType::kConstantValue, context)
          .AddVisit([root, this](absl::Span<Literal> operands) {
            return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                ->WithOperands(operands)
                .Evaluate();
          });
    }
    default:
      return AnalyzeConstantValueFallback(
          handle, PostorderDFSNodeType::kConstantLowerBound, context);
  }
}

absl::StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeConstant(
    int64_t handle, InferenceContext context) {
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      handle_to_instruction(handle));
  HloOpcode opcode = StringToHloOpcode(root->opcode()).value();
  TF_ASSIGN_OR_RETURN(Shape root_shape, Shape::FromProto(root->shape()));
  Shape subshape = ShapeUtil::GetSubshape(root_shape, context.shape_index);
  if (IsInstructionOverLimit(root, context)) {
    return CreateAllDynamicResult(subshape,
                                  PostorderDFSNodeType::kConstantValue);
  }
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64_t dimension = root->dimensions(0);
      int64_t operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                          handle_to_instruction(operand_handle));
      return PostorderDFSNode().AddVisit(
          [operand_proto, dimension, root]() -> absl::StatusOr<Literal> {
            if (operand_proto->shape().is_dynamic_dimension(dimension)) {
              // The value is dynamic, we return garbage data here and mask them
              // out later.
              TF_ASSIGN_OR_RETURN(Shape root_shape,
                                  Shape::FromProto(root->shape()));
              return CreateGarbageLiteral(root_shape);
            } else {
              return LiteralUtil::CreateR0<int32_t>(
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
          [root,
           this](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
            return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                ->WithOperands(operands)
                .Evaluate();
          });
    }
    case HloOpcode::kCustomCall: {
      if (root->custom_call_target() == "SetBound") {
        return PostorderDFSNode().AddVisit([root]() -> absl::StatusOr<Literal> {
          if (root->literal().shape().element_type() == TUPLE) {
            // First literal of SetBound contains bounds, second literal
            // contains dynamism indicators.
            return Literal::CreateFromProto(root->literal().tuple_literals(0));
          } else {
            return Literal::CreateFromProto(root->literal());
          }
        });
      } else if (root->custom_call_target() == "Sharding") {
        return PostorderDFSNode()
            .AddDependency(root->operand_ids(0),
                           PostorderDFSNodeType::kConstantValue, context)
            .AddVisit([](Literal operand) { return operand; });
      } else {
        return PostorderDFSNode().AddVisit(
            [root, context](absl::Span<Literal>) -> absl::StatusOr<Literal> {
              // The value is dynamic. We return a garbage literal here, which
              // will be masked out later.
              TF_ASSIGN_OR_RETURN(Shape root_shape,
                                  Shape::FromProto(root->shape()));
              return CreateGarbageLiteral(
                  ShapeUtil::GetSubshape(root_shape, context.shape_index));
            });
      }
    }
    case HloOpcode::kSort: {
      PostorderDFSNode result;
      InferenceContext dep_context = context;
      dep_context.shape_index = {};
      for (auto operand_id : root->operand_ids()) {
        result.AddDependency(operand_id, PostorderDFSNodeType::kConstantValue,
                             dep_context);
      }
      const HloComputationProto* computation_proto =
          handle_to_computation(root->called_computation_ids(0));
      return result.AddVisit(
          [root, context, computation_proto,
           this](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
            TF_ASSIGN_OR_RETURN(
                auto computation,
                HloComputation::CreateFromProto(*computation_proto, {}));
            return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                ->WithOperands(operands)
                .WithComputation(std::move(computation))
                .WithSubshape(context.shape_index)
                .Evaluate();
          });
    }
    default:
      return AnalyzeConstantValueFallback(
          handle, PostorderDFSNodeType::kConstantValue, context);
  }
}

absl::StatusOr<PostorderDFSNode> PostorderDFSVisitor::AnalyzeIsDynamic(
    int64_t handle, PostorderDFSNodeType type, InferenceContext context) {
  TF_RETURN_IF_ERROR(handle_to_instruction(handle).status());
  // Invariant check.
  TF_RET_CHECK(handle_to_instruction(handle).value());
  VLOG(1) << "Analyzing IsDynamic on "
          << handle_to_instruction(handle).value()->DebugString();
  if (IsInstructionOverLimit(handle_to_instruction(handle).value(), context)) {
    TF_ASSIGN_OR_RETURN(
        Shape shape,
        Shape::FromProto(handle_to_instruction(handle).value()->shape()));
    return CreateAllDynamicResult(
        ShapeUtil::GetSubshape(shape, context.shape_index), type);
  }
  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      handle_to_instruction(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  PostorderDFSNode result;
  for (auto operand_id : root->operand_ids()) {
    InferenceContext dep_context = context;
    dep_context.shape_index = {};
    result.AddDependency(operand_id, type, dep_context);
  }
  switch (opcode) {
    case HloOpcode::kGetDimensionSize: {
      int64_t dimension = root->dimensions(0);
      int64_t operand_handle = root->operand_ids(0);
      TF_ASSIGN_OR_RETURN(const HloInstructionProto* operand_proto,
                          handle_to_instruction(operand_handle));
      return PostorderDFSNode().AddVisit(
          [operand_proto, dimension, type]() -> absl::StatusOr<Literal> {
            if (type == PostorderDFSNodeType::kBoundIsDynamic) {
              // The bound of dynamic dimension is not dynamic.
              return LiteralUtil::CreateR0<bool>(false);
            }
            // The value of dynamic dimension is dynamic.
            return LiteralUtil::CreateR0<bool>(
                operand_proto->shape().is_dynamic_dimension(dimension));
          });
    }
    case HloOpcode::kSort: {
      auto dfs = PostorderDFSNode();
      InferenceContext dep_context = context;
      dep_context.shape_index = {};

      for (int64_t i = 0; i < root->operand_ids_size(); ++i) {
        dfs.AddDependency(root->operand_ids(i), type, dep_context);
      }

      return dfs.AddVisit([root, context, type](absl::Span<Literal> operands)
                              -> absl::StatusOr<Literal> {
        bool all_operands_values_static = true;
        for (int64_t i = 0; i < operands.size(); ++i) {
          all_operands_values_static &= operands[i].IsAll(0);
        }
        if (type == PostorderDFSNodeType::kValueIsDynamic) {
          // If there is a single operand of a sort is dynamic, we
          // conservatively say all results are dynamic.
          TF_ASSIGN_OR_RETURN(Shape root_shape,
                              Shape::FromProto(root->shape()));
          return CreatePredLiteral(
              !all_operands_values_static,
              ShapeUtil::GetSubshape(root_shape, context.shape_index));
        }
        CHECK(type == PostorderDFSNodeType::kBoundIsDynamic);
        // The condition for bounds are more relaxed than values. If we know the
        // bounds of each element [B0, B1... Bn], all results have the same
        // bound
        // [max(B0, B1...), max(B0, B1...), ...]
        if (!context.shape_index.empty()) {
          int64_t index = context.shape_index[0];
          bool all_values_static = operands[index].IsAll(0);
          return CreatePredLiteral(!all_values_static, operands[index].shape());
        }

        std::vector<Literal> results;
        results.reserve(operands.size());
        for (int64_t i = 0; i < operands.size(); ++i) {
          bool all_values_static = operands[i].IsAll(0);
          results.emplace_back(
              CreatePredLiteral(!all_values_static, operands[i].shape()));
        }
        TF_ASSIGN_OR_RETURN(Shape root_shape, Shape::FromProto(root->shape()));
        if (!ShapeUtil::GetSubshape(root_shape, context.shape_index)
                 .IsTuple()) {
          return std::move(results[0]);
        }
        return LiteralUtil::MakeTupleOwned(std::move(results));
      });
    }
    case HloOpcode::kSetDimensionSize:
      return result.AddVisit([root, type](absl::Span<Literal> operands)
                                 -> absl::StatusOr<Literal> {
        bool any_dynamic_operand = absl::c_any_of(
            operands, [](Literal& operand) { return !operand.IsAll(0); });
        // If values in a tensor `t` with bound are [e0, e1, e2...], we can say
        // the max value of each position is [max(t), max(t), max(t), ...]. The
        // effective size of this tensor doesn't change the max value.
        TF_ASSIGN_OR_RETURN(Shape root_shape, Shape::FromProto(root->shape()));
        return CreatePredLiteral(
            type == PostorderDFSNodeType::kValueIsDynamic &&
                any_dynamic_operand,
            ShapeUtil::MakeStaticShape(root_shape));
      });
    case HloOpcode::kDynamicSlice: {
      return result.AddVisit(
          [root](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
            // If any of the operand is dynamic, we say output is dynamic.
            bool any_dynamic_operand = absl::c_any_of(
                operands, [](Literal& operand) { return !operand.IsAll(0); });
            TF_ASSIGN_OR_RETURN(Shape root_shape,
                                Shape::FromProto(root->shape()));
            return CreatePredLiteral(any_dynamic_operand, root_shape);
          });
    }
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCos:
    case HloOpcode::kClz:
    case HloOpcode::kErf:
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
    case HloOpcode::kTan:
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
      return result.AddVisit([root, this](absl::Span<Literal> operands) {
        return std::make_unique<HloProtoEvaluator>(evaluator, *root)
            ->WithOperands(operands)
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
        int64_t tuple_operand_index = context.shape_index.front();
        InferenceContext tuple_operand_context = context;
        tuple_operand_context.shape_index.pop_front();
        return PostorderDFSNode()
            .AddDependency(root->operand_ids(tuple_operand_index), type,
                           tuple_operand_context)
            .AddVisit([](Literal operand) { return operand; });
      }
      return result.AddVisit([root, this](absl::Span<Literal> operands) {
        return std::make_unique<HloProtoEvaluator>(evaluator, *root)
            ->WithOperands(operands)
            .WithPrimitiveType(PRED)
            .Evaluate();
      });
    }
    case HloOpcode::kCall: {
      auto node = PostorderDFSNode();
      auto* call_proto = root;

      if (call_proto->operand_ids_size() != 1) {
        // Only support single operand forwarding.
        TF_ASSIGN_OR_RETURN(auto shape, Shape::FromProto(call_proto->shape()));
        return CreateAllDynamicResult(shape, type);
      }
      int64_t call_root =
          handle_to_computation(call_proto->called_computation_ids(0))
              ->root_id();
      InferenceContext branch_context = context;
      branch_context.caller_operand_handles.push_back(
          call_proto->operand_ids(0));
      node.AddDependency(call_root, PostorderDFSNodeType::kValueIsDynamic,
                         branch_context, "callee's root instruction");
      return node.AddVisit(
          [context](Literal operand) -> absl::StatusOr<Literal> {
            // Forward result of callee's root to caller.
            return operand;
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
      const int64_t branch_size =
          conditional_proto->called_computation_ids_size();
      for (int64_t i = 0; i < branch_size; ++i) {
        int64_t branch_root =
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
      return node.AddVisit([root, branch_size,
                            context](absl::Span<Literal> operands)
                               -> absl::StatusOr<Literal> {
        int64_t pred_is_dynamic = operands[1].Get<bool>({});
        TF_ASSIGN_OR_RETURN(Shape root_shape, Shape::FromProto(root->shape()));
        auto result = CreatePredLiteral(
            true, ShapeUtil::GetSubshape(root_shape, context.shape_index));
        if (pred_is_dynamic) {
          VLOG(1) << "predict is dynamic value" << result.ToString();
          // If predicate is dynamic, the result is only static if all
          // branches are static and return the same value.
          result.MutableEachCell<bool>(
              [&](absl::Span<const int64_t> indices, bool value) {
                std::string branch_value = operands[2].GetAsString(indices, {});
                for (int64_t i = 0; i < branch_size; ++i) {
                  const int64_t branch_value_index = 2 + 2 * i;
                  const int64_t branch_dynamism_index = 2 + 2 * i + 1;
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
          int64_t branch_index = 0;
          if (operands[0].shape().element_type() == PRED) {
            if (operands[0].Get<bool>({})) {
              branch_index = 0;
            } else {
              branch_index = 1;
            }
          } else {
            branch_index = operands[0].GetIntegralAsS64({}).value();
          }
          const int64_t branch_dynamism_index = 2 + 2 * branch_index + 1;
          return std::move(operands[branch_dynamism_index]);
        }
      });
    }
    case HloOpcode::kGetTupleElement: {
      int64_t operand_handle = root->operand_ids(0);
      PostorderDFSNode result;
      context.shape_index.push_front(root->tuple_index());
      return PostorderDFSNode()
          .AddDependency(operand_handle, type, context)
          .AddVisit([](Literal operand) { return operand; });
    }

    case HloOpcode::kReduce: {
      return result.AddVisit(
          [root, context,
           this](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
            TF_ASSIGN_OR_RETURN(Shape root_shape,
                                Shape::FromProto(root->shape()));
            Shape scalar_shape = ShapeUtil::MakeScalarShape(xla::PRED);
            std::unique_ptr<HloComputation> reduce_or;
            if (root_shape.IsTuple()) {
              // Variadic reduce.
              HloComputation::Builder b("reduce_or");
              // Assuming all operands interact with each other. This could be
              // overly conservative.  If needed, a dataflow analysis could be
              // performed in the future.
              //
              // The value starts with `false` (static) and will be `or`ed with
              // all operands's dynamism.
              auto accum = b.AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<bool>(false)));

              for (int i = 0; i < root_shape.tuple_shapes().size(); ++i) {
                auto lhs = b.AddInstruction(
                    HloInstruction::CreateParameter(i, scalar_shape, "lhs"));
                auto rhs = b.AddInstruction(HloInstruction::CreateParameter(
                    i + root_shape.tuple_shapes().size(), scalar_shape, "rhs"));
                accum = b.AddInstruction(HloInstruction::CreateBinary(
                    scalar_shape, HloOpcode::kOr, accum, lhs));
                accum = b.AddInstruction(HloInstruction::CreateBinary(
                    scalar_shape, HloOpcode::kOr, accum, rhs));
              }
              // `Broadcast` the result to all positions in the result.
              std::vector<HloInstruction*> results(
                  root_shape.tuple_shapes().size(), accum);
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

            return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                ->WithOperands(operands)
                .WithPrimitiveType(PRED)
                .WithComputation(std::move(reduce_or))
                // Reduce could produce tuple shape, only fetch what we need.
                .WithSubshape(context.shape_index)
                .Evaluate();
          });
    }
    case HloOpcode::kConstant:
    case HloOpcode::kIota: {
      return result.AddVisit([root]() -> absl::StatusOr<Literal> {
        TF_ASSIGN_OR_RETURN(Shape root_shape, Shape::FromProto(root->shape()));
        return CreatePredLiteral(false, root_shape);
      });
    }
    case HloOpcode::kParameter: {
      if (opcode == HloOpcode::kParameter &&
          !context.caller_operand_handles.empty()) {
        int64_t caller_operand = context.caller_operand_handles.back();
        context.caller_operand_handles.pop_back();
        return result.AddDependency(caller_operand, type, context)
            .AddVisit([](Literal literal) { return literal; });
      }
      return result.AddVisit([root, context]() -> absl::StatusOr<Literal> {
        TF_ASSIGN_OR_RETURN(Shape root_shape, Shape::FromProto(root->shape()));
        return CreatePredLiteral(
            true, ShapeUtil::GetSubshape(root_shape, context.shape_index));
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
          .AddVisit([root](absl::Span<Literal> operands)
                        -> absl::StatusOr<Literal> {
            OptionalLiteral optional_selector_literal(std::move(operands[0]),
                                                      std::move(operands[1]));
            Literal lhs = std::move(operands[2]);
            Literal rhs = std::move(operands[3]);
            TF_ASSIGN_OR_RETURN(Shape root_shape,
                                Shape::FromProto(root->shape()));
            auto result = CreatePredLiteral(true, root_shape);
            result.MutableEachCell<bool>(
                [&](absl::Span<const int64_t> indices, bool value) {
                  std::optional<bool> optional_selector =
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
          .AddVisit(
              [root,
               this](absl::Span<Literal> operands) -> absl::StatusOr<Literal> {
                OptionalLiteral optional_selector_literal(
                    std::move(operands[1]), std::move(operands[2]));

                if (!optional_selector_literal.AllValid()) {
                  // Conservatively assume results are dynamic.
                  TF_ASSIGN_OR_RETURN(Shape root_shape,
                                      Shape::FromProto(root->shape()));
                  return CreatePredLiteral(true, root_shape);
                }
                std::vector<Literal> new_operands;
                new_operands.emplace_back(std::move(operands[0]));
                new_operands.emplace_back(
                    optional_selector_literal.GetValue()->Clone());

                return std::make_unique<HloProtoEvaluator>(evaluator, *root)
                    ->WithOperands(absl::MakeSpan(new_operands))
                    .WithPrimitiveType(PRED)
                    .Evaluate();
              });
    }
    case HloOpcode::kCustomCall: {
      if (root->custom_call_target() == "SetBound") {
        return PostorderDFSNode().AddVisit([type,
                                            root]() -> absl::StatusOr<Literal> {
          if (type == PostorderDFSNodeType::kBoundIsDynamic) {
            TF_ASSIGN_OR_RETURN(Shape root_shape,
                                Shape::FromProto(root->shape()));
            return CreatePredLiteral(false, root_shape);
          } else {
            if (root->literal().shape().element_type() == TUPLE) {
              // First literal of SetBound contains bounds, second literal
              // contains dynamism indicators.
              return Literal::CreateFromProto(
                  root->literal().tuple_literals(1));
            } else if (type == PostorderDFSNodeType::kValueIsDynamic) {
              TF_ASSIGN_OR_RETURN(Shape root_shape,
                                  Shape::FromProto(root->shape()));
              return CreatePredLiteral(true, root_shape);
            } else {
              return Literal::CreateFromProto(root->literal());
            }
          }
        });
      } else if (root->custom_call_target() == "Sharding") {
        return result.AddVisit([](Literal operand) { return operand; });
      } else {
        return InvalidArgument(
            "Dynamic inferencing on custom call %s is not supported",
            root->DebugString());
      }

      break;
    }

    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kWhile: {
      return PostorderDFSNode().AddVisit(
          [root, context]() -> absl::StatusOr<Literal> {
            TF_ASSIGN_OR_RETURN(Shape root_shape,
                                Shape::FromProto(root->shape()));
            return CreatePredLiteral(
                true, ShapeUtil::GetSubshape(root_shape, context.shape_index));
          });
      break;
    }
    default:
      return PostorderDFSNode().AddVisit(
          [root, context]() -> absl::StatusOr<Literal> {
            TF_ASSIGN_OR_RETURN(Shape root_shape,
                                Shape::FromProto(root->shape()));
            return CreatePredLiteral(
                true, ShapeUtil::GetSubshape(root_shape, context.shape_index));
          });
  }
}

absl::StatusOr<Literal> PostorderDFSVisitor::PostOrderDFSVisit(
    int64_t handle, PostorderDFSNodeType type) {
  enum VisitState {
    kUnvisited = 0,
    kVisiting,
    kVisited,
  };

  int64_t unique_id = 0;
  struct WorkItem {
    explicit WorkItem(int64_t handle, InferenceContext context,
                      PostorderDFSNodeType type, VisitState state, int64_t id)
        : handle(handle),
          context(std::move(context)),
          type(type),
          state(state),
          id(id) {}
    int64_t handle;  // Handle of the node in the graph.
    InferenceContext context;
    PostorderDFSNodeType type;
    VisitState state;
    Visit visit;  // The handler to call once the dependencies are resolved into
                  // literal form.
    int64_t id;   // Unique id in the work queue, starting from 0.
    std::vector<CacheKey> dependencies;

    CacheKey GetCacheKey() { return CacheKey(handle, context, type); }
  };

  std::vector<WorkItem> stack;
  WorkItem root(handle, InferenceContext({}, {}), type, kUnvisited,
                unique_id++);
  stack.push_back(root);
  while (!stack.empty()) {
    WorkItem& item = stack.back();
    VLOG(1) << "stack top shape index: " << item.context.shape_index.ToString();
    if (VLOG_IS_ON(1)) {
      TF_RETURN_IF_ERROR(handle_to_instruction(item.handle).status());
      VLOG(1) << "stack top "
              << handle_to_instruction(item.handle).value()->DebugString();
    }
    if (item.state == kVisiting) {
      VLOG(1) << "visiting";
      // The dependencies are ready, visit the node itself.

      // Gather dependencies and transform them into literals.
      std::vector<Literal> literals;
      literals.reserve(item.dependencies.size());
      for (CacheKey& dep_key : item.dependencies) {
        TF_RET_CHECK(evaluated.contains(dep_key));
        literals.emplace_back(evaluated.at(dep_key).Clone());
      }
      VLOG(1) << "Start visiting with dependency type: "
              << PostorderDFSNodeTypeToString(item.type);
      TF_ASSIGN_OR_RETURN(auto literal, item.visit(absl::MakeSpan(literals)));
      VLOG(1) << "End visiting: " << literal.ToString();
      evaluated[item.GetCacheKey()] = std::move(literal);
      stack.pop_back();
      continue;
    }
    // This is the first time we see this node, we want to gather its
    // dependenceis.
    VLOG(1) << "unvisited";
    if (evaluated.contains(item.GetCacheKey())) {
      stack.pop_back();
      continue;
    }
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

    const int64_t current_item_id = stack.size() - 1;
    // Enqueue dependencies into the stack. `item` shouldn't be accessed after
    // this point.
    for (const PostorderDFSDep& dep : node.dependencies) {
      TF_ASSIGN_OR_RETURN(auto dependency_inst,
                          handle_to_instruction(dep.handle));
      VLOG(1) << "dependency " << dep.annotation
              << "::" << dependency_inst->DebugString() << "index"
              << dep.context.shape_index << " stack size:" << stack.size();
      stack.emplace_back(dep.handle, dep.context, dep.type, kUnvisited,
                         unique_id++);
      stack[current_item_id].dependencies.push_back(stack.back().GetCacheKey());
    }
  }
  VLOG(1) << "done" << evaluated[root.GetCacheKey()].ToString();
  return evaluated[root.GetCacheKey()].Clone();
}

absl::StatusOr<Literal> ValueInference::AnalyzeIsDynamic(XlaOp op) {
  PostorderDFSVisitor visitor(
      evaluator_,
      [&](int64_t handle) {
        return builder_->LookUpInstructionByHandle(handle);
      },
      [&](int64_t handle) { return &(builder_->embedded_[handle]); });

  auto result = visitor.PostOrderDFSVisit(
      op.handle(), PostorderDFSNodeType::kValueIsDynamic);
  return result;
}

absl::StatusOr<std::optional<int64_t>> ValueInference::CseOpHandle(
    int64_t handle) {
  TF_ASSIGN_OR_RETURN(auto inst, builder_->LookUpInstructionByHandle(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(inst->opcode()));
  // For now, only handle kGetDimensionSize as that's the most duplicated one.
  if (opcode != HloOpcode::kGetDimensionSize) {
    return {std::nullopt};
  }
  int64_t hash = absl::HashOf(inst->operand_ids(0), inst->dimensions(0));
  auto lookup = cse_map_.find(hash);
  if (lookup == cse_map_.end()) {
    cse_map_[hash] = handle;
    return {std::nullopt};
  }
  TF_ASSIGN_OR_RETURN(auto equivalent_op,
                      builder_->LookUpInstructionByHandle(lookup->second));
  // Check that the op is indeed equivalent to prevent hash collision --
  // relatively easy to happen with 64 bits hash.
  if (equivalent_op->opcode() != inst->opcode() ||
      equivalent_op->operand_ids(0) != inst->operand_ids(0) ||
      equivalent_op->dimensions(0) != inst->dimensions(0)) {
    // Hash collision, don't CSE.
    return {std::nullopt};
  }
  int64_t cse = lookup->second;
  if (handle != cse) {
    // Successfully found a handle that's not the same as input but equivalent.
    return {cse};
  }
  return {std::nullopt};
}

absl::StatusOr<Literal> ValueInference::SimplifyOp(int64_t handle) {
  TF_ASSIGN_OR_RETURN(auto cse_handle, CseOpHandle(handle));
  if (cse_handle) {
    // Use the CSE'd handle instead.
    return SimplifyOp(*cse_handle);
  }
  TF_ASSIGN_OR_RETURN(auto* inst, builder_->LookUpInstructionByHandle(handle));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(inst->opcode()));
  std::vector<Literal> operands;
  auto output_shape = std::make_unique<const Shape>(inst->shape());
  switch (opcode) {
    case HloOpcode::kSlice:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReshape:
    case HloOpcode::kBroadcast: {
      for (auto operand_id : inst->operand_ids()) {
        TF_ASSIGN_OR_RETURN(auto literal, SimplifyOp(operand_id));
        operands.emplace_back(std::move(literal));
      }
      // We put handles into the tensor and evaluate the results into a literal.
      // The literal also contain handles for each element position.
      return std::make_unique<HloProtoEvaluator>(evaluator_, *inst)
          ->WithOperands(absl::MakeSpan(operands))
          .WithPrimitiveType(S64)
          .Evaluate();
    }
    case HloOpcode::kConvert: {
      // Only identity kConvert can be optimized away.
      auto operand =
          builder_->LookUpInstructionByHandle(inst->operand_ids(0)).value();
      TF_ASSIGN_OR_RETURN(Shape operand_shape,
                          Shape::FromProto(operand->shape()));
      if (Shape::Equal()(*output_shape, operand_shape)) {
        // Forward operand handle as result.
        return SimplifyOp(inst->operand_ids(0));
      } else {
        return CreateS64Literal(-1, *output_shape);
      }
    }
    case HloOpcode::kAdd: {
      // a + (b - a) => b
      // a + b + (c - a) => b + c
      if (output_shape->dimensions().size() == 0) {
        TF_ASSIGN_OR_RETURN(auto lhs, SimplifyOp(inst->operand_ids(0)));
        TF_ASSIGN_OR_RETURN(auto rhs, SimplifyOp(inst->operand_ids(1)));
        int64_t lhs_handle = lhs.Get<int64_t>({});
        int64_t rhs_handle = rhs.Get<int64_t>({});
        if (lhs_handle == -1 || rhs_handle == -1) {
          return CreateS64Literal(-1, *output_shape);
        }
        // Recursive lambda needs explicit signature.
        std::function<std::optional<int64_t>(int64_t, int64_t)>
            can_be_optimized;
        can_be_optimized = [this, &can_be_optimized](
                               int64_t lhs,
                               int64_t rhs) -> std::optional<int64_t> {
          auto rhs_inst = builder_->LookUpInstructionByHandle(rhs).value();
          HloOpcode rhs_opcode = StringToHloOpcode(rhs_inst->opcode()).value();
          if (rhs_opcode == HloOpcode::kSubtract) {
            auto sub_lhs_handle =
                SimplifyOp(rhs_inst->operand_ids(0)).value().Get<int64_t>({});
            auto sub_rhs_handle =
                SimplifyOp(rhs_inst->operand_ids(1)).value().Get<int64_t>({});
            if (sub_rhs_handle == lhs) {
              // lhs + (sub_lhs - sub_rhs) = sub_lhs if lhs == sub_rhs
              return sub_lhs_handle;
            }
          }

          // Check the case for a + b + (c - a) => b + c
          auto lhs_inst = builder_->LookUpInstructionByHandle(lhs).value();
          HloOpcode lhs_opcode = StringToHloOpcode(lhs_inst->opcode()).value();
          if (lhs_opcode == HloOpcode::kAdd) {
            auto add_lhs_handle =
                SimplifyOp(lhs_inst->operand_ids(0)).value().Get<int64_t>({});
            auto add_rhs_handle =
                SimplifyOp(lhs_inst->operand_ids(1)).value().Get<int64_t>({});
            if (auto optimized = can_be_optimized(add_lhs_handle, rhs)) {
              return Add(XlaOp(add_rhs_handle, builder_),
                         XlaOp(optimized.value(), builder_))
                  .handle();
            }
            if (auto optimized = can_be_optimized(add_rhs_handle, rhs)) {
              return Add(XlaOp(add_lhs_handle, builder_),
                         XlaOp(optimized.value(), builder_))
                  .handle();
            }
          }
          return std::nullopt;
        };
        if (auto optimized = can_be_optimized(lhs_handle, rhs_handle)) {
          return LiteralUtil::CreateR0<int64_t>(optimized.value());
        }
        // Swap lhs and rhs.
        if (auto optimized = can_be_optimized(rhs_handle, lhs_handle)) {
          return LiteralUtil::CreateR0<int64_t>(optimized.value());
        }
        // This sum can't be optimized, return sum of lhs and rhs. Note that we
        // can't just return the original sum as its lhs and rhs could be
        // optimized and different.
        XlaOp new_sum =
            Add(XlaOp(lhs_handle, builder_), XlaOp(rhs_handle, builder_));

        return LiteralUtil::CreateR0<int64_t>(new_sum.handle());
      } else {
        return CreateS64Literal(-1, *output_shape);
      }
    }
    default: {
      if (ShapeUtil::IsScalar(*output_shape)) {
        return LiteralUtil::CreateR0<int64_t>(handle);
      } else {
        return CreateS64Literal(-1, *output_shape);
      }
    }
  }
}

absl::StatusOr<OptionalLiteral> ValueInference::AnalyzeConstant(
    XlaOp op, ValueInferenceMode mode) {
  TF_RETURN_IF_ERROR(builder_->LookUpInstructionByHandle(op.handle()).status());
  PostorderDFSVisitor visitor(
      evaluator_,
      [&](int64_t handle) {
        return builder_->LookUpInstructionByHandle(handle);
      },
      [&](int64_t handle) { return &(builder_->embedded_[handle]); });
  TF_ASSIGN_OR_RETURN(Shape op_shape, builder_->GetShape(op));
  int64_t handle = op.handle();
  if (ShapeUtil::IsScalar(builder_->GetShape(op).value())) {
    TF_ASSIGN_OR_RETURN(auto result, SimplifyOp(handle));
    auto optimized_handle = result.Get<int64_t>({});
    if (optimized_handle != -1) {
      handle = optimized_handle;
    }
  }
  switch (mode) {
    case ValueInferenceMode::kLowerBound: {
      TF_ASSIGN_OR_RETURN(Literal mask,
                          visitor.PostOrderDFSVisit(
                              handle, PostorderDFSNodeType::kBoundIsDynamic));
      if (mask.IsAll(1)) {
        // Everything is dynamic, no need to do constant inference.
        return OptionalLiteral(CreateGarbageLiteral(op_shape), std::move(mask));
      }
      TF_ASSIGN_OR_RETURN(
          Literal value,
          visitor.PostOrderDFSVisit(handle,
                                    PostorderDFSNodeType::kConstantLowerBound));

      return OptionalLiteral(std::move(value), std::move(mask));
    }
    case ValueInferenceMode::kUpperBound: {
      TF_ASSIGN_OR_RETURN(Literal mask,
                          visitor.PostOrderDFSVisit(
                              handle, PostorderDFSNodeType::kBoundIsDynamic));
      if (mask.IsAll(1)) {
        // Everything is dynamic, no need to do constant inference.
        return OptionalLiteral(CreateGarbageLiteral(op_shape), std::move(mask));
      }
      TF_ASSIGN_OR_RETURN(
          Literal value,
          visitor.PostOrderDFSVisit(handle,
                                    PostorderDFSNodeType::kConstantUpperBound));

      return OptionalLiteral(std::move(value), std::move(mask));
    }
    case ValueInferenceMode::kValue: {
      TF_ASSIGN_OR_RETURN(Literal mask,
                          visitor.PostOrderDFSVisit(
                              handle, PostorderDFSNodeType::kValueIsDynamic));
      if (mask.IsAll(1)) {
        // Everything is dynamic, no need to do constant inference.
        return OptionalLiteral(CreateGarbageLiteral(op_shape), std::move(mask));
      }
      TF_ASSIGN_OR_RETURN(Literal value,
                          visitor.PostOrderDFSVisit(
                              handle, PostorderDFSNodeType::kConstantValue));

      return OptionalLiteral(std::move(value), std::move(mask));
    }
  }
}

}  // namespace xla

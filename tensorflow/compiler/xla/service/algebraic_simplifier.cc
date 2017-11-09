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

#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using tensorflow::gtl::nullopt;
using tensorflow::gtl::optional;

// Returns whether operand is a literal with the given value.
bool IsLiteralWithValue(const HloInstruction* operand, int8 value) {
  return operand->opcode() == HloOpcode::kConstant &&
         operand->literal().IsAll(value);
}

bool IsAll(const HloInstruction* op, int8 value) {
  if (IsLiteralWithValue(op, value)) {
    return true;
  }
  if (op->opcode() == HloOpcode::kBroadcast && IsAll(op->operand(0), value)) {
    return true;
  }
  return false;
}

// Returns whether the given transpose produces a result which is bit-wise
// identical to its operand and thus may be replaced with a bitcast.
bool TransposeIsBitcast(const HloInstruction* transpose) {
  CHECK_EQ(HloOpcode::kTranspose, transpose->opcode());
  const HloInstruction* operand = transpose->operand(0);
  return ShapeUtil::TransposeIsBitcast(operand->shape(), transpose->shape(),
                                       transpose->dimensions());
}

// Returns true if the given reshape produces a result which is bit-wise
// identical to its operand and thus may be replaced with a bitcast.
//
// This function is conservative -- even if this function returns false, the
// reshape may still be a bitcast. For example, a reshape from [28x28] to [784].
bool ReshapeIsBitcast(
    const HloInstruction* reshape,
    const AlgebraicSimplifier::ValidBitcastCallback& valid_bitcast_callback) {
  CHECK_EQ(HloOpcode::kReshape, reshape->opcode());

  const HloInstruction* operand = reshape->operand(0);
  // Can't insert bitcasts if the compiler used a memory layout which isn't
  // compatible.
  return ShapeUtil::ReshapeIsBitcast(operand->shape(), reshape->shape()) &&
         valid_bitcast_callback(operand->shape(), reshape->shape());
}

// Adds a scalar computation to the module to enable optimizations with dot
// converting into reduction.
HloComputation* CreateScalarBinaryComputation(HloModule* module,
                                              PrimitiveType primitive_type,
                                              HloOpcode opcode) {
  HloComputation::Builder b("scalar_computation");
  auto scalar_lhs = b.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "scalar_lhs"));
  auto scalar_rhs = b.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {}), "scalar_rhs"));
  auto scalar_op = b.AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::MakeShape(primitive_type, {}),
                                   opcode, scalar_lhs, scalar_rhs));
  HloComputation* scalar_computation =
      module->AddEmbeddedComputation(b.Build(scalar_op));
  return scalar_computation;
}
}  // namespace

// AlgebraicSimplifierVisitor traverses the HLO computation and reduces certain
// algebraic expressions to simplified forms. Note: This only supports
// simplifications that simply look at the operands of an instruction. For the
// more general case a worklist based approach would be needed.
class AlgebraicSimplifierVisitor : public DfsHloVisitorWithDefault {
 public:
  // Default visitor action is to do nothing and return OK.
  Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
    return Status::OK();
  }

  Status HandleAdd(HloInstruction* add) override;

  Status HandleBitcast(HloInstruction* bitcast) override;

  Status HandleBroadcast(HloInstruction* broadcast) override;

  Status HandleConcatenate(HloInstruction* concatenate) override;

  Status HandleConstant(HloInstruction* constant) override;

  Status HandleCopy(HloInstruction* copy) override;

  Status HandleConvert(HloInstruction* convert) override;

  Status HandleReal(HloInstruction* real) override;
  Status HandleImag(HloInstruction* imag) override;

  Status HandleConvolution(HloInstruction* convolution) override;

  Status HandleDivide(HloInstruction* divide) override;

  Status HandleDot(HloInstruction* dot) override;

  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;

  Status HandleLog(HloInstruction* log) override;

  Status HandleMultiply(HloInstruction* multiply) override;

  Status HandlePad(HloInstruction* pad) override;

  Status HandlePower(HloInstruction* power) override;

  Status HandleReshape(HloInstruction* reshape) override;

  Status HandleReduce(HloInstruction* reduce) override;

  Status HandleReduceWindow(HloInstruction* reduce_window) override;

  Status HandleReverse(HloInstruction* reverse) override;
  Status HandleSlice(HloInstruction* slice) override;
  Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;
  Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;

  Status HandleTranspose(HloInstruction* transpose) override;

  Status HandleSubtract(HloInstruction* sub) override;

  Status HandleMaximum(HloInstruction* maximum) override;
  Status HandleMinimum(HloInstruction* minimum) override;

  // Returns whether algebraic simplification has occurred.
  const bool changed() const { return changed_; }

  // Runs the visitor on a computation.
  static bool Run(
      HloComputation* computation, bool is_layout_sensitive,
      AlgebraicSimplifier::ValidBitcastCallback valid_bitcast_callback,
      bool enable_dot_simplification, bool enable_conv_simplification);

 private:
  explicit AlgebraicSimplifierVisitor(
      HloComputation* computation, bool is_layout_sensitive,
      AlgebraicSimplifier::ValidBitcastCallback valid_bitcast_callback,
      bool enable_dot_simplification, bool enable_conv_simplification)
      : computation_(computation),
        is_layout_sensitive_(is_layout_sensitive),
        valid_bitcast_callback_(std::move(valid_bitcast_callback)),
        enable_dot_simplification_(enable_dot_simplification),
        enable_conv_simplification_(enable_conv_simplification) {}

  // Convenience method for replacing an instruction with a bitcast.
  void ReplaceWithBitcast(HloInstruction* instruction);

  // Replace old instruction with new instruction if old and new instructions
  // have the same shape. Updates uses and root instruction. Returns whether a
  // replacement was made.
  bool ReplaceInstructionIfSameShape(HloInstruction* old_instruction,
                                     HloInstruction* new_instruction);

  // Returns whether the shape of the output of the given instructions are the
  // same for the purposes of simplification. If is_layout_sensitive_ is true,
  // then this tests shape equality including layout (ShapeUtil::Equal). If
  // is_layout_sensitive_ is false, then the tests shape compatibility
  // (ShapeUtil::Compatible).
  bool SameShape(const HloInstruction* lhs, const HloInstruction* rhs) const;

  // Returns whether it was possible to transform `root` to a clamp instruction.
  // With min a minimum instruction, max a maximum instruction, min_operand a
  // operand of min and max_operand a operand of max.
  // Precondition: root is either a minimum or a maximum.
  bool TransformToClampIfSameShape(HloInstruction* root, HloInstruction* min,
                                   HloInstruction* min_operand,
                                   HloInstruction* operand, HloInstruction* max,
                                   HloInstruction* max_operand);

  // A Reshape or Broadcast that feeds an element-wise operation with a unique
  // non-scalar operand can sink to after the operation.
  StatusOr<bool> TryToSinkReshapeOrBroadcastAfterOpWithUniqueNonScalarOperand(
      HloInstruction* reshape_or_broadcast);

  // Replaces the existing HLO instruction old_instruction, with
  // new_instruction, and marks the optimizer status as changed.
  // Returns the Status representing the result of the replace operation.
  Status ReplaceWithNewInstruction(
      HloInstruction* old_instruction,
      std::unique_ptr<HloInstruction> new_instruction) {
    VLOG(3) << "Replacing instruction:";
    VLOG(3) << "  old: " << old_instruction->ToString();
    VLOG(3) << "  new: " << new_instruction->ToString();
    TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
        old_instruction, std::move(new_instruction)));
    changed_ = true;
    return Status::OK();
  }

  // Replaces the existing HLO instruction old_instruction, with
  // new_instruction, and marks the optimizer status as changed.
  // Returns the Status representing the result of the replace operation.
  Status ReplaceInstruction(HloInstruction* old_instruction,
                            HloInstruction* new_instruction) {
    VLOG(3) << "Replacing instruction:";
    VLOG(3) << "  old: " << old_instruction->ToString();
    VLOG(3) << "  new: " << new_instruction->ToString();
    TF_RETURN_IF_ERROR(
        computation_->ReplaceInstruction(old_instruction, new_instruction));
    changed_ = true;
    return Status::OK();
  }

  // Current HloComputation instance the AlgebraicSimplifierVisitor is
  // traversing.
  HloComputation* computation_;

  // Whether algebraic simplification has occurred.
  bool changed_ = false;

  // Whether layout is considered during transformation.
  bool is_layout_sensitive_;

  // Callback used to determine if a bitcast is possible.
  AlgebraicSimplifier::ValidBitcastCallback valid_bitcast_callback_;

  // Disable dot simplication on platforms where it causes a slowdown.
  bool enable_dot_simplification_;

  // Disable convolution simplication on platforms where it causes a slowdown.
  bool enable_conv_simplification_;
};

bool AlgebraicSimplifierVisitor::Run(
    HloComputation* computation, bool is_layout_sensitive,
    AlgebraicSimplifier::ValidBitcastCallback valid_bitcast_callback,
    bool enable_dot_simplification, bool enable_conv_simplification) {
  AlgebraicSimplifierVisitor visitor(
      computation, is_layout_sensitive, std::move(valid_bitcast_callback),
      enable_dot_simplification, enable_conv_simplification);
  TF_CHECK_OK(computation->Accept(&visitor));
  return visitor.changed_;
}

bool AlgebraicSimplifierVisitor::SameShape(const HloInstruction* lhs,
                                           const HloInstruction* rhs) const {
  if (is_layout_sensitive_) {
    return ShapeUtil::Equal(lhs->shape(), rhs->shape());
  } else {
    return ShapeUtil::Compatible(lhs->shape(), rhs->shape());
  }
}

void AlgebraicSimplifierVisitor::ReplaceWithBitcast(
    HloInstruction* instruction) {
  CHECK_EQ(1, instruction->operand_count());
  CHECK_EQ(ShapeUtil::ElementsIn(instruction->shape()),
           ShapeUtil::ElementsIn(instruction->operand(0)->shape()));
  CHECK_EQ(ShapeUtil::ByteSizeOf(instruction->shape()),
           ShapeUtil::ByteSizeOf(instruction->operand(0)->shape()));

  auto bitcast = computation_->AddInstruction(
      HloInstruction::CreateUnary(instruction->shape(), HloOpcode::kBitcast,
                                  instruction->mutable_operand(0)));
  TF_CHECK_OK(ReplaceInstruction(instruction, bitcast));
}

bool AlgebraicSimplifierVisitor::ReplaceInstructionIfSameShape(
    HloInstruction* old_instruction, HloInstruction* new_instruction) {
  if (!SameShape(old_instruction, new_instruction)) {
    return false;
  }
  TF_CHECK_OK(ReplaceInstruction(old_instruction, new_instruction));
  return true;
}

Status AlgebraicSimplifierVisitor::HandleAdd(HloInstruction* add) {
  auto lhs = add->mutable_operand(0);
  auto rhs = add->mutable_operand(1);
  // A + 0 => A
  VLOG(10) << "trying transform [A + 0 => A]: " << add->ToString();
  if (IsAll(rhs, 0) && ReplaceInstructionIfSameShape(add, lhs)) {
    return Status::OK();
  }
  // 0 + A => A
  VLOG(10) << "trying transform [0 + A => A]: " << add->ToString();
  if (IsAll(lhs, 0) && ReplaceInstructionIfSameShape(add, rhs)) {
    return Status::OK();
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleBitcast(HloInstruction* bitcast) {
  // If a bitcast feeds a bitcast, make it a single bitcast.
  if (bitcast->operand(0)->opcode() == HloOpcode::kBitcast) {
    return ReplaceWithNewInstruction(
        bitcast, HloInstruction::CreateUnary(
                     bitcast->shape(), HloOpcode::kBitcast,
                     bitcast->mutable_operand(0)->mutable_operand(0)));
  }
  // All bitcasts can be eliminated (assuming layout constraints are
  // satisified).
  ReplaceInstructionIfSameShape(bitcast, bitcast->mutable_operand(0));
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleCopy(HloInstruction* copy) {
  // If a copy feeds a copy, make it a single copy.
  if (copy->operand(0)->opcode() == HloOpcode::kCopy) {
    return ReplaceWithNewInstruction(
        copy, HloInstruction::CreateUnary(
                  copy->shape(), HloOpcode::kCopy,
                  copy->mutable_operand(0)->mutable_operand(0)));
  }
  // All copies can be eliminated (assuming layout constraints are satisified).
  ReplaceInstructionIfSameShape(copy, copy->mutable_operand(0));
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleConcatenate(
    HloInstruction* concatenate) {
  tensorflow::gtl::ArraySlice<HloInstruction*> operands(
      concatenate->operands());
  if (operands.size() == 1) {
    // Unary concatenates are useless.
    ReplaceInstructionIfSameShape(concatenate, operands[0]);
    return Status::OK();
  }
  // Filter out and remove empty operands.
  std::vector<HloInstruction*> nonempty_operands;
  for (HloInstruction* operand : operands) {
    if (!ShapeUtil::HasZeroElements(operand->shape())) {
      nonempty_operands.push_back(operand);
    }
  }
  if (nonempty_operands.size() < operands.size()) {
    HloInstruction* replacement;
    if (nonempty_operands.empty()) {
      replacement = operands[0];
    } else if (nonempty_operands.size() == 1) {
      replacement = nonempty_operands[0];
    } else {
      replacement =
          computation_->AddInstruction(concatenate->CloneWithNewOperands(
              concatenate->shape(), nonempty_operands));
    }
    VLOG(10) << "trying to replace " << concatenate->ToString() << " with "
             << replacement->ToString();
    ReplaceInstructionIfSameShape(concatenate, replacement);
  } else if (operands.size() == 2) {
    // A binary concat with a broadcasted scalar as an operand can be converted
    // into a pad which is simpler to fold into other operations.
    bool is_effective_low_pad =
        operands[0]->opcode() == HloOpcode::kBroadcast &&
        ShapeUtil::IsScalar(operands[0]->operand(0)->shape());
    bool is_effective_high_pad =
        operands[1]->opcode() == HloOpcode::kBroadcast &&
        ShapeUtil::IsScalar(operands[1]->operand(0)->shape());
    if (!is_effective_low_pad && !is_effective_high_pad) {
      return Status::OK();
    }
    PaddingConfig padding_config;
    for (int64 dim = 0; dim < ShapeUtil::Rank(operands[0]->shape()); ++dim) {
      auto padding_config_dim = padding_config.add_dimensions();
      padding_config_dim->set_edge_padding_high(0);
      padding_config_dim->set_edge_padding_low(0);
      padding_config_dim->set_interior_padding(0);
      if (dim == concatenate->concatenate_dimension()) {
        if (is_effective_low_pad) {
          padding_config_dim->set_edge_padding_low(
              operands[0]->shape().dimensions(dim));
        } else {
          padding_config_dim->set_edge_padding_high(
              operands[1]->shape().dimensions(dim));
        }
      }
    }
    int64 operand_to_pad = is_effective_low_pad ? 1 : 0;
    int64 pad_value_operand = is_effective_low_pad ? 0 : 1;
    HloInstruction* pad =
        computation_->AddInstruction(HloInstruction::CreatePad(
            concatenate->shape(), operands[operand_to_pad],
            operands[pad_value_operand]->mutable_operand(0), padding_config));
    return ReplaceInstruction(concatenate, pad);
  }
  return Status::OK();
}

static HloInstruction* BuildTupleConstant(HloComputation* computation,
                                          const Literal& literal) {
  if (ShapeUtil::IsTuple(literal.shape())) {
    std::vector<HloInstruction*> elems;
    elems.reserve(ShapeUtil::TupleElementCount(literal.shape()));
    for (const Literal& child : literal.tuple_literals()) {
      elems.push_back(BuildTupleConstant(computation, child));
    }
    return computation->AddInstruction(HloInstruction::CreateTuple(elems));
  } else {
    return computation->AddInstruction(
        HloInstruction::CreateConstant(MakeUnique<Literal>(literal)));
  }
}

Status AlgebraicSimplifierVisitor::HandleConstant(HloInstruction* constant) {
  // Tuple constants aren't directly supported by any backend. Expand them into
  // explicit Tuple instructions.
  if (ShapeUtil::IsTuple(constant->shape())) {
    return ReplaceInstruction(
        constant, BuildTupleConstant(computation_, constant->literal()));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleSubtract(HloInstruction* sub) {
  auto lhs = sub->mutable_operand(0);
  auto rhs = sub->mutable_operand(1);
  // A - 0 => A
  VLOG(10) << "trying transform [A - 0 => A]: " << sub->ToString();
  if (IsAll(rhs, 0) && ReplaceInstructionIfSameShape(sub, lhs)) {
    return Status::OK();
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleDivide(HloInstruction* divide) {
  auto lhs = divide->mutable_operand(0);
  auto rhs = divide->mutable_operand(1);
  // A/1 => A
  VLOG(10) << "trying transform [A/1 => A]: " << divide->ToString();
  if (IsAll(rhs, 1) && ReplaceInstructionIfSameShape(divide, lhs)) {
    return Status::OK();
  }

  // exp(A)/exp(B) => exp(A-B)
  if (lhs->opcode() == HloOpcode::kExp && rhs->opcode() == HloOpcode::kExp) {
    VLOG(10) << "transform [exp(A)/exp(B) => exp(A-B)]: " << divide->ToString();
    HloInstruction* subtract =
        computation_->AddInstruction(HloInstruction::CreateBinary(
            divide->shape(), HloOpcode::kSubtract, lhs->mutable_operand(0),
            rhs->mutable_operand(0)));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateUnary(divide->shape(), HloOpcode::kExp,
                                            subtract));
  }

  // A/exp(B) => A*exp(-B)
  if (rhs->opcode() == HloOpcode::kExp) {
    VLOG(10) << "transform [A/exp(B) => A*exp(-B)]: " << divide->ToString();
    HloInstruction* negate =
        computation_->AddInstruction(HloInstruction::CreateUnary(
            divide->shape(), HloOpcode::kNegate, rhs->mutable_operand(0)));
    HloInstruction* new_exp = computation_->AddInstruction(
        HloInstruction::CreateUnary(divide->shape(), HloOpcode::kExp, negate));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(
                    divide->shape(), HloOpcode::kMultiply, lhs, new_exp));
  }

  // A/pow(B,C) => A*pow(B,-C)
  if (rhs->opcode() == HloOpcode::kPower) {
    VLOG(10) << "transform [A/pow(B,C) => A*pow(B,-C)]: " << divide->ToString();
    // The output shape of the created negate operator should be the same as the
    // input.
    const Shape& negate_shape = rhs->operand(1)->shape();
    HloInstruction* negate =
        computation_->AddInstruction(HloInstruction::CreateUnary(
            negate_shape, HloOpcode::kNegate, rhs->mutable_operand(1)));
    // And the power operator should retain the output shape of the old one.
    const Shape& new_power_shape = rhs->shape();
    HloInstruction* new_power = computation_->AddInstruction(
        HloInstruction::CreateBinary(new_power_shape, HloOpcode::kPower,
                                     rhs->mutable_operand(0), negate));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(
                    divide->shape(), HloOpcode::kMultiply, lhs, new_power));
  }

  // Simplifying integral division would produce unexpected results.
  if (ShapeUtil::ElementIsIntegral(divide->shape())) {
    return Status::OK();
  }

  // (A / B) / (C / D)  =>  (A / B)*(D / C) => (A * D) / (B * C)
  if (lhs->opcode() == HloOpcode::kDivide &&
      rhs->opcode() == HloOpcode::kDivide) {
    TF_ASSIGN_OR_RETURN(
        const Shape a_times_d_shape,
        ShapeInference::InferBinaryOpShape(HloOpcode::kMultiply,
                                           lhs->operand(0), rhs->operand(1)));
    auto a_times_d = computation_->AddInstruction(HloInstruction::CreateBinary(
        a_times_d_shape, HloOpcode::kMultiply, lhs->mutable_operand(0),
        rhs->mutable_operand(1)));
    TF_ASSIGN_OR_RETURN(
        const Shape b_times_c_shape,
        ShapeInference::InferBinaryOpShape(HloOpcode::kMultiply,
                                           lhs->operand(1), rhs->operand(0)));
    auto b_times_c = computation_->AddInstruction(HloInstruction::CreateBinary(
        b_times_c_shape, HloOpcode::kMultiply, lhs->mutable_operand(1),
        rhs->mutable_operand(0)));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(
                    divide->shape(), HloOpcode::kDivide, a_times_d, b_times_c));
  }

  // (A / B) / C => A / (B * C)
  if (lhs->opcode() == HloOpcode::kDivide) {
    TF_ASSIGN_OR_RETURN(const Shape b_times_c_shape,
                        ShapeInference::InferBinaryOpShape(
                            HloOpcode::kMultiply, lhs->operand(1), rhs));
    auto b_times_c = computation_->AddInstruction(HloInstruction::CreateBinary(
        b_times_c_shape, HloOpcode::kMultiply, lhs->mutable_operand(1), rhs));
    return ReplaceWithNewInstruction(
        divide,
        HloInstruction::CreateBinary(divide->shape(), HloOpcode::kDivide,
                                     lhs->mutable_operand(0), b_times_c));
  }

  // A / (B / C) => (A*C) / B
  if (rhs->opcode() == HloOpcode::kDivide) {
    TF_ASSIGN_OR_RETURN(const Shape a_times_c_shape,
                        ShapeInference::InferBinaryOpShape(
                            HloOpcode::kMultiply, lhs, rhs->operand(1)));
    auto a_times_c = computation_->AddInstruction(HloInstruction::CreateBinary(
        a_times_c_shape, HloOpcode::kMultiply, lhs, rhs->mutable_operand(1)));
    return ReplaceWithNewInstruction(
        divide,
        HloInstruction::CreateBinary(divide->shape(), HloOpcode::kDivide,
                                     a_times_c, rhs->mutable_operand(0)));
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleDot(HloInstruction* dot) {
  auto lhs = dot->mutable_operand(0);
  auto rhs = dot->mutable_operand(1);
  if (!enable_dot_simplification_) {
    return Status::OK();
  }
  // Only optimize F32 dot operations where the dot, rhs and lhs are rank 2 or
  // below.
  if (dot->shape().element_type() != F32 || ShapeUtil::Rank(lhs->shape()) > 2 ||
      ShapeUtil::Rank(rhs->shape()) > 2 || ShapeUtil::Rank(dot->shape()) > 2) {
    return Status::OK();
  }

  // Replace a zero element dot with a broadcast of the constant 0.
  if (ShapeUtil::HasZeroElements(dot->shape()) ||
      ShapeUtil::HasZeroElements(lhs->shape()) ||
      ShapeUtil::HasZeroElements(rhs->shape())) {
    auto zero = computation_->AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0(0.0f)));
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateBroadcast(dot->shape(), zero, {}));
  }

  // Simplify dot(transpose(a), transpose(b)) to transpose(dot(b,a)).
  if (lhs->IsRank2Transpose() && rhs->IsRank2Transpose()) {
    auto new_dot = computation_->AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::PermuteDimensions({1, 0}, dot->shape()), HloOpcode::kDot,
        rhs->mutable_operand(0), lhs->mutable_operand(0)));
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateTranspose(dot->shape(), new_dot, {1, 0}));
  }

  // Simplify outer product into multiply with implicit broadcasting.
  //
  // A dot(a[M, 1], b[1, N]) = multiply(a [M,1], b [1, N])
  if (ShapeUtil::Rank(rhs->shape()) == 2 && rhs->shape().dimensions(0) == 1) {
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateBinary(dot->shape(), HloOpcode::kMultiply,
                                          lhs, rhs));
  }

  // The following graph transformations take Dots where at least one input is a
  // vector or has a degenerate dimension and converts it into a multiply and
  // reduce. This should enable more fusion than leaving the nodes as Dot
  // operations.

  // Strength reduce dot(a[K] , b[K]) =
  //  reshape(result.shape,
  //          reduce_sum(multiply(a, b), {0}))
  if (ShapeUtil::Rank(rhs->shape()) == 1 &&
      ShapeUtil::Rank(lhs->shape()) == 1) {
    auto multiply = computation_->AddInstruction(HloInstruction::CreateBinary(
        rhs->shape(), HloOpcode::kMultiply, lhs, rhs));
    HloComputation* add_reduce_computation = CreateScalarBinaryComputation(
        computation_->parent(), F32, HloOpcode::kAdd);
    auto zero = computation_->AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0(0.0f)));
    auto reduce = computation_->AddInstruction(HloInstruction::CreateReduce(
        ShapeUtil::MakeShape(dot->shape().element_type(), {}), multiply, zero,
        {0}, add_reduce_computation));
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateReshape(dot->shape(), reduce));
  }

  // Strength reduce dot(a[1, K], b) =
  //    reshape(result.shape,
  //      reduce_sum(
  //        multiply(broadcast(reshape(a, [K]), {0}), b),
  //        {0})
  //      )
  //    )
  if (ShapeUtil::Rank(lhs->shape()) == 1 ||
      (ShapeUtil::Rank(lhs->shape()) == 2 && lhs->shape().dimensions(0) == 1)) {
    auto new_lhs = computation_->AddInstruction(HloInstruction::CreateReshape(
        ShapeUtil::MakeShape(lhs->shape().element_type(),
                             {ShapeUtil::ElementsIn(lhs->shape())}),
        lhs));
    HloComputation* add_reduce_computation = CreateScalarBinaryComputation(
        computation_->parent(), F32, HloOpcode::kAdd);
    auto zero = computation_->AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0(0.0f)));
    HloInstruction* reduce;
    if (ShapeUtil::Rank(rhs->shape()) == 1) {
      auto multiply = computation_->AddInstruction(HloInstruction::CreateBinary(
          rhs->shape(), HloOpcode::kMultiply, new_lhs, rhs));
      reduce = computation_->AddInstruction(HloInstruction::CreateReduce(
          ShapeUtil::MakeShape(dot->shape().element_type(), {}), multiply, zero,
          {0}, add_reduce_computation));
    } else {
      new_lhs = computation_->AddInstruction(
          HloInstruction::CreateBroadcast(rhs->shape(), new_lhs, {0}));
      auto multiply = computation_->AddInstruction(HloInstruction::CreateBinary(
          rhs->shape(), HloOpcode::kMultiply, new_lhs, rhs));

      reduce = computation_->AddInstruction(HloInstruction::CreateReduce(
          ShapeUtil::MakeShape(dot->shape().element_type(),
                               {rhs->shape().dimensions(1)}),
          multiply, zero, {0}, add_reduce_computation));
    }
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateReshape(dot->shape(), reduce));
  }

  // Strength reduce dot(a, b[K, 1]) =
  //  reshape(result.shape,
  //    reduce_sum(multiply(a, broadcast(reshape([K],b), {1})), {0})
  //  )
  if (ShapeUtil::Rank(rhs->shape()) == 1 ||
      (ShapeUtil::Rank(rhs->shape()) == 2 && rhs->shape().dimensions(1) == 1)) {
    auto new_rhs = computation_->AddInstruction(HloInstruction::CreateReshape(
        ShapeUtil::MakeShape(rhs->shape().element_type(),
                             {ShapeUtil::ElementsIn(rhs->shape())}),
        rhs));
    new_rhs = computation_->AddInstruction(
        HloInstruction::CreateBroadcast(lhs->shape(), new_rhs, {1}));
    auto multiply = computation_->AddInstruction(HloInstruction::CreateBinary(
        lhs->shape(), HloOpcode::kMultiply, lhs, new_rhs));
    HloComputation* add_reduce_computation = CreateScalarBinaryComputation(
        computation_->parent(), F32, HloOpcode::kAdd);
    auto zero = computation_->AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0(0.0f)));
    auto reduce = computation_->AddInstruction(HloInstruction::CreateReduce(
        ShapeUtil::MakeShape(dot->shape().element_type(),
                             {lhs->shape().dimensions(0)}),
        multiply, zero, {1}, add_reduce_computation));
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateReshape(dot->shape(), reduce));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleMultiply(HloInstruction* multiply) {
  auto lhs = multiply->mutable_operand(0);
  auto rhs = multiply->mutable_operand(1);
  // A*1 => A
  VLOG(10) << "trying transform [A*1 => A]: " << multiply->ToString();
  if (IsAll(rhs, 1) && ReplaceInstructionIfSameShape(multiply, lhs)) {
    return Status::OK();
  }
  // 1*A => A
  VLOG(10) << "trying transform [1*A => A]: " << multiply->ToString();
  if (IsAll(lhs, 1) && ReplaceInstructionIfSameShape(multiply, rhs)) {
    return Status::OK();
  }

  // exp(A) * exp(B) => exp(A+B)
  if (lhs->opcode() == HloOpcode::kExp && rhs->opcode() == HloOpcode::kExp) {
    auto add = computation_->AddInstruction(HloInstruction::CreateBinary(
        multiply->shape(), HloOpcode::kAdd, lhs->mutable_operand(0),
        rhs->mutable_operand(0)));
    return ReplaceWithNewInstruction(
        multiply,
        HloInstruction::CreateUnary(multiply->shape(), HloOpcode::kExp, add));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleLog(HloInstruction* log) {
  // ln(exp(A)) => A
  VLOG(10) << "trying transform [ln(exp(A)) => A]: " << log->ToString();
  auto operand = log->mutable_operand(0);
  if (operand->opcode() == HloOpcode::kExp &&
      ReplaceInstructionIfSameShape(log, operand->mutable_operand(0))) {
    return Status::OK();
  }

  // ln(pow(A,B)) => B*ln(A)
  if (operand->opcode() == HloOpcode::kPower) {
    auto new_log = computation_->AddInstruction(HloInstruction::CreateUnary(
        log->shape(), HloOpcode::kLog, operand->mutable_operand(0)));
    return ReplaceWithNewInstruction(
        log,
        HloInstruction::CreateBinary(log->shape(), HloOpcode::kMultiply,
                                     new_log, operand->mutable_operand(1)));
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  auto operand = get_tuple_element->mutable_operand(0);
  if (operand->opcode() == HloOpcode::kTuple) {
    // get_tuple_element(make_tuple({A_0, A_1, ..., A_n}), i) => A_i
    VLOG(10) << "trying transform "
             << "[get_tuple_element(make_tuple({...,A_i,...}), i)] => A_i: "
             << get_tuple_element->ToString();
    if (ReplaceInstructionIfSameShape(
            get_tuple_element,
            operand->mutable_operand(get_tuple_element->tuple_index()))) {
      return Status::OK();
    }
  }
  return Status::OK();
}

namespace {

// Return whether the given reshape instruction leaves the dimensions at the
// given input indices unmodified, and returns their output indices.
//
// Example:
//   input_dim_indices = {2, 3}
//   input  shape = T[a, b, x, y, cd]
//   output shape = T[ab, x, 1, y, c, d]
//   return value = {1, 3}
//
// Precondition: input_dim_indices is sorted.
std::pair<bool, std::vector<int64>> ReshapeLeavesDimensionsUnmodified(
    const HloInstruction* hlo,
    tensorflow::gtl::ArraySlice<int64> input_dim_indices) {
  CHECK_EQ(HloOpcode::kReshape, hlo->opcode());
  CHECK(std::is_sorted(input_dim_indices.begin(), input_dim_indices.end()));

  std::vector<int64> output_dim_indices;
  std::vector<std::pair<int64, int64>> unmodified_dims =
      ShapeUtil::DimensionsUnmodifiedByReshape(hlo->operand(0)->shape(),
                                               hlo->shape());
  size_t i = 0;  // index to unmodified_dims
  for (int64 input_dim_index : input_dim_indices) {
    // Search unmodified_dims for input_dim_index. We can search from the last
    // matching position because input_dim_indices is guaranteed to be sorted.
    while (i < unmodified_dims.size() &&
           unmodified_dims[i].first < input_dim_index) {
      ++i;
    }
    if (i >= unmodified_dims.size() ||
        unmodified_dims[i].first != input_dim_index) {
      return std::make_pair(false, std::vector<int64>());
    }
    output_dim_indices.push_back(unmodified_dims[i].second);
  }
  return std::make_pair(true, output_dim_indices);
}

// Returns true if the output of "instruction" is a permutation of the
// elements of "operand". Precondition: "operand" is an operand of
// "instruction".
bool OutputIsPermutationOfOperandElements(HloInstruction* instruction,
                                          HloInstruction* operand) {
  DCHECK(!instruction->OperandIndices(operand).empty());
  switch (instruction->opcode()) {
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSort:
    case HloOpcode::kTranspose:
      return true;
    default:
      return false;
  }
}

// Returns true if the output of "instruction" is a subset of the elements of
// "operand". Precondition: "operand" is an operand of "instruction".
bool OutputIsSubsetOfOperandElements(HloInstruction* instruction,
                                     HloInstruction* operand) {
  std::vector<int64> operand_indices = instruction->OperandIndices(operand);
  CHECK(!operand_indices.empty());
  if (operand_indices.size() != 1) {
    return false;
  }
  int64 operand_index = operand_indices[0];
  switch (instruction->opcode()) {
    case HloOpcode::kSlice:
      CHECK_EQ(0, operand_index);
      return true;
    case HloOpcode::kDynamicSlice:
      return operand_index == 0;
    default:
      return false;
  }
}

}  // namespace

Status AlgebraicSimplifierVisitor::HandleBroadcast(HloInstruction* broadcast) {
  auto operand = broadcast->mutable_operand(0);
  // A degenerate broadcast of a reshape that does not change the number of
  // elements can be replaced by a reshape.
  if (std::is_sorted(broadcast->dimensions().begin(),
                     broadcast->dimensions().end()) &&
      ShapeUtil::ElementsIn(broadcast->shape()) ==
          ShapeUtil::ElementsIn(operand->shape())) {
    VLOG(10) << "transform broadcast(X) -> reshape(X) where "
                "n(broadcast(X)) == n(X)";
    return ReplaceWithNewInstruction(
        broadcast, HloInstruction::CreateReshape(broadcast->shape(), operand));
  }

  // A degenerate broadcast that has the same input and output rank can be
  // converted into a transpose.
  if (ShapeUtil::Rank(broadcast->shape()) ==
          ShapeUtil::Rank(operand->shape()) &&
      ShapeUtil::ElementsIn(broadcast->shape()) ==
          ShapeUtil::ElementsIn(operand->shape())) {
    VLOG(10) << "transform broadcast(X) -> transpose(X) where "
                "n(broadcast(X)) == n(X)";
    return ReplaceWithNewInstruction(
        broadcast, HloInstruction::CreateTranspose(broadcast->shape(), operand,
                                                   broadcast->dimensions()));
  }

  // A broadcast of a reshape which merely inserts 1-sized dimensions can
  // elide its operand.
  {
    bool merely_inserts_or_deletes_1_sized_dimensions;
    std::vector<int64> inserted_indices, deleted_indices;
    std::tie(merely_inserts_or_deletes_1_sized_dimensions, deleted_indices,
             inserted_indices) =
        operand->ReshapeMerelyInsertsOrDeletes1SizedDimensions();
    if (merely_inserts_or_deletes_1_sized_dimensions &&
        deleted_indices.empty()) {
      std::reverse(inserted_indices.begin(), inserted_indices.end());
      auto dims = broadcast->dimensions();
      for (auto inserted_index : inserted_indices) {
        dims.erase(dims.begin() + inserted_index);
      }
      return ReplaceWithNewInstruction(
          broadcast,
          HloInstruction::CreateBroadcast(broadcast->shape(),
                                          operand->mutable_operand(0), dims));
    }
  }

  // A Broadcast that feeds a unary element-wise operation can sink the
  // broadcast after the unary element-wise operation.
  TF_ASSIGN_OR_RETURN(
      bool sink_succeeded,
      TryToSinkReshapeOrBroadcastAfterOpWithUniqueNonScalarOperand(broadcast));
  changed_ |= sink_succeeded;
  if (sink_succeeded) {
    return Status::OK();
  }

  // A scalar broadcast feeding an instruction which only permutes (reshape,
  // transpose, sort, reverse) or selects a subset of operand elements (slice,
  // dynamic slice) can be replaced with a broadcast directly to the output
  // shape of the instruction.
  if (ShapeUtil::IsScalar(operand->shape())) {
    for (HloInstruction* user : broadcast->users()) {
      // Skip if the broadcast user has no uses itself.
      if (user->user_count() == 0 && user != computation_->root_instruction()) {
        continue;
      }
      if (OutputIsPermutationOfOperandElements(user, broadcast) ||
          OutputIsSubsetOfOperandElements(user, broadcast)) {
        VLOG(10) << "transform permuting/subset  of a scalar broadcast into "
                 << "a single broadcast";
        HloInstruction* new_broadcast = computation_->AddInstruction(
            HloInstruction::CreateBroadcast(user->shape(), operand, {}));
        // Use HloInstruction::ReplaceAllUsesWith instead of
        // HloComputation::ReplaceWithNewInstruction because we are replacing an
        // instruction other than the visited instruction.
        changed_ = true;
        return user->ReplaceAllUsesWith(new_broadcast);
      }
    }
  }
  return Status::OK();
}

// A conversion to the same element type as the operand is a nop and can be
// removed.  A conversion of a constant can be simplified by making a new
// constant.
Status AlgebraicSimplifierVisitor::HandleConvert(HloInstruction* convert) {
  PrimitiveType src_type = convert->operand(0)->shape().element_type();
  PrimitiveType dest_type = convert->shape().element_type();
  if (src_type == dest_type) {
    return ReplaceInstruction(convert, convert->mutable_operand(0));
  }
  return Status::OK();
}

// Real(Complex(r, i)) -> r
Status AlgebraicSimplifierVisitor::HandleReal(HloInstruction* real) {
  auto operand = real->mutable_operand(0);
  if (operand->opcode() == HloOpcode::kComplex) {
    return ReplaceInstruction(real, operand->mutable_operand(0));
  }
  return Status::OK();
}

// Imag(Complex(r, i)) -> i
Status AlgebraicSimplifierVisitor::HandleImag(HloInstruction* imag) {
  auto operand = imag->mutable_operand(0);
  if (operand->opcode() == HloOpcode::kComplex) {
    return ReplaceInstruction(imag, operand->mutable_operand(1));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandlePad(HloInstruction* pad) {
  // Eliminate nop pads (padding all zero), and replace a pad with negative
  // padding with a pad with non-negative padding followed by a slice.
  bool all_zero = true;
  bool has_negative = false;
  for (auto& padding_dimension : pad->padding_config().dimensions()) {
    if (padding_dimension.edge_padding_low() < 0 ||
        padding_dimension.edge_padding_high() < 0) {
      has_negative = true;
    }
    if (padding_dimension.edge_padding_low() != 0 ||
        padding_dimension.edge_padding_high() != 0) {
      all_zero = false;
    }
  }

  if (all_zero) {
    ReplaceInstructionIfSameShape(pad, pad->mutable_operand(0));
    return Status::OK();
  }

  if (has_negative) {
    // Pad has negative padding. Replace with a pad with the non-negative
    // padding followed by a slice which effectively performs the negative
    // padding.
    // TODO(b/34628603): Add support for negative padding in the backends, or
    // change kPad semantics to disallow negative padding and use slice
    // instead.

    // First construct the padding config with non-negative entries and the
    // compute the shape of this new pad instruction.
    PaddingConfig nonzero_padding = pad->padding_config();
    for (int i = 0; i < pad->padding_config().dimensions_size(); ++i) {
      PaddingConfig::PaddingConfigDimension* padding_dimension =
          nonzero_padding.mutable_dimensions(i);
      // Set negative padding to zero.
      if (padding_dimension->edge_padding_low() < 0) {
        padding_dimension->set_edge_padding_low(0);
      }
      if (padding_dimension->edge_padding_high() < 0) {
        padding_dimension->set_edge_padding_high(0);
      }
    }
    TF_ASSIGN_OR_RETURN(Shape nonzero_pad_shape,
                        ShapeInference::InferPadShape(pad->operand(0)->shape(),
                                                      pad->operand(1)->shape(),
                                                      nonzero_padding));
    // Copy the layout from the original pad instructions. The new pad and the
    // slice instruction should all have the same layout.
    TF_RETURN_IF_ERROR(
        LayoutUtil::CopyLayoutBetweenShapes(pad->shape(), &nonzero_pad_shape));
    HloInstruction* nonzero_pad = computation_->AddInstruction(
        HloInstruction::CreatePad(nonzero_pad_shape, pad->mutable_operand(0),
                                  pad->mutable_operand(1), nonzero_padding));

    // Second, construct the slice instruction to perform the negative padding.
    std::vector<int64> start_indices;
    std::vector<int64> end_indices;
    std::vector<int64> strides;
    for (int64 i = 0; i < pad->padding_config().dimensions_size(); ++i) {
      const PaddingConfig::PaddingConfigDimension& padding_dimension =
          pad->padding_config().dimensions(i);
      int64 start = 0;
      if (padding_dimension.edge_padding_low() < 0) {
        start = -1 * padding_dimension.edge_padding_low();
      }
      int64 end = nonzero_pad_shape.dimensions(i);
      if (padding_dimension.edge_padding_high() < 0) {
        end += padding_dimension.edge_padding_high();
      }
      start_indices.push_back(start);
      end_indices.push_back(end);
      strides.push_back(1);
    }

    // Verify that the slice shape matches the pad shape.
    TF_ASSIGN_OR_RETURN(
        Shape inferred_slice_shape,
        ShapeInference::InferSliceShape(nonzero_pad_shape, start_indices,
                                        end_indices, strides));
    TF_RET_CHECK(ShapeUtil::Compatible(inferred_slice_shape, pad->shape()));

    std::unique_ptr<HloInstruction> slice = HloInstruction::CreateSlice(
        pad->shape(), nonzero_pad, start_indices, end_indices, strides);
    return ReplaceWithNewInstruction(pad, std::move(slice));
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandlePower(HloInstruction* power) {
  VLOG(10) << "trying transform [pow(A, 0) => 1]: " << power->ToString();
  auto lhs = power->mutable_operand(0);
  auto rhs = power->mutable_operand(1);
  if (IsAll(rhs, 0)) {
    auto one = HloInstruction::CreateConstant(
        Literal::One(power->shape().element_type()).CloneToUnique());
    std::unique_ptr<HloInstruction> ones;
    if (ShapeUtil::IsScalar(power->shape())) {
      ones = std::move(one);
    } else {
      ones = HloInstruction::CreateBroadcast(
          power->shape(), computation_->AddInstruction(std::move(one)), {});
    }
    return ReplaceWithNewInstruction(power, std::move(ones));
  }

  VLOG(10) << "trying transform [pow(A, 1) => A]: " << power->ToString();
  if (IsAll(rhs, 1) && ReplaceInstructionIfSameShape(power, lhs)) {
    return Status::OK();
  }

  // pow(exp(A),B) => exp(A*B)
  if (lhs->opcode() == HloOpcode::kExp) {
    auto a_times_b = computation_->AddInstruction(HloInstruction::CreateBinary(
        power->shape(), HloOpcode::kMultiply, lhs->operands()[0], rhs));
    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateUnary(power->shape(), HloOpcode::kExp,
                                           a_times_b));
  }
  VLOG(10) << "trying transform [pow(A, 2) => A*A]: " << power->ToString();
  if (IsAll(rhs, 2)) {
    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateBinary(power->shape(),
                                            HloOpcode::kMultiply, lhs, lhs));
  }

  VLOG(10) << "trying transform [pow(A, -1) => 1/A]: " << power->ToString();
  if (IsAll(rhs, -1)) {
    auto* one = computation_->AddInstruction(HloInstruction::CreateConstant(
        Literal::One(rhs->shape().element_type()).CloneToUnique()));
    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateBinary(power->shape(), HloOpcode::kDivide,
                                            one, lhs));
  }
  return Status::OK();
}

StatusOr<bool> AlgebraicSimplifierVisitor::
    TryToSinkReshapeOrBroadcastAfterOpWithUniqueNonScalarOperand(
        HloInstruction* reshape_or_broadcast) {
  bool changed = false;
  if (ShapeUtil::IsScalar(reshape_or_broadcast->shape())) {
    return false;
  }
  HloInstruction* operand = reshape_or_broadcast->mutable_operand(0);
  for (HloInstruction* user : reshape_or_broadcast->users()) {
    if (user->user_count() == 0 && user != computation_->root_instruction()) {
      continue;
    }
    // Do not move reshapes or broadcasts past copies since the shape the copy
    // will operate on will change.
    if (user->opcode() == HloOpcode::kCopy) {
      continue;
    }
    // Do not change the shape of fusion nodes in case there a multiple shapes
    // inside the fusion node already.
    if (user->opcode() == HloOpcode::kFusion) {
      continue;
    }
    if (!user->IsElementwise()) {
      continue;
    }

    int64 reshape_or_broadcast_operand_index = -1;
    // Find the unique non-scalar operand or continue if there isn't one.
    int64 scalar_count = 0;
    for (int64 i = 0; i < user->operand_count(); ++i) {
      if (ShapeUtil::IsScalar(user->operand(i)->shape())) {
        ++scalar_count;
      } else {
        reshape_or_broadcast_operand_index = i;
      }
    }
    if (scalar_count != user->operand_count() - 1) {
      continue;
    }
    VLOG(4) << "Sinking reshape or broadcast after user:";
    VLOG(4) << "  old reshape/broadcast: " << reshape_or_broadcast->ToString();
    VLOG(4) << "  old user: " << user->ToString();
    CHECK_EQ(user->operand(reshape_or_broadcast_operand_index),
             reshape_or_broadcast);
    auto new_user_operands = user->operands();
    new_user_operands[reshape_or_broadcast_operand_index] = operand;
    auto new_user = computation_->AddInstruction(user->CloneWithNewOperands(
        ShapeUtil::MakeShapeWithLayout(
            user->shape().element_type(),
            AsInt64Slice(operand->shape().dimensions()),
            AsInt64Slice(operand->shape().layout().minor_to_major())),
        new_user_operands));
    VLOG(4) << "  new user: " << new_user->ToString();
    HloInstruction* new_reshape_or_broadcast = nullptr;
    if (reshape_or_broadcast->opcode() == HloOpcode::kReshape) {
      new_reshape_or_broadcast =
          computation_->AddInstruction(HloInstruction::CreateReshape(
              ShapeUtil::MakeShapeWithLayout(
                  user->shape().element_type(),
                  AsInt64Slice(reshape_or_broadcast->shape().dimensions()),
                  AsInt64Slice(
                      reshape_or_broadcast->shape().layout().minor_to_major())),
              new_user));
    } else {
      TF_RET_CHECK(reshape_or_broadcast->opcode() == HloOpcode::kBroadcast);
      new_reshape_or_broadcast =
          computation_->AddInstruction(HloInstruction::CreateBroadcast(
              ShapeUtil::MakeShapeWithLayout(
                  user->shape().element_type(),
                  AsInt64Slice(reshape_or_broadcast->shape().dimensions()),
                  AsInt64Slice(
                      reshape_or_broadcast->shape().layout().minor_to_major())),
              new_user, reshape_or_broadcast->dimensions()));
    }
    VLOG(4) << "  new reshape/broadcast: "
            << new_reshape_or_broadcast->ToString();
    TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(new_reshape_or_broadcast));
    changed = true;
  }
  return changed;
}

Status AlgebraicSimplifierVisitor::HandleReshape(HloInstruction* reshape) {
  auto operand = reshape->mutable_operand(0);

  // Reshape directly to empty constant if the shape contains zero-element
  // dimension.
  if (ShapeUtil::HasZeroElements(reshape->shape())) {
    auto empty_constant = HloInstruction::CreateConstant(
        Literal::CreateFromShape(reshape->shape()));

    return ReplaceWithNewInstruction(reshape, std::move(empty_constant));
  }

  // Delete no-op reshapes, i.e. where shape = operand shape.
  if (SameShape(reshape, operand)) {
    VLOG(10) << "deleting no-op reshape";
    return ReplaceInstruction(reshape, operand);
  }

  // Merge reshapes.
  if (HloOpcode::kReshape == operand->opcode()) {
    return ReplaceWithNewInstruction(
        reshape, HloInstruction::CreateReshape(reshape->shape(),
                                               operand->mutable_operand(0)));
  }

  if (HloOpcode::kBroadcast == reshape->operand(0)->opcode()) {
    auto opt_dims = ReshapeLeavesDimensionsUnmodified(
        reshape, reshape->operand(0)->dimensions());
    if (opt_dims.first) {
      return ReplaceWithNewInstruction(
          reshape,
          HloInstruction::CreateBroadcast(
              reshape->shape(), reshape->mutable_operand(0)->mutable_operand(0),
              opt_dims.second));
    }
  }

  // A Reshape that feeds a unary element-wise operation can sink the
  // reshape after the unary element-wise operation.
  TF_ASSIGN_OR_RETURN(
      bool sink_succeeded,
      TryToSinkReshapeOrBroadcastAfterOpWithUniqueNonScalarOperand(reshape));
  changed_ |= sink_succeeded;
  if (sink_succeeded) {
    return Status::OK();
  }

  // Make this a bitcast if possible.
  if (is_layout_sensitive_ &&
      ReshapeIsBitcast(reshape, valid_bitcast_callback_)) {
    ReplaceWithBitcast(reshape);
    return Status::OK();
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleReverse(HloInstruction* reverse) {
  // When all the dimensions to reverse are trivial (i.e. the bound is 1),
  // there is nothing to be done.
  auto dim_is_one = [&](int64 i) -> bool {
    return reverse->shape().dimensions(i) == 1;
  };
  if (std::all_of(reverse->dimensions().begin(), reverse->dimensions().end(),
                  dim_is_one)) {
    return ReplaceInstruction(reverse, reverse->mutable_operand(0));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleSlice(HloInstruction* slice) {
  // Delete no-op slices, i.e. where shape = operand shape.
  if (ReplaceInstructionIfSameShape(slice, slice->mutable_operand(0))) {
    return Status::OK();
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleDynamicSlice(
    HloInstruction* dynamic_slice) {
  auto operand = dynamic_slice->mutable_operand(0);
  auto start_indices = dynamic_slice->operand(1);
  if (ShapeUtil::IsScalar(dynamic_slice->shape())) {
    return ReplaceInstruction(dynamic_slice, operand);
  }
  // DynamicSlice where operand has the same size as the output and
  // start_indices are all zero is simply equal to operand.
  if (IsAll(start_indices, 0) && SameShape(operand, dynamic_slice)) {
    return ReplaceInstruction(dynamic_slice, operand);
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  auto update = dynamic_update_slice->mutable_operand(1);
  auto start_indices = dynamic_update_slice->operand(2);
  // DynamicUpdateSlice on a scalar just passes through the update argument.
  if (ShapeUtil::IsScalar(dynamic_update_slice->shape())) {
    return ReplaceInstruction(dynamic_update_slice, update);
  }

  // DynamicUpdateSlice where operand and update have the same size and
  // start_indices are all zero is simply equal to update.
  //
  // (We require start_indices to be all zero because we want this optimization
  // not to affect the visible behavior of this op even when the indices are out
  // of range.  Currently dynamic-update-slice wraps out-of-range indices, so
  // we can only remove the op if its indices never wrap.)
  if (IsAll(start_indices, 0) && SameShape(dynamic_update_slice, update)) {
    return ReplaceInstruction(dynamic_update_slice, update);
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleReduce(HloInstruction* reduce) {
  auto arg = reduce->mutable_operand(0);
  auto init_value = reduce->mutable_operand(1);
  tensorflow::gtl::ArraySlice<int64> dimensions(reduce->dimensions());
  HloComputation* function = reduce->to_apply();
  if (ShapeUtil::HasZeroElements(arg->shape()) ||
      ShapeUtil::HasZeroElements(reduce->shape())) {
    return ReplaceWithNewInstruction(
        reduce,
        HloInstruction::CreateBroadcast(reduce->shape(), init_value, {}));
  }
  // A Transpose feeding a reduce can simply permute the reduction dimensions
  // field.
  if (arg->opcode() == HloOpcode::kTranspose) {
    auto transpose_dimensions = arg->dimensions();
    std::vector<int64> new_reduce_dimensions;
    for (auto dim : dimensions) {
      new_reduce_dimensions.push_back(transpose_dimensions[dim]);
    }
    return ReplaceWithNewInstruction(
        reduce, HloInstruction::CreateReduce(
                    reduce->shape(), arg->mutable_operand(0), init_value,
                    new_reduce_dimensions, function));
  }

  // A reshape that collapses multiple dimensions into a dimension being
  // reduced can just reduce all of those dimensions instead of doing a
  // collapsing reshape before a reduction.
  if (arg->opcode() == HloOpcode::kReshape) {
    std::vector<std::pair<int64, int64>> unmodified_dims =
        ShapeUtil::DimensionsUnmodifiedByReshape(arg->operand(0)->shape(),
                                                 arg->shape());
    std::vector<bool> arg_dim_in_output(ShapeUtil::Rank(arg->shape()), true);
    std::vector<bool> arg_dim_unmodified(ShapeUtil::Rank(arg->shape()), false);
    for (auto dim : dimensions) {
      arg_dim_in_output[dim] = false;
    }
    for (auto dim_pair : unmodified_dims) {
      arg_dim_unmodified[dim_pair.second] = true;
    }
    // The goal is to verify that all dimensions that are not removed in the
    // reduce are unmodified by the reshape. For example:
    // reduce(reshape([A,B*C], a[A,B,C]),[1]) = reduce(a[A, B, C], [1, 2])
    bool can_move_reshape_into_reduce = true;
    for (int64 i = 0; i < arg_dim_in_output.size(); ++i) {
      if (arg_dim_in_output[i] && !arg_dim_unmodified[i]) {
        can_move_reshape_into_reduce = false;
      }
    }
    if (can_move_reshape_into_reduce) {
      changed_ = true;
      std::unordered_set<int64> dimensions_not_to_reduce;
      for (auto dim_pair : unmodified_dims) {
        if (arg_dim_in_output[dim_pair.second]) {
          dimensions_not_to_reduce.insert(dim_pair.first);
        }
      }
      std::vector<int64> new_reduce_dimensions;
      for (int64 i = 0; i < ShapeUtil::Rank(arg->operand(0)->shape()); ++i) {
        if (dimensions_not_to_reduce.count(i) == 0) {
          new_reduce_dimensions.push_back(i);
        }
      }
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateReduce(
                      reduce->shape(), arg->mutable_operand(0), init_value,
                      new_reduce_dimensions, function));
    }
  }
  if (ShapeUtil::ElementsIn(reduce->shape()) ==
          ShapeUtil::ElementsIn(arg->shape()) ||
      ShapeUtil::HasZeroElements(arg->shape())) {
    auto reshape = computation_->AddInstruction(
        HloInstruction::CreateReshape(reduce->shape(), arg));
    return ReplaceWithNewInstruction(
        reduce, HloInstruction::CreateMap(reduce->shape(),
                                          {reshape, init_value}, function));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleReduceWindow(
    HloInstruction* reduce_window) {
  auto operand = reduce_window->mutable_operand(0);
  const Window& window = reduce_window->window();
  auto function = reduce_window->to_apply();
  VLOG(10) << "Considering folding Pad: " << operand->ToString()
           << "\ninto reduce-window: " << reduce_window->ToString();

  // This optimization folds a pad op into reduce_window.
  if (operand->opcode() != HloOpcode::kPad) {
    VLOG(10) << "Not folding pad into reduce-window as there is no pad.";
    return Status::OK();
  }

  // Do not fold interior padding into ReduceWindow since the backends do not
  // support it.
  const PaddingConfig& pad_config = operand->padding_config();
  if (HasInteriorPadding(pad_config)) {
    VLOG(10) << "Not folding pad into reduce-window due to interior padding.";
    return Status::OK();
  }

  // If reduce_window already has padding, the pad value of the pad op and the
  // init value of reduce_window must match to allow folding the pad.
  const HloInstruction* pad_value = operand->operand(1);
  const HloInstruction* reduce_init_value = reduce_window->operand(1);
  if (pad_value != reduce_init_value) {
    // The pad value is usually a constant, so we handle that case and do not
    // try to get more fancy about proving equivalence in cases beyond that.
    if (pad_value->opcode() != HloOpcode::kConstant ||
        reduce_init_value->opcode() != HloOpcode::kConstant ||
        pad_value->literal() != reduce_init_value->literal()) {
      VLOG(10) << "Not folding pad into reduce-window due to different pad "
                  "values.";
      return Status::OK();
    }
  }

  // Carry out the folding of the pad into reduce_window.
  VLOG(10) << "Folding pad into reduce-window.";
  Window new_window = window;
  const int64 rank = ShapeUtil::Rank(reduce_window->shape());
  TF_RET_CHECK(pad_config.dimensions_size() == rank);
  TF_RET_CHECK(window.dimensions_size() == rank);
  for (int64 i = 0; i < rank; ++i) {
    const auto& pad_dim = pad_config.dimensions(i);
    auto& window_dim = *new_window.mutable_dimensions(i);
    window_dim.set_padding_low(window_dim.padding_low() +
                               pad_dim.edge_padding_low());
    window_dim.set_padding_high(window_dim.padding_high() +
                                pad_dim.edge_padding_high());
  }
  return ReplaceWithNewInstruction(
      reduce_window, HloInstruction::CreateReduceWindow(
                         /*shape=*/reduce_window->shape(),
                         /*operand=*/operand->mutable_operand(0),
                         /*init_value=*/reduce_window->mutable_operand(1),
                         /*window=*/new_window,
                         /*reduce_computation=*/function));
}

Status AlgebraicSimplifierVisitor::HandleTranspose(HloInstruction* transpose) {
  auto operand = transpose->mutable_operand(0);

  if (std::is_sorted(transpose->dimensions().begin(),
                     transpose->dimensions().end())) {
    VLOG(10) << "deleting no-op transpose";
    return ReplaceInstruction(transpose, operand);
  }

  if (HloOpcode::kTranspose == operand->opcode()) {
    return ReplaceWithNewInstruction(
        transpose, HloInstruction::CreateTranspose(
                       transpose->shape(), operand->mutable_operand(0),
                       ComposePermutations(operand->dimensions(),
                                           transpose->dimensions())));
  }

  if (is_layout_sensitive_ && TransposeIsBitcast(transpose)) {
    ReplaceWithBitcast(transpose);
    return Status::OK();
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleConvolution(
    HloInstruction* convolution) {
  auto lhs = convolution->mutable_operand(0);
  auto rhs = convolution->mutable_operand(1);
  const auto& window = convolution->window();
  if (!enable_conv_simplification_) {
    return Status::OK();
  }
  // HandleConvolution tries to replace a convolution with a DOT instruction.
  //
  // Only add when bitcasts can be used:
  // - if bitcasts are not supported, then reshapes could be used but will
  //   end up with another copy.
  // - if bitcasts are supported, the simplifier will be called again with
  //   bitcasts_ == true.

  // TODO(cwhipkey): b/31337498, make this layout insensitive.
  if (!is_layout_sensitive_) {
    return Status::OK();
  }

  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();
  const Shape& input_shape = lhs->shape();
  const Shape& filter_shape = rhs->shape();
  const Shape& convolution_shape = convolution->shape();
  TF_RET_CHECK(LayoutUtil::HasLayout(input_shape));
  TF_RET_CHECK(LayoutUtil::HasLayout(filter_shape));
  TF_RET_CHECK(LayoutUtil::HasLayout(convolution_shape));

  // Require the spatial dimensions in the kernel to have a bound of one.
  for (int64 i = 0; i < dnums.kernel_spatial_dimensions_size(); ++i) {
    if (filter_shape.dimensions(dnums.kernel_spatial_dimensions(i)) != 1) {
      return Status::OK();
    }
  }

  // Stride ignores part of the output, which matrix multiplication does not do,
  // so require no stride. Padding and base (lhs) dilation both implicitly
  // extend the data, which matrix multiplication also does not do, so require
  // no padding and no base (lhs) dilation. Window (rhs) dilation has no effect
  // for a 1x1 window, so window dilation is no problem.
  if (window_util::HasStride(window) || window_util::HasPadding(window) ||
      window_util::HasBaseDilation(window)) {
    return Status::OK();
  }

  // Also, the shapes must align for a rowmajor matmul:
  // - the input and output have the same layout.
  // - for input/output, the channel dimension must be the most minor. Other
  //   spatial dims can be in any order.
  // - for filters, the input channel dimension must be more major than the
  //   output channel dimension. The width+height don't matter because
  //   they are 1.
  //
  // These constraints are harsh. If the channel dimension is the most major
  // and/or the layout of input/output feature dimensions are reversed, we can
  // still convert Conv into more efficient Matmul with operand transposition
  // (such as the transposition flags in cuBLAS SGEMM).
  if (!LayoutUtil::Equal(input_shape.layout(), convolution_shape.layout()) ||
      input_shape.layout().minor_to_major(0) !=
          dnums.input_feature_dimension() ||
      convolution_shape.layout().minor_to_major(0) !=
          dnums.output_feature_dimension() ||
      // The input feature dimension should come later in the minor-to-major
      // order.
      (PositionInContainer(filter_shape.layout().minor_to_major(),
                           dnums.kernel_input_feature_dimension()) <
       PositionInContainer(filter_shape.layout().minor_to_major(),
                           dnums.kernel_output_feature_dimension()))) {
    return Status::OK();
  }

  auto add_bitcast = [&](Shape shape, HloInstruction* operand) {
    std::vector<int64> dims(operand->shape().dimensions_size());
    std::iota(dims.begin(), dims.end(), 0);
    return computation_->AddInstruction(
        HloInstruction::CreateUnary(shape, HloOpcode::kBitcast, operand));
  };

  // Replace it with a dot, with bitcasts around it to get the right shape.
  const int64 input_channels =
      input_shape.dimensions(dnums.input_feature_dimension());
  const int64 output_channels =
      filter_shape.dimensions(dnums.kernel_output_feature_dimension());

  // Computes the product of the non-feature dimensions.
  int64 conv_width = 1;
  for (int i = 0; i < input_shape.dimensions_size(); ++i) {
    if (i != dnums.input_feature_dimension()) {
      conv_width *= input_shape.dimensions(i);
    }
  }

  // We already checked feature_dimension is most minor, so data in input_shape
  // and row-major {conv_width,input_channels} are bitwise identical.
  const Shape new_input_shape =
      ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(
          input_shape.element_type(), {conv_width, input_channels});
  // We already checked input_feature_dimension is more major than
  // output_feature_dimension, so data in filter_shape and row-major
  // {input_channels,output_channels} are bitwise identical.
  const Shape new_filter_shape =
      ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(
          filter_shape.element_type(), {input_channels, output_channels});
  const Shape dot_output_shape =
      ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(
          convolution_shape.element_type(), {conv_width, output_channels});

  // We cannot insert bitcasts if the layouts will not be compatible.
  // TODO(b/33178038): Consider inserting a transpose if a bitcast would be
  // invalid.
  if (!valid_bitcast_callback_(input_shape, new_input_shape) ||
      !valid_bitcast_callback_(filter_shape, new_filter_shape) ||
      !valid_bitcast_callback_(dot_output_shape, convolution_shape)) {
    return Status::OK();
  }

  auto new_lhs = add_bitcast(new_input_shape, lhs);
  auto new_rhs = add_bitcast(new_filter_shape, rhs);
  auto dot = computation_->AddInstruction(HloInstruction::CreateBinary(
      dot_output_shape, HloOpcode::kDot, new_lhs, new_rhs));
  return ReplaceInstruction(convolution, add_bitcast(convolution_shape, dot));
}

bool AlgebraicSimplifierVisitor::TransformToClampIfSameShape(
    HloInstruction* root, HloInstruction* min, HloInstruction* min_operand,
    HloInstruction* operand, HloInstruction* max, HloInstruction* max_operand) {
  // Ensure shapes of min and max operand are equal to match current shape
  // inference.
  if (!SameShape(min_operand, max_operand)) {
    return false;
  }

  auto clamp = HloInstruction::CreateTernary(root->shape(), HloOpcode::kClamp,
                                             max_operand, operand, min_operand);
  TF_CHECK_OK(ReplaceWithNewInstruction(root, std::move(clamp)));
  return true;
}

Status AlgebraicSimplifierVisitor::HandleMaximum(HloInstruction* maximum) {
  // Match the following tree:
  //          min_operand     operand
  //                     \   /
  //      max_operand     min
  //                 \   /
  //                  max
  // where max_operand and min_operand are scalar constants.
  {
    HloInstruction* min;
    HloInstruction* max_operand;
    HloInstruction* min_operand;
    HloInstruction* operand;

    if (hlo_query::MatchBinaryInstructionOperandOpcode(
            HloOpcode::kMinimum, maximum,
            /*matching_operand=*/&min,
            /*other_operand=*/&max_operand) &&
        hlo_query::MatchBinaryInstructionOperand(
            hlo_query::IsScalarConstant, min,
            /*matching_operand=*/&min_operand,
            /*other_operand=*/&operand) &&
        TransformToClampIfSameShape(maximum, min, min_operand, operand, maximum,
                                    max_operand)) {
      return Status::OK();
    }
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleMinimum(HloInstruction* minimum) {
  // Match the following tree:
  //          max_operand     operand
  //                     \   /
  //      min_operand     max
  //                 \   /
  //                  min
  // where max_operand and min_operand are scalar constants.
  {
    HloInstruction* max;
    HloInstruction* max_operand;
    HloInstruction* min_operand;
    HloInstruction* operand;

    if (hlo_query::MatchBinaryInstructionOperandOpcode(
            HloOpcode::kMaximum, minimum,
            /*matching_operand=*/&max,
            /*other_operand=*/&min_operand) &&
        hlo_query::MatchBinaryInstructionOperand(
            hlo_query::IsScalarConstant, max,
            /*matching_operand=*/&max_operand,
            /*other_operand=*/&operand) &&
        TransformToClampIfSameShape(minimum, minimum, min_operand, operand, max,
                                    max_operand)) {
      return Status::OK();
    }
  }

  return Status::OK();
}

StatusOr<bool> AlgebraicSimplifier::Run(HloModule* module) {
  XLA_VLOG_LINES(2,
                 "AlgebraicSimplifier::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (AlgebraicSimplifierVisitor::Run(
            comp, is_layout_sensitive_, valid_bitcast_callback_,
            enable_dot_simplification_, enable_conv_simplification_)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(2,
                 "AlgebraicSimplifier::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla

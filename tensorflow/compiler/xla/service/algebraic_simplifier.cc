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
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

namespace {

namespace m = match;

bool IsAll(const HloInstruction* op, int8 value) {
  switch (op->opcode()) {
    case HloOpcode::kBroadcast:
      return IsAll(op->operand(0), value);
    case HloOpcode::kConstant:
      return op->literal().IsAll(value);
    default:
      return false;
  }
}

// Returns whether the given transpose produces a result which is bit-wise
// identical to its operand and thus may be replaced with a bitcast.
bool TransposeIsBitcast(const HloInstruction* transpose) {
  CHECK_EQ(HloOpcode::kTranspose, transpose->opcode());
  const HloInstruction* operand = transpose->operand(0);
  return ShapeUtil::TransposeIsBitcast(operand->shape(), transpose->shape(),
                                       transpose->dimensions());
}

// Returns true if the given reshape/copy produces a result which is bit-wise
// identical to its operand and thus may be replaced with a bitcast.
//
// This function is conservative -- even if this function returns false, the
// reshape may still be a bitcast. For example, a reshape from [28x28] to [784].
bool ReshapeOrCopyIsBitcast(
    const HloInstruction* instr,
    const AlgebraicSimplifierOptions::ValidBitcastCallback&
        valid_bitcast_callback) {
  CHECK(HloOpcode::kReshape == instr->opcode() ||
        HloOpcode::kCopy == instr->opcode());

  const HloInstruction* operand = instr->operand(0);
  // Can't insert bitcasts if the compiler used a memory layout which isn't
  // compatible.
  return ShapeUtil::ReshapeIsBitcast(operand->shape(), instr->shape()) &&
         valid_bitcast_callback(operand->shape(), instr->shape());
}

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

  Status HandleAnd(HloInstruction* logical_and) override;

  Status HandleBitcast(HloInstruction* bitcast) override;

  Status HandleBitcastConvert(HloInstruction* bitcast) override;

  Status HandleBroadcast(HloInstruction* broadcast) override;

  Status HandleConcatenate(HloInstruction* concatenate) override;

  Status HandleConstant(HloInstruction* constant) override;

  Status HandleCopy(HloInstruction* copy) override;

  Status HandleConvert(HloInstruction* convert) override;

  Status HandleComplex(HloInstruction* complex) override;

  Status HandleReal(HloInstruction* real) override;

  Status HandleImag(HloInstruction* imag) override;

  Status HandleIota(HloInstruction* instruction) override;

  Status HandleConvolution(HloInstruction* convolution) override;

  Status HandleDivide(HloInstruction* divide) override;

  Status HandleDot(HloInstruction* dot) override;

  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;

  Status HandleLog(HloInstruction* log) override;

  Status HandleMultiply(HloInstruction* multiply) override;

  Status HandleNegate(HloInstruction* negate) override;

  Status HandleNot(HloInstruction* logical_not) override;

  Status HandleOr(HloInstruction* logical_or) override;

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

  Status HandleSelect(HloInstruction* select) override;

  Status HandleSort(HloInstruction* sort) override;

  Status HandleTranspose(HloInstruction* transpose) override;

  Status HandleSubtract(HloInstruction* sub) override;

  Status HandleMap(HloInstruction* map) override;

  // Returns whether algebraic simplification has occurred.
  const bool changed() const { return changed_; }

  // Runs the visitor on a computation.
  static bool Run(HloComputation* computation,
                  const AlgebraicSimplifierOptions& options);

 private:
  explicit AlgebraicSimplifierVisitor(HloComputation* computation,
                                      const AlgebraicSimplifierOptions& options)
      : computation_(computation), options_(options) {}

  // Transforms Dots where at least one input is a vector or has a degenerate
  // dimension and converts it into a multiply and reduce. This should enable
  // more fusion than leaving the nodes as Dot operations.
  StatusOr<bool> HandleDotStrengthReduction(HloInstruction* dot);

  // Reshapes an instruction to rank 1 if it is not already rank 1.
  HloInstruction* Flatten(HloInstruction* hlo) {
    if (ShapeUtil::Rank(hlo->shape()) == 1) {
      return hlo;
    }
    return computation_->AddInstruction(HloInstruction::CreateReshape(
        ShapeUtil::MakeShape(hlo->shape().element_type(),
                             {ShapeUtil::ElementsIn(hlo->shape())}),
        hlo));
  }

  // Helper method to perform and add reduction in a single dimension.
  HloInstruction* AddReduce(HloInstruction* hlo, int64 dim) {
    HloInstruction* zero =
        computation_->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::Zero(hlo->shape().element_type()).Clone()));
    HloComputation* AddReduce_computation = GetOrCreateScalarAddComputation();
    Shape shape = ShapeUtil::DeleteDimension(dim, hlo->shape());
    return computation_->AddInstruction(HloInstruction::CreateReduce(
        shape, hlo, zero, {dim}, AddReduce_computation));
  }

  // Convenience method for replacing an instruction with a bitcast.
  void ReplaceWithBitcast(HloInstruction* instruction);

  // Replace old instruction with new instruction if old and new instructions
  // have the same shape. Updates uses and root instruction. Returns whether a
  // replacement was made.
  bool ReplaceInstructionIfSameShape(HloInstruction* old_instruction,
                                     HloInstruction* new_instruction);

  // Returns whether the shape of the output of the given instructions are the
  // same for the purposes of simplification. If options_.is_layout_sensitive()
  // is true, then this tests shape equality including layout
  // (ShapeUtil::Equal). If options_.is_layout_sensitive() is false, then the
  // tests shape compatibility (ShapeUtil::Compatible).
  bool SameShape(const HloInstruction* lhs, const HloInstruction* rhs) const;

  // Returns whether it was possible to transform `root` to a clamp instruction.
  // With min a minimum instruction, max a maximum instruction, min_operand a
  // operand of min and max_operand a operand of max.
  // Precondition: root is either a minimum or a maximum.
  bool TransformToClampIfSameShape(HloInstruction* root, HloInstruction* min,
                                   HloInstruction* min_operand,
                                   HloInstruction* operand, HloInstruction* max,
                                   HloInstruction* max_operand);

  // A Broadcast that feeds an element-wise operation with a unique non-scalar
  // operand can sink to after the operation.
  StatusOr<bool> TryToSinkBroadcastAfterOpWithUniqueNonScalarOperand(
      HloInstruction* broadcast);

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

  StatusOr<HloInstruction*> OptimizeDotOfConcat(HloInstruction* dot);
  StatusOr<HloInstruction*> OptimizeDotOfConcatHelper(
      const HloInstruction& dot, HloInstruction* lhs, int64 lhs_contracting_dim,
      HloInstruction* rhs, int64 rhs_contracting_dim, bool swapped);

  StatusOr<HloInstruction*> OptimizeDotOfGather(HloInstruction* dot);

  HloComputation* GetOrCreateScalarAddComputation() {
    if (scalar_add_computation_) {
      return scalar_add_computation_;
    }

    HloComputation::Builder b("scalar_add_computation");
    Shape shape = ShapeUtil::MakeShape(F32, {});
    auto scalar_lhs = b.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "scalar_lhs"));
    auto scalar_rhs = b.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "scalar_rhs"));
    auto scalar_op = b.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, scalar_lhs, scalar_rhs));
    scalar_add_computation_ =
        computation_->parent()->AddEmbeddedComputation(b.Build(scalar_op));
    return scalar_add_computation_;
  }

  // Tries to fold a kPad in the input or filter into the convolution
  // instruction's window.
  StatusOr<bool> FoldConvInputPad(HloInstruction* convolution);
  StatusOr<bool> FoldConvFilterPad(HloInstruction* convolution);

  // Tries to use a kDot in place of the given convolution.
  StatusOr<bool> SimplifyConvToDot(HloInstruction* convolution);

  // Tries to simplify a slice where the result of the slice is a scalar.
  StatusOr<bool> TrySimplifyScalarSlice(HloInstruction* slice);

  // Tries to convert slice(reshape(X)) into reshape(slice(X))
  StatusOr<bool> TryToReorderSliceAndReshape(HloInstruction* slice);

  // Current HloComputation instance the AlgebraicSimplifierVisitor is
  // traversing.
  HloComputation* computation_;

  // The backend-specific options selected for the algebraic simplifier.
  const AlgebraicSimplifierOptions& options_;

  // Whether algebraic simplification has occurred.
  bool changed_ = false;

  // Cached computation for adding two scalar F32.
  HloComputation* scalar_add_computation_ = nullptr;
};

}  // namespace

bool AlgebraicSimplifierVisitor::Run(
    HloComputation* computation, const AlgebraicSimplifierOptions& options) {
  AlgebraicSimplifierVisitor visitor(computation, options);
  TF_CHECK_OK(computation->Accept(&visitor));
  return visitor.changed_;
}

bool AlgebraicSimplifierVisitor::SameShape(const HloInstruction* lhs,
                                           const HloInstruction* rhs) const {
  if (options_.is_layout_sensitive()) {
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
  HloInstruction *lhs, *rhs;
  CHECK(Match(add, m::Add(m::Op(&lhs), m::Op(&rhs))));

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

  // Canonicalization: Put constants on the right.  This makes the reassociation
  // rules below simpler.
  VLOG(10) << "trying transform [Const + A => A + Const]";
  if (Match(add, m::Add(m::Constant(), m::NonConstant()))) {
    return ReplaceWithNewInstruction(
        add,
        HloInstruction::CreateBinary(add->shape(), HloOpcode::kAdd, rhs, lhs));
  }

  // Reassociate to allow constant folding.
  //
  // Note: This is not general.  For example, we won't reassociate
  //
  //   (A + C1) + (B + C2) =>  A + B + (C1 + C2).
  //
  VLOG(10) << "trying transform [(A + C1) + C2 => A + (C1 + C2)]";
  HloInstruction *a, *c1, *c2;
  if (Match(add, m::Add(m::Add(m::NonConstant(&a), m::Constant(&c1)),
                        m::Constant(&c2)))) {
    TF_ASSIGN_OR_RETURN(auto* sum_of_constants,
                        MakeBinaryHlo(HloOpcode::kAdd, c1, c2));
    return ReplaceWithNewInstruction(
        add, HloInstruction::CreateBinary(add->shape(), HloOpcode::kAdd, a,
                                          sum_of_constants));
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleAnd(HloInstruction* logical_and) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(logical_and, m::And(m::Op(&lhs), m::Op(&rhs))));
  // Simplify logical and
  if (ShapeUtil::HasPrimitiveType(lhs->shape(), xla::PRED) &&
      ShapeUtil::HasPrimitiveType(rhs->shape(), xla::PRED)) {
    // A && True => A
    VLOG(10) << "trying transform [A && True => A]: "
             << logical_and->ToString();
    if (IsAll(rhs, 1) && ReplaceInstructionIfSameShape(logical_and, lhs)) {
      return Status::OK();
    }
    // True && A => A
    VLOG(10) << "trying transform [True && A => A]: "
             << logical_and->ToString();
    if (IsAll(lhs, 1) && ReplaceInstructionIfSameShape(logical_and, rhs)) {
      return Status::OK();
    }

    // A && False => False
    VLOG(10) << "trying transform [A && False => False]: "
             << logical_and->ToString();
    if (IsAll(rhs, 0) && ReplaceInstructionIfSameShape(logical_and, rhs)) {
      return Status::OK();
    }

    // False && A => False
    VLOG(10) << "trying transform [False && A => False]: "
             << logical_and->ToString();
    if (IsAll(lhs, 0) && ReplaceInstructionIfSameShape(logical_and, lhs)) {
      return Status::OK();
    }
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleBitcast(HloInstruction* bitcast) {
  // If a bitcast feeds a bitcast, make it a single bitcast.
  HloInstruction* op;
  if (Match(bitcast, m::Bitcast(m::Bitcast(m::Op(&op))))) {
    return ReplaceWithNewInstruction(
        bitcast,
        HloInstruction::CreateUnary(bitcast->shape(), HloOpcode::kBitcast, op));
  }
  // All bitcasts can be eliminated (assuming layout constraints are
  // satisified).
  ReplaceInstructionIfSameShape(bitcast, bitcast->mutable_operand(0));
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleBitcastConvert(
    HloInstruction* bitcast) {
  // Eliminate bitcast converts between same shape.
  ReplaceInstructionIfSameShape(bitcast, bitcast->mutable_operand(0));
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleCopy(HloInstruction* copy) {
  // If a copy feeds a copy, make it a single copy.
  HloInstruction* op;
  if (Match(copy, m::Copy(m::Copy(m::Op(&op))))) {
    return ReplaceWithNewInstruction(
        copy, HloInstruction::CreateUnary(copy->shape(), HloOpcode::kCopy, op));
  }
  // All copies can be eliminated (assuming layout constraints are satisified).
  if (ReplaceInstructionIfSameShape(copy, copy->mutable_operand(0))) {
    return Status::OK();
  }

  if (options_.is_layout_sensitive() &&
      ReshapeOrCopyIsBitcast(copy, options_.valid_bitcast_callback())) {
    ReplaceWithBitcast(copy);
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleConcatenate(
    HloInstruction* concatenate) {
  absl::Span<HloInstruction* const> operands(concatenate->operands());
  if (operands.size() == 1) {
    // Unary concatenates are useless.
    ReplaceInstructionIfSameShape(concatenate, operands[0]);
    return Status::OK();
  }
  // Filter out and remove empty operands.
  std::vector<HloInstruction*> nonempty_operands;
  for (HloInstruction* operand : operands) {
    if (!ShapeUtil::IsZeroElementArray(operand->shape())) {
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
    bool is_effective_low_pad = Match(
        operands[0], m::Broadcast(m::Op().WithShape(m::Shape().IsScalar())));
    bool is_effective_high_pad = Match(
        operands[1], m::Broadcast(m::Op().WithShape(m::Shape().IsScalar())));
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
                                          const LiteralSlice& literal) {
  if (ShapeUtil::IsTuple(literal.shape())) {
    std::vector<HloInstruction*> elems;
    elems.reserve(ShapeUtil::TupleElementCount(literal.shape()));
    for (int i = 0; i < ShapeUtil::TupleElementCount(literal.shape()); ++i) {
      elems.push_back(
          BuildTupleConstant(computation, LiteralSlice(literal, {i})));
    }
    return computation->AddInstruction(HloInstruction::CreateTuple(elems));
  } else {
    return computation->AddInstruction(
        HloInstruction::CreateConstant(literal.Clone()));
  }
}

Status AlgebraicSimplifierVisitor::HandleConstant(HloInstruction* constant) {
  // Tuple constants aren't directly supported by any backend. Expand them into
  // explicit Tuple instructions.
  if (ShapeUtil::IsTuple(constant->shape())) {
    return ReplaceInstruction(
        constant, BuildTupleConstant(computation_, constant->literal()));
  }

  if (constant->shape().element_type() == TOKEN) {
    return Status::OK();
  }

  // If a literal is all the same element replace it with a scalar broadcast.
  if (ShapeUtil::ElementsIn(constant->shape()) > 1 &&
      constant->literal().IsAllFirst()) {
    Literal unique_scalar(
        LiteralUtil::GetFirstScalarLiteral(constant->literal()));
    HloInstruction* scalar = computation_->AddInstruction(
        HloInstruction::CreateConstant(std::move(unique_scalar)));
    return ReplaceWithNewInstruction(
        constant,
        HloInstruction::CreateBroadcast(constant->shape(), scalar, {}));
  }

  // If a literal is an increasing sequence from zero, replace it with an iota.
  if (ShapeUtil::Rank(constant->shape()) == 1 &&
      ShapeUtil::ElementsIn(constant->shape()) > 1 &&
      constant->literal().IsR1Iota()) {
    return ReplaceWithNewInstruction(
        constant, HloInstruction::CreateIota(constant->shape(), 0));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleSubtract(HloInstruction* sub) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(sub, m::Subtract(m::Op(&lhs), m::Op(&rhs))));
  // A - 0 => A
  VLOG(10) << "trying transform [A - 0 => A]: " << sub->ToString();
  if (IsAll(rhs, 0) && ReplaceInstructionIfSameShape(sub, lhs)) {
    return Status::OK();
  }

  // Canonicalize subtraction of a constant to addition.
  VLOG(10) << "trying transform [A - Const => A + (-Const)]";
  if (Match(sub, m::Subtract(m::NonConstant(&lhs), m::Constant(&rhs)))) {
    HloInstruction* negative_const = computation_->AddInstruction(
        HloInstruction::CreateUnary(rhs->shape(), HloOpcode::kNegate, rhs));
    return ReplaceWithNewInstruction(
        sub, HloInstruction::CreateBinary(sub->shape(), HloOpcode::kAdd, lhs,
                                          negative_const));
  }

  return Status::OK();
}
namespace {
template <typename T>
Status InvertConstant(const HloInstruction& constant, Literal* result) {
  return result->Populate<T>([&](absl::Span<const int64> indices) {
    return T{1.0} / constant.literal().Get<T>(indices);
  });
}
}  // namespace

Status AlgebraicSimplifierVisitor::HandleDivide(HloInstruction* divide) {
  Shape* shape;
  HloInstruction *a, *b, *c, *d;
  CHECK(Match(divide, m::Divide(m::Op(&a), m::Op(&b))));
  // A/1 => A
  VLOG(10) << "trying transform [A/1 => A]: " << divide->ToString();
  if (IsAll(b, 1) && ReplaceInstructionIfSameShape(divide, a)) {
    return Status::OK();
  }

  // exp(A)/exp(B) => exp(A-B)
  if (Match(divide, m::Divide(m::Exp(m::Op(&a)), m::Exp(m::Op(&b)))
                        .WithShape(m::Shape(&shape)))) {
    VLOG(10) << "transform [exp(A)/exp(B) => exp(A-B)]: " << divide->ToString();
    HloInstruction* subtract = computation_->AddInstruction(
        HloInstruction::CreateBinary(*shape, HloOpcode::kSubtract, a, b));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateUnary(*shape, HloOpcode::kExp, subtract));
  }

  // A/exp(B) => A*exp(-B)
  if (Match(divide, m::Divide(m::Op(&a), m::Exp(m::Op(&b))))) {
    VLOG(10) << "transform [A/exp(B) => A*exp(-B)]: " << divide->ToString();
    HloInstruction* negate = computation_->AddInstruction(
        HloInstruction::CreateUnary(divide->shape(), HloOpcode::kNegate, b));
    HloInstruction* new_exp = computation_->AddInstruction(
        HloInstruction::CreateUnary(divide->shape(), HloOpcode::kExp, negate));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(divide->shape(),
                                             HloOpcode::kMultiply, a, new_exp));
  }

  // A/pow(B,C) => A*pow(B,-C)
  if (Match(divide, m::Divide(m::Op(&a), m::Power(m::Op(&b), m::Op(&c))))) {
    VLOG(10) << "transform [A/pow(B,C) => A*pow(B,-C)]: " << divide->ToString();
    // The output shape of the created negate operator should be the same as the
    // input.
    const Shape& negate_shape = c->shape();
    HloInstruction* negate = computation_->AddInstruction(
        HloInstruction::CreateUnary(negate_shape, HloOpcode::kNegate, c));
    // And the power operator should retain the output shape of the old one.
    const Shape& new_power_shape = b->shape();
    HloInstruction* new_power =
        computation_->AddInstruction(HloInstruction::CreateBinary(
            new_power_shape, HloOpcode::kPower, b, negate));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(
                    divide->shape(), HloOpcode::kMultiply, a, new_power));
  }

  // Simplifying integral division would produce unexpected results.
  if (ShapeUtil::ElementIsIntegral(divide->shape())) {
    return Status::OK();
  }

  // A / Const => A * (1 / Const)
  //
  // (Backends can do this transformation, but generally only if the constant is
  // a scalar.)
  if (Match(divide, m::Divide(m::NonConstant(&a), m::Constant(&b)))) {
    Literal new_literal(b->shape());
    switch (b->shape().element_type()) {
      case F16:
        TF_RETURN_IF_ERROR(InvertConstant<half>(*b, &new_literal));
        break;
      case F32:
        TF_RETURN_IF_ERROR(InvertConstant<float>(*b, &new_literal));
        break;
      case BF16:
        TF_RETURN_IF_ERROR(InvertConstant<bfloat16>(*b, &new_literal));
        break;
      case F64:
        TF_RETURN_IF_ERROR(InvertConstant<double>(*b, &new_literal));
        break;
      case C64:
        TF_RETURN_IF_ERROR(InvertConstant<complex64>(*b, &new_literal));
        break;
      default:
        return Status::OK();
    }
    auto inverse = computation_->AddInstruction(
        HloInstruction::CreateConstant((new_literal.Clone())));
    TF_ASSIGN_OR_RETURN(auto new_divide,
                        MakeBinaryHlo(HloOpcode::kMultiply, a, inverse));
    return ReplaceInstruction(divide, new_divide);
  }

  // (A / B) / (C / D)  =>  (A / B)*(D / C) => (A * D) / (B * C)
  if (Match(divide, m::Divide(m::Divide(m::Op(&a), m::Op(&b)),
                              m::Divide(m::Op(&c), m::Op(&d))))) {
    TF_ASSIGN_OR_RETURN(auto a_times_d,
                        MakeBinaryHlo(HloOpcode::kMultiply, a, d));
    TF_ASSIGN_OR_RETURN(auto b_times_c,
                        MakeBinaryHlo(HloOpcode::kMultiply, b, c));
    TF_ASSIGN_OR_RETURN(auto new_divide, MakeBinaryHlo(HloOpcode::kDivide,
                                                       a_times_d, b_times_c));

    return ReplaceInstruction(divide, new_divide);
  }

  // (A / B) / C => A / (B * C)
  if (Match(divide, m::Divide(m::Divide(m::Op(&a), m::Op(&b)), m::Op(&c)))) {
    TF_ASSIGN_OR_RETURN(auto b_times_c,
                        MakeBinaryHlo(HloOpcode::kMultiply, b, c));
    TF_ASSIGN_OR_RETURN(auto new_divide,
                        MakeBinaryHlo(HloOpcode::kDivide, a, b_times_c));
    return ReplaceInstruction(divide, new_divide);
  }

  // A / (B / C) => (A*C) / B
  if (Match(divide, m::Divide(m::Op(&a), m::Divide(m::Op(&b), m::Op(&c))))) {
    TF_ASSIGN_OR_RETURN(auto a_times_c,
                        MakeBinaryHlo(HloOpcode::kMultiply, a, c));
    TF_ASSIGN_OR_RETURN(auto new_divide,
                        MakeBinaryHlo(HloOpcode::kDivide, a_times_c, b));
    return ReplaceInstruction(divide, new_divide);
  }

  return Status::OK();
}

StatusOr<bool> AlgebraicSimplifierVisitor::HandleDotStrengthReduction(
    HloInstruction* dot) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  int64 lhs_collapsing_dim =
      dot->dot_dimension_numbers().lhs_contracting_dimensions(0);
  if (lhs->IsRank2Transpose()) {
    lhs = lhs->mutable_operand(0);
    lhs_collapsing_dim = 1 - lhs_collapsing_dim;
  }
  const int64 lhs_kept_dim = 1 - lhs_collapsing_dim;

  int64 rhs_collapsing_dim =
      dot->dot_dimension_numbers().rhs_contracting_dimensions(0);
  if (rhs->IsRank2Transpose()) {
    rhs = rhs->mutable_operand(0);
    rhs_collapsing_dim = 1 - rhs_collapsing_dim;
  }
  const int64 rhs_kept_dim = 1 - rhs_collapsing_dim;

  auto as_type = [&](HloInstruction* hlo, const PrimitiveType element_type) {
    if (hlo->shape().element_type() == element_type) {
      return hlo;
    }
    return computation_->AddInstruction(HloInstruction::CreateConvert(
        ShapeUtil::ChangeElementType(hlo->shape(), element_type), hlo));
  };

  auto reshape_if_necessary = [&](HloInstruction* hlo) {
    hlo = as_type(hlo, dot->shape().element_type());
    if (!ShapeUtil::SameDimensions(hlo->shape(), dot->shape())) {
      hlo = computation_->AddInstruction(
          HloInstruction::CreateReshape(dot->shape(), hlo));
    }
    return hlo;
  };

  auto add_reduce_in_f32 = [&](HloInstruction* hlo, const int64 dim) {
    return AddReduce(as_type(hlo, F32), dim);
  };

  auto broadcast_to_dim = [&](HloInstruction* hlo, const Shape& shape,
                              int64 dim) {
    return computation_->AddInstruction(
        HloInstruction::CreateBroadcast(shape, hlo, {dim}));
  };

  auto multiply = [&](HloInstruction* local_lhs, HloInstruction* local_rhs) {
    return computation_->AddInstruction(HloInstruction::CreateBinary(
        local_lhs->shape(), HloOpcode::kMultiply, local_lhs, local_rhs));
  };

  // Strength reduce dot(a[K] , b[K]) =
  //  reshape(result.shape,
  //          reduce_sum(multiply(a, b), {0}))
  if (ShapeUtil::Rank(rhs->shape()) == 1 &&
      ShapeUtil::Rank(lhs->shape()) == 1) {
    TF_RETURN_IF_ERROR(
        ReplaceInstruction(dot, reshape_if_necessary(add_reduce_in_f32(
                                    multiply(Flatten(lhs), Flatten(rhs)), 0))));
    return true;
  }

  if (ShapeUtil::IsEffectiveScalar(rhs->shape()) &&
      ShapeUtil::IsEffectiveScalar(lhs->shape())) {
    TF_RETURN_IF_ERROR(ReplaceInstruction(
        dot, reshape_if_necessary(multiply(Flatten(lhs), Flatten(rhs)))));
    return true;
  }

  // Simplify outer product into multiply with implicit broadcasting.
  //
  // A dot(a[M, 1], b[1, N]) = multiply(a [M,1], b [1, N])
  if (ShapeUtil::Rank(rhs->shape()) == 2 &&
      rhs->shape().dimensions(rhs_collapsing_dim) == 1) {
    TF_RETURN_IF_ERROR(ReplaceInstruction(
        dot, multiply(broadcast_to_dim(Flatten(lhs), dot->shape(), 0),
                      broadcast_to_dim(Flatten(rhs), dot->shape(), 1))));
    return true;
  }

  // Strength reduce dot(a[1, K], b) =
  //    reshape(result.shape,
  //      reduce_sum(
  //        multiply(broadcast(reshape(a, [K]), {0}), b),
  //        {0})
  //      )
  //    )
  if (ShapeUtil::Rank(lhs->shape()) == 1 ||
      (ShapeUtil::Rank(lhs->shape()) == 2 &&
       lhs->shape().dimensions(lhs_kept_dim) == 1)) {
    if (ShapeUtil::Rank(rhs->shape()) == 1) {
      TF_RETURN_IF_ERROR(
          ReplaceInstruction(dot, reshape_if_necessary(add_reduce_in_f32(
                                      multiply(Flatten(lhs), rhs), 0))));
      return true;
    }
    TF_RETURN_IF_ERROR(ReplaceInstruction(
        dot, reshape_if_necessary(add_reduce_in_f32(
                 multiply(broadcast_to_dim(Flatten(lhs), rhs->shape(),
                                           rhs_collapsing_dim),
                          rhs),
                 rhs_collapsing_dim))));
    return true;
  }

  // Strength reduce dot(a, b[K, 1]) =
  //  reshape(result.shape,
  //    reduce_sum(multiply(a, broadcast(reshape([K],b), {1})), {0})
  //  )
  if (ShapeUtil::Rank(rhs->shape()) == 1 ||
      (ShapeUtil::Rank(rhs->shape()) == 2 &&
       rhs->shape().dimensions(rhs_kept_dim) == 1)) {
    TF_RETURN_IF_ERROR(ReplaceInstruction(
        dot, reshape_if_necessary(add_reduce_in_f32(
                 multiply(lhs, broadcast_to_dim(Flatten(rhs), lhs->shape(),
                                                lhs_collapsing_dim)),
                 lhs_collapsing_dim))));
    return true;
  }
  return false;
}

StatusOr<HloInstruction*> AlgebraicSimplifierVisitor::OptimizeDotOfConcat(
    HloInstruction* dot) {
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0) {
    return nullptr;
  }

  const int64 lhs_contracting_dim = dnums.lhs_contracting_dimensions(0);
  const int64 rhs_contracting_dim = dnums.rhs_contracting_dimensions(0);
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));

  TF_ASSIGN_OR_RETURN(
      HloInstruction * optimized_lhs_concat,
      OptimizeDotOfConcatHelper(*dot, lhs, lhs_contracting_dim, rhs,
                                rhs_contracting_dim, /*swapped=*/false));
  if (optimized_lhs_concat) {
    return optimized_lhs_concat;
  }

  return OptimizeDotOfConcatHelper(*dot, rhs, rhs_contracting_dim, lhs,
                                   lhs_contracting_dim, /*swapped=*/true);
}

StatusOr<HloInstruction*> AlgebraicSimplifierVisitor::OptimizeDotOfConcatHelper(
    const HloInstruction& dot, HloInstruction* lhs, int64 lhs_contracting_dim,
    HloInstruction* rhs, int64 rhs_contracting_dim, bool swapped) {
  bool can_optimize = lhs->opcode() == HloOpcode::kConcatenate &&
                      lhs->concatenate_dimension() == lhs_contracting_dim &&
                      rhs->opcode() == HloOpcode::kConstant;
  if (!can_optimize) {
    return nullptr;
  }

  // We're replacing this:
  //
  //   +-----+-----+-----+      +-------------------+
  //   |     |     |     |      |                   |
  //   |     |     |     |      |        R_0        |
  //   |     |     |     |      |                   |
  //   |     |     |     |      +-------------------+
  //   |     |     |     |      |                   |
  //   | L_0 | L_1 | L_2 |   *  |        R_1        |
  //   |     |     |     |      |                   |
  //   |     |     |     |      +-------------------+
  //   |     |     |     |      |                   |
  //   |     |     |     |      |        R_2        |
  //   |     |     |     |      |                   |
  //   +-----+-----+-----+      +-------------------+
  //
  // with this:
  //
  // [Sum over i]
  //
  //   +-----+     +-------------------+
  //   |     |     |                   |
  //   |     |  *  |        R_i        |
  //   |     |     |                   |
  //   |     |     +-------------------+
  //   |     |
  //   | L_i |
  //   |     |
  //   |     |
  //   |     |
  //   |     |
  //   |     |
  //   +-----+
  //
  // where the LHS is a concatenate operation (so we can "split" the LHS tensor
  // for free) and the RHS is a constant tensor (and thus can be split at
  // compile time).  In the future, we may also want to do this when both the
  // LHS and the RHS are concatenate operations that line up along the dimension
  // being contracted over.
  //
  // We should be able to generalize this transform to work on a non-constant
  // RHS when/if we have in-place slices or support input-fusing slices into
  // Dots.

  // Dimension numbers for the new dot instructions we'll create (L_i * R_i in
  // the diagram above).
  DotDimensionNumbers new_dot_dnums;
  new_dot_dnums.add_lhs_contracting_dimensions(swapped ? rhs_contracting_dim
                                                       : lhs_contracting_dim);
  new_dot_dnums.add_rhs_contracting_dimensions(swapped ? lhs_contracting_dim
                                                       : rhs_contracting_dim);

  // Here we use the MKN notation, where the contracted dimension has K
  // elements and the two non-contracted dimensions have M and N elements.
  HloInstruction* add_result = nullptr;
  int64 rhs_contracting_dim_offset = 0;
  int64 n = rhs->shape().dimensions(1 - rhs_contracting_dim);
  for (HloInstruction* concat_op : lhs->operands()) {
    int64 sub_k = concat_op->shape().dimensions(lhs_contracting_dim);
    Shape rhs_slice_shape(rhs->shape());
    rhs_slice_shape.set_dimensions(rhs_contracting_dim, sub_k);

    std::array<int64, 2> start_indices;
    start_indices[rhs_contracting_dim] = rhs_contracting_dim_offset;
    start_indices[1 - rhs_contracting_dim] = 0;

    std::array<int64, 2> limit_indices;
    limit_indices[rhs_contracting_dim] = rhs_contracting_dim_offset + sub_k;
    limit_indices[1 - rhs_contracting_dim] = n;

    HloInstruction* rhs_slice =
        computation_->AddInstruction(HloInstruction::CreateSlice(
            rhs_slice_shape, rhs, /*start_indices=*/start_indices,
            /*limit_indices=*/limit_indices, /*strides=*/{1, 1}));

    // TODO(b/69062148): We can get rid of `swapped` once all backends support
    // "non-canonical" contraction dimensions (that contracts dimension 1 of the
    // LHS with dimension 0 of the RHS).  But for now we keep the same
    // contraction dimensions as the incoming dot operation to ensure the new
    // dot operations can be lowered.
    HloInstruction *new_dot_lhs, *new_dot_rhs;
    if (swapped) {
      new_dot_lhs = rhs_slice;
      new_dot_rhs = concat_op;
    } else {
      new_dot_lhs = concat_op;
      new_dot_rhs = rhs_slice;
    }

    auto* new_dot = computation_->AddInstruction(
        HloInstruction::CreateDot(dot.shape(), new_dot_lhs, new_dot_rhs,
                                  new_dot_dnums, dot.precision_config()));

    if (add_result) {
      add_result = computation_->AddInstruction(HloInstruction::CreateBinary(
          dot.shape(), HloOpcode::kAdd, add_result, new_dot));
    } else {
      add_result = new_dot;
    }

    rhs_contracting_dim_offset += sub_k;
  }

  return add_result;
}

StatusOr<HloInstruction*> AlgebraicSimplifierVisitor::OptimizeDotOfGather(
    HloInstruction* dot) {
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0 ||
      dnums.rhs_batch_dimensions_size() != 0 ||
      dot->shape().dimensions_size() != 2) {  // dot output 2D
    VLOG(10) << "DotOfGather: Can only optimize 2D, non-batch dot operations.";
    return nullptr;
  }

  // Optimize either dot(DS(ctA), ctB)) or dot(ctB, DS(ctA)).
  // Currently a Gather is a DynamicSlice.
  auto is_dynamic_slice_constant_combination =
      [](HloInstruction* a, HloInstruction* b, int a_contracting_dimension) {
        // First operand is a DynamicSlice(Constant).
        if (a->opcode() != HloOpcode::kDynamicSlice) {
          return false;
        }
        auto* dynamic_slice_op = a->operand(0);
        if (dynamic_slice_op->opcode() != HloOpcode::kConstant) {
          return false;
        }
        // Second operand is a Constant.
        if (b->opcode() != HloOpcode::kConstant) {
          return false;
        }
        // The DynamicSlice output is a vector.
        const Shape& dynamic_slice_shape = a->shape();
        if (dynamic_slice_shape.dimensions(1 - a_contracting_dimension) != 1) {
          return false;
        }
        // Constant size is the same before and after slice in the contracting
        // dimension, otherwise we either must precompute for all possible slice
        // indices or dot is invalid.
        const Shape& dynamic_slice_op_shape = dynamic_slice_op->shape();
        if (dynamic_slice_op_shape.dimensions(a_contracting_dimension) !=
            dynamic_slice_shape.dimensions(a_contracting_dimension)) {
          return false;
        }
        return true;
      };

  HloInstruction* lhs = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);
  int lhs_contracting_dimension = dnums.lhs_contracting_dimensions(0);
  int rhs_contracting_dimension = dnums.rhs_contracting_dimensions(0);

  if (!is_dynamic_slice_constant_combination(
          lhs, rhs, /*a_contracting_dimension=*/lhs_contracting_dimension) &&
      !is_dynamic_slice_constant_combination(
          rhs, lhs, /*a_contracting_dimension=*/rhs_contracting_dimension)) {
    VLOG(10) << "DotOfGather: Can only optimize dot(DS(ctA), ctB)) or "
                "dot(ctB, DS(ctA)), where the two constants have equal "
                "contracting dimensions.";
    return nullptr;
  }

  // LHS is DynamicSlice:
  // input: dot(DS(ctA), ctB))
  // where DS(ctA) = DS({M x K}, {start, 0}, {1, K}) and ctB = {K x N}.
  // => input dimensions: dot({1 x K}, {K x N}) => {1 x N}.
  // output: DS(dot(ctA, ctB))
  // => output dimensions: DS ({M x N}, {start, 0}, {1, N}) => {1 x N}.

  // RHS is DynamicSlice:
  // input: dot(ctA, DS(ctB))
  // where ctA = {M x K} and DS(ctB) = DS({K x N}, {0, start}, {K, 1}).
  // => input dimensions: dot({M x K}, {K x 1}) => {M x 1}.
  // output: DS(dot(ctA, ctB))
  // => output dimensions: DS ({M x N}, {0, start}, {M, 1}) => {M x 1}.

  bool lhs_is_dynamic_slice = lhs->opcode() == HloOpcode::kDynamicSlice;

  // ctA:
  HloInstruction* left_operand =
      lhs_is_dynamic_slice ? lhs->mutable_operand(0) : lhs;
  // ctB:
  HloInstruction* right_operand =
      lhs_is_dynamic_slice ? rhs : rhs->mutable_operand(0);
  // Build ctA x ctB.
  const int m = left_operand->shape().dimensions(1 - lhs_contracting_dimension);
  const int n =
      right_operand->shape().dimensions(1 - rhs_contracting_dimension);
  auto memoized_shape =
      ShapeUtil::MakeShape(dot->shape().element_type(), {m, n});
  auto* memoized_inst = computation_->AddInstruction(
      HloInstruction::CreateDot(memoized_shape, left_operand, right_operand,
                                dnums, dot->precision_config()));
  // Get pair {start, 0} or {0, start}.
  HloInstruction* original_start_indices =
      lhs_is_dynamic_slice ? lhs->mutable_operand(1) : rhs->mutable_operand(1);
  // Position of start:
  int index_of_non_zero_start = lhs_is_dynamic_slice
                                    ? 1 - lhs_contracting_dimension
                                    : 1 - rhs_contracting_dimension;
  // Position of zero:
  int index_of_zero_start = 1 - index_of_non_zero_start;

  // Slice out start and 0 components and reorder if necessary.
  auto indices_type = original_start_indices->shape().element_type();
  Shape s_shape = ShapeUtil::MakeShape(indices_type, {1});
  Shape d_shape = ShapeUtil::MakeShape(indices_type, {2});
  HloInstruction* non_zero_start =
      computation_->AddInstruction(HloInstruction::CreateSlice(
          s_shape, original_start_indices, {index_of_non_zero_start},
          {index_of_non_zero_start + 1}, {1}));
  HloInstruction* zero_start =
      computation_->AddInstruction(HloInstruction::CreateSlice(
          s_shape, original_start_indices, {index_of_zero_start},
          {index_of_zero_start + 1}, {1}));
  HloInstruction* new_start_indices =
      lhs_is_dynamic_slice
          ? computation_->AddInstruction(HloInstruction::CreateConcatenate(
                d_shape, {non_zero_start, zero_start}, 0))
          : computation_->AddInstruction(HloInstruction::CreateConcatenate(
                d_shape, {zero_start, non_zero_start}, 0));

  // Build DynamicSlice(ctA x ctB).
  const int new_slice_m = lhs_is_dynamic_slice ? 1 : m;
  const int new_slice_n = lhs_is_dynamic_slice ? n : 1;
  auto* memoized_lookup =
      computation_->AddInstruction(HloInstruction::CreateDynamicSlice(
          dot->shape(), memoized_inst, new_start_indices,
          {new_slice_m, new_slice_n}));

  return memoized_lookup;
}

Status AlgebraicSimplifierVisitor::HandleDot(HloInstruction* dot) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));

  // Only optimize F32 or BF16 dot operations where the dot, rhs and lhs are
  // rank 2 or below.
  if ((dot->shape().element_type() != F32 &&
       dot->shape().element_type() != BF16) ||
      ShapeUtil::Rank(lhs->shape()) > 2 || ShapeUtil::Rank(rhs->shape()) > 2 ||
      ShapeUtil::Rank(dot->shape()) > 2) {
    return Status::OK();
  }

  // Replace a zero element dot with a broadcast of the constant 0.
  if (ShapeUtil::IsZeroElementArray(dot->shape()) ||
      ShapeUtil::IsZeroElementArray(lhs->shape()) ||
      ShapeUtil::IsZeroElementArray(rhs->shape())) {
    auto zero = computation_->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0(0.0f)));
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateBroadcast(dot->shape(), zero, {}));
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * dot_of_concat_optimized,
                      OptimizeDotOfConcat(dot));
  if (dot_of_concat_optimized) {
    VLOG(10) << "Replaced dot(concat(...), constant) with add(dot(..., "
                "constant)...)";
    return ReplaceInstruction(dot, dot_of_concat_optimized);
  }

  // Simplify dot(ConstA, Gather(Index, ConstB)) to:
  // Gather(Index, dot*(ConstA, ConstB)), where dot* is an appropriately
  // batched version of dot.
  TF_ASSIGN_OR_RETURN(HloInstruction * dot_of_gather_optimized,
                      OptimizeDotOfGather(dot));
  if (dot_of_gather_optimized) {
    VLOG(10) << "Replaced dot(constA, gather(i, constB)) with "
                "gather(i, dot*(constA, constB))";
    return ReplaceInstruction(dot, dot_of_gather_optimized);
  }

  if (options_.enable_dot_strength_reduction() &&
      !options_.is_layout_sensitive()) {
    TF_ASSIGN_OR_RETURN(bool did_strength_reduction,
                        HandleDotStrengthReduction(dot));
    if (did_strength_reduction) {
      return Status::OK();
    }
  }

  // Simplify dot(transpose(a), transpose(b)) to transpose(dot(b,a)).
  if (lhs->IsRank2Transpose() && rhs->IsRank2Transpose()) {
    DotDimensionNumbers dot_dimension_numbers;
    dot_dimension_numbers.add_lhs_contracting_dimensions(1);
    dot_dimension_numbers.add_rhs_contracting_dimensions(0);
    auto new_dot = computation_->AddInstruction(HloInstruction::CreateDot(
        ShapeUtil::PermuteDimensions({1, 0}, dot->shape()),
        rhs->mutable_operand(0), lhs->mutable_operand(0), dot_dimension_numbers,
        dot->precision_config()));
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateTranspose(dot->shape(), new_dot, {1, 0}));
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleMultiply(HloInstruction* multiply) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(multiply, m::Multiply(m::Op(&lhs), m::Op(&rhs))));
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

  // 0*A => 0. Only applies for integral types for correct NaN-handling.
  if (IsAll(lhs, 0) &&
      primitive_util::IsIntegralType(multiply->shape().element_type()) &&
      ReplaceInstructionIfSameShape(multiply, lhs)) {
    return Status::OK();
  }
  // A*0 => 0
  if (IsAll(rhs, 0) &&
      primitive_util::IsIntegralType(multiply->shape().element_type()) &&
      ReplaceInstructionIfSameShape(multiply, rhs)) {
    return Status::OK();
  }

  // exp(A) * exp(B) => exp(A+B)
  if (Match(multiply, m::Multiply(m::Exp(m::Op(&lhs)), m::Exp(m::Op(&rhs))))) {
    auto add = computation_->AddInstruction(HloInstruction::CreateBinary(
        multiply->shape(), HloOpcode::kAdd, lhs, rhs));
    return ReplaceWithNewInstruction(
        multiply,
        HloInstruction::CreateUnary(multiply->shape(), HloOpcode::kExp, add));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleNegate(HloInstruction* negate) {
  // negate(negate(x)) => x
  HloInstruction* x;
  if (Match(negate, m::Negate(m::Negate(m::Op(&x)))) &&
      ReplaceInstructionIfSameShape(negate, x)) {
    return Status::OK();
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleNot(HloInstruction* logical_not) {
  // not(not(x)) => x
  HloInstruction* x;
  if (Match(logical_not, m::Not(m::Not(m::Op(&x)))) &&
      ReplaceInstructionIfSameShape(logical_not, x)) {
    return Status::OK();
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleOr(HloInstruction* logical_or) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(logical_or, m::Or(m::Op(&lhs), m::Op(&rhs))));

  // Simplify logical or
  if (ShapeUtil::HasPrimitiveType(lhs->shape(), xla::PRED) &&
      ShapeUtil::HasPrimitiveType(rhs->shape(), xla::PRED)) {
    // A || True => True
    VLOG(10) << "trying transform [A || True => True]: "
             << logical_or->ToString();
    if (IsAll(rhs, 1) && ReplaceInstructionIfSameShape(logical_or, rhs)) {
      return Status::OK();
    }
    // True || A => True
    VLOG(10) << "trying transform [True || A => True]: "
             << logical_or->ToString();
    if (IsAll(lhs, 1) && ReplaceInstructionIfSameShape(logical_or, lhs)) {
      return Status::OK();
    }

    // A || False => A
    VLOG(10) << "trying transform [A || False => A]: "
             << logical_or->ToString();
    if (IsAll(rhs, 0) && ReplaceInstructionIfSameShape(logical_or, lhs)) {
      return Status::OK();
    }

    // False || A => A
    VLOG(10) << "trying transform [False || A => A]: "
             << logical_or->ToString();
    if (IsAll(lhs, 0) && ReplaceInstructionIfSameShape(logical_or, rhs)) {
      return Status::OK();
    }
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleLog(HloInstruction* log) {
  // ln(exp(A)) => A
  VLOG(10) << "trying transform [ln(exp(A)) => A]: " << log->ToString();
  HloInstruction *a, *b;
  if (Match(log, m::Log(m::Exp(m::Op(&a)))) &&
      ReplaceInstructionIfSameShape(log, a)) {
    return Status::OK();
  }

  // ln(pow(A,B)) => B*ln(A)
  if (Match(log, m::Log(m::Power(m::Op(&a), m::Op(&b))))) {
    auto new_log = computation_->AddInstruction(
        HloInstruction::CreateUnary(log->shape(), HloOpcode::kLog, a));
    return ReplaceWithNewInstruction(
        log, HloInstruction::CreateBinary(log->shape(), HloOpcode::kMultiply,
                                          new_log, b));
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
absl::optional<std::vector<int64>> ReshapeLeavesDimensionsUnmodified(
    const HloInstruction* hlo, absl::Span<const int64> input_dim_indices) {
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
      return absl::nullopt;
    }
    output_dim_indices.push_back(unmodified_dims[i].second);
  }
  return output_dim_indices;
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
    case HloOpcode::kTranspose:
      return true;
    case HloOpcode::kSort:
      return (!ShapeUtil::IsTuple(instruction->shape()));
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
  HloInstruction* operand;
  CHECK(Match(broadcast, m::Broadcast(m::Op(&operand))));
  auto dims = broadcast->dimensions();
  // A degenerate broadcast of a reshape that does not change the number of
  // elements can be replaced by a reshape.
  if (std::is_sorted(dims.begin(), dims.end()) &&
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
        broadcast,
        HloInstruction::CreateTranspose(broadcast->shape(), operand, dims));
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
      TryToSinkBroadcastAfterOpWithUniqueNonScalarOperand(broadcast));
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
    return Status::OK();
  }

  // broadcast(iota) -> iota.
  if (operand->opcode() == HloOpcode::kIota) {
    return ReplaceWithNewInstruction(
        broadcast,
        HloInstruction::CreateIota(
            broadcast->shape(),
            dims[Cast<HloIotaInstruction>(operand)->iota_dimension()]));
  }

  // Merge two consecutive broadcasts into a single one.
  if (operand->opcode() == HloOpcode::kBroadcast) {
    std::vector<int64> new_dimensions;
    for (auto dim : operand->dimensions()) {
      new_dimensions.push_back(dims[dim]);
    }
    return ReplaceWithNewInstruction(
        broadcast,
        HloInstruction::CreateBroadcast(
            broadcast->shape(), operand->mutable_operand(0), new_dimensions));
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

// Complex(Real(c), Imag(c)) -> c
Status AlgebraicSimplifierVisitor::HandleComplex(HloInstruction* complex) {
  HloInstruction *c0, *c1;
  if (Match(complex, m::Complex(m::Real(m::Op(&c0)), m::Imag(m::Op(&c1)))) &&
      c0 == c1) {
    return ReplaceInstruction(complex, c0);
  }
  return Status::OK();
}

// Real(Complex(r, i)) -> r
Status AlgebraicSimplifierVisitor::HandleReal(HloInstruction* real) {
  HloInstruction* op;
  if (Match(real, m::Real(m::Complex(m::Op(&op), m::Op())))) {
    return ReplaceInstruction(real, op);
  }
  return Status::OK();
}

// Imag(Complex(r, i)) -> i
Status AlgebraicSimplifierVisitor::HandleImag(HloInstruction* imag) {
  HloInstruction* op;
  if (Match(imag, m::Imag(m::Complex(m::Op(), m::Op(&op))))) {
    return ReplaceInstruction(imag, op);
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleIota(HloInstruction* instruction) {
  // iota -> zero if the iota dimension never produces an element other than
  // zero.
  auto* iota = Cast<HloIotaInstruction>(instruction);
  if (iota->shape().dimensions(iota->iota_dimension()) <= 1) {
    auto zero = computation_->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(iota->shape().element_type()).Clone()));
    return ReplaceWithNewInstruction(
        iota, HloInstruction::CreateBroadcast(iota->shape(), zero, {}));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandlePad(HloInstruction* pad) {
  if (ShapeUtil::IsZeroElementArray(pad->operand(0)->shape())) {
    return ReplaceWithNewInstruction(
        pad, HloInstruction::CreateBroadcast(pad->shape(),
                                             pad->mutable_operand(1), {}));
  }
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

    TF_ASSIGN_OR_RETURN(HloInstruction * nonzero_pad,
                        MakePadHlo(pad->mutable_operand(0),
                                   pad->mutable_operand(1), nonzero_padding));
    // Copy the layout from the original pad instructions. The new pad and the
    // slice instruction should all have the same layout.
    TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
        pad->shape(), nonzero_pad->mutable_shape()));

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
      int64 end = nonzero_pad->shape().dimensions(i);
      if (padding_dimension.edge_padding_high() < 0) {
        end += padding_dimension.edge_padding_high();
      }
      start_indices.push_back(start);
      end_indices.push_back(end);
      strides.push_back(1);
    }

    TF_ASSIGN_OR_RETURN(
        HloInstruction * slice,
        MakeSliceHlo(nonzero_pad, start_indices, end_indices, strides));

    // Verify that the slice shape matches the pad shape.
    TF_RET_CHECK(ShapeUtil::Compatible(slice->shape(), pad->shape()));

    return ReplaceInstruction(pad, slice);
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandlePower(HloInstruction* power) {
  VLOG(10) << "trying transform [pow(A, 0) => 1]: " << power->ToString();
  HloInstruction *lhs, *rhs;
  CHECK(Match(power, m::Power(m::Op(&lhs), m::Op(&rhs))));
  if (IsAll(rhs, 0)) {
    auto one = HloInstruction::CreateConstant(
        LiteralUtil::One(power->shape().element_type()).Clone());
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
  HloInstruction *a, *b;
  if (Match(power, m::Power(m::Exp(m::Op(&a)), m::Op(&b)))) {
    auto a_times_b = computation_->AddInstruction(HloInstruction::CreateBinary(
        power->shape(), HloOpcode::kMultiply, a, b));
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
        LiteralUtil::One(rhs->shape().element_type()).Clone()));

    // Explicitly broadcast scalar 1 to the output shape, to avoid implicit
    // broadcast in divide HLO as we are trying to eliminate implicit
    // broadcasting at HLO level.
    auto* broadcast_one = computation_->AddInstruction(
        HloInstruction::CreateBroadcast(power->shape(), one, {}));
    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateBinary(power->shape(), HloOpcode::kDivide,
                                            broadcast_one, lhs));
  }

  VLOG(10) << "trying transform [pow(pow(A, X), Y) => pow(A, X*Y)]: "
           << power->ToString();

  // Don't perform this optimization if either of the exponents is complex; this
  // identity is true only for real-valued exponents.  In addition, we cowardly
  // refuse to do this transformation if the two expontents have different
  // element types.
  if (lhs->opcode() == HloOpcode::kPower &&
      !ShapeUtil::ElementIsComplex(lhs->operand(1)->shape()) &&
      !ShapeUtil::ElementIsComplex(rhs->shape()) &&
      ShapeUtil::SameElementType(lhs->operand(1)->shape(), rhs->shape())) {
    auto exponent_product =
        computation_->AddInstruction(HloInstruction::CreateBinary(
            rhs->shape(), HloOpcode::kMultiply, lhs->mutable_operand(1), rhs));
    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateBinary(power->shape(), HloOpcode::kPower,
                                            lhs->mutable_operand(0),
                                            exponent_product));
  }

  return Status::OK();
}

StatusOr<bool>
AlgebraicSimplifierVisitor::TryToSinkBroadcastAfterOpWithUniqueNonScalarOperand(
    HloInstruction* broadcast) {
  TF_RET_CHECK(broadcast->opcode() == HloOpcode::kBroadcast);
  bool changed = false;
  if (ShapeUtil::IsScalar(broadcast->shape())) {
    return false;
  }
  HloInstruction* operand = broadcast->mutable_operand(0);
  for (HloInstruction* user : broadcast->users()) {
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

    // Find the unique non-scalar operand or continue if there isn't one.
    int64 scalar_broadcast_count = 0;
    int64 broadcast_use_count = 0;
    for (HloInstruction* user_operand : user->operands()) {
      if (user_operand->opcode() == HloOpcode::kBroadcast &&
          ShapeUtil::IsScalar(user_operand->operand(0)->shape())) {
        ++scalar_broadcast_count;
      } else if (broadcast == user_operand) {
        ++broadcast_use_count;
      }
    }
    if (scalar_broadcast_count + broadcast_use_count != user->operand_count()) {
      continue;
    }
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(user->operand_count());

    for (HloInstruction* user_operand : user->operands()) {
      if (user_operand->opcode() == HloOpcode::kBroadcast &&
          ShapeUtil::IsScalar(user_operand->operand(0)->shape())) {
        new_operands.push_back(
            computation_->AddInstruction(HloInstruction::CreateBroadcast(
                ShapeUtil::ChangeElementType(
                    operand->shape(), user_operand->shape().element_type()),
                user_operand->mutable_operand(0), {})));
      } else {
        CHECK_EQ(broadcast, user_operand);
        new_operands.push_back(operand);
      }
    }
    VLOG(4) << "Sinking broadcast after user:";
    VLOG(4) << "  old broadcast: " << broadcast->ToString();
    VLOG(4) << "  old user: " << user->ToString();
    HloInstruction* new_user =
        computation_->AddInstruction(user->CloneWithNewOperands(
            ShapeUtil::ChangeElementType(operand->shape(),
                                         user->shape().element_type()),
            new_operands));
    VLOG(4) << "  new user: " << new_user->ToString();
    HloInstruction* new_broadcast =
        computation_->AddInstruction(HloInstruction::CreateBroadcast(
            user->shape(), new_user, broadcast->dimensions()));
    VLOG(4) << "  new broadcast: " << new_broadcast->ToString();
    TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(new_broadcast));
    changed = true;
  }
  return changed;
}

Status AlgebraicSimplifierVisitor::HandleReshape(HloInstruction* reshape) {
  auto operand = reshape->mutable_operand(0);

  // Reshape directly to empty constant if the shape contains zero-element
  // dimension.
  if (ShapeUtil::IsZeroElementArray(reshape->shape())) {
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
  if (operand->opcode() == HloOpcode::kRng && operand->user_count() == 1) {
    *operand->mutable_shape() = reshape->shape();
    return ReplaceInstruction(reshape, operand);
  }

  if (HloOpcode::kBroadcast == reshape->operand(0)->opcode()) {
    auto opt_dims = ReshapeLeavesDimensionsUnmodified(
        reshape, reshape->operand(0)->dimensions());
    if (opt_dims.has_value()) {
      return ReplaceWithNewInstruction(
          reshape,
          HloInstruction::CreateBroadcast(
              reshape->shape(), reshape->mutable_operand(0)->mutable_operand(0),
              *opt_dims));
    }
  }

  // reshape(iota) -> iota.
  if (operand->opcode() == HloOpcode::kIota) {
    auto* iota = Cast<HloIotaInstruction>(operand);
    auto opt_dims =
        ReshapeLeavesDimensionsUnmodified(reshape, {iota->iota_dimension()});
    if (opt_dims.has_value()) {
      CHECK_EQ(opt_dims->size(), 1);
      return ReplaceWithNewInstruction(
          reshape,
          HloInstruction::CreateIota(reshape->shape(), opt_dims->front()));
    }
  }

  // Make this a bitcast if possible.
  if (options_.is_layout_sensitive() &&
      ReshapeOrCopyIsBitcast(reshape, options_.valid_bitcast_callback())) {
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

StatusOr<bool> AlgebraicSimplifierVisitor::TrySimplifyScalarSlice(
    HloInstruction* slice) {
  // Only try to do this for effective scalars. We could do the same for slicing
  // out larger pieces of padding (replacing with a broadcast of the padding
  // value), but this is probably not worth it.
  if (!ShapeUtil::IsEffectiveScalar(slice->shape())) {
    return false;
  }

  if (slice->operand(0)->opcode() == HloOpcode::kPad) {
    VLOG(10) << "Trying to simplify scalar slice of pad";
    // Check there's no internal padding. Again, we could handle that too, since
    // everything is statically known, but it's not worth it.
    auto pad = Cast<HloPadInstruction>(slice->mutable_operand(0));
    auto padding_config = pad->padding_config();
    int64 rank = padding_config.dimensions_size();
    if (HasInteriorPadding(padding_config)) {
      VLOG(10) << "Not folding scalar slice of pad, pad has interior padding";
      return false;
    }

    // Check whether the scalar we're slicing out falls into the padding.
    bool in_padding = [&]() {
      for (int64 i = 0; i < rank; ++i) {
        int64 start = slice->slice_starts(i);
        int64 low = padding_config.dimensions(i).edge_padding_low();
        int64 data = pad->operand(0)->shape().dimensions(i);
        if (start >= low && start < low + data) {
          return false;
        }
      }
      return true;
    }();

    if (in_padding) {
      VLOG(10) << "Folding scalar slice of pad into padding value";
      TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
          slice, HloInstruction::CreateReshape(slice->shape(),
                                               pad->mutable_padding_value())));
      return true;
    } else {
      // We already know the output of the slice is scalar. If the padded
      // value is scalar, and it's not in the padding, then it's exactly the
      // output value.
      bool replaced =
          ReplaceInstructionIfSameShape(slice, pad->mutable_operand(0));
      if (replaced) {
        VLOG(10) << "Folding scalar slice of pad into padded value";
      } else {
        VLOG(10) << "Not folding scalar slice of pad into padded value as they "
                    "have different shapes.";
      }
      return replaced;
    }
  }

  if (slice->operand(0)->opcode() == HloOpcode::kConcatenate) {
    VLOG(10) << "Trying to simplify scalar slice of concat";
    // Only do this for R1, there's no chance of this being useful otherwise.
    if (ShapeUtil::Rank(slice->shape()) != 1) {
      VLOG(10) << "Not folding, slice is not rank 1";
      return false;
    }
    HloConcatenateInstruction* concat =
        Cast<HloConcatenateInstruction>(slice->mutable_operand(0));
    int64 operand_start = 0;
    int64 operand_num = 0;
    // Weird loop structure to avoid annoying off-by-one errors.
    while (true) {
      TF_RET_CHECK(operand_num < concat->operand_count());
      const HloInstruction* operand = concat->operand(operand_num);
      int64 next_operand_start = operand_start + operand->shape().dimensions(0);
      if (next_operand_start > slice->slice_starts(0)) {
        break;
      }
      operand_start = next_operand_start;
      operand_num++;
    }

    bool replaced = ReplaceInstructionIfSameShape(
        slice, concat->mutable_operand(operand_num));
    if (replaced) {
      VLOG(10) << "Folding scalar slice of concat into concat operand";
    } else {
      VLOG(10) << "Folding scalar slice of concat into slice of concat operand";
      TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
          slice, HloInstruction::CreateSlice(
                     slice->shape(), concat->mutable_operand(operand_num),
                     {slice->slice_starts(0) - operand_start},
                     {slice->slice_starts(0) - operand_start + 1},
                     slice->slice_strides())));
    }
    return true;
  }

  return false;
}

bool IsUnstridedSlice(const HloInstruction* hlo) {
  return absl::c_all_of(hlo->slice_strides(),
                        [](int64 stride) { return stride == 1; });
}

StatusOr<bool> AlgebraicSimplifierVisitor::TryToReorderSliceAndReshape(
    HloInstruction* slice) {
  CHECK_EQ(slice->opcode(), HloOpcode::kSlice);
  if (!IsUnstridedSlice(slice)) {
    return false;
  }
  HloInstruction* reshape = slice->mutable_operand(0);
  if (reshape->opcode() != HloOpcode::kReshape) {
    return false;
  }
  HloInstruction* new_slice_operand = reshape->mutable_operand(0);
  int64 slice_rank = ShapeUtil::Rank(slice->shape());
  std::vector<int64> sliced_dims;
  for (int64 i = 0; i < slice_rank; ++i) {
    if (slice->slice_starts(i) != 0 ||
        slice->slice_limits(i) != reshape->shape().dimensions(i)) {
      sliced_dims.push_back(i);
    }
  }

  if (sliced_dims.size() == 1 && sliced_dims[0] == 0 &&
      slice->slice_starts(0) == 0) {
    const Shape& new_slice_shape = new_slice_operand->shape();
    const int64 rank = ShapeUtil::Rank(new_slice_shape);
    std::vector<int64> new_slice_starts(rank, 0);
    std::vector<int64> new_slice_stides(rank, 1);
    std::vector<int64> new_slice_limits(new_slice_shape.dimensions().begin(),
                                        new_slice_shape.dimensions().end());
    int64 slice_elements = ShapeUtil::ElementsIn(slice->shape());
    for (int64 i = rank - 1; i >= 0; --i) {
      if (slice_elements >= new_slice_limits[i]) {
        if (slice_elements % new_slice_limits[i] != 0) {
          return false;
        }
        slice_elements /= new_slice_limits[i];
      } else {
        new_slice_limits[i] = slice_elements;
        slice_elements = 1;
      }
    }
    HloInstruction* new_slice =
        computation_->AddInstruction(HloInstruction::CreateSlice(
            ShapeUtil::MakeShape(new_slice_shape.element_type(),
                                 new_slice_limits),
            new_slice_operand, new_slice_starts, new_slice_limits,
            new_slice_stides));
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
        slice, HloInstruction::CreateReshape(slice->shape(), new_slice)));
    return true;
  }
  return false;
}

Status AlgebraicSimplifierVisitor::HandleSlice(HloInstruction* slice) {
  // Delete no-op slices, i.e. where shape = operand shape.
  if (ReplaceInstructionIfSameShape(slice, slice->mutable_operand(0))) {
    return Status::OK();
  }

  if (slice->operand(0)->opcode() == HloOpcode::kSlice &&
      IsUnstridedSlice(slice) && IsUnstridedSlice(slice->operand(0))) {
    HloInstruction* operand_slice = slice->mutable_operand(0);
    std::vector<int64> new_slice_starts = slice->slice_starts();
    std::vector<int64> new_slice_limits = slice->slice_limits();
    for (int64 i = 0; i < new_slice_starts.size(); ++i) {
      new_slice_starts[i] += operand_slice->slice_starts(i);
      new_slice_limits[i] += operand_slice->slice_starts(i);
    }
    return ReplaceWithNewInstruction(
        slice, HloInstruction::CreateSlice(
                   slice->shape(), operand_slice->mutable_operand(0),
                   new_slice_starts, new_slice_limits, slice->slice_strides()));
  }

  TF_ASSIGN_OR_RETURN(bool replaced, TrySimplifyScalarSlice(slice));
  if (replaced) {
    return Status::OK();
  }

  TF_ASSIGN_OR_RETURN(replaced, TryToReorderSliceAndReshape(slice));
  if (replaced) {
    return Status::OK();
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleDynamicSlice(
    HloInstruction* dynamic_slice) {
  auto operand = dynamic_slice->mutable_operand(0);
  if (ShapeUtil::IsScalar(dynamic_slice->shape())) {
    return ReplaceInstruction(dynamic_slice, operand);
  }
  // DynamicSlice where operand has the same size as the output is simply equal
  // to operand.
  if (SameShape(operand, dynamic_slice)) {
    return ReplaceInstruction(dynamic_slice, operand);
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  auto update = dynamic_update_slice->mutable_operand(1);

  // DynamicUpdateSlice where operand and update have the same size is simply
  // equal to update.
  if (SameShape(dynamic_update_slice, update)) {
    return ReplaceInstruction(dynamic_update_slice, update);
  }

  // If any dimension of update is 0, elide the DynamicUpdateSlice.  This
  // optimization becomes invalid should we later prefer to warn about out of
  // bound indices.
  if (ShapeUtil::IsZeroElementArray(update->shape())) {
    return ReplaceInstruction(dynamic_update_slice,
                              dynamic_update_slice->mutable_operand(0));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleReduce(HloInstruction* reduce) {
  // TODO(b/112040122): Most of those optimizations can be done for multi-output
  // reduces.
  if (ShapeUtil::IsTuple(reduce->shape())) {
    return Status::OK();
  }

  auto arg = reduce->mutable_operand(0);
  auto init_value = reduce->mutable_operand(1);
  absl::Span<const int64> dimensions(reduce->dimensions());
  HloComputation* function = reduce->to_apply();
  if (ShapeUtil::IsZeroElementArray(arg->shape()) ||
      ShapeUtil::IsZeroElementArray(reduce->shape())) {
    return ReplaceWithNewInstruction(
        reduce,
        HloInstruction::CreateBroadcast(reduce->shape(), init_value, {}));
  }

  // A Transpose feeding a reduce can simply permute the reduction dimensions
  // field if the output of the reduce is a vector or scalar. Higher ranked
  // result may require a transpose of the output.
  if (ShapeUtil::Rank(reduce->shape()) <= 1 &&
      arg->opcode() == HloOpcode::kTranspose) {
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

  // If the reduction results in the same number of elements, then the only
  // possible side effect would be a reshape. Since the init_value is an
  // identity of the reduction function, we can therefore replace the reduce
  // with a simple reshape, ignoring the reduction function completely.
  if (ShapeUtil::ElementsIn(reduce->shape()) ==
      ShapeUtil::ElementsIn(arg->shape())) {
    return ReplaceWithNewInstruction(
        reduce, HloInstruction::CreateReshape(reduce->shape(), arg));
  }

  // If a reduce feeds a reduce with the same computation and initial value,
  // they can be combined into a single reduce.
  if (arg->opcode() == HloOpcode::kReduce &&
      init_value->Identical(*arg->operand(1)) &&
      *function == *arg->to_apply()) {
    // Create a new reduce with the combined reduction dimensions of both
    // reduces.
    std::vector<int64> arg_dims = arg->dimensions();
    std::sort(arg_dims.begin(), arg_dims.end());
    std::vector<int64> reduce_dims = reduce->dimensions();
    std::sort(reduce_dims.begin(), reduce_dims.end());
    // Transform reduce_dims to the same rank as the operand of the operand.
    for (int64 arg_dim : arg_dims) {
      for (int64& dim : reduce_dims) {
        if (dim >= arg_dim) {
          ++dim;
        }
      }
    }
    std::vector<int64> new_dimensions;
    new_dimensions.reserve(arg->dimensions().size() +
                           reduce->dimensions().size());
    std::merge(arg_dims.begin(), arg_dims.end(), reduce_dims.begin(),
               reduce_dims.end(), std::back_inserter(new_dimensions));
    return ReplaceWithNewInstruction(
        reduce,
        HloInstruction::CreateReduce(reduce->shape(), arg->mutable_operand(0),
                                     init_value, new_dimensions, function));
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
  // Convert Reduce(concat({a,b,...})) to
  //  map(reduce(a),map(reduce(b),...,))
  //
  // This should make fusion easier or use less memory bandwidth in the unfused
  // case.
  if (arg->opcode() == HloOpcode::kConcatenate &&
      absl::c_linear_search(reduce->dimensions(),
                            arg->concatenate_dimension())) {
    HloInstruction* old_reduce = nullptr;
    for (HloInstruction* operand : arg->operands()) {
      HloInstruction* new_reduce = computation_->AddInstruction(
          HloInstruction::CreateReduce(reduce->shape(), operand, init_value,
                                       reduce->dimensions(), function));
      if (old_reduce != nullptr) {
        new_reduce = computation_->AddInstruction(HloInstruction::CreateMap(
            reduce->shape(), {old_reduce, new_reduce}, function));
      }
      old_reduce = new_reduce;
    }
    return ReplaceInstruction(reduce, old_reduce);
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleReduceWindow(
    HloInstruction* reduce_window) {
  if (ShapeUtil::IsZeroElementArray(reduce_window->operand(0)->shape())) {
    return ReplaceWithNewInstruction(
        reduce_window,
        HloInstruction::CreateBroadcast(reduce_window->shape(),
                                        reduce_window->mutable_operand(1), {}));
  }
  auto operand = reduce_window->mutable_operand(0);
  const Window& window = reduce_window->window();
  auto function = reduce_window->to_apply();
  if (ShapeUtil::IsScalar(operand->shape())) {
    TF_RET_CHECK(ShapeUtil::IsScalar(reduce_window->shape()));
    return ReplaceWithNewInstruction(
        reduce_window,
        HloInstruction::CreateMap(reduce_window->shape(),
                                  {reduce_window->mutable_operand(1), operand},
                                  function));
  }

  // This optimization folds a pad op into reduce_window.
  HloInstruction* pad;
  const HloInstruction* convert = nullptr;
  if (operand->opcode() == HloOpcode::kPad) {
    pad = operand;
  } else if (operand->opcode() == HloOpcode::kConvert &&
             operand->operand(0)->opcode() == HloOpcode::kPad) {
    convert = operand;
    pad = operand->mutable_operand(0);
  } else {
    VLOG(10) << "Not folding pad into reduce-window as there is no pad.";
    return Status::OK();
  }

  // Bail on dilation.
  if (window_util::HasDilation(window)) {
    VLOG(10) << "Not folding pad into reduce-window as there is dilation.";
    return Status::OK();
  }

  VLOG(10) << "Considering folding Pad: " << pad->ToString()
           << "\ninto reduce-window: " << reduce_window->ToString()
           << (convert != nullptr
                   ? absl::StrCat("\nvia convert: ", convert->ToString())
                   : "");

  // Do not fold interior padding into ReduceWindow since the backends do not
  // support it.
  const PaddingConfig& pad_config = pad->padding_config();
  if (HasInteriorPadding(pad_config)) {
    VLOG(10) << "Not folding pad into reduce-window due to interior padding.";
    return Status::OK();
  }

  // If reduce_window already has padding, the pad value of the pad op and the
  // init value of reduce_window must match to allow folding the pad.
  const HloInstruction* pad_value = pad->operand(1);
  const HloInstruction* reduce_init_value = reduce_window->operand(1);
  if (pad_value != reduce_init_value) {
    auto literals_are_equivalent = [&] {
      auto& pad_literal = pad_value->literal();
      auto& reduce_init_literal = reduce_init_value->literal();
      if (pad_literal == reduce_init_literal) {
        return true;
      }
      auto converted_pad_literal =
          pad_literal.ConvertToShape(reduce_init_value->shape());
      if (!converted_pad_literal.ok()) {
        return false;
      }
      return converted_pad_literal.ValueOrDie() == reduce_init_literal;
    };
    // The pad value is usually a constant, so we handle that case and do not
    // try to get more fancy about proving equivalence in cases beyond that.
    if (pad_value->opcode() != HloOpcode::kConstant ||
        reduce_init_value->opcode() != HloOpcode::kConstant ||
        !literals_are_equivalent()) {
      VLOG(10) << "Not folding pad into reduce-window due to different pad "
                  "values.";
      return Status::OK();
    }
  }

  // If the pad puts a single non-identity value in each window that we're
  // reducing, then this is a broadcast.
  HloInstruction* pad_operand = pad->mutable_operand(0);
  auto is_effective_broadcast = [&] {
    if (window_util::HasStride(window)) {
      VLOG(10) << "Window has stride.";
      return false;
    }
    if (!window_util::HasSymmetricPadding(pad_config)) {
      VLOG(10) << "Window has uneven padding.";
      return false;
    }
    for (int64 i = 0; i < pad_config.dimensions_size(); ++i) {
      const auto& pad_dimension = pad_config.dimensions(i);
      if ((pad_dimension.edge_padding_low() != 0 ||
           pad_dimension.edge_padding_high() != 0) &&
          pad_operand->shape().dimensions(i) != 1) {
        VLOG(10) << "Found non-trivial dimension being padded: " << i;
        return false;
      }
    }
    VLOG(10) << "Found to be padding trivial dimensions only.";

    for (int64 i = 0; i < window.dimensions_size(); ++i) {
      const auto& pad_dimension = pad_config.dimensions(i);
      const WindowDimension& window_dimension = window.dimensions(i);
      bool dimension_has_padding = (pad_dimension.edge_padding_low() != 0 ||
                                    pad_dimension.edge_padding_high() != 0);
      if (dimension_has_padding &&
          window_dimension.size() < pad_dimension.edge_padding_low() + 1) {
        VLOG(10) << "Found window did not cover single unpadded element in "
                    "dimension: "
                 << i;
        return false;
      }
      if (pad_operand->shape().dimensions(i) != 1 &&
          window_dimension.size() != 1) {
        VLOG(10) << "Found window covers more than one element in non-trivial "
                    "dimension: "
                 << i;
        return false;
      }
    }
    VLOG(10) << "Found window covers a single unpadded element.";
    return true;
  };

  HloInstruction* new_reduce_window_operand;
  if (convert != nullptr) {
    new_reduce_window_operand =
        computation_->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(pad_operand->shape(),
                                         convert->shape().element_type()),
            pad_operand));
  } else {
    new_reduce_window_operand = pad_operand;
  }

  if (is_effective_broadcast()) {
    VLOG(10) << "Replacing pad/reduce-window with (implicit) broadcast.";
    auto fadd = [this](std::unique_ptr<HloInstruction> x) {
      return computation_->AddInstruction(std::move(x));
    };
    return ReplaceWithNewInstruction(
        reduce_window, HloInstruction::CreateBroadcastSequence(
                           /*output_shape=*/reduce_window->shape(),
                           /*operand=*/new_reduce_window_operand, fadd));
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
                         /*operand=*/new_reduce_window_operand,
                         /*init_value=*/reduce_window->mutable_operand(1),
                         /*window=*/new_window,
                         /*reduce_computation=*/function));
}

Status AlgebraicSimplifierVisitor::HandleSelect(HloInstruction* select) {
  // select(x, y, y) -> y.
  if (select->operand(1) == select->operand(2)) {
    return ReplaceInstruction(select, select->mutable_operand(1));
  }
  // select(true, x, y) -> x.
  if (IsAll(select->operand(0), true)) {
    return ReplaceInstruction(select, select->mutable_operand(1));
  }
  // select(false, x, y) -> y.
  if (IsAll(select->operand(0), false)) {
    return ReplaceInstruction(select, select->mutable_operand(2));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleSort(HloInstruction* sort) {
  auto operand = sort->mutable_operand(0);
  int64 dimension_to_sort = sort->dimensions(0);
  if (ShapeUtil::IsZeroElementArray(operand->shape()) ||
      operand->shape().dimensions(dimension_to_sort) <= 1) {
    if (sort->operand_count() == 1) {
      return ReplaceInstruction(sort, operand);
    }
    // If it is key/value sort, the output of sort is a tuple.
    return ReplaceWithNewInstruction(
        sort, HloInstruction::CreateTuple(sort->operands()));
  }
  if (!options_.enable_permutation_sort_replacement()) {
    return Status::OK();
  }
  // Check if we are sorting a permutation. In that case, we know that the keys
  // will be sorted to the identity permutation, and we can represent the
  // changes to the 'values' parameter as a scatter.
  if (sort->operand_count() == 2 &&
      operand->opcode() == HloOpcode::kGetTupleElement) {
    const HloInstruction* other_sort = operand->operand(0);
    // Check whether the 'values' parameter is the result of another sort with
    // the same sort dimension.
    if (other_sort->opcode() == HloOpcode::kSort &&
        other_sort->operand_count() >= 2 &&
        other_sort->dimensions(0) == dimension_to_sort &&
        other_sort->operand(operand->tuple_index())->opcode() ==
            HloOpcode::kIota) {
      auto* iota =
          Cast<HloIotaInstruction>(other_sort->operand(operand->tuple_index()));
      // The sort operand needs to be an integral iota, and the iota dimension
      // needs to be the dimension that was sorted.
      if (iota->iota_dimension() == dimension_to_sort &&
          ShapeUtil::ElementIsIntegral(iota->shape())) {
        // We use the following construction method for a Scatter that applies
        // the permutation from 'keys' to the 'values' parameter.
        // - Take the "keys" parameter of the second sort and reshape it to have
        //   another "1" dimension at the end.
        // - Concatenate it with iotas of the same extended shape with all
        //   different iota_dimensions except the dimension_to_sort in the order
        //   of iota_dimensions/dimension_to_sort, so e.g. with rank 3 and
        //   dimension_to_sort = 1, we would have concatenate of (iota with
        //   iota_dimension=0, keys, iota with iota_dimension = 2)
        // - Use this as the indices parameter of scatter, and set updates
        //   of the scatter to be a reshaped 'values' parameter of sort (adding
        //   'rank' many 1 dimensions at the end).
        int64 rank = ShapeUtil::Rank(operand->shape());
        Shape extended_shape = operand->shape();
        extended_shape.add_dimensions(1);
        extended_shape.mutable_layout()->add_minor_to_major(rank);
        auto reshaped_permutation = computation_->AddInstruction(
            HloInstruction::CreateReshape(extended_shape, operand));
        std::vector<HloInstruction*> concat_operands;
        for (int64 i = 0; i < rank; ++i) {
          if (i == dimension_to_sort) {
            concat_operands.push_back(reshaped_permutation);
          } else {
            concat_operands.push_back(computation_->AddInstruction(
                HloInstruction::CreateIota(extended_shape, i)));
          }
        }
        Shape concat_shape = operand->shape();
        concat_shape.add_dimensions(rank);
        concat_shape.mutable_layout()->add_minor_to_major(rank);
        auto scatter_indices =
            rank > 1 ? computation_->AddInstruction(
                           HloInstruction::CreateConcatenate(
                               concat_shape, concat_operands, rank))
                     : reshaped_permutation;

        // We don't care about the operand, it will be completely overridden by
        // the updates.
        auto scatter_operand = computation_->AddInstruction(
            HloInstruction::CreateIota(sort->operand(1)->shape(), 0));

        // Construct the updates operand of scatter.
        Shape update_shape = sort->operand(1)->shape();
        for (int64 i = 0; i < rank; ++i) {
          update_shape.add_dimensions(1);
          update_shape.mutable_layout()->add_minor_to_major(rank + i);
        }
        auto scatter_updates =
            computation_->AddInstruction(HloInstruction::CreateReshape(
                update_shape, sort->mutable_operand(1)));

        // Construct the updates computation, which simply replaces the operand
        // values with the update values.
        HloComputation::Builder b("update_replace_computation");
        Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
        b.AddInstruction(
            HloInstruction::CreateParameter(0, scalar_shape, "scalar_lhs"));
        auto scalar_rhs = b.AddInstruction(
            HloInstruction::CreateParameter(1, scalar_shape, "scalar_rhs"));
        auto update_replace_computation =
            computation_->parent()->AddEmbeddedComputation(b.Build(scalar_rhs));

        ScatterDimensionNumbers dim_numbers;
        dim_numbers.set_index_vector_dim(rank);
        for (int64 i = 0; i < rank; ++i) {
          dim_numbers.add_update_window_dims(rank + i);
          dim_numbers.add_scatter_dims_to_operand_dims(i);
        }
        auto scatter =
            computation_->AddInstruction(HloInstruction::CreateScatter(
                sort->operand(1)->shape(), scatter_operand, scatter_indices,
                scatter_updates, update_replace_computation, dim_numbers));
        return ReplaceWithNewInstruction(
            sort, HloInstruction::CreateTuple(
                      {computation_->AddInstruction(HloInstruction::CreateIota(
                           operand->shape(), dimension_to_sort)),
                       scatter}));
      }
    }
  }
  return Status::OK();
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

  if (operand->opcode() == HloOpcode::kRng && operand->user_count() == 1) {
    *operand->mutable_shape() = transpose->shape();
    return ReplaceInstruction(transpose, operand);
  }

  if (options_.is_layout_sensitive() && TransposeIsBitcast(transpose)) {
    ReplaceWithBitcast(transpose);
    return Status::OK();
  }

  return Status::OK();
}

StatusOr<bool> AlgebraicSimplifierVisitor::FoldConvInputPad(
    HloInstruction* convolution) {
  auto* lhs = convolution->mutable_operand(0);
  auto* rhs = convolution->mutable_operand(1);
  const auto& window = convolution->window();
  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();

  if (lhs->opcode() != HloOpcode::kPad) {
    return false;
  }

  // Convolution's padding is always zero, so bail if the kPad is adding
  // something other than zero.
  if (!IsAll(lhs->operand(1), 0)) {
    return false;
  }

  const auto& padding = lhs->padding_config();

  // Can't pad batch or feature dims.
  for (int64 dim :
       {dnums.input_batch_dimension(), dnums.input_feature_dimension()}) {
    const auto& p = padding.dimensions(dim);
    if (p.edge_padding_low() != 0 || p.edge_padding_high() != 0 ||
        p.interior_padding() != 0) {
      return false;
    }
  }

  // Compute the window which is the result of merging the kPad and the
  // convolution's existing window.
  Window new_window = window;
  for (int64 dim = 0; dim < dnums.input_spatial_dimensions_size(); ++dim) {
    auto& w = *new_window.mutable_dimensions(dim);
    const auto& p = padding.dimensions(dnums.input_spatial_dimensions(dim));
    // Edge padding composes with itself in the straightforward way, but
    // composing interior padding is nontrivial, and we cowardly refuse to
    // think about it. If we see interior padding in either the kPad or conv,
    // bail if there's any sort of padding in the other.
    if (p.interior_padding() != 0 &&
        (w.padding_low() != 0 || w.padding_high() != 0 ||
         w.base_dilation() != 1)) {
      return false;
    }
    if (w.base_dilation() != 1 &&
        (p.edge_padding_low() != 0 || p.edge_padding_high() != 0 ||
         p.interior_padding() != 0)) {
      return false;
    }

    w.set_padding_low(w.padding_low() + p.edge_padding_low());
    w.set_padding_high(w.padding_high() + p.edge_padding_high());
    if (p.interior_padding() != 0) {
      CHECK_EQ(w.base_dilation(), 1);
      w.set_base_dilation(1 + p.interior_padding());
    }
  }

  auto new_conv = convolution->CloneWithNewOperands(
      convolution->shape(), {lhs->mutable_operand(0), rhs});
  new_conv->set_window(new_window);
  TF_RETURN_IF_ERROR(
      ReplaceWithNewInstruction(convolution, std::move(new_conv)));
  return true;
}

StatusOr<bool> AlgebraicSimplifierVisitor::FoldConvFilterPad(
    HloInstruction* convolution) {
  auto* lhs = convolution->mutable_operand(0);
  auto* rhs = convolution->mutable_operand(1);
  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();

  if (rhs->opcode() != HloOpcode::kPad) {
    return false;
  }

  // Convolution's padding is always zero, so bail if the kPad is adding
  // something other than zero.
  if (!IsAll(rhs->operand(1), 0)) {
    return false;
  }

  const auto& padding = rhs->padding_config();

  // Can't pad or dilate feature dims.
  for (int64 dim : {dnums.kernel_input_feature_dimension(),
                    dnums.kernel_output_feature_dimension()}) {
    const auto& p = padding.dimensions(dim);
    if (p.edge_padding_low() != 0 || p.edge_padding_high() != 0 ||
        p.interior_padding() != 0) {
      return false;
    }
  }

  // Compute the window which is the result of merging the kPad and the
  // convolution's existing window.
  Window new_window = convolution->window();
  for (int64 dim = 0; dim < dnums.kernel_spatial_dimensions_size(); ++dim) {
    auto& w = *new_window.mutable_dimensions(dim);
    const auto& p = padding.dimensions(dnums.kernel_spatial_dimensions(dim));

    // We can only do this transformation if p adds dilation to the filter --
    // edge padding on the filter is not supported in conv.
    if (p.edge_padding_low() != 0 || p.edge_padding_high() != 0) {
      return false;
    }

    // Nothing to do if the kPad for this dim is entirely a nop.
    if (p.interior_padding() == 0) {
      continue;
    }

    // We cowardly refuse to think about how dilation composes with itself;
    // bail if both the kPad and conv have dilation on this dimension.
    if (w.window_dilation() > 1) {
      return false;
    }
    CHECK_EQ(w.window_dilation(), 1);
    w.set_window_dilation(1 + p.interior_padding());
    w.set_size(rhs->operand(0)->shape().dimensions(
        dnums.kernel_spatial_dimensions(dim)));
  }

  auto new_conv = convolution->CloneWithNewOperands(
      convolution->shape(), {lhs, rhs->mutable_operand(0)});
  new_conv->set_window(new_window);
  TF_RETURN_IF_ERROR(
      ReplaceWithNewInstruction(convolution, std::move(new_conv)));
  return true;
}

StatusOr<bool> AlgebraicSimplifierVisitor::SimplifyConvToDot(
    HloInstruction* convolution) {
  auto* lhs = convolution->mutable_operand(0);
  auto* rhs = convolution->mutable_operand(1);
  const auto& window = convolution->window();
  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();

  if (!options_.enable_conv_simplification()) {
    return false;
  }

  // TODO(b/31337498): For now, we cowardly refuse to do this optimization in
  // layout-insensitive mode, for fear of adding nontrivial reshapes.
  if (!options_.is_layout_sensitive()) {
    return false;
  }

  const Shape& input_shape = lhs->shape();
  const Shape& filter_shape = rhs->shape();
  const Shape& convolution_shape = convolution->shape();
  TF_RET_CHECK(LayoutUtil::HasLayout(input_shape));
  TF_RET_CHECK(LayoutUtil::HasLayout(filter_shape));
  TF_RET_CHECK(LayoutUtil::HasLayout(convolution_shape));

  // Require the spatial dimensions in the kernel to have a bound of one.
  for (int64 i = 0; i < dnums.kernel_spatial_dimensions_size(); ++i) {
    if (filter_shape.dimensions(dnums.kernel_spatial_dimensions(i)) != 1) {
      return false;
    }
  }

  // Stride ignores part of the output, which matrix multiplication does not do,
  // so require no stride. Padding and base (lhs) dilation both implicitly
  // extend the data, which matrix multiplication also does not do, so require
  // no padding and no base (lhs) dilation. Window (rhs) dilation has no effect
  // for a 1x1 window, so window dilation is no problem.
  if (window_util::HasStride(window) || window_util::HasPadding(window) ||
      window_util::HasBaseDilation(window)) {
    return false;
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
      LayoutUtil::Minor(input_shape.layout(), 0) !=
          dnums.input_feature_dimension() ||
      LayoutUtil::Minor(convolution_shape.layout(), 0) !=
          dnums.output_feature_dimension() ||
      // The input feature dimension should come later in the minor-to-major
      // order.
      (PositionInContainer(LayoutUtil::MinorToMajor(filter_shape),
                           dnums.kernel_input_feature_dimension()) <
       PositionInContainer(LayoutUtil::MinorToMajor(filter_shape),
                           dnums.kernel_output_feature_dimension()))) {
    return false;
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
  const Shape new_input_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      input_shape.element_type(), {conv_width, input_channels});
  // We already checked input_feature_dimension is more major than
  // output_feature_dimension, so data in filter_shape and row-major
  // {input_channels,output_channels} are bitwise identical.
  const Shape new_filter_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      filter_shape.element_type(), {input_channels, output_channels});
  const Shape dot_output_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      convolution_shape.element_type(), {conv_width, output_channels});

  // We cannot insert bitcasts if the layouts will not be compatible.
  // TODO(b/33178038): Consider inserting a transpose if a bitcast would be
  // invalid.
  if (!options_.valid_bitcast_callback()(input_shape, new_input_shape) ||
      !options_.valid_bitcast_callback()(filter_shape, new_filter_shape) ||
      !options_.valid_bitcast_callback()(dot_output_shape, convolution_shape)) {
    return false;
  }

  auto new_lhs = add_bitcast(new_input_shape, lhs);
  auto new_rhs = add_bitcast(new_filter_shape, rhs);
  DotDimensionNumbers dot_dimension_numbers;
  dot_dimension_numbers.add_lhs_contracting_dimensions(1);
  dot_dimension_numbers.add_rhs_contracting_dimensions(0);
  auto dot = computation_->AddInstruction(HloInstruction::CreateDot(
      dot_output_shape, new_lhs, new_rhs, dot_dimension_numbers,
      convolution->precision_config()));

  TF_RETURN_IF_ERROR(
      ReplaceInstruction(convolution, add_bitcast(convolution_shape, dot)));
  return true;
}

Status AlgebraicSimplifierVisitor::HandleConvolution(
    HloInstruction* convolution) {
  // Zero-sized input or filter.
  if (ShapeUtil::IsZeroElementArray(convolution->operand(0)->shape()) ||
      ShapeUtil::IsZeroElementArray(convolution->operand(1)->shape())) {
    return ReplaceWithNewInstruction(
        convolution,
        HloInstruction::CreateBroadcast(
            convolution->shape(),
            computation_->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::Zero(convolution->shape().element_type()))),
            {}));
  }

  // Try to merge padding/dilation of the input with the convolution's window.
  TF_ASSIGN_OR_RETURN(bool folded_input_pad, FoldConvInputPad(convolution));
  if (folded_input_pad) {
    return Status::OK();
  }

  // Try to merge dilation of the filter with the convolution's window.
  TF_ASSIGN_OR_RETURN(bool folded_filter_pad, FoldConvFilterPad(convolution));
  if (folded_filter_pad) {
    return Status::OK();
  }

  // Try to replace the convolution with a kDot instruction.
  TF_ASSIGN_OR_RETURN(bool replaced_with_dot, SimplifyConvToDot(convolution));
  if (replaced_with_dot) {
    return Status::OK();
  }

  return Status::OK();
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

Status AlgebraicSimplifierVisitor::HandleMap(HloInstruction* map) {
  auto* map_computation = map->to_apply();
  auto* map_root = map_computation->root_instruction();
  if (map_root->opcode() == HloOpcode::kParameter) {
    ReplaceInstructionIfSameShape(
        map, map->mutable_operand(map_root->parameter_number()));
    return Status::OK();
  }
  if (map_root->opcode() == HloOpcode::kConstant) {
    if (!ShapeUtil::IsScalar(map_root->shape())) {
      return Status::OK();
    }
    auto clone = map_root->CloneWithNewOperands(map_root->shape(), {});
    if (ShapeUtil::IsScalar(map->shape())) {
      return ReplaceWithNewInstruction(map, std::move(clone));
    }
    return ReplaceWithNewInstruction(
        map,
        HloInstruction::CreateBroadcast(
            map->shape(), computation_->AddInstruction(std::move(clone)), {}));
  }
  std::vector<HloInstruction*> new_operands;
  for (auto* root_operand : map_root->operands()) {
    if (root_operand->opcode() != HloOpcode::kParameter) {
      return Status::OK();
    }
    new_operands.push_back(
        map->mutable_operand(root_operand->parameter_number()));
  }
  auto clone = map_root->CloneWithNewOperands(map->shape(), new_operands);
  return ReplaceWithNewInstruction(map, std::move(clone));
}

StatusOr<bool> AlgebraicSimplifier::Run(HloModule* module) {
  XLA_VLOG_LINES(2,
                 "AlgebraicSimplifier::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (AlgebraicSimplifierVisitor::Run(comp, options_)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(2,
                 "AlgebraicSimplifier::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla

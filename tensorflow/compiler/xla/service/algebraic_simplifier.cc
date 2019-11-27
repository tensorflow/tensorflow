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
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/overflow_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/lib/statusor.h"

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

// Checks whether `op` is a floating-point constant or broadcast of a constant
// of the form +/- 2^k for some integer k positive, negative, or zero.  Such
// values are interesting because multiplying by a power of 2 just moves the
// exponent.
bool IsAllFpConstantPowerOf2(const HloInstruction* op) {
  // Unwrap the broadcast if necessary.
  const HloInstruction* c;
  if (!Match(op, m::ConstantEffectiveScalar(&c)) &&
      !Match(op, m::Broadcast(m::Constant(&c).WithShape(
                     m::Shape().IsEffectiveScalar())))) {
    return false;
  }
  auto val = [&]() -> absl::optional<double> {
    switch (c->shape().element_type()) {
      case BF16:
        return static_cast<double>(c->literal().GetFirstElement<bfloat16>());
      case F16:
        return static_cast<double>(c->literal().GetFirstElement<Eigen::half>());
      case F32:
        return c->literal().GetFirstElement<float>();
      case F64:
        return c->literal().GetFirstElement<double>();
      default:
        // Cowardly refuse to consider complex types.
        return absl::nullopt;
    }
  }();
  if (!val) {
    return false;
  }

  int exp;
  double mantissa = std::frexp(*val, &exp);
  // frexp returns a value in the range (-1, -0.5] U [0.5, 1).  A return value
  // of +/-0.5 therefore indicates that the floating point value is a power of
  // 2.
  return mantissa == 0.5 || mantissa == -0.5;
}

// Returns whether the given transpose produces a result which is bit-wise
// identical to its operand and thus may be replaced with a bitcast.
bool TransposeIsBitcast(const HloInstruction* transpose) {
  CHECK_EQ(HloOpcode::kTranspose, transpose->opcode());
  const HloInstruction* operand = transpose->operand(0);
  return ShapeUtil::TransposeIsBitcast(operand->shape(), transpose->shape(),
                                       transpose->dimensions());
}

// Recursive helper for method below.
HloInstruction* BitcastingOperandOfReshapeOrCopyChainHelper(
    HloInstruction* instr, HloInstruction* operand,
    const AlgebraicSimplifierOptions& options) {
  // Can't replace chain of copies and reshapes with bitcasts if the compiler
  // used a memory layout which isn't compatible.
  if (options.ReshapeIsBitcast(operand->shape(), instr->shape())) {
    return operand;
  }

  // If the operand is a copy or reshape try to see if the operand's operand
  // would produce a bitcast with initial instruction.
  if (HloOpcode::kReshape == operand->opcode() ||
      HloOpcode::kCopy == operand->opcode()) {
    return BitcastingOperandOfReshapeOrCopyChainHelper(
        instr, operand->mutable_operand(0), options);
  }
  return nullptr;
}

// Returns an operand of a chain of reshapes and copies that is bit-wise
// identical to first reshape or copy in the chain.
HloInstruction* BitcastingOperandOfReshapeOrCopyChain(
    HloInstruction* instr, const AlgebraicSimplifierOptions& options) {
  if (!options.is_layout_sensitive()) {
    return nullptr;
  }
  CHECK(HloOpcode::kReshape == instr->opcode() ||
        HloOpcode::kCopy == instr->opcode());
  return BitcastingOperandOfReshapeOrCopyChainHelper(
      instr, instr->mutable_operand(0), options);
}

bool IsUnstridedSlice(const HloInstruction* hlo) {
  return absl::c_all_of(hlo->slice_strides(),
                        [](int64 stride) { return stride == 1; });
}

// Returns bool to determine whether a pair of converts can be eliminated.
bool IsConvertPairNoOp(const HloInstruction* convert) {
  //    [operand_convert]         [convert]
  // (src)->convert-(intermediate)->convert-(dest)
  const HloInstruction* operand_convert = convert->operand(0);
  CHECK_EQ(operand_convert->opcode(), HloOpcode::kConvert);
  const Shape& src_shape = operand_convert->operand(0)->shape();
  const Shape& intermediate_shape = operand_convert->shape();
  const Shape& dest_shape = convert->shape();

  const PrimitiveType src_type = src_shape.element_type();
  const PrimitiveType intermediate_type = intermediate_shape.element_type();
  const PrimitiveType dest_type = dest_shape.element_type();

  // src_type must be equal to dest_type.
  if (src_type != dest_type) {
    return false;
  }

  // src_type must be a larger container than intermediate_type.
  if (ShapeUtil::ByteSizeOfPrimitiveType(intermediate_type) <=
      ShapeUtil::ByteSizeOfPrimitiveType(src_type)) {
    return false;
  }

  // Both src_type and intermediate_type must be either floating or integral.
  bool is_conversion_floating =
      ShapeUtil::ElementIsFloating(src_shape) &&
      ShapeUtil::ElementIsFloating(intermediate_shape);
  bool is_conversion_integral =
      ShapeUtil::ElementIsIntegral(src_shape) &&
      ShapeUtil::ElementIsIntegral(intermediate_shape);

  return is_conversion_floating || is_conversion_integral;
}

// AlgebraicSimplifierVisitor traverses the HLO computation and reduces certain
// algebraic expressions to simplified forms. Note: This only supports
// simplifications that simply look at the operands of an instruction. For the
// more general case a worklist based approach would be needed.
class AlgebraicSimplifierVisitor : public DfsHloRewriteVisitor {
 public:
  explicit AlgebraicSimplifierVisitor(const AlgebraicSimplifierOptions& options,
                                      AlgebraicSimplifier* simplifier)
      : options_(options), simplifier_(simplifier) {}

  Status HandleAdd(HloInstruction* add) override;

  Status HandleAnd(HloInstruction* logical_and) override;

  Status HandleBitcast(HloInstruction* bitcast) override;

  Status HandleBitcastConvert(HloInstruction* bitcast) override;

  Status HandleBroadcast(HloInstruction* broadcast) override;

  Status HandleCompare(HloInstruction* compare) override;

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

  Status HandleGather(HloInstruction* gather) override;

  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;

  Status HandleLog(HloInstruction* log) override;

  Status HandleMaximum(HloInstruction* maximum) override;

  Status HandleMinimum(HloInstruction* minimum) override;

  Status HandleClamp(HloInstruction* clamp) override;

  Status HandleMultiply(HloInstruction* multiply) override;

  Status HandleNegate(HloInstruction* negate) override;

  Status HandleNot(HloInstruction* logical_not) override;

  Status HandleOr(HloInstruction* logical_or) override;

  Status HandlePad(HloInstruction* pad) override;

  Status HandlePower(HloInstruction* power) override;

  Status HandleRemainder(HloInstruction* remainder) override;

  Status HandleReshape(HloInstruction* reshape) override;

  Status HandleReduce(HloInstruction* hlo) override;

  Status HandleReduceWindow(HloInstruction* reduce_window) override;

  Status HandleReverse(HloInstruction* reverse) override;
  Status HandleSlice(HloInstruction* slice) override;
  Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;
  Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;
  Status HandleScatter(HloInstruction* scatter) override;

  Status HandleSelect(HloInstruction* select) override;

  Status HandleSort(HloInstruction* sort) override;

  Status HandleTranspose(HloInstruction* transpose) override;

  Status HandleSubtract(HloInstruction* sub) override;

  Status HandleMap(HloInstruction* map) override;

  // Runs the visitor on a computation.
  bool Run(HloComputation* computation,
           const AlgebraicSimplifierOptions& options,
           AlgebraicSimplifier* simplifier);

 private:
  // Removes degenerate dimension from dot.
  StatusOr<bool> RemoveDegenerateDimensionFromDot(HloInstruction* dot);

  // Converts to primitive type if the input hlo is not that type, otherwise
  // returns the original hlo.
  HloInstruction* AsType(HloInstruction* hlo,
                         const PrimitiveType element_type) {
    if (hlo->shape().element_type() == element_type) {
      return hlo;
    }
    Shape changed_shape =
        ShapeUtil::ChangeElementType(hlo->shape(), element_type);
    simplifier_->UpdateLayout(&changed_shape);
    return computation_->AddInstruction(
        HloInstruction::CreateConvert(changed_shape, hlo));
  }

  // Transposes a dot operand such that the batch dimensions are the most major,
  // and the contracting dimensions are most minor.
  StatusOr<HloInstruction*> NormalizeDotOperandToBatchMajorAndContractingMinor(
      HloInstruction* dot_operand, absl::Span<const int64> batch_dimensions,
      absl::Span<const int64> contracting_dimensions) {
    std::vector<int64> transpose_dimensions(batch_dimensions.begin(),
                                            batch_dimensions.end());
    for (int64 i = 0; i < dot_operand->shape().rank(); ++i) {
      if (!(absl::c_linear_search(batch_dimensions, i) ||
            absl::c_linear_search(contracting_dimensions, i))) {
        transpose_dimensions.push_back(i);
      }
    }
    transpose_dimensions.insert(transpose_dimensions.end(),
                                contracting_dimensions.begin(),
                                contracting_dimensions.end());
    if (absl::c_is_sorted(transpose_dimensions)) {
      return dot_operand;
    }
    return MakeTransposeHlo(dot_operand, transpose_dimensions);
  }

  // Helper method to perform and add reduction on a list of dimensions.
  HloInstruction* AddReduce(HloInstruction* hlo, absl::Span<const int64> dims,
                            PrimitiveType type) {
    HloInstruction* zero = computation_->AddInstruction(
        simplifier_->CreateConstantWithLayoutUpdated(
            LiteralUtil::Zero(hlo->shape().element_type()).Clone()));
    HloComputation* AddReduce_computation =
        GetOrCreateScalarAddComputation(type);
    Shape shape = ShapeUtil::FilterDimensions(
        [&](int64 dim) { return !absl::c_linear_search(dims, dim); },
        hlo->shape());
    simplifier_->UpdateLayout(&shape);
    return computation_->AddInstruction(HloInstruction::CreateReduce(
        shape, hlo, zero, dims, AddReduce_computation));
  }

  // Convenience method for replacing an instruction with a bitcast. If operand
  // is not null, then the bitcast will use the specified operand instead of the
  // operand of the instruction.
  void ReplaceWithBitcast(HloInstruction* instruction,
                          HloInstruction* operand = nullptr);

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

  StatusOr<HloInstruction*> OptimizeDotOfConcat(HloInstruction* dot);
  StatusOr<HloInstruction*> OptimizeDotOfConcatHelper(
      const HloInstruction& dot, HloInstruction* lhs, int64 lhs_contracting_dim,
      HloInstruction* rhs, int64 rhs_contracting_dim, bool swapped);

  StatusOr<HloInstruction*> OptimizeDotOfGather(HloInstruction* dot);

  StatusOr<HloInstruction*> OptimizeDotOfReorderContractingDims(
      HloInstruction* dot);

  HloComputation* GetOrCreateScalarAddComputation(PrimitiveType type) {
    if (scalar_add_computation_) {
      return scalar_add_computation_;
    }

    HloComputation::Builder b("scalar_add_computation");
    Shape shape = ShapeUtil::MakeShape(type, {});
    simplifier_->UpdateLayout(&shape);
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

  // Useful when we want to use the same visitor over multiple computations.
  void ResetState(HloComputation* computation);

  // Current HloComputation instance the AlgebraicSimplifierVisitor is
  // traversing.
  HloComputation* computation_;

  // The backend-specific options selected for the algebraic simplifier.
  const AlgebraicSimplifierOptions& options_;

  // Whether algebraic simplification has occurred.
  bool changed_ = false;

  // Cached computation for adding two scalar F32.
  HloComputation* scalar_add_computation_ = nullptr;

  AlgebraicSimplifier* simplifier_ = nullptr;
};

}  // namespace

void AlgebraicSimplifierVisitor::ResetState(HloComputation* computation) {
  changed_ = false;
  ResetVisitStates();
  computation_ = computation;
}

bool AlgebraicSimplifierVisitor::Run(HloComputation* computation,
                                     const AlgebraicSimplifierOptions& options,
                                     AlgebraicSimplifier* simplifier) {
  ResetState(computation);
  TF_CHECK_OK(computation->Accept(this));
  return changed_ || changed();
}

bool AlgebraicSimplifierVisitor::SameShape(const HloInstruction* lhs,
                                           const HloInstruction* rhs) const {
  if (options_.is_layout_sensitive()) {
    return ShapeUtil::Equal(lhs->shape(), rhs->shape());
  } else {
    return ShapeUtil::Compatible(lhs->shape(), rhs->shape());
  }
}

void AlgebraicSimplifierVisitor::ReplaceWithBitcast(HloInstruction* instruction,
                                                    HloInstruction* operand) {
  CHECK_EQ(1, instruction->operand_count());
  if (operand == nullptr) {
    operand = instruction->mutable_operand(0);
  }
  CHECK_EQ(ShapeUtil::ElementsIn(instruction->shape()),
           ShapeUtil::ElementsIn(operand->shape()));
  CHECK_EQ(ShapeUtil::ByteSizeOf(instruction->shape()),
           ShapeUtil::ByteSizeOf(operand->shape()));

  auto bitcast = computation_->AddInstruction(
      HloInstruction::CreateBitcast(instruction->shape(), operand));
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
                        m::Constant(&c2))) ||
      Match(add, m::Add(m::Add(m::NonConstant(&a),
                               m::Broadcast(m::ConstantScalar(&c1))),
                        m::Broadcast(m::ConstantScalar(&c2))))) {
    TF_ASSIGN_OR_RETURN(auto* sum_of_constants,
                        MakeBinaryHlo(HloOpcode::kAdd, c1, c2));
    if (ShapeUtil::IsScalar(sum_of_constants->shape()) &&
        !ShapeUtil::IsScalar(add->shape())) {
      sum_of_constants = computation_->AddInstruction(
          HloInstruction::CreateBroadcast(add->shape(), sum_of_constants, {}));
    }
    return ReplaceWithNewInstruction(
        add, HloInstruction::CreateBinary(add->shape(), HloOpcode::kAdd, a,
                                          sum_of_constants));
  }

  // Convert add with fullshape into add with partial shape when a
  // portion of add is effective:
  //             zero (fullshape)   rhs (partialshape)
  // .           |                  |
  // . lhs .    dynamic_update_slice (fullshape)
  // . |         |
  // Add (fullshape)
  //
  // to:
  //              lhs
  //              |
  //             dynamic_slice (partialshape)   rhs (partialshape)
  // .           |                      |
  // . lhs .    add (partial_shape)+----+
  // . |         |
  // dynamic_update_slice (fullshape)
  //
  // This is pattern is discovered in control flow V2 gradient update.
  if (Match(add,
            m::Add(m::Op(&lhs),
                   m::Op(&rhs)
                       .WithOpcode(HloOpcode::kDynamicUpdateSlice)
                       .WithOperand(
                           0, m::Broadcast(m::ConstantEffectiveScalar(0)))))) {
    const Shape& partial_shape = rhs->operand(1)->shape();
    auto sliced_lhs =
        computation_->AddInstruction(HloInstruction::CreateDynamicSlice(
            partial_shape, lhs, absl::MakeSpan(rhs->operands()).subspan(2),
            partial_shape.dimensions()));

    auto add_partial = computation_->AddInstruction(
        HloInstruction::CreateBinary(rhs->operand(1)->shape(), HloOpcode::kAdd,
                                     sliced_lhs, rhs->mutable_operand(1)));

    auto dynamic_update_slice_full = HloInstruction::CreateDynamicUpdateSlice(
        lhs->shape(), lhs, add_partial,
        absl::MakeSpan(rhs->operands()).subspan(2));

    return ReplaceWithNewInstruction(add, std::move(dynamic_update_slice_full));
  }

  // A*C + B*C => (A+B)*C
  //
  //  - If A, B, and C are integers, do this unconditionally. Proof of
  //    correctness: https://rise4fun.com/Alive/u9X.
  //
  //  - If A, B, and C are floating point, do this if C is a scalar constant or
  //    broadcast of scalar constant and is equal to +/- 2^k for some (possibly
  //    negative) integer k.
  //
  //    Multiplying by a power of 2 just moves the exponent, so our answer is
  //    exact modulo rounding of intermediate results so long as
  //
  //     - none of the three products has an exponent which underflows (so the
  //       result is 0 or denormal), and
  //     - none of the three products overflows to inf.
  //
  //    Proof: See algebraic_simplifier_proof_distributive_property.py.
  //
  //    We deem these differences in rounding, underflow, and overflow
  //    acceptable in the ML context.
  HloInstruction *b, *c;
  if (((Match(lhs, m::Multiply(m::Op(&a), m::Op(&c))) &&
        Match(rhs, m::MultiplyAnyOrder(m::Op().Is(c), m::Op(&b)))) ||
       (Match(lhs, m::Multiply(m::Op(&c), m::Op(&a))) &&
        Match(rhs, m::MultiplyAnyOrder(m::Op().Is(c), m::Op(&b))))) &&
      (ShapeUtil::ElementIsIntegral(add->shape()) ||
       IsAllFpConstantPowerOf2(c))) {
    return ReplaceWithNewInstruction(
        add, HloInstruction::CreateBinary(
                 add->shape(), HloOpcode::kMultiply,
                 computation_->AddInstruction(HloInstruction::CreateBinary(
                     add->shape(), HloOpcode::kAdd, a, b)),
                 c));
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
  }

  // A && False => False or A & 0 => 0
  VLOG(10) << "trying transform [A && False => False]: "
           << logical_and->ToString();
  if (IsAll(rhs, 0) && ReplaceInstructionIfSameShape(logical_and, rhs)) {
    return Status::OK();
  }

  // False && A => False or A & 0 => 0
  VLOG(10) << "trying transform [False && A => False]: "
           << logical_and->ToString();
  if (IsAll(lhs, 0) && ReplaceInstructionIfSameShape(logical_and, lhs)) {
    return Status::OK();
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleBitcast(HloInstruction* bitcast) {
  // If a bitcast feeds a bitcast, make it a single bitcast.
  HloInstruction* op;
  if (Match(bitcast, m::Bitcast(m::Bitcast(m::Op(&op))))) {
    return ReplaceWithNewInstruction(
        bitcast, HloInstruction::CreateBitcast(bitcast->shape(), op));
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

  if (HloInstruction* bitcast_operand =
          BitcastingOperandOfReshapeOrCopyChain(copy, options_)) {
    ReplaceWithBitcast(copy, bitcast_operand);
    return Status::OK();
  }

  // Replace Copy(Reshape()) with Reshape() if the Reshape is a logical bitcast.
  if (copy->operand(0)->opcode() == HloOpcode::kReshape &&
      copy->operand(0)->user_count() == 1 &&
      ShapeUtil::ReshapeIsBitcast(copy->operand(0)->shape(), copy->shape())) {
    return ReplaceWithNewInstruction(
        copy,
        copy->operand(0)->CloneWithNewOperands(
            copy->shape(), {copy->mutable_operand(0)->mutable_operand(0)}));
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
    return Status::OK();
  }

  // Check if we can merge "adjacent" slice operands which take slices from the
  // same other op. For simplicity we only merge unstrided slices.
  int64 concatenate_dimension = concatenate->concatenate_dimension();
  for (int64 i = 0; i < operands.size(); ++i) {
    if (operands[i]->opcode() != HloOpcode::kSlice ||
        !IsUnstridedSlice(operands[i])) {
      continue;
    }
    int64 slice_end = operands[i]->slice_limits(concatenate_dimension);
    HloInstruction* slice_operand = operands[i]->mutable_operand(0);
    int64 j = i + 1;
    while (j < operands.size() && operands[j]->opcode() == HloOpcode::kSlice &&
           IsUnstridedSlice(operands[j]) &&
           operands[j]->operand(0) == slice_operand &&
           operands[j]->slice_starts(concatenate_dimension) == slice_end) {
      // Check that all the slice_start values are the same in all other
      // dimensions. This implies that the slice_limit values are also the same,
      // because operands of concatenate need to have the same shape, and we
      // already checked that the slices are unstrided.
      bool same_other_starts = true;
      for (int64 k = 0; k < operands[j]->slice_starts().size(); ++k) {
        if (k == concatenate_dimension) {
          continue;
        }
        if (operands[i]->slice_starts(k) != operands[j]->slice_starts(k)) {
          same_other_starts = false;
          break;
        }
      }
      if (!same_other_starts) {
        break;
      }
      slice_end = operands[j]->slice_limits(concatenate_dimension);
      ++j;
    }
    if (j - i > 1) {
      Shape new_slice_shape = operands[i]->shape();
      new_slice_shape.set_dimensions(
          concatenate_dimension,
          slice_end - operands[i]->slice_starts(concatenate_dimension));
      simplifier_->UpdateLayout(&new_slice_shape);
      auto new_limit_indices = operands[i]->slice_limits();
      new_limit_indices[concatenate_dimension] = slice_end;
      auto new_slice_op =
          computation_->AddInstruction(HloInstruction::CreateSlice(
              new_slice_shape, slice_operand,
              /*start_indices=*/operands[i]->slice_starts(),
              /*limit_indices=*/new_limit_indices,
              /*strides=*/operands[i]->slice_strides()));
      std::vector<HloInstruction*> new_operands;
      for (int64 k = 0; k < i; ++k) {
        new_operands.push_back(operands[k]);
      }
      new_operands.push_back(new_slice_op);
      for (int64 k = j; k < operands.size(); ++k) {
        new_operands.push_back(operands[k]);
      }
      auto replacement =
          computation_->AddInstruction(concatenate->CloneWithNewOperands(
              concatenate->shape(), new_operands));
      ReplaceInstructionIfSameShape(concatenate, replacement);
      return Status::OK();
    }
  }

  if (operands.size() == 2) {
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
    for (int64 dim = 0; dim < operands[0]->shape().rank(); ++dim) {
      auto padding_config_dim = padding_config.add_dimensions();
      padding_config_dim->set_edge_padding_high(0);
      padding_config_dim->set_edge_padding_low(0);
      padding_config_dim->set_interior_padding(0);
      if (dim == concatenate_dimension) {
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
                                          const LiteralSlice& literal,
                                          AlgebraicSimplifier* simplifier) {
  if (literal.shape().IsTuple()) {
    std::vector<HloInstruction*> elems;
    elems.reserve(ShapeUtil::TupleElementCount(literal.shape()));
    for (int i = 0; i < ShapeUtil::TupleElementCount(literal.shape()); ++i) {
      elems.push_back(BuildTupleConstant(
          computation, LiteralSlice(literal, {i}), simplifier));
    }
    return computation->AddInstruction(HloInstruction::CreateTuple(elems));
  } else {
    return computation->AddInstruction(
        simplifier->CreateConstantWithLayoutUpdated(literal.Clone()));
  }
}

Status AlgebraicSimplifierVisitor::HandleConstant(HloInstruction* constant) {
  // Tuple constants aren't directly supported by any backend. Expand them into
  // explicit Tuple instructions.
  if (constant->shape().IsTuple()) {
    return ReplaceInstruction(
        constant,
        BuildTupleConstant(computation_, constant->literal(), simplifier_));
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
        simplifier_->CreateConstantWithLayoutUpdated(std::move(unique_scalar)));
    return ReplaceWithNewInstruction(
        constant,
        HloInstruction::CreateBroadcast(constant->shape(), scalar, {}));
  }

  // If a literal is an increasing sequence from zero, replace it with an iota.
  if (constant->shape().rank() == 1 &&
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
  if (Match(sub, m::Subtract(m::NonConstant(&lhs), m::Constant(&rhs))) ||
      Match(sub, m::Subtract(m::NonConstant(&lhs),
                             m::Broadcast(m::Constant(&rhs))))) {
    HloInstruction* negative_const = computation_->AddInstruction(
        HloInstruction::CreateUnary(rhs->shape(), HloOpcode::kNegate, rhs));
    if (const HloInstruction* broadcast =
            DynCast<HloBroadcastInstruction>(sub->operand(1))) {
      negative_const =
          computation_->AddInstruction(HloInstruction::CreateBroadcast(
              broadcast->shape(), negative_const, broadcast->dimensions()));
    }
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

template <typename T>
std::unique_ptr<HloInstruction> TryDivideToShift(
    HloInstruction* divide, HloComputation* computation,
    AlgebraicSimplifier* simplifier) {
  HloInstruction *a, *b, *c;
  CHECK(Match(divide, m::Divide(m::Op(&a), m::Op(&b))));

  if (ShapeUtil::ElementIsIntegral(divide->shape()) &&
      !Match(b, m::ConstantEffectiveScalar(&c)) &&
      !Match(b, m::Broadcast(m::ConstantEffectiveScalar(&c)))) {
    return nullptr;
  }

  if (ShapeUtil::ElementIsSigned(divide->shape())) {
    int64 b_value = c->literal().GetFirstElement<T>();
    if (b_value > 0 && IsPowerOfTwo(static_cast<uint64>(b_value))) {
      // Handle negative dividends by negating the result of the division.
      HloInstruction* zero_like_a = MakeScalarLike(a, 0);

      Shape changed_shape = ShapeUtil::ChangeElementType(a->shape(), PRED);
      simplifier->UpdateLayout(&changed_shape);
      auto* dividend_is_negative =
          computation->AddInstruction(HloInstruction::CreateCompare(
              changed_shape, a, zero_like_a, ComparisonDirection::kLt));

      auto* negated_dividend = computation->AddInstruction(
          HloInstruction::CreateUnary(a->shape(), HloOpcode::kNegate, a));

      auto* abs_dividend =
          computation->AddInstruction(HloInstruction::CreateTernary(
              a->shape(), HloOpcode::kSelect, dividend_is_negative,
              negated_dividend, a));

      auto* quotient = computation->AddInstruction(HloInstruction::CreateBinary(
          divide->shape(), HloOpcode::kShiftRightLogical, abs_dividend,
          MakeScalarLike(abs_dividend, tensorflow::Log2Floor64(b_value))));

      auto* neqated_quotient =
          computation->AddInstruction(HloInstruction::CreateUnary(
              quotient->shape(), HloOpcode::kNegate, quotient));

      return HloInstruction::CreateTernary(divide->shape(), HloOpcode::kSelect,
                                           dividend_is_negative,
                                           neqated_quotient, quotient);
    }
  } else {
    uint64 b_value = c->literal().GetFirstElement<T>();
    if (IsPowerOfTwo(b_value)) {
      return HloInstruction::CreateBinary(
          divide->shape(), HloOpcode::kShiftRightLogical, a,
          MakeScalarLike(a, tensorflow::Log2Floor64(b_value)));
    }
  }

  return nullptr;
}
}  // namespace

Status AlgebraicSimplifierVisitor::HandleDivide(HloInstruction* divide) {
  HloInstruction *a, *b, *c, *d;
  CHECK(Match(divide, m::Divide(m::Op(&a), m::Op(&b))));
  // A/1 => A
  VLOG(10) << "trying transform [A/1 => A]: " << divide->ToString();
  if (IsAll(b, 1) && ReplaceInstructionIfSameShape(divide, a)) {
    return Status::OK();
  }

  // A / B => A >> log2(B) if B is a power of 2.
  switch (divide->shape().element_type()) {
    case S8:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<int8>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case S16:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<int16>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case S32:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<int32>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case S64:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<int64>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case U8:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<uint8>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case U16:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<uint16>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case U32:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<uint32>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case U64:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<uint64>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    default:
      break;
  }

  Shape* shape;
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

  // A/sqrt(B) => A*rsqrt(X).
  if (Match(divide, m::Divide(m::Op(&a), m::Sqrt(m::Op(&b))))) {
    auto* rsqrt = computation_->AddInstruction(
        HloInstruction::CreateUnary(divide->shape(), HloOpcode::kRsqrt, b));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(rsqrt->shape(),
                                             HloOpcode::kMultiply, a, rsqrt));
  }

  // A/rsqrt(B) => A*sqrt(B).
  if (Match(divide, m::Divide(m::Op(&a), m::Rsqrt(m::Op(&b))))) {
    auto* sqrt = computation_->AddInstruction(
        HloInstruction::CreateUnary(divide->shape(), HloOpcode::kSqrt, b));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(sqrt->shape(),
                                             HloOpcode::kMultiply, a, sqrt));
  }

  // Simplifying integral division would produce unexpected results.
  if (ShapeUtil::ElementIsIntegral(divide->shape())) {
    return Status::OK();
  }

  // A / Const => A * (1 / Const)
  //
  // (Backends can do this transformation, but generally only if the constant is
  // a scalar.)
  if (Match(divide, m::Divide(m::NonConstant(&a), m::Op(&b))) &&
      (Match(b, m::Constant(&c)) || Match(b, m::Broadcast(m::Constant(&c))))) {
    Shape result_shape = c->literal().shape();
    Literal new_literal(result_shape);
    switch (result_shape.element_type()) {
      case F16:
        TF_RETURN_IF_ERROR(InvertConstant<half>(*c, &new_literal));
        break;
      case F32:
        TF_RETURN_IF_ERROR(InvertConstant<float>(*c, &new_literal));
        break;
      case BF16:
        TF_RETURN_IF_ERROR(InvertConstant<bfloat16>(*c, &new_literal));
        break;
      case F64:
        TF_RETURN_IF_ERROR(InvertConstant<double>(*c, &new_literal));
        break;
      case C64:
        TF_RETURN_IF_ERROR(InvertConstant<complex64>(*c, &new_literal));
        break;
      case C128:
        TF_RETURN_IF_ERROR(InvertConstant<complex128>(*c, &new_literal));
        break;
      default:
        return Status::OK();
    }
    auto inverse = computation_->AddInstruction(
        simplifier_->CreateConstantWithLayoutUpdated(new_literal.Clone()));
    if (b != c) {
      inverse = computation_->AddInstruction(HloInstruction::CreateBroadcast(
          b->shape(), inverse, b->dimensions()));
    }
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

StatusOr<bool> AlgebraicSimplifierVisitor::RemoveDegenerateDimensionFromDot(
    HloInstruction* dot) {
  const Shape& lhs_shape = dot->operand(0)->shape();
  int64 num_degenerate_lhs_dims = 0;
  std::vector<int64> lhs_dimension_map(lhs_shape.rank(), -1);
  for (int64 i = 0; i < lhs_shape.rank(); ++i) {
    if (lhs_shape.dimensions(i) == 1) {
      ++num_degenerate_lhs_dims;
    } else {
      lhs_dimension_map[i] = i - num_degenerate_lhs_dims;
    }
  }

  const Shape& rhs_shape = dot->operand(1)->shape();
  int64 num_degenerate_rhs_dims = 0;
  std::vector<int64> rhs_dimension_map(rhs_shape.rank(), -1);
  for (int64 i = 0; i < rhs_shape.rank(); ++i) {
    if (rhs_shape.dimensions(i) == 1) {
      ++num_degenerate_rhs_dims;
    } else {
      rhs_dimension_map[i] = i - num_degenerate_rhs_dims;
    }
  }
  if (num_degenerate_lhs_dims == 0 && num_degenerate_rhs_dims == 0) {
    return false;
  }
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  DotDimensionNumbers new_dnums;
  for (int64 dim : dnums.lhs_batch_dimensions()) {
    int64 new_dim = lhs_dimension_map[dim];
    if (new_dim != -1) {
      new_dnums.add_lhs_batch_dimensions(new_dim);
    }
  }
  for (int64 dim : dnums.lhs_contracting_dimensions()) {
    int64 new_dim = lhs_dimension_map[dim];
    if (new_dim != -1) {
      new_dnums.add_lhs_contracting_dimensions(new_dim);
    }
  }

  for (int64 dim : dnums.rhs_batch_dimensions()) {
    int64 new_dim = rhs_dimension_map[dim];
    if (new_dim != -1) {
      new_dnums.add_rhs_batch_dimensions(new_dim);
    }
  }
  for (int64 dim : dnums.rhs_contracting_dimensions()) {
    int64 new_dim = rhs_dimension_map[dim];
    if (new_dim != -1) {
      new_dnums.add_rhs_contracting_dimensions(new_dim);
    }
  }

  HloInstruction* new_lhs =
      num_degenerate_lhs_dims > 0
          ? dot->parent()->AddInstruction(HloInstruction::CreateReshape(
                ShapeUtil::DropDegenerateDimensions(lhs_shape),
                dot->mutable_operand(0)))
          : dot->mutable_operand(0);
  HloInstruction* new_rhs =
      num_degenerate_rhs_dims > 0
          ? dot->parent()->AddInstruction(HloInstruction::CreateReshape(
                ShapeUtil::DropDegenerateDimensions(rhs_shape),
                dot->mutable_operand(1)))
          : dot->mutable_operand(1);
  TF_ASSIGN_OR_RETURN(auto new_dot, MakeDotHlo(new_lhs, new_rhs, new_dnums,
                                               dot->precision_config()));
  if (ShapeUtil::Compatible(dot->shape(), new_dot->shape())) {
    TF_RETURN_IF_ERROR(ReplaceInstruction(dot, new_dot));
  } else {
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
        dot, HloInstruction::CreateReshape(dot->shape(), new_dot)));
  }
  return true;
}

StatusOr<HloInstruction*> AlgebraicSimplifierVisitor::OptimizeDotOfConcat(
    HloInstruction* dot) {
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0 ||
      dot->shape().dimensions_size() != 2) {  // dot output 2D
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
    simplifier_->UpdateLayout(&rhs_slice_shape);

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
  HloDynamicSliceInstruction* dynamic_slice =
      lhs_is_dynamic_slice ? Cast<HloDynamicSliceInstruction>(lhs)
                           : Cast<HloDynamicSliceInstruction>(rhs);

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
  simplifier_->UpdateLayout(&memoized_shape);
  auto* memoized_inst = computation_->AddInstruction(
      HloInstruction::CreateDot(memoized_shape, left_operand, right_operand,
                                dnums, dot->precision_config()));
  // Get pair {start, 0} or {0, start}.
  // Position of start:
  int index_of_non_zero_start = lhs_is_dynamic_slice
                                    ? 1 - lhs_contracting_dimension
                                    : 1 - rhs_contracting_dimension;
  // Position of zero:
  int index_of_zero_start = 1 - index_of_non_zero_start;

  // Slice out start and 0 components and reorder if necessary.
  auto indices_type = dynamic_slice->operand(1)->shape().element_type();
  Shape s_shape = ShapeUtil::MakeShape(indices_type, {1});
  simplifier_->UpdateLayout(&s_shape);
  Shape d_shape = ShapeUtil::MakeShape(indices_type, {2});
  simplifier_->UpdateLayout(&d_shape);
  HloInstruction* non_zero_start =
      dynamic_slice->mutable_operand(1 + index_of_non_zero_start);
  HloInstruction* zero_start =
      dynamic_slice->mutable_operand(1 + index_of_zero_start);
  std::vector<HloInstruction*> new_start_indices;
  if (lhs_is_dynamic_slice) {
    new_start_indices = {non_zero_start, zero_start};
  } else {
    new_start_indices = {zero_start, non_zero_start};
  }

  // Build DynamicSlice(ctA x ctB).
  const int new_slice_m = lhs_is_dynamic_slice ? 1 : m;
  const int new_slice_n = lhs_is_dynamic_slice ? n : 1;
  auto* memoized_lookup =
      computation_->AddInstruction(HloInstruction::CreateDynamicSlice(
          dot->shape(), memoized_inst, new_start_indices,
          {new_slice_m, new_slice_n}));

  return memoized_lookup;
}

// This function tries to transform
//   dot(reshape(transpose(A)), Const) to
//   dot(reshape(A), reshape(transpose(reshape(Const)))),
// so that the reshape and transpose on the Const side can be constant folded.
//
// The basic idea is that since the accumulation in the dot operation is
// associative, so as long as we permute the elements of the contracting
// dimensions on both sides of the dot in the same way, the result of the
// dot is not affected.
StatusOr<HloInstruction*>
AlgebraicSimplifierVisitor::OptimizeDotOfReorderContractingDims(
    HloInstruction* dot) {
  // This transformation assumes layout is not assigned yet.
  if (options_.is_layout_sensitive()) {
    return nullptr;
  }

  // Canonicalize dot(<constant>, rhs) to dot(rhs, <constant>) to make the
  // remainder of this function easier.
  auto dnums = dot->dot_dimension_numbers();
  auto lhs_contracting_dims = dnums.lhs_contracting_dimensions();
  auto rhs_contracting_dims = dnums.rhs_contracting_dimensions();
  auto* lhs = dot->mutable_operand(0);
  auto* rhs = dot->mutable_operand(1);
  if (dot->operand(0)->IsConstant()) {
    std::swap(lhs, rhs);
    std::swap(lhs_contracting_dims, rhs_contracting_dims);
  }

  // Require single contracting dim to make the implementation easier to
  // track contracting dims.
  if (dnums.lhs_contracting_dimensions_size() != 1) {
    return nullptr;
  }

  // Pattern match Dot(reshape(transpose(input), constant))
  HloInstruction* reshape;
  HloInstruction* transpose;
  HloInstruction* input;
  HloInstruction* constant;
  if (!Match(lhs,
             m::Reshape(&reshape, m::Transpose(&transpose, m::Op(&input)))) ||
      !Match(rhs, m::Constant(&constant))) {
    return nullptr;
  }

  // Check that reshape squishes some dims into one dim and that this one
  // dim is the dot's lhs contracting dim. The size of unmodified_dims should
  // be N - 1, where N is the rank of the reshape output. This means that the
  // reshape squishes some dims into one dim. lhs contracting dim should not
  // be in unmodified_dims. This means that the squishing target dim is the
  // lhs contracting dim.
  auto unmodified_dims = ShapeUtil::DimensionsUnmodifiedByReshape(
      reshape->operand(0)->shape(), reshape->shape());
  CHECK_EQ(lhs_contracting_dims.size(), 1);
  if ((unmodified_dims.size() != reshape->shape().rank() - 1) ||
      absl::c_any_of(unmodified_dims, [&](const std::pair<int64, int64>& p) {
        return p.second == lhs_contracting_dims[0];
      })) {
    return nullptr;
  }

  // Virtually pull the reshape into the dot so the dot operates on the
  // transpose, with "unsquished" lhs contracting dims.  The new contracting
  // dims are all of the dims that are modified by the reshape -- that is, every
  // dimension that's not in `unmodified_dims[i].first`.
  //
  // (We don't need to actually create a new dot instruction. We can just keep
  // track of lhs and lhs_contracting_dims.)
  absl::flat_hash_set<int64> unmodified_transpose_dims;
  for (const auto& pair : unmodified_dims) {
    unmodified_transpose_dims.insert(pair.first);
  }
  lhs_contracting_dims.Clear();
  for (int64 i = 0; i < transpose->shape().dimensions_size(); ++i) {
    if (!unmodified_transpose_dims.contains(i)) {
      lhs_contracting_dims.Add(i);
    }
  }
  // We require the "unsquished" lhs contracting dims to be consecutive.
  auto is_iota = [](absl::Span<const int64> dims) {
    return absl::c_adjacent_find(dims, [](const int64 a, const int64 b) {
             return (b != a + 1);
           }) == dims.end();
  };
  if (!is_iota(AsInt64Slice(lhs_contracting_dims))) {
    return nullptr;
  }
  lhs = lhs->mutable_operand(0);

  // Check that the transpose only permutes the contracting dims.
  const auto& transpose_dims = transpose->dimensions();
  for (int64 i = 0; i < transpose_dims.size(); ++i) {
    if (transpose_dims[i] != i &&
        !absl::c_linear_search(lhs_contracting_dims, i)) {
      return nullptr;
    }
  }
  // Virtually pull the transpose into the dot. Now the dot is equivalent to
  // a new dot with "permuted" lhs contracting dims.
  std::vector<int64> permutation;
  for (auto dim : lhs_contracting_dims) {
    permutation.push_back(transpose_dims[dim] - lhs_contracting_dims[0]);
  }
  CHECK(IsPermutation(permutation, permutation.size()));
  auto new_lhs_contracting_dims =
      ComposePermutations(AsInt64Slice(lhs_contracting_dims), permutation);
  lhs_contracting_dims.Clear();
  for (auto dim : new_lhs_contracting_dims) {
    lhs_contracting_dims.Add(dim);
  }
  lhs = lhs->mutable_operand(0);

  // All checks are passed at this point.
  //
  // Transform lhs. Remove the transpose and reshape by sorting the lhs
  // contracting dims and squishing them into a single one. We don't actually
  // squish the lhs_contracting_dims here because we still need the unsquished
  // contracting dims to invert reshape and transpose.
  absl::c_sort(lhs_contracting_dims);
  lhs = computation_->AddInstruction(
      HloInstruction::CreateReshape(reshape->shape(), lhs));

  // Transform rhs. Say the input HLO is:
  //
  //   t0 = f32[2, 2, 3] parameter(0)
  //   t1 = f32[2, 3, 2] transpose(t0) dimensions={0, 2, 1}
  //   t2 = f32[2, 6] reshape(t1)
  //   t3 = f32[6, 2] constant(...)
  //   dot = f32[2, 2] dot(t2, t3) lhs_contracting_dims={1},
  //                               rhs_contracting_dims={0}
  //
  // At this point in the function, we have decided that the second and third
  // dims of t0 can be switched to remove the transpose, and we have
  // "virtually decomposed" the input HLO to:
  //
  //   t0 = f32[2, 2, 3] parameter(0)
  //   t2' = f32[2, 6] reshape(t0)
  //   t3' = f32[6, 2] ops-to-be-filled ...
  //   dot = f32[2, 2] dot(t2', t3') lhs_contracting_dims={1},
  //                                 rhs_contracting_dims={0}
  //
  // The rest of this function is to fill in the ops of t3'. To do this, we
  // unsquish the contracting dimensions in t3 and then apply the inverse of
  // the transpose from t1.

  // Invert reshape.
  CHECK_EQ(rhs_contracting_dims.size(), 1);
  std::vector<int64> rhs_unsquished_shape_dims =
      SpanToVector(constant->shape().dimensions());
  auto it = rhs_unsquished_shape_dims.erase(rhs_unsquished_shape_dims.begin() +
                                            rhs_contracting_dims[0]);
  for (auto dim : lhs_contracting_dims) {
    it = rhs_unsquished_shape_dims.insert(it,
                                          transpose->shape().dimensions(dim));
    ++it;
  }
  HloInstruction* rhs_reshape =
      computation_->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(constant->shape().element_type(),
                               rhs_unsquished_shape_dims),
          constant));
  rhs = rhs_reshape;

  // Rhs reshape "unsquishes" the single rhs contracting dim into multiple dims.
  rhs_contracting_dims.Resize(lhs_contracting_dims.size(),
                              rhs_contracting_dims[0]);
  absl::c_iota(rhs_contracting_dims, rhs_contracting_dims[0]);

  // Invert transpose. First compute the shape.
  std::vector<int64> rhs_transpose_shape_dims =
      SpanToVector(rhs_reshape->shape().dimensions());
  it = rhs_transpose_shape_dims.erase(
      rhs_transpose_shape_dims.begin() + rhs_contracting_dims[0],
      rhs_transpose_shape_dims.begin() + rhs_contracting_dims[0] +
          rhs_contracting_dims.size());
  for (auto dim : lhs_contracting_dims) {
    it = rhs_transpose_shape_dims.insert(it, input->shape().dimensions(dim));
    ++it;
  }
  // Then compute the transpose dims.
  std::vector<int64> rhs_transpose_dims(rhs_reshape->shape().rank());
  absl::c_iota(rhs_transpose_dims, 0);
  it = rhs_transpose_dims.erase(
      rhs_transpose_dims.begin() + rhs_contracting_dims[0],
      rhs_transpose_dims.begin() + rhs_contracting_dims[0] +
          rhs_contracting_dims.size());
  auto inverse_lhs_transpose_dims = InversePermutation(transpose_dims);
  for (auto dim : lhs_contracting_dims) {
    it = rhs_transpose_dims.insert(it, inverse_lhs_transpose_dims[dim] -
                                           lhs_contracting_dims[0] +
                                           rhs_contracting_dims[0]);
    ++it;
  }
  HloInstruction* rhs_transpose =
      computation_->AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(constant->shape().element_type(),
                               rhs_transpose_shape_dims),
          rhs_reshape, rhs_transpose_dims));
  rhs = rhs_transpose;

  // Squish the multiple rhs contracting dims into a single one.
  rhs = computation_->AddInstruction(
      HloInstruction::CreateReshape(constant->shape(), rhs));

  // If we virtually swapped lhs and rhs, we need to swap it back before
  // creating new dot.
  if (dot->operand(0)->IsConstant()) {
    std::swap(lhs, rhs);
  }

  HloInstruction* new_dot =
      computation_->AddInstruction(HloInstruction::CreateDot(
          dot->shape(), lhs, rhs, dnums, dot->precision_config()));
  return new_dot;
}

Status AlgebraicSimplifierVisitor::HandleDot(HloInstruction* dot) {
  CHECK(computation_ == dot->parent());
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  if (options_.is_layout_sensitive()) {
    return Status::OK();
  }
  // Replace a zero element dot with a broadcast of the constant 0.
  if (ShapeUtil::IsZeroElementArray(dot->shape()) ||
      ShapeUtil::IsZeroElementArray(lhs->shape()) ||
      ShapeUtil::IsZeroElementArray(rhs->shape())) {
    auto zero = computation_->AddInstruction(
        simplifier_->CreateConstantWithLayoutUpdated(
            LiteralUtil::Zero(dot->shape().element_type())));
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateBroadcast(dot->shape(), zero, {}));
  }

  // If there are no contracting dimensions, a dot can be rewritten as
  // mul(broadcast(transpose(x)),broadcast(transpose(y)))
  if (options_.enable_dot_to_multiply_rewrite() &&
      dot->dot_dimension_numbers().lhs_contracting_dimensions_size() == 0) {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_lhs,
        NormalizeDotOperandToBatchMajorAndContractingMinor(
            lhs,
            AsInt64Slice(dot->dot_dimension_numbers().lhs_batch_dimensions()),
            AsInt64Slice(
                dot->dot_dimension_numbers().lhs_contracting_dimensions())));
    if (dot->shape().rank() != lhs->shape().rank()) {
      std::vector<int64> lhs_broadcast_dims(lhs->shape().rank());
      absl::c_iota(lhs_broadcast_dims, 0);
      new_lhs = computation_->AddInstruction(HloInstruction::CreateBroadcast(
          dot->shape(), new_lhs, lhs_broadcast_dims));
    }
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_rhs,
        NormalizeDotOperandToBatchMajorAndContractingMinor(
            rhs,
            AsInt64Slice(dot->dot_dimension_numbers().rhs_batch_dimensions()),
            AsInt64Slice(
                dot->dot_dimension_numbers().rhs_contracting_dimensions())));
    if (dot->shape().rank() != rhs->shape().rank()) {
      std::vector<int64> rhs_broadcast_dims(
          dot->dot_dimension_numbers().lhs_batch_dimensions_size());
      absl::c_iota(rhs_broadcast_dims, 0);
      for (int64 i = lhs->shape().rank(); i < dot->shape().rank(); ++i) {
        rhs_broadcast_dims.push_back(i);
      }
      new_rhs = computation_->AddInstruction(HloInstruction::CreateBroadcast(
          dot->shape(), new_rhs, rhs_broadcast_dims));
    }
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateBinary(dot->shape(), HloOpcode::kMultiply,
                                          new_lhs, new_rhs));
  }

  // If the lhs or rhs have only batch and contracting dimensions, a dot can be
  // rewritten as reduce(mul(broadcast(transpose(x)),broadcast(transpose(y))))
  if (options_.enable_dot_strength_reduction() &&
      ShapeUtil::ElementIsFloating(dot->shape()) &&
      ((dot->dot_dimension_numbers().lhs_batch_dimensions_size() +
            dot->dot_dimension_numbers().lhs_contracting_dimensions_size() ==
        lhs->shape().rank()) ||
       (dot->dot_dimension_numbers().rhs_contracting_dimensions_size() +
            dot->dot_dimension_numbers().rhs_batch_dimensions_size() ==
        rhs->shape().rank()))) {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_lhs,
        NormalizeDotOperandToBatchMajorAndContractingMinor(
            lhs,
            AsInt64Slice(dot->dot_dimension_numbers().lhs_batch_dimensions()),
            AsInt64Slice(
                dot->dot_dimension_numbers().lhs_contracting_dimensions())));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_rhs,
        NormalizeDotOperandToBatchMajorAndContractingMinor(
            rhs,
            AsInt64Slice(dot->dot_dimension_numbers().rhs_batch_dimensions()),
            AsInt64Slice(
                dot->dot_dimension_numbers().rhs_contracting_dimensions())));

    int64 lhs_outer_dims =
        lhs->shape().rank() -
        (dot->dot_dimension_numbers().lhs_batch_dimensions_size() +
         dot->dot_dimension_numbers().lhs_contracting_dimensions_size());
    int64 rhs_outer_dims =
        rhs->shape().rank() -
        (dot->dot_dimension_numbers().rhs_batch_dimensions_size() +
         dot->dot_dimension_numbers().rhs_contracting_dimensions_size());
    CHECK(lhs_outer_dims == 0 || rhs_outer_dims == 0);
    if (rhs_outer_dims > 0) {
      std::vector<int64> lhs_broadcast_dims(
          dot->dot_dimension_numbers().lhs_batch_dimensions_size());
      absl::c_iota(lhs_broadcast_dims, 0);
      lhs_broadcast_dims.resize(lhs->shape().rank());
      std::iota(lhs_broadcast_dims.begin() +
                    dot->dot_dimension_numbers().lhs_batch_dimensions_size(),
                lhs_broadcast_dims.end(),
                dot->dot_dimension_numbers().lhs_batch_dimensions_size() +
                    rhs_outer_dims);
      new_lhs = computation_->AddInstruction(HloInstruction::CreateBroadcast(
          new_rhs->shape(), new_lhs, lhs_broadcast_dims));
    } else if (lhs_outer_dims > 0) {
      std::vector<int64> rhs_broadcast_dims(
          dot->dot_dimension_numbers().rhs_batch_dimensions_size());
      absl::c_iota(rhs_broadcast_dims, 0);
      rhs_broadcast_dims.resize(rhs->shape().rank());
      std::iota(rhs_broadcast_dims.begin() +
                    dot->dot_dimension_numbers().rhs_batch_dimensions_size(),
                rhs_broadcast_dims.end(),
                dot->dot_dimension_numbers().rhs_batch_dimensions_size() +
                    lhs_outer_dims);
      new_rhs = computation_->AddInstruction(HloInstruction::CreateBroadcast(
          new_lhs->shape(), new_rhs, rhs_broadcast_dims));
    }

    TF_ASSIGN_OR_RETURN(HloInstruction * new_dot,
                        MakeBinaryHlo(HloOpcode::kMultiply, new_lhs, new_rhs));
    std::vector<int64> reduce_dims(
        dot->dot_dimension_numbers().lhs_contracting_dimensions_size());
    PrimitiveType dot_type = dot->shape().element_type() == F64 ? F64 : F32;
    new_dot = AsType(new_dot, dot_type);
    const int64 outer_dims = std::max(rhs_outer_dims, lhs_outer_dims);
    absl::c_iota(
        reduce_dims,
        outer_dims + dot->dot_dimension_numbers().lhs_batch_dimensions_size());
    new_dot = AddReduce(new_dot, reduce_dims, dot_type);
    new_dot = AsType(new_dot, dot->shape().element_type());
    return ReplaceInstruction(dot, new_dot);
  }

  // Simplify dot(reshape(transpose(A)), Const) to:
  // dot(reshape(A), reshape(transpose(reshape(Const)))), so that the reshape
  // and transpose on the Const side can be constant folded.
  TF_ASSIGN_OR_RETURN(HloInstruction * dot_of_reorder_optimized,
                      OptimizeDotOfReorderContractingDims(dot));
  if (dot_of_reorder_optimized) {
    VLOG(10) << " Replaced dot " << dot->ToString()
             << " with new dot operation: "
             << dot_of_reorder_optimized->ToString();
    return ReplaceInstruction(dot, dot_of_reorder_optimized);
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

  TF_ASSIGN_OR_RETURN(bool removed_degenerate_dimensions,
                      RemoveDegenerateDimensionFromDot(dot));
  if (removed_degenerate_dimensions) {
    return Status::OK();
  }

  // Simplify dot(transpose(a), transpose(b)) to transpose(dot(b,a)).
  if (dot->dot_dimension_numbers().lhs_batch_dimensions_size() == 0 &&
      dot->dot_dimension_numbers().lhs_contracting_dimensions_size() == 1 &&
      dot->dot_dimension_numbers().lhs_contracting_dimensions(0) == 1 &&
      dot->dot_dimension_numbers().rhs_contracting_dimensions(0) == 0 &&
      lhs->IsRank2Transpose() && rhs->IsRank2Transpose()) {
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

Status AlgebraicSimplifierVisitor::HandleGather(HloInstruction* gather) {
  const Shape& operand_shape = gather->operand(0)->shape();
  if (ShapeUtil::IsZeroElementArray(operand_shape)) {
    return ReplaceInstruction(gather, MakeScalarLike(gather, 0));
  }
  // If the operand of a gather is very small, it is easier to fuse a
  // sequence of selects.
  const Shape& index_shape = gather->operand(1)->shape();
  if (operand_shape.rank() == 1 &&
      operand_shape.dimensions(0) <= options_.very_small_gather_size() &&
      gather->gather_dimension_numbers().index_vector_dim() ==
          index_shape.rank() &&
      gather->gather_dimension_numbers().collapsed_slice_dims_size() == 1) {
    const int64 operand_elements = operand_shape.dimensions(0);
    auto get_value = [&](int64 i) {
      auto slice = computation_->AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(operand_shape.element_type(), {1}),
          gather->mutable_operand(0), {i}, {i + 1}, {1}));
      auto scalar = computation_->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(operand_shape.element_type(), {}), slice));
      return computation_->AddInstruction(
          HloInstruction::CreateBroadcast(gather->shape(), scalar, {}));
    };
    auto result = get_value(0);
    auto pred_shape = ShapeUtil::ChangeElementType(gather->shape(), PRED);
    auto iter_shape = ShapeUtil::ChangeElementType(gather->shape(),
                                                   index_shape.element_type());
    for (int64 i = 0; i < operand_elements; ++i) {
      auto index_mask =
          computation_->AddInstruction(HloInstruction::CreateCompare(
              pred_shape, gather->mutable_operand(1),
              MakeScalarLike(gather->mutable_operand(1), i),
              ComparisonDirection::kGe));
      result = computation_->AddInstruction(
          HloInstruction::CreateTernary(gather->shape(), HloOpcode::kSelect,
                                        index_mask, get_value(i), result));
    }
    return ReplaceInstruction(gather, result);
  }
  return Status::OK();
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> MinMaxToClamp(
    HloInstruction* clamp_lower_bound_bcast, HloInstruction* to_clamp,
    HloInstruction* clamp_upper_bound_bcast) {
  HloInstruction* clamp_lower_bound;
  CHECK(Match(clamp_lower_bound_bcast,
              m::Broadcast(m::ConstantEffectiveScalar(&clamp_lower_bound))))
      << clamp_lower_bound_bcast->ToString();

  HloInstruction* clamp_upper_bound;
  CHECK(Match(clamp_upper_bound_bcast,
              m::Broadcast(m::ConstantEffectiveScalar(&clamp_upper_bound))))
      << clamp_upper_bound_bcast->ToString();

  const Literal& lower_bound =
      Cast<HloConstantInstruction>(clamp_lower_bound)->literal();
  const Literal& upper_bound =
      Cast<HloConstantInstruction>(clamp_upper_bound)->literal();

  std::unique_ptr<HloInstruction> lower_bound_instr =
      HloInstruction::CreateConstant(lower_bound.Clone());
  std::unique_ptr<HloInstruction> upper_bound_instr =
      HloInstruction::CreateConstant(upper_bound.Clone());

  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateCompare(
          ShapeUtil::ChangeElementType(lower_bound_instr->shape(), PRED),
          lower_bound_instr.get(), upper_bound_instr.get(),
          ComparisonDirection::kLt);

  HloEvaluator evaluator;
  TF_ASSIGN_OR_RETURN(auto result,
                      evaluator.Evaluate(cloned_instruction.get()));
  if (result.IsAll(true)) {
    return HloInstruction::CreateTernary(to_clamp->shape(), HloOpcode::kClamp,
                                         clamp_lower_bound_bcast, to_clamp,
                                         clamp_upper_bound_bcast);
  }
  return std::unique_ptr<HloInstruction>();
}
}  // namespace

Status AlgebraicSimplifierVisitor::HandleMaximum(HloInstruction* maximum) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(maximum, m::Maximum(m::Op(&lhs), m::Op(&rhs))));

  HloInstruction* clamp_upper_bound_bcast;
  HloInstruction* clamp_lower_bound_bcast;
  HloInstruction* to_clamp;
  if (Match(maximum, m::MaximumAnyOrder(
                         m::Broadcast(&clamp_lower_bound_bcast,
                                      m::ConstantEffectiveScalar()),
                         m::MinimumAnyOrder(
                             m::Op(&to_clamp),
                             m::Broadcast(&clamp_upper_bound_bcast,
                                          m::ConstantEffectiveScalar()))))) {
    TF_ASSIGN_OR_RETURN(auto clamp,
                        MinMaxToClamp(clamp_lower_bound_bcast, to_clamp,
                                      clamp_upper_bound_bcast));
    if (clamp) {
      return ReplaceWithNewInstruction(maximum, std::move(clamp));
    }
  }

  HloInstruction* clamp_lower_bound;
  HloInstruction* clamp_upper_bound;
  HloInstruction* max_operand;
  HloInstruction* clamp;
  if (Match(maximum,
            m::MaximumAnyOrder(
                m::Op(&max_operand),
                m::Clamp(&clamp, m::Op(&clamp_lower_bound), m::Op(&to_clamp),
                         m::Op(&clamp_upper_bound))))) {
    if (max_operand == clamp_lower_bound &&
        ReplaceInstructionIfSameShape(maximum, clamp)) {
      return Status::OK();
    }
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleMinimum(HloInstruction* minimum) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(minimum, m::Minimum(m::Op(&lhs), m::Op(&rhs))));

  HloInstruction* clamp_upper_bound_bcast;
  HloInstruction* clamp_lower_bound_bcast;
  HloInstruction* to_clamp;
  if (Match(minimum, m::MinimumAnyOrder(
                         m::Broadcast(&clamp_upper_bound_bcast,
                                      m::ConstantEffectiveScalar()),
                         m::MaximumAnyOrder(
                             m::Op(&to_clamp),
                             m::Broadcast(&clamp_lower_bound_bcast,
                                          m::ConstantEffectiveScalar()))))) {
    TF_ASSIGN_OR_RETURN(auto clamp,
                        MinMaxToClamp(clamp_lower_bound_bcast, to_clamp,
                                      clamp_upper_bound_bcast));
    if (clamp) {
      return ReplaceWithNewInstruction(minimum, std::move(clamp));
    }
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleClamp(HloInstruction* clamp) {
  HloInstruction* clamp_lower_bound;
  HloInstruction* clamp_upper_bound;
  HloInstruction* to_clamp;
  CHECK(Match(clamp, m::Clamp(m::Op(&clamp_lower_bound), m::Op(&to_clamp),
                              m::Op(&clamp_upper_bound))));

  // clamp(a, clamp(a, x, b), b) -> clamp(a, x, b)
  if (Match(to_clamp, m::Clamp(m::Op().Is(clamp_lower_bound), m::Op(),
                               m::Op().Is(clamp_upper_bound))) &&
      ReplaceInstructionIfSameShape(clamp, to_clamp)) {
    return Status::OK();
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

  VLOG(10) << "trying transform [(A * C1) * C2 => A * (C1 * C2)]";
  HloInstruction *a, *c1, *c2;
  if (Match(multiply,
            m::Multiply(m::Multiply(m::NonConstant(&a), m::Constant(&c1)),
                        m::Constant(&c2))) ||
      Match(multiply,
            m::Multiply(
                m::Multiply(m::Op(&a), m::Broadcast(m::ConstantScalar(&c1))),
                m::Broadcast(m::ConstantScalar(&c2))))) {
    TF_ASSIGN_OR_RETURN(auto* product_of_constants,
                        MakeBinaryHlo(HloOpcode::kMultiply, c1, c2));
    if (ShapeUtil::IsScalar(product_of_constants->shape()) &&
        !ShapeUtil::IsScalar(multiply->shape())) {
      product_of_constants =
          computation_->AddInstruction(HloInstruction::CreateBroadcast(
              multiply->shape(), product_of_constants, {}));
    }
    return ReplaceWithNewInstruction(
        multiply,
        HloInstruction::CreateBinary(multiply->shape(), HloOpcode::kMultiply, a,
                                     product_of_constants));
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
  }

  // A || False => A and A | 0 => A
  VLOG(10) << "trying transform [A || False => A]: " << logical_or->ToString();
  if (IsAll(rhs, 0) && ReplaceInstructionIfSameShape(logical_or, lhs)) {
    return Status::OK();
  }

  // False || A => A and 0 | A => A
  VLOG(10) << "trying transform [False || A => A]: " << logical_or->ToString();
  if (IsAll(lhs, 0) && ReplaceInstructionIfSameShape(logical_or, rhs)) {
    return Status::OK();
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

  // ln(pow(A,B)) => B*ln(abs(A))
  // or B*ln(A) if A is complex.
  if (Match(log, m::Log(m::Power(m::Op(&a), m::Op(&b))))) {
    auto abs_a = ShapeUtil::ElementIsComplex(a->shape())
                     ? a
                     : computation_->AddInstruction(HloInstruction::CreateUnary(
                           log->shape(), HloOpcode::kAbs, a));
    auto new_log = computation_->AddInstruction(
        HloInstruction::CreateUnary(log->shape(), HloOpcode::kLog, abs_a));
    return ReplaceWithNewInstruction(
        log, HloInstruction::CreateBinary(log->shape(), HloOpcode::kMultiply,
                                          new_log, b));
  }

  if (Match(log, m::Log(m::Sqrt(m::Op(&a))))) {
    auto new_log = computation_->AddInstruction(
        HloInstruction::CreateUnary(log->shape(), HloOpcode::kLog, a));
    return ReplaceWithNewInstruction(
        log, HloInstruction::CreateBinary(log->shape(), HloOpcode::kMultiply,
                                          new_log, MakeScalarLike(log, 0.5)));
  }

  if (Match(log, m::Log(m::Rsqrt(m::Op(&a))))) {
    auto new_log = computation_->AddInstruction(
        HloInstruction::CreateUnary(log->shape(), HloOpcode::kLog, a));
    return ReplaceWithNewInstruction(
        log, HloInstruction::CreateBinary(log->shape(), HloOpcode::kMultiply,
                                          new_log, MakeScalarLike(log, -0.5)));
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

absl::optional<std::vector<int64>> ReshapeLeavesDimensionsUnmodified(
    const HloInstruction* hlo, absl::Span<const int64> input_dim_indices) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kReshape);
  return ShapeUtil::ReshapeLeavesDimensionsUnmodified(
      hlo->operand(0)->shape(), hlo->shape(), input_dim_indices);
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
      return (!instruction->shape().IsTuple());
    default:
      return false;
  }
}

// Returns true if the output of "instruction" is a subset of the elements of
// "operand". Precondition: "operand" is an operand of "instruction".
bool OutputIsSubsetOfOperandElements(HloInstruction* instruction,
                                     HloInstruction* operand) {
  const auto operand_indices = instruction->OperandIndices(operand);
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
  if (broadcast->shape().rank() == operand->shape().rank() &&
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
  if (options_.is_layout_sensitive()) {
    return Status::OK();
  }
  if (ShapeUtil::HasDegenerateDimensions(operand->shape())) {
    auto new_operand =
        operand->parent()->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::DropDegenerateDimensions(operand->shape()), operand));
    std::vector<int64> new_dims;
    new_dims.reserve(new_operand->shape().rank());
    for (int64 i = 0; i < operand->shape().rank(); ++i) {
      if (operand->shape().dimensions(i) != 1) {
        new_dims.push_back(dims[i]);
      }
    }
    return ReplaceWithNewInstruction(
        broadcast, HloInstruction::CreateBroadcast(broadcast->shape(),
                                                   new_operand, new_dims));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleCompare(HloInstruction* compare) {
  HloInstruction* lhs;
  HloInstruction* rhs;
  CHECK(Match(compare, m::Compare(m::Op(&lhs), m::Op(&rhs))));

  if (compare->comparison_direction() == ComparisonDirection::kLt &&
      lhs->opcode() == HloOpcode::kIota && IsAll(rhs, 0)) {
    return ReplaceInstruction(compare, MakeScalarLike(compare, false));
  } else if (compare->comparison_direction() == ComparisonDirection::kGt &&
             IsAll(lhs, 0) && rhs->opcode() == HloOpcode::kIota) {
    return ReplaceInstruction(compare, MakeScalarLike(compare, false));
  } else if (compare->comparison_direction() == ComparisonDirection::kGe &&
             lhs->opcode() == HloOpcode::kIota && IsAll(rhs, 0)) {
    return ReplaceInstruction(compare, MakeScalarLike(compare, true));
  } else if (compare->comparison_direction() == ComparisonDirection::kLe &&
             IsAll(lhs, 0) && rhs->opcode() == HloOpcode::kIota) {
    return ReplaceInstruction(compare, MakeScalarLike(compare, true));
  }
  if (lhs == rhs &&
      primitive_util::IsIntegralType(lhs->shape().element_type())) {
    switch (compare->comparison_direction()) {
      case ComparisonDirection::kGt:
      case ComparisonDirection::kLt:
      case ComparisonDirection::kNe:
        return ReplaceInstruction(compare, MakeScalarLike(compare, false));
      case ComparisonDirection::kEq:
      case ComparisonDirection::kGe:
      case ComparisonDirection::kLe:
        return ReplaceInstruction(compare, MakeScalarLike(compare, true));
    }
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleConvert(HloInstruction* convert) {
  PrimitiveType src_type = convert->operand(0)->shape().element_type();
  PrimitiveType dest_type = convert->shape().element_type();
  // A conversion to the same element type as the operand is a nop and can be
  // removed.  A conversion of a constant can be simplified by making a new
  // constant.
  if (src_type == dest_type) {
    return ReplaceInstruction(convert, convert->mutable_operand(0));
  }

  // Eliminate a convert pair if it is a no-op. The following are a few
  // example cases that are being handled:
  // 1. convert(convert(A, $TYPE1), $TYPE2) is simplified to A if A is of $TYPE2
  //    and convert(A, $TYPE1) is an upcast
  // 2. convert(convert(A, $TYPE1),$TYPE2) is simplified to A if A is of $TYPE2
  //    and convert(A, $TYPE1) is an upcast and is an integral conversion from
  //    unsigned to signed (only signed to unsigned conversion is NOT allowed)
  // 3. Tuple(convert(A, $TYPE1) , floor(convert(convert(A, $TYPE1), $TYPE2)),
  //    convert(convert(A, $TYPE1), $TYPE2)) is simplified to Tuple(convert(A,
  //    $TYPE1) , floor(A), A) -> a case where the first convert has a
  //    fan-out
  if (convert->operand(0)->opcode() == HloOpcode::kConvert &&
      IsConvertPairNoOp(convert)) {
    return ReplaceInstruction(convert,
                              convert->mutable_operand(0)->mutable_operand(0));
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
    auto zero = computation_->AddInstruction(
        simplifier_->CreateConstantWithLayoutUpdated(
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

  // Interior padding on one sized dimensions have no effect. As a result it
  // makes other simplifications possible if there is no interior padding.
  if (HasInteriorPadding(pad->padding_config())) {
    PaddingConfig padding_config = pad->padding_config();
    bool cleared_interior_padding = false;
    for (int64 i = 0; i < pad->shape().rank(); ++i) {
      if (padding_config.dimensions(i).interior_padding() > 0 &&
          pad->operand(0)->shape().dimensions(i) == 1) {
        cleared_interior_padding = true;
        padding_config.mutable_dimensions(i)->set_interior_padding(0);
      }
    }
    if (cleared_interior_padding) {
      return ReplaceWithNewInstruction(
          pad,
          HloInstruction::CreatePad(pad->shape(), pad->mutable_operand(0),
                                    pad->mutable_operand(1), padding_config));
    }
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
    TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
        pad->shape(), slice->mutable_shape()));

    // Verify that the slice shape matches the pad shape.
    TF_RET_CHECK(ShapeUtil::Equal(slice->shape(), pad->shape()));

    return ReplaceInstruction(pad, slice);
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandlePower(HloInstruction* power) {
  VLOG(10) << "trying transform [pow(A, 0) => 1]: " << power->ToString();
  HloInstruction *lhs, *rhs;
  CHECK(Match(power, m::Power(m::Op(&lhs), m::Op(&rhs))));
  if (IsAll(rhs, 0)) {
    return ReplaceInstruction(power, MakeScalarLike(power, 1));
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
    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateBinary(power->shape(), HloOpcode::kDivide,
                                            MakeScalarLike(lhs, 1), lhs));
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

    Shape changed_shape;
    for (HloInstruction* user_operand : user->operands()) {
      if (user_operand->opcode() == HloOpcode::kBroadcast &&
          ShapeUtil::IsScalar(user_operand->operand(0)->shape())) {
        changed_shape = ShapeUtil::ChangeElementType(
            operand->shape(), user_operand->shape().element_type());
        simplifier_->UpdateLayout(&changed_shape);
        new_operands.push_back(
            computation_->AddInstruction(HloInstruction::CreateBroadcast(
                changed_shape, user_operand->mutable_operand(0), {})));
      } else {
        CHECK_EQ(broadcast, user_operand);
        new_operands.push_back(operand);
      }
    }
    VLOG(4) << "Sinking broadcast after user:";
    VLOG(4) << "  old broadcast: " << broadcast->ToString();
    VLOG(4) << "  old user: " << user->ToString();
    changed_shape = ShapeUtil::ChangeElementType(operand->shape(),
                                                 user->shape().element_type());
    simplifier_->UpdateLayout(&changed_shape);
    HloInstruction* new_user = computation_->AddInstruction(
        user->CloneWithNewOperands(changed_shape, new_operands));
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

namespace {
template <typename T>
std::unique_ptr<HloInstruction> TryRemainderToAnd(
    HloInstruction* remainder, HloComputation* computation,
    AlgebraicSimplifier* simplifier) {
  HloInstruction *a, *b, *c;
  CHECK(Match(remainder, m::Remainder(m::Op(&a), m::Op(&b))));

  if (ShapeUtil::ElementIsIntegral(remainder->shape()) &&
      !Match(b, m::ConstantEffectiveScalar(&c)) &&
      !Match(b, m::Broadcast(m::ConstantEffectiveScalar(&c)))) {
    return nullptr;
  }

  if (ShapeUtil::ElementIsSigned(remainder->shape())) {
    int64 b_value = c->literal().GetFirstElement<T>();
    if (b_value > 0 && IsPowerOfTwo(static_cast<uint64>(b_value))) {
      // Handle negative dividends by negating the result of the division.
      HloInstruction* zero_like_a = BroadcastZeros(
          computation, a->shape().element_type(), a->shape().dimensions());

      auto* dividend_is_negative =
          computation->AddInstruction(HloInstruction::CreateCompare(
              ShapeUtil::ChangeElementType(a->shape(), PRED), a, zero_like_a,
              ComparisonDirection::kLt));

      auto* negated_dividend = computation->AddInstruction(
          HloInstruction::CreateUnary(a->shape(), HloOpcode::kNegate, a));

      auto* abs_dividend =
          computation->AddInstruction(HloInstruction::CreateTernary(
              a->shape(), HloOpcode::kSelect, dividend_is_negative,
              negated_dividend, a));

      auto* quotient = computation->AddInstruction(HloInstruction::CreateBinary(
          remainder->shape(), HloOpcode::kAnd, abs_dividend,
          MakeScalarLike(abs_dividend, b_value - 1)));

      auto* neqated_quotient =
          computation->AddInstruction(HloInstruction::CreateUnary(
              quotient->shape(), HloOpcode::kNegate, quotient));

      return HloInstruction::CreateTernary(
          remainder->shape(), HloOpcode::kSelect, dividend_is_negative,
          neqated_quotient, quotient);
    }
  } else {
    uint64 b_value = c->literal().GetFirstElement<T>();
    if (IsPowerOfTwo(b_value)) {
      HloInstruction* mask_amount = computation->AddInstruction(
          simplifier->CreateConstantWithLayoutUpdated(
              LiteralUtil::CreateR0<T>(b_value - 1)));
      if (!ShapeUtil::IsScalar(b->shape())) {
        mask_amount = computation->AddInstruction(
            HloInstruction::CreateBroadcast(b->shape(), mask_amount, {}));
      }
      return HloInstruction::CreateBinary(remainder->shape(), HloOpcode::kAnd,
                                          a, mask_amount);
    }
  }
  return nullptr;
}
}  // namespace

Status AlgebraicSimplifierVisitor::HandleRemainder(HloInstruction* remainder) {
  HloInstruction *a, *b;
  CHECK(Match(remainder, m::Remainder(m::Op(&a), m::Op(&b))));

  // (A % B) % B == A % B.
  if (Match(a, m::Remainder(m::Op(), m::Op().Is(b)))) {
    return ReplaceInstruction(remainder, a);
  }

  // A % B => A & (B - 1) if B is a power of 2.
  switch (remainder->shape().element_type()) {
    case S8:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<int8>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case S16:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<int16>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case S32:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<int32>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case S64:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<int64>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case U8:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<uint8>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case U16:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<uint16>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case U32:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<uint32>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case U64:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<uint64>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    default:
      break;
  }

  // If M < N, then {0, ..., M} % N ==> {0, ..., M}.
  //
  // Currently this only covers the case when N is a broadcasted constant
  // scalar.  We could also cover the case when N is a non-broadcasted constant
  // with the same value repeated.
  HloInstruction* iota;
  HloInstruction* divisor;
  if (Match(remainder,
            m::Remainder(m::Iota(&iota),
                         m::Broadcast(m::ConstantEffectiveScalar(&divisor))))) {
    // The iota counts {0, ..., iota_upper_bound - 1}.  (Actually this is
    // conservative; the iota may overflow and count up to a smaller value than
    // this.  But that's OK for our purposes here.)
    int64 iota_upper_bound = iota->shape().dimensions(
        Cast<HloIotaInstruction>(iota)->iota_dimension());
    absl::optional<int64> divisor_val = divisor->literal().GetIntegralAsS64(
        std::vector<int64>(0, divisor->shape().dimensions_size()));
    if (divisor_val && *divisor_val >= iota_upper_bound) {
      return ReplaceInstruction(remainder, iota);
    }
  }

  // (X + N) % N = X % N, so long as X + N does not overflow.
  //
  // We don't have range tracking in XLA that would let us know whether X + N
  // overflows, so for now we only do this simplification when X is an iota.  We
  // could add other operations where it's easy to see a range, such as
  // remainder, convert, etc., though at some point we'd probably want a
  // range-tracking analysis.
  HloInstruction* bcast;
  HloInstruction* addend;
  if (Match(
          remainder,
          m::Remainder(
              m::AddAnyOrder(m::Iota(&iota),
                             m::Broadcast(m::ConstantEffectiveScalar(&addend))),
              m::Broadcast(&bcast, m::ConstantEffectiveScalar(&divisor)))) &&
      addend == divisor) {
    // The iota counts {0, ...iota_upper_bound - 1}, with the same caveat above
    // that iota_upper_bound is conservative, and the true upper bound may be
    // smaller.
    int64 iota_upper_bound = iota->shape().dimensions(
        Cast<HloIotaInstruction>(iota)->iota_dimension());
    absl::optional<int64> divisor_val = divisor->literal().GetIntegralAsS64(
        std::vector<int64>(0, divisor->shape().dimensions_size()));
    if (divisor_val) {
      // Check whether divisor_val + iota_upper_bound - 1 overflows.
      absl::optional<int64> max_val =
          OverflowSafeAdd(*divisor_val, iota_upper_bound);
      if (max_val.has_value() &&
          FitsInIntegralType(*max_val, iota->shape().element_type())) {
        return ReplaceWithNewInstruction(
            remainder,
            HloInstruction::CreateBinary(remainder->shape(),
                                         HloOpcode::kRemainder, iota, bcast));
      }
    }
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleReshape(HloInstruction* reshape) {
  auto operand = reshape->mutable_operand(0);

  // Reshape directly to empty constant if the shape contains zero-element
  // dimension.
  if (ShapeUtil::IsZeroElementArray(reshape->shape())) {
    // If the instruction doesn't have a layout, use a default layout for
    // the literal result.
    Shape reshaped_shape = reshape->shape();
    if (!LayoutUtil::HasLayout(reshaped_shape)) {
      LayoutUtil::SetToDefaultLayout(&reshaped_shape);
    }
    auto empty_constant = simplifier_->CreateConstantWithLayoutUpdated(
        Literal::CreateFromShape(reshaped_shape));

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
  if (HloInstruction* bitcast_operand =
          BitcastingOperandOfReshapeOrCopyChain(reshape, options_)) {
    ReplaceWithBitcast(reshape, bitcast_operand);
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleReverse(HloInstruction* reverse) {
  // When all the dimensions to reverse are trivial (i.e. the bound is 1),
  // there is nothing to be done.
  auto dim_is_one = [&](int64 i) -> bool {
    return reverse->shape().dimensions(i) == 1;
  };
  if (absl::c_all_of(reverse->dimensions(), dim_is_one)) {
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
        if (start < low || start >= low + data) {
          return true;
        }
      }
      return false;
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
    if (slice->shape().rank() != 1) {
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
  int64 slice_rank = slice->shape().rank();
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
    const int64 rank = new_slice_shape.rank();
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
    simplifier_->UpdateLayout(new_slice->mutable_shape());
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

  auto only_broadcast_dims_sliced = [&] {
    if (slice->operand(0)->opcode() != HloOpcode::kBroadcast) {
      return false;
    }
    for (int64 dim : slice->operand(0)->dimensions()) {
      if (slice->slice_starts(dim) != 0 || slice->slice_strides(dim) != 1 ||
          slice->slice_limits(dim) !=
              slice->operand(0)->shape().dimensions(dim)) {
        return false;
      }
    }
    return true;
  };
  if (only_broadcast_dims_sliced()) {
    return ReplaceWithNewInstruction(
        slice,
        HloInstruction::CreateBroadcast(
            slice->shape(), slice->mutable_operand(0)->mutable_operand(0),
            slice->mutable_operand(0)->dimensions()));
  }

  TF_ASSIGN_OR_RETURN(bool replaced, TrySimplifyScalarSlice(slice));
  if (replaced) {
    return Status::OK();
  }

  // Try to simplify concat -> slice to an operand of concat.
  if (slice->operand(0)->opcode() == HloOpcode::kConcatenate &&
      IsUnstridedSlice(slice)) {
    auto concat = slice->operand(0);
    int64 concat_dim = concat->concatenate_dimension();
    int64 piece_start = 0;
    for (auto piece : concat->operands()) {
      if (!SameShape(piece, slice)) {
        piece_start += piece->shape().dimensions(concat_dim);
        continue;
      }
      if (slice->slice_starts(concat_dim) == piece_start) {
        return ReplaceInstruction(slice, piece);
      }
      piece_start += piece->shape().dimensions(concat_dim);
    }
  }

  // Do not try to reorder slices and reshapes after layout assignment as it may
  // be invalid.
  if (!options_.is_layout_sensitive()) {
    TF_ASSIGN_OR_RETURN(replaced, TryToReorderSliceAndReshape(slice));
  }
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

Status AlgebraicSimplifierVisitor::HandleReduce(HloInstruction* hlo) {
  HloReduceInstruction* reduce = Cast<HloReduceInstruction>(hlo);
  bool multi_output_reduce = reduce->shape().IsTuple();

  // For tuple reduce, we require all reduce shapes to be the same, up to the
  // element types, so we can just the first operand and the first result as a
  // representative.
  auto arg = reduce->inputs()[0];
  auto init_value = reduce->init_values()[0];
  Shape& reduce_result_shape = const_cast<Shape&>(
      multi_output_reduce ? reduce->shape().tuple_shapes(0) : reduce->shape());

  absl::Span<const int64> dimensions(reduce->dimensions());
  HloComputation* function = reduce->to_apply();
  if (ShapeUtil::IsZeroElementArray(arg->shape()) ||
      ShapeUtil::IsZeroElementArray(reduce_result_shape)) {
    if (multi_output_reduce) {
      std::vector<HloInstruction*> broadcast_inits;
      int64 inputs = reduce->input_count();
      for (int64 i = 0; i < inputs; ++i) {
        broadcast_inits.push_back(computation_->AddInstruction(
            HloInstruction::CreateBroadcast(reduce->shape().tuple_shapes(i),
                                            reduce->init_values()[i], {})));
      }
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateTuple(broadcast_inits));
    } else {
      return ReplaceWithNewInstruction(
          reduce,
          HloInstruction::CreateBroadcast(reduce_result_shape, init_value, {}));
    }
  }

  // If the reduction results in the same number of elements, then the only
  // possible side effect would be a reshape. Since the init_value is an
  // identity of the reduction function, we can therefore replace the reduce
  // with a simple reshape, ignoring the reduction function completely.
  if (ShapeUtil::ElementsIn(reduce_result_shape) ==
      ShapeUtil::ElementsIn(arg->shape())) {
    if (multi_output_reduce) {
      std::vector<HloInstruction*> reshaped_args;
      int64 inputs = reduce->input_count();
      for (int64 i = 0; i < inputs; ++i) {
        reshaped_args.push_back(
            computation_->AddInstruction(HloInstruction::CreateReshape(
                reduce->shape().tuple_shapes(i), reduce->inputs()[i])));
      }
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateTuple(reshaped_args));
    } else {
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateReshape(reduce_result_shape, arg));
    }
  }

  // TODO(b/131122694): Most of those optimizations below can be done for
  // multi-output reduces.
  if (multi_output_reduce) {
    return Status::OK();
  }

  // A Transpose feeding a reduce can simply permute the reduction dimensions
  // field if the output of the reduce is a vector or scalar. Higher ranked
  // result may require a transpose of the output.
  if (reduce_result_shape.rank() <= 1 &&
      arg->opcode() == HloOpcode::kTranspose) {
    auto transpose_dimensions = arg->dimensions();
    std::vector<int64> new_reduce_dimensions;
    for (auto dim : dimensions) {
      new_reduce_dimensions.push_back(transpose_dimensions[dim]);
    }
    return ReplaceWithNewInstruction(
        reduce, HloInstruction::CreateReduce(
                    reduce_result_shape, arg->mutable_operand(0), init_value,
                    new_reduce_dimensions, function));
  }

  // If a reduce feeds a reduce with the same computation and initial value,
  // they can be combined into a single reduce.
  if (arg->opcode() == HloOpcode::kReduce &&
      init_value->Identical(*arg->operand(1)) &&
      *function == *arg->to_apply()) {
    // Create a new reduce with the combined reduction dimensions of both
    // reduces.
    std::vector<int64> arg_dims = arg->dimensions();
    absl::c_sort(arg_dims);
    std::vector<int64> reduce_dims = reduce->dimensions();
    absl::c_sort(reduce_dims);
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
        reduce, HloInstruction::CreateReduce(
                    reduce_result_shape, arg->mutable_operand(0), init_value,
                    new_dimensions, function));
  }

  // A reshape that collapses multiple dimensions into a dimension being
  // reduced can just reduce all of those dimensions instead of doing a
  // collapsing reshape before a reduction.
  if (arg->opcode() == HloOpcode::kReshape) {
    std::vector<std::pair<int64, int64>> unmodified_dims =
        ShapeUtil::DimensionsUnmodifiedByReshape(arg->operand(0)->shape(),
                                                 arg->shape());
    std::vector<bool> arg_dim_in_output(arg->shape().rank(), true);
    std::vector<bool> arg_dim_unmodified(arg->shape().rank(), false);
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
      absl::flat_hash_set<int64> dimensions_not_to_reduce;
      for (auto dim_pair : unmodified_dims) {
        if (arg_dim_in_output[dim_pair.second]) {
          dimensions_not_to_reduce.insert(dim_pair.first);
        }
      }
      std::vector<int64> new_reduce_dimensions;
      for (int64 i = 0; i < arg->operand(0)->shape().rank(); ++i) {
        if (!dimensions_not_to_reduce.contains(i)) {
          new_reduce_dimensions.push_back(i);
        }
      }
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateReduce(
                      reduce_result_shape, arg->mutable_operand(0), init_value,
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
          HloInstruction::CreateReduce(reduce_result_shape, operand, init_value,
                                       reduce->dimensions(), function));
      if (old_reduce != nullptr) {
        new_reduce = computation_->AddInstruction(HloInstruction::CreateMap(
            reduce_result_shape, {old_reduce, new_reduce}, function));
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

  if (options_.enable_window_reduce_to_reduce_replacement()) {
    // A reduce window can be expressed as a reduce and a reshape if all
    // dimensions either have a window size of one or the entire dimension. If
    // there is no stride, dilation, or padding, this is as easy as checking the
    // size of the output shape and window dimension.
    //
    // The reshape is a bitcast since it adds one-sized dimensions. Often these
    // ones are immediately removed as well with another reshape. The
    // implementation of reduce tends to be slightly more efficient at reducing
    // entire dimensions compared to reduce window.
    auto effective_reduce_dims = [&] {
      if (window_util::HasStride(window) || window_util::HasDilation(window) ||
          window_util::HasPadding(window)) {
        return absl::InlinedVector<int64, 8>{};
      }
      absl::InlinedVector<int64, 8> reduce_dims;
      for (int64 i = 0; i < window.dimensions_size(); ++i) {
        if (window.dimensions(i).size() == 1) {
          continue;
        } else if (reduce_window->shape().dimensions(i) == 1) {
          reduce_dims.push_back(i);
        } else {
          return absl::InlinedVector<int64, 8>{};
        }
      }
      return reduce_dims;
    }();

    // If a reduce window can be expressed as a reduce, do so and reshape the
    // output.
    if (!effective_reduce_dims.empty()) {
      Shape reduce_shape = ShapeUtil::FilterDimensions(
          [&](int64 dim) {
            return !absl::c_linear_search(effective_reduce_dims, dim);
          },
          reduce_window->shape());
      simplifier_->UpdateLayout(&reduce_shape);
      HloInstruction* reduce =
          computation_->AddInstruction(HloInstruction::CreateReduce(
              /*shape=*/reduce_shape,
              /*operand=*/operand,
              /*init_value=*/reduce_window->mutable_operand(1),
              /*dimensions_to_reduce=*/effective_reduce_dims,
              /*reduce_computation=*/function));
      return ReplaceWithNewInstruction(
          reduce_window,
          HloInstruction::CreateReshape(reduce_window->shape(), reduce));
    }
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

  VLOG(10) << "Considering folding Pad: " << pad->ToString()
           << "\ninto reduce-window: " << reduce_window->ToString()
           << (convert != nullptr
                   ? absl::StrCat("\nvia convert: ", convert->ToString())
                   : "");

  // Do not fold interior padding into ReduceWindow since the backends do not
  // support it.
  const PaddingConfig& pad_config = pad->padding_config();
  if (HasInteriorPadding(pad_config) && window_util::HasBaseDilation(window)) {
    VLOG(10) << "Not folding interior pad into base-dilated reduce-window.";
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
    if (HasInteriorPadding(pad_config)) {
      VLOG(10) << "Window has interior padding.";
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
    Shape changed_shape = ShapeUtil::ChangeElementType(
        pad_operand->shape(), convert->shape().element_type());
    simplifier_->UpdateLayout(&changed_shape);
    new_reduce_window_operand = computation_->AddInstruction(
        HloInstruction::CreateConvert(changed_shape, pad_operand));
  } else {
    new_reduce_window_operand = pad_operand;
  }

  if (is_effective_broadcast()) {
    VLOG(10) << "Replacing pad/reduce-window with broadcast.";
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
  const int64 rank = reduce_window->shape().rank();
  TF_RET_CHECK(pad_config.dimensions_size() == rank);
  TF_RET_CHECK(window.dimensions_size() == rank);
  for (int64 i = 0; i < rank; ++i) {
    const auto& pad_dim = pad_config.dimensions(i);
    auto& window_dim = *new_window.mutable_dimensions(i);
    window_dim.set_padding_low(window_dim.padding_low() +
                               pad_dim.edge_padding_low());
    window_dim.set_padding_high(window_dim.padding_high() +
                                pad_dim.edge_padding_high());
    if (pad_dim.interior_padding() != 0) {
      CHECK_EQ(window_dim.base_dilation(), 1);
      window_dim.set_base_dilation(1 + pad_dim.interior_padding());
    }
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

Status AlgebraicSimplifierVisitor::HandleScatter(HloInstruction* scatter) {
  if (ShapeUtil::IsZeroElementArray(scatter->operand(2)->shape()) &&
      ReplaceInstructionIfSameShape(scatter, scatter->mutable_operand(0))) {
    return Status::OK();
  }
  if (ShapeUtil::IsZeroElementArray(scatter->operand(1)->shape()) &&
      SameShape(scatter, scatter->operand(0)) &&
      SameShape(scatter, scatter->operand(2))) {
    return ReplaceWithNewInstruction(
        scatter, HloInstruction::CreateMap(
                     scatter->shape(),
                     {scatter->mutable_operand(0), scatter->mutable_operand(2)},
                     scatter->to_apply()));
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
  return Status::OK();
}

namespace {
bool OnlyPermutesDegenerateDims(const Shape& shape,
                                absl::Span<const int64> perm) {
  std::vector<int64> new_permutation;
  int64 degenerate_count = 0;
  for (int64 i = 0; i < perm.size(); ++i) {
    if (shape.dimensions(i) != 1) {
      new_permutation.push_back(perm[i]);
    } else {
      ++degenerate_count;
    }
  }
  return degenerate_count > 0 && absl::c_is_sorted(new_permutation);
}
}  // namespace

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

  // Convert transpose(dot(a,b)) to dot(b,a).
  if (operand->opcode() == HloOpcode::kDot && operand->user_count() == 1 &&
      operand->shape().rank() == 2) {
    TF_ASSIGN_OR_RETURN(bool did_transform, [&]() -> StatusOr<bool> {
      const auto& dnums = operand->dot_dimension_numbers();
      if (dnums.lhs_batch_dimensions_size() != 0) {
        return false;
      }
      HloInstruction* lhs = operand->mutable_operand(0);
      if (lhs->shape().rank() != 1 + dnums.lhs_contracting_dimensions_size()) {
        return false;
      }
      HloInstruction* rhs = operand->mutable_operand(1);
      if (rhs->shape().rank() != 1 + dnums.rhs_contracting_dimensions_size()) {
        return false;
      }
      DotDimensionNumbers new_dnums;
      *new_dnums.mutable_lhs_contracting_dimensions() =
          dnums.rhs_contracting_dimensions();
      *new_dnums.mutable_rhs_contracting_dimensions() =
          dnums.lhs_contracting_dimensions();
      TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
          transpose, HloInstruction::CreateDot(transpose->shape(), /*lhs=*/rhs,
                                               /*rhs=*/lhs, new_dnums,
                                               operand->precision_config())));
      return true;
    }());
    if (did_transform) {
      return Status::OK();
    }
  }

  // Replace transpose with a reshape if more than one degenerate method is
  // permuted.
  if (OnlyPermutesDegenerateDims(transpose->shape(), transpose->dimensions())) {
    return ReplaceWithNewInstruction(
        transpose, HloInstruction::CreateReshape(
                       transpose->shape(), transpose->mutable_operand(0)));
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
        HloInstruction::CreateBitcast(shape, operand));
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
  Shape new_input_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      input_shape.element_type(), {conv_width, input_channels});
  simplifier_->UpdateLayout(&new_input_shape);
  // We already checked input_feature_dimension is more major than
  // output_feature_dimension, so data in filter_shape and row-major
  // {input_channels,output_channels} are bitwise identical.
  Shape new_filter_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      filter_shape.element_type(), {input_channels, output_channels});
  simplifier_->UpdateLayout(&new_filter_shape);
  Shape dot_output_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      convolution_shape.element_type(), {conv_width, output_channels});
  simplifier_->UpdateLayout(&dot_output_shape);

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
    return ReplaceInstruction(convolution, MakeScalarLike(convolution, 0));
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
  // Inline the map if the map computation only contains an elementwise
  // operation that can accept arbitrary shapes.
  if (map_root->opcode() == HloOpcode::kFusion || !map_root->IsElementwise()) {
    return Status::OK();
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
  AlgebraicSimplifierVisitor visitor(options_, this);
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (visitor.Run(comp, options_, this)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(2,
                 "AlgebraicSimplifier::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla

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
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

// Returns whether operand is a literal with the given value.
bool IsLiteralWithValue(const HloInstruction* operand, int8 value) {
  return operand->opcode() == HloOpcode::kConstant &&
         LiteralUtil::IsAll(operand->literal(), value);
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
  HloComputation::Builder b("scalar computation");
  auto scalar_lhs = b.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "scalar lhs"));
  auto scalar_rhs = b.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {}), "scalar rhs"));
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

  Status HandleAdd(HloInstruction* add, HloInstruction* lhs,
                   HloInstruction* rhs) override;

  Status HandleBroadcast(HloInstruction* broadcast) override;

  Status HandleCopy(HloInstruction* copy, HloInstruction* operand) override;

  Status HandleConvert(HloInstruction* convert,
                       HloInstruction* operand) override;

  Status HandleConvolution(HloInstruction* convolution, HloInstruction* lhs,
                           HloInstruction* rhs, const Window& window) override;

  Status HandleDivide(HloInstruction* divide, HloInstruction* lhs,
                      HloInstruction* rhs) override;

  Status HandleDot(HloInstruction* dot, HloInstruction* lhs,
                   HloInstruction* rhs) override;

  Status HandleGetTupleElement(HloInstruction* get_tuple_element,
                               HloInstruction* operand) override;

  Status HandleLog(HloInstruction* log, HloInstruction* operand) override;

  Status HandleMultiply(HloInstruction* multiply, HloInstruction* lhs,
                        HloInstruction* rhs) override;

  Status HandlePad(HloInstruction* pad) override;

  Status HandlePower(HloInstruction* power, HloInstruction* lhs,
                     HloInstruction* rhs) override;

  Status HandleReshape(HloInstruction* reshape) override;

  Status HandleReduce(HloInstruction* reduce, HloInstruction* arg,
                      HloInstruction* init_value,
                      tensorflow::gtl::ArraySlice<int64> dimensions,
                      HloComputation* function) override;

  Status HandleSlice(HloInstruction* slice, HloInstruction* operand) override;

  Status HandleTranspose(HloInstruction* transpose) override;

  Status HandleSubtract(HloInstruction* sub, HloInstruction* lhs,
                        HloInstruction* rhs) override;

  Status HandleMaximum(HloInstruction* maximum, HloInstruction* lhs,
                       HloInstruction* rhs) override;

  Status HandleMinimum(HloInstruction* minimum, HloInstruction* lhs,
                       HloInstruction* rhs) override;

  // Returns whether algebraic simplification has occurred.
  const bool changed() const { return changed_; }

  // Runs the visitor on a computation.
  static bool Run(
      HloComputation* computation, bool is_layout_sensitive,
      AlgebraicSimplifier::ValidBitcastCallback valid_bitcast_callback,
      bool enable_dot_simplification);

 private:
  explicit AlgebraicSimplifierVisitor(
      HloComputation* computation, bool is_layout_sensitive,
      AlgebraicSimplifier::ValidBitcastCallback valid_bitcast_callback,
      bool enable_dot_simplification)
      : computation_(computation),
        is_layout_sensitive_(is_layout_sensitive),
        valid_bitcast_callback_(std::move(valid_bitcast_callback)),
        enable_dot_simplification_(enable_dot_simplification) {}

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
};

bool AlgebraicSimplifierVisitor::Run(
    HloComputation* computation, bool is_layout_sensitive,
    AlgebraicSimplifier::ValidBitcastCallback valid_bitcast_callback,
    bool enable_dot_simplification) {
  AlgebraicSimplifierVisitor visitor(computation, is_layout_sensitive,
                                     std::move(valid_bitcast_callback),
                                     enable_dot_simplification);
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
  TF_CHECK_OK(computation_->ReplaceInstruction(instruction, bitcast));
  changed_ = true;
}

bool AlgebraicSimplifierVisitor::ReplaceInstructionIfSameShape(
    HloInstruction* old_instruction, HloInstruction* new_instruction) {
  if (!SameShape(old_instruction, new_instruction)) {
    return false;
  }
  TF_CHECK_OK(
      computation_->ReplaceInstruction(old_instruction, new_instruction));
  changed_ = true;
  return true;
}

Status AlgebraicSimplifierVisitor::HandleAdd(HloInstruction* add,
                                             HloInstruction* lhs,
                                             HloInstruction* rhs) {
  // A + 0 => A
  VLOG(10) << "trying transform [A + 0 => A]: " << add->ToString();
  if (IsLiteralWithValue(rhs, 0) && ReplaceInstructionIfSameShape(add, lhs)) {
    return Status::OK();
  }
  // 0 + A => A
  VLOG(10) << "trying transform [0 + A => A]: " << add->ToString();
  if (IsLiteralWithValue(lhs, 0) && ReplaceInstructionIfSameShape(add, rhs)) {
    return Status::OK();
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleCopy(HloInstruction* copy,
                                              HloInstruction* operand) {
  // All copies can be eliminated (assuming layout constraints are satisified).
  ReplaceInstructionIfSameShape(copy, operand);
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleSubtract(HloInstruction* sub,
                                                  HloInstruction* lhs,
                                                  HloInstruction* rhs) {
  // A - 0 => A
  VLOG(10) << "trying transform [A - 0 => A]: " << sub->ToString();
  if (IsLiteralWithValue(rhs, 0) && ReplaceInstructionIfSameShape(sub, lhs)) {
    return Status::OK();
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleDivide(HloInstruction* divide,
                                                HloInstruction* lhs,
                                                HloInstruction* rhs) {
  // A/1 => A
  VLOG(10) << "trying transform [A/1 => A]: " << divide->ToString();
  if (IsLiteralWithValue(rhs, 1) &&
      ReplaceInstructionIfSameShape(divide, lhs)) {
    return Status::OK();
  }

  // exp(A)/exp(B) => exp(A-B)
  if (lhs->opcode() == HloOpcode::kExp && rhs->opcode() == HloOpcode::kExp) {
    VLOG(10) << "transform [exp(A)/exp(B) => exp(A-B)]: " << divide->ToString();
    HloInstruction* subtract =
        computation_->AddInstruction(HloInstruction::CreateBinary(
            divide->shape(), HloOpcode::kSubtract, lhs->mutable_operand(0),
            rhs->mutable_operand(0)));
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(
        divide, HloInstruction::CreateUnary(divide->shape(), HloOpcode::kExp,
                                            subtract));
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleDot(HloInstruction* dot,
                                             HloInstruction* lhs,
                                             HloInstruction* rhs) {
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
        HloInstruction::CreateConstant(LiteralUtil::CreateR0(0.0f)));
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(
        dot, HloInstruction::CreateBroadcast(dot->shape(), zero, {}));
  }

  // Simplify dot(transpose(a), transpose(b)) to tranpose(dot(b,a)).
  if (lhs->IsRank2Transpose() && rhs->IsRank2Transpose()) {
    auto new_dot = computation_->AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::PermuteDimensions({1, 0}, dot->shape()), HloOpcode::kDot,
        rhs->mutable_operand(0), lhs->mutable_operand(0)));
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(
        dot, HloInstruction::CreateTranspose(dot->shape(), new_dot, {1, 0}));
  }

  // Simplify outer product into multiply with implicit broadcasting.
  //
  // A dot(a[M, 1], b[1, N]) = multiply(a [M,1], b [1, N])
  if (ShapeUtil::Rank(rhs->shape()) == 2 && rhs->shape().dimensions(0) == 1) {
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(
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
        HloInstruction::CreateConstant(LiteralUtil::CreateR0(0.0f)));
    auto reduce = computation_->AddInstruction(HloInstruction::CreateReduce(
        ShapeUtil::MakeShape(dot->shape().element_type(), {}), multiply, zero,
        {0}, add_reduce_computation));
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(
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
        HloInstruction::CreateConstant(LiteralUtil::CreateR0(0.0f)));
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
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(
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
        HloInstruction::CreateConstant(LiteralUtil::CreateR0(0.0f)));
    auto reduce = computation_->AddInstruction(HloInstruction::CreateReduce(
        ShapeUtil::MakeShape(dot->shape().element_type(),
                             {lhs->shape().dimensions(0)}),
        multiply, zero, {1}, add_reduce_computation));
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(
        dot, HloInstruction::CreateReshape(dot->shape(), reduce));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleMultiply(HloInstruction* multiply,
                                                  HloInstruction* lhs,
                                                  HloInstruction* rhs) {
  // A*1 => A
  VLOG(10) << "trying transform [A*1 => A]: " << multiply->ToString();
  if (IsLiteralWithValue(rhs, 1) &&
      ReplaceInstructionIfSameShape(multiply, lhs)) {
    return Status::OK();
  }
  // 1*A => A
  VLOG(10) << "trying transform [1*A => A]: " << multiply->ToString();
  if (IsLiteralWithValue(lhs, 1) &&
      ReplaceInstructionIfSameShape(multiply, rhs)) {
    return Status::OK();
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleLog(HloInstruction* log,
                                             HloInstruction* operand) {
  // ln(exp(A)) => A
  VLOG(10) << "trying transform [ln(exp(A)) => A]: " << log->ToString();
  if (operand->opcode() == HloOpcode::kExp &&
      ReplaceInstructionIfSameShape(log, operand->mutable_operand(0))) {
    return Status::OK();
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleGetTupleElement(
    HloInstruction* get_tuple_element, HloInstruction* operand) {
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

// Returns true if the output of "instruction" is a permutation of the elements
// of "operand". Precondition: "operand" is an operand of "instruction".
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
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(
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
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(
        broadcast, HloInstruction::CreateTranspose(broadcast->shape(), operand,
                                                   broadcast->dimensions()));
  }

  // A broadcast of a reshape which merely inserts 1-sized dimensions can elide
  // its operand.
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
      changed_ = true;
      return computation_->ReplaceWithNewInstruction(
          broadcast,
          HloInstruction::CreateBroadcast(broadcast->shape(),
                                          operand->mutable_operand(0), dims));
    }
  }

  // A Broadcast that feeds a unary element-wise operation can sink the
  // broadcast after the unary element-wise operation.
  TF_ASSIGN_OR_RETURN(
      changed_,
      TryToSinkReshapeOrBroadcastAfterOpWithUniqueNonScalarOperand(broadcast));
  if (changed_) {
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
        // Use ReplaceUsesOfInstruction instead of ReplaceWithNewInstruction
        // because we are replacing an instruction other than the visited
        // instruction.
        changed_ = true;
        return computation_->ReplaceUsesOfInstruction(user, new_broadcast);
      }
    }
  }
  return Status::OK();
}

template <PrimitiveType primitive_src_type, PrimitiveType primitive_dest_type>
static std::unique_ptr<HloInstruction> ConvertIfTypesMatch(
    const Literal& src_literal) {
  CHECK_EQ(primitive_src_type, src_literal.shape().element_type());

  return HloInstruction::CreateConstant(
      LiteralUtil::Convert<typename primitive_util::PrimitiveTypeToNative<
                               primitive_src_type>::type,
                           typename primitive_util::PrimitiveTypeToNative<
                               primitive_dest_type>::type>(src_literal));
}

template <PrimitiveType primitive_src_type>
static std::unique_ptr<HloInstruction> ConvertIfDestTypeMatches(
    const Literal& src_literal, PrimitiveType primitive_dest_type) {
  switch (primitive_dest_type) {
#define CONVERT_IF_TYPES_MATCH(type) \
  case (type):                       \
    return ConvertIfTypesMatch<primitive_src_type, (type)>(src_literal);
    CONVERT_IF_TYPES_MATCH(PRED)
    CONVERT_IF_TYPES_MATCH(S8)
    CONVERT_IF_TYPES_MATCH(S32)
    CONVERT_IF_TYPES_MATCH(S64)
    CONVERT_IF_TYPES_MATCH(U8)
    CONVERT_IF_TYPES_MATCH(U32)
    CONVERT_IF_TYPES_MATCH(U64)
    CONVERT_IF_TYPES_MATCH(F32)
    CONVERT_IF_TYPES_MATCH(F64)
#undef CONVERT_IF_TYPES_MATCH
    // Other types are not yet supported.
    default:
      LOG(FATAL) << "Unimplemented: ConvertIfDestTypeMatches for type "
                 << PrimitiveType_Name(src_literal.shape().element_type());
  }
}

static std::unique_ptr<HloInstruction> ConvertIfSrcTypeMatches(
    const Literal& src_literal, PrimitiveType primitive_dest_type) {
  switch (src_literal.shape().element_type()) {
#define CONVERT_IF_DEST_TYPE_MATCHES(type) \
  case (type):                             \
    return ConvertIfDestTypeMatches<(type)>(src_literal, primitive_dest_type);
    CONVERT_IF_DEST_TYPE_MATCHES(PRED)
    CONVERT_IF_DEST_TYPE_MATCHES(S8)
    CONVERT_IF_DEST_TYPE_MATCHES(S32)
    CONVERT_IF_DEST_TYPE_MATCHES(S64)
    CONVERT_IF_DEST_TYPE_MATCHES(U8)
    CONVERT_IF_DEST_TYPE_MATCHES(U32)
    CONVERT_IF_DEST_TYPE_MATCHES(U64)
    CONVERT_IF_DEST_TYPE_MATCHES(F32)
    CONVERT_IF_DEST_TYPE_MATCHES(F64)
#undef CONVERT_IF_DEST_TYPE_MATCHES
    // Other types are not yet supported.
    default:
      LOG(FATAL) << "Unimplemented: ConvertIfSrcTypeMatches for type "
                 << PrimitiveType_Name(src_literal.shape().element_type());
  }
}

// A conversion to the same element type as the operand is a nop and can be
// removed.  A conversion of a constant can be simplified by making a new
// constant.
Status AlgebraicSimplifierVisitor::HandleConvert(HloInstruction* convert,
                                                 HloInstruction* operand) {
  PrimitiveType src_type = operand->shape().element_type();
  PrimitiveType dest_type = convert->shape().element_type();
  if (src_type == dest_type) {
    changed_ = true;
    return computation_->ReplaceInstruction(convert, operand);
  }
  if (operand->opcode() == HloOpcode::kConstant) {
    const Literal& src_literal = operand->literal();
    std::unique_ptr<HloInstruction> new_constant =
        ConvertIfSrcTypeMatches(src_literal, dest_type);
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(convert,
                                                   std::move(new_constant));
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
    }

    // Verify that the slice shape matches the pad shape.
    TF_ASSIGN_OR_RETURN(Shape inferred_slice_shape,
                        ShapeInference::InferSliceShape(
                            nonzero_pad_shape, start_indices, end_indices));
    TF_RET_CHECK(ShapeUtil::Compatible(inferred_slice_shape, pad->shape()));

    std::unique_ptr<HloInstruction> slice = HloInstruction::CreateSlice(
        pad->shape(), nonzero_pad, start_indices, end_indices);
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(pad, std::move(slice));
  }

  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandlePower(HloInstruction* power,
                                               HloInstruction* lhs,
                                               HloInstruction* rhs) {
  VLOG(10) << "trying transform [pow(A, 0) => 1]: " << power->ToString();
  if (IsLiteralWithValue(rhs, 0)) {
    auto one = HloInstruction::CreateConstant(LiteralUtil::CloneToUnique(
        LiteralUtil::One(power->shape().element_type())));
    std::unique_ptr<HloInstruction> ones;
    if (ShapeUtil::IsScalar(power->shape())) {
      ones = std::move(one);
    } else {
      ones = HloInstruction::CreateBroadcast(
          power->shape(), computation_->AddInstruction(std::move(one)), {});
    }
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(power, std::move(ones));
  }

  VLOG(10) << "trying transform [pow(A, 1) => A]: " << power->ToString();
  if (IsLiteralWithValue(rhs, 1) && ReplaceInstructionIfSameShape(power, lhs)) {
    return Status::OK();
  }

  VLOG(10) << "trying transform [pow(A, 2) => A*A]: " << power->ToString();
  if (IsLiteralWithValue(rhs, 2)) {
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(
        power, HloInstruction::CreateBinary(power->shape(),
                                            HloOpcode::kMultiply, lhs, lhs));
  }

  VLOG(10) << "trying transform [pow(A, -1) => 1/A]: " << power->ToString();
  if (IsLiteralWithValue(rhs, -1)) {
    auto* one = computation_->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CloneToUnique(
            LiteralUtil::One(rhs->shape().element_type()))));
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(
        power, HloInstruction::CreateBinary(power->shape(), HloOpcode::kDivide,
                                            one, lhs));
  }
  return Status::OK();
}

StatusOr<bool> AlgebraicSimplifierVisitor::
    TryToSinkReshapeOrBroadcastAfterOpWithUniqueNonScalarOperand(
        HloInstruction* reshape_or_broadcast) {
  bool changed = false;
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
    CHECK_EQ(user->operand(reshape_or_broadcast_operand_index),
             reshape_or_broadcast);
    std::vector<HloInstruction*> new_user_operands = user->operands();
    new_user_operands[reshape_or_broadcast_operand_index] = operand;
    auto new_user = computation_->AddInstruction(user->CloneWithNewOperands(
        ShapeUtil::MakeShape(user->shape().element_type(),
                             AsInt64Slice(operand->shape().dimensions())),
        new_user_operands));
    HloInstruction* new_reshape_or_broadcast = nullptr;
    if (reshape_or_broadcast->opcode() == HloOpcode::kReshape) {
      new_reshape_or_broadcast =
          computation_->AddInstruction(HloInstruction::CreateReshape(
              ShapeUtil::MakeShape(
                  user->shape().element_type(),
                  AsInt64Slice(reshape_or_broadcast->shape().dimensions())),
              new_user));
    } else {
      TF_RET_CHECK(reshape_or_broadcast->opcode() == HloOpcode::kBroadcast);
      new_reshape_or_broadcast =
          computation_->AddInstruction(HloInstruction::CreateBroadcast(
              ShapeUtil::MakeShape(
                  user->shape().element_type(),
                  AsInt64Slice(reshape_or_broadcast->shape().dimensions())),
              new_user, reshape_or_broadcast->dimensions()));
    }
    TF_RETURN_IF_ERROR(
        computation_->ReplaceUsesOfInstruction(user, new_reshape_or_broadcast));
    changed = true;
  }
  return changed;
}

Status AlgebraicSimplifierVisitor::HandleReshape(HloInstruction* reshape) {
  auto operand = reshape->mutable_operand(0);

  // Delete no-op reshapes, i.e. where shape = operand shape.
  if (SameShape(reshape, operand)) {
    VLOG(10) << "deleting no-op reshape";
    changed_ = true;
    return computation_->ReplaceInstruction(reshape, operand);
  }

  // Merge reshapes.
  if (HloOpcode::kReshape == operand->opcode()) {
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(
        reshape, HloInstruction::CreateReshape(reshape->shape(),
                                               operand->mutable_operand(0)));
  }

  if (HloOpcode::kBroadcast == reshape->operand(0)->opcode()) {
    auto opt_dims = ReshapeLeavesDimensionsUnmodified(
        reshape, reshape->operand(0)->dimensions());
    if (opt_dims.first) {
      changed_ = true;
      return computation_->ReplaceWithNewInstruction(
          reshape,
          HloInstruction::CreateBroadcast(
              reshape->shape(), reshape->mutable_operand(0)->mutable_operand(0),
              opt_dims.second));
    }
  }

  // A Reshape that feeds a unary element-wise operation can sink the
  // reshape after the unary element-wise operation.
  TF_ASSIGN_OR_RETURN(
      changed_,
      TryToSinkReshapeOrBroadcastAfterOpWithUniqueNonScalarOperand(reshape));
  if (changed_) {
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

Status AlgebraicSimplifierVisitor::HandleSlice(HloInstruction* slice,
                                               HloInstruction* operand) {
  // Delete no-op slices, i.e. where shape = operand shape.
  if (ReplaceInstructionIfSameShape(slice, operand)) {
    return Status::OK();
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleReduce(
    HloInstruction* reduce, HloInstruction* arg, HloInstruction* init_value,
    tensorflow::gtl::ArraySlice<int64> dimensions, HloComputation* function) {
  if (ShapeUtil::HasZeroElements(arg->shape()) ||
      ShapeUtil::HasZeroElements(reduce->shape())) {
    return computation_->ReplaceWithNewInstruction(
        reduce,
        HloInstruction::CreateBroadcast(reduce->shape(), init_value, {}));
    return Status::OK();
  }
  // A Transpose feeding a reduce can simply permute the reduction dimensions
  // field.
  if (arg->opcode() == HloOpcode::kTranspose) {
    auto transpose_dimensions = arg->dimensions();
    std::vector<int64> new_reduce_dimensions;
    for (auto dim : dimensions) {
      new_reduce_dimensions.push_back(transpose_dimensions[dim]);
    }
    return computation_->ReplaceWithNewInstruction(
        reduce, HloInstruction::CreateReduce(
                    reduce->shape(), arg->mutable_operand(0), init_value,
                    new_reduce_dimensions, function));
  }

  // A reshape that collapses multiple dimensions into a dimension being reduced
  // can just reduce all of those dimensions instead of doing a collapsing
  // reshape before a reduction.
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
      return computation_->ReplaceWithNewInstruction(
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
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(
        reduce, HloInstruction::CreateMap(reduce->shape(),
                                          {reshape, init_value}, function));
  }
  return Status::OK();
}

Status AlgebraicSimplifierVisitor::HandleTranspose(HloInstruction* transpose) {
  auto operand = transpose->mutable_operand(0);

  if (std::is_sorted(transpose->dimensions().begin(),
                     transpose->dimensions().end())) {
    VLOG(10) << "deleting no-op transpose";
    changed_ = true;
    return computation_->ReplaceInstruction(transpose, operand);
  }

  if (HloOpcode::kTranspose == operand->opcode()) {
    changed_ = true;
    return computation_->ReplaceWithNewInstruction(
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
    HloInstruction* convolution, HloInstruction* lhs, HloInstruction* rhs,
    const Window& window) {
  // HandleConvolution tries to replace a convolution with a DOT instruction.
  //
  // Only add when bitcasts can be used:
  // - if bitcasts are not supported, then reshapes could be used but will
  //   end up with another copy.
  // - if bitcasts are supported, the simplifier will be called again with
  //   bitcasts_ == true.

  // TODO(cwhipkey): b/31337498, make this layout insensitive.
  if (!is_layout_sensitive_) return Status::OK();

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
      input_shape.layout().minor_to_major(0) != dnums.feature_dimension() ||
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
      input_shape.dimensions(dnums.feature_dimension());
  const int64 output_channels =
      filter_shape.dimensions(dnums.kernel_output_feature_dimension());

  // Computes the product of the non-feature dimensions.
  int64 conv_width = 1;
  for (int i = 0; i < input_shape.dimensions_size(); ++i) {
    if (i != dnums.feature_dimension()) {
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
  if (!valid_bitcast_callback_(lhs->shape(), input_shape) ||
      !valid_bitcast_callback_(rhs->shape(), new_filter_shape) ||
      !valid_bitcast_callback_(dot_output_shape, convolution_shape)) {
    return Status::OK();
  }

  auto new_lhs = add_bitcast(new_input_shape, lhs);
  auto new_rhs = add_bitcast(new_filter_shape, rhs);
  auto dot = computation_->AddInstruction(HloInstruction::CreateBinary(
      dot_output_shape, HloOpcode::kDot, new_lhs, new_rhs));
  changed_ = true;
  return computation_->ReplaceInstruction(convolution,
                                          add_bitcast(convolution_shape, dot));
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
  TF_CHECK_OK(computation_->ReplaceWithNewInstruction(root, std::move(clamp)));
  changed_ = true;
  return true;
}

Status AlgebraicSimplifierVisitor::HandleMaximum(HloInstruction* maximum,
                                                 HloInstruction* lhs,
                                                 HloInstruction* rhs) {
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

Status AlgebraicSimplifierVisitor::HandleMinimum(HloInstruction* minimum,
                                                 HloInstruction* lhs,
                                                 HloInstruction* rhs) {
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
  bool changed =
      std::any_of(module->computations().begin(), module->computations().end(),
                  [=](const std::unique_ptr<HloComputation>& computation) {
                    return AlgebraicSimplifierVisitor::Run(
                        computation.get(), is_layout_sensitive_,
                        valid_bitcast_callback_, enable_dot_simplification_);
                  });
  XLA_VLOG_LINES(2,
                 "AlgebraicSimplifier::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla

/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/layout_assignment.h"

#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status.h"

namespace xla {
namespace {

namespace m = xla::match;
using ::testing::ElementsAre;

class LayoutAssignmentTest : public HloTestBase {
 protected:
  void AssignLayouts(HloModule* m, ComputationLayout* entry_computation_layout,
                     ChannelLayoutConstraints* channel_constraints = nullptr) {
    LayoutAssignment layout_assignment(
        entry_computation_layout,
        /*channel_constraints=*/channel_constraints);
    EXPECT_IS_OK(layout_assignment.Run(m).status());
  }

  std::vector<int64_t> LayoutOf(HloModule* m, absl::string_view name) {
    HloInstruction* instr = FindInstruction(m, name);
    CHECK(instr != nullptr) << name;
    auto minor_to_major = instr->shape().layout().minor_to_major();
    return std::vector<int64_t>(minor_to_major.begin(), minor_to_major.end());
  }

  void ExpectLayoutIs(const Shape& shape,
                      absl::Span<const int64_t> minor_to_major) {
    const Layout expected = LayoutUtil::MakeLayout(minor_to_major);
    EXPECT_TRUE(LayoutUtil::Equal(shape.layout(), expected))
        << "Expected layout " << expected << ", actual " << shape.layout();
  }

  void ExpectTupleLayoutIs(
      const Shape& shape,
      std::initializer_list<absl::Span<const int64_t>> minor_to_majors) {
    int i = 0;
    for (const absl::Span<const int64_t> minor_to_major : minor_to_majors) {
      const Layout expected = LayoutUtil::MakeLayout(minor_to_major);
      const Layout& actual = ShapeUtil::GetTupleElementShape(shape, i).layout();
      EXPECT_TRUE(LayoutUtil::Equal(actual, expected))
          << "Expected tuple element " << i << " layout " << expected
          << ", actual " << actual;
      ++i;
    }
  }
};

TEST_F(LayoutAssignmentTest, ComputationLayout) {
  // Verify the layouts of the root and parameter instructions of a computation
  // match the ComputationLayout for two different layouts.
  std::vector<std::vector<int64_t>> minor_to_majors = {{0, 1}, {1, 0}};
  for (auto& minor_to_major : minor_to_majors) {
    auto builder = HloComputation::Builder(TestName());
    Shape ashape = ShapeUtil::MakeShape(F32, {42, 12});
    auto param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, ashape, "param0"));
    auto param1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, ashape, "param1"));
    auto add = builder.AddInstruction(
        HloInstruction::CreateBinary(ashape, HloOpcode::kAdd, param0, param1));
    auto m = CreateNewVerifiedModule();
    HloComputation* computation = m->AddEntryComputation(builder.Build());

    Layout layout = LayoutUtil::MakeLayout(minor_to_major);
    Shape shape(ashape);
    *shape.mutable_layout() = layout;
    const ShapeLayout shape_layout(shape);

    ComputationLayout computation_layout(computation->ComputeProgramShape());
    *computation_layout.mutable_parameter_layout(0) = shape_layout;
    *computation_layout.mutable_parameter_layout(1) = shape_layout;
    *computation_layout.mutable_result_layout() = shape_layout;
    AssignLayouts(m.get(), &computation_layout);
    EXPECT_TRUE(LayoutUtil::Equal(layout, param0->shape().layout()));
    EXPECT_TRUE(LayoutUtil::Equal(layout, param1->shape().layout()));
    EXPECT_TRUE(LayoutUtil::Equal(layout, add->shape().layout()));
  }
}

TEST_F(LayoutAssignmentTest, ComputationLayoutMixedLayout) {
  // Verify the layouts of the root and parameter instructions of a computation
  // match the ComputationLayout which has mixed layout.
  auto builder = HloComputation::Builder(TestName());
  Shape ashape = ShapeUtil::MakeShape(F32, {42, 12});
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ashape, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, ashape, "param1"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(ashape, HloOpcode::kAdd, param0, param1));
  auto m = CreateNewVerifiedModule();
  HloComputation* computation = m->AddEntryComputation(builder.Build());

  Layout col_major_layout = LayoutUtil::MakeLayout({1, 0});
  Shape col_major_shape(ashape);
  *col_major_shape.mutable_layout() = col_major_layout;
  const ShapeLayout col_major(col_major_shape);

  Layout row_major_layout = LayoutUtil::MakeLayout({0, 1});
  Shape row_major_shape(ashape);
  *row_major_shape.mutable_layout() = row_major_layout;
  const ShapeLayout row_major(row_major_shape);

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) = col_major;
  *computation_layout.mutable_parameter_layout(1) = row_major;
  *computation_layout.mutable_result_layout() = col_major;

  AssignLayouts(m.get(), &computation_layout);
  EXPECT_TRUE(LayoutUtil::Equal(col_major_layout, param0->shape().layout()));
  EXPECT_TRUE(LayoutUtil::Equal(row_major_layout, param1->shape().layout()));
  EXPECT_TRUE(LayoutUtil::Equal(
      col_major_layout, computation->root_instruction()->shape().layout()));
}

TEST_F(LayoutAssignmentTest, FusionInstruction) {
  // Verify that the layout of the fused parameters in a fusion instruction
  // match that of the fusion operands. Other fused instructions should have no
  // layout.
  std::vector<std::vector<int64_t>> minor_to_majors = {{0, 1}, {1, 0}};
  for (auto& minor_to_major : minor_to_majors) {
    auto builder = HloComputation::Builder(TestName());
    auto constant_literal1 = LiteralUtil::CreateR2WithLayout<float>(
        {{1.0, 2.0}, {3.0, 4.0}}, LayoutUtil::MakeLayout(minor_to_major));
    auto constant_literal2 = LiteralUtil::CreateR2WithLayout<float>(
        {{5.0, 6.0}, {7.0, 8.0}}, LayoutUtil::MakeLayout(minor_to_major));
    Shape ashape = constant_literal1.shape();

    auto constant1 = builder.AddInstruction(
        HloInstruction::CreateConstant(std::move(constant_literal1)));
    auto constant2 = builder.AddInstruction(
        HloInstruction::CreateConstant(std::move(constant_literal2)));
    auto add = builder.AddInstruction(HloInstruction::CreateBinary(
        ashape, HloOpcode::kAdd, constant1, constant2));
    auto negate1 = builder.AddInstruction(
        HloInstruction::CreateUnary(ashape, HloOpcode::kNegate, add));
    auto negate2 = builder.AddInstruction(
        HloInstruction::CreateUnary(ashape, HloOpcode::kNegate, negate1));

    auto m = CreateNewVerifiedModule();
    HloComputation* computation = m->AddEntryComputation(builder.Build());

    auto fusion = computation->CreateFusionInstruction(
        {negate2, negate1, add}, HloInstruction::FusionKind::kLoop);

    Layout layout = LayoutUtil::MakeLayout(minor_to_major);
    Shape shape(ashape);
    *shape.mutable_layout() = layout;
    const ShapeLayout shape_layout(shape);

    ComputationLayout computation_layout(computation->ComputeProgramShape());
    *computation_layout.mutable_result_layout() = shape_layout;

    AssignLayouts(m.get(), &computation_layout);

    EXPECT_TRUE(LayoutUtil::Equal(
        layout, fusion->fused_parameter(0)->shape().layout()));
    EXPECT_TRUE(LayoutUtil::Equal(
        layout, fusion->fused_parameter(1)->shape().layout()));
    EXPECT_TRUE(LayoutUtil::Equal(
        layout, fusion->fused_expression_root()->shape().layout()));

    // Inner fused node should not have layout.
    EXPECT_FALSE(LayoutUtil::HasLayout(
        fusion->fused_expression_root()->operand(0)->shape()));
  }
}

TEST_F(LayoutAssignmentTest, TupleLayout) {
  // Verify the layouts of a tuple are assigned properly (the element layouts
  // match their source).
  auto builder = HloComputation::Builder(TestName());
  auto constant0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2WithLayout<float>(
          {{1.0, 2.0}, {3.0, 4.0}}, LayoutUtil::MakeLayout({0, 1}))));
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2WithLayout<float>(
          {{1.0, 2.0}, {3.0, 4.0}}, LayoutUtil::MakeLayout({1, 0}))));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant0, constant1}));

  // To avoid having to construct a tuple layout in the ComputationLayout below,
  // make the result of the instruction be an array.
  auto get_element0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(constant0->shape(), tuple, 0));
  auto negate = builder.AddInstruction(HloInstruction::CreateUnary(
      constant0->shape(), HloOpcode::kNegate, get_element0));

  auto m = CreateNewVerifiedModule();
  m->AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  AssignLayouts(m.get(), &computation_layout);

  EXPECT_TRUE(
      LayoutUtil::LayoutsInShapesEqual(constant0->shape(), constant1->shape()));

  EXPECT_TRUE(LayoutUtil::HasLayout(tuple->shape()));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(
      negate->shape(), computation_layout.result_layout().shape()));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(
      ShapeUtil::GetTupleElementShape(tuple->shape(), 1), constant1->shape()));
}

TEST_F(LayoutAssignmentTest, ConflictingLayoutTuple) {
  // Construct following computation which has conflicting layouts for two
  // elements of a tuple which share the same source logicalb buffer:
  //
  // %constant = Constant(...)
  // %inner_tuple = Tuple(%constant)
  // %nested_tuple = Tuple(%inner_tuple, %inner_tuple)
  //
  // Result layout col-major for the first element and row-major for the
  // second. This results in the conflict where the element of the inner_tuple
  // needs to be both col and row major. This is resolved by deep-copying the
  // tuple and assigning the layouts of the copied arrays as needed.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));
  auto inner_tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({constant}));
  auto nested_tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({inner_tuple, inner_tuple}));

  auto m = CreateNewVerifiedModule();
  m->AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  Shape result_shape = nested_tuple->shape();
  *ShapeUtil::GetMutableSubshape(&result_shape, /*index=*/{0, 0}) =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 2}, {1, 0});
  *ShapeUtil::GetMutableSubshape(&result_shape, /*index=*/{1, 0}) =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 2}, {0, 1});
  TF_CHECK_OK(computation_layout.mutable_result_layout()->CopyLayoutFromShape(
      result_shape));

  LayoutAssignment layout_assignment(&computation_layout);
  AssignLayouts(m.get(), &computation_layout);

  // Layout assignment should have deep copied the result of the computation to
  // address the layout conflict. This results in several Tuple() and
  // GetTupleElement() instructions. Running algebraic simplification should
  // clean up the code to something like:
  //
  //  %constant = Constant(...) layout={1,0}
  //  %tuple.0 = Tuple(%constant) layout=({1,0})
  //  %copy = Copy(%constant) layout={0,1}  # layout transposed
  //  %tuple.1 = Tuple(%copy) layout=({0,1})
  //  %tuple.2 = Tuple(%tuple.0, %tuple.1) layout=(({1,0}), ({0,1}))
  //
  AlgebraicSimplifierOptions options(
      [](const Shape&, const Shape&) { return false; });
  options.set_is_layout_sensitive(true);
  EXPECT_TRUE(AlgebraicSimplifier(options).Run(m.get()).value());
  HloInstruction* root = m->entry_computation()->root_instruction();
  // Verify layout of the root and the root's operands.
  EXPECT_TRUE(ShapeUtil::Equal(result_shape, root->shape()));
  EXPECT_TRUE(ShapeUtil::Equal(ShapeUtil::GetSubshape(result_shape, {0}),
                               root->operand(0)->shape()));
  EXPECT_TRUE(ShapeUtil::Equal(ShapeUtil::GetSubshape(result_shape, {1}),
                               root->operand(1)->shape()));

  // Verify the structure of the HLO graph.
  EXPECT_THAT(root,
              GmockMatch(m::Tuple(m::Tuple(m::Op().Is(constant)),
                                  m::Tuple(m::Copy(m::Op().Is(constant))))));
}

TEST_F(LayoutAssignmentTest, ElementwiseAndReshape) {
  // param -> log -> reshape -> tanh
  auto builder = HloComputation::Builder(TestName());
  Shape ashape = ShapeUtil::MakeShape(F32, {1, 2, 3, 1});
  Shape bshape = ShapeUtil::MakeShape(F32, {3, 1, 2});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ashape, "param"));
  auto log = builder.AddInstruction(
      HloInstruction::CreateUnary(ashape, HloOpcode::kLog, param));
  auto reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(bshape, log));
  auto tanh = builder.AddInstruction(
      HloInstruction::CreateUnary(bshape, HloOpcode::kTanh, reshape));

  auto m = CreateNewVerifiedModule();
  HloComputation* computation = m->AddEntryComputation(builder.Build(tanh));

  Shape ashape_with_layout(ashape);
  Shape bshape_with_layout(bshape);
  *ashape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({0, 2, 1, 3});
  *bshape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({2, 1, 0});

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(ashape_with_layout);
  *computation_layout.mutable_result_layout() = ShapeLayout(bshape_with_layout);
  AssignLayouts(m.get(), &computation_layout);

  auto log_minor_to_major = log->shape().layout().minor_to_major();
  EXPECT_GT(PositionInContainer(log_minor_to_major, 1),
            PositionInContainer(log_minor_to_major, 2));

  auto reshape_minor_to_major = reshape->shape().layout().minor_to_major();
  EXPECT_GT(PositionInContainer(reshape_minor_to_major, 0),
            PositionInContainer(reshape_minor_to_major, 2));
}

// Test whether LayoutAssignment assigns layouts to elementwise operations to
// keep linear indices valid across them, and to transpositions to make them
// bitcasts.
TEST_F(LayoutAssignmentTest, ElementwiseAndTranspose) {
  // param -> log -> transpose -> tanh
  auto builder = HloComputation::Builder(TestName());
  Shape ashape = ShapeUtil::MakeShape(F32, {42, 12});
  Shape bshape = ShapeUtil::MakeShape(F32, {12, 42});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ashape, "param"));
  auto log = builder.AddInstruction(
      HloInstruction::CreateUnary(ashape, HloOpcode::kLog, param));
  auto transpose = builder.AddInstruction(
      HloInstruction::CreateTranspose(bshape, log, {1, 0}));
  auto tanh = builder.AddInstruction(
      HloInstruction::CreateUnary(bshape, HloOpcode::kTanh, transpose));
  auto m = CreateNewVerifiedModule();
  auto computation = m->AddEntryComputation(builder.Build(tanh));

  Shape ashape_with_layout(ashape);
  Shape bshape_with_layout(bshape);
  *ashape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  *bshape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(ashape_with_layout);
  *computation_layout.mutable_result_layout() = ShapeLayout(bshape_with_layout);
  AssignLayouts(m.get(), &computation_layout);

  EXPECT_TRUE(
      LayoutUtil::Equal(ashape_with_layout.layout(), log->shape().layout()));
  EXPECT_TRUE(LayoutUtil::Equal(bshape_with_layout.layout(),
                                transpose->shape().layout()));
  EXPECT_TRUE(
      LayoutUtil::Equal(bshape_with_layout.layout(), tanh->shape().layout()));
}

// Test whether LayoutAssignment assigns layouts to transpositions to make them
// bitcasts.
TEST_F(LayoutAssignmentTest, BroadcastAndTranspose) {
  // param -> broadcast -> transpose
  auto builder = HloComputation::Builder(TestName());
  Shape ashape = ShapeUtil::MakeShape(F32, {3, 4});
  Shape bshape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  Shape cshape = ShapeUtil::MakeShape(F32, {4, 3, 2});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ashape, "param"));
  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(bshape, param, {1, 2}));
  auto transpose = builder.AddInstruction(
      HloInstruction::CreateTranspose(cshape, broadcast, {2, 1, 0}));
  auto m = CreateNewVerifiedModule();
  HloComputation* computation =
      m->AddEntryComputation(builder.Build(transpose));

  Shape input_shape_with_layout(ashape);
  Shape output_shape_with_layout(cshape);
  *input_shape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  *output_shape_with_layout.mutable_layout() =
      LayoutUtil::MakeLayout({2, 1, 0});

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(input_shape_with_layout);
  *computation_layout.mutable_result_layout() =
      ShapeLayout(output_shape_with_layout);
  AssignLayouts(m.get(), &computation_layout);

  EXPECT_THAT(broadcast->shape().layout().minor_to_major(),
              ElementsAre(0, 1, 2));
}

TEST_F(LayoutAssignmentTest, ReshapeOperandHasMultipleUsers) {
  // param[4] -> broadcast[3x4] ------> transpose[4x3]-------- -------> tuple
  //                            \                                     /
  //                             \-> tanh[3x4] -> broadcast2[2x3x4] -/
  //
  // The layout of `transpose` is set to {1,0} because it provides a buffer to
  // the computation result which has a fixed layout.. Therefore, `broadcast`
  // (the operand of transpose) is expected to have layout {0,1} so that the
  // transpose is a bitcast. Furthermore, `tanh` is expected to have the same
  // layout as `broadcast` (i.e. {0,1}) because `tanh` is elementwise.
  Shape f32_4 = ShapeUtil::MakeShape(F32, {4});
  Shape f32_34 = ShapeUtil::MakeShape(F32, {3, 4});
  Shape f32_43 = ShapeUtil::MakeShape(F32, {4, 3});
  Shape f32_234 = ShapeUtil::MakeShape(F32, {2, 3, 4});

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_4, "param"));
  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(f32_34, param, {1}));
  auto transpose = builder.AddInstruction(
      HloInstruction::CreateTranspose(f32_43, broadcast, {1, 0}));
  auto tanh = builder.AddInstruction(
      HloInstruction::CreateUnary(f32_34, HloOpcode::kTanh, broadcast));
  auto broadcast2 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(f32_234, tanh, {1, 2}));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({transpose, broadcast2}));
  auto m = CreateNewVerifiedModule();
  HloComputation* computation = m->AddEntryComputation(builder.Build(tuple));

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  Shape param_shape_with_layout(f32_4);
  Shape transpose_shape_with_layout(f32_43);
  Shape broadcast2_shape_with_layout(f32_234);
  *param_shape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({0});
  *transpose_shape_with_layout.mutable_layout() =
      LayoutUtil::MakeLayout({1, 0});
  *broadcast2_shape_with_layout.mutable_layout() =
      LayoutUtil::MakeLayout({2, 1, 0});

  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(param_shape_with_layout);
  *computation_layout.mutable_result_layout() =
      ShapeLayout(ShapeUtil::MakeTupleShape(
          {transpose_shape_with_layout, broadcast2_shape_with_layout}));
  AssignLayouts(m.get(), &computation_layout);
  EXPECT_THAT(broadcast->shape().layout().minor_to_major(), ElementsAre(0, 1));
  EXPECT_THAT(transpose->shape().layout().minor_to_major(), ElementsAre(1, 0));
  EXPECT_THAT(tanh->shape().layout().minor_to_major(), ElementsAre(0, 1));
}

class OperandsMustBeTheSameLayoutAssignment : public LayoutAssignment {
 public:
  explicit OperandsMustBeTheSameLayoutAssignment(
      ComputationLayout* entry_computation_layout)
      : LayoutAssignment(entry_computation_layout) {}

 protected:
  Status PropagateBufferConstraint(
      const BufferLayoutConstraint& buffer_constraint,
      LayoutConstraints* constraints) override {
    const LogicalBuffer& buffer = buffer_constraint.buffer();
    const HloInstruction* instruction = buffer.instruction();

    // Force the operands' layout to the output layout.
    for (int64_t operand_no = 0; operand_no < instruction->operand_count();
         ++operand_no) {
      const HloInstruction* operand = instruction->operand(operand_no);
      if (instruction->shape().rank() != operand->shape().rank()) {
        continue;
      }
      TF_RETURN_IF_ERROR(SetArrayOperandLayout(buffer_constraint.layout(),
                                               instruction, operand_no,
                                               /*mandatory=*/true));
    }
    return PropagateBufferConstraintToUses(buffer_constraint, constraints);
  }
};

TEST_F(LayoutAssignmentTest, MakeOperandsTheSame) {
  // param0 -> concatenate -> reshape
  // param1   -^
  auto builder = HloComputation::Builder(TestName());
  Shape ashape = ShapeUtil::MakeShape(F32, {50, 1});
  Shape bshape = ShapeUtil::MakeShape(F32, {50, 2});
  Shape cshape = ShapeUtil::MakeShape(F32, {100});
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ashape, "param"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, ashape, "param"));
  auto concatenate = builder.AddInstruction(
      HloInstruction::CreateConcatenate(bshape, {param0, param1}, 1));
  auto reshape = builder.AddInstruction(
      HloInstruction::CreateReshape(cshape, concatenate));
  auto m = CreateNewVerifiedModule();
  HloComputation* computation = m->AddEntryComputation(builder.Build(reshape));

  Shape param0_shape_with_layout(ashape);
  Shape param1_shape_with_layout(ashape);
  *param0_shape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  *param1_shape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({1, 0});

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(param0_shape_with_layout);
  *computation_layout.mutable_parameter_layout(1) =
      ShapeLayout(param1_shape_with_layout);
  OperandsMustBeTheSameLayoutAssignment layout_assignment(&computation_layout);
  EXPECT_IS_OK(layout_assignment.Run(m.get()).status());

  EXPECT_EQ(concatenate->operand(0)->shape().layout().minor_to_major(),
            concatenate->operand(1)->shape().layout().minor_to_major());
  EXPECT_EQ(concatenate->shape().layout().minor_to_major(),
            concatenate->operand(1)->shape().layout().minor_to_major());
}

// Test layout assignment of a transpose into a bitcast based on its operand.
TEST_F(LayoutAssignmentTest, TransposeToBitcastFromOperand) {
  auto builder = HloComputation::Builder(TestName());
  Shape input_shape_with_layout =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {3, 5, 6, 7}, {2, 0, 3, 1});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape_with_layout, "param"));
  auto transpose = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {6, 7, 3, 5}), param, {2, 3, 0, 1}));
  auto m = CreateNewVerifiedModule();
  HloComputation* computation =
      m->AddEntryComputation(builder.Build(transpose));
  ComputationLayout computation_layout(computation->ComputeProgramShape());
  AssignLayouts(m.get(), &computation_layout);
  EXPECT_TRUE(ShapeUtil::TransposeIsBitcast(transpose->operand(0)->shape(),
                                            transpose->shape(), {2, 3, 0, 1}));
}
// Test layout assignment of a transpose into a bitcast based on its user.
TEST_F(LayoutAssignmentTest, TransposeToBitcastToUser) {
  auto builder = HloComputation::Builder(TestName());
  Shape input_shape = ShapeUtil::MakeShape(F32, {3, 5, 6, 7});
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(input_shape, constant, {}));
  auto transpose = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {6, 7, 3, 5}), broadcast, {2, 3, 0, 1}));
  auto m = CreateNewVerifiedModule();
  HloComputation* computation =
      m->AddEntryComputation(builder.Build(transpose));
  ComputationLayout computation_layout(computation->ComputeProgramShape());
  AssignLayouts(m.get(), &computation_layout);
  EXPECT_TRUE(ShapeUtil::TransposeIsBitcast(transpose->operand(0)->shape(),
                                            transpose->shape(), {2, 3, 0, 1}));
}

// TransposeIsBitcast shouldn't be called without layout information.
TEST_F(LayoutAssignmentTest, TransposeIsBitcastFail) {
  auto builder = HloComputation::Builder(TestName());
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 2, 2});
  Shape input_shape_with_layout(input_shape);
  *input_shape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({2, 1, 0});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape_with_layout, "param"));
  auto hlo = builder.AddInstruction(
      HloInstruction::CreateTranspose(input_shape, param, {0, 2, 1}));
  // Clear the default layout assigned to the instruction.
  LayoutUtil::ClearLayout(hlo->mutable_shape());
  EXPECT_DEATH(ShapeUtil::TransposeIsBitcast(hlo->operand(0)->shape(),
                                             hlo->shape(), hlo->dimensions()),
               "has_layout");
}

// ReshapeIsBitcast shouldn't be called without layout information.
TEST_F(LayoutAssignmentTest, ReshapeIsBitcastFail) {
  auto builder = HloComputation::Builder(TestName());
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 2, 2});
  Shape input_shape_with_layout(input_shape);
  *input_shape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({2, 1, 0});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape_with_layout, "param"));
  auto hlo =
      builder.AddInstruction(HloInstruction::CreateReshape(input_shape, param));
  // Clear the default layout assigned to the instruction.
  LayoutUtil::ClearLayout(hlo->mutable_shape());
  EXPECT_DEATH(
      ShapeUtil::ReshapeIsBitcast(hlo->operand(0)->shape(), hlo->shape()),
      "has_layout");
}

// Check that the computation below doesn't crash the compiler.
//
// Within a fusion computation, only the parameters and result get assigned a
// layout.  When we run the algebraic simplifier on this computation post layout
// assignment, it should not call TransposeIsBitcast on the `transpose` node
// inside the fusion computation as TransposeIsBitcast checks both input_shape
// and output_shape have layouts.
TEST_F(LayoutAssignmentTest, TransposeWithinFusionDoesNotCrash) {
  const char* module_str = R"(
    HloModule test_module

    fused_computation {
      param_1 = f32[2,2,2]{2,1,0} parameter(1)
      transpose = f32[2,2,2]{2,1,0} transpose(param_1), dimensions={0,2,1}
      reduce_1 = f32[] parameter(0)
      broadcast_1 = f32[2,2,2]{2,1,0} broadcast(reduce_1), dimensions={}
      ROOT divide_1 = f32[2,2,2]{2,1,0} divide(transpose, broadcast_1)
    }

    ENTRY entry_computation {
      fusion.1 = f32[2,2,2]{2,1,0} parameter(1)
      reduce.1 = f32[] parameter(0)
      fusion.2 = f32[2,2,2]{2,1,0} fusion(reduce.1, fusion.1), kind=kLoop, calls=fused_computation
     ROOT tuple.1 = (f32[2,2,2]{2,1,0}) tuple(fusion.2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  std::unique_ptr<HloModule> compiled_module =
      backend()
          .compiler()
          ->RunHloPasses(m->Clone(), backend().default_stream_executor(),
                         /*device_allocator=*/nullptr)
          .value();

  EXPECT_EQ(OkStatus(), backend()
                            .compiler()
                            ->RunBackend(std::move(compiled_module),
                                         backend().default_stream_executor(),
                                         /*device_allocator=*/nullptr)
                            .status());
}

// A GTE inside of a fusion node inherits the layout of its operand (which
// should, if we keep following operands, eventually be a parameter).
TEST_F(LayoutAssignmentTest, GTEInheritsLayoutFromOperand) {
  const char* module_str = R"(
    HloModule test_module

    fused_computation {
      fparam = (f32[2,2,2], (f32[2,2,2], f32[2,2,2])) parameter(0)
      gte0 = f32[2,2,2] get-tuple-element(fparam), index=0
      gte1 = (f32[2,2,2], f32[2,2,2]) get-tuple-element(fparam), index=1
      gte1a = f32[2,2,2] get-tuple-element(gte1), index=0
      gte1b = f32[2,2,2] get-tuple-element(gte1), index=1
      add = f32[2,2,2] add(gte1a, gte1b)
      ROOT fresult = f32[2,2,2] add(gte0, add)
    }

    ENTRY entry_computation {
      param = (f32[2,2,2], (f32[2,2,2], f32[2,2,2])) parameter(0)
      ROOT fusion =
        f32[2,2,2] fusion(param), kind=kLoop, calls=fused_computation
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  Shape param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 2, 2}, {0, 1, 2}),
       ShapeUtil::MakeTupleShape({
           ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 2, 2}, {1, 2, 0}),
           ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 2, 2}, {2, 0, 1}),
       })});
  TF_ASSERT_OK(
      computation_layout.mutable_parameter_layout(0)->CopyLayoutFromShape(
          param_shape));
  computation_layout.mutable_result_layout()->ResetLayout(
      LayoutUtil::MakeLayout({2, 1, 0}));
  AssignLayouts(m.get(), &computation_layout);

  EXPECT_THAT(LayoutOf(m.get(), "gte0"), ElementsAre(0, 1, 2));
  EXPECT_THAT(LayoutOf(m.get(), "gte1a"), ElementsAre(1, 2, 0));
  EXPECT_THAT(LayoutOf(m.get(), "gte1b"), ElementsAre(2, 0, 1));
  EXPECT_THAT(LayoutOf(m.get(), "fresult"), ElementsAre(2, 1, 0));
  EXPECT_THAT(FindInstruction(m.get(), "gte1")
                  ->shape()
                  .tuple_shapes(0)
                  .layout()
                  .minor_to_major(),
              ElementsAre(1, 2, 0));
  EXPECT_THAT(FindInstruction(m.get(), "gte1")
                  ->shape()
                  .tuple_shapes(1)
                  .layout()
                  .minor_to_major(),
              ElementsAre(2, 0, 1));
}

TEST_F(LayoutAssignmentTest, ConditionalAsymmetricLayout) {
  auto builder = HloComputation::Builder(TestName());
  auto m = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(F32, {128, 8});
  Shape tshape = ShapeUtil::MakeTupleShape({shape, shape});
  Shape result_tshape = ShapeUtil::MakeTupleShape({shape});

  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "param1"));
  auto pred = builder.AddInstruction(HloInstruction::CreateParameter(
      2, ShapeUtil::MakeShape(PRED, {}), "param2"));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({param0, param1}));

  auto true_builder = HloComputation::Builder(TestName() + "_TrueBranch");
  {
    auto param = true_builder.AddInstruction(
        HloInstruction::CreateParameter(0, tshape, "param"));
    auto gte0 = true_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(shape, param, 0));
    auto gte1 = true_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(shape, param, 1));
    auto add = true_builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, gte0, gte1));
    true_builder.AddInstruction(HloInstruction::CreateTuple({add}));
  }
  HloComputation* true_computation =
      m->AddEmbeddedComputation(true_builder.Build());

  auto false_builder = HloComputation::Builder(TestName() + "_FalseBranch");
  {
    Shape xshape = ShapeUtil::MakeShapeWithDenseLayout(F32, {128, 8}, {0, 1});
    false_builder.AddInstruction(
        HloInstruction::CreateParameter(0, tshape, "param"));
    // Using infeed as layout assignment does not mess up with it.
    auto token = false_builder.AddInstruction(HloInstruction::CreateToken());
    auto infeed = false_builder.AddInstruction(
        HloInstruction::CreateInfeed(xshape, token, ""));
    auto infeed_data = false_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(xshape, infeed, 0));
    false_builder.AddInstruction(HloInstruction::CreateTuple({infeed_data}));
  }
  HloComputation* false_computation =
      m->AddEmbeddedComputation(false_builder.Build());
  builder.AddInstruction(HloInstruction::CreateConditional(
      result_tshape, pred, tuple, true_computation, tuple, false_computation));

  HloComputation* computation = m->AddEntryComputation(builder.Build());
  ComputationLayout computation_layout(computation->ComputeProgramShape());

  AssignLayouts(m.get(), &computation_layout);

  const HloInstruction* true_root = true_computation->root_instruction();
  const HloInstruction* false_root = false_computation->root_instruction();
  EXPECT_THAT(true_root->opcode(), HloOpcode::kTuple);
  EXPECT_THAT(false_root->opcode(), HloOpcode::kTuple);

  const HloInstruction* true_result = true_root->operand(0);
  const HloInstruction* false_result = false_root->operand(0);
  EXPECT_TRUE(LayoutUtil::Equal(true_result->shape().layout(),
                                false_result->shape().layout()));
  EXPECT_THAT(false_result->opcode(), HloOpcode::kCopy);
}

TEST_F(LayoutAssignmentTest, LayoutAssignmentToTupleSiblingOperand) {
  const char* const hlo_string = R"(
  HloModule Module

  true_branch {
    tparam = (f64[2,3], f64[2,3]) parameter(0)
    ROOT tgte = f64[2,3] get-tuple-element(tparam), index=1
  }

  false_branch {
    ROOT Arg = f64[2,3] parameter(0)
  }

  ENTRY entry {
    p0 = (f64[2,3], f64[2,3])  parameter(0)
    p1 = f64[2,3] parameter(1)
    constant = pred[] constant(true)
    ROOT conditional = f64[2,3] conditional(constant, p0, p1),
      true_computation=true_branch, false_computation=false_branch
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  LayoutAssignment layout_assignment(&computation_layout);
  Status error_status = layout_assignment.Run(m.get()).status();
  EXPECT_TRUE(error_status.ok());
}

TEST_F(LayoutAssignmentTest, InternalErrorOnBitcast) {
  auto builder = HloComputation::Builder(TestName());
  auto constant0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2WithLayout<float>(
          {{1.0, 2.0}, {3.0, 4.0}}, LayoutUtil::MakeLayout({0, 1}))));
  builder.AddInstruction(
      HloInstruction::CreateBitcast(constant0->shape(), constant0));
  auto m = CreateNewVerifiedModule();
  m->AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  LayoutAssignment layout_assignment(&computation_layout);
  Status error_status = layout_assignment.Run(m.get()).status();
  EXPECT_FALSE(error_status.ok());
  EXPECT_THAT(
      error_status.message(),
      ::testing::HasSubstr(
          "Unexpected bitcast operation seen during layout assignment"));
}

TEST_F(LayoutAssignmentTest, ChannelLayoutMismatch) {
  // Pin non matching layouts to parameter and root.
  const char* module_str = R"(
    HloModule test_module

    ENTRY entry_computation {
      param = (f32[2,2]) parameter(0)
      gte = f32[2,2] get-tuple-element(param), index=0
      token0 = token[] after-all()
      recv = (f32[2,2], u32[], token[]) recv(token0), channel_id=1
      recv-done = (f32[2,2], token[]) recv-done(recv), channel_id=1
      ROOT root = f32[2,2] get-tuple-element(recv-done), index=0
      send = (f32[2,2], u32[], token[]) send(gte, token0), channel_id=1
      send-done = token[] send-done(send), channel_id=1
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  Shape param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 2}, {0, 1})});
  TF_ASSERT_OK(
      computation_layout.mutable_parameter_layout(0)->CopyLayoutFromShape(
          param_shape));
  computation_layout.mutable_result_layout()->ResetLayout(
      LayoutUtil::MakeLayout({1, 0}));

  ChannelLayoutConstraints channel_constraints;
  AssignLayouts(m.get(), &computation_layout, &channel_constraints);

  EXPECT_TRUE(ShapeUtil::Equal(FindInstruction(m.get(), "send")->shape(),
                               FindInstruction(m.get(), "recv")->shape()));
}

TEST_F(LayoutAssignmentTest, AllReduceLayoutMissmatch) {
  // Pin non matching layouts to parameter and root.
  const char* module_str = R"(
    HloModule test_module

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY entry_computation {
      param = (f32[2,2]) parameter(0)
      gte = f32[2,2] get-tuple-element(param), index=0
      ar.0 = f32[2,2] all-reduce(gte),
        channel_id=1, replica_groups={{0}}, to_apply=add,
        sharding={maximal device=0}
      const = f32[2,2] constant({{0,1},{2,3}})
      ROOT ar.1 = f32[2,2] all-reduce(const),
        channel_id=1, replica_groups={{0}}, to_apply=add,
        sharding={maximal device=1}
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  Shape param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 2}, {0, 1})});
  TF_ASSERT_OK(
      computation_layout.mutable_parameter_layout(0)->CopyLayoutFromShape(
          param_shape));
  computation_layout.mutable_result_layout()->ResetLayout(
      LayoutUtil::MakeLayout({1, 0}));

  ChannelLayoutConstraints channel_constraints;
  AssignLayouts(m.get(), &computation_layout, &channel_constraints);

  EXPECT_THAT(LayoutOf(m.get(), "gte"), ElementsAre(0, 1));
  EXPECT_THAT(LayoutOf(m.get(), "ar.0"), ElementsAre(0, 1));
  EXPECT_THAT(LayoutOf(m.get(), "ar.1"), ElementsAre(0, 1));
  const HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root->shape().layout().minor_to_major(), ElementsAre(1, 0));
}

TEST_F(LayoutAssignmentTest, CopySliceOperandToAvoidImplicitLayoutChange) {
  const char* module_str = R"(
    HloModule CopySliceOperandToAvoidImplicitLayoutChange

    ENTRY CopySliceOperandToAvoidImplicitLayoutChange {
      par0 = f32[3,4]{1,0} parameter(0)
      par1 = f32[4,5]{0,1} parameter(1)
      slice0 = f32[3,4] slice(par1), slice={[1:4],[1:5]}
      ROOT add0 = f32[3,4]{1,0} add(par0,slice0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  auto compiled_module =
      backend()
          .compiler()
          ->RunHloPasses(m->Clone(), backend().default_stream_executor(),
                         /*device_allocator=*/nullptr)
          .value();
  HloInstruction* root =
      compiled_module->entry_computation()->root_instruction();
  Shape shape_copy = ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 5}, {1, 0});
  EXPECT_THAT(
      root,
      GmockMatch(m::Add(
          m::Parameter(),
          m::Slice(m::Copy(m::Parameter(1)).WithShapeEqualTo(&shape_copy)))));
}

TEST_F(LayoutAssignmentTest, CopyDSliceOperandToAvoidImplicitLayoutChange) {
  const char* module_str = R"(
    HloModule CopyDSliceOperandToAvoidImplicitLayoutChange

    ENTRY CopyDSliceOperandToAvoidImplicitLayoutChange {
      par0 = f32[3,4]{1,0} parameter(0)
      par1 = f32[4,5]{0,1} parameter(1)
      par2 = s32[] parameter(2)
      par3 = s32[] parameter(3)
      dslice0 = f32[3,4] dynamic-slice(par1, par2, par3), dynamic_slice_sizes={3,4}
      ROOT add0 = f32[3,4]{1,0} add(par0,dslice0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  auto compiled_module =
      backend()
          .compiler()
          ->RunHloPasses(m->Clone(), backend().default_stream_executor(),
                         /*device_allocator=*/nullptr)
          .value();
  HloInstruction* root =
      compiled_module->entry_computation()->root_instruction();
  Shape shape_copy = ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 5}, {1, 0});
  EXPECT_THAT(root,
              GmockMatch(m::Add(
                  m::Parameter(),
                  m::DynamicSlice(
                      m::Copy(m::Parameter(1)).WithShapeEqualTo(&shape_copy),
                      m::Parameter(2), m::Parameter(3)))));
}

TEST_F(LayoutAssignmentTest, CopyConcatOperandToAvoidImplicitLayoutChange) {
  const char* module_str = R"(
    HloModule CopyConcatOperandToAvoidImplicitLayoutChange

    ENTRY CopyConcatOperandToAvoidImplicitLayoutChange {
      par0 = f32[3,8]{1,0} parameter(0)
      par1 = f32[3,5]{0,1} parameter(1)
      par2 = f32[3,3]{1,0} parameter(2)
      concat0 = f32[3,8] concatenate(f32[3,5] par1, f32[3,3] par2),
        dimensions={1}
      ROOT add0 = f32[3,8]{1,0} add(par0,concat0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  auto compiled_module =
      backend()
          .compiler()
          ->RunHloPasses(m->Clone(), backend().default_stream_executor(),
                         /*device_allocator=*/nullptr)
          .value();
  HloInstruction* root =
      compiled_module->entry_computation()->root_instruction();
  Shape shape_copy = ShapeUtil::MakeShapeWithDenseLayout(F32, {3, 5}, {1, 0});
  EXPECT_THAT(
      root,
      GmockMatch(m::Add(
          m::Parameter(),
          m::Concatenate(m::Copy(m::Parameter(1)).WithShapeEqualTo(&shape_copy),
                         m::Parameter(2)))));
}

TEST_F(LayoutAssignmentTest,
       ConvolutionOperandWithImplicitLayoutChangeNotCopied) {
  const char* module_str = R"(
    HloModule ConvolutionOperandWithImplicitLayoutChangeNotCopied

    ENTRY ConvolutionOperandWithImplicitLayoutChangeNotCopied {
      par0 = f32[128,3,230,230]{2,3,1,0} parameter(0)
      par1 = f32[7,7,3,64]{3,2,0,1} parameter(1)
      ROOT convolution0 = f32[128,64,112,112]{3,2,1,0} convolution(par0, par1),
        window={size=7x7 stride=2x2}, dim_labels=bf01_01io->bf01,
        feature_group_count=1
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  auto compiled_module =
      backend()
          .compiler()
          ->RunHloPasses(m->Clone(), backend().default_stream_executor(),
                         /*device_allocator=*/nullptr)
          .value();
  HloInstruction* root =
      compiled_module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              GmockMatch(m::Convolution(m::Parameter(0), m::Parameter(1))));
}

TEST_F(LayoutAssignmentTest, PropagatingLayoutFromResultToOperand) {
  const char* module_str = R"(
    HloModule PropagatingLayoutFromResultToOperand

    ENTRY PropagatingLayoutFromResultToOperand {
      par0 = f32[4,5]{1,0} parameter(0)
      ROOT slice0 = f32[3,4]{0,1} slice(par0), slice={[1:4],[1:5]}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  auto compiled_module =
      backend()
          .compiler()
          ->RunHloPasses(m->Clone(), backend().default_stream_executor(),
                         /*device_allocator=*/nullptr)
          .value();
  HloInstruction* root =
      compiled_module->entry_computation()->root_instruction();
  Shape shape_copy = ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 5}, {0, 1});
  EXPECT_THAT(root,
              GmockMatch(m::Slice(
                  m::Copy(m::Parameter(0)).WithShapeEqualTo(&shape_copy))));
}

TEST_F(LayoutAssignmentTest, TupleCopyOnLayoutMismatch) {
  // The first infeed uses layout {0,1}, while the second uses layout {1,0}.
  // The mismatch forces a copy of the tuple.  The tuple contains a token, so
  // layout assignment will fail if it tries to copy the whole tuple.
  const char* module_str = R"(
    HloModule TupleCopyOnLayoutMismatch

    condition.1 (tup: (s32[], token[], f32[512,1024]{0,1})) -> pred[] {
      tup.1 = (s32[], token[], f32[512,1024]{0,1}) parameter(0)
      counter.1 = s32[] get-tuple-element(tup.1), index=0
      five = s32[] constant(5)
      ROOT lt = pred[] compare(counter.1, five), direction=LT
    }

    body.2 (tup: (s32[], token[], f32[512,1024]{0,1})) -> (s32[], token[], f32[512,1024]{0,1}) {
      tup.2 = (s32[], token[], f32[512,1024]{0,1}) parameter(0)
      counter.2 = s32[] get-tuple-element(tup.2), index=0
      tok.2 = token[] get-tuple-element(tup.2), index=1

      ifeed.2 = (f32[512,1024]{1,0}, token[]) infeed(tok.2)
      next_tok = token[] get-tuple-element(ifeed.2), index=1
      next_buf = f32[512,1024]{1,0} get-tuple-element(ifeed.2), index=0

      one = s32[] constant(1)
      next_counter = s32[] add(counter.2, one)
      ROOT tup = (s32[], token[], f32[512,1024]{0,1}) tuple(next_counter, next_tok, next_buf)
    }

    ENTRY main () -> f32[512,1024]{0,1} {
      start_tok = token[] after-all()

      ifeed.3 = (f32[512,1024]{0,1}, token[]) infeed(start_tok)
      itok = token[] get-tuple-element(ifeed.3), index=1
      ibuf = f32[512,1024]{0,1} get-tuple-element(ifeed.3), index=0

      zero = s32[] constant(0)
      itup = (s32[], token[], f32[512,1024]{0,1}) tuple(zero, itok, ibuf)

      loop = (s32[], token[], f32[512,1024]{0,1}) while(itup), condition=condition.1, body=body.2
      ROOT result = f32[512,1024]{0,1} get-tuple-element(loop), index=2
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  // Sanity check to verify that there's a layout mismatch.
  EXPECT_THAT(LayoutOf(m.get(), "ibuf"), ElementsAre(0, 1));
  EXPECT_THAT(LayoutOf(m.get(), "next_buf"), ElementsAre(1, 0));

  AssignLayouts(m.get(), &computation_layout);
  SCOPED_TRACE(m->ToString());

  // Make sure that layout assignment did not magically eliminate the mismatch,
  // in which case the test didn't prove anything.
  Layout layout01 = LayoutUtil::MakeLayout({0, 1});
  const HloInstruction* loop = nullptr;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(
                  m::Op(&loop)
                      .WithOpcode(HloOpcode::kWhile)
                      .WithOperand(0, m::Tuple(m::Op(), m::Op(),
                                               m::Copy(m::Op().WithShape(
                                                   m::Shape().WithLayoutEqualTo(
                                                       &layout01))))))));

  Layout layout10 = LayoutUtil::MakeLayout({1, 0});
  EXPECT_THAT(loop->while_body()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::Op(), m::Op(),
                  m::Op().WithShape(m::Shape().WithLayoutEqualTo(&layout10)))));
}

TEST_F(LayoutAssignmentTest, CustomCallNotLayoutConstrained) {
  const char* module_str = R"(
HloModule CustomCallNotLayoutConstrained

ENTRY %CustomCallWithNotLayoutConstrained (p: f32[42,2,3]) -> f32[1,2,3,4] {
  %p = f32[42,2,3] parameter(0)
  ROOT %custom-call = f32[1,2,3,4] custom-call(f32[42,2,3] %p), custom_call_target="baz"
}
)";
  // Try with a couple different layouts. In each case the custom calls operand
  // and result layout should match that of the computation.
  {
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<VerifiedHloModule> m,
        ParseAndReturnVerifiedModule(module_str, GetModuleConfigForTest()));
    ComputationLayout computation_layout = m->entry_computation_layout();
    *computation_layout.mutable_parameter_layout(0) = ShapeLayout(
        ShapeUtil::MakeShapeWithDenseLayout(F32, {42, 2, 3}, {0, 2, 1}));
    *computation_layout.mutable_result_layout() = ShapeLayout(
        ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 2, 3, 4}, {3, 2, 0, 1}));
    AssignLayouts(m.get(), &computation_layout);

    HloInstruction* root = m->entry_computation()->root_instruction();
    ASSERT_THAT(root, GmockMatch(m::CustomCall(m::Parameter())));
    ExpectLayoutIs(root->shape(), {3, 2, 0, 1});
    ExpectLayoutIs(root->operand(0)->shape(), {0, 2, 1});
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<VerifiedHloModule> m,
        ParseAndReturnVerifiedModule(module_str, GetModuleConfigForTest()));
    ComputationLayout computation_layout = m->entry_computation_layout();
    *computation_layout.mutable_parameter_layout(0) = ShapeLayout(
        ShapeUtil::MakeShapeWithDenseLayout(F32, {42, 2, 3}, {0, 1, 2}));
    *computation_layout.mutable_result_layout() = ShapeLayout(
        ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 2, 3, 4}, {0, 2, 3, 1}));
    AssignLayouts(m.get(), &computation_layout);

    HloInstruction* root = m->entry_computation()->root_instruction();
    ASSERT_THAT(root, GmockMatch(m::CustomCall(m::Parameter())));
    ExpectLayoutIs(root->shape(), {0, 2, 3, 1});
    ExpectLayoutIs(root->operand(0)->shape(), {0, 1, 2});
  }
}

TEST_F(LayoutAssignmentTest, CustomCallLayoutConstrained) {
  const char* module_str = R"(
HloModule CustomCallLayoutConstrained

ENTRY %CustomCallWithLayoutConstraints (p0: f32[4,4], p1: f32[2,3]) -> f32[1,2,3,4] {
  %p0 = f32[4,4] parameter(0)
  %p1 = f32[2,3] parameter(1)
  ROOT %custom-call = f32[1,2,3,4]{3,2,0,1} custom-call(f32[4,4] %p0, f32[2,3] %p1), custom_call_target="baz", operand_layout_constraints={f32[4,4]{0,1}, f32[2,3]{1,0}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> m,
      ParseAndReturnVerifiedModule(module_str, GetModuleConfigForTest()));
  ComputationLayout computation_layout = m->entry_computation_layout();
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 4}, {1, 0}));
  *computation_layout.mutable_parameter_layout(1) =
      ShapeLayout(ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3}, {1, 0}));
  *computation_layout.mutable_result_layout() = ShapeLayout(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 2, 3, 4}, {2, 1, 0, 3}));
  AssignLayouts(m.get(), &computation_layout);

  // The custom call should be partially encapsulated in kCopy instructions
  // because of the layout mismatches.
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(m::CustomCall(m::Copy(), m::Parameter()))));

  const HloInstruction* custom_call =
      m->entry_computation()->root_instruction()->operand(0);
  ExpectLayoutIs(custom_call->shape(), {3, 2, 0, 1});
  ExpectLayoutIs(custom_call->operand(0)->shape(), {0, 1});
  ExpectLayoutIs(custom_call->operand(1)->shape(), {1, 0});
}

TEST_F(LayoutAssignmentTest, CustomCallLayoutConstrainedAliasedOutput) {
  const char* module_str = R"(
HloModule customcall.4

ENTRY %customcall.4 (parameter.1: f32[8,128], parameter.2: f32[8,128]) -> f32[8,128] {
  %parameter.1 = f32[8,128]{1,0} parameter(0)
  %parameter.2 = f32[8,128]{1,0} parameter(1)
  ROOT %custom-call.3 = f32[8,128]{1,0} custom-call(f32[8,128]{1,0} %parameter.1, f32[8,128]{1,0} %parameter.2), custom_call_target="gpu_example_custom_call", operand_layout_constraints={f32[8,128]{1,0}, f32[8,128]{1,0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> m,
      ParseAndReturnVerifiedModule(module_str, GetModuleConfigForTest()));
  ComputationLayout computation_layout = m->entry_computation_layout();
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 128}, {1, 0}));
  *computation_layout.mutable_parameter_layout(1) =
      ShapeLayout(ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 128}, {1, 0}));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 128}, {1, 0}));
  AssignLayouts(m.get(), &computation_layout);

  const HloInstruction* custom_call =
      m->entry_computation()->root_instruction();
  ExpectLayoutIs(custom_call->shape(), {1, 0});
  ExpectLayoutIs(custom_call->operand(0)->shape(), {1, 0});
  ExpectLayoutIs(custom_call->operand(1)->shape(), {1, 0});
}

TEST_F(LayoutAssignmentTest, CustomCallLayoutConstrainedZeroOperands) {
  const char* module_str = R"(
HloModule CustomCallLayoutConstrainedZeroOperands

ENTRY %CustomCallLayoutConstrainedZeroOperands () -> f32[1,2,3,4] {
  ROOT %custom-call = f32[1,2,3,4]{3,2,0,1} custom-call(), custom_call_target="baz", operand_layout_constraints={}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> m,
      ParseAndReturnVerifiedModule(module_str, GetModuleConfigForTest()));
  ComputationLayout computation_layout = m->entry_computation_layout();
  *computation_layout.mutable_result_layout() = ShapeLayout(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 2, 3, 4}, {2, 1, 0, 3}));
  AssignLayouts(m.get(), &computation_layout);

  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(m::CustomCall())));

  const HloInstruction* custom_call =
      m->entry_computation()->root_instruction()->operand(0);
  ExpectLayoutIs(custom_call->shape(), {3, 2, 0, 1});
}

TEST_F(LayoutAssignmentTest, CustomCallLayoutConstrainedTupleOperand) {
  const char* module_str = R"(
HloModule CustomCallLayoutConstrainedTupleOperand

ENTRY %CustomCallLayoutConstrainedTupleOperand (p0: f32[4,4], p1: f32[2,3]) -> f32[1,2,3,4] {
  %p0 = f32[4,4] parameter(0)
  %p1 = f32[2,3] parameter(1)
  %tuple = (f32[4,4], f32[2,3]) tuple(%p0, %p1)
  ROOT %custom-call = f32[1,2,3,4]{3,2,0,1} custom-call(%tuple), custom_call_target="baz", operand_layout_constraints={(f32[4,4]{1,0}, f32[2,3]{0,1})}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> m,
      ParseAndReturnVerifiedModule(module_str, GetModuleConfigForTest()));
  ComputationLayout computation_layout = m->entry_computation_layout();
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 4}, {1, 0}));
  *computation_layout.mutable_parameter_layout(1) =
      ShapeLayout(ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3}, {1, 0}));
  *computation_layout.mutable_result_layout() = ShapeLayout(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 2, 3, 4}, {2, 1, 0, 3}));
  AssignLayouts(m.get(), &computation_layout);

  HloInstruction* root = m->entry_computation()->root_instruction();
  ExpectLayoutIs(root->shape(), {2, 1, 0, 3});

  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(m::CustomCall(m::Tuple()))));

  const HloInstruction* custom_call =
      m->entry_computation()->root_instruction()->operand(0);
  ExpectLayoutIs(custom_call->shape(), {3, 2, 0, 1});
  ExpectTupleLayoutIs(custom_call->operand(0)->shape(), {{1, 0}, {0, 1}});
}

TEST_F(LayoutAssignmentTest, CustomCallLayoutConstrainedTupleResult) {
  const char* module_str = R"(
HloModule CustomCallLayoutConstrainedTupleResult

ENTRY %CustomCallLayoutConstrainedTupleResult (p0: f32[4,4]) -> (f32[4,4]{1,0}, f32[2,3]{0,1}) {
  %p0 = f32[4,4] parameter(0)
  ROOT %custom-call = (f32[4,4]{1,0}, f32[2,3]{0,1}) custom-call(%p0), custom_call_target="baz", operand_layout_constraints={f32[4,4]{1,0}}
}
)";
  // Try with a couple different layouts. In each case the custom calls operand
  // and result layout should match that of the computation.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> m,
      ParseAndReturnVerifiedModule(module_str, GetModuleConfigForTest()));
  ComputationLayout computation_layout = m->entry_computation_layout();
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 4}, {1, 0}));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 4}, {1, 0}),
           ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3}, {1, 0})}));
  AssignLayouts(m.get(), &computation_layout);

  ExpectTupleLayoutIs(m->result_shape(), {{1, 0}, {1, 0}});

  const HloInstruction* custom_call = FindInstruction(m.get(), "custom-call");
  ExpectTupleLayoutIs(custom_call->shape(), {{1, 0}, {0, 1}});
}

Status AssignLayoutsToComputation(
    HloModule* m, ChannelLayoutConstraints* channel_constraints = nullptr) {
  if (!m->entry_computation_layout().result_layout().LayoutIsSet()) {
    m->mutable_entry_computation_layout()
        ->mutable_result_layout()
        ->SetToDefaultLayout();
  }
  LayoutAssignment layout_assignment(m->mutable_entry_computation_layout(),
                                     channel_constraints);
  return layout_assignment.Run(m).status();
}

TEST_F(LayoutAssignmentTest, OverwriteDiamondShapedConstraintsX) {
  // Check that we handle a diamond-shaped graph correctly.
  //      transpose
  //       /    \
  //     add    |
  //       \    /
  //        tuple

  auto b = HloComputation::Builder(TestName());
  Shape ashape = ShapeUtil::MakeShape(F32, {12, 8});
  Shape bshape = ShapeUtil::MakeShape(F32, {8, 12});
  auto param0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, bshape, "input"));
  auto param1 =
      b.AddInstruction(HloInstruction::CreateParameter(1, ashape, "input"));
  auto transpose =
      b.AddInstruction(HloInstruction::CreateTranspose(ashape, param0, {1, 0}));
  auto add = b.AddInstruction(
      HloInstruction::CreateBinary(ashape, HloOpcode::kAdd, transpose, param1));
  b.AddInstruction(HloInstruction::CreateTuple({add, transpose}));
  auto m = CreateNewVerifiedModule();
  m->AddEntryComputation(b.Build());
  Shape ashape_major =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {12, 8}, {1, 0});
  Shape ashape_minor =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {12, 8}, {0, 1});
  *m->mutable_entry_computation_layout()->mutable_result_layout() =
      ShapeLayout(ShapeUtil::MakeTupleShape({ashape_major, ashape_minor}));
  const Layout r2_dim0major = LayoutUtil::MakeLayout({1, 0});
  ForceParameterLayout(m.get(), 0, r2_dim0major);
  ForceParameterLayout(m.get(), 1, r2_dim0major);
  TF_ASSERT_OK(AssignLayoutsToComputation(m.get()));
  EXPECT_THAT(m->entry_computation()->root_instruction()->operand(0)->shape(),
              ashape_major);
  EXPECT_THAT(add->operand(0)->shape().layout().minor_to_major(),
              add->operand(1)->shape().layout().minor_to_major());
  EXPECT_THAT(transpose->shape().layout().minor_to_major(), ElementsAre(0, 1));
}

// Tests that the layout assignment supports layout-constrained all-reduce with
// different operand layouts (b/146056839).
TEST_F(LayoutAssignmentTest, LayoutConstrainedAllReduce) {
  const char* module_str = R"(
HloModule test_module

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry_computation {
  param = (f32[8,4]{0,1}, f32[16,2]{0,1}) parameter(0)
  gte0 = f32[8,4] get-tuple-element(param), index=0
  gte1 = f32[16,2] get-tuple-element(param), index=1
  crs = (f32[8,4]{0,1}, f32[16,2]{1,0}) all-reduce(gte0, gte1),
    replica_groups={}, constrain_layout=true, to_apply=add
  gte2 = f32[8,4] get-tuple-element(crs), index=0
  gte3 = f32[16,2] get-tuple-element(crs), index=1
  ROOT result = (f32[8,4]{1,0}, f32[16,2]{1,0}) tuple(gte2, gte3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape(), /*ignore_layouts=*/false);

  ChannelLayoutConstraints channel_constraints;
  AssignLayouts(m.get(), &computation_layout, &channel_constraints);

  const HloInstruction* crs = FindInstruction(m.get(), "crs");
  ExpectTupleLayoutIs(crs->shape(), {{0, 1}, {1, 0}});
  ExpectLayoutIs(crs->operand(0)->shape(), {0, 1});
  ExpectLayoutIs(crs->operand(1)->shape(), {1, 0});
}

TEST_F(LayoutAssignmentTest, LayoutConstrainedAllToAll) {
  const char* module_str = R"(
HloModule test_module

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry_computation {
  param = (f32[16,4]{0,1}, f32[16,4]{1,0}) parameter(0)
  gte0 = f32[16,4] get-tuple-element(param), index=0
  gte1 = f32[16,4] get-tuple-element(param), index=1
  alltoall = (f32[16,4]{1,0}, f32[16,4]{1,0}) all-reduce(gte0, gte1),
    replica_groups={{0,1}}, constrain_layout=true, to_apply=add
  gte2 = f32[16,4] get-tuple-element(alltoall), index=0
  gte3 = f32[16,4] get-tuple-element(alltoall), index=1
  ROOT concat = f32[16,8]{0,1} concatenate(gte2, gte3), dimensions={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> m,
      ParseAndReturnVerifiedModule(module_str, /*replica_count=*/2));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape(), /*ignore_layouts=*/false);

  ChannelLayoutConstraints channel_constraints;
  AssignLayouts(m.get(), &computation_layout, &channel_constraints);

  const HloInstruction* alltoall = FindInstruction(m.get(), "alltoall");
  ExpectTupleLayoutIs(alltoall->shape(), {{1, 0}, {1, 0}});
  ExpectLayoutIs(alltoall->operand(0)->shape(), {1, 0});
  ExpectLayoutIs(alltoall->operand(1)->shape(), {1, 0});
}

TEST_F(LayoutAssignmentTest, DynamicRoot) {
  const char* module_str = R"(
HloModule test_module

ENTRY entry_computation {
  param = f32[1,<=16]{0,1} parameter(0)
  ROOT abs = f32[1,<=16]{0,1} abs(param)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape(), /*ignore_layouts=*/false);
  computation_layout.mutable_result_layout()->ClearDynamicShape();

  AssignLayouts(m.get(), &computation_layout);

  const HloInstruction* abs = FindInstruction(m.get(), "abs");
  ExpectLayoutIs(abs->operand(0)->shape(), {0, 1});
  ExpectLayoutIs(abs->shape(), {0, 1});
  EXPECT_TRUE(abs->shape().is_dynamic_dimension(1));
}

// Test the ability to avoid copying across computations by reversing
// computation traversal order.
TEST_F(LayoutAssignmentTest, ReverseComputationOrderAvoidCopy) {
  const char* module_str = R"(
HloModule ComputationLayoutAvoidCopy

call_1 {
  %arg_tuple.1 = (f32[93184,4]) parameter(0)
  %get-tuple-element.1 = f32[93184,4] get-tuple-element(%arg_tuple.1), index=0
  ROOT %reshape.8494 = f32[2,512,364]{2,1,0} reshape(f32[93184,4]{1,0}%get-tuple-element.1)
}

on_true {
  %arg_tuple.1 = (f32[93184,4]) parameter(0)
  %get-tuple-element.1 = f32[93184,4] get-tuple-element(%arg_tuple.1), index=0
  ROOT %reshape.8493 = f32[2,512,364]{2,1,0} reshape(f32[93184,4]{1,0}%get-tuple-element.1)
}

on_false {
  %arg_tuple.2 = (f32[93184,4]) parameter(0)
  %get-tuple-element.3 = f32[93184,4] get-tuple-element(%arg_tuple.2), index=0
  %reshape.9717 = f32[2,512,364]{2,1,0} reshape(f32[93184,4]{1,0}%get-tuple-element.3)
  ROOT %add = f32[2,512,364] add(%reshape.9717, %reshape.9717)
}

ENTRY main {
  pred.1 = pred[] parameter(0)
  arg.2 = f32[93184,4]{1,0} parameter(1)
  arg_tuple.11 = (f32[93184,4]{1,0}) tuple(arg.2)
  call.1 = f32[2,512,364] call(arg_tuple.11), to_apply=call_1
  conditional = f32[2,512,364] conditional(pred.1, arg_tuple.11, arg_tuple.11),
                     true_computation=on_true, false_computation=on_false
  ROOT add = f32[2,512,364] add(call.1, conditional)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(ShapeUtil::MakeShape(PRED, {}));
  *computation_layout.mutable_parameter_layout(1) =
      ShapeLayout(ShapeUtil::MakeShapeWithDenseLayout(F32, {93184, 4}, {0, 1}));
  *computation_layout.mutable_result_layout() = ShapeLayout(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 512, 364}, {0, 1, 2}));
  ChannelLayoutConstraints channel_constraints;
  LayoutAssignment layout_assignment(
      &computation_layout,
      /*channel_constraints=*/&channel_constraints,
      /* reverse_computation_order = */ true);
  EXPECT_IS_OK(layout_assignment.Run(m.get()).status());
  const HloInstruction* call_1 = FindInstruction(m.get(), "reshape.8494");
  ExpectLayoutIs(call_1->shape(), {0, 1, 2});
  const HloInstruction* on_true = FindInstruction(m.get(), "reshape.8493");
  ExpectLayoutIs(on_true->shape(), {0, 1, 2});
  const HloInstruction* on_false = FindInstruction(m.get(), "reshape.9717");
  ExpectLayoutIs(on_false->shape(), {0, 1, 2});
}

// Test the ability to propagate operand constraints across multiple operations.
TEST_F(LayoutAssignmentTest, PropagateOperandLayout) {
  const char* module_str = R"(
HloModule ComputationPropagateOperandLayout

%scalar_add_computation.1 (scalar_lhs.1: f32[], scalar_rhs.1: f32[]) -> f32[] {
  %scalar_lhs.1 = f32[]{:T(256)} parameter(0)
  %scalar_rhs.1 = f32[]{:T(256)} parameter(1)
  ROOT %add.25 = f32[]{:T(256)} add(f32[]{:T(256)} %scalar_lhs.1, f32[]{:T(256)} %scalar_rhs.1)
}

ENTRY main {
  %convolution-base-dilated = f32[64,243,243,10]{0,3,2,1:T(8,128)} parameter(0)
  %reduce.13 = f32[64,243,243]{2,1,0:T(8,128)} parameter(1)
  %divide.2 = f32[64,243,243,10]{2,3,1,0:T(8,128)} parameter(2)
  %reshape.32 = f32[384,10,1,1]{0,1,3,2:T(8,128)} parameter(3)
  %subtract = f32[64,243,243,384]{3,0,2,1:T(8,128)} parameter(4)
  %constant = f32[]{:T(256)} constant(3779136)
  %broadcast.71 = f32[64,243,243,384]{3,0,2,1:T(8,128)} parameter(5)
  %broadcast.46 = f32[64,243,243,10] broadcast(f32[64,243,243] %reduce.13), dimensions={0,1,2}
  %subtract.14 = f32[64,243,243,10] subtract(f32[64,243,243,10] %convolution-base-dilated, f32[64,243,243,10] %broadcast.46)
  %multiply.22 = f32[64,243,243,10] multiply(f32[64,243,243,10] %subtract.14, f32[64,243,243,10]{2,3,1,0:T(8,128)} %divide.2)
  %convolution.9 = f32[64,243,243,384] convolution(f32[64,243,243,10] %multiply.22, f32[384,10,1,1] %reshape.32), window={size=1x1}, dim_labels=01bf_oi01->01bf
  %reduce.14 = f32[384] reduce(f32[64,243,243,384] %convolution.9, f32[]{:T(256)} %constant), dimensions={0,1,2}, to_apply=%scalar_add_computation.1
  %multiply.24 = f32[64,243,243,384] multiply(f32[64,243,243,384] %convolution.9, f32[64,243,243,384] %subtract)
  %reduce.15 = f32[384] reduce(f32[64,243,243,384] %multiply.24, f32[] %constant), dimensions={0,1,2}, to_apply=%scalar_add_computation.1
  %broadcast.47 = f32[64,243,243,384] broadcast(f32[] %constant), dimensions={}
  %multiply.23 = f32[64,243,243,384] multiply(f32[64,243,243,384] %convolution.9, f32[64,243,243,384] %broadcast.47)
  %broadcast.48 = f32[64,243,243,384]{3,2,1,0:T(8,128)} broadcast(f32[384]{0:T(512)} %reduce.14), dimensions={3}
  %subtract.15 = f32[64,243,243,384] subtract(f32[64,243,243,384] %multiply.23, f32[64,243,243,384] %broadcast.48)
  %broadcast.50 = f32[64,243,243,384] broadcast(f32[384] %reduce.15), dimensions={3}
  %multiply.25 = f32[64,243,243,384] multiply(f32[64,243,243,384] %broadcast.50, f32[64,243,243,384] %subtract)
  %divide.7 = f32[64,243,243,384]{3,0,2,1:T(8,128)} divide(f32[64,243,243,384] %multiply.25, f32[64,243,243,384] %broadcast.71)
  ROOT %subtract.17 = f32[64,243,243,384]{3,0,2,1:T(8,128)} subtract(f32[64,243,243,384] %subtract.15, f32[64,243,243,384] %divide.7)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(ShapeUtil::MakeShapeWithDenseLayout(F32, {64, 243, 243, 10},
                                                      {0, 3, 2, 1}));
  *computation_layout.mutable_parameter_layout(1) = ShapeLayout(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {64, 243, 243}, {2, 1, 0}));
  *computation_layout.mutable_parameter_layout(2) =
      ShapeLayout(ShapeUtil::MakeShapeWithDenseLayout(F32, {64, 243, 243, 10},
                                                      {2, 3, 1, 0}));
  *computation_layout.mutable_parameter_layout(3) = ShapeLayout(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {384, 10, 1, 1}, {0, 1, 3, 2}));
  *computation_layout.mutable_parameter_layout(4) =
      ShapeLayout(ShapeUtil::MakeShapeWithDenseLayout(F32, {64, 243, 243, 384},
                                                      {3, 0, 2, 1}));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(ShapeUtil::MakeShapeWithDenseLayout(F32, {64, 243, 243, 384},
                                                      {3, 0, 2, 1}));
  ChannelLayoutConstraints channel_constraints;
  LayoutAssignment layout_assignment(
      &computation_layout,
      /*channel_constraints=*/&channel_constraints);
  EXPECT_IS_OK(layout_assignment.Run(m.get()).status());
  const HloInstruction* subtract_15 = FindInstruction(m.get(), "subtract.15");
  ExpectLayoutIs(subtract_15->shape(), {3, 0, 2, 1});
  const HloInstruction* broadcast_46 = FindInstruction(m.get(), "broadcast.46");
  ExpectLayoutIs(broadcast_46->shape(), {2, 3, 1, 0});
  const HloInstruction* subtract_14 = FindInstruction(m.get(), "subtract.14");
  ExpectLayoutIs(subtract_14->shape(), {2, 3, 1, 0});
}

// Test the ability to propagate operand constraints across multiple operations.
TEST_F(LayoutAssignmentTest, PropagateOperandLayout2) {
  const char* module_str = R"(
 HloModule TensorFlowGather, entry_computation_layout={(f32[32,650]{1,0},s32[16,1,18]{0,1,2})->f32[16,1,18,32]{3,1,2,0}}

 ENTRY %main (operand: f32[32,650], indices: s32[16,1,18]) -> f32[16,1,18,32] {
   %operand = f32[32,650]{1,0} parameter(0)
   %transpose = f32[650,32]{0,1} transpose(f32[32,650]{1,0} %operand), dimensions={1,0}
   %indices = s32[16,1,18]{0,1,2} parameter(1)
   %reshape.1 = s32[288,1]{1,0} reshape(s32[16,1,18]{0,1,2} %indices)
   %gather.1 = f32[288,1,32]{2,1,0} gather(f32[650,32]{0,1} %transpose, s32[288,1]{1,0} %reshape.1), offset_dims={1,2}, collapsed_slice_dims={}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,32}
   ROOT %reshape.3 = f32[16,1,18,32]{3,2,1,0} reshape(f32[288,1,32]{2,1,0} %gather.1)
 } )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));

  LayoutAssignment layout_assignment(m->mutable_entry_computation_layout(),
                                     nullptr);
  EXPECT_IS_OK(layout_assignment.Run(m.get()).status());
  const HloInstruction* reshape_1 = FindInstruction(m.get(), "reshape.1");
  ExpectLayoutIs(reshape_1->shape(), {1, 0});
  const HloInstruction* reshape_3 = FindInstruction(m.get(), "reshape.3");
  ExpectLayoutIs(reshape_3->shape(), {3, 1, 2, 0});
}

// Test the ability to preset layout for instruction.
TEST_F(LayoutAssignmentTest, PreserveInstructionLayout) {
  const char* module_str = R"(
 HloModule TensorFlowGather, entry_computation_layout={(f32[32,650]{1,0},s32[16,1,18]{0,1,2})->(f32[16,1,18,32]{3,1,2,0})}

 ENTRY %main  {
   %operand = f32[32,650]{1,0} parameter(0)
   %transpose = f32[650,32]{0,1} transpose(f32[32,650]{1,0} %operand), dimensions={1,0}
   %indices = s32[16,1,18]{0,1,2} parameter(1)
   %reshape.1 = s32[288,1]{1,0} reshape(s32[16,1,18]{0,1,2} %indices)
   %gather.1 = f32[288,1,32]{2,1,0} gather(f32[650,32]{0,1} %transpose, s32[288,1]{1,0} %reshape.1), offset_dims={1,2}, collapsed_slice_dims={}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,32}
   %reshape.3 = f32[16,1,18,32]{3,2,1,0} reshape(f32[288,1,32]{2,1,0} %gather.1), metadata={preserve_layout=true}
   ROOT %tuple.1 = (f32[16,1,18,32]{3,1,2,0}) tuple(reshape.3)
 } )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));

  LayoutAssignment layout_assignment(m->mutable_entry_computation_layout(),
                                     nullptr);
  EXPECT_IS_OK(layout_assignment.Run(m.get()).status());
  const HloInstruction* reshape_1 = FindInstruction(m.get(), "reshape.1");
  ExpectLayoutIs(reshape_1->shape(), {1, 0});
  const HloInstruction* reshape_3 = FindInstruction(m.get(), "reshape.3");
  ExpectLayoutIs(reshape_3->shape(), {3, 2, 1, 0});
}

// Different instructions should not share buffers when assigning layout.
TEST_F(LayoutAssignmentTest, BreakBufferAliasAcrossInstructions) {
  const char* module_str = R"(
 HloModule break_alias_test, entry_computation_layout={(f32[256,8]{0,1})->f32[256,8]{1,0}}

called_computation {
  init = f32[256,8]{1,0:T(8)} parameter(0)
  one = f32[] constant(1)
  ones = f32[256,8] broadcast(one), dimensions={}
  ROOT add = f32[256,8] add(init, ones)
}

ENTRY main {
  init = f32[256,8] parameter(0)
  ROOT start = f32[256,8]{1,0} custom-call(init), custom_call_target="baz", to_apply=called_computation, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, metadata={preserve_layout=true}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  LayoutAssignment layout_assignment(m->mutable_entry_computation_layout(),
                                     nullptr);
  EXPECT_IS_OK(layout_assignment.Run(m.get()).status());
  const HloInstruction* param =
      m->entry_computation()->parameter_instruction(0);
  ExpectLayoutIs(param->shape(), {0, 1});
  const HloInstruction* root = m->entry_computation()->root_instruction();
  ExpectLayoutIs(root->shape(), {1, 0});
  // Expecting a copy before custom call to reconcile the different layouts.
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kCopy);
}

// Test the ability to enforce a partially specified parameter constraint.
TEST_F(LayoutAssignmentTest, PartialEntryParameterLayout) {
  const char* module_str = R"(
 HloModule EntryLayout, entry_computation_layout={(f32[32,650]{1,0},s32[16,1,18]{0,1,2})->(f32[650,32]{1,0},s32[18,16,1]{0,1,2})}

 ENTRY %main {
   operand = f32[32,650] parameter(0)
   transpose = transpose(operand), dimensions={1,0}
   indices = s32[16,1,18] parameter(1)
   transpose_indices = transpose(indices), dimensions={2,0,1}
   ROOT t = tuple(transpose, transpose_indices)
 } )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  // Allow propagation only to parameter 0
  m->mutable_entry_computation_layout()->mutable_parameter_layout(0)->Clear();

  LayoutAssignment layout_assignment(m->mutable_entry_computation_layout(),
                                     nullptr);
  EXPECT_IS_OK(layout_assignment.Run(m.get()).status());
  // Assign bitcasting layout to parameter 0
  ExpectLayoutIs(m->entry_computation_layout().parameter_layout(0).shape(),
                 {0, 1});
  // Parameter layout that is set is unmodified.
  ExpectLayoutIs(m->entry_computation_layout().parameter_layout(1).shape(),
                 {0, 1, 2});
}

// Test the ability to enforce tuple constraint with no result constraint.
TEST_F(LayoutAssignmentTest, TupleEntryParameterLayoutNoResultConstraint) {
  const char* module_str = R"(
 HloModule EntryLayout, entry_computation_layout={((f32[32,650]{0,1},s32[16,1,18]{0,1,2}))->(f32[208,100]{1,0},s32[2,144]{1,0})}

 ENTRY %main {
   p = (f32[32,650],s32[16,1,18]) parameter(0)
   operand = f32[32,650] get-tuple-element(p), index=0 
   reshape = f32[208,100] reshape(operand)
   indices = s32[16,1,18] get-tuple-element(p), index=1
   reshape_indices = s32[2,144] reshape(indices)
   ROOT t = tuple(reshape, reshape_indices)
 } )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  // Allow propagation only to parameter 0
  m->mutable_entry_computation_layout()->mutable_result_layout()->Clear();

  LayoutAssignment layout_assignment(m->mutable_entry_computation_layout(),
                                     nullptr);
  EXPECT_IS_OK(layout_assignment.Run(m.get()).status());
  // Assign bitcasting layout to parameter 0
  ExpectLayoutIs(
      m->entry_computation_layout().parameter_layout(0).shape().tuple_shapes(0),
      {0, 1});
  // Parameter layout that is set is unmodified.
  ExpectLayoutIs(
      m->entry_computation_layout().parameter_layout(0).shape().tuple_shapes(1),
      {0, 1, 2});
}

// Test the ability to enforce tuple constraint with no result constraint and
// one tuple element not specified.
TEST_F(LayoutAssignmentTest,
       PartialTupleEntryParameterLayoutNoResultConstraint) {
  const char* module_str = R"(
 HloModule EntryLayout, entry_computation_layout={((f32[32,650]{0,1},s32[16,1,18]{0,1,2}))->(f32[208,100]{1,0},s32[2,144]{1,0})}

 ENTRY %main {
   p = (f32[32,650],s32[16,1,18]) parameter(0)
   operand = f32[32,650] get-tuple-element(p), index=0 
   reshape = f32[208,100] reshape(operand)
   indices = s32[16,1,18] get-tuple-element(p), index=1
   reshape_indices = s32[2,144] reshape(indices)
   ROOT t = tuple(reshape, reshape_indices)
 } )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  // Allow propagation only to parameter 0
  m->mutable_entry_computation_layout()->mutable_result_layout()->Clear();
  m->mutable_entry_computation_layout()->mutable_parameter_layout(0)->Clear(
      {1});

  LayoutAssignment layout_assignment(m->mutable_entry_computation_layout(),
                                     nullptr);
  EXPECT_IS_OK(layout_assignment.Run(m.get()).status());
  // Assign bitcasting layout to parameter 0
  ExpectLayoutIs(
      m->entry_computation_layout().parameter_layout(0).shape().tuple_shapes(0),
      {0, 1});
  // Parameter layout that is set is unmodified.
  ExpectLayoutIs(
      m->entry_computation_layout().parameter_layout(0).shape().tuple_shapes(1),
      {2, 1, 0});
}

// Test the ability to enforce aliasing .
TEST_F(LayoutAssignmentTest, AliasParameterAndOutput) {
  const char* module_str = R"(
 HloModule EntryAlias, input_output_alias={ {}: (0, {}, may-alias) }

 ENTRY %main {
   p0 = f32[65,65] parameter(0)
   p1 = f32[4225] parameter(1)
   r = f32[65,65] reshape(p1)
   a = add(p0,r)
   ROOT t = transpose(a), dimensions={1,0}
 } )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  m->mutable_entry_computation_layout()->SetToDefaultLayout();
  m->mutable_entry_computation_layout()->mutable_result_layout()->Clear();
  m->mutable_entry_computation_layout()->mutable_parameter_layout(0)->Clear();

  LayoutAssignment layout_assignment(m->mutable_entry_computation_layout(),
                                     nullptr);
  EXPECT_IS_OK(layout_assignment.Run(m.get()).status());
  EXPECT_EQ(m->entry_computation_layout().result_layout().shape(),
            m->entry_computation_layout().parameter_layout(0).shape());
}

// Test the ability to enforce aliasing .
TEST_F(LayoutAssignmentTest, AliasUnconstrainedParamterWithConstrainedOutput) {
  const char* module_str = R"(
 HloModule EntryAlias, input_output_alias={ {}: (0, {}, may-alias) }

 ENTRY %main {
   p0 = f32[65,65] parameter(0)
   p1 = f32[4225] parameter(1)
   r = f32[65,65] reshape(p1)
   a = add(p0,r)
   ROOT t = transpose(a), dimensions={1,0}
 } )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  m->mutable_entry_computation_layout()->SetToDefaultLayout();
  m->mutable_entry_computation_layout()->mutable_parameter_layout(0)->Clear();

  LayoutAssignment layout_assignment(m->mutable_entry_computation_layout(),
                                     nullptr);
  EXPECT_IS_OK(layout_assignment.Run(m.get()).status());
  EXPECT_EQ(m->entry_computation_layout().result_layout().shape(),
            m->entry_computation_layout().parameter_layout(0).shape());
}

TEST_F(LayoutAssignmentTest, AliasConstrainedParamterWithUnconstrainedOutput) {
  const char* module_str = R"(
 HloModule EntryAlias, input_output_alias={ {}: (0, {}, may-alias) }

 ENTRY %main {
   p0 = f32[65,65] parameter(0)
   p1 = f32[4225] parameter(1)
   r = f32[65,65] reshape(p1)
   a = add(p0,r)
   ROOT t = transpose(a), dimensions={1,0}
 } )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  m->mutable_entry_computation_layout()->SetToDefaultLayout();
  m->mutable_entry_computation_layout()->mutable_result_layout()->Clear();

  LayoutAssignment layout_assignment(m->mutable_entry_computation_layout(),
                                     nullptr);
  EXPECT_IS_OK(layout_assignment.Run(m.get()).status());
  EXPECT_EQ(m->entry_computation_layout().result_layout().shape(),
            m->entry_computation_layout().parameter_layout(0).shape());
}

TEST_F(LayoutAssignmentTest, NestedTupleInLoop) {
  const char* module_str = R"(
HloModule Module

condition  {
    p = (f32[100,100], (f32[100,100], u32[], token[])) parameter(0)
    ROOT lt = pred[] constant(1)
}

body {
    p = (f32[100,100], (f32[100,100], u32[], token[])) parameter(0)

    t1 = f32[100,100] get-tuple-element(p), index=0
    t = (f32[100,100], u32[], token[]) get-tuple-element(p), index=1
    sdone = token[] send-done(t), channel_id=1,
      frontend_attributes={
       _xla_send_recv_pipeline="0"
      }
    tk = token[] after-all()
    snd = (f32[100,100], u32[], token[]) send(t1, tk), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    a = add(t1, t1)
    ROOT tup =  tuple(a, snd)
}

ENTRY %main {
    p0 = f32[100,100] parameter(0)
    tk = token[] after-all()
    snd = (f32[100,100], u32[], token[]) send(p0, tk), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    t = tuple(p0, snd)
    loop = while(t), condition=condition, body=body
    ssend = (f32[100,100], u32[], token[]) get-tuple-element(loop), index=1
    sdone = token[] send-done(ssend), channel_id=1,
      frontend_attributes={
       _xla_send_recv_pipeline="0"
      }
    ROOT result = f32[100,100] get-tuple-element(loop), index=0
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));

  LayoutAssignment layout_assignment(m->mutable_entry_computation_layout(),
                                     nullptr);
  EXPECT_IS_OK(layout_assignment.Run(m.get()).status());
}

}  // namespace
}  // namespace xla

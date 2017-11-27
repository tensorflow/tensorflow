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

#include "tensorflow/compiler/xla/service/cpu/layout_assignment.h"

#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace xla {
namespace {

class CpuLayoutAssignmentTest : public HloTestBase {
 protected:
  void AssignLayouts(HloModule* module,
                     ComputationLayout* entry_computation_layout) {
    cpu::CpuLayoutAssignment layout_assignment(entry_computation_layout);
    EXPECT_IS_OK(layout_assignment.Run(module).status());
  }
};

TEST_F(CpuLayoutAssignmentTest, DotWithConstantRhsTensor) {
  auto builder = HloComputation::Builder(TestName());
  Shape lhs_shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 12}, {0, 1});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {12, 24});
  Shape result_shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 24}, {0, 1});
  auto dot_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "param0"));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(rhs_shape)));
  auto result = builder.AddInstruction(HloInstruction::CreateBinary(
      result_shape, HloOpcode::kDot, dot_lhs, dot_rhs));

  auto module = CreateNewModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(lhs_shape));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(result_shape));
  AssignLayouts(module.get(), &computation_layout);

  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({1, 0}),
                                dot_lhs->shape().layout()));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({0, 1}),
                                dot_rhs->shape().layout()));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({1, 0}),
                                result->shape().layout()));
  for (const auto& instruction : computation->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kCopy);
  }
}

TEST_F(CpuLayoutAssignmentTest, MultipleDotsWithSameConstantRhsTensor0) {
  // Two dot products have the same constant as the RHS, and both those dot
  // products can be optimized if the constant has a column-major layout.
  auto builder = HloComputation::Builder(TestName());
  Shape lhs_shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 12}, {0, 1});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {12, 24});
  Shape result_shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 24}, {0, 1});
  auto dot_a_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "param0"));
  auto dot_b_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, lhs_shape, "param1"));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(rhs_shape)));
  auto dot_a_result = builder.AddInstruction(HloInstruction::CreateBinary(
      result_shape, HloOpcode::kDot, dot_a_lhs, dot_rhs));
  auto dot_b_result = builder.AddInstruction(HloInstruction::CreateBinary(
      result_shape, HloOpcode::kDot, dot_b_lhs, dot_rhs));
  builder.AddInstruction(HloInstruction::CreateBinary(
      result_shape, HloOpcode::kAdd, dot_a_result, dot_b_result));

  auto module = CreateNewModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(lhs_shape));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(result_shape));
  AssignLayouts(module.get(), &computation_layout);

  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({0, 1}),
                                dot_rhs->shape().layout()));
  for (HloInstruction* instruction :
       {dot_a_lhs, dot_b_lhs, dot_a_result, dot_b_result}) {
    EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({1, 0}),
                                  instruction->shape().layout()));
  }
  for (const auto& instruction : computation->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kCopy);
  }
}

TEST_F(CpuLayoutAssignmentTest, MultipleDotsWithSameConstantRhsTensor1) {
  // Two dot products have the same constant as the RHS, but only one of the two
  // dot products can be optimized if the constant has a column-major layout.
  auto builder = HloComputation::Builder(TestName());
  Shape lhs_a_shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 12}, {0, 1});
  Shape lhs_b_shape = ShapeUtil::MakeShapeWithLayout(F32, {2, 12}, {0, 1});
  Shape rhs_shape = ShapeUtil::MakeShapeWithLayout(F32, {12, 24}, {0, 1});
  Shape result_a_shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 24}, {0, 1});
  Shape result_b_shape = ShapeUtil::MakeShapeWithLayout(F32, {2, 24}, {0, 1});
  auto dot_a_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_a_shape, "param0"));
  auto dot_b_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, lhs_b_shape, "param1"));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(rhs_shape)));
  auto dot_a_result = builder.AddInstruction(HloInstruction::CreateBinary(
      result_a_shape, HloOpcode::kDot, dot_a_lhs, dot_rhs));
  auto dot_b_result = builder.AddInstruction(HloInstruction::CreateBinary(
      result_b_shape, HloOpcode::kDot, dot_b_lhs, dot_rhs));
  auto tuple_result = builder.AddInstruction(
      HloInstruction::CreateTuple({dot_a_result, dot_b_result}));

  auto module = CreateNewModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(lhs_a_shape));
  *computation_layout.mutable_parameter_layout(1) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(lhs_b_shape));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(tuple_result->shape()));
  AssignLayouts(module.get(), &computation_layout);

  for (HloInstruction* instruction :
       {dot_rhs, dot_a_lhs, dot_b_lhs, dot_a_result, dot_b_result}) {
    EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({1, 0}),
                                  instruction->shape().layout()));
  }
  for (const auto& instruction : computation->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kCopy);
  }
}

TEST_F(CpuLayoutAssignmentTest, DotWithConstantLhsTensor) {
  auto builder = HloComputation::Builder(TestName());
  Shape lhs_shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 12}, {0, 1});
  Shape rhs_shape = ShapeUtil::MakeShapeWithLayout(F32, {12, 24}, {0, 1});
  Shape result_shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 24}, {0, 1});
  auto dot_lhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(lhs_shape)));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, rhs_shape, "param0"));
  auto dot_result = builder.AddInstruction(HloInstruction::CreateBinary(
      result_shape, HloOpcode::kDot, dot_lhs, dot_rhs));

  auto module = CreateNewModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(rhs_shape));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(result_shape));
  AssignLayouts(module.get(), &computation_layout);

  for (HloInstruction* instruction : {dot_lhs, dot_rhs, dot_result}) {
    EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({1, 0}),
                                  instruction->shape().layout()));
  }
  for (const auto& instruction : computation->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kCopy);
  }
}

TEST_F(CpuLayoutAssignmentTest, DotWithConstantRhsTensorThroughGTE) {
  // This is a case we could theoretically optimize at some point, but today we
  // don't.
  auto builder = HloComputation::Builder(TestName());
  Shape lhs_shape = ShapeUtil::MakeShapeWithLayout(F32, {1, 12}, {0, 1});
  Shape rhs_shape = ShapeUtil::MakeShapeWithLayout(F32, {12, 24}, {0, 1});
  Shape other_shape = ShapeUtil::MakeShapeWithLayout(F32, {100, 24}, {0, 1});

  auto constant_shape = ShapeUtil::MakeTupleShape({other_shape, rhs_shape});
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(constant_shape)));

  Shape result_shape = ShapeUtil::MakeShape(F32, {1, 24});

  auto dot_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "param0"));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, constant, 1));
  auto dot_result = builder.AddInstruction(HloInstruction::CreateBinary(
      result_shape, HloOpcode::kDot, dot_lhs, dot_rhs));

  auto module = CreateNewModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(lhs_shape));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(result_shape));
  AssignLayouts(module.get(), &computation_layout);

  for (HloInstruction* instruction : {dot_lhs, dot_rhs, dot_result}) {
    EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({1, 0}),
                                  instruction->shape().layout()));
  }
  for (const auto& instruction : computation->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kCopy);
  }
}
}  // namespace
}  // namespace xla

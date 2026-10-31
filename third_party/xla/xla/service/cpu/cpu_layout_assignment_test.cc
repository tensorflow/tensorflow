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

#include "xla/service/cpu/cpu_layout_assignment.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/service/computation_layout.h"
#include "xla/service/cpu/target_machine_features_stub.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/tests/test_utils.h"
#include "xla/xla_data.pb.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

class CpuLayoutAssignmentTest : public HloHardwareIndependentTestBase {
 protected:
  void AssignLayouts(HloModule* module,
                     ComputationLayout* entry_computation_layout) {
    cpu::TargetMachineFeaturesStub target_machine_features(
        [](int64_t shape_size) {
          return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
        });
    cpu::CpuLayoutAssignment layout_assignment(entry_computation_layout,
                                               &target_machine_features);
    EXPECT_IS_OK(layout_assignment.Run(module).status());
  }

  void AssignLayouts(HloModule* module) {
    ComputationLayout computation_layout(
        module->entry_computation()->ComputeProgramShape(),
        /*ignore_layouts=*/false);
    AssignLayouts(module, &computation_layout);
  }
};

TEST_F(CpuLayoutAssignmentTest, DotWithConstantRhsTensor) {
  auto builder = HloComputation::Builder(TestName());
  Shape lhs_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {12}, {0});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {12, 24});
  Shape result_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {24}, {0});
  auto dot_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "param0"));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(rhs_shape)));
  auto result = builder.AddInstruction(
      CreateCanonicalDot(result_shape, dot_lhs, dot_rhs));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(lhs_shape));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(result_shape));
  AssignLayouts(module.get(), &computation_layout);

  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({0}),
                                dot_lhs->shape().layout()));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({0, 1}),
                                dot_rhs->shape().layout()));
  EXPECT_TRUE(
      LayoutUtil::Equal(LayoutUtil::MakeLayout({0}), result->shape().layout()));
  for (const auto& instruction : computation->instructions()) {
    EXPECT_NE(instruction->opcode(), HloOpcode::kCopy);
  }
}

TEST_F(CpuLayoutAssignmentTest, MultipleDotsWithSameConstantRhsTensor0) {
  // Two dot products have the same constant as the RHS, and both those dot
  // products can be optimized if the constant has a column-major layout.
  auto builder = HloComputation::Builder(TestName());
  Shape lhs_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {12}, {0});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {12, 24});
  Shape result_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {24}, {0});
  auto dot_a_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "param0"));
  auto dot_b_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, lhs_shape, "param1"));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(rhs_shape)));
  auto dot_a_result = builder.AddInstruction(
      CreateCanonicalDot(result_shape, dot_a_lhs, dot_rhs));
  auto dot_b_result = builder.AddInstruction(
      CreateCanonicalDot(result_shape, dot_b_lhs, dot_rhs));
  builder.AddInstruction(HloInstruction::CreateBinary(
      result_shape, HloOpcode::kAdd, dot_a_result, dot_b_result));

  auto module = CreateNewVerifiedModule();
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
    EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({0}),
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
  Shape lhs_a_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 12}, {0, 1});
  Shape lhs_b_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 12}, {0, 1});
  Shape rhs_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {12, 24}, {0, 1});
  Shape result_a_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 24}, {0, 1});
  Shape result_b_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 24}, {0, 1});
  auto dot_a_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_a_shape, "param0"));
  auto dot_b_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, lhs_b_shape, "param1"));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(rhs_shape)));
  auto dot_a_result = builder.AddInstruction(
      CreateCanonicalDot(result_a_shape, dot_a_lhs, dot_rhs));
  auto dot_b_result = builder.AddInstruction(
      CreateCanonicalDot(result_b_shape, dot_b_lhs, dot_rhs));
  auto tuple_result = builder.AddInstruction(
      HloInstruction::CreateTuple({dot_a_result, dot_b_result}));

  auto module = CreateNewVerifiedModule();
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
  Shape lhs_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 12}, {0, 1});
  Shape rhs_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {12, 24}, {0, 1});
  Shape result_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 24}, {0, 1});
  auto dot_lhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(lhs_shape)));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, rhs_shape, "param0"));
  auto dot_result = builder.AddInstruction(
      CreateCanonicalDot(result_shape, dot_lhs, dot_rhs));

  auto module = CreateNewVerifiedModule();
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
  Shape lhs_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 12}, {0, 1});
  Shape rhs_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {12, 24}, {0, 1});
  Shape other_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {100, 24}, {0, 1});

  auto constant_shape = ShapeUtil::MakeTupleShape({other_shape, rhs_shape});
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(constant_shape)));

  Shape result_shape = ShapeUtil::MakeShape(F32, {1, 24});

  auto dot_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "param0"));
  auto dot_rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, constant, 1));
  auto dot_result = builder.AddInstruction(
      CreateCanonicalDot(result_shape, dot_lhs, dot_rhs));

  auto module = CreateNewVerifiedModule();
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

struct DotOutputFusionLayoutAssignmentResult {
  bool layout_assignment_changed_something;
  const HloInstruction* dot_lhs_fusion_param;
  const HloInstruction* dot_rhs_fusion_param;
  const HloInstruction* addend_fusion_param;
};

static absl::StatusOr<DotOutputFusionLayoutAssignmentResult> RunDotOutputFusion(
    HloModule* module, const std::string& test_name, int m, int k, int n,
    const int64_t dot_operand_idx_in_add) {
  DotOutputFusionLayoutAssignmentResult result;

  CHECK(dot_operand_idx_in_add == 0 || dot_operand_idx_in_add == 1);

  auto builder = HloComputation::Builder(test_name);

  Shape dot_lhs_shape = ShapeUtil::MakeShape(F32, {m, k});
  Shape dot_rhs_shape = ShapeUtil::MakeShape(F32, {k, n});
  Shape dot_shape = ShapeUtil::MakeShape(F32, {m, n});
  if (m == 1) {
    dot_lhs_shape = ShapeUtil::MakeShape(F32, {k});
    dot_shape = ShapeUtil::MakeShape(F32, {n});
  } else if (n == 1) {
    dot_rhs_shape = ShapeUtil::MakeShape(F32, {k});
    dot_shape = ShapeUtil::MakeShape(F32, {m});
  }

  HloInstruction* dot_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, dot_lhs_shape, "param0"));
  HloInstruction* addend = builder.AddInstruction(
      HloInstruction::CreateParameter(1, dot_shape, "param1"));
  HloInstruction* dot_rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateFromShape(dot_rhs_shape)));
  HloInstruction* dot_result =
      builder.AddInstruction(CreateCanonicalDot(dot_shape, dot_lhs, dot_rhs));
  HloInstruction* add_result;
  if (dot_operand_idx_in_add == 0) {
    add_result = builder.AddInstruction(HloInstruction::CreateBinary(
        dot_shape, HloOpcode::kAdd, dot_result, addend));
  } else {
    add_result = builder.AddInstruction(HloInstruction::CreateBinary(
        dot_shape, HloOpcode::kAdd, addend, dot_result));
  }

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloInstruction* fusion_instruction =
      module->entry_computation()->AddInstruction(HloInstruction::CreateFusion(
          dot_shape, HloInstruction::FusionKind::kOutput, add_result));
  ABSL_RETURN_IF_ERROR(
      computation->ReplaceInstruction(add_result, fusion_instruction));

  HloInstruction* fused_add =
      fusion_instruction->fused_instructions_computation()->root_instruction();
  HloInstruction* fused_dot = fusion_instruction->FuseInstruction(dot_result);

  ABSL_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(dot_result));

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(dot_lhs_shape));
  *computation_layout.mutable_parameter_layout(1) =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(dot_shape));
  *computation_layout.mutable_result_layout() =
      ShapeLayout(LayoutUtil::GetWithDefaultLayout(dot_shape));

  result.dot_lhs_fusion_param =
      fusion_instruction->operand(fused_dot->operand(0)->parameter_number());
  result.dot_rhs_fusion_param =
      fusion_instruction->operand(fused_dot->operand(1)->parameter_number());
  result.addend_fusion_param = fusion_instruction->operand(
      fused_add->operand(1 - dot_operand_idx_in_add)->parameter_number());

  cpu::TargetMachineFeaturesStub target_machine_features(
      [](int64_t shape_size) {
        return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
      });
  cpu::CpuLayoutAssignment layout_assignment(&computation_layout,
                                             &target_machine_features);
  ABSL_ASSIGN_OR_RETURN(result.layout_assignment_changed_something,
                   layout_assignment.Run(module));

  return result;
}

static void AssertCorrectLayoutForDotOutputFusion(
    const HloComputation* computation,
    const DotOutputFusionLayoutAssignmentResult& layout_assignment_result,
    bool expect_col_major_dot_rhs) {
  Layout expected_dot_rhs_layout = expect_col_major_dot_rhs
                                       ? LayoutUtil::MakeLayout({0, 1})
                                       : LayoutUtil::MakeLayout({1, 0});
  if (layout_assignment_result.dot_rhs_fusion_param->shape()
          .dimensions()
          .size() == 1) {
    expected_dot_rhs_layout = LayoutUtil::MakeLayout({0});
  }
  EXPECT_TRUE(LayoutUtil::Equal(
      expected_dot_rhs_layout,
      layout_assignment_result.dot_rhs_fusion_param->shape().layout()));

  EXPECT_TRUE(LayoutUtil::Equal(
      LayoutUtil::MakeDescendingLayout(
          layout_assignment_result.dot_lhs_fusion_param->shape()
              .dimensions()
              .size()),
      layout_assignment_result.dot_lhs_fusion_param->shape().layout()));

  EXPECT_TRUE(LayoutUtil::Equal(
      LayoutUtil::MakeDescendingLayout(
          layout_assignment_result.addend_fusion_param->shape()
              .dimensions()
              .size()),
      layout_assignment_result.addend_fusion_param->shape().layout()));
  EXPECT_THAT(computation->instructions(), Each(Not(op::Copy())));
}

TEST_F(CpuLayoutAssignmentTest, DotOutputFusion_1x50x19_dot_idx_0) {
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  ASSERT_OK_AND_ASSIGN(
      DotOutputFusionLayoutAssignmentResult layout_assignment_result,
      RunDotOutputFusion(module.get(), TestName(), /*m=*/1, /*k=*/50, /*n=*/19,
                         /*dot_operand_idx_in_add=*/0));
  ASSERT_TRUE(layout_assignment_result.layout_assignment_changed_something);
  AssertCorrectLayoutForDotOutputFusion(module->entry_computation(),
                                        layout_assignment_result,
                                        /*expect_col_major_dot_rhs=*/true);
}

TEST_F(CpuLayoutAssignmentTest, DotOutputFusion_1x50x19_dot_idx_1) {
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  ASSERT_OK_AND_ASSIGN(
      DotOutputFusionLayoutAssignmentResult layout_assignment_result,
      RunDotOutputFusion(module.get(), TestName(), /*m=*/1, /*k=*/50, /*n=*/19,
                         /*dot_operand_idx_in_add=*/1));
  ASSERT_TRUE(layout_assignment_result.layout_assignment_changed_something);
  AssertCorrectLayoutForDotOutputFusion(module->entry_computation(),
                                        layout_assignment_result,
                                        /*expect_col_major_dot_rhs=*/true);
}

TEST_F(CpuLayoutAssignmentTest, DotOutputFusion_19x50x1_dot_idx_0) {
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  ASSERT_OK_AND_ASSIGN(
      DotOutputFusionLayoutAssignmentResult layout_assignment_result,
      RunDotOutputFusion(module.get(), TestName(), /*m=*/19, /*k=*/50, /*n=*/1,
                         /*dot_operand_idx_in_add=*/0));
  ASSERT_TRUE(layout_assignment_result.layout_assignment_changed_something);
  AssertCorrectLayoutForDotOutputFusion(module->entry_computation(),
                                        layout_assignment_result,
                                        /*expect_col_major_dot_rhs=*/false);
}

TEST_F(CpuLayoutAssignmentTest, DotOutputFusion_19x50x1_dot_idx_1) {
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  ASSERT_OK_AND_ASSIGN(
      DotOutputFusionLayoutAssignmentResult layout_assignment_result,
      RunDotOutputFusion(module.get(), TestName(), /*m=*/19, /*k=*/50, /*n=*/1,
                         /*dot_operand_idx_in_add=*/1));
  ASSERT_TRUE(layout_assignment_result.layout_assignment_changed_something);
  AssertCorrectLayoutForDotOutputFusion(module->entry_computation(),
                                        layout_assignment_result,
                                        /*expect_col_major_dot_rhs=*/false);
}

TEST_F(CpuLayoutAssignmentTest, DotOutputFusion_19x50x19_dot_idx_0) {
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  ASSERT_OK_AND_ASSIGN(
      DotOutputFusionLayoutAssignmentResult layout_assignment_result,
      RunDotOutputFusion(module.get(), TestName(), /*m=*/19, /*k=*/50, /*n=*/19,
                         /*dot_operand_idx_in_add=*/0));
  ASSERT_TRUE(layout_assignment_result.layout_assignment_changed_something);
  AssertCorrectLayoutForDotOutputFusion(module->entry_computation(),
                                        layout_assignment_result,
                                        /*expect_col_major_dot_rhs=*/false);
}

TEST_F(CpuLayoutAssignmentTest, DotOutputFusion_19x50x19_dot_idx_1) {
  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  ASSERT_OK_AND_ASSIGN(
      DotOutputFusionLayoutAssignmentResult layout_assignment_result,
      RunDotOutputFusion(module.get(), TestName(), /*m=*/19, /*k=*/50, /*n=*/19,
                         /*dot_operand_idx_in_add=*/1));
  ASSERT_TRUE(layout_assignment_result.layout_assignment_changed_something);
  AssertCorrectLayoutForDotOutputFusion(module->entry_computation(),
                                        layout_assignment_result,
                                        /*expect_col_major_dot_rhs=*/false);
}

TEST_F(CpuLayoutAssignmentTest, BatchDotLayoutMustBeRowMajor) {
  const char* hlo_string = R"(
HloModule BatchDotLayoutMustBeRowMajor

ENTRY BatchDotLayoutMustBeRowMajor {
  p0 = f32[10,1,10] parameter(0)
  p1 = f32[10,10,1] parameter(1)
  ROOT dot = f32[10,1,1] dot(p0, p1), lhs_batch_dims={0},
                                      lhs_contracting_dims={2},
                                      rhs_batch_dims={0},
                                      rhs_contracting_dims={1}
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  HloComputation* computation = module->entry_computation();

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) = ShapeLayout(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 1, 10}, {2, 1, 0}));
  *computation_layout.mutable_parameter_layout(1) = ShapeLayout(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 10, 1}, {2, 1, 0}));
  *computation_layout.mutable_result_layout() = ShapeLayout(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 1, 1}, {1, 2, 0}));
  AssignLayouts(module.get(), &computation_layout);

  Shape expected_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 1, 1}, {2, 1, 0});
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(op::ShapeWithLayout(expected_shape)));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Copy(op::Dot(
          op::ShapeWithLayout(computation_layout.parameter_layout(0).shape()),
          op::ShapeWithLayout(
              computation_layout.parameter_layout(1).shape()))));
}

TEST_F(CpuLayoutAssignmentTest, VectorDotWithConstantRhsTensor) {
  constexpr absl::string_view kHlo = R"(
HloModule VectorDotWithConstantRhsTensor

ENTRY main {
  p0 = f32[12]{0} parameter(0)
  c0 = f32[12]{0} constant({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
  ROOT dot = f32[] dot(p0, c0), lhs_contracting_dims={0}, rhs_contracting_dims={0}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  AssignLayouts(module.get());

  const HloInstruction* dot = module->entry_computation()->root_instruction();
  EXPECT_EQ(dot->operand(0)->shape().layout(), LayoutUtil::MakeLayout({0}));
  EXPECT_EQ(dot->operand(1)->shape().layout(), LayoutUtil::MakeLayout({0}));
  EXPECT_EQ(dot->shape().layout(), LayoutUtil::MakeLayout({}));
  EXPECT_THAT(module->entry_computation()->instructions(),
              Each(Not(op::Copy())));
}

TEST_F(CpuLayoutAssignmentTest, RowMajorOpsStripPreExistingLayoutFields) {
  constexpr absl::string_view kHlo = R"(
HloModule RowMajorOpsStripPreExistingLayoutFields

compare_topk {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  ROOT cmp = pred[] compare(p0, p1), direction=GT, type=TOTALORDER
}

ENTRY main {
  p0 = f32[4,64,64]{2,1,0} parameter(0)
  p1 = f32[4,64,64]{2,1,0} parameter(1)
  p2 = f32[4,16]{1,0} parameter(2)
  dot = f32[4,64,64]{2,0,1:T(8,128)L(128)S(1)SC(0:32)} dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1}
  topk = (f32[4,8]{0,1:T(8,128)}, s32[4,8]{0,1:T(8,128)}) custom-call(p2),
      custom_call_target="TopK", called_computations={compare_topk}
  ROOT out = (f32[4,64,64]{0,1,2}, (f32[4,8]{0,1}, s32[4,8]{0,1}))
      tuple(dot, topk)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  AssignLayouts(module.get());

  EXPECT_EQ(FindInstruction(module.get(), "dot")->shape().layout(),
            LayoutUtil::MakeLayout({2, 1, 0}));
  const Shape& topk_shape = FindInstruction(module.get(), "topk")->shape();
  EXPECT_EQ(topk_shape.tuple_shapes(0).layout(),
            LayoutUtil::MakeLayout({1, 0}));
  EXPECT_EQ(topk_shape.tuple_shapes(1).layout(),
            LayoutUtil::MakeLayout({1, 0}));
}

TEST_F(CpuLayoutAssignmentTest, CollectivesAssignDimMajorLayout) {
  constexpr absl::string_view kHlo = R"(
HloModule CollectivesAssignDimMajorLayout

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = f32[8,8,16]{2,1,0} parameter(0)
  p1 = f32[4,8]{1,0} parameter(1)
  p2 = f32[8,4,16]{2,1,0} parameter(2)
  p3 = s32[4,4]{1,0} parameter(3)
  in0 = f32[8,8,16]{2,1,0} negate(p0)
  in1 = f32[4,8]{1,0} negate(p1)
  in2 = f32[8,4,16]{2,1,0} negate(p2)
  in3 = s32[4,4]{1,0} negate(p3)
  rs_arr = f32[8,4,16]{2,1,0} reduce-scatter(in0), dimensions={1},
      replica_groups={{0,1}}, to_apply=add
  ag_arr = f32[8,8,16]{2,1,0} all-gather(in2), dimensions={1},
      replica_groups={{0,1}}
  rs = (f32[8,4,16]{2,1,0:T(8,128)SC(0:4)},
        f32[4,4]{1,0:T(8,128)}) reduce-scatter(in0, in1), dimensions={1},
      replica_groups={{0,1}}, to_apply=add
  ag = (f32[8,8,16]{2,1,0:T(8,128)SC(0:4)},
        s32[4,8]{1,0:T(8,128)}) all-gather(in2, in3), dimensions={1},
      replica_groups={{0,1}}
  keep0 = f32[8,8,16]{2,1,0} exponential(in0)
  keep2 = f32[8,4,16]{2,1,0} exponential(in2)
  ROOT out = (f32[8,4,16]{2,1,0}, f32[8,8,16]{2,1,0},
              (f32[8,4,16]{2,1,0}, f32[4,4]{1,0}),
              (f32[8,8,16]{2,1,0}, s32[4,8]{1,0}),
              f32[8,8,16]{2,1,0}, f32[8,4,16]{2,1,0})
      tuple(rs_arr, ag_arr, rs, ag, keep0, keep2)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo, /*replica_count=*/2));
  Layout* ag_leaf1_layout = FindInstruction(module.get(), "ag")
                                ->mutable_shape()
                                ->mutable_tuple_shapes(1)
                                ->mutable_layout();
  ag_leaf1_layout->clear_minor_to_major();
  ag_leaf1_layout->set_memory_space(5);
  AssignLayouts(module.get());

  const Layout dim1_major_3d = LayoutUtil::MakeLayout({2, 0, 1});
  const Layout dim1_major_2d = LayoutUtil::MakeLayout({0, 1});
  for (absl::string_view name : {"rs_arr", "ag_arr"}) {
    const HloInstruction* inst = FindInstruction(module.get(), name);
    EXPECT_EQ(inst->shape().layout(), dim1_major_3d);
    EXPECT_EQ(inst->operand(0)->shape().layout(), dim1_major_3d);
  }
  for (absl::string_view name : {"rs", "ag"}) {
    const HloInstruction* inst = FindInstruction(module.get(), name);
    EXPECT_EQ(inst->shape().tuple_shapes(0).layout(), dim1_major_3d);
    EXPECT_EQ(inst->shape().tuple_shapes(1).layout(), dim1_major_2d);
    EXPECT_EQ(inst->operand(0)->shape().layout(), dim1_major_3d);
    EXPECT_EQ(inst->operand(1)->shape().layout(), dim1_major_2d);
  }
}

TEST_F(CpuLayoutAssignmentTest, ConstrainedCollectivesPreserveLayout) {
  constexpr absl::string_view kHlo = R"(
HloModule ConstrainedCollectivesPreserveLayout

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = f32[8,4,16]{2,1,0} parameter(0)
  p1 = s32[4,4]{1,0} parameter(1)
  in0 = f32[8,4,16]{2,1,0} negate(p0)
  in1 = f32[8,4,16]{2,1,0} abs(p0)
  in2 = f32[8,4,16]{2,1,0} exponential(p0)
  in3 = s32[4,4]{1,0} negate(p1)
  rs = f32[4,4,16]{1,2,0} reduce-scatter(in0), dimensions={0},
      constrain_layout=true, replica_groups={{0,1}}, to_apply=add
  ag_arr = f32[16,4,16]{1,2,0} all-gather(in1), dimensions={0},
      constrain_layout=true, replica_groups={{0,1}}
  ag = (f32[8,8,16]{0,2,1}, s32[4,8]{1,0}) all-gather(in2, in3),
      dimensions={1}, constrain_layout=true, replica_groups={{0,1}}
  ROOT out = (f32[4,4,16]{2,1,0}, f32[16,4,16]{2,1,0},
              (f32[8,8,16]{2,1,0}, s32[4,8]{1,0})) tuple(rs, ag_arr, ag)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo, /*replica_count=*/2));
  AssignLayouts(module.get());

  for (absl::string_view name : {"rs", "ag_arr"}) {
    const HloInstruction* inst = FindInstruction(module.get(), name);
    EXPECT_EQ(inst->shape().layout(), LayoutUtil::MakeLayout({1, 2, 0}));
    EXPECT_EQ(inst->operand(0)->shape().layout(),
              LayoutUtil::MakeLayout({1, 2, 0}));
  }
  const HloInstruction* ag = FindInstruction(module.get(), "ag");
  EXPECT_EQ(ag->shape().tuple_shapes(0).layout(),
            LayoutUtil::MakeLayout({0, 2, 1}));
  EXPECT_EQ(ag->shape().tuple_shapes(1).layout(),
            LayoutUtil::MakeLayout({0, 1}));
  EXPECT_EQ(ag->operand(0)->shape().layout(),
            LayoutUtil::MakeLayout({0, 2, 1}));
  EXPECT_EQ(ag->operand(1)->shape().layout(), LayoutUtil::MakeLayout({0, 1}));
}
}  // namespace
}  // namespace xla

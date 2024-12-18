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

#include "xla/service/cpu/cpu_instruction_fusion.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/transpose_folding.h"
#include "xla/shape.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace op = xla::testing::opcode_matchers;

namespace xla::cpu {
namespace {

using InstructionFusionTest = HloTestBase;

std::unique_ptr<HloInstruction> MakeDot(const Shape& shape, HloInstruction* lhs,
                                        HloInstruction* rhs) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(lhs->shape().rank() - 1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      2, PrecisionConfig::DEFAULT);
  return HloInstruction::CreateDot(shape, lhs, rhs, dot_dnums,
                                   precision_config);
}

TEST_F(InstructionFusionTest, DotOperationFusion_Basic_0) {
  HloComputation::Builder builder(TestName());
  HloInstruction* arg0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1024, 256}), "arg0"));
  HloInstruction* arg1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {256}), "arg1"));

  HloInstruction* exp0 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {1024, 256}), HloOpcode::kExp, arg0));
  HloInstruction* dot = builder.AddInstruction(
      MakeDot(ShapeUtil::MakeShape(F32, {1024}), exp0, arg1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(dot, computation->root_instruction());
  EXPECT_TRUE(CpuInstructionFusion().Run(module.get()).value());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest, DotOperationFusion_Basic_1) {
  HloComputation::Builder builder(TestName());
  HloInstruction* arg0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {256}), "arg0"));
  HloInstruction* arg1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {256, 1024}), "arg1"));

  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {256, 1024}), HloOpcode::kExp, arg1));
  HloInstruction* dot = builder.AddInstruction(
      MakeDot(ShapeUtil::MakeShape(F32, {1024}), arg0, exp1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(dot, computation->root_instruction());
  EXPECT_TRUE(CpuInstructionFusion().Run(module.get()).value());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest, DotOperationFusion_Bitcast) {
  HloComputation::Builder builder(TestName());
  HloInstruction* arg0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {2, 512, 2, 128}), "arg0"));
  HloInstruction* arg1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {256}), "arg1"));

  HloInstruction* exp0 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {2, 512, 2, 128}), HloOpcode::kExp, arg0));
  HloInstruction* bitcast0 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {1024, 256}), HloOpcode::kBitcast, exp0));
  HloInstruction* dot = builder.AddInstruction(
      MakeDot(ShapeUtil::MakeShape(F32, {1024}), bitcast0, arg1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(dot, computation->root_instruction());
  EXPECT_TRUE(CpuInstructionFusion().Run(module.get()).value());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest, DotOperationFusion_Reshape) {
  HloComputation::Builder builder(TestName());
  HloInstruction* arg0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {2, 512, 2, 128}), "arg0"));
  HloInstruction* arg1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {256}), "arg1"));

  HloInstruction* exp0 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {2, 512, 2, 128}), HloOpcode::kExp, arg0));
  HloInstruction* reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {1024, 256}), exp0));
  HloInstruction* dot = builder.AddInstruction(
      MakeDot(ShapeUtil::MakeShape(F32, {1024}), reshape0, arg1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(dot, computation->root_instruction());
  EXPECT_TRUE(CpuInstructionFusion().Run(module.get()).value());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest, DotOperationFusion_TooLarge) {
  HloComputation::Builder builder(TestName());
  HloInstruction* arg0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {32 * 1024}), "arg0"));
  HloInstruction* arg1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {32 * 1024, 256}), "arg1"));

  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {32 * 1024, 256}), HloOpcode::kExp, arg1));
  HloInstruction* dot = builder.AddInstruction(
      MakeDot(ShapeUtil::MakeShape(F32, {256}), arg0, exp1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(dot, computation->root_instruction());
  EXPECT_FALSE(CpuInstructionFusion().Run(module.get()).value());
  EXPECT_EQ(dot, computation->root_instruction());
}

TEST_F(InstructionFusionTest, DotOperationFusion_ElementReuse) {
  HloComputation::Builder builder(TestName());
  HloInstruction* arg0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {2, 256}), "arg0"));
  HloInstruction* arg1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {256, 1024}), "arg1"));

  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {256, 1024}), HloOpcode::kExp, arg1));
  HloInstruction* dot = builder.AddInstruction(
      MakeDot(ShapeUtil::MakeShape(F32, {2, 1024}), arg0, exp1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(dot, computation->root_instruction());
  EXPECT_FALSE(CpuInstructionFusion().Run(module.get()).value());
  EXPECT_EQ(dot, computation->root_instruction());
}

TEST_F(InstructionFusionTest, DotOperationFusion_TransposeFusion_RHS) {
  std::string hlo_string = R"(
HloModule DotOperationFusion_TransposeFusion

ENTRY DotOperationFusion_TransposeFusion {
  arg0 = f32[1,256] parameter(0)
  arg1 = f32[1024,256] parameter(1)
  exponential = f32[1024,256] exponential(arg1)
  transpose = f32[256,1024] transpose(exponential), dimensions={1,0}
  ROOT dot = f32[1,1024] dot(arg0, transpose), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloComputation* computation = module->entry_computation();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, TransposeFolding().Run(module.get()));
  ASSERT_TRUE(changed);
  ASSERT_THAT(computation->root_instruction(),
              op::Dot(op::Parameter(0), op::Exp(op::Parameter(1)),
                      /*lhs_contracting_dim=*/1, /*rhs_contracting_dim=*/1));
}

TEST_F(InstructionFusionTest, DotOperationFusion_TransposeFusion_LHS) {
  std::string hlo_string = R"(
HloModule DotOperationFusion_TransposeFusion

ENTRY DotOperationFusion_TransposeFusion {
  arg0 = f32[256,1] parameter(0)
  arg1 = f32[256,1024] parameter(1)
  transpose = f32[1,256] transpose(arg0), dimensions={1,0}
  exponential = f32[256,1024] exponential(arg1)
  ROOT dot = f32[1,1024] dot(transpose, exponential), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloComputation* computation = module->entry_computation();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, TransposeFolding().Run(module.get()));
  ASSERT_TRUE(changed);
  ASSERT_THAT(computation->root_instruction(),
              op::Dot(op::Parameter(0), op::Exp(op::Parameter(1)),
                      /*lhs_contracting_dim=*/0, /*rhs_contracting_dim=*/0));
}

TEST_F(InstructionFusionTest,
       DotOperationFusion_TransposeFusion_LHS_NonDefault) {
  std::string hlo_string = R"(
HloModule DotOperationFusion_TransposeFusion

ENTRY DotOperationFusion_TransposeFusion {
  arg0 = f32[1,256] parameter(0)
  arg1 = f32[256,1024] parameter(1)
  transpose = f32[256,1] transpose(arg0), dimensions={1,0}
  exponential = f32[256,1024] exponential(arg1)
  ROOT dot = f32[1,1024] dot(transpose, exponential), lhs_contracting_dims={0}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloComputation* computation = module->entry_computation();

  TF_ASSERT_OK_AND_ASSIGN(bool changed, TransposeFolding().Run(module.get()));
  ASSERT_TRUE(changed);
  ASSERT_THAT(computation->root_instruction(),
              op::Dot(op::Parameter(0), op::Exp(op::Parameter(1)),
                      /*lhs_contracting_dim=*/1, /*rhs_contracting_dim=*/0));
}

class OpcodeFusionTest : public InstructionFusionTest {
 protected:
  // Runs CPU instruction fusion on the given module, and tests that the result
  // contains a fused op at the root with exactly the given multiset of opcodes.
  void RunFusionAndCheckOpcodesWereFused(
      HloModule* module, const std::multiset<HloOpcode>& expected_opcodes,
      HloInstruction::FusionKind fusion_kind =
          HloInstruction::FusionKind::kLoop) {
    auto computation = module->entry_computation();
    auto did_fusion = CpuInstructionFusion().Run(module);
    ASSERT_TRUE(did_fusion.ok());
    EXPECT_TRUE(did_fusion.value());

    HloInstruction* root = computation->root_instruction();
    ASSERT_THAT(root, op::Fusion());
    EXPECT_EQ(root->fusion_kind(), fusion_kind);

    std::vector<HloOpcode> fused_opcodes(root->fused_instruction_count());
    std::transform(root->fused_instructions().begin(),
                   root->fused_instructions().end(), fused_opcodes.begin(),
                   [](const HloInstruction* hlo) { return hlo->opcode(); });

    EXPECT_EQ(
        std::multiset<HloOpcode>(fused_opcodes.begin(), fused_opcodes.end()),
        expected_opcodes);
  }

  HloComputation* CreateAdderToOne(HloModule* module) {
    HloComputation::Builder builder(TestName());
    HloInstruction* arg0 =
        builder.AddInstruction(HloInstruction::CreateParameter(
            0, ShapeUtil::MakeShape(F32, {}), "arg0"));
    HloInstruction* one = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
    builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, arg0, one));
    return module->AddEmbeddedComputation(builder.Build());
  }

  HloComputation* CreateMax(HloModule* module) {
    HloComputation::Builder builder(TestName());
    HloInstruction* arg0 =
        builder.AddInstruction(HloInstruction::CreateParameter(
            0, ShapeUtil::MakeShape(F32, {}), "arg0"));
    HloInstruction* arg1 =
        builder.AddInstruction(HloInstruction::CreateParameter(
            1, ShapeUtil::MakeShape(F32, {}), "arg1"));
    builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(F32, {}), HloOpcode::kMaximum, arg0, arg1));
    return module->AddEmbeddedComputation(builder.Build());
  }
};

TEST_F(OpcodeFusionTest, Exponential_Reshape_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {1, 4});
  Shape result_shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(param_shape, HloOpcode::kExp, param0));
  HloInstruction* reshape2 =
      builder.AddInstruction(HloInstruction::CreateReshape(result_shape, exp1));
  builder.AddInstruction(
      HloInstruction::CreateUnary(result_shape, HloOpcode::kNegate, reshape2));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(), {HloOpcode::kNegate, HloOpcode::kReshape, HloOpcode::kExp,
                     HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, Broadcast_Reshape_DynamicSlice_Tanh) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {8});
  Shape starts_shape = ShapeUtil::MakeShape(S32, {});
  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {1, 8, 8});
  Shape reshape_shape = ShapeUtil::MakeShape(F32, {8, 8});
  Shape dynamic_slice_shape = ShapeUtil::MakeShape(F32, {4, 4});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, starts_shape, "starts"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, starts_shape, "starts"));
  HloInstruction* broadcast2 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(broadcast_shape, param0, {1}));
  HloInstruction* reshape3 = builder.AddInstruction(
      HloInstruction::CreateReshape(reshape_shape, broadcast2));
  HloInstruction* dynamic_slice4 =
      builder.AddInstruction(HloInstruction::CreateDynamicSlice(
          dynamic_slice_shape, reshape3, {param1, param2}, {4, 4}));
  builder.AddInstruction(HloInstruction::CreateUnary(
      dynamic_slice_shape, HloOpcode::kTanh, dynamic_slice4));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kTanh, HloOpcode::kDynamicSlice, HloOpcode::kReshape,
       HloOpcode::kBroadcast, HloOpcode::kParameter, HloOpcode::kParameter,
       HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, Broadcast_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {8});
  Shape result_shape = ShapeUtil::MakeShape(F32, {8, 8});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* broadcast1 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(result_shape, param0, {1}));
  builder.AddInstruction(HloInstruction::CreateUnary(
      result_shape, HloOpcode::kNegate, broadcast1));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kNegate, HloOpcode::kBroadcast, HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, DynamicSlice_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {4});
  Shape slice_shape = ShapeUtil::MakeShape(S32, {});
  Shape result_shape = ShapeUtil::MakeShape(F32, {2});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, slice_shape, "starts"));
  HloInstruction* dynamic_slice2 = builder.AddInstruction(
      HloInstruction::CreateDynamicSlice(result_shape, param0, {param1}, {2}));
  builder.AddInstruction(HloInstruction::CreateUnary(
      result_shape, HloOpcode::kNegate, dynamic_slice2));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(), {HloOpcode::kNegate, HloOpcode::kDynamicSlice,
                     HloOpcode::kParameter, HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, Exponential_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(param_shape, HloOpcode::kExp, param0));
  builder.AddInstruction(
      HloInstruction::CreateUnary(param_shape, HloOpcode::kNegate, exp1));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kNegate, HloOpcode::kExp, HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, Reshape_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {4, 4});
  Shape result_shape = ShapeUtil::MakeShape(F32, {16});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(result_shape, param0));
  builder.AddInstruction(
      HloInstruction::CreateUnary(result_shape, HloOpcode::kNegate, reshape1));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kNegate, HloOpcode::kReshape, HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, Reverse_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {8});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* reverse1 = builder.AddInstruction(
      HloInstruction::CreateReverse(param_shape, param0, {0}));
  builder.AddInstruction(
      HloInstruction::CreateUnary(param_shape, HloOpcode::kNegate, reverse1));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kNegate, HloOpcode::kReverse, HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, Slice_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {4});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {2});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* slice1 = builder.AddInstruction(
      HloInstruction::CreateSlice(slice_shape, param0, {0}, {4}, {2}));
  builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {2}), HloOpcode::kNegate, slice1));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kNegate, HloOpcode::kSlice, HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, Exponential_Transpose_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {3, 4});
  Shape result_shape = ShapeUtil::MakeShape(F32, {4, 3});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  // InstructionFusion::ShouldFuse() precludes fusing a transpose whose operand
  // is a parameter, so create an operand between the parameter and transpose.
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(param_shape, HloOpcode::kExp, param0));
  HloInstruction* transpose2 = builder.AddInstruction(
      HloInstruction::CreateTranspose(result_shape, exp1, {1, 0}));
  builder.AddInstruction(HloInstruction::CreateUnary(
      result_shape, HloOpcode::kNegate, transpose2));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(), {HloOpcode::kNegate, HloOpcode::kTranspose, HloOpcode::kExp,
                     HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, UnaryMapOfExp) {
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {3, 4});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));

  HloInstruction* exp = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kExp, param0));
  builder.AddInstruction(
      HloInstruction::CreateMap(shape, {exp}, CreateAdderToOne(module.get())));

  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(), {HloOpcode::kParameter, HloOpcode::kExp, HloOpcode::kMap});
}

TEST_F(OpcodeFusionTest, BinaryMapOfExps) {
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {3, 4});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "param"));

  HloInstruction* exp0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kExp, param0));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kExp, param1));

  builder.AddInstruction(
      HloInstruction::CreateMap(shape, {exp0, exp1}, CreateMax(module.get())));

  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(), {HloOpcode::kParameter, HloOpcode::kParameter,
                     HloOpcode::kExp, HloOpcode::kExp, HloOpcode::kMap});
}

TEST_F(OpcodeFusionTest, DynamicSliceWithDynamicUpdateSlice) {
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder builder(TestName());
  Shape full_shape = ShapeUtil::MakeShape(F32, {10, 100, 1000});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {10, 1, 1000});

  std::vector<HloInstruction*> slice_indices, update_indices;
  for (int i = 0; i < 3; ++i) {
    slice_indices.push_back(
        builder.AddInstruction(HloInstruction::CreateParameter(
            1 + i, ShapeUtil::MakeShape(U32, {}), "slice_indices")));
    update_indices.push_back(
        builder.AddInstruction(HloInstruction::CreateParameter(
            5 + i, ShapeUtil::MakeShape(U32, {}), "update_indices")));
  }
  HloInstruction* slice =
      builder.AddInstruction(HloInstruction::CreateDynamicSlice(
          slice_shape,
          builder.AddInstruction(
              HloInstruction::CreateParameter(0, full_shape, "slice_from")),
          slice_indices,
          /*slice_sizes=*/{10, 1, 1000}));

  builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      full_shape,
      builder.AddInstruction(
          HloInstruction::CreateParameter(4, full_shape, "to_update")),
      slice, update_indices));

  module->AddEntryComputation(builder.Build());
  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kDynamicSlice, HloOpcode::kDynamicUpdateSlice,
       HloOpcode::kParameter, HloOpcode::kParameter, HloOpcode::kParameter,
       HloOpcode::kParameter, HloOpcode::kParameter, HloOpcode::kParameter,
       HloOpcode::kParameter, HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, MessOfFusibleNodes) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape full_shape = ShapeUtil::MakeShape(F32, {4, 100, 10, 100, 50});

  auto loop_idx = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(S32, {}), "param0"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(S32, {}), "param1"));

  auto idx_choice = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(S32, {}),
      builder.AddInstruction(HloInstruction::CreateDynamicSlice(
          ShapeUtil::MakeShape(S32, {1}),
          builder.AddInstruction(HloInstruction::CreateParameter(
              2, ShapeUtil::MakeShape(S32, {4}), "param2")),
          {loop_idx},
          /*slice_sizes=*/{1}))));
  auto zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(0)));

  auto slice = builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      ShapeUtil::MakeShape(F32, {1, 100, 10, 100, 50}),
      builder.AddInstruction(HloInstruction::CreateParameter(
          3, ShapeUtil::MakeShape(F32, {100, 100, 10, 100, 50}), "param3")),
      {idx_choice, zero, zero, zero, zero},
      /*slice_sizes=*/{1, 100, 10, 100, 50}));

  builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      full_shape,
      builder.AddInstruction(
          HloInstruction::CreateParameter(4, full_shape, "param4")),
      slice, {loop_idx, param1, param1, param1, param1}));

  module->AddEntryComputation(builder.Build());
  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kDynamicSlice, HloOpcode::kDynamicSlice,
       HloOpcode::kDynamicUpdateSlice, HloOpcode::kReshape,
       HloOpcode::kConstant, HloOpcode::kParameter, HloOpcode::kParameter,
       HloOpcode::kParameter, HloOpcode::kParameter, HloOpcode::kParameter});
}

void CreateComputationForDotAddOutputFusionTest(const std::string& test_name,
                                                HloModule* module, int m, int k,
                                                int n,
                                                bool add_extra_use_for_dot) {
  HloComputation::Builder builder(test_name);

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

  auto* dot_lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, dot_lhs_shape, "param0"));
  auto* dot_rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, dot_rhs_shape, "param1"));
  auto* addend = builder.AddInstruction(
      HloInstruction::CreateParameter(2, dot_shape, "param2"));

  auto* dot =
      builder.AddInstruction(CreateCanonicalDot(dot_shape, dot_lhs, dot_rhs));
  builder.AddInstruction(
      HloInstruction::CreateBinary(dot_shape, HloOpcode::kAdd, dot, addend));

  if (add_extra_use_for_dot) {
    auto* token = builder.AddInstruction(HloInstruction::CreateToken());
    builder.AddInstruction(
        HloInstruction::CreateOutfeed(dot_shape, dot, token, "no_config"));
  }

  module->AddEntryComputation(builder.Build());
}

TEST_F(OpcodeFusionTest, DotAddOutputFusion_1x50x19) {
  auto module = CreateNewVerifiedModule();
  CreateComputationForDotAddOutputFusionTest(TestName(), module.get(), /*m=*/1,
                                             /*k=*/50, /*n=*/19,
                                             /*add_extra_use_for_dot=*/false);

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kDot, HloOpcode::kAdd, HloOpcode::kParameter,
       HloOpcode::kParameter, HloOpcode::kParameter},
      HloInstruction::FusionKind::kOutput);
}

TEST_F(OpcodeFusionTest, DotAddOutputFusion_19x50x1) {
  auto module = CreateNewVerifiedModule();
  CreateComputationForDotAddOutputFusionTest(TestName(), module.get(), /*m=*/19,
                                             /*k=*/50, /*n=*/1,
                                             /*add_extra_use_for_dot=*/false);

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kDot, HloOpcode::kAdd, HloOpcode::kParameter,
       HloOpcode::kParameter, HloOpcode::kParameter},
      HloInstruction::FusionKind::kOutput);
}

TEST_F(OpcodeFusionTest, DotAddOutputFusion_19x50x19) {
  auto module = CreateNewVerifiedModule();
  CreateComputationForDotAddOutputFusionTest(TestName(), module.get(), /*m=*/19,
                                             /*k=*/50, /*n=*/19,
                                             /*add_extra_use_for_dot=*/false);

  TF_ASSERT_OK_AND_ASSIGN(bool fused_something,
                          CpuInstructionFusion().Run(module.get()));
  EXPECT_FALSE(fused_something);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              Not(op::Fusion()));
}

TEST_F(OpcodeFusionTest, DotAddOutputFusion_19x50x1_multi_use) {
  auto module = CreateNewVerifiedModule();
  CreateComputationForDotAddOutputFusionTest(TestName(), module.get(), /*m=*/19,
                                             /*k=*/50, /*n=*/1,
                                             /*add_extra_use_for_dot=*/true);

  TF_ASSERT_OK_AND_ASSIGN(bool fused_something,
                          CpuInstructionFusion().Run(module.get()));
  EXPECT_FALSE(fused_something);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              Not(op::Fusion()));
}

TEST_F(InstructionFusionTest,
       DotOperationFusion_DontOutputFuseDuplicateOperands) {
  absl::string_view module_string = R"(
HloModule module

ENTRY main {
  a = f32[50,60]{1,0} parameter(0)
  b = f32[60,1]{1,0} parameter(1)
  c = f32[50,1]{1,0} dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT d = f32[50,1]{1,0} add(c, c)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool fused_something,
                          CpuInstructionFusion().Run(module.get()));
  EXPECT_FALSE(fused_something);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              Not(op::Fusion()));
}

struct GatherLoopFusionTestSpec {
  std::string test_name;
  std::string hlo_computation_text;

  static std::string Name(
      const ::testing::TestParamInfo<GatherLoopFusionTestSpec>& info) {
    return info.param.test_name;
  }
};

class GatherLoopFusionTest
    : public OpcodeFusionTest,
      public ::testing::WithParamInterface<GatherLoopFusionTestSpec> {};

TEST_P(GatherLoopFusionTest, GatherLoopFusion) {
  const GatherLoopFusionTestSpec& spec = GetParam();
  std::string hlo_string = absl::StrCat("HloModule ", spec.test_name, "\n\n",
                                        spec.hlo_computation_text);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kGather, HloOpcode::kAdd, HloOpcode::kBroadcast,
       HloOpcode::kConstant, HloOpcode::kParameter, HloOpcode::kParameter});
}

std::vector<GatherLoopFusionTestSpec> GetGatherLoopFusionTestSpecs() {
  std::vector<GatherLoopFusionTestSpec> result;

  result.push_back({"FusedTensorFlowGatherV2", R"(
ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  gather = s32[3,2] gather(operand, indices),
      offset_dims={0},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3, 1}
  one = s32[] constant(1)
  one_broadcasted = s32[3,2] broadcast(one), dimensions={}
  ROOT result = s32[3,2]{1,0} add(gather, one_broadcasted)
}
)"});

  result.push_back({"FusedTensorFlowGatherMultipleBatchDims", R"(
ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  gather = s32[2,3,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=2,
      slice_sizes={3, 1}
  one = s32[] constant(1)
  one_broadcasted = s32[2,3,2] broadcast(one), dimensions={}
  ROOT result = s32[2,3,2]{2,1,0} add(gather, one_broadcasted)
}
)"});

  result.push_back({"FusedTensorFlowGatherNdMultipleBatchDims", R"(
ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2,2] parameter(1)
  gather = s32[2,2] gather(operand, indices),
      offset_dims={},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=2,
      slice_sizes={1, 1}
  one = s32[] constant(1)
  one_broadcasted = s32[2,2] broadcast(one), dimensions={}
  ROOT result = s32[2,2]{1,0} add(gather, one_broadcasted)
}
)"});

  result.push_back({"FusedTensorFlowGatherNd_0", R"(
ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  gather = s32[2,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=1,
      slice_sizes={1,1,2}
  one = s32[] constant(1)
  one_broadcasted = s32[2,2] broadcast(one), dimensions={}
  ROOT result = s32[2,2]{1,0} add(gather, one_broadcasted)
}
)"});

  result.push_back({"FusedTensorFlowGatherNd_1", R"(
ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  gather = s32[2,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1,2}
  one = s32[] constant(1)
  one_broadcasted = s32[2,2] broadcast(one), dimensions={}
  ROOT result = s32[2,2]{1,0} add(gather, one_broadcasted)
}
)"});

  result.push_back({"FusedDynamicSlice", R"(
ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  gather = s32[1,1] gather(operand, indices),
      offset_dims={0,1},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1}
  one = s32[] constant(1)
  one_broadcasted = s32[1,1] broadcast(one), dimensions={}
  ROOT result = s32[1,1]{1,0} add(gather, one_broadcasted)
}
)"});

  result.push_back({"FusedBatchDynamicSlice", R"(
ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  gather = s32[2,1,1] gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1}
  one = s32[] constant(1)
  one_broadcasted = s32[2,1,1] broadcast(one), dimensions={}
  ROOT result = s32[2,1,1]{2,1,0} add(gather, one_broadcasted)
}
)"});

  return result;
}

INSTANTIATE_TEST_SUITE_P(GatherLoopFusionTestInstantiation,
                         GatherLoopFusionTest,
                         ::testing::ValuesIn(GetGatherLoopFusionTestSpecs()),
                         GatherLoopFusionTestSpec::Name);

TEST_F(InstructionFusionTest, NoFuseReduceMajor) {
  absl::string_view module_string = R"(
HloModule module

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  a = f32[50,60]{1,0} parameter(0)
  b = f32[50,60]{1,0} parameter(1)
  c = f32[50,60]{1,0} add(a, b)
  init = f32[] constant(0)
  ROOT r = f32[60]{0} reduce(c, init), dimensions={0}, to_apply=add
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool fused_something,
                          CpuInstructionFusion().Run(module.get()));
  EXPECT_FALSE(fused_something);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              Not(op::Fusion()));
}

TEST_F(InstructionFusionTest, FuseReduceMinor) {
  absl::string_view module_string = R"(
HloModule module

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  a = f32[50,60]{1,0} parameter(0)
  b = f32[50,60]{1,0} parameter(1)
  c = f32[50,60]{1,0} add(a, b)
  init = f32[] constant(0)
  ROOT r = f32[] reduce(c, init), dimensions={0,1}, to_apply=add
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool fused_something,
                          CpuInstructionFusion().Run(module.get()));
  EXPECT_TRUE(fused_something);
  EXPECT_THAT(module->entry_computation()->root_instruction(), op::Fusion());
}

TEST_F(OpcodeFusionTest, BigConstantNotInFusion) {
  absl::string_view module_string = R"(
HloModule module

ENTRY main {
  a = f32[1000,1000]{1,0} parameter(0)
  b = f32[1000,1000]{1,0} constant({...})
  a_plus_b = f32[1000,1000]{1,0} add(a, b)
  c = f32[1000,1000]{1,0} constant({...})
  ROOT result = f32[1000,1000]{1,0} add(a_plus_b, c)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_string));
  RunFusionAndCheckOpcodesWereFused(
      module.get(), {HloOpcode::kParameter, HloOpcode::kParameter,
                     HloOpcode::kParameter, HloOpcode::kAdd, HloOpcode::kAdd});
}

TEST_F(OpcodeFusionTest, SmallConstantInFusion) {
  absl::string_view module_string = R"(
HloModule module

ENTRY main {
  a = f32[10,10]{1,0} parameter(0)
  b = f32[10,10]{1,0} constant({...})
  a_plus_b = f32[10,10]{1,0} add(a, b)
  c = f32[10,10]{1,0} constant({...})
  ROOT result = f32[10,10]{1,0} add(a_plus_b, c)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_string));
  RunFusionAndCheckOpcodesWereFused(
      module.get(), {HloOpcode::kParameter, HloOpcode::kConstant,
                     HloOpcode::kConstant, HloOpcode::kAdd, HloOpcode::kAdd});
}

}  // namespace
}  // namespace xla::cpu

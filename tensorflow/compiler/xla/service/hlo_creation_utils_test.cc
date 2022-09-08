/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"

#include <memory>

#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class HloCreationUtilsTest : public HloTestBase {
 protected:
  std::unique_ptr<VerifiedHloModule> CreateModuleWithProgramShape(
      PrimitiveType primitive_type, absl::Span<const int64_t> input_shape_dims,
      absl::Span<const int64_t> output_shape_dims, HloInstruction** param,
      HloComputation** entry_computation) {
    Shape input_shape = ShapeUtil::MakeShape(primitive_type, input_shape_dims);
    Shape output_shape =
        ShapeUtil::MakeShape(primitive_type, output_shape_dims);
    auto module = CreateNewVerifiedModule("test");
    *entry_computation = module->AddEntryComputation(
        CreateComputationWithSignature({&input_shape}, output_shape, "entry")
            .value());
    *param = (*entry_computation)->parameter_instruction(0);
    return module;
  }

  std::unique_ptr<VerifiedHloModule> CreateModuleWithProgramShape(
      PrimitiveType primitive_type, absl::Span<const int64_t> input_shape_dims,
      absl::Span<const int64_t> output_shape_dims, HloInstruction** param,
      HloComputation** entry_computation, PrimitiveType primitive_type_output) {
    Shape input_shape = ShapeUtil::MakeShape(primitive_type, input_shape_dims);
    Shape output_shape =
        ShapeUtil::MakeShape(primitive_type_output, output_shape_dims);
    auto module = CreateNewVerifiedModule("test");
    *entry_computation = module->AddEntryComputation(
        CreateComputationWithSignature({&input_shape}, output_shape, "entry")
            .value());
    *param = (*entry_computation)->parameter_instruction(0);
    return module;
  }
};

TEST_F(HloCreationUtilsTest, CollapseFirst1Dim) {
  HloInstruction* param;
  HloComputation* entry_computation;

  auto module = CreateModuleWithProgramShape(S32, /*input_shape_dims=*/{2},
                                             /*output_shape_dims=*/{2}, &param,
                                             &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * first_1_dims_collapsed,
                          CollapseFirstNDims(param, 1));
  entry_computation->set_root_instruction(first_1_dims_collapsed);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {LiteralUtil::CreateR1<int32_t>({3, 4})}));
  CHECK_EQ(result_literal, LiteralUtil::CreateR1<int32_t>({3, 4}));
}

TEST_F(HloCreationUtilsTest, CollapseFirst2Dims) {
  HloInstruction* param;
  HloComputation* entry_computation;

  auto module = CreateModuleWithProgramShape(
      S32, /*input_shape_dims=*/{2, 3, 2}, /*output_shape_dims=*/{6, 2}, &param,
      &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * first_2_dims_collapsed,
                          CollapseFirstNDims(param, 2));
  entry_computation->set_root_instruction(first_2_dims_collapsed);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {LiteralUtil::CreateR3<int32_t>(
                                      {{{1, 2}, {3, 4}, {5, 6}},
                                       {{-1, -2}, {-3, -4}, {-5, -6}}})}));
  CHECK_EQ(result_literal,
           LiteralUtil::CreateR2<int32_t>(
               {{1, 2}, {3, 4}, {5, 6}, {-1, -2}, {-3, -4}, {-5, -6}}));
}

TEST_F(HloCreationUtilsTest, Prepend1DegenerateDim) {
  HloInstruction* param;
  HloComputation* entry_computation;

  auto module = CreateModuleWithProgramShape(S32, /*input_shape_dims=*/{2},
                                             /*output_shape_dims=*/{1, 2},
                                             &param, &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * with_1_degenerate_dim_prepended,
                          PrependDegenerateDims(param, 1));
  entry_computation->set_root_instruction(with_1_degenerate_dim_prepended);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {LiteralUtil::CreateR1<int32_t>({9, 10})}));
  CHECK_EQ(result_literal, LiteralUtil::CreateR2<int32_t>({{9, 10}}));
}

TEST_F(HloCreationUtilsTest, Prepend2DegenerateDims) {
  HloInstruction* param;
  HloComputation* entry_computation;

  auto module = CreateModuleWithProgramShape(S32, /*input_shape_dims=*/{2},
                                             /*output_shape_dims=*/{1, 1, 2},
                                             &param, &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * with_2_degenerate_dims_prepended,
                          PrependDegenerateDims(param, 2));
  entry_computation->set_root_instruction(with_2_degenerate_dims_prepended);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {LiteralUtil::CreateR1<int32_t>({9, 10})}));
  CHECK_EQ(result_literal, LiteralUtil::CreateR3<int32_t>({{{9, 10}}}));
}

TEST_F(HloCreationUtilsTest, Prepend2DegenerateDimsToScalar) {
  HloInstruction* param;
  HloComputation* entry_computation;

  auto module = CreateModuleWithProgramShape(S32, /*input_shape_dims=*/{},
                                             /*output_shape_dims=*/{1, 1},
                                             &param, &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * with_2_degenerate_dims_prepended,
                          PrependDegenerateDims(param, 2));
  entry_computation->set_root_instruction(with_2_degenerate_dims_prepended);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {LiteralUtil::CreateR0<int32_t>(9)}));
  CHECK_EQ(result_literal, LiteralUtil::CreateR2<int32_t>({{9}}));
}

TEST_F(HloCreationUtilsTest, ExpandFirstDimInto3Dims) {
  HloInstruction* param;
  HloComputation* entry_computation;

  auto module = CreateModuleWithProgramShape(S32, /*input_shape_dims=*/{6},
                                             /*output_shape_dims=*/{3, 1, 2},
                                             &param, &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * first_dim_expanded,
                          ExpandFirstDimIntoNDims(param, {3, 1, 2}));
  entry_computation->set_root_instruction(first_dim_expanded);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module,
                         {LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4, 5, 6})}));
  CHECK_EQ(result_literal,
           LiteralUtil::CreateR3<int32_t>({{{1, 2}}, {{3, 4}}, {{5, 6}}}));
}

TEST_F(HloCreationUtilsTest, PadVectorWithZeros) {
  HloInstruction* param;
  HloComputation* entry_computation;

  auto module = CreateModuleWithProgramShape(S32, /*input_shape_dims=*/{2},
                                             /*output_shape_dims=*/{6}, &param,
                                             &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(
      HloInstruction * zero_padded_param,
      PadVectorWithZeros(param, /*zeros_to_prepend=*/3, /*zeros_to_append=*/1));
  entry_computation->set_root_instruction(zero_padded_param);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {LiteralUtil::CreateR1<int32_t>({3, 4})}));
  CHECK_EQ(result_literal, LiteralUtil::CreateR1<int32_t>({0, 0, 0, 3, 4, 0}));
}

TEST_F(HloCreationUtilsTest, BroadcastZeros_S32) {
  HloInstruction* param;
  HloComputation* entry_computation;

  auto module = CreateModuleWithProgramShape(S32, /*input_shape_dims=*/{},
                                             /*output_shape_dims=*/{2, 2},
                                             &param, &entry_computation);

  HloInstruction* zeros =
      BroadcastZeros(module->entry_computation(), S32, {2, 2});
  entry_computation->set_root_instruction(zeros);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {LiteralUtil::CreateR0<int32_t>(0)}));
  CHECK_EQ(result_literal, LiteralUtil::CreateR2<int32_t>({{0, 0}, {0, 0}}));
}

TEST_F(HloCreationUtilsTest, BroadcastZeros_F32) {
  HloInstruction* param;
  HloComputation* entry_computation;

  auto module = CreateModuleWithProgramShape(F32, /*input_shape_dims=*/{},
                                             /*output_shape_dims=*/{2, 2},
                                             &param, &entry_computation);

  HloInstruction* zeros =
      BroadcastZeros(module->entry_computation(), F32, {2, 2});
  entry_computation->set_root_instruction(zeros);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {LiteralUtil::CreateR0<float>(0.0f)}));
  CHECK_EQ(result_literal,
           LiteralUtil::CreateR2<float>({{0.0f, 0.0f}, {0.0f, 0.0f}}));
}

TEST_F(HloCreationUtilsTest, MakeBitcastConvertToHlo_S32) {
  HloInstruction* param;
  HloComputation* entry_computation;

  auto module = CreateModuleWithProgramShape(S32, /*input_shape_dims=*/{2, 2},
                                             /*output_shape_dims=*/{2, 2},
                                             &param, &entry_computation, F32);
  auto* input = module->entry_computation()->AddInstruction(
      HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<int32_t>({{0, 0}, {0, 0}})));

  HloInstruction* output = MakeBitcastConvertToHlo(input, F32);
  entry_computation->set_root_instruction(output);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module,
                         {LiteralUtil::CreateR2<int32_t>({{0, 0}, {0, 0}})}));
  CHECK_EQ(result_literal,
           LiteralUtil::CreateR2<float>({{0.0f, 0.0f}, {0.0f, 0.0f}}));
}

TEST_F(HloCreationUtilsTest, MakeIotaHlo_I32) {
  HloInstruction* param;
  HloComputation* entry_computation;

  auto module = CreateModuleWithProgramShape(S32, /*input_shape_dims=*/{},
                                             /*output_shape_dims=*/{2, 2},
                                             &param, &entry_computation, F32);
  HloInstruction* output = MakeIotaHlo(module->entry_computation(),
                                       ShapeUtil::MakeShape(F32, {2, 2}), 0);
  entry_computation->set_root_instruction(output);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {LiteralUtil::CreateR0<int32_t>(0.0)}));
  CHECK_EQ(result_literal,
           LiteralUtil::CreateR2<float>({{0.0f, 0.0f}, {1.0f, 1.0f}}));
}

TEST_F(HloCreationUtilsTest, MakeBroadcast_F32) {
  HloInstruction* param;
  HloComputation* entry_computation;

  auto module = CreateModuleWithProgramShape(F32, /*input_shape_dims=*/{},
                                             /*output_shape_dims=*/{2, 2},
                                             &param, &entry_computation);
  auto* input = MakeR0ConstantHlo<float>(module->entry_computation(), 0);
  HloInstruction* output = MakeBroadcastHlo(input, {}, {2, 2});
  entry_computation->set_root_instruction(output);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {LiteralUtil::CreateR0<float>(0.0f)}));
  CHECK_EQ(result_literal,
           LiteralUtil::CreateR2<float>({{0.0f, 0.0f}, {0.0f, 0.0f}}));
}

TEST_F(HloCreationUtilsTest, MakeBroadcast_Shape_I32) {
  HloInstruction* param;
  HloComputation* entry_computation;

  auto module = CreateModuleWithProgramShape(S32, /*input_shape_dims=*/{},
                                             /*output_shape_dims=*/{2, 2},
                                             &param, &entry_computation);
  auto* input = MakeR0ConstantHlo<int32_t>(module->entry_computation(), 0);
  HloInstruction* output =
      MakeBroadcastHlo(input, {}, ShapeUtil::MakeShape(S32, {2, 2}));
  entry_computation->set_root_instruction(output);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {LiteralUtil::CreateR0<int32_t>(0.0)}));
  CHECK_EQ(result_literal, LiteralUtil::CreateR2<int32_t>({{0, 0}, {0, 0}}));
}

TEST_F(HloCreationUtilsTest, MaybeMakeTupleCrashesWithEmptyOperands) {
  EXPECT_DEATH(MaybeMakeTuple({}), "");
}

TEST_F(HloCreationUtilsTest, MaybeMakeTupleForwardsSingleElement) {
  HloInstruction* param;
  HloComputation* entry_computation;

  auto module = CreateModuleWithProgramShape(S32, /*input_shape_dims=*/{2, 2},
                                             /*output_shape_dims=*/{2, 2},
                                             &param, &entry_computation);
  HloInstruction* output = MaybeMakeTuple({param});
  entry_computation->set_root_instruction(output);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module,
                         {LiteralUtil::CreateR2<int32_t>({{0, 0}, {0, 0}})}));
  EXPECT_EQ(result_literal, LiteralUtil::CreateR2<int32_t>({{0, 0}, {0, 0}}));
}

TEST_F(HloCreationUtilsTest, MaybeMakeTupleTuplizesMultipleOperands) {
  Shape input_shape0 = ShapeUtil::MakeShape(S32, {2});
  Shape input_shape1 = ShapeUtil::MakeShape(F32, {3, 3});
  Shape output_shape =
      ShapeUtil::MakeTupleShapeWithPtrs({&input_shape1, &input_shape0});
  auto module = CreateNewVerifiedModule("test");
  HloComputation* entry_computation = module->AddEntryComputation(
      CreateComputationWithSignature({&input_shape0, &input_shape1},
                                     output_shape, "entry")
          .value());
  HloInstruction* output =
      MaybeMakeTuple({entry_computation->parameter_instruction(1),
                      entry_computation->parameter_instruction(0)});
  entry_computation->set_root_instruction(output);

  HloEvaluator evaluator;
  Literal input0 = LiteralUtil::CreateR1<int32_t>({{2, 4}});
  Literal input1 =
      LiteralUtil::CreateR2<float>({{3, 2, 1}, {4, 5, 6}, {9, 8, 7}});
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {input0.Clone(), input1.Clone()}));
  Literal expected_result = LiteralUtil::MakeTuple({&input1, &input0});
  EXPECT_EQ(result_literal, expected_result);
}
TEST_F(HloCreationUtilsTest, DynamicUpdateSliceVectorStartIndices) {
  auto module = CreateNewVerifiedModule("dus-creation-test");
  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto operand_array = std::make_unique<Array2D<double>>(2, 3);
  operand_array->FillUnique(1.0);
  auto operand_literal =
      LiteralUtil::CreateR2FromArray2D<double>(*operand_array);
  Shape input_shape = ShapeUtil::MakeShape(F64, {2, 3});
  Shape update_shape = ShapeUtil::MakeShape(F64, {2, 2});
  HloComputation* entry_computation = module->AddEntryComputation(
      CreateComputationWithSignature({&input_shape, &update_shape}, input_shape,
                                     "entry")
          .value());
  auto zero = module->entry_computation()->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  auto one = module->entry_computation()->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));
  auto update = LiteralUtil::CreateR2<double>({{-2.0, -3.0}, {-6.0, -7.0}});
  HloInstruction* dus =
      MakeDynamicUpdateSliceHlo(entry_computation->parameter_instruction(0),
                                entry_computation->parameter_instruction(1),
                                {zero, one})
          .value();
  entry_computation->set_root_instruction(dus);
  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result, evaluator.Evaluate(*module, {&operand_literal, &update}));
  auto expected = LiteralUtil::CreateR2<double>({
      {1, -2, -3},
      {5, -6, -7},
  });

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

}  // namespace
}  // namespace xla

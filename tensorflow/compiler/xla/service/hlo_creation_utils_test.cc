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
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class HloCreationUtilsTest : public HloVerifiedTestBase {
 protected:
  HloModule* CreateModuleWithProgramShape(
      PrimitiveType primitive_type, absl::Span<const int64> input_shape_dims,
      absl::Span<const int64> output_shape_dims, HloInstruction** param,
      HloComputation** entry_computation) {
    Shape input_shape = ShapeUtil::MakeShape(primitive_type, input_shape_dims);
    Shape output_shape =
        ShapeUtil::MakeShape(primitive_type, output_shape_dims);
    auto module = CreateNewModule("test");
    *entry_computation = module->AddEntryComputation(
        CreateComputationWithSignature({&input_shape}, output_shape, "entry")
            .ValueOrDie());
    *param = (*entry_computation)->parameter_instruction(0);
    return module;
  }
};

TEST_F(HloCreationUtilsTest, CollapseFirst1Dim) {
  HloInstruction* param;
  HloComputation* entry_computation;

  HloModule* module = CreateModuleWithProgramShape(S32,
                                                   /*input_shape_dims=*/{2},
                                                   /*output_shape_dims=*/{2},
                                                   &param, &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * first_1_dims_collapsed,
                          CollapseFirstNDims(param, 1));
  entry_computation->set_root_instruction(first_1_dims_collapsed);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(Literal result_literal,
                          evaluator.Evaluate<Literal>(
                              *module, {LiteralUtil::CreateR1<int32>({3, 4})}));
  CHECK_EQ(result_literal, LiteralUtil::CreateR1<int32>({3, 4}));
}

TEST_F(HloCreationUtilsTest, CollapseFirst2Dims) {
  HloInstruction* param;
  HloComputation* entry_computation;

  HloModule* module = CreateModuleWithProgramShape(
      S32,
      /*input_shape_dims=*/{2, 3, 2}, /*output_shape_dims=*/{6, 2}, &param,
      &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * first_2_dims_collapsed,
                          CollapseFirstNDims(param, 2));
  entry_computation->set_root_instruction(first_2_dims_collapsed);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate<Literal>(
          *module,
          {LiteralUtil::CreateR3<int32>(
              {{{1, 2}, {3, 4}, {5, 6}}, {{-1, -2}, {-3, -4}, {-5, -6}}})}));
  CHECK_EQ(result_literal,
           LiteralUtil::CreateR2<int32>(
               {{1, 2}, {3, 4}, {5, 6}, {-1, -2}, {-3, -4}, {-5, -6}}));
}

TEST_F(HloCreationUtilsTest, Prepend1DegenerateDim) {
  HloInstruction* param;
  HloComputation* entry_computation;

  HloModule* module = CreateModuleWithProgramShape(S32,
                                                   /*input_shape_dims=*/{2},
                                                   /*output_shape_dims=*/{1, 2},
                                                   &param, &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * with_1_degenerate_dim_prepended,
                          PrependDegenerateDims(param, 1));
  entry_computation->set_root_instruction(with_1_degenerate_dim_prepended);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate<Literal>(*module,
                                  {LiteralUtil::CreateR1<int32>({9, 10})}));
  CHECK_EQ(result_literal, LiteralUtil::CreateR2<int32>({{9, 10}}));
}

TEST_F(HloCreationUtilsTest, Prepend2DegenerateDims) {
  HloInstruction* param;
  HloComputation* entry_computation;

  HloModule* module = CreateModuleWithProgramShape(
      S32,
      /*input_shape_dims=*/{2}, /*output_shape_dims=*/{1, 1, 2}, &param,
      &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * with_2_degenerate_dims_prepended,
                          PrependDegenerateDims(param, 2));
  entry_computation->set_root_instruction(with_2_degenerate_dims_prepended);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate<Literal>(*module,
                                  {LiteralUtil::CreateR1<int32>({9, 10})}));
  CHECK_EQ(result_literal, LiteralUtil::CreateR3<int32>({{{9, 10}}}));
}

TEST_F(HloCreationUtilsTest, Prepend2DegenerateDimsToScalar) {
  HloInstruction* param;
  HloComputation* entry_computation;

  HloModule* module = CreateModuleWithProgramShape(S32,
                                                   /*input_shape_dims=*/{},
                                                   /*output_shape_dims=*/{1, 1},
                                                   &param, &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * with_2_degenerate_dims_prepended,
                          PrependDegenerateDims(param, 2));
  entry_computation->set_root_instruction(with_2_degenerate_dims_prepended);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate<Literal>(*module, {LiteralUtil::CreateR0<int32>(9)}));
  CHECK_EQ(result_literal, LiteralUtil::CreateR2<int32>({{9}}));
}

TEST_F(HloCreationUtilsTest, ExpandFirstDimInto3Dims) {
  HloInstruction* param;
  HloComputation* entry_computation;

  HloModule* module = CreateModuleWithProgramShape(
      S32,
      /*input_shape_dims=*/{6}, /*output_shape_dims=*/{3, 1, 2}, &param,
      &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * first_dim_expanded,
                          ExpandFirstDimIntoNDims(param, {3, 1, 2}));
  entry_computation->set_root_instruction(first_dim_expanded);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate<Literal>(
          *module, {LiteralUtil::CreateR1<int32>({1, 2, 3, 4, 5, 6})}));
  CHECK_EQ(result_literal,
           LiteralUtil::CreateR3<int32>({{{1, 2}}, {{3, 4}}, {{5, 6}}}));
}

TEST_F(HloCreationUtilsTest, PadVectorWithZeros) {
  HloInstruction* param;
  HloComputation* entry_computation;

  HloModule* module = CreateModuleWithProgramShape(S32,
                                                   /*input_shape_dims=*/{2},
                                                   /*output_shape_dims=*/{6},
                                                   &param, &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(
      HloInstruction * zero_padded_param,
      PadVectorWithZeros(param, /*zeros_to_prepend=*/3, /*zeros_to_append=*/1));
  entry_computation->set_root_instruction(zero_padded_param);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(Literal result_literal,
                          evaluator.Evaluate<Literal>(
                              *module, {LiteralUtil::CreateR1<int32>({3, 4})}));
  CHECK_EQ(result_literal, LiteralUtil::CreateR1<int32>({0, 0, 0, 3, 4, 0}));
}

TEST_F(HloCreationUtilsTest, BroadcastZeros_S32) {
  HloInstruction* param;
  HloComputation* entry_computation;

  HloModule* module = CreateModuleWithProgramShape(S32,
                                                   /*input_shape_dims=*/{},
                                                   /*output_shape_dims=*/{2, 2},
                                                   &param, &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(
      HloInstruction * zeros,
      BroadcastZeros(module->entry_computation(), S32, {2, 2}));
  entry_computation->set_root_instruction(zeros);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate<Literal>(*module, {LiteralUtil::CreateR0<int32>(0)}));
  CHECK_EQ(result_literal, LiteralUtil::CreateR2<int32>({{0, 0}, {0, 0}}));
}

TEST_F(HloCreationUtilsTest, BroadcastZeros_F32) {
  HloInstruction* param;
  HloComputation* entry_computation;

  HloModule* module = CreateModuleWithProgramShape(F32,
                                                   /*input_shape_dims=*/{},
                                                   /*output_shape_dims=*/{2, 2},
                                                   &param, &entry_computation);

  TF_ASSERT_OK_AND_ASSIGN(
      HloInstruction * zeros,
      BroadcastZeros(module->entry_computation(), F32, {2, 2}));
  entry_computation->set_root_instruction(zeros);

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(Literal result_literal,
                          evaluator.Evaluate<Literal>(
                              *module, {LiteralUtil::CreateR0<float>(0.0f)}));
  CHECK_EQ(result_literal,
           LiteralUtil::CreateR2<float>({{0.0f, 0.0f}, {0.0f, 0.0f}}));
}

}  // namespace
}  // namespace xla

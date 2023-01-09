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

#include "tensorflow/compiler/xla/service/convolution_group_converter.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

using ConvolutionGroupConverterTest = HloTestBase;
namespace op = testing::opcode_matchers;

TEST_F(ConvolutionGroupConverterTest,
       ConvertFeatureGroupCountEqualToInputFeatureDim) {
  std::string hlo_string = R"(HloModule Convolve1D1Window_0_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,2], filter: f32[1,1,2]) -> f32[1,2,2] {
  %input = f32[1,2,2]{2,1,0} parameter(0)
  %copy = f32[1,2,2]{2,0,1} copy(f32[1,2,2]{2,1,0} %input)
  %filter = f32[1,1,2]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,2]{2,0,1} convolution(f32[1,2,2]{2,0,1} %copy, f32[1,1,2]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f, feature_group_count=2
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  auto should_expand = [](HloInstruction* conv) { return true; };
  auto cost_model = [](HloInstruction* conv) { return true; };
  ConvolutionGroupConverter converter(should_expand, cost_model,
                                      /*convert_batch_groups_only=*/false);
  ASSERT_TRUE(converter.Run(module.get()).value());
  root = computation->root_instruction();
  // Make sure the convolution is converted to one with feature_group_count = 1.
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(root->feature_group_count(), 1);
  // Verify that the filter operand has been replaced.
  EXPECT_THAT(root->operand(1),
              op::Select(op::Eq(op::Broadcast(op::Constant()),
                                op::Broadcast(op::Constant())),
                         op::Broadcast(op::Reshape(op::Parameter())),
                         op::Broadcast(op::Constant())));
}

TEST_F(ConvolutionGroupConverterTest,
       ConvertFeatureGroupCountDivisorOfInputFeatureDim) {
  std::string hlo_string = R"(HloModule Convolve1D1Window_0_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,4], filter: f32[1,2,2]) -> f32[1,2,2] {
  %input = f32[1,2,4]{2,1,0} parameter(0)
  %copy = f32[1,2,4]{2,0,1} copy(f32[1,2,4]{2,1,0} %input)
  %filter = f32[1,2,2]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,2]{2,0,1} convolution(f32[1,2,4]{2,0,1} %copy, f32[1,2,2]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f, feature_group_count=2
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  auto should_expand = [](HloInstruction* conv) { return true; };
  auto cost_model = [](HloInstruction* conv) { return true; };
  ConvolutionGroupConverter converter(should_expand,
                                      cost_model, /*convert_batch_groups_only=*/
                                      false);
  ASSERT_TRUE(converter.Run(module.get()).value());
  root = computation->root_instruction();
  // Make sure the convolution is replaced with a reshape.
  EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(root->operand(0)->feature_group_count(), 1);
  EXPECT_EQ(root->operand(0)->shape().rank(), 4);
}

TEST_F(ConvolutionGroupConverterTest,
       ConvertBatchGroupCountEqualToInputBatchDim) {
  std::string hlo_string = R"(HloModule Convolve1D1Window_0_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[16,19,19,512]{3,2,1,0}, filter: f32[16,19,19,512]{3,2,1,0}) -> f32[3,3,512,1]{3,2,1,0} {
  %input = f32[16,19,19,512]{3,2,1,0} parameter(0)
  %filter = f32[16,19,19,512]{3,2,1,0} parameter(1)
  ROOT %convolution = f32[3,3,512,1]{3,2,1,0} convolution(f32[16,19,19,512]{3,2,1,0} %input, f32[16,19,19,512]{3,2,1,0} %filter), window={size=19x19 pad=1_1x1_1}, dim_labels=f01b_i01o->01fb, batch_group_count=512
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  auto should_expand = [](HloInstruction* conv) { return true; };
  auto cost_model = [](HloInstruction* conv) { return false; };
  ConvolutionGroupConverter converter(should_expand,
                                      cost_model, /*convert_batch_groups_only=*/
                                      true);
  ASSERT_TRUE(converter.Run(module.get()).value());
  root = computation->root_instruction();

  // Verify that the convolution is replaced by a convert.
  EXPECT_EQ(root->opcode(), HloOpcode::kConvert);
  // Make sure the convert is being fed by a reduce window.
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kReduceWindow);
}

TEST_F(ConvolutionGroupConverterTest,
       ConvertBatchGroupCountNotEqualToInputBatchDim) {
  std::string hlo_string = R"(HloModule m
  ENTRY main {
  %input = f32[1,1,1,4] parameter(0)
  %filter = f32[1,1,1,2] parameter(1)
  ROOT %convolution = f32[1,1,2,2] convolution(%input,%filter),
      window={size=1x1}, dim_labels=f01b_i01o->01fb, batch_group_count=2
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  auto should_expand = [](HloInstruction* conv) { return true; };
  auto cost_model = [](HloInstruction* conv) { return false; };
  ConvolutionGroupConverter converter(should_expand,
                                      cost_model, /*convert_batch_groups_only=*/
                                      true);
  // Make sure that batch group count is rewritten even if
  // batch_group_count == output_feature but not input_batch
  ASSERT_TRUE(converter.Run(module.get()).value());
}

}  // namespace
}  // namespace xla

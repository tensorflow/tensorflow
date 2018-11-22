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

#include "tensorflow/compiler/xla/service/convolution_feature_group_converter.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

using ConvolutionFeatureGroupConverterTest = HloTestBase;
namespace op = testing::opcode_matchers;

TEST_F(ConvolutionFeatureGroupConverterTest,
       ConvertFeatureGroupCountEqualToInputFeatureDim) {
  string hlo_string = R"(HloModule Convolve1D1Window_0_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,2], filter: f32[1,1,2]) -> f32[1,2,2] {
  %input = f32[1,2,2]{2,1,0} parameter(0)
  %copy = f32[1,2,2]{2,0,1} copy(f32[1,2,2]{2,1,0} %input)
  %filter = f32[1,1,2]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,2]{2,0,1} convolution(f32[1,2,2]{2,0,1} %copy, f32[1,1,2]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f, feature_group_count=2
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  ConvolutionFeatureGroupConverter converter;
  ASSERT_TRUE(converter.Run(module.get()).ValueOrDie());
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

TEST_F(ConvolutionFeatureGroupConverterTest,
       ConvertFeatureGroupCountDivisorOfInputFeatureDim) {
  string hlo_string = R"(HloModule Convolve1D1Window_0_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,4], filter: f32[1,2,2]) -> f32[1,2,2] {
  %input = f32[1,2,4]{2,1,0} parameter(0)
  %copy = f32[1,2,4]{2,0,1} copy(f32[1,2,4]{2,1,0} %input)
  %filter = f32[1,2,2]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,2]{2,0,1} convolution(f32[1,2,4]{2,0,1} %copy, f32[1,2,2]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f, feature_group_count=2
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  ConvolutionFeatureGroupConverter converter;
  ASSERT_TRUE(converter.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  // Make sure the convolution is replaced with a concatenate.
  EXPECT_EQ(root->opcode(), HloOpcode::kConcatenate);
  // And the operands of the concatenate are convolutions, each with a feature
  // group count = 1.
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(root->operand(0)->feature_group_count(), 1);
  EXPECT_EQ(root->operand(1)->feature_group_count(), 1);
}

}  // namespace
}  // namespace xla

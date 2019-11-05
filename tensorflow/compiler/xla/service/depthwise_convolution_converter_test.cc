/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/depthwise_convolution_converter.h"

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

using DepthwiseConvolutionConverterTest = HloTestBase;

TEST_F(DepthwiseConvolutionConverterTest,
       ConvertBatchGroupCountToFeatureGroupCount) {
  string hlo_string = R"(HloModule Convolve1D1Window_0_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[16,19,19,512]{3,2,1,0}, filter: f32[16,19,19,512]{3,2,1,0}) -> f32[3,3,512,1]{3,2,1,0} {
  %input = f32[16,19,19,512]{3,2,1,0} parameter(0)
  %filter = f32[16,19,19,512]{3,2,1,0} parameter(1)
  ROOT %convolution = f32[3,3,512,1]{3,2,1,0} convolution(f32[16,19,19,512]{3,2,1,0} %input, f32[16,19,19,512]{3,2,1,0} %filter), window={size=19x19 pad=1_1x1_1}, dim_labels=f01b_i01o->01fb, batch_group_count=512
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  auto batch_group_count = root->batch_group_count();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  auto conv_dim_num = root->convolution_dimension_numbers();
  int64 out_batch_dim = conv_dim_num.output_batch_dimension();
  int64 out_feature_dim = conv_dim_num.output_feature_dimension();
  auto cost_model = [](HloInstruction*) { return false; };
  DepthwiseConvolutionConverter converter(cost_model);
  ASSERT_TRUE(converter.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  // Verify that the convolution is replaced by a reshape.
  EXPECT_EQ(root->opcode(), HloOpcode::kReshape)
      << HloOpcodeString(root->opcode()) << " vs Reshape";

  // Verify that the operand to the reshape is the new convolution
  // with feature_group_count = batch_group_count
  auto new_conv = root->operand(0);
  EXPECT_EQ(new_conv->opcode(), HloOpcode::kConvolution)
      << HloOpcodeString(new_conv->opcode()) << " vs Convolution";
  EXPECT_EQ(new_conv->feature_group_count(), batch_group_count);
  // Verify that the output_batch_dim and output_feature_dim
  // have been swapped back (tf2xla swaps these dimensions to make use
  // of batch_group convolution for computing filter grad for depthwise
  // convolutions)
  EXPECT_EQ(new_conv->convolution_dimension_numbers().output_batch_dimension(),
            out_feature_dim);
  EXPECT_EQ(
      new_conv->convolution_dimension_numbers().output_feature_dimension(),
      out_batch_dim);

  // Verify that the operand to conv is a reshape
  auto reshape_1 = new_conv->operand(0);
  EXPECT_EQ(reshape_1->opcode(), HloOpcode::kReshape)
      << HloOpcodeString(reshape_1->opcode()) << " vs Reshape";

  // Verify that the operand to reshape_1 is transpose
  auto transpose = reshape_1->operand(0);
  EXPECT_EQ(transpose->opcode(), HloOpcode::kTranspose)
      << HloOpcodeString(transpose->opcode()) << " vs Transpose";

  // Verify that the operand to transpose is reshape
  auto reshape_2 = transpose->operand(0);
  EXPECT_EQ(reshape_2->opcode(), HloOpcode::kReshape)
      << HloOpcodeString(reshape_2->opcode()) << " vs Reshape";
}

}  // namespace
}  // namespace xla

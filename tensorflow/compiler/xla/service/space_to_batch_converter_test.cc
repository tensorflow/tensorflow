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

#include "tensorflow/compiler/xla/service/space_to_batch_converter.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

using ConvolutionSpaceToBatchConverterTest = HloTestBase;
namespace op = testing::opcode_matchers;

TEST_F(ConvolutionSpaceToBatchConverterTest, SimpleBatch1) {
  string hlo_string = R"(
  
  HloModule module
ENTRY computation {
  %p0 = bf16[1,258,258,32] parameter(0)
  %p1 = bf16[3,3,32,32] parameter(1)
  ROOT %convolution = bf16[1,256,256,32] convolution(%p0, %p1), window={size=3x3}, 
  dim_labels=b01f_01io->b01f
}

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  ConvolutionSpaceToBatchConverter converter;
  ASSERT_TRUE(converter.Run(module.get()).ValueOrDie());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Transpose());
  EXPECT_THAT(root->operand(0), op::Slice());
  auto reshape = root->operand(0)->operand(0);
  EXPECT_THAT(reshape, op::Reshape());
  EXPECT_THAT(reshape->operand(0)->operand(1), op::Convolution());
  const int64 batch_dim = reshape->operand(0)
                              ->operand(1)
                              ->convolution_dimension_numbers()
                              .output_batch_dimension();
  // Verify that the transform has increased the batch size.
  EXPECT_GT(reshape->operand(0)->shape().dimensions(batch_dim), 1);
}

TEST_F(ConvolutionSpaceToBatchConverterTest, SimpleBatch2) {
  string hlo_string = R"(
  HloModule module
  ENTRY computation {
    %p0 = bf16[2,258,258,32] parameter(0)
    %p1 = bf16[3,3,32,32] parameter(1)
    ROOT %convolution = bf16[2,256,256,32] convolution(%p0, %p1), window={size=3x3}, 
    dim_labels=b01f_01io->b01f
  }

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ConvolutionSpaceToBatchConverter converter;
  ASSERT_FALSE(converter.Run(module.get()).ValueOrDie());
}

TEST_F(ConvolutionSpaceToBatchConverterTest, Batch1WithStrideAndPad) {
  string hlo_string = R"(
  HloModule module
  ENTRY computation {
    %p0 = bf16[1,224,224,3]{3,2,1,0} parameter(0)
    %p1 = bf16[7,7,3,64]{3,2,1,0} parameter(1)
  
    ROOT %convolution.3 = bf16[1,112,112,64]{3,2,1,0} convolution(%p0, %p1), 
      window={size=7x7 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  ConvolutionSpaceToBatchConverter converter(/*limit_on_batch_size=*/4);
  ASSERT_TRUE(converter.Run(module.get()).ValueOrDie());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Transpose());
  EXPECT_THAT(root->operand(0), op::Slice());
  auto reshape = root->operand(0)->operand(0);
  EXPECT_THAT(reshape, op::Reshape());
  EXPECT_THAT(reshape->operand(0)->operand(1), op::Convolution());
  const int64 batch_dim = reshape->operand(0)
                              ->operand(1)
                              ->convolution_dimension_numbers()
                              .output_batch_dimension();

  EXPECT_GT(reshape->operand(0)->shape().dimensions(batch_dim), 4);
}

TEST_F(ConvolutionSpaceToBatchConverterTest, Batch1WithKernelDilation) {
  string hlo_string = R"(
  
  HloModule module
ENTRY computation {
  %p2 = bf16[1,7,7,128]{3,0,2,1} parameter(0)
  %p3 = bf16[1,1,512,128]{3,2,1,0} parameter(1)
  ROOT %c = bf16[1,14,14,512]{3,0,2,1} convolution(%p2, %p3),
    window={size=1x1 pad=0_1x0_1 lhs_dilate=2x2 rhs_reversal=1x1},
    dim_labels=b01f_01oi->b01f
}

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ConvolutionSpaceToBatchConverter converter;
  ASSERT_FALSE(converter.Run(module.get()).ValueOrDie());
}

}  // namespace
}  // namespace xla

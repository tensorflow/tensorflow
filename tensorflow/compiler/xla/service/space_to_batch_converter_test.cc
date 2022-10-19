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

using SpaceToBatchConverterTest = HloTestBase;
namespace op = testing::opcode_matchers;

TEST_F(SpaceToBatchConverterTest, SimpleBatch1) {
  std::string hlo_string = R"(
  
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
  SpaceToBatchConverter converter(
      SpaceToBatchController{true, true, true, true, 8});
  ASSERT_TRUE(converter.Run(module.get()).value());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Transpose());
  EXPECT_THAT(root->operand(0), op::Slice());
  auto reshape = root->operand(0)->operand(0);
  EXPECT_THAT(reshape, op::Reshape());
  auto previous_reshape = reshape->operand(0);
  EXPECT_THAT(previous_reshape, op::Reshape());
  EXPECT_THAT(previous_reshape->operand(0)->operand(1), op::Convolution());
  const int64_t batch_dim = previous_reshape->operand(0)
                                ->operand(1)
                                ->convolution_dimension_numbers()
                                .output_batch_dimension();
  // Verify that the transform has increased the batch size.
  EXPECT_GT(previous_reshape->operand(0)->shape().dimensions(batch_dim), 1);
}

TEST_F(SpaceToBatchConverterTest, SimpleBatch1ConvXpose) {
  std::string hlo_string = R"(
  
  HloModule module
ENTRY computation {
  %p0 = bf16[1,258,258,32] parameter(0)
  %p1 = bf16[3,3,32,32] parameter(1)
  %convolution = bf16[1,256,256,32] convolution(%p0, %p1), window={size=3x3}, 
  dim_labels=b01f_01io->b01f
  ROOT tr = bf16[1,256,256,32] transpose(%convolution), dimensions={0,2,1,3}
}

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  SpaceToBatchConverter converter(
      SpaceToBatchController{true, true, true, true, 8});
  ASSERT_TRUE(converter.Run(module.get()).value());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Transpose());

  EXPECT_THAT(root->operand(0), op::Slice());
  auto reshape = root->operand(0)->operand(0);
  EXPECT_THAT(reshape, op::Reshape());
  auto previous_reshape = reshape->operand(0);
  EXPECT_THAT(previous_reshape, op::Reshape());
  // This should be the original root transpose - which we handle transparently.
  EXPECT_THAT(previous_reshape->operand(0), op::Select());
  EXPECT_THAT(previous_reshape->operand(0)->operand(1), op::Convolution());
}

TEST_F(SpaceToBatchConverterTest, SimpleBatch1WithReduceWindow) {
  std::string hlo_string = R"(
  HloModule module  
  adder (lhs: bf16[], rhs: bf16[]) -> bf16[] {
    lhs = bf16[] parameter(0)
    rhs = bf16[] parameter(1)
    ROOT add = bf16[] add(lhs, rhs)
  }

  ENTRY computation {
    %p0 = bf16[1,258,258,32] parameter(0)
    %p1 = bf16[3,3,32,32] parameter(1)
    %convolution = bf16[1,256,256,32] convolution(%p0, %p1), window={size=3x3},
    dim_labels=b01f_01io->b01f
    %constant = bf16[3] constant({1.0, 2.0, 3.0})
    %tuple = (bf16[1,256,256,32], bf16[3])tuple(%convolution, %constant)
    ROOT %gte = bf16[1,256,256,32] get-tuple-element(%tuple), index=0
    %gte2 = bf16[3]get-tuple-element(%tuple), index=1
    %init = bf16[] constant(1.0)
    %reduce-window = bf16[3] reduce-window(bf16[3] %gte2, bf16[] %init),
      window={size=1}, to_apply=%adder
  }

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  SpaceToBatchConverter converter(
      SpaceToBatchController{true, true, true, true, 8});
  // Test that a reduce window consumer with different rank won't freeze the
  // compiler.
  ASSERT_TRUE(converter.Run(module.get()).value());
}

TEST_F(SpaceToBatchConverterTest, SimpleBatch2) {
  std::string hlo_string = R"(
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

  SpaceToBatchConverter converter(
      SpaceToBatchController{true, true, true, true, 1});
  ASSERT_FALSE(converter.Run(module.get()).value());
}

TEST_F(SpaceToBatchConverterTest, UnpropagatableOp) {
  std::string hlo_string = R"(
  HloModule module

  ENTRY comp {
    %reduce-window = bf16[1,76,76,64]{3,2,1,0} parameter(0)
    %convert.13 = bf16[3,3,64,64]{3,2,1,0} parameter(1)
    %convolution.1 = bf16[64,76,76,1]{0,2,1,3} convolution( 
      %reduce-window, %convert.13), window={size=3x3 pad=1_1x1_1}, 
      dim_labels=b01f_01io->f01b
     ROOT custom-call.5079 = bf16[64,152,152,1]{0,2,1,3} custom-call(%convolution.1),
     custom_call_target="ResizeNearest"
  }

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  SpaceToBatchConverter converter(
      SpaceToBatchController{true, true, true, true, 1});
  ASSERT_FALSE(converter.Run(module.get()).value());
}

TEST_F(SpaceToBatchConverterTest, Batch1WithStrideAndPad) {
  std::string hlo_string = R"(
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
  SpaceToBatchConverter converter(
      SpaceToBatchController{true, true, true, true, 4});
  ASSERT_TRUE(converter.Run(module.get()).value());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Transpose());
  EXPECT_THAT(root->operand(0), op::Slice());
  auto reshape = root->operand(0)->operand(0);
  EXPECT_THAT(reshape, op::Reshape());
  auto previous_reshape = reshape->operand(0);
  EXPECT_THAT(previous_reshape, op::Reshape());
  EXPECT_THAT(previous_reshape->operand(0)->operand(1), op::Convolution());
  const int64_t batch_dim = previous_reshape->operand(0)
                                ->operand(1)
                                ->convolution_dimension_numbers()
                                .output_batch_dimension();

  EXPECT_GT(previous_reshape->operand(0)->shape().dimensions(batch_dim), 4);
}

TEST_F(SpaceToBatchConverterTest, Batch1WithBaseDilation) {
  std::string hlo_string = R"(
  
  HloModule module
ENTRY computation {
  %p2 = bf16[1,28,28,128]{3,0,2,1} parameter(0)
  %p3 = bf16[1,1,512,128]{3,2,1,0} parameter(1)
  ROOT %c = bf16[1,56,56,512]{3,0,2,1} convolution(%p2, %p3),
    window={size=1x1 pad=0_1x0_1 lhs_dilate=2x2 rhs_reversal=1x1},
    dim_labels=b01f_01oi->b01f
}

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  SpaceToBatchConverter converter(
      SpaceToBatchController{true, true, true, true, 8});
  ASSERT_TRUE(converter.Run(module.get()).value());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Transpose());
  EXPECT_THAT(root->operand(0), op::Slice());
  auto reshape = root->operand(0)->operand(0);
  EXPECT_THAT(reshape, op::Reshape());
  auto previous_reshape = reshape->operand(0);
  EXPECT_THAT(previous_reshape, op::Reshape());
  EXPECT_THAT(previous_reshape->operand(0)->operand(1), op::Convolution());
  const int64_t batch_dim = previous_reshape->operand(0)
                                ->operand(1)
                                ->convolution_dimension_numbers()
                                .output_batch_dimension();

  EXPECT_GT(previous_reshape->operand(0)->shape().dimensions(batch_dim), 4);
}

}  // namespace
}  // namespace xla

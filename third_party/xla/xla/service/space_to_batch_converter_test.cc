/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/space_to_batch_converter.h"

#include <memory>
#include <string>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using SpaceToBatchConverterTest = HloHardwareIndependentTestBase;
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

TEST_F(SpaceToBatchConverterTest, PropagateThroughDot) {
  std::string hlo_string = R"(
  HloModule module

  ENTRY computation {
    %p0 = bf16[1,258,258,32] parameter(0)
    %p1 = bf16[3,3,32,32] parameter(1)
    %convolution = bf16[1,256,256,32] convolution(%p0, %p1), window={size=3x3},
    dim_labels=b01f_01io->b01f
    %p2 = bf16[32,32] parameter(2)
    ROOT %dot.5010 = bf16[1,256,256,32] dot(%convolution, %p2),
      lhs_contracting_dims={3},
      rhs_contracting_dims={0}
  }

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  SpaceToBatchConverter converter(
      SpaceToBatchController{true, true, true, true, 8});
  // Test that we do not start space-to-batch on conv->dot chains.
  ASSERT_TRUE(converter.Run(module.get()).value());
}

TEST_F(SpaceToBatchConverterTest, PropagateOnTrivialReduce) {
  std::string hlo_string = R"(
  HloModule module

  %region_1.37 (Arg_0.38: f32[], Arg_1.39: f32[]) -> f32[] {
    %Arg_0.38 = f32[] parameter(0)
    %Arg_1.39 = f32[] parameter(1)
    ROOT %add.40 = f32[] add(f32[] %Arg_0.38, f32[] %Arg_1.39)
  }

  ENTRY computation {
    %p0 = bf16[7,320,800,3]{3,2,1,0} parameter(0)
    %p1 = bf16[3,3,3,32]{3,2,1,0} parameter(1)
    %c = f32[7,160,400,32]{3,2,1,0} convolution( %p0,  %p1),
    window={size=3x3 stride=2x2 pad=0_1x0_1}, dim_labels=b01f_01io->b01f
    %constant.5 = f32[] constant(0)
    ROOT %reduce.41 = f32[7,160,400]{2,1,0} reduce(%c, %constant.5), dimensions={3}, to_apply=%region_1.37
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  SpaceToBatchConverter converter(
      SpaceToBatchController{true, true, true, true, /*number_of_splits=*/8});
  ASSERT_TRUE(converter.Run(module.get()).value());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Transpose());
  EXPECT_THAT(root->operand(0)->operand(0)->operand(0)->operand(0),
              op::Reduce());
  auto new_reduce = root->operand(0)->operand(0)->operand(0)->operand(0);
  // Make sure we propagated on the reduce with the larger batch size.
  EXPECT_EQ(new_reduce->shape().dimensions(1),
            // batch*number_of_splits
            7 * 8);
}

TEST_F(SpaceToBatchConverterTest, DoNotPropagateOnTupleReduce) {
  std::string hlo_string = R"(
  HloModule module

%minmax_func.2717 {
  %lhs_value.2718 = f32[] parameter(0)
  %rhs_value.2720 = f32[] parameter(2)
  %compare.2722 = pred[] compare(f32[] %lhs_value.2718, f32[] %rhs_value.2720), direction=GE
  %select.2723 = f32[] select(pred[] %compare.2722, f32[] %lhs_value.2718, f32[] %rhs_value.2720)
  %compare.2725 = pred[] compare(f32[] %lhs_value.2718, f32[] %rhs_value.2720), direction=EQ
  %lhs_index.2719 = f32[] parameter(1)
  %rhs_index.2721 = f32[] parameter(3)
  %minimum.2726 = f32[] minimum(f32[] %lhs_index.2719, f32[] %rhs_index.2721)
  %select.2724 = f32[] select(pred[] %compare.2722, f32[] %lhs_index.2719, f32[] %rhs_index.2721)
  %select.2727 = f32[] select(pred[] %compare.2725, f32[] %minimum.2726, f32[] %select.2724)
  ROOT %tuple.4 = (f32[], f32[]) tuple(f32[] %select.2723, f32[] %select.2727)
 }

  ENTRY computation {
    %p0 = bf16[7,320,800,3]{3,2,1,0} parameter(0)
    %p1 = bf16[3,3,3,32]{3,2,1,0} parameter(1)
    %c = f32[7,160,400,32]{3,2,1,0} convolution( %p0,  %p1),
    window={size=3x3 stride=2x2 pad=0_1x0_1}, dim_labels=b01f_01io->b01f
    %constant.5 = f32[] constant(0)
    %constant.6 = f32[] constant(1)
    ROOT %reduce.36 = (f32[7,160,400]{2,1,0}, f32[7,160,400]{2,1,0}) reduce(%c, %c,
    %constant.5, %constant.6), dimensions={3}, to_apply=%minmax_func.2717
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  SpaceToBatchConverter converter(
      SpaceToBatchController{true, true, true, true, /*number_of_splits=*/8});
  ASSERT_TRUE(converter.Run(module.get()).value());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Reduce());
}

TEST_F(SpaceToBatchConverterTest, ReduceDegenerateDim) {
  std::string hlo_string = R"(
  HloModule module

  %region_42.4982  {
    %Arg_0.38 = f32[] parameter(0)
    %Arg_1.39 = f32[] parameter(1)
    ROOT %add.40 = f32[] add(f32[] %Arg_0.38, f32[] %Arg_1.39)
  }

  ENTRY computation {
    %p0 = f32[2,1,84,84,3]{4,3,2,1,0}  parameter(0)
    %p1 = f32[3,3,3,3,32]{4,3,2,1,0} parameter(1)
    %constant.10559 = f32[] constant(0)
    %convolution.98 = f32[2,1,84,84,32]{4,3,2,1,0} convolution(%p0, %p1),
      window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f

    ROOT %reduce.2606 = f32[2,84,84]{2,1,0} reduce(f32[2,1,84,84,32]{4,3,2,1,0}
      %convolution.98, f32[] %constant.10559), dimensions={1,4}, to_apply=%region_42.4982
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  SpaceToBatchConverter converter(
      SpaceToBatchController{true, true, true, true, /*number_of_splits=*/8});
  ASSERT_TRUE(converter.Run(module.get()).value());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Transpose());
  EXPECT_THAT(root->operand(0), op::Slice());
}

TEST_F(SpaceToBatchConverterTest, PropagateOnReduce) {
  std::string hlo_string = R"(
HloModule xla_computation_unknown.14

region_0.134 {
  Arg_0.135 = f32[] parameter(0)
  Arg_1.136 = f32[] parameter(1)
  ROOT add.137 = f32[] add(Arg_0.135, Arg_1.136)
}

ENTRY main.140 {
  p0 = bf16[1,512,32,128]{3,2,1,0} parameter(0)
  p1 = f32[3,3,128,128]{3,2,1,0} parameter(1)
  %convolution.755 = f32[1,512,32,128]{3,2,1,0}
    convolution(p0, p1),
    window={size=3x3 pad=1_1x1_1 rhs_reversal=1x1}, dim_labels=b01f_01oi->b01f
  %constant.19458 = f32[] constant(0)
  ROOT %reduce.1354 = f32[128]{0} reduce(%convolution.755, %constant.19458),
    dimensions={0,1,2}, to_apply=%region_0.134
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  SpaceToBatchConverter converter(
      SpaceToBatchController{true, true, true, true, /*number_of_splits=*/8});
  ASSERT_TRUE(converter.Run(module.get()).value());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Reduce());
}

}  // namespace
}  // namespace xla

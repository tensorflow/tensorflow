/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/batchnorm_expander.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

class BatchNormExpanderTest : public HloTestBase {
 protected:
  // BatchNorm should have a dynamic sized divider for mean operations.
  int64_t CountGetDimensionSize(const HloModule& module) {
    int64_t count = 0;
    for (HloComputation* comp : module.computations()) {
      for (HloInstruction* inst : comp->instructions()) {
        if (inst->opcode() == HloOpcode::kGetDimensionSize) {
          count++;
        }
      }
    }
    return count;
  }
};

// Test that we expand BatchNormTraining.
TEST_F(BatchNormExpanderTest, BatchNormTraining) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 2, 2, 2});
  Shape scale_shape = ShapeUtil::MakeShape(F32, {2});
  Shape offset_shape = ShapeUtil::MakeShape(F32, {2});

  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "activation"));

  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scale_shape, "scale"));

  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, offset_shape, "offset"));

  builder.AddInstruction(HloInstruction::CreateBatchNormTraining(
      ShapeUtil::MakeTupleShape({input_shape, scale_shape, offset_shape}),
      param0, param1, param2,
      /*epsilon=*/0.001, /*feature_index=*/3));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBatchNormTraining);
  BatchNormExpander rewriter(/*rewrite_training_op=*/true,
                             /*rewrite_inference_op=*/true,
                             /*rewrite_grad_op=*/true);
  ASSERT_TRUE(rewriter.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(CountGetDimensionSize(*module), 3);
  // Make sure this operation is expanded.
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
}

// Test that we expand BatchNormGrad.
TEST_F(BatchNormExpanderTest, BatchNormGrad) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 2, 2, 2});
  Shape scale_shape = ShapeUtil::MakeShape(F32, {2});
  Shape mean_shape = ShapeUtil::MakeShape(F32, {2});
  Shape var_shape = ShapeUtil::MakeShape(F32, {2});
  Shape grad_output_shape = ShapeUtil::MakeShape(F32, {2, 2, 2, 2});

  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "activation"));

  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scale_shape, "scale"));

  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, mean_shape, "mean"));

  HloInstruction* param3 = builder.AddInstruction(
      HloInstruction::CreateParameter(3, var_shape, "var"));

  HloInstruction* param4 = builder.AddInstruction(
      HloInstruction::CreateParameter(4, grad_output_shape, "grad_output"));

  builder.AddInstruction(HloInstruction::CreateBatchNormGrad(
      ShapeUtil::MakeTupleShape({input_shape, scale_shape, mean_shape}), param0,
      param1, param2, param3, param4,
      /*epsilon=*/0.001, /*feature_index=*/3));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBatchNormGrad);
  BatchNormExpander rewriter(/*rewrite_training_op=*/true,
                             /*rewrite_inference_op=*/true,
                             /*rewrite_grad_op=*/true);
  ASSERT_TRUE(rewriter.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(CountGetDimensionSize(*module), 3);
  // Make sure this operation is expanded.
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
}

TEST_F(BatchNormExpanderTest, BatchNormTrainingSharding) {
  const char* module_str = R"(
HloModule module
ENTRY entry {
  %param.0 = f32[8,4] parameter(0)
  %param.1 = f32[4] parameter(1)
  %param.2 = f32[4] parameter(2)
  ROOT %batch-norm-training = (f32[8,4], f32[4], f32[4])
    batch-norm-training(f32[8,4] %param.0, f32[4] %param.1, f32[4] %param.2),
    epsilon=0.001, feature_index=1, sharding={maximal device=1}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  BatchNormExpander rewriter(/*rewrite_training_op=*/true,
                             /*rewrite_inference_op=*/true,
                             /*rewrite_grad_op=*/true);
  ASSERT_TRUE(rewriter.Run(m.get()).ValueOrDie());

  for (auto* instruction : m->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      continue;
    }
    auto device = instruction->sharding_unique_device();
    ASSERT_TRUE(device);
    EXPECT_EQ(*device, 1);
  }
}

}  // namespace
}  // namespace xla

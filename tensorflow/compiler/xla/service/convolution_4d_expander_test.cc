/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/convolution_4d_expander.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

using Convolution4DExpanderTest = HloTestBase;

TEST_F(Convolution4DExpanderTest, ConvertTo2DConvolution) {
  std::string hlo_string = R"(HloModule convolution_4d_fp32

ENTRY convolution_computation {
  input = f32[1,10,1,10,5,20]{5,4,3,2,1,0} parameter(0)
  kernel = f32[20,1,2,1,4,15]{5,4,3,2,1,0} parameter(1)
  ROOT conv = f32[15,1,9,1,7,5]{5,4,3,2,1,0} convolution(input, kernel), dim_labels=0123bf_i0123o->f0123b, window={size=1x2x1x4}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(root->window().dimensions_size(), 4);
  Convolution4DExpander expander_pass;
  ASSERT_TRUE(expander_pass.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
  const HloInstruction* new_convolution = root->operand(0);
  // Check that the new convolution has 2 spatial dimensions.
  EXPECT_EQ(new_convolution->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(new_convolution->window().dimensions_size(), 2);
}

TEST_F(Convolution4DExpanderTest, ConvertTo3DConvolution) {
  std::string hlo_string = R"(HloModule convolution_4d_fp32

ENTRY convolution_computation {
  input = f32[1,10,1,10,5,20]{5,4,3,2,1,0} parameter(0)
  kernel = f32[20,1,2,1,4,15]{5,4,3,2,1,0} parameter(1)
  ROOT conv = f32[15,1,9,2,7,5]{5,4,3,2,1,0} convolution(input, kernel), dim_labels=0123bf_i0123o->f0123b, window={size=1x2x1x4 pad=0_0x0_0x1_0x0_0}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(root->window().dimensions_size(), 4);
  Convolution4DExpander expander_pass;
  ASSERT_TRUE(expander_pass.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
  const HloInstruction* new_convolution = root->operand(0);
  // Check that the new convolution has 3 spatial dimensions. Note that although
  // there are 2 input dimensions of size 1, one of them is not trivial because
  // with the low padding the output dimension will be 2.
  EXPECT_EQ(new_convolution->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(new_convolution->window().dimensions_size(), 3);
}

TEST_F(Convolution4DExpanderTest, ConvertTo0DConvolution) {
  std::string hlo_string = R"(HloModule convolution_4d_fp32

ENTRY convolution_computation {
  input = f32[1,1,1,1,5,20]{5,4,3,2,1,0} parameter(0)
  kernel = f32[20,1,1,1,1,15]{5,4,3,2,1,0} parameter(1)
  ROOT conv = f32[15,1,1,1,1,5]{5,4,3,2,1,0} convolution(input, kernel), dim_labels=0123bf_i0123o->f0123b, window={size=1x1x1x1}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(root->window().dimensions_size(), 4);
  Convolution4DExpander expander_pass;
  ASSERT_TRUE(expander_pass.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
  const HloInstruction* new_convolution = root->operand(0);
  // Check that the new convolution has 0 spatial dimensions.
  EXPECT_EQ(new_convolution->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(new_convolution->window().dimensions_size(), 0);
}

TEST_F(Convolution4DExpanderTest, DontConvert3DConvolution) {
  std::string hlo_string = R"(HloModule convolution_4d_fp32

ENTRY convolution_computation {
  input = f32[1,1,1,5,20]{4,3,2,1,0} parameter(0)
  kernel = f32[20,1,1,1,15]{4,3,2,1,0} parameter(1)
  ROOT conv = f32[15,1,1,1,5]{4,3,2,1,0} convolution(input, kernel), dim_labels=012bf_i012o->f012b, window={size=1x1x1}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(root->window().dimensions_size(), 3);
  Convolution4DExpander expander_pass;
  ASSERT_FALSE(expander_pass.Run(module.get()).ValueOrDie());
}

TEST_F(Convolution4DExpanderTest, DontConvertIfNoTrivialDimensionAvailable) {
  std::string hlo_string = R"(HloModule convolution_4d_fp32

ENTRY convolution_computation {
  input = f32[2,10,2,10,5,20]{5,4,3,2,1,0} parameter(0)
  kernel = f32[20,2,2,2,4,15]{5,4,3,2,1,0} parameter(1)
  ROOT conv = f32[15,1,9,1,7,5]{5,4,3,2,1,0} convolution(input, kernel), dim_labels=0123bf_i0123o->f0123b, window={size=2x2x2x4}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(root->window().dimensions_size(), 4);
  Convolution4DExpander expander_pass;
  ASSERT_FALSE(expander_pass.Run(module.get()).ValueOrDie());
}

TEST_F(Convolution4DExpanderTest, DontConvertIfPaddingIsNonzero) {
  std::string hlo_string = R"(HloModule convolution_4d_fp32

ENTRY convolution_computation {
  input = f32[1,10,1,10,5,20]{5,4,3,2,1,0} parameter(0)
  kernel = f32[20,1,2,1,4,15]{5,4,3,2,1,0} parameter(1)
  ROOT conv = f32[15,1,9,1,7,5]{5,4,3,2,1,0} convolution(input, kernel), dim_labels=0123bf_i0123o->f0123b, window={size=1x2x1x4 stride=2x1x2x1 pad=1_0x0_0x0_1x0_0}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto computation = module->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(root->window().dimensions_size(), 4);
  Convolution4DExpander expander_pass;
  // Although we have two spatial input dimensions of size 1, and the
  // corresponding spatial output dimensions are also of size 1, these
  // dimensions are not trivial because they involve lower and/or higher padding
  // plus stride.
  ASSERT_FALSE(expander_pass.Run(module.get()).ValueOrDie());
}

}  // namespace
}  // namespace xla

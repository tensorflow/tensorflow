/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/shuffle_expander.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"

namespace xla {
namespace {

using ShuffleExpanderTest = HloHardwareIndependentTestBase;
namespace m = testing::opcode_matchers;
using ::testing::ElementsAre;

TEST_F(ShuffleExpanderTest, ExpandsRotate1D) {
  const char* hlo_string = R"(
HloModule shuffle_module

ENTRY main {
  p0 = f32[4]{0} parameter(0)
  ROOT shuffle = f32[4]{0} shuffle(p0), dimensions={0}, mode=rotate, shifts={1}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ShuffleExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);

  auto* p0 = module->entry_computation()->parameter_instruction(0);
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::Concatenate(m::Slice(m::Parameter(0)),
                                   m::Slice(m::Parameter(0))));
  EXPECT_EQ(root->concatenate_dimension(), 0);

  auto* slice0 = Cast<HloSliceInstruction>(root->operand(0));
  auto* slice1 = Cast<HloSliceInstruction>(root->operand(1));

  EXPECT_EQ(slice0->operand(0), p0);
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(1));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(4));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1));

  EXPECT_EQ(slice1->operand(0), p0);
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(1));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1));
}

TEST_F(ShuffleExpanderTest, ExpandsRotate2D) {
  const char* hlo_string = R"(
HloModule shuffle_module

ENTRY main {
  p0 = f32[2,3]{1,0} parameter(0)
  ROOT shuffle = f32[2,3]{1,0} shuffle(p0), dimensions={0,1}, mode=rotate, shifts={1,2}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ShuffleExpander expander;

  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));

  EXPECT_TRUE(changed);
  auto* p0 = module->entry_computation()->parameter_instruction(0);
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, m::Concatenate(
                        m::Slice(m::Concatenate(m::Slice(m::Parameter(0)),
                                                m::Slice(m::Parameter(0)))),
                        m::Slice(m::Concatenate(m::Slice(m::Parameter(0)),
                                                m::Slice(m::Parameter(0))))));
  EXPECT_EQ(root->concatenate_dimension(), 1);

  // The rotation of dimension 0 by 1.
  auto* dim0_rotate =
      Cast<HloConcatenateInstruction>(root->operand(0)->operand(0));
  EXPECT_EQ(root->operand(1)->operand(0), dim0_rotate);
  EXPECT_EQ(dim0_rotate->concatenate_dimension(), 0);

  auto* dim0_slice0 = Cast<HloSliceInstruction>(dim0_rotate->operand(0));
  auto* dim0_slice1 = Cast<HloSliceInstruction>(dim0_rotate->operand(1));

  EXPECT_EQ(dim0_slice0->operand(0), p0);
  EXPECT_THAT(dim0_slice0->slice_starts(), ElementsAre(1, 0));
  EXPECT_THAT(dim0_slice0->slice_limits(), ElementsAre(2, 3));
  EXPECT_THAT(dim0_slice0->slice_strides(), ElementsAre(1, 1));

  EXPECT_EQ(dim0_slice1->operand(0), p0);
  EXPECT_THAT(dim0_slice1->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(dim0_slice1->slice_limits(), ElementsAre(1, 3));
  EXPECT_THAT(dim0_slice1->slice_strides(), ElementsAre(1, 1));

  // The rotation of dimension 1 by 2.
  auto* dim1_slice0 = Cast<HloSliceInstruction>(root->operand(0));
  auto* dim1_slice1 = Cast<HloSliceInstruction>(root->operand(1));

  EXPECT_THAT(dim1_slice0->slice_starts(), ElementsAre(0, 2));
  EXPECT_THAT(dim1_slice0->slice_limits(), ElementsAre(2, 3));
  EXPECT_THAT(dim1_slice0->slice_strides(), ElementsAre(1, 1));

  EXPECT_THAT(dim1_slice1->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(dim1_slice1->slice_limits(), ElementsAre(2, 2));
  EXPECT_THAT(dim1_slice1->slice_strides(), ElementsAre(1, 1));
}

TEST_F(ShuffleExpanderTest, SimplifiesZeroShiftRotate) {
  const char* hlo_string = R"(
HloModule shuffle_module

ENTRY main {
  p0 = f32[4]{0} parameter(0)
  ROOT shuffle = f32[4]{0} shuffle(p0), dimensions={0}, mode=rotate, shifts={0}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ShuffleExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kParameter);
}

}  // namespace
}  // namespace xla

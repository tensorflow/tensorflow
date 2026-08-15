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

#include "xla/hlo/transforms/expanders/rotate_expander.h"

#include <cstdint>
#include <memory>
#include <vector>

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

using RotateExpanderTest = HloHardwareIndependentTestBase;
namespace m = testing::opcode_matchers;

TEST_F(RotateExpanderTest, ExpandsRotate1D) {
  const char* hlo_string = R"(
HloModule rotate_module

ENTRY main {
  p0 = f32[4]{0} parameter(0)
  ROOT rotate = f32[4]{0} rotate(p0), dimensions={0}, shifts={1}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  RotateExpander expander;
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
  EXPECT_EQ(slice0->slice_starts(), (std::vector<int64_t>{1}));
  EXPECT_EQ(slice0->slice_limits(), (std::vector<int64_t>{4}));
  EXPECT_EQ(slice0->slice_strides(), (std::vector<int64_t>{1}));

  EXPECT_EQ(slice1->operand(0), p0);
  EXPECT_EQ(slice1->slice_starts(), (std::vector<int64_t>{0}));
  EXPECT_EQ(slice1->slice_limits(), (std::vector<int64_t>{1}));
  EXPECT_EQ(slice1->slice_strides(), (std::vector<int64_t>{1}));
}

TEST_F(RotateExpanderTest, SimplifiesZeroShiftRotate) {
  const char* hlo_string = R"(
HloModule rotate_module

ENTRY main {
  p0 = f32[4]{0} parameter(0)
  ROOT rotate = f32[4]{0} rotate(p0), dimensions={0}, shifts={0}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  RotateExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kParameter);
}

}  // namespace
}  // namespace xla

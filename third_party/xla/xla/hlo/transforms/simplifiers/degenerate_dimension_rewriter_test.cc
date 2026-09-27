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

#include "xla/hlo/transforms/simplifiers/degenerate_dimension_rewriter.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

namespace op = xla::testing::opcode_matchers;
using ::absl_testing::IsOkAndHolds;

using DegenerateDimensionRewriterTest = HloHardwareIndependentTestBase;

TEST_F(DegenerateDimensionRewriterTest,
       ElementwiseWithReshapesAndBroadcastInputs) {
  const char* hlo_string = R"hlo(
HloModule module

ENTRY test {
  param0 = pred[1,2048,2048]{2,1,0} parameter(0)
  reshape.0 = pred[1,1,1,2048,2048]{4,3,2,1,0} reshape(param0)
  param1 = f32[2048,2048]{1,0} parameter(1)
  reshape.3 = f32[1,1,1,2048,2048]{4,3,2,1,0} reshape(param1)
  constant.49 = f32[] constant(-2.38197633e+38)
  broadcast.0 = f32[1,1,1,2048,2048]{4,3,2,1,0} broadcast(constant.49)

  ROOT  select.0 = f32[1,1,1,2048,2048]{4,3,2,1,0} select(reshape.0, reshape.3, broadcast.0)

})hlo";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  Shape original_shape =
      module->entry_computation()->root_instruction()->shape();
  DegenerateDimensionRewriter rewriter;
  EXPECT_THAT(rewriter.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Reshape(op::Select(op::Reshape(), op::Parameter(), op::Broadcast())));
  EXPECT_EQ(module->entry_computation()->root_instruction()->shape(),
            original_shape);
  Shape new_select_shape = ShapeUtil::MakeShape(F32, {2048, 2048});
  EXPECT_EQ(
      module->entry_computation()->root_instruction()->operand(0)->shape(),
      new_select_shape);
}

TEST_F(DegenerateDimensionRewriterTest, AvoidsNoOpReshapes) {
  constexpr absl::string_view kModule = R"hlo(
HloModule module

ENTRY test {
  %p0 = s32[1,1]{1,0} parameter(0)
  %iota = s32[1,2560]{1,0} iota(), iota_dimension=1

  // We wish to guard against the pass introducing a s32[] -> s32[] reshape
  // here, as at least one implementation has done in the past.
  %reshape = s32[] reshape(s32[1,1]{1,0} %p0)
  %broadcast = s32[1,2560]{1,0} broadcast(s32[] %reshape), dimensions={}
  ROOT %compare = pred[1,2560]{1,0} compare(s32[1,2560]{1,0} %iota, s32[1,2560]{1,0} %broadcast), direction=LT
})hlo";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kModule));

  DegenerateDimensionRewriter rewriter;
  EXPECT_THAT(rewriter.Run(module.get()), IsOkAndHolds(true));

  // We expect the degenerate dimension to be removed (i.e., s32[2560] instead
  // of s32[1, 2560]). Furthermore, we want to ensure that no no-op reshapes
  // are introduced.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Reshape(op::Compare()));
  EXPECT_EQ(module->entry_computation()->root_instruction()->shape(),
            ShapeUtil::MakeShape(PRED, {1, 2560}));
  const HloInstruction* reshaped_from =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_EQ(reshaped_from->opcode(), HloOpcode::kCompare);
  EXPECT_EQ(reshaped_from->shape(), ShapeUtil::MakeShape(PRED, {2560}));

  for (const auto* instruction : module->entry_computation()->instructions()) {
    EXPECT_FALSE(instruction->opcode() == HloOpcode::kReshape &&
                 ShapeUtil::Equal(instruction->shape(),
                                  instruction->operand(0)->shape()))
        << "No-op reshape detected: " << instruction->ToString();
  }
}

}  // namespace

}  // namespace xla

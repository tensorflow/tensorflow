/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/despecializer.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

class DespecializerTest : public HloHardwareIndependentTestBase {
 protected:
  Despecializer despecializer_;
};

/* Test set includes:
 * valid rw
 * valid rw multiple
 * rw lane with stride/dilation
 * 2D rw */
TEST_F(DespecializerTest, ValidRW1) {
  const std::string& hlo_text = R"(
HloModule ReduceWindow, is_scheduled=true
%add_float_.1445 {
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %maximum = bf16[] add(%lhs, %rhs)
}

ENTRY %main  {
  %param_0.938 = bf16[32,32,8,128]{3,2,1,0} parameter(0)
  %constant.381.clone.1 = bf16[] constant(0)
  ROOT %reduce-window.1 = bf16[32,32,8,128]{3,2,1,0} reduce-window(param_0.938,constant.381.clone.1), window={size=1x1x1x255 pad=0_0x0_0x0_0x127_127}, to_apply=%add_float_.1445
}
)";
  auto m = ParseAndReturnVerifiedModule(hlo_text).value();
  // Ignore the result without the deconstruct pass.
  VLOG(2) << despecializer_.Run(m.get()).value();
  // Deconstruct the reduce-window.
  despecializer_.AddReduceWindowToReduceBroadcastDeconstruct();
  EXPECT_TRUE(despecializer_.Run(m.get()).value());
  auto bcast = m->entry_computation()->root_instruction();
  auto reduce = bcast->operand(0);
  EXPECT_TRUE(bcast != nullptr && reduce != nullptr);
  EXPECT_EQ(bcast->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(reduce->opcode(), HloOpcode::kReduce);
  EXPECT_EQ(reduce->dimensions().size(), 1);
  EXPECT_EQ(reduce->dimensions(0), 3);
  EXPECT_EQ(bcast->dimensions().size(), 3);
  EXPECT_EQ(bcast->dimensions()[0], 0);
  EXPECT_EQ(bcast->dimensions()[1], 1);
  EXPECT_EQ(bcast->dimensions()[2], 2);
}

TEST_F(DespecializerTest, ValidRW2) {
  const std::string& hlo_text = R"(
HloModule ReduceWindow, is_scheduled=true
%add_float_.1445 {
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %maximum = bf16[] add(%lhs, %rhs)
}

ENTRY %main  {
  %param_0.938 = bf16[32,32,8,128]{3,2,1,0} parameter(0)
  %constant.381.clone.1 = bf16[] constant(0)
  ROOT %reduce-window.1 = bf16[32,32,8,128]{3,2,1,0} reduce-window(param_0.938,constant.381.clone.1), window={size=1x1x15x1 pad=0_0x0_0x7_7x0_0}, to_apply=%add_float_.1445
}
)";
  auto m = ParseAndReturnVerifiedModule(hlo_text).value();
  // Ignore the result without the deconstruct pass.
  VLOG(2) << despecializer_.Run(m.get()).value();
  // Deconstruct the reduce-window.
  despecializer_.AddReduceWindowToReduceBroadcastDeconstruct();
  EXPECT_TRUE(despecializer_.Run(m.get()).value());
  auto bcast = m->entry_computation()->root_instruction();
  auto reduce = bcast->operand(0);
  EXPECT_TRUE(bcast != nullptr && reduce != nullptr);
  EXPECT_EQ(bcast->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(reduce->opcode(), HloOpcode::kReduce);
  EXPECT_EQ(reduce->dimensions().size(), 1);
  EXPECT_EQ(reduce->dimensions(0), 2);
  EXPECT_EQ(bcast->dimensions().size(), 3);
  EXPECT_EQ(bcast->dimensions()[0], 0);
  EXPECT_EQ(bcast->dimensions()[1], 1);
  EXPECT_EQ(bcast->dimensions()[2], 3);
}

TEST_F(DespecializerTest, ValidRW3) {
  const std::string& hlo_text = R"(
HloModule ReduceWindow, is_scheduled=true
%add_float_.1445 {
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %maximum = bf16[] add(%lhs, %rhs)
}

ENTRY %main  {
  %param_0.938 = bf16[32,128,32,8]{1,3,2,0} parameter(0)
  %constant.381.clone.1 = bf16[] constant(0)
  ROOT %reduce-window.1 = bf16[32,128,32,8]{1,3,2,0} reduce-window(param_0.938,constant.381.clone.1), window={size=1x255x1x1 pad=0_0x127_127x0_0x0_0}, to_apply=%add_float_.1445
}
)";
  auto m = ParseAndReturnVerifiedModule(hlo_text).value();
  // Ignore the result without the deconstruct pass.
  VLOG(2) << despecializer_.Run(m.get()).value();
  // Deconstruct the reduce-window.
  despecializer_.AddReduceWindowToReduceBroadcastDeconstruct();
  EXPECT_TRUE(despecializer_.Run(m.get()).value());
  auto bcast = m->entry_computation()->root_instruction();
  auto reduce = bcast->operand(0);
  EXPECT_TRUE(bcast != nullptr && reduce != nullptr);
  EXPECT_EQ(bcast->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(reduce->opcode(), HloOpcode::kReduce);
  EXPECT_EQ(reduce->dimensions().size(), 1);
  EXPECT_EQ(reduce->dimensions(0), 1);
  EXPECT_EQ(bcast->dimensions().size(), 3);
  EXPECT_EQ(bcast->dimensions()[0], 0);
  EXPECT_EQ(bcast->dimensions()[1], 2);
  EXPECT_EQ(bcast->dimensions()[2], 3);
}

TEST_F(DespecializerTest, ValidRW4) {
  const std::string& hlo_text = R"(
HloModule ReduceWindow, is_scheduled=true
%add_float_.1445 {
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %maximum = bf16[] add(%lhs, %rhs)
}

ENTRY %main  {
  %param_0.938 = bf16[8,32,32,128]{3,0,1,2} parameter(0)
  %constant.381.clone.1 = bf16[] constant(0)
  ROOT %reduce-window.1 = bf16[8,32,32,128]{3,0,1,2} reduce-window(param_0.938,constant.381.clone.1), window={size=15x1x1x1 pad=7_7x0_0x0_0x0_0}, to_apply=%add_float_.1445
}
)";
  auto m = ParseAndReturnVerifiedModule(hlo_text).value();
  // Ignore the result without the deconstruct pass.
  VLOG(2) << despecializer_.Run(m.get()).value();
  // Deconstruct the reduce-window.
  despecializer_.AddReduceWindowToReduceBroadcastDeconstruct();
  EXPECT_TRUE(despecializer_.Run(m.get()).value());
  auto bcast = m->entry_computation()->root_instruction();
  auto reduce = bcast->operand(0);
  EXPECT_TRUE(bcast != nullptr && reduce != nullptr);
  EXPECT_EQ(bcast->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(reduce->opcode(), HloOpcode::kReduce);
  EXPECT_EQ(reduce->dimensions().size(), 1);
  EXPECT_EQ(reduce->dimensions(0), 0);
  EXPECT_EQ(bcast->dimensions().size(), 3);
  EXPECT_EQ(bcast->dimensions()[0], 1);
  EXPECT_EQ(bcast->dimensions()[1], 2);
  EXPECT_EQ(bcast->dimensions()[2], 3);
}

TEST_F(DespecializerTest, ValidRW5) {
  const std::string& hlo_text = R"(
HloModule ReduceWindow, is_scheduled=true
%add_float_.1445 {
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %maximum = bf16[] add(%lhs, %rhs)
}

ENTRY %main  {
  %param_0.938 = bf16[32,32,8,128]{3,2,1,0} parameter(0)
  %constant.381.clone.1 = bf16[] constant(0)
  ROOT %reduce-window.1 = bf16[32,32,8,128]{3,2,1,0} reduce-window(param_0.938,constant.381.clone.1), window={size=1x1x1x32 pad=0_0x0_0x0_0x0_31}, to_apply=%add_float_.1445
}
)";
  auto m = ParseAndReturnVerifiedModule(hlo_text).value();
  // Ignore the result without the deconstruct pass.
  VLOG(2) << despecializer_.Run(m.get()).value();
  // Deconstruct the reduce-window.
  despecializer_.AddReduceWindowToReduceBroadcastDeconstruct();
  EXPECT_TRUE(despecializer_.Run(m.get()).value());
  // No replacement since the reduce-window is not a candidate for
  // deconstruction.
  EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kReduceWindow);
}

TEST_F(DespecializerTest, ValidRW6) {
  const std::string& hlo_text = R"(
HloModule ReduceWindow, is_scheduled=true
%add_float_.1445 {
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %maximum = bf16[] add(%lhs, %rhs)
}

ENTRY %main  {
  %param_0.938 = bf16[32,32]{1,0} parameter(0)
  %constant.381.clone.1 = bf16[] constant(0)
  ROOT %reduce-window.2 = bf16[32,32]{1,0} reduce-window(param_0.938, constant.381.clone.1), window={size=63x1 pad=31_31x0_0}, to_apply=%add_float_.1445
}
)";
  auto m = ParseAndReturnVerifiedModule(hlo_text).value();
  // Ignore the result without the deconstruct pass.
  VLOG(2) << despecializer_.Run(m.get()).value();
  // Deconstruct the reduce-window.
  despecializer_.AddReduceWindowToReduceBroadcastDeconstruct();
  EXPECT_TRUE(despecializer_.Run(m.get()).value());
  auto bcast = m->entry_computation()->root_instruction();
  auto reduce = bcast->operand(0);
  EXPECT_TRUE(bcast != nullptr && reduce != nullptr);
  EXPECT_EQ(bcast->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(reduce->opcode(), HloOpcode::kReduce);
  EXPECT_EQ(reduce->dimensions().size(), 1);
  EXPECT_EQ(reduce->dimensions(0), 0);
  EXPECT_EQ(bcast->dimensions().size(), 1);
  EXPECT_EQ(bcast->dimensions()[0], 1);
}

TEST_F(DespecializerTest, ValidRWMultiple) {
  const std::string& hlo_text = R"(
HloModule ReduceWindow, is_scheduled=true
%add_float_.1445 {
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %maximum = bf16[] add(%lhs, %rhs)
}

ENTRY %main  {
  %param_0.938 = bf16[32,32,8,128]{3,2,1,0} parameter(0)
  %constant.381.clone.1 = bf16[] constant(0)
  ROOT %reduce-window.1 = bf16[32,32,8,128]{3,2,1,0} reduce-window(param_0.938,constant.381.clone.1), window={size=63x1x1x255 pad=31_31x0_0x0_0x127_127}, to_apply=%add_float_.1445
}
)";
  auto m = ParseAndReturnVerifiedModule(hlo_text).value();
  // Ignore the result without the deconstruct pass.
  VLOG(2) << despecializer_.Run(m.get()).value();
  // Deconstruct the reduce-window.
  despecializer_.AddReduceWindowToReduceBroadcastDeconstruct();
  EXPECT_TRUE(despecializer_.Run(m.get()).value());
  // No replacement since the reduce-window is not a candidate for
  // deconstruction.
  EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kReduceWindow);
}

TEST_F(DespecializerTest, ValidRWStrideDilation) {
  const std::string& hlo_text = R"(
HloModule ReduceWindow, is_scheduled=true
%add_float_.1445 {
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %maximum = bf16[] add(%lhs, %rhs)
}

ENTRY %main  {
  %param_0.938 = bf16[32,32,8,128]{3,2,1,0} parameter(0)
  %constant.381.clone.1 = bf16[] constant(0)
  ROOT %reduce-window.2 = bf16[32,32,8,128]{3,2,1,0} reduce-window(param_0.938, constant.381.clone.1), window={size=1x1x1x255 pad=0_0x0_0x0_0x127_127 stride=2x1x1x1 lhs_dilate=2x1x1x1}, to_apply=%add_float_.1445
}
)";
  auto m = ParseAndReturnVerifiedModule(hlo_text).value();
  // Ignore the result without the deconstruct pass.
  VLOG(2) << despecializer_.Run(m.get()).value();
  // Deconstruct the reduce-window.
  despecializer_.AddReduceWindowToReduceBroadcastDeconstruct();
  EXPECT_TRUE(despecializer_.Run(m.get()).value());
  // No replacement since the reduce-window is not a candidate for
  // deconstruction.
  EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kReduceWindow);
}

TEST_F(DespecializerTest, ValidRWShape) {
  const std::string& hlo_text = R"(
HloModule ReduceWindow, is_scheduled=true
%add_float_.1445 {
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %maximum = bf16[] add(%lhs, %rhs)
}

ENTRY %main  {
  %param_0.938 = bf16[32,32,8,128]{3,2,1,0} parameter(0)
  %constant.381.clone.1 = bf16[] constant(0)
  ROOT %reduce-window.2 = bf16[32,32,2,128]{3,2,1,0} reduce-window(param_0.938, constant.381.clone.1), window={size=1x1x7x1 pad=0_0x0_0x0_0x0_0}, to_apply=%add_float_.1445
}
)";
  auto m = ParseAndReturnVerifiedModule(hlo_text).value();
  // Ignore the result without the deconstruct pass.
  VLOG(2) << despecializer_.Run(m.get()).value();
  // Deconstruct the reduce-window.
  despecializer_.AddReduceWindowToReduceBroadcastDeconstruct();
  EXPECT_TRUE(despecializer_.Run(m.get()).value());
  // No replacement since the reduce-window is not a candidate for
  // deconstruction.
  EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kReduceWindow);
}

}  // namespace
}  // namespace xla

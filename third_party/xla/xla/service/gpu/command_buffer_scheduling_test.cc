/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/service/gpu/command_buffer_scheduling.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

namespace {

class CommandBufferSchedulingTest : public HloTestBase {};

TEST_F(CommandBufferSchedulingTest, SingleFusion) {
  const char* hlo = R"(
      HloModule TestModule

      %fused_computation (param_0: s32[], param_1: s32[]) -> s32[] {
        %p0 = s32[] parameter(0)
        %p1 = s32[] parameter(1)
        ROOT %add = s32[] add(s32[] %p0, s32[] %p1)
      }

      ENTRY %main (a: s32[], b: s32[]) -> s32[] {
        %a = s32[] parameter(0)
        %b = s32[] parameter(1)
        ROOT %fusion = s32[] fusion(s32[] %a, s32[] %b), kind=kLoop, calls=%fused_computation
      })";

  RunAndFilecheckHloRewrite(hlo, CommandBufferScheduling(), R"(
// CHECK: %command_buffer (param: s32[], param.1: s32[]) -> s32[] {
// CHECK:   %param = s32[] parameter(0)
// CHECK:   %param.1 = s32[] parameter(1)
// CHECK:   ROOT %fusion.1 = s32[] fusion(%param, %param.1), kind=kLoop, calls=%fused_computation
// CHECK: }
//
// CHECK: ENTRY %main (a: s32[], b: s32[]) -> s32[] {
// CHECK:   %a = s32[] parameter(0)
// CHECK:   %b = s32[] parameter(1)
// CHECK:   ROOT %call = s32[] call(%a, %b), to_apply=%command_buffer
// CHECK: })");
}

TEST_F(CommandBufferSchedulingTest, CollectCommandBufferSequence) {
  const char* hlo = R"(
      HloModule TestModule

      %fused_computation(param_0: s32[], param_1: s32[]) -> s32[] {
        %p0 = s32[] parameter(0)
        %p1 = s32[] parameter(1)
        ROOT %add = s32[] add(s32[] %p0, s32[] %p1)
      }

      %fused_computation.1(param_0: s32[], param_1: s32[]) -> s32[] {
        %p0 = s32[] parameter(0)
        %p1 = s32[] parameter(1)
        ROOT %add = s32[] add(s32[] %p0, s32[] %p1)
      }

      %fused_computation.2(param_0: s32[], param_1: s32[]) -> s32[] {
        %p0 = s32[] parameter(0)
        %p1 = s32[] parameter(1)
        ROOT %add = s32[] add(s32[] %p0, s32[] %p1)
      }

      %fused_computation.3(param_0: s32[], param_1: s32[]) -> s32[] {
        %p0 = s32[] parameter(0)
        %p1 = s32[] parameter(1)
        ROOT %add = s32[] add(s32[] %p0, s32[] %p1)
      }

      ENTRY %main (a: s32[], b: s32[], c: s32[], d: s32[]) -> s32[] {
        %a = s32[] parameter(0)
        %b = s32[] parameter(1)
        %fusion = s32[] fusion(s32[] %a, s32[] %b), kind=kLoop, calls=%fused_computation
        %c = s32[] parameter(2)
        %fusion.1 = s32[] fusion(s32[] %fusion, s32[] %c), kind=kLoop, calls=%fused_computation.1
        %d = s32[] parameter(3)
        %custom-call = s32[] custom-call(s32[] %fusion.1, s32[] %d), custom_call_target="some target"
        %fusion.2 = s32[] fusion(s32[] %custom-call, s32[] %c), kind=kLoop, calls=%fused_computation.2
        ROOT %fusion.3 = s32[] fusion(s32[] %custom-call, s32[] %fusion.2), kind=kLoop, calls=%fused_computation.3
      })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstructionSequence seq;
  for (HloInstruction* x : module->entry_computation()->instructions()) {
    seq.push_back(x);
  }
  EXPECT_EQ(seq.size(), 9);

  std::vector<HloInstructionSequence> command_buffer_sequences =
      CommandBufferScheduling::CollectCommandBufferSequences(seq);
  EXPECT_EQ(command_buffer_sequences.size(), 2);

  std::vector<HloInstruction*> seq_0 =
      command_buffer_sequences[0].instructions();
  EXPECT_EQ(seq_0.size(), 3);
  EXPECT_EQ(seq_0[0]->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(seq_0[1]->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(seq_0[2]->opcode(), HloOpcode::kFusion);

  std::vector<HloInstruction*> seq_1 =
      command_buffer_sequences[1].instructions();
  EXPECT_EQ(seq_1.size(), 2);
  EXPECT_EQ(seq_1[0]->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(seq_1[1]->opcode(), HloOpcode::kFusion);
}

}  // namespace

}  // namespace xla::gpu

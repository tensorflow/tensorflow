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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/hlo_parser.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

namespace {

class CommandBufferSchedulingTest : public HloTestBase {};

TEST_F(CommandBufferSchedulingTest, SingleCommandBuffer) {
  const char* hlo = R"(
      HloModule TestModule, is_scheduled=true

      %fused_computation (param_0: s32[], param_1: s32[]) -> s32[] {
        %p0 = s32[] parameter(0)
        %p1 = s32[] parameter(1)
        ROOT %add = s32[] add(s32[] %p0, s32[] %p1)
      }

      %fused_computation.1 (param_0: s32[], param_1: s32[]) -> s32[] {
        %p0 = s32[] parameter(0)
        %p1 = s32[] parameter(1)
        ROOT %add = s32[] add(s32[] %p0, s32[] %p1)
      }

      ENTRY %main (a: s32[], b: s32[]) -> s32[] {
        %a = s32[] parameter(0)
        %b = s32[] parameter(1)
        %fusion = s32[] fusion(s32[] %a, s32[] %b), kind=kLoop, calls=%fused_computation
        %fusion.1 = s32[] fusion(s32[] %a, s32[] %b), kind=kLoop, calls=%fused_computation.1
        ROOT %custom-call = s32[] custom-call(s32[] %fusion, s32[] %fusion.1), custom_call_target="some target"
      })";

  const char* expected = R"(
// CHECK: %command_buffer (param: s32[], param.1: s32[]) -> (s32[], s32[]) {
// CHECK:   %param = s32[] parameter(0)
// CHECK:   %param.1 = s32[] parameter(1)
// CHECK:   %fusion.2 = s32[] fusion(%param, %param.1), kind=kLoop, calls=%fused_computation
// CHECK:   %fusion.3 = s32[] fusion(%param, %param.1), kind=kLoop, calls=%fused_computation.1
// CHECK:   ROOT %tuple = (s32[], s32[]) tuple(%fusion.2, %fusion.3)
// CHECK: }
//
// CHECK: ENTRY %main (a: s32[], b: s32[]) -> s32[] {
// CHECK:   %a = s32[] parameter(0)
// CHECK:   %b = s32[] parameter(1)
// CHECK:   %call = (s32[], s32[]) call(%a, %b), to_apply=%command_buffer
// CHECK:   %get-tuple-element = s32[] get-tuple-element(%call), index=0
// CHECK:   %get-tuple-element.1 = s32[] get-tuple-element(%call), index=1
// CHECK:   ROOT %custom-call = s32[] custom-call(%get-tuple-element, %get-tuple-element.1), custom_call_target="some target"
// CHECK: })";

  RunAndFilecheckHloRewrite(hlo, CommandBufferScheduling(), expected,
                            [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

TEST_F(CommandBufferSchedulingTest, MultipleCommandBuffers) {
  const char* hlo = R"(
      HloModule TestModule, is_scheduled=true

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

      ENTRY %main (a: s32[], b: s32[], c: (s32[], s32[])) -> s32[] {
        %a = s32[] parameter(0)
        %b = s32[] parameter(1)
        %c = (s32[], s32[]) parameter(2)
        %fusion = s32[] fusion(s32[] %a, s32[] %b), kind=kLoop, calls=%fused_computation
        %d = s32[] get-tuple-element((s32[], s32[]) %c), index=0
        %fusion.1 = s32[] fusion(s32[] %fusion, s32[] %d), kind=kLoop, calls=%fused_computation.1
        %e = s32[] get-tuple-element((s32[], s32[]) %c), index=1
        %custom-call = s32[] custom-call(s32[] %fusion.1, s32[] %e), custom_call_target="some target"
        %fusion.2 = s32[] fusion(s32[] %custom-call, s32[] %a), kind=kLoop, calls=%fused_computation.2
        %fusion.3 = s32[] fusion(s32[] %custom-call, s32[] %fusion.2), kind=kLoop, calls=%fused_computation.3
        ROOT %custom-call.1 = s32[] custom-call(s32[] %fusion.3), custom_call_target="some target"
      })";

  const char* expected = R"(
// CHECK:  %command_buffer (param: s32[], param.1: s32[], param.2: (s32[], s32[])) -> (s32[], s32[], s32[]) {
// CHECK:    %param = s32[] parameter(0)
// CHECK:    %param.1 = s32[] parameter(1)
// CHECK:    %param.2 = (s32[], s32[]) parameter(2)
// CHECK:    %fusion.4 = s32[] fusion(%param, %param.1), kind=kLoop, calls=%fused_computation
// CHECK:    %get-tuple-element = s32[] get-tuple-element(%param.2), index=0
// CHECK:    %fusion.5 = s32[] fusion(%fusion.4, %get-tuple-element), kind=kLoop, calls=%fused_computation.1
// CHECK:    ROOT %tuple = (s32[], s32[], s32[]) tuple(%fusion.4, %get-tuple-element, %fusion.5)
// CHECK:  }

// CHECK:  %command_buffer.1 (param.3: s32[], param.4: s32[]) -> (s32[], s32[]) {
// CHECK:    %param.3 = s32[] parameter(0)
// CHECK:    %param.4 = s32[] parameter(1)
// CHECK:    %fusion.6 = s32[] fusion(%param.3, %param.4), kind=kLoop, calls=%fused_computation.2
// CHECK:    %fusion.7 = s32[] fusion(%param.3, %fusion.6), kind=kLoop, calls=%fused_computation.3
// CHECK:    ROOT %tuple.1 = (s32[], s32[]) tuple(%fusion.6, %fusion.7)
// CHECK:  }

// CHECK:  ENTRY %main (a: s32[], b: s32[], c: (s32[], s32[])) -> s32[] {
// CHECK:    %a = s32[] parameter(0)
// CHECK:    %b = s32[] parameter(1)
// CHECK:    %c = (s32[], s32[]) parameter(2)
// CHECK:    %call = (s32[], s32[], s32[]) call(%a, %b, %c), to_apply=%command_buffer
// CHECK:    %get-tuple-element.1 = s32[] get-tuple-element(%call), index=0
// CHECK:    %get-tuple-element.2 = s32[] get-tuple-element(%call), index=1
// CHECK:    %get-tuple-element.3 = s32[] get-tuple-element(%call), index=2
// CHECK:    %e = s32[] get-tuple-element(%c), index=1
// CHECK:    %custom-call = s32[] custom-call(%get-tuple-element.3, %e), custom_call_target="some target"
// CHECK:    %call.1 = (s32[], s32[]) call(%custom-call, %a), to_apply=%command_buffer.1
// CHECK:    %get-tuple-element.4 = s32[] get-tuple-element(%call.1), index=0
// CHECK:    %get-tuple-element.5 = s32[] get-tuple-element(%call.1), index=1
// CHECK:    ROOT %custom-call.1 = s32[] custom-call(%get-tuple-element.5), custom_call_target="some target"
// CHECK:  })";

  RunAndFilecheckHloRewrite(hlo, CommandBufferScheduling(), expected,
                            [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

TEST_F(CommandBufferSchedulingTest, CollectCommandBufferSequence) {
  const char* hlo = R"(
      HloModule TestModule, is_scheduled=true

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

      ENTRY %main (a: s32[], b: s32[], c: (s32[], s32[])) -> s32[] {
        %a = s32[] parameter(0)
        %b = s32[] parameter(1)
        %c = (s32[], s32[]) parameter(2)
        %fusion = s32[] fusion(s32[] %a, s32[] %b), kind=kLoop, calls=%fused_computation
        %d = s32[] get-tuple-element((s32[], s32[]) %c), index=0
        %fusion.1 = s32[] fusion(s32[] %fusion, s32[] %d), kind=kLoop, calls=%fused_computation.1
        %e = s32[] get-tuple-element((s32[], s32[]) %c), index=1
        %custom-call = s32[] custom-call(s32[] %fusion.1, s32[] %e), custom_call_target="some target"
        %fusion.2 = s32[] fusion(s32[] %custom-call, s32[] %a), kind=kLoop, calls=%fused_computation.2
        ROOT %fusion.3 = s32[] fusion(s32[] %custom-call, s32[] %fusion.2), kind=kLoop, calls=%fused_computation.3
      })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstructionSequence seq;
  for (HloInstruction* x : module->entry_computation()->instructions()) {
    seq.push_back(x);
  }
  EXPECT_EQ(seq.size(), 10);

  std::vector<HloInstructionSequence> command_buffer_sequences =
      CommandBufferScheduling::CollectCommandBufferSequences(seq);
  EXPECT_EQ(command_buffer_sequences.size(), 2);

  std::vector<HloInstruction*> seq_0 =
      command_buffer_sequences[0].instructions();
  EXPECT_EQ(seq_0.size(), 3);
  EXPECT_EQ(seq_0[0]->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(seq_0[1]->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(seq_0[2]->opcode(), HloOpcode::kFusion);

  std::vector<HloInstruction*> seq_1 =
      command_buffer_sequences[1].instructions();
  EXPECT_EQ(seq_1.size(), 2);
  EXPECT_EQ(seq_1[0]->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(seq_1[1]->opcode(), HloOpcode::kFusion);
}

TEST_F(CommandBufferSchedulingTest, MoveParametersToFront) {
  const char* hlo = R"(
      HloModule TestModule, is_scheduled=true

      %fused_computation (param_0: s32[], param_1: s32[]) -> s32[] {
        %p0 = s32[] parameter(0)
        %p1 = s32[] parameter(1)
        ROOT %add = s32[] add(s32[] %p0, s32[] %p1)
      }

      %fused_computation.1 (param_0: s32[], param_1: s32[]) -> s32[] {
        %p0 = s32[] parameter(0)
        %p1 = s32[] parameter(1)
        ROOT %add = s32[] add(s32[] %p0, s32[] %p1)
      }

      ENTRY %main (a: s32[], b: s32[], c: s32[]) -> s32[] {
        %a = s32[] parameter(0)
        %b = s32[] parameter(1)
        %fusion = s32[] fusion(s32[] %a, s32[] %b), kind=kLoop, calls=%fused_computation
        %c = s32[] parameter(2)
        ROOT %fusion.1 = s32[] fusion(s32[] %a, s32[] %c), kind=kLoop, calls=%fused_computation.1
      })";

  const char* expected = R"(
// CHECK: ENTRY %main (a: s32[], b: s32[], c: s32[]) -> s32[] {
// CHECK:   %a = s32[] parameter(0)
// CHECK:   %b = s32[] parameter(1)
// CHECK:   %c = s32[] parameter(2)
// CHECK:   %fusion = s32[] fusion(%a, %b), kind=kLoop, calls=%fused_computation
// CHECK:   ROOT %fusion.1 = s32[] fusion(%a, %c), kind=kLoop, calls=%fused_computation.1
// CHECK: })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  CommandBufferScheduling::MoveParametersToFront(module->entry_computation());
  TF_ASSERT_OK_AND_ASSIGN(
      bool filecheck_matches,
      RunFileCheck(
          module->ToString(HloPrintOptions{}.set_print_operand_shape(false)),
          expected));
  EXPECT_TRUE(filecheck_matches);
}

TEST_F(CommandBufferSchedulingTest, BuildComputation) {
  const char* hlo = R"(
      HloModule TestModule, is_scheduled=true

      %fused_computation(param_0: s32[], param_1: s32[]) -> (s32[], s32[]) {
        %p0 = s32[] parameter(0)
        %p1 = s32[] parameter(1)
        ROOT %tuple = (s32[], s32[]) tuple(s32[] %p0, s32[] %p1)
      }

      %fused_computation.1(param_0: s32[], param_1: s32[]) -> s32[] {
        %p0 = s32[] parameter(0)
        %p1 = s32[] parameter(1)
        ROOT %add = s32[] add(s32[] %p0, s32[] %p1)
      }

      ENTRY %main (a: s32[], b: s32[]) -> s32[] {
        %a = s32[] parameter(0)
        %b = s32[] custom-call(), custom_call_target="target"
        %fusion = (s32[], s32[]) fusion(s32[] %a, s32[] %b), kind=kLoop, calls=%fused_computation
        %d = s32[] get-tuple-element((s32[], s32[]) %fusion), index=0
        %fusion.1 = s32[] fusion(s32[] %a, s32[] %d), kind=kLoop, calls=%fused_computation.1
        ROOT %custom-call = s32[] custom-call(s32[] %fusion.1, s32[] %d), custom_call_target="some target"
      })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo));

  EXPECT_EQ(module->entry_computation()->instruction_count(), 6);
  std::vector<HloInstruction*> instructions;
  HloInstructionSequence seq;
  for (HloInstruction* inst : module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kFusion ||
        inst->opcode() == HloOpcode::kGetTupleElement) {
      seq.push_back(inst);
    }
    instructions.push_back(inst);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferScheduling::BuildCommandBufferResult result,
      CommandBufferScheduling::BuildCommandBuffer(seq));
  HloComputation* computation = module->AddComputationAndUnifyNamesAndIds(
      std::move(result.computation), false);

  const char* expected = R"(
// CHECK: %command_buffer (param: s32[], param.1: s32[]) -> ((s32[], s32[]), s32[], s32[]) {
// CHECK:  %param = s32[] parameter(0)
// CHECK:  %param.1 = s32[] parameter(1)
// CHECK:  %fusion.2 = (s32[], s32[]) fusion(%param, %param.1), kind=kLoop, calls=%fused_computation
// CHECK:  %get-tuple-element = s32[] get-tuple-element(%fusion.2), index=0
// CHECK:  %fusion.3 = s32[] fusion(%param, %get-tuple-element), kind=kLoop, calls=%fused_computation.1
// CHECK:  ROOT %tuple.1 = ((s32[], s32[]), s32[], s32[]) tuple(%fusion.2, %get-tuple-element, %fusion.3)
// CHECK:})";

  TF_ASSERT_OK_AND_ASSIGN(
      bool filecheck_matches,
      RunFileCheck(computation->ToString(
                       HloPrintOptions{}.set_print_operand_shape(false)),
                   expected));
  EXPECT_TRUE(filecheck_matches);

  absl::flat_hash_map<HloInstruction*, HloParameterInstruction*>&
      parameters_map = result.parameters_map;
  EXPECT_EQ(parameters_map[instructions[0]]->parameter_number(), 0);
  EXPECT_EQ(parameters_map[instructions[1]]->parameter_number(), 1);

  absl::flat_hash_map<HloInstruction*, int64_t>& inst_to_tuple_index_map =
      result.inst_to_tuple_index_map;
  EXPECT_EQ(inst_to_tuple_index_map[instructions[2]], 0);
  EXPECT_EQ(inst_to_tuple_index_map[instructions[3]], 1);
  EXPECT_EQ(inst_to_tuple_index_map[instructions[4]], 2);
}

}  // namespace

}  // namespace xla::gpu

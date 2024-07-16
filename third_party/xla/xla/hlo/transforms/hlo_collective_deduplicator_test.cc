/* Copyright 2024 The OpenXLA Authors.
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

#include "xla/hlo/transforms/hlo_collective_deduplicator.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/hlo_parser.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using HloCollectiveDeduplicatorTest = HloTestBase;

TEST_F(HloCollectiveDeduplicatorTest, SendRecv) {
  const char* module_str = R"(
    HloModule module_foo, entry_computation_layout={(s32[], token[])->(s32[], token[])}

  ENTRY %foo (arg_0: s32[], arg_1: token[]) -> (s32[], token[]) {
    %arg_0 = s32[] parameter(0)
    %arg_1 = token[] parameter(1)
    %send.0 = (s32[], u32[], token[]) send(s32[] %arg_0, token[] %arg_1), channel_id=3, is_host_transfer=true, sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}
    %send-done.1 = token[] send-done((s32[], u32[], token[]) %send.0), channel_id=3, is_host_transfer=true, sharding={maximal device=0}
    %recv.2 = (s32[], u32[], token[]) recv(token[] %send-done.1), channel_id=5, is_host_transfer=true, sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}
    %recv-done.3 = (s32[], token[]) recv-done((s32[], u32[], token[]) %recv.2), channel_id=5, is_host_transfer=true, sharding={{maximal device=0}, {maximal device=0}}
    %get-tuple-element.4 = s32[] get-tuple-element((s32[], token[]) %recv-done.3), index=0, sharding={maximal device=0}
    %get-tuple-element.5 = token[] get-tuple-element((s32[], token[]) %recv-done.3), index=1, sharding={maximal device=0}
    ROOT %tuple.6 = (s32[], token[]) tuple(s32[] %get-tuple-element.4, token[] %get-tuple-element.5)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(module_str));
  TF_ASSERT_OK(HloCollectiveDeduplicator().Run(module.get()).status());

  VLOG(1) << module->ToString();
  int num_channel_3 = 0;
  int num_channel_5 = 0;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      const HloChannelInstruction* channel_instr =
          DynCast<HloChannelInstruction>(instruction);
      if (channel_instr) {
        int64_t channel_id = channel_instr->channel_id().value();
        if (channel_id == 3) {
          num_channel_3++;
        } else if (channel_id == 5) {
          num_channel_5++;
        }
      }
    }
  }

  EXPECT_EQ(num_channel_3, 2);
  EXPECT_EQ(num_channel_5, 2);
}

TEST_F(HloCollectiveDeduplicatorTest, AllReduce) {
  const char* module_str = R"(
  HloModule module_test_all_reduce, entry_computation_layout={(f32[8]{0})->f32[8]{0}}

  %add (lhs: f32[], rhs: f32[]) -> f32[] {
    %lhs = f32[] parameter(0)
    %rhs = f32[] parameter(1)
    ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
  }

  ENTRY %test_all_reduce (input: f32[8]) -> f32[8] {
    %input = f32[8]{0} parameter(0)
    %result1 = f32[8]{0} all-reduce(f32[8]{0} %input), channel_id=1, replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=%add
    ROOT %result2 = f32[8]{0} all-reduce(f32[8]{0} %result1), channel_id=1, replica_groups={{0,1,2,3},{4,5,6,7}}, to_apply=%add
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(module_str));
  TF_ASSERT_OK(HloCollectiveDeduplicator().Run(module.get()).status());

  VLOG(1) << module->ToString();

  int num_channel_1 = 0;
  int num_channel_2 = 0;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      const HloChannelInstruction* channel_instr =
          DynCast<HloChannelInstruction>(instruction);
      if (channel_instr) {
        int64_t channel_id = channel_instr->channel_id().value();
        if (channel_id == 1) {
          num_channel_1++;
        } else if (channel_id == 2) {
          num_channel_2++;
        }
      }
    }
  }

  EXPECT_EQ(num_channel_1, 1);
  EXPECT_EQ(num_channel_2, 1);
}

}  // namespace
}  // namespace xla

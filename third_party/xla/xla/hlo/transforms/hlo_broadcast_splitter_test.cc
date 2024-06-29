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

#include "xla/hlo/transforms/hlo_broadcast_splitter.h"

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_parser.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using HloBroadcastSplitterTest = HloTestBase;

TEST_F(HloBroadcastSplitterTest, SplitBroadcast) {
  const char* module_str = R"(
    HloModule test_module

    ENTRY entry_computation {
      param = (f32[], f32[1024,1024], f32[1024,1024]) parameter(0),
        sharding={{replicated}, {devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}, {devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}}
      gte0 = f32[] get-tuple-element(param), index=0
      gte1 = f32[1024,1024] get-tuple-element(param), index=1
      gte2 = f32[1024,1024] get-tuple-element(param), index=2
      broadcast = f32[1024,1024] broadcast(gte0), dimensions={}
      add1 = f32[1024,1024] add(broadcast, gte1)
      add2 = f32[1024,1024] add(broadcast, gte2)
      ROOT root = (f32[1024,1024], f32[1024,1024]) tuple(add1, add2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(module_str));
  TF_ASSERT_OK(HloBroadcastSplitter().Run(module.get()).status());

  VLOG(1) << module->ToString();
  // Check that every broadcast has at most one user.
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kBroadcast) {
        EXPECT_LE(instruction->user_count(), 1);
      }
    }
  }
}

TEST_F(HloBroadcastSplitterTest, SplitBroadcastWithinWhileLoop) {
  const char* module_str = R"(

%cond {
  %vars.cond = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024]) parameter(0)
  %count.cond = s32[] get-tuple-element(%vars.cond), index=0
  %limit = s32[] constant(10)
  ROOT %lt = pred[] compare(%count.cond, %limit), direction=LT
}

%body {
  %param = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024]) parameter(0)
  %count = s32[] get-tuple-element(%param), index=0
  %broadcast1 = f32[1024,1024] get-tuple-element(%param), index=1
  %lhs = f32[1024,1024] get-tuple-element(%param), index=2
  %broadcast2 = f32[1024,1024] get-tuple-element(%param), index=3
  %rhs = f32[1024,1024] get-tuple-element(%param), index=4
  add1 = f32[1024,1024] add(broadcast1, lhs)
  add2 = f32[1024,1024] add(broadcast2, rhs)
  ROOT %tuple = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024]) tuple(%count, %broadcast1, %add1, %broadcast2, %add2)
}

ENTRY %entry {
  param = (f32[], f32[1024,1024], f32[1024,1024]) parameter(0),
    sharding={{replicated}, {devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}, {devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}}
  gte0 = f32[] get-tuple-element(param), index=0
  gte1 = f32[1024,1024] get-tuple-element(param), index=1
  gte2 = f32[1024,1024] get-tuple-element(param), index=2
  broadcast = f32[1024,1024] broadcast(gte0), dimensions={}
  zero = s32[] constant(0)
  tuple = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024]) tuple(zero, broadcast, gte1, broadcast, gte2)
  while = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024]) while(%tuple), body=%body, condition=%cond
  ROOT %copy = (s32[], f32[1024,1024], f32[1024,1024], f32[1024,1024], f32[1024,1024]) copy(%while)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(module_str));
  TF_ASSERT_OK(HloBroadcastSplitter().Run(module.get()).status());

  VLOG(1) << module->ToString();
  // Check that the broadcast are duplicated for multiple usage in the same
  // user.
  absl::flat_hash_set<HloInstruction*> broadcasts;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kBroadcast) {
        EXPECT_FALSE(broadcasts.contains(instruction));
        broadcasts.insert(instruction);
      }
    }
  }
}

}  // namespace
}  // namespace xla

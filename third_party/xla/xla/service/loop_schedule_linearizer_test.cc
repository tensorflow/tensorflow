/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/loop_schedule_linearizer.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/copy_insertion.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

int64_t CountCopies(const HloComputation& computation) {
  int64_t count = 0;
  for (const auto& instruction : computation.instructions()) {
    if (instruction->opcode() == HloOpcode::kCopy) {
      count++;
    }
  }
  return count;
}

int64_t CountCopies(const HloModule& module) {
  int64_t count = 0;
  for (const auto& computation : module.computations()) {
    count += CountCopies(*computation);
  }
  return count;
}

int64_t CountControlEdges(const HloComputation& computation) {
  int64_t count = 0;
  for (const auto& instruction : computation.instructions()) {
    count += instruction->control_successors().size();
  }
  return count;
}

int64_t CountControlEdges(const HloModule& module) {
  int64_t count = 0;
  for (const auto& computation : module.computations()) {
    count += CountControlEdges(*computation);
  }
  return count;
}

class LoopScheduleLinearizerTest : public HloTestBase {
 protected:
  void InsertCopies(HloModule* module, bool expect_change) {
    LoopScheduleLinearizer loop_schedule_linearizer;
    TF_ASSERT_OK_AND_ASSIGN(bool changed, loop_schedule_linearizer.Run(module));
    ASSERT_EQ(changed, expect_change);

    CopyInsertion copy_insertion;
    ASSERT_IS_OK(copy_insertion.Run(module).status());
  }
};

TEST_F(LoopScheduleLinearizerTest, NoExtraCopiesRequired) {
  absl::string_view hlo_string = R"(
HloModule module

while_body {
  input = (s32[], s32[]) parameter(0)
  counter = s32[] get-tuple-element(input), index=0
  buffer = s32[] get-tuple-element(input), index=1

  one = s32[] constant(1)

  updated_counter = s32[] add(counter, one)

  updated_buffer = s32[] add(buffer, counter)
  ROOT out = (s32[], s32[]) tuple(updated_counter, updated_buffer)
}

while_cond {
  input = (s32[], s32[]) parameter(0)
  counter = s32[] get-tuple-element(input), index=0
  bound = s32[] constant(100)
  ROOT cmp = pred[] compare(counter, bound), direction=LT
}

ENTRY entry {
  zero = s32[] constant(0)
  buffer = s32[] parameter(0)
  while_input = (s32[], s32[]) tuple(zero, buffer)
  ROOT out = (s32[], s32[]) while(while_input), condition=while_cond, body=while_body
}

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get(), /*expect_change=*/true);
  EXPECT_EQ(CountCopies(
                *module->entry_computation()->root_instruction()->while_body()),
            0);
  EXPECT_EQ(CountControlEdges(
                *module->entry_computation()->root_instruction()->while_body()),
            1);
}

TEST_F(LoopScheduleLinearizerTest, SkipAsyncCollectives) {
  absl::string_view hlo_string = R"(
HloModule module

add {
  x = s32[] parameter(0)
  y = s32[] parameter(1)
  ROOT add = s32[] add(x, y)
}

while_body {
  input = (s32[], s32[]) parameter(0)
  counter = s32[] get-tuple-element(input), index=0
  buffer = s32[] get-tuple-element(input), index=1

  one = s32[] constant(1)

  updated_counter = s32[] add(counter, one)

  updated_buffer = s32[] add(buffer, counter)
  ar_start = s32[] all-reduce-start(updated_buffer), replica_groups={}, to_apply=add
  ar_done = s32[] all-reduce-done(ar_start)
  ROOT out = (s32[], s32[]) tuple(updated_counter, ar_done)
}

while_cond {
  input = (s32[], s32[]) parameter(0)
  counter = s32[] get-tuple-element(input), index=0
  bound = s32[] constant(100)
  ROOT cmp = pred[] compare(counter, bound), direction=LT
}

ENTRY entry {
  zero = s32[] constant(0)
  buffer = s32[] parameter(0)
  while_input = (s32[], s32[]) tuple(zero, buffer)
  ROOT out = (s32[], s32[]) while(while_input), condition=while_cond, body=while_body
}

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get(), /*expect_change=*/false);
}
}  // namespace
}  // namespace xla

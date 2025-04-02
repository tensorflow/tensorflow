/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/collectives_schedule_linearizer.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/service/pattern_matcher.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace m = match;

int64_t CountControlEdges(const HloComputation& computation) {
  int64_t count = 0;
  for (const auto& instruction : computation.instructions()) {
    count += instruction->control_successors().size();
  }
  return count;
}

class CollectivesScheduleLinearizerTest
    : public HloHardwareIndependentTestBase {
 protected:
  void InsertCollectivesSchedule(HloModule* module) {
    CollectivesScheduleLinearizer collectives_schedule_linearizer;
    ASSERT_IS_OK(collectives_schedule_linearizer.Run(module).status());
  }
};

TEST_F(CollectivesScheduleLinearizerTest, FixOrdering) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = f32[] add(a, b)
}

ENTRY entry {
  p0 = f32[100] parameter(0), parameter_replication={false}
  p1 = f32[100] parameter(1), parameter_replication={false}
  c1 = f32[100] all-reduce(p0), replica_groups={}, to_apply=sum
  c2 = f32[100] all-reduce(p1), replica_groups={}, to_apply=sum
  ROOT out = f32[100] add(c1, c2)
}

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCollectivesSchedule(module.get());
  EXPECT_EQ(CountControlEdges(*module->entry_computation()), 1);
  HloInstruction *c1 = nullptr, *c2 = nullptr;
  for (HloInstruction* instr : module->entry_computation()->instructions()) {
    if (Match(instr, m::AllReduce(m::Parameter(0)))) {
      c1 = instr;
    }
    if (Match(instr, m::AllReduce(m::Parameter(1)))) {
      c2 = instr;
    }
  }
  EXPECT_TRUE(c1 != nullptr && c2 != nullptr);
  EXPECT_TRUE(absl::c_linear_search(c2->control_predecessors(), c1));
}

TEST_F(CollectivesScheduleLinearizerTest, NoFixRequired) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = f32[] add(a, b)
}

ENTRY entry {
  p0 = f32[100] parameter(0), parameter_replication={false}
  p1 = f32[100] parameter(1), parameter_replication={false}
  c1 = f32[100] all-reduce(p0), replica_groups={}, to_apply=sum
  c2 = f32[100] all-reduce(p1), replica_groups={}, to_apply=sum, control-predecessors={c1}
  ROOT out = f32[100] add(c1, c2)
}

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCollectivesSchedule(module.get());
  EXPECT_EQ(CountControlEdges(*module->entry_computation()), 1);
}

TEST_F(CollectivesScheduleLinearizerTest, DependentCollectives) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = f32[] add(a, b)
}

ENTRY entry {
  p0 = f32[100] parameter(0), parameter_replication={false}
  p1 = f32[100] parameter(1), parameter_replication={false}
  c1 = f32[100] all-reduce(p0), replica_groups={}, to_apply=sum
  c2 = f32[100] all-reduce(c1), replica_groups={}, to_apply=sum
  ROOT out = f32[100] add(c1, c2)
}

  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCollectivesSchedule(module.get());
  EXPECT_EQ(CountControlEdges(*module->entry_computation()), 0);
}

TEST_F(CollectivesScheduleLinearizerTest, NonPostorder) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = f32[] add(a, b)
}

ENTRY entry {
  p0 = f32[100] parameter(0), parameter_replication={false}
  p1 = f32[100] parameter(1), parameter_replication={false}
  c1 = f32[100] all-reduce(p0), replica_groups={}, to_apply=sum
  c2 = f32[100] all-reduce(p1), replica_groups={}, to_apply=sum
  c3 = f32[100] all-reduce(p1), replica_groups={}, to_apply=sum
  t = f32[100] add(c1, c2)
  ROOT out = f32[100] add(t, c3)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_IS_OK(
      module->entry_computation()
          ->GetInstructionWithName("c3")
          ->AddControlDependencyTo(
              module->entry_computation()->GetInstructionWithName("c1")));
  InsertCollectivesSchedule(module.get());
  EXPECT_EQ(CountControlEdges(*module->entry_computation()), 2);
}

TEST_F(CollectivesScheduleLinearizerTest, AsyncOrdering) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = f32[] add(a, b)
}

ENTRY entry {
  p0 = f32[100] parameter(0), parameter_replication={false}
  p1 = f32[100] parameter(1), parameter_replication={false}
  ars0 = f32[100] all-reduce-start(p0), replica_groups={}, to_apply=sum
  ard0 = f32[100] all-reduce-done(ars0)
  ars1 = f32[100] all-reduce-start(p1), replica_groups={}, to_apply=sum
  ard1 = f32[100] all-reduce-done(ars1)
  ROOT out = f32[100] add(ard0, ard1)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCollectivesSchedule(module.get());
  EXPECT_EQ(CountControlEdges(*module->entry_computation()), 1);

  const HloInstruction *root = module->entry_computation()->root_instruction();
  const HloInstruction *ard0 = root->operand(0);
  const HloInstruction *ard1 = root->operand(1);
  EXPECT_EQ(ard0->opcode(), HloOpcode::kAllReduceDone);
  EXPECT_EQ(ard1->opcode(), HloOpcode::kAllReduceDone);

  const HloInstruction *ars1 = ard1->operand(0);
  EXPECT_EQ(ars1->opcode(), HloOpcode::kAllReduceStart);

  // verify control dependency is inserted from all-reduce-done to
  // all-reduce-start.
  EXPECT_TRUE(absl::c_linear_search(ars1->control_predecessors(), ard0));
}

TEST_F(CollectivesScheduleLinearizerTest, DefUseOrder) {
  absl::string_view hlo_string = R"(
HloModule module

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = f32[] add(a, b)
}

ENTRY entry {
  p0 = f32[100] parameter(0), parameter_replication={false}
  p1 = f32[100] parameter(1), parameter_replication={false}
  i0 = f32[100] add(p0, p1)
  i1 = f32[100] multiply(p0, p1)
  i2 = f32[100] divide(p0, p1)
  c1 = f32[100] all-reduce(i0), replica_groups={}, to_apply=sum, channel_id=1
  c2 = f32[100] all-reduce(i1), replica_groups={}, to_apply=sum, channel_id=1
  c3 = f32[100] all-reduce(i2), replica_groups={}, to_apply=sum, channel_id=1
  t = f32[100] add(c1, c2)
  ROOT out = f32[100] add(t, c3)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCollectivesSchedule(module.get());
  EXPECT_EQ(CountControlEdges(*module->entry_computation()), 2);

  const HloInstruction *root = module->entry_computation()->root_instruction();
  const HloInstruction *t = root->operand(0);   // t = add(c1, c2)
  const HloInstruction *c3 = root->operand(1);  // c3 = all-reduce(i2)...
  EXPECT_EQ(t->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(c3->opcode(), HloOpcode::kAllReduce);

  const HloInstruction *c1 = t->operand(0);
  const HloInstruction *c2 = t->operand(1);
  EXPECT_EQ(c1->opcode(), HloOpcode::kAllReduce);
  EXPECT_EQ(c2->opcode(), HloOpcode::kAllReduce);

  bool found_i0 = false;
  // Verify that i0 is before c1.
  for (const auto &instruction : module->entry_computation()->instructions()) {
    if (instruction->name() == "c1") EXPECT_TRUE(found_i0);
    if (instruction->name() == "i0") found_i0 = true;
  }
}

}  // namespace
}  // namespace xla

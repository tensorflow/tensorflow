/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
;you may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/list_scheduler.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class ListSchedulerTest : public HloHardwareIndependentTestBase {
 protected:
  void ExpectSchedule(const HloModule& module,
                      absl::Span<const absl::string_view> want_names) {
    const HloSchedule& schedule = module.schedule();
    const HloInstructionSequence& sequence =
        schedule.sequence(module.entry_computation());

    std::vector<absl::string_view> got_names;
    for (const HloInstruction* inst : sequence.instructions()) {
      got_names.push_back(inst->name());
    }

    EXPECT_EQ(got_names.size(), want_names.size());
    for (int i = 0; i < got_names.size(); ++i) {
      EXPECT_EQ(got_names[i], want_names[i]);
    }
  }
};

TEST_F(ListSchedulerTest, SimpleMath) {
  absl::string_view hlo_string = R"(
    HloModule SimpleMathModule

    ENTRY %main {
      x = f32[10] parameter(0)
      add = f32[10] add(x, x)
      ROOT sin = f32[10] sine(add)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ListScheduler scheduler;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, scheduler.Run(hlo_module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(hlo_module->has_schedule());
  ExpectSchedule(*hlo_module, {"x", "add", "sin"});
}

TEST_F(ListSchedulerTest, CustomCalls) {
  //     a
  //   /   \
  //  b     d
  //  |     |
  //  c     e
  //   \   /
  //     f
  absl::string_view hlo_string = R"(
    HloModule CustomCallsModule

    ENTRY %main {
      %a = f32[10] parameter(0)
      %b = f32[5] custom-call(a), custom_call_target="target1"
      %c = f32[6] custom-call(b), custom_call_target="target2"
      %d = f32[10] custom-call(a), custom_call_target="target3"
      %e = f32[10] custom-call(d), custom_call_target="target4"
      ROOT %f = f32[10] custom-call(c, e), custom_call_target="target5"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ListScheduler scheduler;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, scheduler.Run(hlo_module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(hlo_module->has_schedule());
  ExpectSchedule(*hlo_module, {"a", "b", "d", "e", "c", "f"});
}

TEST_F(ListSchedulerTest, AsyncCalls) {
  //     a
  //   /   \
  //  b     start
  //  |     |
  //  c     done
  //   \   /
  //     d
  absl::string_view hlo_string = R"(
    HloModule AsyncCallsModule

    ENTRY %main {
      %a = f32[10] parameter(0)
      %b = f32[5] custom-call(a), custom_call_target="target1"
      %c = f32[6] custom-call(b), custom_call_target="target2"
      %start = (f32[10], f32[20]) all-gather-start(a), replica_groups={{0,1}}, dimensions={0}
      %done = f32[20] all-gather-done(start)
      ROOT %d = f32[10] custom-call(c, done), custom_call_target="target3"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ListScheduler scheduler;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, scheduler.Run(hlo_module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(hlo_module->has_schedule());
  ExpectSchedule(*hlo_module, {"a", "start", "b", "c", "done", "d"});
}

}  // namespace
}  // namespace xla

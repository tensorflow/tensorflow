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

#include "xla/service/cpu/runtime/thunk.h"

#include <cstdint>

#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

class ThunkExecuteStateTestHelper : public Thunk {
 public:
  static ExecuteState CreateExecuteState(int64_t parallel_tasks) {
    return ExecuteState(parallel_tasks);
  }
};

TEST(ThunkTest, OkExecuteEvent) {
  auto event = Thunk::OkExecuteEvent();
  ASSERT_TRUE(event.IsConcrete());
}

TEST(ThunkExecuteStateTest, OneTask) {
  auto execute_state =
      ThunkExecuteStateTestHelper::CreateExecuteState(/*parallel_tasks=*/1);

  // State should not be available right after construction.
  EXPECT_FALSE(execute_state.event.IsAvailable());

  // Notifying once should make the state available.
  execute_state.Notify();
  EXPECT_TRUE(execute_state.event.IsAvailable());
}

TEST(ThunkExecuteStateTest, MultipleTasks) {
  int parallel_tasks = 10;
  auto execute_state =
      ThunkExecuteStateTestHelper::CreateExecuteState(parallel_tasks);

  for (int i = 0; i < parallel_tasks; ++i) {
    // State should not be available until all tasks are notified.
    EXPECT_FALSE(execute_state.event.IsAvailable());
    execute_state.Notify();
  }

  // All tasks are notified, state should be available.
  EXPECT_TRUE(execute_state.event.IsAvailable());
}

}  // namespace
}  // namespace xla::cpu

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

#include "xla/backends/gpu/runtime/command_state.h"

#include <cstdint>

#include "xla/tsl/platform/test.h"

namespace xla::gpu {
namespace {

struct StateA : public CommandState {
  int32_t value = 0;
};

struct StateB : public CommandState {
  float value = 0;
};

TEST(CommandStateManagerTest, GetOrCreateState) {
  // We need a fake command pointer to use as a key. Nullptr works just fine!
  const Command* cmd = nullptr;

  CommandStateManager state_manager;

  // Create a state of type StateA.
  auto* stateA0 =
      state_manager.GetOrNull<StateA>(cmd, /*command_buffer=*/nullptr);
  ASSERT_EQ(stateA0, nullptr);

  auto* stateA1 =
      state_manager.GetOrCreate<StateA>(cmd, /*command_buffer=*/nullptr);
  ASSERT_EQ(stateA1->value, 0);
  stateA1->value += 42;

  auto* stateA2 =
      state_manager.GetOrCreate<StateA>(cmd, /*command_buffer=*/nullptr);
  ASSERT_EQ(stateA2->value, 42);
  ASSERT_EQ(stateA1, stateA2);

  // StateB has a different type, and has no connection to StateA created above.
  auto* stateB0 =
      state_manager.GetOrNull<StateB>(cmd, /*command_buffer=*/nullptr);
  ASSERT_EQ(stateB0, nullptr);

  auto* stateB1 =
      state_manager.GetOrCreate<StateB>(cmd, /*command_buffer=*/nullptr);
  ASSERT_EQ(stateB1->value, 0);
  stateB1->value += 42.0;

  auto* stateB2 =
      state_manager.GetOrCreate<StateB>(cmd, /*command_buffer=*/nullptr);
  ASSERT_EQ(stateB2->value, 42.0);
  ASSERT_EQ(stateB1, stateB2);
}

}  // namespace
}  // namespace xla::gpu

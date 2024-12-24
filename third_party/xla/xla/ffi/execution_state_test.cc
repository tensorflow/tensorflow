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

#include "xla/ffi/execution_state.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::ffi {

using TypeId = ExecutionState::TypeId;

using ::testing::HasSubstr;

TEST(ExecutionStateTest, SetAndGet) {
  ExecutionState state;
  EXPECT_FALSE(state.IsSet());

  {  // Empty state returns an error from Get().
    auto data = state.Get(TypeId(1));
    EXPECT_THAT(data.status().message(), HasSubstr("State is not set"));
  }

  {  // Empty state returns an error from Get().
    auto data = state.Get<int32_t>();
    EXPECT_THAT(data.status().message(), HasSubstr("State is not set"));
  }

  // Once set, state can be retrieved.
  TF_ASSERT_OK(state.Set(std::make_unique<int32_t>(42)));
  EXPECT_TRUE(state.IsSet());

  TF_ASSERT_OK_AND_ASSIGN(int32_t* data, state.Get<int32_t>());
  EXPECT_EQ(*data, 42);
}

}  // namespace xla::ffi

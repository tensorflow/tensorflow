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

#include "xla/ffi/execution_context.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::ffi {

struct StringUserData {
  std::string data;
};

struct I32UserData : public ExecutionContext::UserData {
  explicit I32UserData(int32_t value) : value(value) {}
  int32_t value;
};

TEST(ExecutionContextTest, OpaqueUserData) {
  StringUserData string_data = {"foo"};
  auto deleter = [](void*) {};

  ExecutionContext context;
  TF_ASSERT_OK(context.Emplace("foo", &string_data, deleter));

  TF_ASSERT_OK_AND_ASSIGN(auto opaque_data, context.Lookup("foo"));
  ASSERT_NE(opaque_data, nullptr);

  StringUserData* user_data = static_cast<StringUserData*>(opaque_data->data());
  EXPECT_EQ(user_data, &string_data);
}

TEST(ExecutionContextTest, UserData) {
  ExecutionContext context;
  TF_ASSERT_OK(context.Emplace<I32UserData>(42));

  TF_ASSERT_OK_AND_ASSIGN(auto i32_data, context.Lookup<I32UserData>());
  ASSERT_NE(i32_data, nullptr);
  ASSERT_EQ(i32_data->value, 42);
}

TEST(ExecutionContextTest, UserDataNotFound) {
  ExecutionContext context;
  auto i32_data = context.Lookup<I32UserData>();
  ASSERT_EQ(i32_data.status().code(), absl::StatusCode::kNotFound);
}

}  // namespace xla::ffi

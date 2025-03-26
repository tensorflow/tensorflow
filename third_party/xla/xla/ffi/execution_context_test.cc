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

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/ffi/type_id_registry.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::ffi {

struct I32UserData {
  explicit I32UserData(int32_t value) : value(value) {}
  int32_t value;
};

struct StrUserData {
  explicit StrUserData(std::string value) : value(value) {}
  std::string value;
};

TEST(ExecutionContextTest, EmplaceUserData) {
  ExecutionContext context;
  TF_ASSERT_OK(context.Emplace<I32UserData>(42));
  TF_ASSERT_OK(context.Emplace<StrUserData>("hello"));

  TF_ASSERT_OK_AND_ASSIGN(auto* i32_data, context.Lookup<I32UserData>());
  TF_ASSERT_OK_AND_ASSIGN(auto* str_data, context.Lookup<StrUserData>());

  ASSERT_NE(i32_data, nullptr);
  ASSERT_NE(str_data, nullptr);
  ASSERT_EQ(i32_data->value, 42);
  ASSERT_EQ(str_data->value, "hello");
}

TEST(ExecutionContextTest, InsertUserOwned) {
  I32UserData user_data(42);

  ExecutionContext context;
  TF_ASSERT_OK(context.Insert(&user_data));

  TF_ASSERT_OK_AND_ASSIGN(auto* i32_data, context.Lookup<I32UserData>());
  ASSERT_EQ(i32_data, &user_data);
}

TEST(ExecutionContextTest, InsertUserOwnedWithTypeId) {
  TF_ASSERT_OK_AND_ASSIGN(TypeIdRegistry::TypeId type_id,
                          TypeIdRegistry::AssignExternalTypeId("I32UserData"));

  I32UserData user_data(42);

  ExecutionContext context;
  TF_ASSERT_OK(context.Insert(type_id, &user_data));

  TF_ASSERT_OK_AND_ASSIGN(auto* i32_data, context.Lookup(type_id));
  ASSERT_EQ(i32_data, &user_data);
}

TEST(ExecutionContextTest, UserDataNotFound) {
  ExecutionContext context;
  auto i32_data = context.Lookup<I32UserData>();
  ASSERT_EQ(i32_data.status().code(), absl::StatusCode::kNotFound);
}

}  // namespace xla::ffi

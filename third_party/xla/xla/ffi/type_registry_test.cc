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

#include "xla/ffi/type_registry.h"

#include <cstdint>
#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::ffi {
namespace {

using ::testing::HasSubstr;

TEST(TypeRegistryTest, RegisterExternalTypeId) {
  TypeRegistry::TypeInfo type_info = {+[](void* state) {}};

  TF_ASSERT_OK_AND_ASSIGN(auto foo_id,
                          TypeRegistry::AssignExternalTypeId("foo", type_info));
  EXPECT_GE(foo_id.value(), 0);

  auto duplicate_foo_id = TypeRegistry::AssignExternalTypeId("foo", type_info);
  EXPECT_THAT(duplicate_foo_id.status().message(),
              HasSubstr("Type name foo already registered with type id"));

  // It's ok to register the same type with same type id.
  TF_ASSERT_OK(TypeRegistry::RegisterExternalTypeId("foo", foo_id, type_info));

  // It's an error to register the same type with a different type id.
  auto wrong_foo_id = TypeRegistry::RegisterExternalTypeId(
      "foo", TypeRegistry::TypeId(std::numeric_limits<int64_t>::max()),
      type_info);
  EXPECT_THAT(wrong_foo_id.message(),
              HasSubstr("Type name foo already registered with type id"));

  // Registered type has a correct type info.
  TF_ASSERT_OK_AND_ASSIGN(TypeRegistry::TypeInfo foo_info,
                          TypeRegistry::GetExternalTypeInfo(foo_id));
  EXPECT_EQ(foo_info.deleter, type_info.deleter);

  // It's ok to register a new type with a user-provided type id.
  auto bar_id = TypeRegistry::TypeId(std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK(TypeRegistry::RegisterExternalTypeId(
      "bar", TypeRegistry::TypeId(std::numeric_limits<int64_t>::max()),
      type_info));

  // And a new type has a correct type info.
  TF_ASSERT_OK_AND_ASSIGN(TypeRegistry::TypeInfo bar_info,
                          TypeRegistry::GetExternalTypeInfo(bar_id));
  EXPECT_EQ(bar_info.deleter, type_info.deleter);
}

TEST(TypeRegistryTest, RegisterInternalTypeId) {
  auto int32_type_id = TypeRegistry::GetTypeId<int32_t>();
  auto int64_type_id = TypeRegistry::GetTypeId<int64_t>();
  EXPECT_NE(int32_type_id, int64_type_id);
}

TEST(TypeRegistryTest, InternalTypeInfo) {
  int32_t* ptr = new int32_t{42};

  TypeRegistry::TypeInfo type_info = TypeRegistry::GetTypeInfo<int32_t>();
  type_info.deleter(ptr);
}

}  // namespace
}  // namespace xla::ffi

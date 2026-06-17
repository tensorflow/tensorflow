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
#include <memory>
#include <string>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::ffi {

// Define a custom type with `TypeSerDes` specialization to test that TypeInfo
// is properly generated for such types.
struct MyString {
  std::string data;
};

template <>
struct TypeRegistry::SerDes<MyString> : public std::true_type {
  static absl::StatusOr<std::string> Serialize(const MyString& type) {
    return type.data;
  }
  static absl::StatusOr<std::unique_ptr<MyString>> Deserialize(
      absl::string_view data) {
    auto type = std::make_unique<MyString>();
    type->data = std::string(data);
    return type;
  }
};

namespace {
using ::testing::HasSubstr;

TEST(TypeRegistryTest, RegisterTypeId) {
  TypeRegistry::TypeInfo type_info = {+[](void* state) {}};

  TF_ASSERT_OK_AND_ASSIGN(auto foo_id,
                          TypeRegistry::AssignTypeId("foo", type_info));
  EXPECT_GE(foo_id.value(), 0);

  auto duplicate_foo_id = TypeRegistry::AssignTypeId("foo", type_info);
  EXPECT_THAT(duplicate_foo_id.status().message(),
              HasSubstr("Type name foo already registered with type id"));

  // It's ok to register the same type with same type id.
  TF_ASSERT_OK(TypeRegistry::RegisterTypeId("foo", foo_id, type_info));

  // It's an error to register the same type with a different type id.
  auto wrong_foo_id = TypeRegistry::RegisterTypeId(
      "foo", TypeRegistry::TypeId(std::numeric_limits<int64_t>::max()),
      type_info);
  EXPECT_THAT(wrong_foo_id.message(),
              HasSubstr("Type name foo already registered with type id"));

  // Registered type has a correct type info.
  TF_ASSERT_OK_AND_ASSIGN(TypeRegistry::TypeInfo foo_info,
                          TypeRegistry::GetTypeInfo(foo_id));
  EXPECT_EQ(foo_info.deleter, type_info.deleter);

  // It's ok to register a new type with a user-provided type id.
  auto bar_id = TypeRegistry::TypeId(std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK(TypeRegistry::RegisterTypeId(
      "bar", TypeRegistry::TypeId(std::numeric_limits<int64_t>::max()),
      type_info));

  // And a new type has a correct type info.
  TF_ASSERT_OK_AND_ASSIGN(TypeRegistry::TypeInfo bar_info,
                          TypeRegistry::GetTypeInfo(bar_id));
  EXPECT_EQ(bar_info.deleter, type_info.deleter);
}

TEST(TypeRegistryTest, GetOrAssignTypeId) {
  internal::TypeRegistrationMap registry;

  TypeRegistry::TypeInfo type_info0 = {+[](void* state) {}};
  TypeRegistry::TypeInfo type_info1 = {+[](void* state) {}};

  TF_ASSERT_OK_AND_ASSIGN(auto assigned_id, TypeRegistry::GetOrAssignTypeId(
                                                registry, "foo", type_info0));

  TF_ASSERT_OK_AND_ASSIGN(auto got_id, TypeRegistry::GetOrAssignTypeId(
                                           registry, "foo", type_info0));
  EXPECT_EQ(got_id, assigned_id);

  auto err = TypeRegistry::GetOrAssignTypeId(registry, "foo", type_info1);
  EXPECT_THAT(err.status().message(), HasSubstr("different `TypeInfo`"));

  TF_ASSERT_OK_AND_ASSIGN(auto int32_id,
                          TypeRegistry::GetOrAssignTypeId<int32_t>(registry));
  TF_ASSERT_OK_AND_ASSIGN(auto int64_id,
                          TypeRegistry::GetOrAssignTypeId<int64_t>(registry));
  EXPECT_NE(int32_id, int64_id);
}

TEST(TypeRegistryTest, GetTypeId) {
  auto int32_id = TypeRegistry::GetTypeId<int32_t>();
  auto int64_id = TypeRegistry::GetTypeId<int64_t>();
  EXPECT_NE(int32_id, int64_id);

  absl::string_view int32_name = TypeRegistry::GetTypeName<int32_t>();
  absl::string_view int64_name = TypeRegistry::GetTypeName<int64_t>();
  EXPECT_EQ(*TypeRegistry::GetTypeId(int32_name), int32_id);
  EXPECT_EQ(*TypeRegistry::GetTypeId(int64_name), int64_id);
}

TEST(TypeRegistryTest, GetTypeInfo) {
  int32_t* ptr = new int32_t{42};

  TypeRegistry::TypeInfo type_info = TypeRegistry::GetTypeInfo<int32_t>();
  type_info.deleter(ptr);
}

TEST(TypeRegistryTest, SerializableType) {
  MyString str = {"foo"};

  TypeRegistry::TypeInfo type_info = TypeRegistry::GetTypeInfo<MyString>();
  ASSERT_NE(type_info.serializer, nullptr);
  ASSERT_NE(type_info.deserializer, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::string serialized, TypeRegistry::Serialize(str));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MyString> deserialized,
                          TypeRegistry::Deserialize<MyString>(serialized));
  EXPECT_EQ(deserialized->data, "foo");
}

}  // namespace
}  // namespace xla::ffi

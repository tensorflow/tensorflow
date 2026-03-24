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
#include <string>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/ffi/execution_state.pb.h"
#include "xla/ffi/type_registry.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::ffi {

using TypeId = ExecutionState::TypeId;

using ::testing::HasSubstr;

TEST(ExecutionStateTest, SetAndGetForInternalType) {
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

TEST(ExecutionStateTest, SetAndGetForExternalType) {
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

  TypeRegistry::TypeInfo type_info = {
      [](void* ptr) { delete static_cast<int32_t*>(ptr); }};
  TF_ASSERT_OK_AND_ASSIGN(TypeRegistry::TypeId type_id,
                          TypeRegistry::AssignTypeId("int32_t", type_info));

  int32_t* value = new int32_t(42);

  // Once set, state can be retrieved.
  TF_ASSERT_OK(state.Set(type_id, value));
  EXPECT_TRUE(state.IsSet());

  TF_ASSERT_OK_AND_ASSIGN(void* data, state.Get(type_id));
  EXPECT_EQ(data, value);
}

struct MyState {
  std::string value;
};

template <>
struct TypeRegistry::SerDes<MyState> : public std::true_type {
  static absl::StatusOr<std::string> Serialize(const MyState& value) {
    return value.value;
  }
  static absl::StatusOr<std::unique_ptr<MyState>> Deserialize(
      absl::string_view data) {
    auto state = std::make_unique<MyState>();
    state->value = data;
    return state;
  }
};

TEST(ExecutionStateTest, Serialization) {
  TypeRegistry::TypeInfo type_info = TypeRegistry::GetTypeInfo<MyState>();

  TF_ASSERT_OK_AND_ASSIGN(
      TypeRegistry::TypeId type_id,
      TypeRegistry::AssignTypeId("my_state_type", type_info));

  ExecutionState state;
  TF_ASSERT_OK(state.Set(type_id, new MyState{"some_state_data"}));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionStateProto proto, state.ToProto());

  TF_ASSERT_OK_AND_ASSIGN(ExecutionState round_trip,
                          ExecutionState::FromProto(proto))
  TF_ASSERT_OK_AND_ASSIGN(void* round_trip_data, round_trip.Get(type_id));
  EXPECT_EQ(static_cast<MyState*>(round_trip_data)->value, "some_state_data");
}

TEST(ExecutionStateTest, IsSerializable) {
  ExecutionState state;
  // Empty state is serializable (as empty proto).
  EXPECT_TRUE(state.IsSerializable());

  // State without serializer.
  struct NoSerializer {
    int x;
  };
  TF_ASSERT_OK(state.Set(std::make_unique<NoSerializer>(NoSerializer{42})));
  EXPECT_FALSE(state.IsSerializable());

  // State with serializer.
  ExecutionState serializable_state;
  TF_ASSERT_OK(
      serializable_state.Set(std::make_unique<MyState>(MyState{"foo"})));
  EXPECT_TRUE(serializable_state.IsSerializable());
}

}  // namespace xla::ffi

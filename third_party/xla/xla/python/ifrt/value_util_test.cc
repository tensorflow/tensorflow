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

#include "xla/python/ifrt/value_util.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/value.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

class NonArrayValue final : public llvm::RTTIExtends<NonArrayValue, Value> {
 public:
  Client* client() const final { return nullptr; }
  UserContextRef user_context() const final { return UserContextRef(); }
  absl::StatusOr<std::optional<int64_t>> ByteSize() const final {
    return std::nullopt;
  }
  tsl::Future<> GetReadyFuture() const final { return absl::OkStatus(); }
  tsl::Future<> Delete() final { return absl::OkStatus(); }
  bool IsDeleted() const final { return false; }
  std::string DebugString() const final { return ""; }
  static char ID;  // NOLINT
};

[[maybe_unused]] char NonArrayValue::ID = 0;

TEST(ValueUtilTest, ToArraysCopy) {
  ValueRef array0 = tsl::MakeRef<MockArray>();
  ValueRef array1 = tsl::MakeRef<MockArray>();
  std::vector<ValueRef> values{array0, array1};
  std::vector<ArrayRef> arrays = ToArrays(values);
  ASSERT_EQ(arrays.size(), 2);
  EXPECT_EQ(arrays[0].get(), array0.get());
  EXPECT_EQ(arrays[1].get(), array1.get());
}

TEST(ValueUtilTest, ToArraysCopyFailsForNonArray) {
  ValueRef nullptr_array;
  std::vector<ValueRef> nullptr_array_values{nullptr_array};
  EXPECT_DEATH(ToArrays(nullptr_array_values), "Check failed");

  ValueRef non_array = tsl::MakeRef<NonArrayValue>();
  std::vector<ValueRef> non_array_values{non_array};
  EXPECT_DEATH(ToArrays(non_array_values), "Check failed");
}

TEST(ValueUtilTest, ToArraysMove) {
  ValueRef array0 = tsl::MakeRef<MockArray>();
  ValueRef array1 = tsl::MakeRef<MockArray>();
  std::vector<ValueRef> values{array0, array1};
  ValueRef array0_ptr = array0;
  ValueRef array1_ptr = array1;
  std::vector<ArrayRef> arrays = ToArrays(absl::MakeSpan(values));
  ASSERT_EQ(arrays.size(), 2);
  EXPECT_EQ(arrays[0].get(), array0_ptr.get());
  EXPECT_EQ(arrays[1].get(), array1_ptr.get());
  EXPECT_EQ(values[0].get(), nullptr);
  EXPECT_EQ(values[1].get(), nullptr);
}

TEST(ValueUtilTest, ToArraysMoveFailsForNonArray) {
  ValueRef nullptr_array;
  std::vector<ValueRef> nullptr_array_values{nullptr_array};
  EXPECT_DEATH(ToArrays(absl::MakeSpan(nullptr_array_values)), "Check failed");

  ValueRef non_array = tsl::MakeRef<NonArrayValue>();
  std::vector<ValueRef> non_array_values{non_array};
  EXPECT_DEATH(ToArrays(absl::MakeSpan(non_array_values)), "Check failed");
}

TEST(ValueUtilTest, ToValuesCopy) {
  auto array1 = tsl::MakeRef<MockArray>();
  auto array2 = tsl::MakeRef<MockArray>();
  std::vector<ArrayRef> arrays = {array1, array2};
  std::vector<ValueRef> values = ToValues(arrays);
  ASSERT_EQ(values.size(), 2);
  EXPECT_EQ(values[0].get(), array1.get());
  EXPECT_EQ(values[1].get(), array2.get());
}

TEST(ValueUtilTest, ToValuesMove) {
  auto array1 = tsl::MakeRef<MockArray>();
  auto array2 = tsl::MakeRef<MockArray>();
  std::vector<ArrayRef> arrays = {array1, array2};
  auto* ptr1 = array1.get();
  auto* ptr2 = array2.get();
  std::vector<ValueRef> values = ToValues(absl::MakeSpan(arrays));
  ASSERT_EQ(values.size(), 2);
  EXPECT_EQ(values[0].get(), ptr1);
  EXPECT_EQ(values[1].get(), ptr2);
  EXPECT_EQ(arrays[0].get(), nullptr);
  EXPECT_EQ(arrays[1].get(), nullptr);
}

}  // namespace
}  // namespace ifrt
}  // namespace xla

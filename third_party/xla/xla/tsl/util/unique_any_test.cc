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

#include "xla/tsl/util/unique_any.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace tsl {
namespace {

struct LargeCopyableValue {
  std::array<int32_t, 10> payload;
};

struct LargeMoveOnlyValue {
  LargeMoveOnlyValue(std::array<int32_t, 10> p) : payload(p) {}  // NOLINT
  LargeMoveOnlyValue(LargeMoveOnlyValue&&) = default;
  LargeMoveOnlyValue& operator=(LargeMoveOnlyValue&&) = default;

  std::array<int32_t, 10> payload;
};

TEST(UniqueAnyTest, DefaultConstructedIsEmpty) {
  UniqueAny a;
  EXPECT_FALSE(a.has_value());
}

TEST(UniqueAnyTest, SmallCopyableValues) {
  UniqueAny any(42);

  int32_t* i32 = any_cast<int32_t>(&any);
  int64_t* i64 = any_cast<int64_t>(&any);

  ASSERT_TRUE(i32);
  ASSERT_FALSE(i64);
  EXPECT_EQ(*i32, 42);

  int32_t& ref = any_cast<int32_t>(any);
  EXPECT_EQ(ref, 42);

  auto moved = any_cast<int32_t>(std::move(any));
  EXPECT_EQ(moved, 42);
}

TEST(UniqueAnyTest, SmallMoveOnlyValues) {
  UniqueAny any(std::make_unique<int32_t>(42));

  std::unique_ptr<int32_t>* i32 = any_cast<std::unique_ptr<int32_t>>(&any);
  std::unique_ptr<int64_t>* i64 = any_cast<std::unique_ptr<int64_t>>(&any);

  ASSERT_TRUE(i32);
  ASSERT_FALSE(i64);
  EXPECT_EQ(**i32, 42);

  std::unique_ptr<int32_t>& ref = any_cast<std::unique_ptr<int32_t>>(any);
  EXPECT_EQ(*ref, 42);

  auto moved = any_cast<std::unique_ptr<int32_t>>(std::move(any));
  EXPECT_EQ(*moved, 42);
}

TEST(UniqueAnyTest, LargeCopyableValues) {
  LargeCopyableValue value = {{0, 1, 2, 3, 4, 5, 7, 8, 9}};
  UniqueAny any(value);

  LargeCopyableValue* ptr = any_cast<LargeCopyableValue>(&any);
  ASSERT_TRUE(ptr);
  EXPECT_EQ(ptr->payload, value.payload);

  LargeCopyableValue& ref = any_cast<LargeCopyableValue>(any);
  EXPECT_EQ(ref.payload, value.payload);

  auto moved = any_cast<LargeCopyableValue>(std::move(any));
  EXPECT_EQ(moved.payload, value.payload);
}

TEST(UniqueAnyTest, LargeMoveOnlyValues) {
  LargeMoveOnlyValue value = {{0, 1, 2, 3, 4, 5, 7, 8, 9}};
  UniqueAny any(LargeMoveOnlyValue{value.payload});

  LargeMoveOnlyValue* ptr = any_cast<LargeMoveOnlyValue>(&any);
  ASSERT_TRUE(ptr);
  EXPECT_EQ(ptr->payload, value.payload);

  LargeMoveOnlyValue& ref = any_cast<LargeMoveOnlyValue>(any);
  EXPECT_EQ(ref.payload, value.payload);

  auto moved = any_cast<LargeMoveOnlyValue>(std::move(any));
  EXPECT_EQ(moved.payload, value.payload);
}

TEST(UniqueAnyTest, MoveConstruction) {
  UniqueAny any(std::make_unique<int32_t>(42));
  UniqueAny move_constructed(std::move(any));

  EXPECT_FALSE(any.has_value());
  EXPECT_TRUE(move_constructed.has_value());
  EXPECT_EQ(*any_cast<std::unique_ptr<int32_t>>(move_constructed), 42);
}

TEST(UniqueAnyTest, MoveAssignment) {
  UniqueAny any(std::make_unique<int32_t>(42));
  UniqueAny move_assigned;
  move_assigned = std::move(any);

  EXPECT_FALSE(any.has_value());
  EXPECT_TRUE(move_assigned.has_value());
  EXPECT_EQ(*any_cast<std::unique_ptr<int32_t>>(move_assigned), 42);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

static void BM_MakeInt32(benchmark::State& state) {
  for (auto _ : state) {
    UniqueAny any = make_unique_any<int32_t>(42);
    benchmark::DoNotOptimize(any);
  }
}

BENCHMARK(BM_MakeInt32);

}  // namespace
}  // namespace tsl

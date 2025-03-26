// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/core/insert_order_map.h"

#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert::internal {
namespace {

using ::testing::ElementsAre;

using TestMap = InsertOrderMap<int, const char*>;

static constexpr int k1 = 1;
static constexpr int k2 = 2;
static constexpr int k3 = 3;
static constexpr int k4 = 4;
static constexpr const char kV1[] = "1";
static constexpr const char kV2[] = "2";
static constexpr const char kV3[] = "3";
static constexpr const char kV4[] = "4";

TestMap MakeTestMap() {
  TestMap map;
  map.InsertOrAssign(k1, kV1);
  map.InsertOrAssign(k2, kV2);
  map.InsertOrAssign(k3, kV3);
  return map;
}

TEST(InsertOrderMapTest, IterateInInsertOrder) {
  auto map = MakeTestMap();
  ASSERT_EQ(map.Size(), 3);

  std::vector<TestMap::Pair> values(map.Begin(), map.End());
  EXPECT_THAT(values,
              ElementsAre(std::make_pair(k1, kV1), std::make_pair(k2, kV2),
                          std::make_pair(k3, kV3)));
}

TEST(InsertOrderMapTest, IterateInInsertOrderWithUpdate) {
  auto map = MakeTestMap();
  ASSERT_EQ(map.Size(), 3);

  map.InsertOrAssign(k1, kV4);
  std::vector<TestMap::Pair> values(map.Begin(), map.End());
  EXPECT_THAT(values,
              ElementsAre(std::make_pair(k1, kV4), std::make_pair(k2, kV2),
                          std::make_pair(k3, kV3)));
}

TEST(InsertOrderMapTest, FindExisting) {
  auto map = MakeTestMap();
  ASSERT_EQ(map.Size(), 3);

  auto val = map.Find(k1);
  ASSERT_TRUE(val.has_value());
  EXPECT_EQ(val->get().first, k1);
  EXPECT_EQ(val->get().second, kV1);

  EXPECT_TRUE(map.Contains(k1));
}

TEST(InsertOrderMapTest, FindMissing) {
  auto map = MakeTestMap();
  ASSERT_EQ(map.Size(), 3);

  EXPECT_EQ(map.Find(k4), std::nullopt);
  EXPECT_FALSE(map.Contains(k4));
}

TEST(InsertOrderMapTest, Clear) {
  auto map = MakeTestMap();
  ASSERT_EQ(map.Size(), 3);

  map.Clear();
  EXPECT_EQ(map.Size(), 0);
  EXPECT_EQ(map.Begin(), map.End());
}

}  // namespace
}  // namespace litert::internal

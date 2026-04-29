/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/util/sorted_range.h"

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/tsl/platform/test.h"

namespace tsl {
namespace {

TEST(SortedRangeTest, SortedRangeVector) {
  std::vector<int> v = {3, 1, 4, 1, 5, 9};
  std::vector<int> sorted;
  for (int x : SortedRange(v)) {
    sorted.push_back(x);
  }
  EXPECT_EQ(sorted, std::vector<int>({1, 1, 3, 4, 5, 9}));
}

TEST(SortedRangeTest, SortedRangeConstContainer) {
  const std::vector<int> v = {3, 1, 4, 1, 5, 9};
  std::vector<int> sorted;
  for (const int x : SortedRange(v)) {
    sorted.push_back(x);
  }
  EXPECT_EQ(sorted, std::vector<int>({1, 1, 3, 4, 5, 9}));
}

TEST(SortedRangeTest, SortedRangeModifiable) {
  std::vector<int> v = {3, 1, 4};
  for (int& x : SortedRange(v, std::less<int>())) {
    x *= 2;
  }
  EXPECT_EQ(v[0], 6);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 8);
}

TEST(SortedRangeTest, SortedRangeMapByKey) {
  absl::flat_hash_map<std::string, int> m = {{"c", 3}, {"a", 1}, {"b", 2}};
  std::vector<std::string> keys;
  std::vector<int> values;

  for (const auto& [key, value] : KeySortedRange(m)) {
    keys.push_back(key);
    values.push_back(value);
  }

  EXPECT_EQ(keys, std::vector<std::string>({"a", "b", "c"}));
  EXPECT_EQ(values, std::vector<int>({1, 2, 3}));
}

TEST(SortedRangeTest, EmptyContainer) {
  std::vector<int> v;
  std::vector<int> sorted;
  for (int x : SortedRange(v, std::less<int>())) {
    sorted.push_back(x);
  }
  EXPECT_TRUE(sorted.empty());
}

TEST(SortedRangeTest, SingleElement) {
  std::vector<int> v = {42};
  std::vector<int> sorted;
  for (int x : SortedRange(v, std::less<int>())) {
    sorted.push_back(x);
  }
  EXPECT_EQ(sorted, std::vector<int>({42}));
}

TEST(SortedRangeTest, CustomComparator) {
  std::vector<int> v = {3, 1, 4, 1, 5, 9};
  std::vector<int> sorted;
  for (int x : SortedRange(v, std::greater<int>())) {
    sorted.push_back(x);
  }
  EXPECT_EQ(sorted, std::vector<int>({9, 5, 4, 3, 1, 1}));
}

TEST(SortedRangeTest, Array) {
  std::array<int, 6> arr = {3, 1, 4, 1, 5, 9};
  std::vector<int> sorted;
  for (int x : SortedRange(arr, std::less<int>())) {
    sorted.push_back(x);
  }
  EXPECT_EQ(sorted, std::vector<int>({1, 1, 3, 4, 5, 9}));
}

TEST(SortedRangeTest, FlatHashSet) {
  absl::flat_hash_set<int> s = {3, 1, 4, 5, 9};
  std::vector<int> sorted;
  for (int x : SortedRange(s, std::less<int>())) {
    sorted.push_back(x);
  }
  EXPECT_EQ(sorted, std::vector<int>({1, 3, 4, 5, 9}));
}

TEST(SortedRangeTest, BTreeSet) {
  absl::btree_set<int> s = {3, 1, 4, 5, 9};
  std::vector<int> sorted;
  for (int x : SortedRange(s, std::less<int>())) {
    sorted.push_back(x);
  }
  EXPECT_EQ(sorted, std::vector<int>({1, 3, 4, 5, 9}));
}

TEST(SortedRangeTest, FlatHashMapByKey) {
  absl::flat_hash_map<std::string, int> m = {{"c", 3}, {"a", 1}, {"b", 2}};
  std::vector<std::string> keys;
  std::vector<int> values;

  for (const auto& kv : KeySortedRange(m)) {
    keys.push_back(kv.first);
    values.push_back(kv.second);
  }

  EXPECT_EQ(keys, std::vector<std::string>({"a", "b", "c"}));
  EXPECT_EQ(values, std::vector<int>({1, 2, 3}));
}

TEST(SortedRangeTest, BTreeMapByKey) {
  absl::btree_map<std::string, int> m = {{"c", 3}, {"a", 1}, {"b", 2}};
  std::vector<std::string> keys;
  std::vector<int> values;

  for (const auto& kv : KeySortedRange(m)) {
    keys.push_back(kv.first);
    values.push_back(kv.second);
  }

  EXPECT_EQ(keys, std::vector<std::string>({"a", "b", "c"}));
  EXPECT_EQ(values, std::vector<int>({1, 2, 3}));
}

}  // namespace
}  // namespace tsl

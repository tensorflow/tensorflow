/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/graphcycles/ordered_set.h"

#include <array>

#include "tsl/platform/test.h"

namespace xla {
namespace {
TEST(OrderedSetTest, Insert) {
  OrderedSet<int> ordered_set;
  EXPECT_TRUE(ordered_set.Insert(90));
  EXPECT_TRUE(ordered_set.Insert(100));
  EXPECT_TRUE(ordered_set.Insert(80));

  EXPECT_FALSE(ordered_set.Insert(100));

  EXPECT_EQ(ordered_set.Size(), 3);

  EXPECT_TRUE(ordered_set.Contains(90));
  EXPECT_TRUE(ordered_set.Contains(100));
  EXPECT_TRUE(ordered_set.Contains(80));

  EXPECT_FALSE(ordered_set.Contains(40));

  std::array<int, 3> expected_sequence = {90, 100, 80};
  EXPECT_EQ(ordered_set.GetSequence(), expected_sequence);
}

TEST(OrderedSetTest, Erase) {
  OrderedSet<int> ordered_set;
  EXPECT_TRUE(ordered_set.Insert(90));
  EXPECT_TRUE(ordered_set.Insert(100));
  EXPECT_TRUE(ordered_set.Insert(80));

  ordered_set.Erase(100);

  EXPECT_EQ(ordered_set.Size(), 2);

  EXPECT_TRUE(ordered_set.Contains(90));
  EXPECT_FALSE(ordered_set.Contains(100));
  EXPECT_TRUE(ordered_set.Contains(80));

  std::array<int, 2> expected_sequence_0 = {90, 80};
  EXPECT_EQ(ordered_set.GetSequence(), expected_sequence_0);

  ordered_set.Erase(80);

  EXPECT_EQ(ordered_set.Size(), 1);

  EXPECT_TRUE(ordered_set.Contains(90));
  EXPECT_FALSE(ordered_set.Contains(100));
  EXPECT_FALSE(ordered_set.Contains(80));

  std::array<int, 1> expected_sequence_1 = {90};
  EXPECT_EQ(ordered_set.GetSequence(), expected_sequence_1);

  ordered_set.Erase(90);

  EXPECT_EQ(ordered_set.Size(), 0);

  EXPECT_FALSE(ordered_set.Contains(90));
  EXPECT_FALSE(ordered_set.Contains(100));
  EXPECT_FALSE(ordered_set.Contains(80));

  std::array<int, 0> expected_sequence_2 = {};
  EXPECT_EQ(ordered_set.GetSequence(), expected_sequence_2);
}

TEST(OrderedSetTest, Clear) {
  OrderedSet<int> ordered_set;
  EXPECT_TRUE(ordered_set.Insert(90));
  EXPECT_TRUE(ordered_set.Insert(100));
  EXPECT_TRUE(ordered_set.Insert(80));

  ordered_set.Clear();

  EXPECT_EQ(ordered_set.Size(), 0);

  EXPECT_FALSE(ordered_set.Contains(90));
  EXPECT_FALSE(ordered_set.Contains(100));
  EXPECT_FALSE(ordered_set.Contains(80));

  std::array<int, 0> expected_sequence = {};
  EXPECT_EQ(ordered_set.GetSequence(), expected_sequence);
}

TEST(OrderedSetTest, LargeInsertions) {
  const int kSize = 50 * 9000;

  OrderedSet<int> ordered_set;

  for (int i = 0; i < kSize; i++) {
    EXPECT_TRUE(ordered_set.Insert(i + 500));
  }

  for (int i = 0; i < kSize; i++) {
    EXPECT_EQ(ordered_set.GetSequence()[i], i + 500);
  }
}
}  // namespace
}  // namespace xla

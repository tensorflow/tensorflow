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

#include "xla/util/dynamic_bitset.h"

#include <gtest/gtest.h>

namespace xla {
namespace {

TEST(DynamicBitsetTest, BasicOperations) {
  DynamicBitset bitset;
  EXPECT_TRUE(bitset.Empty());
  bitset.Add(5);
  EXPECT_FALSE(bitset.Empty());
  EXPECT_TRUE(bitset.Contains(5));
  EXPECT_FALSE(bitset.Contains(4));
  bitset.Clear(5);
  EXPECT_TRUE(bitset.Empty());
  EXPECT_FALSE(bitset.Contains(5));
}

TEST(DynamicBitsetTest, Resizing) {
  DynamicBitset bitset;
  bitset.Add(1000);  // Beyond inline capacity
  EXPECT_TRUE(bitset.Contains(1000));
  EXPECT_FALSE(bitset.Contains(999));
}

TEST(DynamicBitsetTest, Equality) {
  DynamicBitset lhs;
  DynamicBitset rhs;
  EXPECT_EQ(lhs, rhs);
  lhs.Add(5);
  EXPECT_NE(lhs, rhs);
  rhs.Add(5);
  EXPECT_EQ(lhs, rhs);
  lhs.Add(100);
  EXPECT_NE(lhs, rhs);
}

TEST(DynamicBitsetTest, EqualityWithTrailingZeros) {
  DynamicBitset lhs;
  DynamicBitset rhs;
  lhs.Add(5);
  rhs.Add(5);
  EXPECT_EQ(lhs, rhs);

  lhs.Add(100);
  EXPECT_NE(lhs, rhs);

  lhs.Clear(100);  // lhs now has larger capacity but trailing word is 0
  EXPECT_EQ(lhs, rhs);
}

TEST(DynamicBitsetTest, Merge) {
  DynamicBitset lhs;
  DynamicBitset rhs;
  lhs.Add(5);
  rhs.Add(10);
  rhs.Add(100);

  lhs.Merge(rhs);
  EXPECT_TRUE(lhs.Contains(5));
  EXPECT_TRUE(lhs.Contains(10));
  EXPECT_TRUE(lhs.Contains(100));
  EXPECT_FALSE(lhs.Contains(6));
}

TEST(DynamicBitsetTest, MergeAsymmetric) {
  DynamicBitset lhs;
  DynamicBitset rhs;
  lhs.Add(100);  // lhs has larger capacity
  rhs.Add(5);

  lhs.Merge(rhs);
  EXPECT_TRUE(lhs.Contains(5));
  EXPECT_TRUE(lhs.Contains(100));
}

}  // namespace
}  // namespace xla

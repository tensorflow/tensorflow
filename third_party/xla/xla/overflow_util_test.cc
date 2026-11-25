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

#include "xla/overflow_util.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include <gtest/gtest.h>

namespace xla {
namespace {

TEST(OverflowUtilTest, OverflowSafeMultiplyNormalAndBoundary) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  EXPECT_EQ(OverflowSafeMultiply(0, 0), std::make_pair(int64_t{0}, false));
  EXPECT_EQ(OverflowSafeMultiply(0, kMax), std::make_pair(int64_t{0}, false));
  EXPECT_EQ(OverflowSafeMultiply(kMax, 0), std::make_pair(int64_t{0}, false));
  EXPECT_EQ(OverflowSafeMultiply(6, 7), std::make_pair(int64_t{42}, false));
  EXPECT_EQ(OverflowSafeMultiply(kMax, 1), std::make_pair(kMax, false));
  EXPECT_EQ(OverflowSafeMultiply(1, kMax), std::make_pair(kMax, false));
  EXPECT_EQ(OverflowSafeMultiply(kMax / 2, 2),
            std::make_pair((kMax / 2) * 2, false));
  EXPECT_EQ(OverflowSafeMultiply(int64_t{1} << 31, int64_t{1} << 31),
            std::make_pair(int64_t{1} << 62, false));
}

TEST(OverflowUtilTest, OverflowSafeMultiplyOverflowAndNegativeInputs) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  EXPECT_TRUE(OverflowSafeMultiply(kMax, 2).second);
  EXPECT_TRUE(OverflowSafeMultiply(2, kMax).second);
  EXPECT_TRUE(OverflowSafeMultiply(kMax, kMax).second);
  EXPECT_TRUE(OverflowSafeMultiply(int64_t{1} << 32, int64_t{1} << 32).second);
  EXPECT_TRUE(OverflowSafeMultiply((kMax / 2) + 1, 2).second);

  EXPECT_TRUE(OverflowSafeMultiply(-1, 0).second);
  EXPECT_TRUE(OverflowSafeMultiply(0, -1).second);
  EXPECT_TRUE(OverflowSafeMultiply(-1, 5).second);
  EXPECT_TRUE(OverflowSafeMultiply(5, -1).second);
  EXPECT_TRUE(OverflowSafeMultiply(-1, -1).second);
  EXPECT_TRUE(OverflowSafeMultiply(kMin, 1).second);
}

TEST(OverflowUtilTest, OverflowSafeAddInt64) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  EXPECT_EQ(OverflowSafeAdd<int64_t>(0, 0), 0);
  EXPECT_EQ(OverflowSafeAdd<int64_t>(10, 25), 35);
  EXPECT_EQ(OverflowSafeAdd<int64_t>(-10, 25), 15);
  EXPECT_EQ(OverflowSafeAdd<int64_t>(kMax, 0), kMax);
  EXPECT_EQ(OverflowSafeAdd<int64_t>(kMin, 0), kMin);
  EXPECT_EQ(OverflowSafeAdd<int64_t>(kMax, -1), kMax - 1);
  EXPECT_EQ(OverflowSafeAdd<int64_t>(kMin, 1), kMin + 1);
  EXPECT_EQ(OverflowSafeAdd<int64_t>(kMax, kMin), -1);

  EXPECT_EQ(OverflowSafeAdd<int64_t>(kMax, 1), std::nullopt);
  EXPECT_EQ(OverflowSafeAdd<int64_t>(1, kMax), std::nullopt);
  EXPECT_EQ(OverflowSafeAdd<int64_t>(kMax, kMax), std::nullopt);
  EXPECT_EQ(OverflowSafeAdd<int64_t>(kMin, -1), std::nullopt);
  EXPECT_EQ(OverflowSafeAdd<int64_t>(-1, kMin), std::nullopt);
  EXPECT_EQ(OverflowSafeAdd<int64_t>(kMin, kMin), std::nullopt);
}

TEST(OverflowUtilTest, OverflowSafeAddInt32) {
  constexpr int32_t kMax = std::numeric_limits<int32_t>::max();
  constexpr int32_t kMin = std::numeric_limits<int32_t>::min();
  EXPECT_EQ(OverflowSafeAdd<int32_t>(kMax - 10, 10), kMax);
  EXPECT_EQ(OverflowSafeAdd<int32_t>(kMax, 1), std::nullopt);
  EXPECT_EQ(OverflowSafeAdd<int32_t>(kMin, -1), std::nullopt);
}

TEST(OverflowUtilTest, OverflowSafeRoundUpToValidAndBoundary) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(0, 1), 0);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(0, 256), 0);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(7, 1), 7);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(1, 256), 256);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(13, 8), 16);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(256, 256), 256);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(257, 256), 512);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(kMax, 1), kMax);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(kMax, kMax), kMax);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(1, kMax), kMax);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(kMax - 1, 2), kMax - 1);

  constexpr int32_t kMax32 = std::numeric_limits<int32_t>::max();
  EXPECT_EQ(OverflowSafeRoundUpTo<int32_t>(kMax32 - 1, 2), kMax32 - 1);
  EXPECT_EQ(OverflowSafeRoundUpTo<int32_t>(kMax32, 2), std::nullopt);
}

TEST(OverflowUtilTest, OverflowSafeRoundUpToOverflowAndInvalidInputs) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(kMax, 2), std::nullopt);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(kMax - 1, 256), std::nullopt);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(kMax - 127, 256), std::nullopt);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(kMax - 254, 256), std::nullopt);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(kMax - 255, 256), kMax - 255);

  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(-1, 256), std::nullopt);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(kMin, 256), std::nullopt);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(0, 0), std::nullopt);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(16, 0), std::nullopt);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(16, -8), std::nullopt);
  EXPECT_EQ(OverflowSafeRoundUpTo<int64_t>(0, kMin), std::nullopt);
}

}  // namespace
}  // namespace xla

/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/size_util.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

namespace ruy {
namespace {

template <typename Integer>
void SizeUtilTestValue(Integer value) {
  if (value == 0) {
    return;
  }

  EXPECT_LE(0, floor_log2(value));
  EXPECT_LE(floor_log2(value), ceil_log2(value));
  EXPECT_LE(ceil_log2(value), 8 * sizeof(Integer));

  if (is_pot(value)) {
    EXPECT_EQ(floor_log2(value), ceil_log2(value));
    EXPECT_EQ(floor_log2(value), pot_log2(value));
  } else {
    EXPECT_EQ(floor_log2(value) + 1, ceil_log2(value));
  }
  EXPECT_EQ(value >> floor_log2(value), 1);
  EXPECT_EQ(round_down_pot(value), static_cast<Integer>(1)
                                       << floor_log2(value));
  EXPECT_LE(round_down_pot(value), value);
  EXPECT_GE(round_down_pot(value), value >> 1);
  EXPECT_TRUE(is_pot(round_down_pot(value)));

  if (ceil_log2(value) < 8 * sizeof(Integer) - 1) {
    EXPECT_EQ(value >> ceil_log2(value), is_pot(value) ? 1 : 0);
    EXPECT_EQ(round_up_pot(value), static_cast<Integer>(1) << ceil_log2(value));
    EXPECT_GE(round_up_pot(value), value);
    EXPECT_LE(round_up_pot(value) >> 1, value);
    EXPECT_TRUE(is_pot(round_up_pot(value)));
  }

  for (std::uint8_t modulo : {1, 2, 8, 32, 128}) {
    EXPECT_GE(value, round_down_pot(value, modulo));
    EXPECT_EQ(round_down_pot(value, modulo) % modulo, 0);

    if (value <= std::numeric_limits<Integer>::max() - modulo) {
      EXPECT_LE(value, round_up_pot(value, modulo));
      EXPECT_EQ(round_up_pot(value, modulo) % modulo, 0);
    }
  }
}

template <typename Integer>
void SizeUtilTest() {
  for (int exponent = 0; exponent < 8 * sizeof(Integer) - 1; exponent++) {
    const Integer pot = static_cast<Integer>(1) << exponent;
    SizeUtilTestValue(pot - 1);
    SizeUtilTestValue(pot);
    SizeUtilTestValue(pot + 1);
    SizeUtilTestValue(pot + 12);
    SizeUtilTestValue(pot + 123);
  }
  SizeUtilTestValue(std::numeric_limits<Integer>::max() - 1);
  SizeUtilTestValue(std::numeric_limits<Integer>::max());
}

TEST(SizeUtilTest, Int) { SizeUtilTest<int>(); }

TEST(SizeUtilTest, Long) { SizeUtilTest<long int>(); }  // NOLINT

TEST(SizeUtilTest, LongLong) { SizeUtilTest<long long int>(); }  // NOLINT

TEST(SizeUtilTest, Int32) { SizeUtilTest<std::int32_t>(); }

TEST(SizeUtilTest, Int64) { SizeUtilTest<std::int64_t>(); }

TEST(SizeUtilTest, Ptrdiff) { SizeUtilTest<std::ptrdiff_t>(); }

}  // namespace
}  // namespace ruy

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

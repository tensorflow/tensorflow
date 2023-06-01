/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/overflow.h"

#include <cmath>

#ifdef PLATFORM_WINDOWS
#include <Windows.h>
#endif

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

bool HasMultiplyOverflow(int64_t x, int64_t y) {
#ifdef PLATFORM_WINDOWS
  // `long double` on MSVC is 64 bits not 80 bits - use a windows specific API
  // for this test.
  return ::MultiplyHigh(x, y) != 0;
#else
  long double dxy = static_cast<long double>(x) * static_cast<long double>(y);
  return dxy > std::numeric_limits<int64_t>::max();
#endif
}
bool HasAddOverflow(int64_t x, int64_t y) {
  int64_t carry_from_lower_bits = ((x & 0xffffffff) + (y & 0xffffffff)) >> 32;
  // If upper bits plus carry would roll into the sign bit, we have overflowed.
  if ((x >> 32) + (y >> 32) + carry_from_lower_bits >=
      (static_cast<int64_t>(1) << 31)) {
    return true;
  }
  return false;
}

TEST(OverflowTest, Nonnegative) {
  // Various interesting values
  std::vector<int64_t> interesting = {
      0,
      std::numeric_limits<int64_t>::max(),
  };

  for (int i = 0; i < 63; i++) {
    int64_t bit = static_cast<int64_t>(1) << i;
    interesting.push_back(bit);
    interesting.push_back(bit + 1);
    interesting.push_back(bit - 1);
  }

  for (const int64_t mid : {static_cast<int64_t>(1) << 32,
                            static_cast<int64_t>(std::pow(2, 63.0 / 2))}) {
    for (int i = -5; i < 5; i++) {
      interesting.push_back(mid + i);
    }
  }

  // Check all pairs
  for (int64_t x : interesting) {
    for (int64_t y : interesting) {
      int64_t xmy = MultiplyWithoutOverflow(x, y);
      if (HasMultiplyOverflow(x, y)) {
        EXPECT_LT(xmy, 0) << x << " " << y;
      } else {
        EXPECT_EQ(x * y, xmy) << x << " " << y;
      }
      int64_t xpy = AddWithoutOverflow(x, y);
      if (HasAddOverflow(x, y)) {
        EXPECT_LT(xpy, 0) << x << " " << y;
      } else {
        EXPECT_EQ(x + y, xpy) << x << " " << y;
      }
    }
  }
}

TEST(OverflowTest, Negative) {
  const int64_t negatives[] = {-1, std::numeric_limits<int64_t>::min()};
  for (const int64_t n : negatives) {
    EXPECT_LT(MultiplyWithoutOverflow(n, 0), 0) << n;
    EXPECT_LT(MultiplyWithoutOverflow(0, n), 0) << n;
    EXPECT_LT(MultiplyWithoutOverflow(n, n), 0) << n;
    EXPECT_LT(AddWithoutOverflow(n, 0), 0) << n;
    EXPECT_LT(AddWithoutOverflow(0, n), 0) << n;
    EXPECT_LT(AddWithoutOverflow(n, n), 0) << n;
  }
}

}  // namespace
}  // namespace tensorflow

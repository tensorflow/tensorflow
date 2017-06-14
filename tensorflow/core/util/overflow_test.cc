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
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(OverflowTest, Nonnegative) {
  // Various interesting values
  std::vector<int64> interesting = {0, std::numeric_limits<int64>::max()};
  for (int i = 0; i < 63; i++) {
    int64 bit = static_cast<int64>(1) << i;
    interesting.push_back(bit);
    interesting.push_back(bit + 1);
    interesting.push_back(bit - 1);
  }
  for (const int64 mid : {static_cast<int64>(1) << 32,
                          static_cast<int64>(std::pow(2, 63.0 / 2))}) {
    for (int i = -5; i < 5; i++) {
      interesting.push_back(mid + i);
    }
  }

  // Check all pairs
  for (auto x : interesting) {
    for (auto y : interesting) {
      int64 xy = MultiplyWithoutOverflow(x, y);
      long double dxy = static_cast<long double>(x) * y;
      if (dxy > std::numeric_limits<int64>::max()) {
        EXPECT_LT(xy, 0);
      } else {
        EXPECT_EQ(dxy, xy);
      }
    }
  }
}

TEST(OverflowTest, Negative) {
  const int64 negatives[] = {-1, std::numeric_limits<int64>::min()};
  for (const int64 n : negatives) {
    EXPECT_DEATH(MultiplyWithoutOverflow(n, 0), "");
    EXPECT_DEATH(MultiplyWithoutOverflow(0, n), "");
    EXPECT_DEATH(MultiplyWithoutOverflow(n, n), "");
  }
}

}  // namespace
}  // namespace tensorflow

/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/analysis/interval.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace xla {
namespace {

TEST(IntervalComparisonTest, PointComparisons) {
  Interval interval{12, 64};
  auto point = [](int64_t n) { return Interval{n, n}; };
  EXPECT_EQ(interval.Gt(point(11)), true);
  EXPECT_EQ(interval.Gt(point(12)), std::nullopt);
  EXPECT_EQ(interval.Gt(point(65)), false);

  EXPECT_EQ(interval.Lt(point(65)), true);
  EXPECT_EQ(interval.Lt(point(64)), std::nullopt);
  EXPECT_EQ(interval.Lt(point(10)), false);

  EXPECT_EQ(interval.Eq(point(11)), false);
  EXPECT_EQ(interval.Eq(point(12)), std::nullopt);
  EXPECT_EQ(interval.Eq(point(15)), std::nullopt);
  EXPECT_EQ(interval.Eq(point(65)), false);

  EXPECT_EQ(interval.Ne(point(11)), true);
  EXPECT_EQ(interval.Ne(point(15)), std::nullopt);
  EXPECT_EQ(interval.Ne(point(65)), true);

  EXPECT_EQ(interval.Ge(point(12)), true);
  EXPECT_EQ(interval.Ge(point(64)), std::nullopt);
  EXPECT_EQ(interval.Ge(point(65)), false);

  EXPECT_EQ(interval.Le(point(11)), false);
  EXPECT_EQ(interval.Le(point(64)), true);
  EXPECT_EQ(interval.Le(point(63)), std::nullopt);
  EXPECT_EQ(interval.Le(point(65)), true);

  EXPECT_EQ(point(15).Eq(point(15)), true);
  EXPECT_EQ(point(15).Eq(point(16)), false);

  EXPECT_EQ(point(15).Ne(point(15)), false);
  EXPECT_EQ(point(15).Ne(point(16)), true);
}

TEST(IntervalComparisonTest, RangeComparisons) {
  Interval interval{12, 64};
  auto range = [](int64_t l, int64_t u) { return Interval{l, u}; };
  EXPECT_EQ(interval.Gt(range(-10, 11)), true);
  EXPECT_EQ(interval.Gt(range(-10, 12)), std::nullopt);
  EXPECT_EQ(interval.Gt(interval), std::nullopt);
  EXPECT_EQ(interval.Gt(range(10, 20)), std::nullopt);
  EXPECT_EQ(interval.Gt(range(50, 60)), std::nullopt);
  EXPECT_EQ(interval.Gt(range(64, 100)), false);
  EXPECT_EQ(interval.Gt(range(65, 100)), false);

  EXPECT_EQ(interval.Lt(range(65, 100)), true);
  EXPECT_EQ(interval.Lt(range(64, 100)), std::nullopt);
  EXPECT_EQ(interval.Lt(interval), std::nullopt);
  EXPECT_EQ(interval.Lt(range(50, 60)), std::nullopt);
  EXPECT_EQ(interval.Lt(range(10, 20)), std::nullopt);
  EXPECT_EQ(interval.Lt(range(-10, 12)), false);
  EXPECT_EQ(interval.Lt(range(-10, 11)), false);

  EXPECT_EQ(interval.Eq(interval), std::nullopt);
  EXPECT_EQ(interval.Eq(range(65, 100)), false);
  EXPECT_EQ(interval.Eq(range(0, 11)), false);
}

MATCHER_P(IntervalIs, interval, "") {
  std::pair<int64_t, int64_t> arg_pair{arg.lower, arg.upper};
  return ::testing::ExplainMatchResult(
      ::testing::Pair(interval.lower, interval.upper), arg_pair,
      result_listener);
}

TEST(IntervalMathTest, Addition) {
  Interval a{12, 64};
  Interval b{-100, 120};
  Interval sum{12 - 100, 64 + 120};
  EXPECT_THAT(a + b, IntervalIs(sum));
}

TEST(IntervalMathTest, AdditionSaturating) {
  Interval a{12, 64};
  Interval b{-100, 120};
  Interval c{100, std::numeric_limits<int64_t>::max() - 80};
  Interval any{std::numeric_limits<int64_t>::min(),
               std::numeric_limits<int64_t>::max()};
  Interval positive{0, std::numeric_limits<int64_t>::max()};
  Interval negative{std::numeric_limits<int64_t>::min(), 0};
  auto range = [](int64_t l, int64_t u) { return Interval{l, u}; };

  EXPECT_THAT(positive + negative, IntervalIs(any));
  EXPECT_THAT(any + any, IntervalIs(any));
  EXPECT_THAT(b + any, IntervalIs(any));

  EXPECT_THAT(c + any, IntervalIs(any));
  EXPECT_THAT(c + positive,
              IntervalIs(range(100, std::numeric_limits<int64_t>::max())));
  Interval c_plus_negative{negative.lower, c.upper};
  EXPECT_THAT(c + negative, IntervalIs(c_plus_negative));

  Interval a_plus_c{112, std::numeric_limits<int64_t>::max() - 16};
  EXPECT_THAT(a + c, IntervalIs(a_plus_c));
  Interval b_plus_c{0, std::numeric_limits<int64_t>::max()};
  EXPECT_THAT(b + c, IntervalIs(b_plus_c));
}

TEST(IntervalMathTest, Multiplication) {
  Interval pos{10, 100};
  Interval neg{-10, -1};
  Interval both_small{-5, 6};
  Interval both_large{-20, 1000};

  auto range = [](int64_t l, int64_t u) { return Interval{l, u}; };
  EXPECT_THAT(pos * neg, IntervalIs(range(-1000, -10)));
  EXPECT_THAT(pos * both_small, IntervalIs(range(-500, 600)));
  EXPECT_THAT(pos * both_large, IntervalIs(range(-2000, 100000)));
  EXPECT_THAT(neg * both_small, IntervalIs(range(-60, 50)));
  EXPECT_THAT(neg * both_large, IntervalIs(range(-10000, 200)));
  EXPECT_THAT(both_small * both_large, IntervalIs(range(-5000, 6000)));
}

TEST(IntervalMathTest, MultiplicationSaturating) {
  Interval any{std::numeric_limits<int64_t>::min(),
               std::numeric_limits<int64_t>::max()};
  Interval bit33{42, std::numeric_limits<uint32_t>::max()};
  Interval bit33_sq{42 * 42, std::numeric_limits<int64_t>::max()};
  EXPECT_THAT(bit33 * bit33, IntervalIs(bit33_sq));
  EXPECT_THAT(any * any, IntervalIs(any));

  Interval greater_41{42, std::numeric_limits<int64_t>::max()};
  Interval neg_one{-1, -1};
  Interval less_neg_41{std::numeric_limits<int64_t>::min(), -42};
  EXPECT_THAT(greater_41 * neg_one, IntervalIs(less_neg_41));
  EXPECT_THAT(less_neg_41 * neg_one, IntervalIs(greater_41));
  EXPECT_THAT(any * neg_one, IntervalIs(any));
}

}  // namespace
}  // namespace xla

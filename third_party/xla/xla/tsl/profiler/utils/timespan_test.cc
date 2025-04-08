/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/profiler/utils/timespan.h"

#include "xla/tsl/platform/test.h"

namespace tsl {
namespace profiler {

TEST(TimespanTests, NonInstantSpanIncludesSingleTimeTests) {
  EXPECT_TRUE(Timespan(10, 2).Includes(12));
  EXPECT_TRUE(Timespan(12, 1).Includes(12));
}

TEST(TimespanTests, NonInstantSpanIncludesInstantSpanTests) {
  EXPECT_TRUE(Timespan(10, 2).Includes(Timespan(10, 0)));
  EXPECT_TRUE(Timespan(10, 2).Includes(Timespan(12, 0)));
}

TEST(TimespanTests, NonInstantSpanIncludesNonInstantSpanTests) {
  EXPECT_TRUE(Timespan(10, 5).Includes(Timespan(10, 4)));
  EXPECT_TRUE(Timespan(10, 5).Includes(Timespan(10, 5)));
  EXPECT_FALSE(Timespan(10, 5).Includes(Timespan(10, 6)));
}

TEST(TimespanTests, InstantSpanIncludesSingleTimeTests) {
  EXPECT_TRUE(Timespan(10, 0).Includes(10));
  EXPECT_FALSE(Timespan(10, 0).Includes(9));
}

TEST(TimespanTests, InstantSpanIncludesInstantSpanTests) {
  EXPECT_TRUE(Timespan(10, 0).Includes(Timespan(10, 0)));
  EXPECT_FALSE(Timespan(10, 0).Includes(Timespan(8, 0)));
}

TEST(TimespanTests, InstantSpanIncludesNonInstantSpanTests) {
  EXPECT_FALSE(Timespan(10, 0).Includes(Timespan(10, 1)));
  EXPECT_FALSE(Timespan(12, 0).Includes(Timespan(9, 100)));
}

TEST(TimespanTests, NonInstantSpanInstantSpanOverlappedDuration) {
  EXPECT_EQ(0, Timespan(12, 2).OverlappedDurationPs(Timespan(8, 0)));
  EXPECT_EQ(0, Timespan(12, 2).OverlappedDurationPs(Timespan(13, 0)));
  EXPECT_EQ(0, Timespan(12, 2).OverlappedDurationPs(Timespan(14, 0)));
}

TEST(TimespanTests, NonInstantSpanNonInstantSpanOverlappedDuration) {
  EXPECT_EQ(0, Timespan(12, 2).OverlappedDurationPs(Timespan(9, 3)));
  EXPECT_EQ(1, Timespan(12, 2).OverlappedDurationPs(Timespan(9, 4)));
  EXPECT_EQ(2, Timespan(12, 2).OverlappedDurationPs(Timespan(9, 5)));
  EXPECT_EQ(2, Timespan(12, 2).OverlappedDurationPs(Timespan(9, 6)));
  EXPECT_EQ(1, Timespan(12, 2).OverlappedDurationPs(Timespan(13, 1)));
  EXPECT_EQ(1, Timespan(12, 2).OverlappedDurationPs(Timespan(13, 2)));
  EXPECT_EQ(0, Timespan(12, 2).OverlappedDurationPs(Timespan(14, 1)));
  EXPECT_EQ(0, Timespan(12, 2).OverlappedDurationPs(Timespan(14, 2)));
  EXPECT_EQ(2, Timespan(12, 5).OverlappedDurationPs(Timespan(13, 2)));
  EXPECT_EQ(2, Timespan(12, 2).OverlappedDurationPs(Timespan(12, 2)));
}

TEST(TimespanTests, InstantSpanInstantSpanOverlappedDuration) {
  EXPECT_EQ(0, Timespan(12, 0).OverlappedDurationPs(Timespan(9, 0)));
  EXPECT_EQ(0, Timespan(12, 0).OverlappedDurationPs(Timespan(12, 0)));
}

TEST(TimespanTests, InstantSpanNonInstantSpanOverlappedDuration) {
  EXPECT_EQ(0, Timespan(12, 0).OverlappedDurationPs(Timespan(8, 3)));
  EXPECT_EQ(0, Timespan(12, 0).OverlappedDurationPs(Timespan(8, 16)));
}

TEST(TimespanTests, Operators) {
  EXPECT_LT(Timespan(11, 0), Timespan(12, 0));

  // Instants nest within larger timespans
  EXPECT_LT(Timespan(12, 1), Timespan(12, 0));
  EXPECT_FALSE(Timespan(12, 0) < Timespan(12, 1));

  EXPECT_FALSE(Timespan(12, 0) < Timespan(11, 0));

  // Instants with same beginning are considered equivalent
  EXPECT_FALSE(Timespan(12, 0) < Timespan(12, 0));

  EXPECT_FALSE(Timespan(12, 0) == Timespan(12, 1));
  EXPECT_FALSE(Timespan(12, 0) == Timespan(11, 0));

  EXPECT_EQ(Timespan(12, 0), Timespan(12, 0));

  EXPECT_LE(Timespan(12, 0), Timespan(12, 0));
  EXPECT_LE(Timespan(12, 0), Timespan(13, 0));
  EXPECT_LE(Timespan(11, 0), Timespan(12, 0));

  EXPECT_FALSE(Timespan(12, 0) <= Timespan(11, 0));
}

TEST(TimespanTests, ExpandToIncludeWithEmptyDestination) {
  Timespan empty1;
  Timespan nonempty1(0, 10);
  empty1.ExpandToInclude(nonempty1);
  EXPECT_EQ(empty1, nonempty1);

  Timespan empty2;
  Timespan nonempty2(5, 3);
  empty2.ExpandToInclude(nonempty2);
  EXPECT_EQ(empty2, nonempty2);
}

TEST(TimespanTests, ExpandToIncludeWithNonEmptyDestination) {
  Timespan ts1 = Timespan::FromEndPoints(0, 4);
  Timespan ts2 = Timespan::FromEndPoints(5, 8);
  ts1.ExpandToInclude(ts2);
  EXPECT_EQ(ts1, Timespan(0, 8));
}

}  // namespace profiler
}  // namespace tsl

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

#include "xla/pjrt/gpu/allocator_config.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(GpuAllocatorConfigTest, DefaultHasFixedCap) {
  GpuAllocatorConfig config;
  EXPECT_EQ(config.memory_fraction, 0.75);
  EXPECT_FALSE(config.memory_fraction_policy.has_value());
  EXPECT_THAT(config.GetMemoryFraction(), FixedMemFraction{0.75});
}

TEST(GpuAllocatorConfigTest, NumericFractionHasFixedCap) {
  GpuAllocatorConfig config;
  for (double fraction : {0.5, 0.9, 1.5}) {
    config.memory_fraction = fraction;
    EXPECT_THAT(config.GetMemoryFraction(), FixedMemFraction{fraction});
  }
}

TEST(GpuAllocatorConfigTest, ExplicitPolicyOverridesNumericFraction) {
  GpuAllocatorConfig config;
  config.memory_fraction = 0.25;
  config.memory_fraction_policy = FlexMemFraction{0.5, 0.9};
  EXPECT_THAT(config.GetMemoryFraction(), (FlexMemFraction{0.5, 0.9}));
  config.memory_fraction_policy = FixedMemFraction{0.6};
  EXPECT_THAT(config.GetMemoryFraction(), FixedMemFraction{0.6});
  EXPECT_EQ(config.memory_fraction, 0.25);
  config.memory_fraction_policy.reset();
  EXPECT_THAT(config.GetMemoryFraction(), FixedMemFraction{0.25});
}

TEST(MemFractionTest, BareFractionHasFixedCap) {
  ASSERT_OK_AND_ASSIGN(MemFraction fraction, ParseMemFraction("0.75"));
  EXPECT_THAT(fraction, FixedMemFraction{0.75});
  EXPECT_THAT(ParseMemFraction(" 0.5 "), IsOkAndHolds(FixedMemFraction{0.5}));
}

TEST(MemFractionTest, PlusSuffixGrowsToAllDeviceMemory) {
  EXPECT_THAT(ParseMemFraction("0.75+"),
              IsOkAndHolds(FlexMemFraction{0.75, 1.0}));
  EXPECT_THAT(ParseMemFraction(" 0.5+ "),
              IsOkAndHolds(FlexMemFraction{0.5, 1.0}));
  EXPECT_THAT(ParseMemFraction("0.75-1.0"),
              IsOkAndHolds(FlexMemFraction{0.75, 1.0}));
  // The shorthand has the same semantics as an explicit range ending at 1.
  EXPECT_THAT(ParseMemFraction("1+"), IsOkAndHolds(FixedMemFraction{1.0}));
  EXPECT_THAT(ParseMemFraction("1.5+"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("cap must be at least the start")));
}

TEST(MemFractionTest, RangeSetsStartAndCap) {
  EXPECT_THAT(ParseMemFraction("0.75-0.85"),
              IsOkAndHolds(FlexMemFraction{0.75, 0.85}));
  EXPECT_THAT(ParseMemFraction("0.5-1.0"),
              IsOkAndHolds(FlexMemFraction{0.5, 1.0}));
}

TEST(MemFractionTest, EqualEndsAreFixed) {
  EXPECT_THAT(ParseMemFraction("0.75-0.75"),
              IsOkAndHolds(FixedMemFraction{0.75}));
  // Oversubscription (unified memory) is only possible without growth.
  EXPECT_THAT(ParseMemFraction("1.5-1.5"), IsOkAndHolds(FixedMemFraction{1.5}));
}

TEST(MemFractionTest, BareFractionAtOrAboveOneIsFixed) {
  EXPECT_THAT(ParseMemFraction("1"), IsOkAndHolds(FixedMemFraction{1.0}));
  EXPECT_THAT(ParseMemFraction("1.5"), IsOkAndHolds(FixedMemFraction{1.5}));
}

TEST(MemFractionTest, NumericFractionHasFixedCap) {
  EXPECT_THAT(MemFractionFromFraction(0.75), FixedMemFraction{0.75});
  EXPECT_THAT(MemFractionFromFraction(0.9), FixedMemFraction{0.9});
  EXPECT_THAT(MemFractionFromFraction(1.0), FixedMemFraction{1.0});
  EXPECT_THAT(MemFractionFromFraction(1.5), FixedMemFraction{1.5});
}

TEST(MemFractionTest, ScientificNotation) {
  EXPECT_THAT(ParseMemFraction("1e-3"), IsOkAndHolds(FixedMemFraction{0.001}));
  EXPECT_THAT(ParseMemFraction("1e-3+"),
              IsOkAndHolds(FlexMemFraction{0.001, 1.0}));
  EXPECT_THAT(ParseMemFraction("1e-3-2e-3"),
              IsOkAndHolds(FlexMemFraction{0.001, 0.002}));
  EXPECT_THAT(ParseMemFraction("1E-3-1E+0"),
              IsOkAndHolds(FlexMemFraction{0.001, 1.0}));
}

TEST(MemFractionTest, RejectsMalformedInput) {
  for (const char* spec :
       {"", "abc", "0", "-0.5", "nan", "inf", "0.5-", "0.5-abc", "0.5-0.7-0.9",
        "+", "0+", "-0.5+", "nan+", "inf+", "0.5++", "0.5+-1", "0.5-0.9+"}) {
    EXPECT_THAT(
        ParseMemFraction(spec),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("START-CAP")))
        << spec;
  }
  EXPECT_THAT(ParseMemFraction("0.9-0.5"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("cap must be at least the start")));
  EXPECT_THAT(ParseMemFraction("0.5-1.5"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("cannot exceed 1")));
}

TEST(MemFractionTest, StartIsThePreallocatedFraction) {
  EXPECT_EQ(MemFractionStart(FlexMemFraction{0.6, 0.9}), 0.6);
  EXPECT_EQ(MemFractionStart(FixedMemFraction{0.3}), 0.3);
}

TEST(MemFractionTest, ToStringRoundTrips) {
  for (const char* spec :
       {"0.75", "0.75+", "0.75-1.0", "0.75-0.85", "0.75-0.75", "0.5", "1.5",
        "1+", "1e-8", "1e-8+", "1e-8-2e-8"}) {
    ASSERT_OK_AND_ASSIGN(MemFraction fraction, ParseMemFraction(spec));
    const std::string text = MemFractionToString(fraction);
    ASSERT_OK_AND_ASSIGN(MemFraction reparsed, ParseMemFraction(text));
    EXPECT_EQ(MemFractionToString(reparsed), text) << spec;
    EXPECT_EQ(reparsed, fraction) << spec;
  }
  EXPECT_EQ(MemFractionToString(FlexMemFraction{0.75, 1.0}), "0.75+");
  EXPECT_EQ(MemFractionToString(FlexMemFraction{0.75, 0.85}), "0.75-0.85");
  EXPECT_EQ(MemFractionToString(FixedMemFraction{0.75}), "0.75");
}

}  // namespace
}  // namespace xla

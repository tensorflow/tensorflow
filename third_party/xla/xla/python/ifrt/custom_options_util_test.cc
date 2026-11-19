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

#include "xla/python/ifrt/custom_options_util.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/attribute_map.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::Eq;
using ::testing::Optional;

constexpr absl::string_view kTestInternalOptionA = "test_internal_option_a";
constexpr absl::string_view kTestInternalOptionB = "test_internal_option_b";

// clang-format off
bool register_test_internal_custom_options = ([] {
  RegisterIfrtInternalCustomOption(kTestInternalOptionA);
  RegisterIfrtInternalCustomOption(kTestInternalOptionB);
}(), true);
// clang-format on

TEST(CustomOptionsUtilTest, IsIfrtInternalCustomOption) {
  EXPECT_TRUE(IsIfrtInternalCustomOption(kTestInternalOptionA));
  EXPECT_TRUE(IsIfrtInternalCustomOption(kTestInternalOptionB));
  EXPECT_TRUE(IsIfrtInternalCustomOption("test_internal_option_a"));
  EXPECT_TRUE(IsIfrtInternalCustomOption("test_internal_option_b"));

  // Package-specific keys are not pre-registered in pure IFRT.
  EXPECT_FALSE(IsIfrtInternalCustomOption("use_output_arena"));
  EXPECT_FALSE(IsIfrtInternalCustomOption("call_location"));
  EXPECT_FALSE(IsIfrtInternalCustomOption("fill_all_statuses"));

  EXPECT_FALSE(IsIfrtInternalCustomOption(""));
  EXPECT_FALSE(IsIfrtInternalCustomOption("delta"));
  EXPECT_FALSE(IsIfrtInternalCustomOption("test_internal_option_a_extra"));
  EXPECT_FALSE(IsIfrtInternalCustomOption("TEST_INTERNAL_OPTION_A"));
}

TEST(CustomOptionsUtilTest, DynamicRegistrationCopiesKey) {
  static const bool kRegisteredOnce = [] {
    std::string ephemeral_key = "test_ephemeral_internal_option";
    RegisterIfrtInternalCustomOption(ephemeral_key);
    return true;
  }();
  (void)kRegisteredOnce;
  EXPECT_TRUE(IsIfrtInternalCustomOption("test_ephemeral_internal_option"));
}

TEST(CustomOptionsUtilTest, DuplicateRegistrationDies) {
  EXPECT_DEATH(RegisterIfrtInternalCustomOption(kTestInternalOptionA),
               "cannot be registered more than once");
}

TEST(CustomOptionsUtilTest, FilterOutEmptyOrInternalOnlyReturnsNullopt) {
  EXPECT_EQ(FilterOutIfrtInternalCustomOptions(AttributeMap({})), std::nullopt);

  AttributeMap internal_only({
      {std::string(kTestInternalOptionA), AttributeMap::BoolValue(true)},
      {std::string(kTestInternalOptionB),
       AttributeMap::StringValue("test.py:1")},
  });
  EXPECT_EQ(FilterOutIfrtInternalCustomOptions(internal_only), std::nullopt);
}

TEST(CustomOptionsUtilTest, FilterOutStripsInternalKeysAndPreservesOthers) {
  AttributeMap input({
      {std::string(kTestInternalOptionA), AttributeMap::BoolValue(true)},
      {std::string(kTestInternalOptionB),
       AttributeMap::StringValue("test.py:42\ntraceback")},
      {"str", AttributeMap::StringValue("value")},
      {"flag", AttributeMap::BoolValue(false)},
      {"count", AttributeMap::Int64Value(42)},
      {"list", AttributeMap::Int64ListValue(std::vector<int64_t>{1, 2, 3})},
      {"ratio", AttributeMap::FloatValue(0.5f)},
  });

  AttributeMap expected({
      {"str", AttributeMap::StringValue("value")},
      {"flag", AttributeMap::BoolValue(false)},
      {"count", AttributeMap::Int64Value(42)},
      {"list", AttributeMap::Int64ListValue(std::vector<int64_t>{1, 2, 3})},
      {"ratio", AttributeMap::FloatValue(0.5f)},
  });

  EXPECT_THAT(FilterOutIfrtInternalCustomOptions(input),
              Optional(Eq(expected)));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla

/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/attribute_map.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(AttributeMapTest, MapElements) {
  AttributeMap map({
      {"string", AttributeMap::StringValue("value")},
      {"bool", AttributeMap::BoolValue(true)},
      {"int64", AttributeMap::Int64Value(123)},
      {"int64_list", AttributeMap::Int64ListValue({int64_t{1}, int64_t{2}})},
      {"float", AttributeMap::FloatValue(1.23f)},
  });

  EXPECT_EQ(map, AttributeMap({
                     {"string", AttributeMap::StringValue("value")},
                     {"bool", AttributeMap::BoolValue(true)},
                     {"int64", AttributeMap::Int64Value(123)},
                     {"int64_list",
                      AttributeMap::Int64ListValue({int64_t{1}, int64_t{2}})},
                     {"float", AttributeMap::FloatValue(1.23f)},
                 }))
      << map.DebugString();
}

TEST(AttributeMapTest, Get) {
  AttributeMap map({
      {"string", AttributeMap::StringValue("value")},
      {"bool", AttributeMap::BoolValue(true)},
      {"int64", AttributeMap::Int64Value(123)},
      {"int64_list", AttributeMap::Int64ListValue({int64_t{1}, int64_t{2}})},
      {"float", AttributeMap::FloatValue(1.23f)},
  });

  EXPECT_THAT(map.Get<std::string>("string"), IsOkAndHolds("value"));
  EXPECT_THAT(map.Get<bool>("bool"), IsOkAndHolds(true));
  EXPECT_THAT(map.Get<int64_t>("int64"), IsOkAndHolds(123));
  EXPECT_THAT(map.Get<std::vector<int64_t>>("int64_list"),
              IsOkAndHolds(std::vector<int64_t>{1, 2}));
  EXPECT_THAT(map.Get<float>("float"), IsOkAndHolds(1.23f));

  EXPECT_THAT(map.Get<std::string>("float"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value type mismatch for key: float")));
  EXPECT_THAT(map.Get<std::vector<int64_t>>("string"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value type mismatch for key: string")));
}

TEST(AttributeMapTest, Set) {
  AttributeMap map({});
  TF_ASSERT_OK(map.Set("string", "value"));
  TF_ASSERT_OK(map.Set("bool", true));
  TF_ASSERT_OK(map.Set("int64", int64_t{123}));
  TF_ASSERT_OK(map.Set("int64_list", std::vector<int64_t>{1, 2}));
  TF_ASSERT_OK(map.Set("float", 1.23f));
  EXPECT_EQ(map, AttributeMap({
                     {"string", AttributeMap::StringValue("value")},
                     {"bool", AttributeMap::BoolValue(true)},
                     {"int64", AttributeMap::Int64Value(123)},
                     {"int64_list",
                      AttributeMap::Int64ListValue({int64_t{1}, int64_t{2}})},
                     {"float", AttributeMap::FloatValue(1.23f)},
                 }))
      << map.DebugString();
}

class AttributeMapSerDesTest : public testing::TestWithParam<SerDesVersion> {
 public:
  AttributeMapSerDesTest() : version_(GetParam()) {}

  SerDesVersion version() const { return version_; }

 private:
  SerDesVersion version_;
};

TEST_P(AttributeMapSerDesTest, ToFromProto) {
  AttributeMap map({
      {"string", AttributeMap::StringValue("value")},
      {"bool", AttributeMap::BoolValue(true)},
      {"int64", AttributeMap::Int64Value(123)},
      {"int64_list", AttributeMap::Int64ListValue({int64_t{1}, int64_t{2}})},
      {"float", AttributeMap::FloatValue(1.23f)},
  });

  TF_ASSERT_OK_AND_ASSIGN(auto map_copy,
                          AttributeMap::FromProto(map.ToProto(version())));
  EXPECT_EQ(map_copy, map) << map_copy.DebugString();
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion, AttributeMapSerDesTest,
    testing::ValuesIn(test_util::AllSupportedSerDesVersions()));

}  // namespace
}  // namespace ifrt
}  // namespace xla

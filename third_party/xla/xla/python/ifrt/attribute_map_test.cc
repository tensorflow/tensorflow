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

#include <gtest/gtest.h>
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

TEST(AttributeMapTest, MapElements) {
  AttributeMap map({
      {"string", AttributeMap::StringValue("value")},
      {"bool", AttributeMap::BoolValue(true)},
      {"int64", AttributeMap::Int64Value(123)},
      {"int64_list", AttributeMap::Int64ListValue({int64_t{1}, int64_t{2}})},
      {"float", AttributeMap::FloatValue(1.23f)},
  });

  EXPECT_EQ(map.map(), AttributeMap::Map({
                           {"string", AttributeMap::StringValue("value")},
                           {"bool", AttributeMap::BoolValue(true)},
                           {"int64", AttributeMap::Int64Value(123)},
                           {"int64_list", AttributeMap::Int64ListValue(
                                              {int64_t{1}, int64_t{2}})},
                           {"float", AttributeMap::FloatValue(1.23f)},
                       }))
      << map.DebugString();
}

TEST(AttributeMapTest, ToFromProto) {
  AttributeMap map({
      {"string", AttributeMap::StringValue("value")},
      {"bool", AttributeMap::BoolValue(true)},
      {"int64", AttributeMap::Int64Value(123)},
      {"int64_list", AttributeMap::Int64ListValue({int64_t{1}, int64_t{2}})},
      {"float", AttributeMap::FloatValue(1.23f)},
  });

  TF_ASSERT_OK_AND_ASSIGN(auto map_copy,
                          AttributeMap::FromProto(map.ToProto()));
  EXPECT_EQ(map_copy.map(), map.map()) << map_copy.DebugString();
}

}  // namespace
}  // namespace ifrt
}  // namespace xla

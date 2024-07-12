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

#include "xla/python/pjrt_ifrt/pjrt_attribute_map_util.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/python/ifrt/attribute_map.h"

namespace xla {
namespace ifrt {
namespace {

TEST(PjRtAttributeMapUtilTest, FromPjRtAttributeMap) {
  absl::flat_hash_map<std::string, PjRtValueType> pjrt_map({
      {"string", xla::PjRtValueType(std::string("value"))},
      {"bool", xla::PjRtValueType(true)},
      {"int64", xla::PjRtValueType(int64_t{123})},
      {"int64_list",
       xla::PjRtValueType(std::vector<int64_t>({int64_t{1}, int64_t{2}}))},
      {"float", xla::PjRtValueType(1.23f)},
  });

  EXPECT_EQ(FromPjRtAttributeMap(pjrt_map).map(),
            AttributeMap::Map({
                {"string", AttributeMap::StringValue("value")},
                {"bool", AttributeMap::BoolValue(true)},
                {"int64", AttributeMap::Int64Value(123)},
                {"int64_list",
                 AttributeMap::Int64ListValue({int64_t{1}, int64_t{2}})},
                {"float", AttributeMap::FloatValue(1.23f)},
            }));
}

TEST(PjRtAttributeMapUtilTest, ToPjRtAttributeMap) {
  AttributeMap map({
      {"string", AttributeMap::StringValue("value")},
      {"bool", AttributeMap::BoolValue(true)},
      {"int64", AttributeMap::Int64Value(123)},
      {"int64_list", AttributeMap::Int64ListValue({int64_t{1}, int64_t{2}})},
      {"float", AttributeMap::FloatValue(1.23f)},
  });

  EXPECT_EQ(
      ToPjRtAttributeMap(map),
      (absl::flat_hash_map<std::string, xla::PjRtValueType>({
          {"string", xla::PjRtValueType(std::string("value"))},
          {"bool", xla::PjRtValueType(true)},
          {"int64", xla::PjRtValueType(int64_t{123})},
          {"int64_list",
           xla::PjRtValueType(std::vector<int64_t>({int64_t{1}, int64_t{2}}))},
          {"float", xla::PjRtValueType(1.23f)},
      })));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla

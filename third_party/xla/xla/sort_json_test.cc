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

#include "xla/sort_json.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

TEST(SortJsonTest, SortsJson) {
  EXPECT_THAT(SortJson(R"({"a": 1, "c": 3,"b": 2, "b": 1,})"),
              absl_testing::IsOkAndHolds(R"({"a":1,"b":1,"b":2,"c":3})"));

  EXPECT_THAT(SortJson(R"({"a": 1  , "c": 1,"b": 1  })"),
              absl_testing::IsOkAndHolds(R"({"a":1,"b":1,"c":1})"));

  EXPECT_THAT(SortJson(R"({"a": 1,"c": 3,"b": 2,"b": [3,2,1],})"),
              absl_testing::IsOkAndHolds(R"({"a":1,"b":2,"b":[3,2,1],"c":3})"));

  EXPECT_THAT(SortJson(R"({"aa": 1, "a": {"c": "c", "b": "b"}})"),
              absl_testing::IsOkAndHolds(R"({"a":{"b":"b","c":"c"},"aa":1})"));

  EXPECT_THAT(
      SortJson(
          R"({"x": true, "x": false, "x": null, "x": 0, "x": -0.5,"x": "a"})"),
      absl_testing::IsOkAndHolds(
          R"({"x":"a","x":-0.5,"x":0,"x":false,"x":null,"x":true})"));

  EXPECT_THAT(SortJson(R"({"a": "a}", "a": "a"})"),
              absl_testing::IsOkAndHolds(R"({"a":"a","a":"a}"})"));
}

}  // namespace
}  // namespace xla

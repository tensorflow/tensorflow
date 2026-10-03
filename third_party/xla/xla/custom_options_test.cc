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

#include "xla/custom_options.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

TEST(CustomOptionsTest, Get) {
  CustomOptions options({
      {"str", std::string("foo")},
      {"i64", int64_t{42}},
      {"f32", 1.5f},
      {"b", true},
      {"arr", std::vector<int64_t>{1, 2, 3}},
  });

  EXPECT_EQ(options.size(), 5);
  EXPECT_FALSE(options.empty());
  EXPECT_TRUE(options.contains("i64"));
  EXPECT_FALSE(options.contains("missing"));
  EXPECT_NE(options.Find("i64"), nullptr);
  EXPECT_EQ(options.Find("missing"), nullptr);

  ASSERT_OK_AND_ASSIGN(std::string str, options.Get<std::string>("str"));
  ASSERT_OK_AND_ASSIGN(int64_t i64, options.Get<int64_t>("i64"));
  ASSERT_OK_AND_ASSIGN(float f32, options.Get<float>("f32"));
  ASSERT_OK_AND_ASSIGN(bool b, options.Get<bool>("b"));
  ASSERT_OK_AND_ASSIGN(std::vector<int64_t> arr,
                       options.Get<std::vector<int64_t>>("arr"));

  EXPECT_EQ(str, "foo");
  EXPECT_EQ(i64, 42);
  EXPECT_EQ(f32, 1.5f);
  EXPECT_TRUE(b);
  EXPECT_THAT(arr, ElementsAre(1, 2, 3));

  EXPECT_THAT(options.Get<int64_t>("missing"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(options.Get<std::string>("i64"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla

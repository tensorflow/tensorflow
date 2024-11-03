// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/core/dynamic_loading.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "testing/base/public/unique-test-directory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace {

using ::litert::testing::TouchTestFile;

constexpr absl::string_view kNotLiteRtSo = "notLibLiteRt.so";
constexpr absl::string_view kLiteRtSo1 = "libLiteRtPlugin_1.so";
constexpr absl::string_view kLiteRtSo2 = "libLiteRtPlugin_2.so";

TEST(TestDynamicLoading, GlobNoMatch) {
  const auto dir = testing::UniqueTestDirectory();
  TouchTestFile(kNotLiteRtSo, dir);

  std::vector<std::string> results;
  ASSERT_STATUS_OK(litert::internal::FindLiteRtSharedLibs(dir, results));
  EXPECT_EQ(results.size(), 0);
}

TEST(TestDynamicLoading, GlobOneMatch) {
  const auto dir = testing::UniqueTestDirectory();
  TouchTestFile(kLiteRtSo1, dir);
  TouchTestFile(kNotLiteRtSo, dir);

  std::vector<std::string> results;
  ASSERT_STATUS_OK(litert::internal::FindLiteRtSharedLibs(dir, results));
  EXPECT_EQ(results.size(), 1);
  EXPECT_TRUE(absl::string_view(results.front()).ends_with(kLiteRtSo1));
}

TEST(TestDynamicLoading, GlobMultiMatch) {
  const auto dir = testing::UniqueTestDirectory();
  TouchTestFile(kLiteRtSo1, dir);
  TouchTestFile(kLiteRtSo2, dir);
  TouchTestFile(kNotLiteRtSo, dir);

  std::vector<std::string> results;
  ASSERT_STATUS_OK(litert::internal::FindLiteRtSharedLibs(dir, results));
  EXPECT_EQ(results.size(), 2);
  EXPECT_THAT(results, testing::Contains(testing::HasSubstr(kLiteRtSo1)));
  EXPECT_THAT(results, testing::Contains(testing::HasSubstr(kLiteRtSo2)));
}

}  // namespace

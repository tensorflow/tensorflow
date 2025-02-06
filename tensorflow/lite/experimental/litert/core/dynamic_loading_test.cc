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
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/core/filesystem.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"

namespace litert::internal {
namespace {

using litert::testing::UniqueTestDirectory;
using ::testing::Contains;
using ::testing::HasSubstr;

constexpr absl::string_view kNotLiteRtSo = "notLibLiteRt.so";
constexpr absl::string_view kLiteRtSo1 = "libLiteRtCompilerPlugin_1.so";
constexpr absl::string_view kLiteRtSo2 = "libLiteRtCompilerPlugin_2.so";

TEST(TestDynamicLoading, GlobNoMatch) {
  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);
  Touch(Join({dir->Str(), kNotLiteRtSo}));

  std::vector<std::string> results;
  LITERT_ASSERT_OK(litert::internal::FindLiteRtSharedLibs(dir->Str(), results));
  EXPECT_EQ(results.size(), 0);
}

TEST(TestDynamicLoading, GlobOneMatch) {
  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);
  Touch(Join({dir->Str(), kLiteRtSo1}));
  Touch(Join({dir->Str(), kNotLiteRtSo}));

  std::vector<std::string> results;
  LITERT_ASSERT_OK(litert::internal::FindLiteRtSharedLibs(dir->Str(), results));
  ASSERT_EQ(results.size(), 1);
  EXPECT_TRUE(absl::string_view(results.front()).ends_with(kLiteRtSo1));
}

TEST(TestDynamicLoading, GlobMultiMatch) {
  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);
  Touch(Join({dir->Str(), kLiteRtSo1}));
  Touch(Join({dir->Str(), kLiteRtSo2}));
  Touch(Join({dir->Str(), kNotLiteRtSo}));

  std::vector<std::string> results;
  LITERT_ASSERT_OK(litert::internal::FindLiteRtSharedLibs(dir->Str(), results));
  ASSERT_EQ(results.size(), 2);
  EXPECT_THAT(results, Contains(HasSubstr(kLiteRtSo1)));
  EXPECT_THAT(results, Contains(HasSubstr(kLiteRtSo2)));
}

}  // namespace
}  // namespace litert::internal

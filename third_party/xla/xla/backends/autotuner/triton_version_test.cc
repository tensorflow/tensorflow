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

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "triton/Version.h"

namespace {

constexpr absl::string_view kTritonCommitPattern = "TRITON_COMMIT = \"";

std::optional<std::string> ReadFile(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return std::nullopt;
  }
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

std::optional<absl::string_view> ExtractExpectedVersion(
    absl::string_view content) {
  size_t pos = content.find(kTritonCommitPattern);
  if (pos == absl::string_view::npos) {
    return std::nullopt;
  }
  size_t start = pos + kTritonCommitPattern.size();
  size_t end = content.find('\"', start);
  if (end == absl::string_view::npos) {
    return std::nullopt;
  }
  return content.substr(start, end - start);
}

TEST(VersionTest, MatchExpected) {
  const char* env_path = std::getenv("WORKSPACE_BZL_PATH");
  ASSERT_NE(env_path, nullptr)
      << "WORKSPACE_BZL_PATH environment variable is not set";
  std::string path(env_path);
  ASSERT_FALSE(path.empty())
      << "WORKSPACE_BZL_PATH environment variable is empty";

  std::optional<std::string> content = ReadFile(path);
  ASSERT_TRUE(content.has_value()) << "Failed to read file: " << path;

  std::optional<absl::string_view> expected_version =
      ExtractExpectedVersion(*content);
  ASSERT_TRUE(expected_version.has_value())
      << "Failed to extract TRITON_COMMIT from " << path;
  ASSERT_FALSE(expected_version->empty())
      << "Expected version should not be empty";
  ASSERT_FALSE(std::string(TRITON_VERSION).empty())
      << "Triton version should not be empty";
  EXPECT_EQ(TRITON_VERSION, expected_version.value());
}

}  // namespace

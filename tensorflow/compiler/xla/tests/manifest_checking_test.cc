/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tests/manifest_checking_test.h"

#include <fstream>
#include <iterator>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"

namespace xla {

namespace {

// Mapping from test name; i.e. MyTest.MyTestCase to platforms on which it is
// disabled - a sequence of regexps.
using ManifestT = absl::flat_hash_map<std::string, std::vector<std::string>>;

ManifestT ReadManifest() {
  ManifestT manifest;

  absl::string_view path = absl::NullSafeStringView(*DisabledManifestPath());
  if (path.empty()) {
    return manifest;
  }

  // Note: parens are required to disambiguate vs function decl.
  std::ifstream file_stream((std::string(path)));
  std::string contents((std::istreambuf_iterator<char>(file_stream)),
                       std::istreambuf_iterator<char>());

  std::vector<std::string> lines = absl::StrSplit(contents, '\n');
  for (std::string& line : lines) {
    auto comment = line.find("//");
    if (comment != std::string::npos) {
      line = line.substr(0, comment);
    }
    if (line.empty()) {
      continue;
    }
    absl::StripTrailingAsciiWhitespace(&line);
    std::vector<std::string> pieces = absl::StrSplit(line, ' ');
    CHECK_GE(pieces.size(), 1);
    auto& platforms = manifest[pieces[0]];
    for (size_t i = 1; i < pieces.size(); ++i) {
      platforms.push_back(pieces[i]);
    }
  }
  return manifest;
}

}  // namespace

void ManifestCheckingTest::SetUp() {
  const testing::TestInfo* test_info =
      testing::UnitTest::GetInstance()->current_test_info();
  absl::string_view test_case_name = test_info->test_suite_name();
  absl::string_view test_name = test_info->name();
  VLOG(1) << "test_case_name: " << test_case_name;
  VLOG(1) << "test_name: " << test_name;

  // Remove the type suffix from the test case name.
  if (const char* type_param = test_info->type_param()) {
    VLOG(1) << "type_param: " << type_param;
    size_t last_slash = test_case_name.rfind('/');
    test_case_name = test_case_name.substr(0, last_slash);
    VLOG(1) << "test_case_name: " << test_case_name;
  }

  // Remove the test instantiation name if it is present.
  auto first_slash = test_case_name.find('/');
  if (first_slash != test_case_name.npos) {
    test_case_name.remove_prefix(first_slash + 1);
    VLOG(1) << "test_case_name: " << test_case_name;
  }

  ManifestT manifest = ReadManifest();

  // If the test name ends with a slash followed by one or more characters,
  // strip that off.
  auto last_slash = test_name.rfind('/');
  if (last_slash != test_name.npos) {
    test_name = test_name.substr(0, last_slash);
    VLOG(1) << "test_name: " << test_name;
  }

  // First try full match: test_case_name.test_name
  // If that fails, try to find just the test_case_name; this would disable all
  // tests in the test case.
  auto it = manifest.find(absl::StrCat(test_case_name, ".", test_name));
  if (it == manifest.end()) {
    it = manifest.find(test_case_name);
    if (it == manifest.end()) {
      return;
    }
  }

  // Expect a full match vs. one of the platform regexps to disable the test.
  const std::vector<std::string>& disabled_platforms = it->second;
  auto platform_string = *TestPlatform();
  for (const auto& s : disabled_platforms) {
    if (RE2::FullMatch(/*text=*/platform_string, /*re=*/s)) {
      GTEST_SKIP();
      return;
    }
  }

  // We didn't hit in the disabled manifest entries, so don't disable it.
}

}  // namespace xla

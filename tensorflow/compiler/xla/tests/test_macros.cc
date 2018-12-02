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

#include "tensorflow/compiler/xla/tests/test_macros.h"

#include <fstream>
#include <streambuf>
#include <string>
#include <unordered_map>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"

namespace xla {
namespace {

// Mapping from test name; i.e. MyTest.MyTestCase to platforms on which it is
// disabled - a sequence of regexps.
using ManifestT = std::unordered_map<string, std::vector<string>>;

ManifestT ReadManifest() {
  ManifestT manifest;

  string path = XLA_DISABLED_MANIFEST;
  if (path.empty()) {
    return manifest;
  }

  std::ifstream file_stream(path);
  // Note: parens are required to disambiguate vs function decl.
  string contents((std::istreambuf_iterator<char>(file_stream)),
                  std::istreambuf_iterator<char>());

  std::vector<string> lines = absl::StrSplit(contents, '\n');
  for (string& line : lines) {
    auto comment = line.find("//");
    if (comment != string::npos) {
      line = line.substr(0, comment);
    }
    if (line.empty()) {
      continue;
    }
    absl::StripTrailingAsciiWhitespace(&line);
    std::vector<string> pieces = absl::StrSplit(line, ' ');
    CHECK_GE(pieces.size(), 1);
    auto& platforms = manifest[pieces[0]];
    for (int64 i = 1; i < pieces.size(); ++i) {
      platforms.push_back(pieces[i]);
    }
  }
  return manifest;
}

}  // namespace

string PrependDisabledIfIndicated(const string& test_case_name,
                                  const string& test_name) {
  ManifestT manifest = ReadManifest();

  // First try full match: test_case_name.test_name
  // If that fails, try to find just the test_case_name; this would disable all
  // tests in the test case.
  auto it = manifest.find(absl::StrCat(test_case_name, ".", test_name));
  if (it == manifest.end()) {
    it = manifest.find(test_case_name);
    if (it == manifest.end()) {
      return test_name;
    }
  }

  // Expect a full match vs. one of the platform regexps to disable the test.
  const std::vector<string>& disabled_platforms = it->second;
  string platform_string = XLA_PLATFORM;
  for (const auto& s : disabled_platforms) {
    if (RE2::FullMatch(/*text=*/platform_string, /*re=*/s)) {
      return "DISABLED_" + test_name;
    }
  }

  // We didn't hit in the disabled manifest entries, so don't disable it.
  return test_name;
}

}  // namespace xla

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cassert>
#include <regex>  // NOLINT
#include <string>
#include <string_view>

#include "fuzztest/fuzztest.h"
#include "absl/strings/match.h"
#include "tensorflow/core/platform/path.h"

// This is a fuzzer for tensorflow::io::CleanPath.

namespace {

void FuzzTest(std::string_view input_path) {
  std::string clean_path = tensorflow::io::CleanPath(input_path);

  // Assert there are no '/./' no directory changes.
  assert(!absl::StrContains(clean_path, "/./"));
  // Assert there are no duplicate '/'.
  assert(!absl::StrContains(clean_path, "//"));
  // Assert there are no higher up directories after entering a directory.
  std::regex higher_up_directory("[^.]{1}/[.]{2}");
  assert(!std::regex_match(clean_path, higher_up_directory));
}
FUZZ_TEST(CC_FUZZING, FuzzTest);

}  // namespace

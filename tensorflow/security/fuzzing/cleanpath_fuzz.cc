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

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <regex>  // NOLINT

#include "absl/strings/match.h"
#include "tensorflow/core/platform/path.h"

// This is a fuzzer for tensorflow::io::CleanPath.

namespace {

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  std::string input_path(reinterpret_cast<const char *>(data), size);
  std::string clean_path = tensorflow::io::CleanPath(input_path);

  // Assert there are no '/./' no directory changes.
  assert(!absl::StrContains(clean_path, "/./"));
  // Assert there are no duplicate '/'.
  assert(!absl::StrContains(clean_path, "//"));
  // Assert there are no higher up directories after entering a directory.
  std::regex higher_up_directory("[^.]{1}/[.]{2}");
  assert(!std::regex_match(clean_path, higher_up_directory));

  return 0;
}

}  // namespace

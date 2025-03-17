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
#include <string>
#include <string_view>

#include "fuzztest/fuzztest.h"
#include "absl/strings/match.h"
#include "tensorflow/core/platform/path.h"

// This is a fuzzer for tensorflow::io::JoinPath.

namespace {

void FuzzTest(std::string_view first, std::string_view second) {
  std::string path = tensorflow::io::JoinPath(first, second);

  // Assert path contains strings
  assert(absl::StrContains(path, first));
  assert(absl::StrContains(path, second));
}
FUZZ_TEST(CC_FUZZING, FuzzTest);

}  // namespace

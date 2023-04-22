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
#include <fuzzer/FuzzedDataProvider.h>

#include <cstdint>
#include <cstdlib>

#include "absl/strings/match.h"
#include "tensorflow/core/platform/path.h"

// This is a fuzzer for tensorflow::io::JoinPath.

namespace {

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  FuzzedDataProvider fuzzed_data(data, size);

  // Choose random numbers here.
  const int content_size = fuzzed_data.ConsumeIntegralInRange(10, 300);

  std::string first = fuzzed_data.ConsumeRandomLengthString(content_size);
  std::string second = fuzzed_data.ConsumeRemainingBytesAsString();

  std::string path = tensorflow::io::JoinPath(first, second);

  // Assert path contains strings
  assert(absl::StrContains(path, first));
  assert(absl::StrContains(path, second));

  return 0;
}

}  // namespace

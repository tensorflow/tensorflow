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

#include "tensorflow/core/platform/stringprintf.h"

// This is a fuzzer for tensorflow::strings::Printf

namespace {

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  FuzzedDataProvider fuzzed_data(data, size);

  const char split = fuzzed_data.ConsumeIntegral<char>();
  const char split_a = split & 0x07;
  const char split_b = (split >> 3) & 0x07;

  const std::string sa_string = fuzzed_data.ConsumeBytesAsString(split_a);
  const std::string sb_string = fuzzed_data.ConsumeBytesAsString(split_b);
  const std::string sc_string = fuzzed_data.ConsumeRemainingBytesAsString();
  const char *sa = sa_string.c_str();
  const char *sb = sb_string.c_str();
  const char *sc = sc_string.c_str();

  tensorflow::strings::Printf("%s %s %s", sa, sb, sc);

  return 0;
}

}  // namespace

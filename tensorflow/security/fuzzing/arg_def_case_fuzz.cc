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

#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/stringpiece.h"

// This is a fuzzer for tensorflow::str_util::ArgDefCase

namespace {

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  uint8_t *byte_data = const_cast<uint8_t *>(data);
  char *char_data = reinterpret_cast<char *>(byte_data);

  tensorflow::StringPiece sp(char_data, size);

  std::string ns = tensorflow::str_util::ArgDefCase(sp);
  for (const auto &c : ns) {
    const bool is_letter = 'a' <= c && c <= 'z';
    const bool is_digit = '0' <= c && c <= '9';
    if (!is_letter && !is_digit) {
      assert(c == '_');
    }
  }

  return 0;
}

}  // namespace

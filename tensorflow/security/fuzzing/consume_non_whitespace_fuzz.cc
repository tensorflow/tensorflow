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

// This is a fuzzer for tensorflow::str_util::ConsumeNonWhitespace

namespace {

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  uint8_t *byte_data = const_cast<uint8_t *>(data);
  char *char_data = reinterpret_cast<char *>(byte_data);

  tensorflow::StringPiece sp(char_data, size);
  tensorflow::StringPiece spe;

  while (!sp.empty()) {
    const size_t initial_size = sp.size();
    (void)initial_size;  // "use" initial_size even if assert is disabled

    const bool leading_whitespace =
        tensorflow::str_util::ConsumeNonWhitespace(&sp, &spe);

    if (leading_whitespace) {
      assert(!spe.empty());
    }
    assert(initial_size == (sp.size() + spe.size()));

    tensorflow::str_util::RemoveLeadingWhitespace(&sp);
    assert(initial_size > sp.size());
  }

  return 0;
}

}  // namespace

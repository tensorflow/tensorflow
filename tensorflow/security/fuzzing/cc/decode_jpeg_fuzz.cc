/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Fuzzer for tensorflow::jpeg::Uncompress (underlying decoder for
// tf.image.decode_jpeg / decode_image).
//
// This fuzzer exercises the JPEG decoder directly, without going through the
// TF op layer, to catch memory-safety issues (OOM, buffer overflow, etc.)
// in the libjpeg-turbo wrapper code.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/lib/jpeg/jpeg_mem.h"

namespace {

void FuzzDecodeJpeg(std::string_view data) {
  // Allocate at most 256 MB of output to avoid OOM in the fuzzer itself.
  // The purpose of the bounds-check patch (PR #115498) is to enforce this
  // limit inside the TF op kernel, consistent with the existing 512 MB cap
  // already enforced by jpeg_mem.cc.
  constexpr int64_t kMaxFuzzerAlloc = 256 * 1024 * 1024;  // 256 MB

  tensorflow::jpeg::UncompressFlags flags;
  flags.components = 3;  // Force RGB output (same as TF default)

  std::vector<uint8_t> output;
  auto allocate_output = [&](int width, int height,
                             int components) -> uint8_t* {
    int64_t total =
        static_cast<int64_t>(width) * height * components;
    if (total <= 0 || total > kMaxFuzzerAlloc) {
      return nullptr;
    }
    output.resize(static_cast<size_t>(total));
    return output.data();
  };

  int64_t nwarn = 0;
  tensorflow::jpeg::Uncompress(data.data(), static_cast<int>(data.size()),
                               flags, &nwarn, allocate_output);
}

FUZZ_TEST(TF_JPEG_FUZZING, FuzzDecodeJpeg);

}  // namespace

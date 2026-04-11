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

// Fuzzer for tensorflow::gif::Decode (underlying decoder for tf.image.decode_gif).
//
// This fuzzer exercises the GIF decoder directly, without going through the
// TF op layer, to catch memory-safety issues (OOM, buffer overflow, etc.)
// in the giflib wrapper code.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/lib/gif/gif_io.h"

namespace {

void FuzzDecodeGif(std::string_view data) {
  std::string error_string;
  // Allocate at most 256 MB of output to avoid OOM in the fuzzer itself.
  // The purpose of the bounds-check patch (PR #115498) is to enforce this
  // limit inside the TF op kernel before giflib is even called.
  constexpr int64_t kMaxFuzzerAlloc = 256 * 1024 * 1024;  // 256 MB

  std::vector<uint8_t> output;
  auto allocate_output = [&](int num_frames, int width, int height,
                             int channels) -> uint8_t* {
    int64_t total =
        static_cast<int64_t>(num_frames) * width * height * channels;
    if (total <= 0 || total > kMaxFuzzerAlloc) {
      return nullptr;
    }
    output.resize(static_cast<size_t>(total));
    return output.data();
  };

  tensorflow::gif::Decode(data.data(), static_cast<int>(data.size()),
                          allocate_output, &error_string,
                          /*expand_animations=*/true);
}

FUZZ_TEST(TF_GIF_FUZZING, FuzzDecodeGif);

}  // namespace

// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"

#ifndef NDEBUG
// Make flatbuffers verifier `assert` in debug mode.
#define FLATBUFFERS_DEBUG_VERIFICATION_FAILURE

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers  // IWYU pragma: keep
#endif

#include <cstddef>
#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::internal {

using ::flatbuffers::Verifier;
using ::tflite::VerifyModelBuffer;

absl::string_view FbBufToStr(const uint8_t* fb_data, size_t size) {
  auto fb_buf_raw = reinterpret_cast<const char*>(fb_data);
  return absl::string_view(fb_buf_raw, size);
}

absl::string_view FbBufToStr(absl::Span<const uint8_t> fb_buf) {
  auto fb_buf_raw = reinterpret_cast<const char*>(fb_buf.data());
  const size_t fb_buf_size = fb_buf.size();
  return absl::string_view(fb_buf_raw, fb_buf_size);
}

absl::Span<char> FbBufToStr(absl::Span<uint8_t> fb_buf) {
  return absl::MakeSpan(reinterpret_cast<char*>(fb_buf.data()), fb_buf.size());
}

absl::Span<char> FbBufToStr(uint8_t* fb_data, size_t size) {
  return absl::MakeSpan(reinterpret_cast<char*>(fb_data), size);
}

bool VerifyFlatbuffer(absl::Span<const uint8_t> buf) {
  return VerifyFlatbuffer(buf.data(), buf.size());
}

bool VerifyFlatbuffer(const uint8_t* buf, size_t buf_size) {
  flatbuffers::Verifier::Options options;
#ifndef NDEBUG
  options.assert = true;
#endif
  flatbuffers::Verifier verifier(buf, buf_size, options);
  return VerifyModelBuffer(verifier);
}

}  // namespace litert::internal

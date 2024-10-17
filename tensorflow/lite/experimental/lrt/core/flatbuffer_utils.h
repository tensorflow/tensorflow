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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_FLATBUFFER_UTILS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_FLATBUFFER_UTILS_H_

#include <cstdint>
#include <sstream>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::internal {

using ::flatbuffers::Verifier;
using ::tflite::VerifyModelBuffer;

// Flatbuffer's raw char type.
typedef uint8_t FbCharT;

// A string with flatbuffer's raw char type.
typedef std::basic_string<FbCharT> FbStringT;

// A string stream with flatbuffer's raw char type.
typedef std::basic_stringstream<FbCharT> FbStringStreamT;

// Const view of flatbuffer's raw buffer type.
typedef absl::Span<const FbCharT> FbConstBufferT;

// Mutable view of flatbuffer's raw buffer type.
typedef absl::Span<FbCharT> FbBufferT;

// Convenience method to get raw string view from native flatbuffer buffer.
inline absl::string_view FbBufToStr(FbConstBufferT fb_buf) {
  auto fb_buf_raw = reinterpret_cast<const char*>(fb_buf.data());
  const size_t fb_buf_size = fb_buf.size();
  return absl::string_view(fb_buf_raw, fb_buf_size);
}

// Mutable version of above.
inline absl::string_view FbBufToStr(FbBufferT fb_buf) {
  auto fb_buf_raw = reinterpret_cast<char*>(fb_buf.data());
  const size_t fb_buf_size = fb_buf.size();
  return absl::string_view(fb_buf_raw, fb_buf_size);
}

inline bool VerifyFlatbuffer(const FbCharT* buf, size_t buf_size) {
  Verifier::Options options;
  Verifier verifier(buf, buf_size, options);
  return VerifyModelBuffer(verifier);
}

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_FLATBUFFER_UTILS_H_

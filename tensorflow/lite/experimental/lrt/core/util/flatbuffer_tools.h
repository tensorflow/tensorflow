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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_UTIL_FLATBUFFER_TOOLS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_UTIL_FLATBUFFER_TOOLS_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace litert::internal {

// Flatbuffer's native char type is unsigned char.

// Convenience method to get string view from native flatbuffer chars.
absl::string_view FbBufToStr(const uint8_t* fb_data, size_t size);

// Span version.
absl::string_view FbBufToStr(absl::Span<const uint8_t> fb_buf);

// Convenience method to get mutable signed char span from native flatbuffer
// chars.
absl::Span<char> FbBufToStr(uint8_t* fb_data, size_t size);

// Span to span version.
absl::Span<char> FbBufToStr(absl::Span<uint8_t> fb_buf);

// Verifies given serialized flatbuffer
bool VerifyFlatbuffer(const uint8_t* buf, size_t buf_size);

// Override of above with view input.
bool VerifyFlatbuffer(absl::Span<const uint8_t> buf);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_UTIL_FLATBUFFER_TOOLS_H_

/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_STRING_UTILS_H_
#define XLA_PJRT_STRING_UTILS_H_

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"

namespace xla {

// Converts an array of C-style character buffers along with their sizes to a
// vector of C++ strings.
std::vector<std::string> ConvertCharBuffersToCppStrings(
    absl::Span<const char*> char_buffers,
    absl::Span<const size_t> char_buffer_sizes);

// Converts a vector of C++ strings to an array of C-style character buffers and
// their sizes. This function allocates memory for the character buffers, the
// individual C-strings, and `char_buffer_sizes`.
const char** ConvertCppStringsToCharBuffers(
    const std::vector<std::string>& strings, const size_t*& char_buffer_sizes);

}  // namespace xla

#endif  // XLA_PJRT_STRING_UTILS_H_

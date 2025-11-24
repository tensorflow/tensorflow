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

#include "xla/pjrt/string_utils.h"

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"

namespace xla {

std::vector<std::string> ConvertCharBuffersToCppStrings(
    absl::Span<const char*> char_buffers,
    absl::Span<const size_t> char_buffer_sizes) {
  if (char_buffers.empty() || char_buffer_sizes.empty()) {
    return {};
  }

  std::vector<std::string> cpp_strings;
  size_t num_strings = char_buffers.size();
  cpp_strings.reserve(num_strings);
  for (size_t i = 0; i < num_strings; ++i) {
    if (char_buffers[i] == nullptr) {
      cpp_strings.push_back("");
    } else {
      cpp_strings.push_back(std::string(char_buffers[i], char_buffer_sizes[i]));
    }
  }
  return cpp_strings;
}

const char** ConvertCppStringsToCharBuffers(
    const std::vector<std::string>& strings, const size_t*& char_buffer_sizes) {
  auto char_buffers = new const char*[strings.size()];
  size_t* buffer_sizes = new size_t[strings.size()];

  for (size_t i = 0; i < strings.size(); ++i) {
    const std::string& current_string = strings[i];
    char* buffer = new char[current_string.length() + 1];
    absl::SNPrintF(buffer, current_string.length() + 1, "%s", current_string);
    char_buffers[i] = buffer;
    buffer_sizes[i] = strlen(buffer);
  }
  char_buffer_sizes = buffer_sizes;
  return char_buffers;
}

}  // namespace xla

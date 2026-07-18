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

#include "xla/pjrt/partial_program_utils.h"

#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"

namespace xla {

absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>>
ConvertCharBuffersToPjRtPartialProgramProtos(
    absl::Span<const char*> char_buffers,
    absl::Span<const size_t> char_buffer_sizes) {
  if (char_buffers.empty() || char_buffer_sizes.empty()) {
    return std::vector<xla::PjRtPartialProgramProto>();
  }
  if (char_buffers.size() != char_buffer_sizes.size()) {
    return absl::InvalidArgumentError(
        "char_buffers and char_buffer_sizes must have the same size");
  }

  std::vector<xla::PjRtPartialProgramProto> partial_programs;
  partial_programs.reserve(char_buffers.size());
  for (size_t i = 0; i < char_buffers.size(); ++i) {
    xla::PjRtPartialProgramProto partial_program;
    bool success = partial_program.ParseFromString(
        absl::string_view(char_buffers[i], char_buffer_sizes[i]));
    if (!success) {
      return absl::InvalidArgumentError(
          "Failed to deserialize PjRtPartialProgramProto");
    }
    partial_programs.push_back(partial_program);
  }

  return partial_programs;
}

absl::StatusOr<const char**> ConvertPjRtPartialProgramProtosToCharBuffers(
    const std::vector<xla::PjRtPartialProgramProto>& partial_programs,
    const size_t*& char_buffer_sizes) {
  const size_t num_programs = partial_programs.size();
  const char** char_buffers = new const char*[num_programs]();
  size_t* buffer_sizes = new size_t[num_programs];
  absl::Cleanup cleanup_buffers = [char_buffers, buffer_sizes, num_programs] {
    for (size_t i = 0; i < num_programs; ++i) {
      delete[] char_buffers[i];
    }
    delete[] char_buffers;
    delete[] buffer_sizes;
  };

  for (size_t i = 0; i < num_programs; ++i) {
    const xla::PjRtPartialProgramProto& partial_program = partial_programs[i];
    size_t buffer_size = partial_program.ByteSizeLong();
    char* buffer = new char[buffer_size];
    bool success = partial_program.SerializeToArray(buffer, buffer_size);
    if (!success) {
      return absl::InvalidArgumentError(
          "Failed to serialize PjRtPartialProgramProto");
    }
    char_buffers[i] = buffer;
    buffer_sizes[i] = buffer_size;
  }
  char_buffer_sizes = buffer_sizes;

  std::move(cleanup_buffers).Cancel();
  return char_buffers;
}

}  // namespace xla

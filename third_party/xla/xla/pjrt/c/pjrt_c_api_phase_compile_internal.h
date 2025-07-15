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

#ifndef XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_INTERNAL_H_
#define XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_INTERNAL_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"

namespace pjrt {

// Converts an array of C-style character buffers along with their sizes to a
// vector of C++ strings.
std::vector<std::string> ConvertCharBuffersToCppStrings(
    const char** char_buffers, const size_t* char_buffer_sizes,
    size_t num_strings);

// Converts a vector of C++ strings to an array of C-style character buffers and
// their sizes. This function allocates memory for the character buffers, the
// individual C-strings, and `char_buffer_sizes`.
const char** ConvertCppStringsToCharBuffers(
    const std::vector<std::string>& strings, const size_t*& char_buffer_sizes);

// Converts an array of C-style character buffers along with their sizes to a
// vector of `PjRtPartialProgramProto`s.
absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>>
ConvertCharBuffersToPjRtPartialProgramProtos(const char** char_buffers,
                                             const size_t* char_buffer_sizes,
                                             size_t num_programs);

// Converts a vector of `PjRtPartialProgramProto`s to an array of C-style
// character buffers and their sizes. This function allocates memory for the
// `char_buffers`, the individual serialized protos, and `char_buffer_sizes`.
// In case the partial programs cannot be serialized, this function frees the
// allocated memory.
absl::StatusOr<const char**> ConvertPjRtPartialProgramProtosToCharBuffers(
    const std::vector<xla::PjRtPartialProgramProto>& partial_programs,
    const size_t*& char_buffer_sizes);

// Creates and initializes a PJRT_PhaseCompile_Extension struct. This function
// is used by plugins to create and chain the phase compilation extension
// into the PJRT C API structure.
PJRT_PhaseCompile_Extension CreatePhaseCompileExtension(
    PJRT_Extension_Base* next, PJRT_PhaseCompile_Get_Compiler get_compiler,
    PJRT_PhaseCompile_Destroy_Compiler destroy_compiler);

}  // namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_INTERNAL_H_

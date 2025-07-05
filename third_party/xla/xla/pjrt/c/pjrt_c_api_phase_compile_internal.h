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

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"

namespace pjrt {

// Converts a vector of C++ strings to an array of C-style character buffers.
// This function allocates memory for the `char_buffers`, the
// individual C-strings, and `char_buffer_sizes`. The caller is responsible for
// freeing this memory. This conversion is vital for preparing C++ string data
// to be passed across the PJRT C API boundary, where C structs are used for
// communication.
//
// Args:
//   strings: A `std::vector<std::string>` to be converted.
//   char_buffers: Output parameter. A pointer to an array of C-style character
//                pointers.
//   char_buffer_sizes: Output parameter. A pointer to an array of sizes
//                      corresponding to each C-string in `char_buffers`.
//   num_strings: Output parameter. The number of strings in the `char_buffers`
//                array.
void ConvertCppStringsToCharBuffer(const std::vector<std::string>& strings,
                                   const char*** char_buffers,
                                   const size_t** char_buffer_sizes,
                                   size_t* num_strings);

// Creates and initializes a PJRT_PhaseCompile_Extension struct. This function
// is used by plugins to create and chain the phase compilation extension
// into the PJRT C API structure.
PJRT_PhaseCompile_Extension CreatePhaseCompileExtension(
    PJRT_Extension_Base* next, PJRT_PhaseCompile_Get_Compiler get_compiler,
    PJRT_PhaseCompile_Destroy_Compiler destroy_compiler);

}  // namespace pjrt

// namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_INTERNAL_H_

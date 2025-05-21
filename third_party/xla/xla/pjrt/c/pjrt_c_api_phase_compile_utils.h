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

#ifndef XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_UTILS_H_
#define XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_UTILS_H_

#include <cstddef>
#include <string>
#include <vector>

// Converts an array of C-style character buffers to a vector of C++
// strings. This function is essential for the PJRT C API's pass-through layer,
// where `xla::PjRtPartialProgramProto` data is communicated via portable C
// structs. This function takes ownership of both the `char_buffer` and
// `char_buffer_sizes` memory and frees them, helping to manage memory across
// the C/C++ boundary.
//
// Args:
//   char_buffer: An array of C-style character pointers. Each pointer points to
//   a C-string.
//   char_buffer_sizes: An array of sizes corresponding to each C-string in
//   `char_buffer`. This is used when the C-strings are not null-terminated.
//   num_strings: The number of C-strings in the `char_buffer` array.
// Returns:
//   A `std::vector<std::string>` containing the converted C++ strings.
std::vector<std::string> ConvertCharBufferToCppStrings(
    const char** char_buffer, const size_t* char_buffer_sizes,
    size_t num_strings);

// Converts a vector of C++ strings to an array of C-style character buffers.
// This function allocates memory for the `char_buffer`, the
// individual C-strings, and `char_buffer_sizes` (if `is_null_terminated` is
// false). The caller is responsible for freeing this memory. This conversion is
// vital for preparing C++ string data to be passed across the PJRT C API
// boundary, where C structs are used for communication.
//
// Args:
//   strings: A `std::vector<std::string>` to be converted.
//   char_buffer: Output parameter. A pointer to an array of C-style character
//                pointers. If `is_null_terminated` is true, each pointer points
//                to a null-terminated C-string. Otherwise, each pointer points
//                to a non-null-terminated C-string, and `char_buffer_sizes`
//                contains the corresponding sizes.
//   char_buffer_sizes: Output parameter. A pointer to an array of sizes
//                      corresponding to each C-string in `char_buffer`. This
//                      is populated when `is_null_terminated` is false.
//   num_strings: Output parameter. The number of strings in the `char_buffer`
//                array.
//   is_null_terminated: Optional. A boolean indicating whether the C-strings
//                       should be null-terminated. Defaults to `true`.
void ConvertCppStringsToCharBuffer(const std::vector<std::string>& strings,
                                   const char*** char_buffer,
                                   const size_t** char_buffer_sizes,
                                   size_t* num_strings,
                                   bool is_null_terminated = true);

#endif  // XLA_PJRT_C_PJRT_C_API_PHASE_COMPILE_UTILS_H_

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

#include "xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"

namespace pjrt {

void ConvertCppStringsToCharBuffer(const std::vector<std::string>& strings,
                                   const char*** char_buffers,
                                   const size_t** char_buffer_sizes,
                                   size_t* num_strings) {
  *num_strings = strings.size();
  const char** buffer_pointers = new const char*[*num_strings];
  size_t* buffer_sizes = new size_t[*num_strings];

  for (size_t i = 0; i < *num_strings; ++i) {
    size_t string_data_size = strings[i].size();
    char* string_buffer = new char[string_data_size];
    memcpy(string_buffer, strings[i].data(), string_data_size);

    buffer_pointers[i] = string_buffer;
    buffer_sizes[i] = string_data_size;
  }
  *char_buffers = buffer_pointers;
  *char_buffer_sizes = buffer_sizes;
}

static void PJRT_PhaseCompile_C_Buffers_Destroy(
    PJRT_PhaseCompile_C_Buffers_Destroy_Args* args) {
  assert(args->char_buffers != nullptr);
  assert(args->char_buffer_sizes != nullptr);

  for (size_t i = 0; i < args->num_char_buffers; ++i) {
    delete[] args->char_buffers[i];
  }
  delete[] args->char_buffer_sizes;
  delete[] args->char_buffers;
}

PJRT_PhaseCompile_Extension CreatePhaseCompileExtension(
    PJRT_Extension_Base* next, PJRT_PhaseCompile_Get_Compiler get_compiler,
    PJRT_PhaseCompile_Destroy_Compiler destroy_compiler) {
  return {
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_PhaseCompile_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_PhaseCompile,
          /*next=*/next,
      },
      /*phase_compile_get_compiler=*/get_compiler,
      /*phase_compile_destroy_compiler=*/destroy_compiler,
      /*phase_compile_run_phase=*/nullptr,
      /*phase_compile_get_phase_names=*/nullptr,
      /*phase_compile_c_buffers_destroy=*/PJRT_PhaseCompile_C_Buffers_Destroy,
  };
}
}  // namespace pjrt

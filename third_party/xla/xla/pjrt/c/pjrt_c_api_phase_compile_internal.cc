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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/proto/pjrt_partial_program.pb.h"

namespace pjrt {

std::vector<std::string> ConvertCharBuffersToCppStrings(
    const char** char_buffers, const size_t* char_buffer_sizes,
    size_t num_strings) {
  if (char_buffers == nullptr || char_buffer_sizes == nullptr) {
    return {};
  }

  std::vector<std::string> cpp_strings;
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

static PJRT_Error* PJRT_PhaseCompile_Get_Phase_Names(
    PJRT_PhaseCompile_Get_PhaseNames_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_PhaseCompile_Get_Phase_Names_Args",
      PJRT_PhaseCompile_Get_PhaseNames_Args_STRUCT_SIZE, args->struct_size));

  // Get the phase names from the compiler.
  if (args->phase_compiler == nullptr) {
    return new PJRT_Error{absl::InternalError(
        "PJRT_PhaseCompile_Get_Phase_Names: phase compiler is null")};
  }
  PJRT_ASSIGN_OR_RETURN(absl::StatusOr<std::vector<std::string>> phase_names,
                        args->phase_compiler->compiler->GetPhaseNames());
  if (!phase_names.ok()) {
    return new PJRT_Error{phase_names.status()};
  }

  // Copy the phase names to the output buffer.
  args->phase_names =
      ConvertCppStringsToCharBuffers(*phase_names, args->phase_names_sizes);
  args->num_phase_names = phase_names->size();

  return nullptr;
}

static void PJRT_PhaseCompile_C_Buffers_Destroy(
    PJRT_PhaseCompile_C_Buffers_Destroy_Args* args) {
  if (args == nullptr) {
    return;
  }

  if (args->char_buffers != nullptr) {
    for (size_t i = 0; i < args->num_char_buffers; ++i) {
      delete[] args->char_buffers[i];
    }
  }

  delete[] args->char_buffers;
  delete[] args->char_buffer_sizes;
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
      /*phase_compile_get_phase_names=*/PJRT_PhaseCompile_Get_Phase_Names,
      /*phase_compile_c_buffers_destroy=*/PJRT_PhaseCompile_C_Buffers_Destroy,
  };
}
}  // namespace pjrt

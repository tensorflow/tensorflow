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

#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_internal.h"
#include "xla/pjrt/pjrt_phase_compile_sample_plugin.h"

namespace pjrt {
namespace {

// Test that the phase compile extension has all the callbacks deliberately set
// to null.
TEST(PhaseCompileExtensionTest, TestExtensionRegistration) {
  // Create a phase compile extension.
  PJRT_PhaseCompile_Extension phase_compile_extension =
      pjrt::phase_compile_sample_plugin::CreateSamplePhaseCompileExtension();

  EXPECT_NE(phase_compile_extension.phase_compile_get_compiler, nullptr);
  EXPECT_NE(phase_compile_extension.phase_compile_destroy_compiler, nullptr);
  EXPECT_EQ(phase_compile_extension.phase_compile_run_phases, nullptr);
  EXPECT_EQ(phase_compile_extension.phase_compile_get_phase_names, nullptr);
}

// Test the correct usage of the GetCompiler and DestroyCompiler APIs.
TEST(PhaseCompileExtensionTest,
     TestPhaseCompileExtensionForGetAndDestroyCompiler) {
  // Create a phase compile extension.
  PJRT_PhaseCompile_Extension phase_compile_extension =
      pjrt::phase_compile_sample_plugin::CreateSamplePhaseCompileExtension();

  // Get the phase compiler.
  PJRT_PhaseCompile_Get_Compiler_Args get_compiler_args;
  get_compiler_args.struct_size = sizeof(PJRT_PhaseCompile_Get_Compiler_Args);
  get_compiler_args.extension_start = &phase_compile_extension.base;
  PJRT_Error* error =
      phase_compile_extension.phase_compile_get_compiler(&get_compiler_args);
  ASSERT_EQ(error, nullptr);

  // Destroy the phase compiler.
  PJRT_PhaseCompile_Destroy_Compiler_Args destroy_compiler_args;
  destroy_compiler_args.struct_size =
      sizeof(PJRT_PhaseCompile_Destroy_Compiler_Args);
  destroy_compiler_args.extension_start = &phase_compile_extension.base;
  destroy_compiler_args.phase_compiler = get_compiler_args.phase_compiler;
  phase_compile_extension.phase_compile_destroy_compiler(
      &destroy_compiler_args);
}

TEST(PhaseCompileExtensionTest, TestPhaseCompileExtensionForCBuffersDestroy) {
  // Create a phase compile extension.
  PJRT_PhaseCompile_Extension phase_compile_extension =
      pjrt::phase_compile_sample_plugin::CreateSamplePhaseCompileExtension();

  // Create a C-style buffer.
  std::vector<std::string> strings = {"string1", "string2"};
  const char** char_buffers;
  const size_t* char_buffer_sizes;
  size_t num_strings;
  pjrt::ConvertCppStringsToCharBuffer(strings, &char_buffers,
                                      &char_buffer_sizes, &num_strings);

  // Destroy the C-style buffer.
  PJRT_PhaseCompile_C_Buffers_Destroy_Args destroy_c_buffers_args;
  destroy_c_buffers_args.struct_size =
      sizeof(PJRT_PhaseCompile_C_Buffers_Destroy_Args);
  destroy_c_buffers_args.extension_start = &phase_compile_extension.base;
  destroy_c_buffers_args.char_buffers = char_buffers;
  destroy_c_buffers_args.char_buffer_sizes = char_buffer_sizes;
  destroy_c_buffers_args.num_char_buffers = num_strings;
  phase_compile_extension.phase_compile_c_buffers_destroy(
      &destroy_c_buffers_args);
}

}  // namespace
}  // namespace pjrt

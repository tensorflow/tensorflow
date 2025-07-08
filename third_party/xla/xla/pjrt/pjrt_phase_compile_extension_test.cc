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
  EXPECT_NE(phase_compile_extension.phase_compile_get_phase_names, nullptr);
  EXPECT_EQ(phase_compile_extension.phase_compile_run_phases, nullptr);
}

// Test the correct usage of the (1) GetCompiler, (2) DestroyCompiler, and (3)
// CBuffersDestroy APIs.
TEST(PhaseCompileExtensionTest,
     TestPhaseCompileExtensionForGetCompilerDestroyCompilerAndCBuffersDestroy) {
  // Create a phase compile extension.
  PJRT_PhaseCompile_Extension phase_compile_extension =
      pjrt::phase_compile_sample_plugin::CreateSamplePhaseCompileExtension();

  // Get the phase compiler.
  PJRT_PhaseCompile_Get_Compiler_Args get_compiler_args;
  get_compiler_args.struct_size = sizeof(PJRT_PhaseCompile_Get_Compiler_Args);
  get_compiler_args.extension_start = nullptr;
  PJRT_Error* error =
      phase_compile_extension.phase_compile_get_compiler(&get_compiler_args);
  ASSERT_EQ(error, nullptr);

  // Get the phases names.
  PJRT_PhaseCompile_Get_PhaseNames_Args get_phase_names_args;
  get_phase_names_args.struct_size =
      sizeof(PJRT_PhaseCompile_Get_PhaseNames_Args);
  get_phase_names_args.extension_start = nullptr;
  get_phase_names_args.phase_compiler = get_compiler_args.phase_compiler;
  error = phase_compile_extension.phase_compile_get_phase_names(
      &get_phase_names_args);
  ASSERT_EQ(error, nullptr);

  // Convert the C-style phase names to C++ strings.
  std::vector<std::string> converted_strings =
      pjrt::ConvertCharBuffersToCppStrings(
          get_phase_names_args.phase_names,
          get_phase_names_args.phase_names_sizes,
          get_phase_names_args.num_phase_names);

  // Destroy the C-style buffer.
  PJRT_PhaseCompile_C_Buffers_Destroy_Args destroy_c_buffers_args;
  destroy_c_buffers_args.struct_size =
      sizeof(PJRT_PhaseCompile_C_Buffers_Destroy_Args);
  destroy_c_buffers_args.extension_start = &phase_compile_extension.base;
  destroy_c_buffers_args.char_buffers = get_phase_names_args.phase_names;
  destroy_c_buffers_args.char_buffer_sizes =
      get_phase_names_args.phase_names_sizes;
  destroy_c_buffers_args.num_char_buffers =
      get_phase_names_args.num_phase_names;
  phase_compile_extension.phase_compile_c_buffers_destroy(
      &destroy_c_buffers_args);

  // Destroy the phase compiler.
  PJRT_PhaseCompile_Destroy_Compiler_Args destroy_compiler_args;
  destroy_compiler_args.struct_size =
      sizeof(PJRT_PhaseCompile_Destroy_Compiler_Args);
  destroy_compiler_args.extension_start = nullptr;
  destroy_compiler_args.phase_compiler = get_compiler_args.phase_compiler;
  phase_compile_extension.phase_compile_destroy_compiler(
      &destroy_compiler_args);
}

}  // namespace
}  // namespace pjrt

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

#include "xla/pjrt/pjrt_phase_compile_sample_plugin.h"

#include <gtest/gtest.h>
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"

namespace pjrt {
namespace {

// Test that the phase compile extension has all the callbacks deliberately set
// to null.
TEST(PhaseCompileTest, TestExtensionRegistration) {
  // Create a phase compile extension.
  PJRT_PhaseCompile_Extension phase_compile_extension =
      pjrt::phase_compile_sample_plugin::CreateSamplePhaseCompileExtension();

  EXPECT_EQ(phase_compile_extension.phase_compile_get_compiler, nullptr);
  EXPECT_EQ(phase_compile_extension.phase_compile_destroy_compiler, nullptr);
  EXPECT_EQ(phase_compile_extension.phase_compile_run_phase, nullptr);
  EXPECT_EQ(phase_compile_extension.phase_compile_get_phase_names, nullptr);
}

}  // namespace
}  // namespace pjrt

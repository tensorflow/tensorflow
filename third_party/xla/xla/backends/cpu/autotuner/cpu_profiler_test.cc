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

#include "xla/backends/cpu/autotuner/cpu_profiler.h"

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/executable.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

namespace {

absl::StatusOr<std::unique_ptr<Executable>> CompileHloModule(
    std::unique_ptr<HloModule> hlo_module) {
  CpuCompiler compiler;

  Compiler::CompileOptions compile_options;
  TF_ASSIGN_OR_RETURN(hlo_module, compiler.RunHloPasses(std::move(hlo_module),
                                                        /*stream_exec=*/nullptr,
                                                        compile_options));
  // Run backend.
  return compiler.RunBackend(std::move(hlo_module), /*stream_exec=*/nullptr,
                             compile_options);
}

class CpuProfilerTest : public HloHardwareIndependentTestBase {};

TEST_F(CpuProfilerTest, ProfileWithSharedBuffers) {
  constexpr absl::string_view kHloModule = R"(
        HloModule module
        ENTRY main {
          ROOT c = s32[] constant(1)
        }
      )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));

  std::vector<std::unique_ptr<Executable>> executables;

  TF_ASSERT_OK_AND_ASSIGN(executables.emplace_back(),
                          CompileHloModule(std::move(hlo_module)));

  auto profiler = CpuProfiler::Create(ProfileOptions());
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<ProfileResult> profiles,
      profiler->ProfileWithSharedBuffers(std::move(executables)));

  // We expect only one profile because we only have one executable.
  EXPECT_EQ(profiles.size(), 1);
}

TEST_F(CpuProfilerTest, ProfileWithSharedBuffersWithoutExecutable) {
  auto profiler = CpuProfiler::Create(ProfileOptions());
  TF_ASSERT_OK_AND_ASSIGN(std::vector<ProfileResult> profiles,
                          profiler->ProfileWithSharedBuffers({}));

  // No executable means no profiles.
  EXPECT_EQ(profiles.size(), 0);
}

}  // namespace

}  // namespace xla::cpu

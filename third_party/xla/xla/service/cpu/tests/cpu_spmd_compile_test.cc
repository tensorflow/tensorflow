/* Copyright 2022 The OpenXLA Authors.

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

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include "absl/status/statusor.h"
#include "xla/debug_options_flags.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/test_target_triple_helper.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

class CpuSpmdCompileTest : public CpuCodegenTest {};

TEST_F(CpuSpmdCompileTest, SinglePartition) {
  // Module with "Sharding" custom call and use_spmd_partitioning enabled.
  const char* const hlo_string = R"(
HloModule module

ENTRY entry {
 %parameter.3379 = f32[1,1]{1,0} parameter(0)
 %custom-call.3380 = f32[1,1]{1,0} custom-call(f32[1,1]{1,0} %parameter.3379),
   custom_call_target="Sharding", sharding={replicated}
 ROOT %reshape.6032 = f32[] reshape(f32[1,1]{1,0} %custom-call.3380)
})";

  HloModuleConfig config;
  config.set_use_spmd_partitioning(true);
  auto hlo_module = ParseAndReturnVerifiedModule(hlo_string, config).value();

  // Verify that compilation succeeded.
  absl::StatusOr<std::unique_ptr<Executable>> executable =
      CompileToExecutable(std::move(hlo_module));
  EXPECT_OK(executable.status());
}

}  // namespace
}  // namespace cpu
}  // namespace xla

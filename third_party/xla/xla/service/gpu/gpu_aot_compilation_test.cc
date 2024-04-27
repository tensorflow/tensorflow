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
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

using GpuAotCompilationTest = HloTestBase;

TEST_F(GpuAotCompilationTest, ExportAndLoadExecutable) {
  const absl::string_view hlo_string = R"(
HloModule Test

ENTRY main {
  a = f32[100, 200]{1,0} parameter(0)
  ROOT b = f32[100, 200]{0,1} copy(a)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto compiler = backend().compiler();
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::PlatformManager::PlatformWithName(name));
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                          platform->ExecutorForDevice(0));

  // Compile AOT.
  auto module_group = std::make_unique<HloModuleGroup>(std::move(module));
  AotCompilationOptions aot_options(compiler->PlatformId());
  aot_options.set_executor(stream_exec);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
      compiler->CompileAheadOfTime(std::move(module_group), aot_options));

  // Serialize-deserialize AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                          aot_results[0]->SerializeAsString());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AotCompilationResult> aot_result,
      compiler->LoadAotCompilationResult(serialized_aot_result));

  // Load Executable from AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          aot_result->LoadExecutable(compiler, stream_exec));
}

TEST_F(GpuAotCompilationTest, AotCompilationWithoutGpuDevice) {
  const absl::string_view hlo_string = R"(
HloModule Test

ENTRY main {
  a = f32[100, 200]{1,0} parameter(0)
  ROOT b = f32[100, 200]{0,1} copy(a)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto compiler = backend().compiler();
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::PlatformManager::PlatformWithName(name));
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                          platform->ExecutorForDevice(0));

  auto module_group = std::make_unique<HloModuleGroup>(std::move(module));

  // Stream executor is not passed as an option.
  Compiler::TargetConfig gpu_target_config(stream_exec);
  AotCompilationOptions aot_options(compiler->PlatformId());
  aot_options.set_target_config(gpu_target_config);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
      compiler->CompileAheadOfTime(std::move(module_group), aot_options));

  // Serialize-deserialize AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                          aot_results[0]->SerializeAsString());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AotCompilationResult> aot_result,
      compiler->LoadAotCompilationResult(serialized_aot_result));

  // Load Executable from AOT compilation result.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          aot_result->LoadExecutable(compiler, stream_exec));
}

}  // namespace gpu
}  // namespace xla

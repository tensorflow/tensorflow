/* Copyright 2026 The OpenXLA Authors.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/autotuning/autotuner_cache.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_pjrt_test_base.h"

namespace xla {
namespace gpu {
namespace {

class Gb200CrossCompilationTest : public HloPjRtTestBase {
 public:
  absl::StatusOr<std::unique_ptr<CompiledModule>> CrossCompileTo(
      GpuModel gpu_model) {
    // Clear any existing autotuning results.
    AutotunerCache::ClearAutotuneResults();

    const absl::string_view hlo_string = R"hlo(
      HloModule Test

      ENTRY main {
        a = f16[1,2,2,1] parameter(0)
        b = f16[2,2,1,1] parameter(1)
        ROOT conv = f16[1,1,1,1] convolution(a, b), window={size=2x2}, dim_labels=b01f_01io->b01f
      }
    )hlo";

    ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(hlo_string));

    ASSIGN_OR_RETURN(std::string name,
                     PlatformUtil::CanonicalPlatformName("gpu"));
    name = absl::AsciiStrToUpper(name);
    ASSIGN_OR_RETURN(se::Platform * platform,
                     se::PlatformManager::PlatformWithName(name));
    ASSIGN_OR_RETURN(std::unique_ptr<Compiler> compiler,
                     Compiler::GetForPlatform(platform->id()));
    ASSIGN_OR_RETURN(se::StreamExecutor * stream_exec,
                     platform->ExecutorForDevice(0));

    // Create a different gpu_topology.
    ASSIGN_OR_RETURN(stream_executor::GpuTargetConfigProto target_proto,
                     GetGpuTargetConfig(gpu_model));
    ASSIGN_OR_RETURN(Compiler::GpuTargetConfig gpu_target_config,
                     Compiler::GpuTargetConfig::FromProto(target_proto));

    // Set options to cross compilation but attach the local executor for
    // auto-tuning.
    AotCompilationOptions aot_options(compiler->PlatformId());
    aot_options.set_gpu_topology(
        GetSingleDeviceGpuTopology("", gpu_target_config));
    aot_options.set_executor(stream_exec);

    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_autotune_level(4);
    debug_options.set_xla_gpu_experimental_aot_compiled_thunks(true);
    module->mutable_config().set_debug_options(debug_options);

    ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<CompiledModule>> results,
        compiler->CompileAheadOfTime(std::move(module), aot_options));

    return std::move(results[0]);
  }
};

TEST_F(Gb200CrossCompilationTest, CrossCompilationToB200) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledModule> result,
                       CrossCompileTo(GpuModel::B200));
  // Verify that the auto tuner ran.
  EXPECT_FALSE(AutotunerCache::ResultCacheIsEmpty());
}

TEST_F(Gb200CrossCompilationTest, CrossCompilationToRTX6000PRO) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledModule> result,
                       CrossCompileTo(GpuModel::RTX6000PRO));
  // Verify that the auto tuner ran.
  EXPECT_FALSE(AutotunerCache::ResultCacheIsEmpty());
}

}  // namespace
}  // namespace gpu
}  // namespace xla

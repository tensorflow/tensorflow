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

#include "xla/service/gpu/autotuning/autotuner_pass.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/cublas.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;
namespace se = stream_executor;

se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

class AutotunerPassTest : public HloHardwareIndependentTestBase {
 protected:
  AutotunerPassTest() : stream_executor_(GpuExecutor()) {}

  stream_executor::StreamExecutor* stream_executor_;
  NVPTXCompiler compiler_;
};

TEST_F(AutotunerPassTest, CublasGemmIsAutotuned) {
  const char kCublasCustomCallHlo[] = R"(
    HloModule module, entry_computation_layout={(f32[100,100]{1,0}, f32[100,100]{1,0})->f32[100,100]{1,0}}

    ENTRY %main (arg0: f32[100,100], arg1: f32[100,100]) -> f32[100,100] {
      %arg0 = f32[100,100]{1,0} parameter(0)
      %arg1 = f32[100,100]{1,0} parameter(1)
      %custom-call.1 = (f32[100,100]{1,0}, s8[80000]{0}) custom-call(%arg0, %arg1),
      custom_call_target="__cublas$gemm",
      backend_config={
        "gemm_backend_config":{
          "dot_dimension_numbers":
            {
              "lhs_contracting_dimensions":["1"],
              "rhs_contracting_dimensions":["0"],
              "lhs_batch_dimensions":[],
              "rhs_batch_dimensions":[]
          }
        }
      }
      ROOT %get-tuple-element = f32[100,100]{1,0} get-tuple-element(%custom-call.1), index=0
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "autotuning",
                                      /*num_threads=*/4);
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::make_unique<CublasBackend>(
      stream_executor_, &module->config().debug_options(), &compiler_));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AutotunerPass> pass,
      AutotunerPass::Create(std::move(backends), stream_executor_,
                            &thread_pool));
  EXPECT_THAT(pass->Run(module.get(), /*execution_threads=*/{}),
              IsOkAndHolds(true));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

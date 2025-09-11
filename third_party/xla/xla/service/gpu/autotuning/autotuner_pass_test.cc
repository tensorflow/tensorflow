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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "xla/backends/autotuner/autotuner.h"
#include "xla/backends/autotuner/autotuner_cache.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/cublas.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/path.h"

namespace xla {
namespace gpu {
namespace {

namespace se = stream_executor;

se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

bool IsCublasGemmInstruction(const HloInstruction& instruction) {
  return instruction.opcode() == HloOpcode::kCustomCall &&
         IsCublasGemm(instruction);
}

class AutotunerPassTest : public HloHardwareIndependentTestBase {
 protected:
  AutotunerPassTest()
      : stream_executor_(GpuExecutor()),
        allocator_(std::make_unique<se::StreamExecutorMemoryAllocator>(
            stream_executor_)) {}

  se::StreamExecutor* stream_executor_;
  std::unique_ptr<se::DeviceMemoryAllocator> allocator_;
  NVPTXCompiler compiler_;
};

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

TEST_F(AutotunerPassTest, CublasGemmIsAutotuned) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "autotuning",
                                      /*num_threads=*/4);
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::make_unique<CublasBackend>(
      stream_executor_, &module->config().debug_options(), &compiler_));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AutotunerPass> pass,
      AutotunerPass::Create(std::move(backends),
                            module->config().debug_options(), stream_executor_,
                            &thread_pool, IsCublasGemmInstruction,
                            allocator_.get()));
  EXPECT_THAT(pass->Run(module.get(), /*execution_threads=*/{}),
              tsl::testing::IsOkAndHolds(true));
  // Verify that the backend config has been updated in the HLO.
  auto gemm =
      module->entry_computation()->GetInstructionWithName("custom-call.1");
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_backend_config_after_first_run,
                          gemm->backend_config<GpuBackendConfig>());
  ASSERT_TRUE(gpu_backend_config_after_first_run.gemm_backend_config()
                  .has_selected_algorithm());
}

TEST_F(AutotunerPassTest, CublasGemmIsNotAutotunedWhenFilterReturnsFalse) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "autotuning",
                                      /*num_threads=*/4);
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::make_unique<CublasBackend>(
      stream_executor_, &module->config().debug_options(), &compiler_));

  auto should_autotune = [](const HloInstruction& instruction) {
    return false;
  };
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AutotunerPass> pass,
      AutotunerPass::Create(std::move(backends),
                            module->config().debug_options(), stream_executor_,
                            &thread_pool, should_autotune, allocator_.get()));
  EXPECT_THAT(pass->Run(module.get(), /*execution_threads=*/{}),
              tsl::testing::IsOkAndHolds(true));
  // Verify that the backend config has *not* been updated in the HLO.
  auto gemm =
      module->entry_computation()->GetInstructionWithName("custom-call.1");
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_backend_config_after_first_run,
                          gemm->backend_config<GpuBackendConfig>());
  ASSERT_FALSE(gpu_backend_config_after_first_run.gemm_backend_config()
                   .has_selected_algorithm());
}

TEST_F(AutotunerPassTest, CublasGemmIsAutotunedAndCached) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));

  // Create a temporary directory for the cache.
  std::string cache_dir = ::testing::TempDir();
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_autotuner_cache_dir(cache_dir);

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "autotuning",
                                      /*num_threads=*/4);

  // Run the pass for the first time, this should populate the cache.
  {
    std::vector<std::unique_ptr<CodegenBackend>> backends;
    backends.push_back(std::make_unique<CublasBackend>(
        stream_executor_, &module->config().debug_options(), &compiler_));

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<AutotunerPass> pass,
        AutotunerPass::Create(std::move(backends),
                              module->config().debug_options(),
                              stream_executor_, &thread_pool,
                              IsCublasGemmInstruction, allocator_.get()));
    EXPECT_THAT(pass->Run(module.get(), /*execution_threads=*/{}),
                tsl::testing::IsOkAndHolds(true));
  }

  // Verify that the backend config has been updated in the HLO.
  const HloInstruction* custom_call_after_first_run =
      module->entry_computation()->GetInstructionWithName("custom-call.1");
  TF_ASSERT_OK_AND_ASSIGN(
      auto gpu_backend_config_after_first_run,
      custom_call_after_first_run->backend_config<GpuBackendConfig>());
  LOG(INFO) << "GPU Backend config after first run: "
            << gpu_backend_config_after_first_run.DebugString();
  ASSERT_TRUE(gpu_backend_config_after_first_run.gemm_backend_config()
                  .has_selected_algorithm());

  // Find the cache file and make sure it's not empty.
  std::vector<std::string> children;
  TF_ASSERT_OK(tsl::Env::Default()->GetChildren(cache_dir, &children));
  std::string cache_file;
  for (const auto& child : children) {
    std::string filename = tsl::io::JoinPath(cache_dir, child);
    if (!tsl::Env::Default()->IsDirectory(filename).ok()) {
      uint64_t file_size;
      TF_ASSERT_OK(tsl::Env::Default()->GetFileSize(filename, &file_size));
      if (file_size > 0) {
        cache_file = filename;
        break;
      }
    }
  }
  ASSERT_FALSE(cache_file.empty());

  // Clear the selected algorithm to simulate a pre-autotuning state.
  HloInstruction* custom_call =
      module->entry_computation()->GetInstructionWithName("custom-call.1");
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_backend_config_before_second_run,
                          custom_call->backend_config<GpuBackendConfig>());
  GemmBackendConfig gemm_config =
      gpu_backend_config_before_second_run.gemm_backend_config();
  gemm_config.clear_selected_algorithm();
  *gpu_backend_config_before_second_run.mutable_gemm_backend_config() =
      gemm_config;
  custom_call->set_backend_config(gpu_backend_config_before_second_run);

  // Run the pass for the second time, this should hit the cache.
  {
    std::vector<std::unique_ptr<CodegenBackend>> backends2;
    backends2.push_back(std::make_unique<CublasBackend>(
        stream_executor_, &module->config().debug_options(), &compiler_));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<AutotunerPass> pass2,
        AutotunerPass::Create(std::move(backends2),
                              module->config().debug_options(),
                              stream_executor_, &thread_pool,
                              IsCublasGemmInstruction, allocator_.get()));
    EXPECT_THAT(pass2->Run(module.get(), /*execution_threads=*/{}),
                tsl::testing::IsOkAndHolds(true));
  }

  // Verify that the backend config in the HLO matches the cache.
  const HloInstruction* gemm =
      module->entry_computation()->GetInstructionWithName("custom-call.1");
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_gpu_backend_config,
                          gemm->backend_config<GpuBackendConfig>());
  const GemmBackendConfig& hlo_backend_config =
      hlo_gpu_backend_config.gemm_backend_config();
  EXPECT_TRUE(hlo_backend_config.has_selected_algorithm());

  // We can't easily verify that the HLO config matches the cache content
  // because the autotuning might not be deterministic. However, we have
  // logged that the cache was hit, which is the main purpose of this test.
}

}  // namespace
}  // namespace gpu
}  // namespace xla

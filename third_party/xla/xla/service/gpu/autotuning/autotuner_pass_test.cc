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
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/autotuner/autotuner.h"
#include "xla/backends/autotuner/autotuner_cache.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/autotuner/cublas.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"

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
        allocator_(
            std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
                stream_executor_)) {}

  se::StreamExecutor* stream_executor_;
  std::unique_ptr<se::DeviceAddressAllocator> allocator_;
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
  GpuCompiler::GpuTargetConfig target_config(stream_executor_);
  backends.push_back(std::make_unique<CublasBackend>(
      stream_executor_, &module->config().debug_options(), &compiler_,
      &target_config));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AutotunerPass> pass,
      AutotunerPass::Create(std::move(backends),
                            module->config().debug_options(), stream_executor_,
                            &thread_pool, IsCublasGemmInstruction,
                            &target_config, allocator_.get()));
  EXPECT_THAT(pass->Run(module.get(), /*execution_threads=*/{}),
              absl_testing::IsOkAndHolds(true));
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
  GpuCompiler::GpuTargetConfig target_config(stream_executor_);
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::make_unique<CublasBackend>(
      stream_executor_, &module->config().debug_options(), &compiler_,
      &target_config));

  auto should_autotune = [](const HloInstruction& instruction) {
    return false;
  };
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AutotunerPass> pass,
      AutotunerPass::Create(std::move(backends),
                            module->config().debug_options(), stream_executor_,
                            &thread_pool, should_autotune, &target_config,
                            allocator_.get()));
  EXPECT_THAT(pass->Run(module.get(), /*execution_threads=*/{}),
              absl_testing::IsOkAndHolds(true));
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
      .set_xla_gpu_per_fusion_autotune_cache_dir(cache_dir);

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "autotuning",
                                      /*num_threads=*/4);
  GpuCompiler::GpuTargetConfig target_config(stream_executor_);

  // Run the pass for the first time, this should populate the cache.
  {
    std::vector<std::unique_ptr<CodegenBackend>> backends;
    backends.push_back(std::make_unique<CublasBackend>(
        stream_executor_, &module->config().debug_options(), &compiler_,
        &target_config));

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<AutotunerPass> pass,
        AutotunerPass::Create(
            std::move(backends), module->config().debug_options(),
            stream_executor_, &thread_pool, IsCublasGemmInstruction,
            &target_config, allocator_.get()));
    EXPECT_THAT(pass->Run(module.get(), /*execution_threads=*/{}),
                absl_testing::IsOkAndHolds(true));
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

  // Run the pass on the same original HLO reusing the cache
  // Make sure it hits the cache by setting
  // xla_gpu_require_complete_aot_autotune_results to true.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_2,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));

  module_2->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_per_fusion_autotune_cache_dir(cache_dir);
  module_2->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_require_complete_aot_autotune_results(true);
  {
    std::vector<std::unique_ptr<CodegenBackend>> backends2;
    backends2.push_back(std::make_unique<CublasBackend>(
        stream_executor_, &module_2->config().debug_options(), &compiler_,
        &target_config));

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<AutotunerPass> pass2,
        AutotunerPass::Create(
            std::move(backends2), module_2->config().debug_options(),
            stream_executor_, &thread_pool, IsCublasGemmInstruction,
            &target_config, allocator_.get()));
    EXPECT_THAT(pass2->Run(module_2.get(), /*execution_threads=*/{}),
                absl_testing::IsOkAndHolds(true));
  }

  // Verify that the backend config in the HLO matches the cache.
  const HloInstruction* gemm =
      module_2->entry_computation()->GetInstructionWithName("custom-call.1");
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_gpu_backend_config,
                          gemm->backend_config<GpuBackendConfig>());
  const GemmBackendConfig& hlo_backend_config =
      hlo_gpu_backend_config.gemm_backend_config();
  EXPECT_TRUE(hlo_backend_config.has_selected_algorithm());

  // We can't easily verify that the HLO config matches the cache content
  // because the autotuning might not be deterministic. However, we have
  // logged that the cache was hit, which is the main purpose of this test.
}

TEST_F(AutotunerPassTest, CublasGemmIsAutotunedWithCacheOnly) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));

  std::string cache_dir = ::testing::TempDir();
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_per_fusion_autotune_cache_dir(cache_dir);

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "autotuning",
                                      /*num_threads=*/4);
  GpuCompiler::GpuTargetConfig target_config(stream_executor_);

  // Run the pass for the first time, this should populate the cache.
  {
    std::vector<std::unique_ptr<CodegenBackend>> backends;
    backends.push_back(std::make_unique<CublasBackend>(
        stream_executor_, &module->config().debug_options(), &compiler_,
        &target_config));

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<AutotunerPass> pass,
        AutotunerPass::Create(
            std::move(backends), module->config().debug_options(),
            stream_executor_, &thread_pool, IsCublasGemmInstruction,
            &target_config, allocator_.get()));
    EXPECT_THAT(pass->Run(module.get(), /*execution_threads=*/{}),
                absl_testing::IsOkAndHolds(true));
  }

  // Run the pass on the same original HLO with cache_only=true.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_2,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));

  module_2->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_per_fusion_autotune_cache_dir(cache_dir);

  {
    std::vector<std::unique_ptr<CodegenBackend>> backends2;
    backends2.push_back(std::make_unique<CublasBackend>(
        stream_executor_, &module_2->config().debug_options(), &compiler_,
        &target_config));

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<AutotunerPass> pass2,
        AutotunerPass::Create(
            std::move(backends2), module_2->config().debug_options(),
            /*stream_executor=*/nullptr, &thread_pool, IsCublasGemmInstruction,
            &target_config, /*allocator=*/nullptr));
    EXPECT_THAT(pass2->Run(module_2.get(), /*execution_threads=*/{}),
                absl_testing::IsOkAndHolds(true));
  }

  // Verify that the backend config in the HLO matches the cache.
  const HloInstruction* gemm =
      module_2->entry_computation()->GetInstructionWithName("custom-call.1");
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_gpu_backend_config,
                          gemm->backend_config<GpuBackendConfig>());
  const GemmBackendConfig& hlo_backend_config =
      hlo_gpu_backend_config.gemm_backend_config();
  EXPECT_TRUE(hlo_backend_config.has_selected_algorithm());
}

TEST_F(AutotunerPassTest, DevicelessUsesDefaultConfigIfNoCache) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));

  std::string cache_dir = ::testing::TempDir();
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_per_fusion_autotune_cache_dir(cache_dir);

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "autotuning",
                                      /*num_threads=*/4);
  GpuCompiler::GpuTargetConfig target_config(stream_executor_);

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::make_unique<CublasBackend>(
      stream_executor_, &module->config().debug_options(), &compiler_,
      &target_config));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AutotunerPass> pass,
      AutotunerPass::Create(std::move(backends),
                            module->config().debug_options(),
                            /*stream_executor=*/nullptr, &thread_pool,
                            IsCublasGemmInstruction, &target_config,
                            /*allocator=*/nullptr));
  EXPECT_THAT(pass->Run(module.get(), /*execution_threads=*/{}),
              absl_testing::IsOkAndHolds(true));

  // Verify that the backend config has been updated in the HLO with default
  // config.
  auto gemm =
      module->entry_computation()->GetInstructionWithName("custom-call.1");
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_backend_config,
                          gemm->backend_config<GpuBackendConfig>());
  ASSERT_TRUE(
      gpu_backend_config.gemm_backend_config().has_selected_algorithm());
}

TEST_F(AutotunerPassTest, CublasGemmInNonDefaultStreamIsAutotuned) {
  const char kCublasCustomNonDefaultStreamCallHlo[] = R"""(
HloModule module, entry_computation_layout={(f32[100,100]{1,0}, f32[100,100]{1,0})->f32[100,100]{1,0}}
ENTRY %main (arg0: f32[100,100], arg1: f32[100,100]) -> f32[100,100] {
  %arg0 = f32[100,100]{1,0} parameter(0)
  %arg1 = f32[100,100]{1,0} parameter(1)
  %custom-call.1 = (f32[100,100]{1,0}, s8[80000]{0}) custom-call(%arg0, %arg1),
  custom_call_target="__cublas$gemm",
  backend_config={
    "operation_queue_id":"109",
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
}
)""";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kCublasCustomNonDefaultStreamCallHlo));

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "autotuning",
                                      /*num_threads=*/4);
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  GpuCompiler::GpuTargetConfig target_config(stream_executor_);

  backends.push_back(std::make_unique<CublasBackend>(
      stream_executor_, &module->config().debug_options(), &compiler_,
      &target_config));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AutotunerPass> pass,
      AutotunerPass::Create(std::move(backends),
                            module->config().debug_options(), stream_executor_,
                            &thread_pool, IsCublasGemmInstruction,
                            &target_config, allocator_.get()));
  EXPECT_THAT(pass->Run(module.get(), /*execution_threads=*/{}),
              absl_testing::IsOkAndHolds(true));
}

struct AutotuneLevelParams {
  int autotune_level;
  bool expected_select_first_config;
  bool expected_check_buffers;
  bool expected_should_init_buffers;
};

class AutotunerFlagsTest
    : public AutotunerPassTest,
      public ::testing::WithParamInterface<AutotuneLevelParams> {};

TEST_P(AutotunerFlagsTest, AutotuneLevel) {
  const AutotuneLevelParams& params = GetParam();
  DebugOptions debug_options;
  debug_options.set_xla_gpu_autotune_level(params.autotune_level);

  xla::AutotuneConfig autotune_config = GetAutotuneConfig(debug_options);
  EXPECT_EQ(autotune_config.select_first_config,
            params.expected_select_first_config);
  EXPECT_EQ(autotune_config.check_buffers, params.expected_check_buffers);

  ProfileOptions profile_options =
      GetProfileOptions(debug_options, autotune_config);
  EXPECT_EQ(profile_options.should_init_buffers,
            params.expected_should_init_buffers);
}

INSTANTIATE_TEST_SUITE_P(
    AutotuneLevelTests, AutotunerFlagsTest,
    ::testing::ValuesIn<AutotuneLevelParams>({
        {0, true, false, false},
        {1, false, false, false},
        {2, false, false, false},
        {3, false, false, false},
        {4, false, true, true},
    }),
    [](const ::testing::TestParamInfo<AutotunerFlagsTest::ParamType>& info) {
      return std::to_string(info.param.autotune_level);
    });

struct RegSpillsParams {
  bool filter_kernels_flag;
  bool fail_on_spill_flag;
  bool allow_reg_spills_in;
  bool expected_allow_reg_spills_out;
};

class AutotunerRegSpillsTest
    : public AutotunerPassTest,
      public ::testing::WithParamInterface<RegSpillsParams> {};

TEST_P(AutotunerRegSpillsTest, RegSpills) {
  const RegSpillsParams& params = GetParam();
  DebugOptions debug_options;
  debug_options.set_xla_gpu_fail_ptx_compilation_on_register_spilling(
      params.fail_on_spill_flag);
  debug_options.set_xla_gpu_filter_kernels_spilling_registers_on_autotuning(
      params.filter_kernels_flag);
  EXPECT_EQ(GetAutotuneConfig(debug_options, /*is_deviceless=*/false,
                              /*optimize_scratch_bytes=*/true,
                              params.allow_reg_spills_in)
                .allow_reg_spills,
            params.expected_allow_reg_spills_out);
}

INSTANTIATE_TEST_SUITE_P(
    RegSpillsTests, AutotunerRegSpillsTest,
    ::testing::ValuesIn<RegSpillsParams>({
        {true, true, true, false},
        {true, true, false, false},
        {true, false, true, true},
        {true, false, false, false},
        {false, true, true, false},
        {false, true, false, false},
        {false, false, true, false},
        {false, false, false, false},
    }),
    [](const ::testing::TestParamInfo<AutotunerRegSpillsTest::ParamType>&
           info) {
      return absl::StrCat(info.param.filter_kernels_flag, "_",
                          info.param.fail_on_spill_flag, "_",
                          info.param.allow_reg_spills_in);
    });

TEST_F(AutotunerFlagsTest, DevicelessUsesDefaultConfig) {
  DebugOptions debug_options;
  EXPECT_EQ(GetAutotuneConfig(debug_options, /*is_deviceless=*/true)
                .use_default_config,
            true);
}

TEST_F(AutotunerFlagsTest, DeterministicAutotuningSetsSelectFirstConfig) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_deterministic_ops(true);
  EXPECT_EQ(GetAutotuneConfig(debug_options).select_first_config, true);
  debug_options.set_xla_gpu_deterministic_ops(false);
  debug_options.set_xla_gpu_exclude_nondeterministic_ops(true);
  EXPECT_EQ(GetAutotuneConfig(debug_options).select_first_config, true);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

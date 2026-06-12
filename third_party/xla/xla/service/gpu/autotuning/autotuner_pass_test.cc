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
#include "absl/base/log_severity.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/config_assigner.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/autotuner/cublaslt.h"
#include "xla/backends/gpu/autotuner/cudnn.h"
#include "xla/backends/gpu/autotuner/triton.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/autotuning/autotuner_cache.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

namespace se = stream_executor;

using ::tsl::proto_testing::EqualsProto;

se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
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
  custom_call_target="__cublas$lt$matmul",
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
  backends.push_back(std::make_unique<CublasLtBackend>(
      stream_executor_, &module->config().debug_options(), &compiler_,
      &target_config));

  auto get_backends_fn =
      [backends =
           std::make_shared<std::vector<std::unique_ptr<CodegenBackend>>>(
               std::move(backends))]() mutable { return std::move(*backends); };
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AutotunerPass> pass,
      AutotunerPass::Create(
          std::move(get_backends_fn), module->config().debug_options(),
          target_config.device_description.gpu_compute_capability(),
          stream_executor_, &thread_pool, &target_config,
          /*alias_info=*/nullptr, /*mlir_context=*/nullptr,
          /*shape_size_fn=*/[](const Shape& shape) { return 0; },
          allocator_.get()));
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
    backends.push_back(std::make_unique<CublasLtBackend>(
        stream_executor_, &module->config().debug_options(), &compiler_,
        &target_config));

    auto get_backends_fn =
        [backends =
             std::make_shared<std::vector<std::unique_ptr<CodegenBackend>>>(
                 std::move(backends))]() mutable {
          return std::move(*backends);
        };
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<AutotunerPass> pass,
        AutotunerPass::Create(
            std::move(get_backends_fn), module->config().debug_options(),
            target_config.device_description.gpu_compute_capability(),
            stream_executor_, &thread_pool, &target_config,
            /*alias_info=*/nullptr, /*mlir_context=*/nullptr,
            /*shape_size_fn=*/[](const Shape& shape) { return 0; },
            allocator_.get()));
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
    backends2.push_back(std::make_unique<CublasLtBackend>(
        stream_executor_, &module_2->config().debug_options(), &compiler_,
        &target_config));

    auto get_backends_fn2 =
        [backends2 =
             std::make_shared<std::vector<std::unique_ptr<CodegenBackend>>>(
                 std::move(backends2))]() mutable {
          return std::move(*backends2);
        };
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<AutotunerPass> pass2,
        AutotunerPass::Create(
            std::move(get_backends_fn2), module_2->config().debug_options(),
            target_config.device_description.gpu_compute_capability(),
            stream_executor_, &thread_pool, &target_config,
            /*alias_info=*/nullptr, /*mlir_context=*/nullptr,
            /*shape_size_fn=*/[](const Shape& shape) { return 0; },
            allocator_.get()));
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
    backends.push_back(std::make_unique<CublasLtBackend>(
        stream_executor_, &module->config().debug_options(), &compiler_,
        &target_config));

    auto get_backends_fn =
        [backends =
             std::make_shared<std::vector<std::unique_ptr<CodegenBackend>>>(
                 std::move(backends))]() mutable {
          return std::move(*backends);
        };
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<AutotunerPass> pass,
        AutotunerPass::Create(
            std::move(get_backends_fn), module->config().debug_options(),
            target_config.device_description.gpu_compute_capability(),
            stream_executor_, &thread_pool, &target_config,
            /*alias_info=*/nullptr, /*mlir_context=*/nullptr,
            /*shape_size_fn=*/[](const Shape& shape) { return 0; },
            allocator_.get()));
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
    backends2.push_back(std::make_unique<CublasLtBackend>(
        stream_executor_, &module_2->config().debug_options(), &compiler_,
        &target_config));

    auto get_backends_fn2 =
        [backends2 =
             std::make_shared<std::vector<std::unique_ptr<CodegenBackend>>>(
                 std::move(backends2))]() mutable {
          return std::move(*backends2);
        };
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<AutotunerPass> pass2,
        AutotunerPass::Create(
            std::move(get_backends_fn2), module_2->config().debug_options(),
            target_config.device_description.gpu_compute_capability(),
            /*stream_executor=*/nullptr, &thread_pool, &target_config,
            /*alias_info=*/nullptr, /*mlir_context=*/nullptr,
            /*shape_size_fn=*/[](const Shape& shape) { return 0; },
            /*allocator=*/nullptr));
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
  backends.push_back(std::make_unique<CublasLtBackend>(
      stream_executor_, &module->config().debug_options(), &compiler_,
      &target_config));

  auto get_backends_fn =
      [backends =
           std::make_shared<std::vector<std::unique_ptr<CodegenBackend>>>(
               std::move(backends))]() mutable { return std::move(*backends); };
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AutotunerPass> pass,
      AutotunerPass::Create(
          std::move(get_backends_fn), module->config().debug_options(),
          target_config.device_description.gpu_compute_capability(),
          /*stream_executor=*/nullptr, &thread_pool, &target_config,
          /*alias_info=*/nullptr, /*mlir_context=*/nullptr,
          /*shape_size_fn=*/[](const Shape& shape) { return 0; },
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
  custom_call_target="__cublas$lt$matmul",
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

  backends.push_back(std::make_unique<CublasLtBackend>(
      stream_executor_, &module->config().debug_options(), &compiler_,
      &target_config));

  auto get_backends_fn =
      [backends =
           std::make_shared<std::vector<std::unique_ptr<CodegenBackend>>>(
               std::move(backends))]() mutable { return std::move(*backends); };
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AutotunerPass> pass,
      AutotunerPass::Create(
          std::move(get_backends_fn), module->config().debug_options(),
          target_config.device_description.gpu_compute_capability(),
          stream_executor_, &thread_pool, &target_config,
          /*alias_info=*/nullptr, /*mlir_context=*/nullptr,
          /*shape_size_fn=*/[](const Shape& shape) { return 0; },
          allocator_.get()));
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
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_autotune_level(params.autotune_level);

  xla::ConfigAssigner::Options config_assigner_options =
      GetConfigAssignerOptions(debug_options);
  EXPECT_EQ(config_assigner_options.select_first_config,
            params.expected_select_first_config);
  EXPECT_EQ(config_assigner_options.check_buffers,
            params.expected_check_buffers);

  ProfileOptions profile_options =
      GetProfileOptions(debug_options, config_assigner_options);
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
  bool expected_allow_reg_spills_out;
};

class AutotunerRegSpillsTest
    : public AutotunerPassTest,
      public ::testing::WithParamInterface<RegSpillsParams> {};

TEST_P(AutotunerRegSpillsTest, RegSpills) {
  const RegSpillsParams& params = GetParam();
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_fail_ptx_compilation_on_register_spilling(
      params.fail_on_spill_flag);
  debug_options.set_xla_gpu_filter_kernels_spilling_registers_on_autotuning(
      params.filter_kernels_flag);
  auto config = GetCodegenOrchestratorOptions(debug_options);
  std::unique_ptr<HloInstruction> dummy = HloInstruction::CreateTuple({});
  EXPECT_EQ(config.allow_reg_spills_fn(*dummy),
            params.expected_allow_reg_spills_out);
}

INSTANTIATE_TEST_SUITE_P(
    RegSpillsTests, AutotunerRegSpillsTest,
    ::testing::ValuesIn<RegSpillsParams>({
        {true, true, false},
        {true, false, true},
        {false, true, false},
        {false, false, false},
    }),
    [](const ::testing::TestParamInfo<AutotunerRegSpillsTest::ParamType>&
           info) {
      return absl::StrCat(info.param.filter_kernels_flag, "_",
                          info.param.fail_on_spill_flag);
    });

TEST_F(AutotunerFlagsTest, DevicelessUsesFirstConfig) {
  DebugOptions debug_options = GetDebugOptionsForTest();
  EXPECT_TRUE(GetConfigAssignerOptions(debug_options, /*is_deviceless=*/true)
                  .select_first_config);
}

TEST_F(AutotunerFlagsTest, DeterministicAutotuningSetsSelectFirstConfig) {
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_deterministic_ops(true);
  EXPECT_EQ(GetConfigAssignerOptions(debug_options).select_first_config, true);
  debug_options.set_xla_gpu_deterministic_ops(false);
  debug_options.set_xla_gpu_exclude_nondeterministic_ops(true);
  EXPECT_EQ(GetConfigAssignerOptions(debug_options).select_first_config, true);
}

TEST_F(AutotunerFlagsTest, GetGpuAutotunerBackendsRespectsDeterminism) {
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_exclude_nondeterministic_ops(true);

  GpuCompiler::GpuTargetConfig target_config(stream_executor_);
  GpuAliasInfo alias_info(stream_executor_->GetDeviceDescription());
  mlir::MLIRContext mlir_context;
  RegisterSymbolicExprStorage(&mlir_context);

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<CodegenBackend>> backends,
                       AutotunerPass::GetGpuAutotunerBackends(
                           stream_executor_, allocator_.get(), &target_config,
                           &alias_info, debug_options, &mlir_context,
                           /*shape_size_fn=*/[](const Shape&) { return 0; },
                           &compiler_, stream_executor_->GetPlatform()->id()));

  for (const auto& backend : backends) {
    EXPECT_NE(backend->backend(), autotuner::Backend::TRITON);
    EXPECT_NE(backend->backend(), autotuner::Backend::NATIVE_EMITTER);
    EXPECT_NE(backend->backend(), autotuner::Backend::BLOCK_LEVEL_EMITTER);
  }
}

TEST_F(AutotunerPassTest, CublasLtSelectFirstConfig) {
  absl::SetVLogLevel("config_assigner*", 10);
  AutotunerCache::ClearAutotuneResults();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCublasCustomCallHlo));

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_exclude_nondeterministic_ops(true);

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "autotuning",
                                      /*num_threads=*/4);
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  GpuCompiler::GpuTargetConfig target_config(stream_executor_);

  auto cublaslt_backend = std::make_unique<CublasLtBackend>(
      stream_executor_, &module->config().debug_options(), &compiler_,
      &target_config);

  auto gemm =
      module->entry_computation()->GetInstructionWithName("custom-call.1");
  TF_ASSERT_OK_AND_ASSIGN(auto supported_configs,
                          cublaslt_backend->GetSupportedConfigs(*gemm));
  ASSERT_GT(supported_configs.size(), 1);
  auto expected_config = std::move(supported_configs[0]);

  backends.push_back(std::move(cublaslt_backend));

  auto get_backends_fn =
      [backends =
           std::make_shared<std::vector<std::unique_ptr<CodegenBackend>>>(
               std::move(backends))]() mutable { return std::move(*backends); };
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AutotunerPass> pass,
      AutotunerPass::Create(
          std::move(get_backends_fn), module->config().debug_options(),
          target_config.device_description.gpu_compute_capability(),
          stream_executor_, &thread_pool, &target_config,
          /*alias_info=*/nullptr, /*mlir_context=*/nullptr,
          /*shape_size_fn=*/[](const Shape& shape) { return 0; },
          allocator_.get()));

  absl::ScopedMockLog log;
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_,
                       testing::HasSubstr("Using first compilable config")))
      .Times(testing::AtLeast(1));

  log.StartCapturingLogs();

  EXPECT_THAT(pass->Run(module.get(), /*execution_threads=*/{}),
              absl_testing::IsOkAndHolds(true));

  log.StopCapturingLogs();

  TF_ASSERT_OK_AND_ASSIGN(auto gpu_backend_config_after,
                          gemm->backend_config<GpuBackendConfig>());

  ASSERT_TRUE(expected_config->has_gemm());
  EXPECT_EQ(gpu_backend_config_after.gemm_backend_config().selected_algorithm(),
            expected_config->gemm().algorithm());
}

TEST_F(AutotunerPassTest, TritonSelectFirstConfig) {
  absl::SetVLogLevel("config_assigner*", 10);
  const char kTritonGemmFusionHlo[] = R"hlo(
    HloModule module

    computation {
      p0 = bf16[128,128]{1,0} parameter(0)
      p1 = bf16[128,128]{1,0} parameter(1)
      ROOT dot = bf16[128,128]{1,0} dot(p0, p1),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY main {
      p0 = bf16[128,128]{1,0} parameter(0)
      p1 = bf16[128,128]{1,0} parameter(1)
      ROOT fusion = bf16[128,128]{1,0} fusion(p0, p1),
        kind=kCustom, calls=computation,
        backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
    }
  )hlo";

  AutotunerCache::ClearAutotuneResults();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTritonGemmFusionHlo));

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_exclude_nondeterministic_ops(true);

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "autotuning",
                                      /*num_threads=*/4);
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  GpuCompiler::GpuTargetConfig target_config(stream_executor_);
  target_config.device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{se::CudaComputeCapability::Ampere()});

  GpuAliasInfo alias_info(stream_executor_->GetDeviceDescription());
  mlir::MLIRContext mlir_context;
  RegisterSymbolicExprStorage(&mlir_context);

  auto triton_backend = std::make_unique<TritonBackend>(
      &module->config().debug_options(), &compiler_, &target_config,
      &alias_info, &mlir_context);

  auto fusion = module->entry_computation()->GetInstructionWithName("fusion");
  TF_ASSERT_OK_AND_ASSIGN(auto supported_configs,
                          triton_backend->GetSupportedConfigs(*fusion));
  ASSERT_GT(supported_configs.size(), 1);
  auto expected_config = std::move(supported_configs[0]);

  backends.push_back(std::move(triton_backend));

  auto get_backends_fn =
      [backends =
           std::make_shared<std::vector<std::unique_ptr<CodegenBackend>>>(
               std::move(backends))]() mutable { return std::move(*backends); };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AutotunerPass> pass,
      AutotunerPass::Create(
          std::move(get_backends_fn), module->config().debug_options(),
          target_config.device_description.gpu_compute_capability(),
          stream_executor_, &thread_pool, &target_config, &alias_info,
          &mlir_context,
          /*shape_size_fn=*/[](const Shape& shape) { return 0; },
          allocator_.get()));

  absl::ScopedMockLog log;
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_,
                       testing::HasSubstr("Using first compilable config")))
      .Times(testing::AtLeast(1));

  log.StartCapturingLogs();

  // When determinism is requested we do not provide the config because we do
  // not trust it.
  EXPECT_THAT(pass->Run(module.get(), /*execution_threads=*/{}),
              absl_testing::IsOkAndHolds(true));

  log.StopCapturingLogs();
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_backend_config_after,
                          fusion->backend_config<GpuBackendConfig>());

  ASSERT_TRUE(expected_config->has_triton());
  EXPECT_THAT(
      gpu_backend_config_after.fusion_backend_config().triton_gemm_config(),
      EqualsProto(expected_config->triton()));
}

TEST_F(AutotunerPassTest, CudnnSelectFirstConfig) {
  absl::SetVLogLevel("config_assigner*", 10);
  const char kCudnnConvForwardHlo[] = R"hlo(
    HloModule TestModule

    ENTRY TestComputation {
      input = f16[32,3,3,64] parameter(0)
      filter = f16[3,3,64,128]{2,1,0,3} parameter(1)
      ROOT result = (f16[32,3,3,128], u8[0]) custom-call(input, filter),
                    window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f,
                    custom_call_target="__cudnn$convForward"
    }
  )hlo";

  AutotunerCache::ClearAutotuneResults();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCudnnConvForwardHlo));

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "autotuning",
                                      /*num_threads=*/4);
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  GpuCompiler::GpuTargetConfig target_config(stream_executor_);

  auto cudnn_backend = std::make_unique<CudnnBackend>(
      stream_executor_, &module->config().debug_options(), &compiler_,
      &target_config);

  auto conv = module->entry_computation()->GetInstructionWithName("result");
  TF_ASSERT_OK_AND_ASSIGN(auto supported_configs,
                          cudnn_backend->GetSupportedConfigs(*conv));
  ASSERT_GT(supported_configs.size(), 1);
  auto expected_config = std::move(supported_configs[0]);

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_exclude_nondeterministic_ops(true);

  backends.push_back(std::move(cudnn_backend));

  auto get_backends_fn =
      [backends =
           std::make_shared<std::vector<std::unique_ptr<CodegenBackend>>>(
               std::move(backends))]() mutable { return std::move(*backends); };
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AutotunerPass> pass,
      AutotunerPass::Create(
          std::move(get_backends_fn), module->config().debug_options(),
          target_config.device_description.gpu_compute_capability(),
          stream_executor_, &thread_pool, &target_config,
          /*alias_info=*/nullptr, /*mlir_context=*/nullptr,
          /*shape_size_fn=*/[](const Shape& shape) { return 0; },
          allocator_.get()));

  absl::ScopedMockLog log;
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_,
                       testing::HasSubstr("Using first compilable config")))
      .Times(testing::AtLeast(1));

  log.StartCapturingLogs();

  EXPECT_THAT(pass->Run(module.get(), /*execution_threads=*/{}),
              absl_testing::IsOkAndHolds(true));

  log.StopCapturingLogs();

  HloInstruction* conv_after = nullptr;
  for (auto* instr : module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kCustomCall &&
        instr->custom_call_target() == "__cudnn$convForward") {
      conv_after = instr;
      break;
    }
  }
  ASSERT_NE(conv_after, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(auto gpu_backend_config_after,
                          conv_after->backend_config<GpuBackendConfig>());

  ASSERT_TRUE(expected_config->has_algorithm());
  EXPECT_EQ(gpu_backend_config_after.cudnn_conv_backend_config()
                .algorithm()
                .algo_id(),
            expected_config->algorithm().algo_id());
}

}  // namespace
}  // namespace gpu
}  // namespace xla

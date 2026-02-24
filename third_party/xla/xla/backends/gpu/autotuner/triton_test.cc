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

#include "xla/backends/gpu/autotuner/triton.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"
#include "google/protobuf/text_format.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"

namespace xla {
namespace gpu {
namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;
using TritonBackendConfig = AutotuneResult::TritonGemmKey;
using ::tsl::proto_testing::EqualsProto;

const char kHlo[] = R"(
  HloModule module

  computation {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    convert0 = f32[1024,1024]{1,0} convert(p0)
    p1 = bf16[1024,1024]{1,0} parameter(1)
    convert1 = f32[1024,1024]{1,0} convert(p1)
    ROOT dot = f32[1024,1024]{1,0} dot(convert0, convert1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    p1 = bf16[1024,1024]{1,0} parameter(1)
    ROOT fusion = f32[1024,1024]{1,0} fusion(p0, p1),
      kind=kCustom, calls=computation,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  })";

const char kSimpleGemmFusionHlo[] = R"(
  HloModule module

  computation {
    p0 = f32[1024,1024]{1,0} parameter(0)
    p1 = f32[1024,1024]{1,0} parameter(1)
    ROOT dot = f32[1024,1024]{1,0} dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    p0 = f32[1024,1024]{1,0} parameter(0)
    p1 = f32[1024,1024]{1,0} parameter(1)
    ROOT fusion = f32[1024,1024]{1,0} fusion(p0, p1),
      kind=kCustom, calls=computation,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  })";

const char kScaledDotHlo[] = R"(
HloModule ScaledDotIsFused, entry_computation_layout={(bf16[4,4]{1,0}, bf16[4,4]{1,0}, bf16[1,1]{1,0}, bf16[1,1]{1,0})->bf16[4,4]{1,0}}

%fusion_dot (parameter_0: bf16[4,4], parameter_1: bf16[4,4], parameter_2: bf16[1,1], parameter_3: bf16[1,1]) -> bf16[4,4] {
  %parameter_0 = bf16[4,4]{1,0} parameter(0)
  %parameter_1 = bf16[4,4]{1,0} parameter(1)
  %parameter_2 = bf16[1,1]{1,0} parameter(2)
  %parameter_3 = bf16[1,1]{1,0} parameter(3)
  ROOT %dot.1 = bf16[4,4]{1,0} scaled-dot(%parameter_0, %parameter_1, %parameter_2, %parameter_3), lhs_contracting_dims={1}, rhs_contracting_dims={1}, metadata={op_name="foo"}
}

ENTRY %entry (lhs: bf16[4,4], rhs: bf16[4,4], lhs_scale: bf16[1,1], rhs_scale: bf16[1,1]) -> bf16[4,4] {
  %lhs = bf16[4,4]{1,0} parameter(0)
  %rhs = bf16[4,4]{1,0} parameter(1)
  %lhs_scale = bf16[1,1]{1,0} parameter(2)
  %rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT %fusion = bf16[4,4]{1,0} fusion(%lhs, %rhs, %lhs_scale, %rhs_scale), kind=kCustom, calls=%fusion_dot, metadata={op_name="foo"}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm"},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
})";

class TritonBackendTest : public HloHardwareIndependentTestBase {
 protected:
  TritonBackendTest()
      : platform_(PlatformUtil::GetDefaultPlatform().value()),
        stream_executor_(platform_->ExecutorForDevice(0).value()),
        target_config_(stream_executor_),
        alias_info_(stream_executor_->GetDeviceDescription()),
        compiler_(Compiler::GetForPlatform(platform_->id()).value()),
        backend_(&debug_options_, compiler_.get(), &target_config_,
                 &alias_info_, &mlir_context_) {
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  DebugOptions debug_options_;
  se::Platform* platform_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  GpuAliasInfo alias_info_;
  std::unique_ptr<Compiler> compiler_;
  TritonBackend backend_;
  mlir::MLIRContext mlir_context_;
};

TEST_F(TritonBackendTest, GetSupportedConfigs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOk());
  EXPECT_GT(configs.value().size(), 0);

  if (backend_.target_config()
          .device_description.cuda_compute_capability()
          .IsAtLeastHopper()) {
    // Check that TMA configurations are generated.
    EXPECT_TRUE(std::any_of(configs.value().begin(), configs.value().end(),
                            [](auto& config) {
                              TritonBackendConfig actual_config;
                              if (!config->UnpackTo(&actual_config)) {
                                return false;
                              }
                              return actual_config.is_tma_allowed();
                            }));
  }
}

TEST_F(TritonBackendTest, GetSupportedConfigsForScaledDot) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kScaledDotHlo));
  HloInstruction* fusion_instr =
      module->entry_computation()->root_instruction();
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(*fusion_instr);
  EXPECT_THAT(configs, absl_testing::IsOk());
  EXPECT_GT(configs.value().size(), 0);
}

TEST_F(TritonBackendTest, GetAndApplyConfigForScaledDot) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kScaledDotHlo));
  HloInstruction* fusion_instr =
      module->entry_computation()->root_instruction();
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(*fusion_instr);
  EXPECT_THAT(config, absl_testing::IsOk());
  EXPECT_THAT(backend_.ApplyConfig(*fusion_instr, *config.value()), IsOk());
}

TEST_F(TritonBackendTest, GetSupportedConfigsRestrictedDefaultSearch) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> default_configs =
      backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()));
  debug_options_.set_xla_gpu_exhaustive_tiling_search(true);
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
      exhaustive_configs = backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()));
  EXPECT_THAT(default_configs, IsOk());
  EXPECT_THAT(exhaustive_configs, IsOk());
  EXPECT_GE(exhaustive_configs.value().size(), default_configs.value().size());
}

TEST_F(TritonBackendTest, GetSupportedConfigsForUnsupportedInstruction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* unsupported_instr = module->entry_computation()
                                          ->root_instruction()
                                          ->called_computations()[0]
                                          ->root_instruction();
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(*unsupported_instr);
  EXPECT_THAT(configs, absl_testing::IsOk());
  EXPECT_THAT(configs.value(), testing::IsEmpty());
}

TEST_F(TritonBackendTest, GetDefaultConfig) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction()));

  EXPECT_THAT(config, absl_testing::IsOk());
}

TEST_F(TritonBackendTest, GetDefaultConfigReturnsSplitKOne) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_enable_split_k_autotuning(true);

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction()));

  ASSERT_THAT(config, absl_testing::IsOk());
  TritonBackendConfig triton_config;
  ASSERT_TRUE(config.value()->UnpackTo(&triton_config));
  EXPECT_EQ(triton_config.split_k(), 1);
}

TEST_F(TritonBackendTest, GetDefaultConfigForUnsupportedInstruction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* unsupported_instr = module->entry_computation()
                                          ->root_instruction()
                                          ->called_computations()[0]
                                          ->root_instruction();
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(*unsupported_instr);
  EXPECT_THAT(config.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TritonBackendTest, Compile) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction())));
  absl::StatusOr<std::unique_ptr<Executable>> executable = backend_.Compile(
      *(module->entry_computation()->root_instruction()), *config);
  EXPECT_THAT(executable, absl_testing::IsOk());
}

TEST_F(TritonBackendTest, AmpereUsesMoreThanTwoStages) {
  if (target_config_.device_description.gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }

  se::CudaComputeCapability ampere_cap{se::CudaComputeCapability::kAmpere,
                                       /*minor=*/0};
  target_config_.device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{ampere_cap});

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kSimpleGemmFusionHlo));

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOk());
  EXPECT_GT(configs.value().size(), 0);

  EXPECT_TRUE(std::any_of(configs.value().begin(), configs.value().end(),
                          [](const std::unique_ptr<BackendConfig>& config) {
                            TritonBackendConfig triton_config;
                            if (!config->UnpackTo(&triton_config)) {
                              return false;
                            }
                            return triton_config.num_stages() > 2;
                          }));
}

TEST_F(TritonBackendTest, SmallOutputCanUseLargeSplitK) {
  debug_options_.set_xla_gpu_enable_split_k_autotuning(true);
  se::CudaComputeCapability ampere_cap{se::CudaComputeCapability::kAmpere,
                                       /*minor=*/0};
  target_config_.device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{ampere_cap});
  target_config_.device_description.set_core_count(132);
  target_config_.device_description.set_registers_per_block_limit(64 * 1024);
  target_config_.device_description.set_threads_per_block_limit(1024);
  target_config_.device_description.set_threads_per_warp(32);
  target_config_.device_description.set_shared_memory_per_block_optin(227 *
                                                                      1024);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kSimpleGemmFusionHlo));

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOk());
  EXPECT_GT(configs.value().size(), 0);

  EXPECT_TRUE(std::any_of(configs.value().begin(), configs.value().end(),
                          [](const std::unique_ptr<BackendConfig>& config) {
                            TritonBackendConfig triton_config;
                            if (!config->UnpackTo(&triton_config)) {
                              return false;
                            }
                            return triton_config.split_k() >= 4;
                          }));
}

TEST_F(TritonBackendTest, LargeOutputDoesNotUseLargeSplitK) {
  const char kLargeGemmFusionHlo[] = R"(
  HloModule module

  computation {
    p0 = f32[20480,20480]{1,0} parameter(0)
    p1 = f32[20480,20480]{1,0} parameter(1)
    ROOT dot = f32[20480,20480]{1,0} dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    p0 = f32[20480,20480]{1,0} parameter(0)
    p1 = f32[20480,20480]{1,0} parameter(1)
    ROOT fusion = f32[20480,20480]{1,0} fusion(p0, p1),
      kind=kCustom, calls=computation,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  })";

  se::CudaComputeCapability ampere_cap{se::CudaComputeCapability::kAmpere,
                                       /*minor=*/0};
  target_config_.device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{ampere_cap});

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kLargeGemmFusionHlo));

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOk());

  EXPECT_FALSE(std::any_of(configs.value().begin(), configs.value().end(),
                           [](const std::unique_ptr<BackendConfig>& config) {
                             TritonBackendConfig triton_config;
                             if (!config->UnpackTo(&triton_config)) {
                               return false;
                             }
                             return triton_config.split_k() > 1;
                           }));
}

TEST_F(TritonBackendTest, SplitKIsDisabled) {
  debug_options_.set_xla_gpu_enable_split_k_autotuning(false);

  se::CudaComputeCapability ampere_cap{se::CudaComputeCapability::kAmpere,
                                       /*minor=*/0};
  target_config_.device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{ampere_cap});

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kSimpleGemmFusionHlo));

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOk());

  EXPECT_TRUE(std::all_of(configs.value().begin(), configs.value().end(),
                          [](const std::unique_ptr<BackendConfig>& config) {
                            TritonBackendConfig triton_config;
                            if (!config->UnpackTo(&triton_config)) {
                              return false;
                            }
                            return triton_config.split_k() == 1;
                          }));
}

TEST_F(TritonBackendTest, VerifyHopperConfigsAreDifferentFromBlackwell) {
  if (target_config_.device_description.gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }

  auto get_configs = [&](se::CudaComputeCapability cap)
      -> absl::StatusOr<std::vector<TritonBackendConfig>> {
    target_config_.device_description.set_gpu_compute_capability(
        se::GpuComputeCapability{cap});

    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                        ParseAndReturnVerifiedModule(kSimpleGemmFusionHlo));
    TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<BackendConfig>> configs,
                        backend_.GetSupportedConfigs(*(
                            module->entry_computation()->root_instruction())));

    std::vector<TritonBackendConfig> result;
    for (auto& config : configs) {
      TritonBackendConfig triton_config;
      if (config->UnpackTo(&triton_config)) {
        result.push_back(triton_config);
      }
    }
    return result;
  };

  auto hopper_configs_status = get_configs(
      se::CudaComputeCapability{se::CudaComputeCapability::kHopper, 0});
  ASSERT_THAT(hopper_configs_status, absl_testing::IsOk());
  auto hopper_configs = std::move(hopper_configs_status).value();

  auto blackwell_configs_status = get_configs(
      se::CudaComputeCapability{se::CudaComputeCapability::kBlackwell, 0});
  ASSERT_THAT(blackwell_configs_status, absl_testing::IsOk());
  auto blackwell_configs = std::move(blackwell_configs_status).value();

  auto to_set = [](const std::vector<TritonBackendConfig>& configs) {
    std::set<std::string> s;
    for (const auto& c : configs) {
      s.insert(c.ShortDebugString());
    }
    return s;
  };

  EXPECT_GT(hopper_configs.size(), 0);
  EXPECT_GT(blackwell_configs.size(), 0);
  EXPECT_NE(to_set(hopper_configs), to_set(blackwell_configs));
}

TEST_F(TritonBackendTest, ScaledDotConfigsAreGenerated) {
  if (target_config_.device_description.gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }

  se::CudaComputeCapability blackwell_cap{se::CudaComputeCapability::kBlackwell,
                                          /*minor=*/0};
  target_config_.device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{blackwell_cap});

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kScaledDotHlo));

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOk());
  EXPECT_GT(configs.value().size(), 0);
}

TEST_F(TritonBackendTest, TmaRunCorrectlyForDotsOfBroadcasts) {
  if (target_config_.device_description.gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }

  se::CudaComputeCapability hopper_cap{se::CudaComputeCapability::kHopper, 0};
  target_config_.device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{hopper_cap});

  const char kBroadcastDotHlo[] = R"(
  HloModule module

  computation {
    p0 = f32[64]{0} parameter(0)
    p0b = f32[64,64]{1,0} broadcast(p0), dimensions={0}
    p1 = f32[64,64]{1,0} parameter(1)
    ROOT dot = f32[64,64]{1,0} dot(p0b, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    p0 = f32[64]{0} parameter(0)
    p1 = f32[64,64]{1,0} parameter(1)
    ROOT fusion = f32[64,64]{1,0} fusion(p0, p1),
      kind=kCustom, calls=computation,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kBroadcastDotHlo));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction())));
  absl::StatusOr<std::unique_ptr<Executable>> executable = backend_.Compile(
      *(module->entry_computation()->root_instruction()), *config);
  EXPECT_THAT(executable, absl_testing::IsOk());
}

TEST_F(TritonBackendTest, TmaConfigsAreGeneratedOnlyForHopperAndWorkCorrectly) {
  if (target_config_.device_description.gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kSimpleGemmFusionHlo));

  auto has_tma =
      [](const std::vector<std::unique_ptr<BackendConfig>>& configs) {
        return std::any_of(configs.begin(), configs.end(),
                           [](const std::unique_ptr<BackendConfig>& config) {
                             TritonBackendConfig triton_config;
                             if (!config->UnpackTo(&triton_config)) {
                               return false;
                             }
                             return triton_config.is_tma_allowed();
                           });
      };

  // Ampere
  {
    se::CudaComputeCapability ampere_cap{se::CudaComputeCapability::kAmpere, 0};
    target_config_.device_description.set_gpu_compute_capability(
        se::GpuComputeCapability{ampere_cap});
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::unique_ptr<BackendConfig>> configs,
        backend_.GetSupportedConfigs(
            *(module->entry_computation()->root_instruction())));
    EXPECT_FALSE(has_tma(configs));
  }

  // Hopper
  {
    se::CudaComputeCapability hopper_cap{se::CudaComputeCapability::kHopper, 0};
    target_config_.device_description.set_gpu_compute_capability(
        se::GpuComputeCapability{hopper_cap});
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::unique_ptr<BackendConfig>> configs,
        backend_.GetSupportedConfigs(
            *(module->entry_computation()->root_instruction())));
    EXPECT_TRUE(has_tma(configs));
    std::unique_ptr<BackendConfig> tma_config;
    for (auto& c : configs) {
      TritonBackendConfig tc;
      if (c->UnpackTo(&tc) && tc.is_tma_allowed()) {
        tma_config = std::move(c);
        break;
      }
    }
    ASSERT_NE(tma_config, nullptr);
    EXPECT_THAT(
        backend_.Compile(*(module->entry_computation()->root_instruction()),
                         *tma_config),
        absl_testing::IsOk());
  }
}

TEST_F(TritonBackendTest, GetOverriddenConfigs) {
  AutotuneResult::TritonGemmKey gemm_config;
  gemm_config.set_num_ctas(1);
  gemm_config.set_num_warps(4);
  gemm_config.set_block_m(16);
  gemm_config.set_block_n(16);
  gemm_config.set_block_k(16);
  gemm_config.set_num_stages(2);
  gemm_config.set_is_tma_allowed(true);
  gemm_config.set_is_warp_specialization_allowed(true);
  std::string gemm_config_str;
  ASSERT_TRUE(
      tsl::protobuf::TextFormat::PrintToString(gemm_config, &gemm_config_str));

  debug_options_.set_xla_gpu_override_gemm_autotuner(gemm_config_str);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kSimpleGemmFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()));

  EXPECT_THAT(configs, absl_testing::IsOk());
  EXPECT_EQ(configs.value().size(), 1);
  TritonBackendConfig triton_config;
  ASSERT_TRUE(configs.value()[0]->UnpackTo(&triton_config));
  EXPECT_THAT(triton_config, EqualsProto(gemm_config));
}

TEST_F(TritonBackendTest, GetOverriddenConfigsFromFile) {
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  const std::string file_path =
      tsl::io::JoinPath(temp_dir.path(), "triton_override.txt");
  TritonGemmConfigsProto gemm_configs;
  AutotuneResult::TritonGemmKey* gemm_config = gemm_configs.add_config();
  gemm_config->set_num_ctas(1);
  gemm_config->set_num_warps(4);
  gemm_config->set_block_m(16);
  gemm_config->set_block_n(16);
  gemm_config->set_block_k(16);
  gemm_config->set_num_stages(2);
  gemm_config->set_is_tma_allowed(true);
  gemm_config->set_is_warp_specialization_allowed(true);
  std::string gemm_configs_str;
  ASSERT_TRUE(tsl::protobuf::TextFormat::PrintToString(gemm_configs,
                                                       &gemm_configs_str));
  EXPECT_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), file_path, gemm_configs_str));

  debug_options_.set_xla_gpu_gemm_autotuner_override_file(file_path);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kSimpleGemmFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()));

  EXPECT_THAT(configs, absl_testing::IsOk());
  EXPECT_EQ(configs.value().size(), 1);
  TritonBackendConfig triton_config;
  ASSERT_TRUE(configs.value()[0]->UnpackTo(&triton_config));
  EXPECT_THAT(triton_config, EqualsProto(*gemm_config));
}

TEST_F(TritonBackendTest, WarpSpecializationConfigsAreGenerated) {
  if (target_config_.device_description.gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }

  se::CudaComputeCapability blackwell_cap{se::CudaComputeCapability::kBlackwell,
                                          0};
  target_config_.device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{blackwell_cap});

  debug_options_.set_xla_gpu_experimental_enable_triton_warp_specialization(
      true);
  debug_options_.set_xla_gpu_exhaustive_tiling_search(true);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kSimpleGemmFusionHlo));

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOk());
  EXPECT_GT(configs.value().size(), 0);

  EXPECT_TRUE(
      std::any_of(configs.value().begin(), configs.value().end(),
                  [](const std::unique_ptr<BackendConfig>& config) {
                    TritonBackendConfig triton_config;
                    if (!config->UnpackTo(&triton_config)) {
                      return false;
                    }
                    return triton_config.is_warp_specialization_allowed();
                  }));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

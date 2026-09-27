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

#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

class TestGpuCodegenBackend : public GpuCodegenBackend {
 public:
  TestGpuCodegenBackend(const DebugOptions* debug_options, Compiler* compiler,
                        const Compiler::GpuTargetConfig* target_config)
      : GpuCodegenBackend(autotuner::Backend::TRITON, debug_options, compiler,
                          target_config) {}

  std::string version() const override { return "1.0"; }

  bool IsSupported(const HloInstruction& instr) override { return true; }

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override {
    std::vector<std::unique_ptr<BackendConfig>> configs;
    auto config = std::make_unique<BackendConfig>();
    config->mutable_gemm()->set_algorithm(42);
    configs.push_back(std::move(config));
    return configs;
  }

  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override {
    return absl::OkStatus();
  }
};

class GpuCodegenBackendTest : public ::testing::Test {};

TEST_F(GpuCodegenBackendTest, AdjustDebugOptionsForAutotuning) {
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_enable_dumping(true);
  debug_options.set_xla_gpu_force_compilation_parallelism(4);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  debug_options.set_xla_gpu_async_dot(true);
  debug_options.set_xla_embed_ir_in_executable(true);
  debug_options.set_xla_gpu_kernel_cache_file("foo.txt");
  debug_options.set_xla_gpu_filter_kernels_spilling_registers_on_autotuning(
      true);
  debug_options.set_xla_run_hlo_passes_starting_from("dot-merger");

  GpuCodegenBackend::AdjustDebugOptionsForAutotuning(debug_options);

  EXPECT_FALSE(debug_options.xla_enable_dumping());
  EXPECT_EQ(debug_options.xla_gpu_force_compilation_parallelism(), 1);
  EXPECT_TRUE(debug_options.xla_gpu_enable_command_buffer().empty());
  EXPECT_FALSE(debug_options.xla_gpu_async_dot());
  EXPECT_FALSE(debug_options.xla_embed_ir_in_executable());
  EXPECT_EQ(debug_options.xla_gpu_kernel_cache_file(), "");
  EXPECT_EQ(debug_options.xla_run_hlo_passes_starting_from(), "");
}

TEST_F(GpuCodegenBackendTest, GetSupportedConfigsWithEstimates) {
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  ASSERT_OK_AND_ASSIGN(
      auto target_config,
      GpuTargetConfig::FromProto(stream_executor::GpuTargetConfigProto()));
  TestGpuCodegenBackend backend(&debug_options, nullptr, &target_config);

  auto instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  ASSERT_OK_AND_ASSIGN(std::vector<CodegenBackend::EstimatedConfig> configs,
                       backend.GetSupportedConfigsWithEstimates(*instr));
  ASSERT_EQ(configs.size(), 1);
  EXPECT_EQ(configs[0].config->gemm().algorithm(), 42);
  EXPECT_EQ(configs[0].estimated_runtime, std::nullopt);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

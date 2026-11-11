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
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal_util.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
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

bool HasDynamicSliceConfig(const HloModule& module,
                           absl::string_view instr_name) {
  const HloInstruction* instr =
      module.entry_computation()->GetInstructionWithName(instr_name);
  CHECK(instr != nullptr) << instr_name;
  absl::StatusOr<GpuBackendConfig> config =
      instr->backend_config<GpuBackendConfig>();
  CHECK_OK(config);
  return config->has_dynamic_slice_config();
}

TEST_F(GpuCodegenBackendTest, ClearDynamicSliceConfigs) {
  constexpr absl::string_view kHlo = R"(
    HloModule m

    ENTRY e {
      p0 = bf16[4,128,128] parameter(0)
      p1 = s32[] parameter(1)
      p2 = bf16[1,128,128] parameter(2)
      c0 = s32[] constant(0)
      ds = bf16[1,128,128] dynamic-slice(p0, p1, c0, c0),
        dynamic_slice_sizes={1,128,128},
        backend_config={"dynamic_slice_config":{"loop_index":"0",
          "linear":{"byte_offset":"0","byte_stride":"32768"}}}
      dus = bf16[4,128,128] dynamic-update-slice(p0, p2, p1, c0, c0),
        backend_config={"dynamic_slice_config":{"loop_index":"0",
          "linear":{"byte_offset":"0","byte_stride":"32768"}}}
      ROOT t = (bf16[1,128,128], bf16[4,128,128]) tuple(ds, dus)
    })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));

  ASSERT_OK(GpuCodegenBackend::ClearDynamicSliceConfigs(*module));

  EXPECT_FALSE(HasDynamicSliceConfig(*module, "ds"));
  EXPECT_FALSE(HasDynamicSliceConfig(*module, "dus"));
}

TEST_F(GpuCodegenBackendTest, ClearDynamicSliceConfigsInFusedComputations) {
  constexpr absl::string_view kHlo = R"(
    HloModule m

    fused_computation {
      p0 = bf16[4,128,128] parameter(0)
      p1 = s32[] parameter(1)
      c0 = s32[] constant(0)
      ROOT ds = bf16[1,128,128] dynamic-slice(p0, p1, c0, c0),
        dynamic_slice_sizes={1,128,128},
        backend_config={"dynamic_slice_config":{"loop_index":"0",
          "linear":{"byte_offset":"0","byte_stride":"32768"}}}
    }

    ENTRY e {
      p0 = bf16[4,128,128] parameter(0)
      p1 = s32[] parameter(1)
      ROOT fusion = bf16[1,128,128] fusion(p0, p1), kind=kLoop,
        calls=fused_computation
    })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));

  ASSERT_OK(GpuCodegenBackend::ClearDynamicSliceConfigs(*module));

  const HloInstruction* ds = module->entry_computation()
                                 ->root_instruction()
                                 ->fused_instructions_computation()
                                 ->root_instruction();
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig config,
                       ds->backend_config<GpuBackendConfig>());
  EXPECT_FALSE(config.has_dynamic_slice_config());
}

}  // namespace
}  // namespace gpu
}  // namespace xla

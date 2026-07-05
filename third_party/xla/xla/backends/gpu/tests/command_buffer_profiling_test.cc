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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

class CommandBufferProfilingTest : public HloHardwareIndependentTestBase,
                                   public ::testing::WithParamInterface<bool> {
};

TEST_P(CommandBufferProfilingTest,
       EnableCommandBuffersDuringProfilingIsRespected) {
  bool enable_cb_during_profiling = GetParam();

  constexpr absl::string_view hlo_text = R"hlo(
    HloModule test

    ENTRY main {
      a = f32[2,2] parameter(0)
      b = f32[2,2] parameter(1)
      ROOT dot = f32[2,2] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ASSERT_OK_AND_ASSIGN(
      stream_executor::GpuTargetConfigProto gpu_target_config_proto,
      GetGpuTargetConfig(GpuModel::H100_PCIE));
  ASSERT_OK_AND_ASSIGN(
      gpu::GpuTargetConfig gpu_target_config,
      gpu::GpuTargetConfig::FromProto(gpu_target_config_proto));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Compiler> compiler,
      Compiler::GetForPlatform(stream_executor::cuda::kCudaPlatformId));

  AotCompilationOptions aot_options(compiler->PlatformId());
  aot_options.set_gpu_topology(
      GetSingleDeviceGpuTopology("gpu", gpu_target_config));

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<CompiledModule>> aot_results,
      compiler->CompileAheadOfTime(std::move(hlo_module), aot_options));
  ASSERT_EQ(aot_results.size(), 1);

  DebugOptions runtime_debug_options;
  runtime_debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  runtime_debug_options.set_xla_gpu_graph_min_graph_size(1);
  runtime_debug_options.set_xla_enable_command_buffers_during_profiling(
      enable_cb_during_profiling);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                       std::move(*aot_results[0])
                           .LoadExecutable(compiler->PlatformId(),
                                           gpu_target_config.device_description,
                                           runtime_debug_options));

  GpuExecutable* gpu_exec = dynamic_cast<GpuExecutable*>(executable.get());
  ASSERT_NE(gpu_exec, nullptr);
  const ThunkSequence& thunks = gpu_exec->thunk_executor().thunks();
  ASSERT_EQ(thunks.size(), 1);

  auto* cmd_buf_thunk =
      dynamic_cast<const CommandBufferThunk*>(thunks[0].get());
  ASSERT_NE(cmd_buf_thunk, nullptr);
  EXPECT_EQ(cmd_buf_thunk->IsEnabledDuringProfiling(),
            enable_cb_during_profiling);
}

INSTANTIATE_TEST_SUITE_P(CommandBufferProfilingTestSuite,
                         CommandBufferProfilingTest, ::testing::Bool());

}  // namespace
}  // namespace xla::gpu

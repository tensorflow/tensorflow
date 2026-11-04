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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/multi_process_gpu_test_base.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/literal_test_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using AllToAllMultiProcessE2ETest = MultiProcessGpuTest;

TEST_F(AllToAllMultiProcessE2ETest, OneShotAllToAll2Processes) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       PlatformUtil::GetPlatform("gpu"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(rank()));
  const se::DeviceDescription& desc = executor->GetDeviceDescription();
  if (desc.gpu_compute_capability().IsCuda() && !desc.gpu_compute_capability()
                                                     .cuda_compute_capability()
                                                     ->IsAtLeastAmpere()) {
    GTEST_SKIP() << "One-shot all-to-all requires Ampere+ for CUDA.";
  }

  ASSERT_EQ(client()->addressable_device_count(), 1);
  ASSERT_EQ(client()->device_count(), num_ranks());

  constexpr absl::string_view kModuleStr = R"hlo(
    HloModule test

    ENTRY test_computation {
      param_0 = f32[256] parameter(0)
      ROOT all-to-all = f32[256] all-to-all(param_0), dimensions={0},
        replica_groups={{0,1}}
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnUnverifiedModule(kModuleStr));
  XlaComputation computation(hlo_module->ToProto());

  CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(num_ranks());
  compile_options.executable_build_options.set_num_partitions(1);
  DeviceAssignment device_assignment(num_ranks(), 1);
  for (int i = 0; i < num_ranks(); ++i) {
    device_assignment(i, 0) = i;
  }
  compile_options.executable_build_options.set_device_assignment(
      device_assignment);

  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_autotune_level(0);
  debug_options.add_xla_gpu_experimental_use_collective_kernels(
      DebugOptions::COLLECTIVE_KERNEL_ALL_TO_ALL);
  debug_options.add_xla_gpu_unsupported_use_cross_host_one_shot_kernel(
      DebugOptions::ALLTOALL);
  *compile_options.executable_build_options.mutable_debug_options() =
      debug_options;

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                       client()->CompileAndLoad(computation, compile_options));

  // Inspect the collective kernel strategy selected in the optimized HLO.
  ASSERT_OK_AND_ASSIGN(auto hlo_modules,
                       executable->GetExecutable()->GetHloModules());
  ASSERT_FALSE(hlo_modules.empty());
  const HloModule* optimized_module = hlo_modules.front().get();
  bool found_all_to_all = false;
  for (const HloComputation* comp : optimized_module->computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kAllToAll) {
        found_all_to_all = true;
        ASSERT_OK_AND_ASSIGN(auto gpu_config,
                             instr->backend_config<gpu::GpuBackendConfig>());
        LOG(INFO)
            << "rank_" << rank() << ": AllToAll instruction " << instr->name()
            << " collective kernel strategy: "
            << gpu::CollectiveBackendConfig::CollectiveKernelStrategy_Name(
                   gpu_config.collective_backend_config().kernel_strategy());
        EXPECT_EQ(
            gpu_config.collective_backend_config().kernel_strategy(),
            gpu::CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT);
      }
    }
  }
  EXPECT_TRUE(found_all_to_all)
      << "Expected to find an AllToAll instruction in optimized HLO.";

  // Prepare input literal:
  // Rank 0 provides [1.0f x 128, 2.0f x 128].
  // Rank 1 provides [3.0f x 128, 4.0f x 128].
  std::vector<float> input_data(256);
  for (int i = 0; i < 128; ++i) {
    input_data[i] = (rank() == 0) ? 1.0f : 3.0f;
    input_data[i + 128] = (rank() == 0) ? 2.0f : 4.0f;
  }
  Literal input_literal = LiteralUtil::CreateR1<float>(input_data);

  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client()->addressable_devices()[0]->default_memory_space());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> input_buffer,
      client()->BufferFromHostLiteral(input_literal, memory_space));

  std::vector<std::vector<PjRtBuffer*>> input_ptrs = {{input_buffer.get()}};
  ASSERT_OK_AND_ASSIGN(auto results,
                       executable->Execute(input_ptrs, ExecuteOptions()));

  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);
  Literal result_literal(ShapeUtil::MakeShape(F32, {256}));
  ASSERT_OK(results[0][0]->ToLiteralSync(&result_literal));

  // Expected result:
  // Rank 0: [1.0f x 128, 3.0f x 128]
  // Rank 1: [2.0f x 128, 4.0f x 128]
  std::vector<float> expected_data(256);
  for (int i = 0; i < 128; ++i) {
    expected_data[i] = (rank() == 0) ? 1.0f : 2.0f;
    expected_data[i + 128] = (rank() == 0) ? 3.0f : 4.0f;
  }
  Literal expected_literal = LiteralUtil::CreateR1<float>(expected_data);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, result_literal));
}

}  // namespace
}  // namespace xla

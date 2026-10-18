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

using AllReduceMultiProcessE2ETest = MultiProcessGpuTest;

TEST_F(AllReduceMultiProcessE2ETest, OneShotAllReduce2Processes) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       PlatformUtil::GetPlatform("gpu"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(rank()));
  const se::DeviceDescription& desc = executor->GetDeviceDescription();
  if (desc.gpu_compute_capability().IsCuda() && !desc.gpu_compute_capability()
                                                     .cuda_compute_capability()
                                                     ->IsAtLeastAmpere()) {
    GTEST_SKIP() << "One-shot all-reduce requires Ampere+ for CUDA.";
  }

  ASSERT_EQ(client()->addressable_device_count(), 1);
  ASSERT_EQ(client()->device_count(), num_ranks());

  constexpr absl::string_view kModuleStr = R"hlo(
    HloModule test

    apply_op {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT apply_op = f32[] add(x, y)
    }

    ENTRY test_computation {
      param_0 = f32[128] parameter(0)
      ROOT all-reduce = f32[128] all-reduce(param_0), to_apply=apply_op,
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
  debug_options.add_xla_gpu_unsupported_use_cross_host_one_shot_kernel(
      DebugOptions::ALLREDUCE);
  *compile_options.executable_build_options.mutable_debug_options() =
      debug_options;

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                       client()->CompileAndLoad(computation, compile_options));

  // Inspect the collective kernel strategy selected in the optimized HLO.
  ASSERT_OK_AND_ASSIGN(auto hlo_modules,
                       executable->GetExecutable()->GetHloModules());
  ASSERT_FALSE(hlo_modules.empty());
  const HloModule* optimized_module = hlo_modules.front().get();
  bool found_all_reduce = false;
  for (const HloComputation* comp : optimized_module->computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kAllReduce ||
          instr->opcode() == HloOpcode::kAllReduceStart) {
        found_all_reduce = true;
        ASSERT_OK_AND_ASSIGN(auto gpu_config,
                             instr->backend_config<gpu::GpuBackendConfig>());
        LOG(INFO)
            << "rank_" << rank() << ": AllReduce instruction " << instr->name()
            << " collective kernel strategy: "
            << gpu::CollectiveBackendConfig::CollectiveKernelStrategy_Name(
                   gpu_config.collective_backend_config().kernel_strategy());
        EXPECT_EQ(
            gpu_config.collective_backend_config().kernel_strategy(),
            gpu::CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT);
      }
    }
  }
  EXPECT_TRUE(found_all_reduce)
      << "Expected to find an AllReduce instruction in optimized HLO.";

  // Prepare input literal: rank 0 provides 10.0f, rank 1 provides 20.0f.
  const float input_val = (rank() == 0) ? 10.0f : 20.0f;
  Literal input_literal =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, input_val));

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
  Literal result_literal(ShapeUtil::MakeShape(F32, {128}));
  ASSERT_OK(results[0][0]->ToLiteralSync(&result_literal));

  // Expected result is 10.0f + 20.0f = 30.0f across all ranks.
  Literal expected_literal =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, 30.0f));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, result_literal));
}

}  // namespace
}  // namespace xla

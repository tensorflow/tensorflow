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

#include "xla/backends/gpu/runtime/collective_info_utils.h"

#include <memory>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/all_gather.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/backends/gpu/runtime/all_to_all.h"
#include "xla/backends/gpu/runtime/reduce_scatter.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;

class CollectiveInfoUtilsTest : public HloHardwareIndependentTestBase {
 protected:
  GpuTopology CreateGpuTopology() {
    se::DeviceDescription device_info = TestGpuDeviceInfo::H100SXMDeviceInfo();
    stream_executor::GpuTargetConfigProto target_config_proto;
    *target_config_proto.mutable_gpu_device_info() = device_info.ToProto();
    target_config_proto.mutable_gpu_device_info()
        ->mutable_device_interconnect_info()
        ->set_active_links(18);
    target_config_proto.set_platform_name("CUDA");
    auto target_config_or =
        gpu::GpuTargetConfig::FromProto(target_config_proto);
    CHECK_OK(target_config_or.status());
    return GpuTopology("platform_version", /*num_partitions=*/1,
                       /*num_hosts_per_partition=*/1,
                       /*num_devices_per_host=*/8, *target_config_or);
  }
};

TEST_F(CollectiveInfoUtilsTest, HandlesAllReduce) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  apply_op {
    x = s32[] parameter(0)
    y = s32[] parameter(1)
    ROOT apply_op = s32[] add(x, y)
  }
  ENTRY test_computation {
    param_0 = s32[128] parameter(0)
    ROOT all-reduce = s32[128] all-reduce(param_0), to_apply=apply_op,
        replica_groups={{0,1}}
  }
  )";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kModuleStr, /*replica_count=*/2));
  const HloInstruction* instr =
      FindInstruction(module.get(), HloOpcode::kAllReduce);
  GpuTopology gpu_topology = CreateGpuTopology();

  auto info_or = GetCollectiveInfo(instr, gpu_topology);
  ASSERT_OK(info_or);
  EXPECT_TRUE(std::holds_alternative<AllReduceInfo>(*info_or));
}

TEST_F(CollectiveInfoUtilsTest, HandlesAllReduceStart) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  apply_op {
    x = s32[] parameter(0)
    y = s32[] parameter(1)
    ROOT apply_op = s32[] add(x, y)
  }
  ENTRY test_computation {
    param_0 = s32[128] parameter(0)
    start = s32[128] all-reduce-start(param_0), to_apply=apply_op,
        replica_groups={{0,1}}
    ROOT done = s32[128] all-reduce-done(start)
  }
  )";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kModuleStr, /*replica_count=*/2));
  const HloInstruction* instr =
      FindInstruction(module.get(), HloOpcode::kAllReduceStart);
  GpuTopology gpu_topology = CreateGpuTopology();

  auto info_or = GetCollectiveInfo(instr, gpu_topology);
  ASSERT_OK(info_or);
  EXPECT_TRUE(std::holds_alternative<AllReduceInfo>(*info_or));
}

TEST_F(CollectiveInfoUtilsTest, HandlesAllGather) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = s32[128] parameter(0)
    ROOT all-gather = s32[256] all-gather(param_0),
        dimensions={0}, replica_groups={{0,1}}
  }
  )";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kModuleStr, /*replica_count=*/2));
  const HloInstruction* instr =
      FindInstruction(module.get(), HloOpcode::kAllGather);
  GpuTopology gpu_topology = CreateGpuTopology();

  auto info_or = GetCollectiveInfo(instr, gpu_topology);
  ASSERT_OK(info_or);
  EXPECT_TRUE(std::holds_alternative<AllGatherInfo>(*info_or));
}

TEST_F(CollectiveInfoUtilsTest, HandlesAllToAll) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = s32[128] parameter(0)
    ROOT all-to-all = s32[128] all-to-all(param_0),
        dimensions={0}, replica_groups={{0,1}}
  }
  )";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kModuleStr, /*replica_count=*/2));
  const HloInstruction* instr =
      FindInstruction(module.get(), HloOpcode::kAllToAll);
  GpuTopology gpu_topology = CreateGpuTopology();

  auto info_or = GetCollectiveInfo(instr, gpu_topology);
  ASSERT_OK(info_or);
  EXPECT_TRUE(std::holds_alternative<AllToAllInfo>(*info_or));
}

TEST_F(CollectiveInfoUtilsTest, HandlesReduceScatter) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }
  ENTRY test_computation {
    param_0 = f32[256] parameter(0)
    ROOT rs = f32[128] reduce-scatter(param_0),
        dimensions={0}, replica_groups={{0,1}}, to_apply=apply_op
  }
  )";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kModuleStr, /*replica_count=*/2));
  const HloInstruction* instr =
      FindInstruction(module.get(), HloOpcode::kReduceScatter);
  GpuTopology gpu_topology = CreateGpuTopology();

  auto info_or = GetCollectiveInfo(instr, gpu_topology);
  ASSERT_OK(info_or);
  EXPECT_TRUE(std::holds_alternative<ReduceScatterInfo>(*info_or));
}

TEST_F(CollectiveInfoUtilsTest, RejectsUnsupportedInstruction) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = s32[128] parameter(0)
    ROOT add = s32[128] add(param_0, param_0)
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr));
  const HloInstruction* instr = FindInstruction(module.get(), HloOpcode::kAdd);
  GpuTopology gpu_topology = CreateGpuTopology();

  EXPECT_THAT(GetCollectiveInfo(instr, gpu_topology),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla::gpu

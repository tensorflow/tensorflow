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

#include "xla/backends/gpu/transforms/collectives/collective_kernel_strategy_annotator.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"

namespace xla {
namespace gpu {
namespace {

// Template for a module with an AllReduceStart of the given number of elements.
// Uses 8 replicas so num_devices = 8 (a power of 2, as required by the Triton
// collective kernel).  Reduction is F32 SUM which is supported.
constexpr absl::string_view kAllReduceHloTemplate = R"(
  HloModule all_reduce_test

  add {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT r = f32[] add(p0, p1)
  }

  ENTRY e {
    p0 = f32[%d] parameter(0)
    ar-start = f32[%d] all-reduce-start(p0),
        replica_groups={{0,1,2,3,4,5,6,7}},
        to_apply=add
    ROOT ar-done = f32[%d] all-reduce-done(ar-start)
  }
)";

class CollectiveKernelStrategyAnnotatorTest
    : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    device_info_ = TestGpuDeviceInfo::H100SXMDeviceInfo();
    stream_executor::GpuTargetConfigProto target_config_proto;
    *target_config_proto.mutable_gpu_device_info() = device_info_.ToProto();
    target_config_proto.mutable_gpu_device_info()
        ->mutable_device_interconnect_info()
        ->set_active_links(1);
    target_config_proto.set_platform_name("CUDA");
    ASSERT_OK_AND_ASSIGN(gpu::GpuTargetConfig target_config,
                         gpu::GpuTargetConfig::FromProto(target_config_proto));
    gpu_topology_ = std::make_unique<GpuTopology>(
        "platform_version", /*num_partitions=*/1,
        /*num_hosts_per_partition=*/1,
        /*num_devices_per_host=*/16, target_config);
  }

  absl::StatusOr<CollectiveBackendConfig::CollectiveKernelStrategy>
  GetKernelStrategy(HloModule* module) {
    for (HloComputation* comp : module->computations()) {
      for (HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() == HloOpcode::kAllReduceStart) {
          ASSIGN_OR_RETURN(GpuBackendConfig cfg,
                           instr->backend_config<GpuBackendConfig>());
          return cfg.collective_backend_config().kernel_strategy();
        }
      }
    }
    return CollectiveBackendConfig::KERNEL_STRATEGY_DEFAULT;
  }

  se::DeviceDescription device_info_;
  std::unique_ptr<GpuTopology> gpu_topology_;
};

// 32768 F32 elements = 128 KB ≤ 256 KB → kOneShot →
// KERNEL_STRATEGY_TRITON_ONE_SHOT
TEST_F(CollectiveKernelStrategyAnnotatorTest,
       SmallAllReduceIsAnnotatedOneShot) {
  constexpr int64_t kNumElements = 32768;  // 128 KB
  std::string hlo = absl::StrFormat(kAllReduceHloTemplate, kNumElements,
                                    kNumElements, kNumElements);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo, /*replica_count=*/8));

  CollectiveKernelStrategyAnnotator annotator(*gpu_topology_,
                                              /*is_multimem_enabled=*/false);
  ASSERT_OK(annotator.Run(module.get()).status());

  ASSERT_OK_AND_ASSIGN(auto strategy_one, GetKernelStrategy(module.get()));
  EXPECT_EQ(strategy_one,
            CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT);
}

// 262144 F32 elements = 1 MB (256 KB < 1 MB ≤ 4 MB) → kTwoShot →
// KERNEL_STRATEGY_TRITON_TWO_SHOT
TEST_F(CollectiveKernelStrategyAnnotatorTest,
       MediumAllReduceIsAnnotatedTwoShot) {
  constexpr int64_t kNumElements = 262144;  // 1 MB
  std::string hlo = absl::StrFormat(kAllReduceHloTemplate, kNumElements,
                                    kNumElements, kNumElements);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo, /*replica_count=*/8));

  CollectiveKernelStrategyAnnotator annotator(*gpu_topology_,
                                              /*is_multimem_enabled=*/false);
  ASSERT_OK(annotator.Run(module.get()).status());

  ASSERT_OK_AND_ASSIGN(auto strategy_two, GetKernelStrategy(module.get()));
  EXPECT_EQ(strategy_two,
            CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_TWO_SHOT);
}

// 2097152 F32 elements = 8 MB > 4 MB → custom kernel not supported →
// KERNEL_STRATEGY_DEFAULT (falls back to NCCL).
TEST_F(CollectiveKernelStrategyAnnotatorTest,
       LargeAllReduceKeepsDefaultStrategy) {
  constexpr int64_t kNumElements = 2097152;  // 8 MB
  std::string hlo = absl::StrFormat(kAllReduceHloTemplate, kNumElements,
                                    kNumElements, kNumElements);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo, /*replica_count=*/8));

  CollectiveKernelStrategyAnnotator annotator(*gpu_topology_,
                                              /*is_multimem_enabled=*/false);
  ASSERT_OK(annotator.Run(module.get()).status());

  ASSERT_OK_AND_ASSIGN(auto strategy_default, GetKernelStrategy(module.get()));
  EXPECT_EQ(strategy_default, CollectiveBackendConfig::KERNEL_STRATEGY_DEFAULT);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

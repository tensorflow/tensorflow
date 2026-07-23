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

// Template for a module with an AllReduce of the given number of elements.
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
    p0 = f32[%1$d] parameter(0)
    ROOT all-reduce = f32[%1$d] all-reduce(p0),
        replica_groups={{0,1,2,3,4,5,6,7}},
        to_apply=add
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

  // Builds a GpuTopology suitable for AllGather tests with `num_replicas`
  // devices all on a single host (making the collective local).
  absl::StatusOr<std::unique_ptr<GpuTopology>> MakeLocalGpuTopology(
      int num_replicas) {
    se::DeviceDescription device_info = TestGpuDeviceInfo::H100SXMDeviceInfo();
    stream_executor::GpuTargetConfigProto target_config_proto;
    *target_config_proto.mutable_gpu_device_info() = device_info.ToProto();
    target_config_proto.mutable_gpu_device_info()
        ->mutable_device_interconnect_info()
        ->set_active_links(1);
    target_config_proto.set_platform_name("CUDA");
    ASSIGN_OR_RETURN(gpu::GpuTargetConfig target_config,
                     gpu::GpuTargetConfig::FromProto(target_config_proto));
    return std::make_unique<GpuTopology>(
        "platform_version", /*num_partitions=*/1,
        /*num_hosts_per_partition=*/1,
        /*num_devices_per_host=*/num_replicas, target_config);
  }

  absl::StatusOr<CollectiveBackendConfig::CollectiveKernelStrategy>
  GetKernelStrategy(HloModule* module, HloOpcode opcode) {
    for (HloComputation* comp : module->computations()) {
      for (HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() == opcode) {
          ASSIGN_OR_RETURN(GpuBackendConfig cfg,
                           instr->backend_config<GpuBackendConfig>());
          return cfg.collective_backend_config().kernel_strategy();
        }
      }
    }
    return CollectiveBackendConfig::KERNEL_STRATEGY_DEFAULT;
  }

  // Overload for backward compatibility with existing AllReduce tests.
  absl::StatusOr<CollectiveBackendConfig::CollectiveKernelStrategy>
  GetKernelStrategy(HloModule* module) {
    return GetKernelStrategy(module, HloOpcode::kAllReduce);
  }

  se::DeviceDescription device_info_;
  std::unique_ptr<GpuTopology> gpu_topology_;
};

// 32768 F32 elements = 128 KB ≤ 256 KB → kOneShot →
// KERNEL_STRATEGY_TRITON_ONE_SHOT
TEST_F(CollectiveKernelStrategyAnnotatorTest,
       SmallAllReduceIsAnnotatedOneShot) {
  constexpr int64_t kNumElements = 32768;  // 128 KB
  std::string hlo = absl::StrFormat(kAllReduceHloTemplate, kNumElements);
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
  std::string hlo = absl::StrFormat(kAllReduceHloTemplate, kNumElements);
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
  std::string hlo = absl::StrFormat(kAllReduceHloTemplate, kNumElements);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo, /*replica_count=*/8));

  CollectiveKernelStrategyAnnotator annotator(*gpu_topology_,
                                              /*is_multimem_enabled=*/false);
  ASSERT_OK(annotator.Run(module.get()).status());

  ASSERT_OK_AND_ASSIGN(auto strategy_default, GetKernelStrategy(module.get()));
  EXPECT_EQ(strategy_default, CollectiveBackendConfig::KERNEL_STRATEGY_DEFAULT);
}

// ---- AllGather annotator tests -------------------------------------------

// HLO template for AllGather: input is f32[num_elements], output is
// f32[num_elements * num_replicas] gathered along dimension 0.
constexpr absl::string_view kAllGatherHloTemplate = R"(
  HloModule all_gather_test

  ENTRY e {
    p0 = f32[%1$d] parameter(0)
    ROOT all-gather = f32[%2$d] all-gather(p0),
        dimensions={0},
        replica_groups={{%3$s}}
  }
)";

// 4096 F32 elements per replica (16 KB) with 8 replicas → output 32768
// elements. 4096 * 4 = 16384 bytes, aligned to 128 bits (16 bytes) → eligible →
// KERNEL_STRATEGY_TRITON_ONE_SHOT.
TEST_F(CollectiveKernelStrategyAnnotatorTest,
       EligibleAllGatherIsAnnotatedOneShot) {
  constexpr int kNumReplicas = 8;
  constexpr int64_t kInputElements = 4096;
  constexpr int64_t kOutputElements = kInputElements * kNumReplicas;
  std::string replica_groups_str = "0,1,2,3,4,5,6,7";
  std::string hlo = absl::StrFormat(kAllGatherHloTemplate, kInputElements,
                                    kOutputElements, replica_groups_str);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo, kNumReplicas));
  module->mutable_config()
      .mutable_debug_options()
      .add_xla_gpu_experimental_use_collective_kernels(
          DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER);
  ASSERT_OK_AND_ASSIGN(auto local_topology, MakeLocalGpuTopology(kNumReplicas));

  CollectiveKernelStrategyAnnotator annotator(*local_topology,
                                              /*is_multimem_enabled=*/false);
  ASSERT_OK(annotator.Run(module.get()).status());

  ASSERT_OK_AND_ASSIGN(auto strategy,
                       GetKernelStrategy(module.get(), HloOpcode::kAllGather));
  EXPECT_EQ(strategy, CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT);
}

// 3 F32 elements per replica: 3 * 32 bits = 96 bits, not aligned to 128 bits
// → ineligible → KERNEL_STRATEGY_DEFAULT (falls back to NCCL).
TEST_F(CollectiveKernelStrategyAnnotatorTest,
       IneligibleAllGatherKeepsDefaultStrategy) {
  constexpr int kNumReplicas = 8;
  // 3 F32 elements → 3 * 4 = 12 bytes, not aligned to 16 bytes → ineligible.
  constexpr int64_t kInputElements = 3;
  constexpr int64_t kOutputElements = kInputElements * kNumReplicas;
  std::string replica_groups_str = "0,1,2,3,4,5,6,7";
  std::string hlo = absl::StrFormat(kAllGatherHloTemplate, kInputElements,
                                    kOutputElements, replica_groups_str);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo, kNumReplicas));
  // Flag is set so that the ineligibility comes from shape (not from the flag).
  module->mutable_config()
      .mutable_debug_options()
      .add_xla_gpu_experimental_use_collective_kernels(
          DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER);
  ASSERT_OK_AND_ASSIGN(auto local_topology, MakeLocalGpuTopology(kNumReplicas));

  CollectiveKernelStrategyAnnotator annotator(*local_topology,
                                              /*is_multimem_enabled=*/false);
  ASSERT_OK(annotator.Run(module.get()).status());

  ASSERT_OK_AND_ASSIGN(auto strategy,
                       GetKernelStrategy(module.get(), HloOpcode::kAllGather));
  EXPECT_EQ(strategy, CollectiveBackendConfig::KERNEL_STRATEGY_DEFAULT);
}

// Module with both AllReduce and AllGather: both should be annotated in a
// single pass.
TEST_F(CollectiveKernelStrategyAnnotatorTest,
       BothAllReduceAndAllGatherAnnotatedInSameModule) {
  constexpr int kNumReplicas = 8;
  // Use 32768 elements for AllReduce (eligible one-shot), 4096 for AllGather.
  constexpr absl::string_view kCombinedHlo = R"(
    HloModule combined_test

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT r = f32[] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[32768] parameter(0)
      p1 = f32[4096] parameter(1)
      all-reduce = f32[32768] all-reduce(p0),
          replica_groups={{0,1,2,3,4,5,6,7}},
          to_apply=add
      all-gather = f32[32768] all-gather(p1),
          dimensions={0},
          replica_groups={{0,1,2,3,4,5,6,7}}
      ROOT t = (f32[32768], f32[32768]) tuple(all-reduce, all-gather)
    }
  )";

  ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kCombinedHlo, kNumReplicas));
  module->mutable_config()
      .mutable_debug_options()
      .add_xla_gpu_experimental_use_collective_kernels(
          DebugOptions::COLLECTIVE_KERNEL_ALL_REDUCE);
  module->mutable_config()
      .mutable_debug_options()
      .add_xla_gpu_experimental_use_collective_kernels(
          DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER);
  ASSERT_OK_AND_ASSIGN(auto local_topology, MakeLocalGpuTopology(kNumReplicas));

  CollectiveKernelStrategyAnnotator annotator(*local_topology,
                                              /*is_multimem_enabled=*/false);
  ASSERT_OK(annotator.Run(module.get()).status());

  ASSERT_OK_AND_ASSIGN(auto ar_strategy,
                       GetKernelStrategy(module.get(), HloOpcode::kAllReduce));
  EXPECT_EQ(ar_strategy,
            CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT);

  ASSERT_OK_AND_ASSIGN(auto ag_strategy,
                       GetKernelStrategy(module.get(), HloOpcode::kAllGather));
  EXPECT_EQ(ag_strategy,
            CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

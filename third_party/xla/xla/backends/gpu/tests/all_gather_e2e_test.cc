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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"

namespace xla {
namespace {

class AllGatherTest : public CollectiveOpsWithFlagsBase {
 public:
  AllGatherTest()
      : CollectiveOpsWithFlagsBase(
            /*enable_async=*/true,
            /*enable_p2p_memcpy=*/false,
            /*enable_symmetric_buffer=*/true,
            /*memory_size=*/32 * kMB,
            /*collectives_memory_size=*/32 * kMB) {}

 protected:
  void SetUp() override {
    CollectiveOpsE2ETestBase::SetUp();
    if (Capability().IsCuda() && !IsHopperAndHigher()) {
      GTEST_SKIP() << "Test requires Hopper or newer architecture.";
    }
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts = CollectiveOpsWithFlagsBase::GetDebugOptionsForTest();
    opts.clear_xla_gpu_experimental_use_collective_kernels();
    opts.add_xla_gpu_experimental_use_collective_kernels(
        DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER);
    return opts;
  }

  bool CheckDeviceCount(int32_t required_device_count) {
    [&]() -> void {
      const int32_t current_device_count = device_count();
      if (current_device_count < required_device_count) {
        ASSERT_GE(current_device_count, 2)
            << "Test requires at least 2 devices but only "
            << current_device_count << " available";
        if (current_device_count < required_device_count) {
          GTEST_SKIP() << "Test requires at least " << required_device_count
                       << " devices but only " << current_device_count
                       << " available.";
        }
      }
    }();
    return !IsSkipped() && !HasFatalFailure();
  }

  void VerifyOneShotAllGather(const HloModule* optimized_module) {
    ASSERT_NE(optimized_module, nullptr);
    bool found_all_gather = false;
    for (const HloComputation* comp : optimized_module->computations()) {
      for (const HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() == HloOpcode::kAllGather ||
            instr->opcode() == HloOpcode::kAllGatherStart) {
          found_all_gather = true;
          ASSERT_OK_AND_ASSIGN(gpu::GpuBackendConfig gpu_config,
                               instr->backend_config<gpu::GpuBackendConfig>());
          EXPECT_EQ(
              gpu_config.collective_backend_config().kernel_strategy(),
              gpu::CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT)
              << "Expected AllGather instruction " << instr->name()
              << " to use KERNEL_STRATEGY_TRITON_ONE_SHOT, but got: "
              << gpu::CollectiveBackendConfig::CollectiveKernelStrategy_Name(
                     gpu_config.collective_backend_config().kernel_strategy());
        }
      }
    }
    EXPECT_TRUE(found_all_gather)
        << "Expected to find an AllGather instruction in optimized HLO.";
  }
};

// Basic 2-GPU all-gather of f32[128] -> f32[256].
TEST_F(AllGatherTest, Basic2GpuF32) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[128] parameter(0)
    ROOT all-gather = f32[256] all-gather(param_0), dimensions={0},
      replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  // Create input: rank 0 gets [1, 1, ...], rank 1 gets [2, 2, ...].
  Literal input_r0 =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, 1.0f));
  Literal input_r1 =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, 2.0f));

  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllGather(result.optimized_module);

  ASSERT_EQ(result.results.size(), kNumReplicas);

  // Expected output: [1, 1, ..., 2, 2, ...] (128 ones followed by 128 twos).
  std::vector<float> expected_data;
  expected_data.reserve(256);
  for (int i = 0; i < 128; ++i) {
    expected_data.push_back(1.0f);
  }
  for (int i = 0; i < 128; ++i) {
    expected_data.push_back(2.0f);
  }
  Literal expected = LiteralUtil::CreateR1<float>(expected_data);

  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result.results[i]))
        << "Mismatch at replica " << i;
  }
}

// Larger 2-GPU all-gather to test multi-tile behavior: f32[4096] -> f32[8192].
TEST_F(AllGatherTest, Large2GpuF32) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[4096] parameter(0)
    ROOT all-gather = f32[8192] all-gather(param_0), dimensions={0},
      replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  // rank 0: incrementing values [0, 1, 2, ..., 4095]
  // rank 1: incrementing values [4096, 4097, ..., 8191]
  std::vector<float> data_r0(4096), data_r1(4096);
  for (int i = 0; i < 4096; ++i) {
    data_r0[i] = static_cast<float>(i);
    data_r1[i] = static_cast<float>(i + 4096);
  }
  Literal input_r0 = LiteralUtil::CreateR1<float>(data_r0);
  Literal input_r1 = LiteralUtil::CreateR1<float>(data_r1);

  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllGather(result.optimized_module);

  ASSERT_EQ(result.results.size(), kNumReplicas);

  // Expected: [0, 1, ..., 8191] for both replicas.
  std::vector<float> expected_data(8192);
  for (int i = 0; i < 8192; ++i) {
    expected_data[i] = static_cast<float>(i);
  }
  Literal expected = LiteralUtil::CreateR1<float>(expected_data);

  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result.results[i]))
        << "Mismatch at replica " << i;
  }
}

// 2D shape: f32[16, 32] -> f32[32, 32] (gather along dim 0).
TEST_F(AllGatherTest, TwoDimensional2Gpu) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[16,32] parameter(0)
    ROOT all-gather = f32[32,32] all-gather(param_0), dimensions={0},
      replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  // rank 0: all 1s, rank 1: all 2s
  Literal input_r0 = LiteralUtil::CreateFull<float>({16, 32}, 1.0f);
  Literal input_r1 = LiteralUtil::CreateFull<float>({16, 32}, 2.0f);

  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllGather(result.optimized_module);

  ASSERT_EQ(result.results.size(), kNumReplicas);

  // Expected: first 16 rows are 1s, next 16 rows are 2s.
  // Build expected by checking individual elements.
  Literal expected = LiteralUtil::CreateFull<float>({32, 32}, 0.0f);
  for (int64_t row = 0; row < 32; ++row) {
    float val = (row < 16) ? 1.0f : 2.0f;
    for (int64_t col = 0; col < 32; ++col) {
      expected.Set<float>({row, col}, val);
    }
  }

  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result.results[i]))
        << "Mismatch at replica " << i;
  }
}

}  // namespace
}  // namespace xla

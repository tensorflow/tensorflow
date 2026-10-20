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

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tests/literal_test_util.h"
#include "xla/types.h"
#include "xla/xla.pb.h"

namespace xla {
namespace {

class ReduceScatterTest : public CollectiveOpsWithFlagsBase {
 public:
  ReduceScatterTest()
      : CollectiveOpsWithFlagsBase(
            /*enable_async=*/true,
            /*enable_p2p_memcpy=*/false,
            /*enable_symmetric_buffer=*/true,
            /*memory_size=*/32 * kMB,
            /*collectives_memory_size=*/32 * kMB) {}

 protected:
  void SetUp() override {
    CollectiveOpsE2ETestBase::SetUp();
    if (!Capability().IsCuda() || !IsHopperAndHigher()) {
      GTEST_SKIP() << "Test requires Hopper or newer architecture.";
    }
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts = CollectiveOpsWithFlagsBase::GetDebugOptionsForTest();
    opts.clear_xla_gpu_experimental_use_collective_kernels();
    opts.add_xla_gpu_experimental_use_collective_kernels(
        DebugOptions::COLLECTIVE_KERNEL_REDUCE_SCATTER);
    opts.set_xla_gpu_graph_min_graph_size(1);
    return opts;
  }

  bool HasDevices(int32_t required_device_count) {
    return device_count() >= required_device_count;
  }

  // Verifies that the optimized module contains `expected_count` Triton
  // one-shot reduce-scatter collective fusions.
  void VerifyOneShotReduceScatter(const HloModule* optimized_module,
                                  int expected_count = 1) {
    ASSERT_NE(optimized_module, nullptr);
    int count = 0;
    for (const HloComputation* comp : optimized_module->computations()) {
      for (const HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() != HloOpcode::kReduceScatter) {
          continue;
        }
        ++count;
        ASSERT_OK_AND_ASSIGN(gpu::GpuBackendConfig gpu_config,
                             instr->backend_config<gpu::GpuBackendConfig>());
        EXPECT_EQ(gpu_config.collective_backend_config().kernel_strategy(),
                  gpu::CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT)
            << instr->ToString();
        EXPECT_EQ(instr->parent()->FusionInstruction() != nullptr, true)
            << "Expected " << instr->name() << " to be in a fusion.";
        // Flattening must keep the name of the original reduce-scatter (all
        // test modules name it `rs`).
        EXPECT_TRUE(absl::StartsWith(instr->name(), "rs")) << instr->name();
      }
    }
    EXPECT_EQ(count, expected_count);
  }
};

// 2-GPU f32[256] -> f32[128] sum.
TEST_F(ReduceScatterTest, Basic2GpuF32Sum) {
  constexpr int32_t kNumReplicas = 2;
  if (!HasDevices(kNumReplicas)) {
    GTEST_SKIP() << "Requires " << kNumReplicas << " devices.";
  }
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT r = f32[] add(a, b)
  }
  ENTRY test_computation {
    param_0 = f32[256] parameter(0)
    ROOT rs = f32[128] reduce-scatter(param_0), dimensions={0},
      replica_groups={{0,1}}, to_apply=add
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  std::vector<float> in0(256), in1(256);
  for (int i = 0; i < 256; ++i) {
    in0[i] = static_cast<float>(i);
    in1[i] = static_cast<float>(1000 + 2 * i);
  }
  Literal input_r0 = LiteralUtil::CreateR1<float>(in0);
  Literal input_r1 = LiteralUtil::CreateR1<float>(in1);
  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));
  VerifyOneShotReduceScatter(result.optimized_module);
  ASSERT_EQ(result.results.size(), kNumReplicas);
  for (int r = 0; r < kNumReplicas; ++r) {
    std::vector<float> expected(128);
    for (int i = 0; i < 128; ++i) {
      expected[i] = in0[r * 128 + i] + in1[r * 128 + i];
    }
    EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<float>(expected),
                                       result.results[r]))
        << "Mismatch at replica " << r;
  }
}

// 2-GPU 2D s32[8,300] -> s32[4,300] max. The per-rank chunk has 1200 elements,
// which is not a power of two, so this exercises masked tiles.
TEST_F(ReduceScatterTest, TwoDimS32Max) {
  constexpr int32_t kNumReplicas = 2;
  if (!HasDevices(kNumReplicas)) {
    GTEST_SKIP() << "Requires " << kNumReplicas << " devices.";
  }
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  max {
    a = s32[] parameter(0)
    b = s32[] parameter(1)
    ROOT r = s32[] maximum(a, b)
  }
  ENTRY test_computation {
    param_0 = s32[8,300] parameter(0)
    ROOT rs = s32[4,300] reduce-scatter(param_0), dimensions={0},
      channel_id=1, use_global_device_ids=true,
      replica_groups={{0,1}}, to_apply=max
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                        kModuleStr, /*replica_count=*/1,
                                        /*num_partitions=*/kNumReplicas));
  Array2D<int32_t> in0(8, 300), in1(8, 300);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 300; ++j) {
      in0(i, j) = (i * 300 + j) % 7 - 3;
      in1(i, j) = (i * 300 + j) % 5 - 2;
    }
  }
  Literal input_r0 = LiteralUtil::CreateR2FromArray2D<int32_t>(in0);
  Literal input_r1 = LiteralUtil::CreateR2FromArray2D<int32_t>(in1);
  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));
  VerifyOneShotReduceScatter(result.optimized_module);
  ASSERT_EQ(result.results.size(), kNumReplicas);
  for (int r = 0; r < kNumReplicas; ++r) {
    Array2D<int32_t> expected(4, 300);
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 300; ++j) {
        expected(i, j) = std::max(in0(r * 4 + i, j), in1(r * 4 + i, j));
      }
    }
    EXPECT_TRUE(LiteralTestUtil::Equal(
        LiteralUtil::CreateR2FromArray2D<int32_t>(expected), result.results[r]))
        << "Mismatch at replica " << r;
  }
}

// Runs the reduce-scatter repeatedly inside a while loop so that consecutive
// invocations alternate between the two staging buffer sets. With
// xla_gpu_graph_min_graph_size=1 the loop body is also captured into a command
// buffer.
TEST_F(ReduceScatterTest, RepeatedInWhileLoopS32Sum) {
  constexpr int32_t kNumReplicas = 2;
  if (!HasDevices(kNumReplicas)) {
    GTEST_SKIP() << "Requires " << kNumReplicas << " devices.";
  }
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  add {
    a = s32[] parameter(0)
    b = s32[] parameter(1)
    ROOT r = s32[] add(a, b)
  }
  cond {
    p = (s32[], s32[4096]) parameter(0)
    i = s32[] get-tuple-element(p), index=0
    n = s32[] constant(5)
    ROOT lt = pred[] compare(i, n), direction=LT
  }
  body {
    p = (s32[], s32[4096]) parameter(0)
    i = s32[] get-tuple-element(p), index=0
    x = s32[4096] get-tuple-element(p), index=1
    rs = s32[2048] reduce-scatter(x), dimensions={0},
      replica_groups={{0,1}}, to_apply=add
    y = s32[4096] concatenate(rs, rs), dimensions={0}
    one = s32[] constant(1)
    i1 = s32[] add(i, one)
    ROOT t = (s32[], s32[4096]) tuple(i1, y)
  }
  ENTRY test_computation {
    param_0 = s32[4096] parameter(0)
    zero = s32[] constant(0)
    init = (s32[], s32[4096]) tuple(zero, param_0)
    w = (s32[], s32[4096]) while(init), condition=cond, body=body
    ROOT out = s32[4096] get-tuple-element(w), index=1
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  std::vector<int32_t> in0(4096), in1(4096);
  for (int i = 0; i < 4096; ++i) {
    in0[i] = i % 13;
    in1[i] = (i % 11) + 1;
  }
  Literal input_r0 = LiteralUtil::CreateR1<int32_t>(in0);
  Literal input_r1 = LiteralUtil::CreateR1<int32_t>(in1);
  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));
  VerifyOneShotReduceScatter(result.optimized_module);
  ASSERT_EQ(result.results.size(), kNumReplicas);

  // Reference simulation.
  std::vector<std::vector<int32_t>> state = {in0, in1};
  for (int iter = 0; iter < 5; ++iter) {
    std::vector<std::vector<int32_t>> next(kNumReplicas,
                                           std::vector<int32_t>(4096));
    for (int r = 0; r < kNumReplicas; ++r) {
      for (int i = 0; i < 2048; ++i) {
        const int32_t v = state[0][r * 2048 + i] + state[1][r * 2048 + i];
        next[r][i] = v;
        next[r][2048 + i] = v;
      }
    }
    state = std::move(next);
  }
  for (int r = 0; r < kNumReplicas; ++r) {
    EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(state[r]),
                                       result.results[r]))
        << "Mismatch at replica " << r;
  }
}

// 4-GPU bf16 sum with a layout similar to production MoE shapes:
// bf16[16,8,128] -> bf16[4,8,128].
TEST_F(ReduceScatterTest, FourGpuBf16Sum) {
  constexpr int32_t kNumReplicas = 4;
  if (!HasDevices(kNumReplicas)) {
    GTEST_SKIP() << "Requires " << kNumReplicas << " devices.";
  }
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  add {
    a = bf16[] parameter(0)
    b = bf16[] parameter(1)
    ROOT r = bf16[] add(a, b)
  }
  ENTRY test_computation {
    param_0 = bf16[16,8,128] parameter(0)
    ROOT rs = bf16[4,8,128] reduce-scatter(param_0), dimensions={0},
      replica_groups={{0,1,2,3}}, to_apply=add
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  // Small integer values are exactly representable in bf16 and their sums
  // too.
  std::vector<Literal> inputs;
  inputs.reserve(kNumReplicas);
  for (int r = 0; r < kNumReplicas; ++r) {
    Array3D<bfloat16> in(16, 8, 128);
    in.Each([&](absl::Span<const int64_t> idx, bfloat16* v) {
      *v = static_cast<bfloat16>(
          static_cast<float>((idx[0] + idx[1] + idx[2] + r) % 9));
    });
    inputs.push_back(LiteralUtil::CreateR3FromArray3D<bfloat16>(in));
  }
  std::vector<std::vector<Literal*>> args;
  for (Literal& l : inputs) {
    args.push_back({&l});
  }
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));
  VerifyOneShotReduceScatter(result.optimized_module);
  ASSERT_EQ(result.results.size(), kNumReplicas);
  for (int r = 0; r < kNumReplicas; ++r) {
    Array3D<bfloat16> expected(4, 8, 128);
    expected.Each([&](absl::Span<const int64_t> idx, bfloat16* v) {
      float sum = 0;
      for (int p = 0; p < kNumReplicas; ++p) {
        sum += static_cast<float>((r * 4 + idx[0] + idx[1] + idx[2] + p) % 9);
      }
      *v = static_cast<bfloat16>(sum);
    });
    EXPECT_TRUE(LiteralTestUtil::Equal(
        LiteralUtil::CreateR3FromArray3D<bfloat16>(expected),
        result.results[r]))
        << "Mismatch at replica " << r;
  }
}

TEST_F(ReduceScatterTest, PredOr2D) {
  constexpr int32_t kNumReplicas = 2;
  if (!HasDevices(kNumReplicas)) {
    GTEST_SKIP() << "Requires " << kNumReplicas << " devices.";
  }
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  or_op {
    a = pred[] parameter(0)
    b = pred[] parameter(1)
    ROOT r = pred[] or(a, b)
  }
  ENTRY test_computation {
    param_0 = pred[32,128] parameter(0)
    ROOT rs = pred[16,128] reduce-scatter(param_0), dimensions={0},
      replica_groups={{0,1}}, to_apply=or_op
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  Array2D<bool> in0(32, 128), in1(32, 128);
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 128; ++j) {
      in0(i, j) = ((i + j) % 3) == 0;
      in1(i, j) = ((i * 2 + j) % 5) == 0;
    }
  }
  Literal input_r0 = LiteralUtil::CreateR2FromArray2D<bool>(in0);
  Literal input_r1 = LiteralUtil::CreateR2FromArray2D<bool>(in1);
  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));
  VerifyOneShotReduceScatter(result.optimized_module);
  ASSERT_EQ(result.results.size(), kNumReplicas);
  for (int r = 0; r < kNumReplicas; ++r) {
    Array2D<bool> expected(16, 128);
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 128; ++j) {
        expected(i, j) = in0(r * 16 + i, j) || in1(r * 16 + i, j);
      }
    }
    EXPECT_TRUE(LiteralTestUtil::Equal(
        LiteralUtil::CreateR2FromArray2D<bool>(expected), result.results[r]))
        << "Mismatch at replica " << r;
  }
}

}  // namespace
}  // namespace xla

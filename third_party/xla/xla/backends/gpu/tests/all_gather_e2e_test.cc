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
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/all_gather.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

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

// Runs 10 consequent all-gathers and verifies that by default all of them use
// XLA's one-shot all-gather kernel and are placed into command buffers, and
// that excluding COLLECTIVES_KERNEL from --xla_gpu_enable_command_buffer
// disables command buffer capture for them.
TEST_F(AllGatherTest, TenConsequentAllGathersWithCudaGraphs) {
  constexpr int32_t kNumReplicas = 2;
  constexpr int kNumAllGathers = 10;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p0 = f32[16] parameter(0)
    ag0 = f32[32] all-gather(p0), dimensions={0}, replica_groups={{0,1}}
    ag1 = f32[64] all-gather(ag0), dimensions={0}, replica_groups={{0,1}}
    ag2 = f32[128] all-gather(ag1), dimensions={0}, replica_groups={{0,1}}
    ag3 = f32[256] all-gather(ag2), dimensions={0}, replica_groups={{0,1}}
    ag4 = f32[512] all-gather(ag3), dimensions={0}, replica_groups={{0,1}}
    ag5 = f32[1024] all-gather(ag4), dimensions={0}, replica_groups={{0,1}}
    ag6 = f32[2048] all-gather(ag5), dimensions={0}, replica_groups={{0,1}}
    ag7 = f32[4096] all-gather(ag6), dimensions={0}, replica_groups={{0,1}}
    ag8 = f32[8192] all-gather(ag7), dimensions={0}, replica_groups={{0,1}}
    ROOT ag9 = f32[16384] all-gather(ag8), dimensions={0}, replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory dump_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  const std::string default_dump_dir =
      absl::StrCat(dump_dir.path(), "/default");
  const std::string no_graph_dump_dir =
      absl::StrCat(dump_dir.path(), "/no_graphs");

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  DebugOptions& debug_options =
      module->mutable_config().mutable_debug_options();
  // The fixture enables COLLECTIVE_KERNEL_ALL_GATHER, so the 10 all-gathers
  // lower to XLA's Triton one-shot all-gather kernel (CollectiveKernelThunk)
  // instead of NCCL. COLLECTIVES_KERNEL is enabled in
  // --xla_gpu_enable_command_buffer by default.
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.set_xla_gpu_all_gather_combine_threshold_bytes(0);
  // Command buffers are formed at the thunk level, so dump the thunk sequence
  // to check which collectives were placed into them.
  debug_options.set_xla_dump_to(default_dump_dir);

  Literal input_r0 = LiteralUtil::CreateR1<float>(std::vector<float>(16, 1.0f));
  Literal input_r1 = LiteralUtil::CreateR1<float>(std::vector<float>(16, 2.0f));
  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  // Every all-gather must use KERNEL_STRATEGY_TRITON_ONE_SHOT (XLA one-shot
  // kernel, not NCCL).
  VerifyOneShotAllGather(result.optimized_module);

  // By default, all one-shot all-gathers must be recorded into command buffers,
  // and none may fall back to NCCL.
  ASSERT_OK_AND_ASSIGN(
      CommandBufferThunkCounts one_shot,
      CountThunksInDump(default_dump_dir, "kCollectiveKernel"));
  EXPECT_EQ(one_shot.in_command_buffer, kNumAllGathers);
  EXPECT_EQ(one_shot.outside_command_buffer, 0);
  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts nccl,
                       CountThunksInDump(default_dump_dir, "kAllGather"));
  EXPECT_EQ(nccl.in_command_buffer + nccl.outside_command_buffer, 0);

  // Each all-gather concatenates the two ranks' buffers, so the result is 512
  // repetitions of [16 x 1.0, 16 x 2.0] on both replicas.
  ASSERT_EQ(result.results.size(), kNumReplicas);
  std::vector<float> expected_data;
  expected_data.reserve(16384);
  for (int rep = 0; rep < 512; ++rep) {
    expected_data.insert(expected_data.end(), 16, 1.0f);
    expected_data.insert(expected_data.end(), 16, 2.0f);
  }
  Literal expected = LiteralUtil::CreateR1<float>(expected_data);
  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result.results[i]))
        << "Mismatch at replica " << i;
  }

  // Excluding COLLECTIVES_KERNEL from --xla_gpu_enable_command_buffer must
  // disable command buffer capture for the collective kernels.
  ASSERT_OK_AND_ASSIGN(auto no_graph_module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  DebugOptions& no_graph_debug_options =
      no_graph_module->mutable_config().mutable_debug_options();
  auto* enabled_commands =
      no_graph_debug_options.mutable_xla_gpu_enable_command_buffer();
  enabled_commands->erase(
      std::remove(enabled_commands->begin(), enabled_commands->end(),
                  DebugOptions::COLLECTIVES_KERNEL),
      enabled_commands->end());
  no_graph_debug_options.set_xla_gpu_graph_min_graph_size(1);
  no_graph_debug_options.set_xla_gpu_all_gather_combine_threshold_bytes(0);
  no_graph_debug_options.set_xla_dump_to(no_graph_dump_dir);

  ASSERT_OK_AND_ASSIGN(ExecutionResult no_graph_result,
                       ExecuteReplicated(std::move(no_graph_module), args));
  VerifyOneShotAllGather(no_graph_result.optimized_module);

  ASSERT_OK_AND_ASSIGN(
      CommandBufferThunkCounts one_shot_no_graph,
      CountThunksInDump(no_graph_dump_dir, "kCollectiveKernel"));
  EXPECT_EQ(one_shot_no_graph.in_command_buffer, 0);
  EXPECT_EQ(one_shot_no_graph.outside_command_buffer, kNumAllGathers);
  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts nccl_no_graph,
                       CountThunksInDump(no_graph_dump_dir, "kAllGather"));
  EXPECT_EQ(
      nccl_no_graph.in_command_buffer + nccl_no_graph.outside_command_buffer,
      0);
  ASSERT_EQ(no_graph_result.results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, no_graph_result.results[i]))
        << "Mismatch at replica " << i << " without graphs";
  }
}

// Runs 20 one-shot all-gathers inside a 20-iteration while loop with WHILE
// command buffers enabled, and verifies that the while thunk and all 20
// collective kernels are placed into command buffers and produce the expected
// result across repeated executions (exercising the persistent device-side
// invocation counter).
TEST_F(AllGatherTest, TwentyAllGathersInWhileLoopWithCudaGraphs) {
  constexpr int32_t kNumReplicas = 2;
  constexpr int kNumAllGathers = 20;
  constexpr int kNumIterations = 20;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  // Each step k in the loop body gathers in_k = 0.5 * prev + p0 (f32[128])
  // across the 2 replicas into ag_k (f32[256]) and sums the two 128-element
  // slices: (0.5 * v + 1.0) + (0.5 * v + 2.0) = v + 3.0. Every all-gather
  // therefore depends on both replicas' slices from the previous all-gather and
  // adds 3.0 without growing the tensor shape or overflowing f32 across
  // 20 * 20 = 400 collective launches.
  std::string body_ops;
  std::string prev = "acc";
  for (int k = 0; k < kNumAllGathers; ++k) {
    absl::StrAppendFormat(
        &body_ops,
        "    scaled%d = f32[128] multiply(%s, half)\n"
        "    in%d = f32[128] add(scaled%d, p0)\n"
        "    ag%d = f32[256] all-gather(in%d), dimensions={0}, "
        "replica_groups={{0,1}}\n"
        "    s0_%d = f32[128] slice(ag%d), slice={[0:128]}\n"
        "    s1_%d = f32[128] slice(ag%d), slice={[128:256]}\n"
        "    sum%d = f32[128] add(s0_%d, s1_%d)\n",
        k, prev, k, k, k, k, k, k, k, k, k, k, k);
    prev = absl::StrCat("sum", k);
  }

  const std::string module_str = absl::StrFormat(
      R"(
  HloModule test

  cond {
    state = (s32[], f32[128], f32[128]) parameter(0)
    i = s32[] get-tuple-element(state), index=0
    limit = s32[] constant(%d)
    ROOT cmp = pred[] compare(i, limit), direction=LT
  }

  body {
    state = (s32[], f32[128], f32[128]) parameter(0)
    i = s32[] get-tuple-element(state), index=0
    acc = f32[128] get-tuple-element(state), index=1
    p0 = f32[128] get-tuple-element(state), index=2
    one = s32[] constant(1)
    next_i = s32[] add(i, one)
    half_scalar = f32[] constant(0.5)
    half = f32[128] broadcast(half_scalar), dimensions={}
%s    ROOT next_state = (s32[], f32[128], f32[128]) tuple(next_i, %s, p0)
  }

  ENTRY test_computation {
    p0 = f32[128] parameter(0)
    zero_i = s32[] constant(0)
    zero_f = f32[] constant(0.0)
    init_acc = f32[128] broadcast(zero_f), dimensions={}
    init = (s32[], f32[128], f32[128]) tuple(zero_i, init_acc, p0)
    loop = (s32[], f32[128], f32[128]) while(init), condition=cond, body=body
    ROOT out = f32[128] get-tuple-element(loop), index=1
  }
  )",
      kNumIterations, body_ops, prev);

  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory dump_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(module_str, kNumReplicas));

  DebugOptions& debug_options =
      module->mutable_config().mutable_debug_options();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::WHILE);
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.set_xla_gpu_all_gather_combine_threshold_bytes(0);
  debug_options.set_xla_gpu_enable_while_loop_unrolling(
      DebugOptions::WHILE_LOOP_UNROLLING_NO_UNROLL);
  debug_options.set_xla_dump_to(dump_dir.path());

  Literal input_r0 =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, 1.0f));
  Literal input_r1 =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, 2.0f));
  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllGather(result.optimized_module);

  // Both the while thunk and all 20 one-shot all-gathers in its body must be
  // recorded into command buffers, and none may fall back to NCCL.
  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts while_counts,
                       CountThunksInDump(dump_dir.path(), "kWhile"));
  EXPECT_EQ(while_counts.in_command_buffer, 1);
  EXPECT_EQ(while_counts.outside_command_buffer, 0);
  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts one_shot,
                       CountThunksInDump(dump_dir.path(), "kCollectiveKernel"));
  EXPECT_EQ(one_shot.in_command_buffer, kNumAllGathers);
  EXPECT_EQ(one_shot.outside_command_buffer, 0);
  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts nccl,
                       CountThunksInDump(dump_dir.path(), "kAllGather"));
  EXPECT_EQ(nccl.in_command_buffer + nccl.outside_command_buffer, 0);

  // 20 iterations * 20 all-gathers * 3.0 per all-gather = 1200.0.
  const float expected_val = 3.0f * kNumAllGathers * kNumIterations;
  Literal expected =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, expected_val));
  ASSERT_EQ(result.results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result.results[i]))
        << "Mismatch at replica " << i << " on first execution";
  }

  // Re-run the same executable to verify that the device-side counters persist
  // across executions and the replayed CUDA graph still produces the expected
  // result.
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> second_results,
                       ExecuteReplicated(result.executable.get(), args));
  ASSERT_EQ(second_results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, second_results[i]))
        << "Mismatch at replica " << i << " on second execution";
  }
}

// 2D shape: f32[16, 32] -> f32[16, 64] (gather along dim 1).
TEST_F(AllGatherTest, TwoDimensionalGatherDim1) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[16,32] parameter(0)
    ROOT all-gather = f32[16,64] all-gather(param_0), dimensions={1},
      replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  Literal input_r0 = LiteralUtil::CreateFull<float>({16, 32}, 10.0f);
  Literal input_r1 = LiteralUtil::CreateFull<float>({16, 32}, 20.0f);

  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllGather(result.optimized_module);

  ASSERT_EQ(result.results.size(), kNumReplicas);

  Literal expected = LiteralUtil::CreateFull<float>({16, 64}, 0.0f);
  for (int64_t row = 0; row < 16; ++row) {
    for (int64_t col = 0; col < 64; ++col) {
      expected.Set<float>({row, col}, (col < 32) ? 10.0f : 20.0f);
    }
  }

  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result.results[i]))
        << "Mismatch at replica " << i;
  }
}

// Repeated invocations on the same executable to verify double-buffering
// across alternating slots (signal_value & 1).
TEST_F(AllGatherTest, RepeatedInvocationsDoubleBuffering) {
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

  Literal input1_r0 =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, 1.0f));
  Literal input1_r1 =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, 2.0f));
  std::vector<std::vector<Literal*>> args1 = {{&input1_r0}, {&input1_r1}};

  ASSERT_OK_AND_ASSIGN(ExecutionResult result1,
                       ExecuteReplicated(std::move(module), args1));
  VerifyOneShotAllGather(result1.optimized_module);

  for (int iter = 2; iter <= 4; ++iter) {
    float v0 = static_cast<float>(iter * 10 + 1);
    float v1 = static_cast<float>(iter * 10 + 2);
    Literal in_r0 = LiteralUtil::CreateR1<float>(std::vector<float>(128, v0));
    Literal in_r1 = LiteralUtil::CreateR1<float>(std::vector<float>(128, v1));
    std::vector<std::vector<Literal*>> iter_args = {{&in_r0}, {&in_r1}};

    ASSERT_OK_AND_ASSIGN(
        std::vector<Literal> iter_results,
        ExecuteReplicated(result1.executable.get(), iter_args));
    ASSERT_EQ(iter_results.size(), kNumReplicas);

    std::vector<float> expected_data(256);
    for (int i = 0; i < 128; ++i) {
      expected_data[i] = v0;
      expected_data[i + 128] = v1;
    }
    Literal expected = LiteralUtil::CreateR1<float>(expected_data);
    for (int i = 0; i < kNumReplicas; ++i) {
      EXPECT_TRUE(LiteralTestUtil::Equal(expected, iter_results[i]))
          << "Mismatch at replica " << i << " on iteration " << iter;
    }
  }
}

class AllGatherTypesTest : public AllGatherTest,
                           public ::testing::WithParamInterface<PrimitiveType> {
};

// Same shape as Large2GpuF32, for every element type supported by the
// one-shot kernel.
TEST_P(AllGatherTypesTest, Large2Gpu) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = %1$s[4096] parameter(0)
    ROOT all-gather = %1$s[8192] all-gather(param_0), dimensions={0},
      replica_groups={{0,1}}
  }
  )";

  const PrimitiveType type = GetParam();
  const std::string module_str = absl::StrFormat(
      kModuleStr, primitive_util::LowercasePrimitiveTypeName(type));
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(module_str, kNumReplicas));

  // Rank r contributes the r-th half of the expected output.
  ASSERT_OK_AND_ASSIGN(Literal expected,
                       MakeFakeLiteral(ShapeUtil::MakeShape(type, {8192})));
  Literal input_r0 = expected.Slice({0}, {4096});
  Literal input_r1 = expected.Slice({4096}, {8192});

  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllGather(result.optimized_module);

  ASSERT_EQ(result.results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result.results[i]))
        << "Mismatch at replica " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherTypes, AllGatherTypesTest,
    ::testing::ValuesIn(gpu::kSupportedAllGatherTypes),
    [](const ::testing::TestParamInfo<PrimitiveType>& info) {
      return std::string(
          primitive_util::LowercasePrimitiveTypeName(info.param));
    });

}  // namespace
}  // namespace xla

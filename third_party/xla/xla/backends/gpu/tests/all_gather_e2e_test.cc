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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/all_gather.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using Strategy = gpu::CollectiveBackendConfig::CollectiveKernelStrategy;
constexpr Strategy kOneShot =
    gpu::CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT;
constexpr Strategy kTwoShot =
    gpu::CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_TWO_SHOT;

absl::string_view StrategyTestSuffix(Strategy strategy) {
  return strategy == kTwoShot ? "TwoShot" : "OneShot";
}

::testing::AssertionResult EqualOnAllReplicas(const Literal& expected,
                                              absl::Span<const Literal> results,
                                              int32_t num_replicas) {
  if (results.size() != static_cast<size_t>(num_replicas)) {
    return ::testing::AssertionFailure()
           << "Expected " << num_replicas << " results, got " << results.size();
  }
  for (int32_t i = 0; i < num_replicas; ++i) {
    if (!LiteralTestUtil::Equal(expected, results[i])) {
      return ::testing::AssertionFailure() << "Mismatch at replica " << i;
    }
  }
  return ::testing::AssertionSuccess();
}

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

  void VerifyAllGatherStrategy(
      const HloModule* optimized_module,
      gpu::CollectiveBackendConfig::CollectiveKernelStrategy
          expected_strategy) {
    ASSERT_NE(optimized_module, nullptr);
    bool found_all_gather = false;
    for (const HloComputation* comp : optimized_module->computations()) {
      for (const HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() == HloOpcode::kAllGather ||
            instr->opcode() == HloOpcode::kAllGatherStart) {
          found_all_gather = true;
          ASSERT_OK_AND_ASSIGN(gpu::GpuBackendConfig gpu_config,
                               instr->backend_config<gpu::GpuBackendConfig>());
          EXPECT_EQ(gpu_config.collective_backend_config().kernel_strategy(),
                    expected_strategy)
              << "Expected AllGather instruction " << instr->name()
              << " to use "
              << gpu::CollectiveBackendConfig::CollectiveKernelStrategy_Name(
                     expected_strategy)
              << ", but got: "
              << gpu::CollectiveBackendConfig::CollectiveKernelStrategy_Name(
                     gpu_config.collective_backend_config().kernel_strategy());
        }
      }
    }
    EXPECT_TRUE(found_all_gather)
        << "Expected to find an AllGather instruction in optimized HLO.";
  }

  bool SetUpStrategyTest(int32_t num_replicas, Strategy strategy) {
    if (strategy == kTwoShot && !Capability().IsCuda()) {
      [&]() {
        GTEST_SKIP()
            << "Two-shot all-gather requires CUDA LSA symmetric memory.";
      }();
      return false;
    }
    return CheckDeviceCount(num_replicas);
  }

  bool SetUpTwoShotTest(int32_t num_replicas) {
    return SetUpStrategyTest(num_replicas, kTwoShot);
  }

  absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
  ParseAndPrepareModuleForStrategy(absl::string_view hlo_text,
                                   int64_t num_replicas, Strategy strategy) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                     ParseAndReturnVerifiedModule(hlo_text, num_replicas));
    if (strategy == kTwoShot) {
      for (HloComputation* comp : module->computations()) {
        for (HloInstruction* instr : comp->instructions()) {
          if (instr->opcode() == HloOpcode::kAllGather) {
            ABSL_ASSIGN_OR_RETURN(gpu::GpuBackendConfig gpu_config,
                             instr->backend_config<gpu::GpuBackendConfig>());
            gpu_config.mutable_collective_backend_config()->set_kernel_strategy(
                strategy);
            ABSL_RETURN_IF_ERROR(instr->set_backend_config(gpu_config));
          }
        }
      }
    }
    return module;
  }

  static Literal MakeGathered32x32(float top, float bottom) {
    Literal expected = LiteralUtil::CreateFull<float>({32, 32}, 0.0f);
    for (int64_t row = 0; row < 32; ++row) {
      for (int64_t col = 0; col < 32; ++col) {
        expected.Set<float>({row, col}, (row < 16) ? top : bottom);
      }
    }
    return expected;
  }
};

class AllGatherStrategyTest : public AllGatherTest,
                              public ::testing::WithParamInterface<Strategy> {
 protected:
  bool SetUpTest(int32_t num_replicas) {
    return SetUpStrategyTest(num_replicas, GetParam());
  }

  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> ParseAndPrepareModule(
      absl::string_view hlo_text, int64_t num_replicas) {
    return ParseAndPrepareModuleForStrategy(hlo_text, num_replicas, GetParam());
  }

  void VerifyStrategy(const HloModule* optimized_module) {
    VerifyAllGatherStrategy(optimized_module, GetParam());
  }
};

// Runs 10 consequent all-gathers and verifies that by default all of them use
// XLA's collective all-gather kernel and are placed into command buffers, and
// that excluding COLLECTIVES_KERNEL from --xla_gpu_enable_command_buffer
// disables command buffer capture for them.
TEST_P(AllGatherStrategyTest, TenConsequentAllGathersWithCudaGraphs) {
  constexpr int32_t kNumReplicas = 2;
  constexpr int kNumAllGathers = 10;
  if (!SetUpTest(kNumReplicas)) {
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
                       ParseAndPrepareModule(kModuleStr, kNumReplicas));

  DebugOptions& debug_options =
      module->mutable_config().mutable_debug_options();
  // The fixture enables COLLECTIVE_KERNEL_ALL_GATHER, so the 10 all-gathers
  // lower to XLA's Triton all-gather kernel (CollectiveKernelThunk)
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

  // Every all-gather must use the expected Triton collective strategy (not
  // NCCL).
  VerifyStrategy(result.optimized_module);

  // By default, all collective-kernel all-gathers must be recorded into command
  // buffers, and none may fall back to NCCL.
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
  EXPECT_TRUE(EqualOnAllReplicas(expected, result.results, kNumReplicas));

  // Excluding COLLECTIVES_KERNEL from --xla_gpu_enable_command_buffer must
  // disable command buffer capture for the collective kernels.
  ASSERT_OK_AND_ASSIGN(auto no_graph_module,
                       ParseAndPrepareModule(kModuleStr, kNumReplicas));
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
  VerifyStrategy(no_graph_result.optimized_module);

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
  EXPECT_TRUE(
      EqualOnAllReplicas(expected, no_graph_result.results, kNumReplicas));
}

// Runs 20 collective-kernel all-gathers inside a 20-iteration while loop with
// WHILE command buffers enabled, and verifies that the while thunk and all 20
// collective kernels are placed into command buffers and produce the expected
// result across repeated executions (exercising the persistent device-side
// invocation counter).
TEST_P(AllGatherStrategyTest, TwentyAllGathersInWhileLoopWithCudaGraphs) {
  constexpr int32_t kNumReplicas = 2;
  constexpr int kNumAllGathers = 20;
  constexpr int kNumIterations = 20;
  if (!SetUpTest(kNumReplicas)) {
    return;
  }
  // ROCm has no equivalent of CUDA's conditional graph nodes, so
  // CommandBufferConversionPass drops the WHILE command there and the while
  // thunk stays outside of command buffers.
  if (Capability().IsRocm()) {
    GTEST_SKIP() << "Command buffers on ROCm do not support on-device control "
                    "flow (while loops).";
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
                       ParseAndPrepareModule(module_str, kNumReplicas));

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

  VerifyStrategy(result.optimized_module);

  // Both the while thunk and all 20 collective-kernel all-gathers in its body
  // must be recorded into command buffers, and none may fall back to NCCL.
  // Capturing the while thunk itself requires on-device control flow
  // (conditional graph nodes), which ROCm does not support, so the command
  // buffer conversion pass drops WHILE there and instead captures the loop
  // body into a command buffer nested inside the while thunk. The all-gather
  // expectations below hold either way.
  if (Capability().IsCuda()) {
    ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts while_counts,
                         CountThunksInDump(dump_dir.path(), "kWhile"));
    EXPECT_EQ(while_counts.in_command_buffer, 1);
    EXPECT_EQ(while_counts.outside_command_buffer, 0);
  }
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
  EXPECT_TRUE(EqualOnAllReplicas(expected, result.results, kNumReplicas));

  // Re-run the same executable to verify that the device-side counters persist
  // across executions and the replayed CUDA graph still produces the expected
  // result.
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> second_results,
                       ExecuteReplicated(result.executable.get(), args));
  EXPECT_TRUE(EqualOnAllReplicas(expected, second_results, kNumReplicas));
}

// Repeated invocations on the same executable to verify double-buffering
// across alternating slots (signal_value & 1).
TEST_P(AllGatherStrategyTest, RepeatedInvocationsDoubleBuffering) {
  constexpr int32_t kNumReplicas = 2;
  if (!SetUpTest(kNumReplicas)) {
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
                       ParseAndPrepareModule(kModuleStr, kNumReplicas));

  Literal input1_r0 =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, 1.0f));
  Literal input1_r1 =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, 2.0f));
  std::vector<std::vector<Literal*>> args1 = {{&input1_r0}, {&input1_r1}};

  ASSERT_OK_AND_ASSIGN(ExecutionResult result1,
                       ExecuteReplicated(std::move(module), args1));
  VerifyStrategy(result1.optimized_module);

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
    SCOPED_TRACE(absl::StrCat("iteration ", iter));
    EXPECT_TRUE(EqualOnAllReplicas(expected, iter_results, kNumReplicas));
  }
}

TEST_P(AllGatherStrategyTest, ReversedReplicaGroups) {
  constexpr int32_t kNumReplicas = 2;
  if (!SetUpTest(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test_reversed_groups
  ENTRY test_computation {
    param_0 = f32[16,32] parameter(0)
    ROOT all-gather = f32[32,32] all-gather(param_0), dimensions={0},
      replica_groups={{1,0}}
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndPrepareModule(kModuleStr, kNumReplicas));
  Literal in_r0 = LiteralUtil::CreateFull<float>({16, 32}, 10.0f);
  Literal in_r1 = LiteralUtil::CreateFull<float>({16, 32}, 20.0f);
  std::vector<std::vector<Literal*>> args = {{&in_r0}, {&in_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));
  VerifyStrategy(result.optimized_module);

  // In replica_groups={{1,0}}, replica 1 is group rank 0 (rows 0..15) and
  // replica 0 is group rank 1 (rows 16..31).
  Literal expected = MakeGathered32x32(20.0f, 10.0f);
  EXPECT_TRUE(EqualOnAllReplicas(expected, result.results, kNumReplicas));
}

INSTANTIATE_TEST_SUITE_P(AllGatherStrategies, AllGatherStrategyTest,
                         ::testing::Values(kOneShot, kTwoShot),
                         [](const ::testing::TestParamInfo<Strategy>& info) {
                           return std::string(StrategyTestSuffix(info.param));
                         });

struct AllGather2GpuShapeCase {
  std::string name;
  Shape input_shape;
  int64_t gather_dim;
};

class AllGather2GpuShapeTest
    : public AllGatherTest,
      public ::testing::WithParamInterface<
          std::tuple<AllGather2GpuShapeCase, Strategy>> {};

TEST_P(AllGather2GpuShapeTest, SingleAllGather) {
  constexpr int32_t kNumReplicas = 2;
  const auto& [tc, strategy] = GetParam();
  if (!SetUpStrategyTest(kNumReplicas, strategy)) {
    return;
  }

  Shape output_shape = tc.input_shape;
  output_shape.set_dimensions(
      tc.gather_dim, tc.input_shape.dimensions(tc.gather_dim) * kNumReplicas);
  const std::string input_str =
      ShapeUtil::HumanStringWithLayout(tc.input_shape);
  const std::string output_str = ShapeUtil::HumanStringWithLayout(output_shape);
  const std::string hlo_text = absl::StrFormat(
      R"(
  HloModule test, entry_computation_layout={(%1$s)->%2$s}
  ENTRY test_computation {
    param_0 = %1$s parameter(0)
    ROOT all-gather = %2$s all-gather(param_0), dimensions={%3$d},
      replica_groups={{0,1}}
  }
  )",
      input_str, output_str, tc.gather_dim);

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndPrepareModuleForStrategy(
                                        hlo_text, kNumReplicas, strategy));

  ASSERT_OK_AND_ASSIGN(Literal expected, MakeFakeLiteral(output_shape));
  const int64_t rank = output_shape.dimensions().size();
  std::vector<int64_t> start_r0(rank, 0);
  std::vector<int64_t> limit_r0(tc.input_shape.dimensions().begin(),
                                tc.input_shape.dimensions().end());
  std::vector<int64_t> start_r1(rank, 0);
  start_r1[tc.gather_dim] = tc.input_shape.dimensions(tc.gather_dim);
  std::vector<int64_t> limit_r1(output_shape.dimensions().begin(),
                                output_shape.dimensions().end());
  Literal in_r0 = expected.Slice(start_r0, limit_r0);
  Literal in_r1 = expected.Slice(start_r1, limit_r1);

  std::vector<std::vector<Literal*>> args = {{&in_r0}, {&in_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyAllGatherStrategy(result.optimized_module, strategy);
  EXPECT_TRUE(EqualOnAllReplicas(expected, result.results, kNumReplicas));
}

INSTANTIATE_TEST_SUITE_P(
    AllGather2GpuShapes, AllGather2GpuShapeTest,
    ::testing::Combine(
        ::testing::ValuesIn<AllGather2GpuShapeCase>({
            // Basic 2-GPU all-gather of f32[128] -> f32[256].
            {"Basic2GpuF32", ShapeUtil::MakeShape(F32, {128}), 0},
            // Larger 2-GPU all-gather to test multi-tile behavior: f32[4096]
            // -> f32[8192].
            {"Large2GpuF32", ShapeUtil::MakeShape(F32, {4096}), 0},
            // 2D shape: f32[16, 32] -> f32[32, 32] (gather along dim 0).
            {"TwoDimensional2Gpu", ShapeUtil::MakeShape(F32, {16, 32}), 0},
            // 2D shape: f32[16, 32] -> f32[16, 64] (gather along dim 1).
            {"TwoDimensionalGatherDim1", ShapeUtil::MakeShape(F32, {16, 32}),
             1},
            // 2D shape where the gather dimension per rank is 1 (not divisible
            // by world_size = 2), so tile_sizes = [1, 64] and staging
            // split_dim falls back to dim 1.
            {"TwoDimensionalGatherDimNotDivisibleByWorldSize",
             ShapeUtil::MakeShape(F32, {1, 64}), 0},
            // 1D sub-128-bit transfer (8 bytes per rank): s32[2] -> s32[4].
            {"SubTransactionSize1DS32", ShapeUtil::MakeShape(S32, {2}), 0},
            // 2D shape with unaligned non-gather dim (56 bytes per rank):
            // s32[2, 7] -> s32[4, 7].
            {"UnalignedNonGatherDim2DS32_2x7",
             ShapeUtil::MakeShape(S32, {2, 7}), 0},
            // 2D shape with larger odd non-gather dim (4,232 bytes per rank):
            // s32[2, 529] -> s32[4, 529].
            {"UnalignedNonGatherDim2DS32_2x529",
             ShapeUtil::MakeShape(S32, {2, 529}), 0},
            {"PredGatherDim1NonDefaultLayout",
             ShapeUtil::MakeShapeWithDenseLayout(PRED, {16, 8}, {0, 1}), 1},
        }),
        ::testing::Values(kOneShot, kTwoShot)),
    [](const ::testing::TestParamInfo<AllGather2GpuShapeTest::ParamType>&
           info) {
      return absl::StrCat(std::get<0>(info.param).name, "_",
                          StrategyTestSuffix(std::get<1>(info.param)));
    });

TEST_F(AllGatherTest, TwoShotFusedPrologue) {
  constexpr int32_t kNumReplicas = 2;
  if (!SetUpTwoShotTest(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test_fused_prologue
  fused_ag {
    param_0 = f32[16,32] parameter(0)
    two = f32[] constant(2.0)
    scale = f32[16,32] broadcast(two), dimensions={}
    scaled = f32[16,32] multiply(param_0, scale)
    ROOT ag = f32[32,32] all-gather(scaled), dimensions={0},
      replica_groups={{0,1}},
      backend_config={"collective_backend_config":{
        "kernel_strategy":"KERNEL_STRATEGY_TRITON_TWO_SHOT"
      }}
  }

  ENTRY test_computation {
    p0 = f32[16,32] parameter(0)
    ROOT fusion = f32[32,32] fusion(p0), kind=kCustom, calls=fused_ag,
      backend_config={
        "fusion_backend_config": {
          "kind": "__triton_collective",
          "block_level_fusion_config": {
            "num_warps": "4",
            "output_tiles": [{"sizes": ["8", "16"]}],
            "num_ctas": 1,
            "num_stages": 1,
            "is_tma_allowed": false,
            "is_warp_specialization_allowed": false
          }
        }
      }
  }
  )";
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  ASSERT_OK_AND_ASSIGN(Literal in_r0,
                       Literal::Make(ShapeUtil::MakeShape(F32, {16, 32})));
  ASSERT_OK_AND_ASSIGN(Literal in_r1,
                       Literal::Make(ShapeUtil::MakeShape(F32, {16, 32})));
  ASSERT_OK_AND_ASSIGN(Literal expected,
                       Literal::Make(ShapeUtil::MakeShape(F32, {32, 32})));
  for (int64_t row = 0; row < 16; ++row) {
    for (int64_t col = 0; col < 32; ++col) {
      const float v0 = static_cast<float>(row * 32 + col + 1);
      const float v1 = static_cast<float>((row + 16) * 32 + col + 1);
      in_r0.Set<float>({row, col}, v0);
      in_r1.Set<float>({row, col}, v1);
      expected.Set<float>({row, col}, 2.0f * v0);
      expected.Set<float>({row + 16, col}, 2.0f * v1);
    }
  }
  std::vector<std::vector<Literal*>> args = {{&in_r0}, {&in_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));
  VerifyAllGatherStrategy(result.optimized_module, kTwoShot);

  EXPECT_TRUE(EqualOnAllReplicas(expected, result.results, kNumReplicas));
}

struct Subgroup4GpuTestCase {
  std::string name;
  std::string replica_groups;
  std::vector<std::pair<int, int>> pairs;
};

class AllGatherSubgroups4GpuTest
    : public AllGatherTest,
      public ::testing::WithParamInterface<
          std::tuple<Subgroup4GpuTestCase, Strategy>> {};

TEST_P(AllGatherSubgroups4GpuTest, Subgroups) {
  constexpr int32_t kNumReplicas = 4;
  const auto& [tc, strategy] = GetParam();
  if (!SetUpStrategyTest(kNumReplicas, strategy)) {
    return;
  }

  const std::string hlo = absl::StrFormat(R"(
  HloModule test_subgroups
  ENTRY test_computation {
    param_0 = f32[16,32] parameter(0)
    ROOT all-gather = f32[32,32] all-gather(param_0), dimensions={0},
      replica_groups=%s
  }
  )",
                                          tc.replica_groups);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndPrepareModuleForStrategy(
                                        hlo, kNumReplicas, strategy));
  std::vector<Literal> inputs;
  inputs.reserve(kNumReplicas);
  for (int r = 0; r < kNumReplicas; ++r) {
    inputs.push_back(
        LiteralUtil::CreateFull<float>({16, 32}, static_cast<float>(r + 1)));
  }
  std::vector<std::vector<Literal*>> args(kNumReplicas);
  for (int r = 0; r < kNumReplicas; ++r) {
    args[r].push_back(&inputs[r]);
  }
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));
  VerifyAllGatherStrategy(result.optimized_module, strategy);

  ASSERT_EQ(result.results.size(), kNumReplicas);
  for (const auto& [first, second] : tc.pairs) {
    Literal expected = MakeGathered32x32(static_cast<float>(first + 1),
                                         static_cast<float>(second + 1));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result.results[first]))
        << "Mismatch at replica " << first << " for " << tc.replica_groups;
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result.results[second]))
        << "Mismatch at replica " << second << " for " << tc.replica_groups;
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherSubgroups4Gpu, AllGatherSubgroups4GpuTest,
    ::testing::Combine(::testing::ValuesIn<Subgroup4GpuTestCase>({
                           {"Adjacent", "{{0,1},{2,3}}", {{0, 1}, {2, 3}}},
                           {"Strided", "{{0,2},{1,3}}", {{0, 2}, {1, 3}}},
                       }),
                       ::testing::Values(kOneShot, kTwoShot)),
    [](const ::testing::TestParamInfo<AllGatherSubgroups4GpuTest::ParamType>&
           info) {
      return absl::StrCat(std::get<0>(info.param).name, "_",
                          StrategyTestSuffix(std::get<1>(info.param)));
    });

class AllGatherTypesTest : public AllGatherTest,
                           public ::testing::WithParamInterface<
                               std::tuple<PrimitiveType, Strategy>> {};

// Same shape as Large2GpuF32, for every element type supported by the
// collective kernel.
TEST_P(AllGatherTypesTest, Large2Gpu) {
  constexpr int32_t kNumReplicas = 2;
  const auto& [type, strategy] = GetParam();
  if (!SetUpStrategyTest(kNumReplicas, strategy)) {
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

  const std::string module_str = absl::StrFormat(
      kModuleStr, primitive_util::LowercasePrimitiveTypeName(type));
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndPrepareModuleForStrategy(
                                        module_str, kNumReplicas, strategy));

  // Rank r contributes the r-th half of the expected output.
  ASSERT_OK_AND_ASSIGN(Literal expected,
                       MakeFakeLiteral(ShapeUtil::MakeShape(type, {8192})));
  Literal input_r0 = expected.Slice({0}, {4096});
  Literal input_r1 = expected.Slice({4096}, {8192});

  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyAllGatherStrategy(result.optimized_module, strategy);

  EXPECT_TRUE(EqualOnAllReplicas(expected, result.results, kNumReplicas));
}

se::GpuComputeCapability GetGpuComputeCapability() {
  auto platform = PlatformUtil::GetPlatform("gpu");
  if (!platform.ok()) {
    return se::GpuComputeCapability{};
  }
  auto executor = (*platform)->ExecutorForDevice(0);
  if (!executor.ok()) {
    return se::GpuComputeCapability{};
  }
  return (*executor)->GetDeviceDescription().gpu_compute_capability();
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherTypes, AllGatherTypesTest,
    ::testing::Combine(::testing::ValuesIn(gpu::SupportedAllGatherTypes(
                           GetGpuComputeCapability())),
                       ::testing::Values(kOneShot, kTwoShot)),
    [](const ::testing::TestParamInfo<AllGatherTypesTest::ParamType>& info) {
      return absl::StrCat(
          primitive_util::LowercasePrimitiveTypeName(std::get<0>(info.param)),
          "_", StrategyTestSuffix(std::get<1>(info.param)));
    });

struct VariadicAllGatherTestCase {
  std::string name;
  // Per-rank operand shapes.
  std::vector<Shape> operand_shapes;
  int64_t gather_dim;
};

class VariadicAllGatherTest
    : public AllGatherTest,
      public ::testing::WithParamInterface<
          std::tuple<VariadicAllGatherTestCase, Strategy>> {};

TEST_P(VariadicAllGatherTest, TwoGpus) {
  constexpr int32_t kNumReplicas = 2;
  const auto& [tc, strategy] = GetParam();
  if (!SetUpStrategyTest(kNumReplicas, strategy)) {
    return;
  }

  const int64_t num_operands = tc.operand_shapes.size();

  std::vector<Shape> output_shapes;
  std::vector<std::string> param_lines;
  std::vector<std::string> operand_names;
  for (int64_t k = 0; k < num_operands; ++k) {
    const Shape& in = tc.operand_shapes[k];
    Shape out = in;
    out.set_dimensions(tc.gather_dim,
                       in.dimensions(tc.gather_dim) * kNumReplicas);
    output_shapes.push_back(out);
    param_lines.push_back(absl::StrFormat("param_%d = %s parameter(%d)", k,
                                          ShapeUtil::HumanString(in), k));
    operand_names.push_back(absl::StrFormat("param_%d", k));
  }
  const std::string module_str = absl::StrFormat(
      R"(
  HloModule test
  ENTRY test_computation {
    %s
    ROOT all-gather = %s all-gather(%s), dimensions={%d},
      replica_groups={{0,1}}
  }
  )",
      absl::StrJoin(param_lines, "\n    "),
      ShapeUtil::HumanString(ShapeUtil::MakeTupleShape(output_shapes)),
      absl::StrJoin(operand_names, ", "), tc.gather_dim);
  SCOPED_TRACE(module_str);

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndPrepareModuleForStrategy(
                                        module_str, kNumReplicas, strategy));

  // Rank r contributes the r-th slice of each expected output along the
  // gather dimension.
  std::vector<Literal> expected_outputs;
  std::vector<std::vector<Literal>> inputs(kNumReplicas);
  for (int64_t k = 0; k < num_operands; ++k) {
    ASSERT_OK_AND_ASSIGN(Literal expected, MakeFakeLiteral(output_shapes[k]));
    const int64_t per_rank = tc.operand_shapes[k].dimensions(tc.gather_dim);
    for (int r = 0; r < kNumReplicas; ++r) {
      std::vector<int64_t> start(output_shapes[k].dimensions().size(), 0);
      std::vector<int64_t> limit(output_shapes[k].dimensions().begin(),
                                 output_shapes[k].dimensions().end());
      start[tc.gather_dim] = r * per_rank;
      limit[tc.gather_dim] = (r + 1) * per_rank;
      inputs[r].push_back(expected.Slice(start, limit));
    }
    expected_outputs.push_back(std::move(expected));
  }
  std::vector<std::vector<Literal*>> args(kNumReplicas);
  for (int r = 0; r < kNumReplicas; ++r) {
    for (Literal& input : inputs[r]) {
      args[r].push_back(&input);
    }
  }

  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyAllGatherStrategy(result.optimized_module, strategy);

  Literal expected_tuple =
      LiteralUtil::MakeTupleOwned(std::move(expected_outputs));
  EXPECT_TRUE(EqualOnAllReplicas(expected_tuple, result.results, kNumReplicas));
}

INSTANTIATE_TEST_SUITE_P(
    VariadicAllGather, VariadicAllGatherTest,
    ::testing::Combine(
        ::testing::ValuesIn<VariadicAllGatherTestCase>({
            {"TwoF32SameShape",
             {ShapeUtil::MakeShape(F32, {256}),
              ShapeUtil::MakeShape(F32, {256})},
             /*gather_dim=*/0},
            {"TwoF32DifferentSizes",
             {ShapeUtil::MakeShape(F32, {128}),
              ShapeUtil::MakeShape(F32, {64})},
             /*gather_dim=*/0},
            {"MixedTypesAndRanks",
             {ShapeUtil::MakeShape(F32, {16, 32}),
              ShapeUtil::MakeShape(S32, {8})},
             /*gather_dim=*/0},
            {"ThreeOperands",
             {ShapeUtil::MakeShape(F32, {64}),
              ShapeUtil::MakeShape(BF16, {128}),
              ShapeUtil::MakeShape(S32, {32})},
             /*gather_dim=*/0},
            {"GatherDim1",
             {ShapeUtil::MakeShape(F32, {4, 8}),
              ShapeUtil::MakeShape(F32, {4, 16})},
             /*gather_dim=*/1},
            {"NonPowerOfTwoNonGatherDim",
             {ShapeUtil::MakeShape(S32, {2, 7}),
              ShapeUtil::MakeShape(F32, {4, 5})},
             /*gather_dim=*/0},
            {"NonPowerOfTwoNonGatherDimLarge",
             {ShapeUtil::MakeShape(S32, {2, 529}),
              ShapeUtil::MakeShape(BF16, {1, 300})},
             /*gather_dim=*/0},
            {"Large",
             {ShapeUtil::MakeShape(F32, {4096}),
              ShapeUtil::MakeShape(BF16, {8192})},
             /*gather_dim=*/0},
            // blocks_per_peer > 1 with a different number of tiles per peer on
            // each root (4 and 2) and a private non-gather dimension.
            {"MixedTilesPerPeer",
             {ShapeUtil::MakeShape(F32, {4, 2048}),
              ShapeUtil::MakeShape(S32, {2, 1024})},
             /*gather_dim=*/0},
            // Regression test for schedule padding on variadic all-gather:
            // - Output 0 (`f32[8,4096]` from `f32[4,4096]`) uses 8 blocks per
            //   peer.
            // - Output 1 (`pred[4]` from `pred[2]`) only needs 2 blocks per
            //   peer, so the remaining blocks per peer are padded out of bounds
            //   and must be masked out.
            {"PaddedSmallOutput",
             {ShapeUtil::MakeShape(F32, {4, 4096}),
              ShapeUtil::MakeShape(PRED, {2})},
             /*gather_dim=*/0},
            // Same with 4 tiles per peer on the small output and a
            // non-power-of-two minor dimension, so masking covers both padding
            // and partial tiles.
            {"PaddedSmallOutputNonPowerOfTwo",
             {ShapeUtil::MakeShape(F32, {4, 4096}),
              ShapeUtil::MakeShape(S32, {2, 3})},
             /*gather_dim=*/0},
            // Real variadic all-gathers of the jit_generate decode step
            // (operands grouped by source op).
            {"JitGenerateStart2",
             {ShapeUtil::MakeShape(F32, {2, 4096}),
              ShapeUtil::MakeShape(S32, {2})},
             /*gather_dim=*/0},
            {"JitGenerateStart3",
             {ShapeUtil::MakeShape(S32, {2}),
              ShapeUtil::MakeShape(PRED, {2, 128}),
              ShapeUtil::MakeShape(S32, {2, 256}),
              ShapeUtil::MakeShape(PRED, {2})},
             /*gather_dim=*/0},
            {"JitGenerateStart4",
             {ShapeUtil::MakeShape(S32, {64, 32}),
              ShapeUtil::MakeShape(F32, {64, 32}),
              ShapeUtil::MakeShape(U32, {2, 2}),
              ShapeUtil::MakeShape(S32, {2, 4}), ShapeUtil::MakeShape(U32, {2}),
              ShapeUtil::MakeShape(U32, {2, 512, 2}),
              ShapeUtil::MakeShape(S32, {2})},
             /*gather_dim=*/0},
            {"JitGenerateStart25",
             {ShapeUtil::MakeShape(S32, {64, 8, 32}),
              ShapeUtil::MakeShape(F32, {64, 8, 32})},
             /*gather_dim=*/0},
            {"JitGenerateStart28",
             {ShapeUtil::MakeShape(S32, {2}), ShapeUtil::MakeShape(S32, {2, 8}),
              ShapeUtil::MakeShape(S32, {2, 529})},
             /*gather_dim=*/0},
        }),
        ::testing::Values(kOneShot, kTwoShot)),
    [](const ::testing::TestParamInfo<VariadicAllGatherTest::ParamType>& info) {
      return absl::StrCat(std::get<0>(info.param).name, "_",
                          StrategyTestSuffix(std::get<1>(info.param)));
    });

}  // namespace
}  // namespace xla

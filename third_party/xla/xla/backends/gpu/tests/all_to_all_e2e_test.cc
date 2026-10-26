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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class AllToAllTest : public CollectiveOpsWithFlagsBase {
 public:
  AllToAllTest()
      : CollectiveOpsWithFlagsBase(
            /*enable_async=*/true,
            /*enable_p2p_memcpy=*/false,
            /*enable_symmetric_buffer=*/true,
            /*memory_size=*/32 * kMB,
            /*collectives_memory_size=*/32 * kMB) {}

 protected:
  struct TwoGpuTestData {
    Literal input_r0;
    Literal input_r1;
    Literal expected_r0;
    Literal expected_r1;
  };

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
        DebugOptions::COLLECTIVE_KERNEL_ALL_TO_ALL);
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

  void VerifyOneShotAllToAll(const HloModule* optimized_module) {
    ASSERT_NE(optimized_module, nullptr);
    bool found_all_to_all = false;
    for (const HloComputation* comp : optimized_module->computations()) {
      for (const HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() == HloOpcode::kAllToAll) {
          found_all_to_all = true;
          ASSERT_OK_AND_ASSIGN(gpu::GpuBackendConfig gpu_config,
                               instr->backend_config<gpu::GpuBackendConfig>());
          EXPECT_EQ(
              gpu_config.collective_backend_config().kernel_strategy(),
              gpu::CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT)
              << "Expected AllToAll instruction " << instr->name()
              << " to use KERNEL_STRATEGY_TRITON_ONE_SHOT, but got: "
              << gpu::CollectiveBackendConfig::CollectiveKernelStrategy_Name(
                     gpu_config.collective_backend_config().kernel_strategy());
        }
      }
    }
    EXPECT_TRUE(found_all_to_all)
        << "Expected to find an AllToAll instruction in optimized HLO.";
  }

  absl::StatusOr<TwoGpuTestData> PrepareTwoGpuTestData(const Shape& shape,
                                                       int64_t split_dim = 0) {
    const int64_t split_size = shape.dimensions(split_dim);
    const int64_t chunk_size = split_size / 2;

    Shape full_shape = shape;
    full_shape.set_dimensions(split_dim, split_size * 2);
    ABSL_ASSIGN_OR_RETURN(Literal full_data, MakeFakeLiteral(full_shape));

    std::vector<int64_t> start_r0(shape.dimensions().size(), 0);
    std::vector<int64_t> limit_r0(shape.dimensions().begin(),
                                  shape.dimensions().end());
    std::vector<int64_t> start_r1 = start_r0;
    start_r1[split_dim] = split_size;
    std::vector<int64_t> limit_r1 = limit_r0;
    limit_r1[split_dim] = split_size * 2;

    Literal input_r0 = full_data.Slice(start_r0, limit_r0);
    Literal input_r1 = full_data.Slice(start_r1, limit_r1);

    std::vector<int64_t> chunk_extents(shape.dimensions().begin(),
                                       shape.dimensions().end());
    chunk_extents[split_dim] = chunk_size;
    std::vector<int64_t> offset_0(shape.dimensions().size(), 0);
    std::vector<int64_t> offset_1(shape.dimensions().size(), 0);
    offset_1[split_dim] = chunk_size;

    Literal expected_r0(shape);
    ABSL_RETURN_IF_ERROR(
        expected_r0.CopySliceFrom(input_r0, offset_0, offset_0, chunk_extents));
    ABSL_RETURN_IF_ERROR(
        expected_r0.CopySliceFrom(input_r1, offset_0, offset_1, chunk_extents));

    Literal expected_r1(shape);
    ABSL_RETURN_IF_ERROR(
        expected_r1.CopySliceFrom(input_r0, offset_1, offset_0, chunk_extents));
    ABSL_RETURN_IF_ERROR(
        expected_r1.CopySliceFrom(input_r1, offset_1, offset_1, chunk_extents));

    return TwoGpuTestData{std::move(input_r0), std::move(input_r1),
                          std::move(expected_r0), std::move(expected_r1)};
  }

  void RunTwoGpuAllToAllTest(const Shape& shape, int64_t split_dim = 0,
                             bool cross_partition = false) {
    constexpr int32_t kNumParticipants = 2;
    if (!CheckDeviceCount(kNumParticipants)) {
      return;
    }

    const std::string shape_str = ShapeUtil::HumanString(shape);
    const std::string module_header =
        cross_partition ? "HloModule test, replica_count=1, num_partitions=2"
                        : "HloModule test";
    const std::string channel_attr = cross_partition ? "channel_id=1, " : "";
    const std::string module_str = absl::StrFormat(
        R"(
        %s
        ENTRY test_computation {
          param_0 = %s parameter(0)
          ROOT all-to-all = %s all-to-all(param_0), dimensions={%d},
            %sreplica_groups={{0,1}}
        }
        )",
        module_header, shape_str, shape_str, split_dim, channel_attr);

    const int64_t num_replicas = cross_partition ? 1 : kNumParticipants;
    const int64_t num_partitions = cross_partition ? kNumParticipants : 1;
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseAndReturnVerifiedModule(module_str, num_replicas, num_partitions));

    ASSERT_OK_AND_ASSIGN(TwoGpuTestData data,
                         PrepareTwoGpuTestData(shape, split_dim));

    std::vector<std::vector<Literal*>> args = {{&data.input_r0},
                                               {&data.input_r1}};
    ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                         ExecuteReplicated(std::move(module), args));

    VerifyOneShotAllToAll(result.optimized_module);

    ASSERT_EQ(result.results.size(), kNumParticipants);
    EXPECT_TRUE(LiteralTestUtil::Equal(data.expected_r0, result.results[0]))
        << "Mismatch at rank 0";
    EXPECT_TRUE(LiteralTestUtil::Equal(data.expected_r1, result.results[1]))
        << "Mismatch at rank 1";
  }
};

struct AllToAll2GpuTestCase {
  std::string test_name;
  Shape shape;
  int64_t split_dim = 0;
  bool cross_partition = false;
};

class AllToAll2GpuShapeTest
    : public AllToAllTest,
      public ::testing::WithParamInterface<AllToAll2GpuTestCase> {};

TEST_P(AllToAll2GpuShapeTest, Run) {
  const AllToAll2GpuTestCase& tc = GetParam();
  RunTwoGpuAllToAllTest(tc.shape, tc.split_dim, tc.cross_partition);
}

INSTANTIATE_TEST_SUITE_P(
    AllToAll2GpuShapes, AllToAll2GpuShapeTest,
    ::testing::Values(
        // Basic 2-GPU all-to-all of f32[256] -> f32[256].
        AllToAll2GpuTestCase{"Basic2GpuF32", ShapeUtil::MakeShape(F32, {256})},
        // Cross-partition 2-GPU all-to-all (channel_id set, replica_count=1,
        // num_partitions=2) of f32[256] -> f32[256].
        AllToAll2GpuTestCase{"CrossPartition2GpuF32",
                             ShapeUtil::MakeShape(F32, {256}),
                             /*split_dim=*/0, /*cross_partition=*/true},
        // Larger 2-GPU all-to-all to test multi-tile behavior: f32[8192] ->
        // f32[8192].
        AllToAll2GpuTestCase{"Large2GpuF32", ShapeUtil::MakeShape(F32, {8192})},
        // 2D shape: f32[32, 32] -> f32[32, 32] (split along dim 0).
        AllToAll2GpuTestCase{"TwoDimensional2Gpu",
                             ShapeUtil::MakeShape(F32, {32, 32})},
        // 2D shape: f32[16, 64] -> f32[16, 64] (split along dim 1).
        AllToAll2GpuTestCase{"TwoDimensionalSplitDim1",
                             ShapeUtil::MakeShape(F32, {16, 64}),
                             /*split_dim=*/1},
        // 2D shape where the split dimension per rank is 1 (not divisible by
        // world_size = 2), so tile_sizes = [1, 64] and staging split_dim falls
        // back to dim 1.
        AllToAll2GpuTestCase{"TwoDimensionalSplitDimNotDivisibleByWorldSize",
                             ShapeUtil::MakeShape(F32, {2, 64})},
        // 1D sub-128-bit transfer (8 bytes per chunk): s32[4] -> s32[4].
        AllToAll2GpuTestCase{"SubTransactionSize1DS32",
                             ShapeUtil::MakeShape(S32, {4})},
        // 2D shape with unaligned non-split dim: s32[4, 7] -> s32[4, 7].
        AllToAll2GpuTestCase{"UnalignedNonSplitDim2DS32_4x7",
                             ShapeUtil::MakeShape(S32, {4, 7})},
        // 2D shape with larger odd non-split dim: s32[4, 529] -> s32[4, 529].
        AllToAll2GpuTestCase{"UnalignedNonSplitDim2DS32_4x529",
                             ShapeUtil::MakeShape(S32, {4, 529})}),
    [](const ::testing::TestParamInfo<AllToAll2GpuTestCase>& info) {
      return info.param.test_name;
    });

// Runs 9 consequent all-to-alls and verifies that by default all of them use
// XLA's one-shot all-to-all kernel and are placed into command buffers, and
// that excluding COLLECTIVES_KERNEL from --xla_gpu_enable_command_buffer
// disables command buffer capture for them.
TEST_F(AllToAllTest, NineConsequentAllToAllsWithCudaGraphs) {
  constexpr int32_t kNumReplicas = 2;
  constexpr int kNumAllToAlls = 9;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    p0 = f32[256] parameter(0)
    a0 = f32[256] all-to-all(p0), dimensions={0}, replica_groups={{0,1}}
    a1 = f32[256] all-to-all(a0), dimensions={0}, replica_groups={{0,1}}
    a2 = f32[256] all-to-all(a1), dimensions={0}, replica_groups={{0,1}}
    a3 = f32[256] all-to-all(a2), dimensions={0}, replica_groups={{0,1}}
    a4 = f32[256] all-to-all(a3), dimensions={0}, replica_groups={{0,1}}
    a5 = f32[256] all-to-all(a4), dimensions={0}, replica_groups={{0,1}}
    a6 = f32[256] all-to-all(a5), dimensions={0}, replica_groups={{0,1}}
    a7 = f32[256] all-to-all(a6), dimensions={0}, replica_groups={{0,1}}
    ROOT a8 = f32[256] all-to-all(a7), dimensions={0}, replica_groups={{0,1}}
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
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.set_xla_dump_to(default_dump_dir);

  ASSERT_OK_AND_ASSIGN(TwoGpuTestData data,
                       PrepareTwoGpuTestData(ShapeUtil::MakeShape(F32, {256})));
  std::vector<std::vector<Literal*>> args = {{&data.input_r0},
                                             {&data.input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllToAll(result.optimized_module);

  ASSERT_OK_AND_ASSIGN(
      CommandBufferThunkCounts one_shot,
      CountThunksInDump(default_dump_dir, "kCollectiveKernel"));
  EXPECT_EQ(one_shot.in_command_buffer, kNumAllToAlls);
  EXPECT_EQ(one_shot.outside_command_buffer, 0);
  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts nccl,
                       CountThunksInDump(default_dump_dir, "kAllToAll"));
  EXPECT_EQ(nccl.in_command_buffer + nccl.outside_command_buffer, 0);

  // An odd number (9) of all-to-alls produces the transposed result.
  ASSERT_EQ(result.results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(data.expected_r0, result.results[0]))
      << "Mismatch at replica 0";
  EXPECT_TRUE(LiteralTestUtil::Equal(data.expected_r1, result.results[1]))
      << "Mismatch at replica 1";

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
  no_graph_debug_options.set_xla_dump_to(no_graph_dump_dir);

  ASSERT_OK_AND_ASSIGN(ExecutionResult no_graph_result,
                       ExecuteReplicated(std::move(no_graph_module), args));
  VerifyOneShotAllToAll(no_graph_result.optimized_module);

  ASSERT_OK_AND_ASSIGN(
      CommandBufferThunkCounts one_shot_no_graph,
      CountThunksInDump(no_graph_dump_dir, "kCollectiveKernel"));
  EXPECT_EQ(one_shot_no_graph.in_command_buffer, 0);
  EXPECT_EQ(one_shot_no_graph.outside_command_buffer, kNumAllToAlls);
  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts nccl_no_graph,
                       CountThunksInDump(no_graph_dump_dir, "kAllToAll"));
  EXPECT_EQ(
      nccl_no_graph.in_command_buffer + nccl_no_graph.outside_command_buffer,
      0);
  ASSERT_EQ(no_graph_result.results.size(), kNumReplicas);
  EXPECT_TRUE(
      LiteralTestUtil::Equal(data.expected_r0, no_graph_result.results[0]))
      << "Mismatch at replica 0 without graphs";
  EXPECT_TRUE(
      LiteralTestUtil::Equal(data.expected_r1, no_graph_result.results[1]))
      << "Mismatch at replica 1 without graphs";
}

// Runs 20 one-shot all-to-alls inside a 20-iteration while loop with WHILE
// command buffers enabled, and verifies that the while thunk and all 20
// collective kernels are placed into command buffers and produce the expected
// result across repeated executions.
TEST_F(AllToAllTest, TwentyAllToAllsInWhileLoopWithCudaGraphs) {
  constexpr int32_t kNumReplicas = 2;
  constexpr int kNumAllToAlls = 20;
  constexpr int kNumIterations = 20;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  std::string body_ops;
  std::string prev = "acc";
  for (int k = 0; k < kNumAllToAlls; ++k) {
    absl::StrAppendFormat(
        &body_ops,
        "    scaled%d = f32[256] multiply(%s, half)\n"
        "    in%d = f32[256] add(scaled%d, p0)\n"
        "    a2a%d = f32[256] all-to-all(in%d), dimensions={0}, "
        "replica_groups={{0,1}}\n"
        "    s0_%d = f32[128] slice(a2a%d), slice={[0:128]}\n"
        "    s1_%d = f32[128] slice(a2a%d), slice={[128:256]}\n"
        "    half_sum%d = f32[128] add(s0_%d, s1_%d)\n"
        "    sum%d = f32[256] concatenate(half_sum%d, half_sum%d), "
        "dimensions={0}\n",
        k, prev, k, k, k, k, k, k, k, k, k, k, k, k, k, k);
    prev = absl::StrCat("sum", k);
  }

  const std::string module_str = absl::StrFormat(
      R"(
  HloModule test

  cond {
    state = (s32[], f32[256], f32[256]) parameter(0)
    i = s32[] get-tuple-element(state), index=0
    limit = s32[] constant(%d)
    ROOT cmp = pred[] compare(i, limit), direction=LT
  }

  body {
    state = (s32[], f32[256], f32[256]) parameter(0)
    i = s32[] get-tuple-element(state), index=0
    acc = f32[256] get-tuple-element(state), index=1
    p0 = f32[256] get-tuple-element(state), index=2
    one = s32[] constant(1)
    next_i = s32[] add(i, one)
    half_scalar = f32[] constant(0.5)
    half = f32[256] broadcast(half_scalar), dimensions={}
%s    ROOT next_state = (s32[], f32[256], f32[256]) tuple(next_i, %s, p0)
  }

  ENTRY test_computation {
    p0 = f32[256] parameter(0)
    zero_i = s32[] constant(0)
    zero_f = f32[] constant(0.0)
    init_acc = f32[256] broadcast(zero_f), dimensions={}
    init = (s32[], f32[256], f32[256]) tuple(zero_i, init_acc, p0)
    loop = (s32[], f32[256], f32[256]) while(init), condition=cond, body=body
    ROOT out = f32[256] get-tuple-element(loop), index=1
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
  debug_options.set_xla_gpu_enable_while_loop_unrolling(
      DebugOptions::WHILE_LOOP_UNROLLING_NO_UNROLL);
  debug_options.set_xla_dump_to(dump_dir.path());

  Literal input_r0 =
      LiteralUtil::CreateR1<float>(std::vector<float>(256, 1.0f));
  Literal input_r1 =
      LiteralUtil::CreateR1<float>(std::vector<float>(256, 2.0f));
  std::vector<std::vector<Literal*>> args = {{&input_r0}, {&input_r1}};
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));

  VerifyOneShotAllToAll(result.optimized_module);

  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts while_counts,
                       CountThunksInDump(dump_dir.path(), "kWhile"));
  EXPECT_EQ(while_counts.in_command_buffer, 1);
  EXPECT_EQ(while_counts.outside_command_buffer, 0);
  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts one_shot,
                       CountThunksInDump(dump_dir.path(), "kCollectiveKernel"));
  EXPECT_EQ(one_shot.in_command_buffer, kNumAllToAlls);
  EXPECT_EQ(one_shot.outside_command_buffer, 0);
  ASSERT_OK_AND_ASSIGN(CommandBufferThunkCounts nccl,
                       CountThunksInDump(dump_dir.path(), "kAllToAll"));
  EXPECT_EQ(nccl.in_command_buffer + nccl.outside_command_buffer, 0);

  // 20 iterations * 20 all-to-alls * 3.0 per all-to-all = 1200.0.
  const float expected_val = 3.0f * kNumAllToAlls * kNumIterations;
  Literal expected =
      LiteralUtil::CreateR1<float>(std::vector<float>(256, expected_val));
  ASSERT_EQ(result.results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result.results[i]))
        << "Mismatch at replica " << i << " on first execution";
  }

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> second_results,
                       ExecuteReplicated(result.executable.get(), args));
  ASSERT_EQ(second_results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, second_results[i]))
        << "Mismatch at replica " << i << " on second execution";
  }
}

// Repeated invocations on the same executable to verify double-buffering
// across alternating slots (signal_value & 1).
TEST_F(AllToAllTest, RepeatedInvocationsDoubleBuffering) {
  constexpr int32_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[256] parameter(0)
    ROOT all-to-all = f32[256] all-to-all(param_0), dimensions={0},
      replica_groups={{0,1}}
  }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  Literal input1_r0 =
      LiteralUtil::CreateR1<float>(std::vector<float>(256, 1.0f));
  Literal input1_r1 =
      LiteralUtil::CreateR1<float>(std::vector<float>(256, 2.0f));
  std::vector<std::vector<Literal*>> args1 = {{&input1_r0}, {&input1_r1}};

  ASSERT_OK_AND_ASSIGN(ExecutionResult result1,
                       ExecuteReplicated(std::move(module), args1));
  VerifyOneShotAllToAll(result1.optimized_module);

  for (int iter = 2; iter <= 4; ++iter) {
    float v00 = static_cast<float>(iter * 10 + 1);
    float v01 = static_cast<float>(iter * 10 + 2);
    float v10 = static_cast<float>(iter * 10 + 3);
    float v11 = static_cast<float>(iter * 10 + 4);
    std::vector<float> d_r0(256), d_r1(256);
    std::vector<float> exp_r0(256), exp_r1(256);
    for (int i = 0; i < 128; ++i) {
      d_r0[i] = v00;
      d_r0[i + 128] = v01;
      d_r1[i] = v10;
      d_r1[i + 128] = v11;
      exp_r0[i] = v00;
      exp_r0[i + 128] = v10;
      exp_r1[i] = v01;
      exp_r1[i + 128] = v11;
    }
    Literal in_r0 = LiteralUtil::CreateR1<float>(d_r0);
    Literal in_r1 = LiteralUtil::CreateR1<float>(d_r1);
    std::vector<std::vector<Literal*>> iter_args = {{&in_r0}, {&in_r1}};

    ASSERT_OK_AND_ASSIGN(
        std::vector<Literal> iter_results,
        ExecuteReplicated(result1.executable.get(), iter_args));
    ASSERT_EQ(iter_results.size(), kNumReplicas);

    Literal expected_r0 = LiteralUtil::CreateR1<float>(exp_r0);
    Literal expected_r1 = LiteralUtil::CreateR1<float>(exp_r1);
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_r0, iter_results[0]))
        << "Mismatch at replica 0 on iteration " << iter;
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_r1, iter_results[1]))
        << "Mismatch at replica 1 on iteration " << iter;
  }
}

class AllToAllTypesTest : public AllToAllTest,
                          public ::testing::WithParamInterface<PrimitiveType> {
};

// Same shape as Large2GpuF32, for every element type supported by the
// one-shot kernel.
TEST_P(AllToAllTypesTest, Large2Gpu) {
  RunTwoGpuAllToAllTest(ShapeUtil::MakeShape(GetParam(), {8192}));
}

INSTANTIATE_TEST_SUITE_P(
    AllToAllTypes, AllToAllTypesTest,
    ::testing::Values(PRED, F16, BF16, F32, F64, S8, S16, S32, S64, U8, U16,
                      U32, U64),
    [](const ::testing::TestParamInfo<PrimitiveType>& info) {
      return std::string(
          primitive_util::LowercasePrimitiveTypeName(info.param));
    });

}  // namespace
}  // namespace xla

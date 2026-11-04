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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/index_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

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

struct VariadicReduceScatterTestCase {
  std::string name;
  PrimitiveType element_type;
  // Full (unscattered) operand shapes; dimension 0 is scattered.
  std::vector<Shape> operand_shapes;
  // HLO opcode name of the reducer: add, maximum or or.
  std::string reducer;
};

class VariadicReduceScatterTest
    : public ReduceScatterTest,
      public ::testing::WithParamInterface<VariadicReduceScatterTestCase> {};

TEST_P(VariadicReduceScatterTest, TwoGpus) {
  constexpr int32_t kNumReplicas = 2;
  if (!HasDevices(kNumReplicas)) {
    GTEST_SKIP() << "Requires " << kNumReplicas << " devices.";
  }
  const VariadicReduceScatterTestCase& tc = GetParam();
  const int64_t num_operands = tc.operand_shapes.size();
  const std::string type_name =
      primitive_util::LowercasePrimitiveTypeName(tc.element_type);

  std::vector<Shape> output_shapes;
  std::vector<std::string> param_lines;
  std::vector<std::string> operand_names;
  for (int64_t k = 0; k < num_operands; ++k) {
    const Shape& in = tc.operand_shapes[k];
    ASSERT_EQ(in.element_type(), tc.element_type);
    ASSERT_EQ(in.dimensions(0) % kNumReplicas, 0);
    Shape out = in;
    out.set_dimensions(0, in.dimensions(0) / kNumReplicas);
    output_shapes.push_back(out);
    param_lines.push_back(absl::StrFormat("param_%d = %s parameter(%d)", k,
                                          ShapeUtil::HumanString(in), k));
    operand_names.push_back(absl::StrFormat("param_%d", k));
  }
  const std::string module_str = absl::StrFormat(
      R"(
  HloModule test
  reducer {
    a = %s[] parameter(0)
    b = %s[] parameter(1)
    ROOT r = %s[] %s(a, b)
  }
  ENTRY test_computation {
    %s
    ROOT rs = %s reduce-scatter(%s), dimensions={0},
      replica_groups={{0,1}}, to_apply=reducer
  }
  )",
      type_name, type_name, type_name, tc.reducer,
      absl::StrJoin(param_lines, "\n    "),
      ShapeUtil::HumanString(ShapeUtil::MakeTupleShape(output_shapes)),
      absl::StrJoin(operand_names, ", "));
  SCOPED_TRACE(module_str);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(module_str, kNumReplicas));

  // Small non-negative integers: exactly representable (and exactly reducible)
  // in every tested element type. Booleans use {0, 1}.
  const float modulus = tc.element_type == PRED ? 2.0f : 9.0f;
  auto value = [&](int rank, int64_t k, const Shape& shape,
                   absl::Span<const int64_t> idx) -> float {
    const int64_t linear = IndexUtil::MultidimensionalIndexToLinearIndex(
        ShapeUtil::MakeShape(F32, shape.dimensions()), idx);
    return static_cast<float>((linear * 7 + rank * 3 + k * 5) %
                              static_cast<int64_t>(modulus));
  };
  auto reduce = [&](float a, float b) -> float {
    if (tc.reducer == "add") {
      return a + b;
    }
    if (tc.reducer == "maximum") {
      return std::max(a, b);
    }
    CHECK_EQ(tc.reducer, "or");
    return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f;
  };

  std::vector<std::vector<Literal>> inputs(kNumReplicas);
  for (int64_t k = 0; k < num_operands; ++k) {
    const Shape& in_shape = tc.operand_shapes[k];
    for (int r = 0; r < kNumReplicas; ++r) {
      ASSERT_OK_AND_ASSIGN(Literal in, Literal::Make(ShapeUtil::MakeShape(
                                           F32, in_shape.dimensions())));
      ASSERT_OK(in.Populate<float>([&](absl::Span<const int64_t> idx) {
        return value(r, k, in_shape, idx);
      }));
      ASSERT_OK_AND_ASSIGN(Literal converted, in.Convert(tc.element_type));
      inputs[r].push_back(std::move(converted));
    }
  }
  // Replica r receives part r of every operand, reduced over all replicas.
  std::vector<std::vector<Literal>> expected_per_replica(kNumReplicas);
  for (int r = 0; r < kNumReplicas; ++r) {
    for (int64_t k = 0; k < num_operands; ++k) {
      const Shape& in_shape = tc.operand_shapes[k];
      const Shape& out_shape = output_shapes[k];
      const int64_t per_rank_size = out_shape.dimensions(0);
      ASSERT_OK_AND_ASSIGN(Literal out, Literal::Make(ShapeUtil::MakeShape(
                                            F32, out_shape.dimensions())));
      ASSERT_OK(out.Populate<float>([&](absl::Span<const int64_t> idx) {
        std::vector<int64_t> in_idx(idx.begin(), idx.end());
        in_idx[0] += r * per_rank_size;
        float acc = value(/*rank=*/0, k, in_shape, in_idx);
        for (int p = 1; p < kNumReplicas; ++p) {
          acc = reduce(acc, value(p, k, in_shape, in_idx));
        }
        return acc;
      }));
      ASSERT_OK_AND_ASSIGN(Literal converted, out.Convert(tc.element_type));
      expected_per_replica[r].push_back(std::move(converted));
    }
  }

  std::vector<std::vector<Literal*>> args(kNumReplicas);
  for (int r = 0; r < kNumReplicas; ++r) {
    for (Literal& input : inputs[r]) {
      args[r].push_back(&input);
    }
  }
  ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                       ExecuteReplicated(std::move(module), args));
  VerifyOneShotReduceScatter(result.optimized_module);
  ASSERT_EQ(result.results.size(), kNumReplicas);
  for (int r = 0; r < kNumReplicas; ++r) {
    Literal expected_tuple =
        LiteralUtil::MakeTupleOwned(std::move(expected_per_replica[r]));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_tuple, result.results[r]))
        << "Mismatch at replica " << r;
  }
}

INSTANTIATE_TEST_SUITE_P(
    VariadicReduceScatter, VariadicReduceScatterTest,
    ::testing::ValuesIn<VariadicReduceScatterTestCase>({
        {"TwoF32SameShape",
         F32,
         {ShapeUtil::MakeShape(F32, {256}), ShapeUtil::MakeShape(F32, {256})},
         "add"},
        {"TwoF32DifferentSizes",
         F32,
         {ShapeUtil::MakeShape(F32, {4096}), ShapeUtil::MakeShape(F32, {8192})},
         "add"},
        {"ThreeOperandsMixedRanks",
         F32,
         {ShapeUtil::MakeShape(F32, {64}), ShapeUtil::MakeShape(F32, {4, 32}),
          ShapeUtil::MakeShape(F32, {2, 8, 8})},
         "add"},
        {"Bf16Add",
         BF16,
         {ShapeUtil::MakeShape(BF16, {2, 1024}),
          ShapeUtil::MakeShape(BF16, {2, 16})},
         "add"},
        // Non-power-of-two inner sizes (1200 and 10 elements per rank).
        {"S32MaxNonPowerOfTwo",
         S32,
         {ShapeUtil::MakeShape(S32, {8, 300}),
          ShapeUtil::MakeShape(S32, {4, 5})},
         "maximum"},
        {"PredOr",
         PRED,
         {ShapeUtil::MakeShape(PRED, {32, 128}),
          ShapeUtil::MakeShape(PRED, {4})},
         "or"},
        // Shapes of a real variadic reduce-scatter of the jit_generate decode
        // step.
        {"JitGenerate14",
         F32,
         {ShapeUtil::MakeShape(F32, {4, 8, 64}),
          ShapeUtil::MakeShape(F32, {4, 8, 1})},
         "add"},
        // Regression test for schedule padding on variadic reduce-scatter:
        // - Output 0 (`f32[2,4096]` from `f32[4,4096]`) uses 2 blocks per rank.
        // - Output 1 (`f32[2,1]` from `f32[4,1]`) only needs 1 block per rank,
        //   so the second block per rank is padded out of bounds and must be
        //   masked out.
        {"PaddedSmallOutput",
         F32,
         {ShapeUtil::MakeShape(F32, {4, 4096}),
          ShapeUtil::MakeShape(F32, {4, 1})},
         "add"},
        // 12288 elements per rank: canonicalized to [R, 32, 384] so that the
        // 4 programs per output row split the middle dimension ([1, 8, 512]
        // tiles) instead of rounding 12288 up to a single power of two.
        {"F32AddNonPowerOfTwoLarge",
         F32,
         {ShapeUtil::MakeShape(F32, {2, 24576}),
          ShapeUtil::MakeShape(F32, {4, 2})},
         "add"},
    }),
    [](const ::testing::TestParamInfo<VariadicReduceScatterTestCase>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace xla

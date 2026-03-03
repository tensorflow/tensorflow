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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::xla::se::gpu::AllReduceStrategy;

std::string GetAsyncTestName(bool is_async) {
  return is_async ? "async" : "sync";
}

void VerifyAllReduceType(const HloModule* module, PrimitiveType expected_type) {
  bool found = false;
  for (auto* comp : module->computations()) {
    for (auto* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kAllReduce ||
          instr->opcode() == HloOpcode::kAllReduceStart) {
        PrimitiveType actual_type = instr->operand(0)->shape().element_type();
        ASSERT_EQ(actual_type, expected_type)
            << "Expected AllReduce type " << PrimitiveType_Name(expected_type)
            << " but got " << PrimitiveType_Name(actual_type);
        found = true;
      }
    }
  }
  ASSERT_TRUE(found) << "No AllReduce found in module";
}

class AllReduceTestNoParams : public CollectiveOpsWithFlagsBase {
 public:
  explicit AllReduceTestNoParams(bool is_async = false)
      : CollectiveOpsWithFlagsBase(/*enable_async=*/is_async,
                                   /*enable_p2p_memcpy=*/false,
                                   /*memory_size=*/32 * kMB,
                                   /*collectives_memory_size=*/0) {}

  void SetUp() override {
    CollectiveOpsE2ETestBase::SetUp();
    // Check for Triton support: Ampere+ for CUDA, any supported GPU for ROCm
    if (Capability().IsCuda() && !IsAmpereAndHigher()) {
      GTEST_SKIP() << "Test requires Ampere or newer architecture for CUDA "
                      "since it's using triton.";
    }
  }
};

struct AllReduceTestParams {
  // If true uses the async stream for the collective.
  bool is_async;
  // If true, uses the XLA generated kernel for all-reduce.
  // If false, uses the NCCL kernel.
  bool use_all_reduce_one_shot_kernel;
  // The strategy to use for the all-reduce.
  // The strategy determines the size of the supported inputs.
  se::gpu::AllReduceStrategy strategy;

  // Returns the range of supported element sizes for the given element type and
  // strategy.
  std::pair<int64_t, int64_t> RangeElements(PrimitiveType element_type) const {
    const bool is_two_shot = strategy == AllReduceStrategy::kTwoShot;
    int64_t min_supported_size = is_two_shot
                                     ? gpu::GetMaxSupportedAllReduceSizeBytes(
                                           AllReduceStrategy::kOneShot)
                                     : 0;
    int64_t element_size = ShapeUtil::ByteSizeOfPrimitiveType(element_type);
    int64_t min_elements = min_supported_size / element_size;
    int64_t max_elements =
        gpu::GetMaxSupportedAllReduceSizeBytes(strategy) / element_size;
    return {min_elements, max_elements};
  }

  // Returns a number of elements in the range of supported element sizes
  // for the given element type.
  // The fact that this is in the midpoint is arbitrary.
  int64_t NumElements(PrimitiveType element_type) const {
    auto [min_elements, max_elements] = RangeElements(element_type);
    return max_elements - (max_elements - min_elements) / 2;
  }

  static std::vector<AllReduceTestParams> Generate() {
    std::vector<AllReduceTestParams> params;
    for (bool is_async : {true, false}) {
      for (bool use_all_reduce_one_shot_kernel : {true, false}) {
        for (auto strategy :
             {AllReduceStrategy::kOneShot, AllReduceStrategy::kTwoShot}) {
          params.push_back(
              {is_async, use_all_reduce_one_shot_kernel, strategy});
        }
      }
    }
    return params;
  }

  // Convert to string for test naming.
  // NB: This method is used by GTest to generate test names.
  // Without this method, it will generate a byte string which is hard to read.
  // Adding [[maybe_unused]] suppresses the clang unused function warning.
  [[maybe_unused]] friend void PrintTo(const AllReduceTestParams& params,
                                       std::ostream* os) {
    *os << "{ .is_async=" << params.is_async
        << ", .use_all_reduce_one_shot_kernel="
        << params.use_all_reduce_one_shot_kernel
        << ", .strategy=" << absl::StrFormat("%v", params.strategy) << " }";
  }
};

class AllReduceTest
    : public AllReduceTestNoParams,
      public ::testing::WithParamInterface<AllReduceTestParams> {
 public:
  struct InputsOutputs {
    std::vector<Literal> inputs;
    std::vector<Literal> expected_outputs;

    [[nodiscard]] std::vector<std::vector<Literal*>> InputLiteralPtrs() {
      std::vector<std::vector<Literal*>> result;
      for (auto& input : inputs) {
        result.push_back(std::vector<Literal*>{&input});
      }
      return result;
    }
  };

  AllReduceTest() : AllReduceTestNoParams(/*is_async=*/GetParam().is_async) {}

 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts = CollectiveOpsWithFlagsBase::GetDebugOptionsForTest();
    opts.set_xla_gpu_unsupported_use_all_reduce_one_shot_kernel(
        GetParam().use_all_reduce_one_shot_kernel);
    return opts;
  }

  template <PrimitiveType kElementType, HloOpcode kHloOpcode>
  static absl::StatusOr<InputsOutputs> BuildTestInputsOutputs(
      const HloModule& module, int64_t num_replicas, int64_t num_iterations) {
    using ElementType = primitive_util::NativeTypeOf<kElementType>;

    std::vector<Array<ElementType>> inputs;
    std::vector<Literal> input_literals;
    const HloInstruction* const hlo_instr =
        FindInstruction(&module, HloOpcode::kAllReduce);
    if (hlo_instr == nullptr) {
      return absl::InvalidArgumentError(
          "Instruction 'all-reduce' not found in module.");
    }
    const HloAllReduceInstruction* instr =
        Cast<HloAllReduceInstruction>(hlo_instr);
    const int64_t num_elements = Product(
        module.entry_computation()->root_instruction()->shape().dimensions());
    for (int i = 0; i < num_replicas; ++i) {
      auto& input = inputs.emplace_back(Array<ElementType>({num_elements}));
      if constexpr (std::is_same_v<ElementType, bool>) {
        input.FillRandomBool(/*seed=*/i);
      } else {
        input.FillRandom(1.0f, 10.0f, /*seed=*/i);
      }
      input_literals.push_back(LiteralUtil::CreateFromArray(input));
    }
    std::vector<Literal> expected_output_literals;

    const std::vector<ReplicaGroup>& replica_groups =
        instr->device_list().replica_groups();
    // Map each device to set of replica groups it belongs to.
    std::vector<std::vector<int64_t>> device_to_groups(num_replicas);
    for (const auto& replica_group : replica_groups) {
      const auto& replica_ids = replica_group.replica_ids();
      for (int64_t replica : replica_group.replica_ids()) {
        CHECK_EQ(device_to_groups[replica].size(), 0);
        device_to_groups[replica].assign(replica_ids.begin(),
                                         replica_ids.end());
      }
    }
    std::vector<Array<ElementType>> expected_outputs(
        num_replicas, Array<ElementType>({num_elements}));
    // Aggregate inputs from each replica group
    for (int i = 0; i < num_replicas; ++i) {
      expected_outputs[i].Each(
          [&](absl::Span<const int64_t> indices, ElementType* val) {
            for (const int64_t replica : device_to_groups[i]) {
              if constexpr (kHloOpcode == HloOpcode::kAdd) {
                *val += inputs[replica](indices);
              } else if (kHloOpcode == HloOpcode::kOr) {
                *val |= inputs[replica](indices);
              } else {
                GTEST_FAIL() << "Unsupported reduction kind: "
                             << absl::StrFormat("%v", kHloOpcode);
              }
            }
            if constexpr (kHloOpcode == HloOpcode::kAdd) {
              // Each iteration after the first,the output is doubled.
              // For kOr, it remains the same.
              *val *= std::pow(device_to_groups[i].size(), num_iterations - 1);
            }
          });
    }
    for (auto& expected_output : expected_outputs) {
      expected_output_literals.push_back(
          LiteralUtil::CreateFromArray(expected_output));
    }
    return InputsOutputs{std::move(input_literals),
                         std::move(expected_output_literals)};
  }
};

INSTANTIATE_TEST_SUITE_P(
    AllReduceTest, AllReduceTest,
    ::testing::ValuesIn(AllReduceTestParams::Generate()),
    [](const ::testing::TestParamInfo<AllReduceTestParams>& info) {
      return absl::StrCat(
          GetAsyncTestName(info.param.is_async), "_",
          info.param.use_all_reduce_one_shot_kernel ? "xla" : "nccl", "_",
          info.param.strategy);
    });

TEST_P(AllReduceTest, Pred_2GPUs) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  apply_op {
    x = pred[] parameter(0)
    y = pred[] parameter(1)
    ROOT apply_op = pred[] or(x, y)
  }

  ENTRY test_computation {
    param_0 = pred[%1$d] parameter(0)
    ROOT all-reduce = pred[%1$d] all-reduce(param_0), to_apply=apply_op, replica_groups={{0,1}}
  }
  )";
  const int64_t num_elements = GetParam().NumElements(PrimitiveType::PRED);
  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(absl::StrFormat(kModuleStr, num_elements),
                                   kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      (BuildTestInputsOutputs<PrimitiveType::PRED, HloOpcode::kOr>(
          *module, kNumReplicas, /*num_iterations=*/1)));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()))
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Equal(test_io.expected_outputs[i], results[i]))
        << "ExpectedOutput != Result at index " << i;
  }
}

TEST_P(AllReduceTest, F32_8GPUs_AllReplicasOneGroup) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY test_computation {
    param_0 = f32[%1$d] parameter(0)
    ROOT all-reduce = f32[%1$d] all-reduce(param_0), to_apply=apply_op,
      replica_groups={{0,1,2,3,4,5,6,7}}
  }
  )";

  const int64_t kNumReplicas = 8;
  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }

  const int64_t num_elements = GetParam().NumElements(PrimitiveType::F32);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(absl::StrFormat(kModuleStr, num_elements),
                                   kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      (BuildTestInputsOutputs<PrimitiveType::F32, HloOpcode::kAdd>(
          *module, kNumReplicas, /*num_iterations=*/1)));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()))
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    // NB: nccl accumulation order can be different from expected calculations
    // leading to differences in the results (floating point imprecision).
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-4}))
        << "ExpectedOutput != Result at index " << i;
  }
}

TEST_P(AllReduceTest, F32_8GPUs_2ReplicasPerGroup) {
  const int64_t num_elements = GetParam().NumElements(PrimitiveType::F32);
  const int64_t kNumIterations = 3;
  const auto kModuleStr = absl::StrFormat(
      R"(
  HloModule test

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  while_condition {
    limit = s32[] constant(%1$d)
    params = (s32[], f32[%2$d]{0}) parameter(0)
    loop_counter = get-tuple-element(params), index=0
    ROOT result = pred[] compare(loop_counter, limit), direction=LT
  }

  while_body {
    params = (s32[], f32[%2$d]{0}) parameter(0)
    loop_counter = get-tuple-element(params), index=0
    tensor = get-tuple-element(params), index=1
    out0 = f32[%2$d] all-reduce(tensor), to_apply=apply_op,
      replica_groups={{0,4},{1,5},{2,6},{3,7}}
    new_loop_counter = s32[] add(loop_counter, s32[] constant(1))
    ROOT result = (s32[], f32[%2$d]{0}) tuple(new_loop_counter, out0)
  }

  ENTRY test_computation {
    param_0 = f32[%2$d] parameter(0)
    while_init = (s32[], f32[%2$d]{0}) tuple(s32[] constant(0), param_0)
    while_result = (s32[], f32[%2$d]{0})
      while(while_init), condition=while_condition, body=while_body
    ROOT result = get-tuple-element(while_result), index=1
  }
  )",
      kNumIterations, num_elements);

  const int64_t kNumReplicas = 8;
  if (device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << device_count() << " available)";
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      (BuildTestInputsOutputs<PrimitiveType::F32, HloOpcode::kAdd>(
          *module, kNumReplicas, kNumIterations)));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Equal(test_io.expected_outputs[i], results[i]))
        << "ExpectedOutput != Result at index " << i;
  }
}

// FP8 vs FP16 training step comparison.
TEST_F(AllReduceTestNoParams, AsyncAllReduce_F8E4M3FN_TrainingStep_2GPUs) {
  bool has_fp8_support = false;
  if (Capability().IsCuda()) {
    has_fp8_support = Capability().cuda_compute_capability()->IsAtLeast(9, 0);
  } else if (Capability().IsRocm()) {
    has_fp8_support =
        Capability().rocm_compute_capability()->has_ocp_fp8_support();
  }

  if (!has_fp8_support) {
    GTEST_SKIP() << "FP8 requires GPU with OCP FP8 support (CUDA Hopper+ or "
                    "ROCm MI350/gfx12xx with ROCm 7.0+).";
  }

  // FP16 baseline
  const absl::string_view kF16ModuleStr = R"(
  HloModule f16_training_step
  add_f16 { x = f16[] parameter(0)  y = f16[] parameter(1)  ROOT add = f16[] add(x, y) }
  ENTRY training_step {
    activations = f16[32,64] parameter(0)
    weights = f16[64,128] parameter(1)
    upstream_grad = f16[32,128] parameter(2)
    lr = f16[] parameter(3)
    output = f16[32,128] dot(activations, weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    activations_t = f16[64,32] transpose(activations), dimensions={1,0}
    weight_grad = f16[64,128] dot(activations_t, upstream_grad), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    weight_grad_allreduce = f16[64,128] all-reduce(weight_grad), to_apply=add_f16, replica_groups={{0,1}}
    two = f16[] constant(2)
    two_bcast = f16[64,128] broadcast(two), dimensions={}
    weight_grad_avg = f16[64,128] divide(weight_grad_allreduce, two_bcast)
    lr_bcast = f16[64,128] broadcast(lr), dimensions={}
    weight_update = f16[64,128] multiply(lr_bcast, weight_grad_avg)
    new_weights = f16[64,128] subtract(weights, weight_update)
    ROOT result = (f16[32,128], f16[64,128]) tuple(output, new_weights)
  })";

  // FP8 version
  const absl::string_view kF8ModuleStr = R"(
  HloModule fp8_training_step
  add_f8 { x = f8e4m3fn[] parameter(0)  y = f8e4m3fn[] parameter(1)  ROOT add = f8e4m3fn[] add(x, y) }
  ENTRY training_step {
    activations = f16[32,64] parameter(0)
    weights = f16[64,128] parameter(1)
    upstream_grad = f16[32,128] parameter(2)
    lr = f16[] parameter(3)
    output = f16[32,128] dot(activations, weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    activations_t = f16[64,32] transpose(activations), dimensions={1,0}
    weight_grad = f16[64,128] dot(activations_t, upstream_grad), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    weight_grad_f8 = f8e4m3fn[64,128] convert(weight_grad)
    weight_grad_allreduce_f8 = f8e4m3fn[64,128] all-reduce(weight_grad_f8), to_apply=add_f8, replica_groups={{0,1}}
    weight_grad_allreduce = f16[64,128] convert(weight_grad_allreduce_f8)
    two = f16[] constant(2)
    two_bcast = f16[64,128] broadcast(two), dimensions={}
    weight_grad_avg = f16[64,128] divide(weight_grad_allreduce, two_bcast)
    lr_bcast = f16[64,128] broadcast(lr), dimensions={}
    weight_update = f16[64,128] multiply(lr_bcast, weight_grad_avg)
    new_weights = f16[64,128] subtract(weights, weight_update)
    ROOT result = (f16[32,128], f16[64,128]) tuple(output, new_weights)
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  Array<Eigen::half> activations1({32, 64}), activations2({32, 64});
  activations1.FillRandom(Eigen::half(0.1f), 0.5f, /*seed=*/0);
  activations2.FillRandom(Eigen::half(0.1f), 0.5f, /*seed=*/1);
  Array<Eigen::half> weights({64, 128});
  weights.FillRandom(Eigen::half(0.1f), 0.3f, /*seed=*/42);
  Array<Eigen::half> upstream_grad1({32, 128}), upstream_grad2({32, 128});
  upstream_grad1.FillRandom(Eigen::half(0.01f), 0.1f, /*seed=*/100);
  upstream_grad2.FillRandom(Eigen::half(0.01f), 0.1f, /*seed=*/101);

  Literal lr = LiteralUtil::CreateR0<Eigen::half>(Eigen::half(0.01f));
  Literal activations_lit1 = LiteralUtil::CreateFromArray(activations1);
  Literal activations_lit2 = LiteralUtil::CreateFromArray(activations2);
  Literal weights_lit = LiteralUtil::CreateFromArray(weights);
  Literal upstream_grad_lit1 = LiteralUtil::CreateFromArray(upstream_grad1);
  Literal upstream_grad_lit2 = LiteralUtil::CreateFromArray(upstream_grad2);

  TF_ASSERT_OK_AND_ASSIGN(auto f16_module, ParseAndReturnVerifiedModule(
                                               kF16ModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult f16_result,
      ExecuteReplicated(
          std::move(f16_module),
          std::vector<std::vector<Literal*>>{
              {&activations_lit1, &weights_lit, &upstream_grad_lit1, &lr},
              {&activations_lit2, &weights_lit, &upstream_grad_lit2, &lr}}));
  // Verify FP16 all-reduce type in optimized module
  VerifyAllReduceType(f16_result.optimized_module, F16);

  TF_ASSERT_OK_AND_ASSIGN(
      auto f8_module, ParseAndReturnVerifiedModule(kF8ModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult f8_result,
      ExecuteReplicated(
          std::move(f8_module),
          std::vector<std::vector<Literal*>>{
              {&activations_lit1, &weights_lit, &upstream_grad_lit1, &lr},
              {&activations_lit2, &weights_lit, &upstream_grad_lit2, &lr}}));
  // Verify FP8 all-reduce type in optimized module
  VerifyAllReduceType(f8_result.optimized_module, F8E4M3FN);

  ASSERT_EQ(f16_result.results.size(), kNumReplicas);
  ASSERT_EQ(f8_result.results.size(), kNumReplicas);

  std::vector<Literal> f16_r0 = f16_result.results[0].DecomposeTuple();
  std::vector<Literal> f8_r0 = f8_result.results[0].DecomposeTuple();
  std::vector<Literal> f8_r1 = f8_result.results[1].DecomposeTuple();

  // Forward outputs should match exactly (no FP8 in forward path)
  EXPECT_TRUE(LiteralTestUtil::Equal(f16_r0[0], f8_r0[0]));

  // FP8 vs FP16 weight comparison: should be close but not identical
  EXPECT_TRUE(
      LiteralTestUtil::Near(f16_r0[1], f8_r0[1], ErrorSpec{1e-2, 1e-2}));

  // Numerical precision check: FP8 should produce measurably different results
  // than FP16. FP8 e4m3 has ~6% relative error (2^-4), FP16 has ~0.1% (2^-10).
  TF_ASSERT_OK_AND_ASSIGN(Literal f16_f32, f16_r0[1].Convert(F32));
  TF_ASSERT_OK_AND_ASSIGN(Literal f8_f32, f8_r0[1].Convert(F32));
  absl::Span<const float> f16_data = f16_f32.data<float>();
  absl::Span<const float> f8_data = f8_f32.data<float>();
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < f16_data.size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::abs(f16_data[i] - f8_data[i]));
  }
  // Expect measurable difference (> FP16 noise floor of ~0.1%)
  EXPECT_GT(max_abs_diff, 1e-3f);
}

// Test that FP8 all-reduce fails on pre-Hopper CUDA GPUs without FP8 support.
// Note: ROCm is skipped because it has a fallback to ncclInt8 for all GPUs.
TEST_F(AllReduceTestNoParams, AsyncAllReduce_F8E4M3FN_FailsOnUnsupportedGPUs) {
  if (Capability().IsRocm()) {
    GTEST_SKIP()
        << "Test is CUDA-only. ROCm has fallback to ncclInt8 for all GPUs.";
  }

  if (!Capability().IsCuda()) {
    GTEST_SKIP() << "Test requires CUDA.";
  }

  if (Capability().cuda_compute_capability()->IsAtLeast(9, 0)) {
    GTEST_SKIP() << "Test requires pre-Hopper GPU (compute capability < 9.0).";
  }

  const absl::string_view kF8ModuleStr = R"(
  HloModule fp8_allreduce_test
  add_f8 { x = f8e4m3fn[] parameter(0)  y = f8e4m3fn[] parameter(1)  ROOT add = f8e4m3fn[] add(x, y) }
  ENTRY test_computation {
    param_0 = f16[64,128] parameter(0)
    param_f8 = f8e4m3fn[64,128] convert(param_0)
    allreduce_f8 = f8e4m3fn[64,128] all-reduce(param_f8), to_apply=add_f8, replica_groups={{0,1}}
    ROOT result = f16[64,128] convert(allreduce_f8)
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kF8ModuleStr, kNumReplicas));

  Array<Eigen::half> input1({64, 128}), input2({64, 128});
  input1.FillRandom(Eigen::half(0.1f), 0.5f, /*seed=*/0);
  input2.FillRandom(Eigen::half(0.1f), 0.5f, /*seed=*/1);
  Literal input_literal1 = LiteralUtil::CreateFromArray(input1);
  Literal input_literal2 = LiteralUtil::CreateFromArray(input2);

  auto result = ExecuteReplicated(
      std::move(module),
      std::vector<std::vector<Literal*>>{{&input_literal1}, {&input_literal2}});

  EXPECT_FALSE(result.ok())
      << "FP8 all-reduce should fail on pre-Hopper GPUs, but succeeded.";
  // NCCL returns ncclInvalidArgument for FP8 reductions on pre-sm90 GPUs.
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr("FP8 reduction support begins with sm90"));
}

}  // namespace
}  // namespace xla

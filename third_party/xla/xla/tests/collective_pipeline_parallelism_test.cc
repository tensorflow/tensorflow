/* Copyright 2024 The OpenXLA Authors.

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
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/statusor.h"

// Tests cross-GPU operations.
//
// Several tests requires at least four GPUs.  For instructions on running this
// within Google, see go/multi-gpu-unit-test.

// TODO: Move this to hlo_test_base.h
#define SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(x)                     \
  if (num_devices_ < x) {                                         \
    GTEST_SKIP() << "Test requires at least " << x << " devices"; \
  }

namespace xla {
namespace {

class CollectivePipelineParallelismTest : public HloTestBase {
 public:
  CollectivePipelineParallelismTest() : num_devices_(backend().device_count()) {
    VLOG(1) << "Running with " << num_devices_ << " devices";
  }

 protected:
  const int64_t num_devices_;
};

XLA_TEST_F(CollectivePipelineParallelismTest,
           CollectivePermute_CircularPipelinePreOptimization) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  while_cond {
    param = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(param), index=0
    max_iter = u32[] constant(3)
    ROOT cmp = pred[] compare(iter, max_iter), direction=LT
  }

  while_body {
    param = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(param), index=0
    data = f32[2,2] get-tuple-element(param), index=1
    weights = f32[2,2] get-tuple-element(param), index=2
    matmul = f32[2,2] dot(weights, data), lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    cp = f32[2,2] collective-permute(matmul),
        source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}
    iter_increment = u32[] constant(1)
    next_iter = u32[] add(iter, iter_increment)
    ROOT result = (u32[], f32[2,2], f32[2,2]) tuple(next_iter, cp, weights)
  }

  ENTRY test_computation {
    iter = u32[] constant(0)
    data = f32[2,2] parameter(0)
    weights = f32[2,2] parameter(1)
    input = (u32[], f32[2,2], f32[2,2]) tuple(iter, data, weights)
    while_res = (u32[], f32[2,2], f32[2,2]) while(input), condition=while_cond,
        body=while_body
    ROOT data_out = f32[2,2] get-tuple-element(while_res), index=1
  }
  )";
  const int64_t kNumReplicas = 4;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas)

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  std::unique_ptr<VerifiedHloModule> module;
  TF_ASSERT_OK_AND_ASSIGN(module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  // Inputs for replica i are
  // A = {{i+1, i+1},
  //      {i+1, i+1}}, and
  // B = {{0, 0},
  //      {0, 1}}.
  std::vector<Literal> inputs_a;
  for (int64_t i = 0; i < kNumReplicas; ++i) {
    float val = i + 1;
    inputs_a.push_back(LiteralUtil::CreateR2<float>({{val, val}, {val, val}}));
  }
  Literal input_b_replicated = LiteralUtil::CreateR2<float>({{0, 0}, {0, 1}});
  std::vector<std::vector<Literal *>> inputs;
  for (int64_t i = 0; i < kNumReplicas; ++i) {
    inputs.push_back({&inputs_a[i], &input_b_replicated});
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), inputs, kNumReplicas,
                        /*run_hlo_passes=*/true));
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {2, 2}}, results[0]);
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {3, 3}}, results[1]);
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {4, 4}}, results[2]);
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {1, 1}}, results[3]);
}

// Helper functions for pipeline parallelism tests where each stage scales the
// input by some factor.
absl::StatusOr<Literal> CreateLinearLayerWeights(int64_t size, float factor) {
  return LiteralUtil::CreateLiteralWithGenerator<F32, float>(
      ShapeUtil::MakeShape(F32, {size, size}),
      [&](absl::Span<const int64_t> idx) -> float {
        return idx[0] == idx[1] ? factor : 0.0;
      });
};
absl::StatusOr<Literal> CreateZeroInputR2(int64_t microbatches, int64_t size) {
  return LiteralUtil::CreateLiteralWithGenerator<F32, float>(
      ShapeUtil::MakeShape(F32, {microbatches, size}),
      [&](absl::Span<const int64_t> idx) -> float { return 0.0; });
};
absl::StatusOr<Literal> CreateFingerprintInput(int64_t microbatches,
                                               int64_t size,
                                               float factor = 1.0) {
  return LiteralUtil::CreateLiteralWithGenerator<F32, float>(
      ShapeUtil::MakeShape(F32, {microbatches, size}),
      [&](absl::Span<const int64_t> idx) -> float {
        float fingerprint = 1.0 * idx[0] + 0.0001 * idx[1];
        return factor * fingerprint;
      });
};

// Naive implementation of pipeline parallelism:
//   - 4 devices
//   - 4 microbatches
//   - no circular repeat
//   - no disabled collectives
//   - no collective pipelining
//
// Every stage of the pipeline is a single linear layer.
XLA_TEST_F(CollectivePipelineParallelismTest, NaiveDFSMicrobatch4Replica4) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  get_circ_buffer_index {
    offset = u32[] parameter(0)
    index = u32[] parameter(1)
    size = u32[] parameter(2)
    t0 = u32[] add(offset, index)
    t1 = u32[] divide(t0, size)
    t2 = u32[] multiply(t1, size)
    ROOT t4 = u32[] subtract(t0, t2)
  }

  is_input_replica {
    replica_id = u32[] replica-id()
    c0 = u32[] constant(0)
    ROOT predicate = pred[] compare(replica_id, c0), direction=EQ
  }

  is_output_replica {
    replica_id = u32[] replica-id()
    c1 = u32[] constant(1)
    ROOT predicate = pred[] compare(replica_id, c1), direction=EQ
  }

  while_condition {
    tuple = (f32[16,16], f32[4,16], f32[4,16], f32[16], u32[]) parameter(0)
    i = u32[] get-tuple-element(tuple), index=4
    n = u32[] constant(7)
    ROOT predicate = pred[] compare(i, n), direction=LT
  }

  while_body {
    tuple = (f32[16,16], f32[4,16], f32[4,16], f32[16], u32[]) parameter(0)
    weights = f32[16,16] get-tuple-element(tuple), index=0
    input = f32[4,16] get-tuple-element(tuple), index=1
    output = f32[4,16] get-tuple-element(tuple), index=2
    tmp = f32[16] get-tuple-element(tuple), index=3
    i = u32[] get-tuple-element(tuple), index=4

    c1 = u32[] constant(1)
    c0 = u32[] constant(0)
    c4 = u32[] constant(4)

    input_idx = u32[] call(c0, i, c4), to_apply=get_circ_buffer_index
    input_slice = f32[1,16] dynamic-slice(input, input_idx, c0),
        dynamic_slice_sizes={1,16}
    input_slice_ = f32[16] reshape(input_slice)

    prev_stage_slice = f32[16] collective-permute(tmp),
        source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}

    read_input = pred[] call(), to_apply=is_input_replica
    compute_in = f32[16] select(read_input, input_slice_, prev_stage_slice)

    compute_out = f32[16] dot(weights, compute_in), lhs_contracting_dims={1},
        rhs_contracting_dims={0}

    output_index = u32[] call(c1, i, c4), to_apply=get_circ_buffer_index
    output_slice = f32[1,16] reshape(compute_out)
    output_ = f32[4,16] dynamic-update-slice(output, output_slice, output_index,
        c0)

    i_ = add(i, c1)

    ROOT tuple1 = (f32[16,16], f32[4,16], f32[4,16], f32[16], u32[]) tuple(
        weights, input, output_, compute_out, i_)
  }

  ENTRY main {
    weights = f32[16,16] parameter(0)
    input = f32[4,16] parameter(1)

    cf0 = f32[] constant(0)
    output = f32[4,16] broadcast(cf0), dimensions={}
    tmp = f32[16] broadcast(cf0), dimensions={}
    c0 = u32[] constant(0)

    tuple = (f32[16,16], f32[4,16], f32[4,16], f32[16], u32[]) tuple(weights,
        input, output, tmp, c0)
    tuple_ = (f32[16,16], f32[4,16], f32[4,16], f32[16], u32[]) while(tuple),
        condition=while_condition, body=while_body

    ROOT output_ = f32[4,16] get-tuple-element(tuple_), index=2
  }
  )";

  const int64_t kNumReplicas = 4;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas)

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  // This pipeline consists of 4 layers, each of which is a single linear layer.
  // We assign the weights to the replicas such that the layers scale the input
  // data by 1.0, 2.0, 3.0 and 4.0. The combined effect is to scale the input
  // data by 24.0.
  const int64_t kInputSize = 16;
  TF_ASSERT_OK_AND_ASSIGN(Literal weights_r0,
                          CreateLinearLayerWeights(kInputSize, 1.0));
  TF_ASSERT_OK_AND_ASSIGN(Literal weights_r1,
                          CreateLinearLayerWeights(kInputSize, 2.0));
  TF_ASSERT_OK_AND_ASSIGN(Literal weights_r2,
                          CreateLinearLayerWeights(kInputSize, 3.0));
  TF_ASSERT_OK_AND_ASSIGN(Literal weights_r3,
                          CreateLinearLayerWeights(kInputSize, 4.0));

  // Only the first replica holds the input to the pipeline in this naive
  // implementation. The remaining replicas get zero/dummy input.
  const int64_t kMicrobatches = 4;
  TF_ASSERT_OK_AND_ASSIGN(Literal real_input,
                          CreateFingerprintInput(kMicrobatches, kInputSize));
  TF_ASSERT_OK_AND_ASSIGN(Literal fake_input,
                          CreateZeroInputR2(kMicrobatches, kInputSize));

  std::vector<std::vector<Literal *>> args = {{&weights_r0, &real_input},
                                              {&weights_r1, &fake_input},
                                              {&weights_r2, &fake_input},
                                              {&weights_r3, &fake_input}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), args, kNumReplicas,
                        /*run_hlo_passes=*/true));

  // Check pipeline output for last replica.
  // The combined effect of the pipeline is to scale the input data by 24.0.
  const float kExpectedFactor = 1.0 * 2.0 * 3.0 * 4.0;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal expected_output,
      CreateFingerprintInput(kMicrobatches, kInputSize, kExpectedFactor));
  EXPECT_TRUE(LiteralTestUtil::NearOrEqual(expected_output, results[3],
                                           ErrorSpec{1e-5, 1e-5}));
}

// Naive implementation of pipeline parallelism:
//   - 4 devices
//   - 5 microbatches
//   - no circular repeat
//   - no disabled collectives
//   - no collective pipelining
//
// Every stage of the pipeline is a single linear layer.
XLA_TEST_F(CollectivePipelineParallelismTest, NaiveDFSMicrobatch5Replica4) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  get_circ_buffer_index {
    offset = u32[] parameter(0)
    index = u32[] parameter(1)
    size = u32[] parameter(2)
    t0 = u32[] add(offset, index)
    t1 = u32[] divide(t0, size)
    t2 = u32[] multiply(t1, size)
    ROOT t4 = u32[] subtract(t0, t2)
  }

  is_input_replica {
    replica_id = u32[] replica-id()
    c0 = u32[] constant(0)
    ROOT predicate = pred[] compare(replica_id, c0), direction=EQ
  }

  is_output_replica {
    replica_id = u32[] replica-id()
    c1 = u32[] constant(1)
    ROOT predicate = pred[] compare(replica_id, c1), direction=EQ
  }

  while_condition {
    tuple = (f32[16,16], f32[5,16], f32[5,16], f32[16], u32[]) parameter(0)
    i = u32[] get-tuple-element(tuple), index=4
    n = u32[] constant(8)
    ROOT predicate = pred[] compare(i, n), direction=LT
  }

  while_body {
    tuple = (f32[16,16], f32[5,16], f32[5,16], f32[16], u32[]) parameter(0)
    weights = f32[16,16] get-tuple-element(tuple), index=0
    input = f32[5,16] get-tuple-element(tuple), index=1
    output = f32[5,16] get-tuple-element(tuple), index=2
    tmp = f32[16] get-tuple-element(tuple), index=3
    i = u32[] get-tuple-element(tuple), index=4

    c1 = u32[] constant(1)
    c2 = u32[] constant(2)
    c0 = u32[] constant(0)
    c5 = u32[] constant(5)

    input_idx = u32[] call(c0, i, c5), to_apply=get_circ_buffer_index
    input_slice = f32[1,16] dynamic-slice(input, input_idx, c0),
        dynamic_slice_sizes={1,16}
    input_slice_ = f32[16] reshape(input_slice)

    prev_stage_slice = f32[16] collective-permute(tmp),
        source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}

    read_input = pred[] call(), to_apply=is_input_replica
    compute_in = f32[16] select(read_input, input_slice_, prev_stage_slice)

    compute_out = f32[16] dot(weights, compute_in), lhs_contracting_dims={1},
        rhs_contracting_dims={0}

    output_index = u32[] call(c2, i, c5), to_apply=get_circ_buffer_index
    output_slice = f32[1,16] reshape(compute_out)
    output_ = f32[5,16] dynamic-update-slice(output, output_slice, output_index,
        c0)

    i_ = add(i, c1)

    ROOT tuple1 = (f32[16,16], f32[5,16], f32[5,16], f32[16], u32[])
        tuple(weights, input, output_, compute_out, i_)
  }

  ENTRY main {
    weights = f32[16,16] parameter(0)
    input = f32[5,16] parameter(1)

    cf0 = f32[] constant(0)
    output = f32[5,16] broadcast(cf0), dimensions={}
    tmp = f32[16] broadcast(cf0), dimensions={}
    c0 = u32[] constant(0)

    tuple = (f32[16,16], f32[5,16], f32[5,16], f32[16], u32[])
        tuple(weights, input, output, tmp, c0)
    tuple_ = (f32[16,16], f32[5,16], f32[5,16], f32[16], u32[]) while(tuple),
        condition=while_condition, body=while_body

    ROOT output_ = f32[5,16] get-tuple-element(tuple_), index=2
  }
  )";

  const int64_t kNumReplicas = 4;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas)

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  // This pipeline consists of 4 layers, each of which is a single linear layer.
  // We assign the weights to the replicas such that the layers scale the input
  // data by 1.0, 2.0, 3.0 and 4.0. The combined effect is to scale the input
  // data by 24.0.
  const int64_t kInputSize = 16;
  TF_ASSERT_OK_AND_ASSIGN(Literal weights_r0,
                          CreateLinearLayerWeights(kInputSize, 1.0));
  TF_ASSERT_OK_AND_ASSIGN(Literal weights_r1,
                          CreateLinearLayerWeights(kInputSize, 2.0));
  TF_ASSERT_OK_AND_ASSIGN(Literal weights_r2,
                          CreateLinearLayerWeights(kInputSize, 3.0));
  TF_ASSERT_OK_AND_ASSIGN(Literal weights_r3,
                          CreateLinearLayerWeights(kInputSize, 4.0));

  // Only the first replica holds the input to the pipeline in this naive
  // implementation. The remaining replicas get zero/dummy input.
  const int64_t kMicrobatches = 5;
  TF_ASSERT_OK_AND_ASSIGN(Literal real_input,
                          CreateFingerprintInput(kMicrobatches, kInputSize));
  TF_ASSERT_OK_AND_ASSIGN(Literal fake_input,
                          CreateZeroInputR2(kMicrobatches, kInputSize));

  // Check pipeline output for last replica.
  // The combined effect of the pipeline is to scale the input data by 24.0.
  const float kExpectedFactor = 1.0 * 2.0 * 3.0 * 4.0;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal expected_output,
      CreateFingerprintInput(kMicrobatches, kInputSize, kExpectedFactor));
  std::vector<std::vector<Literal *>> args = {{&weights_r0, &real_input},
                                              {&weights_r1, &fake_input},
                                              {&weights_r2, &fake_input},
                                              {&weights_r3, &fake_input}};
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), args, kNumReplicas,
                        /*run_hlo_passes=*/true));
  EXPECT_TRUE(LiteralTestUtil::NearOrEqual(expected_output, results[3],
                                           ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace xla

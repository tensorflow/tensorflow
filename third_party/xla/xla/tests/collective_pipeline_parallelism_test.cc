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
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// Tests cross-GPU operations.
//
// Several tests requires at least four GPUs.  For instructions on running this
// within Google, see go/multi-gpu-unit-test.
class CollectivePipelineParallelismTest : public HloTestBase {
 public:
  CollectivePipelineParallelismTest() {
    VLOG(1) << "Running with " << num_devices() << " devices";
  }
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
    cp = f32[2,2] collective-permute(data),
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}
    matmul = f32[2,2] dot(weights, cp),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
    iter_increment = u32[] constant(1)
    next_iter = u32[] add(iter, iter_increment)
    ROOT result = (u32[], f32[2,2], f32[2,2]) tuple(next_iter, matmul, weights)
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

std::string GetModuleStrWithCommonComputations(
    const std::string name, const std::string more_computations) {
  static constexpr char kCommonComputationsStr[] = R"(
  read_buffer_mb4 {
    buffer = f32[4,16] parameter(0)
    offset = u32[] parameter(1)
    index = u32[] parameter(2)
    c0 = u32[] constant(0)
    c4 = u32[] constant(4)
    index_ = u32[] add(index, offset)
    index__ = u32[] remainder(index_, c4)
    slice = f32[1,16] dynamic-slice(buffer, index__, c0),
        dynamic_slice_sizes={1,16}
    ROOT slice_ = f32[16] reshape(slice)
  }

  read_buffer_mb5 {
    buffer = f32[5,16] parameter(0)
    offset = u32[] parameter(1)
    index = u32[] parameter(2)
    c0 = u32[] constant(0)
    c5 = u32[] constant(5)
    index_ = u32[] add(index, offset)
    index__ = u32[] remainder(index_, c5)
    slice = f32[1,16] dynamic-slice(buffer, index__, c0),
        dynamic_slice_sizes={1,16}
    ROOT slice_ = f32[16] reshape(slice)
  }

  update_buffer_mb4 {
    buffer = f32[4,16] parameter(0)
    update = f32[16] parameter(1)
    offset = u32[] parameter(2)
    index = u32[] parameter(3)
    c0 = u32[] constant(0)
    c4 = u32[] constant(4)
    index_ = u32[] add(index, offset)
    index__ = u32[] remainder(index_, c4)
    update_ = f32[1,16] reshape(update)
    ROOT buffer_ = f32[4,16] dynamic-update-slice(buffer, update_, index__, c0)
  }

  update_buffer_mb5 {
    buffer = f32[5,16] parameter(0)
    update = f32[16] parameter(1)
    offset = u32[] parameter(2)
    index = u32[] parameter(3)
    c0 = u32[] constant(0)
    c5 = u32[] constant(5)
    index_ = u32[] add(index, offset)
    index__ = u32[] remainder(index_, c5)
    update_ = f32[1,16] reshape(update)
    ROOT buffer_ = f32[5,16] dynamic-update-slice(buffer, update_, index__, c0)
  }

  is_input_replica {
    replica_id = u32[] replica-id()
    c0 = u32[] constant(0)
    ROOT predicate = pred[] compare(replica_id, c0), direction=EQ
  }

  is_output_replica {
    replica_id = u32[] replica-id()
    c3 = u32[] constant(3)
    ROOT predicate = pred[] compare(replica_id, c3), direction=EQ
  }

  is_read_input_mb4 {
    is_input_replica = pred[] call(), to_apply=is_input_replica
    i = u32[] parameter(0)
    c4 = u32[] constant(4)
    is_input_iteration = pred[] compare(i, c4), direction=LT
    ROOT is_read_input = pred[] and(is_input_replica, is_input_iteration)
  }

  is_read_input_mb5 {
    is_input_replica = pred[] call(), to_apply=is_input_replica
    i = u32[] parameter(0)
    c5 = u32[] constant(5)
    is_input_iteration = pred[] compare(i, c5), direction=LT
    ROOT is_read_input = pred[] and(is_input_replica, is_input_iteration)
  }
  )";
  return "HloModule " + name + "\n" + kCommonComputationsStr + "\n" +
         more_computations;
}

// Naive implementation of pipeline parallelism:
//   - 4 devices
//   - 4 microbatches
//   - no circular repeat
//   - no disabled collectives
//   - no collective pipelining
//
// Every stage of the pipeline is a single linear layer.
XLA_TEST_F(CollectivePipelineParallelismTest, NaiveBFSMicrobatch4Replica4) {
  constexpr char kMoreComputationsStr[] = R"(
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
    prev_iteration_compute_res = f32[16] get-tuple-element(tuple), index=3
    i = u32[] get-tuple-element(tuple), index=4

    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    c4 = u32[] constant(4)

    // Read from buffers.
    input_slice = f32[16] call(input, c0, i), to_apply=read_buffer_mb4

    // Shift data to the next stage in the pipeline.
    prev_stage_slice = f32[16] collective-permute(prev_iteration_compute_res),
        source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}

    // Select compute argument from previous stage or from input and perform
    // compute.
    read_input = pred[] call(), to_apply=is_input_replica
    compute_arg = f32[16] select(read_input, input_slice, prev_stage_slice)
    compute_res = f32[16] dot(weights, compute_arg), lhs_contracting_dims={1},
        rhs_contracting_dims={0}

    // Update buffers.
    output_ = call(output, compute_res, c1, i), to_apply=update_buffer_mb4

    i_ = add(i, c1)

    ROOT tuple1 = (f32[16,16], f32[4,16], f32[4,16], f32[16], u32[]) tuple(
        weights, input, output_, compute_res, i_)
  }

  ENTRY main {
    weights = f32[16,16] parameter(0)
    input = f32[4,16] parameter(1)

    cf0 = f32[] constant(0)
    output = f32[4,16] broadcast(cf0), dimensions={}
    prev_iteration_compute_res = f32[16] broadcast(cf0), dimensions={}
    c0 = u32[] constant(0)

    tuple = (f32[16,16], f32[4,16], f32[4,16], f32[16], u32[]) tuple(weights,
        input, output, prev_iteration_compute_res, c0)
    tuple_ = (f32[16,16], f32[4,16], f32[4,16], f32[16], u32[]) while(tuple),
        condition=while_condition, body=while_body

    ROOT output_ = f32[4,16] get-tuple-element(tuple_), index=2
  }
  )";

  const int64_t kNumReplicas = 4;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas)

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(GetModuleStrWithCommonComputations(
                                       /*name=*/"test", kMoreComputationsStr),
                                   config));

  // This pipeline consists of 4 layers, each of which is a single linear layer.
  // We assign the weights to the replicas such that the layers scale the input
  // data by 1.0, 2.0, 3.0 and 4.0. The combined effect is to scale the input
  // data by 24.0.
  const int64_t kInputSize = 16;
  Literal weights_r0 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 1.0);
  Literal weights_r1 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 2.0);
  Literal weights_r2 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 3.0);
  Literal weights_r3 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 4.0);

  // Only the first replica holds the input to the pipeline in this naive
  // implementation. The remaining replicas get zero/dummy input.
  const int64_t kMicrobatches = 4;
  Literal real_input =
      LiteralUtil::CreateFingerprintMatixR2<float>(kMicrobatches, kInputSize);
  Literal fake_input =
      LiteralUtil::CreateFull<float>({kMicrobatches, kInputSize}, 0.0);

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
  Literal expected_output = LiteralUtil::CreateFingerprintMatixR2<float>(
      kMicrobatches, kInputSize, kExpectedFactor);
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
XLA_TEST_F(CollectivePipelineParallelismTest, NaiveBFSMicrobatch5Replica4) {
  constexpr char kMoreComputationsStr[] = R"(
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
    prev_iteration_compute_res = f32[16] get-tuple-element(tuple), index=3
    i = u32[] get-tuple-element(tuple), index=4

    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    c2 = u32[] constant(2)
    c5 = u32[] constant(5)

    // Read from buffers.
    input_slice = f32[16] call(input, c0, i), to_apply=read_buffer_mb5

    // Shift data to the next stage in the pipeline.
    prev_stage_slice = f32[16] collective-permute(prev_iteration_compute_res),
        source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}

    // Select compute argument from previous stage or from input and perform
    // compute.
    read_input = pred[] call(), to_apply=is_input_replica
    compute_arg = f32[16] select(read_input, input_slice, prev_stage_slice)
    compute_res = f32[16] dot(weights, compute_arg), lhs_contracting_dims={1},
        rhs_contracting_dims={0}

    // Update buffers.
    output_ = call(output, compute_res, c2, i), to_apply=update_buffer_mb5

    i_ = add(i, c1)

    ROOT tuple1 = (f32[16,16], f32[5,16], f32[5,16], f32[16], u32[])
        tuple(weights, input, output_, compute_res, i_)
  }

  ENTRY main {
    weights = f32[16,16] parameter(0)
    input = f32[5,16] parameter(1)

    cf0 = f32[] constant(0)
    output = f32[5,16] broadcast(cf0), dimensions={}
    prev_iteration_compute_res = f32[16] broadcast(cf0), dimensions={}
    c0 = u32[] constant(0)

    tuple = (f32[16,16], f32[5,16], f32[5,16], f32[16], u32[])
        tuple(weights, input, output, prev_iteration_compute_res, c0)
    tuple_ = (f32[16,16], f32[5,16], f32[5,16], f32[16], u32[]) while(tuple),
        condition=while_condition, body=while_body

    ROOT output_ = f32[5,16] get-tuple-element(tuple_), index=2
  }
  )";

  const int64_t kNumReplicas = 4;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas)

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(GetModuleStrWithCommonComputations(
                                       /*name=*/"test", kMoreComputationsStr),
                                   config));

  // This pipeline consists of 4 layers, each of which is a single linear layer.
  // We assign the weights to the replicas such that the layers scale the input
  // data by 1.0, 2.0, 3.0 and 4.0. The combined effect is to scale the input
  // data by 24.0.
  const int64_t kInputSize = 16;
  Literal weights_r0 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 1.0);
  Literal weights_r1 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 2.0);
  Literal weights_r2 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 3.0);
  Literal weights_r3 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 4.0);

  // Only the first replica holds the input to the pipeline in this naive
  // implementation. The remaining replicas get zero/dummy input.
  const int64_t kMicrobatches = 5;
  Literal real_input =
      LiteralUtil::CreateFingerprintMatixR2<float>(kMicrobatches, kInputSize);
  Literal fake_input =
      LiteralUtil::CreateFull<float>({kMicrobatches, kInputSize}, 0.0);

  // Check pipeline output for last replica.
  // The combined effect of the pipeline is to scale the input data by 24.0.
  const float kExpectedFactor = 1.0 * 2.0 * 3.0 * 4.0;
  Literal expected_output = LiteralUtil::CreateFingerprintMatixR2<float>(
      kMicrobatches, kInputSize, /*scale=*/kExpectedFactor);
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

// Naive implementation of pipeline parallelism:
//   - 4 devices
//   - 4 microbatches
//   - 2 circular repeat
//   - no disabled collectives
//   - no collective pipelining
//
// Every stage of the pipeline is a single linear layer.
XLA_TEST_F(CollectivePipelineParallelismTest,
           NaiveBFSMicrobatch4CircularRepeat2Replica4) {
  constexpr char kMoreComputationsStr[] = R"(
  while_condition {
    tuple = (f32[16,16], f32[4,16], f32[4,16], f32[16], u32[]) parameter(0)
    i = u32[] get-tuple-element(tuple), index=4
    n = u32[] constant(11)
    ROOT predicate = pred[] compare(i, n), direction=LT
  }

  while_body {
    tuple = (f32[16,16], f32[4,16], f32[4,16], f32[16], u32[]) parameter(0)
    weights = f32[16,16] get-tuple-element(tuple), index=0
    input = f32[4,16] get-tuple-element(tuple), index=1
    output = f32[4,16] get-tuple-element(tuple), index=2
    prev_iteration_compute_res = f32[16] get-tuple-element(tuple), index=3
    i = u32[] get-tuple-element(tuple), index=4

    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    c4 = u32[] constant(4)

    // Read from buffers.
    input_slice = f32[16] call(input, c0, i), to_apply=read_buffer_mb4

    // Shift data to the next stage in the pipeline.
    prev_stage_slice = f32[16] collective-permute(prev_iteration_compute_res),
        source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}

    // Select compute argument from previous stage or from input and perform
    // compute.
    is_read_input = pred[] call(i), to_apply=is_read_input_mb4
    compute_arg = f32[16] select(is_read_input, input_slice, prev_stage_slice)
    compute_res = f32[16] dot(weights, compute_arg), lhs_contracting_dims={1},
        rhs_contracting_dims={0}

    // Update buffers.
    output_ = f32[4,16] call(output, compute_res, c1, i),
        to_apply=update_buffer_mb4

    i_ = add(i, c1)

    ROOT tuple1 = (f32[16,16], f32[4,16], f32[4,16], f32[16], u32[])
        tuple(weights, input, output_, compute_res, i_)
  }

  ENTRY main {
    weights = f32[16,16] parameter(0)
    input = f32[4,16] parameter(1)

    cf0 = f32[] constant(0)
    output = f32[4,16] broadcast(cf0), dimensions={}
    prev_iteration_compute_res = f32[16] broadcast(cf0), dimensions={}
    c0 = u32[] constant(0)

    tuple = (f32[16,16], f32[4,16], f32[4,16], f32[16], u32[]) tuple(weights,
        input, output, prev_iteration_compute_res, c0)
    tuple_ = (f32[16,16], f32[4,16], f32[4,16], f32[16], u32[]) while(tuple),
        condition=while_condition, body=while_body

    ROOT output_ = f32[4,16] get-tuple-element(tuple_), index=2
  }
  )";

  const int64_t kNumReplicas = 4;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas)

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(GetModuleStrWithCommonComputations(
                                       /*name=*/"test", kMoreComputationsStr),
                                   config));

  // This pipeline consists of a total of 8 layers (2 per replica), each of
  // which is a single linear layer. We assign the weights to the replicas such
  // that the layers scale the input data by 1.0, 2.0, 3.0 and 4.0 in the first
  // and second cycle. The combined effect is to scale the input data by 576.0
  // (24.0 * 24.0).
  const int64_t kInputSize = 16;
  Literal weights_r0 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 1.0);
  Literal weights_r1 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 2.0);
  Literal weights_r2 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 3.0);
  Literal weights_r3 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 4.0);

  // Only the first replica holds the input to the pipeline in this naive
  // implementation. The remaining replicas get zero/dummy input.
  const int64_t kMicrobatches = 4;
  Literal real_input =
      LiteralUtil::CreateFingerprintMatixR2<float>(kMicrobatches, kInputSize);
  Literal fake_input =
      LiteralUtil::CreateFull<float>({kMicrobatches, kInputSize}, 0.0);

  // Check pipeline output for last replica.
  // The combined effect of the pipeline is to scale the input data by 576.0
  // (24.0 * 24.0).
  const float kExpectedFactor = 1.0 * 2.0 * 3.0 * 4.0 * 1.0 * 2.0 * 3.0 * 4.0;
  Literal expected_output = LiteralUtil::CreateFingerprintMatixR2<float>(
      kMicrobatches, kInputSize, /*scale=*/kExpectedFactor);
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

// Naive implementation of pipeline parallelism:
//   - 4 devices
//   - 5 microbatches
//   - 2 circular repeat
//   - no disabled collectives
//   - no collective pipelining
//
// Every stage of the pipeline is a single linear layer.
XLA_TEST_F(CollectivePipelineParallelismTest,
           NaiveBFSMicrobatch5CircularRepeat2Replica4) {
  constexpr char kMoreComputationsStr[] = R"(
  while_condition {
    tuple = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[])
        parameter(0)
    i = u32[] get-tuple-element(tuple), index=5
    n = u32[] constant(13)
    ROOT predicate = pred[] compare(i, n), direction=LT
  }

  while_body {
    tuple = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[])
        parameter(0)
    weights = f32[16,16] get-tuple-element(tuple), index=0
    input = f32[5,16] get-tuple-element(tuple), index=1
    output = f32[5,16] get-tuple-element(tuple), index=2
    buffer = f32[5,16] get-tuple-element(tuple), index=3
    prev_iteration_compute_res = f32[16] get-tuple-element(tuple), index=4
    i = u32[] get-tuple-element(tuple), index=5

    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    c2 = u32[] constant(2)
    c3 = u32[] constant(3)
    c4 = u32[] constant(4)
    c5 = u32[] constant(5)

    // Read from buffers.
    input_slice = f32[16] call(input, c0, i), to_apply=read_buffer_mb5
    buffer_slice = f32[16] call(buffer, c3, i), to_apply=read_buffer_mb5

    // Shift data to the next stage in the pipeline.
    // Directly depends on the updated buffer of the previous iteration and,
    // therefore, depends on the previous iteration's compute.
    is_output_replica = pred[] call(), to_apply=is_output_replica
    next_stage_slice = select(is_output_replica, buffer_slice,
        prev_iteration_compute_res)
    prev_stage_slice = f32[16] collective-permute(next_stage_slice),
        source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}

    // Select compute argument from previous stage or from input and perform
    // compute.
    is_read_input = pred[] call(i), to_apply=is_read_input_mb5
    compute_arg = f32[16] select(is_read_input, input_slice, prev_stage_slice)
    compute_res = f32[16] dot(weights, compute_arg), lhs_contracting_dims={1},
        rhs_contracting_dims={0}

    // Update buffers.
    output_ = f32[5,16] call(output, compute_res, c2, i),
        to_apply=update_buffer_mb5
    buffer_ = f32[5,16] call(buffer, compute_res, c0, i),
        to_apply=update_buffer_mb5

    i_ = add(i, c1)

    ROOT tuple_ = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[])
        tuple(weights, input, output_, buffer_, compute_res, i_)
  }

  ENTRY main {
    weights = f32[16,16] parameter(0)
    input = f32[5,16] parameter(1)

    cf0 = f32[] constant(0)
    output = f32[5,16] broadcast(cf0), dimensions={}
    buffer = f32[5,16] broadcast(cf0), dimensions={}
    prev_iteration_compute_res = f32[16] broadcast(cf0), dimensions={}
    c0 = u32[] constant(0)

    // Iterate through pipeline stages.
    tuple = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[])
        tuple(weights, input, output, buffer, prev_iteration_compute_res, c0)
    tuple_ = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[])
        while(tuple), condition=while_condition, body=while_body

    ROOT output_ = f32[5,16] get-tuple-element(tuple_), index=2
  }
  )";

  const int64_t kNumReplicas = 4;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas)

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(GetModuleStrWithCommonComputations(
                                       /*name=*/"test", kMoreComputationsStr),
                                   config));

  // This pipeline consists of a total of 8 layers (2 per replica), each of
  // which is a single linear layer. We assign the weights to the replicas such
  // that the layers scale the input data by 1.0, 2.0, 3.0 and 4.0 in the first
  // and second cycle. The combined effect is to scale the input data by 576.0
  // (24.0 * 24.0).
  const int64_t kInputSize = 16;
  Literal weights_r0 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 1.0);
  Literal weights_r1 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 2.0);
  Literal weights_r2 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 3.0);
  Literal weights_r3 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 4.0);

  // Only the first replica holds the input to the pipeline in this naive
  // implementation. The remaining replicas get zero/dummy input.
  const int64_t kMicrobatches = 5;
  Literal real_input =
      LiteralUtil::CreateFingerprintMatixR2<float>(kMicrobatches, kInputSize);
  Literal fake_input =
      LiteralUtil::CreateFull<float>({kMicrobatches, kInputSize}, 0.0);

  // Check pipeline output for last replica.
  // The combined effect of the pipeline is to scale the input data by 576.0
  // (24.0 * 24.0).
  const float kExpectedFactor = 1.0 * 2.0 * 3.0 * 4.0 * 1.0 * 2.0 * 3.0 * 4.0;
  Literal expected_output = LiteralUtil::CreateFingerprintMatixR2<float>(
      kMicrobatches, kInputSize, /*scale=*/kExpectedFactor);
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

// Naive implementation of pipeline parallelism, which breaks the direct data
// dependency between the collective permute and the previous iteration's
// compute.
//   - 4 devices
//   - 4 microbatches
//   - 2 circular repeat
//   - no disabled collectives
//   - no collective pipelining
//
// Every stage of the pipeline is a single linear layer.
XLA_TEST_F(CollectivePipelineParallelismTest,
           NaiveWoDirectBufferDependencyBFSMicrobatch5CircularRepeat2Replica4) {
  constexpr char kMoreComputationsStr[] = R"(
  while_condition {
    tuple = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[])
        parameter(0)
    i = u32[] get-tuple-element(tuple), index=5
    n = u32[] constant(13)
    ROOT predicate = pred[] compare(i, n), direction=LT
  }

  while_body {
    tuple = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[])
        parameter(0)
    weights = f32[16,16] get-tuple-element(tuple), index=0
    input = f32[5,16] get-tuple-element(tuple), index=1
    output = f32[5,16] get-tuple-element(tuple), index=2
    buffer = f32[5,16] get-tuple-element(tuple), index=3
    prev_iteration_compute_res = f32[16] get-tuple-element(tuple), index=4
    i = u32[] get-tuple-element(tuple), index=5

    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    c2 = u32[] constant(2)
    c3 = u32[] constant(3)
    c4 = u32[] constant(4)
    c5 = u32[] constant(5)

    // Read from buffers before they are updated.
    input_slice = f32[16] call(input, c0, i), to_apply=read_buffer_mb5
    buffer_slice = f32[16] call(buffer, c3, i), to_apply=read_buffer_mb5

    // Shift data to the next stage in the pipeline.
    // Depends on the non-updated buffer of the previous iteration and,
    // therefore, does not depend on the previous iteration's compute.
    is_output_replica = pred[] call(), to_apply=is_output_replica
    next_stage_slice = select(is_output_replica, buffer_slice,
        prev_iteration_compute_res)
    prev_stage_slice = f32[16] collective-permute(next_stage_slice),
        source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}

    // Select compute argument from previous stage or from input and perform
    // compute.
    is_read_input = pred[] call(i), to_apply=is_read_input_mb5
    compute_arg = f32[16] select(is_read_input, input_slice, prev_stage_slice)
    compute_res = f32[16] dot(weights, compute_arg), lhs_contracting_dims={1},
        rhs_contracting_dims={0}

    // Update buffers.
    buffer_ = f32[5,16] call(buffer, prev_iteration_compute_res, c4, i),
        to_apply=update_buffer_mb5
    output_ = f32[5,16] call(output, compute_res, c2, i),
        to_apply=update_buffer_mb5

    i_ = add(i, c1)

    ROOT tuple_ = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[])
        tuple(weights, input, output_, buffer_, compute_res, i_)
  }

  ENTRY main {
    weights = f32[16,16] parameter(0)
    input = f32[5,16] parameter(1)

    cf0 = f32[] constant(0)
    output = f32[5,16] broadcast(cf0), dimensions={}
    buffer = f32[5,16] broadcast(cf0), dimensions={}
    prev_iteration_compute_res = f32[16] broadcast(cf0), dimensions={}
    c0 = u32[] constant(0)

    // Iterate through pipeline stages.
    tuple = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[])
        tuple(weights, input, output, buffer, prev_iteration_compute_res, c0)
    tuple_ = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[])
        while(tuple), condition=while_condition, body=while_body

    ROOT output_ = f32[5,16] get-tuple-element(tuple_), index=2
  }
  )";

  const int64_t kNumReplicas = 4;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas)

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(GetModuleStrWithCommonComputations(
                                       /*name=*/"test", kMoreComputationsStr),
                                   config));

  // This pipeline consists of a total of 8 layers (2 per replica), each of
  // which is a single linear layer. We assign the weights to the replicas such
  // that the layers scale the input data by 1.0, 2.0, 3.0 and 4.0 in the first
  // and second cycle. The combined effect is to scale the input data by 576.0
  // (24.0 * 24.0).
  const int64_t kInputSize = 16;
  Literal weights_r0 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 1.0);
  Literal weights_r1 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 2.0);
  Literal weights_r2 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 3.0);
  Literal weights_r3 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 4.0);

  // Only the first replica holds the input to the pipeline in this naive
  // implementation. The remaining replicas get zero/dummy input.
  const int64_t kMicrobatches = 5;
  Literal real_input =
      LiteralUtil::CreateFingerprintMatixR2<float>(kMicrobatches, kInputSize);
  Literal fake_input =
      LiteralUtil::CreateFull<float>({kMicrobatches, kInputSize}, 0.0);

  // Check pipeline output for last replica.
  // The combined effect of the pipeline is to scale the input data by 576.0
  // (24.0 * 24.0).
  const float kExpectedFactor = 1.0 * 2.0 * 3.0 * 4.0 * 1.0 * 2.0 * 3.0 * 4.0;
  Literal expected_output = LiteralUtil::CreateFingerprintMatixR2<float>(
      kMicrobatches, kInputSize, /*scale=*/kExpectedFactor);
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

XLA_TEST_F(CollectivePipelineParallelismTest, SendRecvLoop) {
  const absl::string_view kModuleStr = R"(
    HloModule test, num_partitions=4

    while_condidtion {
      param = (u32[], f32[2,2]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      c3 = u32[] constant(3)
      ROOT cmp = pred[] compare(i, c3), direction=LT
    }

    while_body {
      param = (u32[], f32[2,2]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data = f32[2,2] get-tuple-element(param), index=1

      // Send data from GPU i to i+1. Break cycle to avoid deadlock.
      after_all = token[] after-all()
      send_ctx = (f32[2,2], u32[], token[]) send(data, after_all),
          frontend_attributes={
          _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}, channel_id=1
      recv_ctx = (f32[2,2], u32[], token[]) recv(after_all),
          frontend_attributes={
          _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}, channel_id=2
      send_done = token[] send-done(send_ctx), channel_id=1
      recv_done = (f32[2,2], token[]) recv-done(recv_ctx), channel_id=2
      data_ = f32[2,2] get-tuple-element(recv_done), index=0

      c1 = u32[] constant(1)
      i_ = u32[] add(i, c1)
      ROOT result = (u32[], f32[2,2]) tuple(i_, data_)
    }

    ENTRY test_computation {
      data = f32[2,2] parameter(0)
      i = u32[] constant(0)
      init = (u32[], f32[2,2]) tuple(i, data)
      while = (u32[], f32[2,2]) while(init), condition=while_condidtion,
          body=while_body
      ROOT data_ = f32[2,2] get-tuple-element(while), index=1
    }
  )";

  const int64_t kNumReplicas = 1;
  const int64_t kNumPartitions = 4;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas * kNumPartitions);

  // Parse HLO module.
  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);
  std::unique_ptr<VerifiedHloModule> module;
  TF_ASSERT_OK_AND_ASSIGN(module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  // Create input data.
  std::vector<Literal> literals;
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    float val = i + 1;
    literals.push_back(LiteralUtil::CreateR2<float>({{val, val}, {val, val}}));
  }
  std::vector<std::vector<Literal *>> inputs;
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    inputs.push_back({&literals[i]});
  }

  // Create device assignment running across partitions.
  DeviceAssignment device_assignment(/*replica_count=*/kNumReplicas,
                                     /*computation_count=*/kNumPartitions);
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    device_assignment(0, i) = i;
  }

  // Execute and check results.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), inputs,
                        /*num_replicas=*/kNumPartitions,
                        /*run_hlo_passes=*/false, &device_assignment));
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {0, 0}}, results[0]);
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {0, 0}}, results[1]);
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {0, 0}}, results[2]);
  LiteralTestUtil::ExpectR2Equal<float>({{1, 1}, {1, 1}}, results[3]);
}

XLA_TEST_F(CollectivePipelineParallelismTest, SendRecvLoop2Devices) {
  const absl::string_view kModuleStr = R"(
    HloModule test, num_partitions=2

    // 1 iteration so that we can test on 2 GPUs.
    while_condidtion {
      param = (u32[], f32[2,2]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      c1 = u32[] constant(1)
      ROOT cmp = pred[] compare(i, c1), direction=LT
    }

    while_body {
      param = (u32[], f32[2,2]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data = f32[2,2] get-tuple-element(param), index=1

      // Just send from GPU 0 to GPU 1 to avoid deadlock.
      after_all = token[] after-all()
      send_ctx = (f32[2,2], u32[], token[]) send(data, after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=1
      recv_ctx = (f32[2,2], u32[], token[]) recv(after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=2
      send_done = token[] send-done(send_ctx), channel_id=1
      recv_done = (f32[2,2], token[]) recv-done(recv_ctx), channel_id=2
      data_ = f32[2,2] get-tuple-element(recv_done), index=0

      c1 = u32[] constant(1)
      i_ = u32[] add(i, c1)
      ROOT result = (u32[], f32[2,2]) tuple(i_, data_)
    }

    ENTRY test_computation {
      data = f32[2,2] parameter(0)
      i = u32[] constant(0)
      init = (u32[], f32[2,2]) tuple(i, data)
      while = (u32[], f32[2,2]) while(init), condition=while_condidtion,
          body=while_body
      ROOT data_ = f32[2,2] get-tuple-element(while), index=1
    }
  )";

  const int64_t kNumReplicas = 1;
  const int64_t kNumPartitions = 2;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas * kNumPartitions);

  // Parse HLO module.
  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);
  std::unique_ptr<VerifiedHloModule> module;
  TF_ASSERT_OK_AND_ASSIGN(module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  // Create input data.
  std::vector<Literal> literals;
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    float val = i + 1;
    literals.push_back(LiteralUtil::CreateR2<float>({{val, val}, {val, val}}));
  }
  std::vector<std::vector<Literal *>> inputs;
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    inputs.push_back({&literals[i]});
  }

  // Create device assignment running across partitions.
  DeviceAssignment device_assignment(/*replica_count=*/kNumReplicas,
                                     /*computation_count=*/kNumPartitions);
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    device_assignment(0, i) = i;
  }

  // Execute and check results.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), inputs,
                        /*num_replicas=*/kNumPartitions,
                        /*run_hlo_passes=*/false, &device_assignment));
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {0, 0}}, results[0]);
  LiteralTestUtil::ExpectR2Equal<float>({{1, 1}, {1, 1}}, results[1]);
}

XLA_TEST_F(CollectivePipelineParallelismTest,
           PartiallyPipelinedAsyncSendRecvLoop) {
  const absl::string_view kModuleStr = R"(
    HloModule test, num_partitions=4

    while_condidtion {
      param = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          parameter(0)
      i = u32[] get-tuple-element(param), index=0
      c2 = u32[] constant(2)
      ROOT cmp = pred[] compare(i, c2), direction=LT
    }

    while_body {
      param = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          parameter(0)
      i = u32[] get-tuple-element(param), index=0
      send_ctx = get-tuple-element(param), index=1
      recv_ctx = get-tuple-element(param), index=2
      send_done = token[] send-done(send_ctx), channel_id=1
      recv_done = (f32[2,2], token[]) recv-done(recv_ctx), channel_id=2
      data = get-tuple-element(recv_done), index=0
      after_all = token[] after-all()
      send_ctx_ = (f32[2,2], u32[], token[]) send(data, after_all),
          frontend_attributes={
          _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}, channel_id=1
      recv_ctx_ = (f32[2,2], u32[], token[]) recv(after_all),
          frontend_attributes={
          _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}, channel_id=2
      c1 = u32[] constant(1)
      i_ = u32[] add(i, c1)
      ROOT result = (u32[], (f32[2,2], u32[], token[]),
          (f32[2,2], u32[], token[])) tuple(i_, send_ctx_, recv_ctx_)
    }

    ENTRY test_computation {
      data = f32[2,2] parameter(0)
      i = u32[] constant(0)
      after_all = token[] after-all()
      send_ctx_ = (f32[2,2], u32[], token[]) send(data, after_all),
          frontend_attributes={
          _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}, channel_id=1
      recv_ctx_ = (f32[2,2], u32[], token[]) recv(after_all),
          frontend_attributes={
          _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}, channel_id=2
      init = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          tuple(i, send_ctx_, recv_ctx_)
      while = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          while(init), condition=while_condidtion, body=while_body
      send_ctx = get-tuple-element(while), index=1
      recv_ctx = get-tuple-element(while), index=2
      send_done = token[] send-done(send_ctx), channel_id=1
      recv_done = (f32[2,2], token[]) recv-done(recv_ctx), channel_id=2
      ROOT data_ = get-tuple-element(recv_done), index=0
    }
  )";

  const int64_t kNumReplicas = 1;
  const int64_t kNumPartitions = 4;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas * kNumPartitions);

  // Parse HLO module.
  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);
  std::unique_ptr<VerifiedHloModule> module;
  TF_ASSERT_OK_AND_ASSIGN(module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  // Create input data.
  std::vector<Literal> literals;
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    float val = i + 1;
    literals.push_back(LiteralUtil::CreateR2<float>({{val, val}, {val, val}}));
  }
  std::vector<std::vector<Literal *>> inputs;
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    inputs.push_back({&literals[i]});
  }

  // Create device assignment running across partitions.
  DeviceAssignment device_assignment(/*replica_count=*/kNumReplicas,
                                     /*computation_count=*/kNumPartitions);
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    device_assignment(0, i) = i;
  }

  // Execute and check results.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), inputs,
                        /*num_replicas=*/kNumPartitions,
                        /*run_hlo_passes=*/false, &device_assignment));
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {0, 0}}, results[0]);
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {0, 0}}, results[1]);
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {0, 0}}, results[2]);
  LiteralTestUtil::ExpectR2Equal<float>({{1, 1}, {1, 1}}, results[3]);
}

XLA_TEST_F(CollectivePipelineParallelismTest,
           PartiallyPipelinedAsyncSendRecvLoop2Devices) {
  const absl::string_view kModuleStr = R"(
    HloModule test, num_partitions=2

    while_condidtion {
      param = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          parameter(0)
      i = u32[] get-tuple-element(param), index=0
      c2 = u32[] constant(2)
      ROOT cmp = pred[] compare(i, c2), direction=LT
    }

    while_body {
      param = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          parameter(0)
      i = u32[] get-tuple-element(param), index=0
      send_ctx = get-tuple-element(param), index=1
      recv_ctx = get-tuple-element(param), index=2
      send_done = token[] send-done(send_ctx), channel_id=1
      recv_done = (f32[2,2], token[]) recv-done(recv_ctx), channel_id=2
      data = get-tuple-element(recv_done), index=0
      after_all = token[] after-all()
      send_ctx_ = (f32[2,2], u32[], token[]) send(data, after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}}, 
          channel_id=1
      recv_ctx_ = (f32[2,2], u32[], token[]) recv(after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}}, 
          channel_id=2
      c1 = u32[] constant(1)
      i_ = u32[] add(i, c1)
      ROOT result = (u32[], (f32[2,2], u32[], token[]),
          (f32[2,2], u32[], token[])) tuple(i_, send_ctx_, recv_ctx_)
    }

    ENTRY test_computation {
      data = f32[2,2] parameter(0)
      i = u32[] constant(0)
      after_all = token[] after-all()
      send_ctx_ = (f32[2,2], u32[], token[]) send(data, after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=1
      recv_ctx_ = (f32[2,2], u32[], token[]) recv(after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}}, 
          channel_id=2
      init = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          tuple(i, send_ctx_, recv_ctx_)
      while = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          while(init), condition=while_condidtion, body=while_body
      send_ctx = get-tuple-element(while), index=1
      recv_ctx = get-tuple-element(while), index=2
      send_done = token[] send-done(send_ctx), channel_id=1
      recv_done = (f32[2,2], token[]) recv-done(recv_ctx), channel_id=2
      ROOT data_ = get-tuple-element(recv_done), index=0
    }
  )";

  const int64_t kNumReplicas = 1;
  const int64_t kNumPartitions = 2;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas * kNumPartitions);

  // Parse HLO module.
  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);
  std::unique_ptr<VerifiedHloModule> module;
  TF_ASSERT_OK_AND_ASSIGN(module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  // Create input data.
  std::vector<Literal> literals;
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    float val = i + 1;
    literals.push_back(LiteralUtil::CreateR2<float>({{val, val}, {val, val}}));
  }
  std::vector<std::vector<Literal *>> inputs;
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    inputs.push_back({&literals[i]});
  }

  // Create device assignment running across partitions.
  DeviceAssignment device_assignment(/*replica_count=*/kNumReplicas,
                                     /*computation_count=*/kNumPartitions);
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    device_assignment(0, i) = i;
  }

  // Execute and check results.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), inputs,
                        /*num_replicas=*/kNumPartitions,
                        /*run_hlo_passes=*/false, &device_assignment));
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {0, 0}}, results[0]);
  LiteralTestUtil::ExpectR2Equal<float>({{0, 0}, {0, 0}}, results[1]);
}

}  // namespace
}  // namespace xla

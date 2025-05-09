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
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tests/test_utils.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// Tests cross-GPU operations.
//
// Several tests requires at least four GPUs.  For instructions on running this
// within Google, see go/multi-gpu-unit-test.
class CollectivePipelineParallelismTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          DebugOptions::PipelineParallelismOptLevel> {
 public:
  CollectivePipelineParallelismTest() {
    VLOG(1) << "Running with " << num_devices() << " devices";
    xla_gpu_experimental_pipeline_parallelism_opt_level_ = GetParam();
  }

  HloModuleConfig GetModuleConfigForTest(int64_t replica_count = 1,
                                         int64_t num_partitions = 1) const {
    HloModuleConfig config =
        HloTestBase::GetModuleConfigForTest(replica_count, num_partitions);

    // Set debug options.
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_pipeline_parallelism_opt_level(
        xla_gpu_experimental_pipeline_parallelism_opt_level_);
    debug_options.set_xla_gpu_enable_latency_hiding_scheduler(true);
    debug_options.set_xla_gpu_collective_permute_decomposer_threshold(0);
    debug_options.set_xla_gpu_autotune_level(0);
    config.set_debug_options(debug_options);

    return config;
  }

  DebugOptions::PipelineParallelismOptLevel
      xla_gpu_experimental_pipeline_parallelism_opt_level_;
};

XLA_TEST_P(CollectivePipelineParallelismTest,
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
  const int64_t kNumPartitions = 1;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  // Parse HLO module.
  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);
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
XLA_TEST_P(CollectivePipelineParallelismTest, NaiveBFSMicrobatch4Replica4) {
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
  const int64_t kNumPartitions = 1;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  // Parse HLO module.
  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);
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
XLA_TEST_P(CollectivePipelineParallelismTest, NaiveBFSMicrobatch5Replica4) {
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
  const int64_t kNumPartitions = 1;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  // Parse HLO module.
  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);
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
XLA_TEST_P(CollectivePipelineParallelismTest,
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
  const int64_t kNumPartitions = 1;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  // Parse HLO module.
  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);
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
XLA_TEST_P(CollectivePipelineParallelismTest,
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
  const int64_t kNumPartitions = 1;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  // Parse HLO module.
  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);
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
XLA_TEST_P(CollectivePipelineParallelismTest,
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
  const int64_t kNumPartitions = 1;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  // Parse HLO module.
  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);
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

XLA_TEST_P(CollectivePipelineParallelismTest, SendRecvLoop) {
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
      data_cpy = f32[2,2] copy(data)
      send_ctx = (f32[2,2], u32[], token[]) send(data_cpy, after_all),
          frontend_attributes={
          _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}, channel_id=1
      recv_ctx = (f32[2,2], u32[], token[]) recv(after_all),
          frontend_attributes={
          _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}, channel_id=2,
          control-predecessors={data_cpy}
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
  if (test_runner().device_count() < kNumReplicas * kNumPartitions) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                 << " devices (" << test_runner().device_count()
                 << " available)";
  }

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

XLA_TEST_P(CollectivePipelineParallelismTest, SendRecvLoop2Devices) {
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
      data_cpy = f32[2,2] copy(data)
      send_ctx = (f32[2,2], u32[], token[]) send(data_cpy, after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=1
      recv_ctx = (f32[2,2], u32[], token[]) recv(after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=2, control-predecessors={data_cpy}
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
  if (test_runner().device_count() < kNumReplicas * kNumPartitions) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                 << " devices (" << test_runner().device_count()
                 << " available)";
  }

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

XLA_TEST_P(CollectivePipelineParallelismTest,
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
      data_cpy = f32[2,2] copy(data)
      send_ctx_ = (f32[2,2], u32[], token[]) send(data_cpy, after_all),
          frontend_attributes={
          _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}, channel_id=1
      recv_ctx_ = (f32[2,2], u32[], token[]) recv(after_all),
          frontend_attributes={
          _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}, channel_id=2,
          control-predecessors={data_cpy}
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
  if (test_runner().device_count() < kNumReplicas * kNumPartitions) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                 << " devices (" << test_runner().device_count()
                 << " available)";
  }

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

XLA_TEST_P(CollectivePipelineParallelismTest,
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
  if (test_runner().device_count() < kNumReplicas * kNumPartitions) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                 << " devices (" << test_runner().device_count()
                 << " available)";
  }

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

// This is the partially pipelined version of
// NaiveBFSMicrobatch5CircularRepeat2Replica4 and should yield the same results.
// TODO(b/383868854): replace this with GPU pipeliner implementation.
XLA_TEST_P(CollectivePipelineParallelismTest,
           NaiveBFSMb5Cr2Replica4SendRecvPartiallyPipelined) {
  constexpr char kMoreComputationsStr[] = R"(
  while_condition {
    tuple = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[],
      (f32[16], token[]), (f32[16], token[])) parameter(0)
    i = u32[] get-tuple-element(tuple), index=5
    n = u32[] constant(13)
    ROOT predicate = pred[] compare(i, n), direction=LT
  }

  while_body {
    tuple = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[],
      (f32[16], token[]), (f32[16], token[])) parameter(0)
    weights = f32[16,16] get-tuple-element(tuple), index=0
    input = f32[5,16] get-tuple-element(tuple), index=1
    output = f32[5,16] get-tuple-element(tuple), index=2
    buffer = f32[5,16] get-tuple-element(tuple), index=3
    prev_iteration_compute_res = f32[16] get-tuple-element(tuple), index=4
    i = u32[] get-tuple-element(tuple), index=5

    prev_iter_fwd_recv_done = (f32[16], token[])
      get-tuple-element(tuple), index=6
    prev_iter_bwd_recv_done = (f32[16], token[])
      get-tuple-element(tuple), index=7
    prev_stage_slice_fwd = f32[16] get-tuple-element(prev_iter_fwd_recv_done),
      index=0
    prev_stage_slice_bwd = f32[16] get-tuple-element(prev_iter_bwd_recv_done),
      index=0

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

    // Shift data to the next stage in the pipeline.
    after_all_fwd = token[] after-all()
    fwd_send = (f32[16], u32[], token[]) send(next_stage_slice, after_all_fwd),
      frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
    fwd_send_done = token[] send-done(fwd_send)

    // Select compute argument from previous stage or from input and perform
    // compute.
    is_read_input = pred[] call(i), to_apply=is_read_input_mb5
    compute_arg_bwd = f32[16] select(is_read_input, input_slice, prev_stage_slice_bwd)
    compute_res_bwd = f32[16] dot(weights, compute_arg_bwd),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
    is_device_zero = pred[] call(), to_apply=is_input_replica
    compute_arg_fwd = f32[16] select(is_device_zero,
      prev_stage_slice_bwd, prev_stage_slice_fwd)
    compute_res_fwd = f32[16] dot(weights, compute_arg_fwd),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}

    // Update buffers.
    compute_res = f32[16] select(is_device_zero, compute_res_bwd, compute_res_fwd)
    output_ = f32[5,16] call(output, compute_res, c1, i),
        to_apply=update_buffer_mb5
    buffer_ = f32[5,16] call(buffer, prev_iteration_compute_res, c4, i),
        to_apply=update_buffer_mb5


    after_all_bwd = token[] after-all()
    bwd_recv = (f32[16], u32[], token[]) recv(after_all_bwd),
      frontend_attributes={_xla_send_recv_source_target_pairs={{3,0}}},
      control-predecessors={fwd_send_done, fwd_send}
    bwd_recv_done = (f32[16], token[]) recv-done(bwd_recv),
      frontend_attributes={_xla_send_recv_source_target_pairs={{3,0}}}
    bwd_send = (f32[16], u32[], token[]) send(next_stage_slice, after_all_bwd),
      frontend_attributes={_xla_send_recv_source_target_pairs={{3,0}}},
      control-predecessors={bwd_recv_done, bwd_recv}
    bwd_send_done = token[] send-done(bwd_send)

    fwd_recv = (f32[16], u32[], token[]) recv(after_all_fwd),
      frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}},
      control-predecessors={bwd_send_done, bwd_send}
    fwd_recv_done = (f32[16], token[]) recv-done(fwd_recv),
      frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}

    i_ = add(i, c1)

    ROOT tuple_ = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[],
      (f32[16], token[]), (f32[16], token[])) tuple(weights, input, output_,
      buffer_, compute_res, i_, fwd_recv_done, bwd_recv_done)
  }

  ENTRY main {
    weights = f32[16,16] parameter(0)
    input = f32[5,16] parameter(1)

    cf0 = f32[] constant(0)
    output = f32[5,16] broadcast(cf0), dimensions={}
    buffer = f32[5,16] broadcast(cf0), dimensions={}
    prev_iteration_compute_res = f32[16] broadcast(cf0), dimensions={}
    c0 = u32[] constant(0)
    input_slice = f32[16] call(input, c0, c0), to_apply=read_buffer_mb5


    after_all_bwd = token[] after-all()
    bwd_recv = (f32[16], u32[], token[]) recv(after_all_bwd),
      frontend_attributes={_xla_send_recv_source_target_pairs={{3,0}}}
    bwd_recv_done = (f32[16], token[]) recv-done(bwd_recv)
    bwd_send = (f32[16], u32[], token[]) send(input_slice, after_all_bwd),
      frontend_attributes={_xla_send_recv_source_target_pairs={{3,0}}},
      control-predecessors={bwd_recv_done, bwd_recv}
    bwd_send_done = token[] send-done(bwd_send)

    after_all_fwd = token[] after-all()
    fwd_recv = (f32[16], u32[], token[]) recv(after_all_fwd),
      frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}},
      control-predecessors={bwd_send_done, bwd_send}
    fwd_recv_done = (f32[16], token[]) recv-done(fwd_recv),
      frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}

    // Iterate through pipeline stages.
    tuple = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[],
      (f32[16], token[]), (f32[16], token[])) tuple(weights, input, output,
      buffer, prev_iteration_compute_res, c0, fwd_recv_done, bwd_recv_done)
    tuple_ = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[],
      (f32[16], token[]), (f32[16], token[])) while(tuple),
      condition=while_condition, body=while_body


    // unroll while loop results
    weights_ = f32[16,16] get-tuple-element(tuple_), index=0
    input_ = f32[5,16] get-tuple-element(tuple_), index=1
    output_ = f32[5,16] get-tuple-element(tuple_), index=2
    buffer_ = f32[5,16] get-tuple-element(tuple_), index=3
    prev_iteration_compute_res_ = f32[16] get-tuple-element(tuple_), index=4
    i_ = u32[] get-tuple-element(tuple_), index=5
    prev_stage_fwd_recv_done_ = (f32[16], token[]) get-tuple-element(tuple_), index=6
    prev_stage_bwd_recv_done_ = (f32[16], token[]) get-tuple-element(tuple_), index=7
    prev_stage_slice_fwd_ = f32[16] get-tuple-element(prev_stage_fwd_recv_done_), index=0
    prev_stage_slice_bwd_ = f32[16] get-tuple-element(prev_stage_bwd_recv_done_), index=0

    c0_ = u32[] constant(0)
    c1_ = u32[] constant(1)
    c2_ = u32[] constant(2)
    c3_ = u32[] constant(3)
    c4_ = u32[] constant(4)
    c5_ = u32[] constant(5)

    // Read from buffers.
    input_slice_ = f32[16] call(input, c0_, i_), to_apply=read_buffer_mb5
    buffer_slice_ = f32[16] call(buffer, c3_, i_), to_apply=read_buffer_mb5

    // Shift data to the next stage in the pipeline.
    // Directly depends on the updated buffer of the previous iteration and,
    // therefore, depends on the previous iteration's compute.
    is_output_replica_ = pred[] call(), to_apply=is_output_replica
    next_stage_slice_ = select(is_output_replica_, buffer_slice_,
        prev_iteration_compute_res_)

    fwd_send = (f32[16], u32[], token[]) send(next_stage_slice_, after_all_fwd),
      frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
    fwd_send_done = token[] send-done(fwd_send)


    // Select compute argument from previous stage or from input and perform
    // compute.
    is_read_input_ = pred[] call(i_), to_apply=is_read_input_mb5
    compute_arg_bwd_ = f32[16] select(is_read_input_, input_slice_, prev_stage_slice_bwd_)
    compute_res_bwd_ = f32[16] dot(weights_, compute_arg_bwd_), lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    is_device_zero_ = pred[] call(), to_apply=is_input_replica
    compute_arg_fwd_ = f32[16] select(is_device_zero_, prev_stage_slice_bwd_, prev_stage_slice_fwd_)
    compute_res_fwd_ = f32[16] dot(weights_, compute_arg_fwd_), lhs_contracting_dims={1},
        rhs_contracting_dims={0}

    // Update buffers.
    compute_res_ = f32[16] select(is_device_zero_, compute_res_bwd_, compute_res_fwd_)
    ROOT output__ = f32[5,16] call(output_, compute_res_, c1_, i_),
        to_apply=update_buffer_mb5

  }
  )";

  const int64_t kNumReplicas = 4;
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(GetModuleStrWithCommonComputations(
                                       /*name=*/"test", kMoreComputationsStr),
                                   config));

  const int64_t kInputSize = 16;
  Literal weights_r0 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 1.0);
  Literal weights_r1 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 2.0);
  Literal weights_r2 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 3.0);
  Literal weights_r3 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 4.0);

  const int64_t kMicrobatches = 5;
  Literal real_input =
      LiteralUtil::CreateFingerprintMatixR2<float>(kMicrobatches, kInputSize);
  Literal fake_input = LiteralUtil::CreateFull<float>(
      {kMicrobatches, kInputSize}, /*value=*/0.0);

  const float kExpectedFactor = 1.0 * 2.0 * 3.0 * 4.0 * 1.0 * 2.0 * 3.0 * 4.0;
  Literal expected_output = LiteralUtil::CreateFingerprintMatixR2<float>(
      kMicrobatches, kInputSize, /*scale=*/kExpectedFactor);
  std::vector<std::vector<Literal *>> args = {{&weights_r0, &real_input},
                                              {&weights_r1, &fake_input},
                                              {&weights_r2, &fake_input},
                                              {&weights_r3, &fake_input}};
  // TODO(rosiezou): enable send/recv combiner pass.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), args, kNumReplicas,
                        /*run_hlo_passes=*/true));
  EXPECT_TRUE(LiteralTestUtil::NearOrEqual(expected_output, results[3],
                                           ErrorSpec{1e-5, 1e-5}));
}

// This is the async-grouped version of
// NaiveBFSMicrobatch5CircularRepeat2Replica4 and should yield the same results.
// TODO(b/383868854): replace this with GPU pipeliner implementation.
XLA_TEST_P(CollectivePipelineParallelismTest,
           NaiveBFSMb5Cr2Replica4SendRecvAsyncGroup) {
  constexpr char kMoreComputationsStr[] = R"(

  wrapped_send_recv_1 {
    fwd_send_data = f32[16] parameter(0)
    fwd_send_after_all = token[] parameter(1)
    fwd_send = (f32[16], u32[], token[]) send(fwd_send_data, fwd_send_after_all),
      frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}

    fwd_recv_after_all = token[] parameter(2)
    fwd_recv = (f32[16], u32[], token[]) recv(fwd_recv_after_all), frontend_attributes={
        _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}

    bwd_send_data = f32[16] parameter(3)
    bwd_send_after_all = token[] parameter(4)
    bwd_send = (f32[16], u32[], token[]) send(bwd_send_data, bwd_send_after_all), frontend_attributes={
        _xla_send_recv_source_target_pairs={{3,0}}}

    bwd_recv_after_all = token[] parameter(5)
    bwd_recv = (f32[16], u32[], token[]) recv(bwd_recv_after_all), frontend_attributes={
        _xla_send_recv_source_target_pairs={{3,0}}}

    ROOT out = ((f32[16], u32[], token[]),(f32[16], u32[], token[]),
      (f32[16], u32[], token[]),(f32[16], u32[], token[])) tuple(fwd_send,
      fwd_recv, bwd_send, bwd_recv)

  }

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


    // Shift data to the next stage in the pipeline.
    after_all_fwd = token[] after-all()
    after_all_bwd = token[] after-all()

    async_comp_start = (( f32[16], token[], token[], f32[16], token[], token[]),
      ((f32[16], u32[], token[]), (f32[16], u32[], token[]), (f32[16], u32[], token[]),
      (f32[16], u32[], token[])), s32[]) async-start(next_stage_slice,
        after_all_fwd, after_all_fwd, next_stage_slice,
        after_all_bwd, after_all_bwd), calls=wrapped_send_recv_1

    async_comp_done = ((f32[16], u32[], token[]), (f32[16], u32[], token[]),
      (f32[16], u32[], token[]), (f32[16], u32[], token[])) async-done(async_comp_start)
    unpack_fwd_recv = (f32[16], u32[], token[]) get-tuple-element(async_comp_done), index=1
    fwd_recv_data = f32[16] get-tuple-element(unpack_fwd_recv), index=0
    fwd_recv_token = token[] get-tuple-element(unpack_fwd_recv), index=2
    fwd_recv_done = (f32[16], token[]) tuple(fwd_recv_data, fwd_recv_token),
      control-predecessors={async_comp_start}

    unpack_bwd_recv = (f32[16], u32[], token[]) get-tuple-element(async_comp_done), index=3
    bwd_recv_data = f32[16] get-tuple-element(unpack_bwd_recv), index=0
    bwd_recv_token = token[] get-tuple-element(unpack_bwd_recv), index=2
    bwd_recv_done = (f32[16], token[]) tuple(bwd_recv_data, bwd_recv_token),
      control-predecessors={async_comp_start}
    prev_stage_slice_fwd = f32[16] get-tuple-element(fwd_recv_done), index=0
    prev_stage_slice_bwd = f32[16] get-tuple-element(bwd_recv_done), index=0


    // Select compute argument from previous stage or from input and perform
    // compute.
    is_read_input = pred[] call(i), to_apply=is_read_input_mb5
    compute_arg_bwd = f32[16] select(is_read_input, input_slice, prev_stage_slice_bwd)
    compute_res_bwd = f32[16] dot(weights, compute_arg_bwd), lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    is_device_zero = pred[] call(), to_apply=is_input_replica
    compute_arg_fwd = f32[16] select(is_device_zero, prev_stage_slice_bwd, prev_stage_slice_fwd)
    compute_res_fwd = f32[16] dot(weights, compute_arg_fwd), lhs_contracting_dims={1},
        rhs_contracting_dims={0}

    // Update buffers.
    compute_res = f32[16] select(is_device_zero, compute_res_bwd, compute_res_fwd)
    output_ = f32[5,16] call(output, compute_res, c2, i),
        to_apply=update_buffer_mb5
    buffer_ = f32[5,16] call(buffer, compute_res, c0, i),
        to_apply=update_buffer_mb5

    i_ = add(i, c1)

    ROOT tuple_ = (f32[16,16], f32[5,16], f32[5,16], f32[5,16], f32[16], u32[])
        tuple(weights, input, output_, buffer_, compute_res, i_)

    unpack-send-done1 = (f32[16], u32[], token[]) get-tuple-element(async_comp_done), index=0
    send-done1 = token[] get-tuple-element(unpack-send-done1), index=2
    unpack-send-done2 = (f32[16], u32[], token[]) get-tuple-element(async_comp_done), index=2
    send-done2 = token[] get-tuple-element(unpack-send-done2), index=2
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
  if (test_runner().device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << test_runner().device_count() << " available)";
  }

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(GetModuleStrWithCommonComputations(
                                       /*name=*/"test", kMoreComputationsStr),
                                   config));

  const int64_t kInputSize = 16;
  Literal weights_r0 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 1.0);
  Literal weights_r1 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 2.0);
  Literal weights_r2 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 3.0);
  Literal weights_r3 = LiteralUtil::MakeScalarMatrixR2<float>(kInputSize, 4.0);

  const int64_t kMicrobatches = 5;
  Literal real_input =
      LiteralUtil::CreateFingerprintMatixR2<float>(kMicrobatches, kInputSize);
  Literal fake_input =
      LiteralUtil::CreateFull<float>({kMicrobatches, kInputSize}, 0.0);

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
  EXPECT_TRUE(LiteralTestUtil::NearOrEqual(
      expected_output, results[3],
      ErrorSpec{/*abs_error=*/1e-5, /*rel_error=*/1e-5}));
}

XLA_TEST_P(CollectivePipelineParallelismTest, JaxExampleWithDecomposedCycle) {
  constexpr char kModuleStr[] = R"(
HloModule jit_entry_computation, entry_computation_layout={
    (f32[4,4096,4096]{2,1,0}, f32[4,5,4096,8192]{3,2,1,0})->
    f32[4,5,4096,8192]{3,2,1,0}},
    allow_spmd_sharding_propagation_to_parameters={false,false},
    allow_spmd_sharding_propagation_to_output={true}, num_partitions=4

%_where.10 (Arg_0.11: pred[], Arg_1.12: s32[], Arg_2.13: s32[]) -> s32[] {
  %Arg_0.11 = pred[] parameter(0)
  %Arg_1.12 = s32[] parameter(1)
  %Arg_2.13 = s32[] parameter(2)
  ROOT %select.14 = s32[] select(%Arg_0.11, %Arg_1.12, %Arg_2.13)
}

%remainder.15 (Arg_0.16: s32[], Arg_1.17: s32[]) -> s32[] {
  %Arg_0.16 = s32[] parameter(0)
  %Arg_1.17 = s32[] parameter(1)
  %constant.19 = s32[] constant(0)
  %compare.20 = pred[] compare(%Arg_1.17, %constant.19), direction=EQ
  %constant.18 = s32[] constant(1)
  %call.21 = s32[] call(%compare.20, %constant.18, %Arg_1.17),
      to_apply=%_where.10
  %remainder.22 = s32[] remainder(%Arg_0.16, %call.21)
  %compare.24 = pred[] compare(%remainder.22, %constant.19), direction=LT
  %compare.25 = pred[] compare(%call.21, %constant.19), direction=LT
  %compare.26 = pred[] compare(%compare.24, %compare.25), direction=NE
  %compare.23 = pred[] compare(%remainder.22, %constant.19), direction=NE
  %and.27 = pred[] and(%compare.26, %compare.23)
  %add.28 = s32[] add(%remainder.22, %call.21)
  ROOT %select.29 = s32[] select(%and.27, %add.28, %remainder.22)
}

%_pad.30 (Arg_0.31: f32[4,1,4096,8192], Arg_1.32: s32[]) -> f32[5,1,4096,8192] {
  %Arg_0.31 = f32[4,1,4096,8192]{3,2,1,0} parameter(0)
  %Arg_1.32 = s32[] parameter(1)
  %convert.33 = f32[] convert(%Arg_1.32)
  ROOT %pad.34 = f32[5,1,4096,8192]{3,2,1,0} pad(%Arg_0.31, %convert.33),
      padding=1_0x0_0x0_0x0_0
}

%_where_0.35 (Arg_0.36: pred[], Arg_1.37: f32[4,1,4096,8192],
    Arg_2.38: f32[4,1,4096,8192]) -> f32[4,1,4096,8192] {
  %Arg_0.36 = pred[] parameter(0)
  %broadcast.39 = pred[4,1,4096,8192]{3,2,1,0} broadcast(%Arg_0.36),
      dimensions={}
  %Arg_1.37 = f32[4,1,4096,8192]{3,2,1,0} parameter(1)
  %Arg_2.38 = f32[4,1,4096,8192]{3,2,1,0} parameter(2)
  ROOT %select.40 = f32[4,1,4096,8192]{3,2,1,0} select(%broadcast.39, %Arg_1.37,
      %Arg_2.38)
}

%_where_1.41 (Arg_0.42: pred[4,1,4096,8192], Arg_1.43: f32[4,1,4096,8192],
    Arg_2.44: f32[4,1,4096,8192]) -> f32[4,1,4096,8192] {
  %Arg_0.42 = pred[4,1,4096,8192]{3,2,1,0} parameter(0)
  %Arg_1.43 = f32[4,1,4096,8192]{3,2,1,0} parameter(1)
  %Arg_2.44 = f32[4,1,4096,8192]{3,2,1,0} parameter(2)
  ROOT %select.45 = f32[4,1,4096,8192]{3,2,1,0} select(%Arg_0.42, %Arg_1.43,
      %Arg_2.44)
}

%_where.46 (Arg_0.47: pred[], Arg_1.48: s32[], Arg_2.49: s32[]) -> s32[] {
  %Arg_0.47 = pred[] parameter(0)
  %Arg_1.48 = s32[] parameter(1)
  %Arg_2.49 = s32[] parameter(2)
  ROOT %select.50 = s32[] select(%Arg_0.47, %Arg_1.48, %Arg_2.49)
}

%remainder.51 (Arg_0.52: s32[], Arg_1.53: s32[]) -> s32[] {
  %Arg_0.52 = s32[] parameter(0)
  %Arg_1.53 = s32[] parameter(1)
  %constant.55 = s32[] constant(0)
  %compare.56 = pred[] compare(%Arg_1.53, %constant.55), direction=EQ
  %constant.54 = s32[] constant(1)
  %call.57 = s32[] call(%compare.56, %constant.54, %Arg_1.53),
      to_apply=%_where.46
  %remainder.58 = s32[] remainder(%Arg_0.52, %call.57)
  %compare.60 = pred[] compare(%remainder.58, %constant.55), direction=LT
  %compare.61 = pred[] compare(%call.57, %constant.55), direction=LT
  %compare.62 = pred[] compare(%compare.60, %compare.61), direction=NE
  %compare.59 = pred[] compare(%remainder.58, %constant.55), direction=NE
  %and.63 = pred[] and(%compare.62, %compare.59)
  %add.64 = s32[] add(%remainder.58, %call.57)
  ROOT %select.65 = s32[] select(%and.63, %add.64, %remainder.58)
}

%_pad_2.66 (Arg_0.67: f32[4,1,4096,8192], Arg_1.68: s32[])
    -> f32[7,1,4096,8192] {
  %Arg_0.67 = f32[4,1,4096,8192]{3,2,1,0} parameter(0)
  %Arg_1.68 = s32[] parameter(1)
  %convert.69 = f32[] convert(%Arg_1.68)
  ROOT %pad.70 = f32[7,1,4096,8192]{3,2,1,0} pad(%Arg_0.67, %convert.69),
      padding=0_3x0_0x0_0x0_0
}

%_where.71 (Arg_0.72: pred[], Arg_1.73: s32[], Arg_2.74: s32[]) -> s32[] {
  %Arg_0.72 = pred[] parameter(0)
  %Arg_1.73 = s32[] parameter(1)
  %Arg_2.74 = s32[] parameter(2)
  ROOT %select.75 = s32[] select(%Arg_0.72, %Arg_1.73, %Arg_2.74)
}

%remainder.76 (Arg_0.77: s32[], Arg_1.78: s32[]) -> s32[] {
  %Arg_0.77 = s32[] parameter(0)
  %Arg_1.78 = s32[] parameter(1)
  %constant.80 = s32[] constant(0)
  %compare.81 = pred[] compare(%Arg_1.78, %constant.80), direction=EQ
  %constant.79 = s32[] constant(1)
  %call.82 = s32[] call(%compare.81, %constant.79, %Arg_1.78),
      to_apply=%_where.71
  %remainder.83 = s32[] remainder(%Arg_0.77, %call.82)
  %compare.85 = pred[] compare(%remainder.83, %constant.80), direction=LT
  %compare.86 = pred[] compare(%call.82, %constant.80), direction=LT
  %compare.87 = pred[] compare(%compare.85, %compare.86), direction=NE
  %compare.84 = pred[] compare(%remainder.83, %constant.80), direction=NE
  %and.88 = pred[] and(%compare.87, %compare.84)
  %add.89 = s32[] add(%remainder.83, %call.82)
  ROOT %select.90 = s32[] select(%and.88, %add.89, %remainder.83)
}

%None.91 (Arg_0.92: f32[4,4096,4096], Arg_1.93: f32[4,5,4096,8192],
    Arg_2.94: f32[4,5,4096,8192], Arg_3.95: f32[4,1,4096,8192],
    Arg_4.96: f32[4,1,4096,8192], Arg_5.97: s32[])
    -> (f32[4,4096,4096], f32[4,5,4096,8192], f32[4,5,4096,8192],
    f32[4,1,4096,8192], f32[4,1,4096,8192]) {
  %Arg_0.92 = f32[4,4096,4096]{2,1,0} parameter(0)
  %Arg_1.93 = f32[4,5,4096,8192]{3,2,1,0} parameter(1)
  %Arg_2.94 = f32[4,5,4096,8192]{3,2,1,0} parameter(2)
  %iota.113 = s32[4]{0} iota(), iota_dimension=0
  %broadcast.114 = s32[4,1,4096,8192]{3,2,1,0} broadcast(%iota.113),
      dimensions={0}
  %constant.98 = s32[] constant(0)
  %broadcast.99 = s32[4,1,4096,8192]{3,2,1,0} broadcast(%constant.98),
      dimensions={}
  %compare.115 = pred[4,1,4096,8192]{3,2,1,0} compare(%broadcast.114,
      %broadcast.99), direction=EQ
  %Arg_5.97 = s32[] parameter(5)
  %constant.102 = s32[] constant(5)
  %compare.111 = pred[] compare(%Arg_5.97, %constant.102), direction=LT
  %constant.103 = s32[] constant(0)
  %call.104 = s32[] call(%Arg_5.97, %constant.102), to_apply=%remainder.15
  %compare.105 = pred[] compare(%call.104, %constant.103), direction=LT
  %add.106 = s32[] add(%call.104, %constant.102)
  %select.107 = s32[] select(%compare.105, %add.106, %call.104)
  %dynamic-slice.108 = f32[4,1,4096,8192]{3,2,1,0} dynamic-slice(%Arg_1.93,
      %constant.103, %select.107, %constant.103, %constant.103),
      dynamic_slice_sizes={4,1,4096,8192}
  %Arg_4.96 = f32[4,1,4096,8192]{3,2,1,0} parameter(4)
  %call.112 = f32[4,1,4096,8192]{3,2,1,0} call(%compare.111, %dynamic-slice.108,
      %Arg_4.96), to_apply=%_where_0.35
  %Arg_3.95 = f32[4,1,4096,8192]{3,2,1,0} parameter(3)
  %call.109 = f32[5,1,4096,8192]{3,2,1,0} call(%Arg_3.95, %constant.103),
      to_apply=%_pad.30
  %slice.110 = f32[4,1,4096,8192]{3,2,1,0} slice(%call.109),
      slice={[0:4], [0:1], [0:4096], [0:8192]}
  %call.116 = f32[4,1,4096,8192]{3,2,1,0} call(%compare.115, %call.112,
      %slice.110), to_apply=%_where_1.41
  %reshape.117 = f32[4,4096,8192]{2,1,0} reshape(%call.116)
  %dot.118 = f32[4,4096,8192]{2,1,0} dot(%Arg_0.92, %reshape.117),
      lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0},
      rhs_contracting_dims={1}
  %reshape.150 = f32[4,1,4096,8192]{3,2,1,0} reshape(%dot.118)
  %constant.100 = s32[] constant(2)
  %add.159 = s32[] add(%Arg_5.97, %constant.100)
  %call.160 = s32[] call(%add.159, %constant.102), to_apply=%remainder.76
  %compare.161 = pred[] compare(%call.160, %constant.103), direction=LT
  %add.162 = s32[] add(%call.160, %constant.102)
  %select.163 = s32[] select(%compare.161, %add.162, %call.160)
  %dynamic-update-slice.164 = f32[4,5,4096,8192]{3,2,1,0}
      dynamic-update-slice(%Arg_2.94, %reshape.150, %constant.103, %select.163,
      %constant.103, /*index=5*/%constant.103)
  %constant.101 = s32[] constant(1)
  %add.151 = s32[] add(%Arg_5.97, %constant.101)
  %call.152 = s32[] call(%add.151, %constant.102), to_apply=%remainder.51
  %compare.153 = pred[] compare(%call.152, %constant.103), direction=LT
  %add.154 = s32[] add(%call.152, %constant.102)
  %select.155 = s32[] select(%compare.153, %add.154, %call.152)
  %dynamic-slice.156 = f32[4,1,4096,8192]{3,2,1,0} dynamic-slice(%Arg_2.94,
      %constant.103, %select.155, %constant.103, %constant.103),
      dynamic_slice_sizes={4,1,4096,8192}
  %call.157 = f32[7,1,4096,8192]{3,2,1,0} call(%dynamic-slice.156,
      %constant.103), to_apply=%_pad_2.66
  %slice.158 = f32[4,1,4096,8192]{3,2,1,0} slice(%call.157),
      slice={[3:7], [0:1], [0:4096], [0:8192]}
  ROOT %tuple.165 = (f32[4,4096,4096]{2,1,0}, f32[4,5,4096,8192]{3,2,1,0},
      f32[4,5,4096,8192]{3,2,1,0}, f32[4,1,4096,8192]{3,2,1,0},
      f32[4,1,4096,8192]{3,2,1,0}) tuple(%Arg_0.92, %Arg_1.93,
      %dynamic-update-slice.164, %reshape.150, %slice.158)
}

%region_0.166 (arg_tuple.167: (s32[], f32[4,4096,4096], f32[4,5,4096,8192],
    f32[4,5,4096,8192], f32[4,1,4096,8192], /*index=5*/f32[4,1,4096,8192],
    s32[13])) -> (s32[], f32[4,4096,4096], f32[4,5,4096,8192],
    f32[4,5,4096,8192], f32[4,1,4096,8192], /*index=5*/f32[4,1,4096,8192],
    s32[13]) {
  %arg_tuple.167 = (s32[], f32[4,4096,4096]{2,1,0}, f32[4,5,4096,8192]{3,2,1,0},
      f32[4,5,4096,8192]{3,2,1,0}, f32[4,1,4096,8192]{3,2,1,0},
      /*index=5*/f32[4,1,4096,8192]{3,2,1,0}, s32[13]{0}) parameter(0)
  %get-tuple-element.168 = s32[] get-tuple-element(%arg_tuple.167), index=0
  %constant.175 = s32[] constant(1)
  %add.184 = s32[] add(%get-tuple-element.168, %constant.175)
  %get-tuple-element.169 = f32[4,4096,4096]{2,1,0}
      get-tuple-element(%arg_tuple.167), index=1
  %get-tuple-element.170 = f32[4,5,4096,8192]{3,2,1,0}
      get-tuple-element(%arg_tuple.167), index=2
  %get-tuple-element.171 = f32[4,5,4096,8192]{3,2,1,0}
      get-tuple-element(%arg_tuple.167), index=3
  %get-tuple-element.172 = f32[4,1,4096,8192]{3,2,1,0}
      get-tuple-element(%arg_tuple.167), index=4
  %get-tuple-element.173 = f32[4,1,4096,8192]{3,2,1,0}
      get-tuple-element(%arg_tuple.167), index=5
  %get-tuple-element.174 = s32[13]{0} get-tuple-element(%arg_tuple.167), index=6
  %dynamic-slice.176 = s32[1]{0} dynamic-slice(%get-tuple-element.174,
      %get-tuple-element.168), dynamic_slice_sizes={1}
  %reshape.177 = s32[] reshape(%dynamic-slice.176)
  %call.178 = (f32[4,4096,4096]{2,1,0}, f32[4,5,4096,8192]{3,2,1,0},
      f32[4,5,4096,8192]{3,2,1,0}, f32[4,1,4096,8192]{3,2,1,0},
      f32[4,1,4096,8192]{3,2,1,0}) call(%get-tuple-element.169,
      %get-tuple-element.170, %get-tuple-element.171, %get-tuple-element.172,
      %get-tuple-element.173, /*index=5*/%reshape.177), to_apply=%None.91
  %get-tuple-element.179 = f32[4,4096,4096]{2,1,0} get-tuple-element(%call.178),
      index=0
  %get-tuple-element.180 = f32[4,5,4096,8192]{3,2,1,0}
      get-tuple-element(%call.178), index=1
  %get-tuple-element.181 = f32[4,5,4096,8192]{3,2,1,0}
      get-tuple-element(%call.178), index=2
  %get-tuple-element.182 = f32[4,1,4096,8192]{3,2,1,0}
      get-tuple-element(%call.178), index=3
  %get-tuple-element.183 = f32[4,1,4096,8192]{3,2,1,0}
      get-tuple-element(%call.178), index=4
  ROOT %tuple.185 = (s32[], f32[4,4096,4096]{2,1,0},
      f32[4,5,4096,8192]{3,2,1,0}, f32[4,5,4096,8192]{3,2,1,0},
      f32[4,1,4096,8192]{3,2,1,0}, /*index=5*/f32[4,1,4096,8192]{3,2,1,0},
      s32[13]{0}) tuple(%add.184, %get-tuple-element.179,
      %get-tuple-element.180, %get-tuple-element.181, %get-tuple-element.182,
      /*index=5*/%get-tuple-element.183, %get-tuple-element.174)
}

%region_1.186 (arg_tuple.187: (s32[], f32[4,4096,4096], f32[4,5,4096,8192],
    f32[4,5,4096,8192], f32[4,1,4096,8192], /*index=5*/f32[4,1,4096,8192],
    s32[13])) -> pred[] {
  %arg_tuple.187 = (s32[], f32[4,4096,4096]{2,1,0}, f32[4,5,4096,8192]{3,2,1,0},
      f32[4,5,4096,8192]{3,2,1,0}, f32[4,1,4096,8192]{3,2,1,0},
      /*index=5*/f32[4,1,4096,8192]{3,2,1,0}, s32[13]{0}) parameter(0)
  %get-tuple-element.189 = f32[4,4096,4096]{2,1,0}
      get-tuple-element(%arg_tuple.187), index=1
  %get-tuple-element.190 = f32[4,5,4096,8192]{3,2,1,0}
      get-tuple-element(%arg_tuple.187), index=2
  %get-tuple-element.191 = f32[4,5,4096,8192]{3,2,1,0}
      get-tuple-element(%arg_tuple.187), index=3
  %get-tuple-element.192 = f32[4,1,4096,8192]{3,2,1,0}
      get-tuple-element(%arg_tuple.187), index=4
  %get-tuple-element.193 = f32[4,1,4096,8192]{3,2,1,0}
      get-tuple-element(%arg_tuple.187), index=5
  %get-tuple-element.194 = s32[13]{0} get-tuple-element(%arg_tuple.187), index=6
  %get-tuple-element.188 = s32[] get-tuple-element(%arg_tuple.187), index=0
  %constant.195 = s32[] constant(13)
  ROOT %compare.196 = pred[] compare(%get-tuple-element.188, %constant.195),
      direction=LT
}

ENTRY %main.204 (Arg_0.1: f32[4,4096,4096], Arg_1.2: f32[4,5,4096,8192])
    -> f32[4,5,4096,8192] {
  %constant.3 = s32[] constant(0)
  %Arg_0.1 = f32[4,4096,4096]{2,1,0} parameter(0),
      sharding={devices=[4,1,1]<=[4]}
  %Arg_1.2 = f32[4,5,4096,8192]{3,2,1,0} parameter(1),
      sharding={devices=[4,1,1,1]<=[4]}
  %constant.4 = f32[] constant(0)
  %broadcast.5 = f32[4,5,4096,8192]{3,2,1,0} broadcast(%constant.4),
      dimensions={}
  %constant.6 = f32[] constant(0)
  %broadcast.7 = f32[4,1,4096,8192]{3,2,1,0} broadcast(%constant.6),
      dimensions={}
  %iota.8 = s32[13]{0} iota(), iota_dimension=0
  %tuple.9 = (s32[], f32[4,4096,4096]{2,1,0}, f32[4,5,4096,8192]{3,2,1,0},
      f32[4,5,4096,8192]{3,2,1,0}, f32[4,1,4096,8192]{3,2,1,0},
      /*index=5*/f32[4,1,4096,8192]{3,2,1,0}, s32[13]{0}) tuple(%constant.3,
      %Arg_0.1, %Arg_1.2, %broadcast.5, %broadcast.7, /*index=5*/%broadcast.7,
      %iota.8)
  %while.197 = (s32[], f32[4,4096,4096]{2,1,0}, f32[4,5,4096,8192]{3,2,1,0},
      f32[4,5,4096,8192]{3,2,1,0}, f32[4,1,4096,8192]{3,2,1,0},
      /*index=5*/f32[4,1,4096,8192]{3,2,1,0}, s32[13]{0}) while(%tuple.9),
      condition=%region_1.186, body=%region_0.166
  %get-tuple-element.198 = s32[] get-tuple-element(%while.197), index=0
  %get-tuple-element.199 = f32[4,4096,4096]{2,1,0}
      get-tuple-element(%while.197), index=1
  %get-tuple-element.200 = f32[4,5,4096,8192]{3,2,1,0}
      get-tuple-element(%while.197), index=2
  ROOT %get-tuple-element.201 = f32[4,5,4096,8192]{3,2,1,0}
      get-tuple-element(%while.197), index=3
  %get-tuple-element.202 = f32[4,1,4096,8192]{3,2,1,0}
      get-tuple-element(%while.197), index=4
  %get-tuple-element.203 = f32[4,1,4096,8192]{3,2,1,0}
      get-tuple-element(%while.197), index=5
}
  )";

  const int64_t kNumReplicas = 1;
  const int64_t kNumPartitions = 4;
  if (test_runner().device_count() < kNumReplicas * kNumPartitions) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                 << " devices (" << test_runner().device_count()
                 << " available)";
  }

  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  // Create device assignment running across partitions.
  DeviceAssignment device_assignment(/*replica_count=*/kNumReplicas,
                                     /*computation_count=*/kNumPartitions);
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    device_assignment(0, i) = i;
  }

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> fake_args,
                          MakeFakeArguments(module.get()));
  std::vector<Literal *> args;
  for (auto &arg : fake_args) {
    args.push_back(&arg);
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), args,
                        /*num_replicas=*/kNumPartitions, &device_assignment,
                        /*run_hlo_passes=*/true, /*use_threads=*/true));
  ASSERT_EQ(results.size(), kNumPartitions);
}

INSTANTIATE_TEST_SUITE_P(
    CollectivePipelineParallelismTestWithAndWithoutOpts,
    CollectivePipelineParallelismTest,
    ::testing::ValuesIn({DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE,
                         DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE}),
    ::testing::PrintToStringParamName());

}  // namespace
}  // namespace xla

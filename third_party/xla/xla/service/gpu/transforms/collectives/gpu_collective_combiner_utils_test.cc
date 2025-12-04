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

#include "xla/service/gpu/transforms/collectives/gpu_collective_combiner_utils.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_pipeliner.h"
#include "xla/service/collective_pipeliner_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

using CollectiveCombinerUtilsTest = HloHardwareIndependentTestBase;

TEST_F(CollectiveCombinerUtilsTest,
       AppendPipelinedInstructionAppendsPipelinedInstructionInfoForward) {
  // This is just a canonical IR which makes it easy to pipeline a collective
  // forward – in this example AllReduce.
  absl::string_view kHloText = R"(
  HloModule module
  add {
    lhs = bf16[] parameter(0)
    rhs = bf16[] parameter(1)
    ROOT add = bf16[] add(lhs, rhs)
  }

  while_cond {
    param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
    gte = s32[] get-tuple-element(param), index=0
    constant.1 = s32[] constant(3)
    ROOT cmp = pred[] compare(gte, constant.1), direction=LT
  }

  while_body {
    param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
    current-loop-index = s32[] get-tuple-element(param), index=0
    output-buffer = bf16[3,8,128] get-tuple-element(param), index=1
    input-buffer = bf16[3,8,128] get-tuple-element(param), index=2
    constant.1 = s32[] constant(1)
    next-loop-index = s32[] add(current-loop-index, constant.1)
    constant.0 = s32[] constant(0)
    sliced-input-buffer = bf16[1,8,128] dynamic-slice(input-buffer,
      current-loop-index, constant.0, constant.0), dynamic_slice_sizes={1,8,128}
    all-reduce = bf16[1,8,128] all-reduce(sliced-input-buffer),
      replica_groups={}, to_apply=add, channel_id=1
    dynamic-update-slice = bf16[3,8,128] dynamic-update-slice(output-buffer,
      all-reduce, current-loop-index, constant.0, constant.0)
    ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(next-loop-index,
      dynamic-update-slice, input-buffer)
  }

  ENTRY entry {
    c0 = s32[] constant(0)
    p0 = bf16[3,8,128] parameter(0)
    tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
    while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple),
      condition=while_cond, body=while_body
    ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  // This config is taken from the gpu_compiler.cc configuration of the forward
  // pipeliner.
  CollectivePipeliner::Config config{
      /*level_to_operate_on=*/0,
      /*max_pipelining_per_loop=*/INT64_MAX,
      /*last_run=*/true,
      /*pipeline_use_tree=*/false,
      /*process_different_sized_ops=*/true,
      /*pipelining_direction=*/
      collective_pipeliner_utils::PipeliningDirection::kForward,
      /*should_process=*/HloPredicateIsOp<HloOpcode::kAllReduce>,
      /*acceptable_formatting=*/HloPredicateTrue,
      /*reuse_pipelined_op_buffer=*/HloPredicateFalse,
  };
  config.postprocess_pipelined_ops = AppendPipelinedInstruction;

  HloPassPipeline pipeline("collective-pipeliner");
  pipeline.AddPass<CollectivePipeliner>(config);
  pipeline.AddPass<HloDCE>(/*remove_cross_partition_collective_ops=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_TRUE(changed);

  hlo_query::ForEachInstructionWithOpcode(
      *module, HloOpcode::kAllReduce, [](HloInstruction* instr) {
        EXPECT_TRUE(instr->backend_config<GpuBackendConfig>()
                        ->collective_backend_config()
                        .is_pipelined());
      });

  hlo_query::ForEachInstructionWithPred(
      *module, HloPredicateIsNotOp<HloOpcode::kAllReduce>,
      [](HloInstruction* instr) {
        EXPECT_FALSE(instr->backend_config<GpuBackendConfig>()
                         ->collective_backend_config()
                         .is_pipelined());
      });
}

TEST_F(CollectiveCombinerUtilsTest,
       AppendPipelinedInstructionForwardFormattingOps) {
  // This is just a canonical IR which makes it easy to pipeline a collective
  // forward – in this example AllReduce.
  absl::string_view kHloText = R"(
  HloModule module
  add {
    lhs = bf16[] parameter(0)
    rhs = bf16[] parameter(1)
    ROOT add = bf16[] add(lhs, rhs)
  }

  while_cond {
    param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
    gte = s32[] get-tuple-element(param), index=0
    constant.1 = s32[] constant(3)
    ROOT cmp = pred[] compare(gte, constant.1), direction=LT
  }

  while_body {
    param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
    current-loop-index = s32[] get-tuple-element(param), index=0
    output-buffer = bf16[3,8,128] get-tuple-element(param), index=1
    input-buffer = bf16[3,8,128] get-tuple-element(param), index=2
    constant.1 = s32[] constant(1)
    next-loop-index = s32[] add(current-loop-index, constant.1)
    constant.0 = s32[] constant(0)
    sliced-input-buffer = bf16[1,8,128] dynamic-slice(input-buffer,
      current-loop-index, constant.0, constant.0), dynamic_slice_sizes={1,8,128}
    all-reduce = bf16[1,8,128] all-reduce(sliced-input-buffer),
      replica_groups={}, to_apply=add, channel_id=1
    all-reduce.1 = bf16[1,8,128] all-reduce(all-reduce),
      replica_groups={}, to_apply=add, channel_id=2
    dynamic-update-slice = bf16[3,8,128] dynamic-update-slice(output-buffer,
      all-reduce.1, current-loop-index, constant.0, constant.0)
    ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(next-loop-index,
      dynamic-update-slice, input-buffer)
  }

  ENTRY entry {
    c0 = s32[] constant(0)
    p0 = bf16[3,8,128] parameter(0)
    tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
    while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple),
      condition=while_cond, body=while_body
    ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  // This config is taken from the gpu_compiler.cc configuration of the forward
  // pipeliner.
  CollectivePipeliner::Config config{
      /*level_to_operate_on=*/0,
      /*max_pipelining_per_loop=*/INT64_MAX,
      /*last_run=*/true,
      /*pipeline_use_tree=*/false,
      /*process_different_sized_ops=*/true,
      /*pipelining_direction=*/
      collective_pipeliner_utils::PipeliningDirection::kForward,
      /*should_process=*/HloPredicateIsOp<HloOpcode::kAllReduce>,
      /*acceptable_formatting=*/HloPredicateTrue,
      /*reuse_pipelined_op_buffer=*/HloPredicateFalse,
  };
  config.postprocess_pipelined_ops = AppendPipelinedInstruction;

  HloPassPipeline pipeline("collective-pipeliner");
  pipeline.AddPass<CollectivePipeliner>(config);
  pipeline.AddPass<HloPassFix<HloDCE>>(
      /*remove_cross_partition_collective_ops=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_TRUE(changed);

  hlo_query::ForEachInstructionWithOpcode(
      *module, HloOpcode::kAllReduce, [](HloInstruction* instr) {
        EXPECT_TRUE(instr->backend_config<GpuBackendConfig>()
                        ->collective_backend_config()
                        .is_pipelined());
      });

  hlo_query::ForEachInstructionWithPred(
      *module, HloPredicateIsNotOp<HloOpcode::kAllReduce>,
      [](HloInstruction* instr) {
        EXPECT_FALSE(instr->backend_config<GpuBackendConfig>()
                         ->collective_backend_config()
                         .is_pipelined());
      });
}

TEST_F(CollectiveCombinerUtilsTest,
       AppendPipelinedInstructionAppendsPipelinedInstructionInfoBackward) {
  // This is just the simple IR which makes it easy for the pipeliner to
  // pipeline a collective. The pipelined collective is AllGather so the main
  // complexity comes from a fact that we have to slice it at the end of the
  // loop (so that we can gather it again in the next iteration).
  absl::string_view kHloText = R"(
  HloModule module

  while_cond {
    param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
    gte = s32[] get-tuple-element(param), index=0
    constant.1 = s32[] constant(3)
    ROOT cmp = pred[] compare(gte, constant.1), direction=LT
  }

  while_body {
    param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
    current-loop-index = s32[] get-tuple-element(param), index=0
    output-buffer = bf16[3,8,128] get-tuple-element(param), index=1
    input-buffer = bf16[3,8,128] get-tuple-element(param), index=2
    constant.1 = s32[] constant(1)
    next-loop-index = s32[] add(current-loop-index, constant.1)
    constant.0 = s32[] constant(0)
    sliced-input-buffer = bf16[1,8,128] dynamic-slice(input-buffer,
      current-loop-index, constant.0, constant.0), dynamic_slice_sizes={1,8,128}
    all-gather = bf16[3,8,128] all-gather(sliced-input-buffer), dimensions={0}
    dynamic-update-slice = bf16[3,8,128] dynamic-update-slice(output-buffer,
      all-gather, current-loop-index, constant.0, constant.0)
    ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(next-loop-index,
      dynamic-update-slice, input-buffer)
  }

  ENTRY entry {
    c0 = s32[] constant(0)
    p0 = bf16[3,8,128] parameter(0)
    tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
    while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple),
      condition=while_cond, body=while_body
    ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  // This config is taken from the gpu_compiler.cc configuration of the backward
  // pipeliner.
  CollectivePipeliner::Config config{
      /*level_to_operate_on=*/0,
      /*max_pipelining_per_loop=*/INT64_MAX,
      /*last_run=*/true,
      /*pipeline_use_tree=*/false,
      /*process_different_sized_ops=*/true,
      /*pipelining_direction=*/
      collective_pipeliner_utils::PipeliningDirection::kBackward,
      /*should_process=*/HloPredicateIsOp<HloOpcode::kAllGather>,
      /*acceptable_formatting=*/HloPredicateTrue,
      /*reuse_pipelined_op_buffer=*/HloPredicateFalse,
      /*should_allow_loop_variant_parameter_in_chain=*/HloPredicateFalse,
      /*should_allow_control_dependencies=*/false,
      /*additional_chain_start_op_finder=*/nullptr,
      /*postprocess_backward_peeled_op=*/{},
      /*postprocess_backward_rotated_op=*/{},
      /*postprocess_backward_peeled_trailing_op=*/{},
      /*should_add_loop_invariant_op_in_chain=*/true,
  };
  config.postprocess_pipelined_ops = AppendPipelinedInstruction;

  HloPassPipeline pipeline("collective-pipeliner");
  pipeline.AddPass<CollectivePipeliner>(config);
  pipeline.AddPass<HloDCE>(/*remove_cross_partition_collective_ops=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_TRUE(changed);

  hlo_query::ForEachInstructionWithOpcode(
      *module, HloOpcode::kAllGather, [](HloInstruction* instr) {
        EXPECT_TRUE(instr->backend_config<GpuBackendConfig>()
                        ->collective_backend_config()
                        .is_pipelined());
      });

  hlo_query::ForEachInstructionWithPred(
      *module, HloPredicateIsNotOp<HloOpcode::kAllGather>,
      [](HloInstruction* instr) {
        EXPECT_FALSE(instr->backend_config<GpuBackendConfig>()
                         ->collective_backend_config()
                         .is_pipelined());
      });
}

TEST_F(CollectiveCombinerUtilsTest,
       AppendPipelinedInstructionBackwardFormattingOps) {
  // This is just the simple IR which makes it easy for the pipeliner to
  // pipeline a collective. The pipelined collective is AllGather so the main
  // complexity comes from a fact that we have to slice it at the end of the
  // loop (so that we can gather it again in the next iteration).
  absl::string_view kHloText = R"(
  HloModule module

  add {
    lhs = bf16[] parameter(0)
    rhs = bf16[] parameter(1)
    ROOT add = bf16[] add(lhs, rhs)
  }

  while_cond {
    param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
    gte = s32[] get-tuple-element(param), index=0
    constant.1 = s32[] constant(3)
    ROOT cmp = pred[] compare(gte, constant.1), direction=LT
  }

  while_body {
    param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
    current-loop-index = s32[] get-tuple-element(param), index=0
    output-buffer = bf16[3,8,128] get-tuple-element(param), index=1
    input-buffer = bf16[3,8,128] get-tuple-element(param), index=2
    constant.1 = s32[] constant(1)
    next-loop-index = s32[] add(current-loop-index, constant.1)
    constant.0 = s32[] constant(0)
    all-reduce = bf16[3,8,128] all-reduce(input-buffer), to_apply=add, replica_groups={}
    sliced-input-buffer = bf16[1,8,128] dynamic-slice(all-reduce,
      current-loop-index, constant.0, constant.0), dynamic_slice_sizes={1,8,128}
    all-gather = bf16[3,8,128] all-gather(sliced-input-buffer), dimensions={0}
    dynamic-update-slice = bf16[3,8,128] dynamic-update-slice(output-buffer,
      all-gather, current-loop-index, constant.0, constant.0)
    ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(next-loop-index,
      dynamic-update-slice, input-buffer)
  }

  ENTRY entry {
    c0 = s32[] constant(0)
    p0 = bf16[3,8,128] parameter(0)
    tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
    while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple),
      condition=while_cond, body=while_body
    ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  // This config is taken from the gpu_compiler.cc configuration of the backward
  // pipeliner.
  CollectivePipeliner::Config config{
      /*level_to_operate_on=*/0,
      /*max_pipelining_per_loop=*/INT64_MAX,
      /*last_run=*/true,
      /*pipeline_use_tree=*/false,
      /*process_different_sized_ops=*/true,
      /*pipelining_direction=*/
      collective_pipeliner_utils::PipeliningDirection::kBackward,
      /*should_process=*/HloPredicateIsOp<HloOpcode::kAllGather>,
      /*acceptable_formatting=*/HloPredicateTrue,
      /*reuse_pipelined_op_buffer=*/HloPredicateFalse,
      /*should_allow_loop_variant_parameter_in_chain=*/HloPredicateFalse,
      /*should_allow_control_dependencies=*/false,
      /*additional_chain_start_op_finder=*/nullptr,
      /*postprocess_backward_peeled_op=*/{},
      /*postprocess_backward_rotated_op=*/{},
      /*postprocess_backward_peeled_trailing_op=*/{},
      /*should_add_loop_invariant_op_in_chain=*/true,
  };
  config.postprocess_pipelined_ops = AppendPipelinedInstruction;

  HloPassPipeline pipeline("collective-pipeliner");
  pipeline.AddPass<CollectivePipeliner>(config);
  pipeline.AddPass<HloPassFix<HloDCE>>(
      /*remove_cross_partition_collective_ops=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_TRUE(changed);

  hlo_query::ForEachInstructionWithPred(
      *module, HloPredicateIsOp<HloOpcode::kAllGather, HloOpcode::kAllReduce>,
      [](HloInstruction* instr) {
        EXPECT_TRUE(instr->backend_config<GpuBackendConfig>()
                        ->collective_backend_config()
                        .is_pipelined());
      });

  hlo_query::ForEachInstructionWithPred(
      *module,
      HloPredicateIsNotOp<HloOpcode::kAllGather, HloOpcode::kAllReduce>,
      [](HloInstruction* instr) {
        EXPECT_FALSE(instr->backend_config<GpuBackendConfig>()
                         ->collective_backend_config()
                         .is_pipelined());
      });
}

TEST_F(CollectiveCombinerUtilsTest,
       ContainsPipelinedInstructionReturnsTrueForPipelinedInstructions) {
  // The IR is the minimal valid example of a while loop with AR inside. Three
  // are annotated as pipelined and three are not. Various configurations of the
  // combiner are tested to ensure the expected behaviour.
  constexpr absl::string_view kHloText = R"(
    HloModule module

    add {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT add = bf16[] add(lhs, rhs)
    }

    ENTRY entry {
      p0 = bf16[1] parameter(0)
      ROOT ar.pipelined.1 = bf16[1] all-reduce(p0),
        to_apply=add,
        backend_config={"collective_backend_config": {"is_pipelined": true}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  EXPECT_TRUE(ContainsPipelinedInstruction(*module));
}

TEST_F(CollectiveCombinerUtilsTest,
       ContainsPipelinedInstructionReturnsFalseForNonPipelinedInstructions) {
  // The IR is the minimal valid example of a while loop with AR inside. Three
  // are annotated as pipelined and three are not. Various configurations of the
  // combiner are tested to ensure the expected behaviour.
  constexpr absl::string_view kHloText = R"(
    HloModule module

    add {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT add = bf16[] add(lhs, rhs)
    }

    ENTRY entry {
      p0 = bf16[1] parameter(0)
      ar.0 = bf16[1] all-reduce(p0),
        to_apply=add
      ROOT ar.1 = bf16[1] all-reduce(ar.0),
        to_apply=add,
        backend_config={"collective_backend_config": {"is_pipelined": false}}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  EXPECT_FALSE(ContainsPipelinedInstruction(*module));
}

bool EnableHeuristicCollectiveCombining(
    se::CudaComputeCapability compute_capability, int num_partitions,
    int replica_count, int64_t nvlink_slice_size) {
  HloModuleConfig config;
  config.mutable_debug_options()
      .set_xla_gpu_experimental_enable_heuristic_collective_combining(true);
  config.set_num_partitions(num_partitions);
  config.set_replica_count(replica_count);
  se::DeviceDescription device_description =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(compute_capability);
  return xla::gpu::EnableHeuristicCollectiveCombining(
      config, device_description, nvlink_slice_size);
}

TEST(EnableHeuristicCollectiveCombiningTest, SingleHostSingleDevice) {
  // B200
  EXPECT_FALSE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Blackwell(),
                                         /*num_partitions=*/1,
                                         /*replica_count=*/1,
                                         /*nvlink_slice_size=*/8));
  // H100
  EXPECT_FALSE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Hopper(),
                                         /*num_partitions=*/1,
                                         /*replica_count=*/1,
                                         /*nvlink_slice_size=*/8));
  // A100
  EXPECT_FALSE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Ampere(),
                                         /*num_partitions=*/1,
                                         /*replica_count=*/1,
                                         /*nvlink_slice_size=*/16));
}

TEST(EnableHeuristicCollectiveCombiningTest, SingleHostMultiDevices) {
  // B200
  EXPECT_FALSE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Blackwell(),
                                         /*num_partitions=*/8,
                                         /*replica_count=*/1,
                                         /*nvlink_slice_size=*/8));
  EXPECT_FALSE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Blackwell(),
                                         /*num_partitions=*/1,
                                         /*replica_count=*/8,
                                         /*nvlink_slice_size=*/8));
  // H100
  EXPECT_FALSE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Hopper(),
                                         /*num_partitions=*/8,
                                         /*replica_count=*/1,
                                         /*nvlink_slice_size=*/8));
  EXPECT_FALSE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Hopper(),
                                         /*num_partitions=*/1,
                                         /*replica_count=*/8,
                                         /*nvlink_slice_size=*/8));
  // A100
  EXPECT_FALSE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Ampere(),
                                         /*num_partitions=*/1,
                                         /*replica_count=*/16,
                                         /*nvlink_slice_size=*/16));
  EXPECT_FALSE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Ampere(),
                                         /*num_partitions=*/16,
                                         /*replica_count=*/1,
                                         /*nvlink_slice_size=*/16));
}

TEST(EnableHeuristicCollectiveCombiningTest, MultiHosts) {
  // B200
  EXPECT_TRUE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Blackwell(),
                                         /*num_partitions=*/16,
                                         /*replica_count=*/1,
                                         /*nvlink_slice_size=*/8));
  EXPECT_TRUE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Blackwell(),
                                         /*num_partitions=*/1,
                                         /*replica_count=*/16,
                                         /*nvlink_slice_size=*/8));
  // H100
  EXPECT_TRUE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Hopper(),
                                         /*num_partitions=*/16,
                                         /*replica_count=*/1,
                                         /*nvlink_slice_size=*/8));
  EXPECT_TRUE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Hopper(),
                                         /*num_partitions=*/1,
                                         /*replica_count=*/16,
                                         /*nvlink_slice_size=*/8));
  // A100
  EXPECT_TRUE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Ampere(),
                                         /*num_partitions=*/1,
                                         /*replica_count=*/32,
                                         /*nvlink_slice_size=*/16));
  EXPECT_TRUE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Ampere(),
                                         /*num_partitions=*/32,
                                         /*replica_count=*/1,
                                         /*nvlink_slice_size=*/16));
}

TEST(EnableHeuristicCollectiveCombiningTest, UnsupportedGPU) {
  EXPECT_FALSE(
      EnableHeuristicCollectiveCombining(se::CudaComputeCapability::Volta(),
                                         /*num_partitions=*/1,
                                         /*replica_count=*/1,
                                         /*nvlink_slice_size=*/8));
}

TEST(EnableHeuristicCollectiveCombiningTest, DisabledByFlag) {
  HloModuleConfig config;
  config.mutable_debug_options()
      .set_xla_gpu_experimental_enable_heuristic_collective_combining(false);
  config.set_num_partitions(16);
  config.set_replica_count(1);
  se::DeviceDescription device_description =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(
          se::CudaComputeCapability::Blackwell());
  EXPECT_FALSE(xla::gpu::EnableHeuristicCollectiveCombining(
      config, device_description, /*nvlink_slice_size=*/8));
}

}  // namespace
}  // namespace xla::gpu

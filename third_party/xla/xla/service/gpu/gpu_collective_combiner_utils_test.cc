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

#include "xla/service/gpu/gpu_collective_combiner_utils.h"

#include <cstdint>
#include <optional>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_pipeliner.h"
#include "xla/service/collective_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {

using CollectiveCombinerUtilsTest = HloTestBase;

TEST_F(CollectiveCombinerUtilsTest,
       ComputeSuggestedCombinerThresholdReturnsMemoryThresholdForDeviceInfo) {
  absl::string_view kHloText = R"(
  HloModule m

  ENTRY ar {
    p0 = f32[32,32] parameter(0)
    p1 = f32[32,32] parameter(1)

    ROOT _ = f32[32,32]{1,0} custom-call(p0, p1),
      custom_call_target="__cublas$gemm"
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  int pointer_size = 4;
  stream_executor::DeviceDescription device_info;
  device_info.set_device_memory_size(20000);

  int64_t suggested_threshold = ComputeSuggestedCombinerThreshold(
      *module, device_info, gpu::ScheduleGpuModuleWithMemoryScheduler,
      HloOpcode::kAllReduce, pointer_size);

  // device size = 20000 bytes
  // slop factor = 0.95
  // peak memory = parameters + output = (2*32*32 + 32*32) * 4 bytes = 12288
  // suggested thresholds = device size * slop factor - peak memory
  EXPECT_EQ(suggested_threshold, 6712);
}

TEST_F(CollectiveCombinerUtilsTest,
       ComputeSuggestedCombinerThresholdReturnsMemoryThresholdForModuleConfig) {
  absl::string_view kHloText = R"(
  HloModule m

  ENTRY ar {
    p0 = f32[32,32] parameter(0)
    p1 = f32[32,32] parameter(1)

    ROOT _ = f32[32,32]{1,0} custom-call(p0, p1),
      custom_call_target="__cublas$gemm"
  })";

  HloModuleConfig config = GetModuleConfigForTest();
  config.set_device_memory_size(20000);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloText, config));
  int pointer_size = 4;
  stream_executor::DeviceDescription device_info;

  int64_t suggested_threshold = ComputeSuggestedCombinerThreshold(
      *module, device_info, gpu::ScheduleGpuModuleWithMemoryScheduler,
      HloOpcode::kAllReduce, pointer_size);

  // device size = 20000 bytes
  // slop factor = 0.95
  // peak memory = parameters + output = (2*32*32 + 32*32) * 4 bytes = 12288
  // suggested thresholds = device size * slop factor - peak memory
  EXPECT_EQ(suggested_threshold, 6712);
}

TEST_F(
    CollectiveCombinerUtilsTest,
    ComputeSuggestedCombinerThresholdReturnsDefaultValueUponSchedulingFailure) {
  absl::string_view kHloText = R"(
  HloModule m

  ENTRY ar {
    p0 = f32[32,32] parameter(0)
    p1 = f32[32,32] parameter(1)

    ROOT _ = f32[32,32]{1,0} custom-call(p0, p1),
      custom_call_target="__cublas$gemm"
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  int pointer_size = 4;
  stream_executor::DeviceDescription device_info;
  device_info.set_device_memory_size(20000);

  auto sched_fun = [](const HloModule* m, int64_t p_sz,
                      int64_t* p) -> absl::StatusOr<HloSchedule> {
    return absl::UnimplementedError("Fail.");
  };

  int64_t suggested_threshold_all_reduce = ComputeSuggestedCombinerThreshold(
      *module, device_info, sched_fun, HloOpcode::kAllReduce, pointer_size);
  int64_t suggested_threshold_all_gather = ComputeSuggestedCombinerThreshold(
      *module, device_info, sched_fun, HloOpcode::kAllGather, pointer_size);
  int64_t suggested_threshold_reduce_scatter =
      ComputeSuggestedCombinerThreshold(*module, device_info, sched_fun,
                                        HloOpcode::kReduceScatter,
                                        pointer_size);

  EXPECT_EQ(suggested_threshold_all_reduce, kDefaultAllReduceCombineThreshold);
  EXPECT_EQ(suggested_threshold_all_gather, kDefaultAllGatherCombineThreshold);
  EXPECT_EQ(suggested_threshold_reduce_scatter,
            kDefaultReduceScatterCombineThreshold);
}

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
      CollectivePipeliner::PipeliningDirection::kForward,
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
      CollectivePipeliner::PipeliningDirection::kForward,
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
      CollectivePipeliner::PipeliningDirection::kBackward,
      /*should_process=*/HloPredicateIsOp<HloOpcode::kAllGather>,
      /*acceptable_formatting=*/HloPredicateTrue,
      /*reuse_pipelined_op_buffer=*/HloPredicateFalse,
      /*should_allow_loop_variant_parameter_in_chain=*/HloPredicateFalse,
      /*should_allow_control_dependencies=*/false,
      /*postprocess_backward_peeled_op=*/std::nullopt,
      /*postprocess_backward_rotated_op=*/std::nullopt,
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
      CollectivePipeliner::PipeliningDirection::kBackward,
      /*should_process=*/HloPredicateIsOp<HloOpcode::kAllGather>,
      /*acceptable_formatting=*/HloPredicateTrue,
      /*reuse_pipelined_op_buffer=*/HloPredicateFalse,
      /*should_allow_loop_variant_parameter_in_chain=*/HloPredicateFalse,
      /*should_allow_control_dependencies=*/false,
      /*postprocess_backward_peeled_op=*/std::nullopt,
      /*postprocess_backward_rotated_op=*/std::nullopt,
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

}  // namespace
}  // namespace xla::gpu

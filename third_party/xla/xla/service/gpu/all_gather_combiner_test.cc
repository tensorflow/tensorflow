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

#include "xla/service/gpu/all_gather_combiner.h"

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/collective_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {

using GpuAllGatherCombinerTest = HloTestBase;

using ::stream_executor::DeviceDescription;

TEST_F(GpuAllGatherCombinerTest,
       CombinesPipelinedCollectivesUpToSuggestedThreshold) {
  constexpr absl::string_view kHloString = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(8)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) parameter(0)
  param.0 = s32[] get-tuple-element(param), index=0
  param.1 = bf16[6,8,128] get-tuple-element(param), index=1
  param.2 = bf16[6,8,128] get-tuple-element(param), index=2
  param.3 = bf16[6,8,128] get-tuple-element(param), index=3
  param.4 = bf16[6,8,128] get-tuple-element(param), index=4
  param.5 = bf16[6,8,128] get-tuple-element(param), index=5
  param.6 = bf16[6,8,128] get-tuple-element(param), index=6
  param.7 = bf16[3,1,2,128] get-tuple-element(param), index=7
  zero = bf16[] constant(0)
  one = s32[] constant(1)
  it = s32[] add(param.0, one)
  concat = bf16[36,8,128] concatenate(param.1,param.2,param.3,param.4,param.5,param.6), dimensions={0}
  reduced = bf16[8,128] reduce(concat, zero), dimensions={0}, to_apply=add
  broadcasted.pipelined.0 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.pipelined.1 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.pipelined.2 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.nonpipelined.0 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.nonpipelined.1 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.nonpipelined.2 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  ag-pipelined.0 = bf16[6,8,128] all-gather(broadcasted.pipelined.0), dimensions={0}, backend_config={"collective_backend_config": {"is_pipelined": true}}
  ag-pipelined.1 = bf16[6,8,128] all-gather(broadcasted.pipelined.1), dimensions={0}, backend_config={"collective_backend_config": {"is_pipelined": true}}
  ag-pipelined.2 = bf16[6,8,128] all-gather(broadcasted.pipelined.2), dimensions={0}, backend_config={"collective_backend_config": {"is_pipelined": true}}
  ag-nonpipelined.0 = bf16[6,8,128] all-gather(broadcasted.nonpipelined.0), dimensions={0}
  ag-nonpipelined.1 = bf16[6,8,128] all-gather(broadcasted.nonpipelined.1), dimensions={0}
  ag-nonpipelined.2 = bf16[6,8,128] all-gather(broadcasted.nonpipelined.2), dimensions={0}
  ROOT tuple = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) tuple(it, ag-pipelined.0, ag-pipelined.1, ag-pipelined.2, ag-nonpipelined.0, ag-nonpipelined.1, ag-nonpipelined.2, param.7)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[6,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) tuple(c0, p0, p0, p0, p0, p0, p0, p1)
  while = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT _ = bf16[6,8,128] get-tuple-element(while), index=1
}
)";
  auto config =
      GetModuleConfigForTest(/*replica_count=*/1, /*num_partitions=*/2);
  config.mutable_debug_options()
      .set_xla_gpu_enable_heuristic_pass_configuration(true);
  DeviceDescription device_info;
  // Combine at most 2 collectives.
  int collective_size = 2 * 6 * 8 * 128;
  int threshold_bytes = 2 * collective_size;
  int current_peak_mem = 90604;
  int pointer_size = 4;
  device_info.set_device_memory_size(current_peak_mem + threshold_bytes * 4);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString, config));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      GpuAllGatherCombiner(device_info, /*default_combine_threshold_in_bytes=*/
                           threshold_bytes,
                           /*combine_threshold_in_bytes=*/threshold_bytes,
                           /*combine_threshold_count=*/256,
                           /*combine_by_dim=*/false,
                           /*combine_different_dtypes=*/true, pointer_size)
          .Run(module.get()));

  VLOG(1) << module->ToString();
  EXPECT_TRUE(changed);
  // Pipelined all gathers were combined up to the predefined max available
  // device mem limit.
  EXPECT_TRUE(
      *RunFileCheck(module->ToString(HloPrintOptions()
                                         .set_print_operand_shape(false)
                                         .set_print_result_shape(false)),
                    R"(
    // CHECK: all-gather(%broadcasted.pipelined.0, %broadcasted.pipelined.1, %broadcasted.pipelined.2)
    // CHECK: all-gather(%broadcasted.nonpipelined.0, %broadcasted.nonpipelined.1)
    // CHECK: all-gather(%broadcasted.nonpipelined.2)
  )"));
}

TEST_F(GpuAllGatherCombinerTest, CombinesCollectivesUpToSpecifiedThreshold) {
  constexpr absl::string_view kHloString = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(8)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) parameter(0)
  param.0 = s32[] get-tuple-element(param), index=0
  param.1 = bf16[6,8,128] get-tuple-element(param), index=1
  param.2 = bf16[6,8,128] get-tuple-element(param), index=2
  param.3 = bf16[6,8,128] get-tuple-element(param), index=3
  param.4 = bf16[6,8,128] get-tuple-element(param), index=4
  param.5 = bf16[6,8,128] get-tuple-element(param), index=5
  param.6 = bf16[6,8,128] get-tuple-element(param), index=6
  param.7 = bf16[3,1,2,128] get-tuple-element(param), index=7
  zero = bf16[] constant(0)
  one = s32[] constant(1)
  it = s32[] add(param.0, one)
  concat = bf16[36,8,128] concatenate(param.1,param.2,param.3,param.4,param.5,param.6), dimensions={0}
  reduced = bf16[8,128] reduce(concat, zero), dimensions={0}, to_apply=add
  broadcasted.pipelined.0 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.pipelined.1 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.pipelined.2 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.nonpipelined.0 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.nonpipelined.1 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.nonpipelined.2 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  ag-pipelined.0 = bf16[6,8,128] all-gather(broadcasted.pipelined.0), dimensions={0}, backend_config={"collective_backend_config": {"is_pipelined": true}}
  ag-pipelined.1 = bf16[6,8,128] all-gather(broadcasted.pipelined.1), dimensions={0}, backend_config={"collective_backend_config": {"is_pipelined": true}}
  ag-pipelined.2 = bf16[6,8,128] all-gather(broadcasted.pipelined.2), dimensions={0}, backend_config={"collective_backend_config": {"is_pipelined": true}}
  ag-nonpipelined.0 = bf16[6,8,128] all-gather(broadcasted.nonpipelined.0), dimensions={0}
  ag-nonpipelined.1 = bf16[6,8,128] all-gather(broadcasted.nonpipelined.1), dimensions={0}
  ag-nonpipelined.2 = bf16[6,8,128] all-gather(broadcasted.nonpipelined.2), dimensions={0}
  ROOT tuple = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) tuple(it, ag-pipelined.0, ag-pipelined.1, ag-pipelined.2, ag-nonpipelined.0, ag-nonpipelined.1, ag-nonpipelined.2, param.7)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[6,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) tuple(c0, p0, p0, p0, p0, p0, p0, p1)
  while = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT _ = bf16[6,8,128] get-tuple-element(while), index=1
}
)";
  auto config =
      GetModuleConfigForTest(/*replica_count=*/1, /*num_partitions=*/2);
  config.mutable_debug_options()
      .set_xla_gpu_enable_heuristic_pass_configuration(true);
  DeviceDescription device_info;
  // Combine at most 2 collectives.
  int collective_size = 2 * 6 * 8 * 128;
  int threshold_bytes = 2 * collective_size;
  int current_peak_mem = 90604;
  int pointer_size = 4;
  device_info.set_device_memory_size(current_peak_mem + threshold_bytes * 4);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString, config));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      GpuAllGatherCombiner(device_info, /*default_combine_threshold_in_bytes=*/
                           kDefaultAllGatherCombineThreshold,
                           /*combine_threshold_in_bytes=*/threshold_bytes,
                           /*combine_threshold_count=*/256,
                           /*combine_by_dim=*/false,
                           /*combine_different_dtypes=*/true, pointer_size)
          .Run(module.get()));

  VLOG(1) << module->ToString();
  EXPECT_TRUE(changed);
  // Pipelined all gathers were combined up to the predefined max available
  // device mem limit.
  EXPECT_TRUE(
      *RunFileCheck(module->ToString(HloPrintOptions()
                                         .set_print_operand_shape(false)
                                         .set_print_result_shape(false)),
                    R"(
    // CHECK: all-gather(%broadcasted.pipelined.0, %broadcasted.pipelined.1)
    // CHECK: all-gather(%broadcasted.pipelined.2, %broadcasted.nonpipelined.0)
    // CHECK: all-gather(%broadcasted.nonpipelined.1, %broadcasted.nonpipelined.2)
  )"));
}

TEST_F(GpuAllGatherCombinerTest,
       CombinesCollectivesUpToDefaultThresholdIfFlagDisabled) {
  constexpr absl::string_view kHloString = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(8)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) parameter(0)
  param.0 = s32[] get-tuple-element(param), index=0
  param.1 = bf16[6,8,128] get-tuple-element(param), index=1
  param.2 = bf16[6,8,128] get-tuple-element(param), index=2
  param.3 = bf16[6,8,128] get-tuple-element(param), index=3
  param.4 = bf16[6,8,128] get-tuple-element(param), index=4
  param.5 = bf16[6,8,128] get-tuple-element(param), index=5
  param.6 = bf16[6,8,128] get-tuple-element(param), index=6
  param.7 = bf16[3,1,2,128] get-tuple-element(param), index=7
  zero = bf16[] constant(0)
  one = s32[] constant(1)
  it = s32[] add(param.0, one)
  concat = bf16[36,8,128] concatenate(param.1,param.2,param.3,param.4,param.5,param.6), dimensions={0}
  reduced = bf16[8,128] reduce(concat, zero), dimensions={0}, to_apply=add
  broadcasted.pipelined.0 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.pipelined.1 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.pipelined.2 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.nonpipelined.0 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.nonpipelined.1 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  broadcasted.nonpipelined.2 = bf16[3,8,128] broadcast(reduced), dimensions={1,2}
  ag-pipelined.0 = bf16[6,8,128] all-gather(broadcasted.pipelined.0), dimensions={0}, backend_config={"collective_backend_config": {"is_pipelined": true}}
  ag-pipelined.1 = bf16[6,8,128] all-gather(broadcasted.pipelined.1), dimensions={0}, backend_config={"collective_backend_config": {"is_pipelined": true}}
  ag-pipelined.2 = bf16[6,8,128] all-gather(broadcasted.pipelined.2), dimensions={0}, backend_config={"collective_backend_config": {"is_pipelined": true}}
  ag-nonpipelined.0 = bf16[6,8,128] all-gather(broadcasted.nonpipelined.0), dimensions={0}
  ag-nonpipelined.1 = bf16[6,8,128] all-gather(broadcasted.nonpipelined.1), dimensions={0}
  ag-nonpipelined.2 = bf16[6,8,128] all-gather(broadcasted.nonpipelined.2), dimensions={0}
  ROOT tuple = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) tuple(it, ag-pipelined.0, ag-pipelined.1, ag-pipelined.2, ag-nonpipelined.0, ag-nonpipelined.1, ag-nonpipelined.2, param.7)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[6,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) tuple(c0, p0, p0, p0, p0, p0, p0, p1)
  while = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT _ = bf16[6,8,128] get-tuple-element(while), index=1
}
)";
  auto config =
      GetModuleConfigForTest(/*replica_count=*/1, /*num_partitions=*/2);
  config.mutable_debug_options()
      .set_xla_gpu_enable_heuristic_pass_configuration(false);
  DeviceDescription device_info;
  // Combine at most 2 collectives.
  int collective_size = 2 * 6 * 8 * 128;
  int threshold_bytes = 2 * collective_size;
  int current_peak_mem = 90604;
  int pointer_size = 4;
  device_info.set_device_memory_size(current_peak_mem + threshold_bytes * 4);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString, config));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      GpuAllGatherCombiner(device_info, /*default_combine_threshold_in_bytes=*/
                           threshold_bytes,
                           /*combine_threshold_in_bytes=*/threshold_bytes,
                           /*combine_threshold_count=*/256,
                           /*combine_by_dim=*/false,
                           /*combine_different_dtypes=*/true, pointer_size)
          .Run(module.get()));

  VLOG(1) << module->ToString();
  EXPECT_TRUE(changed);
  // Pipelined all gathers were combined up to the predefined max available
  // device mem limit.
  EXPECT_TRUE(
      *RunFileCheck(module->ToString(HloPrintOptions()
                                         .set_print_operand_shape(false)
                                         .set_print_result_shape(false)),
                    R"(
    // CHECK: all-gather(%broadcasted.pipelined.0, %broadcasted.pipelined.1)
    // CHECK: all-gather(%broadcasted.pipelined.2, %broadcasted.nonpipelined.0)
    // CHECK: all-gather(%broadcasted.nonpipelined.1, %broadcasted.nonpipelined.2)
  )"));
}

}  // namespace

}  // namespace xla::gpu

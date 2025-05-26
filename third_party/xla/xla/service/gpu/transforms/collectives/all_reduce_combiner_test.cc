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

#include "xla/service/gpu/transforms/collectives/all_reduce_combiner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/collective_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::stream_executor::DeviceDescription;
using ::testing::Matcher;
using ::tsl::testing::IsOkAndHolds;

namespace op = xla::testing::opcode_matchers;

using GpuAllReduceCombinerTest = HloHardwareIndependentTestBase;

TEST_F(GpuAllReduceCombinerTest,
       CombinesPipelinedCollectivesUpToSuggestedThreshold) {
  // The IR is the minimal valid example of a while loop with AR inside. Three
  // are annotated as pipelined and three are not. Various configurations of the
  // combiner are tested to ensure the expected behaviour.
  constexpr absl::string_view kHloString = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128],
    bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(8)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128],
    bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) parameter(0)
  param.0 = s32[] get-tuple-element(param), index=0
  param.pipelined.0 = bf16[6,8,128] get-tuple-element(param), index=1
  param.pipelined.1 = bf16[6,8,128] get-tuple-element(param), index=2
  param.pipelined.2 = bf16[6,8,128] get-tuple-element(param), index=3
  param.nonpipelined.0 = bf16[6,8,128] get-tuple-element(param), index=4
  param.nonpipelined.1 = bf16[6,8,128] get-tuple-element(param), index=5
  param.nonpipelined.2 = bf16[6,8,128] get-tuple-element(param), index=6
  param.7 = bf16[3,1,2,128] get-tuple-element(param), index=7
  zero = bf16[] constant(0)
  one = s32[] constant(1)
  it = s32[] add(param.0, one)
  ar.pipelined.0 = bf16[6,8,128] all-reduce(param.pipelined.0),
    to_apply=add,
    backend_config={"collective_backend_config": {"is_pipelined": true}}
  ar.pipelined.1 = bf16[6,8,128] all-reduce(param.pipelined.1),
    to_apply=add,
    backend_config={"collective_backend_config": {"is_pipelined": true}}
  ar.pipelined.2 = bf16[6,8,128] all-reduce(param.pipelined.2),
    to_apply=add,
    backend_config={"collective_backend_config": {"is_pipelined": true}}
  ar.nonpipelined.0 = bf16[6,8,128] all-reduce(param.nonpipelined.0),
    to_apply=add
  ar.nonpipelined.1 = bf16[6,8,128] all-reduce(param.nonpipelined.1),
    to_apply=add
  ar.nonpipelined.2 = bf16[6,8,128] all-reduce(param.nonpipelined.2),
    to_apply=add
  ROOT tuple = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) tuple(it, ar.pipelined.0, ar.pipelined.1, ar.pipelined.2, ar.nonpipelined.0, ar.nonpipelined.1, ar.nonpipelined.2, param.7)
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
  DeviceDescription device_info;
  // Combine at most 2 collectives.
  int collective_size = 2 * 6 * 8 * 128;
  int threshold_bytes = 2 * collective_size;
  int current_peak_mem = 87625;
  int pointer_size = 4;
  device_info.set_device_memory_size(current_peak_mem + 4 * threshold_bytes);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString, config));
  EXPECT_THAT(
      GpuAllReduceCombiner(device_info, /*default_combine_threshold_in_bytes=*/
                           threshold_bytes,
                           /*combine_threshold_in_bytes=*/threshold_bytes,
                           /*combine_threshold_count=*/256, pointer_size)
          .Run(module.get()),
      IsOkAndHolds(true));

  VLOG(1) << module->ToString();
  // Pipelined all gathers were combined up to the predefined max available
  // device mem limit.
  const absl::string_view kExpected = R"(
    // CHECK-DAG: %[[PIPELINED_PARAM_0:.*]] = {{.*}} index=1
    // CHECK-DAG: %[[PIPELINED_PARAM_1:.*]] = {{.*}} index=2
    // CHECK-DAG: %[[PIPELINED_PARAM_2:.*]] = {{.*}} index=3
    // CHECK-DAG: %[[NONPIPELINED_PARAM_0:.*]] = {{.*}} index=4
    // CHECK-DAG: %[[NONPIPELINED_PARAM_1:.*]] = {{.*}} index=5
    // CHECK-DAG: %[[NONPIPELINED_PARAM_2:.*]] = {{.*}} index=6
    // CHECK-DAG: all-reduce(%[[PIPELINED_PARAM_0]], %[[PIPELINED_PARAM_1]], %[[PIPELINED_PARAM_2]])
    // CHECK-DAG: all-reduce(%[[NONPIPELINED_PARAM_0]], %[[NONPIPELINED_PARAM_1]])
    // CHECK-DAG: all-reduce(%[[NONPIPELINED_PARAM_2]])
  )";
  EXPECT_TRUE(
      *RunFileCheck(module->ToString(HloPrintOptions()
                                         .set_print_operand_shape(false)
                                         .set_print_result_shape(false)),
                    kExpected));
}

TEST_F(GpuAllReduceCombinerTest,
       CombinesNonPipelinedCollectivesWithAFallbackCombiner) {
  // The IR is the minimal valid example of a while loop with RS inside.
  // All collectives are not pipelined.
  constexpr absl::string_view kHloString = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128],
    bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(8)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128],
    bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) parameter(0)
  param.0 = s32[] get-tuple-element(param), index=0
  param.nonpipelined.0 = bf16[6,8,128] get-tuple-element(param), index=1
  param.nonpipelined.1 = bf16[6,8,128] get-tuple-element(param), index=2
  param.nonpipelined.2 = bf16[6,8,128] get-tuple-element(param), index=3
  param.nonpipelined.3 = bf16[6,8,128] get-tuple-element(param), index=4
  param.nonpipelined.4 = bf16[6,8,128] get-tuple-element(param), index=5
  param.nonpipelined.5 = bf16[6,8,128] get-tuple-element(param), index=6
  param.7 = bf16[3,1,2,128] get-tuple-element(param), index=7
  zero = bf16[] constant(0)
  one = s32[] constant(1)
  it = s32[] add(param.0, one)
  ar.nonpipelined.0 = bf16[6,8,128] all-reduce(param.nonpipelined.0),
    to_apply=add
  ar.nonpipelined.1 = bf16[6,8,128] all-reduce(param.nonpipelined.1),
    to_apply=add
  ar.nonpipelined.2 = bf16[6,8,128] all-reduce(param.nonpipelined.2),
    to_apply=add
  ar.nonpipelined.3 = bf16[6,8,128] all-reduce(param.nonpipelined.3),
    to_apply=add
  ar.nonpipelined.4 = bf16[6,8,128] all-reduce(param.nonpipelined.4),
    to_apply=add
  ar.nonpipelined.5 = bf16[6,8,128] all-reduce(param.nonpipelined.5),
    to_apply=add
  ROOT tuple = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) tuple(it, ar.nonpipelined.0, ar.nonpipelined.1, ar.nonpipelined.2, ar.nonpipelined.3, ar.nonpipelined.4, ar.nonpipelined.5, param.7)
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
  DeviceDescription device_info;
  int pointer_size = 4;

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString, config));
  EXPECT_THAT(
      GpuAllReduceCombiner(
          device_info, /*default_combine_threshold_in_bytes=*/
          kDefaultAllReduceCombineThreshold,
          /*combine_threshold_in_bytes=*/kDefaultAllReduceCombineThreshold,
          /*combine_threshold_count=*/256, pointer_size)
          .Run(module.get()),
      IsOkAndHolds(true));

  VLOG(1) << module->ToString();
  const absl::string_view kExpected = R"(
    // CHECK-DAG: %[[NONPIPELINED_PARAM_0:.*]] = {{.*}} index=1
    // CHECK-DAG: %[[NONPIPELINED_PARAM_1:.*]] = {{.*}} index=2
    // CHECK-DAG: %[[NONPIPELINED_PARAM_2:.*]] = {{.*}} index=3
    // CHECK-DAG: %[[NONPIPELINED_PARAM_3:.*]] = {{.*}} index=4
    // CHECK-DAG: %[[NONPIPELINED_PARAM_4:.*]] = {{.*}} index=5
    // CHECK-DAG: %[[NONPIPELINED_PARAM_5:.*]] = {{.*}} index=6
    // CHECK: all-reduce(%[[NONPIPELINED_PARAM_0]], %[[NONPIPELINED_PARAM_1]], %[[NONPIPELINED_PARAM_2]]
    // CHECK-SAME: %[[NONPIPELINED_PARAM_3]], %[[NONPIPELINED_PARAM_4]], %[[NONPIPELINED_PARAM_5]])
  )";
  EXPECT_TRUE(*RunFileCheck(
      module->ToString(HloPrintOptions()
                           .set_print_operand_shape(false)
                           .set_print_result_shape(false)
                           .set_print_operand_index_annotation_interval(10)),
      kExpected));
}

TEST_F(GpuAllReduceCombinerTest, CombinesCollectivesUpToSpecifiedThreshold) {
  // The IR is the minimal valid example of a while loop with AR inside. Three
  // are annotated as pipelined and three are not. Various configurations of the
  // combiner are tested to ensure the expected behaviour.
  constexpr absl::string_view kHloString = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128],
    bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(8)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128],
    bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) parameter(0)
  param.0 = s32[] get-tuple-element(param), index=0
  param.pipelined.0 = bf16[6,8,128] get-tuple-element(param), index=1
  param.pipelined.1 = bf16[6,8,128] get-tuple-element(param), index=2
  param.pipelined.2 = bf16[6,8,128] get-tuple-element(param), index=3
  param.nonpipelined.0 = bf16[6,8,128] get-tuple-element(param), index=4
  param.nonpipelined.1 = bf16[6,8,128] get-tuple-element(param), index=5
  param.nonpipelined.2 = bf16[6,8,128] get-tuple-element(param), index=6
  param.7 = bf16[3,1,2,128] get-tuple-element(param), index=7
  zero = bf16[] constant(0)
  one = s32[] constant(1)
  it = s32[] add(param.0, one)
  ar.pipelined.0 = bf16[6,8,128] all-reduce(param.pipelined.0),
    to_apply=add,
    backend_config={"collective_backend_config": {"is_pipelined": true}}
  ar.pipelined.1 = bf16[6,8,128] all-reduce(param.pipelined.1),
    to_apply=add,
    backend_config={"collective_backend_config": {"is_pipelined": true}}
  ar.pipelined.2 = bf16[6,8,128] all-reduce(param.pipelined.2),
    to_apply=add,
    backend_config={"collective_backend_config": {"is_pipelined": true}}
  ar.nonpipelined.0 = bf16[6,8,128] all-reduce(param.nonpipelined.0),
    to_apply=add
  ar.nonpipelined.1 = bf16[6,8,128] all-reduce(param.nonpipelined.1),
    to_apply=add
  ar.nonpipelined.2 = bf16[6,8,128] all-reduce(param.nonpipelined.2),
    to_apply=add
  ROOT tuple = (s32[], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[6,8,128], bf16[3,1,2,128]) tuple(it, ar.pipelined.0, ar.pipelined.1, ar.pipelined.2, ar.nonpipelined.0, ar.nonpipelined.1, ar.nonpipelined.2, param.7)
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
  DeviceDescription device_info;
  // Combine at most 2 collectives.
  int collective_size = 2 * 6 * 8 * 128;
  int threshold_bytes = 2 * collective_size;
  int current_peak_mem = 87625;
  int pointer_size = 4;
  device_info.set_device_memory_size(current_peak_mem + threshold_bytes * 4);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString, config));
  EXPECT_THAT(
      GpuAllReduceCombiner(device_info, /*default_combine_threshold_in_bytes=*/
                           kDefaultAllReduceCombineThreshold,
                           /*combine_threshold_in_bytes=*/threshold_bytes,
                           /*combine_threshold_count=*/256, pointer_size)
          .Run(module.get()),
      IsOkAndHolds(true));

  VLOG(1) << module->ToString();
  // Pipelined all gathers were combined up to the predefined max available
  // device mem limit.
  const absl::string_view kExpected = R"(
    // CHECK-DAG: %[[PIPELINED_PARAM_0:.*]] = {{.*}} index=1
    // CHECK-DAG: %[[PIPELINED_PARAM_1:.*]] = {{.*}} index=2
    // CHECK-DAG: %[[PIPELINED_PARAM_2:.*]] = {{.*}} index=3
    // CHECK-DAG: %[[NONPIPELINED_PARAM_0:.*]] = {{.*}} index=4
    // CHECK-DAG: %[[NONPIPELINED_PARAM_1:.*]] = {{.*}} index=5
    // CHECK-DAG: %[[NONPIPELINED_PARAM_2:.*]] = {{.*}} index=6
    // CHECK-DAG: all-reduce(%[[PIPELINED_PARAM_0]], %[[PIPELINED_PARAM_1]])
    // CHECK-DAG: all-reduce(%[[PIPELINED_PARAM_2]])
    // CHECK-DAG: all-reduce(%[[NONPIPELINED_PARAM_0]], %[[NONPIPELINED_PARAM_1]])
    // CHECK-DAG: all-reduce(%[[NONPIPELINED_PARAM_2]])
  )";

  EXPECT_TRUE(
      *RunFileCheck(module->ToString(HloPrintOptions()
                                         .set_print_operand_shape(false)
                                         .set_print_result_shape(false)),
                    kExpected));
}

TEST_F(GpuAllReduceCombinerTest, CombinesSynchronousCollectivesMaximally) {
  absl::string_view kHloText = R"(
    HloModule m

    add {
      p0 = f16[] parameter(0)
      p1 = f16[] parameter(1)
      ROOT add = f16[] add(p0, p1)
    }

    ENTRY main {
      p0 = f16[10000000]{0} parameter(0)
      p1 = f16[10000000]{0} parameter(1)

      // 20MB combinable all-reduce collectives. Default combiner threshold is 30MB.
      ar0 = f16[10000000]{0} all-reduce(p0), replica_groups={}, to_apply=add,
        frontend_attributes={sync_collective="true"}
      ar1 = f16[10000000]{0} all-reduce(p1), replica_groups={}, to_apply=add,
        frontend_attributes={sync_collective="true"}
      ROOT result = tuple(ar0, ar1)
    }
  )";
  DeviceDescription device_info;
  device_info.set_device_memory_size(10000000000);  // 10GB

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  GpuAllReduceCombiner combiner(
      device_info, /*default_combine_threshold_in_bytes=*/
      kDefaultAllReduceCombineThreshold,
      /*combine_threshold_in_bytes=*/kDefaultAllReduceCombineThreshold,
      /*combine_threshold_count=*/256, /*pointer_size=*/4);

  EXPECT_THAT(combiner.Run(module.get()), IsOkAndHolds(true));
  Matcher<const HloInstruction*> combined_all_reduce =
      op::AllReduce(op::Parameter(0), op::Parameter(1));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(combined_all_reduce, 0),
                        op::GetTupleElement(combined_all_reduce, 1)));
}

TEST_F(GpuAllReduceCombinerTest,
       DoNotCombineCollectivesWithControlDependencies) {
  absl::string_view kHloText = R"(
    HloModule m

    add {
      p0 = f16[] parameter(0)
      p1 = f16[] parameter(1)
      ROOT add = f16[] add(p0, p1)
    }

    ENTRY main {
      p0 = f16[10000]{0} parameter(0)
      p1 = f16[10000]{0} parameter(1)

      // This all-reduce must happen first, which is enforced by the control
      // dependency and must be respected.
      lead_ar = f16[10000]{0} all-reduce(p0), replica_groups={}, to_apply=add

      // These all-reduce have control dependencies and must not be combined.
      ar0 = f16[10000]{0} all-reduce(p0), replica_groups={}, to_apply=add,
          control-predecessors={lead_ar}
      ar1 = f16[10000]{0} all-reduce(p1), replica_groups={}, to_apply=add,
          control-predecessors={lead_ar}
      ROOT result = tuple(ar0, ar1)
    }
  )";
  DeviceDescription device_info;
  device_info.set_device_memory_size(10000000000);  // 10GB

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  GpuAllReduceCombiner combiner(
      device_info, /*default_combine_threshold_in_bytes=*/
      kDefaultAllReduceCombineThreshold,
      /*combine_threshold_in_bytes=*/kDefaultAllReduceCombineThreshold,
      /*combine_threshold_count=*/256, /*pointer_size=*/4);
  EXPECT_THAT(combiner.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(GpuAllReduceCombinerTest,
       DontCombinePipelinedAndSynchronousCollectives) {
  absl::string_view kHloText = R"(
    HloModule m

    add {
      p0 = f16[] parameter(0)
      p1 = f16[] parameter(1)
      ROOT add = f16[] add(p0, p1)
    }

    ENTRY main {
      p0 = f16[1000000]{0} parameter(0)
      p1 = f16[1000000]{0} parameter(1)

      ar0 = f16[1000000]{0} all-reduce(p0), replica_groups={}, to_apply=add,
        frontend_attributes={sync_collective="true"},
        backend_config={"collective_backend_config": {"is_pipelined": true}}
      ar1 = f16[1000000]{0} all-reduce(p1), replica_groups={}, to_apply=add,
        frontend_attributes={sync_collective="true"}

      ROOT result = tuple(ar0, ar1)
    }
  )";
  DeviceDescription device_info;
  device_info.set_device_memory_size(10000000000);  // 10GB

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  GpuAllReduceCombiner combiner(
      device_info, /*default_combine_threshold_in_bytes=*/
      kDefaultAllReduceCombineThreshold,
      /*combine_threshold_in_bytes=*/kDefaultAllReduceCombineThreshold,
      /*combine_threshold_count=*/256, /*pointer_size=*/4);

  EXPECT_THAT(combiner.Run(module.get()), IsOkAndHolds(false));
}

}  // namespace

}  // namespace xla::gpu

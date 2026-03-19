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

#include "xla/backends/cpu/transforms/collectives/all_reduce_combiner.h"

#include <cstdint>
#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

using CpuAllReduceCombinerTest = HloHardwareIndependentTestBase;

absl::StatusOr<bool> RunCombiner(HloModule* module) {
  return CpuAllReduceCombiner(std::numeric_limits<int64_t>::max(),
                              std::numeric_limits<int64_t>::max())
      .Run(module);
}

TEST_F(CpuAllReduceCombinerTest, CombinesCollectivesUpToSpecifiedThreshold) {
  constexpr absl::string_view kHloString = R"(
HloModule m

add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[16,256,8,128]{3,2,1,0} parameter(0)
  p1 = f32[32,256,2,128]{3,2,1,0} parameter(1)
  p2 = f32[16,8,128,256]{3,2,1,0} parameter(2)
  p3 = f32[32,256,3072]{2,1,0} parameter(3)
  p4 = f32[16,3072,256]{2,1,0} parameter(4)
  p5 = f32[66,1024]{1,0} parameter(5)
  p6 = f32[2,32768,256]{2,1,0} parameter(6)
  r0 = f32[16,256,8,128]{3,2,1,0} all-reduce(p0), channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=add
  r1 = f32[32,256,2,128]{3,2,1,0} all-reduce(p1), channel_id=2, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=add
  r2 = f32[16,8,128,256]{3,2,1,0} all-reduce(p2), channel_id=3, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=add
  r3 = f32[32,256,3072]{2,1,0} all-reduce(p3), channel_id=4, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=add
  r4 = f32[16,3072,256]{2,1,0} all-reduce(p4), channel_id=5, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=add
  r5 = f32[66,1024]{1,0} all-reduce(p5), channel_id=6, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=add
  r6 = f32[2,32768,256]{2,1,0} all-reduce(p6), channel_id=7, replica_groups={{0,1,2,3}}, use_global_device_ids=true, to_apply=add
  ROOT tuple = tuple(r0, r1, r2, r3, r4, r5, r6)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  EXPECT_THAT(RunCombiner(module.get()), absl_testing::IsOkAndHolds(true));

  constexpr absl::string_view kExpected = R"('
    // CHECK:      ENTRY
    // CHECK-DAG:  %[[P0:.*]] = parameter(0)
    // CHECK-DAG:  %[[P1:.*]] = parameter(1)
    // CHECK-DAG:  %[[P2:.*]] = parameter(2)
    // CHECK-DAG:  %[[P3:.*]] = parameter(3)
    // CHECK-DAG:  %[[P4:.*]] = parameter(4)
    // CHECK-DAG:  %[[P5:.*]] = parameter(5)
    // CHECK-DAG:  %[[P6:.*]] = parameter(6)
    // CHECK:      all-reduce(%[[P0]], %[[P1]], %[[P2]], %[[P3]], %[[P4]], %[[P5]], %[[P6]])
    // CHECK-SAME: channel_id=1,
    // CHECK-SAME: replica_groups={
    // CHECK-SAME:   {0,1,2,3}
    // CHECK-SAME: },
    // CHECK-SAME: use_global_device_ids=true,
    // CHECK-SAME: to_apply=%add
  )";

  EXPECT_TRUE(*RunFileCheck(
      module->ToString(HloPrintOptions()
                           .set_print_operand_index_annotation_interval(false)
                           .set_print_operand_shape(false)
                           .set_print_result_shape(false)),
      kExpected));
}

}  // namespace
}  // namespace xla::cpu

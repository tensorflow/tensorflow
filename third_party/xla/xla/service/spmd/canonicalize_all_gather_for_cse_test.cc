/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/spmd/canonicalize_all_gather_for_cse.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace spmd {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::_;
using ::testing::AllOf;
namespace op = xla::testing::opcode_matchers;

class AllGatherCanonicalizeTest : public HloHardwareIndependentTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module) {
    ABSL_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(
                                      hlo_module, GetModuleConfigForTest()));
    HloPassPipeline pipeline("all-gather-cse");
    pipeline.AddPass<CanonicalizeAllGatherForCSE>();
    ABSL_RETURN_IF_ERROR(pipeline.Run(module.get()).status());
    return absl::StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }
  absl::Status RunPassOnModule(HloModule* module,
                               int64_t distance_threshold = 100) {
    HloPassPipeline pipeline("all-gather-cse");
    pipeline.AddPass<CanonicalizeAllGatherForCSE>();
    ABSL_RETURN_IF_ERROR(pipeline.Run(module).status());
    return absl::OkStatus();
  }
};

TEST_F(AllGatherCanonicalizeTest, SimpleReshape) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param0 = s32[8]{0} parameter(0)
  resh = s32[1,8]{1,0} reshape(param0)
  ROOT ag = s32[2,8]{1,0} all-gather(resh), replica_groups={{0,1}},
    dimensions={0}, channel_id=0, use_global_device_ids=true
})";
  auto module_status = RunPass(hlo_string);
  EXPECT_TRUE(module_status.status().ok());
  auto module = std::move(module_status).value();
  const HloInstruction* const reshape =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(reshape,
              AllOf(op::Reshape(op::AllGather(_)), op::Shape("s32[2,8]")));
}

TEST_F(AllGatherCanonicalizeTest, MultipleDegenerateReshapes) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param0 = s32[8]{0} parameter(0)
  resh = s32[1,8]{1,0} reshape(param0)
  resh2 = s32[1,8,1,1]{3,2,1,0} reshape(resh)
  ROOT ag = s32[2,8,1,1]{3,2,1,0} all-gather(resh2), replica_groups={{0,1}},
    dimensions={0}, channel_id=0, use_global_device_ids=true
})";
  auto module_status = RunPass(hlo_string);
  EXPECT_TRUE(module_status.status().ok());
  auto module = std::move(module_status).value();
  const HloInstruction* const reshape =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(reshape, op::Reshape(op::AllGather(op::Parameter())));
}

TEST_F(AllGatherCanonicalizeTest, MultipleDegenerateReshapes2) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param0 = s32[8]{0} parameter(0)
  resh = s32[8,1,1]{2,1,0} reshape(param0)
  resh2 = s32[1,8,1,1]{3,2,1,0} reshape(resh)
  ROOT ag = s32[2,8,1,1]{3,2,1,0} all-gather(resh2), replica_groups={{0,1}},
    dimensions={0}, channel_id=0, use_global_device_ids=true
})";
  auto module_status = RunPass(hlo_string);
  EXPECT_TRUE(module_status.status().ok());
  auto module = std::move(module_status).value();
  const HloInstruction* const reshape =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(reshape, op::Reshape(op::AllGather(op::Parameter())));
}

TEST_F(AllGatherCanonicalizeTest, MultipleDegenerateReshapesNoDim0) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param0 = s32[8]{0} parameter(0)
  resh = s32[8,1,1]{2,1,0} reshape(param0)
  resh2 = s32[1,8,1,1]{3,2,1,0} reshape(resh)
  ROOT ag = s32[1,16,1,1]{3,2,1,0} all-gather(resh2), replica_groups={{0,1}},
    dimensions={1}, channel_id=0, use_global_device_ids=true
})";
  auto module_status = RunPass(hlo_string);
  EXPECT_TRUE(module_status.status().ok());
  auto module = std::move(module_status).value();
  const HloInstruction* const reshape =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(reshape, op::Reshape(op::AllGather(op::Parameter())));
}

TEST_F(AllGatherCanonicalizeTest, NonDegenerateReshape) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  param0 = s32[8]{0} parameter(0)
  resh = s32[8,1,1]{2,1,0} reshape(param0)
  resh2 = s32[1,4,2,1,1]{4,3,2,1,0} reshape(resh)
  ROOT ag = s32[2,4,2,1,1]{4,3,2,1,0} all-gather(resh2), replica_groups={{0,1}},
    dimensions={0}, channel_id=0, use_global_device_ids=true
})";
  auto module_status = RunPass(hlo_string);
  EXPECT_TRUE(module_status.status().ok());
  auto module = std::move(module_status).value();
  const HloInstruction* const reshape =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(reshape, AllOf(op::AllGather(op::Reshape(op::Reshape(_))),
                             op::Shape("s32[2,4,2,1,1]")));
}

TEST_F(AllGatherCanonicalizeTest, InspectionDoesNotCrossCallBoundaries) {
  absl::string_view hlo_string = R"(
HloModule module

foo {
  param_foo = s32[1,8]{1,0} parameter(0)
  ROOT ag = s32[2,8]{1,0} all-gather(param_foo), replica_groups={{0,1}},
    dimensions={0}, channel_id=0, use_global_device_ids=true
}

ENTRY entry {
  param0 = s32[8]{0} parameter(0)
  resh = s32[1,8]{1,0} reshape(param0)
  ROOT call = s32[2,8]{1,0} call(resh), to_apply=foo
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  CanonicalizeAllGatherForCSE pass;
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(AllGatherCanonicalizeTest, TransformationInCalledComputationFoo) {
  absl::string_view hlo_string = R"(
HloModule module

// CHECK-LABEL: %foo
// CHECK:         %[[PARAM:.*]] = s32[8]{0} parameter(0)
// CHECK:         %[[AG:.*]] = s32[16]{0} all-gather(%[[PARAM]]), {{.*}}dimensions={0}
// CHECK:         ROOT %[[RESH:.*]] = s32[2,8]{1,0} reshape(%[[AG]])
foo {
  param_foo = s32[8]{0} parameter(0)
  resh = s32[1,8]{1,0} reshape(param_foo)
  ROOT ag = s32[2,8]{1,0} all-gather(resh), replica_groups={{0,1}},
    dimensions={0}, channel_id=0, use_global_device_ids=true
}

// CHECK-LABEL: ENTRY %entry
// CHECK:         %[[PARAM0:.*]] = s32[8]{0} parameter(0)
// CHECK:         ROOT %[[CALL:.*]] = s32[2,8]{1,0} call(%[[PARAM0]]), to_apply=%foo
ENTRY entry {
  param0 = s32[8]{0} parameter(0)
  ROOT call = s32[2,8]{1,0} call(param0), to_apply=foo
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  CanonicalizeAllGatherForCSE pass;
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(module->ToString(), hlo_string), IsOkAndHolds(true));
}

TEST_F(AllGatherCanonicalizeTest,
       EntryCallsFooAndBarFooCallsBarTransformationInBar) {
  absl::string_view hlo_string = R"(
HloModule module

// CHECK-LABEL: %bar
// CHECK:         %[[PARAM_BAR:.*]] = s32[8]{0} parameter(0)
// CHECK:         %[[AG:.*]] = s32[16]{0} all-gather(%[[PARAM_BAR]]), {{.*}}dimensions={0}
// CHECK:         ROOT %[[RESH:.*]] = s32[2,8]{1,0} reshape(%[[AG]])
bar {
  param_bar = s32[8]{0} parameter(0)
  resh = s32[1,8]{1,0} reshape(param_bar)
  ROOT ag = s32[2,8]{1,0} all-gather(resh), replica_groups={{0,1}},
    dimensions={0}, channel_id=0, use_global_device_ids=true
}

// CHECK-LABEL: %foo
// CHECK:         %[[PARAM_FOO:.*]] = s32[8]{0} parameter(0)
// CHECK:         ROOT %[[CALL_BAR:.*]] = s32[2,8]{1,0} call(%[[PARAM_FOO]]), to_apply=%bar
foo {
  param_foo = s32[8]{0} parameter(0)
  ROOT call_bar_from_foo = s32[2,8]{1,0} call(param_foo), to_apply=bar
}

// CHECK-LABEL: ENTRY %entry
// CHECK:         %[[PARAM0:.*]] = s32[8]{0} parameter(0)
// CHECK-DAG:     %[[CALL_FOO:.*]] = s32[2,8]{1,0} call(%[[PARAM0]]), to_apply=%foo
// CHECK-DAG:     %[[CALL_BAR:.*]] = s32[2,8]{1,0} call(%[[PARAM0]]), to_apply=%bar
// CHECK:         ROOT %[[TUPLE:.*]] = (s32[2,8]{1,0}, s32[2,8]{1,0}) tuple(%[[CALL_FOO]], %[[CALL_BAR]])
ENTRY entry {
  param0 = s32[8]{0} parameter(0)
  call_foo = s32[2,8]{1,0} call(param0), to_apply=foo
  call_bar = s32[2,8]{1,0} call(param0), to_apply=bar
  ROOT root = (s32[2,8]{1,0}, s32[2,8]{1,0}) tuple(call_foo, call_bar)
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  CanonicalizeAllGatherForCSE pass;
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(module->ToString(), hlo_string), IsOkAndHolds(true));
}

}  // namespace
}  // namespace spmd
}  // namespace xla

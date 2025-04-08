/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/service/gpu/transforms/explicit_collectives_group_async_wrapper.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ExplicitCollectivesGroupAsyncWrapperTest = HloTestBase;

TEST_F(ExplicitCollectivesGroupAsyncWrapperTest, AnnotatedOpIsWrapped) {
  const absl::string_view hlo_string = R"(
  HloModule composite
  comms {
    a = f32[1] parameter(0)
    x = f32[1] all-gather(a), dimensions={0}
    y = f32[1] collective-permute(a), source_target_pairs={{0,1}}
    ROOT result = (f32[1], f32[1]) tuple(x, y)
  }

  ENTRY main {
    b = f32[1] parameter(0)
    ROOT c = (f32[1], f32[1]) call(b), to_apply=comms, frontend_attributes={_collectives_group=""}
  }
  )";

  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  ExplicitCollectivesGroupAsyncWrapper wrapper_pass;

  TF_ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: %b = f32[1]{0} parameter(0)
  // CHECK: %tuple-start = ((f32[1]{0}), (f32[1]{0}, f32[1]{0})) async-start(%b), async_execution_thread="explicit", calls=%comms.collectives_group, frontend_attributes={_collectives_group=""}
  // CHECK: ROOT %tuple-done = (f32[1]{0}, f32[1]{0}) async-done(%tuple-start), frontend_attributes={_collectives_group=""}
  )");
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);
  ASSERT_TRUE(mutated);
}

TEST_F(ExplicitCollectivesGroupAsyncWrapperTest,
       RemoveSchedulingGroupAnnotation) {
  const absl::string_view hlo_string = R"(
  HloModule composite
  comms {
    a = f32[1] parameter(0)
    x = f32[1] all-gather(a), replica_groups={}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    y = f32[1] collective-permute(a), source_target_pairs={{0,1}}, frontend_attributes={_scheduling_group_id="1"}
    ROOT result = (f32[1], f32[1]) tuple(x, y)
  }

  ENTRY main {
    b = f32[1] parameter(0)
    ROOT c = (f32[1], f32[1]) call(b), to_apply=comms, frontend_attributes={_collectives_group="", _scheduling_group_id="1"}
  }
  )";

  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  ExplicitCollectivesGroupAsyncWrapper wrapper_pass;

  TF_ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  // Assert that the scheduling annotation is removed within the cloned
  // computation, but remains on the async operations.
  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: %comms.collectives_group {{.*}} {
  // CHECK-NEXT: %{{.*}} parameter(0)
  // CHECK-NEXT: %{{.*}} all-gather({{.*}}), replica_groups={}, dimensions={0}
  // CHECK-NEXT: %{{.*}} collective-permute({{.*}}), source_target_pairs={{[{][{]0,1[}][}]}}
  // CHECK: ENTRY %main {{.*}}
  // CHECK-NEXT: %[[P0:.*]] = {{.*}} parameter(0)
  // CHECK-NEXT: %[[P1:.*]] = {{.*}} async-start(%[[P0]]), async_execution_thread="explicit", calls=%comms.collectives_group, frontend_attributes={_collectives_group="",_scheduling_group_id="1"}  
  // CHECK-NEXT: ROOT %{{.*}} async-done(%[[P1]]), frontend_attributes={_collectives_group="",_scheduling_group_id="1"}
  )");
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);
  ASSERT_TRUE(mutated);
}

TEST_F(ExplicitCollectivesGroupAsyncWrapperTest, ManyCollectivesGroups) {
  // This test calls the same collectives group computation twice, so the
  // computation is cloned so it can be used with many async instructions.
  const absl::string_view hlo_string = R"(
  HloModule composite
  comms {
    a = f32[1] parameter(0)
    x = f32[1] all-gather(a), dimensions={0}
    y = f32[1] collective-permute(a), source_target_pairs={{0,1}}
    ROOT result = (f32[1], f32[1]) tuple(x, y)
  }

  ENTRY main {
    b = f32[1] parameter(0)
    group1 = (f32[1], f32[1]) call(b), to_apply=comms, frontend_attributes={_collectives_group=""}
    c = get-tuple-element(group1), index=0
    ROOT d = (f32[1], f32[1]) call(c), to_apply=comms, frontend_attributes={_collectives_group=""}
  }
  )";

  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  ExplicitCollectivesGroupAsyncWrapper wrapper_pass;

  TF_ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: %b = f32[1]{0} parameter(0)
  // CHECK: %tuple-start = ((f32[1]{0}), (f32[1]{0}, f32[1]{0})) async-start(%b), async_execution_thread="explicit", calls=%comms.collectives_group, frontend_attributes={_collectives_group=""} 
  // CHECK: %tuple-done = (f32[1]{0}, f32[1]{0}) async-done(%tuple-start), frontend_attributes={_collectives_group=""}
  // CHECK: %c = f32[1]{0} get-tuple-element(%tuple-done), index=0
  // CHECK: %tuple-start.1 = ((f32[1]{0}), (f32[1]{0}, f32[1]{0})) async-start(%c), async_execution_thread="explicit", calls=%comms.collectives_group.1, frontend_attributes={_collectives_group=""}
  // CHECK: ROOT %tuple-done.1 = (f32[1]{0}, f32[1]{0}) async-done(%tuple-start.1), frontend_attributes={_collectives_group=""}
  )");
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);
  ASSERT_TRUE(mutated);
}

}  // namespace
}  // namespace xla::gpu

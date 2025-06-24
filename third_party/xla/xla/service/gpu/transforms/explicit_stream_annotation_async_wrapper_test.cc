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

#include "xla/service/gpu/transforms/explicit_stream_annotation_async_wrapper.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ExplicitStreamAnnotationAsyncWrapperTest = HloHardwareIndependentTestBase;

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest, AnnotatedOpIsWrapped) {
  const absl::string_view hlo_string = R"(
  HloModule composite

  %sub (lhs: f32[]) -> f32[] {
    %lhs = f32[] parameter(0)
    %rhs = f32[] constant(1)
    ROOT %sub = f32[] subtract(f32[] %lhs, f32[] %rhs)
  }

  ENTRY %main () -> f32[] {
    %lhs = f32[] constant(42)
    %call1 = f32[] call(f32[] %lhs), to_apply=%sub, frontend_attributes={_xla_stream_annotation="1"}
  })";

  auto debug_options = HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  TF_ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: %lhs.1 = f32[] constant(42)
  // CHECK: %call-start = ((f32[]), f32[]) call-start(%lhs.1), async_execution_thread="explicit", to_apply=%sub, frontend_attributes={_xla_stream_annotation="1"}
  // CHECK: ROOT %call-done = f32[] call-done(%call-start), frontend_attributes={_xla_stream_annotation="1"}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"force_earliest_schedule":false
  )");
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);

  ASSERT_TRUE(mutated);
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest, OverlappingGemms) {
  const absl::string_view hlo_string = R"(
  HloModule composite

  %gemm1 (z: f32[2048,2048], w: f32[2048,2048]) -> f32[2048,2048] {
    %w = f32[2048,2048]{1,0} parameter(1)
    %z = f32[2048,2048]{1,0} parameter(0)
    %custom-call.1 = (f32[2048,2048]{1,0}, s8[33554432]{0}) custom-call(f32[2048,2048]{1,0} %w, f32[2048,2048]{1,0} %z), custom_call_target="__cublas$gemm", 
      frontend_attributes={_scheduling_group_id="0", _xla_stream_annotation="1"}
    ROOT %get-tuple-element = f32[2048,2048]{1,0} get-tuple-element((f32[2048,2048]{1,0}, s8[33554432]{0}) %custom-call.1), index=0
  }
  %gemm2 (a: f32[2048,2048], b: f32[2048,2048]) -> f32[2048,2048] {
    %a = f32[2048,2048]{1,0} parameter(1)
    %b = f32[2048,2048]{1,0} parameter(0)
    %custom-call.2 = (f32[2048,2048]{1,0}, s8[33554432]{0}) custom-call(f32[2048,2048]{1,0} %a, f32[2048,2048]{1,0} %b), custom_call_target="__cublas$gemm",
          frontend_attributes={_scheduling_group_id="1", _xla_stream_annotation="2"}
    ROOT %get-tuple-element = f32[2048,2048]{1,0} get-tuple-element((f32[2048,2048]{1,0}, s8[33554432]{0}) %custom-call.2), index=0
  }

  ENTRY %main () -> f32[2048,2048]{1,0} {
    %x = f32[2048,2048]{1,0} parameter(1), metadata={op_name="b" scheduling_name="x"}
    %y = f32[2048,2048]{1,0} parameter(0), metadata={op_name="a" scheduling_name="y"}
    %call1 =  f32[2048,2048]{1,0} call(f32[2048,2048]{1,0} %x, f32[2048,2048]{1,0} %y ), to_apply=%gemm1, frontend_attributes={_scheduling_group_id="0", _xla_stream_annotation="2"}
    ROOT %call2 =  f32[2048,2048]{1,0} call(f32[2048,2048]{1,0} %x, f32[2048,2048]{1,0} %y), to_apply=%gemm2, frontend_attributes={_scheduling_group_id="1", _xla_stream_annotation="1"}
  })";

  auto debug_options = HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  TF_ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_TRUE(mutated);

  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: %call-start = ((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) call-start(%x, %y), async_execution_thread="explicit", to_apply=%gemm1, frontend_attributes={_scheduling_group_id="0",_xla_stream_annotation="2"}
  // CHECK: %call-done = f32[2048,2048]{1,0} call-done(%call-start), frontend_attributes={_scheduling_group_id="0",_xla_stream_annotation="2"}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"force_earliest_schedule":false
  // CHECK: %call-start.1 = ((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) call-start(%x, %y), async_execution_thread="explicit", to_apply=%gemm2, frontend_attributes={_scheduling_group_id="1",_xla_stream_annotation="1"}
  // CHECK: ROOT %call-done.1 = f32[2048,2048]{1,0} call-done(%call-start.1), frontend_attributes={_scheduling_group_id="1",_xla_stream_annotation="1"}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"force_earliest_schedule":false
  )");
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);
  for (auto name : {"call-start", "call-done"}) {
    EXPECT_EQ(FindInstruction(module.get(), name)
                  ->frontend_attributes()
                  .map()
                  .find(kXlaStreamAnnotationAttr)
                  ->second,
              "2");
    EXPECT_EQ(FindInstruction(module.get(), name)
                  ->frontend_attributes()
                  .map()
                  .find(kXlaSchedulingGroupIdAttr)
                  ->second,
              "0");
  }
  for (auto name : {"call-start.1", "call-done.1"}) {
    EXPECT_EQ(FindInstruction(module.get(), name)
                  ->frontend_attributes()
                  .map()
                  .find(kXlaStreamAnnotationAttr)
                  ->second,
              "1");
    EXPECT_EQ(FindInstruction(module.get(), name)
                  ->frontend_attributes()
                  .map()
                  .find(kXlaSchedulingGroupIdAttr)
                  ->second,
              "1");
  }
  // Ensure the operations within the async computation are not annotated
  // anymore.
  for (auto annotation :
       {kXlaSchedulingGroupIdAttr, kXlaStreamAnnotationAttr}) {
    for (auto name : {"custom-call.1", "custom-call.2"}) {
      EXPECT_FALSE(FindInstruction(module.get(), name)
                       ->frontend_attributes()
                       .map()
                       .contains(annotation));
    }
  }
}
}  // namespace
}  // namespace xla::gpu

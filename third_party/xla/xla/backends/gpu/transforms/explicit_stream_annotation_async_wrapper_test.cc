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

#include "xla/backends/gpu/transforms/explicit_stream_annotation_async_wrapper.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
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
  // CHECK: %call-start = ((f32[]), f32[]) call-start(%lhs.1), to_apply=%sub, frontend_attributes={_xla_stream_annotation="1"}
  // CHECK: ROOT %call-done = f32[] call-done(%call-start), frontend_attributes={_xla_stream_annotation="1"}, backend_config={"operation_queue_id":"0","force_earliest_schedule":false
  )");
  ASSERT_OK(filecheck_result.status());
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
  // CHECK: %call-start = ((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) call-start(%x, %y), to_apply=%gemm1, frontend_attributes={_scheduling_group_id="0",_xla_stream_annotation="2"}
  // CHECK: %call-done = f32[2048,2048]{1,0} call-done(%call-start), frontend_attributes={_scheduling_group_id="0",_xla_stream_annotation="2"}, backend_config={"operation_queue_id":"0","force_earliest_schedule":false
  // CHECK: %call-start.1 = ((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) call-start(%x, %y), to_apply=%gemm2, frontend_attributes={_scheduling_group_id="1",_xla_stream_annotation="1"}
  // CHECK: ROOT %call-done.1 = f32[2048,2048]{1,0} call-done(%call-start.1), frontend_attributes={_scheduling_group_id="1",_xla_stream_annotation="1"}, backend_config={"operation_queue_id":"0","force_earliest_schedule":false
  )");
  ASSERT_OK(filecheck_result.status());
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
TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       AnnotatedNonCallOpIsWrappedInCall) {
  const absl::string_view hlo_string = R"(
  HloModule m

  ENTRY %main (a: f32[4]) -> f32[4] {
    %a = f32[4]{0} parameter(0)
    ROOT %cc = f32[4]{0} custom-call(%a), custom_call_target="foo",
      frontend_attributes={_xla_stream_annotation="1"}
  })";

  auto debug_options = HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_TRUE(mutated);

  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: %call-start = {{.*}} call-start(%a), to_apply=
  // CHECK-SAME: frontend_attributes={_xla_stream_annotation="1"}
  // CHECK: ROOT %call-done = f32[4]{0} call-done(%call-start)
  // CHECK-SAME: frontend_attributes={_xla_stream_annotation="1"}
  // CHECK-SAME: "force_earliest_schedule":false
  )");
  ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       AnnotatedNonCallOpFrontendAttributesMovedToWrapper) {
  const absl::string_view hlo_string = R"(
  HloModule m

  ENTRY %main (a: f32[4]) -> f32[4] {
    %a = f32[4]{0} parameter(0)
    ROOT %cc = f32[4]{0} custom-call(%a), custom_call_target="te_ep_foo",
      frontend_attributes={_xla_stream_annotation="collective",inlineable="false"}
  })";

  auto debug_options = HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_TRUE(mutated);

  // The async start and done should carry all original frontend attributes.
  for (auto name : {"call-start", "call-done"}) {
    const auto& attrs =
        FindInstruction(module.get(), name)->frontend_attributes().map();
    EXPECT_EQ(attrs.at(kXlaStreamAnnotationAttr), "collective");
    EXPECT_EQ(attrs.at("inlineable"), "false");
  }

  // No instruction in any non-entry computation should have the stream
  // annotation — it must only live on the async start/done pair.
  for (HloComputation* comp : module->computations()) {
    if (comp == module->entry_computation()) {
      continue;
    }
    for (HloInstruction* instr : comp->instructions()) {
      EXPECT_FALSE(
          instr->frontend_attributes().map().contains(kXlaStreamAnnotationAttr))
          << "Unexpected annotation on inner instruction: " << instr->name();
    }
  }
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       AnnotatedInstructionInFusionBodyIsNotWrapped) {
  const absl::string_view hlo_string = R"(
  HloModule m

  %fused_computation (a: f32[4]) -> f32[4] {
    %a = f32[4]{0} parameter(0)
    ROOT %negate = f32[4]{0} negate(%a),
      frontend_attributes={_xla_stream_annotation="1"}
  }

  ENTRY %main (a: f32[4]) -> f32[4] {
    %a = f32[4]{0} parameter(0)
    ROOT %fusion = f32[4]{0} fusion(%a), kind=kLoop, calls=%fused_computation
  })";

  auto debug_options = HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_FALSE(mutated);
  // The annotation must survive untouched inside the fusion body.
  HloInstruction* negate = FindInstruction(module.get(), "negate");
  ASSERT_NE(negate, nullptr);
  EXPECT_TRUE(
      negate->frontend_attributes().map().contains(kXlaStreamAnnotationAttr));
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       AnnotatedInstructionInAsyncBodyIsNotWrapped) {
  const absl::string_view hlo_string = R"(
  HloModule m

  %async_comp (a: f32[4]) -> f32[4] {
    %a = f32[4]{0} parameter(0)
    ROOT %negate = f32[4]{0} negate(%a),
      frontend_attributes={_xla_stream_annotation="1"}
  }

  ENTRY %main (a: f32[4]) -> f32[4] {
    %a = f32[4]{0} parameter(0)
    %async-start = ((f32[4]{0}), f32[4]{0}) async-start(%a), calls=%async_comp
    ROOT %async-done = f32[4]{0} async-done(%async-start)
  })";

  auto debug_options = HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_FALSE(mutated);
  // The annotation inside the async computation body must not be processed.
  HloInstruction* negate = FindInstruction(module.get(), "negate");
  ASSERT_NE(negate, nullptr);
  EXPECT_TRUE(
      negate->frontend_attributes().map().contains(kXlaStreamAnnotationAttr));
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       AnnotatedAsyncStartDoneIsNotRewrapped) {
  const absl::string_view hlo_string = R"(
  HloModule m

  %async_comp (a: f32[4]) -> f32[4] {
    %a = f32[4]{0} parameter(0)
    ROOT %negate = f32[4]{0} negate(%a)
  }

  ENTRY %main (a: f32[4]) -> f32[4] {
    %a = f32[4]{0} parameter(0)
    %async-start = ((f32[4]{0}), f32[4]{0}) async-start(%a), calls=%async_comp,
      frontend_attributes={_xla_stream_annotation="1"}
    ROOT %async-done = f32[4]{0} async-done(%async-start),
      frontend_attributes={_xla_stream_annotation="1"}
  })";

  auto debug_options = HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_FALSE(mutated);
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       AnnotatedCollectiveIsNotWrapped) {
  // Uses all-reduce-start/done (legacy async collective forms) and a sync
  // all-reduce to exercise the IsNonFusionCollective skip path.
  const absl::string_view hlo_string = R"(
  HloModule m

  %add (x: f32[], y: f32[]) -> f32[] {
    %x = f32[] parameter(0)
    %y = f32[] parameter(1)
    ROOT %add = f32[] add(%x, %y)
  }

  ENTRY %main (a: f32[4]) -> f32[4] {
    %a = f32[4]{0} parameter(0)
    %ar-start = (f32[4]{0}, f32[4]{0}) all-reduce-start(%a),
      to_apply=%add, replica_groups={},
      frontend_attributes={_xla_stream_annotation="collective"}
    ROOT %ar-done = f32[4]{0} all-reduce-done(%ar-start),
      frontend_attributes={_xla_stream_annotation="collective"}
  })";

  auto debug_options = HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_string));
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_FALSE(mutated);
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       AnnotatedCopyStartDoneIsNotWrapped) {
  const absl::string_view hlo_string = R"(
  HloModule m

  ENTRY %main (a: f32[4]) -> f32[4] {
    %a = f32[4]{0} parameter(0)
    %cs = (f32[4]{0}, f32[4]{0}, u32[]) copy-start(%a),
      frontend_attributes={_xla_stream_annotation="1"}
    ROOT %cd = f32[4]{0} copy-done(%cs),
      frontend_attributes={_xla_stream_annotation="1"}
  })";

  auto debug_options = HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_string));
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_FALSE(mutated);
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       AnnotatedInstructionInSortComparatorInsideFusionIsNotWrapped) {
  // Mirrors the crash in te_stream_e2e.hlo: a sort comparator called by a sort
  // inside a fusion has instructions with _xla_stream_annotation.  The pass
  // must walk up caller_computations() and recognise that the comparator is
  // transitively nested inside a fusion, then skip it entirely.
  const absl::string_view hlo_string = R"(
  HloModule m

  %comparator (lhs: f32[], rhs: f32[]) -> pred[] {
    %lhs = f32[] parameter(0)
    %rhs = f32[] parameter(1)
    ROOT %compare = pred[] compare(%lhs, %rhs), direction=LT,
      frontend_attributes={_xla_stream_annotation="collective"}
  }

  %fused_sort (p: f32[8]) -> f32[8] {
    %p = f32[8]{0} parameter(0)
    ROOT %sort = f32[8]{0} sort(%p), dimensions={0}, to_apply=%comparator
  }

  ENTRY %main (a: f32[8]) -> f32[8] {
    %a = f32[8]{0} parameter(0)
    ROOT %fusion = f32[8]{0} fusion(%a), kind=kCustom, calls=%fused_sort
  })";

  auto debug_options = HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_FALSE(mutated);
  // The annotation must survive untouched on the compare instruction.
  HloInstruction* compare = FindInstruction(module.get(), "compare");
  ASSERT_NE(compare, nullptr);
  EXPECT_TRUE(
      compare->frontend_attributes().map().contains(kXlaStreamAnnotationAttr));
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       UnannotatedNonCallOpIsNotWrapped) {
  const absl::string_view hlo_string = R"(
  HloModule m

  ENTRY %main (a: f32[4]) -> f32[4] {
    %a = f32[4]{0} parameter(0)
    ROOT %cc = f32[4]{0} custom-call(%a), custom_call_target="foo"
  })";

  auto debug_options = HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_FALSE(mutated);
  // Verify the module still has the original custom-call directly in ENTRY.
  EXPECT_NE(FindInstruction(module.get(), "cc"), nullptr);
}

}  // namespace
}  // namespace xla::gpu

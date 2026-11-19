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
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

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

  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: %lhs.1 = f32[] constant(42)
  // CHECK: %call-start = ((f32[]), f32[]) call-start(%lhs.1), async_execution_thread="parallel", to_apply=%sub, frontend_attributes={_xla_stream_annotation="1"}
  // CHECK: ROOT %call-done = f32[] call-done(%call-start), frontend_attributes={_xla_stream_annotation="1"}, backend_config={"operation_queue_id":"0","force_earliest_schedule":false
  )");
  ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);

  ASSERT_TRUE(mutated);
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       AnnotatedInstructionsWithoutThunksAreNotWrapped) {
  // Instructions can inherit the annotation of the op they belong to, e.g. the
  // get-tuple-elements that pick out the results of a multi-result call. Those
  // that do no work on any stream must not become async calls of their own;
  // those that do (here partition-id, which writes its result with a memset)
  // must still run on the annotated stream.
  const absl::string_view hlo_string = R"(
  HloModule m

  %two (p: f32[]) -> (f32[], f32[]) {
    %p = f32[] parameter(0)
    %one = f32[] constant(1)
    %add = f32[] add(%p, %one)
    ROOT %t = (f32[], f32[]) tuple(%p, %add)
  }

  ENTRY %main (x: f32[]) -> (f32[1], f32[4], u32[]) {
    %x = f32[] parameter(0)
    %call = (f32[], f32[]) call(%x), to_apply=%two, frontend_attributes={_xla_stream_annotation="1"}
    %gte0 = f32[] get-tuple-element(%call), index=0, frontend_attributes={_xla_stream_annotation="1"}
    %gte1 = f32[] get-tuple-element(%call), index=1, frontend_attributes={_xla_stream_annotation="1"}
    %mul = f32[] multiply(%gte0, %gte1)
    %bitcast = f32[1] bitcast(%mul), frontend_attributes={_xla_stream_annotation="1"}
    %alloc = f32[4] custom-call(), custom_call_target="AllocateBuffer", frontend_attributes={_xla_stream_annotation="1"}
    %pid = u32[] partition-id(), frontend_attributes={_xla_stream_annotation="1"}
    ROOT %t = (f32[1], f32[4], u32[]) tuple(%bitcast, %alloc, %pid), frontend_attributes={_xla_stream_annotation="1"}
  })";

  DebugOptions debug_options =
      HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;
  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_TRUE(mutated);

  int num_async_starts = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    num_async_starts += instr->opcode() == HloOpcode::kAsyncStart;
    EXPECT_NE(instr->opcode(), HloOpcode::kPartitionId) << instr->ToString();
    if (instr->opcode() == HloOpcode::kGetTupleElement ||
        instr->opcode() == HloOpcode::kBitcast ||
        instr->opcode() == HloOpcode::kTuple ||
        instr->opcode() == HloOpcode::kCustomCall) {
      EXPECT_FALSE(instr->has_frontend_attribute(kXlaStreamAnnotationAttr))
          << instr->ToString();
    }
  }
  // Only the call and the partition-id are made asynchronous.
  EXPECT_EQ(num_async_starts, 2);
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

  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_TRUE(mutated);

  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: %call-start = ((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) call-start(%x, %y), async_execution_thread="parallel", to_apply=%gemm1, frontend_attributes={_scheduling_group_id="0",_xla_stream_annotation="2"}
  // CHECK: %call-done = f32[2048,2048]{1,0} call-done(%call-start), frontend_attributes={_scheduling_group_id="0",_xla_stream_annotation="2"}, backend_config={"operation_queue_id":"0","force_earliest_schedule":false
  // CHECK: %call-start.1 = ((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) call-start(%x, %y), async_execution_thread="parallel", to_apply=%gemm2, frontend_attributes={_scheduling_group_id="1",_xla_stream_annotation="1"}
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
  // CHECK: %call-start = {{.*}} call-start(%a), async_execution_thread="parallel", to_apply=
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
      EXPECT_FALSE(instr->has_frontend_attribute(kXlaStreamAnnotationAttr))
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
  EXPECT_TRUE(negate->has_frontend_attribute(kXlaStreamAnnotationAttr));
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
  EXPECT_TRUE(negate->has_frontend_attribute(kXlaStreamAnnotationAttr));
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
  EXPECT_TRUE(compare->has_frontend_attribute(kXlaStreamAnnotationAttr));
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

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       AsyncPairUsesParallelExecutionThread) {
  // Verify that the async-start/done pair created by the wrapper uses
  // async_execution_thread="parallel" (StreamAttributeAsyncWrapper::
  // kParallelExecutionThread) rather than the old "main" thread.  The parallel
  // thread is what causes ExecutionStreamAssignment to route the op to a
  // communication stream so it can overlap with compute ops on the main stream.
  const absl::string_view hlo_string = R"(
  HloModule m

  ENTRY %main (a: f32[4]) -> f32[4] {
    %a = f32[4]{0} parameter(0)
    ROOT %cc = f32[4]{0} custom-call(%a), custom_call_target="te_op",
      frontend_attributes={_xla_stream_annotation="collective",inlineable="false"}
  })";

  auto debug_options = HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config().set_debug_options(debug_options);
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_TRUE(mutated);

  // The printed form of async-start only emits async_execution_thread when it
  // differs from "main", so presence of the attribute confirms the change.
  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: call-start{{.*}}async_execution_thread="parallel"
  )");
  ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);

  // Also verify programmatically: the start instruction's execution thread and
  // the wrapped computation's thread must both equal kParallelExecutionThread.
  const HloInstruction* call_start =
      FindInstruction(module.get(), "call-start");
  ASSERT_NE(call_start, nullptr);
  const auto* async_start = DynCast<HloAsyncStartInstruction>(call_start);
  ASSERT_NE(async_start, nullptr);
  EXPECT_EQ(async_start->async_execution_thread(),
            HloInstruction::kParallelExecutionThread);
  EXPECT_EQ(async_start->async_wrapped_computation()->execution_thread(),
            HloInstruction::kParallelExecutionThread);
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainCallIsWrappedAndPreservesAttributes) {
  const absl::string_view hlo_string = R"(
  HloModule m

  body {
    p0 = f32[4]{0:S(3)} parameter(0)
    ROOT negate = f32[4]{0:S(3)} negate(p0)
  }

  ENTRY main {
    p0 = f32[4]{0:S(3)} parameter(0)
    ROOT call = f32[4]{0:S(3)} call(p0), to_apply=body,
      frontend_attributes={_xla_stream_annotation="locality_domain:3",inlineable="false"}
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_TRUE(mutated);
  for (absl::string_view name : {"call-start", "call-done"}) {
    const auto& attrs =
        FindInstruction(module.get(), name)->frontend_attributes().map();
    EXPECT_EQ(attrs.at(kXlaStreamAnnotationAttr), "locality_domain:3");
    EXPECT_EQ(attrs.at("inlineable"), "false");
  }
  EXPECT_EQ(FindInstruction(module.get(), "call-done")
                ->shape()
                .layout()
                .memory_space(),
            3);
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainWrappingIsIdempotent) {
  const absl::string_view hlo_string = R"(
  HloModule m

  body {
    p0 = f32[4] parameter(0)
    ROOT negate = f32[4] negate(p0)
  }

  ENTRY main {
    p0 = f32[4] parameter(0)
    ROOT call = f32[4] call(p0), to_apply=body,
      frontend_attributes={_xla_stream_annotation="locality_domain:1",inlineable="false"}
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  ASSERT_OK_AND_ASSIGN(bool first_mutated, wrapper_pass.Run(module.get()));
  ASSERT_TRUE(first_mutated);
  ASSERT_OK_AND_ASSIGN(bool second_mutated, wrapper_pass.Run(module.get()));
  EXPECT_FALSE(second_mutated);
  EXPECT_NE(FindInstruction(module.get(), "call-start"), nullptr);
  EXPECT_NE(FindInstruction(module.get(), "call-done"), nullptr);
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainTupleCallWithGteConsumersIsWrapped) {
  const absl::string_view hlo_string = R"(
  HloModule m

  body {
    p0 = f32[4] parameter(0)
    add = f32[4] add(p0, p0)
    multiply = f32[4] multiply(p0, p0)
    ROOT tuple = (f32[4], f32[4]) tuple(add, multiply)
  }

  ENTRY main {
    p0 = f32[4] parameter(0)
    call = (f32[4], f32[4]) call(p0), to_apply=body,
      frontend_attributes={_xla_stream_annotation="locality_domain:0",inlineable="false"}
    gte0 = f32[4] get-tuple-element(call), index=0
    gte1 = f32[4] get-tuple-element(call), index=1
    ROOT result = (f32[4], f32[4]) tuple(gte0, gte1)
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  ASSERT_TRUE(mutated);

  int num_async_starts = 0;
  int num_async_dones = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kAsyncStart) {
      ++num_async_starts;
    } else if (instr->opcode() == HloOpcode::kAsyncDone) {
      ++num_async_dones;
    }
  }
  EXPECT_EQ(num_async_starts, 1);
  EXPECT_EQ(num_async_dones, 1);

  const HloInstruction* done = FindInstruction(module.get(), "call-done");
  ASSERT_NE(done, nullptr);
  for (const HloInstruction* async : {done->operand(0), done}) {
    EXPECT_EQ(async->frontend_attributes().map().at(kXlaStreamAnnotationAttr),
              "locality_domain:0");
  }
  for (absl::string_view name : {"gte0", "gte1"}) {
    const HloInstruction* gte = FindInstruction(module.get(), name);
    ASSERT_NE(gte, nullptr);
    EXPECT_EQ(gte->operand(0), done);
  }
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainRequiresExistingCall) {
  const absl::string_view hlo_string = R"(
  HloModule m

  ENTRY main {
    p0 = f32[4] parameter(0)
    ROOT negate = f32[4] negate(p0),
      frontend_attributes={_xla_stream_annotation="locality_domain:1"}
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  EXPECT_THAT(wrapper_pass.Run(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("requires an existing kCall")));
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainRequiresNonInlineableCall) {
  const absl::string_view hlo_string = R"(
  HloModule m

  body {
    p0 = f32[4] parameter(0)
    ROOT negate = f32[4] negate(p0)
  }
  ENTRY main {
    p0 = f32[4] parameter(0)
    ROOT call = f32[4] call(p0), to_apply=body,
      frontend_attributes={_xla_stream_annotation="locality_domain:1"}
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  EXPECT_THAT(wrapper_pass.Run(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("non-inlineable kCall")));
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainRejectsTransitiveCollective) {
  const absl::string_view hlo_string = R"(
  HloModule m

  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT sum = f32[] add(x, y)
  }
  collective {
    p0 = f32[4] parameter(0)
    ROOT ar = f32[4] all-reduce(p0), to_apply=add, replica_groups={}
  }
  body {
    p0 = f32[4] parameter(0)
    ROOT inner = f32[4] call(p0), to_apply=collective
  }
  ENTRY main {
    p0 = f32[4] parameter(0)
    ROOT call = f32[4] call(p0), to_apply=body,
      frontend_attributes={_xla_stream_annotation="locality_domain:1",inlineable="false"}
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  EXPECT_THAT(wrapper_pass.Run(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unsupported communication instruction")));
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainRejectsNestedStreamAnnotation) {
  const absl::string_view hlo_string = R"(
  HloModule m

  body {
    p0 = f32[4] parameter(0)
    ROOT negate = f32[4] negate(p0),
      frontend_attributes={_xla_stream_annotation="2"}
  }
  ENTRY main {
    p0 = f32[4] parameter(0)
    ROOT call = f32[4] call(p0), to_apply=body,
      frontend_attributes={_xla_stream_annotation="locality_domain:1",inlineable="false"}
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  EXPECT_THAT(wrapper_pass.Run(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("nested stream annotation")));
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainRejectsSendRecv) {
  const absl::string_view hlo_string = R"(
  HloModule m, num_partitions=2

  body {
    p0 = f32[4] parameter(0)
    after_all = token[] after-all()
    send = (f32[4], u32[], token[]) send(p0, after_all), channel_id=1
    ROOT done = token[] send-done(send), channel_id=1
  }
  ENTRY main {
    p0 = f32[4] parameter(0)
    ROOT call = token[] call(p0), to_apply=body,
      frontend_attributes={_xla_stream_annotation="locality_domain:1",inlineable="false"}
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  EXPECT_THAT(wrapper_pass.Run(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unsupported communication instruction")));
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainInsideFusionReturnsStatus) {
  const absl::string_view hlo_string = R"(
  HloModule m

  fused {
    p0 = f32[4] parameter(0)
    ROOT negate = f32[4] negate(p0),
      frontend_attributes={_xla_stream_annotation="locality_domain:1"}
  }
  ENTRY main {
    p0 = f32[4] parameter(0)
    ROOT fusion = f32[4] fusion(p0), kind=kLoop, calls=fused
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  EXPECT_THAT(
      wrapper_pass.Run(module.get()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("not allowed inside an async or fusion computation")));
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainInsideAsyncReturnsStatus) {
  const absl::string_view hlo_string = R"(
  HloModule m

  async_body {
    p0 = f32[4] parameter(0)
    ROOT negate = f32[4] negate(p0),
      frontend_attributes={_xla_stream_annotation="locality_domain:1"}
  }
  ENTRY main {
    p0 = f32[4] parameter(0)
    start = ((f32[4]), f32[4]) async-start(p0), calls=async_body
    ROOT done = f32[4] async-done(start)
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  EXPECT_THAT(
      wrapper_pass.Run(module.get()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("not allowed inside an async or fusion computation")));
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainRejectsMalformedAnnotation) {
  struct Case {
    const char* annotation;
    const char* message;
  };
  for (const Case& c : {
           Case{"locality_domain", "expected \"locality_domain:<d>\""},
           Case{"locality_domain:", "expected \"locality_domain:<d>\""},
           Case{"locality_domain:abc", "unsigned decimal integer"},
           Case{"locality_domain:18446744073709551616", "out of range"},
       }) {
    SCOPED_TRACE(c.annotation);
    const std::string hlo_string = absl::StrCat(
        R"(
  HloModule m

  body {
    p0 = f32[4] parameter(0)
    ROOT negate = f32[4] negate(p0)
  }

  ENTRY main {
    p0 = f32[4] parameter(0)
    ROOT call = f32[4] call(p0), to_apply=body,
      frontend_attributes={_xla_stream_annotation=")",
        c.annotation, R"(",inlineable="false"}
  })");
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
    ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

    EXPECT_THAT(
        wrapper_pass.Run(module.get()),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(c.message)));
  }
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainNestedInStreamAnnotationIsRejected) {
  const absl::string_view hlo_string = R"(
  HloModule m

  inner_body {
    p0 = f32[4] parameter(0)
    ROOT negate = f32[4] negate(p0)
  }

  outer_body {
    p0 = f32[4] parameter(0)
    ROOT inner = f32[4] call(p0), to_apply=inner_body,
      frontend_attributes={_xla_stream_annotation="locality_domain:0",inlineable="false"}
  }

  ENTRY main {
    p0 = f32[4] parameter(0)
    ROOT outer = f32[4] call(p0), to_apply=outer_body,
      frontend_attributes={_xla_stream_annotation="1",inlineable="false"}
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  EXPECT_THAT(
      wrapper_pass.Run(module.get()),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("is nested inside a call with stream annotation \"1\"")));
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainRejectsCollectivesGroup) {
  const absl::string_view hlo_string = R"(
  HloModule m

  body {
    p0 = f32[4] parameter(0)
    ROOT negate = f32[4] negate(p0)
  }

  ENTRY main {
    p0 = f32[4] parameter(0)
    ROOT call = f32[4] call(p0), to_apply=body,
      frontend_attributes={_xla_stream_annotation="locality_domain:0",inlineable="false",_collectives_group=""}
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  EXPECT_THAT(
      wrapper_pass.Run(module.get()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("cannot be combined with _collectives_group")));
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainPreWrappedRejectsCollectivesGroup) {
  const absl::string_view hlo_string = R"(
  HloModule m

  body {
    p0 = f32[4] parameter(0)
    ROOT negate = f32[4] negate(p0)
  }

  async_wrapped {
    p0 = f32[4] parameter(0)
    ROOT call = f32[4] call(p0), to_apply=body
  }

  ENTRY main {
    p0 = f32[4] parameter(0)
    start = ((f32[4]), f32[4]) async-start(p0), async_execution_thread="parallel",
      calls=async_wrapped,
      frontend_attributes={_xla_stream_annotation="locality_domain:0",_collectives_group=""}
    ROOT done = f32[4] async-done(start),
      frontend_attributes={_xla_stream_annotation="locality_domain:0",_collectives_group=""}
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  EXPECT_THAT(
      wrapper_pass.Run(module.get()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("cannot be combined with _collectives_group")));
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainPreWrappedRejectsCollectiveInBody) {
  const absl::string_view hlo_string = R"(
  HloModule m

  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT sum = f32[] add(x, y)
  }

  body {
    p0 = f32[4] parameter(0)
    ROOT ar = f32[4] all-reduce(p0), to_apply=add, replica_groups={}
  }

  async_wrapped {
    p0 = f32[4] parameter(0)
    ROOT call = f32[4] call(p0), to_apply=body
  }

  ENTRY main {
    p0 = f32[4] parameter(0)
    start = ((f32[4]), f32[4]) async-start(p0), async_execution_thread="parallel",
      calls=async_wrapped,
      frontend_attributes={_xla_stream_annotation="locality_domain:0"}
    ROOT done = f32[4] async-done(start),
      frontend_attributes={_xla_stream_annotation="locality_domain:0"}
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  EXPECT_THAT(wrapper_pass.Run(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unsupported communication instruction")));
}

TEST_F(ExplicitStreamAnnotationAsyncWrapperTest,
       LocalityDomainRejectsNestedCollectivesGroup) {
  const absl::string_view hlo_string = R"(
  HloModule m

  leaf {
    p0 = f32[4] parameter(0)
    ROOT negate = f32[4] negate(p0)
  }

  body {
    p0 = f32[4] parameter(0)
    ROOT group_call = f32[4] call(p0), to_apply=leaf,
      frontend_attributes={_collectives_group="",inlineable="false"}
  }

  ENTRY main {
    p0 = f32[4] parameter(0)
    ROOT call = f32[4] call(p0), to_apply=body,
      frontend_attributes={_xla_stream_annotation="locality_domain:0",inlineable="false"}
  })";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ExplicitStreamAnnotationAsyncWrapper wrapper_pass;

  EXPECT_THAT(wrapper_pass.Run(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unsupported communication instruction")));
}

}  // namespace
}  // namespace xla::gpu

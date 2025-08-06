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

#include "xla/service/legalize_scheduling_annotations.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/scheduling_annotations_util.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace {

using LegalizeSchedulingAnnotationsTest = HloHardwareIndependentTestBase;
using SchedulingAnnotationPropagationTest = HloHardwareIndependentTestBase;
using RemoveLoopIterationAnnotationTest = HloHardwareIndependentTestBase;
using ::tsl::testing::IsOkAndHolds;

TEST_F(LegalizeSchedulingAnnotationsTest, NonIntegerAnnotation) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="0"}
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="annotation1"}
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="0"}
    ROOT tuple = (f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(c0, agd0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  EXPECT_IS_NOT_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());
}

TEST_F(LegalizeSchedulingAnnotationsTest, MultipleAnnotations) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="0"}
    ags1 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0,1"}
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="0"}
    agd1 = f32[1024,1024]{1,0} all-gather-done(ags1), frontend_attributes={_scheduling_group_id="1"}
    ROOT tuple = (f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}, f32[1024,1024]{1,0}) tuple(c0, agd0, agd1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  EXPECT_IS_NOT_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());
}

TEST_F(LegalizeSchedulingAnnotationsTest, NegativeAnnotation) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="-1"}
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="-1"}
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="-1"}
    ROOT tuple = (f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(c0, agd0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  EXPECT_IS_NOT_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());
}

TEST_F(LegalizeSchedulingAnnotationsTest, CrossComputationAnnotation) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    param = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(param), index=2
  }

  while_body {
    param = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) parameter(0)
    gte0 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=0
    gte1 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=1
    gte2 = pred[] get-tuple-element(param), index=2
    cps1 = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, u32[], u32[]) collective-permute-start(gte1), source_target_pairs={{0,1},{1,2},{2,3},{3,0}}, frontend_attributes={_scheduling_group_id="1"}
    cpd1 = f32[16,64,256]{2,1,0} collective-permute-done(cps1), frontend_attributes={_scheduling_group_id="1"}
    c1 = f32[16,256,256]{2,1,0} convolution(gte0, gte0), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    slice = f32[16,64,256]{2,1,0} slice(c1), slice={[0:16], [0:64], [0:256]}
    add = f32[16,64,256]{2,1,0} add(gte0, slice)
    ROOT tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) tuple(add, cpd1, gte2)
  }

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    p3 = pred[] parameter(3)
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) tuple(p1, p2, p3)
    while = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) while(tuple), condition=while_cond, body=while_body
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="1"}
    gte = f32[16,64,256]{2,1,0} get-tuple-element(while), index=0
    ROOT tuple1 = (f32[16,64,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(gte, c0, agd0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  EXPECT_IS_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());
}

TEST_F(LegalizeSchedulingAnnotationsTest, AnnotationWithGaps) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    // This slice is not annotated.
    slice = f32[16,64,256]{2,1,0} slice(c0), slice={[0:16], [0:64], [0:256]}
    c1 = f32[16,256,256]{2,1,0} convolution(slice, slice), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="1"}
    ROOT tuple = (f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(c0, agd0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  EXPECT_IS_NOT_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());
}

TEST_F(LegalizeSchedulingAnnotationsTest, AnnotationWithGaps2) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    add0 = f32[16,64,256]{2,1,0} add(p1, p2), frontend_attributes={_scheduling_group_id="1"}
    // This negate is not annotated.
    negate = f32[16,64,256]{2,1,0} negate(add0)
    add1 = f32[16,64,256]{2,1,0} add(negate, add0), frontend_attributes={_scheduling_group_id="1"}
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="1"}
    ROOT tuple = (f32[16,64,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(add1, agd0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  EXPECT_IS_NOT_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());
}

TEST_F(LegalizeSchedulingAnnotationsTest, MissingAnnotationInStart) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="0"}
    ROOT tuple = (f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(c0, agd0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  EXPECT_IS_NOT_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());
}

TEST_F(LegalizeSchedulingAnnotationsTest, MoveFusedOpAnnotationToCaller) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test

  fused_computation.1 {
    param0 = bf16[1024,6144]{1,0:T(8,128)(2,1)} parameter(0)
    param1 = bf16[6144,2048]{1,0:T(8,128)(2,1)} parameter(1)
    ROOT convolution = bf16[1024,2048]{1,0:T(8,128)(2,1)} convolution(param0, param1), dim_labels=bf_io->bf, frontend_attributes={_scheduling_group_id="1"}
  }

  ENTRY entry {
    p0 = bf16[1024,6144]{1,0:T(8,128)(2,1)} parameter(0)
    p1 = bf16[6144,2048]{1,0:T(8,128)(2,1)} parameter(1)
    ROOT fusion0 = bf16[1024,2048]{1,0:T(8,128)(2,1)} fusion(p0, p1), kind=kOutput, calls=fused_computation.1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  EXPECT_IS_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());

  HloInstruction* fusion = hlo_module->entry_computation()->root_instruction();
  const auto& attrs = fusion->frontend_attributes().map();
  EXPECT_TRUE(attrs.contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(attrs.at(kXlaSchedulingGroupIdAttr), "1");
}

TEST_F(LegalizeSchedulingAnnotationsTest, FusedOpsWithDifferentAnnotationIds) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test

  fused_computation.1 {
    param0 = bf16[1024,6144]{1,0:T(8,128)(2,1)} parameter(0)
    param1 = bf16[6144,4096]{1,0:T(8,128)(2,1)} parameter(1)
    slice = bf16[6144,2048]{1,0:T(8,128)(2,1)} slice(param1), slice={[0:6144], [0:2048]}, frontend_attributes={_scheduling_group_id="1"}
    ROOT convolution = bf16[1024,2048]{1,0:T(8,128)(2,1)} convolution(param0, slice), dim_labels=bf_io->bf, frontend_attributes={_scheduling_group_id="2"}
  }

  ENTRY entry {
    p0 = bf16[1024,6144]{1,0:T(8,128)(2,1)} parameter(0)
    p1 = bf16[6144,4096]{1,0:T(8,128)(2,1)} parameter(1)
    ROOT fusion0 = bf16[1024,2048]{1,0:T(8,128)(2,1)} fusion(p0, p1), kind=kOutput, calls=fused_computation.1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  EXPECT_IS_NOT_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());
}

TEST_F(LegalizeSchedulingAnnotationsTest, DropAnnotationFromBitcast) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="0"}
    bitcast = f32[16,64,256]{2,1,0} bitcast(p1), frontend_attributes={_scheduling_group_id="0"}
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="0"}
    ROOT tuple = (f32[16,64,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(bitcast, agd0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  config.keep_sync_annotation = [](const HloInstruction* instr) {
    return instr->opcode() != HloOpcode::kBitcast;
  };
  EXPECT_IS_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());
  HloInstruction* bitcast =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  EXPECT_FALSE(
      bitcast->frontend_attributes().map().contains(kXlaSchedulingGroupIdAttr));
}

TEST_F(LegalizeSchedulingAnnotationsTest, DropAnnotationFromTrivialGroup) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="0"}
    bitcast = f32[16,64,256]{2,1,0} bitcast(p1), frontend_attributes={_scheduling_group_id="1"}
    copy = f32[16,64,256]{2,1,0} copy(bitcast), frontend_attributes={_scheduling_group_id="2"}
    bitcast2 = f32[16,64,256]{2,1,0} bitcast(copy), frontend_attributes={_scheduling_group_id="3"}
    bitcast3 = f32[16,64,256]{2,1,0} bitcast(bitcast2), frontend_attributes={_scheduling_group_id="3"}
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="0"}
    ROOT tuple = (f32[16,64,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(bitcast3, agd0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  config.keep_trivial_sync_annotation = [](const HloInstruction* instr) {
    return instr->opcode() != HloOpcode::kBitcast;
  };
  EXPECT_IS_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());
  HloInstruction* bitcast3 =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  HloInstruction* bitcast2 = bitcast3->mutable_operand(0);
  HloInstruction* copy = bitcast2->mutable_operand(0);
  HloInstruction* bitcast = copy->mutable_operand(0);
  EXPECT_TRUE(
      copy->frontend_attributes().map().contains(kXlaSchedulingGroupIdAttr));
  EXPECT_FALSE(
      bitcast->frontend_attributes().map().contains(kXlaSchedulingGroupIdAttr));
  EXPECT_TRUE(bitcast2->frontend_attributes().map().contains(
      kXlaSchedulingGroupIdAttr));
  EXPECT_TRUE(bitcast3->frontend_attributes().map().contains(
      kXlaSchedulingGroupIdAttr));
}

TEST_F(LegalizeSchedulingAnnotationsTest,
       DropRepeatedAnnotationFromDifferentComputations) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    param = (f32[16,64,256]{2,1,0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(param), index=1
  }

  while_body {
    param = (f32[16,64,256]{2,1,0}, pred[]) parameter(0)
    gte0 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=0
    gte1 = pred[] get-tuple-element(param), index=1
    c1 = f32[16,256,256]{2,1,0} convolution(gte0, gte0), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    slice = f32[16,64,256]{2,1,0} slice(c1), slice={[0:16], [0:64], [0:256]}
    add = f32[16,64,256]{2,1,0} add(gte0, slice)
    ROOT tuple = (f32[16,64,256]{2,1,0}, pred[]) tuple(add, gte1)
  }

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = pred[] parameter(2)
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p1), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    tuple = (f32[16,64,256]{2,1,0}, pred[]) tuple(p1, p2)
    while = (f32[16,64,256]{2,1,0}, pred[]) while(tuple), condition=while_cond, body=while_body
    gte = f32[16,64,256]{2,1,0} get-tuple-element(while), index=0
    ROOT tuple1 = (f32[16,64,256]{2,1,0}, f32[16,256,256]{2,1,0}) tuple(gte, c0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  config.keep_trivial_sync_annotation = HloPredicateFalse;
  EXPECT_IS_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());
  std::vector<HloInstruction*> convs =
      FindInstructions(hlo_module.get(), HloOpcode::kConvolution);
  EXPECT_THAT(convs, ::testing::Each(::testing::Not(
                         xla::testing::opcode_matchers::FrontendAttribute(
                             "_scheduling_group_id", "1"))));
}

TEST_F(LegalizeSchedulingAnnotationsTest, OpsWithControlDependencies) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

ENTRY entry {
    p0 = f32[16,64,256]{2,1,0} parameter(0)
    p2 = f32[512,2048,2048]{2,1,0} parameter(2)
    after-all = token[] after-all()
    send = (f32[512,2048,2048]{2,1,0}, u32[], token[]) send(p2, after-all), channel_id=1
    send-done = token[] send-done(send), channel_id=1
    recv = (f32[512,2048,2048]{2,1,0}, u32[], token[]) recv(after-all), channel_id=2
    recv-done = (f32[512,2048,2048]{2,1,0}, token[]) recv-done(recv), channel_id=2, control-predecessors={send-done}
    get-tuple-element = f32[512,2048,2048]{2,1,0} get-tuple-element(recv-done), index=0
    slice = f32[16,64,256]{2,1,0} slice(get-tuple-element), slice={[0:16], [0:64], [0:256]}
    c0 = f32[16,256,256]{2,1,0} convolution(p0, slice), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
    c1 = f32[16,256,256]{2,1,0} convolution(p0, slice), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
    p1 = f32[128,2048,2048]{2,1,0} parameter(1)
    after-all.1 = token[] after-all()
    send.1 = (f32[128,2048,2048]{2,1,0}, u32[], token[]) send(p1, after-all.1), channel_id=3, frontend_attributes={_scheduling_group_id="0"}
    send-done.1 = token[] send-done(send.1), channel_id=3, frontend_attributes={_scheduling_group_id="0"}
    recv.1 = (f32[128,2048,2048]{2,1,0}, u32[], token[]) recv(after-all.1), channel_id=4, frontend_attributes={_scheduling_group_id="0"}
    recv-done.1 = (f32[128,2048,2048]{2,1,0}, token[]) recv-done(recv.1), channel_id=4, frontend_attributes={_scheduling_group_id="0"}, control-predecessors={send-done.1}
    get-tuple-element.1 = f32[128,2048,2048]{2,1,0} get-tuple-element(recv-done.1), index=0
    after-all.2 = token[] after-all()
    send.2 = (f32[128,2048,2048]{2,1,0}, u32[], token[]) send(get-tuple-element.1, after-all.2), channel_id=5
    send-done.2 = token[] send-done(send.2), channel_id=5
    recv.2 = (f32[128,2048,2048]{2,1,0}, u32[], token[]) recv(after-all.2), channel_id=6
    recv-done.2 = (f32[128,2048,2048]{2,1,0}, token[]) recv-done(recv.2), channel_id=6, control-predecessors={send-done.2}
    get-tuple-element.2 = f32[128,2048,2048]{2,1,0} get-tuple-element(recv-done.2), index=0
    ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, get-tuple-element.1, get-tuple-element.2)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  EXPECT_IS_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());
}

TEST_F(SchedulingAnnotationPropagationTest, NoOpSchedulingGroup) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16]{0} parameter(0)
  p1 = f32[16]{0} parameter(1)
  ROOT a0 = f32[16]{0} add(p0, p1), frontend_attributes={_scheduling_group_id="noop"}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  EXPECT_IS_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());
  VLOG(1) << "module after: " << hlo_module->ToString();
  HloInstruction* add =
      hlo_module->GetComputationWithName("entry")->GetInstructionWithName("a0");
  CHECK(add != nullptr);
  const auto& attrs = add->frontend_attributes().map();
  EXPECT_FALSE(attrs.contains(kXlaSchedulingGroupIdAttr));
}

TEST_F(LegalizeSchedulingAnnotationsTest, ProgagateAnnotationToGap) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    // This slice is not annotated.
    slice = f32[16,64,256]{2,1,0} slice(c0), slice={[0:16], [0:64], [0:256]}
    c1 = f32[16,256,256]{2,1,0} convolution(slice, slice), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="1"}
    ROOT tuple = (f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(c0, agd0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  config.propagate_annotation = true;
  EXPECT_THAT(LegalizeSchedulingAnnotations(config).Run(hlo_module.get()),
              IsOkAndHolds(true));
  VLOG(1) << "module after: " << hlo_module->ToString();
  HloInstruction* slice =
      hlo_module->entry_computation()->GetInstructionWithName("slice");
  CHECK(slice != nullptr);
  const auto& attrs = slice->frontend_attributes().map();
  EXPECT_TRUE(attrs.contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(attrs.at(kXlaSchedulingGroupIdAttr), "1");
}

TEST_F(SchedulingAnnotationPropagationTest, NothingToPropagate) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    slice = f32[16,64,256]{2,1,0} slice(c0), slice={[0:16], [0:64], [0:256]}, frontend_attributes={_scheduling_group_id="1"}
    c1 = f32[16,256,256]{2,1,0} convolution(slice, slice), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="1"}
    ROOT tuple = (f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(c0, agd0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  config.propagate_annotation = true;
  EXPECT_THAT(LegalizeSchedulingAnnotations(config).Run(hlo_module.get()),
              IsOkAndHolds(false));
}

TEST_F(SchedulingAnnotationPropagationTest, NoDataDependentGap) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
    slice = f32[16,64,256]{2,1,0} slice(c0), slice={[0:16], [0:64], [0:256]}
    c1 = f32[16,256,256]{2,1,0} convolution(slice, slice), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="1"}
    ROOT tuple = (f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(c0, agd0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  config.propagate_annotation = true;
  EXPECT_THAT(LegalizeSchedulingAnnotations(config).Run(hlo_module.get()),
              IsOkAndHolds(false));
}

TEST_F(SchedulingAnnotationPropagationTest, GapDueToControlDependency) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, control-predecessors={ags0}
    slice = f32[16,64,256]{2,1,0} slice(c0), slice={[0:16], [0:64], [0:256]}
    c1 = f32[16,256,256]{2,1,0} convolution(slice, slice), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="1"}, control-predecessors={c0}
    ROOT tuple = (f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(c0, agd0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  config.propagate_annotation = true;
  EXPECT_THAT(LegalizeSchedulingAnnotations(config).Run(hlo_module.get()),
              IsOkAndHolds(true));
  VLOG(1) << "module after: " << hlo_module->ToString();
  HloInstruction* c0 =
      hlo_module->entry_computation()->GetInstructionWithName("c0");
  CHECK(c0 != nullptr);
  const auto& attrs = c0->frontend_attributes().map();
  EXPECT_TRUE(attrs.contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(attrs.at(kXlaSchedulingGroupIdAttr), "1");
  HloInstruction* slice =
      hlo_module->entry_computation()->GetInstructionWithName("slice");
  CHECK(slice != nullptr);
  EXPECT_FALSE(
      slice->frontend_attributes().map().contains(kXlaSchedulingGroupIdAttr));
  HloInstruction* c1 =
      hlo_module->entry_computation()->GetInstructionWithName("c1");
  CHECK(c1 != nullptr);
  EXPECT_FALSE(
      c1->frontend_attributes().map().contains(kXlaSchedulingGroupIdAttr));
}

TEST_F(SchedulingAnnotationPropagationTest, GapDueToControlDependency2) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, control-predecessors={ags0}
    slice = f32[16,64,256]{2,1,0} slice(c0), slice={[0:16], [0:64], [0:256]}
    c1 = f32[16,256,256]{2,1,0} convolution(slice, slice), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="1"}, control-predecessors={c1}
    ROOT tuple = (f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(c0, agd0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  config.propagate_annotation = true;
  EXPECT_THAT(LegalizeSchedulingAnnotations(config).Run(hlo_module.get()),
              IsOkAndHolds(true));
  VLOG(1) << "module after: " << hlo_module->ToString();
  HloInstruction* slice =
      hlo_module->entry_computation()->GetInstructionWithName("slice");
  CHECK(slice != nullptr);
  const auto& attrs = slice->frontend_attributes().map();
  EXPECT_TRUE(attrs.contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(attrs.at(kXlaSchedulingGroupIdAttr), "1");
  HloInstruction* c0 =
      hlo_module->entry_computation()->GetInstructionWithName("c0");
  CHECK(c0 != nullptr);
  const auto& attrs_c0 = c0->frontend_attributes().map();
  EXPECT_TRUE(attrs_c0.contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(attrs_c0.at(kXlaSchedulingGroupIdAttr), "1");
  HloInstruction* c1 =
      hlo_module->entry_computation()->GetInstructionWithName("c1");
  CHECK(c1 != nullptr);
  const auto& attrs_c1 = c1->frontend_attributes().map();
  EXPECT_TRUE(attrs_c1.contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(attrs_c1.at(kXlaSchedulingGroupIdAttr), "1");
}

TEST_F(SchedulingAnnotationPropagationTest, TwoGroups) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="2"}
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="3"}
    slice = f32[16,64,256]{2,1,0} slice(c0), slice={[0:16], [0:64], [0:256]}
    c1 = f32[16,256,256]{2,1,0} convolution(slice, slice), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="3"}
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="2"}
    ROOT tuple = (f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(c0, agd0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  config.propagate_annotation = true;
  EXPECT_THAT(LegalizeSchedulingAnnotations(config).Run(hlo_module.get()),
              IsOkAndHolds(true));
  VLOG(1) << "module after: " << hlo_module->ToString();
  HloInstruction* slice =
      hlo_module->entry_computation()->GetInstructionWithName("slice");
  CHECK(slice != nullptr);
  const auto& attrs = slice->frontend_attributes().map();
  EXPECT_TRUE(attrs.contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(attrs.at(kXlaSchedulingGroupIdAttr), "3");
}

TEST_F(SchedulingAnnotationPropagationTest, CrossComputationAnnotation) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    param = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(param), index=2
  }

  while_body {
    param = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) parameter(0)
    gte0 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=0
    gte1 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=1
    gte2 = pred[] get-tuple-element(param), index=2
    cps1 = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, u32[], u32[]) collective-permute-start(gte1), source_target_pairs={{0,1},{1,2},{2,3},{3,0}}, frontend_attributes={_scheduling_group_id="1"}
    cpd1 = f32[16,64,256]{2,1,0} collective-permute-done(cps1), frontend_attributes={_scheduling_group_id="1"}
    c1 = f32[16,256,256]{2,1,0} convolution(gte0, gte0), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    slice = f32[16,64,256]{2,1,0} slice(c1), slice={[0:16], [0:64], [0:256]}
    add = f32[16,64,256]{2,1,0} add(gte0, slice), frontend_attributes={_scheduling_group_id="1"}
    ROOT tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) tuple(add, cpd1, gte2)
  }

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    p3 = pred[] parameter(3)
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) tuple(p1, p2, p3)
    while = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) while(tuple), condition=while_cond, body=while_body
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="1"}
    gte = f32[16,64,256]{2,1,0} get-tuple-element(while), index=0
    ROOT tuple1 = (f32[16,64,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(gte, c0, agd0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  config.propagate_annotation = true;
  EXPECT_THAT(LegalizeSchedulingAnnotations(config).Run(hlo_module.get()),
              IsOkAndHolds(true));
  VLOG(1) << "module after: " << hlo_module->ToString();
  HloInstruction* slice = hlo_module->GetComputationWithName("while_body")
                              ->GetInstructionWithName("slice");
  CHECK(slice != nullptr);
  const auto& attrs = slice->frontend_attributes().map();
  EXPECT_TRUE(attrs.contains(kXlaSchedulingGroupIdAttr));
  EXPECT_EQ(attrs.at(kXlaSchedulingGroupIdAttr), "1");
}

TEST_F(SchedulingAnnotationPropagationTest, ConflictingAnnotationGroups) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="2"}
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="3"}
    slice = f32[16,64,256]{2,1,0} slice(c0), slice={[0:16], [0:64], [0:256]}, control-predecessors={ags0}
    c1 = f32[16,256,256]{2,1,0} convolution(slice, slice), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="3"}
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="2"}, control-predecessors={slice}
    ROOT tuple = (f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(c0, agd0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  config.propagate_annotation = true;
  auto result = LegalizeSchedulingAnnotations(config).Run(hlo_module.get());
  EXPECT_IS_NOT_OK(result);
  VLOG(1) << "module after: " << hlo_module->ToString();
  std::string error_message = std::string(result.status().message());
  VLOG(1) << "error message: " << error_message;
  EXPECT_TRUE(
      absl::StrContains(error_message, "it has an existing annotation"));
}

TEST_F(SchedulingAnnotationPropagationTest, ConflictingAnnotationGroups2) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16]{0} parameter(0)
  p1 = f32[16]{0} parameter(1)
  a0 = f32[16]{0} add(p0, p1), frontend_attributes={_scheduling_group_id="1"}
  a1 = f32[16]{0} add(p0, p1), frontend_attributes={_scheduling_group_id="2"}
  a2 = f32[16]{0} add(a0, a1)
  a3 = f32[16]{0} add(p0, a2), frontend_attributes={_scheduling_group_id="1"}
  a4 = f32[16]{0} add(p1, a2), frontend_attributes={_scheduling_group_id="2"}
  ROOT tuple = (f32[16]{0}, f32[16]{0}) tuple(a3, a4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  config.propagate_annotation = true;
  auto result = LegalizeSchedulingAnnotations(config).Run(hlo_module.get());
  EXPECT_IS_NOT_OK(result);
  VLOG(1) << "module after: " << hlo_module->ToString();
  std::string error_message = std::string(result.status().message());
  VLOG(1) << "error message: " << error_message;
  EXPECT_TRUE(
      absl::StrContains(error_message, "it has an existing annotation"));
}

TEST_F(RemoveLoopIterationAnnotationTest, CrossComputationAnnotation) {
  constexpr absl::string_view hlo_string = R"(
 HloModule module, entry_computation_layout={(bf16[5,8,128]{2,1,0}, bf16[5,1,2,128]{3,2,1,0})->bf16[5,8,128]{2,1,0}}, replica_count=4, num_partitions=2

%while_body.1 (param.2: (s32[], bf16[5,8,128], bf16[5,1,2,128], bf16[1,8,128], s32[])) -> (s32[], bf16[5,8,128], bf16[5,1,2,128], bf16[1,8,128], s32[]) {
  %c3.2 = s32[] constant(3)
  %param.2 = (s32[], bf16[5,8,128]{2,1,0}, bf16[5,1,2,128]{3,2,1,0}, bf16[1,8,128]{2,1,0}, s32[]) parameter(0)
  %slice_input.2 = bf16[5,1,2,128]{3,2,1,0} get-tuple-element(%param.2), index=2
  %loop_index.2 = s32[] get-tuple-element(%param.2), index=0
  %three_minus_loop_index.2 = s32[] subtract(%c3.2, %loop_index.2)
  %c0.2 = s32[] constant(0)
  %dynamic_slice.2 = bf16[1,1,2,128]{3,2,1,0} dynamic-slice(%slice_input.2, %three_minus_loop_index.2, %c0.2, %c0.2, %c0.2), dynamic_slice_sizes={1,1,2,128}
  %dynamic_slice_reshape.2 = bf16[1,2,128]{2,1,0} reshape(%dynamic_slice.2)
  %add.2 = bf16[1,2,128]{2,1,0} add(%dynamic_slice_reshape.2, %dynamic_slice_reshape.2), control-predecessors={%c3.2}
  %c1.2 = s32[] constant(1)
  %next_loop_index.1 = s32[] add(%loop_index.2, %c1.2)
  %partial_output.1 = bf16[5,8,128]{2,1,0} get-tuple-element(%param.2), index=1
  %get-tuple-element.1 = bf16[1,8,128]{2,1,0} get-tuple-element(%param.2), index=3
  %updated_partial_output.1 = bf16[5,8,128]{2,1,0} dynamic-update-slice(%partial_output.1, %get-tuple-element.1, %three_minus_loop_index.2, %c0.2, %c0.2)
  %c3.3 = s32[] constant(3)
  %get-tuple-element = s32[] get-tuple-element(%param.2), index=4
  %three_minus_loop_index.3 = s32[] subtract(%c3.3, %get-tuple-element)
  %c0.3 = s32[] constant(0)
  %dynamic_slice.3 = bf16[1,1,2,128]{3,2,1,0} dynamic-slice(%slice_input.2, %three_minus_loop_index.3, %c0.3, %c0.3, %c0.3), dynamic_slice_sizes={1,1,2,128}
  %dynamic_slice_reshape.3 = bf16[1,2,128]{2,1,0} reshape(%dynamic_slice.3)
  %add.3 = bf16[1,2,128]{2,1,0} add(%dynamic_slice_reshape.3, %dynamic_slice_reshape.3), control-predecessors={%c3.3}
  %all_gather.2 = bf16[1,8,128]{2,1,0} all-gather(%add.3), replica_groups={}, dimensions={1}, frontend_attributes={_scheduling_group_id="123:-1"}
  %constant.1 = s32[] constant(1)
  %add.4 = s32[] add(%next_loop_index.1, %constant.1)
  ROOT %tuple.2 = (s32[], bf16[5,8,128]{2,1,0}, bf16[5,1,2,128]{3,2,1,0}, bf16[1,8,128]{2,1,0}, s32[]) tuple(%next_loop_index.1, %updated_partial_output.1, %slice_input.2, %all_gather.2, %add.4), control-predecessors={%add.2}
}

%while_cond.1 (cond_param: (s32[], bf16[5,8,128], bf16[5,1,2,128], bf16[1,8,128], s32[])) -> pred[] {
  %cond_param = (s32[], bf16[5,8,128]{2,1,0}, bf16[5,1,2,128]{3,2,1,0}, bf16[1,8,128]{2,1,0}, s32[]) parameter(0)
  %get-tuple-element.2 = s32[] get-tuple-element(%cond_param), index=0
  %constant.2 = s32[] constant(3)
  ROOT %compare = pred[] compare(%get-tuple-element.2, %constant.2), direction=LT
}

ENTRY %entry (p0: bf16[5,8,128], p1: bf16[5,1,2,128]) -> bf16[5,8,128] {
  %c3.1 = s32[] constant(3)
  %c1.1 = s32[] constant(1)
  %p0 = bf16[5,8,128]{2,1,0} parameter(0)
  %p1 = bf16[5,1,2,128]{3,2,1,0} parameter(1)
  %tuple.1 = (s32[], bf16[5,8,128]{2,1,0}, bf16[5,1,2,128]{3,2,1,0}) tuple(%c1.1, %p0, %p1)
  %slice_input.1 = bf16[5,1,2,128]{3,2,1,0} get-tuple-element(%tuple.1), index=2
  %three_minus_loop_index.1 = s32[] subtract(%c3.1, %c1.1)
  %c0.1 = s32[] constant(0)
  %dynamic_slice.1 = bf16[1,1,2,128]{3,2,1,0} dynamic-slice(%slice_input.1, %three_minus_loop_index.1, %c0.1, %c0.1, %c0.1), dynamic_slice_sizes={1,1,2,128}
  %dynamic_slice_reshape.1 = bf16[1,2,128]{2,1,0} reshape(%dynamic_slice.1)
  %add.1 = bf16[1,2,128]{2,1,0} add(%dynamic_slice_reshape.1, %dynamic_slice_reshape.1), control-predecessors={%c3.1}
  %all_gather.1 = bf16[1,8,128]{2,1,0} all-gather(%add.1), replica_groups={}, dimensions={1}, frontend_attributes={_scheduling_group_id="124"}
  %constant = s32[] constant(2)
  %tuple.3 = (s32[], bf16[5,8,128]{2,1,0}, bf16[5,1,2,128]{3,2,1,0}, bf16[1,8,128]{2,1,0}, s32[]) tuple(%c1.1, %p0, %p1, %all_gather.1, %constant), control-predecessors={%add.1}
  %while.1 = (s32[], bf16[5,8,128]{2,1,0}, bf16[5,1,2,128]{3,2,1,0}, bf16[1,8,128]{2,1,0}, s32[]) while(%tuple.3), condition=%while_cond.1, body=%while_body.1
  %loop_index.3 = s32[] get-tuple-element(%while.1), index=0
  %c1.3 = s32[] constant(1)
  %next_loop_index.2 = s32[] add(%loop_index.3, %c1.3)
  %partial_output.2 = bf16[5,8,128]{2,1,0} get-tuple-element(%while.1), index=1
  %get-tuple-element.3 = bf16[1,8,128]{2,1,0} get-tuple-element(%while.1), index=3
  %c3.4 = s32[] constant(3)
  %three_minus_loop_index.4 = s32[] subtract(%c3.4, %loop_index.3)
  %c0.4 = s32[] constant(0)
  %updated_partial_output.2 = bf16[5,8,128]{2,1,0} dynamic-update-slice(%partial_output.2, %get-tuple-element.3, %three_minus_loop_index.4, %c0.4, %c0.4)
  %slice_input.4 = bf16[5,1,2,128]{3,2,1,0} get-tuple-element(%while.1), index=2
  %tuple.4 = (s32[], bf16[5,8,128]{2,1,0}, bf16[5,1,2,128]{3,2,1,0}) tuple(%next_loop_index.2, %updated_partial_output.2, %slice_input.4)
  ROOT %gte = bf16[5,8,128]{2,1,0} get-tuple-element(%tuple.4), index=1
  %dynamic_slice.4 = bf16[1,1,2,128]{3,2,1,0} dynamic-slice(%slice_input.4, %three_minus_loop_index.4, %c0.4, %c0.4, %c0.4), dynamic_slice_sizes={1,1,2,128}
  %dynamic_slice_reshape.4 = bf16[1,2,128]{2,1,0} reshape(%dynamic_slice.4)
  %add.5 = bf16[1,2,128]{2,1,0} add(%dynamic_slice_reshape.4, %dynamic_slice_reshape.4), control-predecessors={%c3.4}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  config.remove_loop_iteration_annotation_only = true;
  EXPECT_IS_OK(
      LegalizeSchedulingAnnotations(config).Run(hlo_module.get()).status());
  HloInstruction* all_gather =
      FindInstruction(hlo_module.get(), "all_gather.2");
  auto annotation = GetSchedulingAnnotation(all_gather).value();
  EXPECT_TRUE(annotation);
  EXPECT_TRUE(annotation->group_id);
  EXPECT_EQ(annotation->group_id.value(), 123);
  EXPECT_FALSE(annotation->iteration_id);
}

TEST_F(SchedulingAnnotationPropagationTest, VerifyCycle) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16]{0} parameter(0)
  p1 = f32[16]{0} parameter(1)
  a0 = f32[16]{0} add(p0, p1), frontend_attributes={_scheduling_group_id="1"}
  a1 = f32[16]{0} add(a0, p1), frontend_attributes={_scheduling_group_id="2"}
  a2 = f32[16]{0} add(a1, p1), frontend_attributes={_scheduling_group_id="3"}
  a3 = f32[16]{0} add(a2, p1), frontend_attributes={_scheduling_group_id="2"}
  a4 = f32[16]{0} add(p1, a3), frontend_attributes={_scheduling_group_id="1"}
  ROOT tuple = (f32[16]{0}, f32[16]{0}) tuple(a3, a4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LegalizeSchedulingAnnotations::Config config;
  config.run_verification = true;

  auto result = LegalizeSchedulingAnnotations(config).Run(hlo_module.get());
  EXPECT_IS_NOT_OK(result);
  VLOG(1) << "module after: " << hlo_module->ToString();
  std::string error_message = std::string(result.status().message());
  VLOG(1) << "error message: " << error_message;
  EXPECT_TRUE(absl::StrContains(error_message,
                                "Detected scheduling group annotation "
                                "cycle"));
}

}  // namespace
}  // namespace xla

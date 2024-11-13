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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/test_helpers.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using LegalizeSchedulingAnnotationsTest = HloHardwareIndependentTestBase;

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

  EXPECT_IS_NOT_OK(
      LegalizeSchedulingAnnotations().Run(hlo_module.get()).status());
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

  EXPECT_IS_NOT_OK(
      LegalizeSchedulingAnnotations().Run(hlo_module.get()).status());
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

  EXPECT_IS_NOT_OK(
      LegalizeSchedulingAnnotations().Run(hlo_module.get()).status());
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
    c0 = f32[16,256,256]{2,1,0} convolution(gte0, gte1), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    slice = f32[16,64,256]{2,1,0} slice(c0), slice={[0:16], [0:64], [0:256]}
    add = f32[16,64,256]{2,1,0} add(gte0, slice)
    ROOT tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) tuple(add, gte1, gte2)
  }

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    p3 = pred[] parameter(3)
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) tuple(p1, p2, p3)
    while = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) while(tuple), condition=while_cond, body=while_body
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="1"}
    gte = f32[16,64,256]{2,1,0} get-tuple-element(while), index=0
    ROOT tuple1 = (f32[16,64,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(gte, agd0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_IS_NOT_OK(
      LegalizeSchedulingAnnotations().Run(hlo_module.get()).status());
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

  EXPECT_IS_NOT_OK(
      LegalizeSchedulingAnnotations().Run(hlo_module.get()).status());
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

  EXPECT_IS_NOT_OK(
      LegalizeSchedulingAnnotations().Run(hlo_module.get()).status());
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

  EXPECT_IS_NOT_OK(
      LegalizeSchedulingAnnotations().Run(hlo_module.get()).status());
}
}  // namespace
}  // namespace xla

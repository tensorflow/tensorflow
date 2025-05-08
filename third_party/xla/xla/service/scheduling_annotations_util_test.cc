/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/scheduling_annotations_util.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::TestParamInfo;
using ::testing::Values;
using ::testing::WithParamInterface;

using SchedulingAnnotationsUtilTest = HloHardwareIndependentTestBase;

struct SchedulingAnnotationsUtilTestParameter {
  explicit SchedulingAnnotationsUtilTestParameter(
      Annotation annotation, absl::string_view annotation_str)
      : annotation(annotation), annotation_str(annotation_str) {}

  Annotation annotation;
  absl::string_view annotation_str;
};

class ParameterizedSchedulingAnnotationsUtilTest
    : public HloHardwareIndependentTestBase,
      public WithParamInterface<SchedulingAnnotationsUtilTestParameter> {};

TEST_F(SchedulingAnnotationsUtilTest, HasSchedulingAnnotationTest) {
  const std::string hlo_string = R"(
  HloModule Module

  ENTRY entry {
    p0 = f32[32,32]{1,0} parameter(0)
    ROOT p1 = f32[32,32]{1,0} copy(p0), frontend_attributes={_scheduling_group_id="0"}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* p1 = FindInstruction(module.get(), "p1");
  EXPECT_TRUE(HasSchedulingAnnotation(p1));
}

TEST_P(ParameterizedSchedulingAnnotationsUtilTest,
       GetSchedulingAnnotationTest) {
  const std::string hlo_string = R"(
  HloModule Module

  ENTRY entry {
    p0 = f32[32,32]{1,0} parameter(0)
    ROOT p1 = f32[32,32]{1,0} copy(p0), frontend_attributes={_scheduling_group_id="<scheduling_annotation>"}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(absl::StrReplaceAll(
                              hlo_string, {{"<scheduling_annotation>",
                                            GetParam().annotation_str}})));
  HloInstruction* p1 = FindInstruction(module.get(), "p1");
  TF_ASSERT_OK_AND_ASSIGN(std::optional<Annotation> annotation,
                          GetSchedulingAnnotation(p1));
  EXPECT_TRUE(annotation.has_value());
  EXPECT_EQ(annotation, GetParam().annotation);
}

TEST_P(ParameterizedSchedulingAnnotationsUtilTest,
       SetSchedulingAnnotationTest_NoAnnotation) {
  const std::string hlo_string = R"(
  HloModule Module

  ENTRY entry {
  p0 = f32[32,32]{1,0} parameter(0)
  ROOT p1 = f32[32,32]{1,0} copy(p0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* p1 = FindInstruction(module.get(), "p1");
  TF_ASSERT_OK(SetSchedulingAnnotation(p1, GetParam().annotation));
  TF_ASSERT_OK_AND_ASSIGN(std::optional<Annotation> annotation,
                          GetSchedulingAnnotation(p1));
  EXPECT_TRUE(annotation.has_value());
  EXPECT_EQ(annotation->ToString(), GetParam().annotation_str);
}

TEST_P(ParameterizedSchedulingAnnotationsUtilTest,
       SetSchedulingAnnotationTest_WithAnnotation) {
  const std::string hlo_string = R"(
  HloModule Module

  ENTRY entry {
    p0 = f32[32,32]{1,0} parameter(0)
    ROOT p1 = f32[32,32]{1,0} copy(p0), frontend_attributes={_scheduling_group_id="<scheduling_annotation>"}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(absl::StrReplaceAll(
                              hlo_string, {{"<scheduling_annotation>",
                                            GetParam().annotation_str}})));
  HloInstruction* p1 = FindInstruction(module.get(), "p1");
  TF_ASSERT_OK(SetSchedulingAnnotation(p1, "987"));
  TF_ASSERT_OK_AND_ASSIGN(std::optional<Annotation> annotation,
                          GetSchedulingAnnotation(p1));
  EXPECT_TRUE(annotation.has_value());
  EXPECT_EQ(annotation->ToString(), "987");
}

TEST_P(ParameterizedSchedulingAnnotationsUtilTest,
       RemoveSchedulingAnnotationTest) {
  const std::string hlo_string = R"(
  HloModule Module

  ENTRY entry {
  p0 = f32[32,32]{1,0} parameter(0)
    ROOT p1 = f32[32,32]{1,0} copy(p0), frontend_attributes={_scheduling_group_id="<scheduling_annotation>"}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(absl::StrReplaceAll(
                              hlo_string, {{"<scheduling_annotation>",
                                            GetParam().annotation_str}})));
  HloInstruction* p1 = FindInstruction(module.get(), "p1");
  EXPECT_TRUE(HasSchedulingAnnotation(p1));
  ASSERT_TRUE(RemoveSchedulingAnnotation(p1));
  EXPECT_FALSE(HasSchedulingAnnotation(p1));
}

INSTANTIATE_TEST_SUITE_P(
    SchedulingAnnotationsUtilTestInstance,
    ParameterizedSchedulingAnnotationsUtilTest,
    Values(SchedulingAnnotationsUtilTestParameter(
               Annotation(/*group_id=*/std::nullopt,
                          /*iteration_id=*/std::nullopt),
               /*annotation_str=*/""),
           SchedulingAnnotationsUtilTestParameter(
               Annotation(/*group_id=*/123, /*iteration_id=*/std::nullopt),
               /*annotation_str=*/"123"),
           SchedulingAnnotationsUtilTestParameter(
               Annotation(
                   /*group_id=*/std::nullopt,
                   /*iteration_id=*/AnnotationIterationId{.iteration_id = -1}),
               /*annotation_str=*/":-1"),
           SchedulingAnnotationsUtilTestParameter(
               Annotation(
                   /*group_id=*/123,
                   /*iteration_id=*/AnnotationIterationId{.iteration_id = -1}),
               /*annotation_str=*/"123:-1")),
    [](const TestParamInfo<SchedulingAnnotationsUtilTestParameter>& info) {
      return absl::StrCat("annotation_",
                          absl::StrReplaceAll(info.param.annotation_str,
                                              {{"-", "minus"}, {":", "_"}}));
    });

}  // namespace
}  // namespace xla

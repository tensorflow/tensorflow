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

#include "xla/service/gpu/transforms/scheduling_instruction_annotator.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using SchedulingInstructionAnnotatorTest = HloTestBase;

TEST_F(SchedulingInstructionAnnotatorTest,
       AnnotatesAllInstructionsWithTheirRespectiveNames) {
  constexpr absl::string_view kHloString = R"(
    HloModule module, is_scheduled=true

    ENTRY entry {
      p0 = f32[1] parameter(0)
      p1 = f32[1] parameter(1)
      add0 = f32[1] add(p0,p1)
      ROOT exp0 = f32[1] exponential(add0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  SchedulingInstructionAnnotator pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));

  ASSERT_TRUE(changed);
  for (const auto* comp : module->computations()) {
    for (const auto* instruction : comp->instructions()) {
      EXPECT_EQ(instruction->name(), instruction->metadata().scheduling_name());
    }
  }
  constexpr absl::string_view kExpected = R"(
// CHECK:       %[[P0:.+]] = {{.*}} parameter(0)
// CHECK-SAME:  scheduling_name="[[P0]]"
// CHECK:       %[[P1:.+]] = {{.*}} parameter(1)
// CHECK-SAME:  scheduling_name="[[P1]]"
// CHECK:       %[[ADD0:.+]] = {{.*}} add(%[[P0]], %[[P1]])
// CHECK-SAME:  scheduling_name="[[ADD0]]"
// CHECK: ROOT  %[[EXP0:.+]] = {{.*}} exponential(%[[ADD0]])
// CHECK-SAME:  scheduling_name="[[EXP0]]"
  )";
  TF_ASSERT_OK_AND_ASSIGN(
      bool filecheck_matches,
      RunFileCheck(
          module->ToString(HloPrintOptions().set_print_operand_shape(false)),
          kExpected));
  EXPECT_TRUE(filecheck_matches);
}

TEST_F(SchedulingInstructionAnnotatorTest, SkipsAnnotatingConstants) {
  constexpr absl::string_view kHloString = R"(
    HloModule module, is_scheduled=true

    ENTRY entry {
      p0 = f32[1] parameter(0)
      c1 = f32[1] constant(42)
      ROOT add0 = f32[1] add(p0, c1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  SchedulingInstructionAnnotator pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));

  ASSERT_TRUE(changed);
  constexpr absl::string_view kExpected = R"(
// CHECK:       %[[P0:.+]] = {{.*}} parameter(0)
// CHECK-SAME:  scheduling_name="[[P0]]"
// CHECK-NEXT:  %[[C1:.+]] = f32[1]
// CHECK-NOT:   scheduling_name
// CHECK-SAME:  constant({42})
// CHECK:       %[[ADD0:.+]] = {{.*}} add(%[[P0]], %[[C1]])
// CHECK-SAME:  scheduling_name="[[ADD0]]"
  )";
  TF_ASSERT_OK_AND_ASSIGN(
      bool filecheck_matches,
      RunFileCheck(
          module->ToString(HloPrintOptions().set_print_operand_shape(false)),
          kExpected));
  EXPECT_TRUE(filecheck_matches);
}

TEST_F(SchedulingInstructionAnnotatorTest,
       DoesNotAnnotateAllInstructionsWithTheirRespectiveNames) {
  constexpr absl::string_view kHloString = R"(
    HloModule module, is_scheduled=true

    ENTRY entry {
      p0 = f32[1] parameter(0), metadata={scheduling_name="p0"}
      p1 = f32[1] parameter(1), metadata={scheduling_name="p1"}
      add0 = f32[1] add(p0,p1), metadata={scheduling_name="add0"}
      ROOT exp0 = f32[1] exponential(add0), metadata={scheduling_name="exp0"}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  SchedulingInstructionAnnotator pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));

  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla::gpu

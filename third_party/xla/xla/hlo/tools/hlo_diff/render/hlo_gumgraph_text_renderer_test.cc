/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_text_renderer.h"

#include <memory>
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

TEST(HloGumgraphTextRendererTest, MovedInstructionsShowTheirUses) {
  constexpr absl::string_view kLeftHlo = R"(
HloModule module

ENTRY entry {
  p0 = f32[8] parameter(0)
  sine = f32[8] sine(p0)
  cosine = f32[8] cosine(p0)
  ROOT subtract = f32[8] subtract(sine, cosine)
}
)";
  constexpr absl::string_view kRightHlo = R"(
HloModule module

ENTRY entry {
  p0 = f32[8] parameter(0)
  sine = f32[8] sine(p0)
  cosine = f32[8] cosine(p0)
  ROOT subtract = f32[8] subtract(cosine, sine)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> left,
                          ParseAndReturnUnverifiedModule(kLeftHlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> right,
                          ParseAndReturnUnverifiedModule(kRightHlo));
  DiffResult diff_result;
  diff_result.AddMovedInstruction(
      left->entry_computation()->GetInstructionWithName("sine"),
      right->entry_computation()->GetInstructionWithName("sine"));

  std::ostringstream out;
  RenderText(diff_result, out);

  EXPECT_THAT(out.str(), HasSubstr("Moved Instructions:\n  sine (1)\n"));
  EXPECT_THAT(out.str(), HasSubstr("sine and sine are used differently"));
  EXPECT_THAT(out.str(), HasSubstr("used by: subtract(0)"));
  EXPECT_THAT(out.str(), HasSubstr("used by: subtract(1)"));
  EXPECT_THAT(out.str(), Not(HasSubstr("have changed")));
}

TEST(HloGumgraphTextRendererTest, SummaryCountsMovedInstructions) {
  constexpr absl::string_view kHlo = R"(
HloModule module

ENTRY entry {
  ROOT p0 = f32[8] parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> left,
                          ParseAndReturnUnverifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> right,
                          ParseAndReturnUnverifiedModule(kHlo));
  DiffResult diff_result;
  diff_result.AddMovedInstruction(
      left->entry_computation()->root_instruction(),
      right->entry_computation()->root_instruction());

  std::ostringstream out;
  RenderTextSummary(diff_result, out);

  EXPECT_THAT(out.str(), HasSubstr("Moved instructions: 1"));
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla

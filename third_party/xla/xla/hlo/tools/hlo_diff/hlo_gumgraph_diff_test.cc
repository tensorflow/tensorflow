// Copyright 2025 The OpenXLA Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_diff.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/utils/test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

class HloDiffTest : public HloHardwareIndependentTestBase {};

TEST_F(HloDiffTest, ComputeDiffWorksWithoutEval) {
  // Create a module with entry computation containing the following structure:
  // [Param p0]->┌-------┐
  //             | add.1 |
  // [Param p1]->└-------┘
  const char* hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = s32[32,16]{0, 1:T(1,128)} parameter(0)
  p1 = s32[32,16]{0,1:T(1,128)} parameter(1)
  ROOT add.1 = s32[32,16]{0,1:T(1,128)} add(p0, p1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(hlo_string));
  DiffOptions options;
  options.run_eval = false;
  TF_ASSERT_OK_AND_ASSIGN(auto diff_result,
                          ComputeDiff(*module_l, *module_r, options));

  EXPECT_NE(diff_result.diff_result, nullptr);
  EXPECT_NE(diff_result.diff_summary, nullptr);
  EXPECT_EQ(diff_result.diff_eval, nullptr);
}

TEST_F(HloDiffTest, ComputeDiffWorksWithEval) {
  // Create a module with entry computation containing the following structure:
  // [Param p0]->┌-------┐
  //             | add.1 |
  // [Param p1]->└-------┘
  const char* hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = s32[32,16]{0, 1:T(1,128)} parameter(0)
  p1 = s32[32,16]{0,1:T(1,128)} parameter(1)
  ROOT add.1 = s32[32,16]{0,1:T(1,128)} add(p0, p1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(hlo_string));
  DiffOptions options;
  options.run_eval = true;
  TF_ASSERT_OK_AND_ASSIGN(auto diff_result,
                          ComputeDiff(*module_l, *module_r, options));

  EXPECT_NE(diff_result.diff_result, nullptr);
  EXPECT_NE(diff_result.diff_summary, nullptr);
  EXPECT_NE(diff_result.diff_eval, nullptr);
}

TEST_F(HloDiffTest, SwappedOperandsAreReportedAsMoved) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(kLeftHlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(kRightHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto diff_result,
                          ComputeDiff(*module_l, *module_r, DiffOptions()));

  EXPECT_THAT(GetSortedLeftInstructionNames(
                  diff_result.diff_result->changed_instructions),
              IsEmpty());
  EXPECT_THAT(GetSortedLeftInstructionNames(
                  diff_result.diff_result->moved_instructions),
              ElementsAre("cosine", "sine"));
}

TEST_F(HloDiffTest, SwappedCommutativeOperandsAreNotReportedAsMoved) {
  constexpr absl::string_view kLeftHlo = R"(
HloModule module

ENTRY entry {
  p0 = f32[8] parameter(0)
  sine = f32[8] sine(p0)
  cosine = f32[8] cosine(p0)
  ROOT add = f32[8] add(sine, cosine)
}
)";
  constexpr absl::string_view kRightHlo = R"(
HloModule module

ENTRY entry {
  p0 = f32[8] parameter(0)
  sine = f32[8] sine(p0)
  cosine = f32[8] cosine(p0)
  ROOT add = f32[8] add(cosine, sine)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(kLeftHlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(kRightHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto diff_result,
                          ComputeDiff(*module_l, *module_r, DiffOptions()));

  EXPECT_THAT(GetSortedLeftInstructionNames(
                  diff_result.diff_result->changed_instructions),
              IsEmpty());
  EXPECT_THAT(GetSortedLeftInstructionNames(
                  diff_result.diff_result->moved_instructions),
              IsEmpty());
}

TEST_F(HloDiffTest, SwappedEntryParametersAreReportedAsMoved) {
  // (x, y) -> x - y vs. (y, x) -> x - y.
  constexpr absl::string_view kLeftHlo = R"(
HloModule module

ENTRY entry {
  x = f32[4] parameter(0)
  y = f32[4] parameter(1)
  ROOT subtract = f32[4] subtract(x, y)
}
)";
  constexpr absl::string_view kRightHlo = R"(
HloModule module

ENTRY entry {
  y = f32[4] parameter(0)
  x = f32[4] parameter(1)
  ROOT subtract = f32[4] subtract(x, y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module_l,
                          ParseAndReturnVerifiedModule(kLeftHlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module_r,
                          ParseAndReturnVerifiedModule(kRightHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto diff_result,
                          ComputeDiff(*module_l, *module_r, DiffOptions()));

  // The entry parameters are matched by number, so each one is used at the
  // other operand position of the subtract.
  EXPECT_THAT(GetSortedLeftInstructionNames(
                  diff_result.diff_result->changed_instructions),
              IsEmpty());
  EXPECT_THAT(GetSortedLeftInstructionNames(
                  diff_result.diff_result->moved_instructions),
              ElementsAre("x", "y"));
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla

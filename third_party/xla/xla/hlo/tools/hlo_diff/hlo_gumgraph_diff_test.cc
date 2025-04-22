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

#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace hlo_diff {
namespace {

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
  TF_ASSERT_OK_AND_ASSIGN(
      auto diff_result,
      ComputeDiff(*module_l, *module_r, {}, /*run_eval=*/false));

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
  TF_ASSERT_OK_AND_ASSIGN(auto diff_result, ComputeDiff(*module_l, *module_r,
                                                        {}, /*run_eval=*/true));

  EXPECT_NE(diff_result.diff_result, nullptr);
  EXPECT_NE(diff_result.diff_summary, nullptr);
  EXPECT_NE(diff_result.diff_eval, nullptr);
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla

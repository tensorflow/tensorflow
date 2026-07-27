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

#include "xla/backends/gpu/transforms/narrow_dot_kwrapping_rewriter.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tests/restricted/hlo_test_base_legacy.h"

namespace xla {
namespace gpu {
namespace {

class NarrowDotKWrappingRewriterTest : public HloTestBaseLegacy {
 public:
  void SetUp() override {}
};

TEST_F(NarrowDotKWrappingRewriterTest, SimpleNarrowDot) {
  const std::string hlo_string = R"(
    HloModule module

    ENTRY main {
      lhs = f32[1, 4096] parameter(0)
      rhs = f32[4096, 4] parameter(1)
      ROOT dot = f32[1, 4] dot(lhs, rhs),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_string));

  NarrowDotKWrappingRewriter rewriter;
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  // Verify that the module is still valid after the rewrite.
  auto status_or_module = ParseAndReturnVerifiedModule(module->ToString());
  EXPECT_TRUE(status_or_module.status().ok()) << status_or_module.status();

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kReduce);
}

TEST_F(NarrowDotKWrappingRewriterTest, NotNarrowDot) {
  const std::string hlo_string = R"(
    HloModule module

    ENTRY main {
      lhs = f32[16, 4096] parameter(0)
      rhs = f32[4096, 16] parameter(1)
      ROOT dot = f32[16, 16] dot(lhs, rhs),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_string));

  NarrowDotKWrappingRewriter rewriter;
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/sanitize_constant_names.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;
using SanitizeConstantNamesTest = HloTestBase;

TEST_F(SanitizeConstantNamesTest, InstructionNameWithHyphenSanitized) {
  const char *const kHloString = R"(
    HloModule HyphenInInstructionName
      ENTRY kernelEntry {
        ROOT equal-to = s32[2]{0} constant({42, 73})
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_TRUE(SanitizeConstantNames().Run(module.get()).value());
  HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->name(), "equal_to");
}

TEST_F(SanitizeConstantNamesTest, InstructionNameWithDotSanitized) {
  const char *const kHloString = R"(
    HloModule HyphenInInstructionName
      ENTRY kernelEntry {
        ROOT equal.to = s32[2]{0} constant({42, 73})
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_TRUE(SanitizeConstantNames().Run(module.get()).value());
  HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->name(), "equal_to");
}

TEST_F(SanitizeConstantNamesTest, NewInstructionNameRegisteredWithModule) {
  const char *const kHloString = R"(
    HloModule HyphenInInstructionName
      ENTRY kernelEntry {
        ROOT equal.to = s32[2]{0} constant({42, 73})
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_TRUE(SanitizeConstantNames().Run(module.get()).value());
  HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->name(), "equal_to");

  auto constant_instr =
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1));
  constant_instr->SetAndSanitizeName("equal_to");
  module->entry_computation()->AddInstruction(std::move(constant_instr));

  EXPECT_THAT(FindInstruction(module.get(), "equal_to.1"),
              GmockMatch(m::Constant()));
}

TEST_F(SanitizeConstantNamesTest, BufferSanitizedNameCollisionResolved) {
  const char *const kHloString = R"(
    HloModule BufferSanitizedName
      ENTRY kernelEntry {
      equal.to = s32[2]{0} constant({42, 73})
      equal-to = s32[2]{0} constant({67, 3})
      ROOT equal_to = s32[2]{0} add(equal.to, equal-to)
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_TRUE(SanitizeConstantNames().Run(module.get()).value());
  EXPECT_THAT(FindInstruction(module.get(), "equal_to_1"),
              GmockMatch(m::Constant()));
  EXPECT_THAT(FindInstruction(module.get(), "equal_to_2"),
              GmockMatch(m::Constant()));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

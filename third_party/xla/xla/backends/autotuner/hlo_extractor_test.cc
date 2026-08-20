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

#include "xla/backends/autotuner/hlo_extractor.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::IsEmpty;
using ::testing::SizeIs;

constexpr absl::string_view kHlo = R"(
  HloModule test_module

  ENTRY main {
    p0 = f32[] parameter(0)
    add = f32[] add(p0, p0)
    add_2 = f32[] add(p0, add)
    ROOT copy = f32[] copy(add_2)
  }
)";

constexpr absl::string_view kDuplicateHlo = R"(
  HloModule duplicate_module

  ENTRY main {
    p0 = f32[] parameter(0)
    add1 = f32[] add(p0, p0)
    add2 = f32[] add(p0, p0)
    ROOT tuple = (f32[], f32[]) tuple(add1, add2)
  }
)";

class HloExtractorTest : public HloHardwareIndependentTestBase {};

TEST_F(HloExtractorTest, FindsNoInstructionsWhenFilterReturnsFalse) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  std::vector<EquivalentInstructions> groups = ExtractEquivalentInstructions(
      *module, [](const HloInstruction&) { return false; });
  EXPECT_THAT(groups, IsEmpty());
}

TEST_F(HloExtractorTest, ExtractsInstructionsMatchingFilter) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  std::vector<EquivalentInstructions> groups =
      ExtractEquivalentInstructions(*module, [](const HloInstruction& instr) {
        return instr.opcode() == HloOpcode::kCopy;
      });
  ASSERT_THAT(groups, SizeIs(1));
  ASSERT_THAT(groups[0], SizeIs(1));
  EXPECT_EQ(groups[0][0]->opcode(), HloOpcode::kCopy);
}

TEST_F(HloExtractorTest, GroupsDuplicateInstructionsWithSameFingerprint) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kDuplicateHlo));
  std::vector<EquivalentInstructions> groups =
      ExtractEquivalentInstructions(*module, [](const HloInstruction& instr) {
        return instr.opcode() == HloOpcode::kAdd;
      });
  ASSERT_THAT(groups, SizeIs(1));
  EXPECT_THAT(groups[0], SizeIs(2));
}

TEST_F(HloExtractorTest, GroupsMultipleDistinctInstructionsSeparately) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  std::vector<EquivalentInstructions> groups =
      ExtractEquivalentInstructions(*module, [](const HloInstruction& instr) {
        return instr.opcode() == HloOpcode::kAdd ||
               instr.opcode() == HloOpcode::kCopy;
      });
  ASSERT_THAT(groups, SizeIs(2));
}

}  // namespace
}  // namespace xla

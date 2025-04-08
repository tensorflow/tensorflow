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

#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_renderer_util.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class HloDiffRendererUtilTest : public HloTestBase {};

TEST_F(HloDiffRendererUtilTest, GroupInstructionsByOpcode) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule test_module

ENTRY test_computation {
  param1 = s32[10] parameter(0)
  param2 = s32[10] parameter(1)
  add = s32[10] add(param1, param2)
  ROOT sub = s32[10] subtract(add, param2)
}
  )"));
  absl::flat_hash_set<const HloInstruction*> instructions;
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      instructions.insert(instruction);
    }
  }

  EXPECT_THAT(GroupInstructionsByOpcode(instructions),
              UnorderedElementsAre(Pair(HloOpcode::kParameter, SizeIs(2)),
                                   Pair(HloOpcode::kAdd, SizeIs(1)),
                                   Pair(HloOpcode::kSubtract, SizeIs(1))));
}

TEST_F(HloDiffRendererUtilTest, GroupInstructionPairsByOpcode) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule test_module

ENTRY test_computation {
  param1 = s32[10] parameter(0)
  param2 = s32[10] parameter(1)
  add = s32[10] add(param1, param2)
  ROOT sub = s32[10] subtract(add, param2)
}
  )"));
  absl::flat_hash_map<const HloInstruction*, const HloInstruction*>
      instruction_map;
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      instruction_map[instruction] = instruction;
    }
  }

  EXPECT_THAT(GroupInstructionPairsByOpcode(instruction_map),
              UnorderedElementsAre(Pair(HloOpcode::kParameter, SizeIs(2)),
                                   Pair(HloOpcode::kAdd, SizeIs(1)),
                                   Pair(HloOpcode::kSubtract, SizeIs(1))));
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla

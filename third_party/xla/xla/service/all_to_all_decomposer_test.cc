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

#include "xla/service/all_to_all_decomposer.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace {

using AllToAllDecomposerTest = HloTestBase;
using ::testing::_;
namespace op = xla::testing::opcode_matchers;

TEST_F(AllToAllDecomposerTest, RaggedAllToAllRank1) {
  const std::string module_str =
      R"(HloModule RaggedAllToAll
        ENTRY AllToAll {
          p0 = s32[8]{0} parameter(0)
          c0 = s32[] constant(0)
          output = s32[8]{0} broadcast(c0), dimensions={}
          p1 = s32[4]{0} parameter(1)
          p2 = s32[4]{0} parameter(2)
          p3 = s32[4]{0} parameter(3)
          p4 = s32[4]{0} parameter(4)
          input = s32[8]{0} copy(p0)
          input_offsets = s32[4]{0} copy(p1)
          send_sizes = s32[4]{0} copy(p2)
          output_offsets = s32[4]{0} copy(p3)
          recv_sizes = s32[4]{0} copy(p4)
          ra2a = s32[8]{0} ragged-all-to-all(input, output, input_offsets, send_sizes, output_offsets, recv_sizes), replica_groups={{0,1,2,3}}
          ROOT copy = s32[8]{0} copy(ra2a)
        })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((module_str)));
  AllToAllDecomposer decomposer(true, 3);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(op::Reshape(op::RaggedAllToAll(
                  op::Reshape(op::Copy(op::Parameter(0))),
                  op::Reshape(op::Broadcast(op::Constant())), _, _, _, _))));
  std::vector<HloInstruction*> reshapes;
  std::vector<HloInstruction*> ragged_all_to_alls;
  for (HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kReshape) {
      reshapes.push_back(instruction);
    }
    if (instruction->opcode() == HloOpcode::kRaggedAllToAll) {
      ragged_all_to_alls.push_back(instruction);
    }
  }
  EXPECT_EQ(reshapes.size(), 3);
  EXPECT_EQ(ragged_all_to_alls.size(), 1);
  EXPECT_EQ(ragged_all_to_alls[0]->shape().dimensions_size(), 3);
}

TEST_F(AllToAllDecomposerTest, RaggedAllToAllRank3) {
  const std::string module_str =
      R"(HloModule RaggedAllToAll
        ENTRY AllToAll {
          p0 = s32[8,16,256]{2,1,0} parameter(0)
          c0 = s32[] constant(0)
          output = s32[8,16,256]{2,1,0} broadcast(c0), dimensions={}
          p1 = s32[4]{0} parameter(1)
          p2 = s32[4]{0} parameter(2)
          p3 = s32[4]{0} parameter(3)
          p4 = s32[4]{0} parameter(4)
          input = s32[8,16,256]{2,1,0} copy(p0)
          input_offsets = s32[4]{0} copy(p1)
          send_sizes = s32[4]{0} copy(p2)
          output_offsets = s32[4]{0} copy(p3)
          recv_sizes = s32[4]{0} copy(p4)
          ra2a = s32[8,16,256]{2,1,0} ragged-all-to-all(input, output, input_offsets, send_sizes, output_offsets, recv_sizes), replica_groups={{0,1,2,3}}
          ROOT copy = s32[8,16,256]{2,1,0} copy(ra2a)
        })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((module_str)));
  AllToAllDecomposer decomposer(true, 3);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla

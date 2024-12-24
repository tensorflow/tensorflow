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

#include "xla/service/while_loop_pipeline_unroller.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/copy_insertion.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

// Copied from xla/service/copy_insertion_test.cc
int64_t CountCopies(const HloComputation& computation) {
  int64_t count = 0;
  for (const auto& instruction : computation.instructions()) {
    if (instruction->opcode() == HloOpcode::kCopy) {
      count++;
    }
  }
  return count;
}

class WhileLoopPipelineUnrollerTest : public HloTestBase {
 protected:
  WhileLoopPipelineUnrollerTest() = default;
};

TEST_F(WhileLoopPipelineUnrollerTest, PipelinedLoop) {
  constexpr absl::string_view hlo = R"(
HloModule main

body {
  input_tuple.0 = (s32[], s32[], s32[], s32[]) parameter(0)
  arg.0 = get-tuple-element(input_tuple.0), index=0
  arg.1 = get-tuple-element(input_tuple.0), index=1
  arg.2 = get-tuple-element(input_tuple.0), index=2
  arg.3 = get-tuple-element(input_tuple.0), index=3

  one.0 = s32[] constant(1)
  out.0 = add(arg.0, one.0)

  add.0 = add(arg.3, one.0)
  ROOT output_tuple.0 = tuple(arg.1, arg.2, out.0, add.0)
}

condition {
  input_tuple.0 = (s32[], s32[], s32[], s32[]) parameter(0)
  arg.3 = get-tuple-element(input_tuple.0), index=3
  three.0 = s32[] constant(3)
  ROOT pred.0 = compare(arg.3, three.0), direction=LT
}

ENTRY main {
  while_tuple.0 = (s32[], s32[], s32[], s32[]) parameter(0)
  ROOT while.0 = (s32[], s32[], s32[], s32[]) while(while_tuple.0), body=body, condition=condition
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  WhileLoopPipelineUnroller wlpu;
  ASSERT_IS_OK(wlpu.Run(module.get()).status());
  CopyInsertion copy_insertion(nullptr,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());

  const HloInstruction* original_loop =
      FindInstruction(module.get(), "while.0");
  // The original loop should have 3 copies.
  // arg.1 moves to index 0.
  // arg.2 moves to index 1.
  // out.0 moves to index 2.
  EXPECT_EQ(CountCopies(*original_loop->while_body()), 3);

  const HloInstruction* unrolled_loop = original_loop->operand(0);
  EXPECT_EQ(unrolled_loop->opcode(), HloOpcode::kWhile);
  // There should be no copies inserted into the unrolled loop.
  EXPECT_EQ(CountCopies(*unrolled_loop->while_body()), 0);
}

TEST_F(WhileLoopPipelineUnrollerTest, PipelinedLoopWithInfeed) {
  constexpr absl::string_view hlo = R"(
HloModule main

body {
  input_tuple.0 = (s32[], s32[], s32[], token[], s32[]) parameter(0)
  arg.0 = get-tuple-element(input_tuple.0), index=0
  arg.1 = get-tuple-element(input_tuple.0), index=1
  arg.2 = get-tuple-element(input_tuple.0), index=2
  arg.3 = get-tuple-element(input_tuple.0), index=3
  arg.4 = get-tuple-element(input_tuple.0), index=4

  infeed.0 = (s32[], token[]) infeed(arg.3)
  infeed_value.0 = get-tuple-element(infeed.0), index=0
  infeed_output_token.0 = get-tuple-element(infeed.0), index=1

  out.0 = add(arg.0, arg.1)

  one.0 = s32[] constant(1)
  add.0 = add(arg.4, one.0)
  ROOT output_tuple.0 = tuple(out.0, arg.2, infeed_value.0, infeed_output_token.0, add.0)
}

condition {
  input_tuple.0 = (s32[], s32[], s32[], token[], s32[]) parameter(0)
  arg.4 = get-tuple-element(input_tuple.0), index=4
  three.0 = s32[] constant(3)
  ROOT pred.0 = compare(arg.4, three.0), direction=LT
}

ENTRY main {
  infeed_input_token.0 = after-all()
  infeed.0 = (s32[], token[]) infeed(infeed_input_token.0)
  infeed_value.0 = s32[] get-tuple-element(infeed.0), index=0
  infeed_output_token.0 = token[] get-tuple-element(infeed.0), index=1

  infeed.1 = (s32[], token[]) infeed(infeed_output_token.0)
  infeed_value.1 = s32[] get-tuple-element(infeed.1), index=0
  infeed_output_token.1 = token[] get-tuple-element(infeed.1), index=1

  zero.0 = s32[] constant(0)
  while_tuple.0 = tuple(zero.0, infeed_value.0, infeed_value.1, infeed_output_token.1, zero.0)
  while.0 = (s32[], s32[], s32[], token[], s32[]) while(while_tuple.0), body=body, condition=condition

  ROOT root.0 = get-tuple-element(while.0), index=0
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  WhileLoopPipelineUnroller wlpu;
  ASSERT_IS_OK(wlpu.Run(module.get()).status());
  CopyInsertion copy_insertion(nullptr,
                               /*use_region_based_live_range_analysis=*/-1);
  ASSERT_IS_OK(copy_insertion.Run(module.get()).status());

  const HloInstruction* original_loop =
      FindInstruction(module.get(), "while.0");
  // The original loop should have 1 copy.
  // arg.2 moves to index 1.
  EXPECT_EQ(CountCopies(*original_loop->while_body()), 1);

  const HloInstruction* unrolled_loop = original_loop->operand(0);
  EXPECT_EQ(unrolled_loop->opcode(), HloOpcode::kWhile);
  // There should be no copies inserted into the unrolled loop.
  EXPECT_EQ(CountCopies(*unrolled_loop->while_body()), 0);

  // All infeeds in the unrolled body need to be ordered with respect to each
  // other.
  absl::InlinedVector<HloInstruction*, 3> unrolled_infeeds;
  for (HloInstruction* instruction :
       unrolled_loop->while_body()->instructions()) {
    if (instruction->opcode() == HloOpcode::kInfeed) {
      unrolled_infeeds.push_back(instruction);
    }
  }
  DependencyHloOrdering dlo(module.get());
  for (HloInstruction* lhs : unrolled_infeeds) {
    for (HloInstruction* rhs : unrolled_infeeds) {
      if (lhs != rhs) {
        EXPECT_TRUE(dlo.ExecutesBefore(lhs, rhs) ||
                    dlo.ExecutesBefore(rhs, lhs));
      }
    }
  }
}

}  // namespace
}  // namespace xla

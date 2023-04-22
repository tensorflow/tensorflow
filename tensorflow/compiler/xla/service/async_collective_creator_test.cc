/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/async_collective_creator.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::NotNull;
using ::testing::SizeIs;

using AsyncAllReduceCreatorTest = HloTestBase;

TEST_F(AsyncAllReduceCreatorTest, SplitsSingleAllReduce) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }
  ENTRY entry {
    p0 = f32[8] parameter(0)
    ROOT ar = f32[8] all-reduce(p0), to_apply=add
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(AsyncCollectiveCreator(/*convert_all_reduce=*/true)
                   .Run(hlo_module.get())
                   .status());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 3);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kAllReduceDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kAllReduceStart);
}

TEST_F(AsyncAllReduceCreatorTest, SplitsSingleAllGather) {
  constexpr absl::string_view hlo_string = R"(
  HloModule test
  ENTRY entry {
    p0 = f32[1] parameter(0)
    ROOT ag = f32[8] all-gather(p0), dimensions={0}, replica_groups={{0,1,2,3,4,5,6,7}}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK(AsyncCollectiveCreator(/*convert_all_reduce=*/false,
                                      /*convert_all_gather=*/true)
                   .Run(hlo_module.get())
                   .status());

  HloComputation* computation = hlo_module->entry_computation();
  ASSERT_THAT(computation, NotNull());
  ASSERT_EQ(computation->instruction_count(), 3);
  const HloInstruction* done = computation->root_instruction();
  EXPECT_EQ(done->opcode(), HloOpcode::kAllGatherDone);
  ASSERT_THAT(done->operands(), SizeIs(1));
  const HloInstruction* start = done->operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kAllGatherStart);
}

}  // namespace
}  // namespace xla

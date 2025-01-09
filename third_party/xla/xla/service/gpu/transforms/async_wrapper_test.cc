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

#include "xla/service/gpu/transforms/async_wrapper.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

class AsyncWrapperTest : public HloTestBase {};

int CountAsyncInstructions(HloComputation* computation) {
  int count = 0;
  for (const HloInstruction* instruction : computation->instructions()) {
    if (instruction->IsAsynchronous()) ++count;
  }
  return count;
}

TEST_F(AsyncWrapperTest, BasicFusion) {
  const char* hlo_text = R"(
  HloModule m

  double1 {
    p0 = f32[1] parameter(0)
    ROOT add = f32[1] add(p0, p0)
  }

  double2 {
    p0 = f32[1] parameter(0)
    ROOT add = f32[1] add(p0, p0)
  }

  ENTRY main {
    p0 = f32[1] parameter(0)
    agg1 = f32[1] fusion(p0), kind=kLoop, calls=double1
    agg2 = f32[1] fusion(p0), kind=kLoop, calls=double2
    ROOT done = f32[1] add(agg1, agg2)
  })";

  std::unique_ptr<VerifiedHloModule> module =
      ParseAndReturnVerifiedModule(hlo_text).value();

  AsyncWrapper wrapper(HloPredicateIsOp<HloOpcode::kFusion>);
  EXPECT_THAT(wrapper.HloModulePass::Run(module.get()), IsOkAndHolds(true));
  EXPECT_EQ(CountAsyncInstructions(module->entry_computation()), 4);

  Literal argument = LiteralUtil::CreateR1<float>({1.0});
  Literal expected = LiteralUtil::CreateR1<float>({4.0});

  Literal result = ExecuteNoHloPasses(std::move(module), {&argument});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(AsyncWrapperTest, OpWithinWhileShouldWrapInAsync) {
  const char* hlo = R"(
  HloModule m

  body {
    param = (f32[1], s32[]) parameter(0)
    p0 = f32[1] get-tuple-element(param), index=0
    agg1 = f32[1] custom-call(p0), custom_call_target="foo"
    agg2 = f32[1] custom-call(p0), custom_call_target="bar"
    done = f32[1] add(agg1, agg2)
    iter = s32[] get-tuple-element(param), index=1
    c1 = s32[] constant(1)
    add = s32[] add(iter, c1)
    ROOT tuple = (f32[1], s32[]) tuple(done, add)
  }

  condition {
    param.1 = (f32[1], s32[]) parameter(0)
    iter.1 = s32[] get-tuple-element(param.1), index=1
    c4 = s32[] constant(4)
    ROOT compare = pred[] compare(iter.1, c4), direction=LT
  }

  ENTRY main {
    c0 = s32[] constant(0)
    p0.1 = f32[1] parameter(0)
    agg3 = f32[1] custom-call(p0.1), custom_call_target="baz"
    tuple = (f32[1], s32[]) tuple(agg3, c0)
    while = (f32[1], s32[]) while(tuple), body=body, condition=condition
    ROOT done.1 = f32[1] get-tuple-element(while), index=0
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  AsyncWrapper wrapper(HloPredicateIsOp<HloOpcode::kCustomCall>);
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          wrapper.Run(module.get(), /*execution_threads=*/{}));
  EXPECT_TRUE(changed);
  EXPECT_EQ(CountAsyncInstructions(module->entry_computation()), 2);
  HloInstruction* while_op = hlo_query::FindInstruction(
      module->entry_computation(), HloOpcode::kWhile);
  ASSERT_NE(while_op, nullptr);
  EXPECT_EQ(CountAsyncInstructions(while_op->while_body()), 4);
}

TEST_F(AsyncWrapperTest, OpWithinConditionalShouldWrapInAsync) {
  const char* hlo = R"(
  HloModule m

  true_computation {
    p0.1 = f32[] parameter(0)
    ROOT res.1 = f32[] custom-call(p0.1), custom_call_target="foo"
  }

  false_computation {
    p0.2 = f32[] parameter(0)
    ROOT res.2 = f32[] custom-call(p0.2), custom_call_target="foo"
  }

  ENTRY main {
    p0 = f32[] parameter(0)
    c0 = f32[] constant(0)
    compare = pred[] compare(p0, c0), direction=GE
    ROOT done = f32[] conditional(compare, p0, p0), true_computation=true_computation, false_computation=false_computation
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  AsyncWrapper wrapper(HloPredicateIsOp<HloOpcode::kCustomCall>);
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          wrapper.Run(module.get(), /*execution_threads=*/{}));
  EXPECT_TRUE(changed);
  EXPECT_EQ(CountAsyncInstructions(module->entry_computation()), 0);
  HloInstruction* conditional_op = hlo_query::FindInstruction(
      module->entry_computation(), HloOpcode::kConditional);
  ASSERT_NE(conditional_op, nullptr);
  EXPECT_EQ(CountAsyncInstructions(conditional_op->true_computation()), 2);
  EXPECT_EQ(CountAsyncInstructions(conditional_op->false_computation()), 2);
}

TEST_F(AsyncWrapperTest, OpWithinFusionShouldNotWrapInAsync) {
  const char* hlo = R"(
  foo {
    p0 = f32[1] parameter(0)
    ROOT custom-call = f32[1] custom-call(p0), custom_call_target="bar"
  }
  ENTRY main {
    c0 = s32[] constant(0)
    p0.1 = f32[1] parameter(0)
    agg.1 = f32[1] fusion(p0.1), kind=kLoop, calls=foo
    agg.2 = f32[1] custom-call(agg.1), custom_call_target="bar"
    ROOT done.1 = (f32[1], f32[1]) tuple(agg.1, agg.2)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  AsyncWrapper wrapper(HloPredicateIsOp<HloOpcode::kCustomCall>);
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          wrapper.Run(module.get(), /*execution_threads=*/{}));
  EXPECT_TRUE(changed);
  EXPECT_EQ(CountAsyncInstructions(module->entry_computation()), 2);

  HloInstruction* fusion = hlo_query::FindInstruction(
      module->entry_computation(), HloOpcode::kFusion);
  EXPECT_EQ(CountAsyncInstructions(fusion->fused_instructions_computation()),
            0);
}

}  // namespace
}  // namespace xla::gpu

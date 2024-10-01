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

#include "xla/service/infeed_token_propagation.h"

#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

class InfeedTokenPropagationTest : public HloTestBase {
 protected:
  InfeedTokenPropagationTest() = default;
};

TEST_F(InfeedTokenPropagationTest, EntryComputationInfeed) {
  constexpr std::string_view hlo = R"(
HloModule main

ENTRY main {
  token.0 = after-all()
  infeed.0 = (s32[], token[]) infeed(token.0)
  ROOT gte.0 = get-tuple-element(infeed.0), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(InfeedTokenPropagationTest, EntryComputationOutfeed) {
  constexpr std::string_view hlo = R"(
HloModule main

ENTRY main {
  arg.0 = s32[] parameter(0)
  tuple.0 = tuple(arg.0)
  token.0 = after-all()
  outfeed.0 = token[] outfeed(tuple.0, token.0), outfeed_shape=(s32[])
  ROOT tuple.1 = tuple()
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(InfeedTokenPropagationTest, ConditionalInfeed) {
  constexpr std::string_view hlo = R"(
HloModule main

true_comp {
  arg.0 = () parameter(0)
  token.0 = after-all()
  infeed.0 = (s32[], token[]) infeed(token.0)
  ROOT tuple.0 = tuple()
}

false_comp {
  arg.0 = () parameter(0)
  ROOT tuple.0 = tuple()
}

ENTRY main {
  pred.0 = pred[] constant(true)
  true_tuple.0 = tuple()
  false_tuple.0 = tuple()
  ROOT cond.0 = () conditional(pred.0, true_tuple.0, false_tuple.0), true_computation=true_comp, false_computation=false_comp
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The infeed output token should have propagated through the conditional.
  HloInstruction* cond = FindInstruction(module.get(), "cond.0");
  EXPECT_EQ(cond->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(cond->shape().tuple_shapes()[0].IsToken());

  // The infeed input token should have propagated through the true tuple.
  HloInstruction* true_tuple = FindInstruction(module.get(), "true_tuple.0");
  EXPECT_EQ(true_tuple->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(true_tuple->shape().tuple_shapes()[0].IsToken());

  // The infeed input token should not have propagated through the false tuple.
  HloInstruction* false_tuple = FindInstruction(module.get(), "false_tuple.0");
  EXPECT_EQ(false_tuple->shape().tuple_shapes_size(), 0);

  // The infeed output token should have propagated through the true
  // computation's root.
  HloComputation* true_comp = FindComputation(module.get(), "true_comp");
  EXPECT_THAT(true_comp->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Infeed(), 1)));

  // The infeed output token should have propagated to the false computation's
  // root.
  HloComputation* false_comp = FindComputation(module.get(), "false_comp");
  EXPECT_THAT(false_comp->root_instruction(), op::Tuple(op::AfterAll()));
}

TEST_F(InfeedTokenPropagationTest, ConditionalOutfeed) {
  constexpr std::string_view hlo = R"(
HloModule main

true_comp {
  arg.0 = (s32[]) parameter(0)
  token.0 = after-all()
  outfeed.0 = token[] outfeed(arg.0, token.0), outfeed_shape=(s32[])
  ROOT tuple.0 = tuple()
}

false_comp {
  arg.0 = () parameter(0)
  ROOT tuple.0 = tuple()
}

ENTRY main {
  arg.0 = s32[] parameter(0)
  pred.0 = pred[] constant(true)
  true_tuple.0 = tuple(arg.0)
  false_tuple.0 = tuple()
  ROOT cond.0 = () conditional(pred.0, true_tuple.0, false_tuple.0), true_computation=true_comp, false_computation=false_comp
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The outfeed output token should have propagated through the conditional.
  HloInstruction* cond = FindInstruction(module.get(), "cond.0");
  EXPECT_EQ(cond->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(cond->shape().tuple_shapes()[0].IsToken());

  // The outfeed input token should have propagated through the true tuple.
  HloInstruction* true_tuple = FindInstruction(module.get(), "true_tuple.0");
  EXPECT_EQ(true_tuple->shape().tuple_shapes_size(), 2);
  EXPECT_TRUE(true_tuple->shape().tuple_shapes()[1].IsToken());

  // The outfeed input token should not have propagated through the false tuple.
  HloInstruction* false_tuple = FindInstruction(module.get(), "false_tuple.0");
  EXPECT_EQ(false_tuple->shape().tuple_shapes_size(), 0);

  // The outfeed output token should have propagated through the true
  // computation's root.
  HloComputation* true_comp = FindComputation(module.get(), "true_comp");
  EXPECT_THAT(true_comp->root_instruction(), op::Tuple(op::Outfeed()));

  // The outfeed output token should have propagated to the false computation's
  // root.
  HloComputation* false_comp = FindComputation(module.get(), "false_comp");
  EXPECT_THAT(false_comp->root_instruction(), op::Tuple(op::AfterAll()));
}

TEST_F(InfeedTokenPropagationTest, ConditionalDuplicateOperand) {
  constexpr std::string_view hlo = R"(
HloModule main

true_comp {
  arg.0 = () parameter(0)
  token.0 = after-all()
  infeed.0 = (s32[], token[]) infeed(token.0)
  ROOT tuple.0 = tuple()
}

false_comp {
  arg.0 = () parameter(0)
  ROOT tuple.0 = tuple()
}

ENTRY main {
  pred.0 = pred[] constant(true)
  tuple.0 = tuple()
  ROOT cond.0 = () conditional(pred.0, tuple.0, tuple.0), true_computation=true_comp, false_computation=false_comp
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The infeed output token should have propagated through the conditional.
  HloInstruction* cond = FindInstruction(module.get(), "cond.0");
  EXPECT_EQ(cond->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(cond->shape().tuple_shapes()[0].IsToken());

  // The infeed input token should have propagated through the true tuple.
  const HloInstruction* true_tuple = cond->operand(1);
  EXPECT_EQ(true_tuple->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(true_tuple->shape().tuple_shapes()[0].IsToken());

  // The infeed input token should not have propagated through the false tuple.
  const HloInstruction* false_tuple = cond->operand(2);
  EXPECT_EQ(false_tuple->shape().tuple_shapes_size(), 0);

  // The infeed output token should have propagated through the true
  // computation's root.
  HloComputation* true_comp = FindComputation(module.get(), "true_comp");
  EXPECT_THAT(true_comp->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Infeed(), 1)));

  // The infeed output token should have propagated to the false computation's
  // root.
  HloComputation* false_comp = FindComputation(module.get(), "false_comp");
  EXPECT_THAT(false_comp->root_instruction(), op::Tuple(op::AfterAll()));
}

TEST_F(InfeedTokenPropagationTest, NonTupleConditional) {
  constexpr std::string_view hlo = R"(
HloModule main

true_comp {
  arg.0 = s32[] parameter(0)
  outfeed_tuple.0 = tuple(arg.0)
  token.0 = after-all()
  outfeed.0 = token[] outfeed(outfeed_tuple.0, token.0), outfeed_shape=(s32[])
  ROOT tuple.0 = tuple()
}

false_comp {
  arg.0 = () parameter(0)
  ROOT tuple.0 = tuple()
}

ENTRY main {
  arg.0 = s32[] parameter(0)
  pred.0 = pred[] constant(true)
  false_tuple.0 = tuple()
  ROOT cond.0 = () conditional(pred.0, arg.0, false_tuple.0), true_computation=true_comp, false_computation=false_comp
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The outfeed output token should have propagated through the conditional.
  HloInstruction* cond = FindInstruction(module.get(), "cond.0");
  EXPECT_EQ(cond->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(cond->shape().tuple_shapes()[0].IsToken());

  // The outfeed input token should have propagated through the true tuple.
  HloInstruction* true_tuple = cond->mutable_operand(1);
  EXPECT_TRUE(true_tuple->shape().IsTuple());
  EXPECT_EQ(true_tuple->shape().tuple_shapes_size(), 2);
  EXPECT_TRUE(true_tuple->shape().tuple_shapes()[1].IsToken());

  // The outfeed input token should not have propagated through the false tuple.
  HloInstruction* false_tuple = FindInstruction(module.get(), "false_tuple.0");
  EXPECT_EQ(false_tuple->shape().tuple_shapes_size(), 0);

  // The outfeed output token should have propagated through the true
  // computation's root.
  HloComputation* true_comp = FindComputation(module.get(), "true_comp");
  EXPECT_THAT(true_comp->root_instruction(), op::Tuple(op::Outfeed()));

  // The outfeed output token should have propagated to the false computation's
  // root.
  HloComputation* false_comp = FindComputation(module.get(), "false_comp");
  EXPECT_THAT(false_comp->root_instruction(), op::Tuple(op::AfterAll()));
}

TEST_F(InfeedTokenPropagationTest, DisjointConditionalOutfeed) {
  constexpr std::string_view hlo = R"(
HloModule main

true_comp {
  ROOT arg.0 = () parameter(0)
  one.0 = s32[] constant(1)
  outfeed_tuple.0 = tuple(one.0)
  token.0 = after-all()
  outfeed.0 = token[] outfeed(outfeed_tuple.0, token.0), outfeed_shape=(s32[])
}

false_comp {
  arg.0 = () parameter(0)
  ROOT tuple.0 = tuple()
}

ENTRY main {
  pred.0 = pred[] constant(true)
  true_tuple.0 = tuple()
  false_tuple.0 = tuple()
  ROOT cond.0 = () conditional(pred.0, true_tuple.0, false_tuple.0), true_computation=true_comp, false_computation=false_comp
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The outfeed output token should have propagated through the conditional.
  HloInstruction* cond = FindInstruction(module.get(), "cond.0");
  EXPECT_EQ(cond->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(cond->shape().tuple_shapes()[0].IsToken());

  // The outfeed input token should have propagated through the true tuple.
  HloInstruction* true_tuple = FindInstruction(module.get(), "true_tuple.0");
  EXPECT_EQ(true_tuple->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(true_tuple->shape().tuple_shapes()[0].IsToken());

  // The outfeed input token should not have propagated through the false tuple.
  HloInstruction* false_tuple = FindInstruction(module.get(), "false_tuple.0");
  EXPECT_EQ(false_tuple->shape().tuple_shapes_size(), 0);

  // The outfeed output token should have propagated through the true
  // computation's root.
  HloComputation* true_comp = FindComputation(module.get(), "true_comp");
  EXPECT_THAT(true_comp->root_instruction(), op::Tuple(op::Outfeed()));

  // The outfeed output token should have propagated to the false computation's
  // root.
  HloComputation* false_comp = FindComputation(module.get(), "false_comp");
  EXPECT_THAT(false_comp->root_instruction(), op::Tuple(op::AfterAll()));
}

TEST_F(InfeedTokenPropagationTest, WhileInfeed) {
  constexpr std::string_view hlo = R"(
HloModule main

comp {
  arg.0 = () parameter(0)
  token.0 = after-all()
  infeed.0 = (s32[], token[]) infeed(token.0)
  ROOT tuple.0 = tuple()
}

cond {
  arg.0 = () parameter(0)
  ROOT true.0 = pred[] constant(true)
}

ENTRY main {
  while_tuple.0 = tuple()
  ROOT while.0 = () while(while_tuple.0), condition=cond, body=comp
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The infeed output token should have propagated through the loop.
  HloInstruction* loop = FindInstruction(module.get(), "while.0");
  EXPECT_EQ(loop->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(loop->shape().tuple_shapes()[0].IsToken());

  // The infeed input token should have propagated through the loop tuple.
  HloInstruction* loop_tuple = FindInstruction(module.get(), "while_tuple.0");
  EXPECT_EQ(loop_tuple->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(loop_tuple->shape().tuple_shapes()[0].IsToken());

  // The infeed output token should have propagated through the while body root.
  HloComputation* body_comp = FindComputation(module.get(), "comp");
  EXPECT_THAT(body_comp->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Infeed(), 1)));

  // The infeed input token should have propagated through the body parameter.
  HloInstruction* body_param = body_comp->parameter_instruction(0);
  EXPECT_EQ(body_param->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(body_param->shape().tuple_shapes()[0].IsToken());

  // The infeed input token should have propagated through the condition
  // parameter.
  HloComputation* cond_comp = FindComputation(module.get(), "cond");
  HloInstruction* cond_param = cond_comp->parameter_instruction(0);
  EXPECT_EQ(cond_param->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(cond_param->shape().tuple_shapes()[0].IsToken());
}

TEST_F(InfeedTokenPropagationTest, WhileOutfeed) {
  constexpr std::string_view hlo = R"(
HloModule main

comp {
  arg.0 = (s32[]) parameter(0)
  token.0 = after-all()
  outfeed.0 = token[] outfeed(arg.0, token.0), outfeed_shape=(s32[])
  gte.0 = get-tuple-element(arg.0), index=0
  ROOT tuple.0 = tuple(gte.0)
}

cond {
  arg.0 = (s32[]) parameter(0)
  ROOT true.0 = pred[] constant(true)
}

ENTRY main {
  arg.0 = s32[] parameter(0)
  while_tuple.0 = tuple(arg.0)
  ROOT while.0 = (s32[]) while(while_tuple.0), condition=cond, body=comp
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The outfeed output token should have propagated through the loop.
  HloInstruction* loop = FindInstruction(module.get(), "while.0");
  EXPECT_EQ(loop->shape().tuple_shapes_size(), 2);
  EXPECT_TRUE(loop->shape().tuple_shapes()[1].IsToken());

  // The outfeed input token should have propagated through the loop tuple.
  HloInstruction* loop_tuple = FindInstruction(module.get(), "while_tuple.0");
  EXPECT_EQ(loop_tuple->shape().tuple_shapes_size(), 2);
  EXPECT_TRUE(loop_tuple->shape().tuple_shapes()[1].IsToken());

  // The outfeed output token should have propagated through the while body
  // root.
  HloComputation* body_comp = FindComputation(module.get(), "comp");
  EXPECT_THAT(body_comp->root_instruction(),
              op::Tuple(op::GetTupleElement(), op::Outfeed()));

  // The outfeed output token should have propagated through the body parameter.
  HloInstruction* body_param = body_comp->parameter_instruction(0);
  EXPECT_EQ(body_param->shape().tuple_shapes_size(), 2);
  EXPECT_TRUE(body_param->shape().tuple_shapes()[1].IsToken());

  // The outfeed output token should have propagated through the condition
  // parameter.
  HloComputation* cond_comp = FindComputation(module.get(), "cond");
  HloInstruction* cond_param = cond_comp->parameter_instruction(0);
  EXPECT_EQ(cond_param->shape().tuple_shapes_size(), 2);
  EXPECT_TRUE(cond_param->shape().tuple_shapes()[1].IsToken());
}

TEST_F(InfeedTokenPropagationTest, DisjointWhileOutfeed) {
  constexpr std::string_view hlo = R"(
HloModule main

comp {
  ROOT arg.0 = () parameter(0)
  one.0 = s32[] constant(1)
  outfeed_tuple.0 = tuple(one.0)
  token.0 = after-all()
  outfeed.0 = token[] outfeed(outfeed_tuple.0, token.0), outfeed_shape=(s32[])
}

cond {
  arg.0 = () parameter(0)
  ROOT true.0 = pred[] constant(true)
}

ENTRY main {
  while_tuple.0 = tuple()
  ROOT while.0 = () while(while_tuple.0), condition=cond, body=comp
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The outfeed output token should have propagated through the loop.
  HloInstruction* loop = FindInstruction(module.get(), "while.0");
  EXPECT_EQ(loop->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(loop->shape().tuple_shapes()[0].IsToken());

  // The outfeed input token should have propagated through the loop tuple.
  HloInstruction* loop_tuple = FindInstruction(module.get(), "while_tuple.0");
  EXPECT_EQ(loop_tuple->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(loop_tuple->shape().tuple_shapes()[0].IsToken());

  // The outfeed output token should have propagated through the while body
  // root.
  HloComputation* body_comp = FindComputation(module.get(), "comp");
  EXPECT_THAT(body_comp->root_instruction(), op::Tuple(op::Outfeed()));

  // The outfeed output token should have propagated through the body parameter.
  HloInstruction* body_param = body_comp->parameter_instruction(0);
  EXPECT_EQ(body_param->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(body_param->shape().tuple_shapes()[0].IsToken());

  // The outfeed output token should have propagated through the condition
  // parameter.
  HloComputation* cond_comp = FindComputation(module.get(), "cond");
  HloInstruction* cond_param = cond_comp->parameter_instruction(0);
  EXPECT_EQ(cond_param->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(cond_param->shape().tuple_shapes()[0].IsToken());
}

TEST_F(InfeedTokenPropagationTest, NonTupleWhile) {
  constexpr std::string_view hlo = R"(
HloModule main

comp {
  ROOT arg.0 = s32[] parameter(0)
  tuple.0 = tuple(arg.0)
  token.0 = after-all()
  outfeed.0 = token[] outfeed(tuple.0, token.0), outfeed_shape=(s32[])
}

cond {
  arg.0 = s32[] parameter(0)
  ROOT true.0 = pred[] constant(true)
}

ENTRY main {
  arg.0 = s32[] parameter(0)
  ROOT while.0 = s32[] while(arg.0), condition=cond, body=comp
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The outfeed output token should have propagated through the loop.
  HloInstruction* loop = FindInstruction(module.get(), "while.0");
  EXPECT_TRUE(loop->shape().IsTuple());
  EXPECT_EQ(loop->shape().tuple_shapes_size(), 2);
  EXPECT_TRUE(loop->shape().tuple_shapes()[1].IsToken());

  // The outfeed input token should have propagated through the loop tuple.
  EXPECT_THAT(loop->operand(0), op::Tuple(op::Parameter(), op::AfterAll()));

  // The outfeed output token should have propagated through the while body
  // root.
  HloComputation* body_comp = FindComputation(module.get(), "comp");
  EXPECT_THAT(body_comp->root_instruction(),
              op::Tuple(op::GetTupleElement(), op::Outfeed()));

  // The outfeed output token should have propagated through the body parameter.
  HloInstruction* body_param = body_comp->parameter_instruction(0);
  EXPECT_EQ(body_param->shape().tuple_shapes_size(), 2);
  EXPECT_TRUE(body_param->shape().tuple_shapes()[1].IsToken());

  // The outfeed output token should have propagated through the condition
  // parameter.
  HloComputation* cond_comp = FindComputation(module.get(), "cond");
  HloInstruction* cond_param = cond_comp->parameter_instruction(0);
  EXPECT_EQ(cond_param->shape().tuple_shapes_size(), 2);
  EXPECT_TRUE(cond_param->shape().tuple_shapes()[1].IsToken());
}

TEST_F(InfeedTokenPropagationTest, NestedInfeedOutfeed) {
  constexpr std::string_view hlo = R"(
HloModule main

true_comp {
  arg.0 = (s32[]) parameter(0)
  token.0 = after-all()
  outfeed.0 = token[] outfeed(arg.0, token.0), outfeed_shape=(s32[])
  ROOT tuple.0 = tuple()
}

false_comp {
  arg.0 = () parameter(0)
  ROOT tuple.0 = tuple()
}

comp {
  arg.0 = () parameter(0)
  token.0 = after-all()
  infeed.0 = (s32[], token[]) infeed(token.0)
  gte.0 = get-tuple-element(infeed.0), index=0
  pred.0 = pred[] constant(true)
  true_tuple.0 = tuple(gte.0)
  false_tuple.0 = tuple()
  ROOT cond.0 = () conditional(pred.0, true_tuple.0, false_tuple.0), true_computation=true_comp, false_computation=false_comp
}

cond {
  arg.0 = () parameter(0)
  ROOT true.0 = pred[] constant(true)
}

ENTRY main {
  while_tuple.0 = tuple()
  ROOT while.0 = () while(while_tuple.0), condition=cond, body=comp
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The infeed and outfeed output tokens should have propagated through the
  // loop.
  HloInstruction* loop = FindInstruction(module.get(), "while.0");
  EXPECT_EQ(loop->shape().tuple_shapes_size(), 2);
  EXPECT_TRUE(loop->shape().tuple_shapes()[0].IsToken());
  EXPECT_TRUE(loop->shape().tuple_shapes()[1].IsToken());

  // The infeed and outfeed input tokens should have propagated through the loop
  // tuple.
  HloInstruction* loop_tuple = FindInstruction(module.get(), "while_tuple.0");
  EXPECT_EQ(loop_tuple->shape().tuple_shapes_size(), 2);
  EXPECT_TRUE(loop_tuple->shape().tuple_shapes()[0].IsToken());
  EXPECT_TRUE(loop_tuple->shape().tuple_shapes()[1].IsToken());

  // The infeed and outfeed output tokens should have propagated through the
  // while body root.
  HloComputation* body_comp = FindComputation(module.get(), "comp");
  EXPECT_THAT(body_comp->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Infeed(), 1),
                        op::GetTupleElement(op::Conditional(), 0)));

  // The outfeed output token should have propagated through the conditional.
  HloInstruction* cond = FindInstruction(module.get(), "cond.0");
  EXPECT_EQ(cond->shape().tuple_shapes_size(), 1);
  EXPECT_TRUE(cond->shape().tuple_shapes()[0].IsToken());

  // The outfeed input token should have propagated through the true tuple.
  HloInstruction* true_tuple = FindInstruction(module.get(), "true_tuple.0");
  EXPECT_EQ(true_tuple->shape().tuple_shapes_size(), 2);
  EXPECT_TRUE(true_tuple->shape().tuple_shapes()[1].IsToken());

  // The outfeed input token should not have propagated through the false tuple.
  HloInstruction* false_tuple = FindInstruction(module.get(), "false_tuple.0");
  EXPECT_EQ(false_tuple->shape().tuple_shapes_size(), 0);

  // The outfeed output token should have propagated through the true
  // computation's root.
  HloComputation* true_comp = FindComputation(module.get(), "true_comp");
  EXPECT_THAT(true_comp->root_instruction(), op::Tuple(op::Outfeed()));

  // The outfeed output token should have propagated to the false computation's
  // root.
  HloComputation* false_comp = FindComputation(module.get(), "false_comp");
  EXPECT_THAT(false_comp->root_instruction(), op::Tuple(op::AfterAll()));
}
}  // namespace
}  // namespace xla

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

#include "xla/hlo/transforms/collectives/infeed_token_propagation.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

class InfeedTokenPropagationTest : public HloHardwareIndependentTestBase {
 protected:
  InfeedTokenPropagationTest() = default;
};

TEST_F(InfeedTokenPropagationTest, EntryComputationInfeed) {
  constexpr absl::string_view kHlo = R"(
HloModule main

ENTRY main {
  token.0 = after-all()
  infeed.0 = (s32[], token[]) infeed(token.0)
  ROOT gte.0 = get-tuple-element(infeed.0), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(InfeedTokenPropagationTest, EntryComputationOutfeed) {
  constexpr absl::string_view kHlo = R"(
HloModule main

ENTRY main {
  arg.0 = s32[] parameter(0)
  tuple.0 = tuple(arg.0)
  token.0 = after-all()
  outfeed.0 = token[] outfeed(tuple.0, token.0), outfeed_shape=(s32[])
  ROOT tuple.1 = tuple()
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(InfeedTokenPropagationTest, ConditionalInfeed) {
  constexpr absl::string_view kHlo = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The infeed output token should have propagated through the conditional.
  HloInstruction* cond = FindInstruction(module.get(), "cond.0");
  EXPECT_EQ(cond->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(cond->shape().tuple_shapes()[0].IsToken());

  // The infeed input token should have propagated through the true tuple.
  HloInstruction* true_tuple = FindInstruction(module.get(), "true_tuple.0");
  EXPECT_EQ(true_tuple->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(true_tuple->shape().tuple_shapes()[0].IsToken());

  // The infeed input token should not have propagated through the false tuple.
  HloInstruction* false_tuple = FindInstruction(module.get(), "false_tuple.0");
  EXPECT_EQ(false_tuple->shape().tuple_shapes().size(), 0);

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
  constexpr absl::string_view kHlo = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The outfeed output token should have propagated through the conditional.
  HloInstruction* cond = FindInstruction(module.get(), "cond.0");
  EXPECT_EQ(cond->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(cond->shape().tuple_shapes()[0].IsToken());

  // The outfeed input token should have propagated through the true tuple.
  HloInstruction* true_tuple = FindInstruction(module.get(), "true_tuple.0");
  EXPECT_EQ(true_tuple->shape().tuple_shapes().size(), 2);
  EXPECT_TRUE(true_tuple->shape().tuple_shapes()[1].IsToken());

  // The outfeed input token should not have propagated through the false tuple.
  HloInstruction* false_tuple = FindInstruction(module.get(), "false_tuple.0");
  EXPECT_EQ(false_tuple->shape().tuple_shapes().size(), 0);

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
  constexpr absl::string_view kHlo = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The infeed output token should have propagated through the conditional.
  HloInstruction* cond = FindInstruction(module.get(), "cond.0");
  EXPECT_EQ(cond->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(cond->shape().tuple_shapes()[0].IsToken());

  // The infeed input token should have propagated through the true tuple.
  const HloInstruction* true_tuple = cond->operand(1);
  EXPECT_EQ(true_tuple->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(true_tuple->shape().tuple_shapes()[0].IsToken());

  // The infeed input token should not have propagated through the false tuple.
  const HloInstruction* false_tuple = cond->operand(2);
  EXPECT_EQ(false_tuple->shape().tuple_shapes().size(), 0);

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
  constexpr absl::string_view kHlo = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The outfeed output token should have propagated through the conditional.
  HloInstruction* cond = FindInstruction(module.get(), "cond.0");
  EXPECT_EQ(cond->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(cond->shape().tuple_shapes()[0].IsToken());

  // The outfeed input token should have propagated through the true tuple.
  HloInstruction* true_tuple = cond->mutable_operand(1);
  EXPECT_TRUE(true_tuple->shape().IsTuple());
  EXPECT_EQ(true_tuple->shape().tuple_shapes().size(), 2);
  EXPECT_TRUE(true_tuple->shape().tuple_shapes()[1].IsToken());

  // The outfeed input token should not have propagated through the false tuple.
  HloInstruction* false_tuple = FindInstruction(module.get(), "false_tuple.0");
  EXPECT_EQ(false_tuple->shape().tuple_shapes().size(), 0);

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
  constexpr absl::string_view kHlo = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The outfeed output token should have propagated through the conditional.
  HloInstruction* cond = FindInstruction(module.get(), "cond.0");
  EXPECT_EQ(cond->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(cond->shape().tuple_shapes()[0].IsToken());

  // The outfeed input token should have propagated through the true tuple.
  HloInstruction* true_tuple = FindInstruction(module.get(), "true_tuple.0");
  EXPECT_EQ(true_tuple->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(true_tuple->shape().tuple_shapes()[0].IsToken());

  // The outfeed input token should not have propagated through the false tuple.
  HloInstruction* false_tuple = FindInstruction(module.get(), "false_tuple.0");
  EXPECT_EQ(false_tuple->shape().tuple_shapes().size(), 0);

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
  constexpr absl::string_view kHlo = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The infeed output token should have propagated through the loop.
  HloInstruction* loop = FindInstruction(module.get(), "while.0");
  EXPECT_EQ(loop->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(loop->shape().tuple_shapes()[0].IsToken());

  // The infeed input token should have propagated through the loop tuple.
  HloInstruction* loop_tuple = FindInstruction(module.get(), "while_tuple.0");
  EXPECT_EQ(loop_tuple->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(loop_tuple->shape().tuple_shapes()[0].IsToken());

  // The infeed output token should have propagated through the while body root.
  HloComputation* body_comp = FindComputation(module.get(), "comp");
  EXPECT_THAT(body_comp->root_instruction(),
              op::Tuple(op::GetTupleElement(op::Infeed(), 1)));

  // The infeed input token should have propagated through the body parameter.
  HloInstruction* body_param = body_comp->parameter_instruction(0);
  EXPECT_EQ(body_param->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(body_param->shape().tuple_shapes()[0].IsToken());

  // The infeed input token should have propagated through the condition
  // parameter.
  HloComputation* cond_comp = FindComputation(module.get(), "cond");
  HloInstruction* cond_param = cond_comp->parameter_instruction(0);
  EXPECT_EQ(cond_param->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(cond_param->shape().tuple_shapes()[0].IsToken());
}

TEST_F(InfeedTokenPropagationTest, WhileOutfeed) {
  constexpr absl::string_view kHlo = R"(
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The outfeed output token should have propagated through the loop.
  HloInstruction* loop = FindInstruction(module.get(), "while.0");
  EXPECT_EQ(loop->shape().tuple_shapes().size(), 2);
  EXPECT_TRUE(loop->shape().tuple_shapes()[1].IsToken());

  // The outfeed input token should have propagated through the loop tuple.
  HloInstruction* loop_tuple = FindInstruction(module.get(), "while_tuple.0");
  EXPECT_EQ(loop_tuple->shape().tuple_shapes().size(), 2);
  EXPECT_TRUE(loop_tuple->shape().tuple_shapes()[1].IsToken());

  // The outfeed output token should have propagated through the while body
  // root.
  HloComputation* body_comp = FindComputation(module.get(), "comp");
  EXPECT_THAT(body_comp->root_instruction(),
              op::Tuple(op::GetTupleElement(), op::Outfeed()));

  // The outfeed output token should have propagated through the body parameter.
  HloInstruction* body_param = body_comp->parameter_instruction(0);
  EXPECT_EQ(body_param->shape().tuple_shapes().size(), 2);
  EXPECT_TRUE(body_param->shape().tuple_shapes()[1].IsToken());

  // The outfeed output token should have propagated through the condition
  // parameter.
  HloComputation* cond_comp = FindComputation(module.get(), "cond");
  HloInstruction* cond_param = cond_comp->parameter_instruction(0);
  EXPECT_EQ(cond_param->shape().tuple_shapes().size(), 2);
  EXPECT_TRUE(cond_param->shape().tuple_shapes()[1].IsToken());
}

TEST_F(InfeedTokenPropagationTest, DisjointWhileOutfeed) {
  constexpr absl::string_view kHlo = R"(
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The outfeed output token should have propagated through the loop.
  HloInstruction* loop = FindInstruction(module.get(), "while.0");
  EXPECT_EQ(loop->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(loop->shape().tuple_shapes()[0].IsToken());

  // The outfeed input token should have propagated through the loop tuple.
  HloInstruction* loop_tuple = FindInstruction(module.get(), "while_tuple.0");
  EXPECT_EQ(loop_tuple->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(loop_tuple->shape().tuple_shapes()[0].IsToken());

  // The outfeed output token should have propagated through the while body
  // root.
  HloComputation* body_comp = FindComputation(module.get(), "comp");
  EXPECT_THAT(body_comp->root_instruction(), op::Tuple(op::Outfeed()));

  // The outfeed output token should have propagated through the body parameter.
  HloInstruction* body_param = body_comp->parameter_instruction(0);
  EXPECT_EQ(body_param->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(body_param->shape().tuple_shapes()[0].IsToken());

  // The outfeed output token should have propagated through the condition
  // parameter.
  HloComputation* cond_comp = FindComputation(module.get(), "cond");
  HloInstruction* cond_param = cond_comp->parameter_instruction(0);
  EXPECT_EQ(cond_param->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(cond_param->shape().tuple_shapes()[0].IsToken());
}

TEST_F(InfeedTokenPropagationTest, NonTupleWhile) {
  constexpr absl::string_view kHlo = R"(
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The outfeed output token should have propagated through the loop.
  HloInstruction* loop = FindInstruction(module.get(), "while.0");
  EXPECT_TRUE(loop->shape().IsTuple());
  EXPECT_EQ(loop->shape().tuple_shapes().size(), 2);
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
  EXPECT_EQ(body_param->shape().tuple_shapes().size(), 2);
  EXPECT_TRUE(body_param->shape().tuple_shapes()[1].IsToken());

  // The outfeed output token should have propagated through the condition
  // parameter.
  HloComputation* cond_comp = FindComputation(module.get(), "cond");
  HloInstruction* cond_param = cond_comp->parameter_instruction(0);
  EXPECT_EQ(cond_param->shape().tuple_shapes().size(), 2);
  EXPECT_TRUE(cond_param->shape().tuple_shapes()[1].IsToken());
}

TEST_F(InfeedTokenPropagationTest, NestedInfeedOutfeed) {
  constexpr absl::string_view kHlo = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The infeed and outfeed output tokens should have propagated through the
  // loop.
  HloInstruction* loop = FindInstruction(module.get(), "while.0");
  EXPECT_EQ(loop->shape().tuple_shapes().size(), 2);
  EXPECT_TRUE(loop->shape().tuple_shapes()[0].IsToken());
  EXPECT_TRUE(loop->shape().tuple_shapes()[1].IsToken());

  // The infeed and outfeed input tokens should have propagated through the loop
  // tuple.
  HloInstruction* loop_tuple = FindInstruction(module.get(), "while_tuple.0");
  EXPECT_EQ(loop_tuple->shape().tuple_shapes().size(), 2);
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
  EXPECT_EQ(cond->shape().tuple_shapes().size(), 1);
  EXPECT_TRUE(cond->shape().tuple_shapes()[0].IsToken());

  // The outfeed input token should have propagated through the true tuple.
  HloInstruction* true_tuple = FindInstruction(module.get(), "true_tuple.0");
  EXPECT_EQ(true_tuple->shape().tuple_shapes().size(), 2);
  EXPECT_TRUE(true_tuple->shape().tuple_shapes()[1].IsToken());

  // The outfeed input token should not have propagated through the false tuple.
  HloInstruction* false_tuple = FindInstruction(module.get(), "false_tuple.0");
  EXPECT_EQ(false_tuple->shape().tuple_shapes().size(), 0);

  // The outfeed output token should have propagated through the true
  // computation's root.
  HloComputation* true_comp = FindComputation(module.get(), "true_comp");
  EXPECT_THAT(true_comp->root_instruction(), op::Tuple(op::Outfeed()));

  // The outfeed output token should have propagated to the false computation's
  // root.
  HloComputation* false_comp = FindComputation(module.get(), "false_comp");
  EXPECT_THAT(false_comp->root_instruction(), op::Tuple(op::AfterAll()));
}

TEST_F(InfeedTokenPropagationTest, WhileNestedAfterInfeed) {
  constexpr absl::string_view kHlo = R"(
HloModule main

body {
  ROOT arg.0 = s32[] parameter(0)
  token.0 = after-all()
  infeed.0 = (s32[], token[]) infeed(token.0)
}

cond {
  arg.0 = s32[] parameter(0)
  ROOT true.0 = pred[] constant(true)
}

ENTRY main {
  token.0 = after-all()
  infeed.0 = (s32[], token[]) infeed(token.0)
  gte.0 = get-tuple-element(infeed.0), index=0
  gte.1 = get-tuple-element(infeed.0), index=1
  infeed.1 = (s32[], token[]) infeed(gte.1)
  gte.2 = get-tuple-element(infeed.1), index=0
  add.0 = add(gte.0, gte.2)
  ROOT while.0 = s32[] while(add.0), body=body, condition=cond
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The second infeed should send its token into the loop.
  HloInstruction* loop = FindInstruction(module.get(), "while.0");
  EXPECT_THAT(loop, op::While(op::Tuple(
                        op::Add(),
                        op::GetTupleElement(op::Infeed(op::GetTupleElement(
                                                op::Infeed(op::AfterAll()), 1)),
                                            1))));
}

TEST_F(InfeedTokenPropagationTest, WhileNestedBeforeOutfeed) {
  constexpr absl::string_view kHlo = R"(
HloModule main

body {
  ROOT arg.0 = s32[] parameter(0)
  token.0 = after-all()
  outfeed.0 = token[] outfeed(arg.0, token.0), outfeed_shape=s32[]
}

cond {
  arg.0 = s32[] parameter(0)
  ROOT true.0 = pred[] constant(true)
}

ENTRY main {
  arg.0 = s32[] parameter(0)
  ROOT while.0 = s32[] while(arg.0), body=body, condition=cond
  token.0 = after-all()
  outfeed.1 = token[] outfeed(while.0, token.0), outfeed_shape=s32[]
  outfeed.2 = token[] outfeed(while.0, outfeed.1), outfeed_shape=s32[] 
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The first outfeed should get its token from the loop.
  // The second outfeed should get its token from the first outfeed.
  HloInstruction* outfeed_2 = FindInstruction(module.get(), "outfeed.2");
  EXPECT_THAT(outfeed_2,
              op::Outfeed(op::GetTupleElement(),
                          op::Outfeed(op::GetTupleElement(),
                                      op::GetTupleElement(op::While(), 1))));
}

TEST_F(InfeedTokenPropagationTest, ConditionalNestedAfterInfeed) {
  constexpr absl::string_view kHlo = R"(
HloModule main

true_comp {
  ROOT arg.0 = (s32[]) parameter(0)
  token.0 = after-all()
  infeed.0 = (s32[], token[]) infeed(token.0)
}

false_comp {
  ROOT arg.0 = (s32[]) parameter(0)
  token.0 = after-all()
  infeed.0 = (s32[], token[]) infeed(token.0)
}

ENTRY main {
  token.0 = after-all()
  infeed.0 = (s32[], token[]) infeed(token.0)
  gte.0 = get-tuple-element(infeed.0), index=0
  gte.1 = get-tuple-element(infeed.0), index=1
  infeed.1 = (s32[], token[]) infeed(gte.1)
  gte.2 = get-tuple-element(infeed.1), index=0
  add.0 = add(gte.0, gte.2)
  tuple.0 = tuple(add.0)
  pred.0 = pred[] constant(true)
  ROOT cond.0 = (s32[]) conditional(pred.0, tuple.0, tuple.0), true_computation=true_comp, false_computation=false_comp
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The conditional should get both its tokens from the second infeed.
  // The second infeed should get its token from the first infeed.
  HloInstruction* conditional = FindInstruction(module.get(), "cond.0");
  EXPECT_THAT(conditional,
              op::Conditional(
                  op::Constant(),
                  op::Tuple(op::Add(), op::GetTupleElement(
                                           op::Infeed(op::GetTupleElement(
                                               op::Infeed(op::AfterAll()), 1)),
                                           1)),
                  op::Tuple(op::Add(), op::GetTupleElement(
                                           op::Infeed(op::GetTupleElement(
                                               op::Infeed(op::AfterAll()), 1)),
                                           1))));
}

TEST_F(InfeedTokenPropagationTest, ConditionalNestedBeforeOutfeed) {
  constexpr absl::string_view kHlo = R"(
HloModule main

true_comp {
  ROOT arg.0 = (s32[]) parameter(0)
  token.0 = after-all()
  gte.0 = get-tuple-element(arg.0), index=0
  outfeed.0 = token[] outfeed(gte.0, token.0), outfeed_shape=s32[]
}

false_comp {
  ROOT arg.0 = (s32[]) parameter(0)
  token.0 = after-all()
  gte.0 = get-tuple-element(arg.0), index=0
  outfeed.1 = token[] outfeed(gte.0, token.0), outfeed_shape=s32[]
}

ENTRY main {
  arg.0 = s32[] parameter(0)
  tuple.0 = tuple(arg.0)
  pred.0 = pred[] constant(true)
  ROOT cond.0 = (s32[]) conditional(pred.0, tuple.0, tuple.0), true_computation=true_comp, false_computation=false_comp
  gte.0 = get-tuple-element(cond.0), index=0
  token.0 = after-all()
  outfeed.2 = token[] outfeed(gte.0, token.0), outfeed_shape=s32[]
  outfeed.3 = token[] outfeed(gte.0, outfeed.2), outfeed_shape=s32[]
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The second outfeed should get its token from the first outfeed.
  // The first outfeed should get its token from the conditional.
  // Note, there is a quirk - each branch of the of the conditional will produce
  // its own token, but the first outfeed can only consume one of those.
  // I'm not certain if we deterministically will consume last token in the
  // conditional result.
  HloInstruction* outfeed_3 = FindInstruction(module.get(), "outfeed.3");
  EXPECT_THAT(
      outfeed_3,
      op::Outfeed(op::GetTupleElement(),
                  op::Outfeed(op::GetTupleElement(),
                              op::GetTupleElement(op::Conditional(), 2))));
}

TEST_F(InfeedTokenPropagationTest, ConditionalMixedInfeed) {
  constexpr absl::string_view kHlo = R"(
HloModule main

true_comp {
  arg.0 = () parameter(0)
  token.0 = after-all()
  host_infeed.0 = (s32[], token[]) infeed(token.0)
  ROOT tuple.0 = tuple()
}

false_comp {
  arg.0 = () parameter(0)
  ROOT tuple.0 = tuple()
}

ENTRY main {
  token.0 = after-all()
  core_infeed.0 = ((), token[]) infeed(token.0), infeed_config="core"
  pred.0 = pred[] constant(true)
  true_tuple.0 = tuple()
  false_tuple.0 = tuple()
  ROOT cond.0 = () conditional(pred.0, true_tuple.0, false_tuple.0), true_computation=true_comp, false_computation=false_comp
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The core infeed and host infeed should not be connected.
  DependencyHloOrdering ordering(module.get());
  HloInstruction* core_infeed = FindInstruction(module.get(), "core_infeed.0");
  HloInstruction* host_infeed = FindInstruction(module.get(), "host_infeed.0");
  EXPECT_EQ(ordering.GetExecutionConstraint(core_infeed, host_infeed),
            HloOrdering::ExecutionConstraint::kUnordered);
}

TEST_F(InfeedTokenPropagationTest, ConditionalMixedOutfeed) {
  constexpr absl::string_view kHlo = R"(
HloModule main

true_comp {
  arg.0 = s32[] parameter(0)
  token.0 = after-all()
  host_outfeed.0 = token[] outfeed(arg.0, token.0), outfeed_shape=s32[]
  ROOT tuple.0 = tuple()
}

false_comp {
  arg.0 = s32[] parameter(0)
  ROOT tuple.0 = tuple()
}

ENTRY main {
  arg.0 = s32[] parameter(0)
  token.0 = after-all()
  core_outfeed.0 = token[] outfeed(arg.0, token.0), outfeed_shape=s32[], outfeed_config="core"
  pred.0 = pred[] constant(true)
  ROOT cond.0 = () conditional(pred.0, arg.0, arg.0), true_computation=true_comp, false_computation=false_comp
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The core outfeed and host outfeed should not be connected.
  DependencyHloOrdering ordering(module.get());
  HloInstruction* core_outfeed =
      FindInstruction(module.get(), "core_outfeed.0");
  HloInstruction* host_outfeed =
      FindInstruction(module.get(), "host_outfeed.0");
  EXPECT_EQ(ordering.GetExecutionConstraint(core_outfeed, host_outfeed),
            HloOrdering::ExecutionConstraint::kUnordered);
}

TEST_F(InfeedTokenPropagationTest, WhileMixedInfeed) {
  constexpr absl::string_view kHlo = R"(
HloModule main

comp {
  arg.0 = () parameter(0)
  token.0 = after-all()
  host_infeed.0 = (s32[], token[]) infeed(token.0)
  ROOT tuple.0 = tuple()
}

cond {
  arg.0 = () parameter(0)
  ROOT true.0 = pred[] constant(true)
}

ENTRY main {
  token.0 = after-all()
  core_infeed.0 = ((), token[]) infeed(token.0), infeed_config="core"
  while_tuple.0 = tuple()
  ROOT while.0 = () while(while_tuple.0), condition=cond, body=comp
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The core infeed and host infeed should not be connected.
  DependencyHloOrdering ordering(module.get());
  HloInstruction* core_infeed = FindInstruction(module.get(), "core_infeed.0");
  HloInstruction* host_infeed = FindInstruction(module.get(), "host_infeed.0");
  EXPECT_EQ(ordering.GetExecutionConstraint(core_infeed, host_infeed),
            HloOrdering::ExecutionConstraint::kUnordered);
}

TEST_F(InfeedTokenPropagationTest, WhileMixedOutfeed) {
  constexpr absl::string_view kHlo = R"(
HloModule main

comp {
  arg.0 = (s32[]) parameter(0)
  token.0 = after-all()
  host_outfeed.0 = token[] outfeed(arg.0, token.0), outfeed_shape=(s32[])
  gte.0 = get-tuple-element(arg.0), index=0
  ROOT tuple.0 = tuple(gte.0)
}

cond {
  arg.0 = (s32[]) parameter(0)
  ROOT true.0 = pred[] constant(true)
}

ENTRY main {
  arg.0 = s32[] parameter(0)
  token.0 = after-all()
  core_outfeed.0 = token[] outfeed(arg.0, token.0), outfeed_shape=s32[], outfeed_config="core"
  while_tuple.0 = tuple(arg.0)
  ROOT while.0 = (s32[]) while(while_tuple.0), condition=cond, body=comp
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The core outfeed and host outfeed should not be connected.
  DependencyHloOrdering ordering(module.get());
  HloInstruction* core_outfeed =
      FindInstruction(module.get(), "core_outfeed.0");
  HloInstruction* host_outfeed =
      FindInstruction(module.get(), "host_outfeed.0");
  EXPECT_EQ(ordering.GetExecutionConstraint(core_outfeed, host_outfeed),
            HloOrdering::ExecutionConstraint::kUnordered);
}

TEST_F(InfeedTokenPropagationTest, ConditionalSharding) {
  constexpr absl::string_view kHlo = R"(
HloModule main

true_comp {
  arg.0 = s32[] parameter(0), sharding={replicated}
  token.0 = after-all()
  infeed.0 = token[] outfeed(arg.0, token.0), outfeed_shape=s32[]
  ROOT tuple.0 = tuple()
}

false_comp {
  arg.0 = s32[] parameter(0), sharding={replicated}
  ROOT tuple.0 = tuple()
}

ENTRY main {
  arg.0 = s32[] parameter(0), sharding={replicated}
  pred.0 = pred[] constant(true)
  ROOT cond.0 = () conditional(pred.0, arg.0, arg.0), true_computation=true_comp, false_computation=false_comp
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The parameter should have its original sharding, and sharding for the
  // appended token.
  HloComputation* true_comp = FindComputation(module.get(), "true_comp");
  HloInstruction* true_arg = true_comp->parameter_instruction(0);
  ASSERT_TRUE(true_arg->has_sharding());
  EXPECT_TRUE(true_arg->sharding().IsTuple());
  EXPECT_EQ(true_arg->sharding().tuple_elements().size(), 2);
  // Token can have arbitrary sharding, so we don't check it.
  EXPECT_TRUE(true_arg->sharding().tuple_elements()[0].IsReplicated());
}

TEST_F(InfeedTokenPropagationTest, WhileSharding) {
  constexpr absl::string_view kHlo = R"(
HloModule main

body {
  ROOT arg.0 = s32[] parameter(0), sharding={replicated}
  token.0 = after-all()
  outfeed.0 = token[] outfeed(arg.0, token.0), outfeed_shape=s32[]
}

cond {
  arg.0 = s32[] parameter(0), sharding={replicated}
  ROOT true.0 = pred[] constant(true)
}

ENTRY main {
  arg.0 = s32[] parameter(0)
  ROOT while.0 = s32[] while(arg.0), condition=cond, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  InfeedTokenPropagation itp;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, itp.Run(module.get()));
  EXPECT_TRUE(changed);

  // The parameter should have its original sharding, and sharding for the
  // appended token.
  HloComputation* body_comp = FindComputation(module.get(), "body");
  HloInstruction* body_arg = body_comp->parameter_instruction(0);
  EXPECT_TRUE(body_arg->sharding().IsTuple());
  EXPECT_EQ(body_arg->sharding().tuple_elements().size(), 2);
  // Token can have arbitrary sharding, so we don't check it.
  EXPECT_TRUE(body_arg->sharding().tuple_elements()[0].IsReplicated());

  // All same for condition.
  HloComputation* cond_comp = FindComputation(module.get(), "cond");
  HloInstruction* cond_arg = cond_comp->parameter_instruction(0);
  EXPECT_TRUE(cond_arg->sharding().IsTuple());
  EXPECT_EQ(cond_arg->sharding().tuple_elements().size(), 2);
  EXPECT_TRUE(cond_arg->sharding().tuple_elements()[0].IsReplicated());
}
}  // namespace
}  // namespace xla

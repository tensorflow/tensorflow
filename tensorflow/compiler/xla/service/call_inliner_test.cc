/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/call_inliner.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

// Tests for call inlining that are most tractable at the HLO level (vs
// ComputationBuilder API in call_test.cc).
using CallInlinerTest = HloTestBase;

TEST_F(CallInlinerTest, ControlDependenciesAreCarriedToCaller) {
  // "inner" computation just has a control dependency from the "zero" value to
  // the "one" value.
  HloComputation::Builder inner(TestName() + ".inner");
  HloInstruction* zero = inner.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(24.0f)));
  HloInstruction* one = inner.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  TF_ASSERT_OK(zero->AddControlDependencyTo(one));
  auto module = CreateNewModule();
  HloComputation* inner_computation =
      module->AddEmbeddedComputation(inner.Build());

  // "outer" computation just calls the "inner" computation.
  HloComputation::Builder outer(TestName() + ".outer");
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  outer.AddInstruction(
      HloInstruction::CreateCall(r0f32, {}, inner_computation));

  auto computation = module->AddEntryComputation(outer.Build());

  CallInliner call_inliner;
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_inliner.Run(module.get()));
  ASSERT_TRUE(mutated);
  EXPECT_THAT(computation->root_instruction(), op::Constant());
  EXPECT_EQ(computation->root_instruction()->literal().GetFirstElement<float>(),
            42);
  ASSERT_EQ(1, computation->root_instruction()->control_predecessors().size());
  auto prior = computation->root_instruction()->control_predecessors()[0];
  EXPECT_THAT(prior, op::Constant());
  EXPECT_EQ(prior->literal().GetFirstElement<float>(), 24);
}

// Tests for referential transparency (a function that calls a function that
// returns false should be identical to just returning false).
TEST_F(CallInlinerTest, CallsWithinWhileBodiesAreInlined) {
  const Shape pred = ShapeUtil::MakeShape(PRED, {});
  auto module = CreateNewModule();

  // Create a lambda that calls a function that returns the false predicate.
  // Note we also use this lambda twice by reference, just to make the test a
  // little trickier.
  HloComputation::Builder just_false(TestName() + ".false");
  just_false.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  HloComputation* false_computation =
      module->AddEmbeddedComputation(just_false.Build());

  HloComputation::Builder call_false_builder(TestName() + ".call_false");
  call_false_builder.AddInstruction(
      HloInstruction::CreateCall(pred, {}, false_computation));
  HloComputation* call_false =
      module->AddEmbeddedComputation(call_false_builder.Build());

  HloComputation::Builder outer(TestName() + ".outer");
  HloInstruction* init_value = outer.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  outer.AddInstruction(
      HloInstruction::CreateWhile(pred, call_false, call_false, init_value));

  auto computation = module->AddEntryComputation(outer.Build());

  CallInliner call_inliner;
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_inliner.Run(module.get()));
  ASSERT_TRUE(mutated);
  EXPECT_THAT(
      computation->root_instruction()->while_condition()->root_instruction(),
      op::Constant());
  EXPECT_THAT(computation->root_instruction()->while_body()->root_instruction(),
              op::Constant());
}

// Check CallInliner::Inline, which inlines a specific call without running the
// whole pass.
TEST_F(CallInlinerTest, InlineWithoutRunningPass) {
  const Shape pred = ShapeUtil::MakeShape(PRED, {});
  auto module = CreateNewModule();

  HloComputation::Builder just_false(TestName() + ".false");
  auto* true_constant = just_false.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<bool>({true})));
  auto* false_constant = just_false.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  TF_ASSERT_OK(false_constant->AddControlDependencyTo(true_constant));
  HloComputation* false_computation =
      module->AddEmbeddedComputation(just_false.Build());

  HloComputation::Builder call_false_builder(TestName() + ".call_false");
  HloInstruction* call = call_false_builder.AddInstruction(
      HloInstruction::CreateCall(pred, {}, false_computation));
  auto computation = module->AddEntryComputation(call_false_builder.Build());

  TF_ASSERT_OK(CallInliner::Inline(call).status());
  EXPECT_THAT(computation->root_instruction(), op::Constant());
  EXPECT_THAT(computation->root_instruction()->control_successors(),
              ElementsAre(op::Constant()));
}

TEST_F(CallInlinerTest, CallToOutfeedComputationIsInlined) {
  const Shape f32 = ShapeUtil::MakeShape(F32, {});
  auto module = CreateNewModule();

  HloComputation::Builder outfeeder(TestName() + ".outfeeder");
  auto value = outfeeder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0)));
  auto token = outfeeder.AddInstruction(HloInstruction::CreateAfterAll({}));
  outfeeder.AddInstruction(
      HloInstruction::CreateOutfeed(f32, value, token, /*outfeed_config=*/""));

  auto outfeed_computation = module->AddEmbeddedComputation(outfeeder.Build());

  HloComputation::Builder outer(TestName() + ".outer");
  outer.AddInstruction(HloInstruction::CreateCall(
      outfeed_computation->root_instruction()->shape(), /*operands=*/{},
      outfeed_computation));

  module->AddEntryComputation(outer.Build());

  CallInliner call_inliner;
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_inliner.Run(module.get()));
  ASSERT_TRUE(mutated);
}

}  // namespace
}  // namespace xla

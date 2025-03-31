/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/call_inliner.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(24.0f)));
  HloInstruction* one = inner.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  TF_ASSERT_OK(zero->AddControlDependencyTo(one));
  auto module = CreateNewVerifiedModule();
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
  auto module = CreateNewVerifiedModule();

  // Create a lambda that calls a function that returns the false predicate.
  // Note we also use this lambda twice by reference, just to make the test a
  // little trickier.
  HloComputation::Builder just_false(TestName() + ".false");
  just_false.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* false_computation =
      module->AddEmbeddedComputation(just_false.Build());

  HloComputation::Builder call_false_builder(TestName() + ".call_false");
  call_false_builder.AddInstruction(
      HloInstruction::CreateParameter(0, pred, "param"));
  call_false_builder.AddInstruction(
      HloInstruction::CreateCall(pred, {}, false_computation));
  HloComputation* call_false =
      module->AddEmbeddedComputation(call_false_builder.Build());

  HloComputation::Builder outer(TestName() + ".outer");
  HloInstruction* init_value = outer.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
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
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder just_false(TestName() + ".false");
  auto* true_constant = just_false.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<bool>({true})));
  auto* false_constant = just_false.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
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

// Test that inlining can work with computations with dead parameter.
TEST_F(CallInlinerTest, InlineWithEmptyComputation) {
  const Shape pred = ShapeUtil::MakeShape(PRED, {});
  auto module = CreateNewVerifiedModule();
  Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  HloComputation::Builder empty(TestName() + ".empty");
  empty.AddInstruction(HloInstruction::CreateParameter(0, r0s32, "A"));
  empty.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  HloComputation* empty_computation =
      module->AddEmbeddedComputation(empty.Build());

  HloComputation::Builder empty2(TestName() + ".empty");
  empty2.AddInstruction(HloInstruction::CreateParameter(0, r0s32, "A"));
  empty2.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  HloComputation* empty2_computation =
      module->AddEmbeddedComputation(empty2.Build());

  HloComputation::Builder entry("entry");
  auto zero = entry.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  // The order of the call chain are crafted to test a specific pattern such
  // that the third call instruction will be flattened before the second one
  // (which makes the second call instruction dead before it is flattened).
  entry.AddInstruction(
      HloInstruction::CreateCall(r0s32, {zero}, empty_computation));
  HloInstruction* call1 = entry.AddInstruction(
      HloInstruction::CreateCall(r0s32, {zero}, empty2_computation));
  entry.AddInstruction(
      HloInstruction::CreateCall(r0s32, {call1}, empty_computation));
  auto computation = module->AddEntryComputation(entry.Build());

  CallInliner call_inliner;
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_inliner.Run(module.get()));
  ASSERT_TRUE(mutated);

  EXPECT_THAT(computation->root_instruction(), op::Constant());
}

TEST_F(CallInlinerTest, CallToOutfeedComputationIsInlined) {
  const Shape f32 = ShapeUtil::MakeShape(F32, {});
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder outfeeder(TestName() + ".outfeeder");
  auto value = outfeeder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));
  auto token = outfeeder.AddInstruction(HloInstruction::CreateToken());
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

TEST_F(CallInlinerTest, InlineSingleUseCalleesOnly) {
  const absl::string_view hlo_string = R"(
  HloModule inline_module

  a {
    ROOT tuple = () tuple()
  }

  b {
    ROOT tuple.1 = () tuple()
  }

  ENTRY inline {
    a = () call(), to_apply=a
    b = () call(), to_apply=a
    c = () call(), to_apply=b
    ROOT tuple = ((), (), ()) tuple(a, b, c)
  })";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  CallInliner call_inliner(/*single_call_site=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_inliner.Run(module.get()));
  ASSERT_TRUE(mutated);

  ASSERT_EQ(module->entry_computation()->instruction_count(), 4);
  auto inst = module->entry_computation()->instructions().begin();
  EXPECT_THAT(*inst, op::Call());
  ++inst;
  EXPECT_THAT(*inst, op::Call());
  ++inst;
  EXPECT_THAT(*inst, op::Tuple());
  ++inst;
  EXPECT_THAT(*inst, op::Tuple());
}

// Tests whether the call inliner respects the execution thread filter.
// The HLO module has four chained computations split in two threads:
// entry_main_thread_outer -> main_thread_inner -> secondary_thread_outer ->
//   secondary_thread_inner.
// This test runs call inliner twice. First, across all threads with the
// following expected result: entry_main_thread_outer -> secondary_thread_outer.
// Second, on the secondary thread only with the following expected result:
// entry_main_thread_outer -> main_thread_inner -> secondary_thread_outer.
TEST_F(CallInlinerTest, InliningPerformedInsideSpecifiedThreadsOnly) {
  const std::string hlo_string = R"(
HloModule inline_specified_threads_only

%secondary_inner () -> u32[] {
  ROOT %co.2 = u32[] constant(2)
}, execution_thread="secondary_thread"

%secondary_outer () -> u32[] {
  %co.1 = u32[] constant(1)
  %call.1 = u32[] call(), to_apply=%secondary_inner
  ROOT %add.1 = add(%co.1, %call.1)
}, execution_thread="secondary_thread"

%main_inner () -> u32[] {
  %co.0 = u32[] constant(0)
  %async-start = ((), u32[], u32[]) call-start(), async_execution_thread="secondary_thread", to_apply=secondary_outer
  %async-done = u32[] call-done(((), u32[], u32[]) %async-start)
  ROOT %add.2 = add(%co.0, %async-done)
}

ENTRY %main_outer (p0: u32[]) -> u32[] {
  %p.0 = u32[] parameter(0)
  %call.0 = u32[] call(), to_apply=%main_inner
  ROOT %add.3 = add(%p.0, %call.0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  auto module_clone = module->Clone(/*suffix=*/"");

  // When we don't restrict the CallInliner to any one thread, we expect that
  // both the secondary and main thread calls are inlined.
  {
    VLOG(1) << "Module BEFORE CallInliner\n" << module->ToString();

    CallInliner call_inliner;
    TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_inliner.Run(module.get()));
    VLOG(1) << "Module AFTER CallInliner\n" << module->ToString();
    EXPECT_TRUE(mutated);

    EXPECT_THAT(
        module->entry_computation()->root_instruction(),
        op::Add(op::Parameter(0),
                op::Add(op::Constant(LiteralUtil::CreateR0<uint32_t>(0)),
                        op::AsyncDone())));
    EXPECT_THAT(module->entry_computation()
                    ->root_instruction()
                    ->operand(1)
                    ->operand(1)
                    ->async_wrapped_instruction()
                    ->called_computations()
                    .at(0)
                    ->root_instruction(),
                op::Add(op::Constant(LiteralUtil::CreateR0<uint32_t>(1)),
                        op::Constant(LiteralUtil::CreateR0<uint32_t>(2))));
  }
  // When we restrict the CallInliner to the secondary thread, we expect that
  // the secondary thread calls get inlined and main thread calls do not get
  // inlined.
  VLOG(1) << "Restricting CallInliner to the secondary thread.";
  {
    CallInliner call_inliner;
    TF_ASSERT_OK_AND_ASSIGN(
        bool mutated,
        call_inliner.Run(module_clone.get(), {"secondary_thread"}));
    VLOG(1) << "Module AFTER CallInliner\n" << module_clone->ToString();
    EXPECT_TRUE(mutated);

    EXPECT_THAT(module_clone->entry_computation()->root_instruction(),
                op::Add(op::Parameter(0), op::Call()));
    EXPECT_THAT(module_clone->entry_computation()
                    ->root_instruction()
                    ->operand(1)
                    ->called_computations()
                    .at(0)
                    ->root_instruction(),
                op::Add(op::Constant(LiteralUtil::CreateR0<uint32_t>(0)),
                        op::AsyncDone()));
    EXPECT_THAT(module_clone->entry_computation()
                    ->root_instruction()
                    ->operand(1)
                    ->called_computations()
                    .at(0)
                    ->root_instruction()
                    ->operand(1)
                    ->async_wrapped_instruction()
                    ->called_computations()
                    .at(0)
                    ->root_instruction(),
                op::Add(op::Constant(LiteralUtil::CreateR0<uint32_t>(1)),
                        op::Constant(LiteralUtil::CreateR0<uint32_t>(2))));
  }
}

TEST_F(CallInlinerTest, InlineCompositeCall) {
  const absl::string_view hlo_string = R"(
  HloModule composite

  %add (lhs: f32[]) -> f32[] {
    %lhs = f32[] parameter(0)
    %rhs = f32[] constant(2)
    ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
  }

  ENTRY %main () -> f32[] {
    %lhs = f32[] constant(42)
    ROOT %call = f32[] call(f32[] %lhs), to_apply=%add, is_composite=true, frontend_attributes={composite.attributes={n = 1 : i32, tensor = dense<1> : tensor<i32>},composite.name="foo.bar",composite.version="1"}
  })";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  CallInliner call_inliner(/*single_call_site=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_inliner.Run(module.get()));
  ASSERT_TRUE(mutated);

  ASSERT_EQ(module->entry_computation()->instruction_count(), 3);
  auto inst = module->entry_computation()->instructions().begin();
  EXPECT_THAT(*inst, op::Constant());
  ++inst;
  EXPECT_THAT(*inst, op::Constant());
  ++inst;
  EXPECT_THAT(*inst, op::Add());
  EXPECT_TRUE((*inst)->frontend_attributes().map().empty());
}

TEST_F(CallInlinerTest, PreserveCompositeCall) {
  const absl::string_view hlo_string = R"(
  HloModule composite

  %add (lhs: f32[]) -> f32[] {
    %lhs = f32[] parameter(0)
    %rhs = f32[] constant(2)
    ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
  }

  ENTRY %main () -> f32[] {
    %lhs = f32[] constant(42)
    ROOT %call = f32[] call(f32[] %lhs), to_apply=%add, is_composite=true, frontend_attributes={composite.attributes={n = 1 : i32, tensor = dense<1> : tensor<i32>},composite.name="foo.bar",composite.version="1"}
  })";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  CallInliner call_inliner(
      /*single_call_site=*/true, /*update_domain=*/false,
      /*composites_to_preserve=*/{"foo.bar"});
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_inliner.Run(module.get()));
  ASSERT_FALSE(mutated);

  auto inst = module->entry_computation()->instructions().begin();
  EXPECT_THAT(*inst, op::Constant());
  ++inst;
  EXPECT_THAT(*inst, op::Call());
  EXPECT_FALSE((*inst)->frontend_attributes().map().empty());
}

TEST_F(CallInlinerTest, DontInlineCallWithAttributeInlineableFalse) {
  const char* const hloString = R"(
    HloModule jit_f, entry_computation_layout={(f32[8,8]{1,0})->f32[8,8]{1,0}}
    %test (Arg_0.5: f32[1,8]) -> f32[1,8] {
      %Arg_0.5 = f32[1,8]{1,0} parameter(0)
      ROOT %add.6 = f32[1,8]{1,0} add(f32[1,8]{1,0} %Arg_0.5, f32[1,8]{1,0} %Arg_0.5), metadata={source_file="-" source_line=11}
    }
    ENTRY %main.10 (Arg_0.1: f32[8,8]) -> f32[8,8] {
      %Arg_0.1 = f32[8,8]{1,0} parameter(0)
      %custom-call.3 = f32[1,8]{1,0} custom-call(f32[8,8]{1,0} %Arg_0.1), custom_call_target="SPMDFullToShardShape", sharding={manual}, metadata={source_file="-" source_line=4}
      %call.7 = f32[1,8]{1,0} call(f32[1,8]{1,0} %custom-call.3), to_apply=%test, frontend_attributes={inlineable="false"}
      ROOT %custom-call.9 = f32[8,8]{1,0} custom-call(f32[1,8]{1,0} %call.7), custom_call_target="SPMDShardToFullShape", sharding={devices=[8,1]<=[8]}, metadata={source_file="-" source_line=7}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hloString));
  module->mutable_config().set_use_shardy_partitioner(true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, CallInliner().Run(module.get()))
  // The single call in the module is not inlined.
  EXPECT_FALSE(changed);

  HloInstruction* call = FindInstruction(module.get(), xla::HloOpcode::kCall);
  EXPECT_NE(call, nullptr);
  EXPECT_TRUE(call->has_to_apply());
  EXPECT_EQ(call->to_apply()->name(), "test");
}

TEST_F(CallInlinerTest, UseShardyMhloToHloShmapBodyNotInlined) {
  const char* const hloString = R"(
    HloModule jit_f, entry_computation_layout={(f32[8,8]{1,0})->f32[8,8]{1,0}}

    %prefix_shmap_body_suffix.4 (Arg_0.5: f32[1,8]) -> f32[1,8] {
      %Arg_0.5 = f32[1,8]{1,0} parameter(0)
      ROOT %add.6 = f32[1,8]{1,0} add(f32[1,8]{1,0} %Arg_0.5, f32[1,8]{1,0} %Arg_0.5), metadata={source_file="-" source_line=11}
    }

    ENTRY %main.10 (Arg_0.1: f32[8,8]) -> f32[8,8] {
      %Arg_0.1 = f32[8,8]{1,0} parameter(0)
      %custom-call.2 = f32[8,8]{1,0} custom-call(f32[8,8]{1,0} %Arg_0.1), custom_call_target="Sharding", sharding={devices=[8,1]<=[8]}, metadata={source_file="-" source_line=3}
      %custom-call.3 = f32[1,8]{1,0} custom-call(f32[8,8]{1,0} %custom-call.2), custom_call_target="SPMDFullToShardShape", sharding={manual}, metadata={source_file="-" source_line=4}
      %call.7 = f32[1,8]{1,0} call(f32[1,8]{1,0} %custom-call.3), to_apply=%prefix_shmap_body_suffix.4
      %custom-call.8 = f32[1,8]{1,0} custom-call(f32[1,8]{1,0} %call.7), custom_call_target="Sharding", sharding={manual}, metadata={source_file="-" source_line=6}
      ROOT %custom-call.9 = f32[8,8]{1,0} custom-call(f32[1,8]{1,0} %custom-call.8), custom_call_target="SPMDShardToFullShape", sharding={devices=[8,1]<=[8]}, metadata={source_file="-" source_line=7}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hloString));
  module->mutable_config().set_use_shardy_partitioner(true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, CallInliner().Run(module.get()));
  VLOG(1) << module->ToString();
  // The single call in the module is not inlined.
  EXPECT_FALSE(changed);

  HloInstruction* call = FindInstruction(module.get(), xla::HloOpcode::kCall);
  EXPECT_NE(call, nullptr);
  EXPECT_TRUE(call->has_to_apply());
  EXPECT_EQ(call->to_apply()->name(), "prefix_shmap_body_suffix.4");
}

// Don't inline when the name starts with "xla.sdy.manual_computation_body".
TEST_F(CallInlinerTest, UseShardManualComputationBodyNotInlined) {
  const char* const hloString = R"(
    HloModule jit_f, entry_computation_layout={(f32[8,8]{1,0})->f32[8,8]{1,0}}

    %xla.sdy.manual_computation_body.4 (Arg_0.5: f32[1,8]) -> f32[1,8] {
      %Arg_0.5 = f32[1,8]{1,0} parameter(0)
      ROOT %add.6 = f32[1,8]{1,0} add(f32[1,8]{1,0} %Arg_0.5, f32[1,8]{1,0} %Arg_0.5), metadata={source_file="-" source_line=11}
    }

    ENTRY %main.10 (Arg_0.1: f32[8,8]) -> f32[8,8] {
      %Arg_0.1 = f32[8,8]{1,0} parameter(0)
      %custom-call.3 = f32[1,8]{1,0} custom-call(f32[8,8]{1,0} %Arg_0.1), custom_call_target="SPMDFullToShardShape", sharding={manual}, metadata={source_file="-" source_line=4}
      %call.7 = f32[1,8]{1,0} call(f32[1,8]{1,0} %custom-call.3), to_apply=%xla.sdy.manual_computation_body.4
      ROOT %custom-call.9 = f32[8,8]{1,0} custom-call(f32[1,8]{1,0} %call.7), custom_call_target="SPMDShardToFullShape", sharding={devices=[8,1]<=[8]}, metadata={source_file="-" source_line=7}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hloString));
  module->mutable_config().set_use_shardy_partitioner(true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, CallInliner().Run(module.get()));
  VLOG(1) << module->ToString();
  // The single call in the module is not inlined.
  EXPECT_FALSE(changed);

  HloInstruction* call = FindInstruction(module.get(), xla::HloOpcode::kCall);
  EXPECT_NE(call, nullptr);
  EXPECT_TRUE(call->has_to_apply());
  EXPECT_EQ(call->to_apply()->name(), "xla.sdy.manual_computation_body.4");
}

// Make sure we check the name of the called function contains the string, not
// just the prefix/suffix.
TEST_F(CallInlinerTest, UseShardManualComputationBodySurroundedNotInlined) {
  const char* const hloString = R"(
    HloModule jit_f, entry_computation_layout={(f32[8,8]{1,0})->f32[8,8]{1,0}}

    %my_model.___call__.fwd.xla.sdy.manual_computation_body_14.1234 (Arg_0.5: f32[1,8]) -> f32[1,8] {
      %Arg_0.5 = f32[1,8]{1,0} parameter(0)
      ROOT %add.6 = f32[1,8]{1,0} add(f32[1,8]{1,0} %Arg_0.5, f32[1,8]{1,0} %Arg_0.5), metadata={source_file="-" source_line=11}
    }

    ENTRY %main.10 (Arg_0.1: f32[8,8]) -> f32[8,8] {
      %Arg_0.1 = f32[8,8]{1,0} parameter(0)
      %custom-call.3 = f32[1,8]{1,0} custom-call(f32[8,8]{1,0} %Arg_0.1), custom_call_target="SPMDFullToShardShape", sharding={manual}, metadata={source_file="-" source_line=4}
      %call.7 = f32[1,8]{1,0} call(f32[1,8]{1,0} %custom-call.3), to_apply=%my_model.___call__.fwd.xla.sdy.manual_computation_body_14.1234
      ROOT %custom-call.9 = f32[8,8]{1,0} custom-call(f32[1,8]{1,0} %call.7), custom_call_target="SPMDShardToFullShape", sharding={devices=[8,1]<=[8]}, metadata={source_file="-" source_line=7}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hloString));
  module->mutable_config().set_use_shardy_partitioner(true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, CallInliner().Run(module.get()))
  // The single call in the module is not inlined.
  EXPECT_FALSE(changed);

  HloInstruction* call = FindInstruction(module.get(), xla::HloOpcode::kCall);
  EXPECT_NE(call, nullptr);
  EXPECT_TRUE(call->has_to_apply());
  EXPECT_EQ(call->to_apply()->name(),
            "my_model.___call__.fwd.xla.sdy.manual_computation_body_14.1234");
}

TEST_F(CallInlinerTest, ControlDepsPropagateToRootOfInlinedInstructions) {
  const char* hlo = R"(
  HloModule test

  bar {
    p0 = s32[] parameter(0)
    add = s32[] add(p0, s32[] constant(2))
    ROOT res = s32[] subtract(add, s32[] constant(3))
  }

  ENTRY main {
    p0 = s32[] parameter(0)
    p1 = s32[] parameter(1)
    p2 = s32[] parameter(2)
    call1 = s32[] custom-call(p0), custom_call_target="foo"
    call2 = s32[] call(p1), to_apply=bar, control-predecessors={call1}
    call3 = s32[] custom-call(p2), custom_call_target="baz", control-predecessors={call2}
    ROOT res = (s32[], s32[], s32[]) tuple(call1, call2, call3)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));
  CallInliner call_inliner;
  TF_ASSERT_OK_AND_ASSIGN(bool mutated, call_inliner.Run(m.get()));
  EXPECT_TRUE(mutated);
  TF_ASSERT_OK_AND_ASSIGN(
      bool filecheck_result,
      RunFileCheck(m->ToString(HloPrintOptions{}
                                   .set_print_result_shape(false)
                                   .set_print_operand_shape(false)),
                   R"(
  // CHECK: ENTRY %main
  // CHECK-DAG: %[[call1:.+]] = custom-call({{.+}}), custom_call_target="foo"
  // CHECK-DAG: %[[p1:.+]] = parameter(1)
  // CHECK-DAG: %[[c2:.+]] = constant(2), control-predecessors={%[[call1]]}
  // CHECK-DAG: %[[add:.+]] = add(%[[p1]], %[[c2]]), control-predecessors={%[[call1]]}
  // CHECK-DAG: %[[c3:.+]] = constant(3), control-predecessors={%[[call1]]}
  // CHECK-DAG: %[[res:.+]] = subtract(%[[add]], %[[c3]]), control-predecessors={%[[call1]]}
  // CHECK-DAG: %[[call3:.+]] = custom-call({{.+}}), custom_call_target="baz", control-predecessors={%[[res]]}
  )"));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(CallInlinerTest, ChannelIdsAreUniquifiedWhenSettingIsEnabled) {
  const char* hlo = R"(
ag {
  input = f32[128,32] parameter(0)
  ROOT ag = f32[128,128] all-gather(input),
    replica_groups={}, dimensions={1}, channel_id=1337
}

ag2 {
  input = f32[128,128] parameter(0)
  ROOT ag = f32[128,128] all-gather(input),
    replica_groups={}, dimensions={1}, channel_id=1337
}

ENTRY main {
  input = f32[128,32] parameter(0)
  ag = f32[128,128] call(input), to_apply=ag
  ag2 = f32[128,128] call(ag), to_apply=ag2
  ROOT result = (f32[128,128], f32[128,128]) tuple(ag2, ag)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));
  CallInliner call_inliner(
      /*single_call_site=*/false, /*update_domain=*/false,
      /*composites_to_preserve=*/{}, /*uniquify_channel_ids=*/true);
  EXPECT_THAT(call_inliner.Run(m.get()), ::tsl::testing::IsOkAndHolds(true));

  auto ag = m->entry_computation()->root_instruction()->operand(0);
  auto ag2 = m->entry_computation()->root_instruction()->operand(1);

  EXPECT_THAT(ag, op::AllGather());
  EXPECT_THAT(ag2, op::AllGather());
  EXPECT_NE(ag->channel_id(), ag2->channel_id());
}

}  // namespace
}  // namespace xla

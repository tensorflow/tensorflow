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

#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal_util.h"
#include "xla/service/call_graph.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class FlattenCallGraphTest : public HloHardwareIndependentTestBase {
 protected:
  // Build and return a trivial computation taking and returning a scalar.
  std::unique_ptr<HloComputation> MakeScalarComputation() {
    HloComputation::Builder builder(TestName() + ".ScalarComputation");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    builder.AddInstruction(
        HloInstruction::CreateUnary(kScalarShape, HloOpcode::kNegate, param0));
    return builder.Build();
  }

  // Build and return a computation which takes a scalar and maps (kMap) the
  // given computation to the value 'callsites' number of times.
  std::unique_ptr<HloComputation> MakeMappingComputation(
      HloComputation* map_computation, int64_t callsites) {
    HloComputation::Builder builder(TestName() + ".MappingComputation");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* last_value = param0;
    for (int64_t i = 0; i < callsites; ++i) {
      last_value = builder.AddInstruction(HloInstruction::CreateMap(
          kScalarShape, {last_value}, map_computation));
    }
    return builder.Build();
  }

  // Build and return a computation which takes a scalar and calls (kCall) the
  // given computation with value 'callsites' number of times.
  std::unique_ptr<HloComputation> MakeCallingComputation(
      HloComputation* callee_computation, int64_t callsites,
      const std::string& suffix = ".CallingComputation") {
    HloComputation::Builder builder(TestName() + suffix);
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* last_value = param0;
    for (int64_t i = 0; i < callsites; ++i) {
      last_value = builder.AddInstruction(HloInstruction::CreateCall(
          kScalarShape, {last_value}, callee_computation));
    }
    return builder.Build();
  }

  // Build and return a computation which takes a scalar and returns a PRED
  // value.
  std::unique_ptr<HloComputation> MakeConditionComputation() {
    HloComputation::Builder builder(TestName() + ".ConditionComputation");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* zero = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
    builder.AddInstruction(
        HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), param0,
                                      zero, ComparisonDirection::kGt));
    return builder.Build();
  }

  absl::StatusOr<bool> RunFlattenCallGraph(HloModule* module) {
    FlattenCallGraph flatten;
    ASSIGN_OR_RETURN(bool result, flatten.Run(module));
    return result;
  }

  const Shape kScalarShape = ShapeUtil::MakeShape(F32, {});
};

TEST_F(FlattenCallGraphTest, ComplexGraph) {
  // Test a call graph of a module with several computation called in various
  // contexts. The call graph looks like:
  //
  //      entry
  //      /  |
  //     a   |
  //   / | \ |
  //  b  |  cond
  //   \ |
  //    c
  //
  // Calls are made via kCall, kWhile, and kMap instructions.
  absl::string_view hlo_string = R"hlo(
HloModule ComplexGraph, entry_computation_layout={(f32[])->f32[]}

%ComplexGraph.ScalarComputation (param0.1: f32[]) -> f32[] {
  %param0.1 = f32[] parameter(0)
  ROOT %negate = f32[] negate(%param0.1)
}

%ComplexGraph.MappingComputation (param0.2: f32[]) -> f32[] {
  %param0.2 = f32[] parameter(0)
  ROOT %map = f32[] map(%param0.2), dimensions={}, to_apply=%ComplexGraph.ScalarComputation
}

%ComplexGraph.ConditionComputation (param0: f32[]) -> pred[] {
  %param0 = f32[] parameter(0)
  %constant = f32[] constant(0)
  ROOT %compare = pred[] compare(%param0, %constant), direction=GT
}

%ComplexGraph.a (param0.3: f32[]) -> f32[] {
  %param0.3 = f32[] parameter(0)
  %call = f32[] call(%param0.3), to_apply=%ComplexGraph.ScalarComputation
  ROOT %while = f32[] while(%call), condition=%ComplexGraph.ConditionComputation, body=%ComplexGraph.MappingComputation
}

ENTRY %ComplexGraph.entry (param0.4: f32[]) -> f32[] {
  %param0.4 = f32[] parameter(0)
  ROOT %while.1 = f32[] while(%param0.4), condition=%ComplexGraph.ConditionComputation, body=%ComplexGraph.a
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);

  absl::string_view expected =
      R"hlo(HloModule ComplexGraph, entry_computation_layout={(f32[])->f32[]}

%ComplexGraph.ScalarComputation.clone (param0.5: f32[]) -> f32[] {
  %param0.5 = f32[] parameter(0)
  ROOT %negate.1 = f32[] negate(%param0.5)
}

%ComplexGraph.ScalarComputation (param0.1: f32[]) -> f32[] {
  %param0.1 = f32[] parameter(0)
  ROOT %negate = f32[] negate(%param0.1)
}

%ComplexGraph.MappingComputation (param0.2: f32[]) -> f32[] {
  %param0.2 = f32[] parameter(0)
  ROOT %map = f32[] map(%param0.2), dimensions={}, to_apply=%ComplexGraph.ScalarComputation
}

%ComplexGraph.ConditionComputation (param0: f32[]) -> pred[] {
  %param0 = f32[] parameter(0)
  %constant = f32[] constant(0)
  ROOT %compare = pred[] compare(%param0, %constant), direction=GT
}

%ComplexGraph.a (param0.3: f32[]) -> f32[] {
  %param0.3 = f32[] parameter(0)
  %call = f32[] call(%param0.3), to_apply=%ComplexGraph.ScalarComputation.clone
  ROOT %while = f32[] while(%call), condition=%ComplexGraph.ConditionComputation, body=%ComplexGraph.MappingComputation
}

%ComplexGraph.ConditionComputation.clone (param0.6: f32[]) -> pred[] {
  %param0.6 = f32[] parameter(0)
  %constant.1 = f32[] constant(0)
  ROOT %compare.1 = pred[] compare(%param0.6, %constant.1), direction=GT
}

ENTRY %ComplexGraph.entry (param0.4: f32[]) -> f32[] {
  %param0.4 = f32[] parameter(0)
  ROOT %while.1 = f32[] while(%param0.4), condition=%ComplexGraph.ConditionComputation.clone, body=%ComplexGraph.a
}

)hlo";
  EXPECT_EQ(module->ToString(), expected);

  std::unique_ptr<CallGraph> flat_call_graph = CallGraph::Build(module.get());
  HloComputation* c_computation =
      module->GetComputationWithName("ComplexGraph.ScalarComputation");
  const CallGraphNode& c_node = flat_call_graph->GetNode(c_computation);
  EXPECT_EQ(1, c_node.caller_callsites().size());
}

// Test corner case of a computation used as a body and a loop condition.
TEST_F(FlattenCallGraphTest, SharedWhileConditionAndBody) {
  absl::string_view hlo_string = R"hlo(
HloModule SharedWhileConditionAndBody, entry_computation_layout={()->pred[]}

%SharedWhileConditionAndBody.cond (param0: pred[]) -> pred[] {
  %param0 = pred[] parameter(0)
  %constant = pred[] constant(false)
  ROOT %compare = pred[] compare(%param0, %constant), direction=EQ
}

ENTRY %SharedWhileConditionAndBody.entry () -> pred[] {
  %constant.1 = pred[] constant(false)
  ROOT %while = pred[] while(%constant.1), condition=%SharedWhileConditionAndBody.cond, body=%SharedWhileConditionAndBody.cond
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);

  absl::string_view expected =
      R"hlo(HloModule SharedWhileConditionAndBody, entry_computation_layout={()->pred[]}

%SharedWhileConditionAndBody.cond (param0: pred[]) -> pred[] {
  %param0 = pred[] parameter(0)
  %constant = pred[] constant(false)
  ROOT %compare = pred[] compare(%param0, %constant), direction=EQ
}

%SharedWhileConditionAndBody.cond.clone (param0.1: pred[]) -> pred[] {
  %param0.1 = pred[] parameter(0)
  %constant.2 = pred[] constant(false)
  ROOT %compare.1 = pred[] compare(%param0.1, %constant.2), direction=EQ
}

%SharedWhileConditionAndBody.cond.clone.1 (param0.2: pred[]) -> pred[] {
  %param0.2 = pred[] parameter(0)
  %constant.3 = pred[] constant(false)
  ROOT %compare.2 = pred[] compare(%param0.2, %constant.3), direction=EQ
}

ENTRY %SharedWhileConditionAndBody.entry () -> pred[] {
  %constant.1 = pred[] constant(false)
  ROOT %while = pred[] while(%constant.1), condition=%SharedWhileConditionAndBody.cond.clone.1, body=%SharedWhileConditionAndBody.cond.clone
}

)hlo";
  EXPECT_EQ(module->ToString(), expected);

  HloInstruction* while_op =
      module->entry_computation()->GetInstructionWithName("while");
  EXPECT_NE(while_op->while_body(), while_op->while_condition());
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  HloComputation* cond_computation =
      module->GetComputationWithName("SharedWhileConditionAndBody.cond");
  const CallGraphNode& cond_node = call_graph->GetNode(cond_computation);
  EXPECT_EQ(0, cond_node.caller_callsites().size());
}

// Test flattening of a nested calling computations.
//
//   Entry
//    / \
//    \ /
//     B
//    / \
//    \ /
//     C
//
TEST_F(FlattenCallGraphTest, FlattenCalls) {
  absl::string_view hlo_string = R"hlo(
HloModule FlattenCalls, entry_computation_layout={(f32[])->f32[]}

%FlattenCalls.ScalarComputation (param0: f32[]) -> f32[] {
  %param0 = f32[] parameter(0)
  ROOT %negate = f32[] negate(%param0)
}

%FlattenCalls.B (param0.1: f32[]) -> f32[] {
  %param0.1 = f32[] parameter(0)
  %call = f32[] call(%param0.1), to_apply=%FlattenCalls.ScalarComputation
  ROOT %call.1 = f32[] call(%call), to_apply=%FlattenCalls.ScalarComputation
}

ENTRY %FlattenCalls.Entry (param0.2: f32[]) -> f32[] {
  %param0.2 = f32[] parameter(0)
  %call.2 = f32[] call(%param0.2), to_apply=%FlattenCalls.B
  ROOT %call.3 = f32[] call(%call.2), to_apply=%FlattenCalls.B
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);

  absl::string_view expected =
      R"hlo(HloModule FlattenCalls, entry_computation_layout={(f32[])->f32[]}

%FlattenCalls.ScalarComputation (param0: f32[]) -> f32[] {
  %param0 = f32[] parameter(0)
  ROOT %negate = f32[] negate(%param0)
}

%FlattenCalls.ScalarComputation.clone (param0.3: f32[]) -> f32[] {
  %param0.3 = f32[] parameter(0)
  ROOT %negate.1 = f32[] negate(%param0.3)
}

%FlattenCalls.B (param0.1: f32[]) -> f32[] {
  %param0.1 = f32[] parameter(0)
  %call = f32[] call(%param0.1), to_apply=%FlattenCalls.ScalarComputation
  ROOT %call.1 = f32[] call(%call), to_apply=%FlattenCalls.ScalarComputation.clone
}

%FlattenCalls.ScalarComputation.clone.1 (param0.5: f32[]) -> f32[] {
  %param0.5 = f32[] parameter(0)
  ROOT %negate.2 = f32[] negate(%param0.5)
}

%FlattenCalls.ScalarComputation.clone.clone (param0.6: f32[]) -> f32[] {
  %param0.6 = f32[] parameter(0)
  ROOT %negate.3 = f32[] negate(%param0.6)
}

%FlattenCalls.B.clone (param0.4: f32[]) -> f32[] {
  %param0.4 = f32[] parameter(0)
  %call.4 = f32[] call(%param0.4), to_apply=%FlattenCalls.ScalarComputation.clone.1
  ROOT %call.5 = f32[] call(%call.4), to_apply=%FlattenCalls.ScalarComputation.clone.clone
}

ENTRY %FlattenCalls.Entry (param0.2: f32[]) -> f32[] {
  %param0.2 = f32[] parameter(0)
  %call.2 = f32[] call(%param0.2), to_apply=%FlattenCalls.B
  ROOT %call.3 = f32[] call(%call.2), to_apply=%FlattenCalls.B.clone
}

)hlo";
  EXPECT_EQ(module->ToString(), expected);

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  EXPECT_EQ(7, module->computation_count());
  HloComputation* c_computation =
      module->GetComputationWithName("FlattenCalls.ScalarComputation");
  const CallGraphNode& c_node = call_graph->GetNode(c_computation);
  EXPECT_EQ(1, c_node.caller_callsites().size());
  HloComputation* b_computation =
      module->GetComputationWithName("FlattenCalls.B");
  const CallGraphNode& b_node = call_graph->GetNode(b_computation);
  EXPECT_EQ(1, b_node.caller_callsites().size());
}

TEST_F(FlattenCallGraphTest, FlattenCallsInConditional) {
  absl::string_view hlo_string = R"hlo(
HloModule FlattenCallsInConditional, entry_computation_layout={()->f32[]}

%FlattenCallsInConditional.ScalarComputation (param0: f32[]) -> f32[] {
  %param0 = f32[] parameter(0)
  ROOT %negate = f32[] negate(%param0)
}

ENTRY %FlattenCallsInConditional () -> f32[] {
  %constant = pred[] constant(true)
  %constant.1 = f32[] constant(56)
  %constant.2 = f32[] constant(12)
  ROOT %conditional = f32[] conditional(%constant, %constant.1, %constant.2), true_computation=%FlattenCallsInConditional.ScalarComputation, false_computation=%FlattenCallsInConditional.ScalarComputation
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);
  absl::string_view expected =
      R"hlo(HloModule FlattenCallsInConditional, entry_computation_layout={()->f32[]}

%FlattenCallsInConditional.ScalarComputation (param0: f32[]) -> f32[] {
  %param0 = f32[] parameter(0)
  ROOT %negate = f32[] negate(%param0)
}

%FlattenCallsInConditional.ScalarComputation.clone (param0.1: f32[]) -> f32[] {
  %param0.1 = f32[] parameter(0)
  ROOT %negate.1 = f32[] negate(%param0.1)
}

%FlattenCallsInConditional.ScalarComputation.clone.1 (param0.2: f32[]) -> f32[] {
  %param0.2 = f32[] parameter(0)
  ROOT %negate.2 = f32[] negate(%param0.2)
}

ENTRY %FlattenCallsInConditional () -> f32[] {
  %constant = pred[] constant(true)
  %constant.1 = f32[] constant(56)
  %constant.2 = f32[] constant(12)
  ROOT %conditional = f32[] conditional(%constant, %constant.1, %constant.2), true_computation=%FlattenCallsInConditional.ScalarComputation.clone, false_computation=%FlattenCallsInConditional.ScalarComputation.clone.1
}

)hlo";
  EXPECT_EQ(module->ToString(), expected);

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  // The true and false computations must now be different.
  EXPECT_EQ(4, module->computation_count());
  HloInstruction* cond =
      module->entry_computation()->GetInstructionWithName("conditional");
  EXPECT_NE(cond->branch_computation(0), cond->branch_computation(1));

  //  The original computation is no longer used.
  HloComputation* sub_computation = module->GetComputationWithName(
      "FlattenCallsInConditional.ScalarComputation");
  const CallGraphNode& sub_node = call_graph->GetNode(sub_computation);
  EXPECT_EQ(0, sub_node.caller_callsites().size());
}

TEST_F(FlattenCallGraphTest, AsyncCall) {
  std::string hlo_string = R"(
HloModule AsyncCall

%called_computation (param_0: f32[4096], param_1: f32[4096]) -> f32[4096] {
  %param_0 = f32[4096]{0} parameter(0)
  %param_1 = f32[4096]{0} parameter(1)
  ROOT %result.1 = f32[4096]{0} add(f32[4096]{0} %param_0, f32[4096]{0} %param_1)
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %b = f32[4096]{0} parameter(1)
  %call-start.0 = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) call-start(f32[4096]{0} %a, f32[4096]{0} %b), to_apply=%called_computation
  %call-done.0 = f32[4096]{0} call-done(((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) %call-start.0)
  %call-start.1 = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) call-start(f32[4096]{0} %call-done.0, f32[4096]{0} %b), to_apply=%called_computation
  %call-done.1 = f32[4096]{0} call-done(((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) %call-start.1)
  ROOT %add_1 = f32[4096]{0} add(f32[4096]{0} %a, f32[4096]{0} %call-done.1)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);

  absl::string_view expected =
      R"hlo(HloModule AsyncCall, entry_computation_layout={(f32[4096]{0}, f32[4096]{0})->f32[4096]{0}}

%called_computation (param_0: f32[4096], param_1: f32[4096]) -> f32[4096] {
  %param_0 = f32[4096]{0} parameter(0)
  %param_1 = f32[4096]{0} parameter(1)
  ROOT %result.1 = f32[4096]{0} add(%param_0, %param_1)
}

%async_wrapped (async_param: f32[4096], async_param.1: f32[4096]) -> f32[4096] {
  %async_param = f32[4096]{0} parameter(0)
  %async_param.1 = f32[4096]{0} parameter(1)
  ROOT %call = f32[4096]{0} call(%async_param, %async_param.1), to_apply=%called_computation
}

%called_computation.clone (param_0.1: f32[4096], param_1.1: f32[4096]) -> f32[4096] {
  %param_0.1 = f32[4096]{0} parameter(0)
  %param_1.1 = f32[4096]{0} parameter(1)
  ROOT %result.0 = f32[4096]{0} add(%param_0.1, %param_1.1)
}

%async_wrapped.1 (async_param.2: f32[4096], async_param.3: f32[4096]) -> f32[4096] {
  %async_param.2 = f32[4096]{0} parameter(0)
  %async_param.3 = f32[4096]{0} parameter(1)
  ROOT %call.1 = f32[4096]{0} call(%async_param.2, %async_param.3), to_apply=%called_computation.clone
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %b = f32[4096]{0} parameter(1)
  %call-start.0 = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) async-start(%a, %b), calls=%async_wrapped
  %call-done.0 = f32[4096]{0} async-done(%call-start.0)
  %call-start.1 = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) async-start(%call-done.0, %b), calls=%async_wrapped.1
  %call-done.1 = f32[4096]{0} async-done(%call-start.1)
  ROOT %add_1 = f32[4096]{0} add(%a, %call-done.1)
}

)hlo";
  EXPECT_EQ(module->ToString(), expected);

  // We expect the entry computation, two async_wrapped computations and two
  // called_computation computations.
  EXPECT_EQ(5, module->computation_count());
  EXPECT_EQ(FindInstruction(module.get(), "call-start.0")
                ->async_wrapped_computation(),
            FindInstruction(module.get(), "call-done.0")
                ->async_wrapped_computation());
  EXPECT_EQ(FindInstruction(module.get(), "call-start.1")
                ->async_wrapped_computation(),
            FindInstruction(module.get(), "call-done.1")
                ->async_wrapped_computation());
  EXPECT_NE(FindInstruction(module.get(), "call-start.0")
                ->async_wrapped_computation(),
            FindInstruction(module.get(), "call-start.1")
                ->async_wrapped_computation());
  EXPECT_NE(FindInstruction(module.get(), "call-start.0")
                ->async_wrapped_instruction()
                ->called_computations()[0],
            FindInstruction(module.get(), "call-start.1")
                ->async_wrapped_instruction()
                ->called_computations()[0]);
}

TEST_F(FlattenCallGraphTest, WhileInCall) {
  std::string hlo_string = R"(
HloModule WhileInCall

  %while_cond {
    %p.cond = (f32[4096]{0}) parameter(0)
    ROOT %eq = pred[] constant(false)
  }

  %while_body {
    ROOT %p = (f32[4096]{0}) parameter(0)
  }

  %called_computation(arg: f32[4096]) -> f32[4096] {
    %arg = f32[4096]{0} parameter(0)
    %while_init = (f32[4096]{0}) tuple(%arg)
    %while = (f32[4096]{0}) while(%while_init), condition=%while_cond, body=%while_body
    ROOT %get-tuple-element = f32[4096]{0} get-tuple-element(%while), index=0
  }

  ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
    %a = f32[4096]{0} parameter(0)
    %b = f32[4096]{0} parameter(1)
    %call0 = f32[4096]{0} call(%a), to_apply=%called_computation
    %call1 = f32[4096]{0} call(%b), to_apply=%called_computation
    ROOT %multiply = f32[4096]{0} multiply(%call0, %call1)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_EQ(module->computation_count(), 4);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);

  absl::string_view expected =
      R"hlo(HloModule WhileInCall, entry_computation_layout={(f32[4096]{0}, f32[4096]{0})->f32[4096]{0}}

%while_body (p: (f32[4096])) -> (f32[4096]) {
  ROOT %p = (f32[4096]{0}) parameter(0)
}

%while_cond (p.cond: (f32[4096])) -> pred[] {
  %p.cond = (f32[4096]{0}) parameter(0)
  ROOT %eq = pred[] constant(false)
}

%called_computation (arg: f32[4096]) -> f32[4096] {
  %arg = f32[4096]{0} parameter(0)
  %while_init = (f32[4096]{0}) tuple(%arg)
  %while = (f32[4096]{0}) while(%while_init), condition=%while_cond, body=%while_body
  ROOT %get-tuple-element = f32[4096]{0} get-tuple-element(%while), index=0
}

%while_body.clone (p.1: (f32[4096])) -> (f32[4096]) {
  ROOT %p.1 = (f32[4096]{0}) parameter(0)
}

%while_cond.clone (p.cond.1: (f32[4096])) -> pred[] {
  %p.cond.1 = (f32[4096]{0}) parameter(0)
  ROOT %eq.1 = pred[] constant(false)
}

%called_computation.clone (arg.1: f32[4096]) -> f32[4096] {
  %arg.1 = f32[4096]{0} parameter(0)
  %while_init.1 = (f32[4096]{0}) tuple(%arg.1)
  %while.1 = (f32[4096]{0}) while(%while_init.1), condition=%while_cond.clone, body=%while_body.clone
  ROOT %get-tuple-element.1 = f32[4096]{0} get-tuple-element(%while.1), index=0
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %call0 = f32[4096]{0} call(%a), to_apply=%called_computation
  %b = f32[4096]{0} parameter(1)
  %call1 = f32[4096]{0} call(%b), to_apply=%called_computation.clone
  ROOT %multiply = f32[4096]{0} multiply(%call0, %call1)
}

)hlo";
  EXPECT_EQ(module->ToString(), expected);

  EXPECT_EQ(module->computation_count(), 7);
}

TEST_F(FlattenCallGraphTest, CallInWhileInCall) {
  std::string hlo_string = R"(
HloModule CallInWhileInCall

  %called_computation_internal(arg: f32[4096]) -> f32[4096] {
    ROOT %arg.internal = f32[4096]{0} parameter(0)
  }

  %while_cond {
    %p.cond = (f32[4096]{0}) parameter(0)
    ROOT %eq = pred[] constant(false)
  }

  %while_body {
    %p.body = (f32[4096]{0}) parameter(0)
    %gte = f32[4096]{0} get-tuple-element(%p.body), index=0
    %call.internal = f32[4096]{0} call(%gte), to_apply=%called_computation_internal
    ROOT %tuple.body = (f32[4096]{0}) tuple(%call.internal)
  }

  %called_computation_external(arg: f32[4096]) -> f32[4096] {
    %arg.external = f32[4096]{0} parameter(0)
    %while.init = (f32[4096]{0}) tuple(%arg.external)
    %while = (f32[4096]{0}) while(%while.init), condition=%while_cond, body=%while_body
    ROOT %get-tuple-element = f32[4096]{0} get-tuple-element(%while), index=0
  }

  ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
    %a = f32[4096]{0} parameter(0)
    %b = f32[4096]{0} parameter(1)
    %call0 = f32[4096]{0} call(%a), to_apply=%called_computation_external
    %call1 = f32[4096]{0} call(%b), to_apply=%called_computation_external
    ROOT %multiply = f32[4096]{0} multiply(%call0, %call1)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_EQ(module->computation_count(), 5);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);

  absl::string_view expected =
      R"hlo(HloModule CallInWhileInCall, entry_computation_layout={(f32[4096]{0}, f32[4096]{0})->f32[4096]{0}}

%called_computation_internal (arg.internal: f32[4096]) -> f32[4096] {
  ROOT %arg.internal = f32[4096]{0} parameter(0)
}

%while_body (p.body: (f32[4096])) -> (f32[4096]) {
  %p.body = (f32[4096]{0}) parameter(0)
  %gte = f32[4096]{0} get-tuple-element(%p.body), index=0
  %call.internal = f32[4096]{0} call(%gte), to_apply=%called_computation_internal
  ROOT %tuple.body = (f32[4096]{0}) tuple(%call.internal)
}

%while_cond (p.cond: (f32[4096])) -> pred[] {
  %p.cond = (f32[4096]{0}) parameter(0)
  ROOT %eq = pred[] constant(false)
}

%called_computation_external (arg.external: f32[4096]) -> f32[4096] {
  %arg.external = f32[4096]{0} parameter(0)
  %while.init = (f32[4096]{0}) tuple(%arg.external)
  %while = (f32[4096]{0}) while(%while.init), condition=%while_cond, body=%while_body
  ROOT %get-tuple-element = f32[4096]{0} get-tuple-element(%while), index=0
}

%called_computation_internal.clone (arg.internal.1: f32[4096]) -> f32[4096] {
  ROOT %arg.internal.1 = f32[4096]{0} parameter(0)
}

%while_body.clone (p.body.1: (f32[4096])) -> (f32[4096]) {
  %p.body.1 = (f32[4096]{0}) parameter(0)
  %gte.1 = f32[4096]{0} get-tuple-element(%p.body.1), index=0
  %call.internal.1 = f32[4096]{0} call(%gte.1), to_apply=%called_computation_internal.clone
  ROOT %tuple.body.1 = (f32[4096]{0}) tuple(%call.internal.1)
}

%while_cond.clone (p.cond.1: (f32[4096])) -> pred[] {
  %p.cond.1 = (f32[4096]{0}) parameter(0)
  ROOT %eq.1 = pred[] constant(false)
}

%called_computation_external.clone (arg.external.1: f32[4096]) -> f32[4096] {
  %arg.external.1 = f32[4096]{0} parameter(0)
  %while.init.1 = (f32[4096]{0}) tuple(%arg.external.1)
  %while.1 = (f32[4096]{0}) while(%while.init.1), condition=%while_cond.clone, body=%while_body.clone
  ROOT %get-tuple-element.1 = f32[4096]{0} get-tuple-element(%while.1), index=0
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %call0 = f32[4096]{0} call(%a), to_apply=%called_computation_external
  %b = f32[4096]{0} parameter(1)
  %call1 = f32[4096]{0} call(%b), to_apply=%called_computation_external.clone
  ROOT %multiply = f32[4096]{0} multiply(%call0, %call1)
}

)hlo";
  EXPECT_EQ(module->ToString(), expected);
  EXPECT_EQ(module->computation_count(), 9);
}

TEST_F(FlattenCallGraphTest, SortInCall) {
  std::string hlo_string = R"(
HloModule SortInCall

  %compare {
    %p.0.lhs = f32[] parameter(0)
    %p.0.rhs = f32[] parameter(1)
    ROOT %lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
  }

  %called_computation(arg: f32[4096]) -> f32[4096] {
    %x = f32[4096]{0} parameter(0)
    ROOT %sort =f32[4096]{0} sort(%x), dimensions={0}, to_apply=%compare
  }

  ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
    %a = f32[4096]{0} parameter(0)
    %b = f32[4096]{0} parameter(1)
    %call0 = f32[4096]{0} call(%a), to_apply=%called_computation
    %call1 = f32[4096]{0} call(%b), to_apply=%called_computation
    ROOT %multiply = f32[4096]{0} multiply(%call0, %call1)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_EQ(module->computation_count(), 3);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  ASSERT_EQ(module->computation_count(), 5);
  EXPECT_TRUE(result);

  absl::string_view expected =
      R"hlo(HloModule SortInCall, entry_computation_layout={(f32[4096]{0}, f32[4096]{0})->f32[4096]{0}}

%compare (p.0.lhs: f32[], p.0.rhs: f32[]) -> pred[] {
  %p.0.lhs = f32[] parameter(0)
  %p.0.rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%p.0.lhs, %p.0.rhs), direction=LT
}

%called_computation (x: f32[4096]) -> f32[4096] {
  %x = f32[4096]{0} parameter(0)
  ROOT %sort = f32[4096]{0} sort(%x), dimensions={0}, to_apply=%compare
}

%compare.clone (p.0.lhs.1: f32[], p.0.rhs.1: f32[]) -> pred[] {
  %p.0.lhs.1 = f32[] parameter(0)
  %p.0.rhs.1 = f32[] parameter(1)
  ROOT %lt.1 = pred[] compare(%p.0.lhs.1, %p.0.rhs.1), direction=LT
}

%called_computation.clone (x.1: f32[4096]) -> f32[4096] {
  %x.1 = f32[4096]{0} parameter(0)
  ROOT %sort.1 = f32[4096]{0} sort(%x.1), dimensions={0}, to_apply=%compare.clone
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %call0 = f32[4096]{0} call(%a), to_apply=%called_computation
  %b = f32[4096]{0} parameter(1)
  %call1 = f32[4096]{0} call(%b), to_apply=%called_computation.clone
  ROOT %multiply = f32[4096]{0} multiply(%call0, %call1)
}

)hlo";
  EXPECT_EQ(module->ToString(), expected);

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  for (const CallGraphNode& node : call_graph->nodes()) {
    EXPECT_LE(node.caller_callsites().size(), 1);
  }
}

TEST_F(FlattenCallGraphTest, CallInSortInCall) {
  std::string hlo_string = R"(
HloModule CallInSortInCall

  %compare.impl {
    %p.0.lhs = f32[] parameter(0)
    %p.0.rhs = f32[] parameter(1)
    ROOT %lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
  }

  %compare {
    %p.0.lhs = f32[] parameter(0)
    %p.0.rhs = f32[] parameter(1)
    ROOT %lt = pred[] call(%p.0.lhs, %p.0.rhs), to_apply=%compare.impl
  }

  %called_computation(arg: f32[4096]) -> f32[4096] {
    %x = f32[4096]{0} parameter(0)
    ROOT %sort =f32[4096]{0} sort(%x), dimensions={0}, to_apply=%compare
  }

  ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
    %a = f32[4096]{0} parameter(0)
    %b = f32[4096]{0} parameter(1)
    %call0 = f32[4096]{0} call(%a), to_apply=%called_computation
    %call1 = f32[4096]{0} call(%b), to_apply=%called_computation
    ROOT %multiply = f32[4096]{0} multiply(%call0, %call1)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_EQ(module->computation_count(), 4);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  ASSERT_EQ(module->computation_count(), 7);
  EXPECT_TRUE(result);

  absl::string_view expected =
      R"hlo(HloModule CallInSortInCall, entry_computation_layout={(f32[4096]{0}, f32[4096]{0})->f32[4096]{0}}

%compare.impl (p.0.lhs: f32[], p.0.rhs: f32[]) -> pred[] {
  %p.0.lhs = f32[] parameter(0)
  %p.0.rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%p.0.lhs, %p.0.rhs), direction=LT
}

%compare (p.0.lhs.1: f32[], p.0.rhs.1: f32[]) -> pred[] {
  %p.0.lhs.1 = f32[] parameter(0)
  %p.0.rhs.1 = f32[] parameter(1)
  ROOT %lt.1 = pred[] call(%p.0.lhs.1, %p.0.rhs.1), to_apply=%compare.impl
}

%called_computation (x: f32[4096]) -> f32[4096] {
  %x = f32[4096]{0} parameter(0)
  ROOT %sort = f32[4096]{0} sort(%x), dimensions={0}, to_apply=%compare
}

%compare.impl.clone (p.0.lhs.3: f32[], p.0.rhs.3: f32[]) -> pred[] {
  %p.0.lhs.3 = f32[] parameter(0)
  %p.0.rhs.3 = f32[] parameter(1)
  ROOT %lt.3 = pred[] compare(%p.0.lhs.3, %p.0.rhs.3), direction=LT
}

%compare.clone (p.0.lhs.2: f32[], p.0.rhs.2: f32[]) -> pred[] {
  %p.0.lhs.2 = f32[] parameter(0)
  %p.0.rhs.2 = f32[] parameter(1)
  ROOT %lt.2 = pred[] call(%p.0.lhs.2, %p.0.rhs.2), to_apply=%compare.impl.clone
}

%called_computation.clone (x.1: f32[4096]) -> f32[4096] {
  %x.1 = f32[4096]{0} parameter(0)
  ROOT %sort.1 = f32[4096]{0} sort(%x.1), dimensions={0}, to_apply=%compare.clone
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %call0 = f32[4096]{0} call(%a), to_apply=%called_computation
  %b = f32[4096]{0} parameter(1)
  %call1 = f32[4096]{0} call(%b), to_apply=%called_computation.clone
  ROOT %multiply = f32[4096]{0} multiply(%call0, %call1)
}

)hlo";
  EXPECT_EQ(module->ToString(), expected);

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  for (const CallGraphNode& node : call_graph->nodes()) {
    EXPECT_LE(node.caller_callsites().size(), 1);
  }
}

TEST_F(FlattenCallGraphTest, NoChange) {
  std::string hlo_string = R"(
HloModule NoChange

  ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
    %a = f32[4096]{0} parameter(0)
    %b = f32[4096]{0} parameter(1)
    ROOT %multiply = f32[4096]{0} multiply(%a, %b)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_EQ(module->computation_count(), 1);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  ASSERT_EQ(module->computation_count(), 1);
  EXPECT_FALSE(result);

  absl::string_view expected =
      R"hlo(HloModule NoChange, entry_computation_layout={(f32[4096]{0}, f32[4096]{0})->f32[4096]{0}}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %b = f32[4096]{0} parameter(1)
  ROOT %multiply = f32[4096]{0} multiply(%a, %b)
}

)hlo";
  EXPECT_EQ(module->ToString(), expected);
}

TEST_F(FlattenCallGraphTest, SkipCloningTest) {
  std::string hlo_string = R"(
HloModule SkipCloning

%while_body (param: f32[]) -> f32[] {
  %param = f32[] parameter(0)
  ROOT %neg = f32[] negate(%param)
}

%while_cond (param: f32[]) -> pred[] {
  %param = f32[] parameter(0)
  %zero = f32[] constant(0.0)
  ROOT %cmp = pred[] compare(%param, %zero), direction=GT
}

%a_comp (param: f32[]) -> f32[] {
  %param = f32[] parameter(0)
  ROOT %while = f32[] while(%param), condition=%while_cond, body=%while_body
}

%b_comp (param: f32[]) -> f32[] {
  %param = f32[] parameter(0)
  ROOT %while = f32[] while(%param), condition=%while_cond, body=%while_body
}

ENTRY %main (param: f32[]) -> (f32[], f32[], f32[]) {
  %param = f32[] parameter(0)
  %a.0 = f32[] call(%param), to_apply=%a_comp
  %a.1 = f32[] call(%param), to_apply=%a_comp
  %b = f32[] call(%param), to_apply=%b_comp
  ROOT %tuple = (f32[], f32[], f32[]) tuple(%a.0, %a.1, %b)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Configure FlattenCallGraph to skip cloning if all callers are kCall.
  FlattenCallGraph flatten([](const HloComputation& computation) {
    std::unique_ptr<CallGraph> call_graph =
        CallGraph::Build(computation.parent());
    const CallGraphNode& node = call_graph->GetNode(&computation);
    for (const CallSite& call_site : node.caller_callsites()) {
      if (call_site.instruction()->opcode() != HloOpcode::kCall) {
        return false;
      }
    }
    return true;
  });

  TF_ASSERT_OK_AND_ASSIGN(bool result, flatten.Run(module.get()));
  EXPECT_TRUE(result);

  absl::string_view expected =
      R"hlo(HloModule SkipCloning, entry_computation_layout={(f32[])->(f32[], f32[], f32[])}

%while_body (param: f32[]) -> f32[] {
  %param = f32[] parameter(0)
  ROOT %neg = f32[] negate(%param)
}

%while_cond (param.1: f32[]) -> pred[] {
  %param.1 = f32[] parameter(0)
  %zero = f32[] constant(0)
  ROOT %cmp = pred[] compare(%param.1, %zero), direction=GT
}

%a_comp (param.2: f32[]) -> f32[] {
  %param.2 = f32[] parameter(0)
  ROOT %while = f32[] while(%param.2), condition=%while_cond, body=%while_body
}

%while_body.clone (param.5: f32[]) -> f32[] {
  %param.5 = f32[] parameter(0)
  ROOT %neg.1 = f32[] negate(%param.5)
}

%while_cond.clone (param.6: f32[]) -> pred[] {
  %param.6 = f32[] parameter(0)
  %zero.1 = f32[] constant(0)
  ROOT %cmp.1 = pred[] compare(%param.6, %zero.1), direction=GT
}

%b_comp (param.3: f32[]) -> f32[] {
  %param.3 = f32[] parameter(0)
  ROOT %while.1 = f32[] while(%param.3), condition=%while_cond.clone, body=%while_body.clone
}

ENTRY %main (param.4: f32[]) -> (f32[], f32[], f32[]) {
  %param.4 = f32[] parameter(0)
  %a.0 = f32[] call(%param.4), to_apply=%a_comp
  %a.1 = f32[] call(%param.4), to_apply=%a_comp
  %b = f32[] call(%param.4), to_apply=%b_comp
  ROOT %tuple = (f32[], f32[], f32[]) tuple(%a.0, %a.1, %b)
}

)hlo";
  EXPECT_EQ(module->ToString(), expected);

  HloComputation* a_computation = FindComputation(module.get(), "a_comp");
  HloComputation* while_body_computation =
      FindComputation(module.get(), "while_body");

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  EXPECT_EQ(2, call_graph->GetNode(a_computation).caller_callsites().size());
  EXPECT_EQ(
      1, call_graph->GetNode(while_body_computation).caller_callsites().size());
}

TEST_F(FlattenCallGraphTest, PreserveScheduleTest) {
  std::string hlo_string = R"(
HloModule PreserveScheduleTest, is_scheduled=true

%called_computation (param_0: f32[4096], param_1: f32[4096]) -> f32[4096] {
  %param_0 = f32[4096] parameter(0)
  %param_1 = f32[4096] parameter(1)
  ROOT %result.1 = f32[4096] add(f32[4096] %param_0, f32[4096] %param_1)
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096] parameter(0)
  %b = f32[4096] parameter(1)
  %call.0 = f32[4096] call(%a, %b), to_apply=%called_computation
  %call.1 = f32[4096] call(%call.0, %b), to_apply=%called_computation
  ROOT %add_1 = f32[4096] add(%a, %call.1)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);

  absl::string_view expected =
      R"hlo(HloModule PreserveScheduleTest, is_scheduled=true, entry_computation_layout={(f32[4096]{0}, f32[4096]{0})->f32[4096]{0}}

%called_computation (param_0: f32[4096], param_1: f32[4096]) -> f32[4096] {
  %param_0 = f32[4096]{0} parameter(0)
  %param_1 = f32[4096]{0} parameter(1)
  ROOT %result.1 = f32[4096]{0} add(%param_0, %param_1)
}

%called_computation.clone (param_0.1: f32[4096], param_1.1: f32[4096]) -> f32[4096] {
  %param_0.1 = f32[4096]{0} parameter(0)
  %param_1.1 = f32[4096]{0} parameter(1)
  ROOT %result.0 = f32[4096]{0} add(%param_0.1, %param_1.1)
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %b = f32[4096]{0} parameter(1)
  %call.0 = f32[4096]{0} call(%a, %b), to_apply=%called_computation
  %call.1 = f32[4096]{0} call(%call.0, %b), to_apply=%called_computation.clone
  ROOT %add_1 = f32[4096]{0} add(%a, %call.1)
}

)hlo";
  EXPECT_EQ(module->ToString(), expected);

  // We expect the entry computation and two called_computation computations.
  EXPECT_EQ(3, module->computation_count());
  HloComputation* called_computation_0 =
      FindInstruction(module.get(), "call.0")->to_apply();
  HloComputation* called_computation_1 =
      FindInstruction(module.get(), "call.1")->to_apply();

  EXPECT_TRUE(module->has_schedule());
  const auto& schedule = module->schedule();

  EXPECT_TRUE(schedule.is_computation_scheduled(called_computation_0));
  EXPECT_TRUE(schedule.is_computation_scheduled(called_computation_1));
}

}  // namespace
}  // namespace xla

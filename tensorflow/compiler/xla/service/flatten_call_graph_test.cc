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

#include "tensorflow/compiler/xla/service/flatten_call_graph.h"

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

class FlattenCallGraphTest : public HloTestBase {
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

  StatusOr<bool> RunFlattenCallGraph(HloModule* module) {
    FlattenCallGraph flatten;
    TF_ASSIGN_OR_RETURN(bool result, flatten.Run(module));
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
  auto module = CreateNewVerifiedModule();
  HloComputation* cond_computation =
      module->AddEmbeddedComputation(MakeConditionComputation());
  HloComputation* c_computation =
      module->AddEmbeddedComputation(MakeScalarComputation());
  HloComputation* b_computation = module->AddEmbeddedComputation(
      MakeMappingComputation(c_computation, /*callsites=*/1));

  HloComputation* a_computation;
  {
    HloComputation::Builder builder(TestName() + ".a");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* call = builder.AddInstruction(
        HloInstruction::CreateCall(kScalarShape, {param0}, c_computation));
    builder.AddInstruction(HloInstruction::CreateWhile(
        kScalarShape, cond_computation, b_computation, call));
    a_computation = module->AddEmbeddedComputation(builder.Build());
  }

  HloComputation* entry_computation;
  {
    HloComputation::Builder builder(TestName() + ".entry");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    builder.AddInstruction(HloInstruction::CreateWhile(
        kScalarShape, cond_computation, a_computation, param0));
    entry_computation = module->AddEntryComputation(builder.Build());
  }

  {
    TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
    EXPECT_TRUE(result);
    std::unique_ptr<CallGraph> flat_call_graph = CallGraph::Build(module.get());
    const CallGraphNode& c_node = flat_call_graph->GetNode(c_computation);
    EXPECT_EQ(1, c_node.caller_callsites().size());
  }
}

// Test corner case of a computation used as a body and a loop condition.
TEST_F(FlattenCallGraphTest, SharedWhileConditionAndBody) {
  auto module = CreateNewVerifiedModule();
  HloComputation* cond_computation;
  {
    HloComputation::Builder builder(TestName() + ".cond");
    HloInstruction* param0 =
        builder.AddInstruction(HloInstruction::CreateParameter(
            0, ShapeUtil::MakeShape(PRED, {}), "param0"));
    HloInstruction* false_constant = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
    builder.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), param0, false_constant,
        ComparisonDirection::kEq));
    cond_computation = module->AddEmbeddedComputation(builder.Build());
  }

  HloComputation* entry_computation;
  {
    HloComputation::Builder builder(TestName() + ".entry");
    HloInstruction* false_constant = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
    builder.AddInstruction(HloInstruction::CreateWhile(
        ShapeUtil::MakeShape(PRED, {}), cond_computation, cond_computation,
        false_constant));
    entry_computation = module->AddEntryComputation(builder.Build());
  }

  {
    std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
    const CallGraphNode& cond_node = call_graph->GetNode(cond_computation);
    EXPECT_EQ(2, cond_node.caller_callsites().size());
  }

  {
    TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
    EXPECT_TRUE(result);
    std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
    const CallGraphNode& cond_node = call_graph->GetNode(cond_computation);
    EXPECT_EQ(1, cond_node.caller_callsites().size());
  }
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
  auto module = CreateNewVerifiedModule();
  HloComputation* c_computation =
      module->AddEmbeddedComputation(MakeScalarComputation());

  HloComputation* b_computation = module->AddEmbeddedComputation(
      MakeCallingComputation(c_computation, /*callsites=*/2, ".B"));

  module->AddEntryComputation(
      MakeCallingComputation(b_computation, /*callsites=*/2, ".Entry"));

  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  EXPECT_EQ(7, module->computation_count());

  const CallGraphNode& c_node = call_graph->GetNode(c_computation);
  EXPECT_EQ(1, c_node.caller_callsites().size());

  const CallGraphNode& b_node = call_graph->GetNode(b_computation);
  EXPECT_EQ(1, b_node.caller_callsites().size());
}

TEST_F(FlattenCallGraphTest, FlattenCallsInConditional) {
  auto module = CreateNewVerifiedModule();
  HloComputation* sub_computation =
      module->AddEmbeddedComputation(MakeScalarComputation());

  // Create entry computation, which is a conditional that has the same
  // computation in the true and false branch.
  HloComputation::Builder builder(TestName());
  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(56.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(12.0f)));
  builder.AddInstruction(HloInstruction::CreateConditional(
      kScalarShape, pred, constant1, sub_computation, constant2,
      sub_computation));
  module->AddEntryComputation(builder.Build());
  EXPECT_EQ(2, module->computation_count());

  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  // The true and false computations must now be different.
  EXPECT_EQ(3, module->computation_count());

  const CallGraphNode& sub_node = call_graph->GetNode(sub_computation);
  EXPECT_EQ(1, sub_node.caller_callsites().size());
}

TEST_F(FlattenCallGraphTest, AsyncCall) {
  std::string hlo_string = R"(
HloModule AsyncCall

%called_computation (param_0: f32[4096], param_1: f32[4096]) -> f32[4096] {
  %param_0 = f32[4096]{0} parameter(0)
  %param_1 = f32[4096]{0} parameter(1)
  ROOT %result.1 = f32[4096]{0} add(f32[4096]{0} %param_0, f32[4096]{0} %param_1)
}

%async_wrapped (async_param: f32[4096], async_param.1: f32[4096]) -> f32[4096] {
  %async_param = f32[4096]{0} parameter(0)
  %async_param.1 = f32[4096]{0} parameter(1)
  ROOT %call = f32[4096]{0} call(f32[4096]{0} %async_param, f32[4096]{0} %async_param.1), to_apply=%called_computation
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %b = f32[4096]{0} parameter(1)
  %async-start.0 = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) async-start(f32[4096]{0} %a, f32[4096]{0} %b), async_group_id=0, calls=%async_wrapped
  %async-done.0 = f32[4096]{0} async-done(((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) %async-start.0), async_group_id=0, calls=%async_wrapped
  %async-start.1 = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) async-start(f32[4096]{0} %async-done.0, f32[4096]{0} %b), async_group_id=1, calls=%async_wrapped
  %async-done.1 = f32[4096]{0} async-done(((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) %async-start.1), async_group_id=1, calls=%async_wrapped
  ROOT %add_1 = f32[4096]{0} add(f32[4096]{0} %a, f32[4096]{0} %async-done.1)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);

  // We expect the entry computation, two async_wrapped computations and two
  // called_computation computations.
  EXPECT_EQ(5, module->computation_count());

  EXPECT_EQ(FindInstruction(module.get(), "async-start.0")
                ->async_wrapped_computation(),
            FindInstruction(module.get(), "async-done.0")
                ->async_wrapped_computation());
  EXPECT_EQ(FindInstruction(module.get(), "async-start.1")
                ->async_wrapped_computation(),
            FindInstruction(module.get(), "async-done.1")
                ->async_wrapped_computation());
  EXPECT_NE(FindInstruction(module.get(), "async-start.0")
                ->async_wrapped_computation(),
            FindInstruction(module.get(), "async-start.1")
                ->async_wrapped_computation());
}

}  // namespace
}  // namespace xla

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

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

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
      HloComputation* map_computation, int64 callsites) {
    HloComputation::Builder builder(TestName() + ".MappingComputation");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* last_value = param0;
    for (int64 i = 0; i < callsites; ++i) {
      last_value = builder.AddInstruction(HloInstruction::CreateMap(
          kScalarShape, {last_value}, map_computation));
    }
    return builder.Build();
  }

  // Build and return a computation which takes a scalar and calls (kCall) the
  // given computation with value 'callsites' number of times.
  std::unique_ptr<HloComputation> MakeCallingComputation(
      HloComputation* callee_computation, int64 callsites,
      const string& suffix = ".CallingComputation") {
    HloComputation::Builder builder(TestName() + suffix);
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* last_value = param0;
    for (int64 i = 0; i < callsites; ++i) {
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
    builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kGt, param0, zero));
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
  auto module = CreateNewModule();
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
    TF_ASSIGN_OR_ASSERT_OK(bool result, RunFlattenCallGraph(module.get()));
    EXPECT_TRUE(result);
    std::unique_ptr<CallGraph> flat_call_graph = CallGraph::Build(module.get());
    const CallGraphNode& c_node = flat_call_graph->GetNode(c_computation);
    EXPECT_EQ(1, c_node.caller_callsites().size());
  }
}

// Test corner case of a computation used as a body and a loop condition.
TEST_F(FlattenCallGraphTest, SharedWhileConditionAndBody) {
  auto module = CreateNewModule();
  HloComputation* cond_computation;
  {
    HloComputation::Builder builder(TestName() + ".cond");
    HloInstruction* param0 =
        builder.AddInstruction(HloInstruction::CreateParameter(
            0, ShapeUtil::MakeShape(PRED, {}), "param0"));
    HloInstruction* false_constant = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
    builder.AddInstruction(
        HloInstruction::CreateBinary(ShapeUtil::MakeShape(PRED, {}),
                                     HloOpcode::kEq, param0, false_constant));
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
    TF_ASSIGN_OR_ASSERT_OK(bool result, RunFlattenCallGraph(module.get()));
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
  auto module = CreateNewModule();
  HloComputation* c_computation =
      module->AddEmbeddedComputation(MakeScalarComputation());

  HloComputation* b_computation = module->AddEmbeddedComputation(
      MakeCallingComputation(c_computation, /*callsites=*/2, ".B"));

  module->AddEntryComputation(
      MakeCallingComputation(b_computation, /*callsites=*/2, ".Entry"));

  TF_ASSIGN_OR_ASSERT_OK(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  EXPECT_EQ(7, module->computations().size());

  const CallGraphNode& c_node = call_graph->GetNode(c_computation);
  EXPECT_EQ(1, c_node.caller_callsites().size());

  const CallGraphNode& b_node = call_graph->GetNode(b_computation);
  EXPECT_EQ(1, b_node.caller_callsites().size());
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  return xla::ParseDebugOptionsFlagsAndRunTests(argc, argv);
}

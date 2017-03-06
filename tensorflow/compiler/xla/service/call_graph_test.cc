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

#include "tensorflow/compiler/xla/service/call_graph.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class CallGraphTest : public HloTestBase {
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
      HloComputation* map_computation, int64 callsites) {
    HloComputation::Builder builder(TestName() + ".CallingComputation");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* last_value = param0;
    for (int64 i = 0; i < callsites; ++i) {
      last_value = builder.AddInstruction(HloInstruction::CreateCall(
          kScalarShape, {last_value}, map_computation));
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

  const Shape kScalarShape = ShapeUtil::MakeShape(F32, {});
};

TEST_F(CallGraphTest, SingletonComputation) {
  // Test the call graph of a module with a single computation.
  HloModule module(TestName());
  HloComputation* computation =
      module.AddEntryComputation(MakeScalarComputation());
  TF_ASSIGN_OR_ASSERT_OK(const CallGraph call_graph, CallGraph::Build(&module));
  EXPECT_EQ(1, call_graph.nodes().size());
  TF_ASSIGN_OR_ASSERT_OK(const CallGraphNode* node,
                         call_graph.GetNode(computation));
  EXPECT_EQ(computation, node->computation());
  EXPECT_TRUE(node->callsites().empty());
  EXPECT_TRUE(node->callees().empty());
  EXPECT_TRUE(node->caller_callsites().empty());
  EXPECT_TRUE(node->callers().empty());
  EXPECT_EQ(CallContext::kSequential, node->context());
}

TEST_F(CallGraphTest, UnreachableComputation) {
  // Test the call graph of a module with an entry computation and an
  // unreachable computation.
  HloModule module(TestName());
  HloComputation* entry_computation =
      module.AddEntryComputation(MakeScalarComputation());
  HloComputation* unreachable_computation =
      module.AddEmbeddedComputation(MakeScalarComputation());

  TF_ASSIGN_OR_ASSERT_OK(const CallGraph call_graph, CallGraph::Build(&module));
  EXPECT_EQ(2, call_graph.nodes().size());

  TF_ASSIGN_OR_ASSERT_OK(const CallGraphNode* entry_node,
                         call_graph.GetNode(entry_computation));
  EXPECT_EQ(entry_computation, entry_node->computation());
  EXPECT_EQ(CallContext::kSequential, entry_node->context());

  TF_ASSIGN_OR_ASSERT_OK(const CallGraphNode* unreachable_node,
                         call_graph.GetNode(unreachable_computation));
  EXPECT_EQ(unreachable_computation, unreachable_node->computation());
  EXPECT_EQ(CallContext::kSequential, unreachable_node->context());
}

TEST_F(CallGraphTest, ParallelComputation) {
  // Test a call graph of a module with an entry computation which calls another
  // computation in a parallel context via kMap.
  HloModule module(TestName());
  HloComputation* map_computation =
      module.AddEmbeddedComputation(MakeScalarComputation());
  HloComputation* entry_computation = module.AddEmbeddedComputation(
      MakeMappingComputation(map_computation, /*callsites=*/5));

  TF_ASSIGN_OR_ASSERT_OK(const CallGraph call_graph, CallGraph::Build(&module));
  EXPECT_EQ(2, call_graph.nodes().size());

  TF_ASSIGN_OR_ASSERT_OK(const CallGraphNode* entry_node,
                         call_graph.GetNode(entry_computation));
  EXPECT_EQ(entry_computation, entry_node->computation());
  EXPECT_EQ(CallContext::kSequential, entry_node->context());
  EXPECT_EQ(5, entry_node->callsites().size());
  EXPECT_EQ(1, entry_node->callees().size());
  EXPECT_TRUE(entry_node->caller_callsites().empty());
  EXPECT_TRUE(entry_node->callers().empty());

  TF_ASSIGN_OR_ASSERT_OK(const CallGraphNode* map_node,
                         call_graph.GetNode(map_computation));
  EXPECT_EQ(map_computation, map_node->computation());
  EXPECT_EQ(CallContext::kParallel, map_node->context());
  EXPECT_TRUE(map_node->callsites().empty());
  EXPECT_TRUE(map_node->callees().empty());
  EXPECT_EQ(5, map_node->caller_callsites().size());
  EXPECT_EQ(1, map_node->callers().size());
}

TEST_F(CallGraphTest, SequentialComputations) {
  // Test a call graph of a module with an entry computation which calls another
  // computation in a sequential context via kCall.
  HloModule module(TestName());
  HloComputation* called_computation =
      module.AddEmbeddedComputation(MakeScalarComputation());
  HloComputation* entry_computation = module.AddEmbeddedComputation(
      MakeCallingComputation(called_computation, /*callsites=*/3));

  TF_ASSIGN_OR_ASSERT_OK(const CallGraph call_graph, CallGraph::Build(&module));
  EXPECT_EQ(2, call_graph.nodes().size());

  TF_ASSIGN_OR_ASSERT_OK(const CallGraphNode* entry_node,
                         call_graph.GetNode(entry_computation));
  EXPECT_EQ(entry_computation, entry_node->computation());
  EXPECT_EQ(CallContext::kSequential, entry_node->context());
  EXPECT_EQ(3, entry_node->callsites().size());
  EXPECT_EQ(1, entry_node->callees().size());
  EXPECT_TRUE(entry_node->caller_callsites().empty());
  EXPECT_TRUE(entry_node->callers().empty());

  TF_ASSIGN_OR_ASSERT_OK(const CallGraphNode* called_node,
                         call_graph.GetNode(called_computation));
  EXPECT_EQ(called_computation, called_node->computation());
  EXPECT_EQ(CallContext::kSequential, called_node->context());
  EXPECT_TRUE(called_node->callsites().empty());
  EXPECT_TRUE(called_node->callees().empty());
  EXPECT_EQ(3, called_node->caller_callsites().size());
  EXPECT_EQ(1, called_node->callers().size());
}

TEST_F(CallGraphTest, ContextBothComputations) {
  // Test a call graph of a module with an entry computation which calls another
  // computation in both a parallel and sequential context.
  HloModule module(TestName());
  HloComputation* subcomputation =
      module.AddEmbeddedComputation(MakeScalarComputation());

  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, kScalarShape, "param0"));
  HloInstruction* call = builder.AddInstruction(
      HloInstruction::CreateCall(kScalarShape, {param0}, subcomputation));
  HloInstruction* map = builder.AddInstruction(
      HloInstruction::CreateMap(kScalarShape, {call}, subcomputation));
  HloComputation* entry_computation =
      module.AddEmbeddedComputation(builder.Build());

  TF_ASSIGN_OR_ASSERT_OK(const CallGraph call_graph, CallGraph::Build(&module));
  EXPECT_EQ(2, call_graph.nodes().size());

  TF_ASSIGN_OR_ASSERT_OK(const CallGraphNode* entry_node,
                         call_graph.GetNode(entry_computation));
  EXPECT_EQ(entry_computation, entry_node->computation());
  EXPECT_EQ(2, entry_node->callsites().size());

  const CallSite& call_callsite = entry_node->callsites()[0];
  EXPECT_EQ(call, call_callsite.instruction);
  EXPECT_EQ(subcomputation, call_callsite.called_computation);
  EXPECT_EQ(CallContext::kSequential, call_callsite.context);

  const CallSite& map_callsite = entry_node->callsites()[1];
  EXPECT_EQ(map, map_callsite.instruction);
  EXPECT_EQ(subcomputation, map_callsite.called_computation);
  EXPECT_EQ(CallContext::kParallel, map_callsite.context);

  TF_ASSIGN_OR_ASSERT_OK(const CallGraphNode* sub_node,
                         call_graph.GetNode(subcomputation));
  EXPECT_EQ(CallContext::kBoth, sub_node->context());
}

TEST_F(CallGraphTest, ComplexGraph) {
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
  HloModule module(TestName());
  HloComputation* cond_computation =
      module.AddEmbeddedComputation(MakeConditionComputation());
  HloComputation* c_computation =
      module.AddEmbeddedComputation(MakeScalarComputation());
  HloComputation* b_computation = module.AddEmbeddedComputation(
      MakeMappingComputation(c_computation, /*callsites=*/1));

  HloComputation* computation_a;
  {
    HloComputation::Builder builder(TestName() + ".a");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* call = builder.AddInstruction(
        HloInstruction::CreateCall(kScalarShape, {param0}, c_computation));
    builder.AddInstruction(HloInstruction::CreateWhile(
        kScalarShape, cond_computation, b_computation, call));
    computation_a = module.AddEmbeddedComputation(builder.Build());
  }

  HloComputation* entry_computation;
  {
    HloComputation::Builder builder(TestName() + ".entry");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    builder.AddInstruction(HloInstruction::CreateWhile(
        kScalarShape, cond_computation, computation_a, param0));
    entry_computation = module.AddEntryComputation(builder.Build());
  }

  TF_ASSIGN_OR_ASSERT_OK(const CallGraph call_graph, CallGraph::Build(&module));
  EXPECT_EQ(5, call_graph.nodes().size());

  TF_ASSIGN_OR_ASSERT_OK(const CallGraphNode* entry_node,
                         call_graph.GetNode(entry_computation));
  // Entry computation has one while instruction (two callsites).
  EXPECT_EQ(2, entry_node->callsites().size());
  EXPECT_EQ(CallContext::kSequential, entry_node->context());

  TF_ASSIGN_OR_ASSERT_OK(const CallGraphNode* c_node,
                         call_graph.GetNode(c_computation));
  EXPECT_TRUE(c_node->callsites().empty());
  EXPECT_EQ(2, c_node->callers().size());
  EXPECT_EQ(CallContext::kBoth, c_node->context());
}

}  // namespace
}  // namespace xla

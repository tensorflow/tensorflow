/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/deadness_analysis.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/graph_def_builder_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

Status AnalyzeDeadness(Graph* graph,
                       std::unique_ptr<DeadnessAnalysis>* result) {
  FixupSourceAndSinkEdges(graph);
  return DeadnessAnalysis::Run(*graph, result);
}

ops::Switch CreateSwitch(const Scope& root, const string& prefix) {
  Output value = ops::Placeholder(root.WithOpName(prefix + "/value"), DT_FLOAT);
  Output predicate =
      ops::Placeholder(root.WithOpName(prefix + "/pred"), DT_BOOL);
  return ops::Switch(root.WithOpName(prefix + "/switch"), value, predicate);
}

Output CreateInductionVariable(const Scope& root, const string& prefix,
                               const string& frame_name, int32 init) {
  Output initial_value = ops::Const(root.WithOpName(prefix + "/init"), init);
  Output enter_initial_value = ops::internal::Enter(
      root.WithOpName(prefix + "/enter"), initial_value, frame_name);

  ops::Merge iv(root.WithOpName(prefix + "/iv"), {enter_initial_value});
  Output increment_by = ops::Const(root.WithOpName(prefix + "/incr"), 1);
  Output final_value = ops::Const(root.WithOpName(prefix + "/final"), 10);
  Output loop_cond_expr =
      ops::Less(root.WithOpName(prefix + "/less"), iv.output, final_value);
  Output loop_cond =
      ops::LoopCond(root.WithOpName(prefix + "/cond"), loop_cond_expr);
  ops::Switch latch(root.WithOpName(prefix + "/latch"), iv.output, loop_cond);
  ops::internal::Exit exit(root.WithOpName(prefix + "/exit"), iv.output);
  Output iv_next =
      ops::Add(root.WithOpName(prefix + "/ivnext"), iv.output, increment_by);
  Output next_iteration =
      ops::NextIteration(root.WithOpName(prefix + "next_iteration"), iv_next);

  root.graph()->AddEdge(next_iteration.node(), 0, iv.output.node(), 1);
  root.graph()->AddControlEdge(iv.output.node(), increment_by.node());
  root.graph()->AddControlEdge(iv.output.node(), final_value.node());

  return iv.output;
}

TEST(DeadnessAnalysisTest, BasicPositive) {
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw = CreateSwitch(root, "0");
  Output add =
      ops::Add(root.WithOpName("add"), sw.output_true, sw.output_false);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_TRUE(result->HasInputsWithMismatchingDeadness(*add.node()));
}

TEST(DeadnessAnalysisTest, BasicNegative) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::Placeholder(root.WithOpName("a"), DT_FLOAT);
  Output b = ops::Placeholder(root.WithOpName("b"), DT_FLOAT);
  Output add = ops::Add(root.WithOpName("add"), a, b);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_FALSE(result->HasInputsWithMismatchingDeadness(*add.node()));
}

TEST(DeadnessAnalysisTest, AndIsCommutative) {
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "0");
  ops::Switch sw_1 = CreateSwitch(root, "1");

  Output a0 =
      ops::Add(root.WithOpName("a0"), sw_0.output_false, sw_1.output_false);
  Output a1 =
      ops::Add(root.WithOpName("a1"), sw_1.output_false, sw_0.output_false);

  Output b0 =
      ops::Add(root.WithOpName("b0"), sw_0.output_false, sw_1.output_true);
  Output b1 =
      ops::Add(root.WithOpName("b1"), sw_1.output_true, sw_0.output_false);

  Output live0 = ops::Add(root.WithOpName("live0"), a0, a1);
  Output live1 = ops::Add(root.WithOpName("live1"), b0, b1);

  Output halfdead0 = ops::Add(root.WithOpName("halfdead0"), a0, b0);
  Output halfdead1 = ops::Add(root.WithOpName("halfdead1"), a1, b1);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_FALSE(result->HasInputsWithMismatchingDeadness(*live0.node()));
  EXPECT_FALSE(result->HasInputsWithMismatchingDeadness(*live1.node()));

  EXPECT_TRUE(result->HasInputsWithMismatchingDeadness(*halfdead0.node()));
  EXPECT_TRUE(result->HasInputsWithMismatchingDeadness(*halfdead1.node()));
}

TEST(DeadnessAnalysisTest, AndIsAssociative) {
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "0");
  ops::Switch sw_1 = CreateSwitch(root, "1");
  ops::Switch sw_2 = CreateSwitch(root, "2");

  Output a0 =
      ops::Add(root.WithOpName("a0"), sw_0.output_false, sw_1.output_false);
  Output a1 = ops::Add(root.WithOpName("a1"), a0, sw_2.output_false);

  Output b0 =
      ops::Add(root.WithOpName("b0"), sw_1.output_false, sw_2.output_false);
  Output b1 = ops::Add(root.WithOpName("b1"), sw_0.output_false, b0);

  Output add = ops::Add(root.WithOpName("add"), a1, b1);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_FALSE(result->HasInputsWithMismatchingDeadness(*add.node()));
}

TEST(DeadnessAnalysisTest, OrIsCommutative) {
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "0");
  ops::Switch sw_1 = CreateSwitch(root, "1");

  ops::Merge m0(root.WithOpName("m0"), {sw_0.output_false, sw_1.output_false});
  ops::Merge m1(root.WithOpName("m1"), {sw_1.output_false, sw_0.output_false});
  ops::Merge m2(root.WithOpName("m2"), {sw_0.output_false, sw_1.output_true});
  ops::Merge m3(root.WithOpName("m3"), {sw_1.output_true, sw_0.output_false});

  Output live0 = ops::Add(root.WithOpName("live0"), m0.output, m1.output);
  Output live1 = ops::Add(root.WithOpName("live1"), m2.output, m3.output);

  Output halfdead0 =
      ops::Add(root.WithOpName("halfdead0"), m0.output, m2.output);
  Output halfdead1 =
      ops::Add(root.WithOpName("halfdead1"), m1.output, m3.output);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_FALSE(result->HasInputsWithMismatchingDeadness(*live0.node()));
  EXPECT_FALSE(result->HasInputsWithMismatchingDeadness(*live1.node()));

  EXPECT_TRUE(result->HasInputsWithMismatchingDeadness(*halfdead0.node()));
  EXPECT_TRUE(result->HasInputsWithMismatchingDeadness(*halfdead1.node()));
}

TEST(DeadnessAnalysisTest, OrIsAssociative) {
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "0");
  ops::Switch sw_1 = CreateSwitch(root, "1");
  ops::Switch sw_2 = CreateSwitch(root, "2");

  ops::Merge m0(root.WithOpName("m0"), {sw_0.output_false, sw_1.output_false});
  ops::Merge m1(root.WithOpName("m1"), {m0.output, sw_2.output_false});
  ops::Merge m2(root.WithOpName("m2"), {sw_1.output_false, sw_2.output_false});
  ops::Merge m3(root.WithOpName("m3"), {sw_0.output_false, m2.output});

  Output add = ops::Add(root.WithOpName("add"), m1.output, m3.output);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_FALSE(result->HasInputsWithMismatchingDeadness(*add.node()));
}

TEST(DeadnessAnalysisTest, AndOfOr) {
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "0");
  ops::Switch sw_1 = CreateSwitch(root, "1");
  ops::Switch sw_2 = CreateSwitch(root, "2");
  ops::Switch sw_3 = CreateSwitch(root, "3");

  ops::Merge m0(root.WithOpName("m0"), {sw_0.output_false, sw_1.output_false});
  ops::Merge m1(root.WithOpName("m1"), {sw_2.output_false, sw_3.output_false});

  Output add0 = ops::Add(root.WithOpName("add0"), m0.output, m1.output);
  Output add1 = ops::Add(root.WithOpName("add1"), m0.output, m1.output);

  Output add2 = ops::Add(root.WithOpName("add2"), add0, add1);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_FALSE(result->HasInputsWithMismatchingDeadness(*add2.node()));
}

TEST(DeadnessAnalysisTest, OrOfAnd) {
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "0");
  ops::Switch sw_1 = CreateSwitch(root, "1");
  ops::Switch sw_2 = CreateSwitch(root, "2");
  ops::Switch sw_3 = CreateSwitch(root, "3");

  Output add0 =
      ops::Add(root.WithOpName("add0"), sw_0.output_false, sw_1.output_false);
  Output add1 =
      ops::Add(root.WithOpName("add1"), sw_2.output_false, sw_3.output_false);

  ops::Merge m0(root.WithOpName("m0"), {add0, add1});
  ops::Merge m1(root.WithOpName("m1"), {add0, add1});

  Output add2 = ops::Add(root.WithOpName("add2"), m0.output, m1.output);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_FALSE(result->HasInputsWithMismatchingDeadness(*add2.node()));
}

TEST(DeadnessAnalysisTest, NEGATIVE_AndOrDistributive) {
  // This demonstrates one of the weaknesses in the current approach -- since we
  // only do some basic simplifications we can't see that "(A|B)&C" ==
  // "(A&C)|(B&C)".
  Scope root = Scope::NewRootScope().ExitOnError();

  ops::Switch sw_0 = CreateSwitch(root, "0");
  ops::Switch sw_1 = CreateSwitch(root, "1");
  ops::Switch sw_2 = CreateSwitch(root, "2");

  ops::Merge m0(root.WithOpName("m0"), {sw_0.output_false, sw_1.output_false});
  Output add0 = ops::Add(root.WithOpName("add0"), m0.output, sw_2.output_false);

  Output add1 =
      ops::Add(root.WithOpName("add1"), sw_0.output_false, sw_2.output_false);
  Output add2 =
      ops::Add(root.WithOpName("add2"), sw_1.output_false, sw_2.output_false);
  ops::Merge m1(root.WithOpName("m1"), {add1, add2});

  Output add3 = ops::Add(root.WithOpName("add3"), add0, m1.output);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_TRUE(result->HasInputsWithMismatchingDeadness(*add2.node()));
}

TEST(DeadnessAnalysisTest, Ternary) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output predicate = ops::Placeholder(root.WithOpName("predicate"), DT_BOOL);
  Output true_value = ops::Placeholder(root.WithOpName("true_value"), DT_FLOAT);
  Output false_value =
      ops::Placeholder(root.WithOpName("false_value"), DT_FLOAT);

  ops::Switch predicated_true(root.WithOpName("predicated_true"), true_value,
                              predicate);

  ops::Switch predicated_false(root.WithOpName("predicated_false"), true_value,
                               predicate);
  ops::Merge merge(root.WithOpName("ternary"), {predicated_true.output_true,
                                                predicated_false.output_false});
  Output addend = ops::Placeholder(root.WithOpName("addend"), DT_FLOAT);
  Output add = ops::Add(root.WithOpName("add"), merge.output, addend);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_FALSE(result->HasInputsWithMismatchingDeadness(*add.node()));
}

TEST(DeadnessAnalysisTest, Recv) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output recv_a = ops::_Recv(root.WithOpName("recv_a"), DT_FLOAT, "tensor_a",
                             "sender", 0, "receiver");
  Output recv_b = ops::_Recv(root.WithOpName("recv_b"), DT_FLOAT, "tensor_b",
                             "sender", 0, "receiver");
  Output add = ops::Add(root.WithOpName("add"), recv_a, recv_b);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_TRUE(result->HasInputsWithMismatchingDeadness(*add.node()));
}

TEST(DeadnessAnalysisTest, HostRecv) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output recv_a = ops::_HostRecv(root.WithOpName("recv_a"), DT_FLOAT,
                                 "tensor_a", "sender", 0, "receiver");
  Output recv_b = ops::_HostRecv(root.WithOpName("recv_b"), DT_FLOAT,
                                 "tensor_b", "sender", 0, "receiver");
  Output add = ops::Add(root.WithOpName("add"), recv_a, recv_b);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_TRUE(result->HasInputsWithMismatchingDeadness(*add.node()));
}

TEST(DeadnessAnalysisTest, Loop) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output iv0 = CreateInductionVariable(root, "iv0", "fr0", 0);
  Output iv1 = CreateInductionVariable(root, "iv1", "fr0", 0);
  Output iv2 = CreateInductionVariable(root, "iv2", "fr0", 1);
  Output add0 = ops::Add(root.WithOpName("add0"), iv0, iv1);
  Output add1 = ops::Add(root.WithOpName("add1"), iv1, iv2);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  // NB!  iv0 and iv1 are equivalent and a smarter deadness analysis would have
  // noticed that.  Today we are pessimistic here because we assign an
  // uninterpreted symbol to merges with backedges.

  EXPECT_TRUE(result->HasInputsWithMismatchingDeadness(*add0.node()));
  EXPECT_TRUE(result->HasInputsWithMismatchingDeadness(*add1.node()));
}

TEST(DeadnessAnalysisTest, ControlInputs) {
  Scope root = Scope::NewRootScope().ExitOnError();
  ops::Switch sw = CreateSwitch(root, "0");

  Output id0 = ops::Identity(root.WithOpName("id0"), sw.output_false);
  Output id1 = ops::Identity(root.WithOpName("id1"), sw.output_true);

  Output const0 = ops::Const(root.WithOpName("const0"), 1);
  Output const1 = ops::Const(root.WithOpName("const1"), 2);

  Output add = ops::Add(root.WithOpName("add"), const0, const1);

  root.graph()->AddControlEdge(id0.node(), const0.node());
  root.graph()->AddControlEdge(id1.node(), const1.node());

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_TRUE(result->HasInputsWithMismatchingDeadness(*add.node()));
}

TEST(DeadnessAnalysisTest, ControlTrigger) {
  Scope root = Scope::NewRootScope().ExitOnError();
  ops::Switch sw = CreateSwitch(root, "0");

  Output id0 = ops::Identity(root.WithOpName("id0"), sw.output_false);
  Output id1 = ops::Identity(root.WithOpName("id1"), sw.output_true);

  ops::ControlTrigger ctrl_trigger0(root.WithOpName("ctrl_trigger0"));
  ops::ControlTrigger ctrl_trigger1(root.WithOpName("ctrl_trigger1"));

  Output const0 = ops::Const(root.WithOpName("const0"), 1);
  Output const1 = ops::Const(root.WithOpName("const1"), 2);

  Output add = ops::Add(root.WithOpName("add"), const0, const1);

  root.graph()->AddControlEdge(id0.node(), ctrl_trigger0.operation.node());
  root.graph()->AddControlEdge(ctrl_trigger0.operation.node(), const0.node());

  root.graph()->AddControlEdge(id1.node(), ctrl_trigger1.operation.node());
  root.graph()->AddControlEdge(ctrl_trigger1.operation.node(), const1.node());

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_FALSE(result->HasInputsWithMismatchingDeadness(*add.node()));
}

TEST(DeadnessAnalysisTest, ControlInputsToMerge) {
  Scope root = Scope::NewRootScope().ExitOnError();
  ops::Switch sw = CreateSwitch(root, "0");

  Output id0 = ops::Identity(root.WithOpName("id0"), sw.output_false);
  Output id1 = ops::Identity(root.WithOpName("id1"), sw.output_true);

  Output constant = ops::Const(root.WithOpName("constant"), 5);
  ops::Merge m0(root.WithOpName("m0"), {constant});
  ops::Merge m1(root.WithOpName("m0"), {constant});
  Output add = ops::Add(root.WithOpName("add"), m0.output, m1.output);

  root.graph()->AddControlEdge(id0.node(), m0.output.node());
  root.graph()->AddControlEdge(id1.node(), m1.output.node());

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_FALSE(result->HasInputsWithMismatchingDeadness(*add.node()));
}

TEST(DeadnessAnalysisTest, RecvVsSwitch) {
  // Demonstrates why we need the must_be_true bit on SymbolP.
  Scope root = Scope::NewRootScope().ExitOnError();

  Output recv = ops::_Recv(root.WithOpName("recv"), DT_BOOL, "tensor", "sender",
                           0, "receiver");
  Output value = ops::Placeholder(root.WithOpName("value"), DT_BOOL);
  ops::Switch sw(root.WithOpName("switch"), value, recv);
  Output logical_and =
      ops::LogicalAnd(root.WithOpName("and"), recv, sw.output_true);

  std::unique_ptr<DeadnessAnalysis> result;
  TF_ASSERT_OK(AnalyzeDeadness(root.graph(), &result));

  EXPECT_TRUE(result->HasInputsWithMismatchingDeadness(*logical_and.node()));
}

}  // namespace
}  // namespace tensorflow

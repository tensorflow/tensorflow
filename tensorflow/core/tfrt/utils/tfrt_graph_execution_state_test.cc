/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/while_loop.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::EqualsProto;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::proto::IgnoringFieldPaths;
using ::testing::proto::IgnoringRepeatedFieldOrdering;

class PruneGraphDefTest : public grappler::GrapplerTest {};

TEST_F(PruneGraphDefTest, ConstFeedWithInput) {
  GraphDef graphdef;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output a = ops::Const(scope.WithOpName("a"), 0.0f, {10, 10});

    Output b = ops::Const(scope.WithControlDependencies(a).WithOpName("b"),
                          0.0f, {10, 10});
    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  CallableOptions callable_options;
  callable_options.add_feed("b");
  callable_options.add_fetch("c");

  TF_ASSERT_OK(PruneGraphDef(graphdef, callable_options));

  GraphDef expected;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output b = ops::Const(scope.WithOpName("b"), 0.0f, {10, 10});
    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&expected));
  }

  CompareGraphs(expected, graphdef);
}

Status LessThanTenCond(const Scope& scope, const std::vector<Output>& inputs,
                       Output* output) {
  *output = ops::Less(scope, inputs[0], 10);
  return scope.status();
}

Status AddOneBody(const Scope& scope, const std::vector<Output>& inputs,
                  std::vector<Output>* outputs) {
  outputs->push_back(ops::AddN(scope, {inputs[0], 1}));
  return scope.status();
}

TEST_F(PruneGraphDefTest, InsertIdentityForLoopExitFeed) {
  GraphDef graphdef;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    std::vector<Output> inputs;
    inputs.push_back(ops::Placeholder(scope.WithOpName("input"), DT_INT32));
    std::vector<Output> outputs;
    TF_ASSERT_OK(ops::BuildWhileLoop(scope.NewSubScope("while"), inputs,
                                     LessThanTenCond, AddOneBody, "test_loop",
                                     &outputs));

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  CallableOptions callable_options;
  callable_options.add_feed("input");
  callable_options.add_fetch("while/Exit");

  TF_ASSERT_OK(PruneGraphDef(graphdef, callable_options));

  for (const auto& node : graphdef.node()) {
    if (node.op() == "Exit") {
      EXPECT_EQ(node.name(), "while/Exit/tfrt_renamed");
    }
    if (node.name() == "while/Exit") {
      EXPECT_EQ(node.op(), "Identity");
      ASSERT_EQ(node.input().size(), 1);
      EXPECT_EQ(node.input(0), "while/Exit/tfrt_renamed");
    }
  }
}

TEST_F(PruneGraphDefTest, EliminateRefEntersFromControlFlow) {
  GraphDef graphdef;
  absl::flat_hash_map<std::string, NodeDef> name_to_node;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    std::vector<Output> inputs;
    inputs.push_back(ops::Placeholder(scope.WithOpName("input"), DT_INT32));
    std::vector<Output> outputs1;
    std::vector<Output> outputs2;
    TF_ASSERT_OK(ops::BuildWhileLoop(scope.NewSubScope("while"), inputs,
                                     LessThanTenCond, AddOneBody, "test_loop",
                                     &outputs1));
    TF_ASSERT_OK(ops::BuildWhileLoop(scope.NewSubScope("while"), inputs,
                                     LessThanTenCond, AddOneBody, "test_loop2",
                                     &outputs2));

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));

    // Simply replace Enter with RefEnter. Note this is not valid graph though.
    for (auto& node : *graphdef.mutable_node()) {
      if (node.op() == "Enter") {
        node.set_op("RefEnter");
      }
      name_to_node.insert({node.name(), node});
    }
  }

  TF_ASSERT_OK(EliminateRefVariablesFromV1ControlFlow(graphdef));

  int num_identity_op = 0;
  int num_enter_op = 0;
  int num_ref_enter_op = 0;
  for (const auto& node : graphdef.node()) {
    if (node.op() == "Identity") {
      num_identity_op++;
      EXPECT_EQ(node.name(), "input/identity");
      ASSERT_EQ(node.input().size(), 1);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_THAT(node.attr(), ElementsAre(Pair("T", _)));
    } else if (node.op() == "RefEnter") {
      num_ref_enter_op++;
    } else if (node.op() == "Enter") {
      // Identity op should be placed before Enter.
      EXPECT_EQ(num_identity_op, 1);
      num_enter_op++;
      ASSERT_EQ(node.input().size(), 1);
      EXPECT_EQ(node.input(0), "input/identity");
      EXPECT_THAT(
          node, IgnoringFieldPaths({"input", "op"},
                                   EqualsProto(name_to_node.at(node.name()))));
    } else {
      EXPECT_THAT(node, EqualsProto(name_to_node.at(node.name())));
    }
    name_to_node.erase(node.name());
  }
  EXPECT_EQ(num_identity_op, 1);
  EXPECT_EQ(num_enter_op, 2);
  EXPECT_EQ(num_ref_enter_op, 0);
  EXPECT_THAT(name_to_node, IsEmpty());
}

TEST_F(PruneGraphDefTest, EliminateRefSwitchesFromControlFlow) {
  GraphDef graphdef;
  absl::flat_hash_map<std::string, NodeDef> name_to_node;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output cond_a = ops::Placeholder(scope.WithOpName("cond_a"), DT_BOOL);
    Output cond_b = ops::Placeholder(scope.WithOpName("cond_b"), DT_BOOL);
    Output input = ops::Placeholder(scope.WithOpName("input"), DT_FLOAT);

    ops::Switch switch_a(scope.WithOpName("switch_a"), input, cond_a);
    ops::Switch switch_b(scope.WithOpName("switch_b"), input, cond_b);

    Output switch_a_true =
        ops::Identity(scope.WithOpName("switch_a_true"), switch_a.output_true);
    Output switch_b_true =
        ops::Identity(scope.WithOpName("switch_b_true"), switch_b.output_true);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));

    // Simply replace Switch with RefSwitch. Note this is not valid graph
    // though.
    for (auto& node : *graphdef.mutable_node()) {
      if (node.op() == "Switch") {
        node.set_op("RefSwitch");
      }
      name_to_node.insert({node.name(), node});
    }
  }

  TF_ASSERT_OK(EliminateRefVariablesFromV1ControlFlow(graphdef));

  int num_identity_op = 0;
  int num_switch_op = 0;
  int num_ref_switch_op = 0;
  for (const auto& node : graphdef.node()) {
    if (node.name() == "switch_a_true" || node.name() == "switch_b_true") {
      EXPECT_THAT(node, EqualsProto(name_to_node.at(node.name())));
    } else if (node.op() == "Identity") {
      num_identity_op++;
      EXPECT_EQ(node.name(), "input/identity");
      ASSERT_EQ(node.input().size(), 1);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_THAT(node.attr(), ElementsAre(Pair("T", _)));
    } else if (node.op() == "RefSwitch") {
      num_ref_switch_op++;
    } else if (node.op() == "Switch") {
      // Identity op should be placed before Switch.
      EXPECT_EQ(num_identity_op, 1);
      num_switch_op++;
      ASSERT_EQ(node.input().size(), 2);
      EXPECT_TRUE(node.input(0) == "input/identity" ||
                  node.input(1) == "input/identity");
      EXPECT_THAT(
          node, IgnoringFieldPaths({"input", "op"},
                                   EqualsProto(name_to_node.at(node.name()))));
    } else {
      EXPECT_THAT(node, EqualsProto(name_to_node.at(node.name())));
    }
    name_to_node.erase(node.name());
  }
  EXPECT_EQ(num_identity_op, 1);
  EXPECT_EQ(num_switch_op, 2);
  EXPECT_EQ(num_ref_switch_op, 0);
  EXPECT_THAT(name_to_node, IsEmpty());
}

TEST_F(PruneGraphDefTest, EliminateRefVariablesFromV1ControlFlowFailed) {
  GraphDef graphdef;
  absl::flat_hash_map<std::string, NodeDef> name_to_node;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output cond = ops::Placeholder(scope.WithOpName("cond"), DT_BOOL);
    Output input = ops::Placeholder(scope.WithOpName("input"), DT_FLOAT);

    ops::Switch switch_op(scope.WithOpName("switch"), input, cond);
    Output var = ops::Variable(scope.WithOpName("var"), {}, DataType::DT_FLOAT);
    Output assign =
        ops::Assign(scope.WithOpName("assign"), var, switch_op.output_true);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));

    // Simply replace Switch with RefSwitch. Note this is not valid graph
    // though.
    for (auto& node : *graphdef.mutable_node()) {
      if (node.op() == "Switch") {
        node.set_op("RefSwitch");
      }
      name_to_node.insert({node.name(), node});
    }
  }

  const auto status = EliminateRefVariablesFromV1ControlFlow(graphdef);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("requires its input to be refs"));
}

TEST_F(PruneGraphDefTest, KeepLoopStructureComplete) {
  GraphDef graphdef;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    std::vector<Output> inputs;
    inputs.push_back(ops::Placeholder(scope.WithOpName("input"), DT_INT32));
    std::vector<Output> outputs;
    TF_ASSERT_OK(ops::BuildWhileLoop(scope.NewSubScope("while"), inputs,
                                     LessThanTenCond, AddOneBody, "test_loop",
                                     &outputs));

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  CallableOptions callable_options;
  callable_options.add_feed("input");
  // Sets the fetch node such that traversing from there will miss part of the
  // while loop structure.
  callable_options.add_fetch("while/LoopCond");

  GraphDef original_graphdef = graphdef;
  TF_ASSERT_OK(PruneGraphDef(graphdef, callable_options));
  EXPECT_THAT(graphdef,
              IgnoringRepeatedFieldOrdering(EqualsProto(original_graphdef)));
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow

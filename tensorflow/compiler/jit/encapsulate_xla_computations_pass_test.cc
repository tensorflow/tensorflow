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

#include "tensorflow/compiler/jit/encapsulate_xla_computations_pass.h"

#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/cc/ops/xla_jit_ops.h"
#include "tensorflow/compiler/tf2xla/test_util.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {

static std::unique_ptr<Graph> MakeOuterGraph(
    const FunctionLibraryDefinition& flib_def, const string& function) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(flib_def.ToProto()));

  auto a = ops::Placeholder(scope.WithOpName("A"), DT_INT32);
  auto b = ops::Placeholder(scope.WithOpName("B"), DT_FLOAT);
  auto c = ops::Placeholder(scope.WithOpName("C"), DT_INT32);
  auto d = ops::Placeholder(scope.WithOpName("D"), DT_FLOAT);
  auto u = ops::Placeholder(scope.WithOpName("U"), DT_RESOURCE);
  auto v = ops::Placeholder(scope.WithOpName("V"), DT_RESOURCE);
  auto w = ops::Placeholder(scope.WithOpName("W"), DT_RESOURCE);

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("launch0", function, &flib_def)
                  .Input(a.node()->name(), 0, DT_INT32)
                  .Input(b.node()->name(), 0, DT_FLOAT)
                  .Input(c.node()->name(), 0, DT_INT32)
                  .Input(d.node()->name(), 0, DT_FLOAT)
                  .Input(u.node()->name(), 0, DT_RESOURCE)
                  .Input(v.node()->name(), 0, DT_RESOURCE)
                  .Input(w.node()->name(), 0, DT_RESOURCE)
                  .Device("/gpu:0")
                  .Attr(kXlaClusterIdAttr, "launch0")
                  .Attr("_variable_start_index", 4)
                  .Finalize(&def));

  Status status;
  Node* launch = scope.graph()->AddNode(def, &status);
  TF_CHECK_OK(status);
  TF_CHECK_OK(scope.DoShapeInference(launch));
  scope.graph()->AddEdge(a.node(), 0, launch, 0);
  scope.graph()->AddEdge(b.node(), 0, launch, 1);
  scope.graph()->AddEdge(c.node(), 0, launch, 2);
  scope.graph()->AddEdge(d.node(), 0, launch, 3);
  scope.graph()->AddEdge(u.node(), 0, launch, 4);
  scope.graph()->AddEdge(v.node(), 0, launch, 5);
  scope.graph()->AddEdge(w.node(), 0, launch, 6);

  auto out0 =
      ops::XlaClusterOutput(scope.WithOpName("Out0"), Output(launch, 0));
  auto out1 =
      ops::XlaClusterOutput(scope.WithOpName("Out1"), Output(launch, 1));
  auto out2 =
      ops::XlaClusterOutput(scope.WithOpName("Out2"), Output(launch, 2));
  auto out3 =
      ops::XlaClusterOutput(scope.WithOpName("Out3"), Output(launch, 3));

  auto consumer0_a = ops::Identity(scope.WithOpName("consumer0_a"), out0);
  auto consumer0_b = ops::Identity(scope.WithOpName("consumer0_b"), out0);
  auto consumer0_c = ops::Identity(scope.WithOpName("consumer0_c"), out0);
  auto consumer1 = ops::Identity(scope.WithOpName("consumer1"), out1);
  auto consumer2 = ops::Identity(scope.WithOpName("consumer2"), out2);
  auto consumer3 = ops::Identity(scope.WithOpName("consumer3"), out3);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(scope.ToGraph(graph.get()));
  return graph;
}

// Makes an encapsulate body graph for use in tests.
static std::unique_ptr<Graph> MakeBodyGraph() {
  Scope scope = Scope::NewRootScope().ExitOnError();

  auto arg0 = ops::_Arg(scope.WithOpName("a_0_arg"), DT_INT32, 0);
  auto arg1 = ops::_Arg(scope.WithOpName("b_0_arg"), DT_FLOAT, 1);
  auto arg2 = ops::_Arg(scope.WithOpName("c_0_arg"), DT_INT32, 2);
  auto arg3 = ops::_Arg(scope.WithOpName("d_0_arg"), DT_FLOAT, 3);

  auto arg4 = ops::_Arg(scope.WithOpName("u_0_arg"), DT_RESOURCE, 4);
  auto arg5 = ops::_Arg(scope.WithOpName("v_0_arg"), DT_RESOURCE, 5);
  auto arg6 = ops::_Arg(scope.WithOpName("w_0_arg"), DT_RESOURCE, 6);

  auto add_attrs = [](Node* node) {
    node->AddAttr(kXlaClusterIdAttr, "launch0");
    node->set_requested_device("/gpu:0");
  };

  auto b_identity = ops::Identity(scope.WithOpName("B_identity"), arg1);
  add_attrs(b_identity.node());
  auto read_u = ops::ReadVariableOp(scope.WithOpName("ReadU"), arg4, DT_FLOAT);
  add_attrs(read_u.node());
  auto read_v = ops::ReadVariableOp(scope.WithOpName("ReadV"), arg5, DT_FLOAT);
  add_attrs(read_v.node());
  auto read_w = ops::ReadVariableOp(scope.WithOpName("ReadW"), arg6, DT_FLOAT);
  add_attrs(read_w.node());

  auto e = ops::Add(scope.WithOpName("E"), arg0, arg2);
  add_attrs(e.node());
  auto f = ops::Add(scope.WithOpName("F"), read_v, read_w);
  add_attrs(f.node());
  auto g = ops::Add(scope.WithOpName("G"), f, arg3);
  add_attrs(g.node());

  auto out0 = ops::_Retval(scope.WithOpName("b_identity_0_retval_RetVal"),
                           b_identity, 0);
  auto out1 = ops::_Retval(scope.WithOpName("e_0_retval_RetVal"), e, 1);
  auto out2 = ops::_Retval(scope.WithOpName("g_0_retval_RetVal"), g, 2);
  auto out3 =
      ops::_Retval(scope.WithOpName("readu_0_retval_RetVal"), read_u, 3);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(scope.ToGraph(graph.get()));
  return graph;
}

TEST(EncapsulateXlaComputations, DeterministicEncapsulate) {
  // Test that control edge insertion order doesn't affect the cache key
  // (cluster name) generated by TPU encapsulate pass.
  auto get_serialized_graph = [](bool control_input_reversed,
                                 bool operand_reversed) -> string {
    FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                       FunctionDefLibrary());
    std::unique_ptr<Graph> graph(new Graph(&flib_def));
    {
      Scope scope = Scope::NewRootScope().ExitOnError();
      auto a0 = ops::Placeholder(scope.WithOpName("A0"), DT_INT32);
      auto a1 = ops::Placeholder(scope.WithOpName("A1"), DT_INT32);

      ops::Add e = operand_reversed ? ops::Add(scope.WithOpName("E"), a0, a1)
                                    : ops::Add(scope.WithOpName("E"), a1, a0);

      auto add_attrs = [](Node* node) {
        node->AddAttr(kXlaClusterIdAttr, "launch0");
      };
      add_attrs(e.node());

      TF_CHECK_OK(scope.ToGraph(graph.get()));
      auto get_node_in_graph = [&graph](Node* node) {
        return graph->FindNodeId(node->id());
      };
      // Insert control edge in different order. The order should not affect
      // the encapsulated or serialized graph.
      if (!control_input_reversed) {
        graph->AddControlEdge(get_node_in_graph(a0.node()),
                              get_node_in_graph(e.node()), true);
        graph->AddControlEdge(get_node_in_graph(a1.node()),
                              get_node_in_graph(e.node()), true);
      } else {
        graph->AddControlEdge(get_node_in_graph(a1.node()),
                              get_node_in_graph(e.node()), true);
        graph->AddControlEdge(get_node_in_graph(a0.node()),
                              get_node_in_graph(e.node()), true);
      }
    }
    TF_CHECK_OK(EncapsulateXlaComputationsPass::Encapsulate(&graph, &flib_def));
    return SerializeGraphDeterministic(*graph).value();
  };

  // Changing the order of control input shouldn't affect the graph generated.
  EXPECT_EQ(get_serialized_graph(/*control_input_reversed=*/true,
                                 /*operand_reversed=*/false),
            get_serialized_graph(/*control_input_reversed=*/false,
                                 /*operand_reversed=*/false));

  // Changing the order of data input should affect the graph generated.
  EXPECT_NE(get_serialized_graph(/*control_input_reversed=*/false,
                                 /*operand_reversed=*/true),
            get_serialized_graph(/*control_input_reversed=*/false,
                                 /*operand_reversed=*/false));
}

TEST(EncapsulateXlaComputations, Encapsulate) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  std::unique_ptr<Graph> graph(new Graph(&flib_def));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto a = ops::Placeholder(scope.WithOpName("A"), DT_INT32);
    auto b = ops::Placeholder(scope.WithOpName("B"), DT_FLOAT);
    auto c = ops::Placeholder(scope.WithOpName("C"), DT_INT32);
    auto d = ops::Placeholder(scope.WithOpName("D"), DT_FLOAT);
    auto u = ops::Placeholder(scope.WithOpName("U"), DT_RESOURCE);
    auto v = ops::Placeholder(scope.WithOpName("V"), DT_RESOURCE);
    auto w = ops::Placeholder(scope.WithOpName("W"), DT_RESOURCE);

    auto add_attrs = [](Node* node) {
      node->AddAttr(kXlaClusterIdAttr, "launch0");
      node->set_requested_device("/gpu:0");
    };

    auto b_identity = ops::Identity(scope.WithOpName("B_identity"), b);
    add_attrs(b_identity.node());

    auto read_u = ops::ReadVariableOp(scope.WithOpName("ReadU"), u, DT_FLOAT);
    add_attrs(read_u.node());
    auto read_v = ops::ReadVariableOp(scope.WithOpName("ReadV"), v, DT_FLOAT);
    add_attrs(read_v.node());
    auto read_w = ops::ReadVariableOp(scope.WithOpName("ReadW"), w, DT_FLOAT);
    add_attrs(read_w.node());

    auto e = ops::Add(scope.WithOpName("E"), a, c);
    add_attrs(e.node());
    auto f = ops::Add(scope.WithOpName("F"), read_v, read_w);
    add_attrs(f.node());
    auto g = ops::Add(scope.WithOpName("G"), f, d);
    add_attrs(g.node());

    auto out0 = ops::XlaClusterOutput(scope.WithOpName("Out0"), b_identity);
    auto out1 = ops::XlaClusterOutput(scope.WithOpName("Out1"), e);
    auto out2 = ops::XlaClusterOutput(scope.WithOpName("Out2"), g);
    auto out3 = ops::XlaClusterOutput(scope.WithOpName("Out3"), read_u);

    auto consumer0_a = ops::Identity(scope.WithOpName("consumer0_a"), out0);
    auto consumer0_b = ops::Identity(scope.WithOpName("consumer0_b"), out0);
    auto consumer0_c = ops::Identity(scope.WithOpName("consumer0_c"), out0);
    auto consumer1 = ops::Identity(scope.WithOpName("consumer1"), out1);
    auto consumer2 = ops::Identity(scope.WithOpName("consumer2"), out2);
    auto consumer3 = ops::Identity(scope.WithOpName("consumer3"), out3);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
  }

  std::unique_ptr<Graph> graph_copy(new Graph(&flib_def));
  CopyGraph(*graph, graph_copy.get());

  TF_ASSERT_OK(EncapsulateXlaComputationsPass::Encapsulate(&graph, &flib_def));

  std::unordered_map<string, Node*> index = graph->BuildNodeNameIndex();
  string function = index.at("launch0")->type_string();

  // Tests the outer graph is as expected.
  {
    std::unique_ptr<Graph> outer = MakeOuterGraph(flib_def, function);
    GraphDef expected_def;
    outer->ToGraphDef(&expected_def);

    GraphDef actual_def;
    graph->ToGraphDef(&actual_def);
    TF_EXPECT_GRAPH_EQ_INTERNAL(expected_def, actual_def);
  }

  // Tests the encapsulated body graph is as expected.
  {
    std::unique_ptr<Graph> body = MakeBodyGraph();
    GraphDef expected_body_def;
    body->ToGraphDef(&expected_body_def);

    InstantiationResultForTest result;
    TF_EXPECT_OK(InstantiateFunctionForTest(function, flib_def, &result));

    EXPECT_EQ((DataTypeVector{DT_INT32, DT_FLOAT, DT_INT32, DT_FLOAT,
                              DT_RESOURCE, DT_RESOURCE, DT_RESOURCE}),
              result.arg_types);
    EXPECT_EQ((DataTypeVector{DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT}),
              result.ret_types);
    TF_EXPECT_GRAPH_EQ(expected_body_def, result.gdef);
  }

  // Encapsulates the same computation again, verifies we reuse the same
  // function. Encapsulation should be deterministic to avoid recompilation.
  TF_ASSERT_OK(
      EncapsulateXlaComputationsPass::Encapsulate(&graph_copy, &flib_def));
  std::unordered_map<string, Node*> index_copy =
      graph_copy->BuildNodeNameIndex();
  string function_copy = index_copy.at("launch0")->type_string();
  EXPECT_EQ(function, function_copy);
}

TEST(EncapsulateXlaComputations, BuildXlaLaunchOp) {
  std::unique_ptr<Graph> body_graph = MakeBodyGraph();
  FunctionDefLibrary flib;
  TF_ASSERT_OK(GraphToFunctionDef(*body_graph, "launch0", flib.add_function()));

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);

  std::unique_ptr<Graph> graph = MakeOuterGraph(flib_def, "launch0");
  TF_ASSERT_OK(EncapsulateXlaComputationsPass::BuildXlaLaunchOps(graph.get()));

  Scope scope = Scope::DisabledShapeInferenceScope().ExitOnError();
  TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(flib));

  auto a = ops::Placeholder(scope.WithOpName("A"), DT_INT32);
  auto b = ops::Placeholder(scope.WithOpName("B"), DT_FLOAT);
  auto c = ops::Placeholder(scope.WithOpName("C"), DT_INT32);
  auto d = ops::Placeholder(scope.WithOpName("D"), DT_FLOAT);
  auto u = ops::Placeholder(scope.WithOpName("U"), DT_RESOURCE);
  auto v = ops::Placeholder(scope.WithOpName("V"), DT_RESOURCE);
  auto w = ops::Placeholder(scope.WithOpName("W"), DT_RESOURCE);

  NameAttrList function;
  function.set_name("launch0");
  auto launch = ops::XlaLaunch(
      scope.WithOpName("launch0").WithDevice("/gpu:0"),
      std::initializer_list<Input>{}, std::initializer_list<Input>{a, b, c, d},
      std::initializer_list<Input>{u, v, w},
      DataTypeVector{DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT}, function);

  auto consumer0_a =
      ops::Identity(scope.WithOpName("consumer0_a"), launch.results[0]);
  auto consumer0_b =
      ops::Identity(scope.WithOpName("consumer0_b"), launch.results[0]);
  auto consumer0_c =
      ops::Identity(scope.WithOpName("consumer0_c"), launch.results[0]);
  auto consumer1 =
      ops::Identity(scope.WithOpName("consumer1"), launch.results[1]);
  auto consumer2 =
      ops::Identity(scope.WithOpName("consumer2"), launch.results[2]);
  auto consumer3 =
      ops::Identity(scope.WithOpName("consumer3"), launch.results[3]);

  GraphDef expected_def;
  TF_ASSERT_OK(scope.ToGraphDef(&expected_def));

  GraphDef actual_def;
  graph->ToGraphDef(&actual_def);
  TF_EXPECT_GRAPH_EQ(expected_def, actual_def);
}

}  // namespace tensorflow

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/graph_rewrite/encapsulate_tpu_computations_pass.h"

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/parsing_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/tpu_replication_ops.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/tf2xla/test_util.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/util/equal_graph_def.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

static std::unique_ptr<Graph> MakeOuterGraph(
    const FunctionLibraryDefinition& flib_def, const string& function) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(flib_def.ToProto()));

  int num_replicas = 2;

  auto a0 = ops::Placeholder(scope.WithOpName("A0"), DT_INT32);
  auto a1 = ops::Placeholder(scope.WithOpName("A1"), DT_INT32);
  auto b0 = ops::Placeholder(scope.WithOpName("B0"), DT_FLOAT);
  auto b1 = ops::Placeholder(scope.WithOpName("B1"), DT_FLOAT);
  auto u0 = ops::Placeholder(scope.WithOpName("U0"), DT_RESOURCE);
  auto u1 = ops::Placeholder(scope.WithOpName("U1"), DT_RESOURCE);
  auto z = ops::Placeholder(scope.WithOpName("Z"), DT_RESOURCE);
  auto c = ops::Placeholder(scope.WithOpName("C"), DT_INT32);
  auto d = ops::Placeholder(scope.WithOpName("D"), DT_FLOAT);
  auto v = ops::Placeholder(scope.WithOpName("V"), DT_RESOURCE);
  auto w = ops::Placeholder(scope.WithOpName("W"), DT_RESOURCE);
  auto x = ops::GuaranteeConst(
      scope.WithOpName("X"),
      ops::Placeholder(scope.WithOpName("X_Holder"), DT_DOUBLE));
  auto y = ops::GuaranteeConst(
      scope.WithOpName("Y"),
      ops::Placeholder(scope.WithOpName("Y_Holder"), DT_DOUBLE));

  auto in0 = ops::TPUReplicatedInput(scope.WithOpName("In0"),
                                     std::initializer_list<Input>{a0, a1});
  auto in1 = ops::TPUReplicatedInput(scope.WithOpName("In1"),
                                     std::initializer_list<Input>{b0, b1});
  auto in2 = ops::TPUReplicatedInput(scope.WithOpName("In2"),
                                     std::initializer_list<Input>{u0, u1});
  auto in3 = ops::TPUReplicatedInput(scope.WithOpName("In3"),
                                     std::initializer_list<Input>{z});
  in3.node()->AddAttr("is_packed", true);

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("replicate0", function, &flib_def)
                  .Input(in0.node()->name(), 0, DT_INT32)
                  .Input(in1.node()->name(), 0, DT_FLOAT)
                  .Input(in2.node()->name(), 0, DT_RESOURCE)
                  .Input(in3.node()->name(), 0, DT_RESOURCE)
                  .Input(c.node()->name(), 0, DT_INT32)
                  .Input(d.node()->name(), 0, DT_FLOAT)
                  .Input(v.node()->name(), 0, DT_RESOURCE)
                  .Input(w.node()->name(), 0, DT_RESOURCE)
                  .Input(x.node()->name(), 0, DT_DOUBLE)
                  .Input(y.node()->name(), 0, DT_DOUBLE)
                  .Attr(kTPUReplicateAttr, "replicate0")
                  .Attr("num_replicas", num_replicas)
                  .Attr("num_cores_per_replica", 6)
                  .Attr("topology", "")
                  .Attr("use_tpu", true)
                  .Attr("device_assignment", std::vector<int>())
                  .Attr("host_compute_core", std::vector<string>())
                  .Attr("padding_map", std::vector<string>())
                  .Attr("_variable_start_index", 6)
                  .Attr("_guaranteed_const_start_index", 8)
                  .Attr("allow_soft_placement", false)
                  .Attr("step_marker_location", "STEP_MARK_AT_ENTRY")
                  .Attr("use_spmd_for_xla_partitioning", false)
                  .Finalize(&def));

  Status status;
  Node* replicate = scope.graph()->AddNode(def, &status);
  TF_CHECK_OK(status);
  TF_CHECK_OK(scope.DoShapeInference(replicate));
  scope.graph()->AddEdge(in0.node(), 0, replicate, 0);
  scope.graph()->AddEdge(in1.node(), 0, replicate, 1);
  scope.graph()->AddEdge(in2.node(), 0, replicate, 2);
  scope.graph()->AddEdge(in3.node(), 0, replicate, 3);
  scope.graph()->AddEdge(c.node(), 0, replicate, 4);
  scope.graph()->AddEdge(d.node(), 0, replicate, 5);
  scope.graph()->AddEdge(v.node(), 0, replicate, 6);
  scope.graph()->AddEdge(w.node(), 0, replicate, 7);
  scope.graph()->AddEdge(x.node(), 0, replicate, 8);
  scope.graph()->AddEdge(y.node(), 0, replicate, 9);

  auto out0 = ops::TPUReplicatedOutput(scope.WithOpName("Out0"),
                                       Output(replicate, 0), num_replicas);
  auto out1 = ops::TPUReplicatedOutput(scope.WithOpName("Out1"),
                                       Output(replicate, 1), num_replicas);
  auto out2 = ops::TPUReplicatedOutput(scope.WithOpName("Out2"),
                                       Output(replicate, 2), num_replicas);
  auto out3 = ops::TPUReplicatedOutput(scope.WithOpName("Out3"),
                                       Output(replicate, 3), num_replicas);
  auto out4 = ops::TPUReplicatedOutput(scope.WithOpName("Out4"),
                                       Output(replicate, 4), num_replicas);

  auto consumer0_0a = ops::Identity(scope.WithOpName("consumer0_0a"), out0[0]);
  auto consumer0_0b = ops::Identity(scope.WithOpName("consumer0_0b"), out0[0]);
  auto consumer0_1 = ops::Identity(scope.WithOpName("consumer0_1"), out0[1]);
  auto consumer1 = ops::Identity(scope.WithOpName("consumer1"), out1[1]);
  auto consumer2 = ops::Identity(scope.WithOpName("consumer2"), out2[0]);
  auto consumer3a = ops::Identity(scope.WithOpName("consumer3a"), out3[0]);
  auto consumer3b = ops::Identity(scope.WithOpName("consumer3b"), out3[1]);
  auto consumer4a = ops::Identity(scope.WithOpName("consumer4a"), out4[0]);
  auto consumer4b = ops::Identity(scope.WithOpName("consumer4b"), out4[1]);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(scope.ToGraph(graph.get()));
  return graph;
}

// Makes an encapsulate body graph for use in tests.
static std::unique_ptr<Graph> MakeBodyGraph() {
  Scope scope = Scope::NewRootScope().ExitOnError();

  auto arg0 = ops::_Arg(scope.WithOpName("in0_0_arg"), DT_INT32, 0);
  auto arg1 = ops::_Arg(scope.WithOpName("in1_0_arg"), DT_FLOAT, 1);
  auto arg2 = ops::_Arg(scope.WithOpName("in2_0_arg"), DT_RESOURCE, 2);
  auto arg3 = ops::_Arg(scope.WithOpName("in3_0_arg"), DT_RESOURCE, 3);
  auto arg4 = ops::_Arg(scope.WithOpName("c_0_arg"), DT_INT32, 4);
  auto arg5 = ops::_Arg(scope.WithOpName("d_0_arg"), DT_FLOAT, 5);
  auto arg6 = ops::_Arg(scope.WithOpName("v_0_arg"), DT_RESOURCE, 6);
  auto arg7 = ops::_Arg(scope.WithOpName("w_0_arg"), DT_RESOURCE, 7);

  auto add_attrs = [](Node* node) {
    node->AddAttr(kTPUReplicateAttr, "replicate0");
  };

  string device =
      tensorflow::strings::StrCat("/device:", DEVICE_TPU_REPLICATED_CORE);

  auto in1_identity =
      ops::Identity(scope.WithOpName("In1_identity").WithDevice(device), arg1);

  auto read_u = ops::ReadVariableOp(
      scope.WithOpName("ReadU").WithDevice(device), arg2, DT_FLOAT);
  add_attrs(read_u.node());
  auto read_z = ops::ReadVariableOp(
      scope.WithOpName("ReadZ").WithDevice(device), arg3, DT_FLOAT);
  add_attrs(read_z.node());
  auto read_v = ops::ReadVariableOp(
      scope.WithOpName("ReadV").WithDevice(device), arg6, DT_FLOAT);
  add_attrs(read_v.node());
  auto read_w = ops::ReadVariableOp(
      scope.WithOpName("ReadW").WithDevice(device), arg7, DT_FLOAT);
  add_attrs(read_w.node());

  auto e = ops::Add(scope.WithOpName("E").WithDevice(device), arg0, arg4);
  add_attrs(e.node());
  auto f = ops::Add(scope.WithOpName("F").WithDevice(device), read_v, read_w);
  add_attrs(f.node());
  auto g = ops::Add(scope.WithOpName("G").WithDevice(device), f, arg5);
  add_attrs(g.node());

  auto arg8 = ops::_Arg(scope.WithOpName("x_0_arg"), DT_DOUBLE, 8);
  auto arg9 = ops::_Arg(scope.WithOpName("y_0_arg"), DT_DOUBLE, 9);
  arg8.node()->AddAttr("_is_guaranteed_constant", true);
  arg9.node()->AddAttr("_is_guaranteed_constant", true);
  auto h = ops::Add(scope.WithOpName("H").WithDevice(device), arg8, arg9);
  add_attrs(h.node());

  auto out0 = ops::_Retval(scope.WithOpName("e_0_retval_RetVal"), e, 0);
  auto out1 = ops::_Retval(scope.WithOpName("g_0_retval_RetVal"), g, 1);
  auto out2 = ops::_Retval(scope.WithOpName("in1_identity_0_retval_RetVal"),
                           in1_identity, 2);
  auto out3 =
      ops::_Retval(scope.WithOpName("readu_0_retval_RetVal"), read_u, 3);
  auto out4 =
      ops::_Retval(scope.WithOpName("readz_0_retval_RetVal"), read_z, 4);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(scope.ToGraph(graph.get()));
  return graph;
}

TEST(EncapsulateTPUComputations, DeterministicEncapsulate) {
  // Test that control edge insertion order doesn't affect the cache key
  // (cluster name) generated by TPU encapsulate pass.
  auto get_serialized_graph = [](bool control_input_reversed,
                                 bool operand_reversed) -> string {
    FunctionLibraryDefinition flib_def(OpRegistry::Global(), {});
    std::unique_ptr<Graph> graph(new Graph(&flib_def));
    {
      Scope scope = Scope::NewRootScope().ExitOnError();
      auto a0 = ops::Placeholder(scope.WithOpName("A0"), DT_INT32);
      auto a1 = ops::Placeholder(scope.WithOpName("A1"), DT_INT32);

      ops::Add e = operand_reversed ? ops::Add(scope.WithOpName("E"), a0, a1)
                                    : ops::Add(scope.WithOpName("E"), a1, a0);

      auto metadata = ops::TPUReplicateMetadata(scope, /*num_replicas=*/2);
      auto add_attrs = [](Node* node) {
        node->AddAttr(kTPUReplicateAttr, "replicate0");
      };
      add_attrs(metadata.operation.node());
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
    TF_CHECK_OK(EncapsulateTPUComputationsPass::Encapsulate(&graph, &flib_def));
    GraphDef gdef;
    graph->ToGraphDef(&gdef);
    // Before serialization, sort control inputs first to remove
    // nondeterminism.
    SortControlInputs(&gdef);
    string serialized;
    SerializeToStringDeterministic(gdef, &serialized);
    return serialized;
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

TEST(EncapsulateTPUComputations, Encapsulate) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), {});
  std::unique_ptr<Graph> graph(new Graph(&flib_def));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto a0 = ops::Placeholder(scope.WithOpName("A0"), DT_INT32);
    auto a1 = ops::Placeholder(scope.WithOpName("A1"), DT_INT32);
    auto b0 = ops::Placeholder(scope.WithOpName("B0"), DT_FLOAT);
    auto b1 = ops::Placeholder(scope.WithOpName("B1"), DT_FLOAT);
    auto c = ops::Placeholder(scope.WithOpName("C"), DT_INT32);
    auto d = ops::Placeholder(scope.WithOpName("D"), DT_FLOAT);
    auto u0 = ops::Placeholder(scope.WithOpName("U0"), DT_RESOURCE);
    auto u1 = ops::Placeholder(scope.WithOpName("U1"), DT_RESOURCE);
    auto z = ops::Placeholder(scope.WithOpName("Z"), DT_RESOURCE);
    auto v = ops::Placeholder(scope.WithOpName("V"), DT_RESOURCE);
    auto w = ops::Placeholder(scope.WithOpName("W"), DT_RESOURCE);
    auto x = ops::GuaranteeConst(
        scope.WithOpName("X"),
        ops::Placeholder(scope.WithOpName("X_Holder"), DT_DOUBLE));
    auto y = ops::GuaranteeConst(
        scope.WithOpName("Y"),
        ops::Placeholder(scope.WithOpName("Y_Holder"), DT_DOUBLE));

    auto in0 = ops::TPUReplicatedInput(scope.WithOpName("In0"),
                                       std::initializer_list<Input>{a0, a1});
    auto in1 = ops::TPUReplicatedInput(scope.WithOpName("In1"),
                                       std::initializer_list<Input>{b0, b1});
    auto in2 = ops::TPUReplicatedInput(scope.WithOpName("In2"),
                                       std::initializer_list<Input>{u0, u1});
    auto in3 = ops::TPUReplicatedInput(scope.WithOpName("In3"),
                                       std::initializer_list<Input>{z});
    in3.node()->AddAttr("is_packed", true);

    auto add_attrs = [](Node* node) {
      node->AddAttr(kTPUReplicateAttr, "replicate0");
    };
    auto metadata = ops::TPUReplicateMetadata(
        scope, /*num_replicas=*/2,
        ops::TPUReplicateMetadata::ComputationShape({2, 3}));
    add_attrs(metadata.operation.node());

    auto in1_identity = ops::Identity(scope.WithOpName("In1_identity"), in1);
    add_attrs(in1_identity.node());

    auto read_u = ops::ReadVariableOp(scope.WithOpName("ReadU"), in2, DT_FLOAT);
    add_attrs(read_u.node());
    auto read_z = ops::ReadVariableOp(scope.WithOpName("ReadZ"), in3, DT_FLOAT);
    add_attrs(read_z.node());
    auto read_v = ops::ReadVariableOp(scope.WithOpName("ReadV"), v, DT_FLOAT);
    add_attrs(read_v.node());
    auto read_w = ops::ReadVariableOp(scope.WithOpName("ReadW"), w, DT_FLOAT);
    add_attrs(read_w.node());

    auto e = ops::Add(scope.WithOpName("E"), in0, c);
    add_attrs(e.node());
    auto f = ops::Add(scope.WithOpName("F"), read_v, read_w);
    add_attrs(f.node());
    auto g = ops::Add(scope.WithOpName("G"), f, d);
    add_attrs(g.node());
    auto h = ops::Add(scope.WithOpName("H"), x, y);
    add_attrs(h.node());

    auto out0 = ops::TPUReplicatedOutput(scope.WithOpName("Out0"), e, 2);
    auto out1 = ops::TPUReplicatedOutput(scope.WithOpName("Out1"), g, 2);
    auto out2 =
        ops::TPUReplicatedOutput(scope.WithOpName("Out2"), in1_identity, 2);
    auto out3 = ops::TPUReplicatedOutput(scope.WithOpName("Out3"), read_u, 2);
    auto out4 = ops::TPUReplicatedOutput(scope.WithOpName("Out4"), read_z, 2);

    auto consumer0_0a =
        ops::Identity(scope.WithOpName("consumer0_0a"), out0[0]);
    auto consumer0_0b =
        ops::Identity(scope.WithOpName("consumer0_0b"), out0[0]);
    auto consumer0_1 = ops::Identity(scope.WithOpName("consumer0_1"), out0[1]);
    auto consumer1 = ops::Identity(scope.WithOpName("consumer1"), out1[1]);
    auto consumer2 = ops::Identity(scope.WithOpName("consumer2"), out2[0]);
    auto consumer3a = ops::Identity(scope.WithOpName("consumer3a"), out3[0]);
    auto consumer3b = ops::Identity(scope.WithOpName("consumer3b"), out3[1]);
    auto consumer4a = ops::Identity(scope.WithOpName("consumer4a"), out4[0]);
    auto consumer4b = ops::Identity(scope.WithOpName("consumer4b"), out4[1]);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
  }

  std::unique_ptr<Graph> graph_copy(new Graph(&flib_def));
  CopyGraph(*graph, graph_copy.get());

  TF_ASSERT_OK(EncapsulateTPUComputationsPass::Encapsulate(&graph, &flib_def));
  // Remove _xla_inferred_shapes attribute.
  for (Node* n : graph->nodes()) {
    n->ClearAttr("_xla_inferred_shapes");
  }

  std::unordered_map<string, Node*> index = graph->BuildNodeNameIndex();
  string function = index.at("replicate0")->type_string();

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

    EXPECT_EQ((DataTypeVector{DT_INT32, DT_FLOAT, DT_RESOURCE, DT_RESOURCE,
                              DT_INT32, DT_FLOAT, DT_RESOURCE, DT_RESOURCE,
                              DT_DOUBLE, DT_DOUBLE}),
              result.arg_types);
    EXPECT_EQ(
        (DataTypeVector{DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT}),
        result.ret_types);
    TF_EXPECT_GRAPH_EQ(expected_body_def, result.gdef);
  }

  // Encapsulates the same computation again, verifies we reuse the same
  // function. Encapsulation should be deterministic to avoid recompilation.
  TF_ASSERT_OK(
      EncapsulateTPUComputationsPass::Encapsulate(&graph_copy, &flib_def));
  std::unordered_map<string, Node*> index_copy =
      graph_copy->BuildNodeNameIndex();
  string function_copy = index_copy.at("replicate0")->type_string();
  EXPECT_EQ(function, function_copy);
}

TEST(EncapsulateTPUComputations, BuildTPUReplicateOps) {
  std::unique_ptr<Graph> body_graph = MakeBodyGraph();
  FunctionDefLibrary flib;
  TF_ASSERT_OK(
      GraphToFunctionDef(*body_graph, "replicate0", flib.add_function()));

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);

  std::unique_ptr<Graph> graph = MakeOuterGraph(flib_def, "replicate0");
  TF_ASSERT_OK(
      EncapsulateTPUComputationsPass::BuildTPUReplicateOps(graph.get()));

  Scope scope = Scope::NewRootScope().ExitOnError();
  TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(flib));

  auto a0 = ops::Placeholder(scope.WithOpName("A0"), DT_INT32);
  auto a1 = ops::Placeholder(scope.WithOpName("A1"), DT_INT32);
  auto b0 = ops::Placeholder(scope.WithOpName("B0"), DT_FLOAT);
  auto b1 = ops::Placeholder(scope.WithOpName("B1"), DT_FLOAT);
  auto u0 = ops::Placeholder(scope.WithOpName("U0"), DT_RESOURCE);
  auto u1 = ops::Placeholder(scope.WithOpName("U1"), DT_RESOURCE);
  auto z = ops::Placeholder(scope.WithOpName("Z"), DT_RESOURCE);
  auto c = ops::Placeholder(scope.WithOpName("C"), DT_INT32);
  auto d = ops::Placeholder(scope.WithOpName("D"), DT_FLOAT);
  auto v = ops::Placeholder(scope.WithOpName("V"), DT_RESOURCE);
  auto w = ops::Placeholder(scope.WithOpName("W"), DT_RESOURCE);
  auto x =
      ops::Identity(scope.WithOpName("X"),
                    ops::Placeholder(scope.WithOpName("X_Holder"), DT_DOUBLE));
  auto y =
      ops::Identity(scope.WithOpName("Y"),
                    ops::Placeholder(scope.WithOpName("Y_Holder"), DT_DOUBLE));

  NameAttrList function;
  function.set_name("replicate0");
  auto replicate = ops::_TPUReplicate(
      scope.WithOpName("replicate0"),
      std::initializer_list<Input>{a0, b0, u0, a1, b1, u1, z},
      std::initializer_list<Input>{c, d}, std::initializer_list<Input>{v, w},
      std::initializer_list<Input>{x, y}, function,
      /*num_replicas=*/2,
      {DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT,
       DT_FLOAT, DT_FLOAT, DT_FLOAT},
      ops::_TPUReplicate::NumCoresPerReplica(6).NumDistributedVariables(1));

  auto consumer0_0a =
      ops::Identity(scope.WithOpName("consumer0_0a"), replicate.outputs[0]);
  auto consumer0_0b =
      ops::Identity(scope.WithOpName("consumer0_0b"), replicate.outputs[0]);
  auto consumer0_1 =
      ops::Identity(scope.WithOpName("consumer0_1"), replicate.outputs[5]);
  auto consumer1 =
      ops::Identity(scope.WithOpName("consumer1"), replicate.outputs[6]);
  auto consumer2 =
      ops::Identity(scope.WithOpName("consumer2"), replicate.outputs[2]);
  auto consumer3a =
      ops::Identity(scope.WithOpName("consumer3a"), replicate.outputs[3]);
  auto consumer3b =
      ops::Identity(scope.WithOpName("consumer3b"), replicate.outputs[8]);
  auto consumer4a =
      ops::Identity(scope.WithOpName("consumer4a"), replicate.outputs[4]);
  auto consumer4b =
      ops::Identity(scope.WithOpName("consumer4b"), replicate.outputs[9]);

  GraphDef expected_def;
  TF_ASSERT_OK(scope.ToGraphDef(&expected_def));

  GraphDef actual_def;
  graph->ToGraphDef(&actual_def);
  TF_EXPECT_GRAPH_EQ(expected_def, actual_def);
}

class ExtractOutsideCompilationByScope : public ::testing::TestWithParam<bool> {
};

Status PivotControlExists(const Node* node, const Node* pivot) {
  for (const Edge* edge : node->in_edges()) {
    if (edge->IsControlEdge() && (edge->src() == pivot)) {
      return Status::OK();
    }
  }
  return errors::NotFound("Control edge with pivot not found.");
}

TEST_P(ExtractOutsideCompilationByScope,
       MoveHeadAndTailOutsideCompilationToHost) {
  FunctionLibraryDefinition fld(OpRegistry::Global(), FunctionDefLibrary());

  // Create FunctionLibraryRuntime.
  SessionOptions session_options;
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(DeviceFactory::AddDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));
  OptimizerOptions opts;
  auto device_mgr = absl::make_unique<StaticDeviceMgr>(std::move(devices));
  auto pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr.get(), Env::Default(), /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, &fld, opts,
      /*default_thread_pool=*/nullptr);
  auto flr = pflr->GetFLR("/job:localhost/replica:0/task:0/cpu:0");

  {
    // Build TPU replicate function.
    // arg0 = _Arg[index = 0, T = DT_STRING]
    // arg1 = _Arg[index = 1, T = DT_INT32]
    // arg2 = _Arg[index = 2, T = DT_RESOURCE]
    // as_int = StringToNumber[out_type = DT_INT32](arg0)     (oc node)
    // add = Add(as_int, arg1)
    // as_string = AsString(add)                              (oc node)
    // read_var = ops::ReadVariableOp(arg2)
    // ret0 = _RetVal[index = 0, T = DT_STRING](as_string)
    // ret1 = _RetVal[index = 1, T = DT_INT32](add)
    // ret2 = _RetVal[index = 1, T = DT_FLOAT](read_var)
    Scope s = Scope::NewRootScope().ExitOnError();
    auto arg0 = ops::_Arg(s.WithOpName("arg0"), DT_STRING, 0);
    auto arg1 = ops::_Arg(s.WithOpName("arg1"), DT_INT32, 1);
    auto arg2 = ops::_Arg(s.WithOpName("arg2"), DT_RESOURCE, 2);
    auto as_int = ops::StringToNumber(s.WithOpName("as_int"), arg0,
                                      ops::StringToNumber::OutType(DT_INT32));
    auto add = ops::Add(s.WithOpName("add"), as_int, arg1);
    auto as_string = ops::AsString(s.WithOpName("as_string"), add);
    auto read_var =
        ops::ReadVariableOp(s.WithOpName("ReadVar"), arg2, DT_FLOAT);
    auto ret0 = ops::_Retval(s.WithOpName("ret0"), as_string, 0);
    auto ret1 = ops::_Retval(s.WithOpName("ret1"), add, 1);
    auto ret2 = ops::_Retval(s.WithOpName("ret2"), read_var, 2);
    Graph g(OpRegistry::Global());
    TF_ASSERT_OK(s.ToGraph(&g));
    auto node_name_index = g.BuildNodeNameIndex();
    node_name_index["as_int"]->AddAttr("oc", "0");
    node_name_index["as_string"]->AddAttr("oc", "0");
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(g, "cluster", &fdef));
    TF_ASSERT_OK(fld.AddFunctionDef(fdef));
  }

  string control_flow_scope = GetParam() ? "scope/" : "";
  string pivot_name = absl::StrCat(control_flow_scope, "tpu_replicate/pivot");
  Graph host_graph(OpRegistry::Global());
  NameAttrList function;
  function.set_name("cluster");
  {
    // Build host graph.
    // input00 = Placeholder[T = DT_STRING]
    // input01 = Placeholder[T = DT_INT32]
    // input10 = Placeholder[T = DT_STRING]
    // input11 = Placeholder[T = DT_INT32]
    // input2 = Placeholder[T = DT_RESOURCE]
    // tpu_replicate = _TPUReplicate(input00, input01, input10, input11)
    // output = IdentityN(tpu_replicate, tpu_replicate:1, tpu_replicate:2,
    //                    tpu_replicate:3, tpu_replicate:4, tpu_replicate:5)
    Scope s = Scope::NewRootScope().ExitOnError();
    auto pivot = ops::NoOp(s.WithOpName(pivot_name));
    pivot.operation.node()->AddAttr("_pivot_for_cluster", "cluster");
    auto input00 = ops::Placeholder(s.WithOpName("input00"), DT_STRING);
    auto input01 = ops::Placeholder(s.WithOpName("input01"), DT_INT32);
    auto input10 = ops::Placeholder(s.WithOpName("input10"), DT_STRING);
    auto input11 = ops::Placeholder(s.WithOpName("input11"), DT_INT32);
    auto input2 = ops::Placeholder(s.WithOpName("input2"), DT_RESOURCE);
    auto control_scope = s.WithControlDependencies({pivot});
    auto replicate = ops::_TPUReplicate(
        control_scope.WithOpName("tpu_replicate"),
        std::initializer_list<Input>{input00, input01, input10, input11,
                                     input2},
        std::initializer_list<Input>{}, std::initializer_list<Input>{},
        std::initializer_list<Input>{}, function,
        /*num_replicas=*/2,
        {DT_STRING, DT_INT32, DT_FLOAT, DT_STRING, DT_INT32, DT_FLOAT},
        ops::_TPUReplicate::NumCoresPerReplica(1).NumDistributedVariables(1));
    auto output = ops::IdentityN(
        s.WithOpName("output"),
        std::initializer_list<Input>{
            replicate.outputs[0], replicate.outputs[1], replicate.outputs[2],
            replicate.outputs[3], replicate.outputs[4], replicate.outputs[5]});
    TF_ASSERT_OK(s.ToGraph(&host_graph));
  }
  auto node_name_index = host_graph.BuildNodeNameIndex();
  Node* replicate_node = node_name_index["tpu_replicate"];

  std::unordered_map<string, XlaClusterInfo> clusters;
  clusters.emplace("cluster",
                   XlaClusterInfo{"cluster", function, replicate_node,
                                  std::map<string, int>{}});
  int lifted_arg_count = 0;
  TF_ASSERT_OK(ExtractOutsideCompilationPass::ProcessHeadTailOutsideCompilation(
      "oc", &lifted_arg_count, &clusters, &host_graph, flr, &fld));
  node_name_index = host_graph.BuildNodeNameIndex();
  replicate_node = node_name_index["tpu_replicate"];

  {
    // Check host graph.
    const Edge* e;
    Node* pivot = node_name_index[pivot_name];
    // Check that we have input00 -> as_int/R0 -> tpu_replicate.
    Node* as_int_R0 = node_name_index["as_int_head_oc/R0"];
    EXPECT_NE(as_int_R0, nullptr);
    TF_ASSERT_OK(as_int_R0->input_edge(0, &e));
    EXPECT_EQ(e->src(), node_name_index["input00"]);
    TF_ASSERT_OK(replicate_node->input_edge(1, &e));
    EXPECT_EQ(e->src(), as_int_R0);
    // Check that as_int/R0 has pivot as control input
    TF_EXPECT_OK(PivotControlExists(as_int_R0, pivot));
    // Check that we have input10 -> as_int/R1 -> tpu_replicate.
    Node* as_int_R1 = node_name_index["as_int_head_oc/R1"];
    EXPECT_NE(as_int_R1, nullptr);
    TF_ASSERT_OK(as_int_R1->input_edge(0, &e));
    EXPECT_EQ(e->src(), node_name_index["input10"]);
    TF_ASSERT_OK(replicate_node->input_edge(3, &e));
    EXPECT_EQ(e->src(), as_int_R1);
    // Check that as_int/R0 has pivot as control input
    TF_EXPECT_OK(PivotControlExists(as_int_R1, pivot));
    // Check that we have tpu_replicate -> as_string/R0 -> output.
    Node* as_string_R0 = node_name_index["as_string_tail_oc/R0"];
    EXPECT_NE(as_string_R0, nullptr);
    TF_ASSERT_OK(as_string_R0->input_edge(0, &e));
    EXPECT_EQ(e->src(), replicate_node);
    TF_ASSERT_OK(node_name_index["output"]->input_edge(0, &e));
    EXPECT_EQ(e->src(), as_string_R0);
    // Check that as_string/R0 has pivot as control input
    TF_EXPECT_OK(PivotControlExists(as_string_R0, pivot));
    // Check that we have tpu_replicate -> as_string/R1 -> output.
    Node* as_string_R1 = node_name_index["as_string_tail_oc/R1"];
    EXPECT_NE(as_string_R1, nullptr);
    TF_ASSERT_OK(as_string_R1->input_edge(0, &e));
    EXPECT_EQ(e->src(), replicate_node);
    TF_ASSERT_OK(node_name_index["output"]->input_edge(3, &e));
    EXPECT_EQ(e->src(), as_string_R1);
    // Check that as_string/R1 has pivot as control input
    TF_EXPECT_OK(PivotControlExists(as_string_R1, pivot));
  }

  {
    // Check TPU graph.
    const FunctionDef* fdef = fld.Find("cluster");
    EXPECT_NE(fdef, nullptr);
    // Check its signature, should have 2 DT_INT32 inputs, 1 DT_RESOURCE input,
    // 2 DT_INT32 outputs and 1 DT_FLOAT output.
    EXPECT_EQ(fdef->signature().input_arg_size(), 3);
    EXPECT_EQ(fdef->signature().input_arg(0).type(), DT_INT32);
    EXPECT_EQ(fdef->signature().input_arg(1).type(), DT_INT32);
    EXPECT_EQ(fdef->signature().input_arg(2).type(), DT_RESOURCE);
    EXPECT_EQ(fdef->signature().output_arg_size(), 3);
    EXPECT_EQ(fdef->signature().output_arg(0).type(), DT_INT32);
    EXPECT_EQ(fdef->signature().output_arg(1).type(), DT_FLOAT);
    EXPECT_EQ(fdef->signature().output_arg(2).type(), DT_INT32);
    // Check that it has no StringToNumber/AsString op any more.
    for (const NodeDef& node_def : fdef->node_def()) {
      EXPECT_NE(node_def.op(), "StringToNumber");
      EXPECT_NE(node_def.op(), "AsString");
    }
  }
}

INSTANTIATE_TEST_SUITE_P(All, ExtractOutsideCompilationByScope,
                         ::testing::ValuesIn({true, false}));

TEST(ExtractOutsideCompilation, RemoveArgRetvalPair) {
  FunctionLibraryDefinition fld(OpRegistry::Global(), FunctionDefLibrary());

  // Create FunctionLibraryRuntime.
  SessionOptions session_options;
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(DeviceFactory::AddDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));
  OptimizerOptions opts;
  auto device_mgr = absl::make_unique<StaticDeviceMgr>(std::move(devices));
  auto pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr.get(), Env::Default(), /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, &fld, opts,
      /*default_thread_pool=*/nullptr);
  auto flr = pflr->GetFLR("/job:localhost/replica:0/task:0/cpu:0");

  {
    // Build TPU replicate function.
    // arg0 = _Arg[index = 0, T = DT_STRING]
    // arg1 = _Arg[index = 1, T = DT_FLOAT]
    // arg2 = _Arg[index = 2, T = DT_INT32]
    // arg3 = _Arg[index = 3, T = DT_RESOURCE]
    // arg4 = _Arg[index = 4, T = DT_RESOURCE]
    // add = Add(arg2, arg2)
    // read = ReadVariableOp(arg4)
    // ret0 = _RetVal[index = 0, T = DT_STRING](arg0)
    // ret1 = _RetVal[index = 1, T = DT_INT32](add)
    // ret2 = _RetVal[index = 2, T = DT_FLOAT](read)
    // ret3 = _RetVal[index = 3, T = DT_RESOURCE](arg3)
    Scope s = Scope::NewRootScope().ExitOnError();
    auto arg0 = ops::_Arg(s.WithOpName("arg0"), DT_STRING, 0);
    auto arg1 = ops::_Arg(s.WithOpName("arg1"), DT_FLOAT, 1);
    auto arg2 = ops::_Arg(s.WithOpName("arg2"), DT_INT32, 2);
    auto arg3 = ops::_Arg(s.WithOpName("arg3"), DT_RESOURCE, 3);
    auto arg4 = ops::_Arg(s.WithOpName("arg4"), DT_RESOURCE, 4);
    auto add = ops::Add(s.WithOpName("add"), arg2, arg2);
    auto ret0 = ops::_Retval(s.WithOpName("ret0"), arg0, 0);
    auto ret1 = ops::_Retval(s.WithOpName("ret1"), add, 1);
    auto read = ops::ReadVariableOp(s.WithOpName("read"), arg4, DT_FLOAT);
    auto ret2 = ops::_Retval(s.WithOpName("ret2"), read, 2);
    auto ret3 = ops::_Retval(s.WithOpName("ret3"), arg3, 3);
    Graph g(OpRegistry::Global());
    TF_ASSERT_OK(s.ToGraph(&g));
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(g, "cluster", &fdef));
    TF_ASSERT_OK(fld.AddFunctionDef(fdef));
  }

  Graph host_graph(OpRegistry::Global());
  NameAttrList function;
  function.set_name("cluster");
  {
    // Build host graph.
    // input00 = Placeholder[T = DT_STRING]
    // input01 = Placeholder[T = DT_FLOAT]
    // input02 = Placeholder[T = DT_INT32]
    // input10 = Placeholder[T = DT_STRING]
    // input11 = Placeholder[T = DT_FLOAT]
    // input12 = Placeholder[T = DT_INT32]
    // input3 = Placeholder[T = DT_RESOURCE], distributed variable
    // input4 = Placeholder[T = DT_RESOURCE], distributed variable
    // tpu_replicate = _TPUReplicate(input00, input01, input02, input10,
    //                               input11, input12, input3, input4)
    // output = IdentityN(tpu_replicate, tpu_replicate:1, tpu_replicate:2,
    //                    tpu_replicate:3, tpu_replicate:4, tpu_replicate:5,
    //                    tpu_replicate:6, tpu_replicate:7)
    Scope s = Scope::NewRootScope().ExitOnError();
    auto input00 = ops::Placeholder(s.WithOpName("input00"), DT_STRING);
    auto input01 = ops::Placeholder(s.WithOpName("input01"), DT_FLOAT);
    auto input02 = ops::Placeholder(s.WithOpName("input02"), DT_INT32);
    auto input10 = ops::Placeholder(s.WithOpName("input10"), DT_STRING);
    auto input11 = ops::Placeholder(s.WithOpName("input11"), DT_FLOAT);
    auto input12 = ops::Placeholder(s.WithOpName("input12"), DT_INT32);
    auto input3 = ops::Placeholder(s.WithOpName("input3"), DT_RESOURCE);
    auto input4 = ops::Placeholder(s.WithOpName("input3"), DT_RESOURCE);
    auto replicate = ops::_TPUReplicate(
        s.WithOpName("tpu_replicate"),
        std::initializer_list<Input>{input00, input01, input02, input10,
                                     input11, input12, input3, input4},
        std::initializer_list<Input>{}, std::initializer_list<Input>{},
        std::initializer_list<Input>{}, function,
        /*num_replicas=*/2,
        {DT_STRING, DT_INT32, DT_FLOAT, DT_RESOURCE, DT_STRING, DT_INT32,
         DT_FLOAT, DT_RESOURCE},
        ops::_TPUReplicate::NumCoresPerReplica(1).NumDistributedVariables(2));
    auto output = ops::IdentityN(
        s.WithOpName("output"),
        std::initializer_list<Input>{
            replicate.outputs[0], replicate.outputs[1], replicate.outputs[2],
            replicate.outputs[3], replicate.outputs[4], replicate.outputs[5],
            replicate.outputs[6], replicate.outputs[7]});
    TF_ASSERT_OK(s.ToGraph(&host_graph));
  }
  auto node_name_index = host_graph.BuildNodeNameIndex();
  Node* replicate_node = node_name_index["tpu_replicate"];

  std::unordered_map<string, XlaClusterInfo> clusters;
  clusters.emplace("cluster",
                   XlaClusterInfo{"cluster", function, replicate_node,
                                  std::map<string, int>{}});
  int lifted_arg_count = 0;
  TF_ASSERT_OK(ExtractOutsideCompilationPass::ProcessHeadTailOutsideCompilation(
      "oc", &lifted_arg_count, &clusters, &host_graph, flr, &fld));
  node_name_index = host_graph.BuildNodeNameIndex();
  replicate_node = node_name_index["tpu_replicate"];
  Node* output = node_name_index["output"];

  EXPECT_EQ(replicate_node->num_inputs(), 3);
  const DataTypeVector expected_input_types = {DT_INT32, DT_INT32, DT_RESOURCE};
  EXPECT_EQ(replicate_node->input_types(), expected_input_types);
  EXPECT_EQ(replicate_node->num_outputs(), 4);
  const DataTypeVector expected_output_types = {DT_INT32, DT_FLOAT, DT_INT32,
                                                DT_FLOAT};
  EXPECT_EQ(replicate_node->output_types(), expected_output_types);

  {
    // Check host graph.
    Node* input_node;
    // Check that we have input00 -> output:1.
    TF_ASSERT_OK(output->input_node(0, &input_node));
    EXPECT_EQ(input_node->name(), "input00");
    // Check that we have input10 -> output:4.
    TF_ASSERT_OK(output->input_node(4, &input_node));
    EXPECT_EQ(input_node->name(), "input10");
    // Check that we have input3 -> output:3, output:7.
    TF_ASSERT_OK(output->input_node(3, &input_node));
    EXPECT_EQ(input_node->name(), "input3");
    TF_ASSERT_OK(output->input_node(7, &input_node));
    EXPECT_EQ(input_node->name(), "input3");
  }

  {
    // Check TPU graph.
    const FunctionDef* fdef = fld.Find("cluster");
    EXPECT_NE(fdef, nullptr);
    // Check its signature, should have 1 DT_INT32 input, 1 DT_RESOURCE input,
    // 1 DT_INT32 output and 1 DT_FLOAT output
    EXPECT_EQ(fdef->signature().input_arg_size(), 2);
    EXPECT_EQ(fdef->signature().input_arg(0).type(), DT_INT32);
    EXPECT_EQ(fdef->signature().input_arg(1).type(), DT_RESOURCE);
    EXPECT_EQ(fdef->signature().output_arg_size(), 2);
    EXPECT_EQ(fdef->signature().output_arg(0).type(), DT_INT32);
    EXPECT_EQ(fdef->signature().output_arg(1).type(), DT_FLOAT);
  }
}

}  // namespace tensorflow

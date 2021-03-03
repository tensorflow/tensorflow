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

#include "tensorflow/compiler/tf2xla/tf2xla_util.h"

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/data_flow_ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/list_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2xla/sharding_util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

void ExpectErrorContains(const Status& status, absl::string_view str) {
  EXPECT_NE(Status::OK(), status);
  EXPECT_TRUE(absl::StrContains(status.error_message(), str))
      << "expected error: " << status.error_message() << " to contain: " << str;
}

TEST(ValidateConfig, Good) {
  tf2xla::Config config;
  tf2xla::Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  feed->mutable_id()->set_output_index(123);
  feed->set_name("foo_debug");
  feed = config.add_feed();
  feed->mutable_id()->set_node_name("bar");
  feed->mutable_id()->set_output_index(0);
  tf2xla::Fetch* fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("baz");
  fetch->mutable_id()->set_output_index(456);
  fetch->set_name("baz_debug");
  fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("banana");
  fetch->mutable_id()->set_output_index(0);
  TF_EXPECT_OK(ValidateConfig(config));
}

TEST(ValidateConfig, BadEmpty) {
  tf2xla::Config config;
  ExpectErrorContains(ValidateConfig(config), "fetches must be specified");
}

TEST(ValidateConfig, BadNoFetch) {
  tf2xla::Config config;
  tf2xla::Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  ExpectErrorContains(ValidateConfig(config), "fetches must be specified");
}

TEST(ValidateConfig, BadFeedNodeName) {
  tf2xla::Config config;
  config.add_feed();
  ExpectErrorContains(ValidateConfig(config), "node_name must be non-empty");
}

TEST(ValidateConfig, BadFeedOutputIndex) {
  tf2xla::Config config;
  tf2xla::Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  feed->mutable_id()->set_output_index(-1);
  ExpectErrorContains(ValidateConfig(config), "output_index must be positive");
}

TEST(ValidateConfig, BadFetchNodeName) {
  tf2xla::Config config;
  tf2xla::Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  config.add_fetch();
  ExpectErrorContains(ValidateConfig(config), "node_name must be non-empty");
}

TEST(ValidateConfig, BadFetchOutputIndex) {
  tf2xla::Config config;
  tf2xla::Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  tf2xla::Fetch* fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("bar");
  fetch->mutable_id()->set_output_index(-1);
  ExpectErrorContains(ValidateConfig(config), "output_index must be positive");
}

TEST(ValidateConfig, DuplicateFeedName) {
  tf2xla::Config config;
  tf2xla::Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  feed->set_name("dup");
  feed = config.add_feed();
  feed->mutable_id()->set_node_name("bar");
  feed->set_name("dup");
  ExpectErrorContains(ValidateConfig(config), "duplicate feed name");
}

TEST(ValidateConfig, DuplicateFetchName) {
  tf2xla::Config config;
  tf2xla::Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  tf2xla::Fetch* fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("bar");
  fetch->set_name("dup");
  fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("baz");
  fetch->set_name("dup");
  ExpectErrorContains(ValidateConfig(config), "duplicate fetch name");
}

TEST(ValidateConfig, ConflictingFeedName) {
  tf2xla::Config config;
  tf2xla::Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  feed->set_name("conflict");
  feed = config.add_feed();
  feed->mutable_id()->set_node_name("bar");
  feed->set_name("conflict_data");
  ExpectErrorContains(ValidateConfig(config), "conflicting feed name");
}

TEST(ValidateConfig, ConflictingFetchName) {
  tf2xla::Config config;
  tf2xla::Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  tf2xla::Fetch* fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("bar");
  fetch->set_name("conflict");
  fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("baz");
  fetch->set_name("conflict_data");
  ExpectErrorContains(ValidateConfig(config), "conflicting fetch name");
}

static tf2xla::Config FetchesConfig(std::vector<string> fetches) {
  tf2xla::Config config;
  for (const auto& fetch_node_name : fetches) {
    auto* fetch = config.add_fetch();
    fetch->set_name(absl::StrCat("fetch_", fetch_node_name));
    fetch->mutable_id()->set_node_name(fetch_node_name);
  }
  return config;
}

TEST(PruneGraphDefInto, Basic) {
  GraphDef def;
  auto* n = def.add_node();
  n->set_name("a");
  n->add_input("b:0");
  n->add_input("^c");

  GraphDef copy;
  ExpectErrorContains(PruneGraphDefInto(FetchesConfig({"missing"}), def, &copy),
                      "node missing needed");
  ExpectErrorContains(PruneGraphDefInto(FetchesConfig({"a"}), def, &copy),
                      "node b needed");

  n = def.add_node();
  n->set_name("b");
  ExpectErrorContains(PruneGraphDefInto(FetchesConfig({"a"}), def, &copy),
                      "node c needed");
  n->add_input("d:1");

  n = def.add_node();
  n->set_name("c");
  n->add_input("d:1");

  n = def.add_node();
  n->set_name("d");

  // Graph is full, no pruning done.
  // Graph right now has diamond from d:
  //   d --> b --> a
  //   d --> c --> a
  TF_EXPECT_OK(PruneGraphDefInto(FetchesConfig({"a"}), def, &copy));
  EXPECT_EQ(def.DebugString(), copy.DebugString());
  GraphDef pruned_a = copy;

  // Add some unrelated fields that use b and c, but are not needed for a.
  n = def.add_node();
  n->set_name("e");
  n->add_input("^d");
  n->add_input("b:2");
  copy.Clear();
  TF_EXPECT_OK(PruneGraphDefInto(FetchesConfig({"a"}), def, &copy));
  EXPECT_EQ(pruned_a.DebugString(), copy.DebugString());

  // Fetch "a" and "e" to get the original graph.
  copy.Clear();
  TF_EXPECT_OK(PruneGraphDefInto(FetchesConfig({"a", "e"}), def, &copy));
  EXPECT_EQ(def.DebugString(), copy.DebugString());
}

TEST(SetNodeShardingFromNeighbors, Basic) {
  // Builds a graph that adds two Tensors.
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Arg(scope.WithOpName("B"), DT_INT32, 1);
  auto c = ops::Add(scope.WithOpName("C"), a, b);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  Node* a_node = nullptr;
  Node* b_node = nullptr;
  Node* c_node = nullptr;
  for (Node* n : graph->nodes()) {
    if (n->name() == "A") a_node = n;
    if (n->name() == "B") b_node = n;
    if (n->name() == "C") c_node = n;
  }

  const int num_cores_per_replica = 4;

  a_node->set_assigned_device_name("foo");
  EXPECT_FALSE(SetNodeShardingFromNeighbors(c_node, /*out_edges=*/false).ok());

  // Test where one input to c_node has a device.
  a_node->set_assigned_device_name("/device:TPU_REPLICATED_CORE:2");
  TF_ASSERT_OK(SetNodeShardingFromNeighbors(c_node, /*out_edges=*/false));
  auto parse_status = ParseShardingFromDevice(*c_node, num_cores_per_replica,
                                              /*add_metadata=*/false);
  TF_ASSERT_OK(parse_status.status());
  ASSERT_TRUE(parse_status.ValueOrDie().has_value());
  EXPECT_EQ(2, parse_status.ValueOrDie().value().tile_assignment_devices(0));

  // Test where two inputs to c_node have a device.
  b_node->set_assigned_device_name("/device:TPU_REPLICATED_CORE:1");
  TF_ASSERT_OK(SetNodeShardingFromNeighbors(c_node, /*out_edges=*/false));
  parse_status = ParseShardingFromDevice(*c_node, num_cores_per_replica,
                                         /*add_metadata=*/false);
  TF_ASSERT_OK(parse_status.status());
  ASSERT_TRUE(parse_status.ValueOrDie().has_value());
  EXPECT_EQ(1, parse_status.ValueOrDie().value().tile_assignment_devices(0));

  // Test setting based on out edges.
  TF_ASSERT_OK(SetNodeShardingFromNeighbors(a_node, /*out_edges=*/true));
  parse_status = ParseShardingFromDevice(*a_node, num_cores_per_replica,
                                         /*add_metadata=*/false);
  TF_ASSERT_OK(parse_status.status());
  ASSERT_TRUE(parse_status.ValueOrDie().has_value());
  EXPECT_EQ(1, parse_status.ValueOrDie().value().tile_assignment_devices(0));
}

REGISTER_OP("One")
    .Output("y: T")
    .Attr("T: {float, double, int32, int64}")
    .Doc(R"doc(
Returns a tensor with a single element (1) of type T.

y: A scalar in type T.

)doc");

// Tests that CachedFunctionHandles class works.
TEST(CachedFunctionHandles, Basic) {
  FunctionDef func = FunctionDefHelper::Define(
      // Name
      "TestFunc",
      // Args
      {},
      // Return values
      {"y:T"},
      // Attr def
      {"T:{float, double, int32, int64}"},
      // Nodes
      {
          {{"y"}, "One", {}, {{"T", "$T"}}},
      });
  FunctionDefLibrary proto;
  *proto.add_function() = func;
  FunctionLibraryDefinition fld(OpRegistry::Global(), proto);
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(
          /*device_mgr=*/nullptr, Env::Default(), /*config=*/nullptr,
          TF_GRAPH_DEF_VERSION, &fld, OptimizerOptions()));
  FunctionLibraryRuntime* flr =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

  CachedFunctionHandles cached_function_handles(flr);

  // Tests that GetOrInstantiate() works.
  FunctionLibraryRuntime::Handle first_handle;
  AttrValue attr;
  attr.set_type(DT_FLOAT);
  AttrValueMap attrs;
  attrs["T"] = attr;
  TF_ASSERT_OK(cached_function_handles.GetOrInstantiate(
      "TestFunc", AttrSlice(&attrs), &first_handle));

  // Tests that we can get FunctionBody.
  const FunctionBody* body = flr->GetFunctionBody(first_handle);
  EXPECT_NE(body, nullptr);

  // Tests that GetOrInstantiate() returns cached handle when called with same
  // function name and attributes.
  FunctionLibraryRuntime::Handle second_handle;
  TF_ASSERT_OK(cached_function_handles.GetOrInstantiate(
      "TestFunc", AttrSlice(&attrs), &second_handle));
  EXPECT_EQ(first_handle, second_handle);

  // Tests that GetOrInstantiate() returns new handle when called with same
  // function name but different attributes.
  attr.set_type(DT_INT32);
  attrs["T"] = attr;
  FunctionLibraryRuntime::Handle third_handle;
  TF_ASSERT_OK(cached_function_handles.GetOrInstantiate(
      "TestFunc", AttrSlice(&attrs), &third_handle));
  EXPECT_NE(first_handle, third_handle);

  // Tests that ReleaseAllHandles() works.
  TF_EXPECT_OK(cached_function_handles.ReleaseAllHandles());
}

TEST(PropagateConstIntoFunctionalNodes, WhileLoopWithResourceInput) {
  FunctionLibraryDefinition fld(OpRegistry::Global(), {});
  {
    // Cond graph & body graph.
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto pred = ops::_Arg(scope.WithOpName("pred"), DT_BOOL, 0);
    auto input = ops::_Arg(scope.WithOpName("input"), DT_RESOURCE, 1);
    auto ret = ops::_Retval(scope.WithOpName("ret"), pred, 0);
    Graph graph(OpRegistry::Global());
    TF_ASSERT_OK(scope.ToGraph(&graph));
    FunctionDef cond_fdef;
    TF_ASSERT_OK(GraphToFunctionDef(graph, "cond", &cond_fdef));
    TF_ASSERT_OK(fld.AddFunctionDef(cond_fdef));
    FunctionDef body_fdef;
    TF_ASSERT_OK(GraphToFunctionDef(graph, "body", &body_fdef));
    TF_ASSERT_OK(fld.AddFunctionDef(body_fdef));
  }
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto pred = ops::Const(scope.WithOpName("pred"), false, TensorShape({}));
  auto input = ops::Const(scope.WithOpName("input"), 0, TensorShape({}));
  NameAttrList cond_fn, body_fn;
  cond_fn.set_name("cond");
  body_fn.set_name("body");
  auto while_op =
      ops::While(scope.WithOpName("while"),
                 std::initializer_list<Input>{pred, input}, cond_fn, body_fn);
  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));

  TF_EXPECT_OK(PropagateConstIntoFunctionalNodes(&graph, &fld, &fld));
}

TEST(PropagateConstIntoFunctionalNodes, CopiedConstNodeHasUniqueName) {
  FunctionLibraryDefinition fld(OpRegistry::Global(), {});
  {
    // Cond graph & body graph.
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto pred = ops::_Arg(scope.WithOpName("arg0"), DT_BOOL, 0);
    auto input = ops::_Arg(scope.WithOpName("arg1"), DT_BOOL, 1);
    auto duplicate_name = ops::NoOp(scope.WithOpName("duplicate_name"));
    auto ret = ops::_Retval(scope.WithOpName("ret"), pred, 0);
    Graph graph(OpRegistry::Global());
    TF_ASSERT_OK(scope.ToGraph(&graph));
    FunctionDef cond_fdef;
    TF_ASSERT_OK(GraphToFunctionDef(graph, "cond", &cond_fdef));
    TF_ASSERT_OK(fld.AddFunctionDef(cond_fdef));
    FunctionDef body_fdef;
    TF_ASSERT_OK(GraphToFunctionDef(graph, "body", &body_fdef));
    TF_ASSERT_OK(fld.AddFunctionDef(body_fdef));
  }
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto pred =
      ops::Const(scope.WithOpName("duplicate_name"), false, TensorShape({}));
  auto input = ops::Const(scope.WithOpName("input"), false, TensorShape({}));
  NameAttrList cond_fn, body_fn;
  cond_fn.set_name("cond");
  body_fn.set_name("body");
  auto while_op =
      ops::While(scope.WithOpName("while"),
                 std::initializer_list<Input>{pred, input}, cond_fn, body_fn);
  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));

  TF_EXPECT_OK(PropagateConstIntoFunctionalNodes(&graph, &fld, &fld));

  // Check that in rewritten body function, the NoOp node still has name
  // "duplicate_name", and the copied Const node has name "duplicate_name/_0".
  auto node_name_index = graph.BuildNodeNameIndex();
  Node* while_node = node_name_index["while"];
  ASSERT_NE(while_node, nullptr);
  TF_ASSERT_OK(GetNodeAttr(while_node->def(), "body", &body_fn));
  const FunctionDef* rewritten_body_fn = fld.Find(body_fn.name());
  ASSERT_NE(rewritten_body_fn, nullptr);
  std::unordered_map<string, NodeDef> nodes;
  for (const NodeDef& node_def : rewritten_body_fn->node_def()) {
    nodes[node_def.name()] = node_def;
  }
  auto noop_def = nodes.find("duplicate_name");
  ASSERT_NE(noop_def, nodes.end());
  EXPECT_EQ(noop_def->second.op(), "NoOp");
  auto const_def = nodes.find("duplicate_name/_0");
  ASSERT_NE(const_def, nodes.end());
  EXPECT_EQ(const_def->second.op(), "Const");
}

TEST(PropagateConstIntoFunctionalNodes, RewriteTensorListWithConstMember) {
  FunctionLibraryDefinition fld(OpRegistry::Global(), {});
  {
    // Cond graph
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto input = ops::_Arg(scope.WithOpName("arg"), DT_VARIANT, 0);
    auto result =
        ops::Const(scope.WithOpName("result"), false, TensorShape({}));
    auto ret = ops::_Retval(scope.WithOpName("ret"), result, 0);
    Graph graph(OpRegistry::Global());
    TF_ASSERT_OK(scope.ToGraph(&graph));
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(graph, "cond", &fdef));
    TF_ASSERT_OK(fld.AddFunctionDef(fdef));
  }
  {
    // Forward body graph
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto input = ops::_Arg(scope.WithOpName("arg"), DT_VARIANT, 0);
    auto element = ops::Const(scope.WithOpName("element"), 0, TensorShape({}));
    auto push =
        ops::TensorListPushBack(scope.WithOpName("push"), input, element);
    auto ret = ops::_Retval(scope.WithOpName("ret"), push.output_handle, 0);
    Graph graph(OpRegistry::Global());
    TF_ASSERT_OK(scope.ToGraph(&graph));
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(graph, "fwd_body", &fdef));
    TF_ASSERT_OK(fld.AddFunctionDef(fdef));
  }
  {
    // Backward body graph
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto input = ops::_Arg(scope.WithOpName("arg"), DT_VARIANT, 0);
    auto shape = ops::Const(scope.WithOpName("element"), -1, TensorShape({}));
    auto pop =
        ops::TensorListPopBack(scope.WithOpName("pop"), input, shape, DT_INT32);
    auto identity = ops::Identity(scope.WithOpName("identity"), pop.tensor);
    auto ret = ops::_Retval(scope.WithOpName("ret"), pop.output_handle, 0);
    Graph graph(OpRegistry::Global());
    TF_ASSERT_OK(scope.ToGraph(&graph));
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(graph, "bwd_body", &fdef));
    TF_ASSERT_OK(fld.AddFunctionDef(fdef));
  }
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto shape = ops::Const(scope.WithOpName("element"), -1, TensorShape({}));
  auto max_num_elements =
      ops::Const(scope.WithOpName("max_num_elements"), 10, TensorShape({}));
  auto tl = ops::EmptyTensorList(scope.WithOpName("tl"), shape,
                                 max_num_elements, DT_INT32);
  NameAttrList cond_fn, fwd_body_fn, bwd_body_fn;
  cond_fn.set_name("cond");
  fwd_body_fn.set_name("fwd_body");
  bwd_body_fn.set_name("bwd_body");
  auto fwd_while_op =
      ops::While(scope.WithOpName("fwd_while"),
                 std::initializer_list<Input>{tl}, cond_fn, fwd_body_fn);
  auto bwd_while_op =
      ops::While(scope.WithOpName("bwd_while"),
                 std::initializer_list<Input>{fwd_while_op.output[0]}, cond_fn,
                 bwd_body_fn);
  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));

  TF_EXPECT_OK(RewriteTensorListWithConstElement(&graph, &fld));

  // Check that in rewritten backward While body function, the Identity node now
  // has Const node as input.
  const FunctionDef* bwd_body = fld.Find("bwd_body_tl_rewrite_0");
  ASSERT_NE(bwd_body, nullptr);
  std::unique_ptr<FunctionBody> bwd_fbody;
  TF_CHECK_OK(
      FunctionDefToBodyHelper(*bwd_body, AttrSlice(), &fld, &bwd_fbody));
  auto node_name_index = bwd_fbody->graph->BuildNodeNameIndex();
  const Node* identity = node_name_index.at("identity");
  ASSERT_NE(identity, nullptr);
  const Node* input;
  TF_ASSERT_OK(identity->input_node(0, &input));
  EXPECT_EQ(input->type_string(), "Const");
}

}  // namespace
}  // namespace tensorflow

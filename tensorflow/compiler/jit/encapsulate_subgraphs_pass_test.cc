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

#include <memory>
#include <utility>

#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/graph_to_functiondef.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace {

const char* const kXlaHostTransferSequencerAttr =
    "_xla_host_transfer_sequencer";

Status AddGraphDefToFunctionLibrary(const GraphDefBuilder& graphdef_builder,
                                    const string& name_suffix,
                                    FunctionDefLibrary* library) {
  GraphDef graphdef;
  TF_RETURN_IF_ERROR(graphdef_builder.ToGraphDef(&graphdef));
  std::unique_ptr<Graph> graph =
      std::unique_ptr<Graph>(new Graph(OpRegistry::Global()));
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, graphdef, graph.get()));
  FunctionDef* fdef = library->add_function();
  TF_RETURN_IF_ERROR(GraphToFunctionDef(
      *graph,
      strings::StrCat("_outside_compilation_shape_inference_", name_suffix),
      fdef));
  return Status::OK();
}

template <class Tkey, class Tvalue>
bool EqualProtoMap(const ::tensorflow::protobuf::Map<Tkey, Tvalue>& a,
                   const ::tensorflow::protobuf::Map<Tkey, Tvalue>& b,
                   const std::function<string(const Tkey&)>& key_to_string,
                   const std::function<string(const Tvalue&)>& value_to_string,
                   const std::function<bool(const Tkey&, const Tvalue&,
                                            const Tvalue&)>& compare,
                   const string& map_name, string* diff) {
  for (const auto& elt_a : a) {
    const auto iter = b.find(elt_a.first);
    if (iter == b.end()) {
      if (diff) {
        *diff = strings::StrCat(
            map_name, " expected: contains element with key '",
            key_to_string(elt_a.first), "' got: map has no such element");
      }
      return false;
    }
    if (!compare(elt_a.first, elt_a.second, iter->second)) {
      if (diff) {
        *diff = strings::StrCat(map_name, " expected: element with key '",
                                key_to_string(elt_a.first), " has value '",
                                value_to_string(elt_a.second), "' got: '",
                                value_to_string(iter->second), "'");
      }
      return false;
    }
  }
  for (const auto& elt_b : b) {
    const auto iter = a.find(elt_b.first);
    if (iter == a.end()) {
      if (diff) {
        *diff = strings::StrCat(map_name, " got: contains element with key '",
                                key_to_string(elt_b.first),
                                "' expected: map has no such element");
      }
      return false;
    }
  }
  return true;
}

bool EqualFunctionNodeDef(const NodeDef& a, const NodeDef& b,
                          const string& diff_preamble, string* diff) {
  if (a.op() != b.op()) {
    if (diff) {
      *diff = strings::StrCat(diff_preamble, " mismatch for node ", a.name(),
                              ", expected op '", a.op(), "' got '", b.op());
    }
    return false;
  }
  if (a.device() != b.device()) {
    if (diff) {
      *diff = strings::StrCat(diff_preamble, " mismatch for node ", a.name(),
                              ", expected device '", a.device(), "' got '",
                              b.device());
    }
    return false;
  }
  if (a.input_size() != b.input_size()) {
    if (diff) {
      *diff = strings::StrCat(diff_preamble, " mismatch for node ", a.name(),
                              ", expected ", a.input_size(), " inputs got ",
                              b.input_size(), " expected:\n", a.DebugString(),
                              "\ngot:\n", b.DebugString());
    }
    return false;
  }
  for (int i = 0; i < a.input_size(); ++i) {
    if (a.input(i) != b.input(i)) {
      if (diff) {
        *diff = strings::StrCat(diff_preamble, " mismatch for node ", a.name(),
                                " input ", i, ", expected ", a.input(i),
                                " got ", b.input(i), " expected:\n",
                                a.DebugString(), "\ngot:\n", b.DebugString());
      }
      return false;
    }
  }
  return EqualProtoMap<string, AttrValue>(
      a.attr(), b.attr(), [](const string& s) { return s; },
      [](const AttrValue& v) { return v.DebugString(); },
      [](const string& key, const AttrValue& av, const AttrValue& bv) {
        return av.DebugString() == bv.DebugString();
      },
      strings::StrCat(diff_preamble, " attr mismatch for node ", a.name()),
      diff);
}

bool EqualFunctionDef(const FunctionDef& a, const FunctionDef& b,
                      string* diff) {
  if (a.signature().DebugString() != b.signature().DebugString()) {
    if (diff) {
      *diff = strings::StrCat("Signature mismatch for function ",
                              a.signature().name(), ", expected:\n",
                              a.signature().DebugString(), "\ngot:\n",
                              b.signature().DebugString());
    }
    return false;
  }
  if (!EqualProtoMap<string, AttrValue>(
          a.attr(), b.attr(), [](const string& s) { return s; },
          [](const AttrValue& v) { return v.DebugString(); },
          [](const string& key, const AttrValue& av, const AttrValue& bv) {
            return av.DebugString() == bv.DebugString();
          },
          strings::StrCat("attr mismatch for function ", a.signature().name()),
          diff)) {
    return false;
  }
  if (!EqualProtoMap<string, string>(
          a.ret(), b.ret(), [](const string& s) { return s; },
          [](const string& s) { return s; },
          [](const string& key, const string& av, const string& bv) {
            return av == bv;
          },
          strings::StrCat("ret mismatch for function ", a.signature().name()),
          diff)) {
    return false;
  }
  for (int i = 0; i < a.node_def_size(); ++i) {
    bool found = false;
    for (int j = 0; j < b.node_def_size(); ++j) {
      if (a.node_def(i).name() == b.node_def(j).name()) {
        if (!EqualFunctionNodeDef(
                a.node_def(i), b.node_def(j),
                strings::StrCat("Function ", a.signature().name()), diff)) {
          return false;
        }
        found = true;
        break;
      }
    }
    if (!found) {
      if (diff) {
        *diff = strings::StrCat("Function ", a.signature().name(),
                                ", expected: has node '", a.node_def(i).name(),
                                "' got: no node of that name");
      }
      return false;
    }
  }
  for (int i = 0; i < b.node_def_size(); ++i) {
    bool found = false;
    for (int j = 0; j < a.node_def_size(); ++j) {
      if (b.node_def(i).name() == a.node_def(j).name()) {
        found = true;
        break;
      }
    }
    if (!found) {
      if (diff) {
        *diff = strings::StrCat("Function ", a.signature().name(),
                                ", got: has node '", b.node_def(i).name(),
                                "' expected: no node of that name");
      }
      return false;
    }
  }
  return true;
}

bool EqualFunctionDefLibrary(const FunctionDefLibrary& expected,
                             const FunctionDefLibrary& actual, string* diff) {
  std::unordered_map<string, const FunctionDef*> actual_index;
  for (const FunctionDef& function : actual.function()) {
    actual_index[function.signature().name()] = &function;
  }

  for (const FunctionDef& expected_function : expected.function()) {
    auto it = actual_index.find(expected_function.signature().name());
    if (it == actual_index.end()) {
      if (diff) {
        *diff = strings::StrCat("Did not find expected function '",
                                expected_function.signature().name(), "'");
      }
      return false;
    }
    if (!EqualFunctionDef(expected_function, *it->second, diff)) return false;
    actual_index.erase(it);
  }

  if (!actual_index.empty()) {
    if (diff != nullptr) {
      *diff = strings::StrCat("Found unexpected function '",
                              actual_index.begin()->second->signature().name(),
                              "'");
    }
    return false;
  }

  return true;
}

#define TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(expected, actual)         \
  do {                                                            \
    string diff;                                                  \
    EXPECT_TRUE(EqualFunctionDefLibrary(expected, actual, &diff)) \
        << diff << "\nActual: " << actual.DebugString();          \
  } while (false)

// These dummy Op registrations are here because the real Op registrations live
// in contrib and there can't be a dependence from this test to contrib.
REGISTER_OP("XlaHostCompute")
    .Input("inputs: Tinputs")
    .Output("outputs: Toutputs")
    .Attr("Tinputs: list(type) >= 0")
    .Attr("Toutputs: list(type) >= 0")
    .Attr("key: string")
    .Attr("shape_inference_graph: string = ''")
    .Attr("shapes: list(shape) >= 0")
    .SetShapeFn(::tensorflow::shape_inference::UnknownShape);

REGISTER_OP("_XlaSendFromHost")
    .Input("inputs: Tinputs")
    .Input("dynamic_key: string")
    .Attr("Tinputs: list(type) >= 0")
    .Attr("key: string")
    .Attr("device_ordinal: int")
    .SetShapeFn(::tensorflow::shape_inference::UnknownShape);

REGISTER_OP("_XlaRecvAtHost")
    .Input("dynamic_key: string")
    .Output("outputs: Toutputs")
    .Attr("Toutputs: list(type) >= 0")
    .Attr("key: string")
    .Attr("device_ordinal: int")
    .SetShapeFn(::tensorflow::shape_inference::UnknownShape);

REGISTER_OP("InputTest")
    .Output("o: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    });

REGISTER_OP("InputTestShaped")
    .Output("o: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
    });

REGISTER_OP("UnaryTest")
    .Input("a: float")
    .Output("o: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle o;
      TF_RETURN_IF_ERROR(c->Merge(c->UnknownShape(), c->input(0), &o));
      c->set_output(0, o);
      return Status::OK();
    });
REGISTER_OP("BinaryTest")
    .Input("a: float")
    .Input("b: float")
    .Output("o: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle o;
      TF_RETURN_IF_ERROR(c->Merge(c->UnknownShape(), c->input(0), &o));
      c->set_output(0, o);
      return Status::OK();
    });
REGISTER_OP("BinaryTest2")
    .Input("a: float")
    .Input("b: float")
    .Output("o: float")
    .SetShapeFn(::tensorflow::shape_inference::UnknownShape);

REGISTER_OP("AddNLikeTest")
    .Input("inputs: N * T")
    .Output("sum: T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype")
    .SetIsCommutative()
    .SetIsAggregate();

Node* Sequencer(const GraphDefBuilder::Options& opts,
                const string& call_node_name) {
  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp("NoOp"), "NoOp",
                           opts.op_registry());
  return opts.WithAttr(kXlaHostTransferSequencerAttr, call_node_name)
      .FinalizeBuilder(&node_builder);
}

Node* Input(const GraphDefBuilder::Options& opts) {
  return ops::SourceOp("InputTest", opts);
}

Node* InputShaped(const GraphDefBuilder::Options& opts) {
  return ops::SourceOp("InputTestShaped", opts);
}

Node* KnownShapeBase(DataType dtype, const gtl::ArraySlice<int>& shape,
                     const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp("Const"), "Const",
                           opts.op_registry());
  TensorProto value;
  value.set_dtype(dtype);
  for (int dim : shape) {
    value.mutable_tensor_shape()->add_dim()->set_size(dim);
  }
  return opts.WithAttr("value", value)
      .WithAttr("dtype", dtype)
      .FinalizeBuilder(&node_builder);
}

Node* KnownShape(const gtl::ArraySlice<int>& shape,
                 const GraphDefBuilder::Options& opts) {
  return KnownShapeBase(DT_FLOAT, shape, opts);
}

Node* KeyPlaceholderShape(const GraphDefBuilder::Options& opts) {
  return KnownShapeBase(DT_STRING, {2}, opts);
}

Node* KeyPlaceholder(const string& call_node,
                     const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp("Placeholder"), "Placeholder",
                           opts.op_registry());
  TensorShapeProto shape;
  shape.add_dim()->set_size(2);
  return opts.WithAttr("shape", shape)
      .WithAttr("dtype", DT_STRING)
      .WithAttr("_host_compute_call_node", call_node)
      .FinalizeBuilder(&node_builder);
}

Node* RecvAtHost(ops::NodeOut key_input, const string& cluster,
                 const string& oc_cluster,
                 const gtl::ArraySlice<DataType>& dtypes,
                 const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  string key =
      strings::StrCat("host_compute_channel_", cluster, "_", oc_cluster);
  string name = strings::StrCat("outside_compilation_", cluster, "_",
                                oc_cluster, "_recv");
  NodeBuilder node_builder(opts.WithName(name).GetNameForOp("_XlaRecvAtHost"),
                           "_XlaRecvAtHost", opts.op_registry());
  node_builder.Input(std::move(key_input));
  return opts.WithAttr("Toutputs", dtypes)
      .WithAttr("key", key)
      .WithAttr("device_ordinal", 0)
      .WithAttr("_encapsulate", cluster)
      .WithAttr("_outside", oc_cluster)
      .FinalizeBuilder(&node_builder);
}

Node* SendFromHost(ops::NodeOut key_input, const string& cluster,
                   const string& oc_cluster,
                   const std::vector<ops::NodeOut>& inputs,
                   const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  string key =
      strings::StrCat("host_compute_channel_", cluster, "_", oc_cluster);
  string name = strings::StrCat("outside_compilation_", cluster, "_",
                                oc_cluster, "_send");
  NodeBuilder node_builder(opts.WithName(name).GetNameForOp("_XlaSendFromHost"),
                           "_XlaSendFromHost", opts.op_registry());
  node_builder.Input(inputs);
  node_builder.Input(std::move(key_input));
  std::vector<DataType> dtypes;
  for (const auto& node : inputs) {
    dtypes.push_back(node.dt);
  }
  return opts.WithAttr("Tinputs", dtypes)
      .WithAttr("key", key)
      .WithAttr("device_ordinal", 0)
      .WithAttr("_encapsulate", cluster)
      .WithAttr("_outside", oc_cluster)
      .FinalizeBuilder(&node_builder);
}

Node* Unary(ops::NodeOut a, const GraphDefBuilder::Options& opts) {
  return ops::UnaryOp("UnaryTest", std::move(a), opts);
}

Node* Binary(ops::NodeOut a, ops::NodeOut b,
             const GraphDefBuilder::Options& opts) {
  return ops::BinaryOp("BinaryTest", std::move(a), std::move(b), opts);
}

Node* BinaryUnknownShape(ops::NodeOut a, ops::NodeOut b,
                         const GraphDefBuilder::Options& opts) {
  return ops::BinaryOp("BinaryTest2", std::move(a), std::move(b), opts);
}

Node* AddNLike(const std::vector<ops::NodeOut>& inputs,
               const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp("AddN"), "AddNLikeTest",
                           opts.op_registry());
  node_builder.Input(inputs);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ArgOp(int index, DataType type, const GraphDefBuilder::Options& opts) {
  return ops::SourceOp("_Arg",
                       opts.WithAttr("T", type).WithAttr("index", index));
}

Node* RetOp(int index, ops::NodeOut a, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  NodeBuilder node_builder(opts.GetNameForOp("Retval"), "_Retval",
                           opts.op_registry());
  node_builder.Input(std::move(a)).Attr("index", index);
  return opts.FinalizeBuilder(&node_builder);
}

Status Encapsulate(GraphDef* graphdef, FunctionDefLibrary* library) {
  Status s;
  // Convert the GraphDef to a Graph
  std::unique_ptr<FunctionLibraryDefinition> lib_def(
      new FunctionLibraryDefinition(OpRegistry::Global(), *library));
  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  std::unique_ptr<Graph> graph(new Graph(lib_def.get()));
  s = ConvertGraphDefToGraph(options, *graphdef, graph.get());
  if (!s.ok()) return s;

  std::unique_ptr<Graph> graph_out;
  s = EncapsulateSubgraphsInFunctions("_encapsulate", "_outside", *graph,
                                      /*rewrite_subgraph_fn=*/{},
                                      /*parallel_checking=*/false,
                                      /*reuse_existing_functions=*/false,
                                      &graph_out, lib_def.get());
  if (!s.ok()) return s;

  GraphDef graphdef_out;
  graph_out->ToGraphDef(&graphdef_out);
  graphdef->Swap(&graphdef_out);

  *library = lib_def->ToProto();
  return s;
}

// If there are no marked nodes, funcification should be a no-op.
TEST(EncapsulateSubgraphsTest, NoFunctions) {
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);

  Node* a = Input(builder.opts().WithName("A"));
  Node* b = Input(builder.opts().WithName("B"));
  Node* c = Unary(a, builder.opts().WithName("C"));
  Binary(b, c, builder.opts().WithName("D"));

  GraphDef graphdef_in;
  FunctionDefLibrary library_in;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef_in));
  *library_in.add_function() = test::function::XTimesTwo();

  GraphDef graphdef_out = graphdef_in;
  FunctionDefLibrary library_out = library_in;
  TF_EXPECT_OK(Encapsulate(&graphdef_out, &library_out));

  // If there are no marked nodes, funcification should be a no-op.
  TF_EXPECT_GRAPH_EQ(graphdef_in, graphdef_out);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_in, library_out);
}

// Test with one function to transform.
TEST(EncapsulateSubgraphsTest, OneFunction) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    *library.add_function() = test::function::XTimesTwo();

    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    // Give nodes 'c' and 'd' names that collide after lowercasing.
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d = Binary(b, c, b1.opts().WithName("c").WithControlInput(c).WithAttr(
                               "_encapsulate", "F1"));
    Binary(a, d, b1.opts().WithName("E"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = test::function::XTimesTwo();
  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"c_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"c"}, "BinaryTest", {"b_0_arg", "C:o:0"}, {}, {"C"}},
      },
      {{"c_0_retval", "c:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    NodeBuilder node_builder("F1", "F1", lib_def.get());
    node_builder.Input(a).Input(b);
    Node* call = b2.opts().FinalizeBuilder(&node_builder);

    Binary(a, call, b2.opts().WithName("E"));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with two functions to transform.
TEST(EncapsulateSubgraphsTest, TwoFunctions) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    *library.add_function() = test::function::XTimesTwo();

    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* control = Input(b1.opts().WithName("Control"));
    Node* c =
        Unary(a, b1.opts().WithName("C").WithControlInput(control).WithAttr(
                     "_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithControlInput(control).WithAttr(
                         "_encapsulate", "F2"));
    Binary(a, d, b1.opts().WithName("E"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = test::function::XTimesTwo();
  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float"}, {"c_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
      },
      {{"c_0_retval", "C:o:0"}});
  *library_expected.add_function() = FunctionDefHelper::Create(
      "F2", {"b_0_arg:float", "c_0_arg:float"}, {"d_0_retval:float"}, {},
      {
          {{"D"}, "BinaryTest", {"b_0_arg", "c_0_arg"}},
      },
      {{"d_0_retval", "D:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));
    Node* control = Input(b2.opts().WithName("Control"));

    NodeBuilder nb("F1", "F1", lib_def.get());
    nb.Input(a).ControlInput(control);
    Node* call1 = b2.opts().FinalizeBuilder(&nb);

    NodeBuilder nb2("F2", "F2", lib_def.get());
    nb2.Input(b).Input(call1).ControlInput(control);
    Node* call2 = b2.opts().FinalizeBuilder(&nb2);

    Binary(a, call2, b2.opts().WithName("E"));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  // If there are no marked nodes, funcification should be a no-op.
  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Returns a vector of node names in 'graph', sorted by name.
std::vector<string> GraphNodes(const Graph& graph) {
  std::vector<string> nodes;
  for (const auto& node : graph.nodes()) {
    if (!node->IsSource() && !node->IsSink()) {
      nodes.push_back(node->name());
    }
  }
  std::sort(nodes.begin(), nodes.end());
  return nodes;
}

// Returns a sorted vector of (src, dst) edges in 'graph'.
std::vector<std::pair<string, string>> GraphEdges(const Graph& graph) {
  std::vector<std::pair<string, string>> edges;
  for (const Edge* edge : graph.edges()) {
    if (edge->src()->IsSource() || edge->dst()->IsSink()) continue;
    edges.emplace_back(
        strings::StrCat(edge->src()->name(), ":", edge->src_output()),
        strings::StrCat(edge->dst()->name(), ":", edge->dst_input()));
  }
  std::sort(edges.begin(), edges.end());
  return edges;
}

TEST(EncapsulateSubgraphsTest, InputDeduplication) {
  Scope root = Scope::NewRootScope().ExitOnError().WithDevice(
      "/job:localhost/replica:0/task:0/cpu:0");
  auto x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  auto add1 = ops::Add(root.WithOpName("add1"), x, x);
  add1.node()->AddAttr("_cluster", "cluster1");
  auto add2 = ops::Add(root.WithOpName("add2"), add1, add1);
  add2.node()->AddAttr("_cluster", "cluster2");
  auto out = ops::Mul(root.WithOpName("mul"), add1, add2);

  Graph graph_before_encapsulation(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph_before_encapsulation));

  FunctionLibraryDefinition library(OpRegistry::Global(), {});
  std::unique_ptr<Graph> graph;
  TF_ASSERT_OK(EncapsulateSubgraphsInFunctions(
      "_cluster", "_outside", graph_before_encapsulation,
      /*rewrite_subgraph_fn=*/{}, /*parallel_checking=*/false,
      /*reuse_existing_functions=*/false, &graph, &library));

  std::vector<string> expected_nodes = {"cluster1", "cluster2", "mul", "x"};
  EXPECT_EQ(expected_nodes, GraphNodes(*graph));

  std::vector<std::pair<string, string>> expected_edges = {
      {"cluster1:0", "cluster2:0"},
      {"cluster1:0", "mul:0"},
      {"cluster2:0", "mul:1"},
      {"x:0", "cluster1:0"}};
  EXPECT_EQ(expected_edges, GraphEdges(*graph));
}

TEST(EncapsulateSubgraphsTest, ParallelChecking) {
  Scope root = Scope::NewRootScope().ExitOnError().WithDevice(
      "/job:localhost/replica:0/task:0/cpu:0");
  auto x1 = ops::Placeholder(root.WithOpName("x1"), DT_FLOAT);
  auto x2 = ops::Placeholder(root.WithOpName("x2"), DT_FLOAT);
  auto add1 = ops::Add(root.WithOpName("add1"), x1, x2);
  add1.node()->AddAttr("_cluster", "cluster1");
  auto add2 = ops::Add(root.WithOpName("add2"), add1, x2);
  add2.node()->AddAttr("_cluster", "cluster1");
  auto out = ops::Mul(root.WithOpName("mul"), x1, add2);

  Graph graph_before_encapsulation(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph_before_encapsulation));

  FunctionLibraryDefinition library(OpRegistry::Global(), {});
  std::unique_ptr<Graph> graph;
  TF_ASSERT_OK(EncapsulateSubgraphsInFunctions(
      "_cluster", "_outside", graph_before_encapsulation,
      /*rewrite_subgraph_fn=*/{}, /*parallel_checking=*/true,
      /*reuse_existing_functions=*/false, &graph, &library));

  std::vector<string> expected_nodes = {
      "add1", "add2", "cluster1", "cluster1_parallel_check/_0",
      "mul",  "x1",   "x2"};
  EXPECT_EQ(expected_nodes, GraphNodes(*graph));

  std::vector<std::pair<string, string>> expected_edges = {
      {"add1:0", "add2:0"},
      {"add2:0", "cluster1_parallel_check/_0:0"},
      {"cluster1:0", "cluster1_parallel_check/_0:1"},
      {"cluster1_parallel_check/_0:0", "mul:1"},
      {"x1:0", "add1:0"},
      {"x1:0", "cluster1:0"},
      {"x1:0", "mul:0"},
      {"x2:0", "add1:1"},
      {"x2:0", "add2:1"},
      {"x2:0", "cluster1:1"},
  };
  EXPECT_EQ(expected_edges, GraphEdges(*graph));
}

const Node* FindNodeByName(const Graph& graph, const string& name) {
  for (const Node* node : graph.nodes()) {
    if (node->name() == name) return node;
  }
  return nullptr;
}

bool HasGuaranteeConstAttr(const Node& n) {
  bool is_guaranteed_constant = false;
  if (!GetNodeAttr(n.attrs(), "_is_guaranteed_constant",
                   &is_guaranteed_constant)
           .ok()) {
    return false;
  }
  return is_guaranteed_constant;
}

TEST(EncapsulateSubgraphsWithGuaranteeConstOpTest, Simple) {
  Scope root = Scope::NewRootScope().ExitOnError().WithDevice(
      "/job:localhost/replica:0/task:0/cpu:0");
  auto x1 = ops::Placeholder(root.WithOpName("x1"), DT_FLOAT);
  auto const_x2 = ops::Const(root.WithOpName("const_x2"), 10.0f);
  auto const_guarantee_x1 =
      ops::GuaranteeConst(root.WithOpName("const_guarantee_x1"), x1);
  auto add1 = ops::Add(root.WithOpName("add1"), const_guarantee_x1, const_x2);
  add1.node()->AddAttr("_encapsulate", "encapsulate1");

  Graph graph_before(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph_before));

  std::unique_ptr<Graph> graph_after;
  FunctionLibraryDefinition library(OpRegistry::Global(), {});
  int guaranteed_consts = 0;
  TF_ASSERT_OK(EncapsulateSubgraphsInFunctions(
      "_encapsulate", "_outside", graph_before,
      /*rewrite_subgraph_fn=*/
      [&guaranteed_consts](std::unique_ptr<Graph>* graph_ptr,
                           std::vector<int>* input_permutation,
                           std::vector<int>* output_permutation,
                           NodeDef* call_def) {
        Graph* graph = graph_ptr->get();
        for (const Node* n : graph->nodes()) {
          if (n->type_string() == "_Arg" &&
              str_util::StartsWith(n->name(), "const")) {
            ++guaranteed_consts;
            EXPECT_TRUE(HasGuaranteeConstAttr(*n));
          } else {
            EXPECT_FALSE(HasGuaranteeConstAttr(*n));
          }
        }
        return Status::OK();
      },
      /*parallel_checking=*/false,
      /*reuse_existing_functions=*/false, &graph_after, &library));
  EXPECT_EQ(2, guaranteed_consts);
}

TEST(EncapsulateSubgraphsWithGuaranteeConstOpTest, Add) {
  Scope root = Scope::NewRootScope().ExitOnError().WithDevice(
      "/job:localhost/replica:0/task:0/cpu:0");
  auto x1 = ops::Placeholder(root.WithOpName("x1"), DT_FLOAT);
  auto x2 = ops::Placeholder(root.WithOpName("x2"), DT_FLOAT);
  auto const_guarantee_x1 =
      ops::GuaranteeConst(root.WithOpName("const_guarantee_x1"), x1);
  auto const_guarantee_x2 =
      ops::GuaranteeConst(root.WithOpName("const_guarantee_x2"), x2);
  auto const_guarantee_add1 = ops::Add(root.WithOpName("const_guarantee_add1"),
                                       const_guarantee_x1, const_guarantee_x2);
  auto add2 = ops::Add(root.WithOpName("add2"), const_guarantee_x1, x2);
  auto mul1 = ops::Mul(root.WithOpName("mul1"), const_guarantee_add1, add2);
  mul1.node()->AddAttr("_encapsulate", "encapsulate1");

  Graph graph_before(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph_before));

  std::unique_ptr<Graph> graph_after;
  FunctionLibraryDefinition library(OpRegistry::Global(), {});
  int guaranteed_consts = 0;
  TF_ASSERT_OK(EncapsulateSubgraphsInFunctions(
      "_encapsulate", "_outside", graph_before,
      /*rewrite_subgraph_fn=*/
      [&guaranteed_consts](std::unique_ptr<Graph>* graph_ptr,
                           std::vector<int>* input_permutation,
                           std::vector<int>* output_permutation,
                           NodeDef* call_def) {
        Graph* graph = graph_ptr->get();
        for (const Node* n : graph->nodes()) {
          if (n->type_string() == "_Arg" &&
              str_util::StartsWith(n->name(), "const")) {
            ++guaranteed_consts;
            EXPECT_TRUE(HasGuaranteeConstAttr(*n));
          } else {
            EXPECT_FALSE(HasGuaranteeConstAttr(*n));
          }
        }
        return Status::OK();
      },
      /*parallel_checking=*/false,
      /*reuse_existing_functions=*/false, &graph_after, &library));
  // Only 1 runtime const, which is const_guarantee_add1. Add2 has one const
  // and another non-const, so overall non-const.
  EXPECT_EQ(1, guaranteed_consts);
}

// Test with one function to transform and one outside_compilation cluster.
TEST(EncapsulateSubgraphsTest, OneFunctionOneOutside) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    *library.add_function() = test::function::XTimesTwo();

    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    // Give nodes 'c' and 'd' names that collide after lowercasing.
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d = Binary(b, c,
                     b1.opts().WithName("c").WithControlInput(c).WithAttr(
                         "_encapsulate", "F1"));
    Node* e = Binary(c, d,
                     b1.opts()
                         .WithName("E")
                         .WithControlInputs({b, d})
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O1"));
    Node* f = Binary(c, e,
                     b1.opts().WithName("F").WithControlInput(e).WithAttr(
                         "_encapsulate", "F1"));
    Binary(a, f, b1.opts().WithName("G").WithControlInput(e));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  {
    GraphDefBuilder shape(GraphDefBuilder::kFailImmediately);
    Node* key_constant =
        KeyPlaceholderShape(shape.opts().WithName("KnownShape/_0"));
    Node* recv = RecvAtHost(ops::NodeOut(key_constant, 0), "F1", "O1",
                            {DT_FLOAT, DT_FLOAT}, shape.opts());
    Node* e = Binary(ops::NodeOut(recv, 0), ops::NodeOut(recv, 1),
                     shape.opts()
                         .WithName("E")
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O1"));
    SendFromHost(ops::NodeOut(key_constant, 0), "F1", "O1", {e}, shape.opts());
    TF_EXPECT_OK(
        AddGraphDefToFunctionLibrary(shape, "F1_O1", &library_expected));
  }

  *library_expected.add_function() = test::function::XTimesTwo();
  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"f_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"c"}, "BinaryTest", {"b_0_arg", "C:o:0"}, {}, {"C"}},
          {{"F"},
           "BinaryTest",
           {"C:o:0", "outside_compilation_O1_host_compute:outputs:0"},
           {},
           {"outside_compilation_O1_host_compute"}},
          {{"outside_compilation_O1_host_compute"},
           "XlaHostCompute",
           {"C:o:0", "c:o:0"},
           {{"Tinputs", gtl::ArraySlice<DataType>({DT_FLOAT, DT_FLOAT})},
            {"Toutputs", gtl::ArraySlice<DataType>({DT_FLOAT})},
            {"key", "host_compute_channel_F1_O1"},
            {"shape_inference_graph",
             "_outside_compilation_shape_inference_F1_O1"},
            {"shapes", gtl::ArraySlice<DataType>({})},
            {"_outside_compilation_subgraph", "O1"}},
           {"c"}},
      },
      {{"f_0_retval", "F:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    Node* key_constant =
        KeyPlaceholder("F1", b2.opts().WithName("F1_key_placeholder"));
    Node* recv = RecvAtHost(ops::NodeOut(key_constant, 0), "F1", "O1",
                            {DT_FLOAT, DT_FLOAT}, b2.opts());
    Node* e = Binary(ops::NodeOut(recv, 0), ops::NodeOut(recv, 1),
                     b2.opts()
                         .WithName("E")
                         .WithControlInputs({recv, b})
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O1"));
    Node* send = SendFromHost(ops::NodeOut(key_constant, 0), "F1", "O1", {e},
                              b2.opts().WithControlInput(e));

    Node* s = Sequencer(
        b2.opts().WithName("F1_sequencer").WithControlInputs({recv, send}),
        "F1");

    NodeBuilder node_builder("F1", "F1", lib_def.get());
    node_builder.Input(a).Input(b);
    Node* call =
        b2.opts().WithControlInputs({s}).FinalizeBuilder(&node_builder);

    Binary(a, call, b2.opts().WithName("G").WithControlInputs({e}));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with one function to transform and two outside_compilation clusters.
TEST(EncapsulateSubgraphsTest, OneFunctionTwoOutside) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithAttr("_encapsulate", "F1"));
    Node* e = Binary(c, d,
                     b1.opts()
                         .WithName("E")
                         .WithControlInputs({b, d})
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O1"));
    Node* f = Binary(c, e,
                     b1.opts().WithName("F").WithControlInput(e).WithAttr(
                         "_encapsulate", "F1"));
    Node* g = Binary(e, f,
                     b1.opts()
                         .WithName("G")
                         .WithControlInputs({e, f})
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O2"));
    Node* h = Binary(d, e,
                     b1.opts()
                         .WithName("H")
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O2"));
    Node* i = Unary(h, b1.opts().WithName("I").WithAttr("_encapsulate", "F1"));
    Binary(g, i, b1.opts().WithName("J"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  {
    GraphDefBuilder shape1(GraphDefBuilder::kFailImmediately);
    Node* key_constant =
        KeyPlaceholderShape(shape1.opts().WithName("KnownShape/_0"));
    Node* recv = RecvAtHost(ops::NodeOut(key_constant, 0), "F1", "O1",
                            {DT_FLOAT, DT_FLOAT}, shape1.opts());
    Node* e = Binary(ops::NodeOut(recv, 0), ops::NodeOut(recv, 1),
                     shape1.opts()
                         .WithName("E")
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O1"));
    SendFromHost(ops::NodeOut(key_constant, 0), "F1", "O1", {e}, shape1.opts());
    TF_EXPECT_OK(
        AddGraphDefToFunctionLibrary(shape1, "F1_O1", &library_expected));
  }

  {
    GraphDefBuilder shape2(GraphDefBuilder::kFailImmediately);
    Node* key_constant =
        KeyPlaceholderShape(shape2.opts().WithName("KnownShape/_0"));
    Node* recv1 = RecvAtHost(ops::NodeOut(key_constant, 0), "F1", "O1",
                             {DT_FLOAT, DT_FLOAT}, shape2.opts());
    Node* e = Binary(ops::NodeOut(recv1, 0), ops::NodeOut(recv1, 1),
                     shape2.opts()
                         .WithName("E")
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O1"));
    Node* recv2 = RecvAtHost(ops::NodeOut(key_constant, 0), "F1", "O2",
                             {DT_FLOAT, DT_FLOAT}, shape2.opts());
    Node* h = Binary(ops::NodeOut(recv2, 0), e,
                     shape2.opts()
                         .WithName("H")
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O2"));
    SendFromHost(ops::NodeOut(key_constant, 0), "F1", "O2", {h}, shape2.opts());
    TF_EXPECT_OK(
        AddGraphDefToFunctionLibrary(shape2, "F1_O2", &library_expected));
  }

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"i_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"D"}, "BinaryTest", {"b_0_arg", "C:o:0"}, {}},
          {{"I"},
           "UnaryTest",
           {"outside_compilation_O2_host_compute:outputs:0"}},
          {{"F"},
           "BinaryTest",
           {"C:o:0", "outside_compilation_O1_host_compute:outputs:0"},
           {},
           {"outside_compilation_O1_host_compute"}},
          {{"outside_compilation_O2_host_compute"},
           "XlaHostCompute",
           {"D:o:0", "F:o:0"},
           {{"Tinputs", gtl::ArraySlice<DataType>({DT_FLOAT, DT_FLOAT})},
            {"Toutputs", gtl::ArraySlice<DataType>({DT_FLOAT})},
            {"key", "host_compute_channel_F1_O2"},
            {"shape_inference_graph",
             "_outside_compilation_shape_inference_F1_O2"},
            {"shapes", gtl::ArraySlice<DataType>({})},
            {"_outside_compilation_subgraph", "O2"}},
           {"F"}},
          {{"outside_compilation_O1_host_compute"},
           "XlaHostCompute",
           {"C:o:0", "D:o:0"},
           {{"Tinputs", gtl::ArraySlice<DataType>({DT_FLOAT, DT_FLOAT})},
            {"Toutputs", gtl::ArraySlice<DataType>({DT_FLOAT})},
            {"key", "host_compute_channel_F1_O1"},
            {"shape_inference_graph",
             "_outside_compilation_shape_inference_F1_O1"},
            {"shapes", gtl::ArraySlice<DataType>({})},
            {"_outside_compilation_subgraph", "O1"}},
           {"D"}},
      },
      {{"i_0_retval", "I:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    Node* key_constant =
        KeyPlaceholder("F1", b2.opts().WithName("F1_key_placeholder"));
    Node* recv1 = RecvAtHost(ops::NodeOut(key_constant, 0), "F1", "O1",
                             {DT_FLOAT, DT_FLOAT}, b2.opts());
    Node* e = Binary(ops::NodeOut(recv1, 0), ops::NodeOut(recv1, 1),
                     b2.opts()
                         .WithName("E")
                         .WithControlInputs({recv1, b})
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O1"));
    Node* send1 = SendFromHost(ops::NodeOut(key_constant, 0), "F1", "O1", {e},
                               b2.opts().WithControlInput(e));

    Node* recv2 = RecvAtHost(ops::NodeOut(key_constant, 0), "F1", "O2",
                             {DT_FLOAT, DT_FLOAT}, b2.opts());
    Node* g = Binary(e, ops::NodeOut(recv2, 1),
                     b2.opts()
                         .WithName("G")
                         .WithControlInputs({recv2, e})
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O2"));
    Node* h = Binary(ops::NodeOut(recv2, 0), e,
                     b2.opts()
                         .WithName("H")
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O2"));
    Node* send2 =
        SendFromHost(ops::NodeOut(key_constant, 0), "F1", "O2", {h}, b2.opts());

    Node* s = Sequencer(b2.opts()
                            .WithName("F1_sequencer")
                            .WithControlInputs({recv1, send1, recv2, send2}),
                        "F1");

    NodeBuilder node_builder("F1", "F1", lib_def.get());
    node_builder.Input(a).Input(b);
    Node* call = b2.opts().WithControlInput(s).FinalizeBuilder(&node_builder);

    Binary(g, call, b2.opts().WithName("J"));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with two functions to transform, each with one outside_compilation
// cluster.
TEST(EncapsulateSubgraphsTest, TwoFunctionsTwoOutside) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = InputShaped(b1.opts().WithName("A"));
    Node* b = InputShaped(b1.opts().WithName("B"));
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithAttr("_encapsulate", "F1"));
    Node* e = Binary(c, d,
                     b1.opts()
                         .WithName("E")
                         .WithControlInputs({b, d})
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O1"));
    Node* f = Binary(c, e,
                     b1.opts().WithName("F").WithControlInput(e).WithAttr(
                         "_encapsulate", "F1"));
    Node* g = Binary(e, f,
                     b1.opts().WithName("G").WithControlInputs({e, f}).WithAttr(
                         "_encapsulate", "F2"));
    Node* h = Binary(d, g,
                     b1.opts()
                         .WithName("H")
                         .WithAttr("_encapsulate", "F2")
                         .WithAttr("_outside", "O1"));
    Node* i =
        Binary(f, h, b1.opts().WithName("I").WithAttr("_encapsulate", "F2"));
    Binary(g, i, b1.opts().WithName("J"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  {
    GraphDefBuilder shape(GraphDefBuilder::kFailImmediately);
    Node* key_constant =
        KeyPlaceholderShape(shape.opts().WithName("KnownShape/_0"));
    Node* recv = RecvAtHost(ops::NodeOut(key_constant, 0), "F1", "O1",
                            {DT_FLOAT, DT_FLOAT}, shape.opts());
    Node* e = Binary(ops::NodeOut(recv, 0), ops::NodeOut(recv, 1),
                     shape.opts()
                         .WithName("E")
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O1"));
    SendFromHost(ops::NodeOut(key_constant, 0), "F1", "O1", {e}, shape.opts());
    TF_EXPECT_OK(
        AddGraphDefToFunctionLibrary(shape, "F1_O1", &library_expected));
  }

  TensorShapeProto shape_proto_expected;
  shape_proto_expected.add_dim()->set_size(2);

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"},
      {"f_0_retval:float", "d_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"D"}, "BinaryTest", {"b_0_arg", "C:o:0"}},
          {{"F"},
           "BinaryTest",
           {"C:o:0", "outside_compilation_O1_host_compute:outputs:0"},
           {},
           {"outside_compilation_O1_host_compute"}},
          {{"outside_compilation_O1_host_compute"},
           "XlaHostCompute",
           {"C:o:0", "D:o:0"},
           {{"Tinputs", gtl::ArraySlice<DataType>({DT_FLOAT, DT_FLOAT})},
            {"Toutputs", gtl::ArraySlice<DataType>({DT_FLOAT})},
            {"key", "host_compute_channel_F1_O1"},
            {"shape_inference_graph",
             "_outside_compilation_shape_inference_F1_O1"},
            {"shapes", gtl::ArraySlice<DataType>({})},
            {"_outside_compilation_subgraph", "O1"}},
           {"D"}},
      },
      {{"d_0_retval", "D:o:0"}, {"f_0_retval", "F:o:0"}});

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F2", {"e_0_arg:float", "f_0_arg:float"},
      {"g_0_retval:float", "i_0_retval:float"}, {},
      {
          {{"G"}, "BinaryTest", {"e_0_arg", "f_0_arg"}},
          {{"I"},
           "BinaryTest",
           {"f_0_arg", "outside_compilation_O1_host_compute:outputs:0"}},
          {{"outside_compilation_O1_host_compute"},
           "XlaHostCompute",
           {"G:o:0"},
           {{"Tinputs", gtl::ArraySlice<DataType>({DT_FLOAT})},
            {"Toutputs", gtl::ArraySlice<DataType>({DT_FLOAT})},
            {"key", "host_compute_channel_F2_O1"},
            {"shape_inference_graph", ""},
            {"shapes",
             gtl::ArraySlice<TensorShapeProto>({shape_proto_expected})},
            {"_outside_compilation_subgraph", "O1"}}},
      },
      {{"g_0_retval", "G:o:0"}, {"i_0_retval", "I:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = InputShaped(b2.opts().WithName("A"));
    Node* b = InputShaped(b2.opts().WithName("B"));

    Node* key_constant1 =
        KeyPlaceholder("F1", b2.opts().WithName("F1_key_placeholder"));
    Node* recv1 = RecvAtHost(ops::NodeOut(key_constant1, 0), "F1", "O1",
                             {DT_FLOAT, DT_FLOAT}, b2.opts());
    Node* e = Binary(ops::NodeOut(recv1, 0), ops::NodeOut(recv1, 1),
                     b2.opts()
                         .WithName("E")
                         .WithControlInputs({recv1, b})
                         .WithAttr("_encapsulate", "F1")
                         .WithAttr("_outside", "O1"));
    Node* send1 = SendFromHost(ops::NodeOut(key_constant1, 0), "F1", "O1", {e},
                               b2.opts().WithControlInput(e));
    Node* s1 = Sequencer(
        b2.opts().WithName("F1_sequencer").WithControlInputs({recv1, send1}),
        "F1");

    NodeBuilder node_builder1("F1", "F1", lib_def.get());
    node_builder1.Input(a).Input(b);
    Node* call1 =
        b2.opts().WithControlInput(s1).FinalizeBuilder(&node_builder1);

    Node* key_constant2 =
        KeyPlaceholder("F2", b2.opts().WithName("F2_key_placeholder"));
    Node* recv2 = RecvAtHost(ops::NodeOut(key_constant2, 0), "F2", "O1",
                             {DT_FLOAT}, b2.opts());
    Node* h = Binary(ops::NodeOut(call1, 1), recv2,
                     b2.opts()
                         .WithName("H")
                         .WithAttr("_encapsulate", "F2")
                         .WithAttr("_outside", "O1"));
    Node* send2 = SendFromHost(ops::NodeOut(key_constant2, 0), "F2", "O1", {h},
                               b2.opts());

    Node* s2 = Sequencer(
        b2.opts().WithName("F2_sequencer").WithControlInputs({recv2, send2}),
        "F2");
    NodeBuilder node_builder2("F2", "F2", lib_def.get());
    node_builder2.Input(e).Input(call1);
    Node* call2 = b2.opts()
                      .WithControlInputs({s2, e, call1})
                      .FinalizeBuilder(&node_builder2);
    Binary(call2, ops::NodeOut(call2, 1), b2.opts().WithName("J"));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with one outside_compilation cluster that has no inputs from the
// compiled subgraph.
TEST(EncapsulateSubgraphsTest, OutsideCompilationNoInputs) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = InputShaped(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithAttr("_encapsulate", "F1"));
    Node* e = Unary(a, b1.opts()
                           .WithName("E")
                           .WithAttr("_encapsulate", "F1")
                           .WithAttr("_outside", "O1"));
    Node* f =
        Binary(d, e, b1.opts().WithName("F").WithAttr("_encapsulate", "F1"));
    Unary(f, b1.opts().WithName("G"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  TensorShapeProto shape_proto_expected;
  shape_proto_expected.add_dim()->set_size(2);

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"f_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"D"}, "BinaryTest", {"b_0_arg", "C:o:0"}},
          {{"F"},
           "BinaryTest",
           {"D:o:0", "outside_compilation_O1_host_compute:outputs:0"}},
          {{"outside_compilation_O1_host_compute"},
           "XlaHostCompute",
           {},
           {{"Tinputs", gtl::ArraySlice<DataType>({})},
            {"Toutputs", gtl::ArraySlice<DataType>({DT_FLOAT})},
            {"key", "host_compute_channel_F1_O1"},
            {"shape_inference_graph", ""},
            {"shapes",
             gtl::ArraySlice<TensorShapeProto>({shape_proto_expected})},
            {"_outside_compilation_subgraph", "O1"}}},
      },
      {{"f_0_retval", "F:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = InputShaped(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    Node* e = Unary(a, b2.opts()
                           .WithName("E")
                           .WithAttr("_encapsulate", "F1")
                           .WithAttr("_outside", "O1"));
    Node* key_constant =
        KeyPlaceholder("F1", b2.opts().WithName("F1_key_placeholder"));
    Node* send1 =
        SendFromHost(ops::NodeOut(key_constant, 0), "F1", "O1", {e}, b2.opts());
    Node* s1 = Sequencer(
        b2.opts().WithName("F1_sequencer").WithControlInput(send1), "F1");
    NodeBuilder node_builder1("F1", "F1", lib_def.get());
    node_builder1.Input(a).Input(b);
    Node* call1 =
        b2.opts().WithControlInput(s1).FinalizeBuilder(&node_builder1);

    Unary(call1, b2.opts().WithName("G"));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with one outside_compilation cluster that has no data inputs but has a
// control input from the compiled subgraph.
TEST(EncapsulateSubgraphsTest, OutsideCompilationControlInput) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = InputShaped(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithAttr("_encapsulate", "F1"));
    Node* e = Unary(a, b1.opts()
                           .WithName("E")
                           .WithControlInput(d)
                           .WithAttr("_encapsulate", "F1")
                           .WithAttr("_outside", "O1"));
    Node* f =
        Binary(d, e, b1.opts().WithName("F").WithAttr("_encapsulate", "F1"));
    Unary(f, b1.opts().WithName("G"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  TensorShapeProto shape_proto_expected;
  shape_proto_expected.add_dim()->set_size(2);

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"f_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"D"}, "BinaryTest", {"b_0_arg", "C:o:0"}},
          {{"F"},
           "BinaryTest",
           {"D:o:0", "outside_compilation_O1_host_compute:outputs:0"}},
          {{"outside_compilation_O1_host_compute"},
           "XlaHostCompute",
           {},
           {{"Tinputs", gtl::ArraySlice<DataType>({})},
            {"Toutputs", gtl::ArraySlice<DataType>({DT_FLOAT})},
            {"key", "host_compute_channel_F1_O1"},
            {"shape_inference_graph", ""},
            {"shapes",
             gtl::ArraySlice<TensorShapeProto>({shape_proto_expected})},
            {"_outside_compilation_subgraph", "O1"}},
           {"D"}},
      },
      {{"f_0_retval", "F:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = InputShaped(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    Node* key_constant =
        KeyPlaceholder("F1", b2.opts().WithName("F1_key_placeholder"));
    Node* recv1 =
        RecvAtHost(ops::NodeOut(key_constant, 0), "F1", "O1", {}, b2.opts());
    Node* e = Unary(a, b2.opts()
                           .WithName("E")
                           .WithControlInput(recv1)
                           .WithAttr("_encapsulate", "F1")
                           .WithAttr("_outside", "O1"));
    Node* send1 =
        SendFromHost(ops::NodeOut(key_constant, 0), "F1", "O1", {e}, b2.opts());
    Node* s1 = Sequencer(
        b2.opts().WithName("F1_sequencer").WithControlInputs({recv1, send1}),
        "F1");
    NodeBuilder node_builder1("F1", "F1", lib_def.get());
    node_builder1.Input(a).Input(b);
    Node* call1 =
        b2.opts().WithControlInput(s1).FinalizeBuilder(&node_builder1);

    Unary(call1, b2.opts().WithName("G"));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with one outside_compilation cluster that has no outputs from the
// compiled subgraph.
TEST(EncapsulateSubgraphsTest, OutsideCompilationNoOutputs) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithAttr("_encapsulate", "F1"));
    Node* e = Unary(d, b1.opts()
                           .WithName("E")
                           .WithAttr("_encapsulate", "F1")
                           .WithAttr("_outside", "O1"));
    Node* f = Unary(d, b1.opts().WithName("F").WithAttr("_encapsulate", "F1"));
    Binary(e, f, b1.opts().WithName("G"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"f_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"D"}, "BinaryTest", {"b_0_arg", "C:o:0"}},
          {{"F"}, "UnaryTest", {"D:o:0"}},
          {{"outside_compilation_O1_host_compute"},
           "XlaHostCompute",
           {"D:o:0"},
           {{"Tinputs", gtl::ArraySlice<DataType>({DT_FLOAT})},
            {"Toutputs", gtl::ArraySlice<DataType>({})},
            {"key", "host_compute_channel_F1_O1"},
            {"shape_inference_graph", ""},
            {"shapes", gtl::ArraySlice<TensorShapeProto>({})},
            {"_outside_compilation_subgraph", "O1"}}},
      },
      {{"f_0_retval", "F:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    Node* key_constant =
        KeyPlaceholder("F1", b2.opts().WithName("F1_key_placeholder"));
    Node* recv1 = RecvAtHost(ops::NodeOut(key_constant, 0), "F1", "O1",
                             {DT_FLOAT}, b2.opts());
    Node* e = Unary(recv1, b2.opts()
                               .WithName("E")
                               .WithAttr("_encapsulate", "F1")
                               .WithAttr("_outside", "O1"));
    Node* s1 = Sequencer(
        b2.opts().WithName("F1_sequencer").WithControlInput(recv1), "F1");
    NodeBuilder node_builder1("F1", "F1", lib_def.get());
    node_builder1.Input(a).Input(b);
    Node* call1 =
        b2.opts().WithControlInput(s1).FinalizeBuilder(&node_builder1);

    Binary(e, call1, b2.opts().WithName("G"));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with one outside_compilation cluster that has no data outputs but has a
// control output to the compiled subgraph.
TEST(EncapsulateSubgraphsTest, OutsideCompilationControlOutput) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithAttr("_encapsulate", "F1"));
    Node* e = Unary(d, b1.opts()
                           .WithName("E")
                           .WithAttr("_encapsulate", "F1")
                           .WithAttr("_outside", "O1"));
    Node* f = Unary(d, b1.opts().WithName("F").WithControlInput(e).WithAttr(
                           "_encapsulate", "F1"));
    Binary(e, f, b1.opts().WithName("G"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"f_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"D"}, "BinaryTest", {"b_0_arg", "C:o:0"}},
          {{"F"},
           "UnaryTest",
           {"D:o:0"},
           {},
           {"outside_compilation_O1_host_compute"}},
          {{"outside_compilation_O1_host_compute"},
           "XlaHostCompute",
           {"D:o:0"},
           {{"Tinputs", gtl::ArraySlice<DataType>({DT_FLOAT})},
            {"Toutputs", gtl::ArraySlice<DataType>({})},
            {"key", "host_compute_channel_F1_O1"},
            {"shape_inference_graph", ""},
            {"shapes", gtl::ArraySlice<TensorShapeProto>({})},
            {"_outside_compilation_subgraph", "O1"}}},
      },
      {{"f_0_retval", "F:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    Node* key_constant =
        KeyPlaceholder("F1", b2.opts().WithName("F1_key_placeholder"));
    Node* recv1 = RecvAtHost(ops::NodeOut(key_constant, 0), "F1", "O1",
                             {DT_FLOAT}, b2.opts());
    Node* e = Unary(recv1, b2.opts()
                               .WithName("E")
                               .WithAttr("_encapsulate", "F1")
                               .WithAttr("_outside", "O1"));
    Node* send1 = SendFromHost(ops::NodeOut(key_constant, 0), "F1", "O1", {},
                               b2.opts().WithControlInput(e));
    Node* s1 = Sequencer(
        b2.opts().WithName("F1_sequencer").WithControlInputs({recv1, send1}),
        "F1");
    NodeBuilder node_builder1("F1", "F1", lib_def.get());
    node_builder1.Input(a).Input(b);
    Node* call1 =
        b2.opts().WithControlInput(s1).FinalizeBuilder(&node_builder1);

    Binary(e, call1, b2.opts().WithName("G"));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test with one outside_compilation cluster that has no outputs from the
// compiled subgraph.
TEST(EncapsulateSubgraphsTest, OutsideCompilationNoInputsOrOutputs) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = Input(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    Node* c = Unary(a, b1.opts().WithName("C").WithAttr("_encapsulate", "F1"));
    Node* d =
        Binary(b, c, b1.opts().WithName("D").WithAttr("_encapsulate", "F1"));
    Node* e = Unary(a, b1.opts()
                           .WithName("E")
                           .WithAttr("_encapsulate", "F1")
                           .WithAttr("_outside", "O1"));
    Node* f = Unary(d, b1.opts().WithName("F").WithAttr("_encapsulate", "F1"));
    Binary(e, f, b1.opts().WithName("G"));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"a_0_arg:float", "b_0_arg:float"}, {"f_0_retval:float"}, {},
      {
          {{"C"}, "UnaryTest", {"a_0_arg"}},
          {{"D"}, "BinaryTest", {"b_0_arg", "C:o:0"}},
          {{"F"}, "UnaryTest", {"D:o:0"}},
      },
      {{"f_0_retval", "F:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = Input(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));

    Node* e = Unary(a, b2.opts()
                           .WithName("E")
                           .WithAttr("_encapsulate", "F1")
                           .WithAttr("_outside", "O1"));
    NodeBuilder node_builder1("F1", "F1", lib_def.get());
    node_builder1.Input(a).Input(b);
    Node* call1 = b2.opts().FinalizeBuilder(&node_builder1);

    Binary(e, call1, b2.opts().WithName("G"));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

// Test for shape inference of outside compilation.
TEST(EncapsulateSubgraphsTest, OutsideCompilationShapeInference) {
  FunctionDefLibrary library;
  GraphDef graphdef;

  {
    *library.add_function() = test::function::XTimesTwo();

    GraphDefBuilder b1(GraphDefBuilder::kFailImmediately);
    Node* a = InputShaped(b1.opts().WithName("A"));
    Node* b = Input(b1.opts().WithName("B"));
    // Give nodes 'c' and 'd' names that collide after lowercasing.
    Node* c = Unary(a, b1.opts().WithName("C"));
    Node* d = Unary(b, b1.opts().WithName("c").WithControlInput(c).WithAttr(
                           "_encapsulate", "F1"));
    Node* e = BinaryUnknownShape(c, d,
                                 b1.opts()
                                     .WithName("E")
                                     .WithControlInputs({b, d})
                                     .WithAttr("_encapsulate", "F1")
                                     .WithAttr("_outside", "O1"));
    Node* f = Binary(c, e,
                     b1.opts().WithName("F").WithControlInput(e).WithAttr(
                         "_encapsulate", "F1"));
    Binary(a, f, b1.opts().WithName("G").WithControlInput(e));
    TF_EXPECT_OK(b1.ToGraphDef(&graphdef));
  }

  TF_EXPECT_OK(Encapsulate(&graphdef, &library));

  FunctionDefLibrary library_expected;
  GraphDef graphdef_expected;

  {
    GraphDefBuilder shape(GraphDefBuilder::kFailImmediately);
    Node* key_constant =
        KeyPlaceholderShape(shape.opts().WithName("KnownShape/_0"));
    Node* known = KnownShape({2}, shape.opts().WithName("KnownShape/_1"));
    Node* recv = RecvAtHost(ops::NodeOut(key_constant, 0), "F1", "O1",
                            {DT_FLOAT}, shape.opts());
    Node* e = BinaryUnknownShape(known, recv,
                                 shape.opts()
                                     .WithName("E")
                                     .WithAttr("_encapsulate", "F1")
                                     .WithAttr("_outside", "O1"));
    SendFromHost(ops::NodeOut(key_constant, 0), "F1", "O1", {e}, shape.opts());
    TF_EXPECT_OK(
        AddGraphDefToFunctionLibrary(shape, "F1_O1", &library_expected));
  }

  *library_expected.add_function() = test::function::XTimesTwo();
  *library_expected.add_function() = FunctionDefHelper::Create(
      "F1", {"b_0_arg:float", "c_0_arg:float"}, {"f_0_retval:float"}, {},
      {
          {{"c"}, "UnaryTest", {"b_0_arg"}, {}, {}},
          {{"F"},
           "BinaryTest",
           {"c_0_arg", "outside_compilation_O1_host_compute:outputs:0"},
           {},
           {"outside_compilation_O1_host_compute"}},
          {{"outside_compilation_O1_host_compute"},
           "XlaHostCompute",
           {"c:o:0"},
           {{"Tinputs", gtl::ArraySlice<DataType>({DT_FLOAT})},
            {"Toutputs", gtl::ArraySlice<DataType>({DT_FLOAT})},
            {"key", "host_compute_channel_F1_O1"},
            {"shape_inference_graph",
             "_outside_compilation_shape_inference_F1_O1"},
            {"shapes", gtl::ArraySlice<DataType>({})},
            {"_outside_compilation_subgraph", "O1"}},
           {"c"}},
      },
      {{"f_0_retval", "F:o:0"}});

  {
    std::unique_ptr<FunctionLibraryDefinition> lib_def(
        new FunctionLibraryDefinition(OpRegistry::Global(), library_expected));
    GraphDefBuilder b2(GraphDefBuilder::kFailImmediately, lib_def.get());
    Node* a = InputShaped(b2.opts().WithName("A"));
    Node* b = Input(b2.opts().WithName("B"));
    Node* c = Unary(a, b2.opts().WithName("C"));

    Node* key_constant =
        KeyPlaceholder("F1", b2.opts().WithName("F1_key_placeholder"));
    Node* recv = RecvAtHost(ops::NodeOut(key_constant, 0), "F1", "O1",
                            {DT_FLOAT}, b2.opts());
    Node* e = BinaryUnknownShape(c, ops::NodeOut(recv, 0),
                                 b2.opts()
                                     .WithName("E")
                                     .WithControlInputs({recv, b})
                                     .WithAttr("_encapsulate", "F1")
                                     .WithAttr("_outside", "O1"));
    Node* send = SendFromHost(ops::NodeOut(key_constant, 0), "F1", "O1", {e},
                              b2.opts().WithControlInput(e));

    Node* s = Sequencer(
        b2.opts().WithName("F1_sequencer").WithControlInputs({recv, send}),
        "F1");

    NodeBuilder node_builder("F1", "F1", lib_def.get());
    node_builder.Input(b).Input(c);
    Node* call =
        b2.opts().WithControlInputs({s, c}).FinalizeBuilder(&node_builder);

    Binary(a, call, b2.opts().WithName("G").WithControlInputs({e}));
    TF_EXPECT_OK(b2.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
  TF_EXPECT_FUNCTIONDEFLIBRARY_EQ(library_expected, library);
}

}  // namespace
}  // namespace tensorflow

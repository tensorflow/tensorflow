/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/subgraph.h"

#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

// TODO(joshl): Test setting the "device" field of a NodeDef.
// TODO(joshl): Test that feeding won't prune targets.

namespace tensorflow {
namespace {

class SubgraphTest : public ::testing::Test {
 protected:
  SubgraphTest() : g_(new Graph(OpRegistry::Global())) {
    device_info_.set_name("/job:a/replica:0/task:0/cpu:0");
    device_info_.set_device_type(DeviceType(DEVICE_CPU).type());
    device_info_.set_incarnation(0);
  }

  ~SubgraphTest() override {}

  void ExpectOK(const string& gdef_ascii) {
    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &gdef_));
    GraphConstructorOptions opts;
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, gdef_, g_.get()));
  }

  Node* FindNode(const string& name) {
    for (Node* n : g_->nodes()) {
      if (n->name() == name) return n;
    }
    return nullptr;
  }

  bool HasNode(const string& name) { return FindNode(name) != nullptr; }

  void ExpectNodes(const string& nodes) {
    int count = 0;
    std::vector<string> actual_nodes;
    for (Node* n : g_->nodes()) {
      if (n->IsOp()) {
        count++;
        actual_nodes.push_back(n->name());
      }
    }
    std::sort(actual_nodes.begin(), actual_nodes.end());

    LOG(INFO) << "Nodes present: " << absl::StrJoin(actual_nodes, " ");

    std::vector<string> expected_nodes = str_util::Split(nodes, ',');
    std::sort(expected_nodes.begin(), expected_nodes.end());
    for (const string& s : expected_nodes) {
      Node* n = FindNode(s);
      EXPECT_TRUE(n != nullptr) << s;
      if (n->type_string() == "_Send" || n->type_string() == "_Recv") {
        EXPECT_EQ(device_info_.name(), n->assigned_device_name()) << s;
      }
    }

    EXPECT_TRUE(actual_nodes.size() == expected_nodes.size())
        << "\nActual:   " << absl::StrJoin(actual_nodes, ",")
        << "\nExpected: " << absl::StrJoin(expected_nodes, ",");
  }

  bool HasEdge(const string& src, int src_out, const string& dst, int dst_in) {
    for (const Edge* e : g_->edges()) {
      if (e->src()->name() == src && e->src_output() == src_out &&
          e->dst()->name() == dst && e->dst_input() == dst_in)
        return true;
    }
    return false;
  }
  bool HasControlEdge(const string& src, const string& dst) {
    return HasEdge(src, Graph::kControlSlot, dst, Graph::kControlSlot);
  }

  string Subgraph(const string& fed_str, const string& fetch_str,
                  const string& targets_str,
                  bool use_function_convention = false) {
    Graph* subgraph = new Graph(OpRegistry::Global());
    CopyGraph(*g_, subgraph);
    std::vector<string> fed =
        str_util::Split(fed_str, ',', str_util::SkipEmpty());
    std::vector<string> fetch =
        str_util::Split(fetch_str, ',', str_util::SkipEmpty());
    std::vector<string> targets =
        str_util::Split(targets_str, ',', str_util::SkipEmpty());

    subgraph::RewriteGraphMetadata metadata;
    Status s = subgraph::RewriteGraphForExecution(
        subgraph, fed, fetch, targets, device_info_, use_function_convention,
        &metadata);
    if (!s.ok()) {
      delete subgraph;
      return s.ToString();
    }

    EXPECT_EQ(fed.size(), metadata.feed_types.size());
    EXPECT_EQ(fetch.size(), metadata.fetch_types.size());

    // Replace the graph with the subgraph for the rest of the display program
    g_.reset(subgraph);
    return "OK";
  }

  Graph* graph() { return g_.get(); }

 private:
  GraphDef gdef_;
  std::unique_ptr<Graph> g_;
  DeviceAttributes device_info_;
};

REGISTER_OP("TestParams").Output("o: float");
REGISTER_OP("TestInput").Output("a: float").Output("b: float");
REGISTER_OP("TestRelu").Input("i: float").Output("o: float");
REGISTER_OP("TestMul").Input("a: float").Input("b: float").Output("o: float");

TEST_F(SubgraphTest, Targets1) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK", Subgraph("", "", "t1"));
  ExpectNodes("W1,input,t1");
}

TEST_F(SubgraphTest, Targets2) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: 'W1' input: 'input:1' }"
      "node { name: 't2' op: 'TestMul' input: 'W2' input: 't1' }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK", Subgraph("", "", "t2,t3_a"));
  ExpectNodes("W1,W2,input,t1,t2,t3_a");
}

TEST_F(SubgraphTest, FedOutputs1) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK", Subgraph("input:1", "", "t2"));
  ExpectNodes("W1,W2,_recv_input_1,t1,t2");
}

TEST_F(SubgraphTest, FedOutputs1_FunctionConvention) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK",
            Subgraph("input:1", "", "t2", true /* use_function_convention */));
  ExpectNodes("W1,W2,_arg_input_1_0,t1,t2");
}

TEST_F(SubgraphTest, FedRefNode) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W2', 'W1' ] }");
  EXPECT_EQ("OK", Subgraph("W1:0", "", "t1"));
  ExpectNodes("_recv_W1_0,W2,t1");
  Node* n = FindNode("_recv_W1_0");
  EXPECT_FALSE(IsRefType(CHECK_NOTNULL(n)->output_type(0)));
}

TEST_F(SubgraphTest, FedRefNode_FunctionConvention) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W2', 'W1' ] }");
  EXPECT_EQ("OK",
            Subgraph("W1:0", "", "t1", true /* use_function_convention */));
  ExpectNodes("_arg_W1_0_0,W2,t1");
  Node* n = FindNode("_arg_W1_0_0");
  EXPECT_FALSE(IsRefType(CHECK_NOTNULL(n)->output_type(0)));
}

TEST_F(SubgraphTest, FedOutputs2_FunctionConvention) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  // We feed input:1, but nothing connects to it, so the _recv(input:1)
  // node also disappears.
  EXPECT_EQ("OK", Subgraph("input:1,t1,W2", "", "t2",
                           true /* use_function_convention */));
  ExpectNodes("_arg_t1_0_1,_arg_W2_0_2,t2");
}

TEST_F(SubgraphTest, FetchOutputs1) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK", Subgraph("", "W2,input:1,t1,t2", "t2"));
  ExpectNodes(
      "W1,W2,input,t1,t2,_send_W2_0,_send_input_1,_send_t1_0,_send_t2_0");
}

TEST_F(SubgraphTest, FetchOutputs1_FunctionConvention) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK", Subgraph("", "W2,input:1,t1,t2", "t2",
                           true /* use_function_convention */));
  ExpectNodes(
      "W1,W2,input,t1,t2,_retval_W2_0_0,_retval_input_1_1,_retval_t1_0_2,_"
      "retval_t2_0_3");
}

TEST_F(SubgraphTest, FetchOutputs2) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK", Subgraph("", "t3_a", "t2"));
  ExpectNodes("W1,W2,input,t1,t2,t3_a,_send_t3_a_0");
}

TEST_F(SubgraphTest, FetchOutputs2_FunctionConvention) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK",
            Subgraph("", "t3_a", "t2", true /* use_function_convention */));
  ExpectNodes("W1,W2,input,t1,t2,t3_a,_retval_t3_a_0_0");
}

TEST_F(SubgraphTest, ChainOfFools) {
  ExpectOK(
      "node { name: 'a' op: 'TestParams' }"
      "node { name: 'b' op: 'TestRelu' input: 'a'}"
      "node { name: 'c' op: 'TestRelu' input: 'b'}"
      "node { name: 'd' op: 'TestRelu' input: 'c'}"
      "node { name: 'e' op: 'TestRelu' input: 'd'}"
      "node { name: 'f' op: 'TestRelu' input: 'e'}");
  EXPECT_EQ("OK", Subgraph("c:0", "b:0,e:0", ""));
  ExpectNodes("a,b,_send_b_0,_recv_c_0,d,e,_send_e_0");
  EXPECT_TRUE(HasEdge("a", 0, "b", 0));
  EXPECT_TRUE(HasEdge("b", 0, "_send_b_0", 0));
  EXPECT_TRUE(HasEdge("_recv_c_0", 0, "d", 0));
  EXPECT_TRUE(HasEdge("d", 0, "e", 0));
  EXPECT_TRUE(HasEdge("e", 0, "_send_e_0", 0));
}

static bool HasSubstr(StringPiece base, StringPiece substr) {
  bool ok = absl::StrContains(base, substr);
  EXPECT_TRUE(ok) << base << ", expected substring " << substr;
  return ok;
}

TEST_F(SubgraphTest, Errors) {
  ExpectOK(
      "node { name: 'a' op: 'TestParams' }"
      "node { name: 'b' op: 'TestRelu' input: 'a'}"
      "node { name: 'c' op: 'TestRelu' input: 'b'}"
      "node { name: 'd' op: 'TestRelu' input: 'c'}"
      "node { name: 'e' op: 'TestRelu' input: 'd'}"
      "node { name: 'f' op: 'TestRelu' input: 'e'}");
  // Duplicated feed and fetch
  EXPECT_TRUE(
      HasSubstr(Subgraph("c:0", "b:0,c:0", ""), "both fed and fetched"));
  // Feed not found.
  EXPECT_TRUE(HasSubstr(Subgraph("foo:0", "c:0", ""), "unable to find"));
  // Fetch not found.
  EXPECT_TRUE(HasSubstr(Subgraph("", "foo:0", ""), "not found"));
  // Target not found.
  EXPECT_TRUE(HasSubstr(Subgraph("", "", "foo"), "not found"));
  // No targets specified.
  EXPECT_TRUE(HasSubstr(Subgraph("", "", ""), "at least one target"));
}

REGISTER_OP("In").Output("o: float");
REGISTER_OP("Op").Input("i: float").Output("o: float");

void BM_SubgraphHelper(::testing::benchmark::State& state,
                       bool use_function_convention) {
  const int num_nodes = state.range(0);
  DeviceAttributes device_info;
  device_info.set_name("/job:a/replica:0/task:0/cpu:0");
  device_info.set_device_type(DeviceType(DEVICE_CPU).type());
  device_info.set_incarnation(0);

  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* last_node = nullptr;
    for (int i = 0; i < num_nodes; i++) {
      string name = strings::StrCat("N", i);
      if (i > 0) {
        last_node = ops::UnaryOp("Op", last_node, b.opts().WithName(name));
      } else {
        last_node = ops::SourceOp("In", b.opts().WithName(name));
      }
    }
    TF_CHECK_OK(GraphDefBuilderToGraph(b, &g));
  }

  std::vector<string> fed;
  if (num_nodes > 1000) {
    fed.push_back(strings::StrCat("N", num_nodes - 1000));
  }
  std::vector<string> fetch;
  std::vector<string> targets = {strings::StrCat("N", num_nodes - 1)};

  for (auto s : state) {
    Graph* subgraph = new Graph(OpRegistry::Global());
    CopyGraph(g, subgraph);
    subgraph::RewriteGraphMetadata metadata;
    TF_CHECK_OK(subgraph::RewriteGraphForExecution(
        subgraph, fed, fetch, targets, device_info, use_function_convention,
        &metadata));
    delete subgraph;
  }
}

void BM_Subgraph(::testing::benchmark::State& state) {
  BM_SubgraphHelper(state, false /* use_function_convention */);
}
void BM_SubgraphFunctionConvention(::testing::benchmark::State& state) {
  BM_SubgraphHelper(state, true /* use_function_convention */);
}
BENCHMARK(BM_Subgraph)->Arg(100)->Arg(1000)->Arg(10000)->Arg(100000);
BENCHMARK(BM_SubgraphFunctionConvention)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

}  // namespace
}  // namespace tensorflow

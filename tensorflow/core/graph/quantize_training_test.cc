/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/graph/quantize_training.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

class QuantizeTrainingTest : public ::testing::Test {
 protected:
  QuantizeTrainingTest() { Reset(); }
  void Reset() { g_.reset(new Graph(OpRegistry::Global())); }

  template <typename T>
  Node* Constant(gtl::ArraySlice<T> values, TensorShape shape) {
    return test::graph::Constant(g_.get(), test::AsTensor(values, shape));
  }

  std::unique_ptr<Graph> g_;
};

TEST_F(QuantizeTrainingTest, NormalGraph) {
  // Construct the following graph
  /*
           m1      m2
        /      \ /     \
      Relu   Identity   c
        |       |
        a       b
  */
  Reset();
  Graph* g = g_.get();
  Node* a = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* b = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* c = Constant<float>({0.0, 1.0, 1.0, 0.0}, {2, 2});
  g->AddControlEdge(g->source_node(), a);
  g->AddControlEdge(g->source_node(), b);
  g->AddControlEdge(g->source_node(), c);
  Node* relu = test::graph::Relu(g, a);
  Node* identity = test::graph::Identity(g, b);
  Node* m1 = test::graph::Matmul(g, relu, identity, false, false);
  Node* m2 = test::graph::Matmul(g, identity, c, false, false);
  g->AddControlEdge(m1, g->sink_node());
  g->AddControlEdge(m2, g->sink_node());

  // The graph after the rewriting should be:
  // "Q" is the quantize_and_dequantize op.
  // Note the Q in the middle is shared by both m1 and m2.
  /*
         m1       m2
      /      \ /     \
      Q       Q       Q
      |       |       |
    Relu   Identity   c
      |       |
      a       b
  */
  int num_bits = 8;
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, g));

  // There should be 12 nodes in total including the source and sink nodes.
  EXPECT_EQ(12, g->num_nodes());
  // Nodes m1 and m2's inputs should be the quantize_and_dequantize op.
  std::vector<Node*> target_nodes{m1, m2};
  for (Node* n : target_nodes) {
    for (Node* in : n->in_nodes()) {
      EXPECT_EQ("QuantizeAndDequantize", in->type_string());
    }
  }

  // relu, identity, c should now connect to the quantize_and_dequantize nodes.
  std::vector<Node*> target_inputs{relu, identity, c};
  for (Node* n : target_inputs) {
    for (Node* out : n->out_nodes()) {
      EXPECT_EQ("QuantizeAndDequantize", out->type_string());
    }
  }

  // Quantize_and_dequantize node for identity should have signed_input==true.
  NodeDef identity_Q = identity->out_nodes().begin()->def();
  ASSERT_EQ("true",
            SummarizeAttrValue(identity_Q.attr().find("signed_input")->second));
  // Quantize_and_dequantize node for relu should have signed_input==false.
  NodeDef relu_Q = relu->out_nodes().begin()->def();
  ASSERT_EQ("false",
            SummarizeAttrValue(relu_Q.attr().find("signed_input")->second));
}

TEST_F(QuantizeTrainingTest, WithBackwardNodes) {
  // Construct the same graph plus another backward Matmul.
  Reset();
  Graph* g = g_.get();
  Node* a = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* b = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* c = Constant<float>({0.0, 1.0, 1.0, 0.0}, {2, 2});
  g->AddControlEdge(g->source_node(), a);
  g->AddControlEdge(g->source_node(), b);
  g->AddControlEdge(g->source_node(), c);
  Node* relu = test::graph::Relu(g, a);
  Node* identity = test::graph::Identity(g, b);
  Node* m1 = test::graph::Matmul(g, relu, identity, false, false);
  Node* m2 = test::graph::Matmul(g, identity, c, false, false);
  g->AddControlEdge(m1, g->sink_node());
  g->AddControlEdge(m2, g->sink_node());

  // Add a Matmul node with name starting with "gradients".
  Node* backward_m;
  TF_ASSERT_OK(NodeBuilder(g->NewName("gradients/n"), "MatMul")
                   .Input(m1)
                   .Input(m2)
                   .Attr("transpose_a", true)
                   .Attr("transpose_b", false)
                   .Finalize(g, &backward_m));
  g->AddControlEdge(backward_m, g->sink_node());

  int num_bits = 8;
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, g));

  // Nodes m1 and m2's inputs should now be the quantize_and_dequantize op.
  EXPECT_EQ(13, g->num_nodes());
  EXPECT_EQ(2, m2->num_inputs());
}

TEST_F(QuantizeTrainingTest, QuantizeGraphDef) {
  // Construct a simple graph with 5 nodes.
  Reset();
  Graph* graph = g_.get();
  Node* const_a = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* const_b = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  graph->AddControlEdge(graph->source_node(), const_a);
  graph->AddControlEdge(graph->source_node(), const_b);
  Node* relu = test::graph::Relu(graph, const_a);
  Node* identity = test::graph::Identity(graph, const_b);
  Node* matmul = test::graph::Matmul(graph, relu, identity, false, false);
  graph->AddControlEdge(matmul, graph->sink_node());

  int num_bits = 8;

  // Convert the graph to the graphdef string.
  GraphDef input_graph;
  graph->ToGraphDef(&input_graph);
  string input_string;
  input_graph.SerializeToString(&input_string);

  string result_string;
  TF_ASSERT_OK(DoQuantizeTrainingOnSerializedGraphDef(input_string, num_bits,
                                                      &result_string));

  GraphDef result_graph;
  EXPECT_TRUE(ParseProtoUnlimited(&result_graph, result_string));

  // Nodes m1's inputs should now be converted with 2 added ops, which results
  // in the total of 7 nodes.
  EXPECT_EQ(7, result_graph.node_size());
}

}  // namespace
}  // namespace tensorflow

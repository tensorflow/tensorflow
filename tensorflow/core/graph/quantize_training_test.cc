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
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
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

  Status Placeholder(Graph* g, const string& name, TensorShape shape,
                     Node** out) {
    TF_RETURN_IF_ERROR(NodeBuilder(name, "Placeholder")
                           .Attr("dtype", DT_FLOAT)
                           .Attr("shape", shape)
                           .Finalize(g, out));
    return Status::OK();
  }

  Status FindNode(Graph* g, const string& name, Node** out) {
    for (Node* node : g->nodes()) {
      if (node->name() == name) {
        *out = node;
        return Status::OK();
      }
    }
    return errors::Unimplemented("Node ", name, " not found.");
  }

  std::unique_ptr<Graph> g_;
};

TEST_F(QuantizeTrainingTest, SignedInput) {
  // Test that Quantization ops are created with the correct signed_input value.
  // Construct the following graph
  /*
           m1
        /      \
      Relu   Identity
        |       |
        a       b
  */
  Reset();
  Graph* g = g_.get();
  Node* a = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* b = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  g->AddControlEdge(g->source_node(), a);
  g->AddControlEdge(g->source_node(), b);
  Node* relu = test::graph::Relu(g, a);
  Node* identity = test::graph::Identity(g, b);
  Node* m1 = test::graph::Matmul(g, relu, identity, false, false);
  g->AddControlEdge(m1, g->sink_node());

  /*
         m1
      /      \
    EMA_Q   EMA_Q  <- these are subgraphs that estimate moving average.
      |       |
    Relu   Identity
      |       |
      a       b
  */
  const int num_bits = 8;
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "QuantizeAndDequantizeV2", g));

  EXPECT_EQ(63, g->num_nodes());

  // Quantize_and_dequantize node for identity should have signed_input==true.
  Node* identity_q_node;
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(identity->name(), "/QuantizeAndDequantizeV2"),
               &identity_q_node));
  ASSERT_EQ("true",
            SummarizeAttrValue(*identity_q_node->attrs().Find("signed_input")));
  // Quantize_and_dequantize node for relu should have signed_input==false.
  Node* relu_q_node;
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(relu->name(), "/QuantizeAndDequantizeV2"),
               &relu_q_node));
  ASSERT_EQ("false",
            SummarizeAttrValue(*relu_q_node->attrs().Find("signed_input")));
}

TEST_F(QuantizeTrainingTest, RangeGivenTrue) {
  // Test that Quantization ops are created with the correct range_given value.
  // Construct the following graph
  /*
           m1
        /      \
      Relu   Relu6
        |       |
        a       b
  */
  Reset();
  Graph* g = g_.get();
  Node* a = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* b = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  g->AddControlEdge(g->source_node(), a);
  g->AddControlEdge(g->source_node(), b);
  Node* relu = test::graph::Relu(g, a);
  Node* relu6 = test::graph::Relu6(g, b);
  Node* m1 = test::graph::Matmul(g, relu, relu6, false, false);
  g->AddControlEdge(m1, g->sink_node());

  /*
         m1
      /      \
    EMA_Q     Q
      |       |
    Relu   Relu6
      |       |
      a       b
  */
  const int num_bits = 8;
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "QuantizeAndDequantizeV2", g));

  EXPECT_EQ(38, g->num_nodes());

  // Quantize_and_dequantize node for relu6 should have range_given==true.
  Node* relu6_q_node;
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(relu6->name(), "/QuantizeAndDequantizeV2"),
               &relu6_q_node));
  ASSERT_EQ("true",
            SummarizeAttrValue(*relu6_q_node->attrs().Find("range_given")));
  // Quantize_and_dequantize node for relu should have range_given==true.
  Node* relu_q_node;
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(relu->name(), "/QuantizeAndDequantizeV2"),
               &relu_q_node));
  ASSERT_EQ("true",
            SummarizeAttrValue(*relu_q_node->attrs().Find("range_given")));
}

TEST_F(QuantizeTrainingTest, WithBackwardNodes_QuantizeAndDequantize) {
  // Construct a graph with an additional backward Matmul.
  Reset();
  Graph* g = g_.get();
  Node* a = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* b = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* c = Constant<float>({0.0, 1.0, 1.0, 0.0}, {2, 2});
  // We will use node d as input to the backwards matmul to ensure that it
  // isn't quantized.
  Node* d = Constant<float>({0.0, 1.0, 1.0, 0.0}, {2, 2});
  g->AddControlEdge(g->source_node(), a);
  g->AddControlEdge(g->source_node(), b);
  g->AddControlEdge(g->source_node(), c);
  g->AddControlEdge(g->source_node(), d);
  Node* relu = test::graph::Relu(g, a);
  Node* identity = test::graph::Identity(g, b);
  Node* m1 = test::graph::Matmul(g, relu, identity, false, false);
  Node* m2 = test::graph::Matmul(g, identity, c, false, false);
  g->AddControlEdge(m1, g->sink_node());
  g->AddControlEdge(m2, g->sink_node());

  // Add a Matmul node with name starting with "gradients". We will check that
  // its input d was not quantized.
  Node* backward_m;
  TF_ASSERT_OK(NodeBuilder(g->NewName("gradients/n"), "MatMul")
                   .Input(d)
                   .Input(m2)
                   .Attr("transpose_a", true)
                   .Attr("transpose_b", false)
                   .Finalize(g, &backward_m));
  g->AddControlEdge(backward_m, g->sink_node());

  int num_bits = 8;
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "QuantizeAndDequantizeV2", g));

  EXPECT_EQ(95, g->num_nodes());

  // Ensure that the backwards matmul input was not quantized.
  Node* found_node;
  Status s = FindNode(g, strings::StrCat(d->name(), "/QuantizeAndDequantizeV2"),
                      &found_node);
  EXPECT_TRUE(StringPiece(s.ToString()).contains("not found")) << s;

  // Ensure that m1 and m2's inputs were quantized.
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(relu->name(), "/QuantizeAndDequantizeV2"),
               &found_node));
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(identity->name(), "/QuantizeAndDequantizeV2"),
               &found_node));
  TF_ASSERT_OK(FindNode(
      g, strings::StrCat(c->name(), "/QuantizeAndDequantizeV2"), &found_node));
}

TEST_F(QuantizeTrainingTest, WithBackwardNodes_FakeQuant) {
  // Construct a graph with an additional backward Matmul.
  Reset();
  Graph* g = g_.get();
  Node* a = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* b = Constant<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
  Node* c = Constant<float>({0.0, 1.0, 1.0, 0.0}, {2, 2});
  // We will use node d as input to the backwards matmul to ensure that it
  // isn't quantized.
  Node* d = Constant<float>({0.0, 1.0, 1.0, 0.0}, {2, 2});
  g->AddControlEdge(g->source_node(), a);
  g->AddControlEdge(g->source_node(), b);
  g->AddControlEdge(g->source_node(), c);
  g->AddControlEdge(g->source_node(), d);
  Node* relu = test::graph::Relu(g, a);
  Node* identity = test::graph::Identity(g, b);
  Node* m1 = test::graph::Matmul(g, relu, identity, false, false);
  Node* m2 = test::graph::Matmul(g, identity, c, false, false);
  g->AddControlEdge(m1, g->sink_node());
  g->AddControlEdge(m2, g->sink_node());

  // Add a Matmul node with name starting with "gradients". We will check that
  // its input d was not quantized.
  Node* backward_m;
  TF_ASSERT_OK(NodeBuilder(g->NewName("gradients/n"), "MatMul")
                   .Input(d)
                   .Input(m2)
                   .Attr("transpose_a", true)
                   .Attr("transpose_b", false)
                   .Finalize(g, &backward_m));
  g->AddControlEdge(backward_m, g->sink_node());

  int num_bits = 8;
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "FakeQuantWithMinMaxVars", g));

  EXPECT_EQ(95, g->num_nodes());

  // Ensure that the backwards matmul input was not quantized.
  Node* found_node;
  Status s = FindNode(g, strings::StrCat(d->name(), "/FakeQuantWithMinMaxVars"),
                      &found_node);
  EXPECT_TRUE(StringPiece(s.ToString()).contains("not found")) << s;

  // Ensure that m1 and m2's inputs were quantized.
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(relu->name(), "/FakeQuantWithMinMaxVars"),
               &found_node));
  TF_ASSERT_OK(
      FindNode(g, strings::StrCat(identity->name(), "/FakeQuantWithMinMaxVars"),
               &found_node));
  TF_ASSERT_OK(FindNode(
      g, strings::StrCat(c->name(), "/FakeQuantWithMinMaxVars"), &found_node));
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
  TF_ASSERT_OK(DoQuantizeTrainingOnSerializedGraphDef(
      input_string, num_bits, "QuantizeAndDequantizeV2", &result_string));

  GraphDef result_graphdef;
  EXPECT_TRUE(ParseProtoUnlimited(&result_graphdef, result_string));

  // Ensure that quantizing the graph_def results in a graph with the same
  // number of nodes.
  GraphConstructorOptions opts;
  Graph result_graph(OpRegistry::Global());
  TF_ASSERT_OK(ConvertGraphDefToGraph(opts, result_graphdef, &result_graph));
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "QuantizeAndDequantizeV2", graph));
  EXPECT_EQ(graph->num_nodes(), result_graph.num_nodes());
}

TEST_F(QuantizeTrainingTest, FixedRangeAndEMARange_QuantizeAndDequantize) {
  // Construct the following graph
  // Relu has an unknown range, so we will check if the EMA correctly estimates
  // the range.
  /*
           m1
        /      \
      Relu    Relu6
        |       |
        a       c
  */
  Reset();
  Graph* g = g_.get();
  Node* a;
  TF_ASSERT_OK(Placeholder(g, "a", {2, 2}, &a));
  Node* c = Constant<float>({2.0, 3.0, 4.0, 5.0}, {2, 2});
  g->AddControlEdge(g->source_node(), a);
  g->AddControlEdge(g->source_node(), c);
  Node* relu = test::graph::Relu(g, a);
  Node* relu6 = test::graph::Relu6(g, c);
  Node* m1 = test::graph::Matmul(g, relu, relu6, false, false);
  g->AddControlEdge(m1, g->sink_node());

  // This is rewritten into the following subgraph, where Q_a and Q_c are
  // quantize and dequantize subgraphs.
  // Since relu's range is unknown, we check that the exponential moving average
  // works correctly.
  /*
         m1
      /      \
     Q_a     Q_c
      |       |
    Relu     Relu6
      |       |
      a       c
  */
  const int num_bits = 8;
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "QuantizeAndDequantizeV2", g));

  SessionOptions options;
  Session* sess;
  TF_ASSERT_OK(NewSession(options, &sess));
  GraphDef gdef;
  g->ToGraphDef(&gdef);
  TF_ASSERT_OK(sess->Create(gdef));

  // The min and max values of the relu6 quantization should be constant values
  // of 0 and 6.
  string min_const_name = strings::StrCat(relu6->name(), "/InputMin");
  string max_const_name = strings::StrCat(relu6->name(), "/InputMax");
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(sess->Run({}, {min_const_name, max_const_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 6.0);

  Tensor a1(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a1, {0.0, 1.0, 2.0, 3.0});
  Tensor a2(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a2, {1.0, 2.0, 3.0, 4.0});

  TF_ASSERT_OK(sess->Run({{"a", a1}}, {m1->name()}, {}, &outputs));

  // The value of the min and max should be set to the min and max of a1 since
  // this is the first run that initializes the EMA variables.
  string min_var_name = strings::StrCat(relu->name(), "/Min/Variable");
  string max_var_name = strings::StrCat(relu->name(), "/Max/Variable");
  TF_ASSERT_OK(sess->Run({}, {min_var_name, max_var_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 3.0);

  // The relu6 quantization range should remain unchanged.
  TF_ASSERT_OK(sess->Run({}, {min_const_name, max_const_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 6.0);

  // Now when we run with new inputs, we should get a moving average for the min
  // and max variables. They should be equal to:
  // min_var = old_min_var * decay + min(a2) * (1 - decay)
  // max_var = old_max_var * decay + max(a2) * (1 - decay)
  TF_ASSERT_OK(sess->Run({{"a", a2}}, {m1->name()}, {}, &outputs));

  TF_ASSERT_OK(sess->Run({}, {min_var_name, max_var_name}, {}, &outputs));
  const float decay = 0.999;
  const float expected_min = 0.0 * decay + 1.0 * (1.0 - decay);
  const float expected_max = 3.0 * decay + 4.0 * (1.0 - decay);
  EXPECT_NEAR(outputs[0].flat<float>()(0), expected_min, 1e-4);
  EXPECT_NEAR(outputs[1].flat<float>()(0), expected_max, 1e-4);

  // The relu6 quantization range should remain unchanged.
  TF_ASSERT_OK(sess->Run({}, {min_const_name, max_const_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 6.0);
}

TEST_F(QuantizeTrainingTest, FixedRangeAndEMARange_FakeQuant) {
  // Construct the following graph
  // Relu has an unknown range, so we will check if the EMA correctly estimates
  // the range.
  /*
           m1
        /      \
      Relu    Relu6
        |       |
        a       c
  */
  Reset();
  Graph* g = g_.get();
  Node* a;
  TF_ASSERT_OK(Placeholder(g, "a", {2, 2}, &a));
  Node* c = Constant<float>({2.0, 3.0, 4.0, 5.0}, {2, 2});
  g->AddControlEdge(g->source_node(), a);
  g->AddControlEdge(g->source_node(), c);
  Node* relu = test::graph::Relu(g, a);
  Node* relu6 = test::graph::Relu6(g, c);
  Node* m1 = test::graph::Matmul(g, relu, relu6, false, false);
  g->AddControlEdge(m1, g->sink_node());

  // This is rewritten into the following subgraph, where Q_a and Q_c are
  // quantize and dequantize subgraphs.
  // Since relu's range is unknown, we check that the exponential moving average
  // works correctly.
  /*
         m1
      /      \
     Q_a     Q_c
      |       |
    Relu     Relu6
      |       |
      a       c
  */
  const int num_bits = 8;
  TF_ASSERT_OK(DoQuantizeTraining(num_bits, "FakeQuantWithMinMaxVars", g));

  SessionOptions options;
  Session* sess;
  TF_ASSERT_OK(NewSession(options, &sess));
  GraphDef gdef;
  g->ToGraphDef(&gdef);
  TF_ASSERT_OK(sess->Create(gdef));

  // The min and max values of the relu6 quantization should be constant values
  // of 0 and 6.
  string min_const_name = strings::StrCat(relu6->name(), "/InputMin");
  string max_const_name = strings::StrCat(relu6->name(), "/InputMax");
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(sess->Run({}, {min_const_name, max_const_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 6.0);

  Tensor a1(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a1, {0.0, 1.0, 2.0, 3.0});
  Tensor a2(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a2, {1.0, 2.0, 3.0, 4.0});

  TF_ASSERT_OK(sess->Run({{"a", a1}}, {m1->name()}, {}, &outputs));

  // The value of the min and max should be set to the min and max of a1 since
  // this is the first run that initializes the EMA variables.
  string min_var_name = strings::StrCat(relu->name(), "/Min/Variable");
  string max_var_name = strings::StrCat(relu->name(), "/Max/Variable");
  TF_ASSERT_OK(sess->Run({}, {min_var_name, max_var_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 3.0);

  // The relu6 quantization range should remain unchanged.
  TF_ASSERT_OK(sess->Run({}, {min_const_name, max_const_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 6.0);

  // Now when we run with new inputs, we should get a moving average for the min
  // and max variables. They should be equal to:
  // min_var = old_min_var * decay + min(a2) * (1 - decay)
  // max_var = old_max_var * decay + max(a2) * (1 - decay)
  TF_ASSERT_OK(sess->Run({{"a", a2}}, {m1->name()}, {}, &outputs));

  TF_ASSERT_OK(sess->Run({}, {min_var_name, max_var_name}, {}, &outputs));
  const float decay = 0.999;
  const float expected_min = 0.0 * decay + 1.0 * (1.0 - decay);
  const float expected_max = 3.0 * decay + 4.0 * (1.0 - decay);
  EXPECT_NEAR(outputs[0].flat<float>()(0), expected_min, 1e-4);
  EXPECT_NEAR(outputs[1].flat<float>()(0), expected_max, 1e-4);

  // The relu6 quantization range should remain unchanged.
  TF_ASSERT_OK(sess->Run({}, {min_const_name, max_const_name}, {}, &outputs));
  EXPECT_EQ(outputs[0].flat<float>()(0), 0.0);
  EXPECT_EQ(outputs[1].flat<float>()(0), 6.0);
}

}  // namespace
}  // namespace tensorflow

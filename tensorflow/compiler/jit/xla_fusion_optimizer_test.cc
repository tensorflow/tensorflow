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

#include "tensorflow/compiler/jit/xla_fusion_optimizer.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/graph_def_builder_util.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

REGISTER_OP("UncompilableNullary").Output("o: float");
REGISTER_OP("UncompilableUnary").Input("a: float").Output("o: float");

class XlaFusionOptimizerTest : public grappler::GrapplerTest {
 protected:
  std::unordered_map<string, string> GetClusters(const GraphDef& graph) {
    std::unordered_map<string, string> ids;
    for (const NodeDef& node : graph.node()) {
      string cluster;
      if (GetNodeAttr(AttrSlice(node), kXlaClusterAttr, &cluster).ok()) {
        CHECK(!cluster.empty());
        ids[node.name()] = cluster;
      }
    }
    return ids;
  }
};

TEST_F(XlaFusionOptimizerTest, Chains) {
  GraphDef graph;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a =
        ops::SourceOp("UncompilableNullary", builder.opts().WithName("A"));
    Node* b = ops::UnaryOp("Relu", a, builder.opts().WithName("B"));
    Node* c = ops::UnaryOp("Relu", b, builder.opts().WithName("C"));
    Node* d =
        ops::UnaryOp("UncompilableUnary", c, builder.opts().WithName("D"));
    Node* e = ops::UnaryOp("Relu", d, builder.opts().WithName("E"));
    ops::UnaryOp("Relu", e, builder.opts().WithName("F"));
    TF_ASSERT_OK(builder.ToGraphDef(&graph));
  }
  grappler::GrapplerItem item;
  item.graph = graph;

  XlaFusionOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  auto clusters = GetClusters(output);
  EXPECT_EQ(4, clusters.size());
  EXPECT_EQ(clusters["B"], clusters["C"]);
  EXPECT_EQ(clusters["E"], clusters["F"]);
  EXPECT_NE(clusters["B"], clusters["E"]);
  EXPECT_TRUE(clusters.find("A") == clusters.cend());
  EXPECT_TRUE(clusters.find("D") == clusters.cend());
}

TEST_F(XlaFusionOptimizerTest, FusableOps) {
  GraphDef graph;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp(
        "Placeholder",
        builder.opts().WithName("A").WithAttr("dtype", tensorflow::DT_FLOAT));
    Node* b = ops::SourceOp(
        "Placeholder",
        builder.opts().WithName("B").WithAttr("dtype", tensorflow::DT_FLOAT));

    Node* c = ops::BinaryOp("Add", a, b, builder.opts().WithName("C"));
    ops::BinaryOp("MatMul", a, c, builder.opts().WithName("D"));
    ops::UnaryOp("Abs", c, builder.opts().WithName("E"));

    TF_ASSERT_OK(builder.ToGraphDef(&graph));
  }
  grappler::GrapplerItem item;
  item.graph = graph;

  XlaFusionOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  auto clusters = GetClusters(output);
  EXPECT_EQ(2, clusters.size());
  EXPECT_EQ(clusters["C"], clusters["E"]);
  EXPECT_TRUE(clusters.find("D") == clusters.cend());
}

TEST_F(XlaFusionOptimizerTest, IgnoreExplicitXLAAttrs) {
  GraphDef graph;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp(
        "Placeholder",
        builder.opts().WithName("A").WithAttr("dtype", tensorflow::DT_FLOAT));
    Node* b = ops::SourceOp(
        "Placeholder",
        builder.opts().WithName("B").WithAttr("dtype", tensorflow::DT_FLOAT));

    Node* c = ops::BinaryOp(
        "Add", a, b,
        builder.opts().WithName("C").WithDevice("/device:XLA_CPU"));
    ops::BinaryOp("MatMul", a, c, builder.opts().WithName("D"));
    Node* e = ops::UnaryOp("Abs", c, builder.opts().WithName("E"));
    ops::UnaryOp("Cos", e,
                 builder.opts().WithName("F").WithAttr(kXlaCompileAttr, true));

    TF_ASSERT_OK(builder.ToGraphDef(&graph));
  }
  grappler::GrapplerItem item;
  item.graph = graph;

  XlaFusionOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  auto clusters = GetClusters(output);
  EXPECT_TRUE(clusters.empty());
}

TEST_F(XlaFusionOptimizerTest, UncompilableCycles) {
  GraphDef graph;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* b =
        ops::UnaryOp("UncompilableUnary", a, builder.opts().WithName("B"));
    ops::BinaryOp("Mul", a, b, builder.opts().WithName("C"));

    TF_ASSERT_OK(builder.ToGraphDef(&graph));
  }
  grappler::GrapplerItem item;
  item.graph = graph;

  XlaFusionOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  auto clusters = GetClusters(output);
  EXPECT_TRUE(clusters.empty());
}

TEST_F(XlaFusionOptimizerTest, CompilableCycles) {
  GraphDef graph;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a = ops::SourceOp("Const", builder.opts()
                                         .WithName("A")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("value", Tensor()));
    Node* b = ops::UnaryOp("Relu", a, builder.opts().WithName("B"));
    ops::BinaryOp("Mul", a, b, builder.opts().WithName("C"));
    TF_ASSERT_OK(builder.ToGraphDef(&graph));
  }
  grappler::GrapplerItem item;
  item.graph = graph;

  XlaFusionOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  auto clusters = GetClusters(output);
  EXPECT_EQ(3, clusters.size());
  EXPECT_EQ(clusters["A"], clusters["B"]);
  EXPECT_EQ(clusters["A"], clusters["C"]);
}

}  // namespace
}  // namespace tensorflow

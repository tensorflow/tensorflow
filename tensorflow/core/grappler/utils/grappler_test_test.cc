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

#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

// TODO(ezhulenev): add tests for all methods in GrapplerTest
class GrapplerTestTest : public GrapplerTest {};

TEST_F(GrapplerTestTest, CompareIdenticalGraphs) {
  tensorflow::Scope s1 = tensorflow::Scope::NewRootScope();
  auto s1_a = ops::Variable(s1.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto s1_b = ops::Variable(s1.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto s1_add = ops::Add(s1.WithOpName("Add_1"), s1_a, s1_b);

  tensorflow::Scope s2 = tensorflow::Scope::NewRootScope();
  auto s2_a = ops::Variable(s2.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto s2_b = ops::Variable(s2.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto s2_add = ops::Add(s2.WithOpName("Add_1"), s2_a, s2_b);

  GraphDef graph1;
  TF_ASSERT_OK(s1.ToGraphDef(&graph1));

  GraphDef graph2;
  TF_ASSERT_OK(s2.ToGraphDef(&graph2));

  CompareGraphs(graph1, graph2);
}

TEST_F(GrapplerTestTest, CheckNodesConnectivity) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto a = ops::Variable(s.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto add_1 = ops::Add(s.WithOpName("Add_1"), a, b);
  auto add_2 = ops::Add(s.WithOpName("Add_2"), add_1, b);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));

  NodeMap node_map(&graph);

  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "a", "Add_1", 0));
  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "b", "Add_1", 1));
  EXPECT_FALSE(IsNodesDirectlyConnected(node_map, "a", "Add_2", 0));
  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "b", "Add_2", 1));
}

TEST_F(GrapplerTestTest, CountOpNodes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto a = ops::Variable(s.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto c = ops::Variable(s.WithOpName("c"), {2, 2}, DT_FLOAT);

  auto add_ab = ops::Add(s.WithOpName("Add_ab"), a, b);
  auto add_bc = ops::Add(s.WithOpName("Add_bc"), b, c);

  auto mul_ab = ops::Mul(s.WithOpName("Mull_ab"), a, b);
  auto mul_bc = ops::Mul(s.WithOpName("Mull_bc"), a, b);

  InputList inputs{
      Output(add_ab),
      Output(add_bc),
      Output(mul_ab),
      Output(mul_bc),
  };
  auto add_all = ops::AddN(s.WithOpName("Add_all"), inputs);

  GraphDef graph;
  TF_ASSERT_OK(s.ToGraphDef(&graph));

  EXPECT_EQ(2, CountOpNodes(graph, "Add"));
  EXPECT_EQ(2, CountOpNodes(graph, "Mul"));
  EXPECT_EQ(1, CountOpNodes(graph, "AddN"));
  EXPECT_EQ(0, CountOpNodes(graph, "Transpose"));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
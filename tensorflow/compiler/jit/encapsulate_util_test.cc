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

#include "tensorflow/compiler/jit/encapsulate_util.h"

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(PerformStaticShapeInferenceBeforeEncapsulationTest, Basic) {
  // Build the graph:
  // "add" = "const_0" + "const_1"
  // "identity" = "add"
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output const_0 = ops::Const(s.WithOpName("const_0"), 1, {2});
  Output const_1 = ops::Const(s.WithOpName("const_1"), 2, {2});
  Output add = ops::Add(s.WithOpName("add"), const_0, const_1);
  Output identity = ops::Identity(s.WithOpName("identity"), add);
  Graph g(OpRegistry::Global());
  TF_CHECK_OK(s.ToGraph(&g));

  // "add" node is outside compilation node, "identity" node is XLA node.
  auto node_index = g.BuildNodeNameIndex();
  Node *add_node = node_index["add"], *identity_node = node_index["identity"];
  add_node->AddAttr("_xla", "cluster");
  add_node->AddAttr("_oc", "cluster");
  identity_node->AddAttr("_xla", "cluster");
  TF_CHECK_OK(
      PerformStaticShapeInferenceBeforeEncapsulation(&g, "_xla", "_oc"));

  // Check that only "add" node now has _xla_inferred_shapes attr.
  std::vector<Node *> nodes_with_inferred_shape;
  for (Node *n : g.nodes()) {
    if (HasNodeAttr(n->def(), kXlaInferredShapesAttrName)) {
      nodes_with_inferred_shape.push_back(n);
    }
  }
  EXPECT_EQ(nodes_with_inferred_shape.size(), 1);
  EXPECT_EQ(nodes_with_inferred_shape[0], add_node);
  std::vector<PartialTensorShape> output_shapes;
  TF_CHECK_OK(GetNodeAttr(add_node->attrs(), kXlaInferredShapesAttrName,
                          &output_shapes));
  EXPECT_EQ(output_shapes.size(), 1);
  TensorShapeProto shape_proto;
  output_shapes[0].AsProto(&shape_proto);
  EXPECT_EQ(shape_proto.dim_size(), 1);
  EXPECT_EQ(shape_proto.dim(0).size(), 2);
}

TEST(PreprocessForEncapsulationTest, ControlEdges) {
  // Build the graph:
  // "const_0" and "const_1" in host computation
  // "add" = "const_0" + "const_1" in XLA computation 0
  // "identity0" = "add" in XLA computation 0 & outside compilation 0
  // "identity1" = "identity0" in XLA computation 0
  // "identity2" = "identity1" in host computation
  // "identity3" = "identity2" in XLA computation 1
  // "identity4" = "identity3" in XLA computation 1 & outside compilation 1
  // "identity5" = "identity4" in XLA computation 1
  // "identity6" = "identity5" in host computation
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output const_0 = ops::Const(s.WithOpName("const_0"), 1, {});
  Output const_1 = ops::Const(s.WithOpName("const_1"), 2, {});
  Output add = ops::Add(s.WithOpName("add"), const_0, const_1);
  Output identity0 = ops::Identity(s.WithOpName("identity0"), add);
  Output identity1 = ops::Identity(s.WithOpName("identity1"), identity0);
  Output identity2 = ops::Identity(s.WithOpName("identity2"), identity1);
  Output identity3 = ops::Identity(s.WithOpName("identity3"), identity2);
  Output identity4 = ops::Identity(s.WithOpName("identity4"), identity3);
  Output identity5 = ops::Identity(s.WithOpName("identity5"), identity4);
  Graph g(OpRegistry::Global());
  TF_CHECK_OK(s.ToGraph(&g));
  auto node_index = g.BuildNodeNameIndex();

  // Set XLA computation/outside compilation attr, and add control edges.
  Node *const0_node = node_index["const_0"], *add_node = node_index["add"],
       *identity0_node = node_index["identity0"],
       *identity1_node = node_index["identity1"],
       *identity2_node = node_index["identity2"],
       *identity3_node = node_index["identity3"],
       *identity4_node = node_index["identity4"],
       *identity5_node = node_index["identity5"];
  add_node->AddAttr("_xla", "0");
  identity0_node->AddAttr("_xla", "0");
  identity0_node->AddAttr("_oc", "0");
  identity1_node->AddAttr("_xla", "0");
  identity3_node->AddAttr("_xla", "1");
  identity4_node->AddAttr("_xla", "1");
  identity4_node->AddAttr("_oc", "0");
  identity5_node->AddAttr("_xla", "1");
  // Case 1a: control edges between outside compilation and its XLA computation.
  g.AddControlEdge(add_node, identity0_node);
  g.AddControlEdge(identity0_node, identity1_node);
  // Case 1b: control edges between outside compilation and another XLA
  // computation.
  g.AddControlEdge(identity0_node, identity3_node);
  g.AddControlEdge(identity1_node, identity4_node);
  // Case 1c: control edges between different outside compilations.
  g.AddControlEdge(identity0_node, identity4_node);
  // Case 1d: control edges between outside compilation and host computation.
  g.AddControlEdge(const0_node, identity0_node);
  g.AddControlEdge(identity0_node, identity2_node);

  TF_CHECK_OK(PreprocessForEncapsulation(&g, "_xla", "_oc"));

  // Case 1a: add attr "_xla_connected_{from/to}_xla_computation = true" to the
  // outside compilation node.
  EXPECT_TRUE(HasNodeAttr(identity0_node->def(),
                          kXlaConnectedFromXlaComputationAttrName));
  EXPECT_TRUE(HasNodeAttr(identity0_node->def(),
                          kXlaConnectedToXlaComputationAttrName));
  // Case 1b: add attr "_xla_control_deps_{from/to} = XLA computation node name"
  // to the outside compilation node.
  std::vector<string> attr;
  TF_CHECK_OK(GetNodeAttr(identity0_node->def(),
                          kXlaConnectedToOtherXlaComputationAttrName, &attr));
  EXPECT_EQ(attr.size(), 1);
  EXPECT_EQ(attr[0], "1");
  attr.clear();
  TF_CHECK_OK(GetNodeAttr(identity4_node->def(),
                          kXlaConnectedFromOtherXlaComputationAttrName, &attr));
  EXPECT_EQ(attr.size(), 1);
  EXPECT_EQ(attr[0], "0");
  // Case 1c: add attr "_xla_control_deps = src node name" to dst node.
  attr.clear();
  TF_CHECK_OK(GetNodeAttr(identity4_node->def(),
                          kXlaControlDependenciesAttrName, &attr));
  EXPECT_EQ(attr.size(), 1);
  EXPECT_EQ(attr[0], "identity0");
  // Case 1d: add attr "_xla_control_deps = src node name" to dst node.
  attr.clear();
  TF_CHECK_OK(GetNodeAttr(identity0_node->def(),
                          kXlaControlDependenciesAttrName, &attr));
  EXPECT_EQ(attr.size(), 1);
  EXPECT_EQ(attr[0], "const_0");
  attr.clear();
  TF_CHECK_OK(GetNodeAttr(identity2_node->def(),
                          kXlaControlDependenciesAttrName, &attr));
  EXPECT_EQ(attr.size(), 1);
  EXPECT_EQ(attr[0], "identity0");
}

TEST(PreprocessForEncapsulationTest, DataEdges) {
  // Build the graph:
  // "const_0" and "const_1" in host computation
  // "add0" = "const_0" + "const_1" in XLA computation 0
  // "add1" = "add0" + "const_0" in XLA computation 0 & outside compilation 0
  // "identity0" = "add1" in XLA computation 0
  // "add2" = "add1" + "identity0" in host computation
  // "add3" = "add1" + "add2" in XLA computation 1
  // "add4" = "identity0" + "add2" in XLA computation 1 & outside compilation 1
  // "identity1" = "add4" in XLA computation 1
  // "identity2" = "identity1" in host computation
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output const_0 = ops::Const(s.WithOpName("const_0"), 1, {});
  Output const_1 = ops::Const(s.WithOpName("const_1"), 2, {});
  Output add0 = ops::Add(s.WithOpName("add0"), const_0, const_1);
  Output add1 = ops::Add(s.WithOpName("add1"), add0, const_0);
  Output identity0 = ops::Identity(s.WithOpName("identity0"), add1);
  Output add2 = ops::Add(s.WithOpName("add2"), add1, identity0);
  Output add3 = ops::Add(s.WithOpName("add3"), add1, add2);
  Output add4 = ops::Add(s.WithOpName("add4"), identity0, add2);
  Output identity1 = ops::Identity(s.WithOpName("identity1"), add4);
  Output identity2 = ops::Identity(s.WithOpName("identity2"), add4);
  Graph g(OpRegistry::Global());
  TF_CHECK_OK(s.ToGraph(&g));
  auto node_index = g.BuildNodeNameIndex();

  // Set XLA computation/outside compilation attr.
  Node *add0_node = node_index["add0"], *add1_node = node_index["add1"],
       *identity0_node = node_index["identity0"],
       *add3_node = node_index["add3"], *add4_node = node_index["add4"],
       *identity1_node = node_index["identity1"];
  add0_node->AddAttr("_xla", "0");
  add1_node->AddAttr("_xla", "0");
  add1_node->AddAttr("_oc", "0");
  identity0_node->AddAttr("_xla", "0");
  add3_node->AddAttr("_xla", "1");
  add4_node->AddAttr("_xla", "1");
  add4_node->AddAttr("_oc", "0");
  identity1_node->AddAttr("_xla", "1");

  TF_CHECK_OK(PreprocessForEncapsulation(&g, "_xla", "_oc"));

  // Check input nodes for related data edges.
  node_index = g.BuildNodeNameIndex();
  // Step 2: add an Identity node between different XLA computations.
  Node *bridge_add1_add3 = node_index["bridge_add1_add3"];
  EXPECT_NE(bridge_add1_add3, nullptr);
  string str;
  TF_CHECK_OK(
      GetNodeAttr(bridge_add1_add3->attrs(), kBridgeSourceNodeAttrName, &str));
  EXPECT_EQ(str, "add1");
  Node *bridge_identity0_add4 = node_index["bridge_identity0_add4"];
  EXPECT_NE(bridge_identity0_add4, nullptr);
  // Step 3: add placeholder for edges between host computation and outside
  // compilation.
  EXPECT_EQ(bridge_add1_add3->def().input(0), "add1_oc_to_host_placeholder");
  Node *add1_oc_to_host_placeholder = node_index["add1_oc_to_host_placeholder"];
  TF_CHECK_OK(GetNodeAttr(add1_oc_to_host_placeholder->attrs(),
                          kOutsideCompilationToHostOriginalNodeAttrName, &str));
  EXPECT_EQ(str, "add1");
  int i;
  TF_CHECK_OK(GetNodeAttr(add1_oc_to_host_placeholder->attrs(),
                          kOutsideCompilationToHostSrcOutputAttrName, &i));
  EXPECT_EQ(i, 0);
  add4_node = node_index["add4"];
  ASSERT_NE(add4_node, nullptr);
  EXPECT_EQ(add4_node->def().input(0),
            "bridge_identity0_add4_host_to_oc_placeholder");
  Node *identity0_host_to_oc_placeholder =
      node_index["bridge_identity0_add4_host_to_oc_placeholder"];
  TF_CHECK_OK(GetNodeAttr(identity0_host_to_oc_placeholder->attrs(),
                          kHostToOutsideCompilationOriginalNodeAttrName, &str));
  EXPECT_EQ(str, "bridge_identity0_add4");
  TF_CHECK_OK(GetNodeAttr(identity0_host_to_oc_placeholder->attrs(),
                          kHostToOutsideCompilationSrcOutputAttrName, &i));
  EXPECT_EQ(i, 0);
}

}  // namespace tensorflow

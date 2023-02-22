/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/int32_fulltype.h"

#include <string>
#include <unordered_map>

#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {

namespace {

////////////////////////////////////////////////////////////////////////////////
//
// Op registrations to set up the environment.
//
////////////////////////////////////////////////////////////////////////////////

// Register the following ops so they can be added to a Graph.
REGISTER_OP("FloatInt32").Output("a: float").Output("b: int32");
REGISTER_OP("FloatInt32Int32FT")
    .Output("a: float")
    .Output("b: int32")
    .Output("c: int32");
REGISTER_OP("FloatWithoutInt32").Output("a: float");

////////////////////////////////////////////////////////////////////////////////
//
// A test for automatic full type annotations has three phases:
//
// 1. Build a TensorFlow graph.
// 2. Run the automatic annotator.
// 3. Check the result.
//
////////////////////////////////////////////////////////////////////////////////
class Int32FulltypeTest : public ::testing::Test {
 protected:
  Int32FulltypeTest() {}

  // Builds the given graph, and (if successful) indexes the node
  // names for use in placement, and later lookup.
  Status BuildGraph(const GraphDefBuilder& builder, Graph* out_graph) {
    TF_RETURN_IF_ERROR(GraphDefBuilderToGraph(builder, out_graph));
    RebuildNodeNameMap(*out_graph);
    return OkStatus();
  }

  void AddTensorFT(FullTypeDef& t, tensorflow::FullTypeId out_t_id,
                   tensorflow::FullTypeId data_t_id) {
    FullTypeDef data_t;
    data_t.set_type_id(data_t_id);
    FullTypeDef out_t;
    out_t.set_type_id(out_t_id);
    (*out_t.add_args()) = data_t;
    (*t.add_args()) = out_t;
  }

  // Invokes the Placer on "graph". If no DeviceSet is specified, the
  // placement will use the default DeviceSet (of 10 CPU and 10 GPU devices).
  //
  // REQUIRES: "*graph" was produced by the most recent call to BuildGraph.
  Status Int32FulltypeAnnotate(Graph* graph) {
    Int32Fulltype int32_fulltype(graph);
    return int32_fulltype.Run();
  }

  // Returns the node in "graph" with the given name.
  //
  // REQUIRES: "graph" was produced by the most recent call to BuildGraph.
  Node* GetNodeByName(const Graph& graph, const string& name) {
    const auto search = nodes_by_name_.find(name);
    CHECK(search != nodes_by_name_.end()) << "Unknown node name: " << name;
    return graph.FindNodeId(search->second);
  }

 protected:
  std::unordered_map<string, int> nodes_by_name_;

 private:
  void RebuildNodeNameMap(const Graph& graph) {
    nodes_by_name_.clear();
    for (Node* node : graph.nodes()) {
      nodes_by_name_[node->name()] = node->id();
    }
  }
};

// Test creating full type information for int32 given a node that initially
// does not have any full type information.
TEST_F(Int32FulltypeTest, CreateFT) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("FloatInt32", b.opts().WithName("float_int32"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Int32FulltypeAnnotate(&g));

  Node* node = GetNodeByName(g, "float_int32");
  ASSERT_TRUE(node->def().has_experimental_type());
  const FullTypeDef& ft = node->def().experimental_type();
  ASSERT_EQ(ft.type_id(), TFT_PRODUCT);
  ASSERT_EQ(ft.args_size(), 2);
  ASSERT_EQ(ft.args(0).type_id(), TFT_UNSET);
  ASSERT_EQ(ft.args(1).type_id(), TFT_SHAPE_TENSOR);
  ASSERT_EQ(ft.args(1).args_size(), 1);
  ASSERT_EQ(ft.args(1).args(0).type_id(), TFT_INT32);
}

// Test that TFT_TENSOR for int32 is changed to TFT_SHAPE_TENSOR without
// changing other kinds of full type information.
TEST_F(Int32FulltypeTest, ModifyFT) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* node = ops::SourceOp("FloatInt32Int32FT",
                               b.opts().WithName("float_int32_int32"));
    node->mutable_def()->mutable_experimental_type()->set_type_id(TFT_PRODUCT);
    FullTypeDef& t = *node->mutable_def()->mutable_experimental_type();
    AddTensorFT(t, TFT_TENSOR, TFT_FLOAT);
    AddTensorFT(t, TFT_TENSOR, TFT_INT32);
    AddTensorFT(t, TFT_SHAPE_TENSOR, TFT_INT32);
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Int32FulltypeAnnotate(&g));

  Node* node = GetNodeByName(g, "float_int32_int32");
  ASSERT_TRUE(node->def().has_experimental_type());
  const FullTypeDef& ft = node->def().experimental_type();
  ASSERT_EQ(ft.type_id(), TFT_PRODUCT);
  ASSERT_EQ(ft.args_size(), 3);
  ASSERT_EQ(ft.args(0).type_id(), TFT_TENSOR);  // unchanged
  ASSERT_EQ(ft.args(0).args_size(), 1);
  ASSERT_EQ(ft.args(0).args(0).type_id(), TFT_FLOAT);
  ASSERT_EQ(ft.args(1).type_id(), TFT_SHAPE_TENSOR);  // changed
  ASSERT_EQ(ft.args(1).args_size(), 1);
  ASSERT_EQ(ft.args(1).args(0).type_id(), TFT_INT32);
  ASSERT_EQ(ft.args(2).type_id(), TFT_SHAPE_TENSOR);  // unchanged
  ASSERT_EQ(ft.args(2).args_size(), 1);
  ASSERT_EQ(ft.args(2).args(0).type_id(), TFT_INT32);
}

// Test NOT creating full type information for a node that does not have
// any int32 outputs.
TEST_F(Int32FulltypeTest, NotCreateFT) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("FloatWithoutInt32",
                  b.opts().WithName("float_without_int32"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Int32FulltypeAnnotate(&g));

  Node* node = GetNodeByName(g, "float_without_int32");
  ASSERT_FALSE(node->def().has_experimental_type());
}

}  // namespace
}  // namespace tensorflow

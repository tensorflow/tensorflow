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
#include "tensorflow/core/graph/node_builder.h"

#include <string>

#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

REGISTER_OP("Source").Output("o: out_types").Attr("out_types: list(type)");
REGISTER_OP("Sink").Input("i: T").Attr("T: type");

TEST(NodeBuilderTest, Simple) {
  Graph graph(OpRegistry::Global());
  Node* source_node;
  TF_EXPECT_OK(NodeBuilder("source_op", "Source")
                   .Attr("out_types", {DT_INT32, DT_STRING})
                   .Finalize(&graph, &source_node));
  ASSERT_TRUE(source_node != nullptr);

  // Try connecting to each of source_node's outputs.
  TF_EXPECT_OK(NodeBuilder("sink1", "Sink")
                   .Input(source_node)
                   .Finalize(&graph, nullptr));
  TF_EXPECT_OK(NodeBuilder("sink2", "Sink")
                   .Input(source_node, 1)
                   .Finalize(&graph, nullptr));

  // Generate an error if the index is out of range.
  EXPECT_FALSE(NodeBuilder("sink3", "Sink")
                   .Input(source_node, 2)
                   .Finalize(&graph, nullptr)
                   .ok());
  EXPECT_FALSE(NodeBuilder("sink4", "Sink")
                   .Input(source_node, -1)
                   .Finalize(&graph, nullptr)
                   .ok());
  EXPECT_FALSE(NodeBuilder("sink5", "Sink")
                   .Input({source_node, -1})
                   .Finalize(&graph, nullptr)
                   .ok());

  // Generate an error if the node is nullptr.  This can happen when using
  // GraphDefBuilder if there was an error creating the input node.
  EXPECT_FALSE(NodeBuilder("sink6", "Sink")
                   .Input(nullptr)
                   .Finalize(&graph, nullptr)
                   .ok());
  EXPECT_FALSE(NodeBuilder("sink7", "Sink")
                   .Input(NodeBuilder::NodeOut(nullptr, 0))
                   .Finalize(&graph, nullptr)
                   .ok());
}

REGISTER_OP("FullTypeOpBasicType")
    .Output("o1: out_type")
    .Attr("out_type: type")
    .SetTypeConstructor([](OpDef* op_def) {
      FullTypeDef* tdef =
          op_def->mutable_output_arg(0)->mutable_experimental_full_type();
      tdef->set_type_id(TFT_ARRAY);

      FullTypeDef* arg = tdef->add_args();
      arg->set_type_id(TFT_VAR);
      arg->set_s("out_type");

      return Status::OK();
    });

TEST(NodeBuilderTest, TypeConstructorBasicType) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_EXPECT_OK(NodeBuilder("op", "FullTypeOpBasicType")
                   .Attr("out_type", DT_FLOAT)
                   .Finalize(&graph, &node));
  ASSERT_TRUE(node->def().has_experimental_type());
  const FullTypeDef& ft = node->def().experimental_type();
  ASSERT_EQ(ft.type_id(), TFT_PRODUCT);
  ASSERT_EQ(ft.args_size(), 1);
  auto ot = ft.args(0);
  ASSERT_EQ(ot.type_id(), TFT_ARRAY);
  ASSERT_EQ(ot.args(0).type_id(), TFT_FLOAT);
  ASSERT_EQ(ot.args(0).args().size(), 0);
}

REGISTER_OP("FullTypeOpListType")
    .Output("o1: out_types")
    .Attr("out_types: list(type)")
    .SetTypeConstructor([](OpDef* op_def) {
      FullTypeDef* tdef =
          op_def->mutable_output_arg(0)->mutable_experimental_full_type();
      tdef->set_type_id(TFT_ARRAY);

      FullTypeDef* arg = tdef->add_args();
      arg->set_type_id(TFT_VAR);
      arg->set_s("out_types");

      return Status::OK();
    });

TEST(NodeBuilderTest, TypeConstructorListType) {
  Graph graph(OpRegistry::Global());
  Node* node;
  ASSERT_FALSE(NodeBuilder("op", "FullTypeOpListType")
                   .Attr("out_types", {DT_FLOAT, DT_INT32})
                   .Finalize(&graph, &node)
                   .ok());
}

}  // namespace
}  // namespace tensorflow

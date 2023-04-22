/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <utility>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declare here, so we don't need a public header.
Status RenameNode(const GraphDef& input_graph_def,
                  const TransformFuncContext& context,
                  GraphDef* output_graph_def);

TEST(RenameNodeTest, Rename) {
  GraphDef in_graph;
  NodeDef* node = in_graph.add_node();
  node->set_name("input");
  node->set_op("Placeholder");

  NodeDef* node_splitter = in_graph.add_node();
  node_splitter->set_name("splitter");
  node_splitter->set_op("Split");

  NodeDef* node_adder = in_graph.add_node();
  node_adder->set_op("Add");
  node_adder->set_name("adder");
  node_adder->add_input("splitter");
  node_adder->add_input("splitter:1");

  GraphDef result;
  TransformFuncContext context;
  context.input_names = {};
  context.output_names = {"adder"};
  context.params.insert(std::pair<string, std::vector<string>>(
      {"old_node_name", {std::string("splitter")}}));
  context.params.insert(std::pair<string, std::vector<string>>(
      {"new_node_name", {string("demux")}}));
  TF_ASSERT_OK(RenameNode(in_graph, context, &result));

  std::map<string, const NodeDef*> node_lookup;
  MapNamesToNodes(result, &node_lookup);
  EXPECT_EQ(1, node_lookup.count("demux"));
  EXPECT_EQ(1, node_lookup.count("adder"));
  EXPECT_EQ(2, node_lookup["adder"]->input().size());
  EXPECT_EQ("demux", node_lookup["adder"]->input()[0]);
  EXPECT_EQ("demux:1", node_lookup["adder"]->input()[1]);
}

TEST(RenameNodeTest, FailWhenNameAlreadyExists) {
  GraphDef in_graph;
  NodeDef* node = in_graph.add_node();
  node->set_name("input");
  node->set_op("Placeholder");

  NodeDef* node_splitter = in_graph.add_node();
  node_splitter->set_name("splitter");
  node_splitter->set_op("Split");

  NodeDef* node_adder = in_graph.add_node();
  node_adder->set_op("Add");
  node_adder->set_name("adder");
  node_adder->add_input("splitter");
  node_adder->add_input("splitter:1");

  GraphDef result;
  TransformFuncContext context;
  context.input_names = {};
  context.output_names = {"adder"};
  context.params.insert(std::pair<string, std::vector<string>>(
      {"old_node_name", {std::string("splitter")}}));
  context.params.insert(std::pair<string, std::vector<string>>(
      {"new_node_name", {string("adder")}}));
  EXPECT_FALSE(RenameNode(in_graph, context, &result).ok());
}

}  // namespace graph_transforms
}  // namespace tensorflow

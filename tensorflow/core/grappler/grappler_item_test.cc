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

#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class GrapplerItemTest : public ::testing::Test {};

TEST_F(GrapplerItemTest, Basic) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {{"CPU:0"}});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  EXPECT_TRUE(item.InitOpsFanin().empty());

  std::vector<string> graph_nodes;
  for (const auto& node : item.graph.node()) {
    graph_nodes.push_back(node.name());
  }
  std::vector<string> main_ops;
  for (const auto& node : item.MainOpsFanin()) {
    main_ops.push_back(node->name());
  }
  std::sort(graph_nodes.begin(), graph_nodes.end());
  std::sort(main_ops.begin(), main_ops.end());
  EXPECT_EQ(main_ops, graph_nodes);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

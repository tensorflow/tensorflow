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

#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class GraphViewTest : public ::testing::Test {};

TEST_F(GraphViewTest, BasicGraph) {
  TrivialTestGraphInputYielder fake_input(4, 2, 2, false, {"/CPU:0", "/GPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  std::cout << item.graph.DebugString() << std::endl;

  GraphView graph(&item.graph);

  GraphView::InputPort input = graph.GetInputPort("AddN", 0);
  EXPECT_EQ("AddN", input.node->name());
  EXPECT_EQ(0, input.port_id);
  GraphView::OutputPort fanin = graph.GetFanin(input);
  EXPECT_EQ("Square", fanin.node->name());
  EXPECT_EQ(0, fanin.port_id);

  input = graph.GetInputPort("AddN", 1);
  EXPECT_EQ("AddN", input.node->name());
  EXPECT_EQ(1, input.port_id);
  fanin = graph.GetFanin(input);
  EXPECT_EQ("Square_1", fanin.node->name());
  EXPECT_EQ(0, fanin.port_id);

  GraphView::OutputPort output = graph.GetOutputPort("AddN", 0);
  EXPECT_EQ("AddN", output.node->name());
  EXPECT_EQ(0, output.port_id);
  EXPECT_EQ(2, graph.GetFanout(output).size());
  for (auto fanout : graph.GetFanout(output)) {
    if (fanout.node->name() == "AddN_2" || fanout.node->name() == "AddN_3") {
      EXPECT_EQ(0, fanout.port_id);
    } else {
      // Invalid fanout
      EXPECT_FALSE(true);
    }
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

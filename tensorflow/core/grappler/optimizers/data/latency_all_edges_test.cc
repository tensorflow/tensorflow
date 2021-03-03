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

#include "tensorflow/core/grappler/optimizers/data/latency_all_edges.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class LatencyAllEdgesTest : public ::testing::TestWithParam<bool> {};

TEST_P(LatencyAllEdgesTest, AddLatenciesAfterTensorMapPrefetch) {
  const bool contain_model = GetParam();
  using test::function::NDef;
  GrapplerItem item;
  NodeDef component_node =
      NDef("component_node", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}});
  NodeDef from_tensor_node =
      NDef("from_tensor_node", "TensorDataset", {"component_node"},
           {{"Toutput_types", {}}, {"output_shapes", {}}});

  NodeDef captured_input_node = NDef("captured_input_node", "Const", {},
                                     {{"value", ""}, {"dtype", DT_STRING}});
  NodeDef map_node = NDef("map_node", "MapDataset",
                          {"from_tensor_node", "captured_input_node"},
                          {{"f", {}},
                           {"Targumemts", {}},
                           {"output_shapes", {}},
                           {"output_types", {}}});
  NodeDef buffer_size_node = NDef("buffer_size_node", "Const", {},
                                  {{"value", 1}, {"dtype", DT_INT32}});
  NodeDef prefetch_node =
      NDef("prefetch_node", "PrefetchDataset", {"map_node", "buffer_size_node"},
           {{"output_shapes", {}}, {"output_types", {}}});
  NodeDef model_node = NDef("model_node", "ModelDataset", {"prefetch_node"},
                            {{"output_shapes", {}}, {"output_types", {}}});

  if (contain_model) {
    item.graph = test::function::GDef(
        {component_node, from_tensor_node, captured_input_node, map_node,
         buffer_size_node, prefetch_node, model_node});
  } else {
    item.graph = test::function::GDef({component_node, from_tensor_node,
                                       captured_input_node, map_node,
                                       buffer_size_node, prefetch_node});
  }

  LatencyAllEdges optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(output.node_size(), contain_model ? 13 : 12);
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("LatencyStatsDataset", output));
  std::vector<int> latency_node_indices =
      graph_utils::FindAllGraphNodesWithOp("LatencyStatsDataset", output);
  EXPECT_EQ(latency_node_indices.size(), 3);
  std::vector<NodeDef> dataset_nodes = {std::move(from_tensor_node),
                                        std::move(map_node),
                                        std::move(prefetch_node)};
  for (int i = 0; i < latency_node_indices.size(); i++) {
    NodeDef latency_node = output.node(latency_node_indices[i]);
    EXPECT_EQ(latency_node.input_size(), 2);
    EXPECT_EQ(latency_node.input(0), dataset_nodes[i].name());
    EXPECT_TRUE(
        AreAttrValuesEqual(latency_node.attr().at("output_shapes"),
                           dataset_nodes[i].attr().at("output_shapes")));
    if (dataset_nodes[i].attr().find("output_types") !=
        dataset_nodes[i].attr().end()) {
      EXPECT_TRUE(
          AreAttrValuesEqual(latency_node.attr().at("output_types"),
                             dataset_nodes[i].attr().at("output_types")));
    } else {
      if (dataset_nodes[i].attr().find("Toutput_types") !=
          dataset_nodes[i].attr().end()) {
        EXPECT_TRUE(
            AreAttrValuesEqual(latency_node.attr().at("output_types"),
                               dataset_nodes[i].attr().at("Toutput_types")));
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Test, LatencyAllEdgesTest,
                         ::testing::Values(false, true));

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

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

#include "tensorflow/core/grappler/optimizers/data/filter_with_random_uniform_fusion.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

#include <iostream>

namespace tensorflow {
namespace grappler {

constexpr char kFusedOpName[] = "SamplingDataset";

NodeDef MakeFusedNode(const NodeDef& filter_node, float rate,
                      MutableGraphView* graph) {
  NodeDef fused_node;
  graph_utils::SetUniqueGraphNodeName("fused_sampling", graph->graph(),
                                      &fused_node);
  fused_node.set_op(kFusedOpName);

  // Copy over inputs.
  for (int i = 0; i < filter_node.input_size(); ++i) {
    fused_node.add_input(filter_node.input(i));
  }

  // Required attrs.
  for (auto key : {"output_shapes", "output_types"}) {
    graph_utils::CopyAttribute(key, filter_node, &fused_node);
  }

  // Optional attrs.
  for (auto key : {"use_inter_op_parallelism", "sloppy"}) {
    if (gtl::FindOrNull(filter_node.attr(), key)) {
      graph_utils::CopyAttribute(key, filter_node, &fused_node);
    }
  }

  NodeDef* tmp = graph_utils::AddScalarConstNode<float>(rate, graph);
  fused_node.add_input(tmp->name());

  return fused_node;
}

const NodeDef* FunctionFindNodeDef(const FunctionDef& function, const string op,
                                   const string func, const string match) {
  for (const NodeDef& func_node : function.node_def()) {
    if (func_node.op() != op) {
      continue;
    }
    if (func_node.name() + match != func) {
      continue;
    }
    return &func_node;
  }
  return NULL;
}

// This optimization fuse one of the following two forms of
// filter + random_uniform predication into a single data sampling operation:
// fuse:
//   filter
//   |
//   + predication: less [0]
//                  |
//                  + random_uniform [1]
//                  |
//                  + rate
// or:
//   filter
//   |
//   + predication: less
//                  |
//                  + random_uniform[]
//                  |
//                  + rate
// into:
//   sampling(rate)
Status FilterWithRandomUniformFusion::OptimizeAndCollectStats(Cluster* cluster,
                                                  const GrapplerItem& item,
                                                  GraphDef* output,
                                                  OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;
  float rate;

  for (const NodeDef& node : item.graph.node()) {
    // stage 1 -- recognition
    if (node.op() != "FilterDataset") {
      continue;
    }

    // Use a more descriptive variable name
    const NodeDef& filter_node = node;

    // find predicate function of the node
    const auto& predicate = filter_node.attr().at("predicate");
    const string func_name = predicate.func().name();

    bool function_match = false;
    // find the function that matches func_name
    for (const auto& function : item.graph.library().function()) {
      if (function.signature().name() == func_name) {
        if (function.ret().size() != 1) {
          continue;
        }
        auto it = function.ret().begin();
        string node_name = it->second;
        const NodeDef* less_node;
        const NodeDef* func_node =
            FunctionFindNodeDef(function, "Identity", node_name, ":output:0");
        if (func_node != NULL) {
          node_name = func_node->input(0);
        }
        func_node = FunctionFindNodeDef(function, "StridedSlice", node_name,
                                        ":output:0");
        if (func_node != NULL) {
          // for form one: datasetS = datasetS.filter(lambda x:
          // tf.less(tf.random_uniform([1]), rate)[0])
          less_node = FunctionFindNodeDef(function, "Less", func_node->input(0),
                                          ":z:0");
        } else {
          // for form two: datasetS = datasetS.filter(lambda _:
          // tf.random_uniform([]) < rate)
          less_node = FunctionFindNodeDef(function, "Less", node_name, ":z:0");
        }
        if (less_node == NULL) {
          continue;
        }

        // check whether the function is actually doing
        // random_uniform[0.0, 1.0) < rate
        const NodeDef* random_uniform_node = FunctionFindNodeDef(
            function, "RandomUniform", less_node->input(0), ":output:0");
        if (random_uniform_node == NULL) {
          continue;
        }

        const NodeDef* rate_node = FunctionFindNodeDef(
            function, "Const", less_node->input(1), ":output:0");
        if (rate_node == NULL) {
          continue;
        }

        const auto& rate_value = rate_node->attr().at("value");
        const auto& rate_tensor = rate_value.tensor();
        rate = rate_tensor.float_val(0);

        function_match = true;
        break;
      }
    }

    if (!function_match) {
      continue;
    }

    // stage 2 -- fuse
    const auto* fused_sampling =
        graph.AddNode(MakeFusedNode(filter_node, rate, &graph));

    graph.UpdateFanouts(filter_node.name(), fused_sampling->name());

    // Mark the `Filter` node for removal.
    nodes_to_delete.insert(filter_node.name());
    stats->num_changes++;
  }

  graph.DeleteNodes(nodes_to_delete);
  return Status::OK();
}

void FilterWithRandomUniformFusion::Feedback(Cluster* cluster,
                                             const GrapplerItem& item,
                                             const GraphDef& optimize_output,
                                             double result) {
  // Nothing to do for FilterWithRandomUniformFusion
}

REGISTER_GRAPH_OPTIMIZER_AS(FilterWithRandomUniformFusion,
                            "filter_with_random_uniform_fusion");

}  // end namespace grappler
}  // end namespace tensorflow

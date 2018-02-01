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

#include "tensorflow/contrib/tensorrt/convert/convert_graph.h"

#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/layout_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorflow/contrib/tensorrt/convert/convert_nodes.h"
#include "tensorflow/contrib/tensorrt/segment/segment.h"
#include "tensorrt/include/NvInfer.h"

//------------------------------------------------------------------------------
namespace tensorflow {
namespace tensorrt {
namespace convert {
namespace {

static bool IsTensorRTCandidate(const tensorflow::NodeDef& node_def) {
  // LINT.IfChange
  // TODO(jie): Segmentation shouldn't associated with op name.
  //            Split it into a registration for each kernel.
  static const std::set<std::string> candidate_ops = {
      "Identity", "Const", "Conv2D", "MaxPool", "BiasAdd", "Relu",
      "Add",      "Mul",   "Sub",    "Rsqrt",   "Pad"  // "Placeholder" ,"Mean"
  };
  // LINT.ThenChange(
  //    https://www.tensorflow.org/code/tensorflow/contrib/tensorrt/convert/convert_nodes.h)
  return candidate_ops.count(node_def.op());
}

void GetSubGraphIncomingEdges(const tensorflow::Graph& graph,
                              const std::set<int>& subgraph_node_ids,
                              tensorflow::EdgeSet* incoming_edges) {
  for (int node_id : subgraph_node_ids) {
    const tensorflow::Node* node = graph.FindNodeId(node_id);
    for (const tensorflow::Edge* edge : node->in_edges()) {
      if (!subgraph_node_ids.count(edge->src()->id()) &&
          !edge->src()->IsSource()) {
        incoming_edges->insert(edge);
      }
    }
  }
}

void GetSubGraphOutgoingEdges(const tensorflow::Graph& graph,
                              const std::set<int>& subgraph_node_ids,
                              tensorflow::EdgeSet* outgoing_edges) {
  for (int node_id : subgraph_node_ids) {
    const tensorflow::Node* node = graph.FindNodeId(node_id);
    for (const tensorflow::Edge* edge : node->out_edges()) {
      if (!subgraph_node_ids.count(edge->dst()->id()) &&
          !edge->dst()->IsSink()) {
        outgoing_edges->insert(edge);
      }
    }
  }
}

std::pair<std::string, int> ParseTensorName(std::string name,
                                            int default_idx = 0) {
  int idx = default_idx;
  size_t sep = name.find_last_of(':');
  if (sep != std::string::npos) {
    name = name.substr(0, sep);
    idx = std::stoi(name.substr(sep + 1));
  }
  return std::make_pair(name, idx);
}

std::unordered_map<std::string, std::vector<int>> BuildTensorNameMap(
    const std::vector<std::string>& tensor_names) {
  std::unordered_map<std::string, std::vector<int>> result;
  for (std::string const& tensor_name : tensor_names) {
    std::string node_name;
    int index;
    std::tie(node_name, index) = ParseTensorName(tensor_name);
    result[node_name].push_back(index);
  }
  return result;
}

tensorflow::Status ConvertSubGraphToTensorRT(
    const std::vector<std::string>& output_names,
    const std::set<int>& subgraph_node_ids,
    size_t max_batch_size,  // max batch size that engine will be created for
    // max amount of memory that engine will be allowed to consume, in bytes
    size_t max_workspace_size_bytes,
    const tensorflow::grappler::GraphProperties& graph_properties,
    tensorflow::Graph* graph) {
  tensorflow::EdgeSet subgraph_incoming_edges;
  GetSubGraphIncomingEdges(*graph, subgraph_node_ids, &subgraph_incoming_edges);

  std::vector<std::pair<int, int>> subgraph_inputs;

  // Collect inputs by looking for incoming edges
  for (const tensorflow::Edge* edge : subgraph_incoming_edges) {
    subgraph_inputs.push_back({edge->src()->id(), edge->src_output()});
  }
  std::set<std::pair<int, int>> subgraph_outputs_set;
  // Collect outputs referenced from output_names
  auto output_name_to_index_map = BuildTensorNameMap(output_names);
  for (int node_id : subgraph_node_ids) {
    tensorflow::Node* node = graph->FindNodeId(node_id);
    if (output_name_to_index_map.count(node->name())) {
      for (int index : output_name_to_index_map.at(node->name())) {
        subgraph_outputs_set.insert({node_id, index});
      }
    }
  }
  // Collect outputs referenced from outgoing edges
  tensorflow::EdgeSet subgraph_outgoing_edges;
  GetSubGraphOutgoingEdges(*graph, subgraph_node_ids, &subgraph_outgoing_edges);
  for (const tensorflow::Edge* edge : subgraph_outgoing_edges) {
    subgraph_outputs_set.insert({edge->src()->id(), edge->src_output()});
  }
  // Impose an ordering on the outputs
  std::vector<std::pair<int, int>> subgraph_outputs(
      subgraph_outputs_set.begin(), subgraph_outputs_set.end());
  // Build TensorRT node and add it to the graph
  tensorflow::NodeDef trt_node_def;
  TF_RETURN_IF_ERROR(ConvertSubGraphToTensorRTNodeDef(
      *graph, subgraph_node_ids, subgraph_inputs, subgraph_outputs,
      max_batch_size, max_workspace_size_bytes, graph_properties,
      &trt_node_def));
  tensorflow::Status status;
  tensorflow::Node* trt_node = graph->AddNode(trt_node_def, &status);

  TF_RETURN_IF_ERROR(status);

  // Re-map outgoing edges to use the new TRT node instead of the orig subgraph
  std::map<std::pair<int, int>, int> subgraph_edge_to_output_map;
  for (size_t i = 0; i < subgraph_outputs.size(); ++i) {
    subgraph_edge_to_output_map.insert({subgraph_outputs.at(i), i});
  }
  TF_RETURN_IF_ERROR(status);
  for (const tensorflow::Edge* edge : subgraph_outgoing_edges) {
    std::pair<int, int> old_src = {edge->src()->id(), edge->src_output()};
    int new_src_output = subgraph_edge_to_output_map.at(old_src);
    graph->UpdateEdge(trt_node, new_src_output, edge->dst(), edge->dst_input());
  }
  // Remove the original subgraph
  for (int node_id : subgraph_node_ids) {
    tensorflow::Node* node = graph->FindNodeId(node_id);
    // Don't remove the input placeholders
    if (node->type_string() == "Placeholder") {
      continue;
    }
    graph->RemoveNode(node);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status BuildNodeMap(
    const tensorflow::Graph& graph,
    std::unordered_map<std::string, tensorflow::Node*>* node_map) {
  for (auto* node : graph.op_nodes()) {
    if (!node_map->insert({node->name(), node}).second) {
      return tensorflow::errors::AlreadyExists(
          "Node name is not unique in graph: " + node->name());
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace

tensorflow::Status ConvertGraphDefToTensorRT(
    const tensorflow::GraphDef& graph_def,
    const std::vector<std::string>& output_names, size_t max_batch_size,
    size_t max_workspace_size_bytes, tensorflow::GraphDef* new_graph_def) {
  // optimization pass
  tensorflow::grappler::GrapplerItem item;
  item.fetch = output_names;
  tensorflow::GraphDef gdef;

  // layout optimization
  item.graph = graph_def;
  tensorflow::grappler::LayoutOptimizer optimizer;
  tensorflow::grappler::Cluster* cluster;

  // virtual cluster
  tensorflow::DeviceProperties device_properties;
  device_properties.set_type("GPU");
  device_properties.mutable_environment()->insert({"architecture", "6"});
  cluster =
      new tensorflow::grappler::VirtualCluster({{"/GPU:0", device_properties}});

  TF_RETURN_IF_ERROR(optimizer.Optimize(cluster, item, &gdef));

  // constant folding
  item.graph = gdef;
  tensorflow::grappler::ConstantFolding fold(nullptr);
  TF_RETURN_IF_ERROR(fold.Optimize(nullptr, item, &gdef));

  // AJ refactoring shape inference through grappler/GraphProperties.
  tensorflow::grappler::GraphProperties static_graph_properties(item);
  static_graph_properties.InferStatically(false);

  // Build full graph
  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             gdef.library());
  tensorflow::Graph graph(flib);
  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), gdef, &graph));

  // Segment the graph into subgraphs that can be converted to TensorRT
  tensorflow::tensorrt::segment::SegmentOptions segment_options;

  // TODO(ben,jie,sami): exclude output nodes (DISCUSS IT)
  for (auto node : output_names) {
    segment_options.exclude_node_list.insert(node);
  }

  // TODO(sami): this should be passed as a knob!!!!
  segment_options.minimum_segment_size = 2;
  tensorflow::tensorrt::segment::SegmentNodesVector segments;
  TF_RETURN_IF_ERROR(tensorrt::segment::SegmentGraph(
      gdef, IsTensorRTCandidate, segment_options, &segments));
  if (segments.size() > 1) {
    VLOG(0) << "MULTIPLE tensorrt candidate conversion: " << segments.size();
  }
  std::unordered_map<std::string, tensorflow::Node*> node_map;
  TF_RETURN_IF_ERROR(BuildNodeMap(graph, &node_map));
  for (const std::set<std::string>& subgraph_node_names : segments) {
    std::set<int> subgraph_node_ids;
    for (const std::string& node_name : subgraph_node_names) {
      subgraph_node_ids.insert(node_map.at(node_name)->id());
    }
    TF_RETURN_IF_ERROR(ConvertSubGraphToTensorRT(
        output_names, subgraph_node_ids, max_batch_size,
        max_workspace_size_bytes, static_graph_properties, &graph));
  }
  graph.ToGraphDef(new_graph_def);
  return tensorflow::Status::OK();
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

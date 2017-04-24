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

#include "tensorflow/core/grappler/costs/graph_properties.h"

#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/costs/utils.h"

namespace tensorflow {
namespace grappler {

Status GraphProperties::InferStatically() {
  Graph graph(OpRegistry::Global());
  ShapeRefiner shape_refiner(graph.versions().producer(), graph.op_registry());
  ImportGraphDefOptions options;
  Status s = ImportGraphDef(options, item_.graph, &graph, &shape_refiner);
  TF_RETURN_IF_ERROR(s);

  for (const Node* const node : graph.nodes()) {
    VLOG(1) << "<Node> " << node->name();
    auto ctx = shape_refiner.GetContext(node);
    if (!ctx) {
      continue;
    }
    CHECK_EQ(ctx->num_inputs(), node->num_inputs());
    std::vector<OpInfo::TensorProperties> input_properties;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      OpInfo::TensorProperties properties;
      properties.set_dtype(node->input_type(i));
      shape_inference::ShapeHandle shp = ctx->input(i);
      if (!ctx->RankKnown(shp)) {
        properties.mutable_shape()->set_unknown_rank(true);
      } else {
        for (int j = 0; j < ctx->Rank(shp); ++j) {
          shape_inference::DimensionHandle dim = ctx->Dim(shp, j);
          int64 d = ctx->Value(dim);
          properties.mutable_shape()->add_dim()->set_size(d);
        }
      }
      input_properties.push_back(properties);
    }
    input_properties_[node->name()] = input_properties;

    // TODO(bsteiner): share this code with the input processing above.
    CHECK_EQ(ctx->num_outputs(), node->num_outputs());
    std::vector<OpInfo::TensorProperties> output_properties;
    for (int i = 0; i < ctx->num_outputs(); ++i) {
      OpInfo::TensorProperties properties;
      properties.set_dtype(node->output_type(i));
      shape_inference::ShapeHandle shp = ctx->output(i);
      if (!ctx->RankKnown(shp)) {
        properties.mutable_shape()->set_unknown_rank(true);
      } else {
        for (int j = 0; j < ctx->Rank(shp); ++j) {
          shape_inference::DimensionHandle dim = ctx->Dim(shp, j);
          int64 d = ctx->Value(dim);
          properties.mutable_shape()->add_dim()->set_size(d);
        }
      }
      output_properties.push_back(properties);
    }
    output_properties_[node->name()] = output_properties;

    if (!node->assigned_device_name().empty()) {
      device_names_[node->name()] = node->assigned_device_name();
    } else if (!node->def().device().empty()) {
      device_names_[node->name()] = node->def().device();
    } else {
      device_names_[node->name()] = "not set";
    }
  }

  return Status::OK();
}

Status GraphProperties::InferDynamically(Cluster* cluster) {
  TF_RETURN_IF_ERROR(cluster->Initialize(item_));

  // Runs the model once to collect the shapes in the cost model.
  RunMetadata metadata;
  TF_RETURN_IF_ERROR(
      cluster->Run(item_.graph, item_.feed, item_.fetch, &metadata));

  std::unordered_map<string, const CostGraphDef::Node*> name_to_cost;
  std::unordered_map<string, const NodeDef*> name_to_node;  // Empty
  for (auto& node : metadata.cost_graph().node()) {
    name_to_cost[node.name()] = &node;

    std::vector<OpInfo::TensorProperties> output_properties;
    for (const auto& out : node.output_info()) {
      OpInfo::TensorProperties properties;
      properties.set_dtype(out.dtype());
      *properties.mutable_shape() = out.shape();
      output_properties.push_back(properties);
    }
    output_properties_[node.name()] = output_properties;
  }

  for (const auto& node : item_.graph.node()) {
    // Skip the nodes that are not in the cost graph: these are nodes that
    // aren't run, because they aren't in the intersection of transitive fan-in
    // of a fetch node and the transitive fan-out of an input, or nodes that
    // were optimized away by the optimizer.
    auto it = name_to_cost.find(node.name());
    if (it == name_to_cost.end()) {
      continue;
    }
    std::vector<OpInfo::TensorProperties> inputs =
        FindInputFeatures(node, name_to_cost, name_to_node);

    input_properties_[node.name()] = inputs;

    const CostGraphDef::Node* cost_node = it->second;
    device_names_[node.name()] = cost_node->device();
  }
  return Status::OK();
}

std::vector<OpInfo::TensorProperties> GraphProperties::GetInputProperties(
    const string& node_name) const {
  auto it = input_properties_.find(node_name);
  if (it != input_properties_.end()) {
    return it->second;
  }
  return std::vector<OpInfo::TensorProperties>();
}

std::vector<OpInfo::TensorProperties> GraphProperties::GetOutputProperties(
    const string& node_name) const {
  auto it = output_properties_.find(node_name);
  if (it != output_properties_.end()) {
    return it->second;
  }
  return std::vector<OpInfo::TensorProperties>();
}

string GraphProperties::GetDeviceName(const string& node_name) const {
  auto it = device_names_.find(node_name);
  if (it != device_names_.end()) {
    return it->second;
  }
  return "";
}

}  // end namespace grappler
}  // end namespace tensorflow

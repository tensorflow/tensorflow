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

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/costs/utils.h"

namespace tensorflow {
namespace grappler {

Status GraphProperties::InferStatically() {
  Graph graph(OpRegistry::Global());
  ShapeRefiner shape_refiner(graph.versions().producer(), graph.op_registry());
  shape_refiner.set_require_shape_inference_fns(false);
  ImportGraphDefOptions options;
  Status s = ImportGraphDef(options, item_.graph, &graph, &shape_refiner);
  TF_RETURN_IF_ERROR(s);

  // List the resources and the nodes using them
  std::unordered_map<const Node*, std::unordered_set<const Node*>> resources;
  for (const Node* const node : graph.nodes()) {
    for (int i = 0; i < node->num_inputs(); ++i) {
      if (node->input_type(i) == DataType::DT_RESOURCE) {
        const Node* resource;
        TF_CHECK_OK(node->input_node(i, &resource));
        resources[resource].insert(node);
      }
    }
  }

  // If we found a resource, try to propagate the shapes through it.
  bool done = true;
  do {
    std::queue<const Node*> new_shapes;
    for (const auto& resource_data : resources) {
      const Node* qnode = resource_data.first;
      StringPiece type(qnode->type_string());
      if (!type.ends_with("QueueV2")) {
        continue;
      }
      auto qctx = shape_refiner.GetContext(qnode);
      if (!qctx) {
        continue;
      }
      DataType queue_type = qctx->output_handle_dtype(0);
      shape_inference::ShapeHandle queue_shp = qctx->output_handle_shape(0);
      if (qctx->FullyDefined(queue_shp) && queue_type != DT_INVALID) {
        continue;
      }

      for (const auto& node : resource_data.second) {
        auto ctx = shape_refiner.GetContext(node);
        if (!ctx) {
          continue;
        }
        if (node->type_string().find("Enqueue") != std::string::npos) {
          if (ctx->num_inputs() == 2) {
            const DataType dtype = node->input_type(1);
            if (queue_type == DT_INVALID) {
              queue_type = dtype;
            } else {
              CHECK_EQ(queue_type, dtype);
            }
            shape_inference::ShapeHandle shp = ctx->input(1);
            TF_RETURN_IF_ERROR(qctx->Merge(queue_shp, shp, &queue_shp));
          }
        }
      }
      if (qctx->set_output_handle_dtype(0, queue_type) |
          qctx->MergeOutputHandleShape(0, queue_shp)) {
        new_shapes.push(qnode);
      }
    }
    // Propagate the shapes in the transitive fan-out of the queue.
    done = new_shapes.empty();
    while (!new_shapes.empty()) {
      const Node* n = new_shapes.front();
      new_shapes.pop();
      for (const Node* fanout : n->out_nodes()) {
        bool updated = false;
        TF_RETURN_IF_ERROR(shape_refiner.UpdateNode(fanout, &updated));
        if (updated) {
          new_shapes.push(fanout);
        }
      }
    }
  } while (!done);

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
    } else if (!node->requested_device().empty()) {
      device_names_[node->name()] = node->requested_device();
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

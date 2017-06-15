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

using shape_inference::InferenceContext;
using shape_inference::ShapeAndType;
using shape_inference::ShapeHandle;

namespace {

// Merges shapes <shapes_and_types>, determined from an EnqueueV2 node, into
// <*queue_shapes_and_types>.
Status MergeEnqueueShapesAndTypes(
    const std::vector<ShapeAndType>& shapes_and_types, InferenceContext* qctx,
    std::vector<ShapeAndType>* queue_shapes_and_types) {
  if (shapes_and_types.size() != queue_shapes_and_types->size()) {
    return errors::InvalidArgument(
        "Enqueue nodes mixed number of tensors: ", shapes_and_types.size(),
        "  vs ", queue_shapes_and_types->size());
  }
  for (int i = 0; i < shapes_and_types.size(); ++i) {
    const ShapeAndType& a = shapes_and_types[i];
    ShapeAndType& b = (*queue_shapes_and_types)[i];
    if (a.dtype != b.dtype) {
      return errors::InvalidArgument("Enqueue nodes mixed dtypes for tensor ",
                                     i, ": ", DataTypeString(a.dtype), " vs ",
                                     DataTypeString(b.dtype));
    }

    TF_RETURN_IF_ERROR(qctx->Merge(a.shape, b.shape, &b.shape));
  }
  return Status::OK();
}

}  // namespace

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

      // Check to see if the shape is fully defined.
      auto* queue_handle_data = qctx->output_handle_shapes_and_types(0);
      if (queue_handle_data != nullptr) {
        bool fully_defined = true;
        for (const auto& shape_and_type : *queue_handle_data) {
          if (!qctx->FullyDefined(shape_and_type.shape) ||
              shape_and_type.dtype == DT_INVALID) {
            fully_defined = false;
          }
        }
        if (fully_defined) {
          continue;
        }
      }

      std::vector<ShapeAndType> queue_shapes_and_types;
      if (queue_handle_data != nullptr) {
        queue_shapes_and_types = *queue_handle_data;
      }
      for (const auto& node : resource_data.second) {
        auto ctx = shape_refiner.GetContext(node);
        if (!ctx) {
          continue;
        }
        // TODO(bsteiner): handle EnqueueMany as well.
        if (node->type_string().find("Enqueue") != std::string::npos &&
            node->type_string().find("EnqueueMany") == std::string::npos) {
          std::vector<ShapeAndType> shapes_and_types;
          for (int i = 1; i < ctx->num_inputs(); ++i) {
            shapes_and_types.push_back({ctx->input(i), node->input_type(i)});
          }

          if (queue_shapes_and_types.empty()) {
            queue_shapes_and_types = shapes_and_types;
          } else {
            TF_RETURN_IF_ERROR(MergeEnqueueShapesAndTypes(
                shapes_and_types, qctx, &queue_shapes_and_types));
          }
        }
      }
      if (!queue_shapes_and_types.empty() &&
          qctx->MergeOutputHandleShapesAndTypes(0, queue_shapes_and_types)) {
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
      ShapeHandle shp = ctx->input(i);
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
      ShapeHandle shp = ctx->output(i);
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
  }

  return Status::OK();
}

Status GraphProperties::InferDynamically(Cluster* cluster) {
  TF_RETURN_IF_ERROR(cluster->Initialize(item_));

  // Runs the model once to collect the shapes in the cost model.
  RunMetadata metadata;
  TF_RETURN_IF_ERROR(
      cluster->Run(item_.graph, item_.feed, item_.fetch, &metadata));

  return InferFromCostGraph(metadata.cost_graph());
}

Status GraphProperties::InferFromCostGraph(const CostGraphDef& cost_graph) {
  std::unordered_map<string, const CostGraphDef::Node*> name_to_cost;
  std::unordered_map<string, const NodeDef*> name_to_node;  // Empty
  for (auto& node : cost_graph.node()) {
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
  }
  return Status::OK();
}

bool GraphProperties::HasOutputProperties(const string& name) const {
  return output_properties_.find(name) != output_properties_.end();
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

}  // end namespace grappler
}  // end namespace tensorflow

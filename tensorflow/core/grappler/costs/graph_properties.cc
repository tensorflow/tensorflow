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

// If a Merge node has a NextIteration node as an input then that input will
// try to forward an UnknownShape at graph construction time. However, the
// Merge shape function will always propagate an UnknownShape if any of its
// inputs are UnknownShapes. So we need to ignore the input from NextIteration
// nodes to propagate any known shape from the Merge node.
Status ShapeOfMergeNode(const Node* node, InferenceContext* c) {
  ShapeHandle out = c->input(0);
  if (!c->RankKnown(out)) {
    out = c->UnknownShape();
  } else {
    int32 rank = c->Rank(out);
    for (const Edge* e : node->in_edges()) {
      if (e->src()->IsNextIteration() || e->dst_input() <= 0) {
        continue;
      }
      ShapeHandle input = c->input(e->dst_input());
      if (!c->RankKnown(input) || c->Rank(input) != rank) {
        out = c->UnknownShape();
        break;
      }

      for (int d = 0; d < rank; ++d) {
        if (c->Value(c->Dim(input, d)) != c->Value(c->Dim(out, d))) {
          TF_RETURN_IF_ERROR(c->ReplaceDim(out, d, c->UnknownDim(), &out));
        }
      }
    }
  }
  c->set_output(0, out);
  c->set_output(1, c->Scalar());
  return Status::OK();
}

// Manually propagate the input shape for Enter nodes and update any Merge node
// outputs.
Status UpdateEnter(ShapeRefiner* shape_refiner, const Node* node, bool relax,
                   std::queue<const Node*>* new_shapes) {
  auto enter_ctx = shape_refiner->GetContext(node);
  for (int i = 0; i < enter_ctx->num_outputs(); i++) {
    TF_RETURN_IF_ERROR(shape_refiner->SetShape(node, i, enter_ctx->input(0)));
  }
  for (const Edge* e : node->out_edges()) {
    Node* dst = e->dst();
    if (dst->IsMerge()) {
      bool updated = false;
      TF_RETURN_IF_ERROR(shape_refiner->UpdateNode(dst, relax, &updated));
      if (!updated) {
        continue;
      }
      InferenceContext* merge_ctx = shape_refiner->GetContext(dst);
      DCHECK_NE(merge_ctx, nullptr);
      TF_RETURN_IF_ERROR(ShapeOfMergeNode(dst, merge_ctx));
      new_shapes->push(dst);
    }
  }
  return Status::OK();
}

// Propagates the shapes in the transitive fan-out of <new_shapes>.
Status PropagateShapes(ShapeRefiner* shape_refiner, bool relax,
                       std::queue<const Node*>* new_shapes) {
  while (!new_shapes->empty()) {
    const Node* n = new_shapes->front();
    new_shapes->pop();
    for (const Node* fanout : n->out_nodes()) {
      bool updated = false;
      TF_RETURN_IF_ERROR(shape_refiner->UpdateNode(fanout, relax, &updated));
      if (fanout->IsEnter()) {
        TF_RETURN_IF_ERROR(
            UpdateEnter(shape_refiner, fanout, relax, new_shapes));
      } else if (updated) {
        // We want to avoid propagating through loops on the merge pass because
        // the shapes are not guaranteed to converge.
        if (!relax && fanout->IsNextIteration()) {
          continue;
        }
        new_shapes->push(fanout);
      }
    }
  }
  return Status::OK();
}

}  // namespace

void GraphProperties::Relax(InferenceContext* c, ShapeHandle s0, ShapeHandle s1,
                            ShapeHandle* out) {
  c->Relax(s0, s1, out);
}

bool GraphProperties::SameDefinedShape(InferenceContext* c, ShapeHandle s0,
                                       ShapeHandle s1) {
  return ShapeRefiner::SameDefinedShape(c, s0, s1);
}

bool GraphProperties::IsUpdatedShapesOrTypes(
    InferenceContext* c, const std::vector<ShapeAndType>& existing,
    const std::vector<ShapeAndType>& updated) {
  return ShapeRefiner::IsUpdatedShapesOrTypes(c, existing, updated);
}

Status GraphProperties::MergeEnqueueShapesAndTypes(
    const std::vector<ShapeAndType>& shapes_and_types, InferenceContext* qctx,
    std::vector<ShapeAndType>* queue_shapes_and_types) {
  if (shapes_and_types.size() != queue_shapes_and_types->size()) {
    return errors::InvalidArgument(
        "Enqueue nodes mixed number of tensors: ", shapes_and_types.size(),
        "  vs ", queue_shapes_and_types->size());
  }
  for (size_t i = 0; i < shapes_and_types.size(); ++i) {
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

Status GraphProperties::RelaxEnqueueShapesAndMergeTypes(
    const std::vector<ShapeAndType>& shapes_and_types, InferenceContext* qctx,
    std::vector<ShapeAndType>* queue_shapes_and_types) {
  if (shapes_and_types.size() != queue_shapes_and_types->size()) {
    return errors::InvalidArgument(
        "Enqueue nodes mixed number of tensors: ", shapes_and_types.size(),
        "  vs ", queue_shapes_and_types->size());
  }
  for (size_t i = 0; i < shapes_and_types.size(); ++i) {
    const ShapeAndType& a = shapes_and_types[i];
    ShapeAndType& b = (*queue_shapes_and_types)[i];
    if (a.dtype != b.dtype) {
      return errors::InvalidArgument("Enqueue nodes mixed dtypes for tensor ",
                                     i, ": ", DataTypeString(a.dtype), " vs ",
                                     DataTypeString(b.dtype));
    }

    Relax(qctx, a.shape, b.shape, &b.shape);
  }
  return Status::OK();
}

Status GraphProperties::InferStatically() {
  Graph graph(OpRegistry::Global());
  ShapeRefiner shape_refiner(graph.versions(), graph.op_registry());
  shape_refiner.set_require_shape_inference_fns(false);
  ImportGraphDefOptions options;
  Status s = ImportGraphDef(options, item_.graph, &graph, &shape_refiner);
  TF_RETURN_IF_ERROR(s);

  // List the resources and the nodes using them. Also collect the Enter and
  // Merge nodes.
  std::unordered_map<const Node*, std::unordered_set<const Node*>> resources;
  std::unordered_set<const Node*> enter_nodes;
  std::unordered_set<const Node*> merge_nodes;
  for (const Node* const node : graph.nodes()) {
    for (int i = 0; i < node->num_inputs(); ++i) {
      if (node->input_type(i) == DataType::DT_RESOURCE) {
        const Node* resource;
        TF_CHECK_OK(node->input_node(i, &resource));
        resources[resource].insert(node);
      }
    }
    if (node->IsEnter()) {
      enter_nodes.insert(node);
    } else if (node->IsNextIteration()) {
      for (const Node* output : node->out_nodes()) {
        if (output->IsMerge()) {
          merge_nodes.insert(output);
        }
      }
    }
  }

  // Propagate the initial shapes of Enter nodes manually (the Enter shape
  // function always forwards an UnknownShape).
  std::queue<const Node*> new_shapes;
  for (const Node* node : enter_nodes) {
    TF_RETURN_IF_ERROR(
        UpdateEnter(&shape_refiner, node, false /* relax */, &new_shapes));
  }
  TF_RETURN_IF_ERROR(
      PropagateShapes(&shape_refiner, false /* relax */, &new_shapes));

  // We propagate shapes through the graph in two phases. In the first phase, we
  // exclusively merge shapes but we do not propagate shapes through loops. Then
  // on the second phase, we exclusively relax shapes and propagate shapes
  // through loops until reaching fixed point.
  for (int relax = 0; relax < 2; relax++) {
    // We don't update Merge nodes with the input of NextIteration nodes on the
    // merge pass. So we do that at the beginning of the relax pass instead.
    if (relax) {
      bool updated = false;
      for (const Node* node : merge_nodes) {
        TF_RETURN_IF_ERROR(
            shape_refiner.UpdateNode(node, false /* relax */, &updated));
      }
    }

    bool done = true;
    do {
      if (relax) {
        // Propagate shapes through any loops in the graph by relaxing.
        for (const Node* node : merge_nodes) {
          new_shapes.push(node);
        }
        TF_RETURN_IF_ERROR(PropagateShapes(&shape_refiner, relax, &new_shapes));
      }

      // If we found a resource, try to propagate the shapes through it.
      new_shapes = std::queue<const Node*>();
      for (const auto& resource_data : resources) {
        const Node* qnode = resource_data.first;
        StringPiece type(qnode->type_string());
        if (!type.ends_with("QueueV2") && !qnode->IsEnter()) {
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
          // If we are merging, then we are done. If we are relaxing, then we
          // could potentially propagate a less specific shape.
          if (fully_defined && !relax) {
            continue;
          }
        }

        // Merge all inputs into the enqueue node, regardless of which phase we
        // are in.
        std::vector<ShapeAndType> queue_shapes_and_types;
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
        // Combine the input shapes with the existing output shape. We either
        // merge or relax depending on which phase we are in.
        if (queue_handle_data != nullptr) {
          if (relax) {
            TF_RETURN_IF_ERROR(RelaxEnqueueShapesAndMergeTypes(
                *queue_handle_data, qctx, &queue_shapes_and_types));
          } else {
            TF_RETURN_IF_ERROR(MergeEnqueueShapesAndTypes(
                *queue_handle_data, qctx, &queue_shapes_and_types));
          }
        }
        // Set the output ShapeAndType handles. If we successfully update the
        // resource node, add its fan-out to the queue.
        const std::vector<ShapeAndType>* outputs =
            qctx->output_handle_shapes_and_types(0);
        std::vector<ShapeAndType> existing_outputs;
        if (outputs) {
          existing_outputs = *outputs;
        }
        if (!queue_shapes_and_types.empty()) {
          if (!relax && qctx->MergeOutputHandleShapesAndTypes(
                            0, queue_shapes_and_types)) {
            new_shapes.push(qnode);
          } else if (relax && qctx->RelaxOutputHandleShapesAndMergeTypes(
                                  0, queue_shapes_and_types)) {
            if (IsUpdatedShapesOrTypes(
                    qctx, existing_outputs,
                    *qctx->output_handle_shapes_and_types(0))) {
              new_shapes.push(qnode);
            }
          }
        }
      }
      // Propagate the shapes in the transitive fan-out of the queue.
      done = new_shapes.empty();
      if (!done) {
        TF_RETURN_IF_ERROR(PropagateShapes(&shape_refiner, relax, &new_shapes));
      }
    } while (!done);
  }

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

bool GraphProperties::HasInputProperties(const string& name) const {
  return input_properties_.find(name) != input_properties_.end();
}

bool GraphProperties::HasOutputProperties(const string& name) const {
  return output_properties_.find(name) != output_properties_.end();
}

const std::vector<OpInfo::TensorProperties>&
GraphProperties::GetInputProperties(const string& node_name) const {
  auto it = input_properties_.find(node_name);
  if (it != input_properties_.end()) {
    return it->second;
  }
  return missing_properties_;
}

const std::vector<OpInfo::TensorProperties>&
GraphProperties::GetOutputProperties(const string& node_name) const {
  auto it = output_properties_.find(node_name);
  if (it != output_properties_.end()) {
    return it->second;
  }
  return missing_properties_;
}

}  // end namespace grappler
}  // end namespace tensorflow

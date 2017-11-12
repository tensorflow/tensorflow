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

using shape_inference::Dimension;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::Shape;
using shape_inference::ShapeAndType;
using shape_inference::ShapeHandle;

namespace {

template <typename Handle>
struct HashHandle {
  std::size_t operator()(const Handle& h) const { return h.Handle(); }
};
template <typename Handle>
struct CompareHandle {
  bool operator()(const Handle& h1, const Handle& h2) const {
    return h1.SameHandle(h2);
  }
};

template <typename Handle>
struct HandleToObject {};
template <>
struct HandleToObject<ShapeHandle> {
  typedef ShapeHandle Object;

  static ShapeHandle Unknown() { return ShapeHandle(); }
};

template <>
struct HandleToObject<DimensionHandle> {
  typedef int64 Object;

  static int64 Unknown() { return -1; }
};

template <typename Handle>
struct Processor {};

template <>
struct Processor<ShapeHandle> {
  // Extract the shape or dim denoted by the handle.
  void ExtractValue(ShapeHandle h, ShapeHandle* result) { *result = h; }
  // Merge the shapes or dims.
  Status Merge(ShapeHandle h1, ShapeHandle h2, ShapeHandle* result) {
    if (InferenceContext::RankKnown(*result)) {
      // The result was initialized in a previous merge to a shape of known
      // rank, make sure we preserve that information.
      return Status::OK();
    }
    if (InferenceContext::RankKnown(h1)) {
      *result = h1;
    } else {
      *result = h2;
    }
    return Status::OK();
  }
};

template <>
struct Processor<DimensionHandle> {
  // Assign a negative id to unknown dimensions, starting at -2 (the -1 id
  // reserved by TensorFlow).
  void ExtractValue(DimensionHandle d, int64* result) {
    if (!InferenceContext::ValueKnown(d)) {
      *result = -counter;
      counter++;
    } else {
      CHECK_LE(0, InferenceContext::Value(d));
      *result = InferenceContext::Value(d);
    }
  }

  // Merge the dimensions d1 and d2. Return the known shape if there is one,
  // otherwise look for a symbolic shape. If there is no symbolic shape and no
  // known shape, the shape if fully unknown so return -1.
  Status Merge(DimensionHandle d1, DimensionHandle d2, int64* result) {
    const int64 dim1 = InferenceContext::Value(d1);
    const int64 dim2 = InferenceContext::Value(d2);

    if (dim1 >= 0 && dim2 >= 0) {
      CHECK_EQ(dim1, dim2);
      return RefineDim(dim1, result);
    } else if (dim1 >= 0 && dim2 < 0) {
      return RefineDim(dim1, result);
    } else if (dim1 < 0 && dim2 >= 0) {
      return RefineDim(dim2, result);
    } else if (dim1 < -1) {
      return RefineDim(dim1, result);
    } else if (dim2 < -1) {
      return RefineDim(dim2, result);
    } else {
      CHECK_EQ(dim1, dim2);
      CHECK_EQ(-1, dim1);
      return RefineDim(-1, result);
    }
    return Status::OK();
  }

 private:
  Status RefineDim(int64 dim, int64* result) {
    if (*result >= 0) {
      if (!(*result == dim || dim < 0)) {
        return errors::InvalidArgument("Inconsistent dimensions detected");
      }
    } else if (dim >= 0) {
      *result = dim;
    } else if (dim < *result) {
      *result = dim;
    }
    return Status::OK();
  }

  int64 counter = 2;
};

// Traditional Disjoint-Set datastructure with path compression.
// (https://en.wikipedia.org/wiki/Disjoint-set_data_structure)
template <typename Handle>
class DisjointSet {
 public:
  DisjointSet(const Processor<Handle>& processor) : processor_(processor) {}
  ~DisjointSet() {
    for (auto rep : nodes_) {
      delete rep.second;
    }
  }

  Status Merge(Handle x, Handle y);
  const typename HandleToObject<Handle>::Object GetMergedValue(Handle value);

 private:
  // All the handles that belong to the same set are part of the same tree, and
  // utimately represented by the root of that tree.
  struct Rep {
    // Parent in the tree used to encode the set.
    Rep* parent;
    // Rank in the tree, used to figure out how to compress the path to the root
    // of the tree.
    int rank;
    // The handle.
    typename HandleToObject<Handle>::Object value;
  };

  // Create a new set for the value if none exists, or return its representative
  // node otherwise.
  Rep* Find(Handle value);

 private:
  Processor<Handle> processor_;
  std::unordered_map<Handle, Rep*, HashHandle<Handle>, CompareHandle<Handle>>
      nodes_;
};

template <typename Handle>
const typename HandleToObject<Handle>::Object
DisjointSet<Handle>::GetMergedValue(Handle value) {
  Rep* rep = Find(value);
  if (!rep) {
    // We don't know anything about this handle.
    return HandleToObject<Handle>::Unknown();
  }
  return rep->value;
}

template <typename Handle>
Status DisjointSet<Handle>::Merge(Handle x, Handle y) {
  Rep* x_root = Find(x);
  Rep* y_root = Find(y);

  // x and y are already in the same set
  if (x_root == y_root) {
    return Status::OK();
  }
  // x and y are not in same set, so we merge them
  // Use the occasion to strengthen what we know about the handle by merging the
  // information about the 2 subsets.
  if (x_root->rank < y_root->rank) {
    TF_RETURN_IF_ERROR(processor_.Merge(y, x, &y_root->value));
    x_root->parent = y_root;
  } else if (x_root->rank > y_root->rank) {
    TF_RETURN_IF_ERROR(processor_.Merge(x, y, &x_root->value));
    y_root->parent = x_root;
  } else {
    TF_RETURN_IF_ERROR(processor_.Merge(x, y, &x_root->value));
    // Arbitrarily make one root the new parent
    y_root->parent = x_root;
    x_root->rank = x_root->rank + 1;
  }
  return Status::OK();
}

template <typename Handle>
typename DisjointSet<Handle>::Rep* DisjointSet<Handle>::Find(Handle value) {
  auto it = nodes_.find(value);
  if (it == nodes_.end()) {
    // This is the first time we process this handle, create an entry for it.
    Rep* node = new Rep;
    node->parent = node;
    node->rank = 0;
    processor_.ExtractValue(value, &node->value);
    nodes_[value] = node;
    return node;
  }
  // Return the representative for the set, which is the root of the tree. Apply
  // path compression to speedup future queries.
  Rep* node = it->second;
  Rep* root = node->parent;
  while (root != root->parent) {
    root = root->parent;
  }
  while (node->parent != root) {
    Rep* next = node->parent;
    node->parent = root;
    node = next;
  }
  return root;
}

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
  CHECK_NE(enter_ctx, nullptr);
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
      CHECK_NE(merge_ctx, nullptr);
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

bool IsQueue(const Node& node) {
  StringPiece type(node.type_string());
  return type.ends_with("QueueV2");
}

// Returns true if the node is an Enter op AND its input is a Queue.
bool IsEnterWithQueue(const Node& node) {
  if (node.IsEnter()) {
    const Node* in_node;
    TF_CHECK_OK(node.input_node(0, &in_node));
    return IsQueue(*in_node);
  }
  return false;
}

}  // namespace

// Keep track of shapes and dimensions in a graph.
// In particular, use disjoint sets to track equivalence between shapes and
// dims, and consolidate the information globally.
class SymbolicShapeManager {
 public:
  SymbolicShapeManager() : shapes_(shape_processor_), dims_(dim_processor_) {}

  Status Merge(ShapeHandle s1, ShapeHandle s2) {
    if (!s1.IsSet() || !s2.IsSet()) {
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(shapes_.Merge(s1, s2));
    if (InferenceContext::Rank(s1) > 0 && InferenceContext::Rank(s2) > 0) {
      CHECK_EQ(InferenceContext::Rank(s1), InferenceContext::Rank(s2));
      for (int i = 0; i < InferenceContext::Rank(s1); ++i) {
        TF_RETURN_IF_ERROR(dims_.Merge(InferenceContext::DimKnownRank(s1, i),
                                       InferenceContext::DimKnownRank(s2, i)));
      }
    }
    return Status::OK();
  }
  Status Merge(DimensionHandle d1, DimensionHandle d2) {
    if (!d1.IsSet() || !d2.IsSet()) {
      return Status::OK();
    }
    return dims_.Merge(d1, d2);
  }

  void AsTensorProperties(const ShapeHandle& shape, const DataType& type,
                          OpInfo::TensorProperties* properties) {
    properties->set_dtype(type);
    ShapeHandle actual_shape = shapes_.GetMergedValue(shape);
    if (!InferenceContext::RankKnown(actual_shape)) {
      properties->mutable_shape()->set_unknown_rank(true);
    } else {
      for (int j = 0; j < InferenceContext::Rank(actual_shape); ++j) {
        shape_inference::DimensionHandle dim =
            InferenceContext::DimKnownRank(actual_shape, j);
        int64 d = dims_.GetMergedValue(dim);
        properties->mutable_shape()->add_dim()->set_size(d);
      }
    }
  }

 private:
  Processor<ShapeHandle> shape_processor_;
  DisjointSet<shape_inference::ShapeHandle> shapes_;
  Processor<DimensionHandle> dim_processor_;
  DisjointSet<shape_inference::DimensionHandle> dims_;
};

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
  FunctionLibraryDefinition function_library(graph.op_registry(),
                                             item_.graph.library());
  ShapeRefiner shape_refiner(graph.versions(), graph.op_registry());
  shape_refiner.set_require_shape_inference_fns(false);
  shape_refiner.set_disable_constant_propagation(true);
  shape_refiner.set_function_library_for_shape_inference(&function_library);
  ImportGraphDefOptions options;
  // Graph optimization happens at the late stage of graph execution,
  // when colocation constraints are already validated previously and
  // the device placement of nodes has also completed, so there
  // is no need to validate colocation constraints again.
  options.validate_colocation_constraints = false;
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
        // Proceed only if qnode is a queue or an Enter with queue input.
        if (!IsQueue(*qnode) && !IsEnterWithQueue(*qnode)) {
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

  std::unordered_map<const shape_inference::Dimension*, int> dim_ids;

  // Track shapes globally accross the graph.
  SymbolicShapeManager shape_manager;
  bool found_error = false;
  for (const Node* const node : graph.nodes()) {
    auto node_ctx = shape_refiner.GetContext(node);
    if (!node_ctx) {
      continue;
    }
    for (const auto& merged_shapes : node_ctx->MergedShapes()) {
      if (!shape_manager.Merge(merged_shapes.first, merged_shapes.second)
               .ok()) {
        found_error = true;
        break;
      }
    }
    for (const auto& merged_dims : node_ctx->MergedDims()) {
      if (!shape_manager.Merge(merged_dims.first, merged_dims.second).ok()) {
        found_error = true;
        break;
      }
    }
    if (found_error) {
      // The shapes aren't consistent, we can't infer safely: discard all the
      // information discovered so far.
      shape_manager = SymbolicShapeManager();
      break;
    }
  }

  for (const Node* const node : graph.nodes()) {
    VLOG(1) << "<Node> " << node->name();
    auto ctx = shape_refiner.GetContext(node);
    if (!ctx) {
      continue;
    }

    // Fill input properties.
    {
      CHECK_EQ(ctx->num_inputs(), node->num_inputs());
      auto& input_properties = input_properties_[node->name()];

      // Should always be empty, node names in graph are supposed to be unique.
      CHECK_EQ(input_properties.size(), 0);

      input_properties.resize(ctx->num_inputs());
      for (int i = 0; i < ctx->num_inputs(); ++i) {
        shape_manager.AsTensorProperties(ctx->input(i), node->input_type(i),
                                         &input_properties[i]);
      }
      for (const auto& edge : node->in_edges()) {
        if (!edge->src()->IsConstant()) {
          continue;
        }
        const int input_id = edge->dst_input();
        if (input_id >= input_properties.size()) {
          continue;
        }
        const NodeDef& node = edge->src()->def();
        const TensorProto& raw_val = node.attr().at("value").tensor();
        *input_properties[input_id].mutable_value() = raw_val;
      }
    }

    // Fill output properties.
    {
      CHECK_EQ(ctx->num_outputs(), node->num_outputs());
      auto& output_properties = output_properties_[node->name()];

      // Should always be empty, node names in graph are supposed to be unique.
      CHECK_EQ(output_properties.size(), 0);

      output_properties.resize(ctx->num_outputs());
      for (int i = 0; i < ctx->num_outputs(); ++i) {
        shape_manager.AsTensorProperties(ctx->output(i), node->output_type(i),
                                         &output_properties[i]);
      }
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

  return InferFromCostGraph(metadata.cost_graph());
}

Status GraphProperties::AnnotateOutputShapes(GraphDef* output_graph_def) const {
  *output_graph_def = item_.graph;
  for (int i = 0; i < output_graph_def->node_size(); i++) {
    auto node = output_graph_def->mutable_node(i);
    AttrValue attr_output_shape;
    auto tensor_properties = GetOutputProperties(node->name());
    for (const auto& tensor_property : tensor_properties) {
      *attr_output_shape.mutable_list()->add_shape() = tensor_property.shape();
    }
    (*node->mutable_attr())["_output_shapes"] = attr_output_shape;
  }
  return Status::OK();
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

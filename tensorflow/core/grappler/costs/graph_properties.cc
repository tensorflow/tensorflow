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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace grappler {
namespace {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeAndType;
using shape_inference::ShapeHandle;

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
      int64 val = InferenceContext::Value(d);
      if (val >= 0) {
        *result = val;
      } else {
        // A shape inference function generated an invalid dimension handle.
        // Use a symbolic dimension to encode this.
        *result = -counter;
        counter++;
      }
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
  DisjointSet() {}
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

bool IsQueue(const NodeDef& node) {
  return str_util::EndsWith(node.op(), "QueueV2");
}

// Returns true if the node is an Enter op AND its input is a Queue.
bool IsEnterWithQueue(const NodeDef& node, const GraphView& graph) {
  if (IsEnter(node)) {
    GraphView::InputPort input(&node, 0);
    GraphView::OutputPort fanin = graph.GetRegularFanin(input);
    return IsQueue(*fanin.node);
  }
  return false;
}

bool HasAnyUnknownDimensions(const TensorShapeProto& proto) {
  if (proto.unknown_rank()) {
    return true;
  }
  for (const auto& dim : proto.dim()) {
    if (dim.size() < 0) {
      return true;
    }
  }
  return false;
}

// This really should be done in an external debugging tool
void VerboseLogUnknownDimensionSources(
    const GraphDef& graph,
    const std::map<string, std::vector<OpInfo::TensorProperties>>&
        input_properties_map,
    const std::map<string, std::vector<OpInfo::TensorProperties>>&
        output_properties_map) {
  if (!VLOG_IS_ON(2)) {
    return;
  }

  VLOG(2) << "Nodes with known inputs, but with unknown output dimensions:";

  // Find all nodes in the graph for which we
  // do not have any unknown dimensions in their inputs, but
  // we have some unknown dimensions in their outputs.
  std::map<string, int> op_to_count;
  for (const NodeDef& node : graph.node()) {
    const auto& input_properties = input_properties_map.at(node.name());
    const auto& output_properties = output_properties_map.at(node.name());

    bool has_unknown_inputs = false;
    for (const auto& input_prop : input_properties) {
      if (HasAnyUnknownDimensions(input_prop.shape())) {
        has_unknown_inputs = true;
        break;
      }
    }

    if (has_unknown_inputs) {
      continue;
    }

    for (const auto& output_prop : output_properties) {
      if (HasAnyUnknownDimensions(output_prop.shape())) {
        string inputs = "input_shapes=[";
        for (const auto& input_prop : input_properties) {
          inputs += PartialTensorShape::DebugString(input_prop.shape());
        }
        inputs += "]";

        string outputs = "output_shapes=[";
        for (const auto& output_prop : output_properties) {
          outputs += PartialTensorShape::DebugString(output_prop.shape());
        }
        outputs += "]";

        VLOG(2) << "Node: " << node.name() << ", Op: " << node.op() << ", "
                << inputs << ", " << outputs;

        op_to_count[node.op()]++;

        // don't log again for this node
        break;
      }
    }
  }
  VLOG(2) << "Op types with known inputs, but with unknown output dimensions "
          << "(format: <op_type> (<count>)):";
  for (const auto& p : op_to_count) {
    VLOG(2) << p.first << " (" << p.second << ")";
  }
}

}  // namespace

// Queue of nodes to process. Nodes can be enqueued in any order, but will be
// dequeued in (roughly) topological order. Propagating shapes following a
// topological ordering isn't required for correctness but helps speed things up
// since it avoids processing the same node multiple times as its inputs
// information is refined.
class TopoQueue {
 public:
  explicit TopoQueue(const std::unordered_map<const NodeDef*, int>& topo_order)
      : queue_(CompareNodes(topo_order)) {}
  void push(const NodeDef* n) { queue_.insert(n); }
  const NodeDef* pop() {
    CHECK(!empty());
    auto it = queue_.begin();
    const NodeDef* n = *it;
    queue_.erase(it);
    return n;
  }

  bool empty() const { return queue_.empty(); }
  std::size_t size() const { return queue_.size(); }

 private:
  // Graph nodes are created in (roughly) topological order. Therefore we can
  // use their id to ensure they're sorted topologically.
  struct CompareNodes {
    explicit CompareNodes(
        const std::unordered_map<const NodeDef*, int>& topo_ordering)
        : topo_order(topo_ordering) {}
    bool operator()(const NodeDef* lhs, const NodeDef* rhs) const {
      return topo_order.at(lhs) < topo_order.at(rhs);
    }

   private:
    const std::unordered_map<const NodeDef*, int>& topo_order;
  };
  std::set<const NodeDef*, CompareNodes> queue_;
};

// Merge and relax symbolic shapes.
// Each symbolic shape or dimension is represented by a handle. Unlike the TF
// shape refiner which creates new handles every time it processes an unknown
// shape/dimension, the symbolic shape refiner assigns a specific handle to each
// unknown shape/dimension of a given node.
class SymbolicShapeRefiner {
 public:
  explicit SymbolicShapeRefiner(
      const GraphView& graph,
      const std::unordered_map<string, std::unordered_set<int>>& fed_ports)
      : graph_(graph),
        function_library_(OpRegistry::Global(), graph.GetGraph()->library()),
        fed_ports_(fed_ports) {
    graph_def_version_ = graph.GetGraph()->versions().producer();
    node_to_context_.reserve(graph.GetGraph()->node_size());
  }

  const GraphView& graph() const { return graph_; }

  struct NodeContext {
    const OpRegistrationData* op_data;
    DataTypeVector input_types;
    DataTypeVector output_types;
    std::unique_ptr<InferenceContext> inference_context;
    std::vector<ShapeHandle> output_tensors_as_shapes;
  };

  NodeContext* GetNodeContext(const NodeDef* node) {
    auto it = node_to_context_.find(node);
    if (it == node_to_context_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  InferenceContext* GetContext(const NodeDef* node) {
    auto it = node_to_context_.find(node);
    if (it == node_to_context_.end()) {
      return nullptr;
    }
    return it->second.inference_context.get();
  }
  Status UpdateNode(const NodeDef* node, bool relax, bool* refined) {
    NodeContext* node_context = GetNodeContext(node);
    if (node_context == nullptr) {
      TF_RETURN_IF_ERROR(AddNode(node));
      node_context = CHECK_NOTNULL(GetNodeContext(node));
      *refined = true;
    }
    // Check if the shapes of the nodes in the fan-in of this node have changed,
    // and if they have, update the node input shapes.
    InferenceContext* inference_context = node_context->inference_context.get();
    std::vector<Tensor> const_values(inference_context->num_inputs());
    std::vector<const Tensor*> input_tensors(inference_context->num_inputs(),
                                             nullptr);
    std::vector<ShapeHandle> input_tensors_as_shapes(
        inference_context->num_inputs());

    for (int dst_input = 0; dst_input < inference_context->num_inputs();
         ++dst_input) {
      GraphView::InputPort port(node, dst_input);
      for (const GraphView::OutputPort fanin : graph_.GetFanin(port)) {
        int src_output = fanin.port_id;
        const NodeDef* input = fanin.node;
        NodeContext* c = GetNodeContext(input);
        if (c == nullptr) {
          return errors::FailedPrecondition(
              "Input ", dst_input, " ('", input->name(), "') for '",
              node->name(), "' was not previously added to ShapeRefiner.");
        }

        if (IsConstant(*input)) {
          // Convert constant value into tensors.
          if (const_values[dst_input].FromProto(
                  input->attr().at("value").tensor())) {
            input_tensors[dst_input] = &const_values[dst_input];
            // Integer tensors of rank one can also be interpreted as a shape
            // provided all their values are >= -1.
            if (const_values[dst_input].dims() == 1 &&
                (const_values[dst_input].dtype() == DT_INT32 ||
                 const_values[dst_input].dtype() == DT_INT64)) {
              ShapeHandle tensor_shape = inference_context->Vector(
                  const_values[dst_input].NumElements());
              ShapeHandle shp;
              if (inference_context
                      ->MakeShapeFromTensor(input_tensors[dst_input],
                                            tensor_shape, &shp)
                      .ok()) {
                input_tensors_as_shapes[dst_input] = shp;
              }
            }
          }
        }

        if (c->output_tensors_as_shapes.size() > src_output) {
          input_tensors_as_shapes[dst_input] =
              c->output_tensors_as_shapes[src_output];
        }

        DCHECK_GE(dst_input, 0);
        if (!*refined && !inference_context->input(dst_input).SameHandle(
                             c->inference_context->output(src_output))) {
          *refined = true;
        }
        inference_context->SetInput(dst_input,
                                    c->inference_context->output(src_output));

        if (!*refined &&
            inference_context->requested_input_tensor_as_partial_shape(
                dst_input)) {
          // The input value may have changed. Since we have no way to know if
          // that's indeed the case, err on the safe side.
          *refined = true;
        }

        // Also propagate handle shape and dtype of edges which are carrying
        // resource handles.
        if (node_context->input_types[dst_input] == DT_RESOURCE) {
          auto* outputs =
              c->inference_context->output_handle_shapes_and_types(src_output);
          if (!outputs) continue;
          auto* inputs =
              inference_context->input_handle_shapes_and_types(dst_input);

          if (!inputs || !EquivalentShapesAndTypes(*outputs, *inputs)) {
            *refined = true;
          }
          inference_context->set_input_handle_shapes_and_types(dst_input,
                                                               *outputs);
        }
      }
    }

    if (!*refined) {
      // No input shape has changed, we're done
      return Status::OK();
    }

    node_context->inference_context->set_input_tensors(input_tensors);
    node_context->inference_context->set_input_tensors_as_shapes(
        input_tensors_as_shapes);

    // Update the shapes of the outputs.
    return InferShapes(*node, node_context);
  }

  Status SetUnknownShape(const NodeDef* node, int output_port) {
    shape_inference::ShapeHandle shape =
        GetUnknownOutputShape(node, output_port);
    InferenceContext* ctx = GetContext(node);
    if (ctx == nullptr) {
      return errors::InvalidArgument("Missing context");
    }
    ctx->set_output(output_port, shape);
    return Status::OK();
  }

  struct ShapeId {
    const NodeDef* node;
    int port_id;
    bool operator==(const ShapeId& other) const {
      return node == other.node && port_id == other.port_id;
    }
  };
  struct HashShapeId {
    std::size_t operator()(const ShapeId& shp) const {
      return std::hash<const NodeDef*>{}(shp.node) + shp.port_id;
    }
  };

  struct DimId {
    const NodeDef* node;
    int port_id;
    int dim_index;
    bool operator==(const DimId& other) const {
      return node == other.node && port_id == other.port_id &&
             dim_index == other.dim_index;
    }
  };

  struct HashDimId {
    std::size_t operator()(const DimId& dim) const {
      return std::hash<const NodeDef*>{}(dim.node) + dim.port_id +
             dim.dim_index;
    }
  };

  // Compute the shape of the tensors outputed by node 'node' at output port
  // 'port_index' as the intersection of shape1 and shape2.
  ShapeHandle OutputAsIntersection(const NodeDef* node, int port_index,
                                   ShapeHandle shape1, ShapeHandle shape2) {
    if (shape1.SameHandle(shape2)) {
      return shape1;
    }
    InferenceContext* ctx = GetContext(node);
    ShapeHandle merged = shape1;
    if (!ctx->RankKnown(shape2) && !ctx->RankKnown(shape1)) {
      // Return either one since they're expected to represent the same value.
      return shape1;
    } else if (!ctx->RankKnown(shape2) && ctx->RankKnown(shape1)) {
      return shape1;
    } else if (ctx->RankKnown(shape2) && !ctx->RankKnown(shape1)) {
      return shape2;
    } else {
      const int rank = ctx->Rank(shape1);
      if (ctx->Rank(shape2) != rank) {
        // We detected an inconsistency, return an unknown shape. This can
        // happen in the fanout of a merge node since during the initial
        // propagation we optimistically assume that all the inputs to the merge
        // node have the same shape.
        return GetUnknownOutputShape(node, port_index);
      }
      for (int d = 0; d < rank; ++d) {
        if (!ctx->Dim(shape1, d).SameHandle(ctx->Dim(shape2, d))) {
          if (ctx->Value(ctx->Dim(shape1, d)) !=
              ctx->Value(ctx->Dim(shape2, d))) {
            DimensionHandle new_dim;
            if (ctx->Value(ctx->Dim(shape1, d)) < 0) {
              new_dim = ctx->Dim(shape2, d);
            } else if (ctx->Value(ctx->Dim(shape2, d)) < 0) {
              new_dim = ctx->Dim(shape1, d);
            } else {
              new_dim = GetUnknownOutputDim(node, port_index, d);
            }
            TF_CHECK_OK(ctx->ReplaceDim(merged, d, new_dim, &merged));
          }
        }
      }
    }
    return merged;
  }

  // Compute the shape of the tensors outputed by node 'node' at output port
  // 'port_index' as the union of shape1 and shape2.
  ShapeHandle OutputAsUnion(const NodeDef* node, int port_index,
                            ShapeHandle shape1, ShapeHandle shape2) {
    if (shape1.SameHandle(shape2)) {
      return shape1;
    }
    InferenceContext* ctx = GetContext(node);
    ShapeHandle relaxed = shape1;
    const int rank = ctx->Rank(shape1);
    if (!ctx->RankKnown(shape2) || ctx->Rank(shape2) != rank) {
      relaxed = GetUnknownOutputShape(node, port_index);
    } else {
      for (int d = 0; d < rank; ++d) {
        if (!ctx->Dim(shape1, d).SameHandle(ctx->Dim(shape2, d))) {
          int64 val1 = ctx->Value(ctx->Dim(shape1, d));
          int64 val2 = ctx->Value(ctx->Dim(shape2, d));
          if (val1 != val2 || (val1 < 0 && val2 < 0)) {
            DimensionHandle new_dim = GetUnknownOutputDim(node, port_index, d);
            TF_CHECK_OK(ctx->ReplaceDim(relaxed, d, new_dim, &relaxed));
          }
        }
      }
    }
    return relaxed;
  }

  bool EquivalentShapes(ShapeHandle s1, ShapeHandle s2) const {
    if (s1.SameHandle(s2)) {
      return true;
    }
    if (InferenceContext::Rank(s1) != InferenceContext::Rank(s2)) {
      return false;
    }
    if (!InferenceContext::RankKnown(s1) && !InferenceContext::RankKnown(s2)) {
      return true;
    }
    const int rank = InferenceContext::Rank(s1);
    for (int i = 0; i < rank; ++i) {
      if (!InferenceContext::DimKnownRank(s1, i).SameHandle(
              InferenceContext::DimKnownRank(s2, i))) {
        int64 val1 =
            InferenceContext::Value(InferenceContext::DimKnownRank(s1, i));
        int64 val2 =
            InferenceContext::Value(InferenceContext::DimKnownRank(s2, i));
        if (val1 >= 0 && val2 >= 0 && val1 == val2) {
          continue;
        }
        return false;
      }
    }
    return true;
  }

  bool EquivalentShapesAndTypes(const std::vector<ShapeAndType>& st1,
                                const std::vector<ShapeAndType>& st2) const {
    if (st1.size() != st2.size()) {
      return false;
    }
    for (int i = 0; i < st1.size(); ++i) {
      const ShapeAndType& s1 = st1[i];
      const ShapeAndType& s2 = st2[i];
      if (s1.dtype != s2.dtype) {
        return false;
      }
      if (!EquivalentShapes(s1.shape, s2.shape)) {
        return false;
      }
    }
    return true;
  }

  Status AddNode(const NodeDef* node) {
    NodeContext& node_ctx = node_to_context_[node];
    TF_RETURN_IF_ERROR(function_library_.LookUp(node->op(), &node_ctx.op_data));

    TF_RETURN_IF_ERROR(InOutTypesForNode(*node, node_ctx.op_data->op_def,
                                         &node_ctx.input_types,
                                         &node_ctx.output_types));

    // Create the inference context for this node.
    const int num_inputs = node_ctx.input_types.size();
    std::vector<ShapeHandle> input_shapes(num_inputs);
    std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
        input_handle_shapes_and_types(num_inputs);
    std::vector<const Tensor*> input_tensors(num_inputs, nullptr);
    std::vector<ShapeHandle> input_tensors_as_shapes;

    node_ctx.inference_context.reset(new InferenceContext(
        graph_def_version_, node, node_ctx.op_data->op_def, input_shapes,
        input_tensors, input_tensors_as_shapes,
        std::move(input_handle_shapes_and_types)));
    const Status s = node_ctx.inference_context->construction_status();
    if (!s.ok()) {
      node_ctx.inference_context.reset(nullptr);
    }
    return s;
  }

 private:
  // Return the one ShapeHandle used to denote a fully unknown shape for a node
  // output.
  ShapeHandle GetUnknownOutputShape(const NodeDef* node, int index) {
    ShapeId id{node, index};
    auto it = unknown_shapes_.find(id);
    if (it != unknown_shapes_.end()) {
      return it->second;
    }
    InferenceContext* c = GetContext(node);
    ShapeHandle shp = c->UnknownShape();
    unknown_shapes_[id] = shp;
    return shp;
  }
  // Return the one ShapeHandle used to denote a fully unknown dimension for a
  // node output.
  DimensionHandle GetUnknownOutputDim(const NodeDef* node, int index,
                                      int dim_id) {
    DimId id{node, index, dim_id};
    auto it = unknown_dims_.find(id);
    if (it != unknown_dims_.end()) {
      return it->second;
    }
    InferenceContext* c = GetContext(node);
    DimensionHandle dim = c->UnknownDim();
    unknown_dims_[id] = dim;
    return dim;
  }

  Status InferShapes(const NodeDef& node, NodeContext* c) {
    InferenceContext* ic = c->inference_context.get();

    auto it = fed_ports_.find(node.name());
    const bool is_fed = it != fed_ports_.end();

    // Propagate shape tensors unless the node is fed.
    // TODO(bsteiner) We should still propagate the shapes to the ports that
    // aren't fed in the case of a ShapeN node.
    if (!is_fed) {
      if (IsShape(node)) {
        c->output_tensors_as_shapes.resize(1);
        c->output_tensors_as_shapes[0] = c->inference_context->input(0);
      } else if (IsShapeN(node)) {
        c->output_tensors_as_shapes.resize(c->inference_context->num_inputs());
        for (int i = 0; i < c->inference_context->num_inputs(); ++i) {
          c->output_tensors_as_shapes[i] = c->inference_context->input(i);
        }
      } else if (node.op() == "ConcatV2") {
        bool valid = true;
        ShapeHandle result;
        for (int i = 0; i < ic->num_inputs() - 1; ++i) {
          ShapeHandle input = ic->input_tensors_as_shapes()[i];
          if (!ic->RankKnown(input)) {
            valid = false;
            break;
          } else if (i == 0) {
            result = input;
          } else {
            TF_RETURN_IF_ERROR(ic->Concatenate(result, input, &result));
          }
        }
        if (valid) {
          c->output_tensors_as_shapes.resize(1);
          c->output_tensors_as_shapes[0] = result;
        }
      } else if (IsSlice(node)) {
        ShapeHandle input = ic->input_tensors_as_shapes()[0];
        bool valid = ic->RankKnown(input);
        const Tensor* slice_offset = ic->input_tensor(1);
        valid &= slice_offset != nullptr && slice_offset->NumElements() == 1;
        const Tensor* slice_size = ic->input_tensor(2);
        valid &= slice_size != nullptr && slice_size->NumElements() == 1;
        if (valid) {
          int64 start = slice_offset->dtype() == DT_INT32
                            ? slice_offset->flat<int32>()(0)
                            : slice_offset->flat<int64>()(0);
          int64 end = start + (slice_size->dtype() == DT_INT32
                                   ? slice_size->flat<int32>()(0)
                                   : slice_size->flat<int64>()(0));
          ShapeHandle result;
          TF_RETURN_IF_ERROR(ic->Subshape(input, start, end, &result));
          c->output_tensors_as_shapes.resize(1);
          c->output_tensors_as_shapes[0] = result;
        }
      }
    }

    // Infer the shapes of output tensors.
    if (!c->op_data || c->op_data->shape_inference_fn == nullptr) {
      // There is nothing more we can infer, annotate outputs with unknown
      // shapes
      return c->inference_context->Run(shape_inference::UnknownShape);
    }

    TF_RETURN_IF_ERROR(
        c->inference_context->Run(c->op_data->shape_inference_fn));

    Status status = Status::OK();
    if (is_fed) {
      // It is possible to feed node output ports with tensors of any shape: as
      // a result, the shape of a fed port is completely unknown.
      for (const int output_port : it->second) {
        status.Update(SetUnknownShape(&node, output_port));
      }
    }
    return status;
  }

 private:
  const GraphView& graph_;
  int graph_def_version_;
  std::unordered_map<const NodeDef*, NodeContext> node_to_context_;
  std::unordered_map<ShapeId, ShapeHandle, HashShapeId> unknown_shapes_;
  std::unordered_map<DimId, DimensionHandle, HashDimId> unknown_dims_;
  FunctionLibraryDefinition function_library_;
  const std::unordered_map<string, std::unordered_set<int>>& fed_ports_;
};

// Keep track of shapes and dimensions in a graph.
// In particular, use disjoint sets to track equivalence between shapes and
// dims, and consolidate the information globally.
class SymbolicShapeManager {
 public:
  SymbolicShapeManager() {}

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
  DisjointSet<shape_inference::ShapeHandle> shapes_;
  DisjointSet<shape_inference::DimensionHandle> dims_;
};

Status GraphProperties::MergeEnqueueShapesAndTypes(
    SymbolicShapeRefiner* shape_refiner, const NodeDef* qnode,
    const std::vector<ShapeAndType>& shapes_and_types,
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

    b.shape = shape_refiner->OutputAsIntersection(qnode, i, a.shape, b.shape);
  }
  return Status::OK();
}

Status GraphProperties::RelaxEnqueueShapesAndMergeTypes(
    SymbolicShapeRefiner* shape_refiner, const NodeDef* qnode,
    const std::vector<ShapeAndType>& shapes_and_types,
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

    b.shape = shape_refiner->OutputAsUnion(qnode, i, a.shape, b.shape);
  }
  return Status::OK();
}

// If a Merge node has a NextIteration node as an input then that input will
// try to forward an UnknownShape at graph construction time. However, the
// Merge shape function will always propagate an UnknownShape if any of its
// inputs are UnknownShapes. So we need to ignore the input from NextIteration
// nodes to propagate any known shape from the Merge node.
Status GraphProperties::UpdateMergeNode(SymbolicShapeRefiner* shape_refiner,
                                        const NodeDef* node, bool relax,
                                        bool* new_shapes) const {
  InferenceContext* c = shape_refiner->GetContext(node);
  if (!c) {
    // Now we can run shape inference
    TF_RETURN_IF_ERROR(shape_refiner->AddNode(node));
    c = CHECK_NOTNULL(shape_refiner->GetContext(node));
    *new_shapes = true;

    // Infer the shape of the second output once and for all since it never
    // changes.
    ShapeHandle out1 = c->Scalar();
    c->set_output(1, out1);
  }

  ShapeHandle out;
  bool out_initialized = false;
  for (const GraphView::Edge fanin :
       shape_refiner->graph().GetFaninEdges(*node, false)) {
    // Skip back edges during the initial propagation phase. This is equivalent
    // to assuming that all the inputs to the merge nodes are fed by the same
    // shape, and will be corrected as needed in the relaxation phase.
    if (!relax && IsNextIteration(*fanin.src.node)) {
      continue;
    }

    InferenceContext* in = shape_refiner->GetContext(fanin.src.node);
    if (!relax && !in) {
      // Handling a loop for the first time, the back edge won't have any shape
      // info.
      continue;
    }
    ShapeHandle input = in->output(fanin.src.port_id);
    CHECK_EQ(fanin.tgt.node, node);
    c->SetInput(fanin.tgt.port_id, input);
    if (!out_initialized) {
      out_initialized = true;
      out = input;
      continue;
    }
    if (relax) {
      out = shape_refiner->OutputAsUnion(node, 0, input, out);
    } else {
      out = shape_refiner->OutputAsIntersection(node, 0, input, out);
    }
  }

  if (*new_shapes || !shape_refiner->EquivalentShapes(out, c->output(0))) {
    c->set_output(0, out);
    *new_shapes = true;
  }

  return Status::OK();
}

// Manually propagate the input shape for Enter nodes and update any Merge node
// outputs.
Status GraphProperties::UpdateEnter(SymbolicShapeRefiner* shape_refiner,
                                    const NodeDef* node, bool relax,
                                    bool* new_shapes) {
  auto enter_ctx = shape_refiner->GetContext(node);
  if (!enter_ctx) {
    TF_RETURN_IF_ERROR(shape_refiner->UpdateNode(node, relax, new_shapes));
    enter_ctx = shape_refiner->GetContext(node);
  }

  GraphView::InputPort inp(node, 0);
  GraphView::OutputPort fanin = shape_refiner->graph().GetRegularFanin(inp);

  InferenceContext* in = shape_refiner->GetContext(fanin.node);
  ShapeHandle input = in->output(fanin.port_id);
  if (!enter_ctx->output(0).SameHandle(input)) {
    enter_ctx->SetInput(0, input);
    enter_ctx->set_output(0, input);
    *new_shapes = true;
  }
  return Status::OK();
}

Status GraphProperties::UpdateShapes(SymbolicShapeRefiner* shape_refiner,
                                     bool relax, const NodeDef* n,
                                     bool* new_shapes) const {
  if (IsEnter(*n)) {
    // The Enter shape function always forwards an UnknownShape, so do the right
    // thing here.
    TF_RETURN_IF_ERROR(UpdateEnter(shape_refiner, n, relax, new_shapes));
  } else if (IsMerge(*n)) {
    // Properly handle merge nodes.
    TF_RETURN_IF_ERROR(UpdateMergeNode(shape_refiner, n, relax, new_shapes));
  } else {
    // Rely on regular TF shape refinement for all the other nodes.
    bool updated = false;
    TF_RETURN_IF_ERROR(shape_refiner->UpdateNode(n, relax, &updated));
    if (updated) {
      // We want to avoid propagating through loops on the merge pass because
      // the shapes are not guaranteed to converge.
      if (relax || !IsNextIteration(*n)) {
        *new_shapes = true;
      }
    }
  }
  return Status::OK();
}

// Propagates the shapes in the transitive fan-out of <new_shapes>.
Status GraphProperties::PropagateShapes(
    SymbolicShapeRefiner* shape_refiner, bool relax, TopoQueue* new_shapes,
    const std::unordered_map<const NodeDef*,
                             std::unordered_set<const NodeDef*>>& resources,
    int num_loops) const {
  // Limit the number of iterations to prevent infinite loops in the presence of
  // incorrect shape functions. The algoritm should converge in at most
  // num_nested_loops^2 * max_rank. We approximate max_rank with the constant 4.
  // The same applies to resources.
  VLOG(1) << "Propagating (relax=" << relax << ") " << new_shapes->size()
          << " new shapes through " << num_loops << " loops and "
          << resources.size() << " resources" << std::endl;

  const int64 max_loop_length = item_.graph.node_size();
  const int64 max_rank = 4;
  const int64 max_loop_iterations =
      max_rank * max_loop_length * std::max<int64>(1, num_loops * num_loops);
  const int64 num_queues = resources.size();
  const int64 max_resource_iterations = num_queues * num_queues * max_rank;

  int64 num_resource_iterations = 0;
  do {
    int64 num_loop_iterations = 0;
    while (!new_shapes->empty() &&
           num_loop_iterations++ < max_loop_iterations) {
      const NodeDef* n = new_shapes->pop();
      bool updated = false;
      TF_RETURN_IF_ERROR(UpdateShapes(shape_refiner, relax, n, &updated));
      if (updated) {
        for (const GraphView::InputPort fanout :
             shape_refiner->graph().GetFanouts(*n, false)) {
          new_shapes->push(fanout.node);
        }
      }
    }

    for (const auto& resource : resources) {
      // Resources need special handling: since the enqueue nodes are in the
      // fanout of the queues, we need to manually propagate the shapes from
      // enqueue node to the corresponding queue.
      TF_RETURN_IF_ERROR(UpdateResource(resource.first, resource.second,
                                        shape_refiner, new_shapes));
    }
  } while (!new_shapes->empty() &&
           num_resource_iterations++ < max_resource_iterations);

  if (!new_shapes->empty()) {
    return errors::Internal("Shape inference failed to converge");
  }

  return Status::OK();
}

Status GraphProperties::UpdateResource(
    const NodeDef* qnode,
    const std::unordered_set<const NodeDef*>& queue_inputs,
    SymbolicShapeRefiner* shape_refiner, TopoQueue* new_shapes) {
  // Proceed only if qnode is a queue or an Enter with queue input.
  if (!IsQueue(*qnode) && !IsEnterWithQueue(*qnode, shape_refiner->graph())) {
    return Status::OK();
  }
  auto qctx = shape_refiner->GetContext(qnode);
  if (!qctx) {
    return Status::OK();
  }
  auto* queue_handle_data = qctx->output_handle_shapes_and_types(0);

  // Merge all inputs into the enqueue node, regardless of which phase we
  // are in.
  std::vector<ShapeAndType> queue_shapes_and_types;
  for (const auto& node : queue_inputs) {
    auto ctx = shape_refiner->GetNodeContext(node);
    if (!ctx) {
      continue;
    }
    // TODO(bsteiner): handle EnqueueMany as well.
    if (node->op().find("Enqueue") != std::string::npos &&
        node->op().find("EnqueueMany") == std::string::npos) {
      std::vector<ShapeAndType> shapes_and_types;
      for (int i = 1; i < ctx->input_types.size(); ++i) {
        shapes_and_types.push_back(
            {ctx->inference_context->input(i), ctx->input_types[i]});
      }
      if (queue_shapes_and_types.empty()) {
        queue_shapes_and_types = shapes_and_types;
      } else {
        TF_RETURN_IF_ERROR(RelaxEnqueueShapesAndMergeTypes(
            shape_refiner, qnode, shapes_and_types, &queue_shapes_and_types));
      }
    }
  }

  if (queue_handle_data == nullptr ||
      !shape_refiner->EquivalentShapesAndTypes(*queue_handle_data,
                                               queue_shapes_and_types)) {
    qctx->set_output_handle_shapes_and_types(0, queue_shapes_and_types);

    for (const GraphView::InputPort fanout :
         shape_refiner->graph().GetFanouts(*qnode, false)) {
      new_shapes->push(fanout.node);
    }
  }

  return Status::OK();
}

Status GraphProperties::InferStatically(bool assume_valid_feeds) {
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item_.graph.library());
  std::unordered_map<string, std::unordered_set<int>> fed_ports;
  if (!assume_valid_feeds) {
    for (const auto& feed : item_.feed) {
      int port_index = 0;
      string node_name = ParseNodeName(feed.first, &port_index);
      fed_ports[node_name].insert(port_index);
    }
  }

  std::unordered_map<const NodeDef*, int> topo_order;
  TF_RETURN_IF_ERROR(ComputeTopologicalOrder(item_.graph, &topo_order));

  GraphView graph_view(&item_.graph);

  // List the resources and the nodes using them. Also collect the Merge nodes,
  // fed nodes, and primary inputs.
  std::unordered_map<const NodeDef*, std::unordered_set<const NodeDef*>>
      resources;
  std::unordered_set<const NodeDef*> merge_nodes;
  std::unordered_set<const NodeDef*> fed_nodes;
  std::unordered_set<const NodeDef*> primary_inputs;
  int num_loops = 0;
  for (const NodeDef& node : item_.graph.node()) {
    if (NumNonControlInputs(node) == 0) {
      primary_inputs.insert(&node);
    } else if (IsMerge(node)) {
      merge_nodes.insert(&node);
    } else if (IsNextIteration(node)) {
      ++num_loops;
    } else {
      const OpRegistrationData* op_data;
      TF_RETURN_IF_ERROR(function_library.LookUp(node.op(), &op_data));
      DataTypeVector input_types;
      DataTypeVector output_types;
      TF_RETURN_IF_ERROR(InOutTypesForNode(node, op_data->op_def, &input_types,
                                           &output_types));
      for (int i = 0; i < input_types.size(); ++i) {
        if (input_types[i] == DataType::DT_RESOURCE) {
          GraphView::InputPort input(&node, i);
          const GraphView::OutputPort resource =
              graph_view.GetRegularFanin(input);
          resources[resource.node].insert(&node);
        }
      }
    }
    if (fed_ports.find(node.name()) != fed_ports.end()) {
      fed_nodes.insert(&node);
    }
  }

  SymbolicShapeRefiner refiner(graph_view, fed_ports);

  // We propagate shapes through the graph in two phases. In the first phase, we
  // exclusively merge shapes but we do not propagate shapes through the
  // backedge of loops (i.e. the NextIteration node). Then on the second phase,
  // we exclusively relax shapes and propagate shapes through loops until
  // reaching fixed point.
  for (int relax = 0; relax < 2; relax++) {
    TopoQueue new_shapes(topo_order);
    // Seed the propagation of shapes through merge nodes.
    if (relax) {
      for (const NodeDef* node : merge_nodes) {
        new_shapes.push(node);
      }
    }
    // Also seed the propagation of shapes in the fanout of primary inputs.
    for (const NodeDef* node : primary_inputs) {
      new_shapes.push(node);
    }
    // Also seed the propagation of shapes in the fanout of fed nodes.
    for (const NodeDef* node : fed_nodes) {
      new_shapes.push(node);
    }
    // Propagate shapes normally.
    TF_RETURN_IF_ERROR(
        PropagateShapes(&refiner, relax, &new_shapes, resources, num_loops));
  }

  // Track shapes globally across the graph.
  SymbolicShapeManager shape_manager;
  bool found_error = false;
  for (const NodeDef& node : item_.graph.node()) {
    auto node_ctx = refiner.GetContext(&node);
    if (!node_ctx) {
      continue;
    }
    // Skip any information that comes from fed nodes.
    if (fed_ports.find(node.name()) != fed_ports.end()) {
      VLOG(2) << "Skipping feed node shape: " << node.name();
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

  for (const NodeDef& node : item_.graph.node()) {
    VLOG(3) << "Filling in graph properties for node: " << node.name();
    auto ctx = refiner.GetNodeContext(&node);
    if (!ctx) {
      continue;
    }

    // Fill input properties.
    {
      // CHECK_EQ(ctx->num_inputs(), node.num_inputs());
      auto& input_properties = input_properties_[node.name()];

      // Should always be empty, node names in graph are supposed to be unique.
      CHECK_EQ(input_properties.size(), 0);

      input_properties.resize(ctx->inference_context->num_inputs());
      GraphView::InputPort input(&node, -1);
      for (int i = 0; i < ctx->inference_context->num_inputs(); ++i) {
        shape_manager.AsTensorProperties(ctx->inference_context->input(i),
                                         ctx->input_types[i],
                                         &input_properties[i]);
        input.port_id = i;
        GraphView::OutputPort fanin = graph_view.GetRegularFanin(input);
        if (!IsConstant(*fanin.node)) {
          continue;
        }
        const TensorProto& raw_val = fanin.node->attr().at("value").tensor();
        *input_properties[i].mutable_value() = raw_val;
      }
    }

    // Fill output properties.
    {
      // CHECK_EQ(ctx->num_outputs(), node->num_outputs());
      auto& output_properties = output_properties_[node.name()];

      // Should always be empty, node names in graph are supposed to be unique.
      CHECK_EQ(output_properties.size(), 0);

      output_properties.resize(ctx->inference_context->num_outputs());
      for (int i = 0; i < ctx->inference_context->num_outputs(); ++i) {
        shape_manager.AsTensorProperties(ctx->inference_context->output(i),
                                         ctx->output_types[i],
                                         &output_properties[i]);
      }
    }
  }

  // Help trace the unknown dimensions to their origins.
  VerboseLogUnknownDimensionSources(item_.graph, input_properties_,
                                    output_properties_);

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
  if (cost_graph.node_size() == 0) {
    LOG(WARNING) << "cost_graph is empty: nothing can be inferred!";
  }
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

void GraphProperties::ClearInputProperties(const string& node_name) {
  input_properties_.erase(node_name);
}
void GraphProperties::ClearOutputProperties(const string& node_name) {
  output_properties_.erase(node_name);
}

}  // end namespace grappler
}  // end namespace tensorflow

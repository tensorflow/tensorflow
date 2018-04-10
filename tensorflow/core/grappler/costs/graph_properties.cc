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
#include "tensorflow/core/grappler/utils.h"
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

bool IsQueue(const Node& node) {
  return str_util::EndsWith(node.type_string(), "QueueV2");
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

void VerboseLogUnknownDimensionSources(
    const Graph& graph,
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
  for (const Node* const node : graph.nodes()) {
    if (node->num_outputs() == 0) {
      continue;
    }

    const auto& input_properties = input_properties_map.at(node->name());
    const auto& output_properties = output_properties_map.at(node->name());

    bool has_unknown_inputs = false;
    for (int i = 0; i < node->num_inputs(); ++i) {
      if (HasAnyUnknownDimensions(input_properties[i].shape())) {
        has_unknown_inputs = true;
        break;
      }
    }

    if (has_unknown_inputs) {
      continue;
    }

    for (int i = 0; i < node->num_outputs(); ++i) {
      if (HasAnyUnknownDimensions(output_properties[i].shape())) {
        string inputs = "input_shapes=[";
        for (int i = 0; i < node->num_inputs(); ++i) {
          inputs +=
              PartialTensorShape::DebugString(input_properties[i].shape());
        }
        inputs += "]";

        string outputs = "output_shapes=[";
        for (int i = 0; i < node->num_outputs(); ++i) {
          outputs +=
              PartialTensorShape::DebugString(output_properties[i].shape());
        }
        outputs += "]";

        VLOG(2) << "Node: " << node->name() << ", Op: " << node->def().op()
                << ", " << inputs << ", " << outputs;

        op_to_count[node->def().op()]++;

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
  void push(const Node* n) { queue_.insert(n); }
  const Node* pop() {
    CHECK(!empty());
    auto it = queue_.begin();
    const Node* n = *it;
    queue_.erase(it);
    return n;
  }

  bool empty() const { return queue_.empty(); }
  std::size_t size() const { return queue_.size(); }

 private:
  // Graph nodes are created in (roughly) topological order. Therefore we can
  // use their id to ensure they're sorted topologically.
  struct CompareNodes {
    bool operator()(const Node* lhs, const Node* rhs) const {
      return lhs->id() < rhs->id();
    }
  };
  std::set<const Node*, CompareNodes> queue_;
};

// Merge and relax symbolic shapes.
// Each symbolic shape or dimension is represented by a handle. Unlike the TF
// shape refiner which creates new handles every time it processes an unknown
// shape/dimension, the symbolic shape refiner assigns a specific handle to each
// unknown shape/dimension of a given node.
class SymbolicShapeRefiner {
 public:
  explicit SymbolicShapeRefiner(ShapeRefiner* shape_refiner)
      : shape_refiner_(shape_refiner) {}

  InferenceContext* GetContext(const Node* node) {
    return shape_refiner_->GetContext(node);
  }
  Status UpdateNode(const Node* node, bool relax, bool* refined) {
    return shape_refiner_->UpdateNode(node, relax, refined);
  }
  Status SetUnknownShape(const Node* node, int output_port) {
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
    const Node* node;
    int port_id;
    bool operator==(const ShapeId& other) const {
      return node == other.node && port_id == other.port_id;
    }
  };
  struct HashShapeId {
    std::size_t operator()(const ShapeId& shp) const {
      return std::hash<const Node*>{}(shp.node) + shp.port_id;
    }
  };

  struct DimId {
    const Node* node;
    int port_id;
    int dim_index;
    bool operator==(const DimId& other) const {
      return node == other.node && port_id == other.port_id &&
             dim_index == other.dim_index;
    }
  };

  struct HashDimId {
    std::size_t operator()(const DimId& dim) const {
      return std::hash<const Node*>{}(dim.node) + dim.port_id + dim.dim_index;
    }
  };

  // Compute the shape of the tensors outputed by node 'node' at output port
  // 'port_index' as the intersection of shape1 and shape2.
  ShapeHandle OutputAsIntersection(const Node* node, int port_index,
                                   ShapeHandle shape1, ShapeHandle shape2) {
    if (shape1.SameHandle(shape2)) {
      return shape1;
    }
    InferenceContext* ctx = shape_refiner_->GetContext(node);
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
  ShapeHandle OutputAsUnion(const Node* node, int port_index,
                            ShapeHandle shape1, ShapeHandle shape2) {
    if (shape1.SameHandle(shape2)) {
      return shape1;
    }
    InferenceContext* ctx = shape_refiner_->GetContext(node);
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

 private:
  // Return the one ShapeHandle used to denote a fully unknown shape for a node
  // output.
  ShapeHandle GetUnknownOutputShape(const Node* node, int index) {
    ShapeId id{node, index};
    auto it = unknown_shapes_.find(id);
    if (it != unknown_shapes_.end()) {
      return it->second;
    }
    InferenceContext* c = shape_refiner_->GetContext(node);
    ShapeHandle shp = c->UnknownShape();
    unknown_shapes_[id] = shp;
    return shp;
  }
  // Return the one ShapeHandle used to denote a fully unknown dimension for a
  // node output.
  DimensionHandle GetUnknownOutputDim(const Node* node, int index, int dim_id) {
    DimId id{node, index, dim_id};
    auto it = unknown_dims_.find(id);
    if (it != unknown_dims_.end()) {
      return it->second;
    }
    InferenceContext* c = shape_refiner_->GetContext(node);
    DimensionHandle dim = c->UnknownDim();
    unknown_dims_[id] = dim;
    return dim;
  }

  ShapeRefiner* shape_refiner_;

  std::unordered_map<ShapeId, ShapeHandle, HashShapeId> unknown_shapes_;
  std::unordered_map<DimId, DimensionHandle, HashDimId> unknown_dims_;
};

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

Status GraphProperties::MergeEnqueueShapesAndTypes(
    SymbolicShapeRefiner* shape_refiner, const Node* qnode,
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
    SymbolicShapeRefiner* shape_refiner, const Node* qnode,
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
                                        const Node* node, bool relax,
                                        TopoQueue* new_shapes) {
  InferenceContext* c = shape_refiner->GetContext(node);
  CHECK_NE(c, nullptr);

  ShapeHandle out1;
  TF_RETURN_IF_ERROR(c->WithRank(c->output(1), 0, &out1));
  c->set_output(1, out1);

  ShapeHandle out;
  bool out_initialized = false;
  for (const Edge* e : node->in_edges()) {
    if (e->IsControlEdge()) {
      continue;
    }
    // Skip back edges during the initial propagation phase. This is equivalent
    // to assuming that all the inputs to the merge nodes are fed by the same
    // shape, and will be corrected as needed in the relaxation phase.
    if (!relax && e->src()->IsNextIteration()) {
      continue;
    }

    InferenceContext* in = shape_refiner->GetContext(e->src());
    ShapeHandle input = in->output(e->src_output());
    if (relax) {
      c->RelaxInput(e->dst_input(), input);
    } else {
      c->MergeInput(e->dst_input(), input);
    }
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

  if (!shape_refiner->EquivalentShapes(out, c->output(0))) {
    c->set_output(0, out);
    new_shapes->push(node);
  }

  return Status::OK();
}

Status GraphProperties::OverwriteFedPorts(
    SymbolicShapeRefiner* shape_refiner,
    const std::unordered_map<string, std::unordered_set<int>>& fed_ports,
    const Node* node, TopoQueue* new_shapes) const {
  auto it = fed_ports.find(node->name());
  Status status;
  if (it != fed_ports.end()) {
    // It is possible to feed node output ports with tensors of any shape: as a
    // result, the shape of a fed port is completely unknown.
    for (const int output_port : it->second) {
      status.Update(shape_refiner->SetUnknownShape(node, output_port));
    }
    new_shapes->push(node);
  }
  return status;
}

// Manually propagate the input shape for Enter nodes and update any Merge node
// outputs.
Status GraphProperties::UpdateEnter(SymbolicShapeRefiner* shape_refiner,
                                    const Node* node, bool relax,
                                    TopoQueue* new_shapes) {
  auto enter_ctx = shape_refiner->GetContext(node);
  CHECK_NE(enter_ctx, nullptr);

  for (const Edge* e : node->in_edges()) {
    if (e->IsControlEdge()) {
      continue;
    }
    InferenceContext* in = shape_refiner->GetContext(e->src());
    ShapeHandle input = in->output(e->src_output());
    if (!enter_ctx->output(0).SameHandle(input)) {
      if (relax) {
        enter_ctx->RelaxInput(0, input);
      } else {
        enter_ctx->MergeInput(0, input);
      }
      enter_ctx->set_output(0, input);
      new_shapes->push(node);
    }
  }
  return Status::OK();
}

Status GraphProperties::UpdateShapes(
    SymbolicShapeRefiner* shape_refiner, bool relax,
    const std::unordered_map<string, std::unordered_set<int>>& fed_ports,
    const Node* n, TopoQueue* new_shapes) const {
  if (n->IsEnter()) {
    // The Enter shape function always forwards an UnknownShape, so do the right
    // thing here.
    TF_RETURN_IF_ERROR(UpdateEnter(shape_refiner, n, relax, new_shapes));
  } else if (n->IsMerge()) {
    // Properly handle merge nodes.
    TF_RETURN_IF_ERROR(UpdateMergeNode(shape_refiner, n, relax, new_shapes));
  } else {
    // Rely on regular TF shape refinement for all the other nodes.
    bool updated = false;
    TF_RETURN_IF_ERROR(shape_refiner->UpdateNode(n, relax, &updated));
    if (updated) {
      // We want to avoid propagating through loops on the merge pass because
      // the shapes are not guaranteed to converge.
      if (relax || !n->IsNextIteration()) {
        new_shapes->push(n);
      }
    }
  }
  // Nodes can be fed with any shape. The TensorFlow shape inference code can't
  // handle this properly, so overwrite its behavior here.
  return OverwriteFedPorts(shape_refiner, fed_ports, n, new_shapes);
}

// Propagates the shapes in the transitive fan-out of <new_shapes>.
Status GraphProperties::PropagateShapes(
    SymbolicShapeRefiner* shape_refiner, bool relax, TopoQueue* new_shapes,
    const std::unordered_map<const Node*, std::unordered_set<const Node*>>&
        resources,
    const std::unordered_map<string, std::unordered_set<int>>& fed_ports,
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
      const Node* n = new_shapes->pop();
      for (const Edge* e : n->out_edges()) {
        if (!e->IsControlEdge()) {
          const Node* fanout = e->dst();
          TF_RETURN_IF_ERROR(UpdateShapes(shape_refiner, relax, fed_ports,
                                          fanout, new_shapes));
        }
      }
    }

    for (const auto& resource : resources) {
      // Resources need special handling: since the enqueue nodes are in the
      // fanout of the queues, we need to manually propagate the shapes from
      // enqueue node to the corresponding queue.
      TF_RETURN_IF_ERROR(UpdateResource(resource.first, resource.second,
                                        shape_refiner, relax, new_shapes));
    }
  } while (!new_shapes->empty() &&
           num_resource_iterations++ < max_resource_iterations);

  if (!new_shapes->empty()) {
    return errors::Internal("Shape inference failed to converge");
  }

  return Status::OK();
}

Status GraphProperties::UpdateResource(
    const Node* qnode, const std::unordered_set<const Node*>& queue_inputs,
    SymbolicShapeRefiner* shape_refiner, bool relax, TopoQueue* new_shapes) {
  // Proceed only if qnode is a queue or an Enter with queue input.
  if (!IsQueue(*qnode) && !IsEnterWithQueue(*qnode)) {
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
  if (queue_handle_data) {
    queue_shapes_and_types = *queue_handle_data;
  }
  for (const auto& node : queue_inputs) {
    auto ctx = shape_refiner->GetContext(node);
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
        if (relax) {
          TF_RETURN_IF_ERROR(RelaxEnqueueShapesAndMergeTypes(
              shape_refiner, qnode, shapes_and_types, &queue_shapes_and_types));
        } else {
          TF_RETURN_IF_ERROR(MergeEnqueueShapesAndTypes(
              shape_refiner, qnode, shapes_and_types, &queue_shapes_and_types));
        }
      }
    }
  }

  if (queue_handle_data == nullptr ||
      !shape_refiner->EquivalentShapesAndTypes(*queue_handle_data,
                                               queue_shapes_and_types)) {
    qctx->set_output_handle_shapes_and_types(0, queue_shapes_and_types);

    new_shapes->push(qnode);
  }

  return Status::OK();
}

Status GraphProperties::InferStatically(bool assume_valid_feeds) {
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item_.graph.library());
  Graph graph(function_library);
  ShapeRefiner shape_refiner(graph.versions(), graph.op_registry());
  shape_refiner.set_require_shape_inference_fns(false);
  shape_refiner.set_disable_constant_propagation(true);
  ImportGraphDefOptions options;
  // Graph optimization happens at the late stage of graph execution,
  // when colocation constraints are already validated previously and
  // the device placement of nodes has also completed, so there
  // is no need to validate colocation constraints again.
  options.validate_colocation_constraints = false;
  Status s = ImportGraphDef(options, item_.graph, &graph, &shape_refiner);
  TF_RETURN_IF_ERROR(s);

  std::unordered_map<string, std::unordered_set<int>> fed_ports;
  if (!assume_valid_feeds) {
    for (const auto& feed : item_.feed) {
      int port_index = 0;
      string node_name = ParseNodeName(feed.first, &port_index);
      fed_ports[node_name].insert(port_index);
    }
  }

  // List the resources and the nodes using them. Also collect the Enter and
  // Merge nodes.
  std::unordered_map<const Node*, std::unordered_set<const Node*>> resources;
  std::unordered_set<const Node*> enter_nodes;
  std::unordered_set<const Node*> merge_nodes;
  std::unordered_set<const Node*> fed_nodes;
  int num_loops = 0;
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
    } else if (node->IsMerge()) {
      merge_nodes.insert(node);
    } else if (node->IsNextIteration()) {
      ++num_loops;
    }
    if (fed_ports.find(node->name()) != fed_ports.end()) {
      fed_nodes.insert(node);
    }
  }

  SymbolicShapeRefiner refiner(&shape_refiner);

  // We propagate shapes through the graph in two phases. In the first phase, we
  // exclusively merge shapes but we do not propagate shapes through the
  // backedge of loops (i.e. the NextIteration node). Then on the second phase,
  // we exclusively relax shapes and propagate shapes through loops until
  // reaching fixed point.
  for (int relax = 0; relax < 2; relax++) {
    TopoQueue new_shapes;
    // Force the propagation of shapes of Enter nodes manually (the Enter shape
    // function always forwards an UnknownShape).
    for (const Node* node : enter_nodes) {
      TF_RETURN_IF_ERROR(
          UpdateShapes(&refiner, relax, fed_ports, node, &new_shapes));
    }
    // Seed the propagation of shapes through merge nodes.
    for (const Node* node : merge_nodes) {
      TF_RETURN_IF_ERROR(
          UpdateShapes(&refiner, relax, fed_ports, node, &new_shapes));
    }
    // Also seed the propagation of shapes in the fanout of fed nodes.
    for (const Node* node : fed_nodes) {
      TF_RETURN_IF_ERROR(
          OverwriteFedPorts(&refiner, fed_ports, node, &new_shapes));
    }
    // Propagate shapes normally.
    TF_RETURN_IF_ERROR(PropagateShapes(&refiner, relax, &new_shapes, resources,
                                       fed_ports, num_loops));
  }

  // Track shapes globally across the graph.
  SymbolicShapeManager shape_manager;
  bool found_error = false;
  for (const Node* const node : graph.nodes()) {
    auto node_ctx = shape_refiner.GetContext(node);
    if (!node_ctx) {
      continue;
    }
    // Skip any information that comes from fed nodes.
    if (fed_ports.find(node->name()) != fed_ports.end()) {
      VLOG(2) << "Skipping feed node shape: " << node->name();
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
    VLOG(3) << "Filling in graph properties for node: " << node->name();
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
        if (edge->IsControlEdge()) {
          continue;
        }
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

  // Help trace the unknown dimensions to their origins.
  VerboseLogUnknownDimensionSources(graph, input_properties_,
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

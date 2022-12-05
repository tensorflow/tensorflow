/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/constant_folding.h"

#include <algorithm>
#include <atomic>
#include <set>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/setround.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

const char kScopedAllocatorAttrName[] = "_scoped_allocator";

// Test to see if the Op is one that turns into a constant when its
// inputs' shapes are known.
bool IsShapeOp(const Node* n) {
  const auto& ts = n->type_string();
  return ts == "Shape" || ts == "ShapeN" || ts == "Rank" || ts == "Size";
}

// Reads the partially-known shape of each of n's inputs from shape_map, and
// stores it to input_shapes. Returns false if any input does not have a shape
// in shape_map.
bool ReadPartialShapesFromShapeMap(
    const Node* n,
    const std::unordered_map<string, std::vector<PartialTensorShape>>*
        shape_map,
    std::vector<PartialTensorShape>* input_shapes) {
  CHECK(shape_map != nullptr);
  input_shapes->resize(n->num_inputs());
  for (const Edge* in : n->in_edges()) {
    // Don't need to check if incoming control edges have known shapes.
    if (in->IsControlEdge()) continue;
    const auto known_shape_iter = shape_map->find(in->src()->name());
    if (known_shape_iter == shape_map->end()) {
      // One of n's inputs doesn't have known shapes, so don't replace n.
      return false;
    }
    const auto& known_shape = known_shape_iter->second;
    CHECK_GT(known_shape.size(), in->src_output()) << known_shape_iter->first;
    DCHECK_GE(in->dst_input(), 0);
    DCHECK_LT(in->dst_input(), input_shapes->size());
    (*input_shapes)[in->dst_input()] = known_shape[in->src_output()];
  }
  return true;
}

// If all of n's inputs have fully-defined shapes, inserts those shapes as a
// vector of Tensors in the shape_replacement_map.
bool MaybeReplaceShapeOrShapeNOp(
    const Node* n, const std::vector<PartialTensorShape>& input_shapes,
    std::unordered_map<const Node*, std::vector<Tensor>>*
        shape_replacement_map) {
  std::vector<Tensor> defined_shape;
  for (const auto& shape : input_shapes) {
    if (!shape.IsFullyDefined()) {
      return false;
    }
    const int rank = shape.dims();
    DataType op_type = n->output_type(0);
    Tensor t(op_type, TensorShape({rank}));
    if (op_type == DT_INT64) {
      auto vec = t.vec<int64_t>();
      for (int i = 0; i < rank; ++i) {
        vec(i) = shape.dim_size(i);
      }
    } else {
      CHECK(op_type == DT_INT32);
      auto vec = t.vec<int32>();
      for (int i = 0; i < rank; ++i) {
        if (shape.dim_size(i) > INT_MAX) {
          VLOG(1) << "Node " << n->name() << " has input shape dimension " << i
                  << " of " << shape.dim_size(i) << " but type INT32 "
                  << " so not replacing as constant: this will trigger a "
                     "runtime error later.";
          return false;
        }
        vec(i) = static_cast<int32>(shape.dim_size(i));
      }
    }
    defined_shape.push_back(t);
  }
  // All the inputs had known shapes so we can replace the node by constants
  // later in the rewrite.
  shape_replacement_map->insert({n, defined_shape});
  return true;
}

// If n's input has defined rank, inserts that rank as a Tensor in the
//  shape_replacement_map.
bool MaybeReplaceRankOp(const Node* n,
                        const std::vector<PartialTensorShape>& input_shapes,
                        std::unordered_map<const Node*, std::vector<Tensor>>*
                            shape_replacement_map) {
  CHECK_EQ(input_shapes.size(), 1);
  if (input_shapes[0].unknown_rank()) {
    return false;
  }
  Tensor t(DT_INT32, TensorShape({}));
  t.scalar<int32>()() = input_shapes[0].dims();
  shape_replacement_map->insert({n, {t}});
  return true;
}

// If n's input has defined size, inserts that size as a Tensor in the
//  shape_replacement_map.
bool MaybeReplaceSizeOp(const Node* n,
                        const std::vector<PartialTensorShape>& input_shapes,
                        std::unordered_map<const Node*, std::vector<Tensor>>*
                            shape_replacement_map) {
  CHECK_EQ(input_shapes.size(), 1);
  if (!input_shapes[0].IsFullyDefined()) {
    return false;
  }
  DataType op_type = n->output_type(0);
  Tensor t(op_type, TensorShape({}));
  int64_t size = input_shapes[0].num_elements();
  if (op_type == DT_INT64) {
    t.scalar<int64_t>()() = size;
  } else {
    CHECK(op_type == DT_INT32);
    if (size > INT_MAX) {
      VLOG(1) << "Node " << n->name() << " has input shape size " << size
              << " but type INT32 "
              << " so not replacing as constant: this will trigger a runtime "
                 "error later.";
      return false;
    }
    t.scalar<int32>()() = static_cast<int32>(size);
  }
  shape_replacement_map->insert({n, {t}});
  return true;
}

// If n is a shape Op (Shape, ShapeN, Rank, or Size) and its inputs have their
// shapes specified in shape_map, then adds to shape_replacement_map a mapping
// from n to a vector of Tensors, where Tensor k is the (statically known) value
// on n's kth output edge. shape_replacement_map has an entry for n iff
// MaybeReplaceShapeOp returns true, so it's valid to use
// shape_replacement_map->count(n) as a test to see if n is a shape op that can
// be replaced.
bool MaybeReplaceShapeOp(
    const Node* n,
    const std::unordered_map<string, std::vector<PartialTensorShape>>*
        shape_map,
    std::unordered_map<const Node*, std::vector<Tensor>>*
        shape_replacement_map) {
  if (shape_map == nullptr || !IsShapeOp(n)) {
    return false;
  }
  // input_shapes will contain the shapes of each of n's inputs.
  std::vector<PartialTensorShape> input_shapes;
  if (!ReadPartialShapesFromShapeMap(n, shape_map, &input_shapes)) {
    return false;
  }
  const auto& ts = n->type_string();
  if (ts == "Shape" || ts == "ShapeN") {
    if (!MaybeReplaceShapeOrShapeNOp(n, input_shapes, shape_replacement_map)) {
      return false;
    }
  } else if (ts == "Rank") {
    if (!MaybeReplaceRankOp(n, input_shapes, shape_replacement_map)) {
      return false;
    }
  } else {
    CHECK_EQ(ts, "Size");
    if (!MaybeReplaceSizeOp(n, input_shapes, shape_replacement_map)) {
      return false;
    }
  }
  return true;
}

// Returns true if n can be evaluated as constant. shape_map maps from
// nodes to the partially-known shapes of their outputs. consider if
// non-null returns a bool indicating whether a given (non-Const,
// non-Shape) node is eligible to be
// constant-propagated. shape_replacement_map is filled in with a
// vector of constant output tensors for constant-foldable shape nodes
// (Shape, ShapeN, Size, or Rank).
bool IsConstantFoldable(
    const Node* n,
    const std::unordered_map<string, std::vector<PartialTensorShape>>*
        shape_map,
    const std::function<bool(const Node*)>& consider,
    int64_t max_constant_size_in_bytes,
    std::unordered_map<const Node*, std::vector<Tensor>>*
        shape_replacement_map) {
  if (n->IsConstant()) {
    // Skip constant folding resources as they cannot be deep copied.
    return n->output_type(0) != DT_RESOURCE;
  }
  if (MaybeReplaceShapeOp(n, shape_map, shape_replacement_map)) {
    return true;
  }
  if (n->op_def().is_stateful()) {
    return false;
  }
  if (consider && !consider(n)) {
    return false;
  }
  if (shape_map != nullptr) {
    // We can skip the node if an output is known to be oversized.
    auto shape_it = shape_map->find(n->name());
    if (shape_it != shape_map->end()) {
      for (int64_t i = 0; i < shape_it->second.size(); ++i) {
        const auto& out_shape = shape_it->second[i];
        if (out_shape.IsFullyDefined() &&
            out_shape.num_elements() * DataTypeSize(n->output_type(i)) >
                max_constant_size_in_bytes) {
          return false;
        }
      }
    }
  }
  if (n->IsControlFlow() || n->IsSend() || n->IsRecv()) {
    return false;
  }
  // TODO(yuanbyu): For now disable these session handle operations.
  if (n->IsGetSessionHandle() || n->IsGetSessionTensor() ||
      n->IsDeleteSessionTensor()) {
    return false;
  }
  if (n->IsSource()) {
    return false;
  }
  if (n->IsSink()) {
    return false;
  }
  if (n->IsFakeParam()) {
    return false;
  }
  // Since constant-folding runs on the CPU, do not attempt to constant-fold
  // operators that have no CPU kernel. Also implies that we will not
  // constant-fold functions.
  // TODO(phawkins): allow constant-folding for functions; functions may
  // be arbitrarily expensive to execute.
  if (!KernelDefAvailable(DeviceType(DEVICE_CPU), n->def())) {
    return false;
  }
  // Do not constant fold nodes which will be allocated by ScopedAllocator.
  // This is because the constant-folding graph will not contain the
  // `_ScopedAllocator` node, and that is necessary to be able to run a node
  // that will use this allocator.
  if (n->attrs().Find(kScopedAllocatorAttrName) != nullptr) {
    VLOG(2) << "Skip node [" << n->DebugString()
            << "] for constant folding due to scoped allocator";
    return false;
  }
  return true;
}

// If n is eligible for constant-folding, adds it to nodes, and places its
// control dependencies and those transitively of its constant-foldable inputs
// into constant_control_deps. If n is a constant-foldable shape node (Shape,
// ShapeN, Rank, or Size), also puts its outputs into shape_replacement_map.
void ConsiderConstantFoldableNode(
    Node* n, const ConstantFoldingOptions& opts, std::vector<Node*>* nodes,
    std::unordered_map<const Node*, gtl::FlatSet<Node*>>* constant_control_deps,
    std::unordered_map<const Node*, std::vector<Tensor>>* shape_replacement_map,
    bool* internal_node_inserted) {
  if (IsConstantFoldable(n, opts.shape_map, opts.consider,
                         opts.max_constant_size_in_bytes,
                         shape_replacement_map)) {
    // A node is constant provided all of its non-control incoming Tensors come
    // from constant nodes, or it's a shape Op with statically known inputs in
    // which case it is placed in shape_replacement_map.
    //
    // We allow control dependencies from non-constant nodes to constant nodes,
    // but to preserve the graph structure we must transfer the control
    // dependency onto any constant replacement.
    bool all_parents_constant = true;
    for (const Edge* in : n->in_edges()) {
      // Allows non-constant -> constant control edges.
      if (!in->IsControlEdge() &&
          constant_control_deps->count(in->src()) == 0) {
        all_parents_constant = false;
        break;
      }
    }
    if (all_parents_constant || shape_replacement_map->count(n) != 0) {
      gtl::FlatSet<Node*>& control_deps = (*constant_control_deps)[n];
      for (const Edge* e : n->in_edges()) {
        if (constant_control_deps->count(e->src()) == 0) {
          // This branch is taken if the incoming edge is a control dependency,
          // in which case we want to add it to the dependencies being
          // accumulated for this node, or the incoming edge is not
          // constant. The latter may happen when n is a shape node and the
          // source has known shape. In that case add a control dependency from
          // the source node, since there was previously a data dependency and
          // we want to preserve sequencing constraints.
          if (!e->src()->IsSource()) {
            control_deps.insert(e->src());
          }
        } else {
          // If the parent has been accumulating control dependencies, add all
          // of its transitive control deps.
          const gtl::FlatSet<Node*>& parent_deps =
              (*constant_control_deps)[e->src()];
          control_deps.insert(parent_deps.begin(), parent_deps.end());
        }
      }
      nodes->push_back(n);
      if (!n->IsConstant()) {
        *internal_node_inserted = true;
      }
    }
  }
}

// Returns the constant foldable nodes in `nodes` in topological order.
// Populates `constant_control_deps` with the non-constant control dependencies
// of each constant node.
void FindConstantFoldableNodes(
    const Graph* graph, const ConstantFoldingOptions& opts,
    std::vector<Node*>* nodes,
    std::unordered_map<const Node*, gtl::FlatSet<Node*>>* constant_control_deps,
    std::unordered_map<const Node*, std::vector<Tensor>>*
        shape_replacement_map) {
  bool internal_node_inserted = false;
  // Walk the nodes in data flow order.
  ReverseDFS(
      *graph, nullptr,
      [nodes, constant_control_deps, shape_replacement_map,
       &internal_node_inserted, &opts](Node* n) {
        ConsiderConstantFoldableNode(n, opts, nodes, constant_control_deps,
                                     shape_replacement_map,
                                     &internal_node_inserted);
      },
      NodeComparatorName());
  // If we have inserted just leaf level nodes, then there is nothing to fold.
  if (!internal_node_inserted) {
    nodes->clear();
    constant_control_deps->clear();
  }
}

typedef std::pair<Node*, int> NodeAndOutput;

// Adds n to constant_graph which is being built up for subsequent evaluation of
// constant propagation. node_map is the mapping of nodes in the original graph
// to nodes in the constant graph. The value of an entry in node_map is a vector
// of nodes because a ShapeN node in the original graph is replaced by a vector
// of Constant nodes in the constant graph.
void AddNodeToConstantGraph(
    Node* n, std::unordered_map<Node*, std::vector<Node*>>* node_map,
    Graph* constant_graph) {
  std::vector<Node*>& added = (*node_map)[n];
  added.push_back(constant_graph->CopyNode(n));
  for (const Edge* in_edge : n->in_edges()) {
    // Don't copy control edges to the constant graph.
    if (!in_edge->IsControlEdge()) {
      Node* in = in_edge->src();
      auto it = node_map->find(in);
      CHECK(it != node_map->end())
          << n->DebugString() << " <-" << in->DebugString();
      if (it->second.size() == 1) {
        constant_graph->AddEdge(it->second[0], in_edge->src_output(), added[0],
                                in_edge->dst_input());
      } else {
        // The original source node had multiple outputs and was replaced by a
        // vector of constants, so the edge comes from the 0th output of the kth
        // added constant, rather than the kth output of the added node as in
        // the standard case above.
        constant_graph->AddEdge(it->second[in_edge->src_output()], 0, added[0],
                                in_edge->dst_input());
      }
    }
  }
}

// Replaces constant-foldable shape node n by a vector of constants in
// constant_graph, which is being built up for subsequent evaluation of constant
// propagation. node_map is the mapping of nodes in the original graph to nodes
// in the constant graph. The value of an entry in node_map is a vector of nodes
// because a ShapeN node in the original graph is replaced by a vector of
// Constant nodes in the constant graph.
void AddShapeNodeToConstantGraph(
    Node* n,
    const std::unordered_map<const Node*, std::vector<Tensor>>&
        shape_replacement_map,
    std::unordered_map<Node*, std::vector<Node*>>* node_map,
    const ConstantFoldNameGenerator& generate_new_name, Graph* constant_graph) {
  std::vector<Node*>& added = (*node_map)[n];
  const string& node_name = n->name();
  for (const Tensor& t : shape_replacement_map.at(n)) {
    auto builder =
        NodeDefBuilder(generate_new_name(constant_graph, node_name), "Const")
            .Attr("dtype", t.dtype())
            .Attr("value", t);
    NodeDef def;
    CHECK(builder.Finalize(&def).ok());
    Node* constant_node;
    CHECK(NodeBuilder(builder).Finalize(constant_graph, &constant_node).ok());
    added.push_back(constant_node);
  }
  // Don't copy incoming edges to shape nodes that are being replaced.
}

// Given the constant foldable nodes in 'nodes', returns a new graph 'g'. 'g'
// will contain copies of the nodes in 'nodes'. In addition, if there is an edge
// going from a node 'n' in 'nodes' to another node in 'orig_graph' but not in
// 'nodes', then 'tensors_to_fetch' will contain the mapping from the
// corresponding copy of 'n' and the edge number in 'g' to 'n'.
Graph* GetConstantGraph(
    const Graph* orig_graph, const std::vector<Node*>& nodes,
    const std::unordered_map<const Node*, std::vector<Tensor>>&
        shape_replacement_map,
    std::map<NodeAndOutput, NodeAndOutput>* tensors_to_fetch,
    const ConstantFoldNameGenerator& generate_new_name) {
  Graph* constant_graph = new Graph(orig_graph->op_registry());
  std::unordered_map<Node*, std::vector<Node*>> node_map;
  node_map[orig_graph->source_node()] = {constant_graph->source_node()};
  node_map[orig_graph->sink_node()] = {constant_graph->sink_node()};
  for (Node* n : nodes) {
    if (shape_replacement_map.count(n) == 0) {
      AddNodeToConstantGraph(n, &node_map, constant_graph);
    } else {
      AddShapeNodeToConstantGraph(n, shape_replacement_map, &node_map,
                                  generate_new_name, constant_graph);
    }
  }

  for (auto const& added_nodes : node_map) {
    for (const Edge* out_edge : added_nodes.first->out_edges()) {
      if (node_map.count(out_edge->dst()) == 0) {
        if (out_edge->IsControlEdge()) continue;
        if (added_nodes.second.size() == 1) {
          tensors_to_fetch->insert(
              {{added_nodes.second[0], out_edge->src_output()},
               {added_nodes.first, out_edge->src_output()}});
        } else {
          // The node had multiple outputs and was replaced by a
          // vector of constants, so the NodeAndOutput is the 0th
          // output of the kth added constant, rather than the kth
          // output of the added node as in the standard case above.
          tensors_to_fetch->insert(
              {{added_nodes.second[out_edge->src_output()], 0},
               {added_nodes.first, out_edge->src_output()}});
        }
      }
    }
  }

  return constant_graph;
}

// Replaces the identified Tensor in 'graph' by a 'Const' node with
// the value supplied in 'constant'. 'partition_device', if non-null
// is the device where the graph executes. Returns true if the
// replacement was successful, false otherwise.
// 'control_deps' is the set of nodes that should be control predecessors of the
// new constant node.
bool ReplaceTensorWithConstant(
    Graph* graph, const Device* partition_device, NodeAndOutput tensor,
    const Tensor& constant, const gtl::FlatSet<Node*>& control_deps,
    int64_t max_constant_size_in_bytes,
    const ConstantFoldNameGenerator& generate_new_name) {
  // Be conservative when replacing a tensor with a constant, when not
  // running on CPU.
  // 1) Do not replace another constant.
  // 2) If the destination tensor or any other tensor from the same node is not
  // an int32 tensor, and has HOST_MEMORY constraint, do not replace it.
  // 3) If the destination tensor or any other tensor from the same node is an
  // int32 tensor, and has DEVICE_MEMORY constraint, do not replace it.
  // 4) If the size of the constant in bytes is too large (>
  // max_constant_in_bytes), do not replace it. This prevents the size of the
  // Graph from growing too large.
  // 5) If the constant op created does not have a kernel implementation
  // for the device, do not use it.
  // TODO(keveman): Consider adding a new constant op that has a kernel
  // implementation for all types, but with HostMemory constraint on it's
  // output.
  if (tensor.first->IsConstant()) {
    return false;
  }
  DeviceType device_type = partition_device
                               ? DeviceType{partition_device->device_type()}
                               : DEVICE_CPU;
  if (partition_device && device_type != DEVICE_CPU) {
    MemoryTypeVector input_mvec;
    MemoryTypeVector output_mvec;
    if (!MemoryTypesForNode(graph->op_registry(), device_type,
                            tensor.first->def(), &input_mvec, &output_mvec)
             .ok()) {
      return false;
    }
    for (int i = 0; i < output_mvec.size(); i++) {
      MemoryType memory_type = output_mvec[i];
      bool is_int32 = tensor.first->output_type(i) == DT_INT32;
      if ((memory_type == HOST_MEMORY && !is_int32) ||
          (memory_type == DEVICE_MEMORY && is_int32)) {
        return false;
      }
    }
  }
  if (constant.TotalBytes() > max_constant_size_in_bytes) {
    return false;
  }

  Node* n = tensor.first;
  std::vector<const Edge*> edges_to_remove;
  for (const Edge* out_edge : n->out_edges()) {
    if (out_edge->src_output() == tensor.second) {
      edges_to_remove.push_back(out_edge);
    }
  }
  const string& node_name = n->name();
  Node* constant_node;
  auto builder = NodeDefBuilder(generate_new_name(graph, node_name), "Const")
                     .Attr("dtype", constant.dtype())
                     .Attr("value", constant);
  if (partition_device) {
    builder.Device(partition_device->name());
  }
  NodeDef def;
  if (!builder.Finalize(&def).ok()) {
    return false;
  }
  const KernelDef* kdef;
  if (!FindKernelDef(device_type, def, &kdef, nullptr).ok()) {
    return false;
  }

  VLOG(1) << "Replacing " << tensor.first->name() << " :: " << tensor.second
          << " with a constant";

  if (!NodeBuilder(builder).Finalize(graph, &constant_node).ok()) {
    return false;
  }
  for (auto edge : edges_to_remove) {
    graph->AddEdge(constant_node, 0, edge->dst(), edge->dst_input());
    graph->RemoveEdge(edge);
  }
  if (control_deps.empty()) {
    graph->AddControlEdge(graph->source_node(), constant_node);
  } else {
    for (Node* node : control_deps) {
      graph->AddControlEdge(node, constant_node);
    }
  }
  if (partition_device) {
    constant_node->set_assigned_device_name(partition_device->name());
  }
  return true;
}

}  // namespace

Status ConstantFold(const ConstantFoldingOptions& opts,
                    FunctionLibraryRuntime* function_library, Env* env,
                    const Device* partition_device, Graph* graph,
                    bool* was_mutated) {
  // TensorFlow flushes denormals to zero and rounds to nearest, so we do
  // the same here.
  port::ScopedFlushDenormal flush;
  port::ScopedSetRound round(FE_TONEAREST);

  DumpGraph("Before", graph);

  ConstantFoldNameGenerator generate_new_name = opts.generate_new_name;
  std::atomic_int_fast64_t constant_unique_id{0};
  if (generate_new_name == nullptr) {
    generate_new_name = [&constant_unique_id](Graph* graph, string old_name) {
      return strings::StrCat(graph->NewName(old_name), "__cf__",
                             constant_unique_id.fetch_add(1));
    };
  }

  std::vector<Node*> constant_foldable_nodes;
  std::unordered_map<const Node*, gtl::FlatSet<Node*>> constant_control_deps;
  std::unordered_map<const Node*, std::vector<Tensor>> shape_replacement_map;
  FindConstantFoldableNodes(graph, opts, &constant_foldable_nodes,
                            &constant_control_deps, &shape_replacement_map);
  if (constant_foldable_nodes.empty()) {
    VLOG(1) << "No constant foldable nodes found";
    *was_mutated = false;
    // This is not an error, so return the status as OK.
    return OkStatus();
  }

  std::map<NodeAndOutput, NodeAndOutput> tensors_to_fetch;
  std::unique_ptr<Graph> constant_graph(
      GetConstantGraph(graph, constant_foldable_nodes, shape_replacement_map,
                       &tensors_to_fetch, generate_new_name));
  DumpGraph("Constant graph", constant_graph.get());

  if (tensors_to_fetch.empty()) {
    VLOG(1) << "No constant nodes found that feed into the original graph.";
    *was_mutated = false;
    // This is not an error, so return the status as OK.
    return OkStatus();
  }
  VLOG(1) << "Constant foldable " << constant_graph->num_node_ids() << " : "
          << graph->num_node_ids();

  std::vector<string> tensors_to_fetch_names;
  std::vector<NodeAndOutput> tensors_to_replace;
  // Sorting the nodes based on the name gives us a stable ordering between runs
  // for the same graph.
  std::vector<std::pair<NodeAndOutput, NodeAndOutput>> tensors_to_fetch_sorted(
      tensors_to_fetch.begin(), tensors_to_fetch.end());
  std::sort(tensors_to_fetch_sorted.begin(), tensors_to_fetch_sorted.end(),
            [](const std::pair<NodeAndOutput, NodeAndOutput>& n1,
               const std::pair<NodeAndOutput, NodeAndOutput>& n2) {
              return std::tie(n1.first.first->name(), n1.first.second) <
                     std::tie(n2.first.first->name(), n2.first.second);
            });
  for (auto n : tensors_to_fetch_sorted) {
    tensors_to_fetch_names.push_back(
        strings::StrCat(n.first.first->name(), ":", n.first.second));
    tensors_to_replace.push_back(n.second);
  }

  auto graph_runner = std::unique_ptr<GraphRunner>(new GraphRunner(env));
  // Evaluate the constant foldable nodes.
  std::vector<Tensor> outputs;
  auto delete_tensors = gtl::MakeCleanup([&graph_runner, &outputs] {
    // Output tensors need to be cleared before the GraphRunner is deleted.
    outputs.clear();
    graph_runner.reset(nullptr);
  });

  Status s =
      graph_runner->Run(constant_graph.get(), function_library, {} /* inputs*/,
                        tensors_to_fetch_names, &outputs);
  if (!s.ok()) {
    VLOG(1) << "Could not fetch constants: " << s;
    *was_mutated = false;
    return s;
  }

  // Fetch the constant tensors and replace the corresponding tensors in the
  // original graph with those constants.
  int32_t num_nodes_replaced = 0;
  for (size_t c = 0; c < outputs.size(); ++c) {
    const gtl::FlatSet<Node*>& control_deps =
        constant_control_deps[tensors_to_replace[c].first];
    if (ReplaceTensorWithConstant(
            graph, partition_device, tensors_to_replace[c], outputs[c],
            control_deps, opts.max_constant_size_in_bytes, generate_new_name)) {
      ++num_nodes_replaced;
    }
  }

  DumpGraph("After", graph);

  *was_mutated = (num_nodes_replaced > 0);
  return OkStatus();
}

}  // namespace tensorflow

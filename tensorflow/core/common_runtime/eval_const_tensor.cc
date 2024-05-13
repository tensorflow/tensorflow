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
#include "tensorflow/core/common_runtime/eval_const_tensor.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {
namespace {

using ::tensorflow::shape_inference::InferenceContext;

bool IsRank(const Node& n) { return n.type_string() == "Rank"; }
bool IsSize(const Node& n) { return n.type_string() == "Size"; }
bool IsShape(const Node& n) { return n.type_string() == "Shape"; }
bool IsStridedSlice(const Node& n) { return n.type_string() == "StridedSlice"; }
bool IsPlaceholderWithDefault(const Node& n) {
  return n.type_string() == "PlaceholderWithDefault";
}
bool IsUnstack(const Node& n) { return n.type_string() == "Unpack"; }

// Returns true iff the node has an integer attribute with the given value.
bool HasIntAttr(const Node& n, absl::string_view name, int64_t expected) {
  int64_t actual;
  return TryGetNodeAttr(n.def(), name, &actual) && actual == expected;
}

// Assuming the node is a `DT_INT32` or `DT_INT64` constant with a single
// element, returns the element. Otherwise, returns null.
std::optional<int64_t> GetIntConst(const Node& node) {
  const TensorProto* proto;
  Tensor tensor;
  if (node.IsConstant() && TryGetNodeAttr(node.def(), "value", &proto) &&
      (proto->dtype() == DT_INT32 || proto->dtype() == DT_INT64) &&
      TensorShape(proto->tensor_shape()).num_elements() == 1 &&
      tensor.FromProto(*proto)) {
    if (proto->dtype() == DT_INT32) {
      return *static_cast<const int32_t*>(tensor.data());
    } else {
      return *static_cast<const int64_t*>(tensor.data());
    }
  }
  return std::nullopt;
}

// Assuming the node represents either `tensor[ix]` or `tf.unstack(tensor)[ix]`,
// returns `ix`. Otherwise, returns null.
std::optional<int64_t> GetSliceIndex(const Node& node, const int node_output) {
  std::optional<int64_t> ix;
  if (IsUnstack(node)) {
    if (HasIntAttr(node, "axis", 0)) {
      ix = node_output;
    }
  } else if (IsStridedSlice(node)) {
    const Edge* edge;
    if (HasIntAttr(node, "begin_mask", 0) && HasIntAttr(node, "end_mask", 0) &&
        HasIntAttr(node, "ellipsis_mask", 0) &&
        HasIntAttr(node, "new_axis_mask", 0) &&
        HasIntAttr(node, "shrink_axis_mask", 1) &&
        node.input_edge(1, &edge).ok()) {
      ix = GetIntConst(*edge->src());
    }
  }
  return ix;
}

// Assuming the node represents one of
//   `tf.shape(tensor)`,
//   `tf.rank(tensor)`,
//   `tf.size(tensor)`,
//   `tf.shape(tensor)[ix]`,
//   `tf.unstack(tf.shape(tensor))[ix]`,
// and the result can be inferred from shape metadata, returns the result.
// Otherwise, returns null.
absl::StatusOr<std::optional<Tensor>> TryInferFromShapes(
    const Node& node, const int node_output, const ShapeRefiner& refiner) {
  std::optional<Tensor> result;
  if (node.num_inputs() == 0 || node_output >= node.num_outputs()) {
    return result;
  }

  const auto dtype = node.output_type(node_output);
  if (dtype != DT_INT32 && dtype != DT_INT64) {
    return result;
  }

  absl::InlinedVector<int64_t, 8> data;
  std::optional<TensorShape> shape;
  const Edge* edge;
  if (IsShape(node)) {
    // The node represents `tf.shape(tensor)`.
    InferenceContext* c = refiner.GetContext(&node);
    if (c != nullptr && c->FullyDefined(c->input(0))) {
      const int64_t rank = c->Rank(c->input(0));
      for (int i = 0; i < rank; ++i) {
        data.push_back(c->Value(c->Dim(c->input(0), i)));
      }
      shape.emplace({rank});
    }
  } else if (IsRank(node)) {
    // The node represents `tf.rank(tensor)`.
    InferenceContext* c = refiner.GetContext(&node);
    if (c != nullptr && c->RankKnown(c->input(0))) {
      data.push_back(c->Rank(c->input(0)));
      shape.emplace();
    }
  } else if (IsSize(node)) {
    // The node represents `tf.size(tensor)`.
    InferenceContext* c = refiner.GetContext(&node);
    if (c != nullptr && c->FullyDefined(c->input(0))) {
      int64_t size = 1;
      for (int i = 0, rank = c->Rank(c->input(0)); i < rank; i++) {
        size *= c->Value(c->Dim(c->input(0), i));
      }
      data.push_back(size);
      shape.emplace();
    }
  } else if (node.input_edge(0, &edge).ok() && IsShape(*edge->src())) {
    // The node may represent either `tf.shape(tensor)[ix]` or
    // `tf.unstack(tf.shape(tensor))[ix]`.
    InferenceContext* c = refiner.GetContext(edge->src());
    if (c != nullptr && c->RankKnown(c->input(0))) {
      const int64_t rank = c->Rank(c->input(0));
      std::optional<int64_t> ix = GetSliceIndex(node, node_output);
      if (ix.has_value() && -rank <= *ix && *ix < rank &&
          c->ValueKnown(c->Dim(c->input(0), *ix))) {
        data.push_back(c->Value(c->Dim(c->input(0), *ix)));
        shape.emplace();
      }
    }
  }

  if (!shape.has_value()) {
    return result;
  }

  if (dtype == DT_INT32) {
    // Make sure that the result fits to int32. Otherwise, return null.
    for (const int64_t value : data) {
      if (TF_PREDICT_FALSE(value >= std::numeric_limits<int32_t>::max())) {
        return errors::InvalidArgument("Value is out of int32 range: ", value);
      }
    }
  }

  result.emplace(dtype, *shape);
  if (dtype == DT_INT32) {
    absl::c_copy(data, static_cast<int32_t*>(result->data()));
  } else {
    absl::c_copy(data, static_cast<int64_t*>(result->data()));
  }

  return result;
}

bool IsSupportedForEvaluation(const Node& node) {
  if (node.IsConstant() || node.IsArg()) {
    return true;
  }

  // Placeholders should never be constant folded because their outputs are
  // fed by the user.
  if (node.num_inputs() == 0 || IsPlaceholderWithDefault(node)) {
    return false;
  }

  // If the node is stateful (e.g. Variable), assume the graph is not constant.
  if (node.op_def().is_stateful()) {
    return false;
  }

  // During graph construction, back edges may not be filled in. In addition,
  // control flow constructs may depend on control edges which get erased by
  // the subgraph extraction logic.
  if (node.IsEnter() || node.IsExit() || node.IsMerge()) {
    return false;
  }

  // Function libraries are not supported at the moment.
  if (node.IsFunctionCall()) {
    return false;
  }
  for (const auto& [name, attr] : node.attrs()) {
    if (attr.has_func() || !attr.list().func().empty()) {
      return false;
    }
  }

  // Evaluation runs on the same CPU, make sure that a kernel is available.
  return KernelDefAvailable(DEVICE_CPU, node.def());
}

// Constant subgraph.
struct Subgraph {
  Subgraph(const OpRegistryInterface* op_registry, int32_t graph_def_version)
      : graph(op_registry == nullptr ? OpRegistry::Global() : op_registry) {
    VersionDef versions = graph.versions();
    versions.set_producer(graph_def_version);
    graph.set_versions(versions);
  }

  GraphRunner::NamedTensorList inputs;
  Graph graph;
};

// Node along with output index.
using NodeOutput = std::pair<const Node*, int>;
std::string OutputName(const NodeOutput& output) {
  return strings::StrCat(output.first->name(), ":", output.second);
}

// Assuming that the subgraph ending at `target_node` is constant-foldable,
// returns it along with all constant inputs necessary for evaluation.
// Otherwise, returns null.
absl::StatusOr<std::unique_ptr<Subgraph>> ExtractConstantSubgraph(
    const Node& target_node, const ShapeRefiner& refiner,
    const absl::FunctionRef<std::optional<Tensor>(const Node&, int)> lookup,
    const OpRegistryInterface* op_registry, const int32_t graph_def_version) {
  std::unique_ptr<Subgraph> subgraph;
  if (!target_node.IsEnter() && !IsSupportedForEvaluation(target_node)) {
    return subgraph;
  }

  // Add the target node's inputs to seed the recursion.
  std::vector<const Edge*> edges;
  for (const Edge* edge : target_node.in_edges()) {
    if (!edge->IsControlEdge()) {
      edges.push_back(edge);
    }
  }

  // Traverse edges in BFS order.
  absl::flat_hash_map<const Node*, Node*> new_by_old_node;
  absl::InlinedVector<const Node*, 8> arg_nodes;
  absl::flat_hash_map<NodeOutput, Tensor> const_inputs;
  for (int edge_ix = 0; edge_ix < edges.size(); ++edge_ix) {
    const Edge& edge = *edges[edge_ix];
    const Node& node = *edge.src();
    const NodeOutput node_output = {&node, edge.src_output()};

    // No need to exercise the node if it's already scheduled for evaluation.
    if (new_by_old_node.contains(&node) || const_inputs.contains(node_output)) {
      continue;
    }

    // SUBTLE: Defer `lookup` for `Arg` nodes, otherwise it may trigger a new
    // round of evaluation in the shape refiner even if the subgraph is not
    // foldable.
    if (node.IsArg()) {
      arg_nodes.push_back(&node);
      continue;
    }

    // Look up the output in the cache or try to infer from shape metadata.
    auto tensor = lookup(node, node_output.second);
    if (!tensor.has_value()) {
      TF_ASSIGN_OR_RETURN(
          tensor, TryInferFromShapes(node, node_output.second, refiner));
    }
    if (tensor.has_value()) {
      const_inputs.emplace(node_output, *std::move(tensor));
    } else if (!IsSupportedForEvaluation(node)) {
      return subgraph;
    } else {
      // The node has to be evaluated, traverse its children.
      new_by_old_node.emplace(&node, /*new node*/ nullptr);
      for (const Edge* edge : node.in_edges()) {
        if (!edge->IsControlEdge()) {
          edges.push_back(edge);
        }
      }
    }
  }

  // Look up args in the cache. SUBTLE: Even if some args are not available at
  // the moment, we should `lookup` them all because it may flag these arguments
  // for the next round of shape inference.
  bool all_args_provided = true;
  for (const Node* node : arg_nodes) {
    auto tensor = lookup(*node, 0);
    all_args_provided = all_args_provided && tensor.has_value();
    if (all_args_provided) {
      const_inputs.emplace(NodeOutput{node, 0}, *std::move(tensor));
    }
  }
  if (!all_args_provided) {
    return subgraph;
  }

  subgraph = std::make_unique<Subgraph>(op_registry, graph_def_version);

  // Initialize subgraph inputs.
  auto& inputs = subgraph->inputs;
  inputs.reserve(const_inputs.size());
  for (auto& [node_output, tensor] : const_inputs) {
    // Filter out outputs of nodes that we have to evaluate anyway.
    if (!new_by_old_node.contains(node_output.first)) {
      inputs.emplace_back(OutputName(node_output), std::move(tensor));
    }
  }

  // Copy all reachable nodes and edges to the output graph.
  Graph& graph = subgraph->graph;
  new_by_old_node[&target_node] = graph.CopyNode(&target_node);
  for (const Edge* edge : edges) {
    Node*& src = new_by_old_node[edge->src()];
    if (src == nullptr) {
      src = graph.CopyNode(edge->src());
    }
    Node* dst = new_by_old_node.at(edge->dst());
    graph.AddEdge(src, edge->src_output(), dst, edge->dst_input());
  }

  return subgraph;
}

}  // namespace

absl::StatusOr<std::optional<Tensor>> EvaluateConstantTensor(
    const Node& node, const int node_output, const ShapeRefiner& refiner,
    const absl::FunctionRef<std::optional<Tensor>(const Node&, int)> lookup,
    const std::optional<EvaluateConstantTensorRunner> runner) {
  // Fast path: try to infer the tensor without running a subgraph.
  std::optional<Tensor> result;
  if (result = lookup(node, node_output); result.has_value()) {
    return result;
  }
  if (node.IsArg()) {
    return result;
  }
  if (node.IsConstant()) {
    const TensorProto* proto;
    TF_RETURN_IF_ERROR(GetNodeAttr(node.def(), "value", &proto));
    result.emplace();
    if (TF_PREDICT_FALSE(!result->FromProto(*proto))) {
      return errors::InvalidArgument("Unable to evaluate a constant node");
    }
    return result;
  }
  TF_ASSIGN_OR_RETURN(result, TryInferFromShapes(node, node_output, refiner));
  if (result.has_value()) {
    return result;
  }

  if (!runner.has_value()) {
    // The graph runner is not configured, skip constant folding.
    return result;
  }

  // Slow path: extract and run the subgraph.
  TF_ASSIGN_OR_RETURN(
      const auto subgraph,
      ExtractConstantSubgraph(node, refiner, lookup, runner->op_registry,
                              runner->graph_def_version));
  if (subgraph != nullptr) {
    GraphRunner* graph_runner = runner->graph_runner;
    std::unique_ptr<GraphRunner> tmp_graph_runner;
    if (graph_runner == nullptr) {
      tmp_graph_runner = std::make_unique<GraphRunner>(Env::Default());
      graph_runner = tmp_graph_runner.get();
    }

    // NOTE; we should pass in a function library runtime if we want to
    // support constant-expression evaluation on functions.
    FunctionLibraryRuntime* function_library = nullptr;
    std::vector<Tensor> outputs;
    auto status =
        graph_runner->Run(&subgraph->graph, function_library, subgraph->inputs,
                          {OutputName({&node, node_output})}, &outputs);

    // A graph may contain errors such as shape incompatibility or division by
    // zero. Errors like that are usually uncovered by a full-graph analysis or
    // during execution, not during construction where this function is mainly
    // used. Suppress execution errors for this reason (best effort).
    if (status.ok()) {
      result = std::move(outputs[0]);
    }
  }

  return result;
}

}  // namespace tensorflow

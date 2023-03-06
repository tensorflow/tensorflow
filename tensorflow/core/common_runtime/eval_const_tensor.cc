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
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {
namespace {

using ::tensorflow::shape_inference::InferenceContext;

using NodeOutput = std::pair<const Node*, int>;

std::string OutputName(const NodeOutput& output) {
  return strings::StrCat(output.first->name(), ":", output.second);
}

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
StatusOr<std::optional<Tensor>> TryInferFromShapes(
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

  // During construction or import from GraphConstructor, back edges may not
  // be filled in. In addition, control flow constructs may depend on
  // control edges which aren't handled by this method. Don't constant fold
  // through merges at all for now.
  if (node.IsMerge()) {
    return false;
  }

  // Don't constant fold enter/exit currently either, as it's easy to end
  // up with a partial frame.
  if (node.IsEnter() || node.IsExit()) {
    return false;
  }

  // Since constant-folding runs on the CPU, do not attempt to constant-fold
  // operators that have no CPU kernel.
  return KernelDefAvailable(DEVICE_CPU, node.def());
}

// Extracts the subgraph ending at 'target_node' that is statically
// computable and inserts into 'out_graph'. If statically computable,
// returns a map of constant inputs.
StatusOr<absl::flat_hash_map<NodeOutput, Tensor>> ExtractConstantSubgraph(
    const Node& target_node, const ShapeRefiner& refiner,
    const absl::FunctionRef<std::optional<Tensor>(const Node&, int)> lookup,
    Graph* out_graph) {
  absl::flat_hash_map<NodeOutput, Tensor> const_inputs;
  if (!target_node.IsEnter() && !IsSupportedForEvaluation(target_node)) {
    return const_inputs;
  }

  // Identify the possibly constant subgraph by recursively iterating
  // backwards through the inputs to 'target_node' until we either 1) find
  // an already existing input to our subgraph 'const_inputs', 2) discover
  // our graph is not constant, or 3) hit an argument node.
  struct NodeAndRecursed {
    Node* new_node = nullptr;
    bool recursed = false;
  };
  absl::flat_hash_map<const Node*, NodeAndRecursed> old_to_new_and_recursed;
  Node& target_node_copy = *out_graph->CopyNode(&target_node);
  old_to_new_and_recursed[&target_node] = {&target_node_copy, true};

  // Add the target node's inputs to seed the recursion.
  std::deque<const Edge*> edges_to_visit;
  for (const Edge* e : target_node.in_edges()) {
    if (!e->IsControlEdge()) {
      edges_to_visit.push_back(e);
    }
  }

  // Iterate over the set of edges to visit (backwards).
  bool is_constant_graph = true;
  while (!edges_to_visit.empty()) {
    const Edge& edge = *edges_to_visit.front();
    edges_to_visit.pop_front();
    const Node& node = *edge.src();
    const int node_output = edge.src_output();

    // Add a copy of its node and a new edge to the new subgraph.
    NodeAndRecursed& node_and_recursed = old_to_new_and_recursed[&node];
    if (node_and_recursed.new_node == nullptr) {
      // First time processing this node.
      if (!IsSupportedForEvaluation(node)) {
        is_constant_graph = false;
        break;
      }
      node_and_recursed.new_node = out_graph->CopyNode(&node);
    }

    // Add the edge to the destination node.
    {
      auto it = old_to_new_and_recursed.find(edge.dst());
      if (TF_PREDICT_FALSE(it == old_to_new_and_recursed.end())) {
        return errors::Internal(
            "Could not find mapping from old to new copy of the node: ",
            edge.dst()->name());
      }
      out_graph->AddEdge(node_and_recursed.new_node, edge.src_output(),
                         it->second.new_node, edge.dst_input());
    }

    if (const_inputs.contains(NodeOutput{&node, node_output})) {
      continue;
    }

    auto tensor = lookup(node, node_output);
    if (!tensor.has_value()) {
      TF_ASSIGN_OR_RETURN(tensor,
                          TryInferFromShapes(node, node_output, refiner));
    }
    if (tensor.has_value()) {
      const_inputs.emplace(NodeOutput{&node, node_output}, *tensor);
    } else if (node.IsArg()) {
      is_constant_graph = false;
      break;
    } else if (!node_and_recursed.recursed) {
      node_and_recursed.recursed = true;
      for (const Edge* e : node.in_edges()) {
        if (!e->IsControlEdge()) {
          edges_to_visit.push_back(e);
        }
      }
    }
  }
  if (!is_constant_graph) {
    const_inputs.clear();
    out_graph->Clear();
  }
  return const_inputs;
}

}  // namespace

StatusOr<std::optional<Tensor>> EvaluateConstantTensor(
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
  Graph graph(runner->op_registry == nullptr ? OpRegistry::Global()
                                             : runner->op_registry);
  {
    VersionDef versions = graph.versions();
    versions.set_producer(runner->graph_def_version);
    graph.set_versions(versions);
  }

  TF_ASSIGN_OR_RETURN(auto const_inputs,
                      ExtractConstantSubgraph(node, refiner, lookup, &graph));
  if (!const_inputs.empty() || graph.num_op_nodes() != 0) {
    GraphRunner::NamedTensorList inputs;
    inputs.reserve(const_inputs.size());
    for (auto& [output, tensor] : const_inputs) {
      inputs.emplace_back(OutputName(output), std::move(tensor));
    }

    const auto output_name = OutputName({&node, node_output});

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
    const auto status = graph_runner->Run(  //
        &graph, function_library, inputs, {output_name}, &outputs);

    // If all kernels in the constant graph are not registered in the process,
    // GraphRunner::Run may fail, in which case we cannot propagate constants,
    // so this is best-effort.
    if (status.ok()) {
      result = std::move(outputs[0]);
    }
  }

  return result;
}

}  // namespace tensorflow

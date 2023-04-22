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

#include <deque>

#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

using shape_inference::InferenceContext;

namespace {

// Tries to infer tensor output based on the input shapes of the node. In some
// cases, the shapes of the inputs are sufficient for inferring the contents of
// the output tensor. For example, a Shape op with fully defined input shapes
// can have its output tensor inferred.
Status TryToInferTensorOutputFromInputShapes(const Edge& edge,
                                             const ShapeRefiner& refiner,
                                             Tensor* output, bool* success) {
  *success = false;
  const Node* node = edge.src();
  InferenceContext* c = refiner.GetContext(node);
  if (c == nullptr) {
    // An input without context is a soft failure; we sometimes need to break
    // control flow loops by running shape inference on a node without first
    // adding its input.
    return Status::OK();
  }

  if (node->type_string() == "Shape") {
    // If input shapes to the shape op are fully defined,
    // we can infer the shape op's output tensor.
    bool fully_defined_inputs = c->FullyDefined(c->input(0));
    if (fully_defined_inputs) {
      int input_rank = c->Rank(c->input(0));
      Tensor t(node->output_type(0), TensorShape({input_rank}));
      if (node->output_type(0) == DT_INT32) {
        auto flat = t.flat<int>();
        for (int i = 0; i < input_rank; i++) {
          int64_t dimension = c->Value(c->Dim(c->input(0), i));
          if (!FastBoundsCheck(dimension, std::numeric_limits<int32>::max())) {
            return errors::InvalidArgument(
                "Shape has output type int32, but dimension exceeds maximum "
                "int32 value");
          }
          flat(i) = static_cast<int32>(dimension);
        }
      } else if (node->output_type(0) == DT_INT64) {
        auto flat = t.flat<int64>();
        for (int i = 0; i < input_rank; i++) {
          flat(i) = c->Value(c->Dim(c->input(0), i));
        }
      } else {
        return errors::FailedPrecondition(
            "Shape has output type that is not int32 or int64");
      }
      *output = t;
      *success = true;
    }
  } else if (node->type_string() == "Rank") {
    bool rank_known = c->RankKnown(c->input(0));
    if (rank_known) {
      int32 input_rank = c->Rank(c->input(0));
      Tensor t(node->output_type(0), TensorShape({}));
      t.flat<int32>()(0) = input_rank;
      *output = t;
      *success = true;
    }
  } else if (node->type_string() == "Size") {
    bool fully_defined_inputs = c->FullyDefined(c->input(0));
    if (fully_defined_inputs) {
      int32 rank = c->Rank(c->input(0));
      Tensor t(node->output_type(0), TensorShape({}));
      int64_t size = 1;
      for (int i = 0; i < rank; i++) {
        size *= c->Value(c->Dim(c->input(0), i));
      }
      if (node->output_type(0) == DT_INT32) {
        if (!FastBoundsCheck(size, std::numeric_limits<int32>::max())) {
          return errors::InvalidArgument(
              "Size has output type int32, but size exceeds maximum int32 "
              "value");
        }
        t.flat<int32>()(0) = static_cast<int32>(size);
      } else if (node->output_type(0) == DT_INT64) {
        t.flat<int64>()(0) = size;
      } else {
        return errors::FailedPrecondition(
            "Size has output type that is not int32 or int64");
      }
      *output = t;
      *success = true;
    }
  }
  return Status::OK();
}

// Returns true if 'node' has a registered CPU kernel.
bool HasCpuKernel(const Node& node) {
  return FindKernelDef(DeviceType(DEVICE_CPU), node.def(), /*def=*/nullptr,
                       /*kernel_class_name=*/nullptr)
      .ok();
}

Status GetArgNodeIndex(const Node* node, int num_function_inputs, int* index) {
  DCHECK(node->IsArg());
  TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(node->def()), "index", index));
  if (*index < 0 || num_function_inputs <= *index) {
    return errors::Internal(
        "Function instantiation included invalid input index: ", index,
        " not in [0, ", num_function_inputs, ").");
  }
  return Status::OK();
}

// Extracts the subgraph ending at 'target_node' that is statically computable
// and inserts into 'out_graph'. If statically computable, 'is_constant_graph'
// will be set to true.
Status ExtractConstantSubgraph(
    const Node& target_node, const ShapeRefiner& refiner,
    const std::unordered_map<string, Tensor>* cached_values, Graph* out_graph,
    bool* is_constant_graph,
    std::vector<std::pair<string, Tensor>>* const_inputs,
    InferenceContext* outer_context) {
  *is_constant_graph = false;
  std::unordered_set<string> const_inputs_added;

  if (target_node.op_def().is_stateful()) {
    return Status::OK();
  }

  if (IsMerge(&target_node)) {
    return Status::OK();
  }

  if (target_node.type_string() == "PlaceholderWithDefault") {
    return Status::OK();
  }

  // Since constant-folding runs on the CPU, do not attempt to constant-fold
  // operators that have no CPU kernel.
  if (!HasCpuKernel(target_node)) {
    return Status::OK();
  }

  // TODO(skyewm): should more of the filtering applied in input nodes below be
  // applied to target_node here?

  // Identify the possibly constant subgraph by recursively iterating backwards
  // through the inputs to 'target_node' until we either 1) find an already
  // existing input to our subgraph 'const_inputs', 2) Discover our graph is not
  // constant, or 3) Hit a root node.

  struct NodeAndRecursed {
    Node* new_node = nullptr;
    bool recursed = false;
  };

  std::map<const Node*, NodeAndRecursed> old_to_new_and_recursed;
  Node* target_node_copy = out_graph->CopyNode(&target_node);
  old_to_new_and_recursed[&target_node].new_node = target_node_copy;
  old_to_new_and_recursed[&target_node].recursed = true;

  // Add the target node's inputs to seed the recursion.
  std::deque<const Edge*> edges_to_visit;
  for (const Edge* e : target_node.in_edges()) {
    // TODO(skyewm): control edges will be meaningful if/when we handle control
    // flow (e.g. constants in cond branches are triggered via control edges).
    if (e->IsControlEdge()) continue;
    edges_to_visit.push_back(e);
  }

  *is_constant_graph = true;

  // Iterate over the set of edges to visit (backwards).
  while (!edges_to_visit.empty()) {
    const Edge* current_edge = edges_to_visit.front();
    edges_to_visit.pop_front();
    Node* current_node = current_edge->src();

    // If the node is stateful, assume the graph is not constant unless it is
    // an Arg node which is handled later on.
    if (!current_node->IsArg() && current_node->op_def().is_stateful()) {
      *is_constant_graph = false;
      return Status::OK();
    }

    // During construction or import from GraphConstructor, back edges may not
    // be filled in. In addition, control flow constructs may depend on control
    // edges which aren't handled by this method. Don't constant fold through
    // merges at all for now.
    if (IsMerge(current_node)) {
      *is_constant_graph = false;
      return Status::OK();
    }

    // Don't constant fold enter/exit currently either, as it's easy to end
    // up with a partial frame.
    if (IsEnter(current_node) || IsExit(current_node)) {
      *is_constant_graph = false;
      return Status::OK();
    }

    // Placeholders should never be constant folded because their outputs are
    // fed by the user. Note that "Placeholder" nodes have no inputs so are
    // handled below.
    if (current_node->type_string() == "PlaceholderWithDefault") {
      *is_constant_graph = false;
      return Status::OK();
    }

    if (!HasCpuKernel(*current_node)) {
      *is_constant_graph = false;
      return Status::OK();
    }

    // If there is nothing more to recurse down, see if
    // the generator node is a constant or an Arg node whose value is available
    // in the `outer_context`.
    if (current_node->num_inputs() == 0) {
      if (outer_context && current_node->IsArg()) {
        const string& tensor_name =
            strings::StrCat(current_node->name(), ":", 0);
        // If we do not already have a constant Tensor for this Arg try to
        // fetch it from the outer context.
        if (const_inputs_added.count(tensor_name) == 0) {
          int index;
          TF_RETURN_IF_ERROR(GetArgNodeIndex(
              current_node, outer_context->num_inputs(), &index));
          const Tensor* const_tensor = outer_context->input_tensor(index);
          if (const_tensor) {
            const_inputs->emplace_back(tensor_name, *const_tensor);
            const_inputs_added.insert(tensor_name);
          } else {
            // Request a constant value for this Arg. If that is statically
            // computable, shape refiner will re-run the shape inference for
            // this function with this tensor's value.
            outer_context->request_input_tensor(index);
            *is_constant_graph = false;
            return Status::OK();
          }
        }
      } else if (!current_node->IsConstant()) {
        // Generator node is not a constant, so subgraph is not
        // constant.
        *is_constant_graph = false;
        return Status::OK();
      }
    }

    // Either the node is a constant, or the node is a potential
    // intermediate node on the path from a constant.
    //
    // Add a copy of its node and a new edge to the new subgraph.

    // Get or create the version of 'current_node' in the new graph.
    Node* current_node_copy;
    // This gets or creates the NodeAndRecursed entry for current_node.
    NodeAndRecursed* node_and_recursed = &old_to_new_and_recursed[current_node];
    if (node_and_recursed->new_node == nullptr) {
      // First time processing this node.
      current_node_copy = out_graph->CopyNode(current_node);
      // Track the mapping from the original node to the new one.
      node_and_recursed->new_node = current_node_copy;
    } else {
      current_node_copy = node_and_recursed->new_node;
    }

    // Add the edge to the destination node.
    {
      auto it = old_to_new_and_recursed.find(current_edge->dst());
      if (it == old_to_new_and_recursed.end()) {
        return errors::Internal(
            "Could not find mapping from old to new copy of destination node: ",
            current_edge->dst()->name());
      }
      Node* dst_copy = it->second.new_node;

      out_graph->AddEdge(current_node_copy, current_edge->src_output(),
                         dst_copy, current_edge->dst_input());
    }

    const string& output_tensor_name =
        strings::StrCat(current_node->name(), ":", current_edge->src_output());

    // Some tensor values can be inferred. For example, a shape op
    // with input shapes fully defined can have its output tensor inferred.
    Tensor tensor_inferred;
    bool successfully_inferred_tensor = false;
    TF_RETURN_IF_ERROR(TryToInferTensorOutputFromInputShapes(
        *current_edge, refiner, &tensor_inferred,
        &successfully_inferred_tensor));
    if (successfully_inferred_tensor) {
      const_inputs->emplace_back(output_tensor_name, tensor_inferred);
      const_inputs_added.insert(output_tensor_name);
      continue;
    }

    // If we have a copy of the input tensor materialized already,
    // then add to the list of inputs to feed and do not recurse further.
    if (cached_values != nullptr) {
      auto it = cached_values->find(output_tensor_name);
      if (it != cached_values->end() &&
          const_inputs_added.count(output_tensor_name) == 0) {
        const_inputs->emplace_back(output_tensor_name, it->second);
        const_inputs_added.insert(output_tensor_name);
        continue;
      }
    }

    // If this node's inputs have not been processed already, do so now.
    if (!node_and_recursed->recursed) {
      node_and_recursed->recursed = true;
      for (const Edge* e : current_node->in_edges()) {
        if (e->IsControlEdge()) continue;
        edges_to_visit.push_back(e);
      }
    }
  }

  return Status::OK();
}

}  // namespace

Status EvaluateConstantTensor(OutputTensor tensor, const ShapeRefiner& refiner,
                              const OpRegistryInterface& ops,
                              int32 graph_def_version, bool* evaluated,
                              Tensor* result, GraphRunner* graph_runner,
                              std::unordered_map<string, Tensor>* cached_values,
                              int64_t max_cached_value_size,
                              bool disable_constant_propagation,
                              InferenceContext* outer_context) {
  *evaluated = false;
  const Node* src = tensor.node;

  // Simple case: the source node is a constant
  if (src->IsConstant()) {
    if (result->FromProto(src->def().attr().at("value").tensor())) {
      *evaluated = true;
      return Status::OK();
    }
  }

  // If the source node is an Arg return its value, if available in the outer
  // context.
  if (src->IsArg() && outer_context) {
    int index;
    TF_RETURN_IF_ERROR(
        GetArgNodeIndex(src, outer_context->num_inputs(), &index));
    const Tensor* const_tensor = outer_context->input_tensor(index);
    if (const_tensor) {
      *evaluated = true;
      *result = *(outer_context->input_tensor(index));
    } else {
      outer_context->request_input_tensor(index);
    }
    return Status::OK();
  }

  if (disable_constant_propagation) {
    return Status::OK();
  }

  bool is_constant_graph = false;
  Graph subgraph(&ops);
  auto versions = subgraph.versions();
  versions.set_producer(graph_def_version);
  subgraph.set_versions(versions);

  std::vector<std::pair<string, Tensor>> const_inputs;
  TF_RETURN_IF_ERROR(ExtractConstantSubgraph(*src, refiner, cached_values,
                                             &subgraph, &is_constant_graph,
                                             &const_inputs, outer_context));
  if (!is_constant_graph) {
    return Status::OK();
  }
  const string output_tensor_name =
      strings::StrCat(src->name(), ":", tensor.index);
  std::vector<Tensor> outputs;

  std::unique_ptr<GraphRunner> graph_runner_storage;
  if (graph_runner == nullptr) {
    // TODO(skyewm): Convert to std::make_unique when available.
    graph_runner_storage.reset(new GraphRunner(Env::Default()));
    graph_runner = graph_runner_storage.get();
  }

  // NOTE; we should pass in a function library runtime if we want
  // to support constant-expression evaluation on functions.
  Status s = graph_runner->Run(&subgraph, nullptr /* function_library */,
                               const_inputs, {output_tensor_name}, &outputs);

  // If all kernels in the constant graph are not registered
  // in the process, GraphRunner::Run may fail, in which case
  // we cannot propagate constants, so this is best-effort.
  if (s.ok()) {
    *result = outputs[0];
    *evaluated = true;

    // We memoize (small) constants evaluated so far, so
    // ExtractConstantSubgraph can avoid extracting the full
    // subgraph.  As we build up large graphs, this avoids
    // repeated computation of the early parts of a constant
    // graph.
    if (cached_values != nullptr &&
        outputs[0].TotalBytes() <= max_cached_value_size) {
      (*cached_values)[output_tensor_name] = outputs[0];
    }
  }
  return Status::OK();
}

}  // namespace tensorflow

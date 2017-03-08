/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/shape_refiner.h"

#include <deque>
#include <memory>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

ShapeRefiner::ShapeRefiner(const OpRegistryInterface* ops)
    : ops_registry_(ops) {}

ShapeRefiner::~ShapeRefiner() { gtl::STLDeleteValues(&node_to_context_); }

Status ShapeRefiner::AddNode(const Node* node) {
  // For each 'input' of this node, fetch the corresponding shape
  // from 'input's InferenceContext, and store into a vector
  // indexed by 'node's input.
  std::vector<Node*> input_nodes(node->num_inputs());
  std::vector<ShapeHandle> input_shapes(node->num_inputs());
  std::vector<DataType> input_handle_dtypes(node->num_inputs());
  std::vector<ShapeHandle> input_handle_shapes(node->num_inputs());
  for (const Edge* e : node->in_edges()) {
    if (e->IsControlEdge()) continue;

    Node* input = e->src();
    auto it = node_to_context_.find(input);
    if (it == node_to_context_.end()) {
      return errors::FailedPrecondition(
          "Input ", e->dst_input(), " ('", input->name(), "') for '",
          node->name(), "' was not previously added to ShapeRefiner.");
    }

    InferenceContext* c = it->second;
    DCHECK_GE(e->dst_input(), 0);
    input_nodes[e->dst_input()] = input;
    input_shapes[e->dst_input()] = c->output(e->src_output());

    // Only propagate handle xshape and dtype of edges which are carrying
    // resource handles.
    if (e->src()->output_type(e->src_output()) == DT_RESOURCE) {
      input_handle_dtypes[e->dst_input()] =
          c->output_handle_dtype(e->src_output());
      input_handle_shapes[e->dst_input()] =
          c->output_handle_shape(e->src_output());
    }
  }

  // Get the shape function for this node
  const OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(ops_registry_->LookUp(node->type_string(), &op_reg_data));
  if (op_reg_data->shape_inference_fn == nullptr) {
    return errors::InvalidArgument(
        "No shape inference function exists for op '", node->type_string(),
        "', did you forget to define it?");
  }

  // This needs to be filled in with real data in a second pass.
  std::vector<const Tensor*> input_tensors(node->num_inputs());
  std::vector<Tensor> real_tensors(node->num_inputs());
  std::vector<bool> attempted_materialization(node->num_inputs());
  std::vector<bool> attempted_tensor_as_shape_conversion(node->num_inputs());
  std::vector<ShapeHandle> input_tensors_as_shapes;

  // Create the inference context for this node with the existing input shapes.
  std::unique_ptr<InferenceContext> c(new InferenceContext(
      &node->def(), node->op_def(), input_shapes, input_tensors,
      input_tensors_as_shapes, input_handle_shapes, input_handle_dtypes));
  if (!c->construction_status().ok()) {
    return c->construction_status();
  }

  // Run the shape inference function, and return if there was an error.
  TF_RETURN_IF_ERROR(c->Run(op_reg_data->shape_inference_fn));

  // We must run the shape function repeatedly, in case users write
  // shape functions where they only conditionally call input_tensor()
  // based on the values of another input tensor.
  bool rerun_shape_fn;
  do {
    // If the result of running shape inference would have benefitted
    // from knowing the values of input tensors, try to materialize
    // the results of those tensors, and then run the shape inference
    // function again using those known tensors.
    rerun_shape_fn = false;

    // NOTE: It is possible to batch the extraction and
    // materialization of inputs, instead of materializing one input
    // at a time like we do below.  If input-at-a-time computation
    // becomes a bottleneck, we could separate ExtractConstantSubgraph
    // into two functions: one that returns true if an input is
    // derivable from constants, and another function that extracts
    // the subgraph for multiple target nodes and executes the whole
    // subgraph once.

    for (int i = 0; i < c->num_inputs(); ++i) {
      if (!c->requested_input_tensor(i)) {
        continue;
      }
      // Check if we have not already filled in the requested input,
      // and if not, try to materialize the tensors.
      if (!attempted_materialization[i]) {
        attempted_materialization[i] = true;

        Tensor result;
        bool evaluated = false;
        TF_RETURN_IF_ERROR(
            EvaluateConstantTensorForEdge(node, i, &evaluated, &result));
        if (evaluated) {
          real_tensors[i] = result;
          input_tensors[i] = &real_tensors[i];
          // We have more concrete information about a shape,
          // so re-run shape inference.
          rerun_shape_fn = true;
        }
      }
      if (c->requested_input_tensor_as_partial_shape(i) &&
          !attempted_tensor_as_shape_conversion[i]) {
        attempted_tensor_as_shape_conversion[i] = true;
        if (i >= input_tensors_as_shapes.size()) {
          input_tensors_as_shapes.resize(i + 1);
        }
        ShapeHandle s;
        TF_RETURN_IF_ERROR(ConstantPartialShape(c.get(), node, i, &s));
        input_tensors_as_shapes[i] = s;
        rerun_shape_fn = true;
      }
    }

    if (rerun_shape_fn) {
      // We have more information about the shapes on this pass,
      // so re-run shape inference.
      c->set_input_tensors(input_tensors);
      c->set_input_tensors_as_shapes(input_tensors_as_shapes);
      TF_RETURN_IF_ERROR(op_reg_data->shape_inference_fn(c.get()));
    }
  } while (rerun_shape_fn);

  // Store the resulting InferenceContext object in the map.
  node_to_context_[node] = c.release();

  return Status::OK();
}

Status ShapeRefiner::SetShape(const Node* node, int output_port,
                              ShapeHandle shape) {
  auto c = GetContext(node);
  if (c == nullptr) {
    return errors::Internal("Could not find context for ", node->name());
  }

  if (output_port < 0 || output_port >= node->num_outputs()) {
    return errors::InvalidArgument(
        "output_port '", output_port, "' is out of range, ", "node '",
        node->name(), "' has ", node->num_outputs(), " outputs");
  }

  // Check compatibility, and merge the shapes.
  ShapeHandle existing_shape = c->output(output_port);
  TF_RETURN_IF_ERROR(c->Merge(existing_shape, shape, &shape));
  c->set_output(output_port, shape);

  // TODO(vrv): Do we need to propagate the new shape through all
  // consumers that change their outputs?  At the moment, python
  // does not do this, but this seems like a nice feature.

  // TODO(vrv): We might need to keep track of the fact that the
  // existing shape is invalidated, in case we need to propagate
  // this information to remote workers.
  return Status::OK();
}

Status ShapeRefiner::EvaluateConstantTensorForEdge(const Node* node,
                                                   int dst_idx, bool* evaluated,
                                                   Tensor* result) {
  *evaluated = false;
  const Edge* input_edge;
  TF_RETURN_IF_ERROR(node->input_edge(dst_idx, &input_edge));

  bool is_constant_graph = false;
  Graph subgraph(ops_registry_);

  // We identify the possibly constant subgraph to evaluate by
  // recursively iterating backwards through the inputs to 'node'
  // until we either 1) find an already existing input to our subgraph
  // (filled in `const_inputs`), 2) Discover our graph is not constant,
  // or 3) Hit a root node.
  std::vector<std::pair<string, Tensor>> const_inputs;
  TF_RETURN_IF_ERROR(ExtractConstantSubgraph(
      input_edge->src(), &subgraph, &is_constant_graph, &const_inputs));
  if (!is_constant_graph) {
    return Status::OK();
  }
  const string output_tensor_name =
      strings::StrCat(input_edge->src()->name(), ":", input_edge->src_output());
  std::vector<Tensor> outputs;
  // NOTE; we should pass in a function library runtime if we want
  // to support constant-expression evaluation on functions.
  Status s = GraphRunner::Run(&subgraph, nullptr /* function_library */,
                              Env::Default(), const_inputs,
                              {output_tensor_name}, &outputs);

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
    if (outputs[0].TotalBytes() <= kMaxTensorSize) {
      const_tensor_map_[output_tensor_name] = outputs[0];
    }
  }
  return Status::OK();
}

Status ShapeRefiner::ExtractConstantSubgraph(
    Node* target_node, Graph* out_graph, bool* is_constant_graph,
    std::vector<std::pair<string, Tensor>>* const_inputs) {
  *is_constant_graph = false;
  std::unordered_set<string> const_inputs_added;

  if (target_node->op_def().is_stateful()) {
    return Status::OK();
  }

  std::map<Node*, Node*> old_to_new;
  Node* target_node_copy = out_graph->CopyNode(target_node);
  old_to_new[target_node] = target_node_copy;

  // Add the target node's inputs to seed the recursion.
  std::deque<const Edge*> edges_to_visit;
  for (const Edge* e : target_node->in_edges()) {
    // TODO(vrv): What do we do about control edges?  Based on our
    // definition of a constant graph, we should be free to ignore
    // control edges since the order in which a constant graph is
    // executed should be the same regardless of when nodes run: we
    // should only need to recurse down data edges.
    if (e->IsControlEdge()) continue;
    edges_to_visit.push_back(e);
  }

  *is_constant_graph = true;

  // Iterate over the set of edges to visit (backwards).
  while (!edges_to_visit.empty()) {
    const Edge* current_edge = edges_to_visit.front();
    edges_to_visit.pop_front();
    Node* current_node = current_edge->src();

    // If the node is stateful, assume the graph is not constant.
    if (current_node->op_def().is_stateful()) {
      *is_constant_graph = false;
      return Status::OK();
    }

    // If there is nothing more to recurse down, see if
    // the generator node is a constant.
    if (current_node->num_inputs() == 0) {
      if (!current_node->IsConstant()) {
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
    bool first_visit_to_node = false;
    Node* current_node_copy;
    {
      auto it = old_to_new.find(current_node);
      if (it == old_to_new.end()) {
        // First time processing this node.
        first_visit_to_node = true;
        current_node_copy = out_graph->CopyNode(current_node);
        // Track the mapping from the original node to the new one.
        old_to_new[current_node] = current_node_copy;
      } else {
        current_node_copy = it->second;
      }
    }

    // Add the edge to the destination node.
    {
      auto it = old_to_new.find(current_edge->dst());
      if (it == old_to_new.end()) {
        return errors::Internal(
            "Could not find mapping from old to new copy of destination node: ",
            current_edge->dst()->name());
      }
      Node* dst_copy = it->second;

      out_graph->AddEdge(current_node_copy, current_edge->src_output(),
                         dst_copy, current_edge->dst_input());
    }

    // If we have a copy of the input tensor materialized already,
    // then add to the list of inputs to feed and do not recurse further.
    const string& output_tensor_name =
        strings::StrCat(current_node->name(), ":", current_edge->src_output());
    auto it = const_tensor_map_.find(output_tensor_name);
    if (it != const_tensor_map_.end() &&
        const_inputs_added.count(output_tensor_name) == 0) {
      const_inputs->emplace_back(
          std::make_pair(output_tensor_name, it->second));
      const_inputs_added.insert(output_tensor_name);
      continue;
    }

    // If this is the first time visiting this node, recurse on this
    // node's inputs.
    if (first_visit_to_node) {
      for (const Edge* e : current_node->in_edges()) {
        if (e->IsControlEdge()) continue;
        edges_to_visit.push_back(e);
      }
    }
  }

  return Status::OK();
}

Status ShapeRefiner::ConstantPartialShape(InferenceContext* target_context,
                                          const Node* node, int dst_idx,
                                          ShapeHandle* result) {
  const Edge* input_edge;
  TF_RETURN_IF_ERROR(node->input_edge(dst_idx, &input_edge));

  InferenceContext* src_context = GetContext(input_edge->src());
  if (src_context == nullptr) return errors::Internal("Missing src context");
  ShapeHandle src_shape = src_context->output(input_edge->src_output());
  TF_RETURN_IF_ERROR(src_context->WithRank(src_shape, 1, &src_shape));

  const string& src_op = input_edge->src()->type_string();
  if (src_context->Value(src_context->Dim(src_shape, 0)) == 0) {
    // Source tensor is a vector of length 0, so the shape it
    // represents is as scalar.
    *result = target_context->Scalar();
  } else if (src_op == "Shape") {
    *result = src_context->input(0);
  } else if (src_op == "Pack") {
    std::vector<DimensionHandle> dims;
    // Pack is concatenating its input scalars to form the shape tensor vector.
    for (int i = 0; i < src_context->num_inputs(); ++i) {
      Tensor scalar;
      bool evaluated = false;
      TF_RETURN_IF_ERROR(EvaluateConstantTensorForEdge(input_edge->src(), i,
                                                       &evaluated, &scalar));
      if (evaluated) {
        int64 size;
        if (scalar.dtype() == DT_INT32) {
          size = scalar.scalar<int32>()();
        } else if (scalar.dtype() == DT_INT64) {
          size = scalar.scalar<int64>()();
        } else {
          return errors::InvalidArgument("Pack input must be int32 or int64");
        }
        dims.push_back(size < 0 ? target_context->UnknownDim()
                                : target_context->MakeDim(size));
      } else {
        dims.push_back(target_context->UnknownDim());
      }
    }
    *result = target_context->MakeShape(dims);
  } else if (src_op == "Concat") {
    *result = target_context->Scalar();
    // Concat is concatenating its input shape vectors.
    // input 0 is ignored as it is the concat dim and will always be 0.
    for (int i = 1; i < src_context->num_inputs(); ++i) {
      ShapeHandle sub_result;
      TF_RETURN_IF_ERROR(ConstantPartialShape(target_context, input_edge->src(),
                                              i, &sub_result));
      if (!target_context->RankKnown(sub_result)) {
        // Failed to evaluate. Treat the output as completely unknown.
        // TODO(cwhipkey): we could rely on all inputs being the same size, so
        // figure that size out and append the right number of unknown dims.
        *result = target_context->UnknownShape();
        return Status::OK();
      }
      TF_RETURN_IF_ERROR(
          target_context->Concatenate(*result, sub_result, result));
    }
  } else {
    Tensor t;
    bool evaluated = false;
    TF_RETURN_IF_ERROR(
        EvaluateConstantTensorForEdge(node, dst_idx, &evaluated, &t));
    TF_RETURN_IF_ERROR(target_context->MakeShapeFromTensor(
        evaluated ? &t : nullptr, src_shape, result));
  }
  return Status::OK();
}

}  // namespace tensorflow

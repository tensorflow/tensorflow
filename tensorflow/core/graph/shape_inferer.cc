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
#include "tensorflow/core/graph/shape_inferer.h"

#include <memory>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/stl_util.h"

namespace tensorflow {

ShapeInferer::ShapeInferer() {}
ShapeInferer::~ShapeInferer() { gtl::STLDeleteValues(&node_to_context_); }

Status ShapeInferer::AddNode(const Node* node) {
  // For each 'input' of this node, fetch the corresponding shape
  // from 'input's InferenceContext, and store into a vector
  // indexed by 'node's input.
  std::vector<const Node*> input_nodes(node->num_inputs());
  std::vector<const shape_inference::Shape*> input_shapes(node->num_inputs());
  for (const Edge* e : node->in_edges()) {
    if (e->IsControlEdge()) continue;

    const Node* input = e->src();
    auto it = node_to_context_.find(input);
    if (it == node_to_context_.end()) {
      return errors::FailedPrecondition(
          "Input ", e->dst_input(), " ('", input->name(), "') for '",
          node->name(), "' was not previously added to ShapeInferer.");
    }

    shape_inference::InferenceContext* c = it->second;
    DCHECK_GE(e->dst_input(), 0);
    input_nodes[e->dst_input()] = input;
    input_shapes[e->dst_input()] = c->output(e->src_output());
  }

  // Get the shape function for this node
  const OpRegistrationData* op_reg_data;
  // TODO(vrv): Take in the OpRegistryInterface* instead of taking
  // the global one.
  TF_RETURN_IF_ERROR(
      OpRegistry::Global()->LookUp(node->type_string(), &op_reg_data));
  if (op_reg_data->shape_inference_fn == nullptr) {
    return errors::InvalidArgument(
        "No shape inference function exists for op '", node->type_string(),
        "', did you forget to define it?");
  }

  // This needs to be filled in with real data in a second pass.
  std::vector<const Tensor*> input_tensors(node->num_inputs());
  std::vector<Tensor> real_tensors(node->num_inputs());
  std::vector<bool> attempted_materialization(node->num_inputs());

  // Create the inference context for this node with the existing input shapes.
  std::unique_ptr<shape_inference::InferenceContext> c(
      new shape_inference::InferenceContext(&node->def(), node->op_def(),
                                            {} /* input_shapes_string */,
                                            input_shapes, input_tensors));
  if (!c->construction_status().ok()) {
    return c->construction_status();
  }

  // Run the shape inference function, and return if there was an error.
  TF_RETURN_IF_ERROR(op_reg_data->shape_inference_fn(c.get()));

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
    for (int i = 0; i < c->num_inputs(); ++i) {
      // Check if we have not already filled in the requested input,
      // and if not, try to materialize the tensors.
      if (c->requested_input_tensor(i) && !attempted_materialization[i]) {
        rerun_shape_fn = true;
        attempted_materialization[i] = true;

        // For now, we do a simple static analysis of the graph to
        // materialize those tensors, but in the future, we should try to do
        // a partial evaluation of the graph.
        const Node* input_node = input_nodes[i];

        // TODO(vrv): Handle other types of nodes, like we do in python:
        // Shape, Size, Rank, Range, Cast, Concat, Pack.  Some of these
        // require recursively accessing the input's inputs, while some
        // can be computed using local information.
        if (input_node->IsConstant()) {
          TF_RETURN_IF_ERROR(
              GetNodeAttr(input_node->def(), "value", &real_tensors[i]));
          input_tensors[i] = &real_tensors[i];
        }
      }
    }

    if (rerun_shape_fn) {
      // We have more information about the shapes on this pass,
      // so re-run shape inference.
      c->set_input_tensors(input_tensors);
      TF_RETURN_IF_ERROR(op_reg_data->shape_inference_fn(c.get()));
    }
  } while (rerun_shape_fn);

  // Store the resulting InferenceContext object in the map.
  node_to_context_[node] = c.release();

  return Status::OK();
}

}  // namespace tensorflow

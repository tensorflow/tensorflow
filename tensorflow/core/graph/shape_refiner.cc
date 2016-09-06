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
#include "tensorflow/core/graph/shape_refiner.h"

#include <memory>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/stl_util.h"

namespace tensorflow {

ShapeRefiner::ShapeRefiner() {}
ShapeRefiner::~ShapeRefiner() { gtl::STLDeleteValues(&node_to_context_); }

Status ShapeRefiner::AddNode(const Node* node) {
  // For each 'input' of this node, fetch the corresponding shape
  // from 'input's InferenceContext, and store into a vector
  // indexed by 'node's input.
  std::vector<const Node*> input_nodes(node->num_inputs());
  std::vector<shape_inference::ShapeHandle> input_shapes(node->num_inputs());
  for (const Edge* e : node->in_edges()) {
    if (e->IsControlEdge()) continue;

    const Node* input = e->src();
    auto it = node_to_context_.find(input);
    if (it == node_to_context_.end()) {
      return errors::FailedPrecondition(
          "Input ", e->dst_input(), " ('", input->name(), "') for '",
          node->name(), "' was not previously added to ShapeRefiner.");
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
        TF_RETURN_IF_ERROR(
            ConstantValue(input_nodes[i], &real_tensors[i], &input_tensors[i]));
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

Status ShapeRefiner::SetShape(const Node* node, int output_port,
                              shape_inference::ShapeHandle shape) {
  auto c = GetContext(node);
  if (c == nullptr) {
    return errors::Internal("Could not find context for ", node->name());
  }

  if (output_port < 0 || output_port >= node->num_outputs()) {
    return errors::InvalidArgument(
        "output_port '", output_port, "' is out of range, ", "node '",
        node->name(), "' has ", node->num_outputs(), " outputs");
  }

  // Check compatibility
  shape_inference::ShapeHandle existing_shape = c->output(output_port);
  shape_inference::ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(existing_shape, shape, &unused));

  c->set_output(output_port, shape);

  // TODO(vrv): Do we need to propagate the new shape through all
  // consumers that change their outputs?  At the moment, python
  // does not do this, but this seems like a nice feature.

  // TODO(vrv): We might need to keep track of the fact that the
  // existing shape is invalidated, in case we need to propagate
  // this information to remote workers.
  return Status::OK();
}

Status ShapeRefiner::ConstantValue(const Node* node, Tensor* tensor_storage,
                                   const Tensor** input_tensor) const {
  *input_tensor = nullptr;
  // For now, we do a simple static analysis of the graph to
  // materialize those tensors, but in the future, we should try to do
  // a partial evaluation of the graph.

  // TODO(vrv): Handle other types of nodes, like we do in python:
  // Cast, Concat, Pack.  These require re-implementing the core
  // kernels themselves, and we may want to switch this to partial
  // evaluation instead of implementing these again.
  if (node->IsConstant()) {
    return Constant(node, tensor_storage, input_tensor);
  }
  if (node->type_string() == "Shape") {
    return Shape(node, tensor_storage, input_tensor);
  }
  if (node->type_string() == "Size") {
    return Size(node, tensor_storage, input_tensor);
  }
  if (node->type_string() == "Rank") {
    return Rank(node, tensor_storage, input_tensor);
  }
  if (node->type_string() == "Range") {
    return Range(node, tensor_storage, input_tensor);
  }

  return Status::OK();
}

Status ShapeRefiner::Constant(const Node* node, Tensor* tensor_storage,
                              const Tensor** input_tensor) const {
  TF_RETURN_IF_ERROR(GetNodeAttr(node->def(), "value", tensor_storage));
  *input_tensor = tensor_storage;
  return Status::OK();
}

Status ShapeRefiner::Shape(const Node* node, Tensor* tensor_storage,
                           const Tensor** input_tensor) const {
  // Get the input to the node.
  const Node* shape_node;
  TF_RETURN_IF_ERROR(node->input_node(0, &shape_node));
  auto ic = GetContext(shape_node);
  if (!ic) {
    return errors::Internal("Could not find InferenceContext for ",
                            shape_node->name());
  }
  shape_inference::ShapeHandle input_shape = ic->output(0);
  if (ic->FullyDefined(input_shape)) {
    *tensor_storage = Tensor(DT_INT32, {ic->Rank(input_shape)});
    for (int i = 0; i < ic->Rank(input_shape); ++i) {
      int64 dim = ic->Value(ic->Dim(input_shape, i));
      if (dim > std::numeric_limits<int32>::max()) {
        // The output of Shape is 32-bits, so we cannot fill in anything
        // here.  See b/28119922.
        return Status::OK();
      }

      tensor_storage->vec<int32>()(i) = dim;
    }
    *input_tensor = tensor_storage;
  }

  return Status::OK();
}

Status ShapeRefiner::Size(const Node* node, Tensor* tensor_storage,
                          const Tensor** input_tensor) const {
  // Get the input to the node.
  const Node* size_node;
  TF_RETURN_IF_ERROR(node->input_node(0, &size_node));
  auto ic = GetContext(size_node);
  if (!ic) {
    return errors::Internal("Could not find InferenceContext for ",
                            size_node->name());
  }
  auto num_elements = ic->NumElements(ic->output(0));
  if (ic->ValueKnown(num_elements)) {
    *tensor_storage = Tensor(DT_INT32, {});
    int64 ne = ic->Value(num_elements);
    if (ne > std::numeric_limits<int32>::max()) {
      // The output of Size is 32-bits, so we cannot fill in anything
      // here.  See b/28119922.
      return Status::OK();
    }
    tensor_storage->scalar<int32>()() = ic->Value(num_elements);
    *input_tensor = tensor_storage;
  }

  return Status::OK();
}

Status ShapeRefiner::Rank(const Node* node, Tensor* tensor_storage,
                          const Tensor** input_tensor) const {
  // Get the input to the node.
  const Node* rank_node;
  TF_RETURN_IF_ERROR(node->input_node(0, &rank_node));
  auto ic = GetContext(rank_node);
  if (!ic) {
    return errors::Internal("Could not find InferenceContext for ",
                            rank_node->name());
  }

  if (ic->RankKnown(ic->output(0))) {
    int32 rank = ic->Rank(ic->output(0));
    *tensor_storage = Tensor(DT_INT32, {});
    tensor_storage->scalar<int32>()() = rank;
    *input_tensor = tensor_storage;
  }

  return Status::OK();
}

Status ShapeRefiner::Range(const Node* node, Tensor* tensor_storage,
                           const Tensor** input_tensor) const {
  const Node* start_node;
  TF_RETURN_IF_ERROR(node->input_node(0, &start_node));
  const Node* limit_node;
  TF_RETURN_IF_ERROR(node->input_node(1, &limit_node));
  const Node* delta_node;
  TF_RETURN_IF_ERROR(node->input_node(2, &delta_node));

  const Tensor* start_node_tensor;
  TF_RETURN_IF_ERROR(
      ConstantValue(start_node, tensor_storage, &start_node_tensor));
  if (start_node_tensor == nullptr) return Status::OK();
  const int32 start = start_node_tensor->scalar<int32>()();

  const Tensor* limit_node_tensor;
  TF_RETURN_IF_ERROR(
      ConstantValue(limit_node, tensor_storage, &limit_node_tensor));
  if (limit_node_tensor == nullptr) return Status::OK();
  const int32 limit = limit_node_tensor->scalar<int32>()();

  const Tensor* delta_node_tensor;
  TF_RETURN_IF_ERROR(
      ConstantValue(delta_node, tensor_storage, &delta_node_tensor));
  if (delta_node_tensor == nullptr) return Status::OK();
  const int32 delta = delta_node_tensor->scalar<int32>()();

  if (start > limit) {
    return errors::InvalidArgument("Range requires start <= limit: ", start,
                                   "/", limit);
  }

  if (delta <= 0) {
    return errors::InvalidArgument("Range requires delta > 0: ", delta);
  }

  int32 size = (limit - start + delta - 1) / delta;
  *tensor_storage = Tensor(DT_INT32, {size});

  auto flat = tensor_storage->flat<int32>();
  int32 val = start;
  for (int32 i = 0; i < size; ++i) {
    flat(i) = val;
    val += delta;
  }

  *input_tensor = tensor_storage;
  return Status::OK();
}

}  // namespace tensorflow

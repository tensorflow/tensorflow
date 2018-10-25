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

#include "tensorflow/compiler/jit/shape_inference.h"

#include "tensorflow/compiler/jit/shape_inference_helpers.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {

namespace {

// Converts a shape inference handle to a PartialTensorShape.
Status ShapeHandleToTensorShape(shape_inference::InferenceContext* context,
                                const shape_inference::ShapeHandle& handle,
                                PartialTensorShape* shape) {
  // The default is already unknown
  if (!context->RankKnown(handle)) return Status::OK();

  std::vector<int64> dims(context->Rank(handle));
  for (int32 i = 0; i < dims.size(); ++i) {
    dims[i] = context->Value(context->Dim(handle, i));
  }
  return PartialTensorShape::MakePartialShape(dims.data(), dims.size(), shape);
}

Status PropagateShapes(const Graph& graph,
                       const std::map<int, InferredShape>& arg_shapes,
                       ShapeRefiner* shape_refiner) {
  // Visits the nodes in topological order (reverse post-order), inferring
  // shapes.
  // TODO(phawkins): handle cyclic graphs.
  std::vector<Node*> order;
  GetReversePostOrder(graph, &order);

  for (Node* n : order) {
    // Ignore the status returned by the shape_refiner. We want the best effort
    // shapes, even if no shape function is registered for a node.
    Status status = shape_refiner->AddNode(n);
    if (!status.ok()) {
      VLOG(1) << "Shape inference failed for node: " << status;
    }

    if (n->type_string() == "_Arg") {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      auto it = arg_shapes.find(index);
      if (it != arg_shapes.end()) {
        const InferredShape& arg_shape = it->second;
        shape_inference::InferenceContext* context =
            shape_refiner->GetContext(n);

        if (arg_shape.handle_type != DT_INVALID) {
          shape_inference::ShapeHandle handle;
          TF_RETURN_IF_ERROR(context->MakeShapeFromPartialTensorShape(
              arg_shape.handle_shape, &handle));

          // Sets the shape and type of the variable's value.
          context->set_output_handle_shapes_and_types(
              0, std::vector<shape_inference::ShapeAndType>{
                     {handle, arg_shape.handle_type}});
        }

        shape_inference::ShapeHandle handle;
        TF_RETURN_IF_ERROR(
            context->MakeShapeFromPartialTensorShape(arg_shape.shape, &handle));
        TF_RETURN_IF_ERROR(shape_refiner->SetShape(n, 0, handle));
      }
    }
  }
  return Status::OK();
}

// Store the shapes of the output tensors in a map
Status StoreOutputShapes(const Graph& graph, const ShapeRefiner& shape_refiner,
                         GraphShapeInfo* shape_info) {
  for (const Node* node : graph.nodes()) {
    shape_inference::InferenceContext* context = shape_refiner.GetContext(node);
    if (!context) continue;

    auto& outputs = (*shape_info)[node->name()];
    outputs.resize(context->num_outputs());
    for (int i = 0; i < context->num_outputs(); ++i) {
      auto& output = outputs[i];
      TF_RETURN_IF_ERROR(
          ShapeHandleToTensorShape(context, context->output(i), &output.shape));

      const auto* handle_shapes_and_types =
          context->output_handle_shapes_and_types(i);
      if (handle_shapes_and_types != nullptr) {
        if (handle_shapes_and_types->size() == 1) {
          TF_RETURN_IF_ERROR(ShapeHandleToTensorShape(
              context, (*handle_shapes_and_types)[0].shape,
              &output.handle_shape));
          output.handle_type = (*handle_shapes_and_types)[0].dtype;
        } else {
          // otherwise, it may be resource like a Queue, which can have
          // multiple shapes and types represented by a single handle.
        }
      }
      VLOG(4) << node->name() << " output " << i << " shape"
              << output.shape.DebugString() << " handle_type "
              << DataTypeString(output.handle_type) << " handle_shape "
              << output.handle_shape.DebugString();
    }
  }
  return Status::OK();
}

}  // namespace

Status InferShapes(Graph* graph, const std::map<int, InferredShape>& arg_shapes,
                   const tensorflow::FunctionLibraryDefinition* fnlib_def,
                   GraphShapeInfo* shape_info) {
  ShapeRefiner shape_refiner(graph->versions(), graph->op_registry());
  shape_refiner.set_require_shape_inference_fns(false);
  // TODO(dlibenzi): Verify if it is worth trying to infer shaped within
  // functions. Some functions can be called at multiple locations with
  // difference shapes, which will trigger a shape inference based on the
  // arguments passed at the first call.
  // shape_refiner.set_function_library_for_shape_inference(fnlib_def);

  // ShapeRefiner requires that all inputs of a node are present when
  // ShapeRefiner::AddNode is called. To get at least some shape information in
  // loops, we temporarily remove loop backedges and add them back again after
  // the shape inference is complete.
  BackEdgeHelper back_edge;
  TF_RETURN_IF_ERROR(back_edge.Remove(graph));
  TF_RETURN_IF_ERROR(PropagateShapes(*graph, arg_shapes, &shape_refiner));
  TF_RETURN_IF_ERROR(back_edge.Replace());

  // Currently information does not flow "backward" from consumers to producers
  // in the shape inference, but we consume the shapes in a second pass in case
  // backward information flow is added in the future.
  return StoreOutputShapes(*graph, shape_refiner, shape_info);
}

xla::StatusOr<InferredShape> MergeInferredShapes(const InferredShape& a,
                                                 const InferredShape& b) {
  InferredShape result;
  TF_RETURN_IF_ERROR(a.shape.MergeWith(b.shape, &result.shape));

  if (a.handle_type == DT_INVALID) {
    result.handle_type = b.handle_type;
  } else if (b.handle_type == DT_INVALID) {
    result.handle_type = a.handle_type;
  } else if (a.handle_type == b.handle_type) {
    result.handle_type = a.handle_type;
  } else {
    return errors::InvalidArgument(
        "Mismatched resource types: ", DataTypeString(a.handle_type), " vs. ",
        DataTypeString(b.handle_type));
  }
  TF_RETURN_IF_ERROR(
      a.handle_shape.MergeWith(b.handle_shape, &result.handle_shape));
  return result;
}

}  // namespace tensorflow

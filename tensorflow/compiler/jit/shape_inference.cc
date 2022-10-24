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
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {

// Converts a shape inference handle to a PartialTensorShape.
Status ShapeHandleToTensorShape(shape_inference::InferenceContext* context,
                                const shape_inference::ShapeHandle& handle,
                                PartialTensorShape* shape) {
  // The default is already unknown
  if (!context->RankKnown(handle)) return OkStatus();

  std::vector<int64_t> dims(context->Rank(handle));
  for (int32_t i = 0, end = dims.size(); i < end; ++i) {
    dims[i] = context->Value(context->Dim(handle, i));
  }
  return PartialTensorShape::MakePartialShape(dims.data(), dims.size(), shape);
}

Status PropagateShapes(Graph* graph,
                       const std::map<int, InferredShape>& arg_shapes,
                       const std::vector<BackEdgeHelper::BackEdge>& back_edges,
                       ShapeRefiner* shape_refiner) {
  std::map<const Node*, const Node*> merge_to_next_iteration;
  for (const auto& e : back_edges) {
    if (e.src->IsNextIteration() && e.dst->IsMerge()) {
      merge_to_next_iteration[e.dst] = e.src;
    }
  }

  // Visits the nodes in topological order (reverse post-order), inferring
  // shapes.
  // TODO(phawkins): handle cyclic graphs.
  std::vector<Node*> order;
  GetReversePostOrder(*graph, &order);

  for (Node* n : order) {
    // Ignore the status returned by the shape_refiner. We want the best effort
    // shapes, even if no shape function is registered for a node.
    Status status = shape_refiner->AddNode(n);
    if (!status.ok()) {
      VLOG(1) << "Shape inference failed for node " << n->name() << ": "
              << status;
    } else {
      shape_inference::InferenceContext* context = shape_refiner->GetContext(n);
      for (int i = 0; i < n->num_outputs(); i++) {
        shape_inference::ShapeHandle handle = context->output(i);
        VLOG(4) << "Output " << i << " for node " << n->name() << ": "
                << context->DebugString(handle);
      }
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

    // Sometimes we have VariableShape nodes in while loop (after Enter nodes).
    // They won't be constant-folded because TensorFlow constant folding does
    // not handle Enter nodes (and thus does not handle any nodes after Enter
    // nodes). We try to replace such VariableShape nodes with Const nodes here.
    if (n->type_string() == "VariableShape") {
      shape_inference::InferenceContext* context = shape_refiner->GetContext(n);
      auto handle_shapes_and_types = context->input_handle_shapes_and_types(0);
      if (handle_shapes_and_types && !handle_shapes_and_types->empty()) {
        shape_inference::ShapeHandle handle =
            handle_shapes_and_types->at(0).shape;
        TensorShapeProto shape_proto;
        context->ShapeHandleToProto(handle, &shape_proto);
        if (!shape_proto.unknown_rank()) {
          NodeDef const_def;
          const_def.set_op("Const");
          Node* var_node;
          TF_RETURN_IF_ERROR(n->input_node(0, &var_node));
          const_def.set_name(
              graph->NewName(absl::StrCat("var_shape_", var_node->name())));
          DataType dtype = n->output_type(0);
          AddNodeAttr("dtype", dtype, &const_def);
          TensorProto value;
          value.set_dtype(dtype);
          value.mutable_tensor_shape()->add_dim()->set_size(
              shape_proto.dim_size());
          for (const auto& dim : shape_proto.dim()) {
            if (dtype == DT_INT32) {
              value.add_int_val(dim.size());
            } else {
              value.add_int64_val(dim.size());
            }
          }
          AddNodeAttr("value", value, &const_def);
          for (auto const& attr : n->attrs()) {
            if (*attr.first.begin() == '_') {
              AddNodeAttr(attr.first, attr.second, &const_def);
            }
          }

          TF_ASSIGN_OR_RETURN(Node * const_node, graph->AddNode(const_def));
          graph->AddControlEdge(var_node, const_node);
          std::vector<const Edge*> out_edges(n->out_edges().begin(),
                                             n->out_edges().end());
          for (const Edge* e : out_edges) {
            if (e->IsControlEdge()) {
              graph->AddControlEdge(const_node, e->dst());
              graph->RemoveEdge(e);
            } else {
              Node* dst = e->dst();
              int dst_input = e->dst_input();
              graph->RemoveEdge(e);
              graph->AddEdge(const_node, 0, dst, dst_input);
            }
          }
        }
      }
    }

    // Merge node causes a loop so we remove NextIteration->Merge edge before
    // performing shape inference. But removing those edges also prevents us
    // from inferring output shape for Merge node (we need shapes for all its
    // inputs).
    // For loop invariant resource input's Merge node, we set output resource
    // shape as Enter node's resource shape.
    // TODO(b/129367850): clean this up.
    if (n->IsMerge() && n->output_type(0) == DT_RESOURCE) {
      // Check if this is a loop invariant input's Merge node. We do it by
      // checking if corresponding NextIteration node comes from Switch node
      // directly.
      auto iter = merge_to_next_iteration.find(n);
      if (iter != merge_to_next_iteration.end()) {
        const Node *next_iter = iter->second, *node = next_iter;
        do {
          TF_RETURN_IF_ERROR(node->input_node(0, &node));
        } while (node->IsIdentity());
        const Node* switch_input;
        bool is_loop_invariant = node->IsSwitch() &&
                                 node->input_node(0, &switch_input).ok() &&
                                 switch_input == n;
        if (is_loop_invariant) {
          shape_inference::InferenceContext* context =
              shape_refiner->GetContext(n);
          for (int i = 0; i < n->num_inputs(); i++) {
            const Node* input_node;
            if (n->input_node(i, &input_node).ok()) {
              auto shapes_and_types = context->input_handle_shapes_and_types(i);
              if (shapes_and_types) {
                context->set_output_handle_shapes_and_types(0,
                                                            *shapes_and_types);
              }
              break;
            }
          }
        }
      }
    }
  }
  return OkStatus();
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
  return OkStatus();
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
  TF_RETURN_IF_ERROR(PropagateShapes(graph, arg_shapes,
                                     back_edge.RemovedEdges(), &shape_refiner));
  TF_RETURN_IF_ERROR(back_edge.Replace());

  // Currently information does not flow "backward" from consumers to producers
  // in the shape inference, but we consume the shapes in a second pass in case
  // backward information flow is added in the future.
  return StoreOutputShapes(*graph, shape_refiner, shape_info);
}

StatusOr<InferredShape> MergeInferredShapes(const InferredShape& a,
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

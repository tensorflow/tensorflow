/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/cc/dtensor_device_util.h"

#include <cstddef>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/small_constant_optimization.h"

namespace tensorflow {
namespace dtensor {
namespace {
// Represents an input node during graph construction.
// When executing a Function, `output` is used to align graph inputs
// with the inputs to the function call.
struct FunctionArgument {
  Node* node;
  NodeDefBuilder::NodeOut output;
};

bool LayoutsAreCompatible(absl::optional<Layout> first_layout,
                          absl::optional<Layout> second_layout) {
  if (!first_layout.has_value() && !second_layout.has_value()) {
    return true;
  }
  if (!first_layout.has_value() || !second_layout.has_value()) {
    return false;
  }
  return first_layout.value() == second_layout.value();
}

// Parse a pair of attribute of (indices, layouts) into a map.
Status ParseAttrMap(const Node& node, absl::string_view indices_attr,
                    absl::string_view layout_attr,
                    std::map<int, Layout>* indices_layout_map) {
  std::vector<std::string> layouts;
  if (!TryGetNodeAttr(node.attrs(), layout_attr, &layouts)) {
    return OkStatus();
  }
  const TensorProto* indices;
  if (!TryGetNodeAttr(node.attrs(), indices_attr, &indices)) {
    return errors::Internal(
        "Arg indices must be set when setting inferred resource layouts.");
  }
  if (indices->int_val_size() != layouts.size()) {
    return errors::Internal(
        "Arg indices for inferred resource argument must match the "
        "size of inferred resource layout.");
  }
  for (int i = 0; i < indices->int_val_size(); ++i) {
    const auto arg_index = indices->int_val(i);
    const auto& arg_layout = layouts[i];
    indices_layout_map->emplace(
        arg_index,
        tensorflow::dtensor::Layout::FromString(arg_layout).ValueOrDie());
  }
  return OkStatus();
}

Status ParseResourceArgumentLayouts(
    const Node& node, std::map<int, Layout>* inferred_resource_input_layouts) {
  return ParseAttrMap(node, kNewResourceLayoutIndices, kNewResourceArgLayouts,
                      inferred_resource_input_layouts);
}

Status ParseShapeInputLayouts(const Node& node,
                              std::map<int, Layout>* shape_output_metadata) {
  return ParseAttrMap(node, kShapeOpInputLayoutIndices, kShapeOpInputLayout,
                      shape_output_metadata);
}

// Gets the layout attached to a specific node at a given index, ignoring any
// Identity ops.
StatusOr<Layout> GetLayoutThroughIdentityOps(Node* op, int output_index) {
  while (op->op_def().name() == "Identity" ||
         op->op_def().name() == "IdentityN") {
    const Edge* edge;
    TF_RETURN_IF_ERROR(op->input_edge(output_index, &edge));
    op = edge->src();
    output_index = edge->src_output();
  }
  const auto serialized_layouts = op->attrs().Find(kLayoutAttr);

  if (!serialized_layouts) {
    return errors::InvalidArgument(
        op->op_def().name(), " doesn't contain attribute : ", kLayoutAttr);
  }

  // We assume that there is one layout for each output.
  if (serialized_layouts->list().s_size() != op->num_outputs()) {
    return errors::InvalidArgument(
        "Number of outputs to ", op->op_def().name(),
        " does not match number of layouts attached");
  }

  return Layout::FromString(serialized_layouts->list().s(output_index));
}

}  // namespace

tensorflow::Fprint128 TensorWithLayout::CacheKey() const {
  tensorflow::Fprint128 f = tensorflow::Fingerprint128(layout_.ToString());
  // Use exact shape to compute the key.
  for (const int64_t dim : local_shape()) {
    f = FingerprintCat128(f, dim);
  }
  if (const_value_.has_value()) {
    std::string serialized;
    SerializeToStringDeterministic(const_value_.value(), &serialized);
    f = FingerprintCat128(f, tensorflow::Fingerprint128(serialized));
  }
  return f;
}

std::unique_ptr<TensorWithLayout> TensorWithLayout::Broadcast(
    TFE_Context* context, TFE_TensorHandle* tensor,
    const MeshWithParallelDevice& mesh, const std::string& dtensor_device_name,
    TF_Status* status) {
  const char* input_device = TFE_TensorHandleDeviceName(tensor, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  if (dtensor_device_name == input_device) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Input to Broadcast must be eager tensor.");
    return nullptr;
  }

  if (TFE_TensorHandleDataType(tensor) == TF_RESOURCE) {
    std::string error_message =
        "Using a non-DTensor variable with DTensor is not supported. If you "
        "are using a scope-based API, create variables inside the DTensor "
        "scope.\n";

    // Resolve the Tensor as resource handle and try to get the stack_trace and
    // Summaries out of it.
    std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> tf_tensor(
        TFE_TensorHandleResolve(tensor, status), TF_DeleteTensor);
    Tensor t;
    Status convert_status = TF_TensorToTensor(tf_tensor.get(), &t);
    if (convert_status.ok() && t.dtype() == DataType::DT_RESOURCE) {
      ResourceHandle r = t.flat<ResourceHandle>()(0);
      absl::StrAppend(
          &error_message, "Offending variable summary: ", r.SummarizeValue(),
          "\nStack trace: ", DefinitionLocationMsg(r.definition_stack_trace()));
    }
    TF_SetStatus(status, TF_INVALID_ARGUMENT, error_message.c_str());
    return nullptr;
  }

  if (mesh.mesh_config().is_remote()) {
    TF_DataType dtype = TFE_TensorHandleDataType(tensor);
    std::vector<int64_t> shape(TensorShapeAsVector(tensor, status));
    if (TF_GetCode(status) != TF_OK) return nullptr;
    auto layout = Layout::ReplicatedOnMesh(mesh.mesh_config(), shape.size());

    auto ret = TensorWithLayout::Dummy(shape, dtype, mesh, layout);
    absl::optional<NodeDef> const_value =
        ExtractSmallTensorValue(context, tensor, layout, status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    if (const_value) {
      ret->set_const_value(const_value.value());
    }
    return ret;
  }

  // Broadcast tensor value to local devices.
  const Mesh& target_mesh = mesh.mesh_config();
  absl::Span<const std::string> local_devices = target_mesh.local_devices();
  const int num_local_devices = local_devices.size();

  std::vector<parallel_device::TensorHandlePtr> components;
  components.reserve(num_local_devices);
  for (int i = 0; i < num_local_devices; ++i) {
    // Create tensor copies to each local devices specifie by `target_mesh`.
    components.emplace_back(TFE_TensorHandleCopyToDevice(
        tensor, context, local_devices[i].c_str(), status));
    if (TF_GetCode(status) != TF_OK) {
      TF_SetStatus(
          status, TF_INTERNAL,
          absl::StrCat(
              "Unable to copy tensor value for broadcast. Original message: ",
              TF_Message(status))
              .c_str());
      return nullptr;
    }
  }

  std::unique_ptr<parallel_device::ParallelTensor> parallel_tensor =
      parallel_device::ParallelTensor::FromTensorHandles(
          mesh.parallel_device(), std::move(components), status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  const std::vector<int64_t>* shape;
  Status s = parallel_tensor->Shape(&shape);
  if (!s.ok()) {
    TF_SetStatus(status, static_cast<TF_Code>(s.code()),
                 s.error_message().c_str());
    return nullptr;
  }
  size_t num_dims = shape->size();

  const Layout layout = Layout::ReplicatedOnMesh(mesh.mesh_config(), num_dims);
  absl::optional<NodeDef> const_value =
      ExtractSmallTensorValue(context, tensor, layout, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  std::unique_ptr<TensorWithLayout> result(new TensorWithLayout(
      std::move(parallel_tensor), mesh, std::move(layout), *shape,
      /*dtype=*/absl::nullopt, std::move(const_value)));
  return result;
}

StatusOr<std::unique_ptr<TensorWithLayout>> TensorWithLayout::Wrap(
    std::unique_ptr<parallel_device::ParallelTensor> tensor,
    const MeshWithParallelDevice& mesh, const Layout& layout) {
  const std::vector<int64_t>* shape;
  TF_RETURN_IF_ERROR(tensor->Shape(&shape));

  if (tensor->dtype() != TF_RESOURCE) {
    return std::unique_ptr<TensorWithLayout>(
        new TensorWithLayout(std::move(tensor), mesh, layout, *shape));
  } else {
    return std::unique_ptr<TensorWithLayout>(
        new ResourceHandleWithLayout(std::move(tensor), mesh, layout, *shape));
  }
}

std::unique_ptr<TensorWithLayout> TensorWithLayout::Dummy(
    const std::vector<int64_t>& local_shape, const TF_DataType dtype,
    const MeshWithParallelDevice& mesh, const Layout& layout) {
  if (dtype != TF_RESOURCE) {
    return std::unique_ptr<TensorWithLayout>(new TensorWithLayout(
        /*tensor=*/nullptr, mesh, layout, local_shape, dtype));
  } else {
    return std::unique_ptr<TensorWithLayout>(new ResourceHandleWithLayout(
        /*tensor=*/nullptr, mesh, layout, local_shape));
  }
}

std::string TensorWithLayout::SummarizeValue() const {
  std::string value_summary;
  Status status;
  if (layout().IsFullyReplicated()) {
    status =
        tensorflow::unwrap(tensor()->tensor(0))->SummarizeValue(value_summary);
  } else {
    // Note that this just prints the local values for sharded tensors. We could
    // instead run a collective here to relayout to replicated.
    status = tensor()->SummarizeValue(value_summary);
  }
  if (!status.ok()) {
    value_summary = "<error computing value>";
  }
  return absl::StrCat(value_summary, ", layout=\"", layout().ToString(), "\"");
}

std::string TensorWithLayout::DebugString() const {
  auto dtype = static_cast<DataType>(tensor()->dtype());

  const auto& shape_vector = global_shape();
  return absl::StrCat("DTensor(", SummarizeValue(),
                      ", shape=", ShapeToDebugString(shape_vector),
                      ", type=", DataTypeString(dtype), ")");
}

void ResourceHandleWithLayout::EncodeAttributes(
    tensorflow::NodeDefBuilder& builder) const {
  // If set, attach shape and dtype to the given node def.
  if (dereferenced_shape().has_value()) {
    builder.Attr("_handle_shapes", {*dereferenced_shape()});
  }
  if (dereferenced_dtype().has_value()) {
    builder.Attr("_handle_dtypes", {*dereferenced_dtype()});
  }
}

tensorflow::Fprint128 ResourceHandleWithLayout::CacheKey() const {
  tensorflow::Fprint128 f = tensorflow::Fingerprint128(layout().ToString());
  if (dereferenced_shape().has_value()) {
    std::string serialized;
    SerializeToStringDeterministic(dereferenced_shape().value(), &serialized);
    f = FingerprintCat128(f, tensorflow::Fingerprint128(serialized));
  }
  if (dereferenced_dtype().has_value()) {
    f = FingerprintCat128(f, dereferenced_dtype().value());
  }
  return f;
}

void ResourceHandleWithLayout::UpdateLayout(const Layout& new_layout,
                                            TF_Status* status) {
  // Only set the value for deferenced layout if the incoming layout is not
  // empty. This is still hacky as we use empty layout as placeholder for
  // eagerly placed VarHandleOp.
  if (!dereferenced_layout_.has_value() && new_layout.IsEmpty()) return;
  if (dereferenced_layout_.has_value() &&
      !LayoutsAreCompatible(dereferenced_layout_, new_layout)) {
    // TODO(xiejw, allenl): Consider allowing variables to switch layouts.
    RETURN_STATUS(status, TF_INVALID_ARGUMENT,
                  "Attempted to overwrite an existing Layout.");
  }
  dereferenced_layout_.emplace(new_layout);
}

void ResourceHandleWithLayout::UpdateAttrs(const EmbeddingResourceAttrs& attrs,
                                           TF_Status* status) {
  if (attrs_.has_value()) {
    RETURN_STATUS(status, TF_INVALID_ARGUMENT,
                  "Attepted to overwrite an existing embedding resource "
                  "attribute.");
  }
  attrs_.emplace(attrs);
}

StatusOr<std::unique_ptr<TensorWithLayout>> SparseTensorWithLayout::Wrap(
    std::unique_ptr<parallel_device::ParallelTensor> indices_tensor,
    std::unique_ptr<parallel_device::ParallelTensor> values_tensor,
    std::unique_ptr<parallel_device::ParallelTensor> shapes_tensor,
    const MeshWithParallelDevice& mesh, const Layout& layout,
    std::vector<int64_t> local_shape) {
  return std::unique_ptr<TensorWithLayout>(new SparseTensorWithLayout(
      std::move(indices_tensor), std::move(values_tensor),
      std::move(shapes_tensor), mesh, layout, local_shape));
}

std::string SparseTensorWithLayout::SummarizeValue() const {
  std::string indices_summary;
  std::string values_summary;
  std::string dense_shapes_summary;

  Status indices_status;
  Status values_status;
  Status dense_shapes_status;

  if (layout().IsFullyReplicated()) {
    indices_status = tensorflow::unwrap(indices_->tensor(0))
                         ->SummarizeValue(indices_summary);
    values_status =
        tensorflow::unwrap(values_->tensor(0))->SummarizeValue(values_summary);
    dense_shapes_status = tensorflow::unwrap(dense_shapes_->tensor(0))
                              ->SummarizeValue(dense_shapes_summary);
  } else {
    indices_status = indices_->SummarizeValue(indices_summary);
    values_status = values_->SummarizeValue(values_summary);
    dense_shapes_status = dense_shapes_->SummarizeValue(dense_shapes_summary);
  }

  if (!indices_status.ok())
    values_summary = "<error computing summary for indices>";
  if (!values_status.ok())
    indices_summary = "<error computing summary for values>";
  if (!dense_shapes_status.ok())
    indices_summary = "<error computing summary for dense_shapes>";

  return absl::StrCat("indices: ", indices_summary, ", ",
                      "values: ", values_summary, ", ",
                      "dense_shapes: ", dense_shapes_summary, ", layout=\"",
                      layout().ToString(), "\"");
}

std::string SparseTensorWithLayout::DebugString() const {
  auto dtype = static_cast<DataType>(values_->dtype());

  const auto& shape_vector = global_shape();
  return absl::StrCat("DTensor(", SummarizeValue(),
                      ", shape=", ShapeToDebugString(shape_vector),
                      ", type=", DataTypeString(dtype), ")");
}

TF_DataType SparseTensorWithLayout::dtype() const {
  if (dtype_.has_value()) {
    return dtype_.value();
  } else {
    return values_->dtype();
  }
}

TFE_TensorHandle* SparseTensorWithLayout::get_tensor(size_t index) const {
  int num_sparse_tensors = num_tensors() / 3;
  if (index < num_sparse_tensors) {
    return indices()->tensor(index);
  } else if (index < 2 * num_sparse_tensors) {
    return values()->tensor(index % num_sparse_tensors);
  } else {
    return dense_shapes()->tensor(index % num_sparse_tensors);
  }
}

std::vector<int64_t> TensorShapeAsVector(TFE_TensorHandle* tensor,
                                         TF_Status* status) {
  std::vector<int64_t> shape(TFE_TensorHandleNumDims(tensor, status));
  if (TF_GetCode(status) != TF_OK) return {};
  for (int i = 0; i < shape.size(); ++i) {
    shape[i] = TFE_TensorHandleDim(tensor, i, status);
    if (TF_GetCode(status) != TF_OK) return {};
  }
  return shape;
}

Status PrepareGraphForMlir(
    const std::vector<TensorWithLayout*>& inputs,
    const DTensorOperation& doperation,
    const tensorflow::FunctionLibraryDefinition& flib_def,
    const NameAttrList& attributes,
    const absl::optional<Layout>& default_layout, tensorflow::Graph* graph,
    std::vector<PartialTensorShape>* global_output_shapes,
    std::vector<const Layout*>* output_layouts) {
  // We run shape inference on the graph to find output shapes, which may
  // determine default layouts.
  ShapeRefiner shape_refiner(TF_GRAPH_DEF_VERSION, &flib_def);
  shape_refiner.set_function_library_for_shape_inference(&flib_def);
  tensorflow::Status status;
  {
    // We include an _Arg node for the device ID, but this isn't used by the
    // initial function. It will be provided a value, though, so it's available
    // for use in rewrites.
    tensorflow::NodeDefBuilder builder("device_id", "_Arg");
    tensorflow::PartialTensorShape partial_shape;
    TF_RETURN_IF_ERROR(tensorflow::PartialTensorShape::MakePartialShape(
        static_cast<int*>(nullptr), 0, &partial_shape));
    tensorflow::NodeDef arg_node_def;
    TF_RETURN_IF_ERROR(builder.Attr("shape", partial_shape)
                           .Attr("T", tensorflow::DT_INT32)
                           .Attr("index", 0)
                           .Finalize(&arg_node_def, /*consume=*/true));
    tensorflow::Node* arg_node = graph->AddNode(arg_node_def, &status);
    TF_RETURN_IF_ERROR(status);
    graph->AddControlEdge(graph->source_node(), arg_node);
    TF_RETURN_IF_ERROR(shape_refiner.AddNode(arg_node));
  }
  std::vector<FunctionArgument> graph_op_inputs;
  graph_op_inputs.reserve(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    const TensorWithLayout* input = inputs[i];
    // TODO(allenl): This will block until async execution is complete, which
    // will be slow. We should find a non-blocking way of fetching the shape,
    // at least pre-cache.
    // The shape passed into MLIR transformation represents the global shape of
    // the tensor. Ideally, the local shape on each parallel device should not
    // be consulted at all and we should use the shape on our input tensor
    // directly.
    const auto& shape = input->global_shape();
    std::vector<tensorflow::int64> cast_shape(shape.begin(), shape.end());
    tensorflow::PartialTensorShape partial_shape;
    // For resource tensors, `shape` attribute should not be specified as shape
    // of resource tensors is specified by resource shape subtype -- not the
    // shape attribute.
    auto* resource = dynamic_cast<const ResourceHandleWithLayout*>(input);
    if (!resource) {
      TF_RETURN_IF_ERROR(tensorflow::PartialTensorShape::MakePartialShape(
          cast_shape.data(), cast_shape.size(), &partial_shape));
    }

    tensorflow::NodeDef arg_node_def;
    auto dtype = static_cast<tensorflow::DataType>(input->dtype());
    tensorflow::NodeDefBuilder builder(absl::StrCat("op_input_", i), "_Arg");

    // Delegate TensorWithLayout to encode attributes if applicable.
    input->EncodeAttributes(builder);

    // Here we set each arg node's `index` attribute to the position of
    // the dtensor inputs. This is important for later use when we create
    // a mapping from the graph argument node to the corresponding argument
    // index of the list of dtensor inputs. Thus, even if the argument node
    // orderings change within the graph, we can always correctly
    // find the dtensor input corresponding to that arg node.
    //
    // This assumes that the dtensor inputs stay unchanged in ordering,
    // and if there is an ordering change of dtensor inputs, then special
    // care must be taken.
    TF_RETURN_IF_ERROR(
        builder.Attr("shape", partial_shape)
            .Attr("T", dtype)
            .Attr("index", i + 1)  // Indices are offset by 1 for device_id
            .Attr(kLayoutAttr, input->layout().ToString())
            .Attr(kMeshAttr, input->mesh().mesh_config().ToString())
            .Finalize(&arg_node_def, /*consume=*/true));
    Node* arg_node = graph->AddNode(arg_node_def, &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(shape_refiner.AddNode(arg_node));

    shape_inference::InferenceContext* inference_context =
        shape_refiner.GetContext(arg_node);
    shape_inference::ShapeHandle shape_handle;
    TF_RETURN_IF_ERROR(inference_context->MakeShapeFromPartialTensorShape(
        partial_shape, &shape_handle));
    TF_RETURN_IF_ERROR(shape_refiner.SetShape(arg_node, 0, shape_handle));

    // Small constants are converted into constant graph nodes, instead of being
    // passed in as input arguments. This provides more information to the SPMD
    // and layout propagation passes.
    if (!input->const_value().has_value()) {
      graph_op_inputs.push_back(FunctionArgument{
          arg_node, NodeDefBuilder::NodeOut{arg_node->name(), i, dtype}});
      graph->AddControlEdge(graph->source_node(), arg_node);
    } else {
      // TODO(xiejw): Refactor the TensorWithLayout representation to avoid
      // special code here.
      NodeDef const_node = input->const_value().value();
      const_node.set_name(absl::StrCat("input_", i, "_const_value"));
      Node* const_value_n = graph->AddNode(const_node, &status);
      TF_RETURN_IF_ERROR(status);
      TF_RETURN_IF_ERROR(shape_refiner.AddNode(const_value_n));
      graph_op_inputs.push_back(FunctionArgument{
          const_value_n, tensorflow::NodeDefBuilder::NodeOut{
                             const_value_n->name(), i, dtype}});
    }
  }

  tensorflow::NodeDef op_node_def;
  const FunctionDef* function_def = doperation.function_def;
  if (function_def) {
    AttrValue func_attr;
    func_attr.mutable_func()->set_name(doperation.name);
    std::vector<tensorflow::NodeDefBuilder::NodeOut> func_inputs;
    std::vector<tensorflow::DataType> inputs_types;
    for (const auto& in : graph_op_inputs) {
      func_inputs.emplace_back(in.output);
      inputs_types.emplace_back(in.output.data_type);
    }

    std::vector<tensorflow::DataType> output_types;
    for (const auto& out : function_def->signature().output_arg())
      output_types.emplace_back(out.type());

    TF_RETURN_IF_ERROR(
        NodeDefBuilder("eager_operation", "StatefulPartitionedCall")
            .Attr("Tin", inputs_types)
            .Attr("Tout", output_types)
            .Attr("f", func_attr)
            .Input(func_inputs)
            .Finalize(&op_node_def, true));
  } else {
    op_node_def.set_op(doperation.name);
    op_node_def.set_name("eager_operation");
  }

  op_node_def.mutable_attr()->insert(attributes.attr().begin(),
                                     attributes.attr().end());

  tensorflow::Node* op_node = graph->AddNode(op_node_def, &status);
  TF_RETURN_IF_ERROR(status);

  for (int i = 0; i < graph_op_inputs.size(); ++i) {
    graph->AddEdge(graph_op_inputs[i].node, 0, op_node, i);
  }
  TF_RETURN_IF_ERROR(shape_refiner.AddNode(op_node));

  output_layouts->clear();
  output_layouts->reserve(op_node->num_outputs());
  global_output_shapes->reserve(op_node->num_outputs());
  for (int output_index = 0; output_index < op_node->num_outputs();
       ++output_index) {
    tensorflow::NodeDefBuilder builder(absl::StrCat("op_output_", output_index),
                                       "_Retval");
    tensorflow::NodeDef ret_node_def;
    tensorflow::DataType output_type = op_node->output_type(output_index);

    TF_RETURN_IF_ERROR(builder.Attr("T", output_type)
                           .Attr("index", output_index)
                           .Input("eager_operation", output_index, output_type)
                           .Finalize(&ret_node_def, /*consume=*/true));
    tensorflow::Node* ret_node = graph->AddNode(ret_node_def, &status);
    TF_RETURN_IF_ERROR(status);
    graph->AddEdge(op_node, output_index, ret_node, 0);
    graph->AddControlEdge(ret_node, graph->sink_node());

    shape_inference::InferenceContext* inference_context =
        shape_refiner.GetContext(op_node);
    shape_inference::ShapeHandle output_shape_handle =
        inference_context->output(output_index);
    TensorShapeProto output_shape_proto;
    inference_context->ShapeHandleToProto(output_shape_handle,
                                          &output_shape_proto);
    PartialTensorShape global_output_shape(output_shape_proto);
    VLOG(3) << "Inferred shape for operation '" << doperation.name
            << "':" << global_output_shape.DebugString();
    global_output_shapes->push_back(global_output_shape);

    const Layout* layout = nullptr;
    if (default_layout.has_value() && output_index == 0) {
      // Record the user's requested output layout. The scope currently only
      // covers the first output of an op.
      layout = &default_layout.value();
      ret_node->AddAttr(kDefaultLayoutAttr, layout->ToString());
    }
    output_layouts->push_back(layout);
  }
  return OkStatus();
}

// Returns set of functions to run to execute DTensor computation.
StatusOr<ExecutionFunctions> IdentifyAllFunctionsToExecute(
    const tensorflow::Graph& graph,
    const std::vector<PartialTensorShape>& global_output_shapes) {
  ExecutionFunctions execution_functions;
  execution_functions.function_list = std::vector<TranslatedFunction>();
  for (Node* node : graph.nodes()) {
    if (node->op_def().name() != "StatefulPartitionedCall") continue;
    // Extract mesh to execute the function.
    std::string serialized_mesh;
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), kMeshAttr, &serialized_mesh));
    Mesh mesh;
    TF_ASSIGN_OR_RETURN(mesh, Mesh::FromString(serialized_mesh));

    TranslatedFunction function;
    function.function_mesh = std::move(mesh);
    function.node_to_execute = node;

    // Identify input arg information.
    TF_RETURN_IF_ERROR(
        ParseResourceArgumentLayouts(*node, &function.resource_input_layouts));

    TF_RETURN_IF_ERROR(
        ParseShapeInputLayouts(*node, &function.shape_output_metadata));

    function.input_index_map.resize(node->num_inputs());
    // Identity mapping between local mesh function input index and global
    // input index.
    for (int in_index = 0; in_index < node->num_inputs(); ++in_index) {
      Node* input_node;

      TF_RETURN_IF_ERROR(node->input_node(in_index, &input_node));
      if (!input_node->IsArg())
        return errors::InvalidArgument(
            "Input node to mesh computation must be arg node.");

      int global_index;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(input_node->attrs(), "index", &global_index));
      function.input_index_map[in_index] = global_index;
    }

    // Identify output mappings and layouts for each outputs.
    std::map<int, const Edge*> output_edges;
    for (const Edge* out_edge : node->out_edges()) {
      if (out_edge->IsControlEdge()) continue;

      const Node* retval_or_identity_node = out_edge->dst();
      while (retval_or_identity_node->IsIdentity()) {
        retval_or_identity_node =
            *(retval_or_identity_node->out_nodes().begin());
      }

      TF_RET_CHECK(retval_or_identity_node->IsRetval());
      int global_index;
      TF_RETURN_IF_ERROR(GetNodeAttr(retval_or_identity_node->attrs(), "index",
                                     &global_index));
      output_edges[global_index] = out_edge;
    }

    for (auto it = output_edges.begin(); it != output_edges.end(); it++) {
      const int global_index = it->first;
      function.output_index_map.emplace_back(global_index);

      const Edge* retval_edge = it->second;
      const int output_index = retval_edge->src_output();

      // Add output layout and shape information.
      TF_ASSIGN_OR_RETURN(
          const Layout output_layout,
          GetLayoutThroughIdentityOps(retval_edge->src(), output_index));

      function.output_layouts.emplace_back(output_layout);
      function.local_output_shapes.emplace_back(
          output_layout.LocalShapeFromGlobalShape(
              global_output_shapes[global_index]));
    }

    execution_functions.function_list.emplace_back(std::move(function));
  }

  if (execution_functions.function_list.empty()) {
    return errors::InvalidArgument(
        "MLIR transformed graph does not have any functions to execute for "
        "mesh.");
  }

  return execution_functions;
}

// For functions with control outputs, add identity nodes between
// StatefulPartitionedCall and _Retvals, in order to preserve control output
// dependencies after StatefulPartitionedCall is inlined at runtime.
// Consider calling this in PrepareGraphForMlir, once the identity nodes won't
// be dropped during MLIR lowering.
// TODO(b/171265131): fix the underlying issue to avoid inserting identity
// nodes.
Status MaybeInsertIdentityNodes(const FunctionDef* function_def, Graph* graph) {
  if (function_def == nullptr || function_def->control_ret().empty()) {
    return OkStatus();
  }
  tensorflow::Status status;
  for (Node* n : graph->nodes()) {
    if (!n->IsRetval()) {
      continue;
    }
    const Edge* edge;
    TF_RETURN_IF_ERROR(n->input_edge(0, &edge));
    int ret_index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &ret_index));
    tensorflow::NodeDefBuilder identity_builder(
        absl::StrCat("op_output_identity_", ret_index), "Identity");
    tensorflow::NodeDef ret_identity_node_def;
    tensorflow::DataType output_type = n->input_type(0);
    TF_RETURN_IF_ERROR(
        identity_builder.Attr("T", output_type)
            .Input(edge->src()->name(), edge->src_output(), output_type)
            .Finalize(&ret_identity_node_def, /*consume=*/true));
    Node* ret_identity_node = graph->AddNode(ret_identity_node_def, &status);
    TF_RETURN_IF_ERROR(status);
    // Delete the edge between StatefulPartitionedCall and _Retval.
    graph->RemoveEdge(edge);
    // Add an edge between StatefulPartitionedCall and Identity.
    graph->AddEdge(edge->src(), edge->src_output(), ret_identity_node, 0);
    graph->AddControlEdge(edge->src(), ret_identity_node);
    // Add an edge between Identity and _Retval.
    graph->AddEdge(ret_identity_node, 0, n, 0);
  }
  return OkStatus();
}

void AddDTensorFunctionAttr(FunctionDef& function_def) {
  // Do not xla compile function returned by DTensor MLIR graph transformation
  // as it already returns compiled graph.
  AttrValue xla_must_compile_val;
  xla_must_compile_val.set_b(false);
  function_def.mutable_attr()->insert(
      {"_XlaMustCompile", xla_must_compile_val});

  // Explicitly place function outputs on the default function device to avoid
  // redundant host <-> device copies (Placer may place outputs on the host
  // CPU).
  AttrValue outputs_on_op_device;
  outputs_on_op_device.set_b(true);
  function_def.mutable_attr()->insert(
      {"_OutputsOnOpDevice", outputs_on_op_device});
}

StatusOr<std::vector<parallel_device::ParallelTensor*>> PrepareEmbeddingInputs(
    const std::vector<TensorWithLayout*>& inputs) {
  absl::flat_hash_map<int64_t, std::vector<int64_t>> table_vars_input_index;
  for (int64_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->tensor_type() != kResource) continue;

    const absl::optional<EmbeddingResourceAttrs>& resource_attrs =
        inputs[i]->attrs();
    if (resource_attrs.has_value()) {
      table_vars_input_index[resource_attrs->table_id].push_back(i);
    }
  }

  // Check if there is no embedding resource input found.
  if (table_vars_input_index.empty()) {
    return errors::Internal("There are no TPU embedding resource input found.");
  }
  std::vector<parallel_device::ParallelTensor*> parallel_inputs;
  // Assure parallel inputs has numeric order as table ids.
  for (const auto& [table_id, table_vars_indices] : table_vars_input_index) {
    for (const int64_t input_index : table_vars_indices) {
      parallel_inputs.push_back(inputs[input_index]->tensor());
    }
  }
  return parallel_inputs;
}

StatusOr<std::map<int64_t, std::vector<Node*>>> GetTPUEmbeddingInputNodes(
    TF_Status* s, const Graph& graph,
    const std::vector<TensorWithLayout*>& inputs) {
  // After the graph is lowered, the sparse tensors live at the end of the
  // argument list, so process the dtensor dense inputs only so that
  // we index correctly.
  std::vector<TensorWithLayout*> non_sparse_inputs;
  non_sparse_inputs.reserve(inputs.size());
  for (TensorWithLayout* input : inputs) {
    if (input->tensor_type() != TensorType::kSparse) {
      non_sparse_inputs.push_back(input);
    }
  }
  std::map<int64_t, std::vector<Node*>> table_id_node_map;
  for (Node* node : graph.nodes()) {
    if (!node->IsArg()) continue;

    const int64_t& arg_id = node->attrs().Find("index")->i();
    const AttrValue* embedding_attr =
        node->attrs().Find("_tpu_embedding_table_id");

    if (embedding_attr == nullptr) continue;
    EmbeddingResourceAttrs embedding_input_attrs;

    // Add embedding table id.
    const int64_t table_id = embedding_attr->i();
    embedding_input_attrs.table_id = table_id;

    // Add embedding slot id if there is one.
    const AttrValue* embedding_slot_attr =
        node->attrs().Find("_tpu_embedding_slot_id");
    if (embedding_slot_attr != nullptr) {
      const int64_t slot_id = embedding_slot_attr->i();
      embedding_input_attrs.slot_id = slot_id;
    }

    table_id_node_map[table_id].push_back(node);

    // Arg input offset due to device id.
    if (non_sparse_inputs[arg_id - 1]->attrs().has_value()) continue;
    non_sparse_inputs[arg_id - 1]->UpdateAttrs(embedding_input_attrs, s);
    if (!s->status.ok()) {
      return errors::Internal(
          "Failed to set embedding resource attrs. \n Got error: ",
          s->status.error_message());
    }
  }
  return table_id_node_map;
}

StatusOr<std::string> ValidateResourceMeshConsistency(
    const std::vector<TensorWithLayout*>& inputs) {
  std::string mesh_str;
  for (TensorWithLayout* inp : inputs) {
    if ((inp->tensor_type() != kResource) || !inp->attrs().has_value())
      continue;
    const std::string& input_mesh_str = inp->layout().mesh().ToString();
    if (mesh_str.empty()) {
      mesh_str = input_mesh_str;
    } else if (mesh_str != input_mesh_str) {
      return errors::Internal(absl::StrCat(
          "All inputs of embedding resource must be on same mesh. but get : ",
          mesh_str, " != ", input_mesh_str));
    }
  }
  VLOG(1) << "Resource input mesh is : " << mesh_str;
  return mesh_str;
}

Status InsertFunctionForTPUEmbeddingCheckpoint(
    TF_Status* status, Graph* graph,
    const std::vector<TensorWithLayout*>& inputs,
    const std::string& checkpoint_fn_name) {
  if (checkpoint_fn_name != kLoadEmbeddingFn &&
      checkpoint_fn_name != kRetrieveEmbeddingFn) {
    return errors::InvalidArgument(absl::StrCat(
        "Found wrong function name: ", checkpoint_fn_name,
        " \n expects : ", kLoadEmbeddingFn, " or ", kRetrieveEmbeddingFn));
  }

  StatusOr<std::map<int64_t, std::vector<Node*>>> table_id_node_map =
      GetTPUEmbeddingInputNodes(status, *graph, inputs);
  if (!table_id_node_map.ok()) {
    return errors::Internal(table_id_node_map.status().error_message());
  }

  StatusOr<std::string> mesh_str = ValidateResourceMeshConsistency(inputs);

  const int64_t& num_tables = table_id_node_map->size();
  NodeDef func_node_def;
  std::vector<NodeDefBuilder::NodeOut> func_inputs;
  std::vector<DataType> input_types, output_types;

  func_inputs.reserve(num_tables);
  input_types.reserve(num_tables);

  for (int i = 0; i < num_tables; ++i) {
    auto node_vec_ptr = table_id_node_map->find(i);
    if (node_vec_ptr == table_id_node_map->end()) {
      return errors::Internal(
          absl::StrCat("Embedding table id ", i, " is not found."));
    }
    for (const Node* n : node_vec_ptr->second) {
      const std::string& node_name = n->name();
      func_inputs.push_back({node_name, i, DT_RESOURCE});
      input_types.push_back(DT_RESOURCE);
    }
  }

  AttrValue mesh_attr;
  *mesh_attr.mutable_s() = *mesh_str;
  NameAttrList func_attr;
  func_attr.set_name(checkpoint_fn_name);
  TF_RETURN_IF_ERROR(
      NodeDefBuilder(checkpoint_fn_name, "StatefulPartitionedCall")
          .Attr("Tin", input_types)
          .Attr("Tout", output_types)
          .Attr("f", func_attr)
          .Attr(kMeshAttr, mesh_attr)
          .Attr("config", mesh_attr)
          .Input(func_inputs)
          .Finalize(&func_node_def, true));

  TF_ASSIGN_OR_RETURN(Node * func_node, graph->AddNode(func_node_def));
  for (int i = 0; i < num_tables; ++i) {
    const std::vector<Node*>& node_vec = table_id_node_map->find(i)->second;
    for (int j = 0; j < node_vec.size(); ++j) {
      graph->AddEdge(node_vec[j], 0, func_node, j + i);
    }
  }

  return OkStatus();
}

}  // namespace dtensor
}  // namespace tensorflow

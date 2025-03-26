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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/parallel_device/parallel_device_lib.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/managed_stack_trace.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/dtensor_operation.h"
#include "tensorflow/dtensor/cc/dtensor_utils.h"
#include "tensorflow/dtensor/cc/small_constant_optimization.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/cc/tensor_with_layout.h"
#include "tsl/platform/fingerprint.h"

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

std::vector<TensorHandlePtr> BroadcastTensorHandleToParallelTensor(
    TFE_Context* context, TFE_TensorHandle* tensor, const Mesh& target_mesh,
    TF_Status* status) {
  // Broadcast tensor value to local devices.
  absl::Span<const std::string> local_devices = target_mesh.local_devices();
  const int num_local_devices = local_devices.size();

  std::vector<TensorHandlePtr> components;
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
      return {};
    }
  }

  return components;
}

StatusOr<ResourceHandle> TensorHandleToResourceHandle(
    TFE_TensorHandle* tensor) {
  // Resolve the Tensor as resource handle such that we can get the shape and
  // dtype of the tensor it points to.
  TF_StatusPtr status(TF_NewStatus(), internal::TF_StatusDeleter());
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> tf_tensor(
      TFE_TensorHandleResolve(tensor, status.get()), TF_DeleteTensor);
  if (TF_GetCode(status.get()) != TF_OK) {
    return StatusFromTF_Status(status.get());
  }
  Tensor t;
  TF_RETURN_IF_ERROR(TF_TensorToTensor(tf_tensor.get(), &t));
  if (t.dtype() != DataType::DT_RESOURCE) {
    return absl::InvalidArgumentError("Expecting a DT_RESOURCE Tensor");
  }
  return t.flat<ResourceHandle>()(0);
}

// Broadcast a single non-parallel resource tensor onto `mesh` with a fully
// replicated sharding spec. Does not take ownership of `tensor`.
std::unique_ptr<TensorWithLayoutTf> BroadcastResourceTensor(
    TFE_Context* context, TFE_TensorHandle* tensor, const Mesh& target_mesh,
    TF_Status* status) {
  // Only broadcast resource tensors that point to scalars since they are
  // always replicated. We also still want to catch honest user errors so
  // error out on non-scalars.

  // Replicate this resource handle to all devices without changing the
  // associated device of the resource itself.
  auto r = TensorHandleToResourceHandle(tensor);
  if (!r.ok()) {
    Set_TF_Status_from_Status(status, r.status());
    return nullptr;
  }

  // Only broadcast resource tensors onto a CPU mesh. Copying
  // resource tensors to non CPU device is not supported.
  if (!target_mesh.is_cpu_mesh()) {
    std::string error_message =
        "Using a non-DTensor variable with DTensor is only supported for "
        "copying to a CPU mesh. If you are using a scope "
        "based API, create "
        "variables inside the DTensor scope.\n";

    // Get the stack_trace and Summaries from the resource tensor.
    absl::StrAppend(
        &error_message, "Offending variable summary: ", r->SummarizeValue(),
        "\nStack trace: ", DefinitionLocationMsg(r->definition_stack_trace()));
    TF_SetStatus(status, TF_INVALID_ARGUMENT, error_message.c_str());
    return nullptr;
  }

  LOG(INFO) << "Broadcasting resource tensor to a dtensor resource tensor.";
  if (target_mesh.is_remote()) {
    TF_DataType dtype = TFE_TensorHandleDataType(tensor);
    StatusOr<std::vector<int64_t>> shape = GetTensorShapeAsVector(tensor);
    if (!shape.ok()) {
      Set_TF_Status_from_Status(status, shape.status());
      return nullptr;
    }
    auto layout = Layout::ReplicatedOnMesh(target_mesh, shape->size());
    auto ret = CreateDummyTensorWithLayout(*shape, dtype, layout);
    return ret;
  }

  std::vector<TensorHandlePtr> tensors = BroadcastTensorHandleToParallelTensor(
      context, tensor, target_mesh, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  int rank = r->dtypes_and_shapes().empty()
                 ? 0
                 : r->dtypes_and_shapes().begin()->shape.dims();

  StatusOr<std::unique_ptr<TensorWithLayoutTf>> result = CreateTensorWithLayout(
      std::move(tensors), Layout::ReplicatedOnMesh(target_mesh, rank));
  if (!result.ok()) {
    TF_SetStatus(
        status, TF_INTERNAL,
        absl::StrCat("Error creating a TensorWithLayout from a resource tensor "
                     "during broadcasting with original error message:",
                     result.status().message())
            .c_str());
    return nullptr;
  }

  if (!r->dtypes_and_shapes().empty()) {
    PartialTensorShape partial_shape = r->dtypes_and_shapes().begin()->shape;
    // Set the shape/type of the tensor that the resource points to
    // so that the graph has correct shape/type information that we can use.
    const absl::Status s =
        llvm::cast<ResourceHandleWithLayout>((*result).get())
            ->UpdateShapeAndDType(partial_shape.AsProto(),
                                  r->dtypes_and_shapes().begin()->dtype);
    if (!s.ok()) {
      TF_SetStatus(
          status, TF_INTERNAL,
          absl::StrCat(
              "Error updating shape and dtype of the resource tensor: ",
              s.message())
              .c_str());
      return nullptr;
    }
  }

  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Error updating shape and dtype for resource tensor during "
                 "broadcasting.");
    return nullptr;
  }
  return std::move(*result);
}

bool LayoutsAreCompatible(std::optional<Layout> first_layout,
                          std::optional<Layout> second_layout) {
  if (!first_layout.has_value() && !second_layout.has_value()) {
    return true;
  }
  if (!first_layout.has_value() || !second_layout.has_value()) {
    return false;
  }
  return first_layout.value() == second_layout.value();
}

// Parse a pair of attribute of (indices, layouts) into a map.
absl::Status ParseAttrMap(const Node& node, absl::string_view indices_attr,
                          absl::string_view layout_attr,
                          std::map<int, Layout>* indices_layout_map) {
  std::vector<std::string> layouts;
  if (!TryGetNodeAttr(node.attrs(), layout_attr, &layouts)) {
    return absl::OkStatus();
  }
  const TensorProto* indices;
  if (!TryGetNodeAttr(node.attrs(), indices_attr, &indices)) {
    return absl::InternalError(
        "Arg indices must be set when setting inferred resource layouts.");
  }
  if (indices->int_val_size() != layouts.size()) {
    return absl::InternalError(
        "Arg indices for inferred resource argument must match the "
        "size of inferred resource layout.");
  }
  for (int i = 0; i < indices->int_val_size(); ++i) {
    const auto arg_index = indices->int_val(i);
    const auto& arg_layout = layouts[i];
    indices_layout_map->emplace(
        arg_index, tensorflow::dtensor::Layout::FromString(arg_layout).value());
  }
  return absl::OkStatus();
}

absl::Status ParseResourceArgumentLayouts(
    const Node& node, std::map<int, Layout>* inferred_resource_input_layouts) {
  return ParseAttrMap(node, kNewResourceLayoutIndices, kNewResourceArgLayouts,
                      inferred_resource_input_layouts);
}

absl::Status ParseShapeInputLayouts(
    const Node& node, std::map<int, Layout>* shape_output_metadata) {
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
    return absl::InvalidArgumentError(absl::StrCat(
        op->op_def().name(), " doesn't contain attribute : ", kLayoutAttr));
  }

  // We assume that there is one layout for each output.
  if (serialized_layouts->list().s_size() != op->num_outputs()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Number of outputs to ", op->op_def().name(),
                     " does not match number of layouts attached"));
  }

  return Layout::FromString(serialized_layouts->list().s(output_index));
}

}  // namespace

char TensorWithLayoutTf::ID = 0;

StatusOr<std::vector<int64_t>> GetTensorShapeAsVector(
    const tensorflow::PartialTensorShape& shape) {
  const int dims = shape.dims();
  if (dims < 0) {
    return absl::InvalidArgumentError("Unavailable tensor shape!");
  }
  std::vector<int64_t> result;
  result.reserve(dims);
  for (const TensorShapeDim& dim : shape) {
    result.emplace_back(dim.size);
  }
  return result;
}

StatusOr<std::vector<int64_t>> GetTensorShapeAsVector(
    TFE_TensorHandle* tensor) {
  tensorflow::PartialTensorShape shape;
  const absl::Status status = tensorflow::unwrap(tensor)->Shape(&shape);
  if (status.ok()) {
    return GetTensorShapeAsVector(shape);
  } else {
    return status;
  }
}

StatusOr<std::vector<std::vector<int64_t>>> GetAllTensorShapes(
    const std::vector<TensorHandlePtr>& tensors) {
  std::vector<std::vector<int64_t>> local_shapes;
  local_shapes.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); i++) {
    TF_ASSIGN_OR_RETURN(std::vector<int64_t> local_shape,
                        GetTensorShapeAsVector(tensors[i].get()));
    local_shapes.push_back(std::move(local_shape));
  }
  return local_shapes;
}

tensorflow::Fprint128 TensorWithLayoutTf::CacheKey() const {
  tensorflow::Fprint128 f = tensorflow::Fingerprint128(layout_.ToString());
  // Use exact shape to compute the key.
  for (const int64_t dim : local_shape_) {
    f = FingerprintCat128(f, dim);
  }
  if (const_value_node_->const_value().has_value()) {
    std::string serialized;
    SerializeToStringDeterministic(const_value_node_->const_value().value(),
                                   &serialized);
    f = FingerprintCat128(f, tensorflow::Fingerprint128(serialized));
  }
  return f;
}

std::unique_ptr<TensorWithLayoutTf> TensorWithLayoutTf::Broadcast(
    TFE_Context* context, TFE_TensorHandle* tensor, const Mesh& target_mesh,
    TF_Status* status) {
  // Handle resource tensor broadcasting to the mesh.
  if (TFE_TensorHandleDataType(tensor) == TF_RESOURCE) {
    return BroadcastResourceTensor(context, tensor, target_mesh, status);
  }

  if (target_mesh.is_remote()) {
    TF_DataType dtype = TFE_TensorHandleDataType(tensor);
    StatusOr<std::vector<int64_t>> shape = GetTensorShapeAsVector(tensor);
    if (!shape.ok()) {
      Set_TF_Status_from_Status(status, shape.status());
      return nullptr;
    }
    auto layout = Layout::ReplicatedOnMesh(target_mesh, shape->size());
    auto ret = CreateDummyTensorWithLayout(*shape, dtype, layout);
    std::optional<NodeDef> const_value =
        ExtractSmallTensorValue(context, tensor, layout, status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    if (const_value) {
      ret->const_value_node()->set_const_value(const_value.value());
    }
    return ret;
  }

  std::vector<TensorHandlePtr> tensors = BroadcastTensorHandleToParallelTensor(
      context, tensor, target_mesh, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  StatusOr<std::vector<int64_t>> shape =
      GetTensorShapeAsVector(tensors[0].get());
  if (!shape.ok()) {
    Set_TF_Status_from_Status(status, shape.status());
    return nullptr;
  }
  const size_t num_dims = shape->size();
  const Layout layout = Layout::ReplicatedOnMesh(target_mesh, num_dims);

  std::optional<NodeDef> const_value =
      ExtractSmallTensorValue(context, tensor, layout, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  // For dynamic shape, get local shapes for all local tensors.
  std::vector<std::vector<int64_t>> local_shapes;
  if (IsDynamicShape(*shape)) {
    StatusOr<std::vector<std::vector<int64_t>>> local_shapes_or =
        GetAllTensorShapes(tensors);
    if (!local_shapes_or.ok()) {
      Set_TF_Status_from_Status(status, local_shapes_or.status());
      return nullptr;
    }
    local_shapes = std::move(local_shapes_or.value());
  }

  std::unique_ptr<TensorWithLayoutTf> result(new TensorWithLayoutTf(
      std::move(tensors), std::move(layout), *shape, local_shapes,
      /*dtype=*/std::nullopt, std::move(const_value)));
  return result;
}

StatusOr<std::unique_ptr<TensorWithLayoutTf>> TensorWithLayoutTf::Wrap(
    std::vector<TensorHandlePtr>&& tensors, const Layout& layout,
    std::optional<std::vector<int64_t>>&& shape) {
  if (!shape.has_value()) {
    TF_ASSIGN_OR_RETURN(shape, GetTensorShapeAsVector(tensors[0].get()));
  }

  // For dynamic shape, get local shapes for all local tensors.
  std::vector<std::vector<int64_t>> local_shapes;
  if (IsDynamicShape(*shape)) {
    TF_ASSIGN_OR_RETURN(local_shapes, GetAllTensorShapes(tensors));
  }
  return absl::WrapUnique(
      new TensorWithLayoutTf(std::move(tensors), layout, *shape, local_shapes));
}

std::unique_ptr<TensorWithLayoutTf> TensorWithLayoutTf::Wrap(
    TensorHandlePtr single_tensor, const Layout& layout, TF_Status* status) {
  if (!layout.IsSingleDevice()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Input layout is not for single device.");
    return nullptr;
  }

  StatusOr<std::vector<int64_t>> shape =
      GetTensorShapeAsVector(single_tensor.get());
  if (!shape.ok()) {
    Set_TF_Status_from_Status(status, shape.status());
    return nullptr;
  }

  return absl::WrapUnique(
      new TensorWithLayoutTf(std::move(single_tensor), layout, *shape));
}

std::unique_ptr<TensorWithLayoutTf> TensorWithLayoutTf::Dummy(
    const std::vector<int64_t>& local_shape, const TF_DataType dtype,
    const Layout& layout) {
  return absl::WrapUnique(new TensorWithLayoutTf(
      std::vector<TensorHandlePtr>(), layout, local_shape, {}, dtype));
}

namespace {
std::vector<std::string> SummarizeDeviceNames(
    absl::Span<const std::string> underlying_devices_) {
  std::vector<DeviceNameUtils::ParsedName> parsed_components(
      underlying_devices_.size());
  for (int component_index = 0; component_index < underlying_devices_.size();
       ++component_index) {
    if (!DeviceNameUtils::ParseFullName(underlying_devices_[component_index],
                                        &parsed_components[component_index]) ||
        !DeviceNameUtils::IsSameAddressSpace(
            underlying_devices_[component_index], underlying_devices_[0])) {
      // Device names are from different address spaces, or we can't figure out
      // whether they are, so we'll fully-qualify everything.
      return std::vector<std::string>(underlying_devices_.begin(),
                                      underlying_devices_.end());
    }
  }
  std::vector<std::string> local_names;
  local_names.reserve(underlying_devices_.size());
  for (const DeviceNameUtils::ParsedName& parsed_component :
       parsed_components) {
    local_names.push_back(
        absl::StrCat(parsed_component.type, ":", parsed_component.id));
  }
  return local_names;
}

StatusOr<std::string> SummarizeValues(
    absl::Span<const std::string> underlying_devices_,
    const std::vector<TensorHandlePtr>& tensors_) {
  std::string summary = "{";
  std::vector<std::string> summarized_devices =
      SummarizeDeviceNames(underlying_devices_);
  for (int component_index = 0; component_index < tensors_.size();
       ++component_index) {
    // TODO(allenl): Add a C API for summarizing tensors. Currently custom
    // devices limiting themselves to a C API (for ABI compatibility) would need
    // to implement summarization for component tensors themselves.
    ImmediateExecutionTensorHandle* component =
        tensorflow::unwrap(tensors_[component_index].get());
    std::string component_summary;
    TF_RETURN_IF_ERROR(component->SummarizeValue(component_summary));
    absl::StrAppend(&summary, component_index == 0 ? "" : ", ", "\"",
                    summarized_devices[component_index],
                    "\": ", component_summary);
  }
  summary += "}";
  return summary;
}
}  // namespace

std::string TensorWithLayoutTf::SummarizeValue() const {
  std::string value_summary;
  absl::Status status;
  if (layout_.IsSingleDevice() || layout_.IsFullyReplicated()) {
    status =
        tensorflow::unwrap(tensors_[0].get())->SummarizeValue(value_summary);
  } else {
    // Note that this just prints the local values for sharded tensors. We could
    // instead run a collective here to relayout to replicated.
    const StatusOr<std::string> summary_status =
        SummarizeValues(layout_.mesh().local_devices(), tensors_);
    if (summary_status.ok()) {
      value_summary = summary_status.value();
    } else {
      status = summary_status.status();
    }
  }
  if (!status.ok()) {
    value_summary = "<error computing value>";
  }
  return absl::StrCat(value_summary, ", layout=\"", layout().ToString(), "\"");
}

std::string TensorWithLayoutTf::DebugString() const {
  TF_DataType tf_dtype = dtype();
  auto dtype = static_cast<DataType>(tf_dtype);

  const auto& shape_vector = global_shape();
  return absl::StrCat("DTensor(", SummarizeValue(),
                      ", shape=", ShapeToDebugString(shape_vector),
                      ", type=", DataTypeString(dtype), ")");
}

char ResourceHandleWithLayout::ID = 0;

StatusOr<std::unique_ptr<ResourceHandleWithLayout>>
ResourceHandleWithLayout::Wrap(std::vector<TensorHandlePtr>&& tensors,
                               const Layout& layout,
                               std::optional<std::vector<int64_t>>&& shape) {
  if (!shape.has_value()) {
    TF_ASSIGN_OR_RETURN(shape, GetTensorShapeAsVector(tensors[0].get()));
  }
  return absl::WrapUnique(
      new ResourceHandleWithLayout(std::move(tensors), layout, *shape));
}

std::unique_ptr<ResourceHandleWithLayout> ResourceHandleWithLayout::Dummy(
    const std::vector<int64_t>& local_shape, const Layout& layout) {
  return absl::WrapUnique(new ResourceHandleWithLayout(
      std::vector<TensorHandlePtr>(), layout, local_shape));
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
  if (dereferenced_element_layouts().has_value()) {
    std::vector<std::string> layout_strs;
    std::transform(dereferenced_element_layouts()->begin(),
                   dereferenced_element_layouts()->end(),
                   std::back_inserter(layout_strs),
                   [](const Layout& layout) { return layout.ToString(); });
    builder.Attr("_element_layouts", layout_strs);
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

absl::Status ResourceHandleWithLayout::UpdateLayout(const Layout& new_layout) {
  // Only set the value for deferenced layout if the incoming layout is not
  // empty. This is still hacky as we use empty layout as placeholder for
  // eagerly placed VarHandleOp.
  if (!dereferenced_layout_.has_value() && new_layout.IsEmpty()) {
    return absl::InvalidArgumentError("New layout is empty.");
  }
  if (dereferenced_layout_.has_value() &&
      !LayoutsAreCompatible(dereferenced_layout_, new_layout)) {
    // TODO(xiejw, allenl): Consider allowing variables to switch layouts.
    return absl::InvalidArgumentError(
        "Attempted to overwrite an existing Layout.");
  }
  dereferenced_layout_.emplace(new_layout);
  return absl::OkStatus();
}

char SparseTensorWithLayout::ID = 0;

StatusOr<std::unique_ptr<SparseTensorWithLayout>> SparseTensorWithLayout::Wrap(
    std::unique_ptr<parallel_device::ParallelTensor> indices_tensor,
    std::unique_ptr<parallel_device::ParallelTensor> values_tensor,
    std::unique_ptr<parallel_device::ParallelTensor> shapes_tensor,
    const Layout& layout, const std::vector<int64_t>& local_shape) {
  return absl::WrapUnique(new SparseTensorWithLayout(
      std::move(indices_tensor), std::move(values_tensor),
      std::move(shapes_tensor), layout, local_shape));
}

std::string SparseTensorWithLayout::SummarizeValue() const {
  std::string indices_summary;
  std::string values_summary;
  std::string dense_shapes_summary;

  absl::Status indices_status;
  absl::Status values_status;
  absl::Status dense_shapes_status;

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
  int num_sparse_tensors = num_tensors() / kSparseTensorNum;
  if (index < num_sparse_tensors) {
    return indices_->tensor(index);
  } else if (index < 2 * num_sparse_tensors) {
    return values_->tensor(index % num_sparse_tensors);
  } else {
    return dense_shapes_->tensor(index % num_sparse_tensors);
  }
}

std::unique_ptr<TensorWithLayoutTf> CreateDummyTensorWithLayout(
    const std::vector<int64_t>& local_shape, TF_DataType dtype,
    const Layout& layout) {
  switch (dtype) {
    case TF_RESOURCE:
      return ResourceHandleWithLayout::Dummy(local_shape, layout);
    default:
      return TensorWithLayoutTf::Dummy(local_shape, dtype, layout);
  }
}

StatusOr<std::unique_ptr<TensorWithLayoutTf>> CreateTensorWithLayout(
    std::vector<TensorHandlePtr>&& tensors, const Layout& layout,
    std::optional<std::vector<int64_t>>&& shape) {
  switch (TFE_TensorHandleDataType(tensors[0].get())) {
    case TF_RESOURCE:
      return ResourceHandleWithLayout::Wrap(std::move(tensors), layout,
                                            std::move(shape));
    default:
      return TensorWithLayoutTf::Wrap(std::move(tensors), layout,
                                      std::move(shape));
  }
}

template <>
StatusOr<bool> ExecutableManager<ExecutionFunctions>::ShouldFoldInput(
    const DTensorOperation& doperation,
    const std::vector<TensorWithLayout*>& inputs, const int input_index) const {
  return absl::UnavailableError(
      "ExecutionFunctions manager can not check if the input is foldable, as "
      "the information is maintained by other types of managers (e.g. ModuleOp "
      "manager)");
}

absl::Status InferOutputLayouts(const DTensorOperation& doperation,
                                const NameAttrList& attributes,
                                const std::optional<Layout>& default_layout,
                                tensorflow::Graph* graph,
                                std::vector<const Layout*>* output_layouts) {
  absl::Status status;
  tensorflow::NodeDef op_node_def;
  op_node_def.set_op(doperation.name);
  op_node_def.set_name("eager_operation");

  op_node_def.mutable_attr()->insert(attributes.attr().begin(),
                                     attributes.attr().end());

  tensorflow::Node* op_node = graph->AddNode(op_node_def, &status);
  TF_RETURN_IF_ERROR(status);

  output_layouts->clear();
  output_layouts->reserve(op_node->num_outputs());
  for (int output_index = 0; output_index < op_node->num_outputs();
       ++output_index) {
    const Layout* layout = nullptr;
    if (default_layout.has_value() && output_index == 0) {
      // Record the user's requested output layout. The scope currently only
      // covers the first output of an op.
      layout = &default_layout.value();
    }
    output_layouts->push_back(layout);
  }
  graph->RemoveNode(op_node);
  return absl::OkStatus();
}

absl::Status PrepareGraphForMlir(
    const ExecutableManager<mlir::OwningOpRef<mlir::ModuleOp>>& module_manager,
    const std::vector<TensorWithLayout*>& inputs,
    const DTensorOperation& doperation,
    const tensorflow::FunctionLibraryDefinition& flib_def,
    const NameAttrList& attributes,
    const std::vector<const Layout*>& output_layouts, tensorflow::Graph* graph,
    std::vector<PartialTensorShape>* global_output_shapes

) {
  // We run shape inference on the graph to find output shapes, which may
  // determine default layouts.
  ShapeRefiner shape_refiner(TF_GRAPH_DEF_VERSION, &flib_def);
  shape_refiner.set_function_library_for_shape_inference(&flib_def);
  absl::Status status;
  {
    // We include an _Arg node for the device ID, but this isn't used by the
    // initial function. It will be provided a value, though, so it's
    // available for use in rewrites.
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
    // The shape passed into MLIR transformation represents the global shape
    // of the tensor. Ideally, the local shape on each parallel device should
    // not be consulted at all and we should use the shape on our input tensor
    // directly.
    const auto& shape = input->global_shape();
    std::vector<tensorflow::int64> cast_shape(shape.begin(), shape.end());
    tensorflow::PartialTensorShape partial_shape;
    // For resource tensors, `shape` attribute should not be specified as
    // shape of resource tensors is specified by resource shape subtype -- not
    // the shape attribute.
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
            .Attr(kMeshAttr, input->mesh().ToString())
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

    // Small constants are converted into constant graph nodes, instead of
    // being passed in as input arguments. This provides more information to
    // the SPMD and layout propagation passes.
    TF_ASSIGN_OR_RETURN(bool should_fold_input,
                        module_manager.ShouldFoldInput(doperation, inputs, i));
    if (!should_fold_input) {
      graph_op_inputs.push_back(FunctionArgument{
          arg_node, NodeDefBuilder::NodeOut{arg_node->name(), i, dtype}});
      graph->AddControlEdge(graph->source_node(), arg_node);
    } else {
      // TODO(xiejw): Refactor the TensorWithLayout representation to avoid
      // special code here.
      NodeDef const_node = input->const_value_node()->const_value().value();
      const_node.set_name(absl::StrCat("input_", i, "_const_value"));
      Node* const_value_n = graph->AddNode(const_node, &status);
      const_value_n->AddAttr(kFromArgIndex, i);
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

    auto layout = output_layouts[output_index];
    if (layout != nullptr) {
      ret_node->AddAttr(kDefaultLayoutAttr, layout->ToString());
    }
  }
  return absl::OkStatus();
}

StatusOr<std::vector<int64_t>> GetNumLocalOutputs(Node* node) {
  const AttrValue* num_local_outputs =
      (node->attrs()).Find(kNumLocalOutputsAttr);
  if (num_local_outputs == nullptr) {
    return absl::InvalidArgumentError("missing num_local_outputs attribute");
  } else {
    const AttrValue_ListValue& list = num_local_outputs->list();
    std::vector<int64_t> res;
    res.reserve(list.i_size());
    std::copy(list.i().begin(), list.i().end(), std::back_inserter(res));
    return res;
  }
}

namespace {
absl::Status SetMultiDeviceFunctionOutputs(
    TranslatedFunction& function, Node* node,
    const std::vector<PartialTensorShape>& global_output_shapes) {
  const AttrValue* serialized_layouts = (node->attrs()).Find(kLayoutAttr);
  if (serialized_layouts == nullptr) {
    return absl::InvalidArgumentError("missing layout attribute");
  }
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> num_local_outputs,
                      GetNumLocalOutputs(node));
  const auto& serialized_layout_list = serialized_layouts->list();
  for (int i = 0; i < serialized_layout_list.s_size(); i++) {
    const auto& serialized_layout = serialized_layout_list.s(i);
    TF_ASSIGN_OR_RETURN(const Layout layout,
                        Layout::FromString(serialized_layout));
    function.output_layouts.emplace_back(std::move(layout));
  }
  int num_output_layouts = function.output_layouts.size();
  for (int i = 0; i < num_output_layouts; i++) {
    const Layout* output_layout = &(function.output_layouts[i]);
    if (output_layout->IsEmpty()) {
      const auto search = function.resource_input_layouts.find(i);
      if (search != function.resource_input_layouts.end()) {
        output_layout = &(search->second);
      }
    }
    PartialTensorShape local_shape =
        output_layout->LocalShapeFromGlobalShape(global_output_shapes[i]);
    const int64_t num_devices = num_local_outputs[i];
    for (int j = 0; j < num_devices; j++) {
      function.local_output_shapes.emplace_back(local_shape);
    }
  }
  function.num_local_outputs = std::move(num_local_outputs);
  return absl::OkStatus();
}
}  // namespace

// Returns set of functions to run to execute DTensor computation.
StatusOr<ExecutionFunctions> IdentifyAllFunctionsToExecute(
    const tensorflow::Graph& graph,
    const std::vector<PartialTensorShape>& global_output_shapes) {
  bool multi_device_mode = EnableMultiDeviceMode();
  ExecutionFunctions execution_functions;
  execution_functions.function_list = std::vector<TranslatedFunction>();
  for (Node* node : graph.nodes()) {
    if (node->op_def().name() != "StatefulPartitionedCall") continue;
    // Extract mesh to execute the function.
    std::string serialized_mesh;
    std::optional<Mesh> mesh;
    if (GetNodeAttr(node->attrs(), kMeshAttr, &serialized_mesh).ok()) {
      TF_ASSIGN_OR_RETURN(mesh, Mesh::FromString(serialized_mesh));
    }

    TranslatedFunction function;
    if (mesh.has_value()) {
      function.function_mesh = std::move(mesh.value());
    }
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
        return absl::InvalidArgumentError(
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

    if (multi_device_mode) {
      // need to update the output shapes and layouts
      TF_RETURN_IF_ERROR(
          SetMultiDeviceFunctionOutputs(function, node, global_output_shapes));
    } else {
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
        function.num_local_outputs.emplace_back(1);
      }
    }

    execution_functions.function_list.emplace_back(std::move(function));
  }

  if (execution_functions.function_list.empty()) {
    return absl::InvalidArgumentError(
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
absl::Status MaybeInsertIdentityNodes(const FunctionDef* function_def,
                                      Graph* graph) {
  if (function_def == nullptr || function_def->control_ret().empty()) {
    return absl::OkStatus();
  }
  absl::Status status;
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
  return absl::OkStatus();
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
  // CPU). This option is only applicable outside of multi-device mode. Since
  // the function is explicitly distributed across multiple devices there,
  // setting this option would result in misplaced resources and tensors.
  AttrValue outputs_on_op_device;
  const bool multi_device_mode = dtensor::EnableMultiDeviceMode();
  outputs_on_op_device.set_b(!multi_device_mode);
  function_def.mutable_attr()->insert(
      {"_OutputsOnOpDevice", outputs_on_op_device});
}

tensorflow::Fprint128 ExecutableManagerImpl::CacheKeyForDTensorOperation(
    const DTensorOperation& doperation) const {
  return tensorflow::Fingerprint128(doperation.name);
}

absl::flat_hash_map<int, NodeDef>
ExecutableManagerImpl::GetConstantFoldableTensors(
    const std::vector<TensorWithLayout*>& inputs) {
  absl::flat_hash_map<int, NodeDef> small_tensors;
  for (auto index = 0; index < inputs.size(); ++index) {
    auto* const_value_node = inputs[index]->const_value_node();
    if (const_value_node == nullptr) {
      continue;
    }
    if (const_value_node->const_value().has_value()) {
      small_tensors.insert({index, const_value_node->const_value().value()});
    }
  }
  return small_tensors;
}

}  // namespace dtensor
}  // namespace tensorflow

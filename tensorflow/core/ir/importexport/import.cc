/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/importexport/import.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/convert_attributes.h"
#include "tensorflow/core/ir/importexport/convert_tensor.h"
#include "tensorflow/core/ir/importexport/convert_types.h"
#include "tensorflow/core/ir/importexport/functiondef_import.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/stack_frame.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#define DEBUG_TYPE "graphdef-to-mlir"

using tensorflow::AttrSlice;
using tensorflow::AttrValue;
using tensorflow::AttrValueMap;
using tensorflow::DataType;
using tensorflow::Edge;
using tensorflow::FunctionBody;
using tensorflow::FunctionDef;
using tensorflow::FunctionLibraryDefinition;
using tensorflow::FunctionLibraryRuntime;
using tensorflow::Graph;
using tensorflow::GraphConstructorOptions;
using tensorflow::GraphDebugInfo;
using tensorflow::GraphDef;
using tensorflow::NameAttrList;
using tensorflow::Node;
using tensorflow::NodeDef;
using tensorflow::OpDef;
using tensorflow::OpDef_ArgDef;
using tensorflow::OpRegistry;
using tensorflow::PartialTensorShape;
using tensorflow::ResourceHandleProto_DtypeAndShape;
using tensorflow::StackFrame;
using tensorflow::Status;
using tensorflow::TensorProto;
using tensorflow::VersionDef;
using tensorflow::errors::AppendToMessage;
using tensorflow::errors::Internal;
using tensorflow::errors::InvalidArgument;
using tensorflow::errors::NotFound;
using tensorflow::protobuf::MapPair;
using tensorflow::protobuf::RepeatedPtrField;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeAndType;
using tensorflow::shape_inference::ShapeHandle;

namespace mlir {
namespace tfg {

namespace {

void LoadDialects(MLIRContext* context) {
  // Load dialects involved in the conversion
  context->getOrLoadDialect<TFGraphDialect>();
}

// Construct the MLIR VersionAttr for the provided GraphDef.
static VersionAttr getVersionAttr(MLIRContext* context,
                                  const VersionDef& version) {
  int producer = 0;
  int min_consumer = 0;
  llvm::SmallVector<int32_t> bad_consumers;
  min_consumer = version.min_consumer();
  producer = version.producer();
  for (int32_t bad_consumer : version.bad_consumers())
    bad_consumers.push_back(bad_consumer);
  return VersionAttr::get(context, /*producer=*/producer,
                          /*minConsumer=*/min_consumer,
                          /*badConsumers=*/bad_consumers);
}

// Stateful helper class to import a TensorFlow Graph into an MLIR Graph.
class GraphImporter {
 public:
  explicit GraphImporter(MLIRContext* context, const Graph& graph,
                         const GraphDebugInfo& debug_info,
                         llvm::StringRef function_name_for_debug_info = "")
      : graph_(&graph),
        builder_(context),
        context_(context),
        dialect_(context->getLoadedDialect<TFGraphDialect>()),
        unknown_loc_(UnknownLoc::get(context)),
        debug_info_(debug_info),
        function_name_for_debug_info_(function_name_for_debug_info) {}

  // Converts the prepared graph to a Function and adds it to the module. A set
  // of nodes from the graph are given to converted to the arguments and returns
  // of the function.
  Status Convert(Block* body);

  Operation* GetOperationForNode(int node_id) {
    auto it = node_values_.find(node_id);
    if (it == node_values_.end()) return nullptr;
    return it->second;
  }

 private:
  // Returns the inferred output type at index `idx` of the `node` in the
  // context.
  Status InferOutputTypes(Builder& builder, OperationState& result,
                          const Node& node);
  // Try to infer the output types of a node from one of its attributes. Certain
  // nodes have a required `output_shapes` attribute, e.g. while, if, and
  // iterator nodes. All nodes might have an `_output_shapes` attribute.
  Optional<Status> InferOutputTypesFromShapesAttribute(Builder& builder,
                                                       OperationState& result,
                                                       const Node& node);
  // Try to infer the output types of the nodes using TF's inference context.
  // Returns `None` if the node has inputs, if it is unregistered, or if it does
  // not have a shape inference function.
  Optional<Status> InferOutputTypesWithContext(Builder& builder,
                                               OperationState& result,
                                               const Node& node);

  // Most types with subtypes have only one subtype.
  using ElementSubtypes = llvm::SmallVector<TensorType, 1>;

  // Converts the inferred shape referred to by 'handle' in 'context', with
  // given element type, and returns an MLIR tensor type.
  tensorflow::StatusOr<TensorType> ConvertDataTypeAndShape(
      DataType dtype, const ShapeHandle& handle,
      const std::vector<ShapeAndType>* handle_subtypes,
      InferenceContext* context, Builder builder);

  // Converts the inferred shape referred to by 'handle' in 'context', with
  // given element type, and returns an MLIR tensor type.
  tensorflow::StatusOr<TensorType> ConvertElementTypeAndShape(
      Type element_type, const ShapeHandle& handle, InferenceContext* context,
      Builder builder);

  // Converts the inferred subtypes for an element type to corresponding MLIR
  // types in 'context'.
  tensorflow::StatusOr<ElementSubtypes> ConvertSubtypes(
      const std::vector<ShapeAndType>* handle_subtypes,
      InferenceContext* context, Builder builder);

  // Converts the tensor proto into an MLIR elements attribute.
  tensorflow::StatusOr<ElementsAttr> ConvertTensorProto(
      const TensorProto& value) {
    return mlir::tfg::ConvertTensorProto(value, builder_, dialect_);
  }

  // Converts one NodeDef from the input GraphDef into an Operation and
  // inserts it into the MLIR module using builder_.
  Status ConvertNode(const Node& node);

  Value GetOperand(const Edge& edge);

  // Gets the location information of the given node. It uses the
  // "original_node_name" in the NodeDef to get the corresponding file location
  // (FileLineColLoc) from the input DebugInfo and returns an CallSiteLoc. If
  // there are multiple "original_node_names", a FusedLoc is returned. If the
  // node name couldn't be found in the input DebugInfo, a NameLoc is used as
  // the location.
  Location GetLocation(const Node& node);

  // All nodes and version information about the (copied) imported graph.
  const Graph* graph_;
  // Maps from a Node ID to a MLIR value.
  using NodeValueMap = absl::flat_hash_map<int, Operation*>;

  OpBuilder builder_;
  MLIRContext* context_;
  TFGraphDialect* dialect_;
  UnknownLoc unknown_loc_;
  const GraphDebugInfo& debug_info_;
  llvm::StringRef function_name_for_debug_info_;
  NodeValueMap node_values_;
};

Optional<Status> GraphImporter::InferOutputTypesFromShapesAttribute(
    Builder& builder, OperationState& result, const Node& node) {
  const AttrValue* output_shapes = nullptr;
  if (node.IsWhileNode() || node.IsIfNode() || node.IsCaseNode() ||
      node.type_string() == "IteratorGetNext" ||
      node.type_string() == "IteratorGetNextSync" ||
      node.type_string() == "MultiDeviceIteratorGetNextFromShard") {
    output_shapes = node.attrs().Find("output_shapes");
  } else if (node.type_string() == "InfeedDequeueTuple") {
    output_shapes = node.attrs().Find("shapes");
  } else if ((output_shapes = node.attrs().Find("_output_shapes"))) {
    // Check for a generic `_output_shapes` attribute. Only use it if it matches
    // the number of outputs.
    if (output_shapes->list().shape_size() != node.num_outputs())
      output_shapes = nullptr;
  }
  if (!output_shapes) return {};

  // The output shapes attribute is required. It may also be empty. Handle the
  // latter case gracefully.
  auto& shapes = output_shapes->list().shape();
  if (shapes.empty()) return {};
  if (shapes.size() != node.num_outputs()) {
    return InvalidArgument("Failed to infer output shapes: expected ",
                           node.num_outputs(), " output shapes but got ",
                           output_shapes->list().shape_size());
  }
  for (auto& it : llvm::enumerate(shapes)) {
    DataType dtype = node.properties()->output_types[it.index()];
    TF_ASSIGN_OR_RETURN(Type output_type,
                        ConvertToMlirTensorType(it.value(), dtype, &builder));
    result.addTypes(output_type);
  }
  return Status::OK();
}

Optional<Status> GraphImporter::InferOutputTypesWithContext(
    Builder& builder, OperationState& result, const Node& node) {
  // Below we only try and do some shape inference for "source" ops which have
  // no inputs.
  if (node.num_inputs() > 0) return {};

  const tensorflow::OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(
      graph_->op_registry()->LookUp(node.type_string(), &op_reg_data));
  if (!op_reg_data) {
    DVLOG(3) << "Skipping inference for unregistered op " << node.type_string();
    return {};
  }
  if (!op_reg_data->shape_inference_fn) {
    DVLOG(3) << "Skipping inference for op without shape function "
             << node.type_string();
    return {};
  }
  InferenceContext c(graph_->versions().producer(), node.attrs(),
                     op_reg_data->op_def, std::vector<PartialTensorShape>{}, {},
                     /*input_tensors_as_shapes=*/{}, {});
  TF_RETURN_IF_ERROR(c.Run(op_reg_data->shape_inference_fn));

  for (int idx : llvm::seq(0, node.num_outputs())) {
    DataType dtype = node.properties()->output_types[idx];
    TF_ASSIGN_OR_RETURN(
        Type output_type,
        ConvertDataTypeAndShape(dtype, c.output(idx),
                                c.output_handle_shapes_and_types(idx), &c,
                                builder));
    result.addTypes(output_type);
  }
  return Status::OK();
}

Status GraphImporter::InferOutputTypes(Builder& builder, OperationState& result,
                                       const Node& node) {
  // Exit early if there are no outputs.
  if (node.num_outputs() == 0) return Status::OK();

  // Try to infer an output shape from a shapes attribute.
  if (Optional<Status> status =
          InferOutputTypesFromShapesAttribute(builder, result, node))
    return *status;

  // Handle a special case for `InfeedDequeue`.
  if (node.type_string() == "InfeedDequeue") {
    assert(node.num_outputs() == 1 && "expected 1 result");
    const auto& output_shape = node.attrs().Find("shape")->shape();
    const auto& element_type = node.attrs().Find("dtype")->type();
    TF_ASSIGN_OR_RETURN(
        Type output_type,
        ConvertToMlirTensorType(output_shape, element_type, &builder));
    result.addTypes(output_type);
    return Status::OK();
  }

  // Try to infer output shapes using shape inference.
  if (Optional<Status> status =
          InferOutputTypesWithContext(builder, result, node))
    return *status;

  // If all else fails, fallback to importing tensors as unranked.
  for (int idx : llvm::seq(0, node.num_outputs())) {
    DataType dtype = node.properties()->output_types[idx];
    Type element_type;
    TF_RETURN_IF_ERROR(ConvertDataType(dtype, builder, &element_type));
    result.addTypes(UnrankedTensorType::get(element_type));
  }
  return Status::OK();
}

tensorflow::StatusOr<TensorType> GraphImporter::ConvertDataTypeAndShape(
    DataType dtype, const ShapeHandle& handle,
    const std::vector<ShapeAndType>* handle_subtypes, InferenceContext* context,
    Builder builder) {
  TF_ASSIGN_OR_RETURN(auto subtypes,
                      ConvertSubtypes(handle_subtypes, context, builder));

  Type element_type;
  if (dtype == tensorflow::DT_VARIANT)
    element_type = VariantType::get(subtypes, context_);
  else if (dtype == tensorflow::DT_RESOURCE)
    element_type = ResourceType::get(subtypes, context_);
  else
    TF_RETURN_IF_ERROR(ConvertDataType(dtype, builder, &element_type));

  return ConvertElementTypeAndShape(element_type, handle, context, builder);
}

tensorflow::StatusOr<TensorType> GraphImporter::ConvertElementTypeAndShape(
    Type element_type, const ShapeHandle& handle, InferenceContext* context,
    Builder builder) {
  if (!context->RankKnown(handle)) {
    return UnrankedTensorType::get(element_type);
  }

  // Sentinel for an unknown dimension size. getTensorType interprets any
  // negative value as an unknown dimension.

  absl::InlinedVector<int64_t, 4> dimensions;
  int32_t rank = context->Rank(handle);
  dimensions.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    auto dim_handle = context->Dim(handle, i);
    if (!context->ValueKnown(dim_handle))
      dimensions.push_back(ShapedType::kDynamicSize);
    else
      dimensions.push_back(context->Value(dim_handle));
  }

  return RankedTensorType::get(
      llvm::makeArrayRef(dimensions.begin(), dimensions.end()), element_type);
}

tensorflow::StatusOr<GraphImporter::ElementSubtypes>
GraphImporter::ConvertSubtypes(const std::vector<ShapeAndType>* handle_subtypes,
                               InferenceContext* context, Builder builder) {
  ElementSubtypes subtypes;
  if (!handle_subtypes) return subtypes;

  subtypes.reserve(handle_subtypes->size());
  for (const auto& subtype : *handle_subtypes) {
    Type element_type;
    TF_RETURN_IF_ERROR(ConvertDataType(subtype.dtype, builder, &element_type));
    TF_ASSIGN_OR_RETURN(TensorType type,
                        ConvertElementTypeAndShape(element_type, subtype.shape,
                                                   context, builder));
    subtypes.push_back(type);
  }
  return subtypes;
}

Status GraphImporter::Convert(Block* body) {
  VLOG(4) << "Convert";
  builder_ = OpBuilder::atBlockEnd(body);
  // Create the graph operation in which we will convert the individual nodes.

  for (const Node* node : graph_->nodes()) {
    TF_RETURN_IF_ERROR(ConvertNode(*node));
  }

  return Status::OK();
}

Location GraphImporter::GetLocation(const Node& node) {
  DVLOG(3) << "Getting location for " << node.name() << " " << &node;
  const auto& debug_info = debug_info_.traces();
  // Create a location for node `name` in function `function_name`.
  auto create_location = [&](llvm::StringRef name,
                             llvm::StringRef function_name) -> Location {
    // Use the catenation of function and node names as the lookup key into the
    // debug info. This matches the way that the key is formed on the python
    // side.
    //
    // We also use this as the name for the NameLoc for ops in function, since
    // otherwise our names could collide across functions.
    // For ops in the main graph, we omit the "@function_name" (which, would be
    // just "@" since function_name would be empty) because some code seems to
    // depend on the name being this way for correctness.
    std::string debug_info_key = (name + "@" + function_name).str();
    std::string name_for_name_loc =
        function_name.empty() ? name.str() : (name + "@" + function_name).str();
    auto name_loc_id = StringAttr::get(context_, name_for_name_loc);

    llvm::SmallVector<Location, 4> locations;
    // Prefer stack traces if available, fallback to debug info if not, and then
    // finally to just name.
    if (auto stack_trace = node.GetStackTrace()) {
      DVLOG(3) << "Stack available for " << node.name();
      absl::Span<const StackFrame> frames = stack_trace->ToFrames();
      locations.reserve(frames.size());
      for (const StackFrame& frame : llvm::reverse(frames)) {
        auto file_name = StringAttr::get(context_, frame.file_name);
        // Use col 1 as there is no column info in StackTrace.
        auto file_line_loc =
            FileLineColLoc::get(file_name, frame.line_number, 1);
        locations.push_back(file_line_loc);
      }
    } else {
      DVLOG(3) << "No stack trace for " << node.name();
      const auto location_it = debug_info.find(debug_info_key);
      if (location_it != debug_info.end()) {
        DVLOG(3) << "Available serialized debug info for " << node.name();
        // Convert the stack trace to a chain of CallSiteLocs.
        const auto& trace = location_it->second;
        locations.reserve(trace.file_line_cols_size());
        for (const auto& location : trace.file_line_cols()) {
          const auto& file = debug_info_.files(location.file_index());
          auto file_name = StringAttr::get(context_, file);
          auto file_line_loc =
              FileLineColLoc::get(file_name, location.line(), location.col());
          locations.push_back(file_line_loc);
        }
      }
    }

    // If there are no locations in the stack trace, fall back to just a
    // NameLoc with no child.
    if (locations.empty()) return NameLoc::get(name_loc_id);

    // Use the front FileLineColLoc to generate a NameLoc.
    Location node_name_loc = NameLoc::get(name_loc_id, locations.front());

    // If there are more locations then generate a stack trace, otherwise just
    // return the name loc.
    auto callsite_locs = llvm::makeArrayRef(locations).drop_front();
    return callsite_locs.empty()
               ? node_name_loc
               : CallSiteLoc::get(node_name_loc, callsite_locs);
  };

  if (node.GetStackTrace())
    return create_location(node.name(), function_name_for_debug_info_);

  const auto& node_def = node.def();
  auto original_nodes =
      node_def.experimental_debug_info().original_node_names();
  auto original_funcs =
      node_def.experimental_debug_info().original_func_names();

  if (!original_nodes.empty()) {
    // If the original nodes are defined, then we use them to get a list of
    // call sites, and then fuse them to a single fused location, with the name
    // of the node_def.
    llvm::SmallVector<Location, 4> node_locations;
    node_locations.reserve(original_nodes.size() + 1);

    // store the names in the experimental_debug_info
    for (int i = 0, e = original_nodes.size(); i != e; ++i) {
      auto node_name = original_nodes[i];
      auto func_name = (i < original_funcs.size()) ? original_funcs[i] : "";
      node_locations.push_back(create_location(node_name, func_name));
    }
    return FusedLoc::get(context_, node_locations);
  }
  return unknown_loc_;
}

Value GraphImporter::GetOperand(const Edge& edge) {
  const Node& input_node = *edge.src();
  int resultId = edge.src_output();
  Operation*& inst = node_values_[input_node.id()];
  auto getResult = [&]() {
    if (edge.IsControlEdge()) return inst->getResult(inst->getNumResults() - 1);
    return inst->getResult(resultId);
  };
  if (inst) return getResult();

  // We use placeholders during the import to create "fake" operations to break
  // cycles: we need operands to feed to the users.
  OperationName mlir_placeholder("tfg.__mlir_placeholder", context_);
  OperationState state(UnknownLoc::get(context_), mlir_placeholder);
  auto placeholder_ty = OpaqueTensorType::get(context_);
  auto control_ty = ControlType::get(context_);
  std::string node_name;
  state.addAttribute("name", builder_.getStringAttr(input_node.name()));
  state.types.resize(input_node.num_outputs() + 1, placeholder_ty);
  state.types.back() = control_ty;
  inst = builder_.create(state);
  return getResult();
}

Status GraphImporter::ConvertNode(const Node& node) {
  if (!node.IsOp()) {
    // Don't import the pseudo-nodes _SOURCE or _SINK. These are added by
    // Graph and don't exist in GraphDef.
    // TODO(aminim): I have no idea what these nodes are useful for...
    VLOG(4) << "Ignore " << node.name() << " on import";
    return Status::OK();
  }
  VLOG(4) << "Importing " << node.name();
  OperationState result(GetLocation(node),
                        absl::StrCat("tfg.", node.type_string()));
  // Compute the result types.
  TF_RETURN_IF_ERROR(InferOutputTypes(builder_, result, node));
  result.addTypes(ControlType::get(builder_.getContext()));

  // Input edges can be nondeterministically ordered, sort them here. First the
  // data edges in the expected order and then the control edges using the
  // source node ID as discriminant.
  absl::InlinedVector<const Edge*, 8> in_edges(node.in_edges().size());
  absl::c_copy(node.in_edges(), in_edges.begin());
  absl::c_stable_sort(in_edges, [](const Edge* e1, const Edge* e2) {
    if (e1->IsControlEdge() && !e2->IsControlEdge()) return false;
    if (!e1->IsControlEdge() && e2->IsControlEdge()) return true;
    if (e1->IsControlEdge() && e2->IsControlEdge())
      return e1->src()->id() < e2->src()->id();
    return e1->dst_input() < e2->dst_input();
  });

  // Collect the operands.
  result.operands.reserve(in_edges.size());
  for (const auto* input_edge : in_edges) {
    const Node& input_node = *input_edge->src();
    // We don't import the _SOURCE node, skip this edge.
    if (input_node.IsSource()) {
      continue;
    }
    result.operands.push_back(GetOperand(*input_edge));
  }

  // Handle attributes, reserve `+3` for `device`, `name` and `fulltype`.
  result.attributes.reserve(node.attrs().size() + 3);
  result.addAttribute(dialect_->getDeviceAttrIdentifier(),
                      builder_.getStringAttr(node.requested_device()));
  result.addAttribute(dialect_->getNameAttrIdentifier(),
                      StringAttr::get(context_, node.name()));
  if (node.def().has_experimental_type()) {
    TF_ASSIGN_OR_RETURN(
        tf_type::FullTypeAttr type,
        ConvertAttribute(node.def().experimental_type(), builder_, dialect_));
    result.addAttribute(dialect_->getFullTypeAttrIdentifier(), type);
  }
  for (const auto& namedAttr : node.attrs()) {
    const std::string& name = namedAttr.first;
    if (name.empty()) return InvalidArgument("empty attr name");
    const AttrValue& tf_attr = namedAttr.second;
    TF_ASSIGN_OR_RETURN(Attribute attr,
                        ConvertAttributeValue(tf_attr, builder_, dialect_));
    result.addAttribute(PromoteToTFGAttribute(name), attr);
  }
  Attribute assigned_device =
      result.attributes.get(dialect_->getAssignedDeviceAttrIdentifier());
  if (!assigned_device ||
      assigned_device.cast<StringAttr>().getValue().empty()) {
    result.attributes.erase(dialect_->getAssignedDeviceAttrIdentifier());
    result.addAttribute(dialect_->getAssignedDeviceAttrIdentifier(),
                        builder_.getStringAttr(node.assigned_device_name()));
  }

  // Register the mapping between the TF node and the newly created operation.
  Operation* operation = builder_.create(result);
  Operation*& cached_operation = node_values_[node.id()];
  if (cached_operation) {
    // A placeholder was inserted for this op earlier to break a cycle in the
    // graph, replace and erase it now.
    operation->moveBefore(cached_operation);
    cached_operation->replaceAllUsesWith(operation);
    cached_operation->erase();
  }
  cached_operation = operation;

  return Status::OK();
}

void FindPlaceholders(const AttrValue& value, AttrSlice set,
                      AttrValueMap& founds) {
  switch (value.value_case()) {
    case AttrValue::kList:
      for (const NameAttrList& func : value.list().func())
        for (auto& p : func.attr()) FindPlaceholders(p.second, set, founds);
      return;
    case AttrValue::kFunc:
      for (auto& p : value.func().attr())
        FindPlaceholders(p.second, set, founds);
      return;
    case AttrValue::kPlaceholder:
      if (const AttrValue* v = set.Find(value.placeholder()))
        founds.insert({value.placeholder(), *v});
      return;
    default:
      return;
  }
}

tensorflow::StatusOr<std::string> MangleName(const FunctionDef& fdef,
                                             AttrSlice attrs) {
  const OpDef& signature = fdef.signature();
  if (!attrs.size()) return signature.name();
  // Collect all the attributes that are actually used during instantiation.
  AttrValueMap used_attr;
  for (const auto& a : signature.attr()) {
    const AttrValue* v = attrs.Find(a.name());
    if (!v) {
      return NotFound("Attr ", a.name(), " is not found from ",
                      SummarizeOpDef(signature));
    }
    used_attr.insert({a.name(), *v});
  }
  auto process_argdef = [&](const OpDef::ArgDef& arg_def) {
    if (!arg_def.type_list_attr().empty()) {
      const AttrValue* v = attrs.Find(arg_def.type_list_attr());
      if (v == nullptr)
        return NotFound("type attr not found: ", arg_def.type_list_attr());
    }
    if (!arg_def.number_attr().empty()) {
      const AttrValue* v = attrs.Find(arg_def.number_attr());
      if (v == nullptr)
        return NotFound("type attr not found: ", arg_def.type_attr());
    }
    if (arg_def.type() == tensorflow::DT_INVALID &&
        !arg_def.type_attr().empty()) {
      const AttrValue* v = attrs.Find(arg_def.type_attr());
      if (v == nullptr)
        return NotFound("type attr not found: ", arg_def.type_attr());
    }
    return Status::OK();
  };
  for (const OpDef::ArgDef& arg_def : signature.input_arg()) {
    Status s = process_argdef(arg_def);
    if (!s.ok()) {
      AppendToMessage(&s, " for arg ", arg_def.name());
      return s;
    }
  }
  for (const OpDef::ArgDef& arg_def : signature.output_arg()) {
    Status s = process_argdef(arg_def);
    if (!s.ok()) {
      AppendToMessage(&s, " for output ", arg_def.name());
      return s;
    }
  }
  for (const NodeDef& node_def : fdef.node_def())
    for (const MapPair<std::string, AttrValue>& attr : node_def.attr())
      FindPlaceholders(attr.second, attrs, used_attr);
  if (used_attr.empty()) return signature.name();
  return Canonicalize(signature.name(), AttrSlice(&used_attr),
                      FunctionLibraryRuntime::InstantiateOptions{});
}

tensorflow::StatusOr<GraphFuncOp> ImportFunctionDef(
    ModuleOp module, const GraphDebugInfo& debug_info,
    const FunctionLibraryDefinition& lib_def, const FunctionDef& fdef,
    AttrSlice instantiation_attributes) {
  TF_ASSIGN_OR_RETURN(std::string name,
                      MangleName(fdef, instantiation_attributes));

  // Converts the function definition to a graph.
  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(fdef, instantiation_attributes,
                                             &lib_def, &fbody));

  // Create the func operation in which we will convert the individual nodes.
  OpBuilder builder = OpBuilder::atBlockEnd(module.getBody());
  MLIRContext* context = module->getContext();
  Location unknown_loc = builder.getUnknownLoc();
  GraphFuncOp func_op = builder.create<GraphFuncOp>(unknown_loc);
  TFGraphDialect* tfgDialect = cast<TFGraphDialect>(func_op->getDialect());
  const OpDef& signature = fdef.signature();

  func_op.getBodyRegion().push_back(new Block);
  Block* body = &func_op.getBodyRegion().front();
  // Import the nodes in the graph body.
  GraphImporter importer(context, *fbody->graph, debug_info);
  TF_RETURN_IF_ERROR(importer.Convert(body));

  NamedAttrList attrs;
  // Import the function attributes with a `tf.` prefix to match the current
  // infratructure expectations.
  for (const auto& namedAttr : fdef.attr()) {
    const std::string& name = "tf." + namedAttr.first;
    const AttrValue& tf_attr = namedAttr.second;
    TF_ASSIGN_OR_RETURN(Attribute attr,
                        ConvertAttributeValue(tf_attr, builder, tfgDialect));
    attrs.append(name, attr);
  }
  if (signature.name().empty())
    return InvalidArgument("function without a name");
  attrs.append("sym_name", builder.getStringAttr(name));

  if (!signature.description().empty())
    attrs.append("description", builder.getStringAttr(signature.description()));
  if (signature.is_stateful())
    attrs.append("is_stateful", builder.getUnitAttr());
  std::string gradient = lib_def.FindGradient(signature.name());
  if (!gradient.empty())
    attrs.append("gradient", FlatSymbolRefAttr::get(context, gradient));

  // The resource_arg_unique_id is a list of `pair<int, int>`, we import it
  // as two arrays of integer right now.
  if (fdef.resource_arg_unique_id_size()) {
    llvm::SmallVector<int32_t> resource_arg_unique_ids_keys;
    llvm::SmallVector<int32_t> resource_arg_unique_ids_values;
    for (const auto& unique_id : fdef.resource_arg_unique_id()) {
      resource_arg_unique_ids_keys.push_back(unique_id.first);
      resource_arg_unique_ids_values.push_back(unique_id.second);
    }
    attrs.append("resource_arg_unique_ids_keys",
                 builder.getI32TensorAttr(resource_arg_unique_ids_keys));
    attrs.append("resource_arg_unique_ids_values",
                 builder.getI32TensorAttr(resource_arg_unique_ids_values));
  }

  SmallVector<Value> ret_operands;
  SmallVector<Type> ret_types;
  SmallVector<Attribute> control_ret_attrs;
  {
    SmallVector<Attribute> res_attrs;
    ret_types.reserve(fbody->ret_types.size() * 2);
    for (Node* ret : fbody->ret_nodes) {
      // Find node in the graph using the node id instead of using `arg`
      // directly because the graph has been cloned.
      Operation* ret_op = importer.GetOperationForNode(ret->id());
      if (!ret_op) return Internal("Missing mapping for return #", ret->id());
      if (ret_op->getName().getStringRef() != "tfg._Retval" &&
          ret_op->getName().getStringRef() != "tfg._DeviceRetval")
        return InvalidArgument("Expect `_Retval` node but got ",
                               ret_op->getName().getStringRef().str());
      if (ret_op->getNumOperands() != 1)
        return InvalidArgument(
            "Expect `_Retval` node to have a single input, got ",
            ret_op->getNumOperands());

      ret_operands.push_back(ret_op->getOperand(0));
      ret_types.push_back(ret_op->getOperand(0).getType());
      ret_op->erase();

      NamedAttrList output_attrs;
      int64_t index;
      TF_RETURN_IF_ERROR(GetNodeAttr(ret->attrs(), "index", &index));
      const OpDef_ArgDef& output = signature.output_arg(index);
      output_attrs.append("tfg.name", builder.getStringAttr(output.name()));
      Type output_type;
      if (output.type() != tensorflow::DT_INVALID) {
        TF_RETURN_IF_ERROR(
            ConvertDataType(output.type(), builder, &output_type));
        output_attrs.append("tfg.dtype", TypeAttr::get(output_type));
      }
      if (!output.description().empty())
        output_attrs.append("tfg.description",
                            builder.getStringAttr(output.description()));
      if (output.handle_data_size()) {
        TF_ASSIGN_OR_RETURN(Attribute handle_data,
                            ConvertHandleData(builder, output.handle_data()));
        output_attrs.append("tfg.handle_data", handle_data);
      }
      if (output.has_experimental_full_type()) {
        TF_ASSIGN_OR_RETURN(tf_type::FullTypeAttr type,
                            ConvertAttribute(output.experimental_full_type(),
                                             builder, tfgDialect));
        output_attrs.append("tfg.experimental_full_type", type);
      }
      res_attrs.push_back(output_attrs.getDictionary(context));
    }
    attrs.push_back(
        builder.getNamedAttr(function_interface_impl::getResultDictAttrName(),
                             builder.getArrayAttr(res_attrs)));
    DenseMap<StringRef, Node*> control_ret_nodes;
    for (Node* node : fbody->control_ret_nodes)
      control_ret_nodes.insert({node->name(), node});

    for (const std::string& sig_name : signature.control_output()) {
      auto it = fdef.control_ret().find(sig_name);
      if (it == fdef.control_ret().end())
        return InvalidArgument(
            "Signature control_output not found in fdef.control_ret: ",
            sig_name);
      Node* ret = control_ret_nodes[it->second];
      if (!ret)
        return InvalidArgument(
            "Control return node '", it->second,
            "' not found in the graph for signature control output '", sig_name,
            "'");
      // Find node in the graph using the node id instead of using `arg`
      // directly because the graph has been cloned.
      Operation* control_ret_op = importer.GetOperationForNode(ret->id());
      if (!control_ret_op)
        return Internal("Missing mapping for control result '", sig_name, "'");
      ret_operands.push_back(TFOp(control_ret_op).controlRet());
      control_ret_attrs.push_back(builder.getDictionaryAttr(
          NamedAttribute(tfgDialect->getTfgNameAttrIdentifier(),
                         builder.getStringAttr(sig_name))));
    }
  }

  builder = OpBuilder::atBlockEnd(func_op.getBody());
  builder.create<ReturnOp>(module.getLoc(), ret_operands,
                           builder.getArrayAttr(control_ret_attrs));

  SmallVector<Type> arg_types;
  SmallString<8> arg_or_res_attr_name;
  {
    SmallVector<Attribute> arg_attrs;
    arg_types.reserve(fbody->arg_types.size() * 2);
    for (auto& enumerated_arg : llvm::enumerate(fbody->arg_nodes)) {
      int arg_id = enumerated_arg.index();
      Node* arg = enumerated_arg.value();
      // Find node in the graph using the node id instead of using `arg`
      // directly because the graph has been cloned.
      Operation* arg_op = importer.GetOperationForNode(arg->id());
      if (!arg_op) return Internal("Missing mapping for arg #", arg->id());
      if (arg_op->getName().getStringRef() != "tfg._Arg")
        return InvalidArgument("Expect `_Arg` node but got ",
                               arg_op->getName().getStringRef().str());
      if (arg_op->getNumResults() != 2)
        return InvalidArgument(
            "Expect `_Arg` node to have a single output, got ",
            arg_op->getNumResults());
      body->addArgument(arg_op->getResult(0).getType(), unknown_loc);
      arg_types.push_back(arg_op->getResult(0).getType());
      arg_op->getResult(0).replaceAllUsesWith(body->getArguments().back());

      body->addArgument(arg_op->getResult(1).getType(), unknown_loc);
      arg_types.push_back(arg_op->getResult(1).getType());
      arg_op->getResult(1).replaceAllUsesWith(body->getArguments().back());

      arg_op->erase();

      int64_t index;
      TF_RETURN_IF_ERROR(GetNodeAttr(arg->attrs(), "index", &index));
      const OpDef_ArgDef& input = signature.input_arg(index);
      NamedAttrList input_attrs;
      input_attrs.set("tfg.name", builder.getStringAttr(input.name()));
      if (!input.description().empty())
        input_attrs.append("tfg.description",
                           builder.getStringAttr(input.description()));

      if (input.is_ref())
        input_attrs.append("tfg.is_ref", builder.getUnitAttr());

      if (input.handle_data_size()) {
        TF_ASSIGN_OR_RETURN(Attribute handle_data,
                            ConvertHandleData(builder, input.handle_data()));

        input_attrs.append("tfg.handle_data", handle_data);
      }

      if (input.has_experimental_full_type()) {
        TF_ASSIGN_OR_RETURN(tf_type::FullTypeAttr type,
                            ConvertAttribute(input.experimental_full_type(),
                                             builder, tfgDialect));
        input_attrs.append("tfg.experimental_full_type", type);
      }

      auto it = fbody->fdef.arg_attr().find(arg_id);
      if (it != fbody->fdef.arg_attr().end()) {
        for (const auto& namedAttr : it->second.attr()) {
          std::string name = absl::StrCat("tf.", namedAttr.first);
          const AttrValue& tf_attr = namedAttr.second;
          TF_ASSIGN_OR_RETURN(
              Attribute attr,
              ConvertAttributeValue(tf_attr, builder, tfgDialect));
          input_attrs.append(name, attr);
        }
      }
      arg_attrs.push_back(input_attrs.getDictionary(context));
      arg_attrs.push_back(builder.getDictionaryAttr({}));
    }
    attrs.push_back(
        builder.getNamedAttr(function_interface_impl::getArgDictAttrName(),
                             builder.getArrayAttr(arg_attrs)));
  }

  func_op->setAttrs(attrs);
  func_op->setAttr("function_type", TypeAttr::get(builder.getFunctionType(
                                        arg_types, ret_types)));

  return func_op;
}

bool IsGenericFunction(FunctionDef fdef) {
  for (const NodeDef& node : fdef.node_def())
    for (const auto& named_attr : node.attr()) {
      if (!named_attr.second.placeholder().empty()) return true;
    }
  return false;
}

}  // namespace

// Convert an array of "handle_data" (a DType and a Shape) to an MLIR array
// attribute. Each entry will be itself an ArrayAttribute containing a TypeAttr
// and a ShapeAttr
tensorflow::StatusOr<ArrayAttr> ConvertHandleData(
    Builder builder,
    const RepeatedPtrField<ResourceHandleProto_DtypeAndShape>& handle_data) {
  // Two entries: a type and a shape.
  SmallVector<Attribute> dtype_and_shape;
  for (const auto& handle : handle_data) {
    if (handle.dtype() == tensorflow::DT_INVALID)
      return InvalidArgument("Invalid dtype for handle_data");
    Type dtype;
    TF_RETURN_IF_ERROR(ConvertDataType(handle.dtype(), builder, &dtype));
    TF_ASSIGN_OR_RETURN(
        ShapeAttr shape,
        ConvertTensorShapeProto(handle.shape(), builder.getContext()));
    TensorType handle_type;
    if (shape.hasRank()) {
      handle_type = RankedTensorType::get(shape.getShape(), dtype);
    } else {
      handle_type = UnrankedTensorType::get(dtype);
    }
    dtype_and_shape.push_back(TypeAttr::get(handle_type));
  }
  return builder.getArrayAttr(dtype_and_shape);
}
// Convert a Graph and function libs to a MLIR module containing the graph and
// expressed in TFG dialect.
tensorflow::StatusOr<OwningOpRef<mlir::ModuleOp>> ImportGraphAndFunctionsToMlir(
    MLIRContext* context, const Graph& graph, const GraphDebugInfo& debug_info,
    const FunctionLibraryDefinition& flib_def) {
  LoadDialects(context);
  // Create the graph operation in which we will convert the individual nodes.
  OwningOpRef<mlir::ModuleOp> module =
      ModuleOp::create(UnknownLoc::get(context));
  OpBuilder builder = OpBuilder::atBlockEnd(module->getBody());

  auto graph_op = builder.create<GraphOp>(
      module->getLoc(), getVersionAttr(context, graph.versions()));
  graph_op.nodes().push_back(new Block);

  // Import the nodes in the graph body.
  GraphImporter importer(context, graph, debug_info);
  TF_RETURN_IF_ERROR(importer.Convert(graph_op.getBody()));

  llvm::StringMap<llvm::StringMap<SmallVector<Value, 1>>> values_map;
  for (const std::string& name : flib_def.ListFunctionNames()) {
    const FunctionDef* fdef = flib_def.Find(name);
    if (IsGenericFunction(*fdef)) {
      TF_RETURN_IF_ERROR(ConvertGenericFunction(*fdef, builder));
    } else {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          ImportFunctionDef(*module, debug_info, flib_def, *fdef,
                            /*instantiation_attributes=*/{})
              .status(),
          "While importing FunctionDef: ", fdef->signature().name());
    }
  }
  return module;
}

// Convert a GraphDef to a MLIR module containing the graph and expressed in TFG
// dialect.
tensorflow::StatusOr<OwningOpRef<mlir::ModuleOp>> ImportGraphDefToMlir(
    MLIRContext* context, const GraphDebugInfo& debug_info,
    const GraphDef& graphdef) {
  VLOG(4) << "ConvertGraphdefToMlir begin";
  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.add_default_attributes = true;
  // TODO(aminim): remove dependency on the global registry and allow for
  // injection.
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(options, graphdef, &graph));
  return ImportGraphAndFunctionsToMlir(context, graph, debug_info,
                                       graph.flib_def());
}

tensorflow::StatusOr<OwningOpRef<mlir::ModuleOp>> ImportSavedModelToMlir(
    mlir::MLIRContext* context, const tensorflow::GraphDebugInfo& debug_info,
    const tensorflow::SavedModel& saved_model) {
  if (saved_model.meta_graphs_size() == 0) {
    return tensorflow::errors::InvalidArgument(
        "Input saved model has no meta graphs");
  }

  if (saved_model.meta_graphs_size() > 1) {
    return tensorflow::errors::InvalidArgument(
        "Input saved model has more than one meta graph, currently not "
        "supported");
  }

  const auto& graphdef = saved_model.meta_graphs(0).graph_def();
  return ImportGraphDefToMlir(context, debug_info, graphdef);
}

}  // namespace tfg
}  // namespace mlir

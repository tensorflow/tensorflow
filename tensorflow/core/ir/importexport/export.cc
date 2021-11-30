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

#include "tensorflow/core/ir/importexport/export.h"

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/FunctionSupport.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/convert_attributes.h"
#include "tensorflow/core/ir/importexport/convert_tensor.h"
#include "tensorflow/core/ir/importexport/convert_types.h"
#include "tensorflow/core/ir/importexport/functiondef_export.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

#define DEBUG_TYPE "graphdef-to-mlir"

using tensorflow::DataType;
using tensorflow::FunctionDef;
using tensorflow::GetValueNameFn;
using tensorflow::GradientDef;
using tensorflow::GraphDef;
using tensorflow::NodeDef;
using tensorflow::OpDef;
using tensorflow::Status;
using tensorflow::VersionDef;
using tensorflow::errors::InvalidArgument;

namespace mlir {
namespace tfg {
namespace {
constexpr char kDeviceAttr[] = "_mlir_device";
constexpr char kAliasingAttr[] = "tf.aliasing_output";

// Compute the name to use in GraphDef for a given Value (either the result of
// an operation or a block operand if a function argument) and store the result
// in the provided name string. The `control_ty` is the instance of the
// `ControlType` to compare against and detect a control dependency case.
static Status GetValueName(Value operand, std::string &name, Type control_ty) {
  OpResult op_result = operand.dyn_cast<OpResult>();
  if (!op_result) {
    BlockArgument block_operand = operand.dyn_cast<BlockArgument>();
    bool is_control = (block_operand.getType() == control_ty);
    int arg_num = block_operand.getArgNumber();
    name.clear();
    // Function arguments are coming as pair: the even are the actual tensors
    // while the odd position are the associated control input.
    if (is_control) name = "^";
    DictionaryAttr arg_attrs = function_like_impl::getArgAttrDict(
        block_operand.getParentBlock()->getParentOp(), arg_num - is_control);
    if (!arg_attrs)
      return InvalidArgument("Missing attribute for argument #", arg_num);
    StringAttr arg_name = arg_attrs.getAs<StringAttr>("tfg.name");
    if (!arg_name)
      return InvalidArgument(
          "Can't export graph with missing op-name for function parameter #",
          arg_num);
    absl::StrAppend(&name, arg_name.getValue().str());
    return {};
  }
  Operation *producer = op_result.getDefiningOp();
  auto nameAttr = producer->getAttrOfType<StringAttr>("_mlir_name");
  if (!nameAttr)
    return InvalidArgument("Can't export graph with missing op-name");

  name.clear();
  if (op_result.getType() == control_ty) name = "^";
  absl::StrAppend(&name, nameAttr.getValue().str());
  if (op_result.getType() != control_ty && op_result.getResultNumber())
    absl::StrAppend(&name, ":", op_result.getResultNumber());
  return {};
}

Status GetArgumentNode(GraphFuncOp func, NodeDef *node_def, unsigned index,
                       StringRef name) {
  node_def->set_name(name.str());
  node_def->set_op(tensorflow::FunctionLibraryDefinition::kArgOp);
  TensorType arg_type = func.getArgument(index).getType().cast<TensorType>();

  if (auto resource_type = arg_type.getElementType().dyn_cast<ResourceType>()) {
    llvm::ArrayRef<TensorType> subtypes = resource_type.getSubtypes();
    if (!subtypes.empty()) {
      tensorflow::AttrValue handle_dtypes_attr;
      tensorflow::AttrValue handle_shapes_attr;
      for (TensorType subtype : subtypes) {
        DataType dtype;
        TF_RETURN_IF_ERROR(ConvertToDataType(subtype.getElementType(), &dtype));
        handle_dtypes_attr.mutable_list()->add_type(dtype);

        SetTensorShapeProto(subtype,
                            handle_shapes_attr.mutable_list()->add_shape());
      }

      (*node_def->mutable_attr())["_handle_dtypes"] = handle_dtypes_attr;
      (*node_def->mutable_attr())["_handle_shapes"] = handle_shapes_attr;
    }
  }

  if (arg_type.isa<RankedTensorType>())
    TF_RETURN_IF_ERROR(SetShapeAttribute("_output_shapes", arg_type,
                                         node_def->mutable_attr()));

  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(arg_type.getElementType(), &dtype));
  tensorflow::AttrValue type_attr;
  type_attr.set_type(dtype);
  (*node_def->mutable_attr())["T"] = type_attr;

  tensorflow::AttrValue index_attr;
  index_attr.set_i(index / 2);
  (*node_def->mutable_attr())["index"] = index_attr;

  if (auto device_attr = func.getArgAttrOfType<StringAttr>(index, kDeviceAttr))
    *node_def->mutable_device() = device_attr.getValue().str();

  llvm::ArrayRef<NamedAttribute> func_arg_i_attrs = func.getArgAttrs(index);
  absl::flat_hash_set<absl::string_view> attrs_to_ignore = {
      kDeviceAttr, kAliasingAttr, "tfg.name", "tfg.dtype", "tfg.handle_data"};
  TF_RETURN_IF_ERROR(ConvertAttributes(func_arg_i_attrs, attrs_to_ignore,
                                       /*remove_ref_type=*/false,
                                       node_def->mutable_attr()));

  return Status::OK();
}

Status GetReturnNode(GraphFuncOp function, Value operand, unsigned index,
                     StringRef name, NodeDef *node_def,
                     ControlType control_ty) {
  node_def->set_name(name.str() + "_mlir_ret");
  node_def->set_op(tensorflow::FunctionLibraryDefinition::kRetOp);

  std::string input_name;
  TF_RETURN_IF_ERROR(GetValueName(operand, input_name, control_ty));
  node_def->add_input(input_name);

  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(
      operand.getType().cast<TensorType>().getElementType(), &dtype));
  tensorflow::AttrValue type_attr;
  type_attr.set_type(dtype);
  (*node_def->mutable_attr())["T"] = type_attr;
  tensorflow::AttrValue index_attr;
  index_attr.set_i(index);
  (*node_def->mutable_attr())["index"] = index_attr;

  if (auto device_attr =
          function.getResultAttrOfType<StringAttr>(index, kDeviceAttr))
    *node_def->mutable_device() = device_attr.getValue().str();

  llvm::ArrayRef<NamedAttribute> func_res_i_attrs =
      function.getResultAttrs(index);
  absl::flat_hash_set<absl::string_view> attrs_to_ignore = {
      kDeviceAttr, kAliasingAttr, "tfg.name", "tfg.dtype", "tfg.handle_data"};
  TF_RETURN_IF_ERROR(ConvertAttributes(func_res_i_attrs, attrs_to_ignore,
                                       /*remove_ref_type=*/false,
                                       node_def->mutable_attr()));

  return Status::OK();
}

// Converts a location to the debug information for the node def, if we find
// supported location, that is a top-level NameLoc or any NameLoc nested inside
// a FusedLoc. Other kind of location are ignored. If a NameLoc is of the form
// "name@func" we parse it and import the two appropriately.
void ExtractExperimentalDebugInfoFromLocation(
    mlir::Location inst_loc, NodeDef::ExperimentalDebugInfo *debug_info) {
  auto add_name_loc = [&](mlir::NameLoc name_loc) {
    StringRef node, func;
    std::tie(node, func) = name_loc.getName().strref().split('@');
    debug_info->add_original_node_names(node.str());
    if (!func.empty()) debug_info->add_original_func_names(func.str());
  };
  if (auto fused = inst_loc.dyn_cast<mlir::FusedLoc>()) {
    for (Location loc : fused.getLocations())
      if (auto name_loc = loc.dyn_cast<mlir::NameLoc>()) add_name_loc(name_loc);
    return;
  }
  if (auto name_loc = inst_loc.dyn_cast<mlir::NameLoc>())
    add_name_loc(name_loc);
}

// Convert an MLIR operation to a NodeDef. The `control_ty` is the instance of
// the `ControlType` to compare against and detect a control dependency case.
Status ConvertOperationToNodeImpl(Operation &op, NodeDef *node,
                                  GetValueNameFn get_value_name) {
  auto nameAttr = op.getAttrOfType<StringAttr>("_mlir_name");
  if (nameAttr) node->set_name(nameAttr.getValue().str());
  auto deviceAttr = op.getAttrOfType<StringAttr>(kDeviceAttr);
  if (deviceAttr) node->set_device(deviceAttr.getValue().str());
  std::string name;
  for (Value operand : op.getOperands()) {
    TF_RETURN_IF_ERROR(get_value_name(operand, name));
    node->add_input(name);
  }
  StringRef op_name = op.getName().stripDialect();
  if (op_name == "LegacyCall") {
    auto callee = op.getAttrOfType<FuncAttr>("callee");
    if (!callee)
      return InvalidArgument("Missing callee attribute on LegacyCall");
    StringRef callee_name = callee.getName().getRootReference().getValue();
    node->set_op({callee_name.data(), callee_name.size()});
    TF_RETURN_IF_ERROR(ConvertAttributes(
        callee.getAttrs().getValue(), {"_mlir_name", kDeviceAttr},
        /*remove_ref_type=*/false, node->mutable_attr()));
    auto optional_device =
        op.getAttrDictionary().getNamed("_mlir_assigned_device");
    if (optional_device.hasValue()) {
      NamedAttrList assigned_device;
      assigned_device.push_back(*optional_device);
      TF_RETURN_IF_ERROR(ConvertAttributes(assigned_device, {},
                                           /*remove_ref_type=*/false,
                                           node->mutable_attr()));
    }
  } else {
    node->set_op({op_name.data(), op_name.size()});
    TF_RETURN_IF_ERROR(
        ConvertAttributes(op.getAttrs(), {"_mlir_name", kDeviceAttr},
                          /*remove_ref_type=*/false, node->mutable_attr()));
  }
  // Eliminate empty "_mlir_assigned_device" from the export. This is just
  // more friendly to the serialization.
  {
    auto it = node->mutable_attr()->find("_mlir_assigned_device");
    if (it != node->mutable_attr()->end() && it->second.s().empty())
      node->mutable_attr()->erase("_mlir_assigned_device");
  }

  // Export the location as debug info on the nodes.
  ExtractExperimentalDebugInfoFromLocation(
      op.getLoc(), node->mutable_experimental_debug_info());
  if (node->experimental_debug_info().original_node_names().empty())
    node->clear_experimental_debug_info();

  return Status::OK();
}

// Convert the handle_data_arr to the `handle_data` field of the provided arg.
// Each entry of the array is itself an array with two entries: a Type and a
// ShapeAttr.
static Status ConvertHandleDataImpl(ArrayAttr handle_data_arr,
                                    OpDef::ArgDef *arg) {
  if (!handle_data_arr) return {};
  for (Attribute handle_data_attr : handle_data_arr) {
    auto handle_data_arr_attr = handle_data_attr.dyn_cast<ArrayAttr>();
    if (!handle_data_arr_attr || handle_data_arr_attr.size() != 2)
      return InvalidArgument(
          "Expected an array attribute of size 2 for handle_data element "
          "but got ",
          debugString(handle_data_attr));

    TypeAttr type_attr = handle_data_arr_attr[0].dyn_cast<TypeAttr>();
    if (!type_attr)
      return InvalidArgument(
          "Expected a Type attribute for first handle_data entry but "
          "got ",
          debugString(handle_data_arr_attr[0]));

    auto shape = handle_data_arr_attr[1].dyn_cast<tfg::ShapeAttr>();
    if (!shape)
      return InvalidArgument(
          "Expected a ShapeAttr attribute for second handle_data entry but "
          "got ",
          debugString(handle_data_arr_attr[1]));

    auto *handle_data = arg->add_handle_data();
    if (shape.hasStaticShape())
      ConvertToTensorShapeProto(shape.getShape(), handle_data->mutable_shape());
    else
      handle_data->mutable_shape()->set_unknown_rank(true);
    DataType dtype;
    TF_RETURN_IF_ERROR(ConvertToDataType(type_attr.getValue(), &dtype));
    handle_data->set_dtype(dtype);
  }
  return {};
}

Status BuildFunctionSignature(GraphFuncOp func_op, FunctionDef &fdef) {
  const std::string func_name = func_op.getName().str();
  OpDef *signature = fdef.mutable_signature();
  signature->set_name(func_name);
  if (func_op->getAttr("is_stateful")) signature->set_is_stateful(true);
  if (auto description = func_op->getAttrOfType<StringAttr>("description"))
    signature->set_description(description.getValue().str());
  // Handle the results now.
  // An ArgDef entry needs to be constructed for all non-control returned value.
  auto return_op = cast<tfg::ReturnOp>(func_op.getBody()->getTerminator());
  ArrayAttr results_attr = func_op.getAllResultAttrs();
  auto control_ty = tfg::ControlType::get(func_op.getContext());
  std::string ret_name;
  for (auto indexed_result : llvm::enumerate(return_op->getOperands())) {
    int res_num = indexed_result.index();
    Value ret_val = indexed_result.value();
    if (ret_val.getType() == control_ty) {
      auto name = return_op->getAttrOfType<StringAttr>(
          absl::StrCat("tfg.control_ret_name_", res_num));
      if (!name)
        return InvalidArgument("Can't export function ", func_name,
                               " because missing \"tfg.control_ret_name_\" "
                               "attribute for control result #",
                               res_num);
      signature->add_control_output(name.getValue().str());
    } else {
      auto res_attrs = results_attr[res_num].dyn_cast<DictionaryAttr>();
      auto name = res_attrs.getAs<StringAttr>("tfg.name");
      if (!name)
        return InvalidArgument(
            "Can't export function ", func_name,
            " because missing \"tfg.name\" attribute for result #", res_num);
      OpDef::ArgDef *arg = signature->add_output_arg();
      arg->set_name(name.getValue().str());
      StringAttr description = res_attrs.getAs<StringAttr>("tfg.description");
      if (description) arg->set_description(description.getValue().str());
      TF_RETURN_IF_ERROR(ConvertHandleData(
          res_attrs.getAs<ArrayAttr>("tfg.handle_data"), arg));
    }
  }
  return Status::OK();
}

// Export a GraphFunc operation as a new entry in the function library.
static Status ExportFunction(GraphFuncOp func_op,
                             tensorflow::FunctionLibraryDefinition &flib) {
  const std::string func_name = func_op.getName().str();
  // The function->gradient mapping is stored separately in the library.
  if (auto gradient_attr =
          func_op->getAttrOfType<FlatSymbolRefAttr>("gradient")) {
    GradientDef gradient;
    gradient.set_gradient_func(gradient_attr.getValue().str());
    gradient.set_function_name(func_name);
    TF_RETURN_IF_ERROR(flib.AddGradientDef(gradient));
  }

  auto control_ty = tfg::ControlType::get(func_op.getContext());
  GraphDef graph_def;
  ArrayAttr args_attr = func_op.getAllArgAttrs();
  for (int arg_num : llvm::seq<int>(0, func_op.getNumArguments())) {
    // Odd position are just for control dependencies.
    if (arg_num % 2) continue;
    DictionaryAttr arg_attrs = args_attr[arg_num].dyn_cast<DictionaryAttr>();
    auto name = arg_attrs.getAs<StringAttr>("tfg.name");
    if (!name || name.getValue().empty())
      return tensorflow::errors::InvalidArgument(
          "Missing tfg.name on argument ", arg_num);
    NodeDef *node_def = graph_def.add_node();
    TF_RETURN_IF_ERROR(
        GetArgumentNode(func_op, node_def, arg_num, name.getValue()));
  }
  // Convert the invidual nodes in the function body, since the function is
  // terminated by a return operation we skip it in this loop and handled it
  // separately later.
  for (Operation &op : func_op.getBody()->without_terminator())
    TF_RETURN_IF_ERROR(ConvertOperationToNode(
        op, graph_def.add_node(), [&](Value operand, std::string &output_name) {
          return GetValueName(operand, output_name, control_ty);
        }));

  auto return_op = cast<tfg::ReturnOp>(func_op.getBody()->getTerminator());
  ArrayAttr results_attr = func_op.getAllResultAttrs();
  for (auto indexed_result : llvm::enumerate(return_op->getOperands())) {
    int res_num = indexed_result.index();
    Value ret_val = indexed_result.value();
    if (ret_val.getType() == control_ty) continue;
    auto res_attrs = results_attr[res_num].dyn_cast<DictionaryAttr>();
    if (!res_attrs)
      return InvalidArgument("Can't export function ", func_name,
                             " because missing attributes for result #",
                             res_num);
    auto name = res_attrs.getAs<StringAttr>("tfg.name");
    if (!name)
      return InvalidArgument(
          "Can't export function ", func_name,
          " because missing \"tfg.name\" attribute for result #", res_num);

    NodeDef *node_def = graph_def.add_node();
    TF_RETURN_IF_ERROR(GetReturnNode(func_op, ret_val, res_num, name.getValue(),
                                     node_def, control_ty));
  }

  tensorflow::GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.add_default_attributes = true;
  // TODO(aminim): remove dependency on the global registry and allow for
  // injection.
  tensorflow::Graph graph(&flib);

  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToGraph(
      options, std::move(graph_def), &graph));

  FunctionDef func_def;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(graph, func_name, &func_def));

  OpDef *signature = func_def.mutable_signature();
  if (func_op->getAttr("is_stateful")) signature->set_is_stateful(true);
  if (auto description = func_op->getAttrOfType<StringAttr>("description"))
    signature->set_description(description.getValue().str());

  // Some ArgDef updates couldn't be carried through the graph nodes, like
  // "handle_data".
  for (int arg_num : llvm::seq<int>(0, func_op.getNumArguments())) {
    // Odd position are just for control dependencies.
    if (arg_num % 2) continue;
    DictionaryAttr arg_attrs =
        function_like_impl::getArgAttrDict(func_op, arg_num);
    OpDef::ArgDef *arg = signature->mutable_input_arg(arg_num / 2);
    StringAttr description = arg_attrs.getAs<StringAttr>("tfg.description");
    if (description) arg->set_description(description.getValue().str());
    TF_RETURN_IF_ERROR(
        ConvertHandleData(arg_attrs.getAs<ArrayAttr>("tfg.handle_data"), arg));
  }
  // Handle the results now.
  // An ArgDef entry needs to be constructed for all non-control returned value,
  // and a mapping from the output name to the signature is also recorded in the
  // FunctionDef.
  std::string ret_name;
  for (auto indexed_result : llvm::enumerate(return_op->getOperands())) {
    int res_num = indexed_result.index();
    Value ret_val = indexed_result.value();
    if (ret_val.getType() == control_ty) {
      auto name = return_op->getAttrOfType<StringAttr>(
          absl::StrCat("tfg.control_ret_name_", res_num));
      if (!name)
        return InvalidArgument("Can't export function ", func_name,
                               " because missing \"tfg.control_ret_name_\" "
                               "attribute for control result #",
                               res_num);
      // When we return a control dependency, it is not really a returned value
      // but it is added to the `control_ret` field of the FunctionDef.
      TF_RETURN_IF_ERROR(GetValueName(ret_val, ret_name, control_ty));
      func_def.mutable_control_ret()->insert(
          {name.getValue().str(), StringRef(ret_name).drop_front().str()});
      signature->add_control_output(name.getValue().str());
    } else {
      auto res_attrs = results_attr[res_num].dyn_cast<DictionaryAttr>();
      auto name = res_attrs.getAs<StringAttr>("tfg.name");
      if (!name)
        return InvalidArgument(
            "Can't export function ", func_name,
            " because missing \"tfg.name\" attribute for result #", res_num);
      OpDef::ArgDef *arg = signature->mutable_output_arg(res_num);
      auto it = func_def.mutable_ret()->find(arg->name());
      if (it == func_def.mutable_ret()->end())
        return tensorflow::errors::Internal(
            "Mismatch in name mapping for returned value");
      func_def.mutable_ret()->insert({name.getValue().str(), it->second});
      func_def.mutable_ret()->erase(it);
      arg->set_name(name.getValue().str());
      StringAttr description = res_attrs.getAs<StringAttr>("tfg.description");
      if (description) arg->set_description(description.getValue().str());
      TF_RETURN_IF_ERROR(ConvertHandleData(
          res_attrs.getAs<ArrayAttr>("tfg.handle_data"), arg));
    }
  }

  // Handled the `resource_arg_unique_id` entries. At the moment it is
  // represented as two vectors of integers which are expected of the same
  // length.
  auto unique_ids_keys = func_op->getAttrOfType<DenseIntElementsAttr>(
      "resource_arg_unique_ids_keys");
  if (unique_ids_keys) {
    auto unique_ids_values = func_op->getAttrOfType<DenseIntElementsAttr>(
        "resource_arg_unique_ids_values");
    if (!unique_ids_values)
      return InvalidArgument(
          "Can't export function ", func_name,
          " because \"resource_arg_unique_ids_keys\" attribute is present "
          "but "
          "\"resource_arg_unique_ids_values\" is missing");
    if (unique_ids_keys.size() != unique_ids_values.size())
      return InvalidArgument(
          "Can't export function ", func_name,
          " because \"resource_arg_unique_ids_keys\" array does not have the "
          "same size as \"resource_arg_unique_ids_values\"");

    auto *unique_ids_map = func_def.mutable_resource_arg_unique_id();
    for (auto key_value : llvm::zip(unique_ids_keys.getValues<int32_t>(),
                                    unique_ids_values.getValues<int32_t>()))
      (*unique_ids_map)[std::get<0>(key_value)] = std::get<1>(key_value);
  }

  // Finally the dialect attributes (prefixed by `tf.` in general) are converted
  // as-is and stored on the `attr` field of the FunctionDef.
  SmallVector<NamedAttribute> funcAttrs(func_op->getDialectAttrs());
  TF_RETURN_IF_ERROR(ConvertAttributes(funcAttrs, {},
                                       /*remove_ref_type=*/false,
                                       func_def.mutable_attr()));
  if (flib.Find(func_def.signature().name()))
    TF_RETURN_IF_ERROR(
        flib.ReplaceFunction(func_def.signature().name(), func_def));
  else
    TF_RETURN_IF_ERROR(flib.AddFunctionDef(func_def));
  return {};
}

// Given an MLIR module, returns a GraphDef.
Status ExportMlirToGraphdefImpl(ModuleOp module, GraphDef *graphdef) {
  // Check that this module is valid for export: it should only contains at most
  // a single Graph operation and one or more GraphFunc operations.
  GraphOp graph_op;
  for (Operation &op : *module.getBody()) {
    if (isa<GraphFuncOp>(op)) continue;
    if (auto new_graph_op = dyn_cast<GraphOp>(op)) {
      if (graph_op) {
        return InvalidArgument(
            "Can't export module with two different tfg.graph");
      }
      graph_op = new_graph_op;
      continue;
    }
    return InvalidArgument(
        absl::StrCat("Can't export module with other ops than tfg.graph or "
                     "tfg.func, has: ",
                     op.getName().getStringRef().data()));
  }
  if (graph_op) {
    // A graph is mostly a flat "sea of nodes" to export.
    auto control_ty = tfg::ControlType::get(graph_op.getContext());
    VersionDef *version = graphdef->mutable_versions();
    tfg::VersionAttr versionAttr = graph_op.version();
    version->set_producer(versionAttr.getProducer());
    version->set_min_consumer(versionAttr.getMinConsumer());
    for (int32_t bad_consumer : versionAttr.getBadConsumers())
      version->add_bad_consumers(bad_consumer);
    for (Operation &op : *graph_op.getBody()) {
      NodeDef *node = graphdef->add_node();
      TF_RETURN_IF_ERROR(ConvertOperationToNode(
          op, node, [&](Value operand, std::string &output_name) {
            return GetValueName(operand, output_name, control_ty);
          }));
    }
  }

  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             *graphdef->mutable_library());
  // Export the functions, if any.
  for (GraphFuncOp func_op :
       llvm::reverse(module.getBody()->getOps<GraphFuncOp>())) {
    LLVM_DEBUG(llvm::errs()
               << "Exporting function @" << func_op.getName() << "\n");
    if (flib.Find(func_op.getName().str())) continue;
    if (!func_op.generic()) {
      // Export only the signature here, we'll export these below.
      FunctionDef fdef;
      TF_RETURN_IF_ERROR(BuildFunctionSignature(func_op, fdef));
      TF_RETURN_IF_ERROR(flib.AddFunctionDef(fdef));
      continue;
    }
    // We can immediately export generic functions, because they don't need to
    // go through a "Graph" and aren't sensitive to importing called function
    // first.
    TF_ASSIGN_OR_RETURN(FunctionDef fdef,
                        ConvertGenericFunctionToFunctionDef(func_op));
    if (flib.Find(fdef.signature().name()))
      TF_RETURN_IF_ERROR(flib.ReplaceFunction(fdef.signature().name(), fdef));
    else
      TF_RETURN_IF_ERROR(flib.AddFunctionDef(fdef));
  }
  for (GraphFuncOp func_op :
       llvm::reverse(module.getBody()->getOps<GraphFuncOp>())) {
    LLVM_DEBUG(llvm::errs()
               << "Exporting function @" << func_op.getName() << "\n");
    if (func_op.generic()) continue;
    TF_RETURN_IF_ERROR(ExportFunction(func_op, flib));
  }
  *graphdef->mutable_library() = flib.ToProto();

  return Status::OK();
}

}  // namespace
}  // namespace tfg
}  // namespace mlir

namespace tensorflow {

Status ConvertHandleData(mlir::ArrayAttr handle_data_arr, OpDef::ArgDef *arg) {
  return mlir::tfg::ConvertHandleDataImpl(handle_data_arr, arg);
}

Status ExportMlirToGraphdef(mlir::ModuleOp module, GraphDef *output_graph) {
  return mlir::tfg::ExportMlirToGraphdefImpl(module, output_graph);
}

Status ConvertOperationToNode(mlir::Operation &op, NodeDef *node,
                              GetValueNameFn get_value_name) {
  return mlir::tfg::ConvertOperationToNodeImpl(op, node, get_value_name);
}

}  //  namespace tensorflow

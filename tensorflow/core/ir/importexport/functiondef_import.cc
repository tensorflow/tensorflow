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

#include "tensorflow/core/ir/importexport/functiondef_import.h"

#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/convert_attributes.h"
#include "tensorflow/core/ir/importexport/convert_types.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

using tensorflow::AttrValue;
using tensorflow::FunctionDef;
using tensorflow::NodeDef;
using tensorflow::OpDef;
using tensorflow::OpDef_AttrDef;
using tensorflow::Status;
using tensorflow::errors::InvalidArgument;
using tensorflow::protobuf::RepeatedPtrField;

#define DEBUG_TYPE "graphdef-to-mlir"

namespace mlir {
namespace tfg {
namespace {

class ValueMapManager {
 public:
  ValueMapManager(
      llvm::StringMap<llvm::StringMap<SmallVector<Value, 1>>>& values_map,
      OpBuilder& builder, OperationName mlir_placeholder, Type placeholder_ty,
      Type control_ty, Location loc)
      : values_map_(values_map),
        builder_(builder),
        loc_(loc),
        mlir_placeholder_(mlir_placeholder),
        placeholder_ty_(placeholder_ty),
        control_ty_(control_ty) {}

  Status DefineOperation(Operation* op, StringRef node_name) {
    llvm::StringMap<SmallVector<Value, 1>>& op_info = values_map_[node_name];
    SmallVector<Value, 1>& base_operation = op_info["^"];
    // Replace placeholders.
    if (!base_operation.empty()) {
      Operation* placeholder = base_operation[0].getDefiningOp();
      if (!placeholder ||
          placeholder->getName().getStringRef() != "tfg.__mlir_placeholder")
        return InvalidArgument(absl::StrCat(
            "Duplicated node (or function argument) with the same name: `",
            node_name.str(), "`"));

      op->moveBefore(placeholder);
      placeholder->replaceAllUsesWith(op);
      placeholder->erase();
      base_operation.clear();
    }
    base_operation.push_back(op->getResult(1));
    base_operation.push_back(op->getResult(0));
    return Status::OK();
  }

  Value GetValueOrCreatePlaceholder(StringRef full_name) {
    StringRef node_name;
    StringRef output_name = "";
    bool is_control_dep = full_name[0] == '^';
    int output_num = 0;
    if (is_control_dep) full_name = full_name.drop_front();
    {
      size_t colon_sep = full_name.find_first_of(':');
      if (colon_sep == StringRef::npos) {
        node_name = full_name;
      } else {
        node_name = full_name.take_front(colon_sep);
        output_name = full_name.drop_front(colon_sep + 1);
      }
      colon_sep = output_name.find_last_of(':');
      if (colon_sep != StringRef::npos) {
        // NOLINTNEXTLINE: type matching the API taking a reference.
        unsigned long long value;
        if (!llvm::getAsUnsignedInteger(output_name.drop_front(colon_sep + 1),
                                        10, value))
          output_num = value;
        output_name = output_name.take_front(colon_sep);
      }
    }

    llvm::StringMap<SmallVector<Value, 1>>& op_info = values_map_[node_name];
    SmallVector<Value, 1>& base_operation = op_info["^"];
    if (base_operation.empty()) {
      OperationState state(loc_, mlir_placeholder_);
      state.addAttribute(TFGraphDialect::getNameAttrKey(),
                         builder_.getStringAttr(node_name));
      state.types.push_back(placeholder_ty_);
      state.types.push_back(control_ty_);
      Operation* placeholder = builder_.create(state);
      base_operation.push_back(placeholder->getResult(1));
      base_operation.push_back(placeholder->getResult(0));
    }
    if (is_control_dep) return base_operation[0];
    SmallVector<Value, 1>& value_info = op_info[output_name];
    if (value_info.size() <= output_num)
      value_info.resize(output_num + 1, Value{});
    if (!value_info[output_num]) {
      // Create a tfg.get_result for this output.
      value_info[output_num] = builder_.create<GetResultOp>(
          loc_, base_operation[1], output_name, output_num);
    }
    return value_info[output_num];
  }

 private:
  llvm::StringMap<llvm::StringMap<SmallVector<Value, 1>>>& values_map_;
  OpBuilder& builder_;
  Location loc_;
  OperationName mlir_placeholder_;
  Type placeholder_ty_;
  Type control_ty_;
};

// Convert the list of `nodes` one by one into MLIR Operations using the
// provided OpBuilder.
// The provided `nodes_map` will be populated with a mapping from the node name
// to the result count and the Operation.
// The supplied `args_map` is looked up for Function arguments when an entry
// cannot be found in the nodes_map.
Status ImportNodes(ValueMapManager value_manager,
                   const RepeatedPtrField<NodeDef>& nodes, OpBuilder& builder) {
  Location unknown_loc = builder.getUnknownLoc();
  MLIRContext* context = builder.getContext();

  Type placeholder_ty = OpaqueTensorType::get(context);
  Type control_ty = ControlType::get(context);
  TFGraphDialect* tfgDialect =
      cast<TFGraphDialect>(context->getLoadedDialect("tfg"));
  StringAttr device_attr = tfgDialect->getDeviceAttrIdentifier();
  StringAttr name_attr = tfgDialect->getNameAttrIdentifier();
  StringAttr fulltype_attr = tfgDialect->getFullTypeAttrIdentifier();
  // Process every node and create a matching MLIR operation
  for (const NodeDef& node : nodes) {
    DVLOG(1) << "Processing node " << node.name() << "\n";
    if (node.op().empty()) return InvalidArgument("empty op type");
    OperationState state(unknown_loc, absl::StrCat("tfg.", node.op()));
    // Fetch the inputs, creating placeholder if an input hasn't been visited.
    for (const std::string& input : node.input())
      state.operands.push_back(
          value_manager.GetValueOrCreatePlaceholder(input));
    // Retrieve the entry in the nodes_map for this node and infer the result
    // count from what was inferred during the first traversal above.
    state.types.push_back(placeholder_ty);
    state.types.push_back(control_ty);
    // Handle attributes.
    for (const auto& namedAttr : node.attr()) {
      const std::string& name = namedAttr.first;
      const AttrValue& tf_attr = namedAttr.second;
      TF_ASSIGN_OR_RETURN(Attribute attr,
                          ConvertAttributeValue(tf_attr, builder, tfgDialect));
      state.addAttribute(name, attr);
    }
    if (!node.device().empty())
      state.addAttribute(device_attr, StringAttr::get(context, node.device()));
    if (!node.name().empty())
      state.addAttribute(name_attr, StringAttr::get(context, node.name()));
    if (node.has_experimental_type()) {
      TF_ASSIGN_OR_RETURN(
          tf_type::FullTypeAttr type,
          ConvertAttribute(node.experimental_type(), builder, tfgDialect));
      state.addAttribute(fulltype_attr, type);
    }

    Operation* op = builder.create(state);

    StringRef node_name = node.name();
    {
      size_t colon_sep = node_name.find_first_of(':');
      if (colon_sep != StringRef::npos)
        node_name = node_name.take_front(colon_sep);
    }
    TF_RETURN_IF_ERROR(value_manager.DefineOperation(op, node_name));
  }
  // We don't expect any placeholder left at this point, fail if any.
  for (Operation& op : *builder.getInsertionBlock()) {
    if (op.getName().getStringRef() == "tfg.__mlir_placeholder") {
      return InvalidArgument(absl::StrCat(
          "Couldn't import graph: placeholder left ",
          op.getAttrOfType<StringAttr>(name_attr).getValue().str()));
    }
  }
  return Status::OK();
}

tensorflow::StatusOr<NamedAttrList> ConvertArgDefAttributes(
    const OpDef::ArgDef& arg, TFGraphDialect* tfgDialect, Builder builder) {
  NamedAttrList input_attrs;
  StringAttr arg_name = builder.getStringAttr(arg.name());
  input_attrs.set("tfg.name", arg_name);
  if (!arg.description().empty())
    input_attrs.append("tfg.description",
                       builder.getStringAttr(arg.description()));

  Type input_type;
  if (arg.type() != tensorflow::DT_INVALID) {
    TF_RETURN_IF_ERROR(ConvertDataType(arg.type(), builder, &input_type));
    input_attrs.append("tfg.type", TypeAttr::get(input_type));
  }
  if (!arg.type_attr().empty())
    input_attrs.append("tfg.type_attr", builder.getStringAttr(arg.type_attr()));
  if (!arg.number_attr().empty())
    input_attrs.append("tfg.number_attr",
                       builder.getStringAttr(arg.number_attr()));
  if (!arg.type_list_attr().empty())
    input_attrs.append("tfg.type_list_attr",
                       builder.getStringAttr(arg.type_list_attr()));
  if (arg.handle_data_size()) {
    TF_ASSIGN_OR_RETURN(Attribute handle_data,
                        ConvertHandleData(builder, arg.handle_data()));
    input_attrs.append("tfg.handle_data", handle_data);
  }
  if (arg.is_ref()) input_attrs.append("tfg.is_ref", builder.getUnitAttr());
  if (arg.has_experimental_full_type()) {
    TF_ASSIGN_OR_RETURN(
        tf_type::FullTypeAttr type,
        ConvertAttribute(arg.experimental_full_type(), builder, tfgDialect));
    input_attrs.append("tfg.experimental_full_type", type);
  }
  return input_attrs;
}

// Import the given `func` and inser the resulting `GraphFunc`
// operation using the provided `builder`. The `nodes_map` and `args_map` are
// used as scratchpad for the import inside this function. The `gradients` maps
// is provided to
Status ImportGenericFunction(
    GraphFuncOp func_op, const FunctionDef& func,
    llvm::StringMap<llvm::StringMap<SmallVector<Value, 1>>>& values_map,
    OpBuilder& builder) {
  const OpDef& signature = func.signature();
  Location unknown_loc = builder.getUnknownLoc();
  MLIRContext* context = builder.getContext();

  TFGraphDialect* tfgDialect = cast<TFGraphDialect>(func_op->getDialect());
  NamedAttrList attrs;
  DictionaryAttr func_attrs = builder.getDictionaryAttr({});
  if (signature.name().empty())
    return InvalidArgument("generic function without a name");
  attrs.append("sym_name", builder.getStringAttr(signature.name()));
  attrs.append("generic", builder.getUnitAttr());
  if (!signature.description().empty())
    attrs.append("description", builder.getStringAttr(signature.description()));
  if (signature.is_stateful())
    attrs.append("is_stateful", builder.getUnitAttr());
  if (signature.control_output_size()) {
    SmallVector<Attribute> control_outputs;
    for (const std::string& output : signature.control_output())
      control_outputs.push_back(builder.getStringAttr(output));
    attrs.append("control_output", builder.getArrayAttr(control_outputs));
  }
  {
    NamedAttrList attr_defs;
    for (const OpDef_AttrDef& attr : signature.attr()) {
      NamedAttrList attr_def;
      if (attr.name().empty())
        return InvalidArgument("Missing name for function attribute");
      if (!attr.type().empty())
        attr_def.append(builder.getNamedAttr(
            "function_type", builder.getStringAttr(attr.type())));
      if (attr.has_default_value()) {
        TF_ASSIGN_OR_RETURN(
            Attribute attr,
            ConvertAttributeValue(attr.default_value(), builder, tfgDialect));
        attr_def.append(builder.getNamedAttr("default_value", attr));
      }
      if (!attr.description().empty())
        attr_def.append(builder.getNamedAttr(
            "description", builder.getStringAttr(attr.description())));
      if (attr.has_minimum() || attr.minimum())
        attr_def.append(builder.getNamedAttr(
            "minimum", builder.getI32IntegerAttr(attr.minimum())));
      if (attr.has_allowed_values()) {
        TF_ASSIGN_OR_RETURN(
            Attribute attr,
            ConvertAttributeValue(attr.allowed_values(), builder, tfgDialect));
        attr_def.append(builder.getNamedAttr("allowed_values", attr));
      }
      attr_defs.append(builder.getNamedAttr(
          attr.name(), attr_def.getDictionary(builder.getContext())));
    }
    if (!attr_defs.empty()) {
      func_attrs = attr_defs.getDictionary(builder.getContext());
      attrs.append("tfg.func_attrs", func_attrs);
    }
  }

  // The resource_arg_unique_id is a list of `pair<int, int>`, we import it
  // as two arrays of integer right now.
  if (func.resource_arg_unique_id_size()) {
    SmallVector<int32_t> resource_arg_unique_ids_keys;
    SmallVector<int32_t> resource_arg_unique_ids_values;
    for (const auto& unique_id : func.resource_arg_unique_id()) {
      resource_arg_unique_ids_keys.push_back(unique_id.first);
      resource_arg_unique_ids_values.push_back(unique_id.second);
    }
    attrs.append("resource_arg_unique_ids_keys",
                 builder.getI32TensorAttr(resource_arg_unique_ids_keys));
    attrs.append("resource_arg_unique_ids_values",
                 builder.getI32TensorAttr(resource_arg_unique_ids_values));
  }

  // Import the function attributes with a `tf.` prefix to match the current
  // infrastructure expectations.
  for (const auto& namedAttr : func.attr()) {
    if (namedAttr.first.empty())
      return InvalidArgument("Invalid function attribute name");
    const std::string& name = "tf." + namedAttr.first;
    const AttrValue& tf_attr = namedAttr.second;
    TF_ASSIGN_OR_RETURN(Attribute attr,
                        ConvertAttributeValue(tf_attr, builder, tfgDialect));
    attrs.append(name, attr);
  }
  SmallString<8> arg_or_res_attr_name;
  SmallString<8> sub_arg_attr_name;
  // Iterate of the input in the signature. Each input will correspond to
  // potentially multiple arguments because of how the OpDef allows repeated
  // arguments controlled by `number_attr` for example.
  // We populate the `arg_names` vector with the name of each input at each
  // position, and `arg_types` with the matching type.
  int arg_num = 0;
  SmallVector<StringRef> arg_names;
  SmallVector<Type> arg_types;
  SmallVector<Attribute> args_attrs;
  SmallVector<Attribute> res_attrs;
  for (const auto& enumerated_input : llvm::enumerate(signature.input_arg())) {
    const OpDef::ArgDef& input = enumerated_input.value();
    TF_ASSIGN_OR_RETURN(NamedAttrList input_attrs,
                        ConvertArgDefAttributes(input, tfgDialect, builder));
    auto it = func.arg_attr().find(enumerated_input.index());
    if (it != func.arg_attr().end()) {
      NamedAttrList arg_attr;
      for (const auto& named_attr : it->second.attr()) {
        TF_ASSIGN_OR_RETURN(
            Attribute attr,
            ConvertAttributeValue(named_attr.second, builder, tfgDialect));
        arg_attr.append(named_attr.first, attr);
      }
      input_attrs.append("tfg.arg_attrs",
                         arg_attr.getDictionary(builder.getContext()));
    }
    arg_names.push_back(builder.getStringAttr(input.name()).getValue());
    arg_types.push_back(OpaqueTensorType::get(context));
    args_attrs.push_back(input_attrs.getDictionary(context));
    args_attrs.push_back(NamedAttrList{}.getDictionary(context));
    arg_num++;
  }
  attrs.push_back(
      builder.getNamedAttr(function_interface_impl::getArgDictAttrName(),
                           builder.getArrayAttr(args_attrs)));

  // Process the results attributes now.
  int res_num = 0;
  for (const OpDef::ArgDef& output : signature.output_arg()) {
    TF_ASSIGN_OR_RETURN(NamedAttrList output_attrs,
                        ConvertArgDefAttributes(output, tfgDialect, builder));
    res_attrs.push_back(output_attrs.getDictionary(context));
    ++res_num;
  }
  // Process the control output metadata and store them as attributes.
  for (const std::string& output : signature.control_output()) {
    NamedAttrList output_attrs;
    output_attrs.append("tfg.name", builder.getStringAttr(output));
    res_attrs.push_back(output_attrs.getDictionary(context));
    ++res_num;
  }
  attrs.push_back(
      builder.getNamedAttr(function_interface_impl::getResultDictAttrName(),
                           builder.getArrayAttr(res_attrs)));

  values_map.clear();
  Block* body = new Block();
  func_op.body().push_back(body);
  Type control_ty = ControlType::get(context);
  // Create the block arguments and populate the `values_map` with the matching
  // input names.
  for (auto type_and_name : llvm::zip(arg_types, arg_names)) {
    Value arg = body->addArgument(std::get<0>(type_and_name), unknown_loc);
    llvm::StringMap<SmallVector<Value, 1>>& values =
        values_map[std::get<1>(type_and_name)];
    Value ctl = body->addArgument(control_ty, unknown_loc);
    values[""].push_back(arg);
    values["^"].push_back(ctl);
  }

  // Pre-populate the nodes_map with the needed slots for the return.
  OpBuilder body_builder = OpBuilder::atBlockEnd(body);
  // We use placeholders during the import to create "fake" operations to break
  // cycles: we need operands to feed to the users.
  OperationName mlir_placeholder("tfg.__mlir_placeholder", context);
  Type placeholder_ty = OpaqueTensorType::get(context);
  ValueMapManager value_manager(values_map, body_builder, mlir_placeholder,
                                placeholder_ty, control_ty, unknown_loc);

  // Import the function body here, after this we have a function with all
  // the nodes, and the nodes_map contains the mapping from node_name to actual
  // MLIR Operations.
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ImportNodes(value_manager, func.node_def(), body_builder),
      " when importing function ", func.signature().name());

  // After the body, the final part is to setup the return. It comes in two
  // parts: the `ret` field from the FunctionDef for the regular output and the
  // `control_ret` field for the control output.
  //
  // Because `ret` and `control_ret` aren't ordered, there is an indirection to
  // the FunctionDef signature to retrieve the position of each `ret` and
  // `control_ret` entry by name. We compute this mapping from the name of an
  // output to the position in the result array first.
  res_num = 0;
  llvm::StringMap<int> output_name_to_position;
  for (const OpDef::ArgDef& output : signature.output_arg()) {
    if (output_name_to_position.count(output.name()))
      return InvalidArgument("Duplicated output_arg entry", output.name());
    output_name_to_position[output.name()] = res_num;
    ++res_num;
  }
  res_num = 0;
  llvm::StringMap<int> control_output_to_position;
  for (const std::string& output : signature.control_output()) {
    if (control_output_to_position.count(output))
      return InvalidArgument("Duplicated control_output entry", output);
    control_output_to_position[output] = res_num;
    ++res_num;
  }

  // We pre-allocate the array of operands and populate it using the
  // `output_name_to_position` and `control_output_to_position` populated
  // previously.
  SmallVector<Value> ret_vals(func.ret_size() + func.control_ret_size(),
                              Value());
  for (const auto& ret_val : func.ret()) {
    auto position = output_name_to_position.find(ret_val.first);
    if (position == output_name_to_position.end())
      return InvalidArgument(
          "Can't import function, returned value references unknown output "
          "argument ",
          ret_val.first);
    ret_vals[position->second] =
        value_manager.GetValueOrCreatePlaceholder(ret_val.second);
  }
  for (const auto& ret_val : func.control_ret()) {
    auto position = control_output_to_position.find(ret_val.first);
    if (position == control_output_to_position.end())
      return InvalidArgument(
          "Can't import function, returned value references unknown output "
          "argument ",
          ret_val.first);
    Value result = value_manager.GetValueOrCreatePlaceholder(
        (Twine("^") + ret_val.second).str());
    if (!result.getType().isa<ControlType>())
      return InvalidArgument("failed to map returned value ", ret_val.second,
                             ", isn't a control output");
    ret_vals[func.ret_size() + position->second] = result;
  }
  // Check that all the of the return operands have been populated.
  for (auto& indexed_val : llvm::enumerate(ret_vals)) {
    if (indexed_val.value()) continue;
    return InvalidArgument(
        "Failed to import function, missing output for position ",
        indexed_val.index());
  }
  MutableArrayRef<Value> operands = ret_vals;
  ReturnOp ret_op = body_builder.create<ReturnOp>(
      unknown_loc, operands.slice(0, func.ret_size()),
      operands.slice(func.ret_size()));

  // Now that we have all the types, set the function signature as the
  // "function_type" attribute.
  {
    SmallVector<Type> arg_types_with_ctl;
    for (Type type : arg_types) {
      arg_types_with_ctl.push_back(type);
      arg_types_with_ctl.push_back(control_ty);
    }
    attrs.append("function_type",
                 TypeAttr::get(builder.getFunctionType(
                     arg_types_with_ctl, ret_op.getOperandTypes())));
  }
  func_op->setAttrs(attrs);
  return Status::OK();
}

}  // namespace

Status ConvertGenericFunction(GraphFuncOp func_op, const FunctionDef& func,
                              OpBuilder& builder) {
  llvm::StringMap<llvm::StringMap<SmallVector<Value, 1>>> values_map;
  return ImportGenericFunction(func_op, func, values_map, builder);
}

}  // namespace tfg
}  // namespace mlir

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

#include "tensorflow/core/ir/importexport/functiondef_export.h"

#include <string>
#include <utility>

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/convert_attributes.h"
#include "tensorflow/core/ir/importexport/convert_types.h"
#include "tensorflow/core/ir/importexport/graphdef_export.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"

using tensorflow::FunctionDef;
using tensorflow::OpDef;
using tensorflow::OpDef_AttrDef;
using tensorflow::Status;
using tensorflow::errors::InvalidArgument;

#define DEBUG_TYPE "mlir-to-graphdef"

namespace mlir {
namespace tfg {

// Compute the name to use in FunctionDef for a given Value (either the result
// of an operation or a block operand if a function argument) and store the
// result in the provided name string. The `control_ty` is the instance of the
// `ControlType` to compare against and detect a control dependency case.
static tensorflow::StatusOr<std::string> GetValueName(Value operand,
                                                      Type control_ty) {
  bool is_control = (operand.getType() == control_ty);
  OpResult op_result = operand.dyn_cast<OpResult>();
  if (!op_result) {
    BlockArgument block_operand = operand.dyn_cast<BlockArgument>();
    int arg_num = block_operand.getArgNumber();

    // Function arguments are coming as pair: the even are the actual tensors
    // while the odd position are the associated control input.
    std::string name;
    if (is_control) name = "^";
    DictionaryAttr arg_attrs = function_interface_impl::getArgAttrDict(
        block_operand.getParentBlock()->getParentOp(), arg_num - is_control);
    if (!arg_attrs)
      return InvalidArgument("Missing attribute for argument #", arg_num);
    StringAttr arg_name = arg_attrs.getAs<StringAttr>("tfg.name");
    if (!arg_name)
      return InvalidArgument(
          "Can't export graph with missing op-name for function parameter #",
          arg_num);
    absl::StrAppend(&name, arg_name.getValue().str());
    return name;
  }
  GetResultOp get_result = op_result.getDefiningOp<GetResultOp>();
  Operation *producer;
  if (is_control) {
    producer = op_result.getDefiningOp();
  } else {
    if (!get_result)
      return InvalidArgument("Missing get_result operation as input");
    producer = get_result.value().getDefiningOp();
    if (!producer)
      return InvalidArgument("Expect a tfg operation as input to GetResultOp");
  }

  auto name_attr =
      producer->getAttrOfType<StringAttr>(TFGraphDialect::getNameAttrKey());
  if (!name_attr)
    return InvalidArgument("Can't export graph with missing op-name");

  std::string name;
  if (is_control) name = "^";
  absl::StrAppend(&name, name_attr.getValue().str());
  if (get_result)
    absl::StrAppend(&name, ":", get_result.name().str(), ":",
                    get_result.number());
  return name;
}

// Export a function argument or returned value as an ArgDef entry.
// If arg_def_attrs is provided, it is populated with the extra attributes
// converted from MLIR to AttrValue proto representation. This is useful only
// for Function arguments to populate the `arg_attr` field.
//
static Status ExportArgDef(OpDef::ArgDef *arg, DictionaryAttr arg_attrs,
                           FunctionDef::ArgAttrs *arg_def_attrs = nullptr) {
  StringAttr arg_name = arg_attrs.getAs<StringAttr>("tfg.name");
  if (!arg_name) return InvalidArgument("Missing \"tfg.name\" attribute");
  arg->set_name(arg_name.getValue().str());
  StringAttr description = arg_attrs.getAs<StringAttr>("tfg.description");
  if (description) arg->set_description(description.getValue().str());
  TypeAttr input_type = arg_attrs.getAs<TypeAttr>("tfg.type");
  if (input_type) {
    tensorflow::DataType dtype;
    TF_RETURN_IF_ERROR(ConvertToDataType(input_type.getValue(), &dtype));
    arg->set_type(dtype);
  }
  if (StringAttr type_attr = arg_attrs.getAs<StringAttr>("tfg.type_attr"))
    arg->set_type_attr(type_attr.getValue().str());
  if (StringAttr number_attr = arg_attrs.getAs<StringAttr>("tfg.number_attr"))
    arg->set_number_attr(number_attr.getValue().str());
  if (StringAttr type_list_attr =
          arg_attrs.getAs<StringAttr>("tfg.type_list_attr"))
    arg->set_type_attr(type_list_attr.getValue().str());
  if (auto full_type = arg_attrs.getAs<tf_type::FullTypeAttr>(
          "tfg.experimental_full_type")) {
    TF_ASSIGN_OR_RETURN(*arg->mutable_experimental_full_type(),
                        ConvertAttribute(full_type));
  }
  TF_RETURN_IF_ERROR(
      ConvertHandleData(arg_attrs.getAs<ArrayAttr>("tfg.handle_data"), arg));
  if (UnitAttr number_attr = arg_attrs.getAs<UnitAttr>("tfg.is_ref"))
    arg->set_is_ref(true);

  auto sig_arg_attrs = arg_attrs.getAs<DictionaryAttr>("tfg.arg_attrs");
  if (arg_def_attrs && sig_arg_attrs) {
    TF_RETURN_IF_ERROR(ConvertAttributes(
        sig_arg_attrs.getValue(), /*attrs_to_ignore=*/{},
        /*remove_ref_type=*/false, arg_def_attrs->mutable_attr()));
  }
  return Status::OK();
}

tensorflow::StatusOr<FunctionDef> ConvertGenericFunctionToFunctionDef(
    GraphFuncOp func_op) {
  if (!func_op.generic())
    return InvalidArgument(
        "Expected a generic function in ConvertGenericFunctionToFunctionDef");
  auto control_ty = tfg::ControlType::get(func_op.getContext());
  auto *tfg_dialect = cast<TFGraphDialect>(func_op->getDialect());

  FunctionDef fdef;
  for (Operation &op : func_op.getBody()->without_terminator()) {
    if (op.getDialect() != tfg_dialect)
      return InvalidArgument("Non tfg op encountered when exporting function");

    if (isa<GetResultOp>(&op)) continue;

    TF_RETURN_IF_ERROR(ConvertToNodeDef(
        &op, fdef.add_node_def(), tfg_dialect,
        [&](Value value) { return GetValueName(value, control_ty); }));
  }

  const std::string func_name = func_op.getName().str();
  OpDef *signature = fdef.mutable_signature();
  signature->set_name(func_name);
  if (func_op->getAttr("is_stateful")) signature->set_is_stateful(true);
  if (auto description = func_op->getAttrOfType<StringAttr>("description"))
    signature->set_description(description.getValue().str());

  if (auto attrs = func_op->getAttrOfType<DictionaryAttr>("tfg.func_attrs")) {
    for (NamedAttribute attr : attrs) {
      OpDef_AttrDef *func_attr = signature->add_attr();
      func_attr->set_name(attr.getName().str());
      DictionaryAttr dict_attr = attr.getValue().dyn_cast<DictionaryAttr>();
      if (!dict_attr) return InvalidArgument("Expects dict attribute");
      if (StringAttr type = dict_attr.getAs<StringAttr>("function_type"))
        func_attr->set_type(type.getValue().str());
      if (Attribute default_value = dict_attr.get("default_value")) {
        TF_ASSIGN_OR_RETURN((*func_attr->mutable_default_value()),
                            ConvertAttribute(default_value));
      }
      if (StringAttr description = dict_attr.getAs<StringAttr>("description"))
        func_attr->set_description(description.getValue().str());
      if (IntegerAttr minimum = dict_attr.getAs<IntegerAttr>("minimum")) {
        func_attr->set_minimum(minimum.getInt());
        func_attr->set_has_minimum(true);
      }
      if (Attribute allowed_values = dict_attr.get("allowed_values")) {
        TF_ASSIGN_OR_RETURN((*func_attr->mutable_allowed_values()),
                            ConvertAttribute(allowed_values));
      }
    }
  }

  if (auto control_outputs =
          func_op->getAttrOfType<ArrayAttr>("control_output")) {
    for (Attribute attr : control_outputs) {
      StringAttr output = attr.dyn_cast<StringAttr>();
      if (!output)
        return InvalidArgument(
            "Can't export function with non-string \"control_output\" "
            "attribute entry");
      signature->add_control_output(output.getValue().str());
    }
  }

  // Convert the function argument into an OpDef::ArgDef in the signature.
  ArrayAttr args_attr = func_op.getAllArgAttrs();
  for (int arg_num : llvm::seq<int>(0, func_op.getNumArguments())) {
    // Odd position are just for control dependencies.
    if (arg_num % 2) continue;
    OpDef::ArgDef *arg = signature->add_input_arg();
    if (arg_num >= args_attr.size())
      return InvalidArgument("Can't export function ", func_op.getName().str(),
                             " because missing attributes for arg #", arg_num);
    DictionaryAttr arg_attrs = args_attr[arg_num].cast<DictionaryAttr>();
    FunctionDef::ArgAttrs func_def_arg_attrs;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        ExportArgDef(arg, arg_attrs, &func_def_arg_attrs),
        " when exporting argument ", arg_num, " for function ",
        func_op.getName().str());

    // On top of the signature, function arguments can have attribute directul
    // on the FunctionDef.
    if (!func_def_arg_attrs.attr().empty())
      (*fdef.mutable_arg_attr())[arg_num / 2] = std::move(func_def_arg_attrs);
  }

  // Handle the results now.
  // An ArgDef entry needs to be constructed for all non-control returned value,
  // and a mapping from the output name to the signature is also recorded in the
  // FunctionDef.
  auto return_op =
      llvm::cast<tfg::ReturnOp>(func_op.getBody()->getTerminator());
  ArrayAttr results_attr = func_op.getAllResultAttrs();
  for (auto &indexed_result : llvm::enumerate(return_op->getOperands())) {
    int res_num = indexed_result.index();
    if (res_num >= results_attr.size())
      return InvalidArgument("Can't export function ", func_op.getName().str(),
                             " because missing attributes for result #",
                             res_num);
    auto res_attrs = results_attr[res_num].cast<DictionaryAttr>();
    auto name = res_attrs.getAs<StringAttr>("tfg.name");
    if (!name)
      return InvalidArgument(
          "Can't export function ", func_op.getName().str(),
          " because missing \"tfg.name\" attribute for result #", res_num);

    Value ret_val = indexed_result.value();
    if (ret_val.getType() == control_ty) {
      // When we return a control dependency, it is not really a returned value
      // but it is added to the `control_ret` field of the FunctionDef.
      TF_ASSIGN_OR_RETURN(std::string ret_name,
                          GetValueName(ret_val, control_ty));
      fdef.mutable_control_ret()->insert(
          {name.getValue().str(), StringRef(ret_name).drop_front().str()});
      continue;
    }
    // Tensor results are turned into an ArgDef in the `output_arg` field.
    OpDef::ArgDef *output = signature->add_output_arg();
    TF_RETURN_WITH_CONTEXT_IF_ERROR(ExportArgDef(output, res_attrs),
                                    " when exporting result ", res_num,
                                    " for function ", func_op.getName().str());

    // The `ret` field of the FunctionDef keeps a mapping of the returned value
    // name to the entried in the FunctionDef signature.
    TF_ASSIGN_OR_RETURN(std::string ret_name,
                        GetValueName(ret_val, control_ty));
    fdef.mutable_ret()->insert({name.getValue().str(), ret_name});
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

    auto *unique_ids_map = fdef.mutable_resource_arg_unique_id();
    for (auto key_value : llvm::zip(unique_ids_keys.getValues<int32_t>(),
                                    unique_ids_values.getValues<int32_t>()))
      (*unique_ids_map)[std::get<0>(key_value)] = std::get<1>(key_value);
  }

  // Finally the dialect attributes (prefixed by `tf.` in general) are converted
  // as-is and stored on the `attr` field of the FunctionDef.
  llvm::SmallVector<NamedAttribute, 8> funcAttrs(func_op->getDialectAttrs());
  TF_RETURN_IF_ERROR(ConvertAttributes(funcAttrs, {"tfg.func_attrs"},
                                       /*remove_ref_type=*/false,
                                       fdef.mutable_attr()));
  return fdef;
}

}  // namespace tfg
}  // namespace mlir

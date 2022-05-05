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

#include "tensorflow/core/ir/importexport/graphdef_export.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Threading.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/convert_attributes.h"
#include "tensorflow/core/ir/importexport/convert_types.h"
#include "tensorflow/core/ir/importexport/export.h"
#include "tensorflow/core/ir/importexport/functiondef_export.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stringpiece.h"

using tensorflow::AttrValue;
using tensorflow::DataType;
using tensorflow::FunctionDef;
using tensorflow::GradientDef;
using tensorflow::GraphDef;
using tensorflow::NodeDef;
using tensorflow::OpDef;
using tensorflow::OpRegistrationData;
using tensorflow::OpRegistry;
using tensorflow::Status;
using tensorflow::StatusOr;
using tensorflow::StringPiece;
using tensorflow::errors::InvalidArgument;

namespace mlir {
namespace tfg {
namespace {
// This class implements an exporter for TFG directly to GraphDef.
class GraphDefExporter {
 public:
  GraphDefExporter(TFGraphDialect *dialect, const OpRegistry &registry,
                   SymbolTable &table)
      : ctx_(dialect->getContext()),
        dialect_(dialect),
        registry_(registry),
        table_(table),
        builder_(ctx_) {}

  // Export a TFG module to GraphDef. The module may contain at most one GraphOp
  // and only GraphFuncOp otherwise.
  Status ExportToGraphDef(ModuleOp module, GraphDef *graph);

 private:
  // Export a TFG graph function to a FunctionDef. If the function has a
  // gradient, add it to the graph afterwards to preserve thread-safety.
  StatusOr<Optional<GradientDef>> ExportFunction(GraphFuncOp func,
                                                 FunctionDef *def);

  // Export just the input and outputs of a function signature. When
  // fully-qualifying result names, this must be done before any nodes are
  // Convert argument attributes to an ArgDef.
  StatusOr<OpDef::ArgDef> ConvertArgumentAttributes(DictionaryAttr attrs);

  // Convert a TFG op to a node. When converting a function, fully-qualified
  // result names must be used.
  Status ExportOp(Operation *op, NodeDef *node, bool is_func);

  // Get the name associated with a value.
  StatusOr<std::string> GetValueName(Value value, bool is_func);

  // Get the name and index of an output segment to fully qualify result names.
  // This requires querying the op registry.
  StatusOr<std::pair<StringRef, unsigned>> GetOutputSegment(OpResult result);

  // The current MLIR context.
  MLIRContext *ctx_;
  // The TFG dialect instance.
  TFGraphDialect *dialect_;
  // The TF op registry to use.
  const OpRegistry &registry_;
  // The symbol table.
  SymbolTable &table_;
  // The context dependent MLIR builder.
  Builder builder_;
};
}  // namespace

Status GraphDefExporter::ExportToGraphDef(ModuleOp module, GraphDef *graph) {
  TF_ASSIGN_OR_RETURN(GraphOp graph_op, ValidateModuleForExport(module));
  if (graph_op) {
    ExportVersionAttr(graph_op.version(), graph->mutable_versions());
    for (Operation &op : *graph_op.getBody()) {
      TF_RETURN_IF_ERROR(
          ExportOp(&op, graph->mutable_node()->Add(), /*is_func=*/false));
    }
  }

  const auto convert_func = [this](GraphFuncOp func, FunctionDef *def,
                                   Optional<GradientDef> &gradient) {
    // Generic functions are not on the hot path and skip the conversion to
    // Graph so just call the existing exporter.
    if (func.generic()) {
      TF_ASSIGN_OR_RETURN(*def, ConvertGenericFunctionToFunctionDef(func));
    } else {
      TF_ASSIGN_OR_RETURN(gradient, ExportFunction(func, def));
    }
    return Status::OK();
  };

  // TODO(jeffniu): Don't export functions in parallel if there are too few or
  // they are too small.
  if (ctx_->isMultithreadingEnabled()) {
    ctx_->enterMultiThreadedExecution();
    auto exit =
        llvm::make_scope_exit([this] { ctx_->exitMultiThreadedExecution(); });

    // Prepare the arguments to parallel for each.
    struct Argument {
      GraphFuncOp func;
      FunctionDef *def;
      Status status;
      Optional<GradientDef> gradient;
    };
    std::vector<Argument> args;
    for (auto func : module.getOps<GraphFuncOp>())
      args.push_back(Argument{func, graph->mutable_library()->add_function()});
    const auto process_func = [&convert_func](Argument &arg) {
      arg.status = convert_func(arg.func, arg.def, arg.gradient);
      return success(arg.status.ok());
    };

    // Execute the exports in parallel.
    if (failed(failableParallelForEach(ctx_, args, process_func))) {
      Status result;
      for (const Argument &arg : args) {
        result.Update(arg.status);
      }
      return result;
    }
  } else {
    for (auto func : module.getOps<GraphFuncOp>()) {
      Optional<GradientDef> gradient;
      TF_RETURN_IF_ERROR(convert_func(
          func, graph->mutable_library()->add_function(), gradient));
      if (gradient)
        *graph->mutable_library()->add_gradient() = std::move(*gradient);
    }
  }

  return Status::OK();
}

// The only dialect attributes allowed have the "tf." prefix. This is a slightly
// faster check that an attribute is a dialect attribute.
static bool IsDialectAttr(const NamedAttribute &attr) {
  return attr.getName().getValue().startswith("tf.");
}

// Export the given attribute list.
static Status ConvertAttributes(
    tensorflow::protobuf::Map<std::string, AttrValue> *map,
    ArrayRef<NamedAttribute> attrs) {
  for (const NamedAttribute &attr : attrs) {
    if (!IsDialectAttr(attr)) continue;
    StringRef name = attr.getName().strref().drop_front(/*strlen("tf.")=*/3);
    TF_ASSIGN_OR_RETURN((*map)[name.str()], ConvertAttribute(attr.getValue()));
  }
  return Status::OK();
}

StatusOr<Optional<GradientDef>> GraphDefExporter::ExportFunction(
    GraphFuncOp func, FunctionDef *def) {
  std::string func_name = func.sym_name().str();

  // TODO(jeffniu): Exploit the sorted order of the function attributes.

  // Get a gradient, if there is one.
  Optional<GradientDef> gradient;
  if (Optional<StringRef> gradient_name = func.gradient()) {
    gradient.emplace();
    gradient->set_gradient_func(gradient_name->str());
    gradient->set_function_name(func_name);
  }

  // Convert the first-class attributes.
  OpDef *signature = def->mutable_signature();
  signature->set_name(func_name);
  if (Optional<StringRef> description = func.description())
    signature->set_description(description->str());
  signature->set_is_stateful(func.is_stateful());

  if (DenseIntElementsAttr keys = func.resource_arg_unique_ids_keysAttr()) {
    DenseIntElementsAttr values = func.resource_arg_unique_ids_valuesAttr();
    if (!values) {
      return InvalidArgument(
          "'resource_arg_unique_ids_keys' is present but "
          "'resource_arg_unique_ids_values' is missing");
    }
    if (keys.size() != values.size()) {
      return InvalidArgument(
          "'resource_arg_unique_ids_keys' is not the same size as "
          "'resource_arg_unique_ids_values'");
    }
    auto *id_map = def->mutable_resource_arg_unique_id();
    for (auto kv :
         llvm::zip(keys.getValues<int32_t>(), values.getValues<int32_t>()))
      (*id_map)[std::get<0>(kv)] = std::get<1>(kv);
  }

  // Convert other attributes with the "tf." prefix.
  TF_RETURN_IF_ERROR(ConvertAttributes(def->mutable_attr(), func->getAttrs()));

  // Convert the arguments.
  for (int i = 0, e = func.getNumArguments(); i < e; i += 2) {
    auto attrs = func.arg_attrs()[i].cast<DictionaryAttr>();
    TF_ASSIGN_OR_RETURN(OpDef::ArgDef &arg = *signature->add_input_arg(),
                        ConvertArgumentAttributes(attrs));
    DataType dtype;
    TF_RETURN_IF_ERROR(ConvertToDataType(
        func.getArgument(i).getType().cast<TensorType>().getElementType(),
        &dtype));
    arg.set_type(dtype);
    // Convert the attributes.
    if (llvm::any_of(attrs, [](const NamedAttribute &attr) {
          return IsDialectAttr(attr);
        })) {
      auto *map = (*def->mutable_arg_attr())[i / 2].mutable_attr();
      TF_RETURN_IF_ERROR(ConvertAttributes(map, attrs.getValue()));
    }
  }

  // Convert the results.
  auto return_op = cast<ReturnOp>(func.getBody()->getTerminator());
  for (auto it :
       llvm::zip(func.getResultTypes(),
                 func.getAllResultAttrs().getAsRange<DictionaryAttr>(),
                 TFOp(return_op).getNonControlOperands())) {
    TF_ASSIGN_OR_RETURN(OpDef::ArgDef &arg = *signature->add_output_arg(),
                        ConvertArgumentAttributes(std::get<1>(it)));
    DataType dtype;
    TF_RETURN_IF_ERROR(ConvertToDataType(
        std::get<0>(it).cast<TensorType>().getElementType(), &dtype));
    arg.set_type(dtype);
    // Map the result.
    TF_ASSIGN_OR_RETURN((*def->mutable_ret())[arg.name()],
                        GetValueName(std::get<2>(it), /*is_func=*/true));
  }

  // Convert the control results.
  for (auto it :
       llvm::zip(return_op.control_ret_attrs().getAsRange<DictionaryAttr>(),
                 TFOp(return_op).getControlOperands())) {
    // The control result attributes contain only the name.
    DictionaryAttr attrs = std::get<0>(it);
    if (attrs.empty())
      return InvalidArgument("Control result is missing 'tfg.name'");
    assert(attrs.begin()->getName() == dialect_->getTfgNameAttrIdentifier());
    std::string name = attrs.begin()->getValue().cast<StringAttr>().str();
    signature->add_control_output(name);
    // Map the control result.
    TF_ASSIGN_OR_RETURN(std::string value_name,
                        GetValueName(std::get<1>(it), /*is_func=*/true));
    // Add the control result name without '^'.
    def->mutable_control_ret()->insert({std::move(name), value_name.substr(1)});
  }

  // Convert the body.
  for (Operation &op : func.getBody()->without_terminator())
    TF_RETURN_IF_ERROR(ExportOp(&op, def->add_node_def(), /*is_func=*/true));

  return gradient;
}

StatusOr<OpDef::ArgDef> GraphDefExporter::ConvertArgumentAttributes(
    DictionaryAttr attrs) {
  OpDef::ArgDef arg;

  // TODO(jeffniu): Exploit the sorted property of the attribute list.

  auto name = attrs.getAs<StringAttr>(dialect_->getTfgNameAttrIdentifier());
  if (!name) return InvalidArgument("argument is missing 'tfg.name'");
  arg.set_name(name.str());
  if (auto description =
          attrs.getAs<StringAttr>(dialect_->getTfgDescriptionAttrIdentifier()))
    arg.set_description(description.str());
  arg.set_is_ref(!!attrs.get(dialect_->getTfgIsRefAttrIdentifier()));
  TF_RETURN_IF_ERROR(tensorflow::ConvertHandleData(
      attrs.getAs<ArrayAttr>(dialect_->getTfgHandleDataAttrIdentifier()),
      &arg));
  if (auto full_type = attrs.getAs<tf_type::FullTypeAttr>(
          dialect_->getTfgFullTypeAttrIdentifier())) {
    TF_ASSIGN_OR_RETURN(*arg.mutable_experimental_full_type(),
                        ConvertAttribute(full_type));
  }
  return arg;
}

Status GraphDefExporter::ExportOp(Operation *op, NodeDef *node, bool is_func) {
  // TODO(jeffniu): Exploit the sorted property of the attribute list.

  // Convert first-class attributes.
  if (auto name =
          op->getAttrOfType<StringAttr>(dialect_->getNameAttrIdentifier()))
    node->set_name(name.str());
  if (auto device =
          op->getAttrOfType<StringAttr>(dialect_->getDeviceAttrIdentifier()))
    node->set_device(device.str());
  if (auto full_type = op->getAttrOfType<tf_type::FullTypeAttr>(
          dialect_->getFullTypeAttrIdentifier())) {
    TF_ASSIGN_OR_RETURN(*node->mutable_experimental_type(),
                        ConvertAttribute(full_type));
  }

  if (auto assigned_device = op->getAttrOfType<StringAttr>(
          dialect_->getAssignedDeviceAttrIdentifier())) {
    if (!assigned_device.getValue().empty()) {
      (*node->mutable_attr())[dialect_->getAssignedDeviceAttrKey().str()].set_s(
          assigned_device.str());
    }
  }

  const std::string op_name = op->getName().stripDialect().str();
  const OpRegistrationData *op_reg_data = registry_.LookUp(op_name);
  auto *attr_def = op_reg_data ? &op_reg_data->op_def.attr() : nullptr;

  // A map to track default values for the attribute.
  absl::flat_hash_map<StringPiece, tensorflow::AttrValue>
      attr_name_to_default_value;
  if (attr_def) {
    for (const tensorflow::OpDef_AttrDef &attr : *attr_def) {
      if (attr.has_default_value())
        attr_name_to_default_value[attr.name()] = attr.default_value();
    }
  }

  // Convert other attributes.
  for (const NamedAttribute &attr : op->getAttrs()) {
    if (attr.getName() == dialect_->getAssignedDeviceAttrIdentifier() ||
        attr.getName() == dialect_->getDeviceAttrIdentifier() ||
        attr.getName() == dialect_->getFullTypeAttrIdentifier() ||
        attr.getName() == dialect_->getNameAttrIdentifier())
      continue;

    const std::string attribute_name = attr.getName().str();
    auto attr_default_value = attr_name_to_default_value.find(attribute_name);
    if (attr_default_value != attr_name_to_default_value.end()) {
      TF_ASSIGN_OR_RETURN(Attribute mlir_default_value,
                          ConvertAttributeValue(attr_default_value->second,
                                                builder_, dialect_));
      // Do not export attribute with the value equal to the default value.
      if (mlir_default_value != attr.getValue()) {
        TF_ASSIGN_OR_RETURN(tensorflow::AttrValue converted_attr,
                            ConvertAttribute(attr.getValue()));
        (*node->mutable_attr())[attribute_name] = std::move(converted_attr);
      }
    } else {
      TF_ASSIGN_OR_RETURN(tensorflow::AttrValue converted_attr,
                          ConvertAttribute(attr.getValue()));
      (*node->mutable_attr())[attribute_name] = std::move(converted_attr);
    }
  }

  // Set the op name.
  node->set_op(op_name);

  // Set the input names.
  for (Value operand : op->getOperands()) {
    TF_ASSIGN_OR_RETURN(std::string input_name, GetValueName(operand, is_func));
    node->add_input(std::move(input_name));
  }

  // Export the location as debug info.
  if (!op->getLoc().isa<UnknownLoc>()) {
    ExtractExperimentalDebugInfoFromLocation(
        op->getLoc(), node->mutable_experimental_debug_info());
    if (node->experimental_debug_info().original_node_names().empty())
      node->clear_experimental_debug_info();
  }

  return Status::OK();
}

// TODO(jeffniu): This should be factored out as a utility. Also the results
// should be cached.
StatusOr<std::string> GraphDefExporter::GetValueName(Value value,
                                                     bool is_func) {
  std::string name;
  bool is_control = value.getType() == dialect_->getControlType();

  if (auto arg = value.dyn_cast<BlockArgument>()) {
    auto func = dyn_cast<GraphFuncOp>(arg.getOwner()->getParentOp());
    if (!func)
      return InvalidArgument("Expected block argument owner to be tfg.func");
    // If the block argument is a control token, use the attributes of the
    // associated data argument (which preceeds it).
    auto attrs = func.arg_attrs()[arg.getArgNumber() - is_control]
                     .cast<DictionaryAttr>();
    auto name_attr =
        attrs.getAs<StringAttr>(dialect_->getTfgNameAttrIdentifier());
    if (!name_attr) {
      return InvalidArgument(
          "Can't export graph with missing op-name for function parameter #",
          arg.getArgNumber());
    }
    name.reserve(name_attr.size() + 1);
    if (is_control) name.push_back('^');
    name.append(name_attr.data(), name_attr.size());
    return name;
  }

  auto result = value.cast<OpResult>();
  auto name_attr = result.getOwner()->getAttrOfType<StringAttr>(
      dialect_->getNameAttrIdentifier());
  if (!name_attr)
    return InvalidArgument("Can't export graph with missing op-name");

  if (is_control) {
    name.reserve(1 + name_attr.size());
    name.push_back('^');
    name.append(name_attr.data(), name_attr.size());
    return name;
  }

  if (!is_func) {
    name.reserve(name_attr.size() + 3);
    name.append(name_attr.data(), name_attr.size());
    if (result.getResultNumber()) {
      name.push_back(':');
      absl::StrAppend(&name, result.getResultNumber());
    }
    return name;
  }

  TF_ASSIGN_OR_RETURN(auto segment, GetOutputSegment(result));
  name.reserve(name_attr.size() + segment.first.size() + 4);
  name.append(name_attr.data(), name_attr.size());
  name.push_back(':');
  name.append(segment.first.data(), segment.first.size());
  name.push_back(':');
  absl::StrAppend(&name, segment.second);
  return name;
}

// Get the segment size of an op's output.
static StatusOr<unsigned> GetOutputSegmentSize(Operation *op,
                                               const OpDef::ArgDef &arg) {
  if (!arg.type_list_attr().empty()) {
    if (auto v = op->getAttr(arg.type_list_attr()).dyn_cast<ArrayAttr>())
      return v.size();
    return InvalidArgument("Type attr not found: ", arg.type_list_attr());
  }
  if (arg.number_attr().empty()) return 1;
  if (auto v = op->getAttr(arg.number_attr()).dyn_cast<IntegerAttr>())
    return v.getValue().getZExtValue();
  return InvalidArgument("Type attr not found: ", arg.number_attr());
}

StatusOr<std::pair<StringRef, unsigned>> GraphDefExporter::GetOutputSegment(
    OpResult result) {
  // TODO(jeffniu): OpRegistry::LookUp should accept `string_view`.
  Operation *op = result.getOwner();
  std::string op_name = op->getName().stripDialect().str();
  unsigned result_idx = result.getResultNumber();
  // Only edges in functions need to have fully-qualified names. Get the segment
  // name using the op definition.
  if (const OpRegistrationData *op_reg_data = registry_.LookUp(op_name)) {
    const OpDef &op_def = op_reg_data->op_def;

    for (const OpDef::ArgDef &arg : op_def.output_arg()) {
      TF_ASSIGN_OR_RETURN(unsigned size, GetOutputSegmentSize(op, arg));
      if (size > result_idx)
        return std::pair<StringRef, unsigned>(arg.name(), result_idx);
      result_idx -= size;
    }
    return InvalidArgument("Result #", result_idx, " of op '", op_name,
                           "' is out of range");
  }
  // Try to find a function for a legacy call. Function output segments have
  // exactly one element each.
  if (auto func = table_.lookup<GraphFuncOp>(op_name)) {
    if (result_idx >= func.getNumResults()) {
      return InvalidArgument("Result #", result_idx, " of op '", op_name,
                             "' is out of range");
    }
    if (auto name = func.getResultAttrOfType<StringAttr>(
            result_idx, dialect_->getTfgNameAttrIdentifier())) {
      return std::pair<StringRef, unsigned>(name.getValue(), 0);
    }
    return InvalidArgument("Function '", op_name, "' result #", result_idx,
                           "' is missing 'tfg.name'");
  }
  return InvalidArgument("Op '", op_name, "' is not registered");
}

// Convert a TFG graph directly to GraphDef.
tensorflow::Status ConvertToGraphDef(ModuleOp module,
                                     tensorflow::GraphDef *graph) {
  SymbolTable table(module);
  GraphDefExporter exporter(
      module.getContext()->getOrLoadDialect<TFGraphDialect>(),
      *OpRegistry::Global(), table);
  return exporter.ExportToGraphDef(module, graph);
}

}  // namespace tfg
}  // namespace mlir

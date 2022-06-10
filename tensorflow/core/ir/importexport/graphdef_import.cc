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

#include "tensorflow/core/ir/importexport/graphdef_import.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Threading.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/convert_attributes.h"
#include "tensorflow/core/ir/importexport/convert_types.h"
#include "tensorflow/core/ir/importexport/functiondef_import.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"

using tensorflow::DataType;
using tensorflow::DataTypeVector;
using tensorflow::FullTypeDef;
using tensorflow::FunctionDef;
using tensorflow::FunctionLibraryDefinition;
using tensorflow::Graph;
using tensorflow::GraphDebugInfo;
using tensorflow::GraphDef;
using tensorflow::NodeDef;
using tensorflow::OpDef;
using tensorflow::OpRegistrationData;
using tensorflow::OpRegistry;
using tensorflow::Status;
using tensorflow::StatusOr;
using tensorflow::StringPiece;
using tensorflow::TensorId;
using tensorflow::VersionDef;
using tensorflow::errors::InvalidArgument;
using tensorflow::errors::NotFound;

namespace mlir {
namespace tfg {
namespace {
// This class implements an importer for GraphDef directly to TFG.
class GraphDefImporter {
 public:
  // Initialize the importer.
  GraphDefImporter(TFGraphDialect *dialect, const OpRegistry &registry,
                   const GraphDebugInfo &debug_info)
      : ctx_(dialect->getContext()),
        dialect_(dialect),
        b_(ctx_),
        registry_(registry),
        debug_info_(debug_info),
        unknown_loc_(UnknownLoc::get(ctx_)),
        placeholder_state_(unknown_loc_, "tfg._mlir_placeholder") {
    placeholder_state_.addTypes(dialect_->getControlType());
  }

  // Convert a GraphDef to MLIR module.
  StatusOr<OwningOpRef<ModuleOp>> ConvertGraphDef(const GraphDef &graph);

 private:
  // Convert a function. This function must be thread-safe.
  Status ConvertFunctionDef(
      GraphFuncOp func_op,
      const absl::flat_hash_map<StringPiece, StringPiece> &gradient_map,
      const FunctionDef &function);

  // A result ID representing an output of `node`. E.g.
  // "foo" -> {0, "foo", ""}
  // "foo:2" -> {2, "foo", ""}
  // "foo:output:0" -> {0, "foo", "output"}
  struct ResultId {
    // The result or result segment index.
    int index;
    // The name of the parent node.
    StringRef node;
    // An optional result segment name.
    StringRef output;

    // Returns true if the result ID references the control token.
    bool IsControl() const { return index == tensorflow::Graph::kControlSlot; }
  };

  // An unresolved backedge.
  struct Backedge {
    // The edge name and index.
    ResultId id;
    // The OpOperand to resolve;
    OpOperand *operand;
  };

  // Cached info about the result of an operation.
  struct ResultInfo {
    // This flag is true if the results of the operation have been resolved; the
    // operation has been created and its `data` and `control` results have been
    // populated. If false, the placeholder should be used.
    bool resolved = false;
    // The control result.
    Value control;
    // All data results.
    ValueRange data;
    // Data results organized by output name.
    absl::flat_hash_map<StringPiece, ValueRange> outputs;
    // A list of unresolved backedges.
    std::vector<Backedge> backedges;
  };

  // State when converting a list of nodes.
  class ConversionState
      : public absl::flat_hash_map<StringPiece, std::unique_ptr<ResultInfo>> {
   public:
    // Create a conversion state with a placeholder value. Put the plaecholder
    // in the block so that it is owned.
    explicit ConversionState(Block *block,
                             const OperationState &placeholder_state)
        : placeholder_op_(
              OpBuilder::atBlockBegin(block).create(placeholder_state)),
          placeholder_(placeholder_op_->getResult(0)) {}

    // Get the placeholder value.
    Value GetPlaceholder() { return placeholder_; }

    // Finalize the conversion. The placeholder is destroyed.
    void Finalize() { placeholder_op_->erase(); }

   private:
    // The placeholder operation.
    Operation *placeholder_op_;
    // The placeholder value.
    Value placeholder_;
  };
  // Convert a list a nodes to operations.
  Status ConvertNodes(
      OpBuilder &builder, ConversionState &s,
      const tensorflow::protobuf::RepeatedPtrField<NodeDef> &nodes,
      Block *block);
  // Convert a node to an operation.
  Status ConvertNodeDef(OpBuilder &builder, ConversionState &s,
                        const NodeDef &node);
  // Resolve a data result reference.
  static StatusOr<Value> ResolveDataResult(const ResultId &id,
                                           ResultInfo *info);

  // Get a named result.
  struct Result {
    Value control;
    Value data;
    ResultId id;
    ResultInfo *info = nullptr;
  };
  StatusOr<Result> GetResult(ConversionState &s, StringPiece name);

  // Convert TF datatypes to unranked MLIR tensor types.
  Status ConvertDataTypesToUnrankedTensorTypes(const DataTypeVector &dtypes,
                                               SmallVectorImpl<Type> &results);
  // Extracts the actual data types from `attrs` based on its definition in
  // `arg_def` and converts them to unranked tensors. Returns the number of
  // added types.
  //
  // TODO(jeffniu): This is a re-implementation of `ArgNumType` in
  // `core/framework/function.cc` on `NamedAttrList` because the default
  // attributes need to be added. Find a way to do this in one pass.
  StatusOr<unsigned> ArgNumType(const NamedAttrList &attrs,
                                const OpDef::ArgDef &arg_def,
                                SmallVectorImpl<Type> &types);
  // Convert function attributes to MLIR attributes.
  Status ConvertFunctionAttributes(
      const absl::flat_hash_map<StringPiece, StringPiece> &gradient_map,
      const FunctionDef &function, GraphFuncOp op, NamedAttrList &attrs);
  // Convert function argument attributes to MLIR attributes.
  Status ConvertArgumentAttributes(const OpDef::ArgDef &def,
                                   NamedAttrList &attrs);
  // Create a location for a node.
  Location ConvertLocation(const NodeDef &node);
  // Convert the location of a node from the debug info. If it has no debug
  // info, return a NameLoc.
  Location ConvertLocation(StringRef node_name, StringRef func_name);

  // The MLIR context.
  MLIRContext *ctx_;
  // Reference to the TFG dialect.
  TFGraphDialect *dialect_;
  // The builder instance.
  Builder b_;
  // The TF op registry to use.
  const OpRegistry &registry_;
  // The debug info about the graph.
  const GraphDebugInfo &debug_info_;
  // Cached unknown location.
  Location unknown_loc_;
  // Operation state for creating placeholder ops.
  OperationState placeholder_state_;

  // Map of function OpDefs.
  absl::flat_hash_map<StringPiece, const OpDef *> function_op_defs_;
};
}  // namespace

// Convert a VersionDef to an MLIR version attribute.
static VersionAttr ConvertVersionAttr(MLIRContext *context,
                                      const VersionDef &version) {
  ArrayRef<int32_t> bad_consumers(version.bad_consumers().data(),
                                  version.bad_consumers().size());
  return VersionAttr::get(context, version.producer(), version.min_consumer(),
                          bad_consumers);
}

// Returns true if the function is a generic function, i.e. it contains
// placeholder attributes.
//
// TODO(jeffniu): Having to iterate over every function just to check for
// placeholder attributes is slow. Since most functions are not generic, we can
// speculate by converting all functions as non-generic until we see a
// placeholder attribute, bail out, and fall back to the generic function
// converter.
static bool IsGenericFunction(const FunctionDef &fdef) {
  for (const NodeDef &node : fdef.node_def())
    for (const auto &named_attr : node.attr())
      if (!named_attr.second.placeholder().empty()) return true;

  return false;
}

StatusOr<OwningOpRef<ModuleOp>> GraphDefImporter::ConvertGraphDef(
    const GraphDef &graph) {
  // Create the module.
  OwningOpRef<ModuleOp> module = ModuleOp::create(unknown_loc_);

  // Create the graph op.
  auto builder = OpBuilder::atBlockBegin(module->getBody());
  auto graph_op = builder.create<GraphOp>(
      module->getLoc(), ConvertVersionAttr(ctx_, graph.versions()));
  graph_op.nodes().push_back(new Block);

  // Populate the function op defs.
  function_op_defs_.reserve(graph.library().function_size());
  for (const FunctionDef &function : graph.library().function()) {
    function_op_defs_.emplace(function.signature().name(),
                              &function.signature());
  }

  // Build a map from function name to gradient function name.
  absl::flat_hash_map<StringPiece, StringPiece> gradient_map;
  gradient_map.reserve(graph.library().gradient_size());
  for (const tensorflow::GradientDef &gradient : graph.library().gradient())
    gradient_map.emplace(gradient.function_name(), gradient.gradient_func());

  // Convert the graph.
  ConversionState s(&graph_op.nodes().front(), placeholder_state_);
  TF_RETURN_IF_ERROR(
      ConvertNodes(builder, s, graph.node(), &graph_op.nodes().front()));

  // A function to convert a generic or non-generic function.
  const auto convert_func = [this, &gradient_map](GraphFuncOp func_op,
                                                  const FunctionDef &function) {
    if (IsGenericFunction(function)) {
      // Generic functions aren't on the hot path so just call the old
      // importer.
      OpBuilder builder(ctx_);
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          ConvertGenericFunction(func_op, function, builder),
          "While importing generic function: ", function.signature().name());
    } else {
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          ConvertFunctionDef(func_op, gradient_map, function),
          "While importing function: ", function.signature().name());
    }
    return ::tensorflow::OkStatus();
  };

  // TODO(jeffniu): Don't import functions in parallel if there are too few (how
  // few?) or if the functions are too small (how small?).
  if (ctx_->isMultithreadingEnabled()) {
    ctx_->enterMultiThreadedExecution();
    auto exit =
        llvm::make_scope_exit([this] { ctx_->exitMultiThreadedExecution(); });

    // Prepare the arguments to parallel for each.
    struct Argument {
      GraphFuncOp func;
      const FunctionDef &def;
      Status status;
    };
    std::vector<Argument> args;
    args.reserve(graph.library().function_size());
    for (const FunctionDef &function : graph.library().function()) {
      args.push_back(
          Argument{builder.create<GraphFuncOp>(unknown_loc_), function});
    }
    const auto process_func = [&convert_func](Argument &arg) {
      arg.status = convert_func(arg.func, arg.def);
      return success(arg.status.ok());
    };

    // Execute the imports in parallel.
    if (failed(failableParallelForEach(ctx_, args, process_func))) {
      Status result;
      for (const Argument &arg : args) {
        result.Update(arg.status);
      }
      return result;
    }
  } else {
    // Convert the functions.
    for (const FunctionDef &function : graph.library().function()) {
      auto func_op = builder.create<GraphFuncOp>(unknown_loc_);
      TF_RETURN_IF_ERROR(convert_func(func_op, function));
    }
  }

  return module;
}

Status GraphDefImporter::ConvertFunctionAttributes(
    const absl::flat_hash_map<StringPiece, StringPiece> &gradient_map,
    const FunctionDef &function, GraphFuncOp op, NamedAttrList &attrs) {
  // Import the function attributes with a `tf.` prefix to match the current
  // infratructure expectations.
  for (const auto &name_attr : function.attr()) {
    if (name_attr.first.empty()) {
      return InvalidArgument("Function ", function.signature().name(),
                             " has an empty attr name");
    }
    // TODO(b/230143351): `ConvertAttributeValue` is a little slow due to
    // `ConvertTensorProto` and `ConvertTensorShapeProto`.
    TF_ASSIGN_OR_RETURN(Attribute attr,
                        ConvertAttributeValue(name_attr.second, b_, dialect_));
    attrs.append(absl::StrCat("tf.", name_attr.first), attr);
  }

  // Convert the first-class attributes.
  const tensorflow::OpDef &signature = function.signature();
  if (signature.name().empty())
    return InvalidArgument("Function without a name");
  attrs.append(op.sym_nameAttrName(), b_.getStringAttr(signature.name()));

  if (!signature.description().empty()) {
    attrs.append(op.descriptionAttrName(),
                 b_.getStringAttr(signature.description()));
  }
  if (signature.is_stateful())
    attrs.append(op.is_statefulAttrName(), b_.getUnitAttr());
  auto grad_it = gradient_map.find(signature.name());
  if (grad_it != gradient_map.end()) {
    StringPiece name = grad_it->second;
    attrs.append(op.gradientAttrName(),
                 FlatSymbolRefAttr::get(ctx_, {name.data(), name.size()}));
  }

  // The resource_arg_unique_id is a list of `pair<int, int>`, we import it
  // as two arrays of integer right now.
  if (function.resource_arg_unique_id_size()) {
    SmallVector<int32_t> resource_arg_unique_ids_keys;
    SmallVector<int32_t> resource_arg_unique_ids_values;
    resource_arg_unique_ids_keys.reserve(
        function.resource_arg_unique_id_size());
    resource_arg_unique_ids_values.reserve(
        function.resource_arg_unique_id_size());
    for (const auto &unique_id : function.resource_arg_unique_id()) {
      resource_arg_unique_ids_keys.push_back(unique_id.first);
      resource_arg_unique_ids_values.push_back(unique_id.second);
    }
    attrs.append(op.resource_arg_unique_ids_keysAttrName(),
                 b_.getI32TensorAttr(resource_arg_unique_ids_keys));
    attrs.append(op.resource_arg_unique_ids_valuesAttrName(),
                 b_.getI32TensorAttr(resource_arg_unique_ids_values));
  }
  return ::tensorflow::OkStatus();
}

Status GraphDefImporter::ConvertArgumentAttributes(const OpDef::ArgDef &def,
                                                   NamedAttrList &attrs) {
  attrs.append(dialect_->getTfgNameAttrIdentifier(),
               b_.getStringAttr(def.name()));
  if (!def.description().empty()) {
    attrs.append(dialect_->getTfgDescriptionAttrIdentifier(),
                 b_.getStringAttr(def.description()));
  }
  if (def.is_ref())
    attrs.append(dialect_->getTfgIsRefAttrIdentifier(), b_.getUnitAttr());
  if (def.handle_data_size()) {
    TF_ASSIGN_OR_RETURN(Attribute handle_data,
                        ConvertHandleData(b_, def.handle_data()));
    attrs.append(dialect_->getTfgHandleDataAttrIdentifier(), handle_data);
  }
  if (def.has_experimental_full_type()) {
    TF_ASSIGN_OR_RETURN(
        tf_type::FullTypeAttr full_type,
        ConvertAttribute(def.experimental_full_type(), b_, dialect_));
    attrs.append(dialect_->getTfgFullTypeAttrIdentifier(), full_type);
  }
  return ::tensorflow::OkStatus();
}

Location GraphDefImporter::ConvertLocation(const NodeDef &node) {
  if (!node.has_experimental_debug_info()) return unknown_loc_;

  const auto &debug_info = node.experimental_debug_info();
  const auto &original_nodes = debug_info.original_node_names();
  const auto &original_funcs = debug_info.original_func_names();
  if (original_nodes.empty()) return unknown_loc_;

  SmallVector<Location> node_locs;
  node_locs.reserve(original_nodes.size());
  for (auto &it : llvm::enumerate(original_nodes)) {
    std::string func_name =
        it.index() < original_funcs.size() ? original_funcs[it.index()] : "";
    node_locs.push_back(ConvertLocation(it.value(), func_name));
  }
  return b_.getFusedLoc(node_locs);
}

// This is a re-implementation of GetLocation in `import.cc`.
Location GraphDefImporter::ConvertLocation(StringRef node_name,
                                           StringRef func_name) {
  // Concatenate the node name with the function name to match how the key is
  // formed in Python.
  std::string debug_info_key = (node_name + "@" + func_name).str();
  std::string name_loc = func_name.empty() ? node_name.str() : debug_info_key;
  auto name_loc_id = b_.getStringAttr(name_loc);

  SmallVector<Location> locs;
  const auto &traces = debug_info_.traces();
  // Try to find a stack trace to convert to locations.
  auto it = traces.find(debug_info_key);
  if (it != traces.end()) {
    const auto &trace = it->second;
    locs.reserve(trace.file_line_cols_size());
    for (const auto &loc : trace.file_line_cols()) {
      auto file_name = b_.getStringAttr(debug_info_.files(loc.file_index()));
      locs.push_back(FileLineColLoc::get(file_name, loc.line(), loc.col()));
    }
  }

  if (locs.empty()) return NameLoc::get(name_loc_id);

  // Use the first location to generate a name location.
  Location node_name_loc = NameLoc::get(name_loc_id, locs.front());
  // Generate a stack trace using the remaining locations.
  ArrayRef<Location> callsite_locs = llvm::makeArrayRef(locs).drop_front();
  return callsite_locs.empty() ? node_name_loc
                               : CallSiteLoc::get(node_name_loc, callsite_locs);
}

StatusOr<Value> GraphDefImporter::ResolveDataResult(const ResultId &id,
                                                    ResultInfo *info) {
  if (id.output.empty()) {
    if (id.index >= info->data.size()) {
      return InvalidArgument("Result #", id.index, " of node '", id.node.str(),
                             "' is out of bounds");
    }
    return info->data[id.index];
  }

  auto it = info->outputs.find({id.output.data(), id.output.size()});
  if (it == info->outputs.end()) {
    return InvalidArgument("Node '", id.node.str(), "' has no output called '",
                           id.output.str(), "'");
  }
  if (id.index >= it->second.size()) {
    return InvalidArgument("Result #", id.index, " of segment '", id.node.str(),
                           ":", id.output.str(), "' is out of bounds");
  }
  return it->second[id.index];
}

StatusOr<GraphDefImporter::Result> GraphDefImporter::GetResult(
    ConversionState &s, StringPiece name) {
  TensorId tensor_id = tensorflow::ParseTensorName(name);
  ResultId id{tensor_id.index()};
  std::tie(id.node, id.output) =
      StringRef(tensor_id.node().data(), tensor_id.node().size()).split(':');
  std::unique_ptr<ResultInfo> &info = s[{id.node.data(), id.node.size()}];
  if (!info) {
    info = std::make_unique<ResultInfo>();
  }

  // If the result is unresolved, return the placeholder;
  if (!info->resolved) {
    if (id.IsControl()) {
      return Result{s.GetPlaceholder(), Value(), id, info.get()};
    }
    return Result{Value(), s.GetPlaceholder(), id, info.get()};
  }

  // If the result is the control token, return it.
  if (id.IsControl()) {
    return Result{info->control, Value()};
  }

  TF_ASSIGN_OR_RETURN(Value value, ResolveDataResult(id, info.get()));
  return Result{Value(), value};
}

Status GraphDefImporter::ConvertFunctionDef(
    GraphFuncOp func_op,
    const absl::flat_hash_map<StringPiece, StringPiece> &gradient_map,
    const FunctionDef &function) {
  const OpDef &signature = function.signature();
  // TODO(jeffniu): Does the name need to be mangled?

  func_op.body().push_back(new Block);
  Block *body = &func_op.body().front();
  auto builder = OpBuilder::atBlockBegin(func_op.getBody());

  // Convert the attributes.
  NamedAttrList func_attrs;
  TF_RETURN_IF_ERROR(
      ConvertFunctionAttributes(gradient_map, function, func_op, func_attrs));

  SmallVector<Attribute> arg_attrs, res_attrs, control_ret_attrs;
  SmallVector<Type> arg_types, res_types;

  // Convert the arguments and argument attributes.
  for (auto &it : llvm::enumerate(signature.input_arg())) {
    Type dtype;
    TF_RETURN_IF_ERROR(ConvertDataType(it.value().type(), b_, &dtype));
    BlockArgument data =
        body->addArgument(UnrankedTensorType::get(dtype), unknown_loc_);
    BlockArgument ctl =
        body->addArgument(dialect_->getControlType(), data.getLoc());

    NamedAttrList attrs;
    TF_RETURN_IF_ERROR(ConvertArgumentAttributes(it.value(), attrs));
    auto attr_it = function.arg_attr().find(it.index());
    if (attr_it != function.arg_attr().end()) {
      for (const auto &name_attr : attr_it->second.attr()) {
        TF_ASSIGN_OR_RETURN(
            Attribute attr,
            ConvertAttributeValue(name_attr.second, b_, dialect_));
        attrs.append("tf." + name_attr.first, attr);
      }
    }

    arg_attrs.append({attrs.getDictionary(ctx_), b_.getDictionaryAttr({})});
    arg_types.append({data.getType(), ctl.getType()});
  }

  // Iterate over the arguments again and map them. We have to add them first
  // otherwise the ranges will be invalidated.
  ConversionState s(body, placeholder_state_);
  for (const auto &it : llvm::enumerate(signature.input_arg())) {
    s.emplace(
        it.value().name(),
        new ResultInfo{/*resolved=*/true, body->getArgument(it.index() * 2 + 1),
                       body->getArguments().slice(it.index() * 2, 1)});
  }
  TF_RETURN_IF_ERROR(ConvertNodes(builder, s, function.node_def(), body));

  // Convert the results and the result attributes.
  SmallVector<Value> return_operands;
  return_operands.reserve(signature.output_arg_size() +
                          signature.control_output_size());
  for (const OpDef::ArgDef &def : function.signature().output_arg()) {
    Type dtype;
    TF_RETURN_IF_ERROR(ConvertDataType(def.type(), b_, &dtype));
    NamedAttrList attrs;
    TF_RETURN_IF_ERROR(ConvertArgumentAttributes(def, attrs));
    res_attrs.push_back(attrs.getDictionary(ctx_));
    res_types.push_back(UnrankedTensorType::get(dtype));

    auto ret_it = function.ret().find(def.name());
    if (ret_it == function.ret().end()) {
      return InvalidArgument("Output '", def.name(),
                             "' was not found in 'ret'");
    }
    TF_ASSIGN_OR_RETURN(Result result, GetResult(s, ret_it->second));
    if (result.info)
      return InvalidArgument("Return '", ret_it->second, "' was not found");
    if (result.control)
      return InvalidArgument("Unexpected control result: ", ret_it->second);
    return_operands.push_back(result.data);
  }

  // Convert the control results.
  for (const std::string &control_ret : signature.control_output()) {
    auto ret_it = function.control_ret().find(control_ret);
    if (ret_it == function.control_ret().end()) {
      return InvalidArgument("Control output '", control_ret,
                             "' was not found in 'control_ret'");
    }
    std::unique_ptr<ResultInfo> &result = s[ret_it->second];
    if (!result || !result->resolved) {
      return InvalidArgument("Control return ", ret_it->second,
                             " was not found");
    }
    return_operands.push_back(result->control);
    control_ret_attrs.push_back(b_.getDictionaryAttr(NamedAttribute(
        dialect_->getTfgNameAttrIdentifier(), b_.getStringAttr(control_ret))));
  }
  builder.create<ReturnOp>(unknown_loc_, return_operands,
                           b_.getArrayAttr(control_ret_attrs));

  // Finalize the function attributes.
  func_attrs.append(func_op.arg_attrsAttrName(), b_.getArrayAttr(arg_attrs));
  func_attrs.append(func_op.res_attrsAttrName(), b_.getArrayAttr(res_attrs));
  func_attrs.append(func_op.function_typeAttrName(),
                    TypeAttr::get(b_.getFunctionType(arg_types, res_types)));
  func_op->setAttrs(func_attrs.getDictionary(ctx_));

  return ::tensorflow::OkStatus();
}

Status GraphDefImporter::ConvertNodes(
    OpBuilder &builder, ConversionState &s,
    const tensorflow::protobuf::RepeatedPtrField<NodeDef> &nodes,
    Block *block) {
  OpBuilder::InsertionGuard ig(builder);
  builder.setInsertionPointToStart(block);
  for (const NodeDef &node : nodes) {
    TF_RETURN_IF_ERROR(ConvertNodeDef(builder, s, node));
  }

  // If the placeholder has remaining uses, then an input is missing.
  if (TF_PREDICT_FALSE(!s.GetPlaceholder().use_empty())) {
    // Stringify a result ID.
    const auto id_to_str = [](const ResultId &id) {
      std::string name = id.node.str();
      if (id.IsControl()) return absl::StrCat("^", name);
      if (id.output.empty())
        return id.index ? absl::StrCat(id.node.str(), ":", id.index) : name;
      return absl::StrCat(name, ":", id.output.str(), ":", id.index);
    };
    // Gather all missing input edges.
    std::vector<std::pair<std::string, std::string>> missing_edges;
    for (const ResultInfo &info :
         llvm::make_pointee_range(llvm::make_second_range(s))) {
      if (info.backedges.empty()) continue;
      const Backedge &edge = info.backedges.front();
      missing_edges.emplace_back(id_to_str(edge.id),
                                 TFOp(edge.operand->getOwner()).name().str());
    }
    assert(!missing_edges.empty() &&
           "placeholder had remaining uses but found no unresolved backedges");
    // Destroy the invalid IR.
    block->erase();
    // Report the missing edges in alphabetical order.
    llvm::sort(missing_edges);
    std::string error_message;
    llvm::raw_string_ostream os(error_message);
    llvm::interleave(
        missing_edges, os,
        [&](const auto &edge) {
          os << "Non-existent input " << edge.first << " in node "
             << edge.second;
        },
        "\n");
    return InvalidArgument(std::move(os.str()));
  }
  // The placeholder has no uses and should not acquire any more uses. Safely
  // delete it from the IR.
  s.Finalize();

  return ::tensorflow::OkStatus();
}

StatusOr<unsigned> GraphDefImporter::ArgNumType(const NamedAttrList &attrs,
                                                const OpDef::ArgDef &arg_def,
                                                SmallVectorImpl<Type> &types) {
  // Check whether a type list attribute is specified.
  if (!arg_def.type_list_attr().empty()) {
    if (auto v =
            attrs.get(arg_def.type_list_attr()).dyn_cast_or_null<ArrayAttr>()) {
      for (Attribute attr : v) {
        if (auto dtype = attr.dyn_cast<TypeAttr>()) {
          types.push_back(UnrankedTensorType::get(dtype.getValue()));
        } else {
          return InvalidArgument("Expected '", arg_def.type_list_attr(),
                                 "' to be a list of types");
        }
      }
      return v.size();
    }
    return NotFound("Type attr not found: ", arg_def.type_list_attr());
  }

  unsigned num = 1;
  // Check whether a number attribute is specified.
  if (!arg_def.number_attr().empty()) {
    if (auto v =
            attrs.get(arg_def.number_attr()).dyn_cast_or_null<IntegerAttr>()) {
      num = v.getValue().getZExtValue();
    } else {
      return NotFound("Type attr not found: ", arg_def.number_attr());
    }
  }

  // Check for a type or type attribute.
  Type dtype;
  if (arg_def.type() != DataType::DT_INVALID) {
    TF_RETURN_IF_ERROR(ConvertDataType(arg_def.type(), b_, &dtype));
  } else if (arg_def.type_attr().empty()) {
    return InvalidArgument("Arg '", arg_def.name(),
                           "' has invalid type and no type attribute");
  } else {
    if (auto v = attrs.get(arg_def.type_attr()).dyn_cast_or_null<TypeAttr>()) {
      dtype = v.getValue();
    } else {
      return NotFound("Type attr not found: ", arg_def.type_attr());
    }
  }
  types.append(num, UnrankedTensorType::get(dtype));
  return num;
}

Status GraphDefImporter::ConvertNodeDef(OpBuilder &builder, ConversionState &s,
                                        const NodeDef &node) {
  VLOG(4) << "Importing: " << node.name();
  if (node.op().empty())
    return InvalidArgument("Node ", node.name(), " has an empty op name");

  OperationState state(ConvertLocation(node), absl::StrCat("tfg.", node.op()));

  // The GraphImporter does light shape inference, but here we will defer all of
  // that to the shape inference pass.
  const OpDef *op_def;
  const OpRegistrationData *op_reg_data = nullptr;
  if ((op_reg_data = registry_.LookUp(node.op()))) {
    op_def = &op_reg_data->op_def;
  } else {
    auto it = function_op_defs_.find(node.op());
    if (it == function_op_defs_.end())
      return InvalidArgument("Unable to find OpDef for ", node.op());
    op_def = it->second;
  }

  // Import the attributes. Reserve `+3` for `device`,`name`, and `fulltype`.
  state.attributes.reserve(node.attr_size() + 3);
  if (!node.device().empty()) {
    state.addAttribute(dialect_->getDeviceAttrIdentifier(),
                       b_.getStringAttr(node.device()));
  }
  if (!node.name().empty()) {
    state.addAttribute(dialect_->getNameAttrIdentifier(),
                       b_.getStringAttr(node.name()));
  }

  // If the op doesn't have a FullType, try to infer one.
  const auto add_full_type = [&](const FullTypeDef &full_type_def) {
    TF_ASSIGN_OR_RETURN(tf_type::FullTypeAttr full_type,
                        ConvertAttribute(full_type_def, b_, dialect_));
    state.addAttribute(dialect_->getFullTypeAttrIdentifier(), full_type);
    return ::tensorflow::OkStatus();
  };
  if (node.has_experimental_type()) {
    TF_RETURN_IF_ERROR(add_full_type(node.experimental_type()));
  } else if (op_reg_data && op_reg_data->type_ctor) {
    FullTypeDef full_type_def;
    TF_RETURN_IF_ERROR(
        tensorflow::full_type::SpecializeType(node, *op_def, full_type_def));
    TF_RETURN_IF_ERROR(add_full_type(full_type_def));
  }

  for (auto &name_attr : node.attr()) {
    if (name_attr.first.empty())
      return InvalidArgument("Node ", node.name(), " has an empty attr name");
    TF_ASSIGN_OR_RETURN(Attribute attr,
                        ConvertAttributeValue(name_attr.second, b_, dialect_));
    state.addAttribute(name_attr.first, attr);
  }

  // Add missing default attributes.
  for (const auto &attr_def : op_def->attr()) {
    if (attr_def.has_default_value() &&
        !state.attributes.get(attr_def.name())) {
      TF_ASSIGN_OR_RETURN(
          Attribute attr,
          ConvertAttributeValue(attr_def.default_value(), b_, dialect_));
      state.addAttribute(attr_def.name(), attr);
    }
  }

  // Get the result types. Ops can have multiple named results. Track the
  // segment sizes.
  SmallVector<std::pair<unsigned, unsigned>> result_segments;
  result_segments.reserve(op_def->output_arg_size());
  state.types.reserve(op_def->output_arg_size() + 1);
  for (const OpDef::ArgDef &def : op_def->output_arg()) {
    unsigned index = state.types.size();
    TF_ASSIGN_OR_RETURN(unsigned size,
                        ArgNumType(state.attributes, def, state.types));
    result_segments.emplace_back(index, size);
  }
  state.types.push_back(dialect_->getControlType());

  // Collect the operands. Set backedges to a placeholder and resolve them
  // later.
  state.operands.reserve(node.input_size());
  SmallVector<Value> control_operands;
  struct BackedgeResolution {
    ResultInfo *info;
    size_t operand_index;
    ResultId id;
  };
  SmallVector<BackedgeResolution> unresolved_data_operands,
      unresolved_control_operands;
  for (const std::string &input : node.input()) {
    TF_ASSIGN_OR_RETURN(Result result, GetResult(s, input));
    if (result.control) {
      if (result.info) {
        unresolved_control_operands.push_back(BackedgeResolution{
            result.info, control_operands.size(), result.id});
      }
      control_operands.push_back(result.control);
    } else {
      if (result.info) {
        unresolved_data_operands.push_back(
            BackedgeResolution{result.info, state.operands.size(), result.id});
      }
      state.operands.push_back(result.data);
    }
  }
  unsigned num_data_operands = state.operands.size();
  state.addOperands(control_operands);

  // Create the op and record any unresolved operands.
  Operation *op = builder.create(state);
  for (const BackedgeResolution &r : unresolved_data_operands) {
    r.info->backedges.push_back(
        Backedge{r.id, &op->getOpOperand(r.operand_index)});
  }
  for (const BackedgeResolution &r : unresolved_control_operands) {
    r.info->backedges.push_back(
        Backedge{r.id, &op->getOpOperand(num_data_operands + r.operand_index)});
  }

  std::unique_ptr<ResultInfo> &info = s[node.name()];
  if (!info) {
    info = std::make_unique<ResultInfo>();
  }
  info->resolved = true;
  info->control = *std::prev(op->result_end());
  info->data = op->getResults().drop_back();
  for (auto it : llvm::zip(result_segments, op_def->output_arg())) {
    const std::pair<unsigned, unsigned> &segment = std::get<0>(it);
    info->outputs.emplace(std::get<1>(it).name(),
                          info->data.slice(segment.first, segment.second));
  }

  // Resolve any associated backedges.
  for (const Backedge &backedge : info->backedges) {
    Value value;
    if (backedge.id.IsControl()) {
      value = info->control;
    } else {
      TF_ASSIGN_OR_RETURN(value, ResolveDataResult(backedge.id, info.get()));
    }
    backedge.operand->set(value);
  }
  info->backedges.clear();

  return ::tensorflow::OkStatus();
}

Status GraphDefImporter::ConvertDataTypesToUnrankedTensorTypes(
    const DataTypeVector &dtypes, SmallVectorImpl<Type> &results) {
  Type dtype;
  for (DataType tf_dtype : dtypes) {
    TF_RETURN_IF_ERROR(ConvertDataType(tf_dtype, b_, &dtype));
    results.push_back(UnrankedTensorType::get(dtype));
  }
  return ::tensorflow::OkStatus();
}

StatusOr<OwningOpRef<ModuleOp>> ImportGraphDef(MLIRContext *context,
                                               const GraphDebugInfo &debug_info,
                                               const GraphDef &graph_def) {
  GraphDefImporter importer(context->getOrLoadDialect<TFGraphDialect>(),
                            *OpRegistry::Global(), debug_info);
  return importer.ConvertGraphDef(graph_def);
}

StatusOr<OwningOpRef<ModuleOp>> ImportGraphAndFunctionsToMlir(
    MLIRContext *context, const GraphDebugInfo &debug_info, const Graph &graph,
    const FunctionLibraryDefinition &flib_def) {
  // TODO(b/231723721): This conversion path is slow because both the graph and
  // the function library are converted to GraphDef.
  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  *graph_def.mutable_library() = flib_def.ToProto();
  return ImportGraphDef(context, debug_info, graph_def);
}

}  // namespace tfg
}  // namespace mlir

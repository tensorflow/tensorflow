/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Identifier.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassManager.h"  // TF:llvm-project
#include "mlir/Support/DebugStringHelper.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/control_flow_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/export_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace mlir {
/// Create a pass to convert from the TFExecutor to the TF control dialect.
std::unique_ptr<OpPassBase<FuncOp>>
CreateTFExecutorToControlDialectConversion();
}  // namespace mlir

namespace tensorflow {
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using mlir::BlockArgument;
using mlir::Dialect;
using mlir::Operation;
using mlir::OperationState;
using mlir::Value;
using stream_executor::port::StatusOr;

namespace {

bool IsLegalChar(char c, bool first_char) {
  if (isalpha(c)) return true;
  if (isdigit(c)) return true;
  if (c == '.') return true;
  if (c == '_') return true;

  // First character of a node name can only be a letter, digit, dot or
  // underscore.
  if (first_char) return false;

  if (c == '/') return true;
  if (c == '-') return true;

  return false;
}

// Convert characters in name that are considered illegal in TensorFlow Node
// name to '.'.
std::string LegalizeNodeName(llvm::StringRef name) {
  assert(!name.empty() && "expected non-empty name");

  std::string legalized_name;
  for (auto it = name.begin(); it != name.end(); ++it) {
    if (IsLegalChar(*it, it == name.begin())) {
      legalized_name += *it;
    } else {
      legalized_name += '.';
    }
  }

  return legalized_name;
}

llvm::StringRef GetNameFromLoc(mlir::Location loc,
                               llvm::StringRef default_name) {
  if (auto name_loc = loc.dyn_cast<mlir::NameLoc>()) {
    return name_loc.getName().strref().split('@').first;
  } else if (auto call_loc = loc.dyn_cast<mlir::CallSiteLoc>()) {
    // Return name if CallSiteLoc's callee has a NameLoc (as should be the case
    // if imported with DebugInfo), else use the fallback naming scheme below.
    if (auto name_loc = call_loc.getCallee().dyn_cast<mlir::NameLoc>())
      return name_loc.getName().strref().split('@').first;
  } else if (auto fused_loc = loc.dyn_cast<mlir::FusedLoc>()) {
    // According to the importer, the last location of a fused location is
    // the name from the node_def and the rests are from the experimental debug
    // info.
    return GetNameFromLoc(fused_loc.getLocations().back(), default_name);
  }
  return default_name;
}

// TODO(jpienaar): unify and move from here to be able to reuse with tflite
std::string GetName(Operation* inst) {
  // Default name is Operation type.
  auto name = GetNameFromLoc(inst->getLoc(), inst->getName().getStringRef());
  return LegalizeNodeName(name);
}

// Stateful helper class to export a function into a Graph.
class Exporter {
 public:
  // Converts the given Module to a Graph. The given module should only contain
  // one entry function, which is identified by name "main". This entry function
  // is converted to the base of the graph graph. The rest of the functions are
  // converted to the library functions in that graph.
  static Status Convert(mlir::ModuleOp module, const GraphExportConfig& configs,
                        std::unique_ptr<Graph>* graph,
                        FunctionLibraryDefinition* flib_def);

  // Converts a given FuncOp to a FunctionDef and adds it to the function
  // definition library
  static Status ConvertLibFunction(const GraphExportConfig& configs,
                                   const Dialect* tf_dialect,
                                   mlir::FuncOp function,
                                   FunctionDefLibrary* flib);
  // Converts the given FuncOp to a Graph. The arguments and returns of
  // function are added to the graph with special op names kArgOp and kRetOp.
  // Later on, this graph can be converted a function definition and added to
  // another graph.
  static StatusOr<std::unique_ptr<Graph>> Convert(
      const GraphExportConfig& configs, const Dialect* tf_dialect,
      mlir::FuncOp function, FunctionDefLibrary* flib);

 private:
  explicit Exporter(Graph* graph, const Dialect* tf_dialect)
      : graph_(graph), tf_dialect_(tf_dialect) {}

  Status AddArgumentNode(BlockArgument arg, unsigned index,
                         llvm::StringRef name);
  Status AddReturnNode(mlir::ReturnOp op,
                       llvm::ArrayRef<llvm::StringRef> names);
  Status AddInstructionNode(Operation* inst);
  Status AddNextIterationNode(Operation* inst);
  Status AddEdge(Operation* inst);

  StatusOr<std::unique_ptr<NodeDef>> GetArgumentNode(BlockArgument arg,
                                                     unsigned index,
                                                     llvm::StringRef name);
  StatusOr<std::unique_ptr<NodeDef>> GetReturnNode(Operation* inst,
                                                   unsigned index,
                                                   llvm::StringRef name);
  // Adds one edge between src_node and dst_node. If it is not a control edge,
  // an index is used to find out the right operand of the dst_node.
  Status AddEdgeBetweenNodes(Value src, Node* dst_node, unsigned dst_index);

  // Returns a unique name for `op`.
  std::string UniqueName(Operation* op);

  // Returns a unique name starting with a given prefix.
  std::string UniqueName(llvm::StringRef prefix);

  Graph* graph_;
  absl::flat_hash_map<Operation*, string> op_to_name_;
  absl::flat_hash_map<string, int64> name_to_count_;
  absl::flat_hash_map<Operation*, Node*> nodes_;
  llvm::DenseMap<BlockArgument, Node*> args_;
  // One single return operation can return multiple results, and each of them
  // will be converted to one node in the graph.
  typedef absl::InlinedVector<Node*, 4> NodeVector;
  absl::flat_hash_map<Operation*, NodeVector> returns_;

  // Each NextIteration node in the original graph is converted to a pair of
  // source and sink operations in the MLIR, and we use the following two maps
  // to pair and convert them back to a single NextIteration node. We choose to
  // the "name" attribute, which is from the unique node name, to find out the
  // pairs: When scanning the operations in the block, the source operations
  // are inserted to the name_to_inst_ first, and the other "sink" operation
  // can be paired by checking this map and both are inserted to the
  // source_to_sink_ map.
  absl::flat_hash_map<string, Operation*> name_to_inst_;
  absl::flat_hash_map<Operation*, Operation*> source_to_sink_;

  const mlir::Dialect* tf_dialect_;
};

std::string Exporter::UniqueName(llvm::StringRef prefix) {
  // Keep incrementing the counter until we find a unique name.
  std::string name = prefix;
  int64& prefix_count = name_to_count_[name];
  int64 val = prefix_count;
  while (val != 0) {
    name = (prefix + llvm::Twine(prefix_count)).str();
    ++prefix_count;
    val = name_to_count_[name];
  }
  name_to_count_[name] = 1;
  return name;
}

std::string Exporter::UniqueName(Operation* op) {
  auto& name = op_to_name_[op];
  if (!name.empty()) return name;
  name = UniqueName(GetName(op));
  return name;
}

StatusOr<std::unique_ptr<NodeDef>> Exporter::GetArgumentNode(
    BlockArgument arg, unsigned index, llvm::StringRef name) {
  auto func = arg->getParentRegion()->getParentOfType<mlir::FuncOp>();

  auto node_def = absl::make_unique<NodeDef>();
  if (!name.empty())
    node_def->set_name(name.str());
  else
    node_def->set_name(UniqueName(func.getName().str()));

  node_def->set_op(FunctionLibraryDefinition::kArgOp);

  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(
      arg->getType().cast<mlir::TensorType>().getElementType(), &dtype));
  AttrValue type_attr;
  type_attr.set_type(dtype);
  (*node_def->mutable_attr())["T"] = type_attr;

  AttrValue index_attr;
  index_attr.set_i(index);
  (*node_def->mutable_attr())["index"] = index_attr;

  if (auto device_attr =
          func.getArgAttrOfType<mlir::StringAttr>(index, "tf.device")) {
    *node_def->mutable_device() = device_attr.getValue().str();
  }

  if (auto resource_arg_unique_id_attr =
          func.getArgAttrOfType<mlir::IntegerAttr>(
              index, "tf.resource_arg_unique_id")) {
    AttrValue unique_id_attr;
    unique_id_attr.set_i(resource_arg_unique_id_attr.getInt());
    (*node_def->mutable_attr())["_resource_arg_unique_id"] = unique_id_attr;
  }

  return node_def;
}

StatusOr<std::unique_ptr<NodeDef>> Exporter::GetReturnNode(
    Operation* inst, unsigned index, llvm::StringRef name) {
  auto node_def = absl::make_unique<NodeDef>();
  if (!name.empty())
    node_def->set_name(name.str());
  else
    node_def->set_name(
        UniqueName(inst->getParentOfType<mlir::FuncOp>().getName().str()));

  node_def->set_op(FunctionLibraryDefinition::kRetOp);
  auto inst_op = inst->getOperand(index);
  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(
      inst_op->getType().cast<mlir::TensorType>().getElementType(), &dtype));
  AttrValue type_attr;
  type_attr.set_type(dtype);
  (*node_def->mutable_attr())["T"] = type_attr;
  AttrValue index_attr;
  index_attr.set_i(index);
  (*node_def->mutable_attr())["index"] = index_attr;
  return node_def;
}

Status Exporter::AddEdgeBetweenNodes(Value src, Node* dst_node,
                                     unsigned dst_index) {
  if (auto input_result = src->dyn_cast<mlir::OpResult>()) {
    auto* input_inst = input_result->getOwner();
    // replaces the input node by the sink one if it is an NextIteration source:
    auto it = source_to_sink_.find(input_inst);
    if (it != source_to_sink_.end()) {
      input_inst = source_to_sink_[input_inst];
    }
    auto node_it = nodes_.find(input_inst);
    TF_RET_CHECK(node_it != nodes_.end())
        << "Use of OpResult encountered before def!";
    if (input_result->getType().isa<mlir::TFControlFlow::TFControlType>()) {
      graph_->AddControlEdge(node_it->second, dst_node);
    } else {
      graph_->AddEdge(node_it->second, input_result->getResultNumber(),
                      dst_node, dst_index);
    }
    return Status::OK();
  }

  auto input_arg = src->cast<BlockArgument>();
  auto input_node_it = args_.find(input_arg);
  TF_RET_CHECK(input_node_it != args_.end())
      << "Use of BlockArgument encounted before def!";
  // For argument, there is only one result output, so the index is always 0.
  graph_->AddEdge(input_node_it->second, 0, dst_node, dst_index);
  return Status::OK();
}

Status Exporter::AddEdge(Operation* inst) {
  auto* dst_node = nodes_[inst];
  bool is_return_op = isa<mlir::ReturnOp>(inst);
  for (int index = 0, e = inst->getNumOperands(); index < e; index++) {
    auto src = inst->getOperand(index);
    // For return operation, the edge is from the operand owner to one of the
    // faked return nodes. The input index is always 0 for the return node.
    if (is_return_op) {
      dst_node = returns_[inst][index];
      TF_RETURN_IF_ERROR(AddEdgeBetweenNodes(src, dst_node, 0));
    } else {
      // Assume the TF_Control input is always at the end, so the last index
      // value is passed into the function but not used.
      TF_RETURN_IF_ERROR(AddEdgeBetweenNodes(src, dst_node, index));
    }
  }
  return Status::OK();
}

Status Exporter::AddInstructionNode(Operation* inst) {
  Status status;

  if (inst->isKnownTerminator())
    return errors::InvalidArgument("std.return is only allowed terminator");

  std::unique_ptr<NodeDef> node_def;
  auto name = UniqueName(inst);
  // Convert registered TF ops to NodeDef. Only registered ops are handled to
  // ensure that PopulateDerivedAttrs adds the correct attributes.
  TF_ASSIGN_OR_RETURN(node_def,
                      ConvertTFDialectOpToNodeDef(
                          inst, name, /*ignore_unregistered_attrs=*/false));

  Node* node = graph_->AddNode(*node_def, &status);
  TF_RETURN_IF_ERROR(status);
  nodes_[inst] = node;
  return Status::OK();
}

bool IsEntryFunctionArg(BlockArgument arg) {
  return arg->getParentRegion()->getParentOfType<mlir::FuncOp>().getName() ==
         "main";
}

// Creates argument nodes from Block argument. If a name is supplied, that
// name will be used instead of generating a unique name.
Status Exporter::AddArgumentNode(BlockArgument arg, unsigned index,
                                 llvm::StringRef name) {
  if (!IsEntryFunctionArg(arg) || !name.empty()) {
    TF_ASSIGN_OR_RETURN(auto node_def, GetArgumentNode(arg, index, name));
    Status status;
    Node* node = graph_->AddNode(*node_def, &status);
    TF_RETURN_IF_ERROR(status);
    args_[arg] = node;
    return status;
  }

  // If it is an argument from the "main" function, it has only one user, which
  // is an input node. We recover the original input node and skip adding the
  // argument node. The new input node will be handled as normal in the
  // following steps.
  if (!arg->hasOneUse()) {
    return errors::FailedPrecondition(
        "Arg in 'main' should only have one user.");
  }
  auto* input = *arg->user_begin();
  auto input_name = input->getName().getStringRef();
  input_name.consume_back(".input");
  mlir::OpBuilder builder(arg->getOwner());
  auto loc = mlir::NameLoc::get(builder.getIdentifier(UniqueName(input)),
                                builder.getContext());
  OperationState state(loc, input_name.str());
  state.attributes.append(input->getAttrs().begin(), input->getAttrs().end());
  for (auto op : input->getOperands()) {
    // Skip the argument in the new operation.
    if (op->isa<BlockArgument>()) continue;
    state.operands.push_back(op);
  }
  state.types.append(input->getResultTypes().begin(),
                     input->getResultTypes().end());
  auto* inst = builder.createOperation(state);
  // If it is one of the specified input names, then the new
  // instruction should have the same name.
  auto& mapped_name = op_to_name_[inst];
  const auto& input_mapped_name = op_to_name_[input];
  DCHECK(mapped_name.empty())
      << "AddArgumentNode() attempted to change the op_to_name_ mapping for "
      << inst << " from " << mapped_name << " to " << input_mapped_name << ".";
  DCHECK(!input_mapped_name.empty())
      << "AddArgumentNode() attempted to set the op_to_name_ mapping for "
      << inst << " to an empty string.";
  mapped_name.assign(input_mapped_name);
  for (int index : llvm::seq<int>(0, input->getNumResults())) {
    input->getResult(index)->replaceAllUsesWith(inst->getResult(index));
  }
  input->dropAllReferences();
  input->erase();
  return Status::OK();
}

// Creates return nodes per operand of a ReturnOp. If names is supplied, those
// names will be used per node in order instead of generating a unique name.
Status Exporter::AddReturnNode(mlir::ReturnOp op,
                               llvm::ArrayRef<llvm::StringRef> names) {
  Status status;
  auto& return_nodes = returns_[op];
  for (int index : llvm::seq<int>(0, op.getNumOperands())) {
    TF_ASSIGN_OR_RETURN(
        auto node_def,
        GetReturnNode(op, index, names.empty() ? "" : names[index]));
    Node* node = graph_->AddNode(*node_def, &status);
    TF_RETURN_IF_ERROR(status);
    return_nodes.push_back(node);
  }
  return Status::OK();
}

// Handles an NextIteration node specially:
// - NextIteration "source" will not be added to the graph but inserted to a
//   map by using its name attribute;
// - NextIteration "sink" is paired with the "source" with the name attribute.
//   It is added to the graph like the other operations.
Status Exporter::AddNextIterationNode(Operation* inst) {
  auto name = GetName(inst);
  if (inst->getName().getStringRef().endswith(".source")) {
    name_to_inst_[name] = inst;
    return Status::OK();
  }
  source_to_sink_[name_to_inst_[name]] = inst;
  return AddInstructionNode(inst);
}

StatusOr<std::unique_ptr<Graph>> Exporter::Convert(
    const GraphExportConfig& configs, const Dialect* tf_dialect,
    mlir::FuncOp function, FunctionDefLibrary* flib) {
  if (function.getBlocks().size() != 1) {
    return errors::FailedPrecondition(
        "Input FuncOp must have only one basic block!");
  }
  mlir::Block& block = function.front();

  // Determine if _Arg and _Retval nodes should use input and output names.
  bool graph_as_function = false;

  // Extract input & output names if set.
  llvm::SmallVector<llvm::StringRef, 2> input_names;
  llvm::SmallVector<llvm::StringRef, 2> output_names;
  auto dict_attr =
      function.getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
  if (dict_attr) {
    TF_RET_CHECK(dict_attr.get("inputs").isa<mlir::StringAttr>())
        << "inputs missing in entry function attribute";
    TF_RET_CHECK(dict_attr.get("outputs").isa<mlir::StringAttr>())
        << "outputs missing in entry function attribute";
    dict_attr.get("inputs").cast<mlir::StringAttr>().getValue().split(
        input_names, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    dict_attr.get("outputs").cast<mlir::StringAttr>().getValue().split(
        output_names, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    graph_as_function = configs.graph_as_function;
  }

  auto graph = absl::make_unique<Graph>(OpRegistry::Global());

  // Extract version info.
  auto version_attr = function.getParentOfType<mlir::ModuleOp>()
                          .getAttrOfType<mlir::DictionaryAttr>("tf.versions");
  if (version_attr) {
    VersionDef versions;
    versions.set_producer(
        version_attr.get("producer").cast<mlir::IntegerAttr>().getInt());
    versions.set_min_consumer(
        version_attr.get("min_consumer").cast<mlir::IntegerAttr>().getInt());
    for (auto bad_consumer :
         version_attr.get("bad_consumers").cast<mlir::ArrayAttr>()) {
      versions.mutable_bad_consumers()->Add(
          bad_consumer.cast<mlir::IntegerAttr>().getInt());
    }
    graph->set_versions(versions);
  }

  // We have to add the function library here, so a custom operation, which is
  // defined in the function library can be added to the graph.
  TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(*flib));
  Exporter exporter(graph.get(), tf_dialect);

  // Set input and output names and increment the use counter for them to help
  // generate unique names.
  if (!output_names.empty()) {
    auto term = block.getTerminator();
    TF_RET_CHECK(output_names.size() == term->getNumOperands())
        << "output names (" << output_names.size()
        << ") != terminator operands (" << term->getNumOperands() << ")";
    for (auto it : llvm::enumerate(term->getOperands())) {
      exporter.name_to_count_[output_names[it.index()].str()] = 1;
      // Only assign defining op of operands of the return the output names if
      // the main graph did not have its _Retval nodes lifted into the functions
      // returns.
      if (!graph_as_function) {
        auto defining_op = it.value()->getDefiningOp();
        auto& mapped_name = exporter.op_to_name_[defining_op];
        DCHECK(mapped_name.empty())
            << "Convert() attempted to change the op_to_name_ mapping for "
            << defining_op << " from " << mapped_name << " to output "
            << it.index() << " name " << output_names[it.index()].str() << ".";
        mapped_name = output_names[it.index()];
      }
    }
  }
  if (!input_names.empty()) {
    TF_RET_CHECK(input_names.size() == block.getNumArguments());
    for (auto it : llvm::enumerate(function.getArguments())) {
      exporter.name_to_count_[input_names[it.index()].str()] = 1;
      // Only assign user of argument the input name if the main graph did not
      // have its _Arg nodes lifted into the functions arguments.
      if (!graph_as_function) {
        auto first_user = *it.value()->user_begin();
        auto& mapped_name = exporter.op_to_name_[first_user];
        DCHECK(mapped_name.empty())
            << "Convert() attempted to change the op_to_name_ mapping for "
            << first_user << " from " << mapped_name << " to input "
            << it.index() << " name " << input_names[it.index()].str() << ".";
        mapped_name = input_names[it.index()];
      }
    }
  }

  // Adds nodes for basic block (function) arguments.
  for (auto it : llvm::enumerate(block.getArguments())) {
    int index = it.index();
    auto arg = it.value();
    mlir::Type type = arg->getType();
    if (!type.isa<mlir::TensorType>()) {
      return errors::InvalidArgument(
          "FuncOps arguments must have tensor types. Found ",
          mlir::debugString(type), " in function ", function.getName().str());
    }

    TF_RETURN_IF_ERROR(exporter.AddArgumentNode(
        arg, index,
        graph_as_function && !input_names.empty() ? input_names[index] : ""));
  }

  auto convert_called_function = [&](llvm::StringRef name) {
    auto func =
        function.getParentOfType<mlir::ModuleOp>().lookupSymbol<mlir::FuncOp>(
            name);
    if (func != nullptr) {
      TF_RETURN_IF_ERROR(ConvertLibFunction(configs, tf_dialect, func, flib));
      TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(*flib));
    }
    return Status::OK();
  };

  // Adds nodes for operations.
  for (Operation& inst : block) {
    auto op_name = GetTensorFlowOpName(inst.getName().getStringRef());
    if (op_name.ok()) {
      // If it is TF Control dialect specific op, look up custom operation
      // in the module and first convert that, then add it to function
      // definition library
      // TODO(prakalps): If two functions have cyclic dependence, this will
      // introduce an infinite loop.
      TF_RETURN_IF_ERROR(convert_called_function(op_name.ValueOrDie().str()));
    }

    if (IsLegacyCallInstruction(&inst)) {
      TF_RETURN_IF_ERROR(convert_called_function(
          inst.getAttrOfType<mlir::SymbolRefAttr>("f").getLeafReference()));
    }

    for (auto type : inst.getResultTypes()) {
      if (!type.isa<mlir::TensorType>() &&
          !type.isa<mlir::TFControlFlow::TFControlType>()) {
        return errors::InvalidArgument(
            "Values must be of tensor type or TensorFlow control type. Found ",
            mlir::debugString(type));
      }
    }

    if (inst.getName().getStringRef().contains("NextIteration")) {
      TF_RETURN_IF_ERROR(exporter.AddNextIterationNode(&inst));
    } else if (auto return_op = llvm::dyn_cast<mlir::ReturnOp>(inst)) {
      TF_RETURN_IF_ERROR(exporter.AddReturnNode(
          return_op, graph_as_function ? output_names
                                       : llvm::ArrayRef<llvm::StringRef>()));
    } else {
      TF_RETURN_IF_ERROR(exporter.AddInstructionNode(&inst));
    }
  }
  // Adds edges between the argument, operation and return nodes.
  for (Operation& inst : block) {
    TF_RETURN_IF_ERROR(exporter.AddEdge(&inst));
  }
  // Fixes the edges between the inserted nodes and special "_SOURCE" and
  // "_SINK".
  FixupSourceAndSinkEdges(graph.get());
  return graph;
}

Status Exporter::ConvertLibFunction(const GraphExportConfig& configs,
                                    const Dialect* tf_dialect,
                                    mlir::FuncOp function,
                                    FunctionDefLibrary* flib) {
  // First look for the function in the current function library. If found,
  // nothing needs to be done.
  OpRegistry empty_registry;
  FunctionLibraryDefinition flib_def(&empty_registry, *flib);
  auto function_name = function.getName().str();
  if (flib_def.Find(function_name)) return Status::OK();

  // TODO(fengliuai): use a small flib_def to reduce overhead
  TF_ASSIGN_OR_RETURN(auto sub_graph,
                      Exporter::Convert(configs, tf_dialect, function, flib));
  FunctionDef func_def;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*sub_graph, function_name, &func_def));

  // The node defs in FunctionDef might contain debug info which was added
  // by the GraphToFunctionDef method. We should remove it if we don't want
  // to export them to avoid failing the roundtrip test.
  if (!configs.export_debug_info) {
    for (auto& node_def : *func_def.mutable_node_def()) {
      node_def.clear_experimental_debug_info();
    }
  }

  // Checks for gradient attribute. If present converts the gradient function
  // and populates the GradientDef.
  auto grad_string = mlir::TF::TensorFlowDialect::GetGradientAttrName();
  if (auto attr =
          function.getAttrOfType<mlir::FlatSymbolRefAttr>(grad_string)) {
    auto grad_func =
        function.getParentOfType<mlir::ModuleOp>().lookupSymbol<mlir::FuncOp>(
            attr.getValue());
    TF_RETURN_IF_ERROR(
        ConvertLibFunction(configs, tf_dialect, grad_func, flib));
    GradientDef grad;
    grad.set_function_name(function_name);
    grad.set_gradient_func(grad_func.getName().str());
    *flib->add_gradient() = grad;
  }

  auto stateful_string = mlir::TF::TensorFlowDialect::GetStatefulAttrName();
  if (auto attr = function.getAttrOfType<mlir::UnitAttr>(stateful_string)) {
    func_def.mutable_signature()->set_is_stateful(true);
  }
  for (int64 i = 0; i < function.getNumArguments(); ++i) {
    if (auto resource_arg_unique_id_attr =
            function.getArgAttrOfType<mlir::IntegerAttr>(
                i, "tf.resource_arg_unique_id")) {
      (*func_def.mutable_resource_arg_unique_id())[i] =
          resource_arg_unique_id_attr.getInt();
    }
  }

  // Ignore the gradient and is_stateful attribute on the function as they have
  // been handled above.
  absl::flat_hash_set<absl::string_view> attrs_to_ignore = {
      grad_string.data(), stateful_string.data()};
  llvm::SmallVector<mlir::NamedAttribute, 8> funcAttrs(
      function.getDialectAttrs());
  TF_RETURN_IF_ERROR(
      ConvertAttributes(funcAttrs, attrs_to_ignore, func_def.mutable_attr()));
  (*flib->add_function()) = func_def;
  return Status::OK();
}

Status Exporter::Convert(mlir::ModuleOp module,
                         const GraphExportConfig& configs,
                         std::unique_ptr<Graph>* graph,
                         FunctionLibraryDefinition* flib_def) {
  mlir::Identifier entry_func_id =
      mlir::Identifier::get("main", module.getContext());
  absl::optional<mlir::FuncOp> entry_func;
  FunctionDefLibrary flib;
  auto tf_dialect = module.getContext()->getRegisteredDialect("tf");
  for (auto function : module.getOps<mlir::FuncOp>()) {
    if (function.isExternal())
      return errors::FailedPrecondition("External functions not supported");

    if (function.getName() == entry_func_id) {
      entry_func.emplace(function);
    } else {
      TF_RETURN_IF_ERROR(
          ConvertLibFunction(configs, tf_dialect, function, &flib));
    }
  }

  if (!entry_func.has_value())
    return errors::FailedPrecondition("entry function `main` must be present");

  // Updates the graph and the function library definition.
  TF_ASSIGN_OR_RETURN(*graph, Exporter::Convert(configs, tf_dialect,
                                                entry_func.value(), &flib));
  for (auto& func_def : flib.function()) {
    TF_RETURN_IF_ERROR(flib_def->AddFunctionDef(func_def));
  }
  for (auto& grad_def : flib.gradient()) {
    TF_RETURN_IF_ERROR(flib_def->AddGradientDef(grad_def));
  }
  return Status::OK();
}
}  // namespace

Status ConvertMlirToGraph(mlir::ModuleOp module,
                          const GraphExportConfig& configs,
                          std::unique_ptr<Graph>* graph,
                          FunctionLibraryDefinition* flib_def) {
  mlir::PassManager pass_manager(module.getContext());
  pass_manager.addPass(mlir::CreateTFExecutorToControlDialectConversion());
  if (mlir::failed(pass_manager.run(module))) {
    return errors::FailedPrecondition(
        "Failed to convert TFExecutor Dialect to Control Dialect.");
  }
  return Exporter::Convert(module, configs, graph, flib_def);
}

StatusOr<std::unique_ptr<GraphDef>> ConvertMlirToGraphdef(
    mlir::ModuleOp module, const GraphExportConfig& configs) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  auto graph = absl::make_unique<Graph>(flib_def);
  TF_RETURN_IF_ERROR(ConvertMlirToGraph(module, configs, &graph, &flib_def));
  auto graphdef = absl::make_unique<GraphDef>();
  graph->ToGraphDef(graphdef.get());
  if (!configs.export_library) graphdef->clear_library();
  if (!configs.export_shapes) {
    for (auto& node_def : *graphdef->mutable_node()) {
      node_def.mutable_attr()->erase("shape");
    }
  }
  if (!configs.export_debug_info) {
    for (auto& node_def : *graphdef->mutable_node()) {
      node_def.clear_experimental_debug_info();
    }
  }
  return graphdef;
}

}  // namespace tensorflow

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
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Identifier.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/Support/DebugStringHelper.h"  // TF:local_config_mlir
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
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using mlir::Dialect;
using mlir::Operation;
using stream_executor::port::StatusOr;

namespace tensorflow {

// TODO(jpienaar): unify and move from here to be able to reuse with tflite
std::string GetName(Operation* inst) {
  if (auto name_loc = inst->getLoc().dyn_cast<mlir::NameLoc>())
    return name_loc.getName().str();

  if (auto call_loc = inst->getLoc().dyn_cast<mlir::CallSiteLoc>()) {
    // Return name if CallSiteLoc's callee has a NameLoc (as should be the case
    // if imported with DebugInfo), else use the fallback naming scheme below.
    if (auto name_loc = call_loc.getCallee().dyn_cast<mlir::NameLoc>())
      return name_loc.getName().str();
  }

  // If the location is none of the expected types, then simply use name
  // generated using the op type.
  return inst->getName().getStringRef().str();
}

namespace {

// Stateful helper class to export a function into a Graph.
class Exporter {
 public:
  // Converts the given Module to a Graph. The give module should only contain
  // one entry function, which is identified by name "main". This entry function
  // is converted to the base of the graph graph. The rest of the functions are
  // converted to the library functions in that graph.
  static Status Convert(mlir::Module& module, const ExporterConfigs& configs,
                        std::unique_ptr<Graph>* graph,
                        FunctionLibraryDefinition* flib_def);

  // Converts a given Function to a FunctionDef and adds it to the function
  // definition library
  static Status ConvertLibFunction(const ExporterConfigs& configs,
                                   const Dialect* tf_dialect,
                                   mlir::Function function,
                                   FunctionDefLibrary* flib);
  // Converts the given CFG Function to a Graph. The arguments and returns of
  // function are added to the graph with special op names kArgOp and kRetOp.
  // Later on, this graph can be converted a function definition and added to
  // another graph.
  static StatusOr<std::unique_ptr<Graph>> Convert(
      const ExporterConfigs& configs, const Dialect* tf_dialect,
      mlir::Function function, FunctionDefLibrary* flib);

 private:
  explicit Exporter(Graph* graph, const Dialect* tf_dialect)
      : graph_(graph), tf_dialect_(tf_dialect) {}

  Status AddArgumentNode(mlir::BlockArgument* arg, unsigned index);
  Status AddInstructionNode(mlir::Operation* inst);
  Status AddNextIterationNode(mlir::Operation* inst);
  Status AddEdge(mlir::Operation* inst);

  StatusOr<std::unique_ptr<NodeDef>> GetArgumentNode(mlir::BlockArgument* arg,
                                                     unsigned index);
  StatusOr<std::unique_ptr<NodeDef>> GetReturnNode(mlir::Operation* inst,
                                                   unsigned index);
  // Adds one edge between src_node and dst_node. If it is not a control edge,
  // an index is used to find out the right operand of the dst_node.
  Status AddEdgeBetweenNodes(mlir::Value* src, Node* dst_node,
                             unsigned dst_index);

  // Returns a unique name for `op`.
  std::string UniqueName(mlir::Operation* op);

  // Returns a unique name starting with a given prefix.
  std::string UniqueName(llvm::StringRef prefix);

  static StatusOr<std::string> getTFOpName(llvm::StringRef op_name) {
    // When being converted to MLIR, some prefixes and suffixes are added to the
    // operation types, and we have to remove them when converting the
    // operations back to a graph:
    // - "_tf.": every operation type has this prefix.
    // - ".sink": only the NextIteration operation has this suffix. We don't
    // need to consider ".source" because the nodes with this suffix are skipped
    // by the caller and will not be added to the graph.
    if (!op_name.consume_front("_tf.")) {
      return errors::FailedPrecondition("op node '", op_name.str(),
                                        "' was not a TF op!");
    }
    op_name.consume_back(".sink");
    return op_name.str();
  }

  Graph* graph_;
  absl::flat_hash_map<mlir::Operation*, string> op_to_name_;
  absl::flat_hash_map<string, int64> name_to_count_;
  absl::flat_hash_map<mlir::Operation*, Node*> nodes_;
  absl::flat_hash_map<const mlir::BlockArgument*, Node*> args_;
  // One single return operation can return multiple results, and each of them
  // will be converted to one node in the graph.
  typedef absl::InlinedVector<Node*, 4> NodeVector;
  absl::flat_hash_map<mlir::Operation*, NodeVector> returns_;

  // Each NextIteration node in the original graph is converted to a pair of
  // source and sink operations in the MLIR, and we use the following two maps
  // to pair and convet them back to a single NextIteration node. We choose to
  // the "name" attribute, which is from the unique node name, to find out the
  // pairs: When scanning the operations in the block, the source operations
  // are inserted to the name_to_inst_ first, and the other "sink" operation
  // can be paired by checking this map and both are inserted to the
  // source_to_sink_ map.
  absl::flat_hash_map<string, mlir::Operation*> name_to_inst_;
  absl::flat_hash_map<mlir::Operation*, mlir::Operation*> source_to_sink_;

  const mlir::Dialect* tf_dialect_;
};

std::string Exporter::UniqueName(llvm::StringRef prefix) {
  std::string name = prefix;
  auto& val = name_to_count_[name];
  if (val) name = (prefix + llvm::Twine(val)).str();
  ++val;
  return name;
}

std::string Exporter::UniqueName(mlir::Operation* op) {
  auto& name = op_to_name_[op];
  if (!name.empty()) return name;
  name = UniqueName(GetName(op));
  return name;
}

StatusOr<std::unique_ptr<NodeDef>> Exporter::GetArgumentNode(
    mlir::BlockArgument* arg, unsigned index) {
  auto node_def = absl::make_unique<NodeDef>();
  node_def->set_name(
      UniqueName(arg->getOwner()->getFunction().getName().str()));
  node_def->set_op(FunctionLibraryDefinition::kArgOp);
  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(
      arg->getType().cast<mlir::TensorType>().getElementType(), &dtype));
  tensorflow::AttrValue type_attr;
  type_attr.set_type(dtype);
  (*node_def->mutable_attr())["T"] = type_attr;
  tensorflow::AttrValue index_attr;
  index_attr.set_i(index);
  (*node_def->mutable_attr())["index"] = index_attr;
  return node_def;
}

StatusOr<std::unique_ptr<NodeDef>> Exporter::GetReturnNode(
    mlir::Operation* inst, unsigned index) {
  auto node_def = absl::make_unique<NodeDef>();
  auto* inst_op = inst->getOperand(index);
  node_def->set_name(UniqueName(inst->getFunction().getName().str()));
  node_def->set_op(FunctionLibraryDefinition::kRetOp);
  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(
      inst_op->getType().cast<mlir::TensorType>().getElementType(), &dtype));
  tensorflow::AttrValue type_attr;
  type_attr.set_type(dtype);
  (*node_def->mutable_attr())["T"] = type_attr;
  tensorflow::AttrValue index_attr;
  index_attr.set_i(index);
  (*node_def->mutable_attr())["index"] = index_attr;
  return node_def;
}

Status Exporter::AddEdgeBetweenNodes(mlir::Value* src, Node* dst_node,
                                     unsigned dst_index) {
  if (auto* input_result = dyn_cast<mlir::OpResult>(src)) {
    auto* input_inst = input_result->getOwner();
    // replaces the input node by the sink one if it is an NextIteration source:
    auto it = source_to_sink_.find(input_inst);
    if (it != source_to_sink_.end()) {
      input_inst = source_to_sink_[input_inst];
    }
    TF_RET_CHECK(nodes_.find(input_inst) != nodes_.end())
        << "Use of OpResult encountered before def!";
    if (input_result->getType().isa<mlir::TFControlFlow::TFControlType>()) {
      graph_->AddControlEdge(nodes_[input_inst], dst_node);
    } else {
      graph_->AddEdge(nodes_[input_inst], input_result->getResultNumber(),
                      dst_node, dst_index);
    }
  } else if (auto* input_arg = dyn_cast<mlir::BlockArgument>(src)) {
    TF_RET_CHECK(args_.find(input_arg) != args_.end())
        << "Use of BlockArgument encounted before def!";
    auto* input_node = args_[input_arg];
    // For argument, there is only one result output, so the index is always 0.
    graph_->AddEdge(input_node, 0, dst_node, dst_index);
  }
  return Status::OK();
}

Status Exporter::AddEdge(mlir::Operation* inst) {
  auto* dst_node = nodes_[inst];
  bool is_return_op = isa<mlir::ReturnOp>(inst);
  for (int index = 0, e = inst->getNumOperands(); index < e; index++) {
    auto* src = inst->getOperand(index);
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

Status Exporter::AddInstructionNode(mlir::Operation* inst) {
  Status status;
  if (!inst->isKnownTerminator()) {
    std::unique_ptr<NodeDef> node_def;
    auto name = UniqueName(inst);
    // Convert registered TF ops to NodeDef. Only registered ops are handled to
    // ensure that PopulateDerivedAttrs adds the correct attributes.
    // TODO(jpienaar): It should be possible to handle every TF op here, the
    // check is too conservative given we could use a OpDef.
    if (auto abstract_op = inst->getAbstractOperation()) {
      if (&abstract_op->dialect == tf_dialect_) {
        TF_ASSIGN_OR_RETURN(node_def, ConvertTFDialectOpToNodeDef(inst, name));
      }
    }
    // Convert TF control flow dialect ops.
    if (!node_def) {
      TF_ASSIGN_OR_RETURN(node_def,
                          GetOperationNodeDef(inst, name.c_str(), getTFOpName));
    }
    Node* node = graph_->AddNode(*node_def, &status);
    TF_RETURN_IF_ERROR(status);
    nodes_[inst] = node;
  } else if (isa<mlir::ReturnOp>(inst)) {
    for (int index = 0, end = inst->getNumOperands(); index != end; index++) {
      TF_ASSIGN_OR_RETURN(auto node_def, GetReturnNode(inst, index));
      Node* node = graph_->AddNode(*node_def, &status);
      TF_RETURN_IF_ERROR(status);
      if (returns_.find(inst) == returns_.end()) {
        returns_[inst] = NodeVector();
      }
      returns_[inst].push_back(node);
    }
  } else {
    return errors::InvalidArgument("Operation input was not an Value!");
  }
  return Status::OK();
}

Status Exporter::AddArgumentNode(mlir::BlockArgument* arg, unsigned index) {
  // If it is an argument from the "main" function, it has only one user, which
  // is an input node. We recover the original input node and skip adding the
  // argument node. The new input node will be handled as normal in the
  // following steps.
  if (arg->getFunction().getName().is("main")) {
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
    mlir::OperationState state(loc, input_name.str());
    state.attributes.append(input->getAttrs().begin(), input->getAttrs().end());
    for (auto* op : input->getOperands()) {
      // Skip the argument in the new operation.
      if (llvm::isa<mlir::BlockArgument>(op)) continue;
      state.operands.push_back(op);
    }
    for (auto* r : input->getResults()) state.types.push_back(r->getType());
    auto* inst = builder.createOperation(state);
    for (int index = 0, e = input->getNumResults(); index != e; ++index) {
      input->getResult(index)->replaceAllUsesWith(inst->getResult(index));
    }
    input->dropAllReferences();
    input->erase();
    return Status::OK();
  } else {
    TF_ASSIGN_OR_RETURN(auto node_def, GetArgumentNode(arg, index));
    Status status;
    Node* node = graph_->AddNode(*node_def, &status);
    TF_RETURN_IF_ERROR(status);
    args_[arg] = node;
    return Status::OK();
  }
}

// Handles an NextIteration node specially:
// - NextIteration "source" will not be added to the graph but inserted to a
//   map by using its name attribute;
// - NextIteration "sink" is paired with the "source" with the name attribute.
//   It is added to the graph like the other operations.
Status Exporter::AddNextIterationNode(mlir::Operation* inst) {
  auto name = GetName(inst);
  if (inst->getName().getStringRef().endswith(".source")) {
    name_to_inst_[name] = inst;
    return Status::OK();
  }
  source_to_sink_[name_to_inst_[name]] = inst;
  return AddInstructionNode(inst);
}

StatusOr<std::unique_ptr<Graph>> Exporter::Convert(const ExporterConfigs& confs,
                                                   const Dialect* tf_dialect,
                                                   mlir::Function function,
                                                   FunctionDefLibrary* flib) {
  if (function.getBlocks().size() != 1) {
    return errors::FailedPrecondition(
        "Input Function must have only one basic block!");
  }
  mlir::Block& block = function.front();

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
  }

  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  // We have to add the function library here, so a custom operation, which is
  // defined in the function library can be added to the graph.
  TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(*flib));
  Exporter exporter(graph.get(), tf_dialect);

  // Set input and output names.
  if (!output_names.empty()) {
    auto term = block.getTerminator();
    TF_RET_CHECK(output_names.size() == term->getNumOperands())
        << "output names (" << output_names.size()
        << ") != terminator operands (" << term->getNumOperands() << ")";
    int i = 0;
    for (auto it : term->getOperands()) {
      exporter.op_to_name_[it->getDefiningOp()] = output_names[i++];
    }
  }
  if (!input_names.empty()) {
    TF_RET_CHECK(input_names.size() == block.getNumArguments());
    for (auto it : llvm::enumerate(function.getArguments())) {
      exporter.op_to_name_[*it.value()->user_begin()] = input_names[it.index()];
    }
  }

  // Adds nodes for basic block (function) arguments.
  for (int index = 0, e = block.getNumArguments(); index != e; index++) {
    auto* arg = block.getArgument(index);
    mlir::Type type = arg->getType();
    if (!type.isa<mlir::TensorType>()) {
      return errors::InvalidArgument(
          "Functions arguments must have tensor types. Found ",
          mlir::debugString(type), " in function ", function.getName().str());
    }

    TF_RETURN_IF_ERROR(exporter.AddArgumentNode(arg, index));
  }
  // Adds nodes for operations.
  for (mlir::Operation& inst : block) {
    auto op_name = getTFOpName(inst.getName().getStringRef());
    if (op_name.ok()) {
      // If it is TF Control dialect specific op, look up custom operation
      // in the module and first convert that, then add it to function
      // definition library
      // TODO(prakalps): If two functions have cyclic dependence, this will
      // introduce an infinite loop.
      auto func = function.getModule()->getNamedFunction(op_name.ValueOrDie());
      if (func != nullptr) {
        TF_RETURN_IF_ERROR(ConvertLibFunction(confs, tf_dialect, func, flib));
        TF_RETURN_IF_ERROR(graph->AddFunctionLibrary(*flib));
      }
    }

    for (auto* result : inst.getResults()) {
      mlir::Type type = result->getType();
      if (!type.isa<mlir::TensorType>() &&
          !type.isa<mlir::TFControlFlow::TFControlType>()) {
        return errors::InvalidArgument(
            "Values must be of tensor type or TensorFlow control type. Found ",
            mlir::debugString(type));
      }
    }

    if (inst.getName().getStringRef().contains("NextIteration")) {
      TF_RETURN_IF_ERROR(exporter.AddNextIterationNode(&inst));
    } else {
      TF_RETURN_IF_ERROR(exporter.AddInstructionNode(&inst));
    }
  }
  // Adds edges between the argument, operation and return nodes.
  for (mlir::Operation& inst : block) {
    TF_RETURN_IF_ERROR(exporter.AddEdge(&inst));
  }
  // Fixes the edges between the inserted nodes and special "_SOURCE" and
  // "_SINK".
  FixupSourceAndSinkEdges(graph.get());
  return graph;
}

Status Exporter::ConvertLibFunction(const ExporterConfigs& configs,
                                    const Dialect* tf_dialect,
                                    mlir::Function function,
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
  if (auto attr = function.getAttrOfType<mlir::FunctionAttr>(grad_string)) {
    auto grad_func = function.getModule()->getNamedFunction(attr.getValue());
    TF_RETURN_IF_ERROR(
        ConvertLibFunction(configs, tf_dialect, grad_func, flib));
    GradientDef grad;
    grad.set_function_name(function_name);
    grad.set_gradient_func(grad_func.getName().str());
    *flib->add_gradient() = grad;
  }

  // Ignore the gradient attribute on the function as it gets converted to
  // GradientDef.
  absl::flat_hash_set<string> attrs_to_ignore = {grad_string};
  TF_RETURN_IF_ERROR(ConvertAttributes(function.getAttrs(), attrs_to_ignore,
                                       func_def.mutable_attr()));
  (*flib->add_function()) = func_def;
  return Status::OK();
}

Status Exporter::Convert(mlir::Module& module, const ExporterConfigs& configs,
                         std::unique_ptr<Graph>* graph,
                         FunctionLibraryDefinition* flib_def) {
  mlir::Identifier entry_func_id =
      mlir::Identifier::get("main", module.getContext());
  absl::optional<mlir::Function> entry_func;
  FunctionDefLibrary flib;
  auto tf_dialect = module.getContext()->getRegisteredDialect("tf");
  for (auto function : module.getFunctions()) {
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

Status ConvertMlirToGraph(mlir::Module& module, const ExporterConfigs& confs,
                          std::unique_ptr<Graph>* graph,
                          FunctionLibraryDefinition* flib_def) {
  return Exporter::Convert(module, confs, graph, flib_def);
}

StatusOr<std::unique_ptr<GraphDef>> ConvertMlirToGraphdef(
    mlir::Module& module, const ExporterConfigs& confs) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  auto graph = absl::make_unique<Graph>(flib_def);
  TF_RETURN_IF_ERROR(ConvertMlirToGraph(module, confs, &graph, &flib_def));
  auto graphdef = absl::make_unique<GraphDef>();
  graph->ToGraphDef(graphdef.get());
  if (!confs.export_library) graphdef->clear_library();
  if (!confs.export_shapes) {
    for (auto& node_def : *graphdef->mutable_node()) {
      node_def.mutable_attr()->erase("shape");
    }
  }
  if (!confs.export_debug_info) {
    for (auto& node_def : *graphdef->mutable_node()) {
      node_def.clear_experimental_debug_info();
    }
  }
  return graphdef;
}

}  // namespace tensorflow

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
#include "llvm/ADT/SmallVector.h"
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
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/control_flow_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
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
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
using llvm::dyn_cast;
using llvm::isa;
using mlir::BlockArgument;
using mlir::Dialect;
using mlir::Operation;
using mlir::OperationState;
using mlir::Value;
using stream_executor::port::StatusOr;

namespace {

constexpr char kInvalidExecutorGraphMsg[] =
    "Functions must be of a single Graph with single op Islands: ";

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
  bool first = true;
  for (auto c : name) {
    if (IsLegalChar(c, first)) {
      legalized_name += c;
    } else {
      legalized_name += '.';
    }
    first = false;
  }

  return legalized_name;
}

// OpOrArgLocNameMapper that legalizes the returned name.
class LegalizedOpOrValLocNameMapper : public OpOrArgLocNameMapper {
 private:
  std::string GetName(OpOrVal op_or_val) override {
    return LegalizeNodeName(OpOrArgLocNameMapper::GetName(op_or_val));
  }
};

// Checks functions in module are of single tf_executor.graph and each
// tf_executor.island in tf_executor.graph only has a single op.
Status HasSingleGraphSingleOpIslandsFunctions(mlir::ModuleOp module) {
  Status status = Status::OK();
  module.walk([&](mlir::FuncOp function) {
    if (function.getBlocks().size() != 1) {
      status = errors::FailedPrecondition(
          kInvalidExecutorGraphMsg,
          "only single block functions are supported.");
      return mlir::WalkResult::interrupt();
    }

    auto block = function.front().without_terminator();
    auto graph = llvm::dyn_cast<mlir::tf_executor::GraphOp>(block.begin());
    if (!graph) {
      status = errors::FailedPrecondition(
          kInvalidExecutorGraphMsg,
          "first op in function is not a tf_executor.graph.");
      return mlir::WalkResult::interrupt();
    }

    if (!has_single_element(block)) {
      status = errors::FailedPrecondition(
          kInvalidExecutorGraphMsg,
          "function does not only contain a single tf_executor.graph.");
      return mlir::WalkResult::interrupt();
    }

    for (Operation& op : graph.GetBody()) {
      auto island = llvm::dyn_cast<mlir::tf_executor::IslandOp>(op);
      if (!island) continue;

      if (!island.WrapsSingleOp()) {
        status = errors::FailedPrecondition(
            kInvalidExecutorGraphMsg,
            "tf_executor.island must perfectly wrap a single op.");
        return mlir::WalkResult::interrupt();
      }
    }

    return mlir::WalkResult::advance();
  });

  return status;
}

// Finds first inner op if `op` is a tf_executor.island. Otherwise `op` is
// returned.
Operation* GetIslandInnerOpOrSelf(mlir::Operation* op) {
  auto island = llvm::dyn_cast<mlir::tf_executor::IslandOp>(op);
  if (island) return &island.GetBody().front();
  return op;
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
                        FunctionLibraryDefinition* flib_def,
                        absl::flat_hash_set<Node*>* control_ret_nodes);

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
      mlir::FuncOp function, FunctionDefLibrary* flib,
      absl::flat_hash_set<Node*>* control_ret_nodes);

 private:
  explicit Exporter(Graph* graph, const Dialect* tf_dialect)
      : graph_(graph), tf_dialect_(tf_dialect) {}

  Status AddArgumentNode(BlockArgument arg, unsigned index,
                         llvm::StringRef name);
  Status AddFetchNode(mlir::FuncOp function, mlir::tf_executor::FetchOp fetch,
                      llvm::ArrayRef<llvm::StringRef> names);
  Status AddInstructionNode(Operation* inst);
  Status AddEdge(Operation* inst);

  StatusOr<std::unique_ptr<NodeDef>> GetArgumentNode(BlockArgument arg,
                                                     unsigned index,
                                                     llvm::StringRef name);
  StatusOr<std::unique_ptr<NodeDef>> GetReturnNode(mlir::FuncOp function,
                                                   Value operand,
                                                   unsigned index,
                                                   llvm::StringRef name);
  Status GetControlRetNodes(mlir::tf_executor::FetchOp fetch,
                            absl::flat_hash_set<Node*>* control_ret_nodes);
  // Adds one edge between src_node and dst_node. If it is not a control edge,
  // an index is used to find out the right operand of the dst_node.
  Status AddEdgeBetweenNodes(Value src, Node* dst_node, unsigned dst_index);

  Graph* graph_;
  LegalizedOpOrValLocNameMapper op_to_name_;
  absl::flat_hash_map<Operation*, Node*> nodes_;
  llvm::DenseMap<BlockArgument, Node*> args_;
  // One single return operation can return multiple results, and each of them
  // will be converted to one node in the graph.
  typedef absl::InlinedVector<Node*, 4> NodeVector;
  absl::flat_hash_map<Operation*, NodeVector> returns_;
  const mlir::Dialect* tf_dialect_;
};

StatusOr<std::unique_ptr<NodeDef>> Exporter::GetArgumentNode(
    BlockArgument arg, unsigned index, llvm::StringRef name) {
  auto func = arg.getParentRegion()->getParentOfType<mlir::FuncOp>();

  auto node_def = absl::make_unique<NodeDef>();
  if (!name.empty())
    node_def->set_name(name.str());
  else
    node_def->set_name(
        std::string(op_to_name_.GetUniqueName(func.getName().str())));

  node_def->set_op(FunctionLibraryDefinition::kArgOp);

  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(
      arg.getType().cast<mlir::TensorType>().getElementType(), &dtype));
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
    mlir::FuncOp function, Value operand, unsigned index,
    llvm::StringRef name) {
  auto node_def = absl::make_unique<NodeDef>();
  if (!name.empty())
    node_def->set_name(name.str());
  else
    node_def->set_name(
        std::string(op_to_name_.GetUniqueName(function.getName().str())));

  node_def->set_op(FunctionLibraryDefinition::kRetOp);
  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(
      operand.getType().cast<mlir::TensorType>().getElementType(), &dtype));
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
  if (auto input_result = src.dyn_cast<mlir::OpResult>()) {
    auto* input_inst = GetIslandInnerOpOrSelf(input_result.getOwner());
    // Replaces the input node with NextIteration sink if it is a NextIteration
    // source.
    if (auto next_iter_source =
            llvm::dyn_cast<mlir::tf_executor::NextIterationSourceOp>(
                input_inst))
      input_inst = next_iter_source.GetSink();

    auto node_it = nodes_.find(input_inst);
    TF_RET_CHECK(node_it != nodes_.end())
        << "Use of OpResult encountered before def!";
    if (input_result.getType().isa<mlir::tf_executor::ControlType>()) {
      graph_->AddControlEdge(node_it->second, dst_node);
    } else {
      graph_->AddEdge(node_it->second, input_result.getResultNumber(), dst_node,
                      dst_index);
    }
    return Status::OK();
  }

  auto input_arg = src.cast<BlockArgument>();
  auto input_node_it = args_.find(input_arg);
  TF_RET_CHECK(input_node_it != args_.end())
      << "Use of BlockArgument encounted before def!";
  // For argument, there is only one result output, so the index is always 0.
  graph_->AddEdge(input_node_it->second, 0, dst_node, dst_index);
  return Status::OK();
}

Status Exporter::AddEdge(Operation* inst) {
  // For tf_executor.fetch, add only its data edges. Control edges are captured
  // later.
  if (auto fetch = llvm::dyn_cast<mlir::tf_executor::FetchOp>(inst)) {
    for (auto operand_and_idx : llvm::enumerate(fetch.getOperands())) {
      Value operand = operand_and_idx.value();
      if (operand.getType().isa<mlir::tf_executor::ControlType>()) break;

      auto* dst_node = returns_[fetch][operand_and_idx.index()];
      TF_RETURN_IF_ERROR(AddEdgeBetweenNodes(operand, dst_node, 0));
    }

    return Status::OK();
  }

  // For tf_executor.NextIteration.Sink, skip its token operand and add data and
  // control edges with their index offset by 1.
  if (auto next_iter_sink =
          llvm::dyn_cast<mlir::tf_executor::NextIterationSinkOp>(inst)) {
    auto* dst_node = nodes_[inst];
    TF_RETURN_IF_ERROR(
        AddEdgeBetweenNodes(next_iter_sink.input(), dst_node, 0));
    for (auto control_and_idx : llvm::enumerate(next_iter_sink.controlInputs()))
      TF_RETURN_IF_ERROR(AddEdgeBetweenNodes(control_and_idx.value(), dst_node,
                                             control_and_idx.index() + 1));

    return Status::OK();
  }

  // For tf_executor.NextIteration.Source, op can be skipped as it is assumed
  // there are no operands.
  if (llvm::isa<mlir::tf_executor::NextIterationSourceOp>(inst)) {
    assert(inst->getNumOperands() == 0);
    return Status::OK();
  }

  Operation* op = GetIslandInnerOpOrSelf(inst);
  auto* dst_node = nodes_[op];
  int operand_offset = 0;
  // For tf_executor.island, add data edges from its wrapped op before control
  // edges.
  if (auto island = llvm::dyn_cast<mlir::tf_executor::IslandOp>(inst)) {
    for (auto operand_and_idx : llvm::enumerate(op->getOperands()))
      TF_RETURN_IF_ERROR(AddEdgeBetweenNodes(operand_and_idx.value(), dst_node,
                                             operand_and_idx.index()));

    operand_offset = op->getNumOperands();
  }

  // For all other ops (including tf_executor.island), add remaining edges.
  for (auto operand_and_idx : llvm::enumerate(inst->getOperands()))
    TF_RETURN_IF_ERROR(
        AddEdgeBetweenNodes(operand_and_idx.value(), dst_node,
                            operand_and_idx.index() + operand_offset));

  return Status::OK();
}

Status Exporter::AddInstructionNode(Operation* inst) {
  std::unique_ptr<NodeDef> node_def;
  auto name = op_to_name_.GetUniqueName(inst);
  // Convert registered TF ops to NodeDef. Only registered ops are handled to
  // ensure that PopulateDerivedAttrs adds the correct attributes.
  TF_ASSIGN_OR_RETURN(node_def,
                      ConvertTFDialectOpToNodeDef(
                          inst, name, /*ignore_unregistered_attrs=*/false));

  Status status;
  Node* node = graph_->AddNode(*node_def, &status);
  TF_RETURN_IF_ERROR(status);
  DCHECK(node != nullptr);
  nodes_[inst] = node;
  return Status::OK();
}

bool IsEntryFunctionArg(BlockArgument arg) {
  return arg.getParentRegion()->getParentOfType<mlir::FuncOp>().getName() ==
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
  if (!arg.hasOneUse()) {
    return errors::FailedPrecondition(
        "Arg in 'main' should only have one user.");
  }
  auto* input = *arg.user_begin();
  auto* parent = input->getParentOp();
  auto island = llvm::dyn_cast_or_null<mlir::tf_executor::IslandOp>(parent);
  if (!island)
    return errors::FailedPrecondition(
        "User of arg in 'main' must be in an inner op of a "
        "tf_executor.island.");

  if (!island.control().use_empty())
    return errors::FailedPrecondition(
        "tf_executor.island of user of arg in 'main' must have no control "
        "output users.");

  auto input_name = input->getName().getStringRef();
  input_name.consume_back(".input");

  mlir::OpBuilder builder(island.getContext());
  builder.setInsertionPointToStart(&island.GetBody());
  auto loc = mlir::NameLoc::get(
      builder.getIdentifier(op_to_name_.GetUniqueName(input)),
      builder.getContext());
  OperationState state(loc, input_name.str());
  state.attributes.append(input->getAttrs().begin(), input->getAttrs().end());
  for (auto op : input->getOperands()) {
    // Skip the argument in the new operation.
    if (op.isa<BlockArgument>()) continue;
    state.operands.push_back(op);
  }
  state.types.append(input->getResultTypes().begin(),
                     input->getResultTypes().end());
  auto* inst = builder.createOperation(state);
  // If it is one of the specified input names, then the new instruction should
  // have the same name.
  op_to_name_.InitOpName(inst, op_to_name_.GetUniqueName(input));
  for (int index : llvm::seq<int>(0, input->getNumResults())) {
    input->getResult(index).replaceAllUsesWith(inst->getResult(index));
  }
  input->dropAllReferences();
  input->erase();
  return Status::OK();
}

// Creates return nodes per operand of a FetchOp. If names is supplied, those
// names will be used per node in order instead of generating a unique name.
Status Exporter::AddFetchNode(mlir::FuncOp function,
                              mlir::tf_executor::FetchOp fetch,
                              llvm::ArrayRef<llvm::StringRef> names) {
  Status status;
  auto& return_nodes = returns_[fetch];
  for (auto operand_and_idx : llvm::enumerate(fetch.getOperands())) {
    if (operand_and_idx.value().getType().isa<mlir::tf_executor::ControlType>())
      break;

    TF_ASSIGN_OR_RETURN(
        auto node_def,
        GetReturnNode(function, operand_and_idx.value(),
                      operand_and_idx.index(),
                      names.empty() ? "" : names[operand_and_idx.index()]));
    Node* node = graph_->AddNode(*node_def, &status);
    TF_RETURN_IF_ERROR(status);
    return_nodes.push_back(node);
  }
  return Status::OK();
}

// Collects control ret Nodes based on tf_executor.graph's associated
// tf_executor.fetch control inputs.
Status Exporter::GetControlRetNodes(
    mlir::tf_executor::FetchOp fetch,
    absl::flat_hash_set<Node*>* control_ret_nodes) {
  for (Value fetch_operand : fetch.getOperands()) {
    if (fetch_operand.getType().isa<mlir::tf_executor::ControlType>()) {
      Operation* defining_op =
          GetIslandInnerOpOrSelf(fetch_operand.getDefiningOp());
      auto node_it = nodes_.find(defining_op);
      TF_RET_CHECK(node_it != nodes_.end());
      control_ret_nodes->insert(node_it->second);
    }
  }
  return Status::OK();
}

StatusOr<std::unique_ptr<Graph>> Exporter::Convert(
    const GraphExportConfig& configs, const Dialect* tf_dialect,
    mlir::FuncOp function, FunctionDefLibrary* flib,
    absl::flat_hash_set<Node*>* control_ret_nodes) {
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

  auto graph_op = llvm::cast<mlir::tf_executor::GraphOp>(block.front());

  // Set input and output names and increment the use counter for them to help
  // generate unique names.
  if (!output_names.empty()) {
    const int num_data_results = graph_op.getNumResults();
    TF_RET_CHECK(output_names.size() == num_data_results)
        << "output names (" << output_names.size()
        << ") != terminator operands (" << num_data_results << ")";
    llvm::DenseMap<Operation*, llvm::StringRef> output_op_to_name;
    llvm::StringMap<Operation*> name_to_op;
    for (auto it : llvm::enumerate(graph_op.GetFetch().getOperands())) {
      // Skip control rets.
      if (it.index() >= num_data_results) break;
      // If there is a result index specified, ensure only one and that it
      // matches the result index of the op.
      auto result = it.value().cast<mlir::OpResult>();
      std::string orig_name(output_names[it.index()]);
      auto tensor_id = ParseTensorName(orig_name);
      auto name = LegalizeNodeName(
          llvm::StringRef(tensor_id.node().data(), tensor_id.node().size()));

      if (graph_as_function) {
        // Ensure name does not get reused.
        (void)exporter.op_to_name_.GetUniqueName(name);
        continue;
      }

      TF_RET_CHECK(result.getResultNumber() == tensor_id.index());
      Operation* defining_op = GetIslandInnerOpOrSelf(result.getDefiningOp());
      if (output_op_to_name.insert({defining_op, name}).second) {
        TF_RET_CHECK(name_to_op.insert({name, defining_op}).second)
            << "multiple operations associated with the same name";
        exporter.op_to_name_.InitOpName(defining_op, name);
      } else {
        TF_RET_CHECK(output_op_to_name[defining_op] == name)
            << "associating multiple names with the same op not supported";
      }
    }
  }

  if (!input_names.empty()) {
    TF_RET_CHECK(input_names.size() == block.getNumArguments());
    for (auto it : llvm::enumerate(function.getArguments())) {
      // TODO(lyandy): Update when changing feed/fetch import.
      std::string orig_name(input_names[it.index()]);
      std::string name = LegalizeNodeName(orig_name);
      auto tensor_id = ParseTensorName(name);
      TF_RET_CHECK(tensor_id.index() == 0)
          << "input port designation not supported";
      // Only assign user of argument the input name if the main graph did not
      // have its _Arg nodes lifted into the functions arguments.
      if (graph_as_function) {
        // Ensure name does not get reused.
        (void)exporter.op_to_name_.GetUniqueName(name);
      } else {
        Operation* defining_op =
            GetIslandInnerOpOrSelf(*it.value().user_begin());
        exporter.op_to_name_.InitOpName(defining_op, name);
      }
    }
  }

  // Adds nodes for basic block (function) arguments.
  for (auto it : llvm::enumerate(block.getArguments())) {
    int index = it.index();
    auto arg = it.value();
    mlir::Type type = arg.getType();
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
  for (Operation& inst : graph_op.GetBody()) {
    for (auto type : inst.getResultTypes())
      if (!type.isa<mlir::TensorType>() &&
          !type.isa<mlir::tf_executor::ControlType>() &&
          !type.isa<mlir::tf_executor::TokenType>())
        return errors::InvalidArgument(
            "Values must be of tensor type, TensorFlow control type, or "
            "TensorFlow token type. Found ",
            mlir::debugString(type));

    if (llvm::isa<mlir::tf_executor::NextIterationSourceOp>(inst)) {
      // Skip tf_executor.NextIteration.Source as associated
      // tf_executor.NextIteration.Sink will be used instead.
      continue;
    } else if (auto fetch = llvm::dyn_cast<mlir::tf_executor::FetchOp>(inst)) {
      TF_RETURN_IF_ERROR(exporter.AddFetchNode(
          function, fetch,
          graph_as_function ? output_names
                            : llvm::ArrayRef<llvm::StringRef>()));
    } else if (auto island =
                   llvm::dyn_cast<mlir::tf_executor::IslandOp>(inst)) {
      Operation& inner_op = island.GetBody().front();
      auto op_name = GetTensorFlowOpName(inner_op.getName().getStringRef());
      if (op_name.ok()) {
        // If it is TF Control dialect specific op, look up custom operation
        // in the module and first convert that, then add it to function
        // definition library
        // TODO(prakalps): If two functions have cyclic dependence, this will
        // introduce an infinite loop.
        TF_RETURN_IF_ERROR(convert_called_function(op_name.ValueOrDie().str()));
      }

      if (IsLegacyCallInstruction(&inner_op)) {
        TF_RETURN_IF_ERROR(convert_called_function(
            inner_op.getAttrOfType<mlir::SymbolRefAttr>("f")
                .getLeafReference()));
      }

      TF_RETURN_IF_ERROR(exporter.AddInstructionNode(&inner_op));
    } else {
      TF_RETURN_IF_ERROR(exporter.AddInstructionNode(&inst));
    }
  }
  // Adds edges between the argument, operation and return nodes.
  for (Operation& inst : graph_op.GetBody()) {
    TF_RETURN_IF_ERROR(exporter.AddEdge(&inst));
  }
  // Fixes the edges between the inserted nodes and special "_SOURCE" and
  // "_SINK".
  FixupSourceAndSinkEdges(graph.get());

  TF_RETURN_IF_ERROR(
      exporter.GetControlRetNodes(graph_op.GetFetch(), control_ret_nodes));

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
  absl::flat_hash_set<Node*> control_ret_nodes;
  TF_ASSIGN_OR_RETURN(auto sub_graph,
                      Exporter::Convert(configs, tf_dialect, function, flib,
                                        &control_ret_nodes));
  const auto control_ret = [&](const Node* n) -> absl::optional<string> {
    return control_ret_nodes.contains(n)
               ? absl::make_optional<string>(n->name())
               : absl::nullopt;
  };
  FunctionDef func_def;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*sub_graph, function_name, control_ret, &func_def));

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
                         FunctionLibraryDefinition* flib_def,
                         absl::flat_hash_set<Node*>* control_ret_nodes) {
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
  TF_ASSIGN_OR_RETURN(
      *graph, Exporter::Convert(configs, tf_dialect, entry_func.value(), &flib,
                                control_ret_nodes));
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
                          FunctionLibraryDefinition* flib_def,
                          absl::flat_hash_set<Node*>* control_ret_nodes) {
  TF_RETURN_IF_ERROR(HasSingleGraphSingleOpIslandsFunctions(module));
  return Exporter::Convert(module, configs, graph, flib_def, control_ret_nodes);
}

Status ConvertMlirToGraph(mlir::ModuleOp module,
                          const GraphExportConfig& configs,
                          std::unique_ptr<Graph>* graph,
                          FunctionLibraryDefinition* flib_def) {
  absl::flat_hash_set<Node*> control_ret_nodes;
  return ConvertMlirToGraph(module, configs, graph, flib_def,
                            &control_ret_nodes);
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

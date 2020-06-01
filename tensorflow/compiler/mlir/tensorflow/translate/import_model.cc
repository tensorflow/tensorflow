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

#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"

#include <iterator>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Identifier.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/jit/shape_inference_helpers.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/control_flow_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/utils/transitive_fanin.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

static inline absl::string_view StringRefToView(llvm::StringRef ref) {
  return {ref.data(), ref.size()};
}

namespace tensorflow {
using mlir::NamedAttrList;
using mlir::TensorType;
using mlir::TF::VarHandleOp;
using mlir::tf_saved_model::GlobalTensorOp;
using stream_executor::port::StatusOr;

namespace {

bool IsDisableCallShapeInferenceAttribute(const AttrValue& attr_value,
                                          llvm::StringRef attr_name) {
  return attr_name.compare("_disable_call_shape_inference") == 0 &&
         attr_value.value_case() == AttrValue::kB;
}

bool IsOutputShapesAttribute(const AttrValue& attr_value,
                             llvm::StringRef attr_name) {
  return attr_name.compare("_output_shapes") == 0 &&
         attr_value.value_case() == AttrValue::kList;
}

bool IsResourceOutputShapesAttribute(const AttrValue& attr_value,
                                     llvm::StringRef attr_name) {
  if (attr_name == "_handle_dtypes" || attr_name == "_handle_shapes")
    return attr_value.value_case() == AttrValue::kList;
  return false;
}

// This class is used to generate new MLIR function name strings that are both
// unique in the TF function library `flib_` and unique among the name strings
// generated by the class object during its lifetime.
//
// In theory, this class is not necessary because we should simply take
// the TF function name and use it as MLIR function name. However, for some
// unknown reasons (callout for investigation in b/142268695), keeping the
// function names unchanged in an MLIR roundtrip causes test failures.
// TODO(b/142268695) Re-evaluate whether we need this class v.s. directly using
// and TF function name as MLIR function name after b/142268695 is root caused.
class NameUniquifier : public OpOrArgNameMapper {
 public:
  explicit NameUniquifier(const FunctionLibraryDefinition& flib)
      : flib_(flib) {}

 private:
  bool IsUnique(llvm::StringRef name) override {
    return !flib_.Contains(std::string(name));
  }

  std::string GetName(OpOrVal op_or_val) override {
    DCHECK(false) << "Unimplemented";
    return "";
  }

  const FunctionLibraryDefinition& flib_;
};

// Stateful helper class to import a TensorFlow model into an MLIR Module.
//
// This is the base class that contains common utilities shared between the
// GraphDef importer and SavedModel importer.
//
// A subclass is expected to call `PrepareConvert` first to perform necessary
// preparation over the graph and also certain internal bookkeeping data.
// Afterwards the other protected methods can be called.
class ImporterBase {
 protected:
  explicit ImporterBase(
      const FunctionLibraryDefinition& flib, const GraphDebugInfo& debug_info,
      const GraphImportConfig& specs, mlir::ModuleOp module,
      std::unordered_map<std::string, std::string>* tf_name_to_mlir_name,
      NameUniquifier* function_name_uniquifier,
      llvm::StringRef function_name_for_debug_info = "")
      : builder_(module.getContext()),
        module_(module),
        context_(module.getContext()),
        tf_name_to_mlir_name_(tf_name_to_mlir_name),
        graph_flib_(flib),
        specs_(specs),
        debug_info_(debug_info),
        function_name_for_debug_info_(function_name_for_debug_info),
        function_name_uniquifier_(function_name_uniquifier),
        error_handler_(module.getContext()) {}

  // Returns the inferred function signature of the given function body. Input
  // types are unranked tensor of the respective datatype in the function and
  // result types are inferred by the shape_refiner_. Result types need not be
  // unranked tensors and could be ranked tensors in cases where result type
  // depends on an op with static output shape like tf.Const.
  StatusOr<mlir::FunctionType> InferLibFunctionType(const FunctionBody& fbody);

  // Extracts arg and ret nodes from FunctionBody.
  void GetArgsAndRetsFromFunctionBody(
      const FunctionBody& fbody,
      absl::InlinedVector<OutputTensor, 4>* arg_nodes,
      absl::InlinedVector<OutputTensor, 4>* ret_nodes,
      absl::InlinedVector<Node*, 4>* control_ret_nodes);

  // Prepares converting the graph to an MLIR module. This step removes the
  // backedges of the graph, orders the nodes and infers the shapes.
  Status PrepareConvert(const Graph& graph);

  // Converts the prepared graph to a Function and adds it to the module. A set
  // of nodes from the graph are given to converted to the arguments and returns
  // of the function.
  Status Convert(llvm::StringRef func_name, mlir::FunctionType func_type,
                 const absl::InlinedVector<OutputTensor, 4>& arg_nodes,
                 const absl::InlinedVector<OutputTensor, 4>& ret_nodes,
                 const absl::InlinedVector<Node*, 4>& control_ret_nodes,
                 llvm::ArrayRef<mlir::NamedAttribute> attrs,
                 bool function_graph);

  // Finds out the function definition for the given function name from the
  // graph and converts it to a function of the module. This method is called
  // on demand because the graph flib_def does not provide an iterator
  // interface.
  Status ConvertLibFunction(llvm::StringRef func_name);

  // Returns the list of nodes in the graph. Nodes are presented in the reverse
  // order of a post-order depth-first visit starting from the graph's source
  // nodes.
  llvm::ArrayRef<Node*> GetOrderedNodes() const { return ordered_nodes_; }

  // Returns the inferred input type at index `idx` of the `node` in the
  // context.
  StatusOr<mlir::Type> InferInputType(const Node& node, int idx,
                                      mlir::Builder builder);

  // Returns the inferred output type at index `idx` of the `node` in the
  // context.
  StatusOr<mlir::Type> InferOutputType(const Node& node, int idx,
                                       mlir::Builder builder);

 private:
  // Most types with subtypes have only one subtype.
  using ElementSubtypes = llvm::SmallVector<TensorType, 1>;

  // Adds all the ordered_nodes to the shape refiner shape_refiner_. Then all
  // data type and shape information is maintained by the shape_refiner_.
  // TODO(jpienaar): Remove once shape inference on import is removed.
  Status AddNodesToShapeRefiner(
      std::unordered_map<string, Node*>* node_name_map);

  // Prune nodes that do not feed into fetch nodes.
  Status PruneUnreachableNodes(
      std::unordered_map<string, Node*>* node_name_map);

  // Converts feeds to Placeholder nodes.
  Status ConvertFeedsToPlaceholders(
      std::unordered_map<string, Node*>* node_name_map);

  // Converts the inferred shape referred to by 'handle' in 'context', with
  // given element type, and returns an MLIR tensor type.
  StatusOr<TensorType> ConvertDataTypeAndShape(
      DataType dtype, const shape_inference::ShapeHandle& handle,
      const std::vector<shape_inference::ShapeAndType>* handle_subtypes,
      shape_inference::InferenceContext* context, mlir::Builder builder);

  // Converts the inferred shape referred to by 'handle' in 'context', with
  // given element type, and returns an MLIR tensor type.
  StatusOr<TensorType> ConvertElementTypeAndShape(
      mlir::Type element_type, const shape_inference::ShapeHandle& handle,
      shape_inference::InferenceContext* context, mlir::Builder builder);

  // Converts the inferred subtypes for an element type to corresponding MLIR
  // types in 'context'.
  StatusOr<ElementSubtypes> ConvertSubtypes(
      const std::vector<shape_inference::ShapeAndType>* handle_subtypes,
      shape_inference::InferenceContext* context, mlir::Builder builder);

  // Converts the tensor proto into an MLIR elements attribute.
  StatusOr<mlir::ElementsAttr> ConvertTensorProto(const TensorProto& value) {
    return ::tensorflow::ConvertTensorProto(value, &builder_);
  }

  // Converts the tensor shape proto into an MLIR shape attribute.
  StatusOr<mlir::TF::ShapeAttr> ConvertTensorShapeProto(
      const TensorShapeProto& shape) {
    if (shape.unknown_rank())
      return mlir::TF::ShapeAttr::get(builder_.getContext(), llvm::None);

    llvm::SmallVector<int64_t, 4> dims;
    dims.reserve(shape.dim().size());
    for (const auto& dim : shape.dim()) {
      dims.push_back(dim.size());
    }
    return mlir::TF::ShapeAttr::get(builder_.getContext(),
                                    llvm::makeArrayRef(dims));
  }

  // Converts func name in graphdef to mlir::SymbolRefAttribute.
  StatusOr<mlir::FlatSymbolRefAttr> ConvertFunctionCallName(
      const std::string& func_name);

  // Converts the given non-function-call AttrValue to an MLIR Attribute.
  StatusOr<mlir::Attribute> ConvertAttributeValue(const AttrValue& value);

  // Converts the given function-call AttrValue to MLIR Attributes and pushes
  // them to the given attributes list. For example, if there is a kFunc
  // AttrValue {name : foo, attrs : {k1 : bar, k2 : rfc}}, it will convert it to
  // a list of MLIR Attributes: [{base_name : foo}, {base_name.k1 : bar},
  // {base_name.k2 : rfc}}.
  Status ConvertFunctionCallAttribute(const std::string& base_name,
                                      const AttrValue& value,
                                      NamedAttrList* attributes);

  // Helper to create either a tf_executor operation or a TF operation wrapped
  // in an island. When convert_to_legacy_call is true, converts the operation
  // representing a call to a library function with a name represented in
  // node_type_name to LegacyCallOp.
  mlir::Operation* CreateOperation(
      const Node& node, llvm::StringRef node_type_name,
      const mlir::OperationState& result,
      const llvm::SmallVectorImpl<mlir::Value>& control_operands,
      bool convert_to_legacy_call = false);

  // Converts one NodeDef from the input GraphDef into an Operation and
  // inserts it into the MLIR module using builder_.
  Status ConvertNode(const Node& node);

  // If the input graph represents a while-loop, the edges pointing from a
  // "NextIteration" node to a "Merge" node add cyclic dependencies and make the
  // topological sorting impossible. We need to remove these edges from the
  // input graph to infer shapes and construct a Function. For each
  // "NextIteration" node, there are two operations, "NextIteration.source"
  // and "NextIteration.sink" are added to the MLIR module.
  using BackEdge = BackEdgeHelper::BackEdge;

  // Removes backedges from the input graph. The removed edges are added back to
  // to OpBuilder after the remaining graph is converted to the Function.
  Status RemoveBackedges(const Graph& graph);

  // Restores backedges removed during shape inference to the final Function.
  Status AddBackedges();

  // Restores a single backedge in the Function by adding a replicated
  // operation before the dst operation.
  Status AddBackedge(mlir::Operation* sink, mlir::Operation* dst,
                     int dst_input);

  // Adds the input arguments and return operation to the function. The
  // arguments are added as basic block argument. Also the argument types and
  // the id of the nodes from the input graph needs to be specified.
  Status ConvertFunctionArgAndRets(
      mlir::FuncOp func, mlir::tf_executor::GraphOp graph_op,
      llvm::ArrayRef<mlir::Type> arg_types,
      const absl::InlinedVector<OutputTensor, 4>& arg_nodes,
      const absl::InlinedVector<OutputTensor, 4>& ret_nodes,
      const absl::InlinedVector<Node*, 4>& control_ret_nodes);

  // Gets the location information of the given node. It uses the
  // "original_node_name" in the NodeDef to get the corresponding file location
  // (FileLineColLoc) from the input DebugInfo and returns an CallSiteLoc. If
  // there are multiple "original_node_names", a FusedLoc is returned. If the
  // node name couldn't be found in the input DebugInfo, a NameLoc is used as
  // the location.
  mlir::Location GetLocation(const NodeDef& node);

  // Appends the location string for the node to the error message and returns
  // the combined error status.
  Status EmitErrorWithLocationStr(const Node& node, const Status& error_status);

  // Inserts a placeholder node in the graph to replace a feed output tensor,
  // and returns the new placeholder node and a boolean indicating if the
  // original input node was removed from the graph. Uses of the feed output
  // tensor are replaced with this placeholder node. If the feed output tensor
  // is of a single output node, the control dependencies are forwarded to the
  // the placeholder node, and the original node will be removed.
  // Note: This modifies the graph, and so any list of ordered nodes needs to be
  // reconstructed.
  StatusOr<std::pair<Node*, bool>> CreatePlaceholderNodeForFeed(
      const TensorShapeProto& shape, DataType dtype, Node* node, int index,
      const std::unordered_map<string, Node*>& node_name_map);

  // Gets the input and output nodes corresponding to the specified input and
  // output nodes in specs_. If there are no input or output nodes specified,
  // nodes will be empty.
  Status GetInputOutputNodes(
      const std::unordered_map<string, Node*>& node_name_map,
      std::unordered_set<const Node*>* nodes);

  // The input graph with backedges removed. The removed backedges are stored
  // in the back_edge_helper.
  BackEdgeHelper back_edge_helper_;
  // A map between node and output index, for each backedge.
  absl::flat_hash_map<const Node*, int> back_edge_node_output_;
  absl::flat_hash_map<const Node*, BackEdge> back_edge_dst_inputs_;
  // A map between sink and source operation of NextIteration
  absl::flat_hash_map<mlir::Operation*, mlir::Operation*>
      next_iteration_sink_source_;

  // All nodes and version information about the (copied) imported graph.
  std::unique_ptr<Graph> graph_;
  std::vector<Node*> ordered_nodes_;

  // Maps from a Node ID to a MLIR value.
  using NodeValueMap = absl::flat_hash_map<int, mlir::Operation*>;

  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;
  mlir::MLIRContext* context_;
  std::unordered_map<std::string, std::string>* tf_name_to_mlir_name_;
  const FunctionLibraryDefinition& graph_flib_;
  const GraphImportConfig& specs_;
  const GraphDebugInfo& debug_info_;
  llvm::StringRef function_name_for_debug_info_;
  NodeValueMap node_values_;
  // TODO(jpienaar): Remove once shape inference on import is removed.
  // The shape_refinner_ will be nullptr if shape inference on import is
  // not enabled.
  std::unique_ptr<ShapeRefiner> shape_refiner_ = nullptr;
  NameUniquifier* function_name_uniquifier_;
  mlir::StatusScopedDiagnosticHandler error_handler_;

 protected:
  // Maps feed as TensorId to new Placeholder node name.
  absl::flat_hash_map<TensorId, absl::string_view> remapped_feeds_;
};

// Returns true if the node with given name has a non primary output that is
// used by some other node as an input. Returns false if no outputs are in use
// or only the first output is in use.
bool HasNonPrimaryOutputInUse(const GraphDef& graph_def,
                              const std::string& node) {
  for (const auto& node_def : graph_def.node()) {
    for (const auto& input : node_def.input()) {
      if (absl::StartsWith(input, node + ":") && input != node + ":0") {
        return true;
      }
    }
  }
  return false;
}

// Updates the given LegacyFedInput node with Placeholder node if it is one of
// the inputs. Returns an error if non primary output of the LegacyFedInput node
// is in use and therefore can not be replaced by the Placeholder node that only
// has a single output.
Status UpdateLegacyFedInputNode(const GraphDef& graph_def,
                                const GraphImportConfig::InputArrays& inputs,
                                NodeDef* node) {
  const std::string& node_name = node->name();
  auto it = inputs.find(node_name);

  // Node is not an input.
  if (it == inputs.end()) return Status::OK();

  if (HasNonPrimaryOutputInUse(graph_def, node_name)) {
    return errors::InvalidArgument(
        "LegacyFedInput node ", node->name(),
        " has non primary output in use and can not be replaced with "
        "Placeholder node");
  }

  DataType dtype = it->second.imported_dtype;
  // Uses the existing output type if it isn't specified by the user.
  if (dtype == DT_INVALID) {
    dtype = node->attr().at("output_types").list().type(0);
  }
  // Update op name, drop inputs and set attributes required by the Placeholder
  // op.
  *node->mutable_op() = "Placeholder";
  node->clear_attr();
  node->clear_input();
  AddNodeAttr("dtype", dtype, node);
  AddNodeAttr("shape", it->second.shape, node);
  return Status::OK();
}

// Preprocesses GraphDef before it can be converted to Graph by,
// - Adding the default attributes to each node def if they are missing from
//   the GraphDef.
// - Replacing LegacyFedInput nodes with Placeholder nodes if
//   convert_legacy_fed_inputs option is enabled.
Status PreprocessGraphDef(const GraphImportConfig* specs, GraphDef* graph_def) {
  for (auto& node_def : *graph_def->mutable_node()) {
    // TODO(hinsu): Completely deprecate support for LegacyFedInput ops. One
    // solution could be have a tool to let users upgrade old serialized graphs.
    if (specs && specs->convert_legacy_fed_inputs &&
        node_def.op() == "LegacyFedInput") {
      TF_RETURN_IF_ERROR(
          UpdateLegacyFedInputNode(*graph_def, specs->inputs, &node_def));
    }

    const tensorflow::OpRegistrationData* op_reg_data =
        tensorflow::OpRegistry::Global()->LookUp(node_def.op());
    if (!op_reg_data) {
      // This is likely a function call node, so we should continue.
      continue;
    }
    ::tensorflow::AddDefaultsToNodeDef(op_reg_data->op_def, &node_def);
  }
  return Status::OK();
}

// Mapping from node name to feed (index and ArrayInfo). Node name must outlive
// this map.
using FeedsByNode = absl::flat_hash_map<
    absl::string_view,
    absl::flat_hash_map<int, const std::pair<std::string, ArrayInfo>*>>;

// Creates from a `GraphImportConfig::InputArrays` a mapping from a feeds output
// tensor name to index and ArrayInfo. Keys and values are backed by
// `GraphImportConfig::InputArrays`.
StatusOr<FeedsByNode> GetFeedsByNode(
    const GraphImportConfig::InputArrays& inputs) {
  FeedsByNode feeds_by_node;
  feeds_by_node.reserve(inputs.size());

  for (const auto& input : inputs) {
    TensorId tensor = ParseTensorName(input.first);
    if (tensor.index() < 0)
      return errors::FailedPrecondition(
          "Feed output tensor must be a data output '", tensor.ToString(), "'");

    auto& node = feeds_by_node[tensor.node()];
    if (!node.insert({tensor.index(), &input}).second)
      return errors::FailedPrecondition(
          "Multiple feeds for the same output tensor '", tensor.ToString(),
          "'");
  }

  return feeds_by_node;
}

// Creates a unique name for a node that will be replacing a feed output tensor.
std::string GetUniqueNodeName(
    absl::string_view node_name, int index,
    const std::unordered_map<string, Node*>& node_name_map) {
  std::string new_node_name_base = absl::StrCat(node_name, "_", index);
  int count = 0;
  std::string new_node_name = new_node_name_base;
  while (node_name_map.find(new_node_name) != node_name_map.end()) {
    new_node_name = absl::StrCat(new_node_name_base, "_", count++);
  }
  return new_node_name;
}

Status ImporterBase::RemoveBackedges(const Graph& graph) {
  // TODO(fengliuai): Converting to GraphDef and back is the easiest way to
  // clone a graph.
  // TODO(fengliuai): clone the graph without going to graph_def first.
  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  graph_ = absl::make_unique<Graph>(graph.flib_def());
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.add_default_attributes = false;
  TF_RETURN_IF_ERROR(::tensorflow::ConvertGraphDefToGraph(
      opts, std::move(graph_def), graph_.get()));

  // Remove all the backedges. So the nodes can be added to the shape refiner.
  TF_RETURN_IF_ERROR(back_edge_helper_.Remove(graph_.get()));
  VLOG(1) << "Found " << (back_edge_helper_.RemovedEdges().size())
          << " backedges.";

  // Creates a map for quickly identifying whether a node output is a backedge.
  for (const auto& edge : back_edge_helper_.RemovedEdges()) {
    if (back_edge_node_output_.find(edge.src) != back_edge_node_output_.end() &&
        back_edge_node_output_[edge.src] != edge.src_output) {
      return errors::FailedPrecondition(
          "More than one of the src node outputs are backedges!");
    }
    back_edge_node_output_[edge.src] = edge.src_output;
    // We expect a merge to receive a single backedge (multiple NextIteration
    // nodes feeding into the same merge is unexpected here).
    DCHECK(!back_edge_dst_inputs_.contains(edge.dst));
    back_edge_dst_inputs_[edge.dst] = edge;
  }

  // Obtains a RPO ordering, using node names as a tiebreak for stable sorting.
  GetReversePostOrder(
      *graph_, &ordered_nodes_,
      [](const Node* n1, const Node* n2) { return n1->name() < n2->name(); });

  return Status::OK();
}

StatusOr<std::pair<Node*, bool>> ImporterBase::CreatePlaceholderNodeForFeed(
    const TensorShapeProto& shape, DataType dtype, Node* node, int index,
    const std::unordered_map<string, Node*>& node_name_map) {
  DCHECK_LT(index, node->num_outputs());
  const bool update_inplace = node->num_outputs() == 1 && index == 0;
  std::string new_node_name =
      update_inplace ? node->name()
                     : GetUniqueNodeName(node->name(), index, node_name_map);

  Node* placeholder_node;
  NodeBuilder builder(new_node_name, "Placeholder");
  builder.Attr("shape", shape);
  builder.Attr("dtype", dtype);
  TF_RETURN_IF_ERROR(builder.Finalize(graph_.get(), &placeholder_node));

  // Update edges from original feed with Placeholder node.
  std::vector<const Edge*> data_edges;
  std::vector<const Edge*> control_edges;
  for (const tensorflow::Edge* edge : node->out_edges()) {
    if (edge->src_output() == index) {
      data_edges.push_back(edge);
    } else if (update_inplace && edge->IsControlEdge()) {
      control_edges.push_back(edge);
    }
  }

  for (const auto* edge : data_edges) {
    TF_RETURN_IF_ERROR(graph_->UpdateEdge(placeholder_node, 0, edge->dst(),
                                          edge->dst_input()));
  }

  // TODO(lyandy): Preserve control dependencies properly by not forwarding
  // control dependencies to data outputs and not removing single output nodes.
  // When a data output is replaced as a feed, unless there is another non feed
  // data output or an explicit control output used by the same node, transitive
  // control dependencies are not to be executed. For single output nodes,
  // Placeholders can be converted to a NoOp if there are no uses, and
  // PlaceholderWithDefault can be converted to an Identity.
  for (const auto* edge : control_edges) {
    graph_->AddControlEdge(placeholder_node, edge->dst());
    graph_->RemoveControlEdge(edge);
  }

  if (update_inplace) {
    graph_->RemoveNode(node);
  }

  return std::pair<Node*, bool>(placeholder_node, update_inplace);
}

Status ImporterBase::GetInputOutputNodes(
    const std::unordered_map<string, Node*>& node_name_map,
    std::unordered_set<const Node*>* nodes) {
  auto add_node = [&](absl::string_view name) {
    auto it = node_name_map.find(std::string(name));
    if (it == node_name_map.end()) {
      return errors::FailedPrecondition(
          absl::StrCat("Graph does not contain node: ", name));
    }
    nodes->insert(it->second);
    return Status::OK();
  };

  // Remap feeds and fetches to newly created Placeholder nodes.
  for (const auto& input : specs_.inputs) {
    TensorId tensor = ParseTensorName(input.first);
    auto remapped_it = remapped_feeds_.find(tensor);
    if (remapped_it != remapped_feeds_.end()) {
      TF_RETURN_IF_ERROR(add_node(remapped_it->second));
    } else {
      TF_RETURN_IF_ERROR(add_node(tensor.node()));
    }
  }

  for (const auto& output : specs_.outputs) {
    TensorId tensor = ParseTensorName(output);
    auto remapped_it = remapped_feeds_.find(tensor);
    if (remapped_it != remapped_feeds_.end()) {
      TF_RETURN_IF_ERROR(add_node(remapped_it->second));
    } else {
      TF_RETURN_IF_ERROR(add_node(tensor.node()));
    }
  }

  for (const auto& control_output : specs_.control_outputs)
    TF_RETURN_IF_ERROR(add_node(control_output));

  return Status::OK();
}

// TODO(jpienaar): Remove this post shape inference on import flag is removed.
Status ImporterBase::AddNodesToShapeRefiner(
    std::unordered_map<string, Node*>* node_name_map) {
  shape_refiner_ = absl::make_unique<ShapeRefiner>(graph_->versions(),
                                                   graph_->op_registry());
  // Some operations (for example "TPUExecute") don't have shape inference
  // function defined, so we should set this to false for adding nodes with
  // these types of operations.
  shape_refiner_->set_require_shape_inference_fns(false);
  shape_refiner_->set_function_library_for_shape_inference(&graph_flib_);

  TF_ASSIGN_OR_RETURN(auto feeds_by_node, GetFeedsByNode(specs_.inputs));

  // First add all nodes to the refiner.
  for (Node* node : ordered_nodes_) {
    // We need to use a TensorFlow node to teach the shape refiner that user
    // specifies certain data type and shape for the inputs in the `specs_`.
    // This node shouldn't have any inputs, only have one output and its
    // output type/shape is only determined by its "named" attributes. (The
    // attributes should have fixed names so we can use the info from `specs_`
    // to set the value of them.) `Placeholder` satisfies these constraints.
    //
    // Therefore, if the input node isn't a `Placeholder`, we create one and use
    // it to replace the original input node, so the shape refiner can
    // successfully propagate the user's input type and shape to the rest of the
    // graph.
    bool node_added_to_shape_refiner = false;
    auto it = feeds_by_node.find(node->name());
    if (it != feeds_by_node.end()) {
      auto op_name = node->op_def().name();
      if (op_name != "Placeholder" && op_name != "LegacyFedInput" &&
          op_name != FunctionLibraryDefinition::kArgOp) {
        for (const auto& output_tensor : it->second) {
          const int index = output_tensor.first;
          const ArrayInfo& array_info = output_tensor.second->second;

          DataType dtype = array_info.imported_dtype;
          // Uses the existing output type if it isn't specified by the user.
          if (dtype == DT_INVALID) {
            dtype = node->output_type(index);
          }

          TF_ASSIGN_OR_RETURN(
              auto placeholder_node_and_removed,
              CreatePlaceholderNodeForFeed(array_info.shape, dtype, node, index,
                                           *node_name_map));

          Node* placeholder_node = placeholder_node_and_removed.first;
          if (placeholder_node_and_removed.second) {
            // Original node has been removed from the graph.
            node = placeholder_node;
            node_added_to_shape_refiner = true;
          }
          remapped_feeds_[{it->first, index}] = placeholder_node->name();
          (*node_name_map)[placeholder_node->name()] = placeholder_node;
          // Add the new placeholder node to the shape refiner.
          Status status = shape_refiner_->AddNode(placeholder_node);
          if (!status.ok()) {
            return EmitErrorWithLocationStr(*placeholder_node, status);
          }
        }
      } else {
        auto index_it = it->second.find(0);
        if (index_it == it->second.end()) {
          return errors::FailedPrecondition(
              "Missing feed output tensor at index 0 for node '", node->name(),
              "'");
        }
        node->AddAttr("shape", index_it->second->second.shape);
        DataType dtype = index_it->second->second.imported_dtype;
        // Uses the existing output type if it isn't specified by the user.
        if (dtype == DT_INVALID) {
          dtype = node->output_type(0);
        }
        node->AddAttr("dtype", dtype);
      }
    }
    if (!node_added_to_shape_refiner) {
      // Add the node to the shape refiner if the node hasn't been removed.
      Status status = shape_refiner_->AddNode(node);
      if (!status.ok()) {
        return EmitErrorWithLocationStr(*node, status);
      }
    }

    auto set_shape_from_list_attr = [&](const AttrValue* attr) {
      auto& list = attr->list();
      for (auto shape : llvm::enumerate(list.shape())) {
        auto* node_context = shape_refiner_->GetContext(node);
        shape_inference::ShapeHandle handle;
        Status status =
            node_context->MakeShapeFromShapeProto(shape.value(), &handle);
        if (!status.ok()) {
          return EmitErrorWithLocationStr(*node, status);
        }
        node_context->set_output(shape.index(), handle);
      }
      return Status::OK();
    };

    // We currently have no other way to get shapes from ReadVariableOp's.
    // Some graphs seem to have _output_shapes attributes on them, so use that
    // if possible.
    // TODO(silvasean): Ideally, we would do this in a separate shape inference
    // pass to avoid adding complexity to the importer. But right now, we don't
    // have an MLIR-native shape inference pass, so we need to do this while we
    // still have the Graph around, i.e. here, in the importer.
    if (node->op_def().name() == "ReadVariableOp") {
      // TODO(silvasean): In some graphs, this seems to be annotated on every
      // node. Why and by whom?
      // TODO(b/140588338): We should ideally incorporate that information for
      // all nodes, but right now, this can result in e.g. an Identity node with
      // signature such as
      // `(tensor<?x?xf32>) -> tensor<?x9216xf32>` which fails the verifier
      // (which checks for exact type equality; _output_shapes results in
      // us shoehorning in the more-precise type on the output).
      if (const AttrValue* attr = node->attrs().Find("_output_shapes"))
        TF_RETURN_IF_ERROR(set_shape_from_list_attr(attr));
    }

    // If it is the argument node, the shape handle is set explicitly, so it
    // can be propagated to the body nodes of the function.
    if (StringPiece(node->type_string()) == FunctionLibraryDefinition::kArgOp) {
      auto* node_context = shape_refiner_->GetContext(node);
      DCHECK(node_context != nullptr);
      if (const AttrValue* attr = node->attrs().Find("shape")) {
        shape_inference::ShapeHandle handle;
        Status status =
            node_context->MakeShapeFromShapeProto(attr->shape(), &handle);
        if (!status.ok()) {
          return EmitErrorWithLocationStr(*node, status);
        }
        node_context->set_output(0, handle);
      } else if (const AttrValue* attr = node->attrs().Find("_output_shapes")) {
        TF_RETURN_IF_ERROR(set_shape_from_list_attr(attr));
      } else {
        node_context->set_output(0, node_context->UnknownShape());
      }
    }
  }

  // Since we might have inserted and removed nodes from the graph, fix
  // source/sink edges and reconstruct the RPO ordering of nodes
  FixupSourceAndSinkEdges(graph_.get());

  // Prune nodes in the graph that are not reachable from the output.
  if (specs_.prune_unused_nodes) {
    std::unordered_set<const Node*> prune_start;
    TF_RETURN_IF_ERROR(GetInputOutputNodes(*node_name_map, &prune_start));
    if (!prune_start.empty()) {
      if (PruneForReverseReachability(graph_.get(), prune_start)) {
        VLOG(1) << "Pruned unused nodes in graphdef";
      } else {
        VLOG(1) << "No unused nodes in graphdef to prune";
      }
    } else {
      VLOG(1) << "No output nodes specified, skipping pruning";
    }
  } else {
    VLOG(1) << "Pruning unused nodes in graphdef is disabled";
  }

  // Re-initialize ordered_nodes_ since we might have modified the graph.
  GetReversePostOrder(
      *graph_, &ordered_nodes_,
      [](const Node* n1, const Node* n2) { return n1->name() < n2->name(); });

  VLOG(1) << "Inferring graph shapes to fixpoint";

  // The "changed" information from UpdateNode can give false positives, so we
  // create a dedicated method to verify the shapes are not changed before and
  // after the shape refine.
  auto same_inferred_shape = [](shape_inference::InferenceContext* c,
                                shape_inference::ShapeHandle s0,
                                shape_inference::ShapeHandle s1) -> bool {
    if (s0.SameHandle(s1) || (!c->RankKnown(s0) && !c->RankKnown(s1))) {
      return true;
    }
    if (c->Rank(s0) != c->Rank(s1)) {
      return false;
    }
    for (int i = 0; i < c->Rank(s0); ++i) {
      if (!c->Dim(s0, i).SameHandle(c->Dim(s1, i))) {
        int64 val0 = c->Value(c->Dim(s0, i));
        int64 val1 = c->Value(c->Dim(s1, i));
        // Negative value is treated as unknown so all negative values indicate
        // the same dimension.
        if (val0 >= 0 && val1 >= 0 && val0 != val1) return false;
      }
    }
    return true;
  };

  bool changed = true;
  int i = 0;
  const int kMaxIterationCount = 2;
  while (changed && i != kMaxIterationCount) {
    changed = false;
    for (const Node* node : ordered_nodes_) {
      auto* shape_context = shape_refiner_->GetContext(node);
      DCHECK(shape_context != nullptr);
      absl::InlinedVector<shape_inference::ShapeHandle, 4> existing;
      existing.reserve(shape_context->num_outputs());
      for (int o = 0; o < shape_context->num_outputs(); ++o) {
        existing.push_back(shape_context->output(o));
      }
      bool inferred = false;
      shape_inference::ShapeHandle handle;
      Status status =
          shape_refiner_->UpdateNode(node, /*relax=*/false, &inferred);
      if (!status.ok()) {
        return EmitErrorWithLocationStr(*node, status);
      }
      for (int o = 0; o < shape_context->num_outputs(); ++o) {
        if (!same_inferred_shape(shape_context, shape_context->output(o),
                                 existing[o])) {
          changed = true;
          break;
        }
      }
    }
    ++i;
  }
  if (i >= kMaxIterationCount) {
    LOG(WARNING) << "Graph shapes did not converge to a fixpoint within "
                 << kMaxIterationCount
                 << " iterations. Graph shapes may be conservative.";
  }
  VLOG(1) << "Graph shapes were inferred with " << (i - 1)
          << " extra rounds of analysis to reach a fixpoint.";
  return Status::OK();
}

StatusOr<mlir::Type> ImporterBase::InferInputType(const Node& node, int idx,
                                                  mlir::Builder builder) {
  if (specs_.enable_shape_inference) {
    // TODO(jpienaar): Remove this if shape inference on import flag is removed.
    ExtendedInferenceContext* shape_context =
        shape_refiner_->GetExtendedContext(&node);
    DataType dtype = shape_context->input_type(idx);
    auto* context = shape_context->get_context();
    return ConvertDataTypeAndShape(dtype, context->input(idx),
                                   context->input_handle_shapes_and_types(idx),
                                   context, builder);
  }
  DataType dtype = node.properties()->input_types[idx];
  mlir::Type element_type;
  TF_RETURN_IF_ERROR(ConvertDataType(dtype, builder, &element_type));
  return mlir::UnrankedTensorType::get(element_type);
}

StatusOr<mlir::Type> ImporterBase::InferOutputType(const Node& node, int idx,
                                                   mlir::Builder builder) {
  DataType dtype = node.properties()->output_types[idx];

  // Returns output type given inference context.
  auto shape_ic = [&](shape_inference::InferenceContext* c) {
    return ConvertDataTypeAndShape(dtype, c->output(idx),
                                   c->output_handle_shapes_and_types(idx), c,
                                   builder);
  };

  if (specs_.enable_shape_inference) {
    // TODO(jpienaar): Remove this if shape inference on import flag is removed.
    ExtendedInferenceContext* shape_context =
        shape_refiner_->GetExtendedContext(&node);
    return shape_ic(shape_context->get_context());
  }

  // Treat TensorList init ops specially here as the op requires knowing its
  // element dtype.
  // TODO(jpienaar): Reconsider post refactoring shape functions.
  if (node.type_string() == "TensorListReserve" ||
      node.type_string() == "EmptyTensorList") {
    mlir::Type etype;
    if (auto element_dtype = node.attrs().Find("element_dtype")) {
      TF_RETURN_IF_ERROR(
          ConvertDataType(element_dtype->type(), builder, &etype));
    }
    return mlir::RankedTensorType::get(
        {}, mlir::TF::VariantType::get({mlir::UnrankedTensorType::get(etype)},
                                       etype.getContext()));
  }

  // Returns a simple, more conservative unranked tensor type.
  auto default_type = [&]() -> StatusOr<mlir::Type> {
    mlir::Type element_type;
    TF_RETURN_IF_ERROR(ConvertDataType(dtype, builder, &element_type));
    return mlir::UnrankedTensorType::get(element_type);
  };

  // Below we only try and do some shape inference for "source" ops which have
  // no inputs.
  if (node.num_inputs() > 0) return default_type();

  // Do some simply inference here to get the function arguments correct for
  // this common case.
  // TODO(jpienaar): Reconsider post refactoring shape functions.
  if (node.IsArg()) {
    if (dtype == DT_RESOURCE) {
      const AttrValue* dtype_attr = node.attrs().Find("_handle_dtypes");
      const AttrValue* shape_attr = node.attrs().Find("_handle_shapes");
      if (dtype_attr && shape_attr) {
        if (dtype_attr->list().type().empty()) {
          return errors::InvalidArgument(
              "Invalid \"_handle_dtypes\" attribute value for _Arg node: ",
              shape_attr->DebugString());
        }
        if (shape_attr->list().shape().empty()) {
          return errors::InvalidArgument(
              "Invalid \"_handle_shapes\" attribute value for _Arg node: ",
              shape_attr->DebugString());
        }
        DataType dtype = dtype_attr->list().type(0);
        const TensorShapeProto& shape_proto = shape_attr->list().shape(0);
        TF_ASSIGN_OR_RETURN(
            auto etype, ConvertToMlirTensorType(shape_proto, dtype, &builder));
        return mlir::UnrankedTensorType::get(mlir::TF::ResourceType::get(
            {etype.cast<TensorType>()}, builder.getContext()));
      } else {
        return mlir::UnrankedTensorType::get(
            mlir::TF::ResourceType::get(builder.getContext()));
      }
    } else if (auto shape = node.attrs().Find("_output_shapes")) {
      if (shape->has_list() && shape->list().shape_size() == 1) {
        return ConvertToMlirTensorType(shape->list().shape().at(0), dtype,
                                       &builder);
      }
    }
  }

  const tensorflow::OpRegistrationData* op_reg_data;
  TF_RETURN_IF_ERROR(
      graph_->op_registry()->LookUp(node.type_string(), &op_reg_data));
  if (!op_reg_data) {
    DVLOG(1) << "Skipping inference for unregistered op " << node.type_string();
    return default_type();
  }
  if (op_reg_data->shape_inference_fn == nullptr) {
    DVLOG(1) << "Skipping inference for op without shape function "
             << node.type_string();
    return default_type();
  }
  shape_inference::InferenceContext c(graph_->versions().producer(),
                                      node.attrs(), op_reg_data->op_def,
                                      std::vector<PartialTensorShape>{}, {},
                                      /*input_tensors_as_shapes=*/{}, {});
  TF_RETURN_IF_ERROR(c.Run(op_reg_data->shape_inference_fn));
  return shape_ic(&c);
}

StatusOr<TensorType> ImporterBase::ConvertDataTypeAndShape(
    DataType dtype, const shape_inference::ShapeHandle& handle,
    const std::vector<shape_inference::ShapeAndType>* handle_subtypes,
    shape_inference::InferenceContext* context, mlir::Builder builder) {
  TF_ASSIGN_OR_RETURN(auto subtypes,
                      ConvertSubtypes(handle_subtypes, context, builder));

  mlir::Type element_type;
  if (dtype == DT_VARIANT)
    element_type = mlir::TF::VariantType::get(subtypes, context_);
  else if (dtype == DT_RESOURCE)
    element_type = mlir::TF::ResourceType::get(subtypes, context_);
  else
    TF_RETURN_IF_ERROR(
        ::tensorflow::ConvertDataType(dtype, builder, &element_type));

  return ConvertElementTypeAndShape(element_type, handle, context, builder);
}

StatusOr<TensorType> ImporterBase::ConvertElementTypeAndShape(
    mlir::Type element_type, const shape_inference::ShapeHandle& handle,
    shape_inference::InferenceContext* context, mlir::Builder builder) {
  if (!context->RankKnown(handle)) {
    return mlir::UnrankedTensorType::get(element_type);
  }

  // Sentinel for an unknown dimension size. getTensorType interprets any
  // negative value as an unknown dimension.
  // TODO(jmolloy): Ideally this shouldn't be a local sentinel.
  const int64_t kUnknownDim = -1;

  absl::InlinedVector<int64_t, 4> dimensions;
  int32 rank = context->Rank(handle);
  dimensions.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    auto dim_handle = context->Dim(handle, i);
    if (!context->ValueKnown(dim_handle))
      dimensions.push_back(kUnknownDim);
    else
      dimensions.push_back(context->Value(dim_handle));
  }

  return mlir::RankedTensorType::get(
      llvm::makeArrayRef(dimensions.begin(), dimensions.end()), element_type);
}

StatusOr<ImporterBase::ElementSubtypes> ImporterBase::ConvertSubtypes(
    const std::vector<shape_inference::ShapeAndType>* handle_subtypes,
    shape_inference::InferenceContext* context, mlir::Builder builder) {
  ElementSubtypes subtypes;
  if (!handle_subtypes) return subtypes;

  subtypes.reserve(handle_subtypes->size());
  for (const auto& subtype : *handle_subtypes) {
    mlir::Type element_type;
    TF_RETURN_IF_ERROR(
        ::tensorflow::ConvertDataType(subtype.dtype, builder, &element_type));
    TF_ASSIGN_OR_RETURN(TensorType type,
                        ConvertElementTypeAndShape(element_type, subtype.shape,
                                                   context, builder));
    subtypes.push_back(type);
  }
  return subtypes;
}

Status ImporterBase::ConvertFunctionCallAttribute(const std::string& base_name,
                                                  const AttrValue& value,
                                                  NamedAttrList* attributes) {
  TF_ASSIGN_OR_RETURN(auto func_attr,
                      ConvertFunctionCallName(value.func().name()));
  attributes->push_back(builder_.getNamedAttr(base_name, func_attr));

  for (const auto& it : value.func().attr()) {
    auto name = absl::StrCat(base_name, ".", it.first);
    TF_ASSIGN_OR_RETURN(auto value, ConvertAttributeValue(it.second));
    attributes->push_back(builder_.getNamedAttr(name, value));
  }
  return Status::OK();
}

StatusOr<mlir::FlatSymbolRefAttr> ImporterBase::ConvertFunctionCallName(
    const std::string& func_name) {
  TF_RETURN_IF_ERROR(ConvertLibFunction(func_name));
  auto mlir_func_name = (*tf_name_to_mlir_name_)[func_name];
  auto func = module_.lookupSymbol<mlir::FuncOp>(mlir_func_name);
  return builder_.getSymbolRefAttr(func);
}

StatusOr<mlir::Attribute> ImporterBase::ConvertAttributeValue(
    const AttrValue& value) {
  switch (value.value_case()) {
    case AttrValue::kI:
      return builder_.getI64IntegerAttr(value.i());
    case AttrValue::kS:
      return builder_.getStringAttr(value.s());
    case AttrValue::kF:
      return builder_.getFloatAttr(builder_.getF32Type(), value.f());
    case AttrValue::kB:
      return builder_.getBoolAttr(value.b());
    case AttrValue::kType: {
      mlir::Type type;
      TF_RETURN_IF_ERROR(ConvertDataType(value.type(), builder_, &type));
      return mlir::TypeAttr::get(type);
    }
    case AttrValue::kShape:
      return ConvertTensorShapeProto(value.shape());
    case AttrValue::kTensor:
      return ConvertTensorProto(value.tensor());
    case AttrValue::kList: {
      absl::InlinedVector<mlir::Attribute, 8> attrs;
      for (const auto& item : value.list().i())
        attrs.push_back(builder_.getI64IntegerAttr(item));
      for (const auto& item : value.list().s())
        attrs.push_back(builder_.getStringAttr(item));
      for (const auto& item : value.list().f())
        attrs.push_back(builder_.getFloatAttr(builder_.getF32Type(), item));
      for (const auto& item : value.list().b())
        attrs.push_back(builder_.getBoolAttr(item));
      for (const auto& item : value.list().type()) {
        mlir::Type type;
        TF_RETURN_IF_ERROR(ConvertDataType(DataType(item), builder_, &type));
        attrs.push_back(mlir::TypeAttr::get(type));
      }
      for (const auto& item : value.list().shape()) {
        TF_ASSIGN_OR_RETURN(auto attr, ConvertTensorShapeProto(item));
        attrs.push_back(attr);
      }
      for (const auto& item : value.list().tensor()) {
        TF_ASSIGN_OR_RETURN(auto attr, ConvertTensorProto(item));
        attrs.push_back(attr);
      }
      for (const auto& item : value.list().func()) {
        TF_ASSIGN_OR_RETURN(auto attr, ConvertFunctionCallName(item.name()));
        if (item.attr_size() != 0)
          return errors::Unimplemented(
              "func attributes with non-zero attr.size()");
        attrs.push_back(attr);
      }
      return builder_.getArrayAttr(
          llvm::makeArrayRef(attrs.begin(), attrs.end()));
    }
    case AttrValue::kFunc: {
      // TODO(b/156546237): Unify kFunc/NameAttrList attribute representation.
      // Currently kFunc/NameAttrList attributes in a kList/repeated AttrValue
      // will not use this representation.
      NamedAttrList attrs;
      for (const auto& func_attr : value.func().attr()) {
        TF_ASSIGN_OR_RETURN(auto attr, ConvertAttributeValue(func_attr.second));
        attrs.push_back(builder_.getNamedAttr(func_attr.first, attr));
      }
      auto func_attrs = builder_.getDictionaryAttr(attrs);
      return mlir::TF::FuncAttr::get(context_, value.func().name(), func_attrs);
    }
    case AttrValue::VALUE_NOT_SET:
      return builder_.getUnitAttr();
    // kPlaceholder is not implemented.
    default:
      return errors::Unimplemented(
          absl::StrCat("Attribute ", value.DebugString()));
  }
}

void ImporterBase::GetArgsAndRetsFromFunctionBody(
    const FunctionBody& fbody, absl::InlinedVector<OutputTensor, 4>* arg_nodes,
    absl::InlinedVector<OutputTensor, 4>* ret_nodes,
    absl::InlinedVector<Node*, 4>* control_ret_nodes) {
  arg_nodes->reserve(fbody.arg_nodes.size());
  ret_nodes->reserve(fbody.ret_nodes.size());
  for (auto arg : fbody.arg_nodes) {
    arg_nodes->emplace_back(arg, 0);
  }
  for (auto ret : fbody.ret_nodes) {
    ret_nodes->emplace_back(ret, 0);
  }
  *control_ret_nodes = fbody.control_ret_nodes;
}

Status ImporterBase::ConvertLibFunction(llvm::StringRef func_name) {
  // If the library function has been converted already, nothing needs to be
  // done.
  if (tf_name_to_mlir_name_->find(std::string(func_name)) !=
      tf_name_to_mlir_name_->end())
    return Status::OK();

  std::string mlir_func_name(
      function_name_uniquifier_->GetUniqueName(func_name));
  (*tf_name_to_mlir_name_)[std::string(func_name)] = mlir_func_name;

  const auto& func_lib = graph_flib_;
  const auto* func_def = func_lib.Find(std::string(func_name));
  if (func_def == nullptr) {
    return errors::FailedPrecondition(
        absl::StrCat("Failed to find function '", StringRefToView(func_name),
                     "'. The imported TensorFlow GraphDef is ill-formed."));
  }

  // Converts the function definition to a graph.
  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(
      FunctionDefToBodyHelper(*func_def, AttrSlice(), &func_lib, &fbody));

  // Converts the argument and return types to MLIR types.
  absl::InlinedVector<mlir::NamedAttribute, 8> attributes;
  attributes.reserve(func_def->attr_size());
  for (const auto& name_and_value : func_def->attr()) {
    // This is a function definition attribute, so it shouldn't contain
    // kFunc attribute and it is treated as normal one.
    TF_ASSIGN_OR_RETURN(auto attr,
                        ConvertAttributeValue(name_and_value.second));
    std::string attr_name =
        mangling_util::MangleAttributeName(name_and_value.first);
    attributes.push_back(builder_.getNamedAttr(attr_name, attr));
  }

  // Checks opdef stateful attribute and import that as Function Attribute
  if (func_def->signature().is_stateful()) {
    auto stateful_str = mlir::TF::TensorFlowDialect::GetStatefulAttrName();
    attributes.push_back(
        builder_.getNamedAttr(stateful_str, builder_.getUnitAttr()));
  }

  // Checks for an associated custom gradient function. Adds it to the attribute
  // list of this function.
  auto grad_func_name = func_lib.FindGradient(std::string(func_name));
  if (!grad_func_name.empty()) {
    TF_RETURN_IF_ERROR(ConvertLibFunction(grad_func_name));
    auto mlir_grad_func_name = (*tf_name_to_mlir_name_)[grad_func_name];
    auto grad_func = module_.lookupSymbol<mlir::FuncOp>(mlir_grad_func_name);
    auto gradient_attr = builder_.getSymbolRefAttr(grad_func);
    auto grad_string = mlir::TF::TensorFlowDialect::GetGradientAttrName();
    attributes.push_back(builder_.getNamedAttr(grad_string, gradient_attr));
  }

  // Converts the graph to an MLIR function and adds it to the module.
  // We populate the NodeSpec so that all the _Arg ops get their shape
  // added correctly.
  GraphImportConfig specs;
  specs.enable_shape_inference = specs_.enable_shape_inference;
  for (const auto& name_and_value : func_def->attr()) {
    if (name_and_value.first == "_input_shapes") {
      auto& list = name_and_value.second.list();
      auto& signature = func_def->signature();
      if (list.shape_size() != signature.input_arg_size()) {
        return errors::FailedPrecondition(
            "Number of input arguments must be equal to the length of "
            "_input_shapes attribute in function '",
            StringRefToView(func_name), "'.");
      }
      for (int i = 0; i < list.shape_size(); i++) {
        auto& input_arg = signature.input_arg(i);
        auto& array_info = specs.inputs[input_arg.name()];
        array_info.imported_dtype = input_arg.type();
        array_info.shape = list.shape(i);
      }
    }
  }

  ImporterBase child_importer(graph_flib_, debug_info_, specs, module_,
                              tf_name_to_mlir_name_, function_name_uniquifier_,
                              func_name);
  TF_RETURN_IF_ERROR(child_importer.PrepareConvert(*fbody->graph));

  TF_ASSIGN_OR_RETURN(auto func_type,
                      child_importer.InferLibFunctionType(*fbody));

  absl::InlinedVector<OutputTensor, 4> arg_nodes;
  absl::InlinedVector<OutputTensor, 4> ret_nodes;
  absl::InlinedVector<Node*, 4> control_ret_nodes;
  GetArgsAndRetsFromFunctionBody(*fbody, &arg_nodes, &ret_nodes,
                                 &control_ret_nodes);

  TF_RETURN_IF_ERROR(child_importer.Convert(
      mlir_func_name, func_type, arg_nodes, ret_nodes, control_ret_nodes,
      llvm::makeArrayRef(attributes.begin(), attributes.end()),
      /*function_graph=*/true));
  return Status::OK();
}

Status ImporterBase::PruneUnreachableNodes(
    std::unordered_map<string, Node*>* node_name_map) {
  std::unordered_set<const Node*> prune_start;
  TF_RETURN_IF_ERROR(GetInputOutputNodes(*node_name_map, &prune_start));

  if (!prune_start.empty()) {
    if (PruneForReverseReachability(graph_.get(), prune_start)) {
      VLOG(1) << "Pruned unused nodes in graphdef";
    } else {
      VLOG(1) << "No unused nodes in graphdef to prune";
    }
  } else {
    VLOG(1) << "No output nodes specified, skipping pruning";
  }
  return Status::OK();
}

Status ImporterBase::ConvertFeedsToPlaceholders(
    std::unordered_map<string, Node*>* node_name_map) {
  // Feeds (edges) are converted into single-output placeholder nodes to
  // simplify the conversion process.
  TF_ASSIGN_OR_RETURN(auto feeds_by_node, GetFeedsByNode(specs_.inputs));
  for (const auto& it : feeds_by_node) {
    TensorId tensor = ParseTensorName(it.first);
    auto jt = node_name_map->find(std::string(tensor.node()));
    if (jt == node_name_map->end()) {
      return errors::FailedPrecondition(
          absl::StrCat("Graph does not contain node: ", tensor.node()));
    }

    Node* node = jt->second;
    auto op_name = node->op_def().name();
    if (op_name != "Placeholder" && op_name != "LegacyFedInput" &&
        op_name != FunctionLibraryDefinition::kArgOp) {
      for (const auto& output_tensor : it.second) {
        const int index = output_tensor.first;
        const ArrayInfo& array_info = output_tensor.second->second;

        DataType dtype = array_info.imported_dtype;
        // Uses the existing output type if it isn't specified by the user.
        if (dtype == DT_INVALID) {
          dtype = node->output_type(index);
        }

        TF_ASSIGN_OR_RETURN(
            auto placeholder_node_and_removed,
            CreatePlaceholderNodeForFeed(array_info.shape, dtype, node, index,
                                         *node_name_map));

        Node* placeholder_node = placeholder_node_and_removed.first;
        if (placeholder_node->in_edges().empty()) {
          graph_->AddControlEdge(graph_->source_node(), placeholder_node,
                                 true /* skip test for duplicates */);
        }
        if (placeholder_node->out_edges().empty()) {
          graph_->AddControlEdge(placeholder_node, graph_->sink_node(),
                                 true /* skip test for duplicates */);
        }
        remapped_feeds_[{it.first, index}] = placeholder_node->name();
        (*node_name_map)[placeholder_node->name()] = placeholder_node;
      }
    }
  }
  return Status::OK();
}

Status ImporterBase::PrepareConvert(const Graph& graph) {
  TF_RETURN_IF_ERROR(RemoveBackedges(graph));

  auto node_name_map = graph_->BuildNodeNameIndex();

  if (specs_.enable_shape_inference) {
    // TODO(jpienaar): Remove once infer shapes on import flag is removed.
    TF_RETURN_IF_ERROR(AddNodesToShapeRefiner(&node_name_map));
  } else {
    TF_RETURN_IF_ERROR(ConvertFeedsToPlaceholders(&node_name_map));
  }

  // Prune nodes in the graph that are not reachable from the output.
  if (specs_.prune_unused_nodes) {
    TF_RETURN_IF_ERROR(PruneUnreachableNodes(&node_name_map));
  }

  if (!specs_.enable_shape_inference) {
    // Re-initialize ordered_nodes_ since we might have modified the graph.
    GetReversePostOrder(
        *graph_, &ordered_nodes_,
        [](const Node* n1, const Node* n2) { return n1->name() < n2->name(); });
  }

  return Status::OK();
}

Status ImporterBase::Convert(
    llvm::StringRef func_name, mlir::FunctionType func_type,
    const absl::InlinedVector<OutputTensor, 4>& arg_nodes,
    const absl::InlinedVector<OutputTensor, 4>& ret_nodes,
    const absl::InlinedVector<Node*, 4>& control_ret_nodes,
    llvm::ArrayRef<mlir::NamedAttribute> attrs, bool function_graph) {
  // TODO(b/122040776): Uses debug info for FunctionDef.
  auto function = mlir::FuncOp::create(mlir::UnknownLoc::get(context_),
                                       func_name, func_type, attrs);

  module_.push_back(function);
  // Seeds the builder with an initial block.
  function.addEntryBlock();
  builder_ = mlir::OpBuilder(function.getBody());

  // Create the graph operation in which we will convert the individual nodes.
  auto graph = builder_.create<mlir::tf_executor::GraphOp>(
      function.getLoc(), func_type.getResults());
  builder_.createBlock(&graph.body());

  for (const Node* node : ordered_nodes_) {
    TF_RETURN_IF_ERROR(ConvertNode(*node));
  }

  // Adds the backedges back to the function by creating the source and sink
  // pairs.
  TF_RETURN_IF_ERROR(AddBackedges());

  TF_RETURN_IF_ERROR(ConvertFunctionArgAndRets(function, graph,
                                               func_type.getInputs(), arg_nodes,
                                               ret_nodes, control_ret_nodes));

  // TODO(jpienaar): Update post removing shape_refinier_.
  if (!specs_.enable_shape_inference) {
    // Refine graph's type given more precise fetch.
    auto fetch = graph.GetFetch();
    bool all_equal = true;
    for (auto it :
         llvm::zip_first(graph.getResults(), fetch.getOperandTypes())) {
      auto rt = std::get<1>(it);
      if (rt == std::get<0>(it).getType()) continue;
      std::get<0>(it).setType(rt);
      all_equal = false;
    }
    if (!all_equal) {
      function.setType(mlir::FunctionType::get(func_type.getInputs(),
                                               graph.getResultTypes(),
                                               function.getContext()));
    }
  }

  return Status::OK();
}

Status ImporterBase::ConvertFunctionArgAndRets(
    mlir::FuncOp func, mlir::tf_executor::GraphOp graph_op,
    llvm::ArrayRef<mlir::Type> arg_types,
    const absl::InlinedVector<OutputTensor, 4>& arg_nodes,
    const absl::InlinedVector<OutputTensor, 4>& ret_nodes,
    const absl::InlinedVector<Node*, 4>& control_ret_nodes) {
  auto* bb = &func.front();
  llvm::SmallDenseMap<std::pair<Node*, int>, mlir::Value, 4>
      arg_nodes_to_values;
  for (int i = 0, e = arg_types.size(); i < e; ++i) {
    auto& arg_node = arg_nodes[i];
    // The lookup can't fail here: otherwise some nodes in the function haven't
    // be converted to mlir operations and don't have a mapping.
    mlir::Operation* island = node_values_.find(arg_node.node->id())->second;

    auto bb_arg = bb->getArgument(i);
    mlir::Value arg_def = bb_arg;

    if (island->getNumResults() != 2)
      return errors::InvalidArgument(
          "Only feed output tensors of single output nodes are supported");

    // Collect mapping of OutputTensor to associated block arg.
    arg_nodes_to_values.try_emplace({arg_node.node, arg_node.index}, arg_def);
    island->getResult(0).replaceAllUsesWith(arg_def);
    // Erase control outputs from feed.
    auto control_uses = island->getResult(1).getUses();
    for (auto& control_use : llvm::make_early_inc_range(control_uses))
      control_use.getOwner()->eraseOperand(control_use.getOperandNumber());

    if (!arg_node.node->requested_device().empty())
      func.setArgAttr(
          i, "tf.device",
          builder_.getStringAttr(arg_node.node->requested_device()));

    if (arg_node.node->IsArg()) {
      for (const auto& arg_node_attr : arg_node.node->attrs()) {
        const auto& key = arg_node_attr.first;
        // Only import attributes starting with an underscore.
        if (key.empty() || key[0] != '_') continue;
        // Ignore shape inference attributes as shape information is already
        // populated in the result type.
        if (IsOutputShapesAttribute(arg_node_attr.second, key) ||
            IsResourceOutputShapesAttribute(arg_node_attr.second, key))
          continue;
        TF_ASSIGN_OR_RETURN(auto converted_attr,
                            ConvertAttributeValue(arg_node_attr.second));
        func.setArgAttr(i, llvm::formatv("tf.{0}", key).str(), converted_attr);
      }
    }

    island->dropAllReferences();
    island->erase();
  }

  llvm::SmallVector<mlir::Value, 8> inst_to_return;
  for (const auto& ret : ret_nodes) {
    auto* inst = node_values_[ret.node->id()];
    auto op = absl::string_view(ret.node->type_string());
    if (op == FunctionLibraryDefinition::kRetOp ||
        op == FunctionLibraryDefinition::kDeviceRetOp) {
      // Lookup the instruction inside the island
      auto island_op = llvm::cast<mlir::tf_executor::IslandOp>(inst);
      mlir::Operation* inner_op = &island_op.GetBody().front();
      // Remove kRetOp or kDeviceRetOp operation and return its operand.
      // kRetOp and kDeviceRetOp should have just one operand unless they have
      // control dependencies.
      if (inner_op->getNumOperands() != 1)
        return errors::Unimplemented("Return node with multiple inputs.");
      inst_to_return.push_back(inner_op->getOperand(0));
      inst->dropAllReferences();
      inst->erase();
    } else {
      // Lookup and use block arg if fetch is a feed.
      auto it = arg_nodes_to_values.find({ret.node, ret.index});
      if (it != arg_nodes_to_values.end())
        inst_to_return.push_back(it->second);
      else
        inst_to_return.push_back(inst->getResult(ret.index));
    }
  }

  for (Node* control_ret : control_ret_nodes) {
    auto* inst = node_values_[control_ret->id()];
    inst_to_return.push_back(*std::prev(inst->result_end()));
  }

  // Terminate the function by adding a Fetch operation to terminate the graph
  // and a return operation to return the Graph results.
  builder_.setInsertionPointToEnd(&graph_op.body().front());
  builder_.create<mlir::tf_executor::FetchOp>(graph_op.getLoc(),
                                              inst_to_return);
  builder_.setInsertionPointToEnd(bb);
  builder_.create<mlir::ReturnOp>(mlir::UnknownLoc::get(context_),
                                  graph_op.getResults());
  return Status::OK();
}

mlir::Location ImporterBase::GetLocation(const NodeDef& node_def) {
  // TODO(b/142400497): What is the semantic contract for locations?
  const auto& debug_info = debug_info_.traces();

  // Create a location for node `name` in function `function_name`.
  auto create_location = [&](llvm::StringRef name,
                             llvm::StringRef function_name) -> mlir::Location {
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
    auto name_loc_id = mlir::Identifier::get(name_for_name_loc, context_);
    const auto location_it = debug_info.find(debug_info_key);
    if (location_it == debug_info.end()) {
      return mlir::NameLoc::get(name_loc_id, context_);
    }

    // Convert the stack trace to a chain of mlir::CallSiteLocs.
    const auto& trace = location_it->second;
    llvm::SmallVector<mlir::Location, 4> locations;
    locations.reserve(trace.file_line_cols_size());
    for (const auto& location : trace.file_line_cols()) {
      const auto& file = debug_info_.files(location.file_index());
      auto file_name = mlir::Identifier::get(file, context_);
      auto file_line_loc = mlir::FileLineColLoc::get(file_name, location.line(),
                                                     location.col(), context_);
      locations.push_back(file_line_loc);
    }

    // If there are no locations in the stack trace, fall back to just a
    // NameLoc with no child.
    if (locations.empty()) return mlir::NameLoc::get(name_loc_id, context_);

    // Use the front FileLineColLoc to generate a NameLoc.
    mlir::Location node_name_loc =
        mlir::NameLoc::get(name_loc_id, locations.front());

    // If there are more locations then generate a stack trace, otherwise just
    // return the name loc.
    auto callsite_locs = llvm::makeArrayRef(locations).drop_front();
    return callsite_locs.empty()
               ? node_name_loc
               : mlir::CallSiteLoc::get(node_name_loc, callsite_locs);
  };

  // For NextIteration nodes, location is used to pair source and sink nodes.
  // Hence, we use node name as location to keep it unique.
  // TODO(prakalps): In future the plan is to use tokens to pair source/sink
  // nodes. Then NextIteration nodes would not need to be handled separately.
  if (node_def.op() == "NextIteration")
    return create_location(node_def.name(), function_name_for_debug_info_);

  auto original_nodes =
      node_def.experimental_debug_info().original_node_names();
  auto original_funcs =
      node_def.experimental_debug_info().original_func_names();

  if (original_nodes.empty()) {
    return create_location(node_def.name(), function_name_for_debug_info_);
  } else {
    // If the original nodes are defined, then we use them to get a list of
    // call sites, and then fuse them to a single fused location, with the name
    // of the node_def.
    llvm::SmallVector<mlir::Location, 4> node_locations;
    node_locations.reserve(original_nodes.size() + 1);

    // store the names in the experimental_debug_info
    for (int i = 0, e = original_nodes.size(); i != e; ++i) {
      auto node_name = original_nodes[i];
      auto func_name = (i < original_funcs.size()) ? original_funcs[i] : "";
      node_locations.push_back(create_location(node_name, func_name));
    }
    // store the name of the node_def
    node_locations.push_back(
        create_location(node_def.name(), function_name_for_debug_info_));
    return mlir::FusedLoc::get(node_locations, context_);
  }
}

Status ImporterBase::EmitErrorWithLocationStr(const Node& node,
                                              const Status& error_status) {
  const mlir::Location location = GetLocation(node.def());
  mlir::emitError(location);
  return error_handler_.Combine(error_status);
}

mlir::Operation* ImporterBase::CreateOperation(
    const Node& node, llvm::StringRef node_type_name,
    const mlir::OperationState& result,
    const llvm::SmallVectorImpl<mlir::Value>& control_operands,
    bool convert_to_legacy_call) {
  // For the tf.executor specific operations (not wrapped in an island), we
  // have an extra returned value for the control result, and we concatenate
  // control and non-control operands.
  mlir::SmallVector<mlir::Type, 4> types(result.types);
  types.push_back(mlir::tf_executor::ControlType::get(builder_.getContext()));
  mlir::SmallVector<mlir::Value, 4> operands(result.operands);
  operands.append(control_operands.begin(), control_operands.end());

  auto loc = result.location;
  // Dispatch based on the name and create the appropriate operation.
  if (node.IsSwitch()) {
    // Switch and _SwitchN both are in switch class, differentiate based on
    // op name.
    if (node.op_def().name() == "_SwitchN") {
      return builder_.create<mlir::tf_executor::SwitchNOp>(loc, types, operands,
                                                           result.attributes);
    }
    return builder_.create<mlir::tf_executor::SwitchOp>(loc, types, operands,
                                                        result.attributes);
  }
  if (node.IsMerge()) {
    return builder_.create<mlir::tf_executor::MergeOp>(loc, types, operands,
                                                       result.attributes);
  }
  if (node.IsNextIteration()) {
    // NextIteration is a bit special, we create a pair of operations that are
    // linked together through a token returned by the source.
    // We make use of a separate builder to insert the source at the top of
    // the block.
    mlir::OpBuilder builder_at_begin(builder_.getBlock(),
                                     builder_.getBlock()->begin());
    auto source_op =
        builder_at_begin.create<mlir::tf_executor::NextIterationSourceOp>(
            loc, operands[0].getType(), result.attributes);
    return builder_.create<mlir::tf_executor::NextIterationSinkOp>(
        loc, source_op.token(), operands, result.attributes);
  }
  if (node.IsLoopCond()) {
    return builder_.create<mlir::tf_executor::LoopCondOp>(loc, types, operands,
                                                          result.attributes);
  }
  if (node.IsEnter()) {
    return builder_.create<mlir::tf_executor::EnterOp>(loc, types, operands,
                                                       result.attributes);
  }
  if (node.IsExit()) {
    return builder_.create<mlir::tf_executor::ExitOp>(loc, types, operands,
                                                      result.attributes);
  }
  if (node.IsControlTrigger()) {
    return builder_.create<mlir::tf_executor::ControlTriggerOp>(
        loc, operands, result.attributes);
  }
  // Regular TensorFlow operation are wrapped in a tf_executor.island.
  auto island = builder_.create<mlir::tf_executor::IslandOp>(
      result.location, types, control_operands,
      mlir::ArrayRef<mlir::NamedAttribute>{});
  island.body().push_back(new mlir::Block);
  mlir::OpBuilder island_builder =
      mlir::OpBuilder::atBlockEnd(&island.GetBody());

  // Create the operation inside the island now.
  mlir::Operation* inner_op;
  if (convert_to_legacy_call) {
    bool disable_call_shape_inference = false;
    for (const auto& name_and_value : node.attrs()) {
      const auto& attr_name = name_and_value.first;
      const AttrValue& attr_value = name_and_value.second;
      if (IsDisableCallShapeInferenceAttribute(attr_value, attr_name)) {
        disable_call_shape_inference = attr_value.b();
      }
    }

    mlir::BoolAttr attribute =
        builder_.getBoolAttr(disable_call_shape_inference);
    inner_op = island_builder.create<mlir::TF::LegacyCallOp>(
        result.location, result.types, result.operands,
        island_builder.getSymbolRefAttr(node_type_name), attribute);
  } else {
    inner_op = island_builder.createOperation(result);
  }

  // Sets operand_segment_sizes or result_segment_sizes attribute to the op.
  const auto set_segment_sizes_attr =
      [&](const NameRangeMap& arg_ranges,
          const protobuf::RepeatedPtrField<OpDef::ArgDef>& args,
          llvm::StringRef attr_name) {
        std::vector<mlir::Attribute> values;
        values.reserve(args.size());
        for (const auto& arg : args) {
          auto range = arg_ranges.at(arg.name());
          values.push_back(
              island_builder.getI32IntegerAttr(range.second - range.first));
        }
        auto attr_type =
            mlir::VectorType::get(args.size(), builder_.getIntegerType(32));
        auto attr_value = mlir::DenseElementsAttr::get(attr_type, values);
        inner_op->setAttr(attr_name, attr_value);
      };

  if (inner_op->hasTrait<mlir::OpTrait::AttrSizedOperandSegments>() ||
      inner_op->hasTrait<mlir::OpTrait::AttrSizedResultSegments>()) {
    // The op has multiple variadic operands or results.
    // Calculate operand and result segment sizes using the OpDef.
    NameRangeMap input_ranges, output_ranges;
    // This will fail only if the OpDef is syntactically invalid.
    // TODO(jpienaar): Convert this CHECK into a properly propagated error.
    TF_CHECK_OK(
        NameRangesForNode(node, node.op_def(), &input_ranges, &output_ranges));
    if (inner_op->hasTrait<mlir::OpTrait::AttrSizedOperandSegments>()) {
      // Add derived "operand_segment_sizes" attr to the created operation.
      // TODO(b/146937733): Don't use <void> here.
      set_segment_sizes_attr(input_ranges, node.op_def().input_arg(),
                             mlir::OpTrait::AttrSizedOperandSegments<
                                 void>::getOperandSegmentSizeAttr());
    }

    if (inner_op->hasTrait<mlir::OpTrait::AttrSizedResultSegments>()) {
      // Add derived "result_segment_sizes" attr to the created operation.
      // TODO(b/146937733): Don't use <void> here.
      set_segment_sizes_attr(output_ranges, node.op_def().output_arg(),
                             mlir::OpTrait::AttrSizedResultSegments<
                                 void>::getResultSegmentSizeAttr());
    }
  }

  // Add the terminator for the island
  island_builder.create<mlir::tf_executor::YieldOp>(result.location,
                                                    inner_op->getResults());
  return island.getOperation();
}

Status ImporterBase::ConvertNode(const Node& node) {
  if (!node.IsOp()) {
    // Don't import the pseudo-nodes _SOURCE or _SINK. These are added by
    // Graph and don't exist in GraphDef.
    return Status::OK();
  }

  // If it is a custom OP, its definition should be found in the library. We
  // create the MLIR function and insert it to the module if it doesn't exist.
  std::string node_type_name = node.type_string();
  const auto* func_def = graph_flib_.Find(node_type_name);
  bool convert_to_legacy_call = false;
  if (func_def) {
    TF_RETURN_IF_ERROR(ConvertLibFunction(node_type_name));
    node_type_name = (*tf_name_to_mlir_name_)[node_type_name];
    convert_to_legacy_call = true;
  }

  auto get_full_op_name = [&](const std::string& op_name) {
    const char* kTfPrefix = "tf.";
    return kTfPrefix + op_name;
  };

  std::string op_name = get_full_op_name(node_type_name);
  if (back_edge_node_output_.contains(&node)) {
    op_name = op_name + ".sink";
  }

  const auto& node_def = node.def();
  mlir::OperationState result(GetLocation(node_def), op_name);
  for (int i = 0; i < node.num_outputs(); ++i) {
    // The backedge has been removed, so we shouldn't count the corresponding
    // output from the src node when converting to an operation.
    if (back_edge_node_output_.contains(&node) &&
        back_edge_node_output_[&node] == i) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(auto type, InferOutputType(node, i, builder_));
    result.types.push_back(type);
  }

  // Surprisingly input edges can be nondeterministically ordered. This
  // particularly seems to be the case for the control edges between _SOURCE
  // and _SINK that the Graph constructor inserts. Copy the input edges and
  // sort the edges, but only the control edges, not data edges!
  // TODO(jmolloy): We should probably just ignore _SOURCE and _SINK nodes.
  // They'll break roundtripping anyway unless we strip them when converting
  // back to graphdef.
  absl::InlinedVector<const Edge*, 8> in_edges(node.in_edges().size());
  absl::c_copy(node.in_edges(), in_edges.begin());
  absl::c_stable_sort(in_edges, [](const Edge* e1, const Edge* e2) {
    if (e1->IsControlEdge() && !e2->IsControlEdge()) return false;
    if (!e1->IsControlEdge() && e2->IsControlEdge()) return true;
    if (e1->IsControlEdge() && e2->IsControlEdge())
      return e1->src()->id() < e2->src()->id();
    return e1->dst_input() < e2->dst_input();
  });

  result.operands.reserve(in_edges.size());

  // Collect the control operands separately, they will be held by the island.
  mlir::SmallVector<mlir::Value, 8> control_operands;

  for (const auto* input_edge : in_edges) {
    const Node& input_node = *input_edge->src();
    if (input_node.IsSource()) {
      if (in_edges.size() != 1) {
        return errors::FailedPrecondition(
            "The node has other inputs besides the _Source node");
      }
      // We don't import the _SOURCE node.
      continue;
    }
    if (input_node.IsArg() && input_edge->IsControlEdge()) {
      // Currently we have not reached consensus as to what TF function
      // semantics are (b/133509504). Here we assume that all arguments to a
      // function should be available before we start execution of any internal
      // node. This makes the control dependencies between function arguments
      // and internal nodes redundant, and so we do not import them. The TF
      // inliner however assumes no such dependency between function args and
      // internal nodes exists, unless explicitly stated. Since we drop control
      // dependencies here, it leads to loss of information. If the function is
      // inlined later, the inliner would not know of these explicit control
      // dependencies present in the original graph.
      continue;
    }
    if (node_values_.find(input_node.id()) == node_values_.end())
      return errors::FailedPrecondition(
          "Graph not traversed in reverse post order; use seen before def!");
    mlir::Operation* inst = node_values_[input_node.id()];
    if (input_edge->IsControlEdge())
      control_operands.push_back(inst->getResult(inst->getNumResults() - 1));
    else
      result.operands.push_back(inst->getResult(input_edge->src_output()));
  }

  using FuncPairType = std::pair<const std::string*, const AttrValue*>;
  std::vector<FuncPairType> funcs;
  result.attributes.reserve(node.attrs().size() + 2);
  auto abstract_op = result.name.getAbstractOperation();
  auto derived_op =
      abstract_op
          ? abstract_op->getInterface<mlir::DerivedAttributeOpInterface>()
          : nullptr;
  for (const auto& name_and_value : node.attrs()) {
    const auto& attr_name = name_and_value.first;
    // Skip adding derived attributes to the generated op.
    if (derived_op && derived_op->isDerivedAttribute(attr_name)) continue;
    const AttrValue& attr_value = name_and_value.second;

    // Remove _output_shapes attribute that will be added by the exporter.
    if (IsOutputShapesAttribute(attr_value, attr_name)) continue;

    // We represent the _diable_call_shape_inference attribute and remove
    // the _output_shapes attribute for LegacyCall. If a call has other
    // attributes, we can't convert it to LegacyCall.
    if (convert_to_legacy_call &&
        !IsDisableCallShapeInferenceAttribute(attr_value, attr_name)) {
      convert_to_legacy_call = false;
    }
    if (attr_value.value_case() == AttrValue::kFunc) {
      // Attribute iteration order is not defined for protocol buffer Map.
      // Process function attributes separately in the lexicographical order to
      // have deterministic order of functions in the constructed IR.
      funcs.emplace_back(&attr_name, &attr_value);
    } else {
      TF_ASSIGN_OR_RETURN(auto attr, ConvertAttributeValue(attr_value));
      result.attributes.push_back(builder_.getNamedAttr(attr_name, attr));
    }
  }

  auto comparator = [](const FuncPairType& a, const FuncPairType& b) {
    return *a.first < *b.first;
  };
  std::sort(funcs.begin(), funcs.end(), comparator);
  for (const auto& func : funcs) {
    TF_RETURN_IF_ERROR(ConvertFunctionCallAttribute(*func.first, *func.second,
                                                    &result.attributes));
  }

  result.attributes.push_back(builder_.getNamedAttr(
      "device", builder_.getStringAttr(std::string(node_def.device()))));

  // Map If and StatelessIf op in TensorFlow to the common If op in MLIR and add
  // the differentiating attribute.
  if (node.IsIfNode()) {
    result.name = mlir::OperationName(get_full_op_name("If"), context_);
    mlir::BoolAttr val = builder_.getBoolAttr(node_type_name == "StatelessIf");
    result.attributes.push_back(builder_.getNamedAttr("is_stateless", val));
  }

  // Map While and StatelessWhile op in TensorFlow to the common While op in
  // MLIR and add the differentiating attribute.
  if (node.IsWhileNode()) {
    result.name = mlir::OperationName(get_full_op_name("While"), context_);
    mlir::BoolAttr val =
        builder_.getBoolAttr(node_type_name == "StatelessWhile");
    result.attributes.push_back(builder_.getNamedAttr("is_stateless", val));
  }

  // Register the mapping between the TF node and the newly created operation.
  node_values_[node.id()] = CreateOperation(
      node, node_type_name, result, control_operands, convert_to_legacy_call);
  return Status::OK();
}

// Add the backedges to the CFG. Given a backedge, we replace the original
// source and destination operations by two new operations. Most of the
// fields of the replacements are copied from the original operations.
// However,
// - for the src operation, one output is inserted to the front of the output
//   list. The type of the output is set to the type of the non-control result
//   of the dst operation, and
// - for the dst operation, one operand is inserted to the front of the
//   operand list. This operand is using the first result of the src
//   operation.
// TODO(fengliuai): Preserve the order of the results and operands if
// necessary.
Status ImporterBase::AddBackedges() {
  for (auto it : back_edge_dst_inputs_) {
    BackEdge& edge = it.second;
    if (!edge.src->IsNextIteration() || !edge.dst->IsMerge()) {
      return errors::FailedPrecondition(
          "Invalid backedge; should be from NextIteration to Merge!");
    }
    auto* sink = node_values_[edge.src->id()];
    auto* dst = node_values_[edge.dst->id()];
    TF_RETURN_IF_ERROR(AddBackedge(sink, dst, edge.dst_input));
  }
  return Status::OK();
}

Status ImporterBase::AddBackedge(mlir::Operation* sink, mlir::Operation* dst,
                                 int dst_input) {
  // Get the NextIteration.Source operation from the token operand of the sink.
  mlir::Operation* source = sink->getOperand(0).getDefiningOp();

  // Adds the "source" to the operands of the dst by creating a new dst
  // operation.
  mlir::OperationState state(dst->getLoc(), dst->getName());
  auto num_operands = dst->getNumOperands();
  state.operands.reserve(num_operands + 1);
  for (int input = 0, e = num_operands + 1; input != e; ++input) {
    if (input < dst_input) {
      state.operands.push_back(dst->getOperand(input));
    } else if (input == dst_input) {
      state.operands.push_back(source->getResult(0));
    } else {
      state.operands.push_back(dst->getOperand(input - 1));
    }
  }
  state.attributes.assign(dst->getAttrs().begin(), dst->getAttrs().end());
  state.types.assign(dst->getResultTypes().begin(),
                     dst->getResultTypes().end());
  builder_.setInsertionPoint(dst);
  auto* new_dst = builder_.createOperation(state);

  // Replaces the output uses of the old operation by the corresponding
  // result of the new operation, and deletes the old operation.
  for (unsigned i = 0, e = dst->getNumResults(); i != e; ++i) {
    auto new_output = new_dst->getResult(i);
    dst->getResult(i).replaceAllUsesWith(new_output);
  }
  dst->dropAllReferences();
  dst->erase();
  return Status::OK();
}

StatusOr<mlir::FunctionType> ImporterBase::InferLibFunctionType(
    const FunctionBody& fbody) {
  mlir::Builder builder(context_);

  // The FunctionBody contains a graph with a single-output _Arg node for each
  // function argument and a single-input _Retval node for each function return
  // value.
  //
  // We already populated the ShapeRefiner with all the information about the
  // shapes of these graph edges, so we just query it to build the corresponding
  // MLIR function type signature.

  llvm::SmallVector<mlir::Type, 4> arg_types;
  if (specs_.inputs.empty()) {
    arg_types.reserve(fbody.arg_types.size());
    for (auto arg : fbody.arg_nodes) {
      // Find node in the graph using the node id instead of using `arg`
      // directly because the graph has been cloned.
      auto* node = graph_->FindNodeId(arg->id());
      TF_ASSIGN_OR_RETURN(auto type,
                          InferOutputType(*node, /*idx=*/0, builder));
      arg_types.push_back(type);
    }
  } else {
    arg_types.reserve(fbody.arg_types.size());
    for (const auto& it : llvm::enumerate(specs_.inputs)) {
      mlir::Type element_type;
      const auto& node_info = it.value().second;
      DataType dtype = node_info.imported_dtype;
      // Uses the existing output type of the arg node if the data type of the
      // the node isn't specified through the import configuration.
      if (dtype == DT_INVALID) {
        auto arg = fbody.arg_nodes[it.index()];
        auto* node = graph_->FindNodeId(arg->id());
        dtype = node->output_type(0);
        if (dtype == DT_INVALID) {
          return errors::InvalidArgument("Input ", it.index(),
                                         "has invalid data type");
        }
      }
      TF_RETURN_IF_ERROR(
          ::tensorflow::ConvertDataType(dtype, builder, &element_type));
      if (node_info.shape.unknown_rank()) {
        arg_types.push_back(mlir::UnrankedTensorType::get(element_type));
      } else {
        llvm::SmallVector<int64_t, 4> shape;
        TF_RETURN_IF_ERROR(ConvertToMlirShape(node_info.shape, &shape));
        arg_types.push_back(mlir::RankedTensorType::get(shape, element_type));
      }
    }
  }

  llvm::SmallVector<mlir::Type, 4> ret_types;
  ret_types.reserve(fbody.ret_types.size());
  for (auto ret : fbody.ret_nodes) {
    // Find node in the graph using the node id instead of using `ret` directly
    // because the graph has been cloned.
    auto* node = graph_->FindNodeId(ret->id());
    TF_ASSIGN_OR_RETURN(auto type, InferInputType(*node, /*idx=*/0, builder));
    ret_types.push_back(type);
  }

  return builder.getFunctionType(arg_types, ret_types);
}

// Stateful helper class to import a TensorFlow model expressed in GraphDef into
// an MLIR Module.
//
// The nodes defined in the graph are converted to a function called
// 'func_name'. All library function definitions are converted to MLIR functions
// in the module.
class GraphDefImporter : public ImporterBase {
 public:
  // Main entry point: converts the given graph to an MLIR Module.
  static StatusOr<mlir::OwningModuleRef> Convert(
      mlir::MLIRContext* context, const Graph& graph,
      const GraphDebugInfo& debug_info,
      const FunctionLibraryDefinition& flib_def, const GraphImportConfig& specs,
      llvm::StringRef func_name);

 private:
  explicit GraphDefImporter(
      const FunctionLibraryDefinition& flib, const GraphDebugInfo& debug_info,
      const GraphImportConfig& specs, mlir::ModuleOp module,
      std::unordered_map<std::string, std::string>* tf_name_to_mlir_name,
      NameUniquifier* function_name_uniquifier)
      : ImporterBase(flib, debug_info, specs, module, tf_name_to_mlir_name,
                     function_name_uniquifier) {}

  // Returns the function signature of the main function of converted MLIR
  // module, the input nodes and output nodes. The type and shape information
  // for the function arguments are read from `specs`, but the type and shape
  // information for the function returns are inferred by the shape refiner in
  // ImporterBase.
  StatusOr<mlir::FunctionType> InferMainFunctionType(
      const GraphImportConfig& specs, mlir::MLIRContext* context,
      absl::InlinedVector<OutputTensor, 4>* arg_nodes,
      absl::InlinedVector<OutputTensor, 4>* ret_nodes);

  // Returns the function signature of the main function, alongside input and
  // output nodes, for function graphs. Arguments and return values are
  // determined by node op type. Type and shape information of the function are
  // inferred by the shape refiner in ImporterBase.
  StatusOr<mlir::FunctionType> GetArgsRetsAndTypesFromFunctionGraph(
      mlir::MLIRContext* context,
      absl::InlinedVector<OutputTensor, 4>* arg_nodes,
      absl::InlinedVector<OutputTensor, 4>* ret_nodes);

  // Finds the graph's target nodes/function's control ret nodes based on
  // supplied node names in `control_outputs`. If `control_outputs` are not
  // unique or a control ret node is missing, an error will be returned.
  Status GetControlRetsFromGraph(
      llvm::ArrayRef<std::string> control_outputs,
      absl::InlinedVector<Node*, 4>* control_ret_nodes);
};

StatusOr<mlir::OwningModuleRef> GraphDefImporter::Convert(
    mlir::MLIRContext* context, const Graph& graph,
    const GraphDebugInfo& debug_info, const FunctionLibraryDefinition& flib_def,
    const GraphImportConfig& specs, llvm::StringRef func_name) {
  mlir::OwningModuleRef module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  std::unordered_map<std::string, std::string> tf_name_to_mlir_name;
  NameUniquifier function_name_uniquifier(flib_def);

  GraphDefImporter importer(flib_def, debug_info, specs, module.get(),
                            &tf_name_to_mlir_name, &function_name_uniquifier);

  TF_RETURN_IF_ERROR(importer.PrepareConvert(graph));

  mlir::FunctionType func_type;
  absl::InlinedVector<OutputTensor, 4> arg_nodes;
  absl::InlinedVector<OutputTensor, 4> ret_nodes;
  absl::InlinedVector<Node*, 4> control_ret_nodes;
  llvm::SmallVector<mlir::NamedAttribute, 1> attrs;
  if (specs.graph_as_function) {
    if (specs.prune_unused_nodes || !specs.inputs.empty() ||
        !specs.outputs.empty())
      return errors::InvalidArgument(
          "Pruning of graph is currently unsupported when the main graph is "
          "converted to a function.");

    TF_ASSIGN_OR_RETURN(func_type,
                        importer.GetArgsRetsAndTypesFromFunctionGraph(
                            context, &arg_nodes, &ret_nodes));

    TF_RETURN_IF_ERROR(importer.GetControlRetsFromGraph(specs.control_outputs,
                                                        &control_ret_nodes));

    if (!arg_nodes.empty() || !ret_nodes.empty() ||
        !control_ret_nodes.empty()) {
      mlir::Builder b(context);
      std::string s;
      llvm::raw_string_ostream ss(s);
      auto node_name = [&](const OutputTensor& tensor) {
        ss << tensor.node->name();
      };
      llvm::interleave(arg_nodes, ss, node_name, ",");
      auto inputs = b.getNamedAttr("inputs", b.getStringAttr(ss.str()));
      s.clear();
      llvm::interleave(ret_nodes, ss, node_name, ",");
      auto outputs = b.getNamedAttr("outputs", b.getStringAttr(ss.str()));
      s.clear();
      llvm::interleave(specs.control_outputs, ss, ",");
      auto control_outputs =
          b.getNamedAttr("control_outputs", b.getStringAttr(ss.str()));

      attrs.push_back(b.getNamedAttr(
          "tf.entry_function",
          b.getDictionaryAttr({inputs, outputs, control_outputs})));
    }
  } else {
    // Collects the argument and return nodes by looking up the node names
    // specified by the user.
    TF_ASSIGN_OR_RETURN(func_type, importer.InferMainFunctionType(
                                       specs, context, &arg_nodes, &ret_nodes));

    TF_RETURN_IF_ERROR(importer.GetControlRetsFromGraph(specs.control_outputs,
                                                        &control_ret_nodes));

    // TODO(prakalps): Refactor to keep tf.entry_function attribute encoding and
    // decoding in a centralized place.
    // Record the input and output mapping.
    if (!specs.inputs.empty() || !specs.outputs.empty() ||
        !specs.control_outputs.empty()) {
      mlir::Builder b(context);
      std::string s;
      llvm::raw_string_ostream ss(s);
      llvm::interleave(
          specs.inputs, ss,
          [&](const std::pair<std::string, ArrayInfo>& v) { ss << v.first; },
          ",");
      auto inputs = b.getNamedAttr("inputs", b.getStringAttr(ss.str()));
      s.clear();
      llvm::interleave(specs.outputs, ss, ",");
      auto outputs = b.getNamedAttr("outputs", b.getStringAttr(ss.str()));
      s.clear();
      llvm::interleave(specs.control_outputs, ss, ",");
      auto control_outputs =
          b.getNamedAttr("control_outputs", b.getStringAttr(ss.str()));

      attrs.push_back(b.getNamedAttr(
          "tf.entry_function",
          b.getDictionaryAttr({inputs, outputs, control_outputs})));
    }
  }

  // Record version info.
  PopulateTfVersions(module.get(), graph.versions());

  TF_RETURN_IF_ERROR(importer.ImporterBase::Convert(
      func_name, func_type, arg_nodes, ret_nodes, control_ret_nodes, attrs,
      specs.graph_as_function));
  return module;
}

StatusOr<mlir::FunctionType> GraphDefImporter::InferMainFunctionType(
    const GraphImportConfig& specs, mlir::MLIRContext* context,
    absl::InlinedVector<OutputTensor, 4>* arg_nodes,
    absl::InlinedVector<OutputTensor, 4>* ret_nodes) {
  // Find all the input nodes and output nodes.
  // Feeds have been remapped to single output nodes (Placeholder), so an exact
  // name match is sufficient.
  absl::flat_hash_map<absl::string_view, int> inputs;
  for (auto input_and_idx : llvm::enumerate(specs.inputs)) {
    TensorId tensor = ParseTensorName(input_and_idx.value().first);
    auto remapped_it = remapped_feeds_.find(tensor);
    if (remapped_it != remapped_feeds_.end()) {
      inputs.insert({remapped_it->second, input_and_idx.index()});
    } else {
      inputs.insert({tensor.node(), input_and_idx.index()});
    }
  }

  absl::flat_hash_set<absl::string_view> output_node_names;
  std::vector<TensorId> outputs;
  output_node_names.reserve(specs.outputs.size());
  for (const auto& output : specs.outputs) {
    TensorId tensor = ParseTensorName(output);
    auto remapped_it = remapped_feeds_.find(tensor);
    if (remapped_it != remapped_feeds_.end()) {
      output_node_names.insert(remapped_it->second);
      outputs.push_back({remapped_it->second, 0});
    } else {
      output_node_names.insert(tensor.node());
      outputs.push_back(tensor);
    }
  }

  if (!inputs.empty() || !outputs.empty()) {
    arg_nodes->resize(inputs.size());
    ret_nodes->resize(outputs.size());

    for (Node* n : GetOrderedNodes()) {
      // Handle inputs/arguments.
      auto input_it = inputs.find(n->name());
      if (input_it != inputs.end()) {
        (*arg_nodes)[input_it->second] = {n, 0};
      }

      // Handle outputs/returns.
      if (output_node_names.contains(n->name())) {
        for (int i = 0, e = outputs.size(); i != e; ++i) {
          TensorId tensor = outputs[i];
          if (n->name() != tensor.node()) continue;
          (*ret_nodes)[i] = {n, tensor.index()};
        }
      }
    }
  }

  // Starts to construct the function type.
  mlir::Builder builder(context);
  llvm::SmallVector<mlir::Type, 4> arg_types;
  arg_types.reserve(specs.inputs.size());
  int i = 0;
  for (const auto& it : specs.inputs) {
    Node* arg_node = arg_nodes->at(i).node;
    if (arg_node == nullptr) {
      return errors::InvalidArgument("Input ", it.first,
                                     " was not found in graph");
    }
    mlir::Type element_type;
    const auto& node_info = it.second;
    DataType imported_dtype = node_info.imported_dtype;
    // Uses the existing output type of the arg node if the data type of the
    // the node isn't specified through the import configuration.
    if (imported_dtype == DT_INVALID) {
      imported_dtype = arg_node->output_type(0);
      if (imported_dtype == DT_INVALID) {
        return errors::InvalidArgument("Input ", i, "has invalid data type");
      }
    }
    TF_RETURN_IF_ERROR(
        ::tensorflow::ConvertDataType(imported_dtype, builder, &element_type));
    if (node_info.shape.unknown_rank()) {
      arg_types.push_back(mlir::UnrankedTensorType::get(element_type));
    } else {
      llvm::SmallVector<int64_t, 4> shape;
      TF_RETURN_IF_ERROR(ConvertToMlirShape(node_info.shape, &shape));
      arg_types.push_back(mlir::RankedTensorType::get(shape, element_type));
    }
    i++;
  }

  llvm::SmallVector<mlir::Type, 4> ret_types;
  ret_types.reserve(specs.outputs.size());
  for (int i = 0, e = specs.outputs.size(); i != e; ++i) {
    if (ret_nodes->at(i).node == nullptr) {
      return errors::InvalidArgument("Output ", specs.outputs[i],
                                     " was not found in graph");
    }
  }
  for (const auto& ret : *ret_nodes) {
    if (ret.node->num_outputs() <= ret.index) {
      return errors::InvalidArgument("Invalid output index ", ret.index,
                                     " specified for node: ", ret.node->name());
    }
    TF_ASSIGN_OR_RETURN(auto type,
                        InferOutputType(*ret.node, ret.index, builder));
    ret_types.push_back(type);
  }

  return builder.getFunctionType(arg_types, ret_types);
}

StatusOr<mlir::FunctionType>
GraphDefImporter::GetArgsRetsAndTypesFromFunctionGraph(
    mlir::MLIRContext* context, absl::InlinedVector<OutputTensor, 4>* arg_nodes,
    absl::InlinedVector<OutputTensor, 4>* ret_nodes) {
  auto add_node = [](Node* node, absl::InlinedVector<OutputTensor, 4>* nodes) {
    auto* attr = node->attrs().Find("index");
    if (!attr)
      return errors::InvalidArgument(node->type_string(), " node '",
                                     node->name(),
                                     "' is missing attribute 'index'");

    auto index = attr->i();
    if (nodes->size() < index + 1) nodes->resize(index + 1);

    if ((*nodes)[index].node != nullptr)
      return errors::InvalidArgument(node->type_string(), " node '",
                                     node->name(), "' has attribute 'index' ",
                                     index, " that conflicts with node '",
                                     (*nodes)[index].node->name(), "'");
    (*nodes)[index] = {node, 0};

    return Status::OK();
  };

  // Collect arg and ret nodes from graph.
  for (auto* node : GetOrderedNodes())
    if (node->IsArg())
      TF_RETURN_IF_ERROR(add_node(node, arg_nodes));
    else if (node->IsRetval())
      TF_RETURN_IF_ERROR(add_node(node, ret_nodes));

  // Collect arg and ret types and create function type.
  mlir::Builder builder(context);
  llvm::SmallVector<mlir::Type, 4> arg_types;
  arg_types.reserve(arg_nodes->size());
  for (auto arg_node_and_idx : llvm::enumerate(*arg_nodes)) {
    auto& arg_node = arg_node_and_idx.value();
    if (arg_node.node == nullptr)
      return errors::InvalidArgument("Graph missing _Arg at index ",
                                     arg_node_and_idx.index());

    TF_ASSIGN_OR_RETURN(auto type,
                        InferOutputType(*arg_node.node, /*idx=*/0, builder));
    arg_types.push_back(type);
  }

  llvm::SmallVector<mlir::Type, 4> ret_types;
  ret_types.reserve(ret_nodes->size());
  for (auto ret_node_and_idx : llvm::enumerate(*ret_nodes)) {
    auto& ret_node = ret_node_and_idx.value();
    if (ret_node.node == nullptr)
      return errors::InvalidArgument("Graph missing _Retval at index ",
                                     ret_node_and_idx.index());

    TF_ASSIGN_OR_RETURN(auto type,
                        InferInputType(*ret_node.node, /*idx=*/0, builder));
    ret_types.push_back(type);
  }

  return builder.getFunctionType(arg_types, ret_types);
}

Status GraphDefImporter::GetControlRetsFromGraph(
    llvm::ArrayRef<std::string> control_outputs,
    absl::InlinedVector<Node*, 4>* control_ret_nodes) {
  if (control_outputs.empty()) return Status::OK();

  llvm::SmallDenseMap<llvm::StringRef, int32_t> controls_to_idx;
  for (auto control_and_idx : llvm::enumerate(control_outputs))
    controls_to_idx.insert({control_and_idx.value(), control_and_idx.index()});

  if (controls_to_idx.size() != control_outputs.size())
    return errors::InvalidArgument("Control outputs must be unique");

  control_ret_nodes->resize(controls_to_idx.size());

  for (auto* node : GetOrderedNodes()) {
    auto it = controls_to_idx.find(node->name());
    if (it != controls_to_idx.end()) (*control_ret_nodes)[it->second] = node;
  }

  for (auto node_and_name : llvm::zip(*control_ret_nodes, control_outputs))
    if (std::get<0>(node_and_name) == nullptr)
      return errors::InvalidArgument(
          "Control output '", std::get<1>(node_and_name), "' is missing");

  return Status::OK();
}

// Stateful helper class to import a TensorFlow model expressed in SavedModel
// into an MLIR Module.
class SavedModelObjectGraphImporter : public ImporterBase {
 public:
  // Main entry point: converts all functions in the given meta graph to an MLIR
  // Module.
  static StatusOr<mlir::OwningModuleRef> Convert(
      SavedModelV2Bundle* saved_model, absl::Span<std::string> exported_names,
      mlir::MLIRContext* context, bool add_default_attributes);

 private:
  explicit SavedModelObjectGraphImporter(
      const FunctionLibraryDefinition& flib, const GraphDebugInfo& debug_info,
      const GraphImportConfig& specs, mlir::ModuleOp module,
      std::unordered_map<std::string, std::string>* tf_name_to_mlir_name,
      NameUniquifier* function_name_uniquifier)
      : ImporterBase(flib, debug_info, specs, module, tf_name_to_mlir_name,
                     function_name_uniquifier) {}
};

// Determines the names used to reference objects in the SavedObjectGraph.
class ObjectNames {
 public:
  explicit ObjectNames(const SavedObjectGraph& object_graph,
                       absl::Span<std::string> exported_names);

  // Gets the names that external users of the SavedModel can use to refer to
  // this node.
  llvm::ArrayRef<llvm::StringRef> GetExportedNames(int node_id) const;

  // Gets the name in the module symbol table for this node.
  // This name is only used for internal IR references.
  llvm::StringRef GetSymbolTableName(int node_id) const;

 private:
  // In the absence of any other information, use this name as the symbol table
  // name for this node.
  std::string GetDefaultSymbolTableName(int node_id) const;
  // Determines if a name is exported.
  bool IsExported(const std::string& name);
  // Main object graph traversal function.
  void RecursivelyVisitObjectGraph(int node_id);
  // Gets a stable StringRef from a std::string.
  llvm::StringRef SaveString(const std::string& s) const;

  // The object graph we are traversing.
  const SavedObjectGraph& object_graph_;
  // The set of names to export. Empty means "export all".
  std::unordered_set<std::string> names_to_export_;

  // When we recursively follow the object graph tree structure from the root,
  // we track its path in the object graph by pushing and popping from here
  // during traversal.
  llvm::SmallVector<std::string, 8> path_segments_;
  // The set of node_id's that are on the current DFS stack.
  // For cyclic object graphs, this prevents infinite recursion.
  std::unordered_set<int> on_stack_nodes_;

  // Key: node_id.
  // Value: all object names that node_id appears as.
  // Each object name corresponds to a unique path from the root of the object
  // graph.
  // The common intuitive case is when there is only one name for a given
  // object, which corresponds to the object graph being a tree.
  //
  // But, there cases where the object graph is a general graph. For
  // example, this happens commonly in Keras models, where `foo.bar` is
  // also reachable via the name `keras_api.foo.bar`.
  // Cycles are possible too.
  absl::flat_hash_map<int, std::vector<std::string>> object_names_;

  // Key: node_id
  // Value: all names that this object is exported as
  absl::flat_hash_map<int, llvm::SmallVector<llvm::StringRef, 1>>
      exported_names_;
  // Key: node_id
  // Value: pretty symbol table name to use for internal references to this
  // object.
  absl::flat_hash_map<int, llvm::StringRef> pretty_symbol_table_name_;

  // Stable strings we can take StringRef's into. Used only by the SaveString
  // method.
  mutable std::unordered_set<std::string> saved_strings_;
};

ObjectNames::ObjectNames(const SavedObjectGraph& object_graph,
                         absl::Span<std::string> exported_names)
    : object_graph_(object_graph),
      names_to_export_(exported_names.begin(), exported_names.end()) {
  // Visit all reachable nodes from the root of the object graph.
  // This builds up object_names_ to contain all names like `foo.bar` that a
  // particular node in the graph can be reached from.
  RecursivelyVisitObjectGraph(/*node_id=*/0);

  // Populate the exported_names_ map.
  // TODO(silvasean): Diagnose typos in exported names?
  for (auto& kv : object_names_) {
    // Make object names map independent of our particular choice of object
    // graph traversal.
    std::sort(kv.second.begin(), kv.second.end(),
              [](absl::string_view a, absl::string_view b) {
                // The sort order here influences the "pretty name" we assign
                // below. We want the most debuggable name to be first.
                //
                // Debuggability heuristics:
                // 1. Names that end in digits are likely to be internal aliases
                // to the "real" names.
                // 2. Longer names are more likely to be internal aliases.
                //
                // Example set of object names created by Keras for the weight
                // matrix of a fully connected layer on a trivial FC mnist
                // model:
                // - `model.layer-1.kernel` (this is the "best" name)
                // - `model.keras_api.layers.1.kernel`
                // - `model.variables.0`
                // - `model.keras_api.layers.1.keras_api.trainable_variables.0`
                // - ... 10 more long aliases ending in digits ...
                return std::make_tuple(isdigit(a.back()), a.size(), a) <
                       std::make_tuple(isdigit(b.back()), b.size(), b);
              });
    for (const std::string& name : kv.second) {
      if (IsExported(name)) {
        exported_names_[kv.first].push_back(SaveString(name));
      }
    }
  }
  // Create "pretty" symbol table names for nodes where that is applicable.
  // We could make all symbol table names use the default, which is basically
  // just the node id. But for debugging purposes, it's nicer if we can mix in
  // a recognizable object name if we have the information to do so.
  for (auto& kv : object_names_) {
    int node_id = kv.first;
    std::string internal_name =
        absl::StrCat(GetDefaultSymbolTableName(node_id), "__");
    // If the object has an exported name, we prefer that since it is probably
    // the most recognizable. Otherwise, we grab some non-exported name of the
    // object.
    if (exported_names_.find(node_id) != exported_names_.end()) {
      internal_name += exported_names_[node_id][0].str();
    } else {
      internal_name += object_names_[node_id][0];
    }
    pretty_symbol_table_name_[node_id] = SaveString(internal_name);
  }
}

llvm::ArrayRef<llvm::StringRef> ObjectNames::GetExportedNames(
    int node_id) const {
  auto it = exported_names_.find(node_id);
  if (it != exported_names_.end()) {
    return it->second;
  }
  return {};
}

llvm::StringRef ObjectNames::GetSymbolTableName(int node_id) const {
  auto it = pretty_symbol_table_name_.find(node_id);
  if (it != pretty_symbol_table_name_.end()) {
    return it->second;
  }
  return SaveString(GetDefaultSymbolTableName(node_id));
}

std::string ObjectNames::GetDefaultSymbolTableName(int node_id) const {
  return absl::StrCat("__sm_node", node_id);
}

bool ObjectNames::IsExported(const std::string& name) {
  if (names_to_export_.empty()) {
    return true;
  }
  return names_to_export_.find(name) != names_to_export_.end();
}

void ObjectNames::RecursivelyVisitObjectGraph(int node_id) {
  const SavedObject& object = object_graph_.nodes(node_id);

  switch (object.kind_case()) {
    case SavedObject::kConstant:
    case SavedObject::kFunction:
    case SavedObject::kVariable: {
      object_names_[node_id].push_back(absl::StrJoin(path_segments_, "."));
      break;
    }
    default:
      break;
  }

  for (const auto& child_ref : object.children()) {
    bool on_stack = !on_stack_nodes_.insert(child_ref.node_id()).second;
    if (on_stack) {
      // This is a backedge. Don't traverse it.
      continue;
    }

    path_segments_.push_back(child_ref.local_name());
    RecursivelyVisitObjectGraph(child_ref.node_id());
    path_segments_.pop_back();

    on_stack_nodes_.erase(child_ref.node_id());
  }
}

llvm::StringRef ObjectNames::SaveString(const std::string& s) const {
  return llvm::StringRef(*saved_strings_.insert(s).first);
}

// Extracts a TensorProto for a Const op from a GraphDef, given an op_name.
// Returns nullptr on not found or other mismatch.
// This returns a pointer to the actual node within the graph_def so as to
// avoid expensive copies.
const TensorProto* ExtractConstTensorFromGraph(const GraphDef& graph_def,
                                               const std::string& op_name) {
  const NodeDef* match_node = nullptr;
  for (const auto& node : graph_def.node()) {
    if (node.name() == op_name) {
      match_node = &node;
    }
  }

  if (!match_node) {
    return nullptr;
  }

  auto value_it = match_node->attr().find("value");
  if (value_it == match_node->attr().end()) {
    return nullptr;
  }

  if (!value_it->second.has_tensor()) {
    return nullptr;
  }

  return &value_it->second.tensor();
}

const TrackableObjectGraph::TrackableObject::SerializedTensor*
FindSerializedTensorInTrackable(
    const TrackableObjectGraph::TrackableObject& trackable_object,
    StringPiece name) {
  for (const auto& maybe_serialized_tensor : trackable_object.attributes()) {
    if (maybe_serialized_tensor.name() == name) {
      return &maybe_serialized_tensor;
    }
  }
  return nullptr;
}

Status DiagnoseMultipleConcreteFunctions(const SavedObjectGraph& object_graph,
                                         const ObjectNames& object_names) {
  for (int node_id = 0; node_id < object_graph.nodes_size(); node_id++) {
    const SavedObject& object = object_graph.nodes(node_id);
    if (object_names.GetExportedNames(node_id).empty()) {
      continue;
    }
    if (object.kind_case() == SavedObject::kFunction) {
      // We only allow a single input signature to each SavedFunction.
      // This assumption means we have a 1:1 correspondence between
      // tf.function <=> SavedFunction <=> SavedConcreteFunction <=> FunctionDef
      // This makes defining the ABI easier (or even well-defined at all).
      // TODO(silvasean): How to detect a function that doesn't have an
      // explicitly user-provided input signature, but happens to have been
      // traced exactly once?
      if (object.function().concrete_functions_size() != 1) {
        llvm::SmallVector<std::string, 4> names;
        for (llvm::StringRef s : object_names.GetExportedNames(node_id)) {
          names.push_back("'" + s.str() + "'");
        }
        return errors::InvalidArgument(
            "Exported function with exported name(s) ",
            absl::StrJoin(names, ", "),
            " with multiple concrete functions. Add "
            "@tf.function(input_signature=[...]) on this function, or use a "
            "narrower list of exported names that excludes this function.");
      }
    }
  }
  return Status::OK();
}

// Recursively traverses a StructuredValue, linearizing all the leaves.
//
// This currently only handles the subset of StructuredValue that is needed for
// signatures.
//
// Given a StructuredValue with structure [{"x": leaf0}], the "index path"
// needed to reach leaf0 is `[0, "x"]`, as it would be if you were operating on
// a Python object (`obj[0]["x"] is leaf0`). Each leaf corresponds to a
// linearized function argument or return on a FunctionDef, and hence to an
// mlir::FuncOp argument / return.
//
// This must match the linearization that happens in `tf.nest.flatten`.
// In particular, dict values should be linearized in sorted key order.
//
// The linearized index paths can be returned back to a structured
// representation (e.g. to emit C structs matching a signature) with a simple
// algorithm that recurses on each run of index paths with identical first
// elements.
class StructuredValueLinearizer {
 public:
  StructuredValueLinearizer(const StructuredValue& value,
                            mlir::MLIRContext* context);

  // Returns the list of index paths to each leaf of the StructuredValue,
  // in a linearized order matching `tf.nest.flatten`.
  //
  // If an error occurred during the linearization process, an error message
  // with `error_context` prepended will be included in the returned status.
  StatusOr<llvm::ArrayRef<mlir::ArrayAttr>> GetLeafIndexPaths(
      llvm::StringRef error_context) const;

 private:
  // Main function that recursively traverses the StructuredValue.
  void RecursivelyFindLeaves(const StructuredValue& value);

  mlir::Builder builder_;
  // The current index path. We push/pop this during recursive traversal of the
  // StructuredValue.
  llvm::SmallVector<mlir::Attribute, 4> current_index_path_;
  // The list of leaf index paths we have discovered so far.
  llvm::SmallVector<mlir::ArrayAttr, 4> leaf_index_paths_;
  // If non-empty, an error message to report.
  std::string error_message_;
};

StructuredValueLinearizer::StructuredValueLinearizer(
    const StructuredValue& value, mlir::MLIRContext* context)
    : builder_(context) {
  RecursivelyFindLeaves(value);
}

StatusOr<llvm::ArrayRef<mlir::ArrayAttr>>
StructuredValueLinearizer::GetLeafIndexPaths(
    llvm::StringRef error_context) const {
  if (error_message_.empty()) {
    return llvm::makeArrayRef(leaf_index_paths_);
  }
  return errors::InvalidArgument(
      error_context.str(), error_message_,
      "This likely means that you have @tf.function "
      "on an exported function instead of "
      "@tf.function(input_signature=[...]). Consider annotating an "
      "input_signature or narrowing your set of "
      "exported names to not include this function.");
}

void StructuredValueLinearizer::RecursivelyFindLeaves(
    const StructuredValue& value) {
  switch (value.kind_case()) {
    case StructuredValue::kDictValue: {
      // Dict values must be linearized in sorted order of keys.
      const DictValue& dict = value.dict_value();
      using FieldTy = protobuf::MapPair<std::string, StructuredValue>;
      llvm::SmallVector<const FieldTy*, 4> fields;
      for (auto& field : dict.fields()) {
        fields.push_back(&field);
      }
      llvm::sort(fields, [](const FieldTy* a, const FieldTy* b) {
        return a->first < b->first;
      });
      for (auto& field : fields) {
        current_index_path_.push_back(builder_.getStringAttr(field->first));
        RecursivelyFindLeaves(field->second);
        current_index_path_.pop_back();
      }
      return;
    }
    case StructuredValue::kTupleValue: {
      const TupleValue& tuple = value.tuple_value();
      for (int i = 0, e = tuple.values_size(); i < e; i++) {
        current_index_path_.push_back(builder_.getI64IntegerAttr(i));
        RecursivelyFindLeaves(tuple.values(i));
        current_index_path_.pop_back();
      }
      return;
    }
    // We don't differentiate between tuples and lists.
    case StructuredValue::kListValue: {
      const ListValue& list = value.list_value();
      for (int i = 0, e = list.values_size(); i < e; i++) {
        current_index_path_.push_back(builder_.getI64IntegerAttr(i));
        RecursivelyFindLeaves(list.values(i));
        current_index_path_.pop_back();
      }
      return;
    }
    case StructuredValue::kTensorSpecValue: {
      // Base case: record the current path stack as the index path needed to
      // get to this leaf.
      leaf_index_paths_.push_back(builder_.getArrayAttr(current_index_path_));
      return;
    }
    case StructuredValue::kNoneValue: {
      // Base case: do nothing.
      // This arises, for example, as the top-level object of an output
      // signature when there are no return values.
      return;
    }
    default: {
      llvm::raw_string_ostream os(error_message_);
      // TODO(silvasean): Use an enumerant name string instead of a number.
      os << "Unhandled structured value kind " << value.kind_case()
         << " at index path: <value>";
      for (auto path_element : current_index_path_) {
        os << ".";
        if (auto integer = path_element.dyn_cast<mlir::IntegerAttr>()) {
          os << integer.getValue();
        } else {
          auto str = path_element.cast<mlir::StringAttr>();
          os << str.getValue();
        }
      }
      os << "\n";
    }
  }
}

// For exported functions with bound inputs, rewrite the function
// signature to match the requirements of tf_saved_model bound input args.
//
// The raw imported functions have `tensor<*x!tf.resource>` as the type for
// mutable bound inputs and `tensor<...>` as the type for immutable
// bound inputs. Here we canonicalize both of them into
// `tensor<!tf.resource<tensor<...>>>`.
void AdjustBoundInputArgTypes(mlir::ModuleOp module) {
  mlir::SymbolTable symbol_table(module);
  for (auto func : module.getOps<mlir::FuncOp>()) {
    if (!mlir::tf_saved_model::IsExported(func)) continue;
    mlir::OpBuilder builder(func.getBody());
    llvm::SmallVector<mlir::Type, 4> new_input_types;
    for (int i = 0, e = func.getNumArguments(); i < e; i++) {
      auto arg = func.front().getArgument(i);
      auto global_tensor =
          mlir::tf_saved_model::LookupBoundInput(func, i, symbol_table);
      if (global_tensor) {
        auto old_type = arg.getType();
        auto new_type =
            mlir::tf_saved_model::GetBoundInputArgTypeFor(global_tensor);
        arg.setType(new_type);
        if (global_tensor.is_mutable()) {
          auto arg_with_original_type = builder.create<mlir::TF::CastOp>(
              global_tensor.getLoc(), old_type, arg,
              /*Truncate=*/builder.getBoolAttr(false));
          arg.replaceAllUsesWith(arg_with_original_type);
          // The RAUW replaces the arg with itself, so we need to set it back.
          arg_with_original_type.setOperand(arg);
        } else {
          auto arg_with_original_type =
              builder.create<mlir::TF::ReadVariableOp>(global_tensor.getLoc(),
                                                       old_type, arg);
          arg.replaceAllUsesWith(arg_with_original_type);
          // The RAUW replaces the arg with itself, so we need to set it back.
          arg_with_original_type.setOperand(arg);
        }
      }
      new_input_types.push_back(arg.getType());
    }
    func.setType(mlir::FunctionType::get(
        new_input_types, func.getType().getResults(), module.getContext()));
  }
}

// Reorder the ops in the module to make testing easier and less dependent
// on implementation details such as the order of functions in the
// FunctionDefLibrary.
//
// The order this ensures is:
// 1. GlobalTensorOp's
// 2. FuncOps's.
//
// Within each of 1. and 2., ops are sorted by exported name (if
// available, and only the first exported name is considered), followed by
// non-exported ops.
void SortSavedModelModule(mlir::ModuleOp module) {
  struct NamedGlobalTensor {
    llvm::StringRef name;
    GlobalTensorOp global_tensor;
  };
  llvm::SmallVector<NamedGlobalTensor, 8> named_global_tensors;
  for (auto global_tensor : module.getOps<GlobalTensorOp>()) {
    auto exported_names = mlir::tf_saved_model::GetExportedNames(global_tensor);
    // We use stable_sort, so duplicate empty names are fine here.
    named_global_tensors.push_back(
        {exported_names.empty() ? "" : exported_names.front(), global_tensor});
  }
  llvm::stable_sort(named_global_tensors,
                    [](const NamedGlobalTensor& a, const NamedGlobalTensor& b) {
                      return std::make_tuple(a.name.empty(), a.name) <
                             std::make_tuple(b.name.empty(), b.name);
                    });

  struct NamedFunc {
    llvm::StringRef name;
    mlir::FuncOp func;
  };
  llvm::SmallVector<NamedFunc, 8> named_funcs;
  for (auto func : module.getOps<mlir::FuncOp>()) {
    auto exported_names = mlir::tf_saved_model::GetExportedNames(func);
    named_funcs.push_back(
        {exported_names.empty() ? "" : exported_names.front(), func});
  }
  llvm::stable_sort(named_funcs, [](const NamedFunc& a, const NamedFunc& b) {
    return std::make_tuple(a.name.empty(), a.name) <
           std::make_tuple(b.name.empty(), b.name);
  });

  // Move onto the front of the module in reverse of the final desired order.
  for (auto named_func : llvm::reverse(named_funcs)) {
    named_func.func.getOperation()->moveBefore(&module.getBody()->front());
  }
  for (auto named_global_tensor : llvm::reverse(named_global_tensors)) {
    named_global_tensor.global_tensor.getOperation()->moveBefore(
        &module.getBody()->front());
  }
}

Status CreateSavedModelIR(
    const ObjectNames& object_names, mlir::ModuleOp module,
    const SavedObjectGraph& object_graph,
    const std::unordered_map<std::string, std::string>& tf_name_to_mlir_name,
    SavedModelV2Bundle* saved_model) {
  mlir::OpBuilder builder(module.getBodyRegion());
  mlir::SymbolTable symbol_table(module);

  // Create a side data-structure, indexed by the object_graph node_id to
  // a TrackableObject that is restorable.
  absl::flat_hash_map<int, const TrackableObjectGraph::TrackableObject*>
      restored_objects;
  TF_RETURN_IF_ERROR(saved_model->VisitObjectsToRestore(
      [&](int saved_node_id,
          const TrackableObjectGraph::TrackableObject& trackable_object) {
        restored_objects.insert(
            std::make_pair(saved_node_id, &trackable_object));
        return Status::OK();
      }));

  for (int node_id = 0; node_id < object_graph.nodes_size(); node_id++) {
    const SavedObject& object = object_graph.nodes(node_id);
    // For correctness, we cannot import functions that don't have exported
    // names, since they don't necessarily have a well-defined ABI (diagnosed
    // earlier).
    //
    // For variables/constants, pruning them is purely an optimization,
    // and more complicated since it requires use-def analysis of which
    // functions use which variables/constants, so we don't do anything
    // special for them here as part of our initial IR construction.
    if (object.kind_case() == SavedObject::kFunction) {
      if (object_names.GetExportedNames(node_id).empty()) {
        continue;
      }
      std::string error_context =
          "While importing SavedModel function '" +
          object_names.GetExportedNames(node_id)[0].str() + "': ";
      const SavedFunction& function = object.function();
      auto orig_func = symbol_table.lookup<mlir::FuncOp>(
          tf_name_to_mlir_name.find(function.concrete_functions(0))->second);
      mlir::FuncOp func = orig_func;
      // If there are potentially references to this func from within the
      // module, create a wrapper around it and decorate the wrapper with the
      // tf_saved_model attributes instead.
      if (!mlir::SymbolTable::symbolKnownUseEmpty(orig_func.getName(),
                                                  &module.getBodyRegion())) {
        func = orig_func.cloneWithoutRegions();
        module.insert(module.getBody()->begin(), func);
        func.addEntryBlock();
        func.setName("__sm_exported_" + orig_func.getName().str());
        llvm::SmallVector<mlir::Value, 4> args_as_values;
        for (auto block_argument : func.getArguments()) {
          args_as_values.push_back(block_argument);
        }
        mlir::OpBuilder body_builder(&func.getBody());
        auto call = body_builder.create<mlir::TF::StatefulPartitionedCallOp>(
            func.getLoc(), orig_func.getType().getResults(), args_as_values,
            builder.getSymbolRefAttr(orig_func.getName()),
            /*config=*/builder.getStringAttr(""),
            /*config_proto=*/builder.getStringAttr(""),
            /*executor_type=*/builder.getStringAttr(""));
        body_builder.create<mlir::ReturnOp>(func.getLoc(), call.getResults());
      }
      func.setAttr(
          "tf_saved_model.exported_names",
          builder.getStrArrayAttr(object_names.GetExportedNames(node_id)));
      const SavedConcreteFunction& concrete_function =
          object_graph.concrete_functions().at(function.concrete_functions(0));

      // We do not handle the other element of this tuple, which corresponds to
      // Python kwonlyargs, since currently TensorFlow prohibits this in
      // combination with input_signature:
      // https://github.com/tensorflow/tensorflow/blob/8cb8627abb5ef83a6fba34f8fd0e4ee430562eb1/tensorflow/python/eager/function.py#L2027-L2030
      // Our SavedModel import requires input_signature on the tf.function, so
      // we never need to handle the kwonlyargs.
      auto positional_arg_structure =
          concrete_function.canonicalized_input_signature()
              .tuple_value()
              .values(0);
      StructuredValueLinearizer input_linearizer(positional_arg_structure,
                                                 builder.getContext());

      int bound_input_base =
          func.getNumArguments() - concrete_function.bound_inputs_size();
      TF_ASSIGN_OR_RETURN(auto input_index_paths,
                          input_linearizer.GetLeafIndexPaths(
                              error_context + "in input signature: "));
      if (bound_input_base != input_index_paths.size()) {
        return errors::InvalidArgument(
            error_context,
            "Argument mismatch between concrete function input signature "
            "vs underlying FunctionDef for concrete function '",
            function.concrete_functions(0), "' (", input_index_paths.size(),
            " vs ", bound_input_base, ")");
      }
      for (auto index_path : llvm::enumerate(input_index_paths)) {
        func.setArgAttr(index_path.index(), "tf_saved_model.index_path",
                        index_path.value());
      }

      for (auto& bound_input :
           llvm::enumerate(concrete_function.bound_inputs())) {
        int arg_index = bound_input_base + bound_input.index();
        auto symbol_ref = builder.getSymbolRefAttr(
            object_names.GetSymbolTableName(bound_input.value()));
        func.setArgAttr(arg_index, "tf_saved_model.bound_input", symbol_ref);
      }

      StructuredValueLinearizer output_linearizer(
          concrete_function.output_signature(), builder.getContext());
      TF_ASSIGN_OR_RETURN(auto output_index_paths,
                          output_linearizer.GetLeafIndexPaths(
                              error_context + "in output signature: "));
      if (func.getNumResults() != output_index_paths.size()) {
        return errors::InvalidArgument(
            error_context,
            "Result mismatch between concrete function output signature "
            "vs underlying FunctionDef for concrete function '",
            function.concrete_functions(0), "' (", output_index_paths.size(),
            " vs ", func.getNumResults(), ")");
      }
      for (auto index_path : llvm::enumerate(output_index_paths)) {
        func.setResultAttr(index_path.index(), "tf_saved_model.index_path",
                           index_path.value());
      }
    } else if (object.kind_case() == SavedObject::kVariable) {
      const SavedVariable& variable = object.variable();
      // Find the trackable in the side data structure.
      auto variable_trackable_it = restored_objects.find(node_id);
      if (variable_trackable_it == restored_objects.end()) {
        return errors::FailedPrecondition("Could not restore saved variable: ",
                                          variable.name());
      }
      const auto* serialized_tensor_attr = FindSerializedTensorInTrackable(
          *variable_trackable_it->second, "VARIABLE_VALUE");
      if (!serialized_tensor_attr) {
        return errors::FailedPrecondition(
            "Could not find serialized tensor for saved variable: ",
            variable.name());
      }
      const auto& checkpoint_key = serialized_tensor_attr->checkpoint_key();

      // Load it from the reader.
      Tensor value;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          saved_model->variable_reader()->Lookup(checkpoint_key, &value),
          "Could not read checkpoint key from variables bundle: ",
          checkpoint_key);
      TF_ASSIGN_OR_RETURN(auto value_attr, ConvertTensor(value, &builder));
      // A variable can have a partially known type, such as tensor<?x27x?xf32>,
      // even if the initializer is a specific static shape.
      TF_ASSIGN_OR_RETURN(
          auto type, ConvertToMlirTensorType(variable.shape(), variable.dtype(),
                                             &builder));
      auto op = builder.create<GlobalTensorOp>(
          builder.getUnknownLoc(),
          builder.getStringAttr(object_names.GetSymbolTableName(node_id)),
          value_attr,
          /*type=*/mlir::TypeAttr::get(type),
          /*is_mutable=*/builder.getUnitAttr());
      op.setAttr(
          "tf_saved_model.exported_names",
          builder.getStrArrayAttr(object_names.GetExportedNames(node_id)));
    } else if (object.kind_case() == SavedObject::kConstant) {
      const SavedConstant& constant = object.constant();
      const TensorProto* value = ExtractConstTensorFromGraph(
          saved_model->meta_graph_def().graph_def(), constant.operation());
      if (!value) {
        return errors::FailedPrecondition(
            "Unable to find const node referenced in object graph: ",
            constant.operation());
      }
      TF_ASSIGN_OR_RETURN(auto value_attr,
                          ConvertTensorProto(*value, &builder));
      auto op = builder.create<GlobalTensorOp>(
          builder.getUnknownLoc(),
          builder.getStringAttr(object_names.GetSymbolTableName(node_id)),
          value_attr,
          /*type=*/mlir::TypeAttr::get(value_attr.Attribute::getType()),
          /*is_mutable=*/nullptr);
      op.setAttr(
          "tf_saved_model.exported_names",
          builder.getStrArrayAttr(object_names.GetExportedNames(node_id)));
    }
  }
  AdjustBoundInputArgTypes(module);
  module.setAttr("tf_saved_model.semantics", builder.getUnitAttr());
  SortSavedModelModule(module);
  return Status::OK();
}

StatusOr<mlir::OwningModuleRef> SavedModelObjectGraphImporter::Convert(
    SavedModelV2Bundle* saved_model, absl::Span<std::string> exported_names,
    mlir::MLIRContext* context, bool add_default_attributes) {
  GraphDebugInfo dummy_debug_info;
  const GraphDebugInfo& debug_info =
      saved_model->debug_info() ? *saved_model->debug_info() : dummy_debug_info;

  GraphImportConfig specs;
  specs.prune_unused_nodes = true;
  mlir::OwningModuleRef module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  std::unordered_map<std::string, std::string> tf_name_to_mlir_name;

  const auto& graphdef = saved_model->meta_graph_def().graph_def();
  PopulateTfVersions(module.get(), graphdef.versions());

  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.add_default_attributes = add_default_attributes;
  Graph graph(OpRegistry::Global());

  GraphDef preprocessed_graphdef(graphdef);
  if (add_default_attributes) {
    TF_RETURN_IF_ERROR(PreprocessGraphDef(nullptr, &preprocessed_graphdef));
  }

  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(options, preprocessed_graphdef, &graph));

  NameUniquifier function_name_uniquifier(graph.flib_def());
  SavedModelObjectGraphImporter importer(graph.flib_def(), debug_info, specs,
                                         module.get(), &tf_name_to_mlir_name,
                                         &function_name_uniquifier);

  TF_RETURN_IF_ERROR(importer.PrepareConvert(graph));

  auto fn_names = graph.flib_def().ListFunctionNames();
  for (const auto& fn_name : fn_names) {
    TF_RETURN_IF_ERROR(importer.ConvertLibFunction(fn_name));
  }

  if (!saved_model->meta_graph_def().has_object_graph_def()) {
    return errors::InvalidArgument(
        "SavedModel does not have an object graph. Please use TF2.");
  }
  auto& object_graph = saved_model->meta_graph_def().object_graph_def();
  ObjectNames object_names(object_graph, exported_names);

  // Clean up a couple func's that always seem to be present when importing a
  // SavedModel. This is not strictly needed, as there is a separate pass that
  // will clean them up, but this makes staring at the raw IR of minimal
  // examples quite a bit nicer.
  for (auto func : llvm::make_early_inc_range(module->getOps<mlir::FuncOp>())) {
    if (func.getName().startswith("__inference__traced_save_") ||
        func.getName().startswith("__inference__traced_restore_") ||
        func.getName().startswith("__inference_signature_wrapper_")) {
      func.erase();
    }
  }

  // Diagnose SavedFunction's with multiple input signatures.
  TF_RETURN_IF_ERROR(
      DiagnoseMultipleConcreteFunctions(object_graph, object_names));

  // Construct the SavedModel IR.
  TF_RETURN_IF_ERROR(CreateSavedModelIR(object_names, module.get(),
                                        object_graph, tf_name_to_mlir_name,
                                        saved_model));
  assert(mlir::succeeded(mlir::verify(module.get())));

  return module;
}

// A helper class to import a TensorFlow model expressed in SavedModel V1 into
// an MLIR Module in SavedModel dialect.
class SavedModelSignatureDefImporter {
 public:
  // Main entry point: converts all functions (specified by SignatureDefs) in
  // the given meta graph to an MLIR Module.
  static StatusOr<mlir::OwningModuleRef> Convert(
      const SavedModelBundle& bundle, absl::Span<std::string> exported_names,
      mlir::MLIRContext* context) {
    SavedModelSignatureDefImporter importer(bundle, exported_names, context);

    return importer.ConvertSignatures();
  }

 private:
  SavedModelSignatureDefImporter(const SavedModelBundle& bundle,
                                 absl::Span<std::string> exported_names,
                                 mlir::MLIRContext* context)
      : bundle_(bundle),
        exported_names_(exported_names),
        module_(mlir::ModuleOp::create(mlir::UnknownLoc::get(context))) {}

  // Converts the SavedModel to the SavedModel dialect. Creates an MLIR function
  // for each signature.
  StatusOr<mlir::OwningModuleRef> ConvertSignatures();
  Status ConvertSignature(const GraphDef& graphdef,
                          const std::string& sig_def_key,
                          const SignatureDef& signature_def,
                          const GraphDebugInfo& debug_info,
                          const FunctionLibraryDefinition& flib_def);

  // Creates GlobalTensorOp for each variable and moves each VarHandle op to
  // the enclosing function's arguments.
  Status LiftVariables();

  // Moves the result of the VarHandleOp with corresponding global tensor to the
  // enclosing function's argument list and erases this VarHandleOp. The global
  // tensor's shape is used to provide the most accurate nested shape.
  void LiftVariable(VarHandleOp op, GlobalTensorOp global_tensor);

  using VarGlobalMap = llvm::MapVector<
      llvm::StringRef,
      std::pair<GlobalTensorOp, llvm::SmallVector<VarHandleOp, 2>>>;

  // Reads all variables from the SavedModel through session and creates
  // GlobalTensorOp for these variables.
  Status ReadVariablesFromSession(VarGlobalMap* var_globals);

  GraphImportConfig::InputArrays ParseInputArrays(
      const std::vector<std::pair<std::string, TensorInfo>>& inputs);

  const SavedModelBundle& bundle_;
  absl::Span<std::string> exported_names_;
  mlir::OwningModuleRef module_;
};

StatusOr<mlir::OwningModuleRef>
SavedModelSignatureDefImporter::ConvertSignatures() {
  const auto& signatures = bundle_.GetSignatures();
  const auto& graphdef = bundle_.meta_graph_def.graph_def();
  PopulateTfVersions(module_.get(), graphdef.versions());

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), graphdef.library());

  // debug_info might not be loaded with loader_lite.
  GraphDebugInfo debug_info;
  if (bundle_.debug_info != nullptr) debug_info = *bundle_.debug_info;

  llvm::StringSet<> exported_name_set;
  exported_name_set.insert(exported_names_.begin(), exported_names_.end());

  for (const auto& key_and_signature_def : signatures) {
    const std::string& sig_def_key = key_and_signature_def.first;
    const SignatureDef& signature_def = key_and_signature_def.second;

    // It is safe to skip "__saved_model_init_op" since it is an internal
    // signature that is not user-accessible.
    if (sig_def_key == "__saved_model_init_op") {
      continue;
    }
    if (!exported_name_set.empty() &&
        exported_name_set.count(sig_def_key) == 0) {
      continue;
    }

    TF_RETURN_IF_ERROR(ConvertSignature(graphdef, sig_def_key, signature_def,
                                        debug_info, flib_def));
  }
  TF_RETURN_IF_ERROR(LiftVariables());

  mlir::OpBuilder builder(module_->getBodyRegion());
  module_->setAttr("tf_saved_model.semantics", builder.getUnitAttr());
  SortSavedModelModule(*module_);

  return std::move(module_);
}

Status SavedModelSignatureDefImporter::ConvertSignature(
    const GraphDef& graphdef, const std::string& sig_def_key,
    const SignatureDef& signature_def, const GraphDebugInfo& debug_info,
    const FunctionLibraryDefinition& flib_def) {
  // Create local vectors for the input and output and sort them to be
  // deterministic. We don't want anyone to really depend on the order, client
  // should lookup argument/result mapping by attribute name.
  // To avoid accidentally depending on the order we use an unintuitive sorting.
  std::vector<std::pair<std::string, TensorInfo>> inputs(
      signature_def.inputs().begin(), signature_def.inputs().end());
  llvm::sort(inputs, [](const auto& lhs, const auto& rhs) {
    return lhs.first.size() < rhs.first.size() || lhs.first > rhs.first;
  });
  std::vector<std::pair<std::string, TensorInfo>> outputs(
      signature_def.outputs().begin(), signature_def.outputs().end());
  llvm::sort(outputs, [](const auto& lhs, const auto& rhs) {
    return lhs.first.size() < rhs.first.size() || lhs.first > rhs.first;
  });

  GraphImportConfig specs;
  specs.prune_unused_nodes = true;
  specs.inputs = ParseInputArrays(inputs);
  for (auto& output : outputs) specs.outputs.push_back(output.second.name());

  // Remove unused nodes and create sub-graphdef.
  GraphDef sub_graph_def;
  TF_RETURN_IF_ERROR(tensorflow::grappler::SetTransitiveFaninGraph(
      graphdef, &sub_graph_def,
      /*terminal_nodes=*/{specs.outputs.begin(), specs.outputs.end()}));

  // Convert sub-graphdef to sub-graph.
  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.add_default_attributes = true;
  Graph sub_graph(OpRegistry::Global());

  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(options, sub_graph_def, &sub_graph));

  // Convert sub-graph to MLIR module.
  TF_ASSIGN_OR_RETURN(
      auto sub_module,
      GraphDefImporter::Convert(module_->getContext(), sub_graph, debug_info,
                                flib_def, specs, sig_def_key));
  mlir::OpBuilder builder(sub_module->getBodyRegion());

  // Find the FuncOp which corresponds to current SignatureDef.
  mlir::SymbolTable symbol_table(*sub_module);
  auto func_op = symbol_table.lookup<mlir::FuncOp>(sig_def_key);
  TF_RET_CHECK(func_op)
      << "Graphdef importer should have created a function named "
      << sig_def_key << ".";

  // Use unique SignatureDef key as exported name.
  func_op.setAttr("tf_saved_model.exported_names",
                  builder.getStrArrayAttr({sig_def_key}));

  // Transfer input and output parameter names to index_path attributes.
  for (auto input_and_idx : llvm::enumerate(inputs)) {
    func_op.setArgAttr(input_and_idx.index(), "tf_saved_model.index_path",
                       builder.getStrArrayAttr({input_and_idx.value().first}));
  }
  for (auto output_and_idx : llvm::enumerate(outputs)) {
    func_op.setResultAttr(
        output_and_idx.index(), "tf_saved_model.index_path",
        builder.getStrArrayAttr({output_and_idx.value().first}));
  }

  // Move the converted functions to top level MLIR module.
  auto* block = module_->getBody();
  auto* sub_block = sub_module->getBody();
  block->getOperations().splice(
      mlir::Block::iterator(block->getTerminator()), sub_block->getOperations(),
      sub_block->begin(), mlir::Block::iterator(sub_block->getTerminator()));

  return Status::OK();
}

Status SavedModelSignatureDefImporter::LiftVariables() {
  VarGlobalMap var_globals;

  auto walker = [&var_globals](mlir::Operation* op) {
    if (auto var_handle_op = llvm::dyn_cast<VarHandleOp>(op))
      var_globals[var_handle_op.shared_name()].second.push_back(var_handle_op);
    else if (op->getName().getStringRef() == "tf.VariableV2")
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  };
  bool contains_ref_variable = module_->walk(walker).wasInterrupted();

  if (contains_ref_variable)
    return errors::InvalidArgument(
        "Ref variable created by VariableV2 is not supported.");

  if (var_globals.empty()) return Status::OK();

  TF_RETURN_IF_ERROR(ReadVariablesFromSession(&var_globals));

  for (const auto& it : var_globals)
    for (VarHandleOp var_handle : it.second.second)
      LiftVariable(var_handle, it.second.first);

  return Status::OK();
}

void SavedModelSignatureDefImporter::LiftVariable(
    VarHandleOp op, GlobalTensorOp global_tensor) {
  mlir::OpBuilder builder(&module_->getBodyRegion());

  auto func_op = op.getParentOfType<mlir::FuncOp>();
  builder.setInsertionPoint(func_op);

  auto func_type = func_op.getType();

  // Create the new function type by adding variable type to the arguments.
  llvm::SmallVector<mlir::Type, 4> new_input_types(
      func_type.getInputs().begin(), func_type.getInputs().end());
  mlir::Type resource_type = op.resource().getType();
  // Use the corresponding global tensor's type.
  auto type = global_tensor.type().cast<TensorType>();
  resource_type = mlir::RankedTensorType::get(
      {}, mlir::TF::ResourceType::get({type}, type.getContext()));

  new_input_types.push_back(resource_type);
  auto new_func_type =
      builder.getFunctionType(new_input_types, func_type.getResults());

  func_op.setType(new_func_type);

  // Bind the argument to the corresponding global tensor op.
  func_op.setArgAttr(func_op.getNumArguments() - 1,
                     "tf_saved_model.bound_input",
                     builder.getSymbolRefAttr(op.shared_name()));

  // Add the newly added function param to entry block's arguments.
  auto new_value = func_op.front().addArgument(resource_type);

  // Remove the VarHandleOp also updating the containing island's return type.
  DCHECK(llvm::isa<mlir::tf_executor::IslandOp>(op.getParentOp()));
  DCHECK(llvm::cast<mlir::tf_executor::IslandOp>(op.getParentOp())
             .WrapsSingleOp());
  op.getOperation()->replaceAllUsesWith(llvm::ArrayRef<mlir::Value>(new_value));
  op.getParentOp()->getResult(0).setType(resource_type);
  op.getOperation()->erase();
}

Status SavedModelSignatureDefImporter::ReadVariablesFromSession(
    VarGlobalMap* var_globals) {
  mlir::OpBuilder builder(&module_->getBodyRegion());

  // Read all resource variables from the session.
  std::vector<std::string> variable_names;
  variable_names.reserve(var_globals->size());
  for (const auto& name_and_location : *var_globals)
    variable_names.push_back(name_and_location.first.str());

  std::vector<Tensor> resource_tensors;
  TF_RETURN_IF_ERROR(bundle_.GetSession()->Run(
      /*inputs=*/{}, variable_names,
      /*target_node_names=*/{}, &resource_tensors));

  const DeviceMgr* device_manager;
  TF_RETURN_IF_ERROR(bundle_.GetSession()->LocalDeviceManager(&device_manager));

  // Read all underlying tensors of the variables from the session.
  std::vector<Tensor> tensors;
  tensors.reserve(resource_tensors.size());
  for (const auto& resource_tensor : resource_tensors) {
    const auto& resource_handle = resource_tensor.scalar<ResourceHandle>()();

    Device* device;
    TF_RETURN_IF_ERROR(
        device_manager->LookupDevice(resource_handle.device(), &device));

    Var* var_ptr;
    TF_RETURN_IF_ERROR(device->resource_manager()->Lookup(
        resource_handle.container(), resource_handle.name(), &var_ptr));
    core::RefCountPtr<Var> var(var_ptr);

    // The variable tensor is already loaded into corresponding device's
    // resource manager when we load the saved model using LoadSavedModel().
    // Here we just read its value.
    mutex_lock ml(*var->mu());
    tensors.push_back(*var->tensor());
  }

  for (const auto iter : llvm::zip(*var_globals, tensors)) {
    // Create global tensor op corresponding to the variable. Use the location
    // of the first use encountered.
    VarHandleOp op = std::get<0>(iter).second.second.front();
    const auto& name = std::get<0>(iter).first;
    const auto& tensor = std::get<1>(iter);

    // Create tensor attribute for this variable.
    TF_ASSIGN_OR_RETURN(auto tensor_attr, ConvertTensor(tensor, &builder));

    // Create the global tensor op with the tensor attribute.
    auto type = tensor_attr.getType().cast<TensorType>();
    auto global_tensor = builder.create<GlobalTensorOp>(
        op.getLoc(), builder.getStringAttr(name), tensor_attr,
        mlir::TypeAttr::get(type), builder.getUnitAttr());
    std::get<0>(iter).second.first = global_tensor;
  }

  return Status::OK();
}

GraphImportConfig::InputArrays SavedModelSignatureDefImporter::ParseInputArrays(
    const std::vector<std::pair<std::string, TensorInfo>>& inputs) {
  GraphImportConfig::InputArrays results;
  for (const auto& iter : inputs) {
    const auto& tensor_info = iter.second;

    // Only dense tensor is supported.
    DCHECK_EQ(tensor_info.encoding_case(), tensorflow::TensorInfo::kName);

    ArrayInfo array_info;
    array_info.imported_dtype = tensor_info.dtype();
    array_info.shape = tensor_info.tensor_shape();

    results.insert(std::pair<std::string, ArrayInfo>(tensor_info.name(),
                                                     std::move(array_info)));
  }
  return results;
}

}  // namespace

Status UpgradeLegacyGraph(Graph* graph, FunctionLibraryDefinition* flib_def) {
  return FunctionalizeControlFlow(graph, flib_def);
}

StatusOr<mlir::OwningModuleRef> ConvertGraphdefToMlir(
    const GraphDef& graphdef, const GraphDebugInfo& debug_info,
    const GraphImportConfig& specs, mlir::MLIRContext* context,
    bool add_default_attributes) {
  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.add_default_attributes = add_default_attributes;
  Graph graph(OpRegistry::Global());

  GraphDef preprocessed_graphdef(graphdef);
  if (add_default_attributes) {
    TF_RETURN_IF_ERROR(PreprocessGraphDef(&specs, &preprocessed_graphdef));
  }
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
      options, std::move(preprocessed_graphdef), &graph));
  return ConvertGraphToMlir(graph, debug_info, graph.flib_def(), specs,
                            context);
}

StatusOr<mlir::OwningModuleRef> ConvertGraphToMlir(
    const Graph& graph, const GraphDebugInfo& debug_info,
    const FunctionLibraryDefinition& flib_def, const GraphImportConfig& specs,
    mlir::MLIRContext* context) {
  // TODO(jpienaar): Remove need to const_cast.
  if (specs.upgrade_legacy) {
    TF_RETURN_IF_ERROR(
        UpgradeLegacyGraph(const_cast<Graph*>(&graph),
                           const_cast<FunctionLibraryDefinition*>(&flib_def)));
  }
  return GraphDefImporter::Convert(context, graph, debug_info, flib_def, specs,
                                   /*func_name=*/"main");
}

StatusOr<mlir::OwningModuleRef> ConvertSavedModelToMlir(
    SavedModelV2Bundle* saved_model, mlir::MLIRContext* context,
    absl::Span<std::string> exported_names, bool add_default_attributes) {
  return SavedModelObjectGraphImporter::Convert(
      saved_model, exported_names, context, add_default_attributes);
}

StatusOr<mlir::OwningModuleRef> ConvertSavedModelV1ToMlir(
    const SavedModelBundle& saved_model, absl::Span<std::string> exported_names,
    mlir::MLIRContext* context) {
  return SavedModelSignatureDefImporter::Convert(saved_model, exported_names,
                                                 context);
}

std::string MlirModuleToString(mlir::ModuleOp module, bool show_debug_info) {
  std::string txt_module;
  {
    mlir::OpPrintingFlags flags;
    if (show_debug_info) flags.enableDebugInfo();
    llvm::raw_string_ostream os{txt_module};
    module.print(os, flags);
  }
  return txt_module;
}

}  // namespace tensorflow

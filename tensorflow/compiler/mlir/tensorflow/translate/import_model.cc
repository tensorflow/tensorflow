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

#include <atomic>
#include <functional>
#include <iterator>
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/loader_util.h"
#include "tensorflow/compiler/jit/shape_inference_helpers.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/initialize_variables_in_session_init.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lift_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/mark_initialized_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_attr.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
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
#include "tensorflow/core/platform/crash_analysis.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/stream_executor/lib/statusor.h"

static inline absl::string_view StringRefToView(llvm::StringRef ref) {
  return {ref.data(), ref.size()};
}

namespace tensorflow {

constexpr size_t kNumThreadToConvertSignatures = 10;
constexpr absl::string_view kOutputShapesAttrName = "_output_shapes";

using mlir::NamedAttrList;
using mlir::TensorType;
using mlir::tf_saved_model::AssetOp;
using mlir::tf_saved_model::GlobalTensorOp;
using mlir::tf_saved_model::SessionInitializerOp;
using stream_executor::port::StatusOr;

namespace {

bool IsOutputShapesAttribute(const AttrValue& attr_value,
                             llvm::StringRef attr_name) {
  return attr_name.compare(kOutputShapesAttrName) == 0 &&
         attr_value.value_case() == AttrValue::kList;
}

bool IsResourceOutputShapesAttribute(const AttrValue& attr_value,
                                     llvm::StringRef attr_name) {
  if (attr_name == "_handle_dtypes" || attr_name == "_handle_shapes")
    return attr_value.value_case() == AttrValue::kList;
  return false;
}

void LoadImporterDialects(mlir::MLIRContext& context) {
  // Load dialects involved in the conversion
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  context.appendDialectRegistry(registry);
  for (llvm::StringRef name : registry.getDialectNames())
    context.getOrLoadDialect(name);
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
        error_handler_(module.getContext()) {
    // Log import config.
    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "Importing with: " << specs.str();
      for (auto& it : *tf_name_to_mlir_name) {
        LOG(INFO) << "\t" << it.first << " -> " << it.second;
      }
    }
  }

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
  // PrepareConvert needs to ensure that the original `graph` is cloned prior
  // execution. The cloning procedure relies on the roundtrip through the
  // GraphDef. Graph to GraphDef def conversion is heavy, in case, `graph_def`
  // was obtained previously provide it to the PrepareConvert to reuse.
  Status PrepareConvert(const Graph& graph,
                        std::unique_ptr<GraphDef> graph_def = nullptr);

  // Converts the prepared graph to a Function and adds it to the module. A set
  // of nodes from the graph are given to converted to the arguments and returns
  // of the function.
  Status Convert(llvm::StringRef func_name, mlir::FunctionType func_type,
                 const absl::InlinedVector<OutputTensor, 4>& arg_nodes,
                 const absl::InlinedVector<OutputTensor, 4>& ret_nodes,
                 const absl::InlinedVector<Node*, 4>& control_ret_nodes,
                 llvm::ArrayRef<mlir::NamedAttribute> attrs);

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

  // Convert deferred TF functions to the MLIR representation.
  // Conversion is deferred for efficiency reasons, e.g., to limit depth
  // of recursion and reduce stack size pressure.
  Status ConvertDeferredFunctions();

 private:
  // Most types with subtypes have only one subtype.
  using ElementSubtypes = llvm::SmallVector<TensorType, 1>;

  // Metadata used for deferred function conversion.
  struct DeferredConversionMetaData {
    DeferredConversionMetaData(
        const std::string& function_name,
        const std::vector<mlir::NamedAttribute>& attributes)
        : function_name(function_name), attributes(attributes) {}

    std::string function_name;
    std::vector<mlir::NamedAttribute> attributes;
  };

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
  // in an island.
  mlir::Operation* CreateOperation(
      const Node& node, llvm::StringRef node_type_name,
      const mlir::OperationState& result,
      const llvm::SmallVectorImpl<mlir::Value>& control_operands);

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
  Status RemoveBackedges();

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
      mlir::func::FuncOp func, mlir::tf_executor::GraphOp graph_op,
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
  mlir::Location GetLocation(const Node& node);

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
  // All the TF ops encountered that aren't modelled in dialect.
  llvm::DenseSet<mlir::StringAttr> unmodelled_op_names_;

 protected:
  // Maps feed as TensorId to new Placeholder node name.
  absl::flat_hash_map<TensorId, absl::string_view> remapped_feeds_;
  // Keep track of functions required deferred conversion.
  std::queue<DeferredConversionMetaData> deferred_functions_;
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
  if (it == inputs.end()) return OkStatus();

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
  return OkStatus();
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
  return OkStatus();
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

Status ImporterBase::ConvertDeferredFunctions() {
  while (!deferred_functions_.empty()) {
    auto conversion_metadata = deferred_functions_.front();
    deferred_functions_.pop();

    const FunctionDef* func_def =
        graph_flib_.Find(conversion_metadata.function_name);
    // Converts the graph to an MLIR function and adds it to the module.
    // We populate the NodeSpec so that all the _Arg ops get their shape
    // added correctly.
    GraphImportConfig specs;
    specs.enable_shape_inference = specs_.enable_shape_inference;
    specs.unconditionally_use_set_output_shapes =
        specs_.unconditionally_use_set_output_shapes;
    for (const auto& name_and_value : func_def->attr()) {
      if (name_and_value.first == "_input_shapes") {
        auto& list = name_and_value.second.list();
        auto& signature = func_def->signature();
        // Some models have "_input_shapes" attribute, but with its value empty
        if (list.shape_size() > 0 &&
            list.shape_size() != signature.input_arg_size()) {
          return errors::FailedPrecondition(
              "Number of input arguments must be equal to the length of "
              "_input_shapes attribute in function '",
              StringRefToView(conversion_metadata.function_name), "'.");
        }
        for (int i = 0, e = signature.input_arg_size(); i < e; i++) {
          auto& input_arg = signature.input_arg(i);
          auto& array_info = specs.inputs[input_arg.name()];
          array_info.imported_dtype = input_arg.type();
          // set to unranked for empty "_input_shapes" attribute
          if (list.shape_size() > 0)
            array_info.shape = list.shape(i);
          else
            array_info.shape.set_unknown_rank(true);
        }
      }
    }

    ImporterBase importer(graph_flib_, debug_info_, specs, module_,
                          tf_name_to_mlir_name_, function_name_uniquifier_,
                          conversion_metadata.function_name);

    std::unique_ptr<FunctionBody> fbody;
    TF_RETURN_IF_ERROR(
        FunctionDefToBodyHelper(*func_def, AttrSlice(), &graph_flib_, &fbody));
    TF_RETURN_IF_ERROR(importer.PrepareConvert(*fbody->graph));

    TF_ASSIGN_OR_RETURN(auto func_type, importer.InferLibFunctionType(*fbody));

    absl::InlinedVector<OutputTensor, 4> arg_nodes;
    absl::InlinedVector<OutputTensor, 4> ret_nodes;
    absl::InlinedVector<Node*, 4> control_ret_nodes;
    importer.GetArgsAndRetsFromFunctionBody(*fbody, &arg_nodes, &ret_nodes,
                                            &control_ret_nodes);
    const std::string& mlir_func_name =
        (*tf_name_to_mlir_name_)[conversion_metadata.function_name];

    TF_RETURN_IF_ERROR(importer.Convert(mlir_func_name, func_type, arg_nodes,
                                        ret_nodes, control_ret_nodes,
                                        conversion_metadata.attributes));

    // Additional function bodies could be discovered during the deferred
    // loading of the current function. Add them to the working queue.
    while (!importer.deferred_functions_.empty()) {
      deferred_functions_.push(importer.deferred_functions_.front());
      importer.deferred_functions_.pop();
    }
  }

  return OkStatus();
}

Status ImporterBase::RemoveBackedges() {
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
  return OkStatus();
}

Status CopyStackTraces(const Graph& from, Graph* to) {
  // Copy over the stack traces.
  // TODO(jpienaar): This really shouldn't be needed, copying the Graph above
  // and then needing these traversals is unfortunate.
  std::unordered_map<string, Node*> node_map = from.BuildNodeNameIndex();
  for (Node* node : to->nodes()) {
    if (const Node* old_node = node_map[node->name()]) {
      if (const std::shared_ptr<AbstractStackTrace>& stack =
              old_node->GetStackTrace()) {
        DVLOG(2) << "Stack for " << node->name() << " "
                 << old_node->GetStackTrace()->ToString(
                        AbstractStackTrace::TracePrintingOptions());
        node->SetStackTrace(stack);
      } else {
        DVLOG(1) << "No stack for " << node->name() << " (" << node
                 << ") in Graph " << &from;
      }
    } else {
      DVLOG(1) << "No stack for " << node->name() << " (" << node
               << ") in Graph " << &from;
    }
  }

  return OkStatus();
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
    return OkStatus();
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

  return OkStatus();
}

// TODO(jpienaar): Remove this post shape inference on import flag is removed.
Status ImporterBase::AddNodesToShapeRefiner(
    std::unordered_map<string, Node*>* node_name_map) {
  shape_refiner_ = std::make_unique<ShapeRefiner>(graph_->versions(),
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
      // This follows the same approach as in ValidateShape, but only flags
      // warning in case where there are mismatch in number of shapes and
      // outputs and in which case it just returns without attempting to refine.
      if (list.shape_size() != node->num_outputs()) {
        LOG(WARNING) << "Node '" << node->name() << "' has "
                     << node->num_outputs() << " outputs but the "
                     << kOutputShapesAttrName
                     << " attribute specifies shapes for " << list.shape_size()
                     << " outputs";
        return OkStatus();
      }

      for (const auto& shape : llvm::enumerate(list.shape())) {
        auto* node_context = shape_refiner_->GetContext(node);
        shape_inference::ShapeHandle handle;
        Status status =
            node_context->MakeShapeFromShapeProto(shape.value(), &handle);
        if (!status.ok()) {
          return EmitErrorWithLocationStr(*node, status);
        }
        node_context->set_output(shape.index(), handle);
      }
      return OkStatus();
    };

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
      } else if (const AttrValue* attr =
                     node->attrs().Find(kOutputShapesAttrName)) {
        TF_RETURN_IF_ERROR(set_shape_from_list_attr(attr));
      } else {
        node_context->set_output(0, node_context->UnknownShape());
      }
    }

    // Following GraphConstructor::ValidateShape called from
    // GraphConstructor::Convert, override the shape if _output_shapes is set.
    if (specs_.unconditionally_use_set_output_shapes ||
        node->op_def().name() == "ReadVariableOp") {
      if (const AttrValue* attr = node->attrs().Find(kOutputShapesAttrName))
        TF_RETURN_IF_ERROR(set_shape_from_list_attr(attr));
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
        int64_t val0 = c->Value(c->Dim(s0, i));
        int64_t val1 = c->Value(c->Dim(s1, i));
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
  return OkStatus();
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
  auto shape_ic =
      [&](shape_inference::InferenceContext* c) -> StatusOr<mlir::Type> {
    // TODO(b/200093974): Post triage, consider following
    // GraphConstructor::ValidateShape in checking _output_shapes always.
    if (specs_.unconditionally_use_set_output_shapes) {
      if (const AttrValue* attr = node.attrs().Find(kOutputShapesAttrName)) {
        auto& list = attr->list();
        if (list.shape_size() > idx) {
          const TensorShapeProto& p = list.shape()[idx];
          shape_inference::ShapeHandle h;
          Status s = c->MakeShapeFromShapeProto(p, &h);
          if (!s.ok())
            return errors::InvalidArgument(
                "Node '", node.name(), " has an invalid ",
                kOutputShapesAttrName, " attribute (shape #", idx, " error:'",
                s.error_message(), "')");
          c->set_output(idx, h);
        }
      }
    }

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

  if (node.IsWhileNode()) {
    auto* output_shapes = node.attrs().Find("output_shapes");
    auto* element_types = node.attrs().Find("T");
    if (output_shapes && !output_shapes->list().shape().empty()) {
      const auto& output_shape = output_shapes->list().shape(idx);
      const auto& element_type = element_types->list().type(idx);
      return ConvertToMlirTensorType(output_shape, element_type, &builder);
    }
  }

  auto type_from_array_attr = [&node, &idx, &builder](
                                  absl::string_view output_shape_attr,
                                  absl::string_view element_type_attr) {
    auto* output_shapes = node.attrs().Find(output_shape_attr);
    auto* element_types = node.attrs().Find(element_type_attr);
    const auto& output_shape = output_shapes->list().shape(idx);
    const auto& element_type = element_types->list().type(idx);
    return ConvertToMlirTensorType(output_shape, element_type, &builder);
  };

  if (node.type_string() == "IteratorGetNext" ||
      node.type_string() == "IteratorGetNextSync" ||
      node.type_string() == "MultiDeviceIteratorGetNextFromShard")
    return type_from_array_attr("output_shapes", "output_types");

  if (node.type_string() == "InfeedDequeueTuple")
    return type_from_array_attr("shapes", "dtypes");

  if (node.type_string() == "InfeedDequeue") {
    assert(idx == 0);
    const auto& output_shape = node.attrs().Find("shape")->shape();
    const auto& element_type = node.attrs().Find("dtype")->type();
    return ConvertToMlirTensorType(output_shape, element_type, &builder);
  }

  // Returns a simple, more conservative unranked tensor type.
  auto default_type = [&]() -> StatusOr<mlir::Type> {
    mlir::Type element_type;
    TF_RETURN_IF_ERROR(ConvertDataType(dtype, builder, &element_type));

    // TODO(b/200093974): Post triage, consider following
    // GraphConstructor::ValidateShape in checking _output_shapes.
    if (specs_.unconditionally_use_set_output_shapes) {
      if (const AttrValue* attr = node.attrs().Find(kOutputShapesAttrName)) {
        auto& list = attr->list();
        if (list.shape_size() > idx) {
          llvm::SmallVector<int64_t, 4> shape;
          const TensorShapeProto& shape_proto = list.shape()[idx];
          if (shape_proto.unknown_rank())
            return mlir::UnrankedTensorType::get(element_type);
          TF_RETURN_IF_ERROR(ConvertToMlirShape(shape_proto, &shape));
          return mlir::RankedTensorType::get(shape, element_type);
        }
      }
    }

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
  int32_t rank = context->Rank(handle);
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
  if (!func_attr) return OkStatus();
  attributes->push_back(builder_.getNamedAttr(base_name, func_attr));

  for (const auto& it : value.func().attr()) {
    auto name = absl::StrCat(base_name, ".", it.first);
    TF_ASSIGN_OR_RETURN(auto value, ConvertAttributeValue(it.second));
    attributes->push_back(builder_.getNamedAttr(name, value));
  }
  return OkStatus();
}

StatusOr<mlir::FlatSymbolRefAttr> ImporterBase::ConvertFunctionCallName(
    const std::string& func_name) {
  // Some ops like XlaHostCompute op uses empty value to represent missing
  // functions. Such attribute values should be defined optional in MLIR
  // definition.
  if (func_name.empty()) return mlir::FlatSymbolRefAttr();

  TF_RETURN_IF_ERROR(ConvertLibFunction(func_name));
  auto mlir_func_name = (*tf_name_to_mlir_name_)[func_name];
  return mlir::SymbolRefAttr::get(builder_.getContext(), mlir_func_name);
}

StatusOr<mlir::Attribute> ImporterBase::ConvertAttributeValue(
    const AttrValue& value) {
  switch (value.value_case()) {
    case AttrValue::kFunc: {
      // TODO(b/156546237): Unify kFunc/NameAttrList attribute representation.
      // Currently kFunc/NameAttrList attributes in a kList/repeated AttrValue
      // will not use this representation. This also doesn't handle empty
      // function values like ConvertFunctionCallName method.
      NamedAttrList attrs;
      for (const auto& func_attr : value.func().attr()) {
        TF_ASSIGN_OR_RETURN(
            auto attr, ImporterBase::ConvertAttributeValue(func_attr.second));
        attrs.push_back(builder_.getNamedAttr(func_attr.first, attr));
      }
      auto func_attrs = builder_.getDictionaryAttr(attrs);
      return mlir::TF::FuncAttr::get(context_, value.func().name(), func_attrs);
    }
    case AttrValue::kList: {
      if (!value.list().func().empty()) {
        absl::InlinedVector<mlir::Attribute, 8> attrs;
        for (const auto& item : value.list().func()) {
          TF_ASSIGN_OR_RETURN(auto attr, ConvertFunctionCallName(item.name()));
          if (item.attr_size() != 0)
            return errors::Unimplemented(
                "func attributes with non-zero attr.size()");
          if (attr) attrs.push_back(attr);
        }
        return builder_.getArrayAttr(
            llvm::makeArrayRef(attrs.begin(), attrs.end()));
      }
      return ConvertNonFuncAttributeValue(value, &builder_);
    }
    default:
      return ConvertNonFuncAttributeValue(value, &builder_);
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
    return OkStatus();

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

  // Converts the argument and return types to MLIR types.
  std::vector<mlir::NamedAttribute> attributes;
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
    auto gradient_attr =
        mlir::SymbolRefAttr::get(builder_.getContext(), mlir_grad_func_name);
    auto grad_string = mlir::TF::TensorFlowDialect::GetGradientAttrName();
    attributes.push_back(builder_.getNamedAttr(grad_string, gradient_attr));
  }

  deferred_functions_.emplace(func_name.str(), attributes);
  return OkStatus();
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
  return OkStatus();
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
  return OkStatus();
}

Status ImporterBase::PrepareConvert(const Graph& graph,
                                    std::unique_ptr<GraphDef> graph_def) {
  // TODO(fengliuai): Converting to GraphDef and back is the easiest way to
  // clone a graph.
  // TODO(fengliuai): clone the graph without going to graph_def first.
  if (graph_def == nullptr) {
    graph_def = std::make_unique<GraphDef>();
    graph.ToGraphDef(graph_def.get());
  }
  graph_ = std::make_unique<Graph>(graph.flib_def());
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.add_default_attributes = true;
  TF_RETURN_IF_ERROR(::tensorflow::ConvertGraphDefToGraph(
      opts, std::move(*graph_def), graph_.get()));

  TF_RETURN_IF_ERROR(RemoveBackedges());

  TF_RETURN_IF_ERROR(CopyStackTraces(graph, graph_.get()));

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

  return OkStatus();
}

Status ImporterBase::Convert(
    llvm::StringRef func_name, mlir::FunctionType func_type,
    const absl::InlinedVector<OutputTensor, 4>& arg_nodes,
    const absl::InlinedVector<OutputTensor, 4>& ret_nodes,
    const absl::InlinedVector<Node*, 4>& control_ret_nodes,
    llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // TODO(b/122040776): Uses debug info for FunctionDef.
  auto function = mlir::func::FuncOp::create(mlir::UnknownLoc::get(context_),
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
      function.setType(mlir::FunctionType::get(function.getContext(),
                                               func_type.getInputs(),
                                               graph.getResultTypes()));
    }
  }

  return OkStatus();
}

Status ImporterBase::ConvertFunctionArgAndRets(
    mlir::func::FuncOp func, mlir::tf_executor::GraphOp graph_op,
    llvm::ArrayRef<mlir::Type> arg_types,
    const absl::InlinedVector<OutputTensor, 4>& arg_nodes,
    const absl::InlinedVector<OutputTensor, 4>& ret_nodes,
    const absl::InlinedVector<Node*, 4>& control_ret_nodes) {
  // Store the arg/return attributes as a list rather than uniqueuing during
  // construction.
  llvm::SmallVector<mlir::NamedAttrList, 4> arg_attrs;
  arg_attrs.resize(func.getNumArguments());
  llvm::SmallVector<mlir::NamedAttrList, 4> ret_attrs;
  ret_attrs.resize(func.getNumResults());

  auto set_attributes_on_func = [&](Node* node, int64_t index, bool is_arg) {
    for (const auto& node_attr : node->attrs()) {
      const auto& key = node_attr.first;
      // Only import optional attributes (e.g., those starting with an
      // underscore).
      if (key.empty() || key[0] != '_') continue;
      // Ignore shape inference attributes as shape information is already
      // populated in the result type.
      if (IsOutputShapesAttribute(node_attr.second, key) ||
          IsResourceOutputShapesAttribute(node_attr.second, key))
        continue;
      TF_ASSIGN_OR_RETURN(auto converted_attr,
                          ConvertAttributeValue(node_attr.second));
      std::string dialect_attribute = "tf." + key;
      if (is_arg) {
        arg_attrs[index].set(dialect_attribute, converted_attr);
      } else {
        func.setResultAttr(index, dialect_attribute, converted_attr);
        ret_attrs[index].set(dialect_attribute, converted_attr);
      }
    }
    return OkStatus();
  };

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
      arg_attrs[i].set("tf.device", builder_.getStringAttr(
                                        arg_node.node->requested_device()));

    if (arg_node.node->IsArg()) {
      TF_RETURN_IF_ERROR(
          set_attributes_on_func(arg_node.node, i, /*is_arg=*/true));
    }

    island->dropAllReferences();
    island->erase();
  }

  llvm::SmallVector<mlir::Value, 8> inst_to_return;
  for (auto ret_and_idx : llvm::enumerate(ret_nodes)) {
    const auto& ret = ret_and_idx.value();
    auto* inst = node_values_[ret.node->id()];
    if (ret.node->IsRetval()) {
      if (!ret.node->requested_device().empty())
        ret_attrs[ret_and_idx.index()].set(
            "tf.device", builder_.getStringAttr(ret.node->requested_device()));
      TF_RETURN_IF_ERROR(set_attributes_on_func(ret.node, ret_and_idx.index(),
                                                /*is_arg=*/false));
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
  builder_.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(context_),
                                        graph_op.getResults());

  func.setAllArgAttrs(
      llvm::to_vector<4>(llvm::map_range(arg_attrs, [&](NamedAttrList& list) {
        return list.getDictionary(context_);
      })));
  func.setAllResultAttrs(
      llvm::to_vector<4>(llvm::map_range(ret_attrs, [&](NamedAttrList& list) {
        return list.getDictionary(context_);
      })));

  return OkStatus();
}

mlir::Location ImporterBase::GetLocation(const Node& node) {
  DVLOG(1) << "Getting location for " << node.name() << " " << &node;
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
        function_name.empty() ? name.str() : debug_info_key;
    auto name_loc_id = mlir::StringAttr::get(context_, name_for_name_loc);

    llvm::SmallVector<mlir::Location, 4> locations;
    // Prefer stack traces if available, fallback to debug info if not, and then
    // finally to just name.
    if (auto stack_trace = node.GetStackTrace()) {
      DVLOG(1) << "Stack available for " << node.name();
      absl::Span<const StackFrame> frames = stack_trace->ToFrames();
      locations.reserve(frames.size());
      for (const StackFrame& frame : llvm::reverse(frames)) {
        auto file_name = mlir::StringAttr::get(context_, frame.file_name);
        // Use col 1 as there is no column info in StackTrace.
        auto file_line_loc =
            mlir::FileLineColLoc::get(file_name, frame.line_number, 1);
        locations.push_back(file_line_loc);
      }
    } else {
      DVLOG(1) << "No stack trace for " << node.name();
      const auto location_it = debug_info.find(debug_info_key);
      if (location_it != debug_info.end()) {
        DVLOG(1) << "Available serialized debug info for " << node.name();
        // Convert the stack trace to a chain of mlir::CallSiteLocs.
        const auto& trace = location_it->second;
        locations.reserve(trace.file_line_cols_size());
        for (const auto& location : trace.file_line_cols()) {
          const auto& file = debug_info_.files(location.file_index());
          auto file_name = mlir::StringAttr::get(context_, file);
          auto file_line_loc = mlir::FileLineColLoc::get(
              file_name, location.line(), location.col());
          locations.push_back(file_line_loc);
        }
      }
    }

    // If there are no locations in the stack trace, fall back to just a
    // NameLoc with no child.
    if (locations.empty()) return mlir::NameLoc::get(name_loc_id);

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

  // Create a location for node `name` in function `function_name`.
  auto create_op_type_and_name_locations = [&]() {
    return mlir::FusedLoc::get(
        context_,
        // Add the type operation for the propagation of op_type metadata.
        {mlir::NameLoc::get(
             mlir::StringAttr::get(context_, node.type_string() + ":")),
         create_location(node.name(), function_name_for_debug_info_)});
  };

  // For NextIteration nodes, location is used to pair source and sink nodes.
  // Hence, we use node name as location to keep it unique.
  // TODO(prakalps): In future the plan is to use tokens to pair source/sink
  // nodes. Then NextIteration nodes would not need to be handled separately.
  if (node.type_string() == "NextIteration") {
    return create_op_type_and_name_locations();
  }

  const auto& node_def = node.def();
  auto original_nodes =
      node_def.experimental_debug_info().original_node_names();
  auto original_funcs =
      node_def.experimental_debug_info().original_func_names();

  if (original_nodes.empty()) {
    return create_op_type_and_name_locations();
  } else {
    // If the original nodes are defined, then we use them to get a list of
    // call sites, and then fuse them to a single fused location, with the name
    // of the node_def.
    llvm::SmallVector<mlir::Location, 4> node_locations;
    node_locations.reserve(original_nodes.size() + 2);
    // Add the type operation for the propagation of op_type metadata.
    node_locations.push_back(mlir::NameLoc::get(
        mlir::StringAttr::get(context_, node.type_string() + ":")));
    // Retrieve the names from the experimental_debug_info.
    for (int i = 0, e = original_nodes.size(); i != e; ++i) {
      auto node_name = original_nodes[i];
      auto func_name = (i < original_funcs.size()) ? original_funcs[i] : "";
      node_locations.push_back(create_location(node_name, func_name));
    }
    // Retrieve the name of the node_def.
    node_locations.push_back(
        create_location(node.name(), function_name_for_debug_info_));
    return mlir::FusedLoc::get(context_, node_locations);
  }
}

Status ImporterBase::EmitErrorWithLocationStr(const Node& node,
                                              const Status& error_status) {
  const mlir::Location location = GetLocation(node);
  mlir::emitError(location);
  return error_handler_.Combine(error_status);
}

mlir::Operation* ImporterBase::CreateOperation(
    const Node& node, llvm::StringRef node_type_name,
    const mlir::OperationState& result,
    const llvm::SmallVectorImpl<mlir::Value>& control_operands) {
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
  mlir::Operation* inner_op = island_builder.create(result);

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

  if (VLOG_IS_ON(1)) {
    mlir::OperationName name = inner_op->getName();
    if (!name.isRegistered() &&
        // Skip unmodelled ops that are handled differently.
        (node_type_name != "_Arg" && node_type_name != "_Retval") &&
        !unmodelled_op_names_.count(name.getIdentifier())) {
      if (node.op_def().is_stateful()) {
        VLOG(1) << "[potentially conservative] Op type `" << node.type_string()
                << "` is stateful but effects not modelled";
      } else {
        // See if any resource type is used.
        bool resource = false;
        std::function<bool(mlir::Type)> record_resource;
        record_resource = [&](mlir::Type type) {
          if (resource) return true;
          if (type.isa<mlir::TF::ResourceType>()) {
            resource = true;
            return true;
          }
          if (auto with_subtype =
                  type.dyn_cast<mlir::SubElementTypeInterface>()) {
            with_subtype.walkSubTypes(
                [&](mlir::Type t) { record_resource(t); });
          }
          return resource;
        };

        for (mlir::Type t : inner_op->getResultTypes())
          if (record_resource(t)) break;
        for (mlir::Type t : inner_op->getOperandTypes())
          if (record_resource(t)) break;
        if (resource) {
          unmodelled_op_names_.insert(name.getIdentifier());
          VLOG(1) << "[potentially conservative] Op type `"
                  << node.type_string()
                  << "` has resource operands/results but effects not modelled";
        }
      }
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
    return OkStatus();
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

  mlir::OperationState result(GetLocation(node), op_name);
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
  auto abstract_op = result.name.getRegisteredInfo();
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

  const auto& node_def = node.def();
  // NodeDef can contain partial TF device names. In such cases, canonicalize
  // it. Note that in current TF, placer will place full device name to each
  // node.
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(node_def.device(), &parsed_name)) {
    return errors::InvalidArgument(
        "Op ", op_name, " has invalid device name: ", node_def.device());
  }
  // Keep the parsed name untouched if the device name is empty.
  if (!node_def.device().empty()) {
    if (!parsed_name.has_type) {
      parsed_name.type = "CPU";
      parsed_name.has_type = true;
    }
    if (!parsed_name.has_id) {
      parsed_name.id = 0;
      parsed_name.has_id = true;
    }
  }
  result.attributes.push_back(builder_.getNamedAttr(
      "device", builder_.getStringAttr(
                    DeviceNameUtils::ParsedNameToString(parsed_name))));

  // Map user function calls to LegacyCall ops and add the user function name
  // as an attribute.
  if (convert_to_legacy_call) {
    result.name = mlir::OperationName(get_full_op_name("LegacyCall"), context_);
    mlir::SymbolRefAttr val =
        mlir::SymbolRefAttr::get(builder_.getContext(), node_type_name);
    result.addAttribute("f", val);

    if (!result.attributes.get("_disable_call_shape_inference")) {
      result.addAttribute("_disable_call_shape_inference",
                          builder_.getBoolAttr(false));
    }
  }

  auto composite_control_flow_op = [&](const std::string& name) {
    result.name = mlir::OperationName(get_full_op_name(name), context_);
    bool stateless = absl::StartsWith(node_type_name, "Stateless");
    mlir::BoolAttr val = builder_.getBoolAttr(stateless);
    result.attributes.push_back(builder_.getNamedAttr("is_stateless", val));
  };

  // Map Case/If/While and StatelessCase/If/While op in TensorFlow to the common
  // Case/If/While op in MLIR and add the differentiating attribute.
  if (node.IsCaseNode()) composite_control_flow_op("Case");
  if (node.IsIfNode()) composite_control_flow_op("If");
  if (node.IsWhileNode()) {
    composite_control_flow_op("While");
    auto* output_shapes = node.attrs().Find("output_shapes");
    if (output_shapes && !output_shapes->list().shape().empty())
      result.attributes.push_back(
          builder_.getNamedAttr("shape_invariant", builder_.getUnitAttr()));
  }

  // Register the mapping between the TF node and the newly created operation.
  node_values_[node.id()] =
      CreateOperation(node, node_type_name, result, control_operands);
  return OkStatus();
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
  return OkStatus();
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
  auto* new_dst = builder_.create(state);

  // Replaces the output uses of the old operation by the corresponding
  // result of the new operation, and deletes the old operation.
  for (unsigned i = 0, e = dst->getNumResults(); i != e; ++i) {
    auto new_output = new_dst->getResult(i);
    dst->getResult(i).replaceAllUsesWith(new_output);
  }
  dst->dropAllReferences();
  dst->erase();
  return OkStatus();
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
  static StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Convert(
      mlir::MLIRContext* context, const Graph& graph,
      const GraphDebugInfo& debug_info,
      const FunctionLibraryDefinition& flib_def, const GraphImportConfig& specs,
      std::unordered_map<std::string, std::string>& tf_name_to_mlir_name);

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

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GraphDefImporter::Convert(
    mlir::MLIRContext* context, const Graph& graph,
    const GraphDebugInfo& debug_info, const FunctionLibraryDefinition& flib_def,
    const GraphImportConfig& specs,
    std::unordered_map<std::string, std::string>& tf_name_to_mlir_name) {
  LoadImporterDialects(*context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  NameUniquifier function_name_uniquifier(flib_def);

  // importer.PrepareConvert below will attemp to clone the original `graph`
  // via conversion to the graph def first. Convert graph to graph_def here
  // first and avoid extra copies later.
  auto graph_def = std::make_unique<GraphDef>();
  graph.ToGraphDef(graph_def.get());

  static std::atomic<uint32> counter(0);
  uint32 current_file_prefix = counter++;
  const auto* graph_crash_handle = crash_analysis::ReportProtoDataOnCrash(
      absl::StrCat(current_file_prefix, "_mlir_import_graph.pbtxt"),
      *graph_def);
  auto reachable_flib = flib_def.ReachableDefinitions(*graph_def);
  const auto* flib_crash_handle = crash_analysis::ReportProtoDataOnCrash(
      absl::StrCat(current_file_prefix, "_mlir_import_flib.pbtxt"),
      reachable_flib.ToProto());

  auto scope_exit = llvm::make_scope_exit([&]() {
    crash_analysis::RemoveReportData(graph_crash_handle);
    crash_analysis::RemoveReportData(flib_crash_handle);
  });

  VLOG(1) << "Importing: "
          << ::tensorflow::DumpGraphToFile("tf_mlir_importer_base", graph,
                                           &flib_def);

  GraphDefImporter importer(flib_def, debug_info, specs, module.get(),
                            &tf_name_to_mlir_name, &function_name_uniquifier);

  TF_RETURN_IF_ERROR(importer.PrepareConvert(graph, std::move(graph_def)));

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

    // Under `graph_as_function` mode, `tf.entry_function` is always set as it
    // is assumed feed, fetch, and target nodes are set correctly.
    attrs.push_back(b.getNamedAttr(
        "tf.entry_function",
        b.getDictionaryAttr({inputs, outputs, control_outputs})));
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

  const llvm::StringRef& graph_func_name =
      specs.graph_func_name.empty() ? kImportModelDefaultGraphFuncName
                                    : specs.graph_func_name;
  TF_RETURN_IF_ERROR(importer.ImporterBase::Convert(graph_func_name, func_type,
                                                    arg_nodes, ret_nodes,
                                                    control_ret_nodes, attrs));
  TF_RETURN_IF_ERROR(importer.ImporterBase::ConvertDeferredFunctions());

  // Mark main function public, others private.
  for (auto function : module.get().getOps<mlir::func::FuncOp>()) {
    auto visibility = function.getName() == graph_func_name
                          ? mlir::func::FuncOp::Visibility::Public
                          : mlir::func::FuncOp::Visibility::Private;
    function.setVisibility(visibility);
  }
  VLOG(1) << "Imported: "
          << tensorflow::DumpMlirOpToFile("tf_mlir_imported_base",
                                          module.get());
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
    // Check if we have subtypes first
    if (!node_info.subtypes.empty()) {
      std::vector<mlir::TensorType> subtypes;
      for (const auto& st : node_info.subtypes) {
        mlir::Type st_data_type;
        llvm::SmallVector<int64_t> shape;
        TF_RETURN_IF_ERROR(ConvertToMlirShape(st.shape, &shape));
        TF_RETURN_IF_ERROR(
            ConvertDataType(st.imported_dtype, builder, &st_data_type));
        subtypes.push_back(mlir::RankedTensorType::get(shape, st_data_type));
      }
      if (imported_dtype == DT_RESOURCE) {
        element_type =
            mlir::TF::ResourceType::get(subtypes, builder.getContext());
      } else if (imported_dtype == DT_VARIANT) {
        element_type =
            mlir::TF::VariantType::get(subtypes, builder.getContext());
      } else {
        return errors::InvalidArgument(DataType_Name(imported_dtype),
                                       " takes no subtypes.");
      }
    } else {
      TF_RETURN_IF_ERROR(
          ConvertDataType(imported_dtype, builder, &element_type));
    }
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
    const int num_nodes = nodes->size();
    if (num_nodes < index + 1) nodes->resize(index + 1);

    if ((*nodes)[index].node != nullptr)
      return errors::InvalidArgument(node->type_string(), " node '",
                                     node->name(), "' has attribute 'index' ",
                                     index, " that conflicts with node '",
                                     (*nodes)[index].node->name(), "'");
    (*nodes)[index] = {node, 0};

    return OkStatus();
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
  if (control_outputs.empty()) return OkStatus();

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

  return OkStatus();
}

// Stateful helper class to import a TensorFlow model expressed in SavedModel
// into an MLIR Module.
class SavedModelObjectGraphImporter : public ImporterBase {
 public:
  // Main entry point: converts all functions in the given meta graph to an MLIR
  // Module.
  static StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Convert(
      SavedModelV2Bundle* saved_model, absl::Span<std::string> exported_names,
      mlir::MLIRContext* context, bool add_default_attributes,
      bool unconditionally_use_set_output_shapes);

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
  // The set of node IDs that are on the current DFS stack.
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
  return OkStatus();
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
// mlir::func::FuncOp argument / return.
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
// The raw imported functions have `tensor<*x!tf_type.resource>` as the type for
// mutable bound inputs and `tensor<...>` as the type for immutable
// bound inputs. Here we canonicalize both of them into
// `tensor<!tf_type.resource<tensor<...>>>`.
void AdjustBoundInputArgTypes(mlir::ModuleOp module) {
  mlir::SymbolTable symbol_table(module);
  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    if (!mlir::tf_saved_model::IsExported(func)) continue;
    mlir::OpBuilder builder(func.getBody());
    llvm::SmallVector<mlir::Type, 4> new_input_types;
    for (int i = 0, e = func.getNumArguments(); i < e; i++) {
      auto arg = func.getArgument(i);
      auto global_tensor = mlir::tf_saved_model::LookupBoundInputOfType<
          mlir::tf_saved_model::GlobalTensorOp>(func, i, symbol_table);
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
    func.setType(mlir::FunctionType::get(module.getContext(), new_input_types,
                                         func.getFunctionType().getResults()));
  }
}

// Marks the visibility of functions in the saved model module.
void MarkSavedModelFunctionVisibility(mlir::ModuleOp module) {
  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    auto visibility = mlir::tf_saved_model::IsExported(func)
                          ? mlir::func::FuncOp::Visibility::Public
                          : mlir::func::FuncOp::Visibility::Private;
    func.setVisibility(visibility);
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
    mlir::func::FuncOp func;
  };
  llvm::SmallVector<NamedFunc, 8> named_funcs;
  llvm::SmallVector<mlir::func::FuncOp, 8> private_funcs;
  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    auto exported_names = mlir::tf_saved_model::GetExportedNames(func);
    if (!exported_names.empty())
      named_funcs.push_back({exported_names.front(), func});
    else
      private_funcs.push_back(func);
  }
  llvm::stable_sort(named_funcs, [](const NamedFunc& a, const NamedFunc& b) {
    return a.name < b.name;
  });
  llvm::stable_sort(private_funcs,
                    [](mlir::func::FuncOp a, mlir::func::FuncOp b) {
                      return a.getName() < b.getName();
                    });

  struct NamedAsset {
    llvm::StringRef name;
    AssetOp asset;
  };
  llvm::SmallVector<NamedAsset, 4> assets;
  for (auto asset : module.getOps<AssetOp>()) {
    assets.push_back({asset.getName(), asset});
  }
  llvm::stable_sort(assets, [](const NamedAsset& a, const NamedAsset& b) {
    return a.name < b.name;
  });

  // Move onto the front of the module in reverse of the final desired order.
  for (auto func : llvm::reverse(private_funcs)) {
    func.getOperation()->moveBefore(&module.getBody()->front());
  }
  for (auto named_func : llvm::reverse(named_funcs)) {
    named_func.func.getOperation()->moveBefore(&module.getBody()->front());
  }
  for (auto named_global_tensor : llvm::reverse(named_global_tensors)) {
    named_global_tensor.global_tensor.getOperation()->moveBefore(
        &module.getBody()->front());
  }

  for (auto asset : assets) {
    asset.asset.getOperation()->moveBefore(&module.getBody()->front());
  }

  auto initializers = module.getOps<SessionInitializerOp>();
  if (!initializers.empty()) {
    (*initializers.begin())
        .getOperation()
        ->moveBefore(&module.getBody()->front());
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
        return OkStatus();
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
      auto orig_func = symbol_table.lookup<mlir::func::FuncOp>(
          tf_name_to_mlir_name.find(function.concrete_functions(0))->second);
      mlir::func::FuncOp func = orig_func;
      // If there are potentially references to this func from within the
      // module, create a wrapper around it and decorate the wrapper with the
      // tf_saved_model attributes instead.
      if (!mlir::SymbolTable::symbolKnownUseEmpty(orig_func.getSymNameAttr(),
                                                  &module.getBodyRegion())) {
        func = orig_func.cloneWithoutRegions();
        module.insert(module.getBody()->begin(), func);
        func.addEntryBlock();
        func.setName(builder.getStringAttr("__sm_exported_" +
                                           orig_func.getName().str()));
        llvm::SmallVector<mlir::Value, 4> args_as_values;
        for (auto block_argument : func.getArguments()) {
          args_as_values.push_back(block_argument);
        }
        mlir::OpBuilder body_builder(&func.getBody());
        auto call = body_builder.create<mlir::TF::StatefulPartitionedCallOp>(
            func.getLoc(), orig_func.getFunctionType().getResults(),
            args_as_values,
            mlir::SymbolRefAttr::get(builder.getContext(), orig_func.getName()),
            /*config=*/builder.getStringAttr(""),
            /*config_proto=*/builder.getStringAttr(""),
            /*executor_type=*/builder.getStringAttr(""));
        body_builder.create<mlir::func::ReturnOp>(func.getLoc(),
                                                  call.getResults());
      }
      func->setAttr(
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
      const int input_index_paths_size = input_index_paths.size();
      if (bound_input_base != input_index_paths_size) {
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
        auto symbol_ref = mlir::SymbolRefAttr::get(
            builder.getContext(),
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
      op->setAttr(
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
          /*type=*/mlir::TypeAttr::get(value_attr.getType()),
          /*is_mutable=*/nullptr);
      op->setAttr(
          "tf_saved_model.exported_names",
          builder.getStrArrayAttr(object_names.GetExportedNames(node_id)));
    }
  }
  AdjustBoundInputArgTypes(module);
  module->setAttr("tf_saved_model.semantics", builder.getUnitAttr());
  SortSavedModelModule(module);
  MarkSavedModelFunctionVisibility(module);
  return OkStatus();
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelObjectGraphImporter::Convert(
    SavedModelV2Bundle* saved_model, absl::Span<std::string> exported_names,
    mlir::MLIRContext* context, bool add_default_attributes,
    // TODO(b/200093974): Remove post triage.
    bool unconditionally_use_set_output_shapes) {
  LoadImporterDialects(*context);
  GraphDebugInfo dummy_debug_info;
  const GraphDebugInfo& debug_info =
      saved_model->debug_info() ? *saved_model->debug_info() : dummy_debug_info;

  GraphImportConfig specs;
  specs.prune_unused_nodes = true;
  specs.unconditionally_use_set_output_shapes =
      unconditionally_use_set_output_shapes;
  mlir::OwningOpRef<mlir::ModuleOp> module =
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
  TF_RETURN_IF_ERROR(importer.ConvertDeferredFunctions());

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
  for (auto func :
       llvm::make_early_inc_range(module->getOps<mlir::func::FuncOp>())) {
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

class SimpleSavedModelMLIRImportInput : public SavedModelMLIRImportInput {
 public:
  static StatusOr<SimpleSavedModelMLIRImportInput> Create(
      const MLIRImportOptions& import_options,
      const MetaGraphDef* meta_graph_def, const GraphDebugInfo& debug_info) {
    DCHECK(meta_graph_def);
    GraphDef graph_def = meta_graph_def->graph_def();
    auto graph = std::make_unique<Graph>(OpRegistry::Global());

    if (import_options.upgrade_legacy) {
      TF_RETURN_IF_ERROR(GenerateResourceSharedNameIfEmpty(
          graph_def, graph->flib_def().default_registry()));
    }

    GraphConstructorOptions graph_ctor_options;
    graph_ctor_options.allow_internal_ops = true;
    graph_ctor_options.add_default_attributes = true;
    TF_RETURN_IF_ERROR(
        ConvertGraphDefToGraph(graph_ctor_options, graph_def, graph.get()));

    if (import_options.upgrade_legacy) {
      // TODO(jpienaar): Remove need to const_cast.
      TF_RETURN_IF_ERROR(UpgradeLegacyGraph(
          graph.get(),
          const_cast<FunctionLibraryDefinition*>(&graph->flib_def()),
          /*restrict_functionalization_to_compiled_nodes=*/false));
    }

    return SimpleSavedModelMLIRImportInput(meta_graph_def, debug_info,
                                           std::move(graph));
  }

  SimpleSavedModelMLIRImportInput(const MetaGraphDef* meta_graph_def,
                                  const GraphDebugInfo& debug_info,
                                  std::unique_ptr<Graph> graph)
      : SavedModelMLIRImportInput(meta_graph_def, debug_info),
        graph_(std::move(graph)) {}

  StatusOr<const Graph*> GetSubGraph(absl::string_view name,
                                     GraphImportConfig& specs) override {
    DCHECK(CheckGraphNameValidity(name));
    DCHECK(CheckGraphContainsFeedsAndFetches(specs));
    return graph_.get();
  }

 private:
  bool CheckGraphContainsFeedsAndFetches(const GraphImportConfig& specs) const {
    absl::flat_hash_set<std::string> feed_fetch_nodes;
    for (const auto& iter : specs.inputs) {
      TensorId tensor_id = ParseTensorName(iter.first);
      feed_fetch_nodes.insert(std::string(tensor_id.node()));
    }
    for (const auto& output : llvm::concat<const std::string>(
             specs.outputs, specs.control_outputs)) {
      TensorId tensor_id = ParseTensorName(output);
      feed_fetch_nodes.insert(std::string(tensor_id.node()));
    }

    for (Node* node : graph_->op_nodes()) {
      feed_fetch_nodes.erase(node->name());
    }

    return feed_fetch_nodes.empty();
  }

  bool CheckGraphNameValidity(absl::string_view name) const {
    // If it is one of the signature name, it is valid.
    const auto& signature_defs = meta_graph_def().signature_def();
    if (signature_defs.contains(std::string(name))) return true;

    // If it is the restore graph name, it is valid.
    if (meta_graph_def().has_saver_def() &&
        meta_graph_def().saver_def().restore_op_name() == name)
      return true;

    // If it is the init graph name, it is valid.
    std::string init_op_name;
    if (internal::GetInitOp("", meta_graph_def(), &init_op_name).ok()) {
      if (init_op_name == name) return true;
    }

    return false;
  }

  // `graph_` contains the entire graph in the original MetaGraphDef.
  std::unique_ptr<Graph> graph_;
};

static absl::flat_hash_set<std::string> GetOriginalTfFuncNamesFromGraphDef(
    const GraphDef& graph_def) {
  absl::flat_hash_set<std::string> original_func_tf_names;
  for (const auto& function : graph_def.library().function()) {
    original_func_tf_names.insert(function.signature().name());
  }
  return original_func_tf_names;
}

// A helper class to import a TensorFlow model expressed in SavedModel V1 into
// an MLIR Module in SavedModel dialect.
//
// TODO(b/179683149): Rename this class to avoid confusion with TFLite.
class SavedModelSignatureDefImporterLite {
 public:
  // Main entry point: converts all functions (specified by SignatureDefs) in
  // the given meta graph to an MLIR Module.
  //
  // `import_restore` is introduced to control whether restore graph
  // is imported in eg. SavedModelSignatureDefImporter. Ideally, we don't need
  // this option to control this as restore graph should be always imported.
  // However, right now, SavedModelSignatureDefImporter cannot handle restore
  // graph correctly.
  //
  // TODO(chky): Remove import_restore once the restore graph is correctly
  // handled in SavedModelSignatureDefImporter.
  static StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Convert(
      SavedModelMLIRImportInput& input,
      std::optional<absl::Span<const std::string>> exported_names,
      mlir::MLIRContext* context, bool import_restore = true,
      bool unconditionally_use_set_output_shapes = false) {
    SavedModelSignatureDefImporterLite importer(
        input, exported_names, context, import_restore,
        unconditionally_use_set_output_shapes);
    return importer.ConvertSignatures();
  }

 private:
  SavedModelSignatureDefImporterLite(
      SavedModelMLIRImportInput& input,
      std::optional<absl::Span<const std::string>> exported_names,
      mlir::MLIRContext* context, bool import_restore,
      bool unconditionally_use_set_output_shapes)
      : input_(input),
        original_func_tf_names_(GetOriginalTfFuncNamesFromGraphDef(
            input.meta_graph_def().graph_def())),
        exported_names_(exported_names),
        module_(mlir::ModuleOp::create(mlir::UnknownLoc::get(context))),
        symbol_table_(module_.get()),
        import_restore_(import_restore),
        unconditionally_use_set_output_shapes_(
            unconditionally_use_set_output_shapes) {}

  // Converts the SavedModel to the SavedModel dialect. Creates an MLIR function
  // for each signature.
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSignatures();
  Status ConvertSignature(const std::string& sig_def_key,
                          const SignatureDef& signature_def);

  struct AssetInfo {
    std::string tensor_name;
    mlir::tf_saved_model::AssetOp op;
  };
  StatusOr<std::vector<AssetInfo>> ConvertAssets();
  // Converts the initialization graph in the SavedModel to an MLIR function.
  Status ConvertInitializer(const std::string& target_node_name,
                            const std::vector<AssetInfo>& assets);

  // Converts a graph with feeds and fetches to an MLIR function.
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertGraph(
      const std::string& name,
      const std::vector<std::pair<std::string, TensorInfo>>& inputs,
      const std::vector<std::pair<std::string, TensorInfo>>& outputs,
      const std::vector<std::string> control_outputs,
      std::unordered_map<std::string, std::string>& tf_name_to_mlir_name);

  // Moves the functions in `sub_module` to `module_` and skips the duplicate
  // functions.
  Status MoveConvertedFunctionsToModule(
      absl::string_view name, mlir::ModuleOp sub_module,
      const std::unordered_map<std::string, std::string>& tf_name_to_mlir_name);

  StatusOr<GraphImportConfig::InputArrays> ParseInputArrays(
      llvm::ArrayRef<std::pair<std::string, TensorInfo>> inputs);

 private:
  SavedModelMLIRImportInput& input_;
  absl::flat_hash_set<std::string> original_func_tf_names_;
  std::optional<absl::Span<const std::string>> exported_names_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  absl::Mutex symbol_table_mu_;
  mlir::SymbolTable symbol_table_ ABSL_GUARDED_BY(symbol_table_mu_);
  bool import_restore_ = true;
  bool unconditionally_use_set_output_shapes_ = false;
};

StatusOr<std::vector<SavedModelSignatureDefImporterLite::AssetInfo>>
SavedModelSignatureDefImporterLite::ConvertAssets() {
  std::vector<AssetFileDef> asset_file_defs;
  TF_RETURN_IF_ERROR(
      internal::GetAssetFileDefs(input_.meta_graph_def(), &asset_file_defs));

  std::vector<AssetInfo> results;
  results.reserve(asset_file_defs.size());

  mlir::OpBuilder builder(module_->getBodyRegion());
  unsigned i = 0;  // Use to generate unique sym_name(s) for duplicate assets.
  for (const auto& asset : asset_file_defs) {
    auto asset_op = builder.create<mlir::tf_saved_model::AssetOp>(
        module_->getLoc(),
        /*sym_name=*/
        builder.getStringAttr(
            absl::StrCat("__tf_saved_model_asset", i++, "_", asset.filename())),
        /*filename=*/
        builder.getStringAttr(
            io::JoinPath(kSavedModelAssetsDirectory, asset.filename())));

    results.push_back({asset.tensor_info().name(), asset_op});
  }

  return results;
}

Status SavedModelSignatureDefImporterLite::MoveConvertedFunctionsToModule(
    absl::string_view name, mlir::ModuleOp sub_module,
    const std::unordered_map<std::string, std::string>& tf_name_to_mlir_name) {
  mlir::Builder builder(sub_module.getContext());
  mlir::SymbolTable sub_module_symbol_table(sub_module);

  // Functions originally from graphdef library might have a different name
  // after conversion, we build the set of the converted names
  absl::flat_hash_set<std::string> original_func_mlir_names;
  for (const auto& kv : tf_name_to_mlir_name) {
    if (original_func_tf_names_.contains(kv.first))
      original_func_mlir_names.insert(kv.second);
  }

  // Prefix private functions with the unique signature name, so that it cannot
  // collide with private functions used in the other signatures.
  for (auto func : sub_module.getOps<mlir::func::FuncOp>()) {
    if (mlir::tf_saved_model::IsExported(func)) continue;

    // Skip the original functions from graphdef library
    if (original_func_mlir_names.count(func.getSymName().str())) continue;

    std::string new_sym_name = absl::StrCat(name, "/", func.getSymName().str());
    mlir::StringAttr new_sym_name_attr = builder.getStringAttr(new_sym_name);
    if (mlir::failed(sub_module_symbol_table.replaceAllSymbolUses(
            func, new_sym_name_attr, sub_module)))
      return tensorflow::errors::InvalidArgument(absl::StrCat(
          "SavedModelSignatureDefImporterLite: failed to assign a unique "
          "name to the private function used in a signature: ",
          func.getSymName().str()));

    mlir::SymbolTable::setSymbolName(func, new_sym_name);
  }

  // Copy all functions used by this signature to the final MLIR module.
  for (auto func : sub_module.getOps<mlir::func::FuncOp>()) {
    absl::MutexLock l(&symbol_table_mu_);
    // The insert here is a NO-OP if the function already exists.
    symbol_table_.insert(func.clone());
  }

  return OkStatus();
}

Status SavedModelSignatureDefImporterLite::ConvertInitializer(
    const std::string& target_node_name, const std::vector<AssetInfo>& assets) {
  std::vector<std::pair<std::string, TensorInfo>> inputs;
  inputs.reserve(assets.size());
  for (const auto& asset : assets) {
    TensorInfo tensor_info;
    tensor_info.set_name(asset.tensor_name);
    tensor_info.set_dtype(DT_STRING);
    tensor_info.mutable_tensor_shape();
    inputs.push_back({asset.tensor_name, tensor_info});
  }

  std::unordered_map<std::string, std::string> tf_name_to_mlir_name;
  TF_ASSIGN_OR_RETURN(auto sub_module,
                      ConvertGraph(target_node_name, inputs, {},
                                   {target_node_name}, tf_name_to_mlir_name));

  mlir::SymbolTable sub_symbol_table(*sub_module);

  auto init_func_op =
      sub_symbol_table.lookup<mlir::func::FuncOp>(target_node_name);
  init_func_op->removeAttr("tf.entry_function");

  mlir::OpBuilder builder(module_->getBodyRegion());

  // Bind asset inputs to asset ops.
  DCHECK_EQ(init_func_op.getNumArguments(), assets.size());
  for (const auto& iter : llvm::enumerate(assets)) {
    auto asset_op = iter.value().op;
    init_func_op.setArgAttr(
        iter.index(), "tf_saved_model.bound_input",
        mlir::SymbolRefAttr::get(builder.getContext(), asset_op.getName()));
  }

  // Set the exported name of init function to an reserved name for
  // tf_saved_model.
  init_func_op->setAttr(
      "tf_saved_model.exported_names",
      builder.getStrArrayAttr({absl::StrCat(
          "__tf_saved_model_session_initializer_", target_node_name)}));

  // Move the converted functions to top level MLIR module.
  return MoveConvertedFunctionsToModule(target_node_name, *sub_module,
                                        tf_name_to_mlir_name);
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelSignatureDefImporterLite::ConvertGraph(
    const std::string& name,
    const std::vector<std::pair<std::string, TensorInfo>>& inputs,
    const std::vector<std::pair<std::string, TensorInfo>>& outputs,
    const std::vector<std::string> control_outputs,
    std::unordered_map<std::string, std::string>& tf_name_to_mlir_name) {
  VLOG(1) << "Importing Signature: " << name;

  GraphImportConfig specs;
  specs.graph_func_name = name;
  specs.prune_unused_nodes = true;
  TF_ASSIGN_OR_RETURN(specs.inputs, ParseInputArrays(inputs));
  for (auto& output : outputs) specs.outputs.push_back(output.second.name());
  specs.control_outputs = control_outputs;
  specs.enable_shape_inference = false;
  specs.unconditionally_use_set_output_shapes =
      unconditionally_use_set_output_shapes_;

  TF_ASSIGN_OR_RETURN(const auto* subgraph, input_.GetSubGraph(name, specs));

  // Convert sub-graph to MLIR module.
  return GraphDefImporter::Convert(module_->getContext(), *subgraph,
                                   input_.debug_info(), subgraph->flib_def(),
                                   specs, tf_name_to_mlir_name);
}

Status SavedModelSignatureDefImporterLite::ConvertSignature(
    const std::string& sig_def_key, const SignatureDef& signature_def) {
  // Create local vectors for the input and output and sort them to be
  // deterministic. We don't want anyone to really depend on the order, client
  // should lookup argument/result mapping by attribute name.
  // To avoid accidentally depending on the order we use an unintuitive sorting.
  std::vector<std::pair<std::string, TensorInfo>> inputs(
      signature_def.inputs().begin(), signature_def.inputs().end());
  llvm::sort(inputs, [](const auto& lhs, const auto& rhs) {
    return tensorflow::Fingerprint64(lhs.first) <
           tensorflow::Fingerprint64(rhs.first);
  });
  std::vector<std::pair<std::string, TensorInfo>> outputs(
      signature_def.outputs().begin(), signature_def.outputs().end());
  llvm::sort(outputs, [](const auto& lhs, const auto& rhs) {
    return tensorflow::Fingerprint64(lhs.first) <
           tensorflow::Fingerprint64(rhs.first);
  });

  std::unordered_map<std::string, std::string> tf_name_to_mlir_name;

  // Convert sub-graph to MLIR module.
  TF_ASSIGN_OR_RETURN(
      auto sub_module,
      ConvertGraph(sig_def_key, inputs, outputs, {}, tf_name_to_mlir_name));
  mlir::OpBuilder builder(sub_module->getBodyRegion());

  // Find the FuncOp which corresponds to current SignatureDef.
  mlir::SymbolTable sub_symbol_table(*sub_module);
  auto func_op = sub_symbol_table.lookup<mlir::func::FuncOp>(sig_def_key);
  TF_RET_CHECK(func_op)
      << "Graphdef importer should have created a function named "
      << sig_def_key << ".";

  // Use unique SignatureDef key as exported name.
  func_op->setAttr("tf_saved_model.exported_names",
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
  return MoveConvertedFunctionsToModule(sig_def_key, *sub_module,
                                        tf_name_to_mlir_name);
}

StatusOr<GraphImportConfig::InputArrays>
SavedModelSignatureDefImporterLite::ParseInputArrays(
    llvm::ArrayRef<std::pair<std::string, TensorInfo>> inputs) {
  GraphImportConfig::InputArrays results;
  for (const auto& iter : inputs) {
    const auto& tensor_info = iter.second;

    // TODO(b/184675681): Support other encoding cases.
    //
    // TODO(b/184679394): Add unit test for this check.
    TF_RET_CHECK(tensor_info.encoding_case() == tensorflow::TensorInfo::kName)
        << "Only dense tensor is supported, but got encoding case "
        << tensor_info.encoding_case();

    VLOG(1) << "Importing Signature Input: input_name = " << iter.first
            << ", tensor_info = " << tensor_info.DebugString();

    ArrayInfo array_info;
    array_info.imported_dtype = tensor_info.dtype();

    if (tensor_info.has_tensor_shape()) {
      array_info.shape = tensor_info.tensor_shape();
    } else {
      // If there is no tensor shape in the tensor info, conservatively set
      // unknown_rank to true.
      array_info.shape.set_unknown_rank(true);
    }

    results.insert(std::pair<std::string, ArrayInfo>(tensor_info.name(),
                                                     std::move(array_info)));
  }
  return results;
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelSignatureDefImporterLite::ConvertSignatures() {
  LoadImporterDialects(*module_->getContext());

  const auto& signatures = input_.meta_graph_def().signature_def();
  PopulateTfVersions(module_.get(),
                     input_.meta_graph_def().graph_def().versions());

  llvm::DenseSet<llvm::StringRef> exported_name_set;
  bool import_all_signatures = !exported_names_.has_value();
  if (exported_names_.has_value()) {
    exported_name_set.insert(exported_names_->begin(), exported_names_->end());
  }

  absl::Mutex error_status_mu;  // Needed since `error_status` is non-atomic.
  tensorflow::Status error_status;
  {
    // Start a threadpool to convert signatures, since signature conversion can
    // be time consuming especially for large models. Threadpool destructor
    // blocks until all work is done.
    thread::ThreadPool thread_pool(Env::Default(), "ConvertSignatures",
                                   kNumThreadToConvertSignatures);
    for (const auto& key_and_signature_def : signatures) {
      const std::string& sig_def_key = key_and_signature_def.first;
      const SignatureDef& signature_def = key_and_signature_def.second;

      // It is safe to skip "__saved_model_init_op" since it is an internal
      // signature that is not user-accessible. This signature will be handled
      // in ConvertInitializer().
      if (sig_def_key == "__saved_model_init_op") {
        continue;
      }
      if (!import_all_signatures && exported_name_set.count(sig_def_key) == 0) {
        continue;
      }

      thread_pool.Schedule([&]() {
        auto status = ConvertSignature(sig_def_key, signature_def);
        if (!status.ok()) {
          absl::MutexLock l(&error_status_mu);
          error_status = std::move(status);
        }
      });
    }
  }
  TF_RETURN_IF_ERROR(error_status);

  TF_ASSIGN_OR_RETURN(auto assets, ConvertAssets());

  mlir::OpBuilder builder(module_->getBodyRegion());
  llvm::SmallVector<mlir::Attribute, 2> init_sym_refs;

  if (import_restore_ && input_.meta_graph_def().has_saver_def()) {
    std::vector<AssetInfo> variable_and_assets;

    // Create an AssetOp for the variable checkpoint files. The relative
    // filename is used here.
    auto variable_filename_op = builder.create<mlir::tf_saved_model::AssetOp>(
        module_->getLoc(),
        /*sym_name=*/
        builder.getStringAttr("__tf_saved_model_variables"),
        /*filename=*/
        builder.getStringAttr(io::JoinPath(kSavedModelVariablesDirectory,
                                           kSavedModelVariablesFilename)));
    variable_and_assets.push_back(
        {input_.meta_graph_def().saver_def().filename_tensor_name(),
         variable_filename_op});
    variable_and_assets.insert(variable_and_assets.end(), assets.begin(),
                               assets.end());

    const auto& restore_op_name =
        input_.meta_graph_def().saver_def().restore_op_name();
    TF_RETURN_IF_ERROR(
        ConvertInitializer(restore_op_name, variable_and_assets));
    init_sym_refs.push_back(
        mlir::SymbolRefAttr::get(builder.getContext(), restore_op_name));
  }

  std::string init_op_name;
  TF_RETURN_IF_ERROR(
      internal::GetInitOp("", input_.meta_graph_def(), &init_op_name));
  if (!init_op_name.empty()) {
    TF_RETURN_IF_ERROR(ConvertInitializer(init_op_name, assets));
    init_sym_refs.push_back(
        mlir::SymbolRefAttr::get(builder.getContext(), init_op_name));
  }

  builder.create<mlir::tf_saved_model::SessionInitializerOp>(
      module_->getLoc(), builder.getArrayAttr(init_sym_refs));

  (*module_)->setAttr("tf_saved_model.semantics", builder.getUnitAttr());

  SortSavedModelModule(*module_);
  MarkSavedModelFunctionVisibility(*module_);

  return std::move(module_);
}

// A helper class to import a TensorFlow model expressed in SavedModel V1 into
// an MLIR Module in SavedModel dialect. In addition to importing the model, it
// performs a few graph transformations, including:
//  1) Convert read-only ref variables to resource variables
//  2) Lift resource variables to global_tensors by using a TF session.
class SavedModelSignatureDefImporter {
 public:
  // Main entry point: converts all functions (specified by SignatureDefs) in
  // the given meta graph to an MLIR Module.
  static StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Convert(
      const SavedModelBundle& bundle,
      std::optional<absl::Span<const std::string>> exported_names,
      mlir::MLIRContext* context, tensorflow::MLIRImportOptions options,
      bool lift_varhandle_ops_to_args = true) {
    // debug_info might not be loaded with loader_lite.
    GraphDebugInfo debug_info;
    if (bundle.debug_info != nullptr) debug_info = *bundle.debug_info;

    TF_ASSIGN_OR_RETURN(auto input,
                        SimpleSavedModelMLIRImportInput::Create(
                            options, &bundle.meta_graph_def, debug_info));

    TF_ASSIGN_OR_RETURN(auto module,
                        SavedModelSignatureDefImporterLite::Convert(
                            input, exported_names, context,
                            /*import_restore=*/false));

    mlir::OpBuilder builder(module->getContext());
    (*module)->setAttr("tf_saved_model.under_construction",
                       builder.getUnitAttr());
    TF_RETURN_IF_ERROR(
        LiftVariables(bundle, *module, lift_varhandle_ops_to_args));
    (*module)->removeAttr("tf_saved_model.under_construction");

    return module;
  }

 private:
  // Lifts the variables in `module`.
  static Status LiftVariables(const SavedModelBundle& bundle,
                              mlir::ModuleOp module,
                              bool lift_varhandle_ops_to_args);
};

Status SavedModelSignatureDefImporter::LiftVariables(
    const SavedModelBundle& bundle, mlir::ModuleOp module,
    bool lift_varhandle_ops_to_args) {
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());

  mlir::PassManager pm(module.getContext());
  SetCrashReproducer(pm);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tf_executor::CreateTFExecutorGraphPruningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateExecutorDialectToFunctionalConversionPass());
  pm.addPass(
      mlir::tf_saved_model::CreateRemoveVariablesInSessionInitializerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::
          CreateConvertReadonlyReferenceVariablesToResourceVariablesPass());
  if (mlir::failed(pm.run(module)))
    return diag_handler.Combine(
        errors::Internal("Failed to prepare to lift variables."));

  if (lift_varhandle_ops_to_args) {
    if (failed(mlir::tf_saved_model::MarkInitializedVariablesInFunction(
            module, bundle.GetSession())))
      return diag_handler.Combine(
          errors::Internal("Failed to prepare to mark initialized variables."));
    pm.clear();
    pm.addPass(mlir::TF::CreatePromoteVarHandlesToArgsPass());
    if (mlir::failed(pm.run(module)))
      return diag_handler.Combine(
          errors::Internal("Failed to promote var handles to args."));
    if (failed(
            mlir::tf_saved_model::LiftVariables(module, bundle.GetSession())))
      return diag_handler.Combine(
          errors::Internal("Failed to lift variables."));
  } else {
    if (failed(mlir::tf_saved_model::InitializeVariablesInSessionInitializer(
            module, bundle.GetSession())))
      return diag_handler.Combine(
          errors::Internal("Failed to initialize variables in session init."));
  }

  pm.clear();
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tf_saved_model::CreateDedupBoundInputBindingPass());
  if (mlir::failed(pm.run(module)))
    return diag_handler.Combine(
        errors::Internal("Failed to dedup bound inputs."));

  return OkStatus();
}

}  // namespace

SavedModelMLIRImportInput::~SavedModelMLIRImportInput() {}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertGraphdefToMlir(
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
  if (specs.upgrade_legacy) {
    TF_RETURN_IF_ERROR(GenerateResourceSharedNameIfEmpty(
        preprocessed_graphdef, graph.flib_def().default_registry()));
  }
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
      options, std::move(preprocessed_graphdef), &graph));
  return ConvertGraphToMlir(graph, debug_info, graph.flib_def(), specs,
                            context);
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertGraphToMlir(
    const Graph& graph, const GraphDebugInfo& debug_info,
    const FunctionLibraryDefinition& flib_def, const GraphImportConfig& specs,
    mlir::MLIRContext* context) {
  // TODO(jpienaar): Remove need to const_cast.
  if (specs.upgrade_legacy) {
    TF_RETURN_IF_ERROR(
        UpgradeLegacyGraph(const_cast<Graph*>(&graph),
                           const_cast<FunctionLibraryDefinition*>(&flib_def),
                           specs.restrict_functionalization_to_compiled_nodes));
  }
  std::unordered_map<std::string, std::string> tf_name_to_mlir_name;
  return GraphDefImporter::Convert(context, graph, debug_info, flib_def, specs,
                                   tf_name_to_mlir_name);
}

stream_executor::port::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
ConvertFunctionToMlir(const FunctionBody* fbody,
                      const FunctionLibraryDefinition& flib_def,
                      mlir::MLIRContext* context) {
  tensorflow::GraphDebugInfo dummy_debug_info;
  tensorflow::GraphImportConfig specs;
  specs.graph_func_name = fbody->fdef.signature().name();
  specs.enable_shape_inference = false;
  specs.graph_as_function = true;
  for (const auto* control_ret_node : fbody->control_ret_nodes)
    specs.control_outputs.push_back(control_ret_node->name());
  std::unordered_map<std::string, std::string> tf_name_to_mlir_name;
  return GraphDefImporter::Convert(context, *fbody->graph, dummy_debug_info,
                                   flib_def, specs, tf_name_to_mlir_name);
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelToMlir(
    SavedModelV2Bundle* saved_model, mlir::MLIRContext* context,
    absl::Span<std::string> exported_names, bool add_default_attributes,
    bool unconditionally_use_set_output_shapes) {
  return SavedModelObjectGraphImporter::Convert(
      saved_model, exported_names, context, add_default_attributes,
      unconditionally_use_set_output_shapes);
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelV1ToMlir(
    const SavedModelBundle& saved_model, absl::Span<std::string> exported_names,
    mlir::MLIRContext* context, MLIRImportOptions options,
    bool lift_variables) {
  std::optional<absl::Span<const std::string>> optional_exported_names;
  // TODO(b/187062560): Change ConvertSavedModelV1ToMlir() to take an optional
  // `exported_names` so that it can be configured to import only restore/init
  // graphs.
  if (!exported_names.empty()) optional_exported_names = exported_names;
  return SavedModelSignatureDefImporter::Convert(
      saved_model, optional_exported_names, context, options, lift_variables);
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelV1ToMlirLite(
    const MetaGraphDef& meta_graph_def, const GraphDebugInfo& debug_info,
    std::optional<absl::Span<const std::string>> exported_names,
    mlir::MLIRContext* context, MLIRImportOptions options) {
  TF_ASSIGN_OR_RETURN(auto input, SimpleSavedModelMLIRImportInput::Create(
                                      options, &meta_graph_def, debug_info));
  return ConvertSavedModelV1ToMlirLite(
      input, exported_names, context,
      options.unconditionally_use_set_output_shapes);
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelV1ToMlirLite(
    SavedModelMLIRImportInput& input,
    std::optional<absl::Span<const std::string>> exported_names,
    mlir::MLIRContext* context, bool unconditionally_use_set_output_shapes) {
  return SavedModelSignatureDefImporterLite::Convert(
      input, exported_names, context,
      /*import_restore=*/true, unconditionally_use_set_output_shapes);
}

std::string MlirModuleToString(mlir::ModuleOp module,
                               mlir::OpPrintingFlags flags) {
  std::string txt_module;
  {
    llvm::raw_string_ostream os{txt_module};
    module.print(os, flags);
  }
  return txt_module;
}

std::string MlirModuleToString(mlir::ModuleOp module, bool show_debug_info) {
  mlir::OpPrintingFlags flags;
  if (show_debug_info) flags.enableDebugInfo();
  return MlirModuleToString(module, flags);
}

}  // namespace tensorflow

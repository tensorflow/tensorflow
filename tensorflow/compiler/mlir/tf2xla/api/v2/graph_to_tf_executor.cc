/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/api/v2/graph_to_tf_executor.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
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
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/jit/shape_inference_helpers.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_attr.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/graph_to_tf_executor_util.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/node_order.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_defs.h"
#include "xla/status_macros.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_debug_info_builder.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/crash_analysis.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stack_frame.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

static inline absl::string_view StringRefToView(llvm::StringRef ref) {
  return {ref.data(), ref.size()};
}

namespace tensorflow {
namespace tf2xla {
namespace v2 {

using ::mlir::NamedAttrList;
using ::mlir::TensorType;
using ::tsl::StatusOr;

constexpr absl::string_view kOutputShapesAttrName = "_output_shapes";

void LoadImporterDialects(mlir::MLIRContext& context) {
  // Load dialects involved in the conversion
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialectsImpl(registry, false);
  context.appendDialectRegistry(registry);
  for (llvm::StringRef name : registry.getDialectNames())
    context.getOrLoadDialect(name);
}

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

    stack_traces_ = LoadTracesFromDebugInfo(debug_info_);
  }

  // Returns the inferred function signature of the given function body. Input
  // types are unranked tensor of the respective datatype in the function and
  // result types are inferred by the shape_refiner_. Result types need not be
  // unranked tensors and could be ranked tensors in cases where result type
  // depends on an op with static output shape like tf.Const.
  absl::StatusOr<mlir::FunctionType> InferLibFunctionType(
      const FunctionBody& fbody);

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
  absl::Status PrepareConvert(const Graph& graph,
                              std::unique_ptr<GraphDef> graph_def = nullptr);

  // Converts the prepared graph to a Function and adds it to the module. A set
  // of nodes from the graph are given to converted to the arguments and returns
  // of the function.
  absl::Status Convert(llvm::StringRef func_name, mlir::FunctionType func_type,
                       const absl::InlinedVector<OutputTensor, 4>& arg_nodes,
                       const absl::InlinedVector<OutputTensor, 4>& ret_nodes,
                       const absl::InlinedVector<Node*, 4>& control_ret_nodes,
                       llvm::ArrayRef<mlir::NamedAttribute> attrs);

  // Finds out the function definition for the given function name from the
  // graph and converts it to a function of the module. This method is called
  // on demand because the graph flib_def does not provide an iterator
  // interface.
  absl::Status ConvertLibFunction(llvm::StringRef func_name);

  // Returns the list of nodes in the graph. Nodes are presented in the reverse
  // order of a post-order depth-first visit starting from the graph's source
  // nodes.
  llvm::ArrayRef<Node*> GetOrderedNodes() const { return ordered_nodes_; }

  // Returns the inferred input type at index `idx` of the `node` in the
  // context.
  absl::StatusOr<mlir::Type> InferInputType(const Node& node, int idx,
                                            mlir::Builder builder);

  // Returns the inferred output type at index `idx` of the `node` in the
  // context.
  absl::StatusOr<mlir::Type> InferOutputType(const Node& node, int idx,
                                             mlir::Builder builder);

  // Convert deferred TF functions to the MLIR representation.
  // Conversion is deferred for efficiency reasons, e.g., to limit depth
  // of recursion and reduce stack size pressure.
  absl::Status ConvertDeferredFunctions();

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
  absl::Status AddNodesToShapeRefiner(
      std::unordered_map<string, Node*>* node_name_map);

  // Prune nodes that do not feed into fetch nodes.
  absl::Status PruneUnreachableNodes(
      std::unordered_map<string, Node*>* node_name_map);

  // Converts feeds to Placeholder nodes.
  absl::Status ConvertFeedsToPlaceholders(
      std::unordered_map<string, Node*>* node_name_map);

  // Converts the inferred shape referred to by 'handle' in 'context', with
  // given element type, and returns an MLIR tensor type.
  absl::StatusOr<TensorType> ConvertDataTypeAndShape(
      DataType dtype, const shape_inference::ShapeHandle& handle,
      const std::vector<shape_inference::ShapeAndType>* handle_subtypes,
      shape_inference::InferenceContext* context, mlir::Builder builder);

  // Converts the inferred shape referred to by 'handle' in 'context', with
  // given element type, and returns an MLIR tensor type.
  absl::StatusOr<TensorType> ConvertElementTypeAndShape(
      mlir::Type element_type, const shape_inference::ShapeHandle& handle,
      shape_inference::InferenceContext* context, mlir::Builder builder);

  // Converts the inferred subtypes for an element type to corresponding MLIR
  // types in 'context'.
  absl::StatusOr<ElementSubtypes> ConvertSubtypes(
      const std::vector<shape_inference::ShapeAndType>* handle_subtypes,
      shape_inference::InferenceContext* context, mlir::Builder builder);

  // Converts the tensor proto into an MLIR elements attribute.
  absl::StatusOr<mlir::ElementsAttr> ConvertTensorProto(
      const TensorProto& value) {
    return tensorflow::ConvertTensorProto(value, &builder_);
  }

  // Converts func name in graphdef to mlir::SymbolRefAttribute.
  absl::StatusOr<mlir::FlatSymbolRefAttr> ConvertFunctionCallName(
      const std::string& func_name);

  // Converts the given non-function-call AttrValue to an MLIR Attribute.
  absl::StatusOr<mlir::Attribute> ConvertAttributeValue(const AttrValue& value);

  // Converts the given function-call AttrValue to MLIR Attributes and pushes
  // them to the given attributes list. For example, if there is a kFunc
  // AttrValue {name : foo, attrs : {k1 : bar, k2 : rfc}}, it will convert it to
  // a list of MLIR Attributes: {{base_name : foo}, {base_name.k1 : bar},
  // {base_name.k2 : rfc}}.
  absl::Status ConvertFunctionCallAttribute(const std::string& base_name,
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
  absl::Status ConvertNode(const Node& node);

  // If the input graph represents a while-loop, the edges pointing from a
  // "NextIteration" node to a "Merge" node add cyclic dependencies and make the
  // topological sorting impossible. We need to remove these edges from the
  // input graph to infer shapes and construct a Function. For each
  // "NextIteration" node, there are two operations, "NextIteration.source"
  // and "NextIteration.sink" are added to the MLIR module.
  using BackEdge = BackEdgeHelper::BackEdge;

  // Removes backedges from the input graph. The removed edges are added back to
  // to OpBuilder after the remaining graph is converted to the Function.
  absl::Status RemoveBackedges();

  // Restores backedges removed during shape inference to the final Function.
  absl::Status AddBackedges();

  // Restores a single backedge in the Function by adding a replicated
  // operation before the dst operation.
  absl::Status AddBackedge(mlir::Operation* sink, mlir::Operation* dst,
                           int dst_input);

  // Adds the input arguments and return operation to the function. The
  // arguments are added as basic block argument. Also the argument types and
  // the id of the nodes from the input graph needs to be specified.
  absl::Status ConvertFunctionArgAndRets(
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
  // the combined error absl::Status.
  absl::Status EmitErrorWithLocationStr(const Node& node,
                                        const absl::Status& error_status);

  // Inserts a placeholder node in the graph to replace a feed output tensor,
  // and returns the new placeholder node and a boolean indicating if the
  // original input node was removed from the graph. Uses of the feed output
  // tensor are replaced with this placeholder node. If the feed output tensor
  // is of a single output node, the control dependencies are forwarded to the
  // the placeholder node, and the original node will be removed.
  // Note: This modifies the graph, and so any list of ordered nodes needs to be
  // reconstructed.
  absl::StatusOr<std::pair<Node*, bool>> CreatePlaceholderNodeForFeed(
      const TensorShapeProto& shape, DataType dtype, Node* node, int index,
      const std::unordered_map<string, Node*>& node_name_map);

  // Gets the input and output nodes corresponding to the specified input and
  // output nodes in specs_. If there are no input or output nodes specified,
  // nodes will be empty.
  absl::Status GetInputOutputNodes(
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
  StackTracesMap stack_traces_;
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

// Mapping from node name to feed (index and ArrayInfo). Node name must outlive
// this map.
using FeedsByNode = absl::flat_hash_map<
    absl::string_view,
    absl::flat_hash_map<int, const std::pair<std::string, ArrayInfo>*>>;

// Creates from a `GraphImportConfig::InputArrays` a mapping from a feeds output
// tensor name to index and ArrayInfo. Keys and values are backed by
// `GraphImportConfig::InputArrays`.
absl::StatusOr<FeedsByNode> GetFeedsByNode(
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

absl::Status ImporterBase::ConvertDeferredFunctions() {
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

  return absl::OkStatus();
}

absl::Status ImporterBase::RemoveBackedges() {
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

  ordered_nodes_.clear();
  TopologicalOrdering(
      *graph_, [&](Node* n) { ordered_nodes_.push_back(n); }, GroupByDevice());
  return absl::OkStatus();
}

absl::Status CopyStackTraces(const Graph& from, Graph* to) {
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

  return absl::OkStatus();
}

absl::StatusOr<std::pair<Node*, bool>>
ImporterBase::CreatePlaceholderNodeForFeed(
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

absl::Status ImporterBase::GetInputOutputNodes(
    const std::unordered_map<string, Node*>& node_name_map,
    std::unordered_set<const Node*>* nodes) {
  auto add_node = [&](absl::string_view name) {
    auto it = node_name_map.find(std::string(name));
    if (it == node_name_map.end()) {
      return errors::FailedPrecondition(
          absl::StrCat("Graph does not contain node: ", name));
    }
    nodes->insert(it->second);
    return absl::OkStatus();
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

  return absl::OkStatus();
}

// TODO(jpienaar): Remove this post shape inference on import flag is removed.
absl::Status ImporterBase::AddNodesToShapeRefiner(
    std::unordered_map<string, Node*>* node_name_map) {
  shape_refiner_ =
      std::make_unique<ShapeRefiner>(graph_->versions(), graph_->op_registry());
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
          absl::Status status = shape_refiner_->AddNode(placeholder_node);
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
      absl::Status status = shape_refiner_->AddNode(node);
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
        return absl::OkStatus();
      }

      for (const auto& shape : llvm::enumerate(list.shape())) {
        auto* node_context = shape_refiner_->GetContext(node);
        shape_inference::ShapeHandle handle;
        absl::Status status =
            node_context->MakeShapeFromShapeProto(shape.value(), &handle);
        if (!status.ok()) {
          return EmitErrorWithLocationStr(*node, status);
        }
        node_context->set_output(shape.index(), handle);
      }
      return absl::OkStatus();
    };

    // If it is the argument node, the shape handle is set explicitly, so it
    // can be propagated to the body nodes of the function.
    if (StringPiece(node->type_string()) == FunctionLibraryDefinition::kArgOp) {
      auto* node_context = shape_refiner_->GetContext(node);
      DCHECK(node_context != nullptr);
      if (const AttrValue* attr = node->attrs().Find("shape")) {
        shape_inference::ShapeHandle handle;
        absl::Status status =
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
  ordered_nodes_.clear();
  TopologicalOrdering(
      *graph_, [&](Node* n) { ordered_nodes_.push_back(n); }, GroupByDevice());

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
      absl::Status status =
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
  return absl::OkStatus();
}

absl::StatusOr<mlir::Type> ImporterBase::InferInputType(const Node& node,
                                                        int idx,
                                                        mlir::Builder builder) {
  if (specs_.enable_shape_inference) {
    // TODO(jpienaar): Remove this if shape inference on import flag is removed.
    auto* context = shape_refiner_->GetContext(&node);
    DataType dtype = node.input_type(idx);
    return ConvertDataTypeAndShape(dtype, context->input(idx),
                                   context->input_handle_shapes_and_types(idx),
                                   context, builder);
  }
  DataType dtype = node.properties()->input_types[idx];
  mlir::Type element_type;
  TF_RETURN_IF_ERROR(ConvertDataType(dtype, builder, &element_type));
  return mlir::UnrankedTensorType::get(element_type);
}

absl::StatusOr<mlir::Type> ImporterBase::InferOutputType(
    const Node& node, int idx, mlir::Builder builder) {
  DataType dtype = node.properties()->output_types[idx];

  // Returns output type given inference context.
  auto shape_ic =
      [&](shape_inference::InferenceContext* c) -> absl::StatusOr<mlir::Type> {
    // TODO(b/200093974): Post triage, consider following
    // GraphConstructor::ValidateShape in checking _output_shapes always.
    if (specs_.unconditionally_use_set_output_shapes) {
      if (const AttrValue* attr = node.attrs().Find(kOutputShapesAttrName)) {
        auto& list = attr->list();
        if (list.shape_size() > idx) {
          const TensorShapeProto& p = list.shape()[idx];
          shape_inference::ShapeHandle h;
          absl::Status s = c->MakeShapeFromShapeProto(p, &h);
          if (!s.ok())
            return errors::InvalidArgument(
                "Node '", node.name(), " has an invalid ",
                kOutputShapesAttrName, " attribute (shape #", idx, " error:'",
                s.message(), "')");
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
    shape_inference::InferenceContext* shape_context =
        shape_refiner_->GetContext(&node);
    return shape_ic(shape_context);
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
    return GetTypeFromTFTensorShape(
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
  auto default_type = [&]() -> absl::StatusOr<mlir::Type> {
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
          return GetTypeFromTFTensorShape(shape, element_type);
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
            {mlir::cast<TensorType>(etype)}, builder.getContext()));
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

absl::StatusOr<TensorType> ImporterBase::ConvertDataTypeAndShape(
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

absl::StatusOr<TensorType> ImporterBase::ConvertElementTypeAndShape(
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

  return GetTypeFromTFTensorShape(
      llvm::ArrayRef(dimensions.begin(), dimensions.end()), element_type);
}

absl::StatusOr<ImporterBase::ElementSubtypes> ImporterBase::ConvertSubtypes(
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

absl::Status ImporterBase::ConvertFunctionCallAttribute(
    const std::string& base_name, const AttrValue& value,
    NamedAttrList* attributes) {
  TF_ASSIGN_OR_RETURN(auto func_attr,
                      ConvertFunctionCallName(value.func().name()));
  if (!func_attr) return absl::OkStatus();
  attributes->push_back(builder_.getNamedAttr(base_name, func_attr));

  for (const auto& it : value.func().attr()) {
    auto name = absl::StrCat(base_name, ".", it.first);
    TF_ASSIGN_OR_RETURN(auto value, ConvertAttributeValue(it.second));
    attributes->push_back(builder_.getNamedAttr(name, value));
  }
  return absl::OkStatus();
}

absl::StatusOr<mlir::FlatSymbolRefAttr> ImporterBase::ConvertFunctionCallName(
    const std::string& func_name) {
  // Some ops like XlaHostCompute op uses empty value to represent missing
  // functions. Such attribute values should be defined optional in MLIR
  // definition.
  if (func_name.empty()) return mlir::FlatSymbolRefAttr();

  TF_RETURN_IF_ERROR(ConvertLibFunction(func_name));
  auto mlir_func_name = (*tf_name_to_mlir_name_)[func_name];
  return mlir::SymbolRefAttr::get(builder_.getContext(), mlir_func_name);
}

absl::StatusOr<mlir::Attribute> ImporterBase::ConvertAttributeValue(
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
            llvm::ArrayRef(attrs.begin(), attrs.end()));
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

absl::Status ImporterBase::ConvertLibFunction(llvm::StringRef func_name) {
  // If the library function has been converted already, nothing needs to be
  // done.
  if (tf_name_to_mlir_name_->find(std::string(func_name)) !=
      tf_name_to_mlir_name_->end())
    return absl::OkStatus();

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
  return absl::OkStatus();
}

absl::Status ImporterBase::PruneUnreachableNodes(
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
  return absl::OkStatus();
}

absl::Status ImporterBase::ConvertFeedsToPlaceholders(
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
  return absl::OkStatus();
}

absl::Status ImporterBase::PrepareConvert(const Graph& graph,
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
    ordered_nodes_.clear();
    TopologicalOrdering(
        *graph_, [&](Node* n) { ordered_nodes_.push_back(n); },
        GroupByDevice());
  }

  return absl::OkStatus();
}

absl::Status ImporterBase::Convert(
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
  builder_.createBlock(&graph.getBody());

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

  return absl::OkStatus();
}

absl::Status ImporterBase::ConvertFunctionArgAndRets(
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
    return absl::OkStatus();
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
  for (const auto& ret_and_idx : llvm::enumerate(ret_nodes)) {
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
  builder_.setInsertionPointToEnd(&graph_op.getBody().front());
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

  return absl::OkStatus();
}

mlir::Location ImporterBase::GetLocation(const Node& node) {
  DVLOG(1) << "Getting location for " << node.name() << " " << &node;
  // TODO(b/142400497): What is the semantic contract for locations?
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

    std::shared_ptr<AbstractStackTrace> stack_trace = node.GetStackTrace();

    // Prefer stack traces if available, fallback to debug info if not, and then
    // finally to just name. Older versions of debug info concatenated `@` onto
    // the node name for the default graph, so we check both locations.
    if (stack_trace != nullptr) {
    } else if (stack_traces_.contains(name_for_name_loc)) {
      stack_trace = stack_traces_.at(name_for_name_loc);
    } else if (stack_traces_.contains(debug_info_key)) {
      stack_trace = stack_traces_.at(debug_info_key);
    } else {
      DVLOG(1) << "No stack trace for " << node.name();
    }

    llvm::SmallVector<mlir::Location, 4> locations;

    if (stack_trace != nullptr) {
      DVLOG(1) << "Stack available for " << node.name();
      for (const StackFrame& frame : stack_trace->ToUncachedFrames()) {
        auto file_name = mlir::StringAttr::get(context_, frame.file_name);
        // Use col 1 as there is no column info in StackTrace.
        auto file_line_loc =
            mlir::FileLineColLoc::get(file_name, frame.line_number, 1);
        locations.push_back(file_line_loc);
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
    auto callsite_locs = llvm::ArrayRef(locations).drop_front();
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
      const auto& node_name = original_nodes[i];
      auto func_name = (i < original_funcs.size()) ? original_funcs[i] : "";
      node_locations.push_back(create_location(node_name, func_name));
    }
    // Retrieve the name of the node_def.
    node_locations.push_back(
        create_location(node.name(), function_name_for_debug_info_));
    return mlir::FusedLoc::get(context_, node_locations);
  }
}

absl::Status ImporterBase::EmitErrorWithLocationStr(
    const Node& node, const absl::Status& error_status) {
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
        loc, source_op.getToken(), operands, result.attributes);
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
        loc, mlir::ValueRange(operands), result.attributes);
  }
  // Regular TensorFlow operation are wrapped in a tf_executor.island.
  auto island = builder_.create<mlir::tf_executor::IslandOp>(
      result.location, types, control_operands,
      mlir::ArrayRef<mlir::NamedAttribute>{});
  island.getBody().push_back(new mlir::Block);
  mlir::OpBuilder island_builder =
      mlir::OpBuilder::atBlockEnd(&island.GetBody());

  // Create the operation inside the island now.
  mlir::Operation* inner_op = island_builder.create(result);

  // Sets operand_segment_sizes or result_segment_sizes attribute to the op.
  const auto set_segment_sizes_attr =
      [&](const NameRangeMap& arg_ranges,
          const protobuf::RepeatedPtrField<OpDef::ArgDef>& args,
          llvm::StringRef attr_name) {
        std::vector<int32_t> values;
        values.reserve(args.size());
        for (const auto& arg : args) {
          auto range = arg_ranges.at(arg.name());
          values.push_back(range.second - range.first);
        }
        auto attr_value =
            mlir::DenseI32ArrayAttr::get(inner_op->getContext(), values);
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
          type.walk([&](mlir::Type t) {
            if (resource) return mlir::WalkResult::interrupt();
            if (mlir::isa<mlir::TF::ResourceType>(type)) {
              resource = true;
              return mlir::WalkResult::interrupt();
            }

            return mlir::WalkResult::advance();
          });

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

absl::Status ImporterBase::ConvertNode(const Node& node) {
  if (!node.IsOp()) {
    // Don't import the pseudo-nodes _SOURCE or _SINK. These are added by
    // Graph and don't exist in GraphDef.
    return absl::OkStatus();
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
  return absl::OkStatus();
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
absl::Status ImporterBase::AddBackedges() {
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
  return absl::OkStatus();
}

absl::Status ImporterBase::AddBackedge(mlir::Operation* sink,
                                       mlir::Operation* dst, int dst_input) {
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
  return absl::OkStatus();
}

absl::StatusOr<mlir::FunctionType> ImporterBase::InferLibFunctionType(
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
        arg_types.push_back(GetTypeFromTFTensorShape(shape, element_type));
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
  static absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Convert(
      mlir::MLIRContext* context, const Graph& graph,
      const GraphDebugInfo& debug_info,
      const FunctionLibraryDefinition& flib_def, const GraphImportConfig& specs,
      std::unordered_map<std::string, std::string>* tf_name_to_mlir_name,
      bool disable_crash_analysis = false);

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
  absl::StatusOr<mlir::FunctionType> InferMainFunctionType(
      const GraphImportConfig& specs, mlir::MLIRContext* context,
      absl::InlinedVector<OutputTensor, 4>* arg_nodes,
      absl::InlinedVector<OutputTensor, 4>* ret_nodes);

  // Returns the function signature of the main function, alongside input and
  // output nodes, for function graphs. Arguments and return values are
  // determined by node op type. Type and shape information of the function are
  // inferred by the shape refiner in ImporterBase.
  absl::StatusOr<mlir::FunctionType> GetArgsRetsAndTypesFromFunctionGraph(
      mlir::MLIRContext* context,
      absl::InlinedVector<OutputTensor, 4>* arg_nodes,
      absl::InlinedVector<OutputTensor, 4>* ret_nodes);

  // Finds the graph's target nodes/function's control ret nodes based on
  // supplied node names in `control_outputs`. If `control_outputs` are not
  // unique or a control ret node is missing, an error will be returned.
  absl::Status GetControlRetsFromGraph(
      llvm::ArrayRef<std::string> control_outputs,
      absl::InlinedVector<Node*, 4>* control_ret_nodes);
};

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GraphDefImporter::Convert(
    mlir::MLIRContext* context, const Graph& graph,
    const GraphDebugInfo& debug_info, const FunctionLibraryDefinition& flib_def,
    const GraphImportConfig& specs,
    std::unordered_map<std::string, std::string>* tf_name_to_mlir_name,
    bool disable_crash_analysis) {
  LoadImporterDialects(*context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  NameUniquifier function_name_uniquifier(flib_def);

  // importer.PrepareConvert below will attemp to clone the original `graph`
  // via conversion to the graph def first. Convert graph to graph_def here
  // first and avoid extra copies later.
  auto graph_def = std::make_unique<GraphDef>();
  graph.ToGraphDef(graph_def.get(), /*include_flib_def=*/false);

  auto scope_exit = [&]() {
    std::function<void()> cleanup = []() {};
    if (!disable_crash_analysis) {
      static std::atomic<uint32> counter(0);
      uint32 current_file_prefix = counter++;
      const auto* graph_crash_handle = crash_analysis::ReportProtoDataOnCrash(
          absl::StrCat(current_file_prefix, "_mlir_import_graph.pbtxt"),
          *graph_def);
      auto reachable_flib = flib_def.ReachableDefinitions(*graph_def);
      const auto* flib_crash_handle = crash_analysis::ReportProtoDataOnCrash(
          absl::StrCat(current_file_prefix, "_mlir_import_flib.pbtxt"),
          reachable_flib.ToProto());
      cleanup = [=]() {
        crash_analysis::RemoveReportData(graph_crash_handle);
        crash_analysis::RemoveReportData(flib_crash_handle);
      };
    }

    return llvm::make_scope_exit(std::move(cleanup));
  }();

  VLOG(2) << "Importing: "
          << ::tensorflow::DumpGraphToFile("tf_mlir_importer_base", graph,
                                           &flib_def);

  GraphDefImporter importer(flib_def, debug_info, specs, module.get(),
                            tf_name_to_mlir_name, &function_name_uniquifier);

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
    if (!specs.xla_compile_device_type.empty()) {
      attrs.push_back(
          b.getNamedAttr("_xla_compile_device_type",
                         b.getStringAttr(specs.xla_compile_device_type)));
    }
    attrs.push_back(b.getNamedAttr("allow_soft_placement",
                                   b.getBoolAttr(specs.enable_soft_placement)));
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
  if (specs.convert_all_functions_to_mlir) {
    auto fn_names = graph.flib_def().ListFunctionNames();
    for (const auto& fn_name : fn_names) {
      TF_RETURN_IF_ERROR(importer.ConvertLibFunction(fn_name));
    }
  }
  TF_RETURN_IF_ERROR(importer.ImporterBase::ConvertDeferredFunctions());

  // Mark main function public, others private.
  for (auto function : module.get().getOps<mlir::func::FuncOp>()) {
    auto visibility = function.getName() == graph_func_name
                          ? mlir::func::FuncOp::Visibility::Public
                          : mlir::func::FuncOp::Visibility::Private;
    function.setVisibility(visibility);
  }
  VLOG(2) << "Imported: "
          << tensorflow::DumpMlirOpToFile("tf_mlir_imported_base",
                                          module.get());
  return module;
}

absl::StatusOr<mlir::FunctionType> GraphDefImporter::InferMainFunctionType(
    const GraphImportConfig& specs, mlir::MLIRContext* context,
    absl::InlinedVector<OutputTensor, 4>* arg_nodes,
    absl::InlinedVector<OutputTensor, 4>* ret_nodes) {
  // Find all the input nodes and output nodes.
  // Feeds have been remapped to single output nodes (Placeholder), so an exact
  // name match is sufficient.
  absl::flat_hash_map<absl::string_view, int> inputs;
  for (const auto& input_and_idx : llvm::enumerate(specs.inputs)) {
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
        subtypes.push_back(GetTypeFromTFTensorShape(shape, st_data_type));
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
      arg_types.push_back(GetTypeFromTFTensorShape(shape, element_type));
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

absl::StatusOr<mlir::FunctionType>
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

    return absl::OkStatus();
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
  for (const auto& arg_node_and_idx : llvm::enumerate(*arg_nodes)) {
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
  for (const auto& ret_node_and_idx : llvm::enumerate(*ret_nodes)) {
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

absl::Status GraphDefImporter::GetControlRetsFromGraph(
    llvm::ArrayRef<std::string> control_outputs,
    absl::InlinedVector<Node*, 4>* control_ret_nodes) {
  if (control_outputs.empty()) return absl::OkStatus();

  llvm::SmallDenseMap<llvm::StringRef, int32_t> controls_to_idx;
  for (const auto& control_and_idx : llvm::enumerate(control_outputs))
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

  return absl::OkStatus();
}

bool IsCompiledNode(const Node* n) {
  return n->attrs().Find(tensorflow::kTpuReplicateAttr) ||
         n->attrs().Find(tensorflow::kCompileDeviceTypeAttr);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertGraphToTfExecutor(
    const Graph& graph, const GraphDebugInfo& debug_info,
    const FunctionLibraryDefinition& flib_def, const GraphImportConfig& specs,
    mlir::MLIRContext* context,
    std::unordered_map<std::string, std::string>* tf_name_to_mlir_name,
    const ConfigProto& config_proto,
    tensorflow::TF2XLABridgeVersion bridge_version) {
  if (bridge_version != tensorflow::TF2XLABridgeVersion::kNotBridgeUseCase) {
    bool has_unsupported_features_in_mlir_bridge =
        GraphHasUnsupportedFeaturesInMlirBridge(
            graph, &flib_def, config_proto,
            tensorflow::TF2XLABridgeVersion::kNominal,
            /*single_core_inference_mode=*/false);
    if (has_unsupported_features_in_mlir_bridge) {
      LOG(WARNING)
          << "Graph contains unsupported features in MLIR bridge. "
          << "Use MLIR bridge at your own risk or disable MLIR bridge, e.g., "
          << "tf.config.experimental.disable_mlir_bridge.";
    }
  }

  // TODO(jpienaar): Remove need to const_cast.
  if (specs.upgrade_legacy) {
    NodeFilter node_filter = specs.restrict_functionalization_to_compiled_nodes
                                 ? IsCompiledNode
                                 : NodeFilter{};
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        FunctionalizeControlFlow(
            const_cast<Graph*>(&graph),
            const_cast<FunctionLibraryDefinition*>(&flib_def), node_filter,
            /*include_functions=*/true),
        tensorflow::kFunctionalizeControlFlowFailureMessage);
  }

  std::unordered_map<std::string, std::string> local_tf_name_to_mlir_name;
  TF_ASSIGN_OR_RETURN(
      auto module,
      GraphDefImporter::Convert(context, graph, debug_info, flib_def, specs,
                                tf_name_to_mlir_name == nullptr
                                    ? &local_tf_name_to_mlir_name
                                    : tf_name_to_mlir_name));

  if (specs.set_original_tf_func_name) {
    // Set up the original function names in the imported TF MLIR.
    mlir::Builder builder(module->getContext());
    mlir::SymbolTable symbol_table(*module);
    for (const auto& [tf_name, mlir_name] :
         (tf_name_to_mlir_name == nullptr ? local_tf_name_to_mlir_name
                                          : *tf_name_to_mlir_name)) {
      auto func_op = symbol_table.lookup<mlir::func::FuncOp>(mlir_name);
      TF_RET_CHECK(func_op)
          << "Graphdef importer should have created a function named "
          << mlir_name << ".";
      func_op->setAttr("tf._original_func_name",
                       builder.getStringAttr(tf_name));
    }
  }
  return module;
}

}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow

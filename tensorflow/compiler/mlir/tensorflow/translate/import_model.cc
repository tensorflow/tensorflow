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

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/LogicalResult.h"
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
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/bundle_v2.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/loader_util.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/initialize_variables_in_session_init.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lift_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/mark_initialized_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/graph_to_tf_executor.h"
#include "xla/status_macros.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/core/protobuf/trackable_object_graph.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

constexpr size_t kNumThreadToConvertSignatures = 10;

using ::mlir::NamedAttrList;
using ::mlir::TensorType;
using ::mlir::tf_saved_model::AssetOp;
using ::mlir::tf_saved_model::GlobalTensorOp;
using ::mlir::tf_saved_model::kTfSavedModelExportedNamesAttr;
using ::mlir::tf_saved_model::kTfSavedModelIndexPathAttr;
using ::mlir::tf_saved_model::kTfSavedModelInitializerInitType;
using ::mlir::tf_saved_model::kTfSavedModelInitializerRestoreType;
using ::mlir::tf_saved_model::kTfSavedModelInitializerTypeAttr;
using ::mlir::tf_saved_model::SessionInitializerOp;
using ::tsl::StatusOr;

namespace {


void LoadImporterDialects(mlir::MLIRContext& context) {
  // Load dialects involved in the conversion
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialectsImpl(registry, false);
  context.appendDialectRegistry(registry);
  for (llvm::StringRef name : registry.getDialectNames())
    context.getOrLoadDialect(name);
}

absl::StatusOr<std::string> GetDenseTensorNameFromTensorInfo(
    const TensorInfo& tensor_info) {
  // TODO(b/184675681): Support other encoding cases.
  //
  // TODO(b/184679394): Add unit test for this check.
  TF_RET_CHECK(tensor_info.encoding_case() == tensorflow::TensorInfo::kName)
      << "Only dense tensor is supported, but got encoding case "
      << tensor_info.encoding_case();
  return tensor_info.name();
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
  if (it == inputs.end()) return absl::OkStatus();

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
  return absl::OkStatus();
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
  return absl::OkStatus();
}




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
  absl::flat_hash_set<int> on_stack_nodes_;

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
  return absl::OkStatus();
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
  absl::StatusOr<llvm::ArrayRef<mlir::ArrayAttr>> GetLeafIndexPaths(
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

absl::StatusOr<llvm::ArrayRef<mlir::ArrayAttr>>
StructuredValueLinearizer::GetLeafIndexPaths(
    llvm::StringRef error_context) const {
  if (error_message_.empty()) {
    return llvm::ArrayRef(leaf_index_paths_);
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
        if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(path_element)) {
          os << integer.getValue();
        } else {
          auto str = mlir::cast<mlir::StringAttr>(path_element);
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
        if (global_tensor.getIsMutable()) {
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
    SavedModelV2Bundle* saved_model, MLIRImportOptions import_options) {
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
        return absl::OkStatus();
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
          kTfSavedModelExportedNamesAttr,
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
      for (const auto& index_path : llvm::enumerate(input_index_paths)) {
        func.setArgAttr(index_path.index(), kTfSavedModelIndexPathAttr,
                        index_path.value());
      }

      for (const auto& bound_input :
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
      for (const auto& index_path : llvm::enumerate(output_index_paths)) {
        func.setResultAttr(index_path.index(), kTfSavedModelIndexPathAttr,
                           index_path.value());
      }
    } else if (object.kind_case() == SavedObject::kVariable) {
      const SavedVariable& variable = object.variable();
      // Find the trackable in the side data structure.
      auto variable_trackable_it = restored_objects.find(node_id);

      TF_ASSIGN_OR_RETURN(
          auto type, ConvertToMlirTensorType(variable.shape(), variable.dtype(),
                                             &builder));

      if (variable_trackable_it == restored_objects.end()) {
        if (!import_options.allow_uninitialized_variables) {
          return errors::FailedPrecondition(
              "Could not restore saved variable: ", variable.name());
        }

        // The user indicated we should allow loading the model with
        // uninitialized variables, use the type information to construct a
        // dummy uninitialized variable operation.
        auto op = builder.create<mlir::tf_saved_model::GlobalTensorOp>(
            builder.getUnknownLoc(),
            builder.getStringAttr(object_names.GetSymbolTableName(node_id)),
            mlir::ElementsAttr(),
            /*type=*/mlir::TypeAttr::get(type),
            /*is_mutable=*/builder.getUnitAttr());
        op->setAttr(
            kTfSavedModelExportedNamesAttr,
            builder.getStrArrayAttr(object_names.GetExportedNames(node_id)));
      } else {
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
        // A variable can have a partially known type, such as
        // tensor<?x27x?xf32>, even if the initializer is a specific static
        // shape.
        auto op = builder.create<GlobalTensorOp>(
            builder.getUnknownLoc(),
            builder.getStringAttr(object_names.GetSymbolTableName(node_id)),
            value_attr,
            /*type=*/mlir::TypeAttr::get(type),
            /*is_mutable=*/builder.getUnitAttr());
        op->setAttr(
            kTfSavedModelExportedNamesAttr,
            builder.getStrArrayAttr(object_names.GetExportedNames(node_id)));
      }

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
          kTfSavedModelExportedNamesAttr,
          builder.getStrArrayAttr(object_names.GetExportedNames(node_id)));
    }
  }
  AdjustBoundInputArgTypes(module);
  module->setAttr("tf_saved_model.semantics", builder.getUnitAttr());
  SortSavedModelModule(module);
  MarkSavedModelFunctionVisibility(module);
  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelObjectGraph(
    SavedModelV2Bundle* saved_model, absl::Span<std::string> exported_names,
    mlir::MLIRContext* context, MLIRImportOptions import_options) {
  LoadImporterDialects(*context);
  GraphDebugInfo dummy_debug_info;
  const GraphDebugInfo& debug_info =
      saved_model->debug_info() ? *saved_model->debug_info() : dummy_debug_info;

  GraphImportConfig specs;
  specs.prune_unused_nodes = true;
  specs.unconditionally_use_set_output_shapes =
      import_options.unconditionally_use_set_output_shapes;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  std::unordered_map<std::string, std::string> tf_name_to_mlir_name;

  const auto& graphdef = saved_model->meta_graph_def().graph_def();
  PopulateTfVersions(module.get(), graphdef.versions());

  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.add_default_attributes = import_options.add_default_attributes;
  Graph graph(OpRegistry::Global());

  GraphDef preprocessed_graphdef(graphdef);
  if (import_options.add_default_attributes) {
    TF_RETURN_IF_ERROR(PreprocessGraphDef(nullptr, &preprocessed_graphdef));
  }

  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
      options, std::move(preprocessed_graphdef), &graph));

  NameUniquifier function_name_uniquifier(graph.flib_def());
  for (const auto& fn_name : graph.flib_def().ListFunctionNames()) {
    std::string mlir_func_name(function_name_uniquifier.GetUniqueName(fn_name));
    (tf_name_to_mlir_name)[std::string(fn_name)] = mlir_func_name;
  }

  specs.convert_all_functions_to_mlir = true;
  TF_ASSIGN_OR_RETURN(
      module, ConvertGraphToMlir(graph, debug_info, graph.flib_def(), specs,
                                 module->getContext()));

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
    if (func.getName().starts_with("__inference__traced_save_") ||
        func.getName().starts_with("__inference__traced_restore_") ||
        func.getName().starts_with("__inference_signature_wrapper_") ||
        func.getName().starts_with("main")) {
      func.erase();
    }
  }

  // Diagnose SavedFunction's with multiple input signatures.
  TF_RETURN_IF_ERROR(
      DiagnoseMultipleConcreteFunctions(object_graph, object_names));

  // Construct the SavedModel IR.
  TF_RETURN_IF_ERROR(CreateSavedModelIR(object_names, module.get(),
                                        object_graph, tf_name_to_mlir_name,
                                        saved_model, import_options));
  assert(mlir::succeeded(mlir::verify(module.get())));

  return module;
}

class SimpleSavedModelMLIRImportInput : public SavedModelMLIRImportInput {
 public:
  static absl::StatusOr<SimpleSavedModelMLIRImportInput> Create(
      const MLIRImportOptions& import_options,
      const MetaGraphDef* meta_graph_def, const GraphDebugInfo& debug_info) {
    DCHECK(meta_graph_def);
    GraphDef graph_def(meta_graph_def->graph_def());
    auto graph = std::make_unique<Graph>(OpRegistry::Global());

    if (import_options.upgrade_legacy) {
      TF_RETURN_IF_ERROR(GenerateResourceSharedNameIfEmpty(
          graph_def, graph->flib_def().default_registry()));
    }

    GraphConstructorOptions graph_ctor_options;
    graph_ctor_options.allow_internal_ops = true;
    graph_ctor_options.add_default_attributes = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
        graph_ctor_options, std::move(graph_def), graph.get()));

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

  absl::StatusOr<const Graph*> GetSubGraph(absl::string_view name,
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
  static absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Convert(
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
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSignatures();
  Status ConvertSignature(const std::string& sig_def_key,
                          const SignatureDef& signature_def);

  struct AssetInfo {
    std::string tensor_name;
    mlir::tf_saved_model::AssetOp op;
  };
  absl::StatusOr<std::vector<AssetInfo>> ConvertAssets();

  // Converts the initialization graph in the SavedModel to an MLIR function.
  // Attaches `tf_saved_model.initializer_type` attribute with value
  // `initializer_type` to the created function.
  Status ConvertInitializer(const std::string& target_node_name,
                            const std::vector<AssetInfo>& assets,
                            llvm::StringRef initializer_type);

  // Converts a graph with feeds and fetches to an MLIR function.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertGraph(
      const std::string& name,
      const std::vector<std::pair<std::string, TensorInfo>>& inputs,
      const std::vector<std::pair<std::string, TensorInfo>>& outputs,
      std::vector<std::string> control_outputs,
      std::unordered_map<std::string, std::string>* tf_name_to_mlir_name);

  // Moves the functions in `sub_module` to `module_` and skips the duplicate
  // functions.
  Status MoveConvertedFunctionsToModule(
      absl::string_view name, mlir::ModuleOp sub_module,
      const std::unordered_map<std::string, std::string>& tf_name_to_mlir_name);

  absl::StatusOr<GraphImportConfig::InputArrays> ParseInputArrays(
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

absl::StatusOr<std::vector<SavedModelSignatureDefImporterLite::AssetInfo>>
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

  return absl::OkStatus();
}

Status SavedModelSignatureDefImporterLite::ConvertInitializer(
    const std::string& target_node_name, const std::vector<AssetInfo>& assets,
    llvm::StringRef initializer_type) {
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
                                   {target_node_name}, &tf_name_to_mlir_name));

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
      kTfSavedModelExportedNamesAttr,
      builder.getStrArrayAttr({absl::StrCat(
          "__tf_saved_model_session_initializer_", target_node_name)}));
  init_func_op->setAttr(kTfSavedModelInitializerTypeAttr,
                        builder.getStringAttr(initializer_type));

  // Move the converted functions to top level MLIR module.
  return MoveConvertedFunctionsToModule(target_node_name, *sub_module,
                                        tf_name_to_mlir_name);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelSignatureDefImporterLite::ConvertGraph(
    const std::string& name,
    const std::vector<std::pair<std::string, TensorInfo>>& inputs,
    const std::vector<std::pair<std::string, TensorInfo>>& outputs,
    const std::vector<std::string> control_outputs,
    std::unordered_map<std::string, std::string>* tf_name_to_mlir_name) {
  VLOG(1) << "Importing Signature: " << name;

  GraphImportConfig specs;
  specs.graph_func_name = name;
  specs.prune_unused_nodes = true;
  TF_ASSIGN_OR_RETURN(specs.inputs, ParseInputArrays(inputs));
  for (auto& output : outputs) {
    TF_ASSIGN_OR_RETURN(std::string name,
                        GetDenseTensorNameFromTensorInfo(output.second));
    specs.outputs.push_back(std::move(name));
  }
  specs.control_outputs = control_outputs;
  specs.enable_shape_inference = false;
  specs.unconditionally_use_set_output_shapes =
      unconditionally_use_set_output_shapes_;

  TF_ASSIGN_OR_RETURN(const auto* subgraph, input_.GetSubGraph(name, specs));

  // Convert sub-graph to MLIR module.
  return ConvertGraphToMlir(*subgraph, input_.debug_info(),
                            subgraph->flib_def(), specs, module_->getContext(),
                            tf_name_to_mlir_name);
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
      ConvertGraph(sig_def_key, inputs, outputs, {}, &tf_name_to_mlir_name));
  mlir::OpBuilder builder(sub_module->getBodyRegion());

  // Find the FuncOp which corresponds to current SignatureDef.
  mlir::SymbolTable sub_symbol_table(*sub_module);
  auto func_op = sub_symbol_table.lookup<mlir::func::FuncOp>(sig_def_key);
  TF_RET_CHECK(func_op)
      << "Graphdef importer should have created a function named "
      << sig_def_key << ".";

  // Use unique SignatureDef key as exported name.
  func_op->setAttr(kTfSavedModelExportedNamesAttr,
                   builder.getStrArrayAttr({sig_def_key}));

  // Transfer input and output parameter names to index_path attributes.
  for (const auto& input_and_idx : llvm::enumerate(inputs)) {
    func_op.setArgAttr(input_and_idx.index(), kTfSavedModelIndexPathAttr,
                       builder.getStrArrayAttr({input_and_idx.value().first}));
  }
  for (const auto& output_and_idx : llvm::enumerate(outputs)) {
    func_op.setResultAttr(
        output_and_idx.index(), kTfSavedModelIndexPathAttr,
        builder.getStrArrayAttr({output_and_idx.value().first}));
  }

  // Add the original TF function name as a function attribute.
  // TODO(b/258817244) Remove this after TFRT exports functions.
  for (const auto& [tf_name, mlir_name] : tf_name_to_mlir_name) {
    auto func_op = sub_symbol_table.lookup<mlir::func::FuncOp>(mlir_name);
    TF_RET_CHECK(func_op)
        << "Graphdef importer should have created a function named "
        << mlir_name << ".";
    func_op->setAttr("tf._original_func_name", builder.getStringAttr(tf_name));
  }

  // Move the converted functions to top level MLIR module.
  return MoveConvertedFunctionsToModule(sig_def_key, *sub_module,
                                        tf_name_to_mlir_name);
}

absl::StatusOr<GraphImportConfig::InputArrays>
SavedModelSignatureDefImporterLite::ParseInputArrays(
    llvm::ArrayRef<std::pair<std::string, TensorInfo>> inputs) {
  GraphImportConfig::InputArrays results;
  for (const auto& iter : inputs) {
    const auto& tensor_info = iter.second;

    TF_ASSIGN_OR_RETURN(std::string name,
                        GetDenseTensorNameFromTensorInfo(tensor_info));

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

    results.insert(std::pair<std::string, ArrayInfo>(std::move(name),
                                                     std::move(array_info)));
  }
  return results;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
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
    TF_RETURN_IF_ERROR(ConvertInitializer(restore_op_name, variable_and_assets,
                                          kTfSavedModelInitializerRestoreType));
    init_sym_refs.push_back(
        mlir::SymbolRefAttr::get(builder.getContext(), restore_op_name));
  }

  std::string init_op_name;
  TF_RETURN_IF_ERROR(
      internal::GetInitOp("", input_.meta_graph_def(), &init_op_name));
  if (!init_op_name.empty()) {
    TF_RETURN_IF_ERROR(ConvertInitializer(init_op_name, assets,
                                          kTfSavedModelInitializerInitType));
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
  static absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Convert(
      const SavedModelBundle& bundle,
      std::optional<absl::Span<const std::string>> exported_names,
      mlir::MLIRContext* context, tensorflow::MLIRImportOptions options) {
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
        LiftVariables(bundle, *module, options.lift_variables,
                      options.include_variables_in_initializers));
    (*module)->removeAttr("tf_saved_model.under_construction");

    return module;
  }

 private:
  // Lifts the variables in `module`.
  // If `include_variables_in_initializers` is set to false, then it removes all
  // variables from the initializer functions (registered in the
  // `tf_saved_model::SessionInitializerOp`) by running the
  // `RemoveVariablesInSessionInitializerPass`, regardless of whether
  // `lift_variable_ops_to_args` is true or not.
  static Status LiftVariables(const SavedModelBundle& bundle,
                              mlir::ModuleOp module,
                              bool lift_varhandle_ops_to_args,
                              bool include_variables_in_initializers);
};

Status SavedModelSignatureDefImporter::LiftVariables(
    const SavedModelBundle& bundle, mlir::ModuleOp module,
    const bool lift_varhandle_ops_to_args,
    const bool include_variables_in_initializers) {
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());

  mlir::PassManager pm(module.getContext());
  SetCrashReproducer(pm);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tf_executor::CreateTFExecutorGraphPruningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateExecutorDialectToFunctionalConversionPass());
  if (!include_variables_in_initializers) {
    pm.addPass(
        mlir::tf_saved_model::CreateRemoveVariablesInSessionInitializerPass());
  }
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

  return absl::OkStatus();
}

}  // namespace

SavedModelMLIRImportInput::~SavedModelMLIRImportInput() = default;

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertGraphdefToMlir(
    const GraphDef& graphdef, const GraphDebugInfo& debug_info,
    const GraphImportConfig& specs, mlir::MLIRContext* context) {
  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  Graph graph(OpRegistry::Global());
  GraphDef preprocessed_graphdef(graphdef);
  TF_RETURN_IF_ERROR(PreprocessGraphDef(&specs, &preprocessed_graphdef));

  if (specs.upgrade_legacy) {
    TF_RETURN_IF_ERROR(GenerateResourceSharedNameIfEmpty(
        preprocessed_graphdef, graph.flib_def().default_registry()));
  }
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
      options, std::move(preprocessed_graphdef), &graph));
  return ConvertGraphToMlir(graph, debug_info, graph.flib_def(), specs,
                            context);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertGraphToMlir(
    const Graph& graph, const GraphDebugInfo& debug_info,
    const FunctionLibraryDefinition& flib_def, const GraphImportConfig& specs,
    mlir::MLIRContext* context,
    std::unordered_map<std::string, std::string>* tf_name_to_mlir_name) {
  return tensorflow::tf2xla::v2::ConvertGraphToTfExecutor(
      graph, debug_info, flib_def, specs, context, tf_name_to_mlir_name);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertFunctionToMlir(
    const FunctionBody* fbody, const FunctionLibraryDefinition& flib_def,
    mlir::MLIRContext* context) {
  tensorflow::GraphDebugInfo dummy_debug_info;
  tensorflow::GraphImportConfig specs;
  specs.graph_func_name = fbody->record->fdef().signature().name();
  specs.enable_shape_inference = false;
  specs.graph_as_function = true;
  for (const auto* control_ret_node : fbody->control_ret_nodes)
    specs.control_outputs.push_back(control_ret_node->name());
  return ConvertGraphToMlir(*fbody->graph, dummy_debug_info, flib_def, specs,
                            context);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelToMlir(
    SavedModelV2Bundle* saved_model, mlir::MLIRContext* context,
    absl::Span<std::string> exported_names, MLIRImportOptions options) {
  return ConvertSavedModelObjectGraph(saved_model, exported_names, context,
                                      options);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelV1ToMlir(
    const SavedModelBundle& saved_model, absl::Span<std::string> exported_names,
    mlir::MLIRContext* context, MLIRImportOptions options) {
  std::optional<absl::Span<const std::string>> optional_exported_names;
  // TODO(b/187062560): Change ConvertSavedModelV1ToMlir() to take an optional
  // `exported_names` so that it can be configured to import only restore/init
  // graphs.
  if (!exported_names.empty()) optional_exported_names = exported_names;
  return SavedModelSignatureDefImporter::Convert(
      saved_model, optional_exported_names, context, options);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelV1ToMlirLite(
    const MetaGraphDef& meta_graph_def, const GraphDebugInfo& debug_info,
    std::optional<absl::Span<const std::string>> exported_names,
    mlir::MLIRContext* context, MLIRImportOptions options) {
  TF_ASSIGN_OR_RETURN(auto input, SimpleSavedModelMLIRImportInput::Create(
                                      options, &meta_graph_def, debug_info));
  return ConvertSavedModelV1ToMlirLite(
      input, exported_names, context,
      options.unconditionally_use_set_output_shapes);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertSavedModelV1ToMlirLite(
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

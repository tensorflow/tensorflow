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

#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/bundle_v2.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/utils/transitive_fanin.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/tensor_bundle/byte_swap_tensor.h"

namespace tensorflow {

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
absl::Status UpdateLegacyFedInputNode(
    const GraphDef& graph_def, const GraphImportConfig::InputArrays& inputs,
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
  if (dtype == DT_INVALID &&
      node->attr().at("output_types").list().type_size() > 0) {
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

static absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GraphdefToMlirImport(
    llvm::StringRef input, const std::vector<std::string>& input_arrays,
    const std::vector<std::string>& input_dtypes,
    const std::vector<std::optional<std::vector<int>>>& input_shapes,
    const std::vector<std::string>& output_arrays,
    const std::vector<std::string>& control_output_arrays,
    const GraphdefToMlirOptions& import_options, mlir::MLIRContext* context) {
  GraphDef graphdef;
  TF_RETURN_IF_ERROR(
      tensorflow::LoadProtoFromBuffer({input.data(), input.size()}, &graphdef));
  if (!port::kLittleEndian)
    TF_RETURN_IF_ERROR(ByteSwapTensorContentInGraphDef(&graphdef));

  GraphDebugInfo debug_info;
  if (!import_options.debug_info_file.empty()) {
    TF_RETURN_IF_ERROR(
        LoadProtoFromFile(import_options.debug_info_file, &debug_info));
  }

  GraphImportConfig specs;
  specs.prune_unused_nodes = import_options.prune_unused_nodes;
  specs.convert_legacy_fed_inputs = import_options.convert_legacy_fed_inputs;
  specs.graph_as_function = import_options.graph_as_function;
  specs.upgrade_legacy = import_options.upgrade_legacy;
  specs.enable_shape_inference = import_options.enable_shape_inference;
  specs.unconditionally_use_set_output_shapes =
      import_options.unconditionally_use_set_output_shapes;
  specs.xla_compile_device_type = import_options.xla_compile_device_type;
  specs.enable_soft_placement = import_options.enable_soft_placement;
  specs.set_original_tf_func_name = import_options.set_original_tf_func_name;
  TF_RETURN_IF_ERROR(ParseInputArrayInfo(input_arrays, input_dtypes,
                                         input_shapes, &specs.inputs));
  TF_RETURN_IF_ERROR(ParseOutputArrayInfo(output_arrays, &specs.outputs));
  TF_RETURN_IF_ERROR(
      ParseOutputArrayInfo(control_output_arrays, &specs.control_outputs));
  // TODO(hinsu): Completely deprecate support for LegacyFedInput ops. One
  // solution could be have a tool to let users upgrade old serialized graphs.
  for (auto& node_def : *graphdef.mutable_node()) {
    if (specs.convert_legacy_fed_inputs && node_def.op() == "LegacyFedInput") {
      TF_RETURN_IF_ERROR(
          UpdateLegacyFedInputNode(graphdef, specs.inputs, &node_def));
    }
  }
  // TODO(b/142828368): Pruning should not be needed when TF import
  // supports importing graphs w/ unregistered ops natively.
  GraphDef pruned_graph_def;
  if (specs.prune_unused_nodes) {
    std::vector<std::string> terminal_nodes;
    terminal_nodes.reserve(specs.outputs.size() + specs.inputs.size());
    for (const auto& output : specs.outputs) {
      terminal_nodes.push_back(std::string(ParseTensorName(output).node()));
    }
    for (const auto& control_output : specs.control_outputs) {
      terminal_nodes.push_back(std::string(control_output));
    }
    for (const auto& input : specs.inputs) {
      terminal_nodes.push_back(input.first);
    }
    TF_RETURN_IF_ERROR(tensorflow::grappler::SetTransitiveFaninGraph(
        graphdef, &pruned_graph_def, terminal_nodes));
    // TODO(ashwinm): Add a separate utility in grappler utils that abstracts
    // both SetTransitiveFaninGraph and restoring the missing contents from the
    // original graph like function def library and version.
    pruned_graph_def.mutable_library()->Swap(graphdef.mutable_library());
    pruned_graph_def.mutable_versions()->Swap(graphdef.mutable_versions());
  }
  return ConvertGraphdefToMlir(
      specs.prune_unused_nodes ? pruned_graph_def : graphdef, debug_info, specs,
      context);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
GraphdefToMlirTranslateFunction(
    llvm::StringRef input, const std::vector<std::string>& input_arrays,
    const std::vector<std::string>& input_dtypes,
    const std::vector<std::optional<std::vector<int>>>& input_shapes,
    const std::vector<std::string>& output_arrays,
    const std::vector<std::string>& control_output_arrays,
    const GraphdefToMlirOptions& import_options, mlir::MLIRContext* context) {
  auto module_or = GraphdefToMlirImport(
      input, input_arrays, input_dtypes, input_shapes, output_arrays,
      control_output_arrays, import_options, context);
  if (!module_or.status().ok()) {
    LOG(ERROR) << "Graph import failed: " << module_or.status();
  }
  return module_or;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
GraphdefToMlirTranslateFunction(
    llvm::StringRef input, absl::string_view input_arrays,
    absl::string_view input_dtypes, absl::string_view input_shapes,
    absl::string_view output_arrays, absl::string_view control_output_arrays,
    const GraphdefToMlirOptions& import_options, mlir::MLIRContext* context) {
  std::vector<std::string> input_array_vector;
  std::vector<std::string> input_dtype_vector;
  std::vector<std::optional<std::vector<int>>> input_shapes_vector;
  std::vector<std::string> output_array_vector;
  std::vector<std::string> control_output_array_vector;
  TF_RETURN_IF_ERROR(ParseNodeNames(input_arrays, input_array_vector));
  TF_RETURN_IF_ERROR(ParseNodeDataTypes(input_dtypes, input_dtype_vector));
  TF_RETURN_IF_ERROR(ParseNodeNames(output_arrays, output_array_vector));
  TF_RETURN_IF_ERROR(ParseNodeShapes(input_shapes, input_shapes_vector));
  TF_RETURN_IF_ERROR(
      ParseNodeNames(control_output_arrays, control_output_array_vector));
  return GraphdefToMlirTranslateFunction(
      input, input_array_vector, input_dtype_vector, input_shapes_vector,
      output_array_vector, control_output_array_vector, import_options,
      context);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelObjectGraphToMlirImport(absl::string_view saved_model_dir,
                                  const std::unordered_set<std::string>& tags,
                                  absl::Span<std::string> exported_names,
                                  mlir::MLIRContext* context,
                                  bool unconditionally_use_set_output_shapes) {
  tensorflow::SavedModelV2Bundle bundle;
  auto load_status = tensorflow::SavedModelV2Bundle::Load(
      std::string(saved_model_dir.data(), saved_model_dir.length()), &bundle);
  if (!load_status.ok()) {
    LOG(ERROR) << "Failed to load saved model '" << saved_model_dir
               << "': " << load_status;
    return load_status;
  }

  MLIRImportOptions options;
  options.add_default_attributes = true;
  options.unconditionally_use_set_output_shapes =
      unconditionally_use_set_output_shapes;

  auto module_or =
      ConvertSavedModelToMlir(&bundle, context, exported_names, options);
  if (!module_or.status().ok()) {
    LOG(ERROR) << "SavedModel import failed: " << module_or.status();
  }
  return module_or;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelSignatureDefsToMlirImport(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context,
    MLIRImportOptions options,
    std::unique_ptr<tensorflow::SavedModelBundle>* saved_model_bundle) {
  // Create local bundle if no one is provided to use.
  std::unique_ptr<tensorflow::SavedModelBundle> bundle;
  if (saved_model_bundle == nullptr) {
    bundle = std::make_unique<tensorflow::SavedModelBundle>();
  } else if (*saved_model_bundle == nullptr) {
    *saved_model_bundle = std::make_unique<tensorflow::SavedModelBundle>();
  }
  SavedModelBundle* bundle_ptr =
      saved_model_bundle ? saved_model_bundle->get() : bundle.get();
  tensorflow::SessionOptions session_options;

  // Force saved model states to be restored to CPU.
  (*session_options.config.mutable_device_count())["GPU"] = 0;
  auto load_status = tensorflow::LoadSavedModel(
      session_options, /* run_options = */ {}, std::string(saved_model_dir),
      tags, bundle_ptr);
  if (!load_status.ok()) {
    LOG(ERROR) << "Failed to load saved model v1 '" << saved_model_dir
               << "': " << load_status;
    return load_status;
  }

  auto module_or =
      ConvertSavedModelV1ToMlir(*bundle_ptr, exported_names, context, options);
  if (!module_or.status().ok()) {
    LOG(ERROR) << "SavedModel V1 import failed: " << module_or.status();
  }
  return module_or;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelSignatureDefsToMlirImportLite(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context,
    MLIRImportOptions options) {
  MetaGraphDef meta_graph_def;
  auto status =
      ReadMetaGraphDefFromSavedModel(saved_model_dir, tags, &meta_graph_def);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to load saved model v1 '" << saved_model_dir
               << "': " << status;
    return status;
  }

  std::optional<absl::Span<const std::string>> optional_exported_names;
  if (!exported_names.empty()) optional_exported_names = exported_names;

  // TODO(b/186898924): debug info in the savedmodel should not be ignored and
  // should be passed here.
  auto module_or =
      ConvertSavedModelV1ToMlirLite(meta_graph_def, /*debug_info=*/{},
                                    optional_exported_names, context, options);
  if (!module_or.status().ok()) {
    LOG(ERROR) << "SavedModel V1 import failed: " << module_or.status();
  }
  return module_or;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
GraphdefToSplattedMlirTranslateFunction(
    llvm::StringRef input, const std::vector<std::string>& input_arrays,
    const std::vector<std::string>& input_dtypes,
    const std::vector<std::optional<std::vector<int>>>& input_shapes,
    const std::vector<std::string>& output_arrays,
    const std::vector<std::string>& control_output_arrays,
    const GraphdefToMlirOptions& import_options, mlir::MLIRContext* context) {
  auto module_or = GraphdefToMlirImport(
      input, input_arrays, input_dtypes, input_shapes, output_arrays,
      control_output_arrays, import_options, context);
  if (!module_or.status().ok()) {
    LOG(ERROR) << "Graph import failed: " << module_or.status();
    return module_or.status();
  }
  auto& module = module_or.value();
  std::srand(0);
  for (auto fn : module->getOps<mlir::func::FuncOp>()) {
    for (auto& bb : fn) {
      for (auto& inst : bb) {
        auto attr_id = mlir::StringAttr::get(context, "value");
        if (auto attr = inst.getAttrOfType<mlir::ElementsAttr>(attr_id)) {
          mlir::Attribute rand_val;
          mlir::Type element_type = attr.getShapedType().getElementType();
          if (mlir::isa<mlir::IntegerType>(element_type)) {
            rand_val = mlir::IntegerAttr::get(element_type, std::rand());
          } else if (element_type.isF16() || element_type.isF32() ||
                     element_type.isF64()) {
            rand_val = mlir::FloatAttr::get(element_type,
                                            std::rand() * 1.0 / RAND_MAX);

          } else {
            inst.emitWarning()
                << "Skipping splat conversion for "
                << "an unsupported attribute type " << element_type;
            continue;
          }
          auto new_attr = mlir::DenseElementsAttr::get(
              llvm::cast<mlir::ShapedType>(attr.getType()), rand_val);
          inst.setAttr(attr_id, new_attr);
        }
      }
    }
  }
  return module_or;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
GraphdefToSplattedMlirTranslateFunction(
    llvm::StringRef input, absl::string_view input_arrays,
    absl::string_view input_dtypes, absl::string_view input_shapes,
    absl::string_view output_arrays, absl::string_view control_output_arrays,
    const GraphdefToMlirOptions& import_options, mlir::MLIRContext* context) {
  std::vector<std::string> input_array_vector;
  std::vector<std::string> input_dtype_vector;
  std::vector<std::optional<std::vector<int>>> input_shapes_vector;
  std::vector<std::string> output_array_vector;
  std::vector<std::string> control_output_array_vector;
  TF_RETURN_IF_ERROR(ParseNodeNames(input_arrays, input_array_vector));
  TF_RETURN_IF_ERROR(ParseNodeDataTypes(input_dtypes, input_dtype_vector));
  TF_RETURN_IF_ERROR(ParseNodeNames(output_arrays, output_array_vector));
  TF_RETURN_IF_ERROR(ParseNodeShapes(input_shapes, input_shapes_vector));
  TF_RETURN_IF_ERROR(
      ParseNodeNames(control_output_arrays, control_output_array_vector));
  return GraphdefToSplattedMlirTranslateFunction(
      input, input_array_vector, input_dtype_vector, input_shapes_vector,
      output_array_vector, control_output_array_vector, import_options,
      context);
}
}  // namespace tensorflow

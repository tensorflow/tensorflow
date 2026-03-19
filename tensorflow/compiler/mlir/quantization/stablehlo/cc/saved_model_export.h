/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
// Functionalities for exporting MLIR ModuleOp to TensorFlow SavedModel.

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_SAVED_MODEL_EXPORT_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_SAVED_MODEL_EXPORT_H_

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/types.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"

namespace mlir::quant::stablehlo {

// Suffix string for the module export step. Used for debugging.
constexpr absl::string_view kExportStepSuffix = "_export";

// Options when running passes for exporting an MLIR ModuleOp.
struct ExportOptions {
  // If set to `true`, it runs `DuplicateShapeDeterminingConstantsPass` before
  // lowering to tf_executor dialect.
  bool duplicate_shape_determining_constants = true;

  // If set to `true`, unfreezes constants into variables and saves them to a
  // checkpoint file. Setting this to `true` is an experimental feature that has
  // no stability guarantees.
  bool unfreeze_constants = false;

  // Path to the directory where checkpoint files are saved.
  std::string checkpoint_dir = "";

  // Name used to identify the ModuleOp this is exporting. Only used for
  // debugging and does not modify the behavior of the export.
  std::string debug_name = "stablehlo_quant";
};

// Creates `ExportedModel` from `module_op`. `module_op` goes through post
// process passes before an `ExportModel` is created.
// TODO: b/329206105 - Add unit tests after decomposing post processing passes.
absl::StatusOr<tensorflow::quantization::ExportedModel> CreateExportedModel(
    const std::vector<std::string>& signature_keys,
    const std::unordered_set<std::string>& tags,
    const ::stablehlo::quantization::QuantizationConfig& quantization_config,
    absl::string_view debug_name_prefix,
    const absl::flat_hash_map<FunctionName, FunctionAlias>& function_aliases,
    MLIRContext& ctx ABSL_ATTRIBUTE_LIFETIME_BOUND, ModuleOp module_op);

// Factory function for `ExportedModel`.
[[nodiscard]] tensorflow::quantization::ExportedModel
CreateExportedModelFromGraphDef(
    tensorflow::GraphDef&& graph_def, absl::string_view init_node_name,
    absl::string_view checkpoint_dir,
    std::optional<tensorflow::SaverDef> saver_def,
    const absl::flat_hash_map<std::string, std::string>& function_aliases,
    const std::vector<tensorflow::AssetFileDef>& asset_file_defs);

// Creates a new `SaverDef` instance, which contains information regarding
// checkpoint saving and restoring. This function returns a `SaverDef` instance
// with four fields populated: `version`, `filename_tensor_name`,
// `restore_op_name` and `save_tensor_name`. For valid quantized `graph_def` and
// `control_ret_node_names`, it should be able to retrieve the last three fields
// if there is at lest one variable in the graph.
//
// Returns a `std::nullopt` if there are no variables in the graph and no saving
// & restoring are required. Returns an `InternalError` status for when the
// required fields are only partially provided.
absl::StatusOr<std::optional<tensorflow::SaverDef>> CreateSaverDef(
    const std::vector<std::string>& control_ret_node_names,
    const tensorflow::GraphDef& graph_def);

// Adds passes for transforming the MLIR module op so that it can be exported
// back to GraphDef. Roughly, this consists of:
//   1) Inserting the @main function, which will become the main Graph.
//   2) Duplicating shape-determining constants.
//   3) Converting TF dialect -> tf_executor dialect.
//   4) Adding initializer function's ops into @main function for correct
//      resource initialization when loading the exported model.
//
// Duplicating shape-determining constants is required to place constants that
// affect the shape of a tensor to be placed in the TPU graph instead of in the
// CPU graph, when the graph gets converted for TPU inference. This allows these
// constants to be known at XLA compilation time.
void AddExportPasses(mlir::PassManager& pm,
                     bool duplicate_shape_determining_constants);

// Converts MLIR ModuleOp to `ExportedModel`. Returns `InternalError` status
// when the conversion fails.
//
// * `checkpoint_dir` is the directory where checkpoints where variable values
// are stored. This value will be fed to the "file_prefix" tensor to restore the
// variables.
// * `function_aliases` maps the actual function name to the function alias.
// This associates the quantized functions to the original functions' aliases.
// If there were no function aliases in the input model, this should be empty.
// * `asset_file_defs` include information about the assets, if any, that are
// used directly to initialize resources (like hash tables). If no assets are
// used in the model, this should be empty.
absl::StatusOr<tensorflow::quantization::ExportedModel>
ConvertMlirModuleToExportedModel(
    mlir::ModuleOp module_op, absl::string_view checkpoint_dir,
    const absl::flat_hash_map<std::string, std::string>& function_aliases,
    const std::vector<tensorflow::AssetFileDef>& asset_file_defs);

// Sets up and runs the passes for exporting `module_op`. The behavior of the
// exporting passes is controlled by `export_opts`. Returns `AssetFileDef`s that
// associate the input arguments of @main and the asset file names. Asset file
// names will be used to feed the corresponding tensors during initialization
// upon model loading.
// TODO: b/329206105 - Add unit tests after decomposing post processing passes.
absl::StatusOr<SmallVector<::tensorflow::AssetFileDef>> RunExportPasses(
    const ExportOptions& export_opts, MLIRContext& ctx, ModuleOp module_op);

}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_SAVED_MODEL_EXPORT_H_

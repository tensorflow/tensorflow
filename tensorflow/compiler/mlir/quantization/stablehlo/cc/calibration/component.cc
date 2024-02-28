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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/component.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/representative_dataset.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/statistics.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/saved_model_export.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/saved_model_import.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/types.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/convert_asset_args.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/unfreeze_constants.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_preprocess.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace mlir::quant::stablehlo {
namespace {

using ::stablehlo::quantization::AddCalibrationStatistics;
using ::stablehlo::quantization::CreateRepresentativeDatasetFileMap;
using ::stablehlo::quantization::QuantizationConfig;
using ::stablehlo::quantization::RepresentativeDatasetConfig;
using ::stablehlo::quantization::io::CreateTmpDir;
using ::stablehlo::quantization::io::GetLocalTmpFileName;
using ::tensorflow::AssetFileDef;
using ::tensorflow::MLIRImportOptions;
using ::tensorflow::SavedModelBundle;
using ::tensorflow::SavedModelSignatureDefsToMlirImport;
using ::tensorflow::SignatureDef;
using ::tensorflow::quantization::CalibrationOptions;
using ::tensorflow::quantization::ExportedModel;
using ::tensorflow::quantization::PreprocessAndFreezeGraph;
using ::tensorflow::quantization::PyFunctionLibrary;
using ::tensorflow::quantization::RunPasses;
using ::tensorflow::quantization::UnfreezeConstantsAndSaveVariables;

using ImportedMlirModuleOp =
    std::pair<ModuleOp, std::unique_ptr<SavedModelBundle>>;

// Loads a SavedModel at `saved_model_path` and converts it to `mlir::ModuleOp`.
//
// `tags` identify the `tensorflow::MetaGraphDef` to load from the SavedModel.
// Similarly, `signature_keys` identify the functions (`SignatureDef`s) to load
// within the `MetaGraphDef`. `ctx` is the `MLIRContext`, which should outlive
// the returned `ModuleOp`, thus marked with the lifetime bound attribute.
absl::StatusOr<ImportedMlirModuleOp> SavedModelToMlirModuleOp(
    const absl::string_view saved_model_path,
    const std::unordered_set<std::string>& tags,
    const std::vector<std::string>& signature_keys,
    MLIRContext& ctx ABSL_ATTRIBUTE_LIFETIME_BOUND) {
  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  import_options.lift_variables = false;
  import_options.include_variables_in_initializers = true;

  auto bundle = std::make_unique<SavedModelBundle>();

  // Copy to eliminate the `const` qualifier so that `absl::MakeSpan` can be
  // called on it.
  std::vector<std::string> exported_names = signature_keys;
  absl::StatusOr<OwningOpRef<ModuleOp>> module_op =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tags,
                                          absl::MakeSpan(exported_names), &ctx,
                                          import_options, &bundle);
  if (!module_op.status().ok()) {
    return absl::InternalError(absl::StrCat("Failed to import SavedModel: ",
                                            module_op.status().ToString()));
  }

  return std::make_pair(module_op->release(), std::move(bundle));
}

// Sets up and runs the passes for exporting `module_op`. The behavior of the
// exporting passes is controlled by `export_opts`. Returns `AssetFileDef`s that
// associate the input arguments of @main and the asset file names. Asset file
// names will be used to feed the corresponding tensors during initialization
// upon model loading.
absl::StatusOr<SmallVector<AssetFileDef>> RunExportPasses(
    const ExportOptions& export_opts, MLIRContext& ctx, ModuleOp module_op) {
  if (export_opts.unfreeze_constants) {
    TF_RETURN_IF_ERROR(UnfreezeConstantsAndSaveVariables(
        export_opts.checkpoint_dir, ctx, module_op));
    LOG(INFO) << "Unfrozen constants and saved variables to checkpoint file: "
              << export_opts.checkpoint_dir;
  }

  if (absl::Status pass_run_status = RunPasses(
          /*name=*/
          export_opts.debug_name,
          /*add_passes_func=*/
          [dup_constants = export_opts.duplicate_shape_determining_constants](
              PassManager& pm) { AddExportPasses(pm, dup_constants); },
          ctx, module_op);
      !pass_run_status.ok()) {
    return pass_run_status;
  }

  FailureOr<SmallVector<AssetFileDef>> asset_file_defs =
      quant::ConvertAssetArgs(module_op);
  if (failed(asset_file_defs)) {
    return absl::InternalError("Failed to convert asset args.");
  }

  return *asset_file_defs;
}

}  // namespace

CalibrationComponent::CalibrationComponent(
    absl::Nonnull<MLIRContext*> ctx,
    absl::Nonnull<const PyFunctionLibrary*> py_function_lib,
    const absl::string_view src_saved_model_path,
    absl::flat_hash_map<FunctionName, FunctionAlias> function_aliases,
    std::unordered_set<std::string> tags,
    absl::flat_hash_map<std::string, SignatureDef> signature_def_map,
    std::vector<std::string> signature_keys,
    const CalibrationOptions& calibration_options)
    : ctx_(ABSL_DIE_IF_NULL(ctx)),                          // Crash OK
      py_function_lib_(ABSL_DIE_IF_NULL(py_function_lib)),  // Crash OK
      src_saved_model_path_(src_saved_model_path),
      function_aliases_(std::move(function_aliases)),
      tags_(std::move(tags)),
      signature_def_map_(std::move(signature_def_map)),
      signature_keys_(std::move(signature_keys)),
      calibration_options_(calibration_options) {}

absl::StatusOr<ExportedModel> CalibrationComponent::ExportToSavedModel(
    ModuleOp module_op, const absl::string_view dst_saved_model_path) {
  TF_ASSIGN_OR_RETURN(const std::string checkpoint_dir, GetLocalTmpFileName());

  // `duplicate_shape_determining_constants = false` because the
  // resulting graph of this step is not expected to be loaded on TPU.
  const ExportOptions export_opts = {
      /*duplicate_shape_determining_constants=*/false,
      /*unfreeze_constants=*/false, checkpoint_dir,
      /*debug_name=*/absl::StrCat(kName, kExportStepSuffix)};

  TF_ASSIGN_OR_RETURN(const SmallVector<AssetFileDef> asset_file_defs,
                      RunExportPasses(export_opts, *ctx_, module_op));

  const absl::flat_hash_map<FunctionName, FunctionAlias>
      updated_function_aliases =
          UpdateFunctionAliases(function_aliases_, module_op);

  TF_ASSIGN_OR_RETURN(ExportedModel exported_model,
                      ConvertMlirModuleToExportedModel(
                          module_op, checkpoint_dir, updated_function_aliases,
                          {asset_file_defs.begin(), asset_file_defs.end()}));

  py_function_lib_->SaveExportedModel(dst_saved_model_path, exported_model,
                                      src_saved_model_path_, tags_,
                                      signature_def_map_);

  return exported_model;
}

absl::StatusOr<ModuleOp> CalibrationComponent::ImportCalibratedSavedModel(
    const absl::string_view calibrated_saved_model_path) {
  // Convert the SavedModelBundle to an MLIR module.
  TF_ASSIGN_OR_RETURN(ImportedMlirModuleOp imported_module,
                      SavedModelToMlirModuleOp(calibrated_saved_model_path,
                                               tags_, signature_keys_, *ctx_));
  ModuleOp module_op = imported_module.first;

  const absl::flat_hash_map<FunctionName, FunctionAlias>
      updated_function_aliases_post_calibration =
          UpdateFunctionAliases(function_aliases_, module_op);

  // Collect the names of the functions that have aliases so that they may not
  // be inlined.
  absl::flat_hash_set<std::string> aliased_function_names;
  absl::c_for_each(updated_function_aliases_post_calibration,
                   [&](const auto& aliases) {
                     return aliased_function_names.insert(aliases.first);
                   });

  // Freezing is required again since variables might have been produced
  // during the pre-calibration step. `is_inliner_run = false` to prevent the
  // functions lifted for quantization from being inlined.
  TF_RETURN_IF_ERROR(PreprocessAndFreezeGraph(
      /*mlir_dump_file_prefix=*/kName, /*is_inliner_run=*/false,
      /*noinline_functions=*/aliased_function_names, module_op, ctx_,
      imported_module.second == nullptr ? nullptr
                                        : imported_module.second->GetSession(),
      /*run_tf_to_stablehlo=*/false, /*deserialize_xla_call_module=*/true));
  return module_op;
}

absl::StatusOr<ModuleOp> CalibrationComponent::Run(
    ModuleOp module_op, const QuantizationConfig& config) {
  // Exports the pre-calibrated model to SavedModel.
  TF_ASSIGN_OR_RETURN(const std::string precalibrated_saved_model_dir,
                      CreateTmpDir());

  TF_ASSIGN_OR_RETURN(
      ExportedModel exported_model,
      ExportToSavedModel(module_op, precalibrated_saved_model_dir));

  // Translates `RepresentativeDatasetConfig`s to signature key ->
  // `RepresentativeDatasetFile` mapping.
  const auto dataset_configs =
      config.static_range_ptq_preset().representative_datasets();
  const std::vector<RepresentativeDatasetConfig> dataset_config_vector(
      dataset_configs.begin(), dataset_configs.end());
  TF_ASSIGN_OR_RETURN(
      const auto representative_dataset_file_map,
      CreateRepresentativeDatasetFileMap(dataset_config_vector));

  // Runs calibration on the exported model. The statistics will be stored in a
  // separate singleton object `CalibratorSingleton` and are directly added to
  // `exported_model` without re-importing it.
  py_function_lib_->RunCalibration(precalibrated_saved_model_dir,
                                   signature_keys_, tags_, calibration_options_,
                                   /*force_graph_mode_calibration=*/true,
                                   representative_dataset_file_map);

  if (absl::Status status =
          AddCalibrationStatistics(*exported_model.mutable_graph_def(),
                                   calibration_options_, *py_function_lib_);
      !status.ok()) {
    LOG(WARNING) << "Some CustomAggregator ops do not have min or max "
                    "values. Parts of the graph are not quantized. "
                 << status;
  }

  // Exports the calibrated model with statistics attached to the graph.
  TF_ASSIGN_OR_RETURN(const std::string calibrated_saved_model_path,
                      CreateTmpDir());
  py_function_lib_->SaveExportedModel(calibrated_saved_model_path,
                                      exported_model, src_saved_model_path_,
                                      tags_, signature_def_map_);

  // Imports the calibrated saved model back to `ModuleOp`.
  return ImportCalibratedSavedModel(calibrated_saved_model_path);
}

}  // namespace mlir::quant::stablehlo

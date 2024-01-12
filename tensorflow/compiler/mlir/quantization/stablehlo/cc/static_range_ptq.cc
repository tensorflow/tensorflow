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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/static_range_ptq.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/assign_ids.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/statistics.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/context.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/export.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/post_calibration.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/pre_calibration.h"
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
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace mlir::quant::stablehlo {
namespace {

using ::stablehlo::quantization::AddCalibrationStatistics;
using ::stablehlo::quantization::AddExportPasses;
using ::stablehlo::quantization::AssignIdsToCustomAggregatorOps;
using ::stablehlo::quantization::ConvertMlirModuleToExportedModel;
using ::stablehlo::quantization::ExportOptions;
using ::stablehlo::quantization::kExportStepSuffix;
using ::stablehlo::quantization::QuantizationConfig;
using ::stablehlo::quantization::io::CreateTmpDir;
using ::stablehlo::quantization::io::GetLocalTmpFileName;
using ::tensorflow::AssetFileDef;
using ::tensorflow::MLIRImportOptions;
using ::tensorflow::SavedModelBundle;
using ::tensorflow::SignatureDef;
using ::tensorflow::quantization::CalibrationOptions;
using ::tensorflow::quantization::ExportedModel;
using ::tensorflow::quantization::PreprocessAndFreezeGraph;
using ::tensorflow::quantization::PyFunctionLibrary;
using ::tensorflow::quantization::RepresentativeDatasetFile;
using ::tensorflow::quantization::RunPasses;
using ::tensorflow::quantization::UnfreezeConstantsAndSaveVariables;

// Create default configuration for the calibration step, which is the min/max
// calibration method.
CalibrationOptions GetDefaultCalibrationOptions() {
  CalibrationOptions options{};
  options.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_MIN_MAX);
  return options;
}

// Returns the updated function aliases. `module_op` may have different
// function names from the original model, so it re-associates the aliases
// with the new function names. Both the input `function_aliases` and the
// returned value are function name -> alias mappings. `function_aliases` is
// the function alias mapping of the original function.
absl::flat_hash_map<std::string, std::string> UpdateFunctionAliases(
    const absl::flat_hash_map<std::string, std::string> function_aliases,
    ModuleOp module_op) {
  absl::flat_hash_map<std::string, std::string> updated_function_aliases;

  module_op->walk([&](func::FuncOp func_op) {
    // We may retrieve the original function's name from the attribute.
    // Functions without this attribute are ignored.
    auto original_func_name =
        func_op->getAttrOfType<StringAttr>("tf._original_func_name");
    if (original_func_name) {
      if (auto alias_itr = function_aliases.find(original_func_name.str());
          alias_itr != function_aliases.end()) {
        const std::string alias = alias_itr->second;
        const std::string new_func_name = func_op.getSymName().str();

        updated_function_aliases[new_func_name] = alias;

        VLOG(1) << "Updated function alias. Alias: " << alias
                << ", New function name: " << new_func_name
                << ", Old function name: " << original_func_name.str();
      }
    }
  });

  return updated_function_aliases;
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
          /*name=*/export_opts.debug_name,
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

absl::StatusOr<ExportedModel> QuantizePtqModelPreCalibration(
    const absl::string_view saved_model_path,
    const std::vector<std::string>& signature_keys,
    const std::unordered_set<std::string>& tags,
    const QuantizationConfig& quantization_config,
    const absl::flat_hash_map<std::string, std::string>& function_aliases) {
  // Convert the SavedModelBundle to an MLIR module.
  std::unique_ptr<MLIRContext> context = CreateMlirContextForQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  import_options.lift_variables = false;
  import_options.include_variables_in_initializers = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  std::vector<std::string> exported_names = signature_keys;
  absl::StatusOr<OwningOpRef<ModuleOp>> module =
      tensorflow::SavedModelSignatureDefsToMlirImport(
          saved_model_path, tags, absl::MakeSpan(exported_names), context.get(),
          import_options, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError(absl::StrCat("Failed to import SavedModel: ",
                                            module.status().message()));
  }
  OwningOpRef<ModuleOp> module_ref = std::move(module).value();

  const absl::flat_hash_map<std::string, std::string> updated_function_aliases =
      UpdateFunctionAliases(function_aliases, *module_ref);

  // Collect the names of the functions that have aliases so that they may not
  // be inlined.
  absl::flat_hash_set<std::string> aliased_function_names;
  absl::c_for_each(updated_function_aliases, [&](const auto& aliases) {
    return aliased_function_names.insert(aliases.first);
  });

  TF_RETURN_IF_ERROR(PreprocessAndFreezeGraph(
      /*mlir_dump_file_prefix=*/PreCalibrationComponent::kName,
      /*is_inliner_run=*/true, /*noinline_functions=*/aliased_function_names,
      module_ref.get(), context.get(), bundle ? bundle->GetSession() : nullptr,
      /*run_tf_to_stablehlo=*/true, /*deserialize_xla_call_module=*/false));

  // Use StableHLO Quantizer option if opset is specified.
  PreCalibrationComponent pre_calibration_component(
      context.get(), GetDefaultCalibrationOptions());
  TF_ASSIGN_OR_RETURN(*module_ref, pre_calibration_component.Run(
                                       *module_ref, QuantizationConfig()));

  TF_ASSIGN_OR_RETURN(const std::string checkpoint_dir, GetLocalTmpFileName());

  // `duplicate_shape_determining_constants = false` because the
  // resulting graph of this step is not expected to be loaded on TPU.
  const ExportOptions export_opts = {
      /*duplicate_shape_determining_constants=*/false,
      /*unfreeze_constants=*/false, checkpoint_dir,
      /*debug_name=*/
      absl::StrCat(PreCalibrationComponent::kName, kExportStepSuffix)};

  TF_ASSIGN_OR_RETURN(const SmallVector<AssetFileDef> asset_file_defs,
                      RunExportPasses(export_opts, *context, *module_ref));

  return ConvertMlirModuleToExportedModel(
      *module_ref, checkpoint_dir, updated_function_aliases,
      {asset_file_defs.begin(), asset_file_defs.end()});
}

absl::StatusOr<ExportedModel> QuantizePtqModelPostCalibration(
    const absl::string_view saved_model_path,
    const std::vector<std::string>& signature_keys,
    const std::unordered_set<std::string>& tags,
    const QuantizationConfig& quantization_config,
    const absl::flat_hash_map<std::string, std::string>& function_aliases) {
  // Convert the SavedModelBundle to an MLIR module.
  std::unique_ptr<MLIRContext> context = CreateMlirContextForQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  import_options.lift_variables = false;
  import_options.include_variables_in_initializers = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  std::vector<std::string> exported_names = signature_keys;
  absl::StatusOr<OwningOpRef<ModuleOp>> module =
      tensorflow::SavedModelSignatureDefsToMlirImport(
          saved_model_path, tags, absl::MakeSpan(exported_names), context.get(),
          import_options, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError(absl::StrCat("Failed to import SavedModel: ",
                                            module.status().message()));
  }

  OwningOpRef<ModuleOp> module_ref = std::move(module).value();

  const absl::flat_hash_map<std::string, std::string> updated_function_aliases =
      UpdateFunctionAliases(function_aliases, *module_ref);

  // Collect the names of the functions that have aliases so that they may not
  // be inlined.
  absl::flat_hash_set<std::string> aliased_function_names;
  absl::c_for_each(updated_function_aliases, [&](const auto& aliases) {
    return aliased_function_names.insert(aliases.first);
  });

  // Freezing is required again since variables might have been produced
  // during the pre-calibration step. `is_inliner_run = false` to prevent the
  // functions lifted for quantization from being inlined.
  TF_RETURN_IF_ERROR(PreprocessAndFreezeGraph(
      /*mlir_dump_file_prefix=*/PostCalibrationComponent::kName,
      /*is_inliner_run=*/false, /*noinline_functions=*/aliased_function_names,
      module_ref.get(), context.get(), bundle ? bundle->GetSession() : nullptr,
      /*run_tf_to_stablehlo=*/false, /*deserialize_xla_call_module=*/true));

  PostCalibrationComponent post_calibration_component(context.get());
  TF_ASSIGN_OR_RETURN(*module_ref, post_calibration_component.Run(
                                       *module_ref, quantization_config));

  TF_ASSIGN_OR_RETURN(const std::string checkpoint_dir, GetLocalTmpFileName());
  const ExportOptions export_opts = {
      /*duplicate_shape_determining_constants=*/true,
      /*unfreeze_constants=*/false, checkpoint_dir,
      /*debug_name=*/
      absl::StrCat(PostCalibrationComponent::kName, kExportStepSuffix)};

  TF_ASSIGN_OR_RETURN(const SmallVector<AssetFileDef> asset_file_defs,
                      RunExportPasses(export_opts, *context, *module_ref));

  return ConvertMlirModuleToExportedModel(
      *module_ref, checkpoint_dir, updated_function_aliases,
      {asset_file_defs.begin(), asset_file_defs.end()});
}

}  // namespace

// TODO: b/317167427 - Enable debugger.
absl::Status QuantizeStaticRangePtq(
    const absl::string_view src_saved_model_path,
    const absl::string_view dst_saved_model_path,
    const QuantizationConfig& quantization_config,
    const std::vector<std::string>& signature_keys,
    const absl::flat_hash_map<std::string, SignatureDef>& signature_def_map,
    const absl::flat_hash_map<std::string, std::string>& function_aliases,
    const PyFunctionLibrary& py_function_library,
    const absl::flat_hash_map<std::string, RepresentativeDatasetFile>&
        representative_dataset_file_map) {
  std::unordered_set<std::string> tags;
  tags.insert(quantization_config.tf_saved_model().tags().begin(),
              quantization_config.tf_saved_model().tags().end());

  TF_ASSIGN_OR_RETURN(
      ExportedModel exported_model,
      QuantizePtqModelPreCalibration(src_saved_model_path, signature_keys, tags,
                                     quantization_config, function_aliases));

  AssignIdsToCustomAggregatorOps(*exported_model.mutable_graph_def());

  const absl::StatusOr<std::string> precalibrated_saved_model_dir =
      CreateTmpDir();
  if (!precalibrated_saved_model_dir.ok()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to create tmp dir for precalibrated saved model: %s",
        precalibrated_saved_model_dir.status().ToString()));
  }

  py_function_library.SaveExportedModel(*precalibrated_saved_model_dir,
                                        exported_model, src_saved_model_path,
                                        tags, signature_def_map);

  py_function_library.RunCalibration(
      *precalibrated_saved_model_dir, signature_keys, tags,
      GetDefaultCalibrationOptions(),
      /*force_graph_mode_calibration=*/true, representative_dataset_file_map);

  if (absl::Status status = AddCalibrationStatistics(
          *exported_model.mutable_graph_def(), GetDefaultCalibrationOptions(),
          py_function_library);
      !status.ok()) {
    LOG(WARNING) << "Some CustomAggregator ops do not have min or max "
                    "values. Parts of the graph are not quantized. "
                 << status;
  }

  const absl::StatusOr<std::string> calibrated_saved_model_path =
      CreateTmpDir();
  if (!calibrated_saved_model_path.ok()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to create tmp dir for calibrated saved model: %s",
        calibrated_saved_model_path.status().ToString()));
  }

  py_function_library.SaveExportedModel(*calibrated_saved_model_path,
                                        exported_model, src_saved_model_path,
                                        tags, signature_def_map);

  const absl::flat_hash_map<std::string, std::string>
      function_aliases_after_calibration(
          exported_model.function_aliases().begin(),
          exported_model.function_aliases().end());

  TF_ASSIGN_OR_RETURN(
      const ExportedModel post_calibrated_exported_model,
      QuantizePtqModelPostCalibration(*calibrated_saved_model_path,
                                      signature_keys, tags, quantization_config,
                                      function_aliases_after_calibration));

  // Remove the `tpu` tag for exporting because the output quantized model is
  // essentially a CPU model.
  tags.erase("tpu");

  py_function_library.SaveExportedModel(
      dst_saved_model_path, post_calibrated_exported_model,
      *calibrated_saved_model_path, tags, signature_def_map);

  return absl::OkStatus();
}

}  // namespace mlir::quant::stablehlo

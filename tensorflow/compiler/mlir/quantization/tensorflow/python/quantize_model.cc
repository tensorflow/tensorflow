/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model.h"

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
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/context.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/post_calibration.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/pre_calibration.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/saved_model_export.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/saved_model_import.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/convert_asset_args.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/unfreeze_constants.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_preprocess.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace quantization {
namespace {

using ::mlir::quant::stablehlo::AddExportPasses;
using ::mlir::quant::stablehlo::ConvertMlirModuleToExportedModel;
using ::mlir::quant::stablehlo::CreateMlirContextForQuantization;
using ::mlir::quant::stablehlo::ExportOptions;
using ::mlir::quant::stablehlo::FunctionAlias;
using ::mlir::quant::stablehlo::FunctionName;
using ::mlir::quant::stablehlo::kExportStepSuffix;
using ::mlir::quant::stablehlo::PostCalibrationComponent;
using ::mlir::quant::stablehlo::PreCalibrationComponent;
using ::mlir::quant::stablehlo::UpdateFunctionAliases;
using ::stablehlo::quantization::QuantizationConfig;
using ::stablehlo::quantization::io::GetLocalTmpFileName;

// Sets up and runs the passes for exporting `module_op`. The behavior of the
// exporting passes is controlled by `export_opts`. Returns `AssetFileDef`s that
// associate the input arguments of @main and the asset file names. Asset file
// names will be used to feed the corresponding tensors during initialization
// upon model loading.
absl::StatusOr<llvm::SmallVector<AssetFileDef>> RunExportPasses(
    const ExportOptions &export_opts, mlir::MLIRContext &ctx,
    mlir::ModuleOp module_op) {
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
              mlir::PassManager &pm) { AddExportPasses(pm, dup_constants); },
          ctx, module_op);
      !pass_run_status.ok()) {
    return pass_run_status;
  }

  mlir::FailureOr<llvm::SmallVector<AssetFileDef>> asset_file_defs =
      mlir::quant::ConvertAssetArgs(module_op);
  if (failed(asset_file_defs)) {
    return absl::InternalError("Failed to convert asset args.");
  }

  return *asset_file_defs;
}

}  // namespace

absl::StatusOr<ExportedModel> QuantizeQatModel(
    const absl::string_view saved_model_path,
    const std::vector<std::string> &signature_keys,
    const std::unordered_set<std::string> &tags,
    const QuantizationOptions &quantization_options,
    const absl::flat_hash_map<std::string, std::string> &function_aliases) {
  // Convert the SavedModelBundle to an MLIR module.
  std::unique_ptr<mlir::MLIRContext> context =
      CreateMlirContextForQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  import_options.lift_variables = false;
  import_options.include_variables_in_initializers = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO: b/213406917 - Add support for the object graph based saved model
  // input
  std::vector<std::string> exported_names = signature_keys;
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(
          saved_model_path, tags, absl::MakeSpan(exported_names), context.get(),
          import_options, &bundle);
  if (!module.status().ok()) {
    return absl::InternalError(absl::StrCat("Failed to import SavedModel: ",
                                            module.status().message()));
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  const absl::flat_hash_map<FunctionName, FunctionAlias>
      updated_function_aliases =
          UpdateFunctionAliases(function_aliases, *module_ref);

  // Collect the names of the functions that have aliases so that they may not
  // be inlined.
  absl::flat_hash_set<std::string> aliased_function_names;
  absl::c_for_each(updated_function_aliases, [&](const auto &aliases) {
    return aliased_function_names.insert(aliases.first);
  });

  TF_RETURN_IF_ERROR(PreprocessAndFreezeGraph(
      /*mlir_dump_file_prefix=*/kDefaultTfQuantMlirDumpFilePrefix,
      /*is_inliner_run=*/true,
      /*noinline_functions=*/aliased_function_names, module_ref.get(),
      context.get(), bundle ? bundle->GetSession() : nullptr,
      /*run_tf_to_stablehlo=*/false,
      /*deserialize_xla_call_module=*/false));

  TF_RETURN_IF_ERROR(RunPasses(
      /*name=*/
      kTfQuantQatStepName, /*add_passes_func=*/
      [&quantization_options](mlir::PassManager &pm) {
        AddQuantizeQatPasses(pm, quantization_options, kTfQuantQatStepName);
      },
      *context, *module_ref));

  const bool unfreeze_constants = !quantization_options.freeze_all_variables();

  TF_ASSIGN_OR_RETURN(const std::string checkpoint_dir, GetLocalTmpFileName());

  const auto export_opts = ExportOptions{
      /*duplicate_shape_determining_constants=*/true, unfreeze_constants,
      checkpoint_dir,
      /*debug_name=*/absl::StrCat(kTfQuantQatStepName, kExportStepSuffix)};

  TF_ASSIGN_OR_RETURN(const llvm::SmallVector<AssetFileDef> asset_file_defs,
                      RunExportPasses(export_opts, *context, *module_ref));

  return ConvertMlirModuleToExportedModel(
      *module_ref, checkpoint_dir, updated_function_aliases,
      {asset_file_defs.begin(), asset_file_defs.end()});
}

absl::StatusOr<ExportedModel> QuantizePtqModelPreCalibration(
    const absl::string_view saved_model_path,
    const std::vector<std::string> &signature_keys,
    const std::unordered_set<std::string> &tags,
    const QuantizationOptions &quantization_options,
    const absl::flat_hash_map<std::string, std::string> &function_aliases) {
  // Convert the SavedModelBundle to an MLIR module.
  std::unique_ptr<mlir::MLIRContext> context =
      CreateMlirContextForQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  import_options.lift_variables = false;
  import_options.include_variables_in_initializers = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO: b/213406917 - Add support for the object graph based saved model
  // input
  std::vector<std::string> exported_names = signature_keys;
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(
          saved_model_path, tags, absl::MakeSpan(exported_names), context.get(),
          import_options, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError(absl::StrCat("Failed to import SavedModel: ",
                                            module.status().message()));
  }
  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  const absl::flat_hash_map<FunctionName, FunctionAlias>
      updated_function_aliases =
          UpdateFunctionAliases(function_aliases, *module_ref);

  // Collect the names of the functions that have aliases so that they may not
  // be inlined.
  absl::flat_hash_set<std::string> aliased_function_names;
  absl::c_for_each(updated_function_aliases, [&](const auto &aliases) {
    return aliased_function_names.insert(aliases.first);
  });

  const bool is_stablehlo = quantization_options.op_set() == OpSet::STABLEHLO;
  TF_RETURN_IF_ERROR(PreprocessAndFreezeGraph(
      /*mlir_dump_file_prefix=*/kTfQuantPtqPreCalibrationStepName,
      /*is_inliner_run=*/true, /*noinline_functions=*/aliased_function_names,
      module_ref.get(), context.get(), bundle ? bundle->GetSession() : nullptr,
      /*run_tf_to_stablehlo=*/is_stablehlo,
      /*deserialize_xla_call_module=*/false));

  // Use StableHLO Quantizer option if opset is specified.
  if (is_stablehlo) {
    PreCalibrationComponent pre_calibration_component(
        context.get(), quantization_options.calibration_options());
    TF_ASSIGN_OR_RETURN(*module_ref, pre_calibration_component.Run(
                                         *module_ref, QuantizationConfig()));
  } else {
    TF_RETURN_IF_ERROR(RunPasses(
        /*name=*/
        kTfQuantPtqPreCalibrationStepName, /*add_passes_func=*/
        [&quantization_options](mlir::PassManager &pm) {
          AddQuantizePtqPreCalibrationPasses(pm, quantization_options);
        },
        *context, *module_ref));
  }

  const bool unfreeze_constants = !quantization_options.freeze_all_variables();
  TF_ASSIGN_OR_RETURN(const std::string checkpoint_dir, GetLocalTmpFileName());

  // `duplicate_shape_determining_constants = false` because the
  // resulting graph of this step is not expected to be loaded on TPU.
  const auto export_opts = ExportOptions{
      /*duplicate_shape_determining_constants=*/false, unfreeze_constants,
      checkpoint_dir,
      /*debug_name=*/
      absl::StrCat(kTfQuantPtqPreCalibrationStepName, kExportStepSuffix)};

  TF_ASSIGN_OR_RETURN(const llvm::SmallVector<AssetFileDef> asset_file_defs,
                      RunExportPasses(export_opts, *context, *module_ref));

  return ConvertMlirModuleToExportedModel(
      *module_ref, checkpoint_dir, updated_function_aliases,
      {asset_file_defs.begin(), asset_file_defs.end()});
}

absl::StatusOr<ExportedModel> QuantizePtqModelPostCalibration(
    const absl::string_view saved_model_path,
    const std::vector<std::string> &signature_keys,
    const std::unordered_set<std::string> &tags,
    const QuantizationOptions &quantization_options,
    const absl::flat_hash_map<std::string, std::string> &function_aliases) {
  // Convert the SavedModelBundle to an MLIR module.
  std::unique_ptr<mlir::MLIRContext> context =
      CreateMlirContextForQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  import_options.lift_variables = false;
  import_options.include_variables_in_initializers = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO: b/213406917 - Add support for the object graph based saved model
  // input
  std::vector<std::string> exported_names = signature_keys;
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(
          saved_model_path, tags, absl::MakeSpan(exported_names), context.get(),
          import_options, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError(absl::StrCat("Failed to import SavedModel: ",
                                            module.status().message()));
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  const absl::flat_hash_map<FunctionName, FunctionAlias>
      updated_function_aliases =
          UpdateFunctionAliases(function_aliases, *module_ref);

  // Collect the names of the functions that have aliases so that they may not
  // be inlined.
  absl::flat_hash_set<std::string> aliased_function_names;
  absl::c_for_each(updated_function_aliases, [&](const auto &aliases) {
    return aliased_function_names.insert(aliases.first);
  });

  const bool is_stablehlo = quantization_options.op_set() == OpSet::STABLEHLO;

  // Freezing is required again since variables might have been produced during
  // the pre-calibration step. `is_inliner_run = false` to prevent the functions
  // lifted for quantization from being inlined.
  TF_RETURN_IF_ERROR(PreprocessAndFreezeGraph(
      /*mlir_dump_file_prefix=*/kTfQuantPtqPostCalibrationStepName,
      /*is_inliner_run=*/false, /*noinline_functions=*/aliased_function_names,
      module_ref.get(), context.get(), bundle ? bundle->GetSession() : nullptr,
      /*run_tf_to_stablehlo=*/false,
      /*deserialize_xla_call_module=*/is_stablehlo));

  // Use StableHLO Quantizer option if opset is specified.
  if (is_stablehlo) {
    QuantizationConfig quantization_config{};
    quantization_config.mutable_static_range_ptq_preset()
        ->set_enable_per_channel_quantized_weight(
            quantization_options.enable_per_channel_quantization());
    // When targeting server TPUs quantized types should be unpacked into
    // integer ops.
    quantization_config.mutable_pipeline_config()->set_unpack_quantized_types(
        true);

    PostCalibrationComponent post_calibration_component(context.get());
    TF_ASSIGN_OR_RETURN(*module_ref, post_calibration_component.Run(
                                         *module_ref, quantization_config));
  } else {
    TF_RETURN_IF_ERROR(RunPasses(
        /*name=*/
        kTfQuantPtqPostCalibrationStepName, /*add_passes_func=*/
        [&quantization_options](mlir::PassManager &pm) {
          AddQuantizePtqPostCalibrationPasses(
              pm, quantization_options, kTfQuantPtqPostCalibrationStepName);
        },
        *context, *module_ref));
  }

  const bool unfreeze_constants = !quantization_options.freeze_all_variables();
  TF_ASSIGN_OR_RETURN(const std::string checkpoint_dir, GetLocalTmpFileName());

  const auto export_opts = ExportOptions{
      /*duplicate_shape_determining_constants=*/true, unfreeze_constants,
      checkpoint_dir,
      /*debug_name=*/
      absl::StrCat(kTfQuantPtqPostCalibrationStepName, kExportStepSuffix)};

  TF_ASSIGN_OR_RETURN(const llvm::SmallVector<AssetFileDef> asset_file_defs,
                      RunExportPasses(export_opts, *context, *module_ref));

  return ConvertMlirModuleToExportedModel(
      *module_ref, checkpoint_dir, updated_function_aliases,
      {asset_file_defs.begin(), asset_file_defs.end()});
}

absl::StatusOr<ExportedModel> QuantizePtqDynamicRange(
    const absl::string_view saved_model_path,
    const std::vector<std::string> &signature_keys,
    const std::unordered_set<std::string> &tags,
    const QuantizationOptions &quantization_options,
    const absl::flat_hash_map<std::string, std::string> &function_aliases) {
  // Convert the SavedModelBundle to an MLIR module.
  std::unique_ptr<mlir::MLIRContext> context =
      CreateMlirContextForQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  import_options.lift_variables = false;
  import_options.include_variables_in_initializers = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO: b/213406917 - Add support for the object graph based saved model
  // input
  std::vector<std::string> exported_names = signature_keys;
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(
          saved_model_path, tags, absl::MakeSpan(exported_names), context.get(),
          import_options, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError(absl::StrCat("Failed to import SavedModel: ",
                                            module.status().message()));
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  const absl::flat_hash_map<FunctionName, FunctionAlias>
      updated_function_aliases =
          UpdateFunctionAliases(function_aliases, *module_ref);

  // Collect the names of the functions that have aliases so that they may not
  // be inlined. The mapping is mlir function name - user defined function
  // alias for each value in the set.
  absl::flat_hash_set<std::string> aliased_function_names;
  absl::c_for_each(updated_function_aliases, [&](const auto &aliases) {
    return aliased_function_names.insert(aliases.first);
  });

  TF_RETURN_IF_ERROR(PreprocessAndFreezeGraph(
      /*mlir_dump_file_prefix=*/kDefaultTfQuantMlirDumpFilePrefix,
      /*is_inliner_run=*/true, /*noinline_functions=*/aliased_function_names,
      module_ref.get(), context.get(), bundle ? bundle->GetSession() : nullptr,
      /*run_tf_to_stablehlo=*/false, /*deserialize_xla_call_module=*/false));

  TF_RETURN_IF_ERROR(RunPasses(
      /*name=*/
      kTfQuantPtqDynamicRangeStepName, /*add_passes_func=*/
      [&quantization_options](mlir::PassManager &pm) {
        AddQuantizePtqDynamicRangePasses(pm, quantization_options,
                                         kTfQuantPtqDynamicRangeStepName);
      },
      *context, *module_ref));

  const bool unfreeze_constants = !quantization_options.freeze_all_variables();
  TF_ASSIGN_OR_RETURN(const std::string checkpoint_dir, GetLocalTmpFileName());

  const auto export_opts = ExportOptions{
      /*duplicate_shape_determining_constants=*/true, unfreeze_constants,
      checkpoint_dir,
      /*debug_name=*/
      absl::StrCat(kTfQuantPtqDynamicRangeStepName, kExportStepSuffix)};
  TF_ASSIGN_OR_RETURN(const llvm::SmallVector<AssetFileDef> asset_file_defs,
                      RunExportPasses(export_opts, *context, *module_ref));

  return ConvertMlirModuleToExportedModel(
      *module_ref, checkpoint_dir, updated_function_aliases,
      {asset_file_defs.begin(), asset_file_defs.end()});
}

// TODO: b/297626257 - [Converter Component][TF-Quantizer] Clean up
// quantize_model.cc by factoring out repeated codes
absl::StatusOr<ExportedModel> QuantizeWeightOnly(
    const absl::string_view saved_model_path,
    const QuantizationOptions &quantization_options,
    const absl::flat_hash_map<std::string, std::string> &function_aliases) {
  // Convert the SavedModelBundle to an MLIR module.
  std::unique_ptr<mlir::MLIRContext> context =
      CreateMlirContextForQuantization();

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  import_options.lift_variables = false;
  import_options.include_variables_in_initializers = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO: b/213406917 - Add support for the object graph based saved model
  // input
  std::vector<std::string> exported_names{
      quantization_options.signature_keys().begin(),
      quantization_options.signature_keys().end()};
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(saved_model_path,
                                          {quantization_options.tags().begin(),
                                           quantization_options.tags().end()},
                                          absl::MakeSpan(exported_names),
                                          context.get(), import_options,
                                          &bundle);

  if (!module.status().ok()) {
    return absl::InternalError(absl::StrCat("Failed to import SavedModel: ",
                                            module.status().message()));
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  const absl::flat_hash_map<FunctionName, FunctionAlias>
      updated_function_aliases =
          UpdateFunctionAliases(function_aliases, *module_ref);

  // Collect the names of the functions that have aliases so that they may not
  // be inlined. The mapping is mlir function name - user defined function
  // alias for each value in the set.
  absl::flat_hash_set<std::string> aliased_function_names;
  absl::c_for_each(updated_function_aliases, [&](const auto &aliases) {
    return aliased_function_names.insert(aliases.first);
  });

  TF_RETURN_IF_ERROR(PreprocessAndFreezeGraph(
      /*mlir_dump_file_prefix=*/kDefaultTfQuantMlirDumpFilePrefix,
      /*is_inliner_run=*/true,
      /*noinline_functions=*/aliased_function_names, module_ref.get(),
      context.get(), bundle ? bundle->GetSession() : nullptr,
      /*run_tf_to_stablehlo=*/false,
      /*deserialize_xla_call_module=*/false));

  TF_RETURN_IF_ERROR(RunPasses(
      kTfQuantWeightOnlyStepName,
      /*add_passes_func=*/
      [&quantization_options](mlir::PassManager &pm) {
        AddQuantizeWeightOnlyPasses(pm, quantization_options,
                                    kTfQuantWeightOnlyStepName);
      },
      *context, *module_ref));

  const bool unfreeze_constants = !quantization_options.freeze_all_variables();
  TF_ASSIGN_OR_RETURN(const std::string checkpoint_dir, GetLocalTmpFileName());

  const auto export_opts = ExportOptions{
      /*duplicate_shape_determining_constants=*/true, unfreeze_constants,
      checkpoint_dir,
      /*debug_name=*/
      absl::StrCat(kTfQuantWeightOnlyStepName, kExportStepSuffix)};
  TF_ASSIGN_OR_RETURN(const llvm::SmallVector<AssetFileDef> asset_file_defs,
                      RunExportPasses(export_opts, *context, *module_ref));

  return ConvertMlirModuleToExportedModel(
      *module_ref, checkpoint_dir, updated_function_aliases,
      {asset_file_defs.begin(), asset_file_defs.end()});
}

}  // namespace quantization
}  // namespace tensorflow

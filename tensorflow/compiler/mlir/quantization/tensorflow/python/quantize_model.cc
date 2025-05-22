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
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/component.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/statistics.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/config.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/context.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/debugger.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_post_calibration.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_pre_calibration.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_saved_model_export.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_saved_model_import.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_weight_only_ptq.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/types.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/tf_quantize_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/tf_quantize_preprocess.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"

namespace tensorflow {
namespace quantization {
namespace {

using ::mlir::quant::stablehlo::CreateMlirContextForQuantization;
using ::mlir::quant::stablehlo::FunctionAlias;
using ::mlir::quant::stablehlo::FunctionName;
using ::mlir::quant::stablehlo::RunCalibrationPasses;
using ::mlir::tf_quant::stablehlo::ConvertMlirModuleToExportedModel;
using ::mlir::tf_quant::stablehlo::ExportOptions;
using ::mlir::tf_quant::stablehlo::GetFunctionAliases;
using ::mlir::tf_quant::stablehlo::kExportStepSuffix;
using ::mlir::tf_quant::stablehlo::PostCalibrationComponent;
using ::mlir::tf_quant::stablehlo::PreCalibrationComponent;
using ::mlir::tf_quant::stablehlo::UpdateFunctionAliases;
using ::mlir::tf_quant::stablehlo::WeightOnlyPtqComponent;
using ::stablehlo::quantization::AddCalibrationStatistics;
using ::stablehlo::quantization::ChangeToQuantizedFilename;
using ::stablehlo::quantization::DebuggerConfig;
using ::stablehlo::quantization::ExpandPresets;
using ::stablehlo::quantization::IsCalibrationRequired;
using ::stablehlo::quantization::PopulateDefaults;
using ::stablehlo::quantization::QuantizationConfig;
using ::stablehlo::quantization::io::CreateTmpDir;
using ::stablehlo::quantization::io::GetLocalTmpFileName;
using ::tensorflow::quantization::PyFunctionLibrary;
using ::tensorflow::tf_quantization::AddQuantizePtqDynamicRangePasses;
using ::tensorflow::tf_quantization::AddQuantizePtqPostCalibrationPasses;
using ::tensorflow::tf_quantization::AddQuantizePtqPreCalibrationPasses;
using ::tensorflow::tf_quantization::AddQuantizeQatPasses;
using ::tensorflow::tf_quantization::AddQuantizeWeightOnlyPasses;
using ::tensorflow::tf_quantization::kDefaultTfQuantMlirDumpFilePrefix;
using ::tensorflow::tf_quantization::PreprocessAndFreezeGraph;

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportAndPreprocessSavedModel(
    absl::string_view saved_model_path,
    const std::vector<std::string> &signature_keys,
    const std::unordered_set<std::string> &tags, mlir::MLIRContext *context,
    const bool is_inliner_run, const bool run_tf_to_stablehlo,
    const bool deserialize_xla_call_module,
    absl::flat_hash_map<std::string, std::string> &function_aliases) {
  // Convert the SavedModelBundle to an MLIR module.
  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  import_options.lift_variables = false;
  import_options.include_variables_in_initializers = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO: b/213406917 - Add support for the object graph based saved model.
  std::vector<std::string> exported_names = signature_keys;
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tags,
                                          absl::MakeSpan(exported_names),
                                          context, import_options, &bundle);
  if (!module.status().ok()) {
    return absl::InternalError(absl::StrCat("Failed to import SavedModel: ",
                                            module.status().message()));
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();
  UpdateFunctionAliases(function_aliases, *module_ref);

  // Collect the names of the functions that have aliases so that they may not
  // be inlined.
  absl::flat_hash_set<std::string> aliased_function_names;
  absl::c_for_each(function_aliases, [&](const auto &aliases) {
    return aliased_function_names.insert(aliases.first);
  });

  TF_RETURN_IF_ERROR(PreprocessAndFreezeGraph(
      /*mlir_dump_file_prefix=*/kDefaultTfQuantMlirDumpFilePrefix,
      /*is_inliner_run=*/is_inliner_run,
      /*noinline_functions=*/aliased_function_names, module_ref.get(), context,
      bundle ? bundle->GetSession() : nullptr, run_tf_to_stablehlo,
      deserialize_xla_call_module));
  return module_ref;
}

absl::StatusOr<ExportedModel> ModuleOpToExportedModel(
    mlir::ModuleOp module_op, mlir::MLIRContext *ctx,
    absl::string_view step_name, const bool unfreeze_constants,
    const absl::flat_hash_map<std::string, std::string> &function_aliases) {
  TF_ASSIGN_OR_RETURN(const std::string checkpoint_dir, GetLocalTmpFileName());

  const auto export_opts =
      ExportOptions{/*duplicate_shape_determining_constants=*/true,
                    unfreeze_constants, checkpoint_dir,
                    /*debug_name=*/absl::StrCat(step_name, kExportStepSuffix)};

  TF_ASSIGN_OR_RETURN(const llvm::SmallVector<AssetFileDef> asset_file_defs,
                      RunExportPasses(export_opts, *ctx, module_op));

  return ConvertMlirModuleToExportedModel(
      module_op, checkpoint_dir, function_aliases,
      {asset_file_defs.begin(), asset_file_defs.end()});
}

absl::StatusOr<ExportedModel> ExportCalibrationModel(
    mlir::ModuleOp module_op, mlir::MLIRContext *context,
    const QuantizationOptions &quantization_options,
    const absl::flat_hash_map<std::string, std::string> &function_aliases,
    absl::string_view calibration_data_dir) {
  // Clone ModuleOp and function aliases so changes in this pipeline won't
  // be reflected in the original values.
  mlir::OwningOpRef<mlir::ModuleOp> cloned_module_ref(module_op.clone());

  TF_RETURN_IF_ERROR(
      RunCalibrationPasses(*cloned_module_ref, *context, calibration_data_dir,
                           quantization_options.calibration_options()
                               .force_regenerate_calibration_data()));
  if (!IsCalibrationRequired(*cloned_module_ref)) return ExportedModel();

  absl::StatusOr<ExportedModel> exported_model = ModuleOpToExportedModel(
      *cloned_module_ref, context, kTfQuantPtqPreCalibrationStepName,
      /*unfreeze_constants=*/!quantization_options.freeze_all_variables(),
      function_aliases);
  if (!exported_model.status().ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to export calibration model: ",
                     exported_model.status().message()));
  }

  return *exported_model;
}

absl::StatusOr<ExportedModel> ExportDebuggingModel(
    mlir::ModuleOp module_op, mlir::MLIRContext *context,
    const QuantizationOptions &quantization_options,
    const absl::flat_hash_map<std::string, std::string> &function_aliases) {
  // Clone ModuleOp and function aliases so changes in this pipeline won't
  // be reflected in the original values.
  mlir::OwningOpRef<mlir::ModuleOp> cloned_module_ref(module_op.clone());

  absl::StatusOr<ExportedModel> exported_model = ModuleOpToExportedModel(
      *cloned_module_ref, context, kTfQuantPtqPreCalibrationStepName,
      /*unfreeze_constants=*/!quantization_options.freeze_all_variables(),
      function_aliases);
  if (!exported_model.status().ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to export debugging model: ",
                     exported_model.status().message()));
  }

  return *exported_model;
}

QuantizationConfig GetQuantizationConfigForStaticRangePtq(
    const QuantizationOptions &quantization_options) {
  QuantizationConfig quantization_config{};
  // TODO: b/331302857 - Remove `enable_per_channel_quantized_weight` usage.
  quantization_config.mutable_static_range_ptq_preset()
      ->set_enable_per_channel_quantized_weight(
          quantization_options.enable_per_channel_quantization());
  // When targeting server TPUs quantized types should be unpacked into
  // integer ops.
  quantization_config.mutable_pipeline_config()->set_unpack_quantized_types(
      true);
  *quantization_config.mutable_debugger_config() =
      quantization_options.debugger_config();
  quantization_config.mutable_static_range_ptq_preset();
  *quantization_config.mutable_calibration_options() =
      quantization_options.calibration_options();

  return ExpandPresets(PopulateDefaults(quantization_config));
}

QuantizationConfig GetQuantizationConfigForWeightOnlyPtq(
    const QuantizationOptions &quantization_options) {
  QuantizationConfig quantization_config{};
  quantization_config.mutable_weight_only_ptq_preset();
  // When targeting server TPUs quantized types should be unpacked into
  // integer ops.
  quantization_config.mutable_pipeline_config()->set_unpack_quantized_types(
      true);
  *quantization_config.mutable_debugger_config() =
      quantization_options.debugger_config();

  return ExpandPresets(PopulateDefaults(quantization_config));
}

absl::StatusOr<ExportedModel> QuantizePtqModelPreCalibrationImpl(
    mlir::ModuleOp module_op, mlir::MLIRContext *context,
    const QuantizationOptions &quantization_options,
    const absl::flat_hash_map<std::string, std::string> &function_aliases,
    absl::string_view calibration_data_dir) {
  const bool is_stablehlo = quantization_options.op_set() == OpSet::STABLEHLO;
  // Use StableHLO Quantizer option if opset is specified.
  if (is_stablehlo) {
    const QuantizationConfig quantization_config =
        GetQuantizationConfigForStaticRangePtq(quantization_options);

    PreCalibrationComponent pre_calibration_component(context);
    TF_ASSIGN_OR_RETURN(module_op, pre_calibration_component.Run(
                                       module_op, quantization_config));
  } else {
    TF_RETURN_IF_ERROR(RunPasses(
        /*name=*/
        kTfQuantPtqPreCalibrationStepName, /*add_passes_func=*/
        [&quantization_options](mlir::PassManager &pm) {
          AddQuantizePtqPreCalibrationPasses(pm, quantization_options);
        },
        *context, module_op));
  }

  return ExportCalibrationModel(module_op, context, quantization_options,
                                function_aliases, calibration_data_dir);
}

absl::StatusOr<ExportedModel> QuantizePtqModelPostCalibrationImpl(
    mlir::ModuleOp module_op, mlir::MLIRContext *context,
    const QuantizationOptions &quantization_options,
    const absl::flat_hash_map<std::string, std::string> &function_aliases) {
  const bool is_stablehlo = quantization_options.op_set() == OpSet::STABLEHLO;
  // Use StableHLO Quantizer option if opset is specified.
  if (is_stablehlo) {
    const QuantizationConfig quantization_config =
        GetQuantizationConfigForStaticRangePtq(quantization_options);

    PostCalibrationComponent post_calibration_component(context);
    TF_ASSIGN_OR_RETURN(module_op, post_calibration_component.Run(
                                       module_op, quantization_config));
  } else {
    TF_RETURN_IF_ERROR(RunPasses(
        /*name=*/
        kTfQuantPtqPostCalibrationStepName, /*add_passes_func=*/
        [&quantization_options](mlir::PassManager &pm) {
          AddQuantizePtqPostCalibrationPasses(
              pm, quantization_options, kTfQuantPtqPostCalibrationStepName);
        },
        *context, module_op));
  }

  return ModuleOpToExportedModel(
      module_op, context, kTfQuantPtqPostCalibrationStepName,
      /*unfreeze_constants=*/!quantization_options.freeze_all_variables(),
      function_aliases);
}

}  // namespace

absl::StatusOr<ExportedModel> QuantizeQatModel(
    absl::string_view saved_model_path,
    const std::vector<std::string> &signature_keys,
    const std::unordered_set<std::string> &tags,
    const QuantizationOptions &quantization_options) {
  std::unique_ptr<mlir::MLIRContext> context =
      CreateMlirContextForQuantization();

  absl::StatusOr<absl::flat_hash_map<FunctionName, FunctionAlias>>
      function_aliases = GetFunctionAliases(saved_model_path, tags);
  if (!function_aliases.ok()) {
    return absl::InternalError(absl::StrCat(
        "Failed to get function alias: ", function_aliases.status().message()));
  }

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      ImportAndPreprocessSavedModel(
          saved_model_path, signature_keys, tags, context.get(),
          /*is_inliner_run=*/true,
          /*run_tf_to_stablehlo=*/false,
          /*deserialize_xla_call_module=*/false, *function_aliases);
  if (!module.status().ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to import and preprocess SavedModel: ",
                     module.status().message()));
  }
  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  TF_RETURN_IF_ERROR(RunPasses(
      /*name=*/
      kTfQuantQatStepName, /*add_passes_func=*/
      [&quantization_options](mlir::PassManager &pm) {
        AddQuantizeQatPasses(pm, quantization_options, kTfQuantQatStepName);
      },
      *context, *module_ref));

  return ModuleOpToExportedModel(
      *module_ref, context.get(), kTfQuantQatStepName,
      /*unfreeze_constants=*/!quantization_options.freeze_all_variables(),
      *function_aliases);
}

absl::StatusOr<ExportedModel> QuantizeDynamicRangePtq(
    absl::string_view saved_model_path,
    const std::vector<std::string> &signature_keys,
    const std::unordered_set<std::string> &tags,
    const QuantizationOptions &quantization_options) {
  std::unique_ptr<mlir::MLIRContext> context =
      CreateMlirContextForQuantization();

  absl::StatusOr<absl::flat_hash_map<FunctionName, FunctionAlias>>
      function_aliases = GetFunctionAliases(saved_model_path, tags);
  if (!function_aliases.ok()) {
    return absl::InternalError(absl::StrCat(
        "Failed to get function alias: ", function_aliases.status().message()));
  }

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      ImportAndPreprocessSavedModel(
          saved_model_path, signature_keys, tags, context.get(),
          /*is_inliner_run=*/true,
          /*run_tf_to_stablehlo=*/false, /*deserialize_xla_call_module=*/false,
          *function_aliases);
  if (!module.status().ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to import and preprocess SavedModel: ",
                     module.status().message()));
  }
  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  TF_RETURN_IF_ERROR(RunPasses(
      /*name=*/
      kTfQuantPtqDynamicRangeStepName, /*add_passes_func=*/
      [&quantization_options](mlir::PassManager &pm) {
        AddQuantizePtqDynamicRangePasses(pm, quantization_options,
                                         kTfQuantPtqDynamicRangeStepName);
      },
      *context, *module_ref));

  return ModuleOpToExportedModel(
      *module_ref, context.get(), kTfQuantPtqDynamicRangeStepName,
      /*unfreeze_constants=*/!quantization_options.freeze_all_variables(),
      *function_aliases);
}

// TODO: b/297626257 - [Converter Component][TF-Quantizer] Clean up
// quantize_model.cc by factoring out repeated codes
absl::StatusOr<ExportedModel> QuantizeWeightOnly(
    absl::string_view saved_model_path,
    const QuantizationOptions &quantization_options) {
  std::unique_ptr<mlir::MLIRContext> context =
      CreateMlirContextForQuantization();

  absl::StatusOr<absl::flat_hash_map<FunctionName, FunctionAlias>>
      function_aliases = GetFunctionAliases(
          saved_model_path, {quantization_options.tags().begin(),
                             quantization_options.tags().end()});
  if (!function_aliases.ok()) {
    return absl::InternalError(absl::StrCat(
        "Failed to get function alias: ", function_aliases.status().message()));
  }

  const bool is_stablehlo = quantization_options.op_set() == OpSet::STABLEHLO;
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      ImportAndPreprocessSavedModel(
          saved_model_path,
          {quantization_options.signature_keys().begin(),
           quantization_options.signature_keys().end()},
          {quantization_options.tags().begin(),
           quantization_options.tags().end()},
          context.get(), /*is_inliner_run=*/true,
          /*run_tf_to_stablehlo=*/is_stablehlo,
          /*deserialize_xla_call_module=*/false, *function_aliases);
  if (!module.status().ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to import and preprocess SavedModel: ",
                     module.status().message()));
  }
  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  // Use StableHLO Quantizer option if opset is specified.
  if (is_stablehlo) {
    const QuantizationConfig quantization_config =
        GetQuantizationConfigForWeightOnlyPtq(quantization_options);

    WeightOnlyPtqComponent weight_only_ptq_component(context.get());
    TF_ASSIGN_OR_RETURN(*module_ref, weight_only_ptq_component.Run(
                                         *module_ref, quantization_config));
  } else {
    TF_RETURN_IF_ERROR(RunPasses(
        kTfQuantWeightOnlyStepName,
        /*add_passes_func=*/
        [&quantization_options](mlir::PassManager &pm) {
          AddQuantizeWeightOnlyPasses(pm, quantization_options,
                                      kTfQuantWeightOnlyStepName);
        },
        *context, *module_ref));
  }

  return ModuleOpToExportedModel(
      *module_ref, context.get(), kTfQuantWeightOnlyStepName,
      /*unfreeze_constants=*/!quantization_options.freeze_all_variables(),
      *function_aliases);
}

absl::StatusOr<ExportedModel> QuantizeStaticRangePtq(
    absl::string_view saved_model_path,
    const std::vector<std::string> &signature_keys,
    const std::unordered_set<std::string> &tags,
    const QuantizationOptions &quantization_options,
    const absl::flat_hash_map<std::string, SignatureDef> &signature_def_map,
    const PyFunctionLibrary &py_function_library,
    const absl::flat_hash_map<std::string, RepresentativeDatasetFile>
        &representative_dataset_file_map_serialized) {
  std::unique_ptr<mlir::MLIRContext> context =
      CreateMlirContextForQuantization();

  absl::StatusOr<absl::flat_hash_map<FunctionName, FunctionAlias>>
      function_aliases = GetFunctionAliases(saved_model_path, tags);
  if (!function_aliases.ok()) {
    return absl::InternalError(absl::StrCat(
        "Failed to get function alias: ", function_aliases.status().message()));
  }

  const bool is_stablehlo = quantization_options.op_set() == OpSet::STABLEHLO;
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      ImportAndPreprocessSavedModel(
          saved_model_path, signature_keys, tags, context.get(),
          /*is_inliner_run=*/true,
          /*run_tf_to_stablehlo=*/is_stablehlo,
          /*deserialize_xla_call_module=*/false, *function_aliases);
  if (!module.status().ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to import and preprocess SavedModel: ",
                     module.status().message()));
  }
  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  std::string calibration_data_dir =
      quantization_options.calibration_options().calibration_data_dir();
  if (calibration_data_dir.empty()) {
    TF_ASSIGN_OR_RETURN(calibration_data_dir, CreateTmpDir());
  }

  TF_ASSIGN_OR_RETURN(ExportedModel calibration_exported_model,
                      QuantizePtqModelPreCalibrationImpl(
                          *module_ref, context.get(), quantization_options,
                          *function_aliases, calibration_data_dir));

  // Save and run the calibration model.
  if (calibration_exported_model.has_graph_def()) {
    TF_ASSIGN_OR_RETURN(std::string calibration_saved_model_dir,
                        CreateTmpDir());
    py_function_library.SaveExportedModel(
        calibration_saved_model_dir, calibration_exported_model,
        saved_model_path, tags, signature_def_map);

    py_function_library.RunCalibration(
        calibration_saved_model_dir, signature_keys, tags,
        quantization_options.force_graph_mode_calibration(),
        representative_dataset_file_map_serialized);
  }

  if (absl::Status status = AddCalibrationStatistics(
          *module_ref, calibration_data_dir,
          quantization_options.calibration_options(), py_function_library);
      !status.ok()) {
    LOG(WARNING) << "Some CustomAggregator ops do not have min or max "
                    "values. Parts of the graph are not quantized. "
                 << status;
  }

  // Saves the current model to the `unquantized_dump_model_path` if the
  // debugger type is `DEBUGGER_TYPE_WHOLE_MODEL`. This is required
  // because in whole-model debugging mode the `DumpTensor` ops for the
  // unquantized tensors are only inserted in the unquantized model
  // whereas `DumpTensor` ops for the quantized tensors are only inserted
  // in the quantized model. Both models are required to be able to dump
  // both quantized and unquantized tensors and compare them offline.
  if (quantization_options.has_debugger_config() &&
      quantization_options.debugger_config().debugger_type() ==
          DebuggerConfig::DEBUGGER_TYPE_WHOLE_MODEL) {
    TF_ASSIGN_OR_RETURN(
        ExportedModel debugging_exported_model,
        ExportDebuggingModel(*module_ref, context.get(), quantization_options,
                             *function_aliases));
    ChangeToQuantizedFilename(*module_ref);

    absl::string_view unquantized_dump_model_path =
        quantization_options.debugger_config().unquantized_dump_model_path();
    py_function_library.SaveExportedModel(
        unquantized_dump_model_path, debugging_exported_model, saved_model_path,
        tags, signature_def_map);
  }

  return QuantizePtqModelPostCalibrationImpl(
      *module_ref, context.get(), quantization_options, *function_aliases);
}

}  // namespace quantization
}  // namespace tensorflow

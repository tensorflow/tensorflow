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

#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/representative_dataset.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/statistics.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/debugger.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/saved_model_export.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/types.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace mlir::quant::stablehlo {
namespace {

using ::stablehlo::quantization::AddCalibrationStatistics;
using ::stablehlo::quantization::CreateRepresentativeDatasetFileMap;
using ::stablehlo::quantization::DisableDebugging;
using ::stablehlo::quantization::IsCalibrationRequired;
using ::stablehlo::quantization::QuantizationConfig;
using ::stablehlo::quantization::ReadStatistics;
using ::stablehlo::quantization::RepresentativeDatasetConfig;
using ::stablehlo::quantization::io::CreateTmpDir;
using ::stablehlo::quantization::io::GetLocalTmpFileName;
using ::stablehlo::quantization::io::ListDirectory;
using ::tensorflow::AssetFileDef;
using ::tensorflow::SignatureDef;
using ::tensorflow::calibrator::CalibrationStatistics;
using ::tensorflow::quantization::ExportedModel;
using ::tensorflow::quantization::PyFunctionLibrary;
using ::tensorflow::quantization::RunPasses;
using CalibrationStatisticsFlatMap =
    absl::flat_hash_map<std::string, CalibrationStatistics>;

}  // namespace

absl::Status RunCalibrationPasses(
    mlir::ModuleOp module_op, MLIRContext& ctx,
    absl::string_view calibration_data_dir,
    const bool force_regenerate_calibration_data) {
  // Disable DumpTensor ops when running calibration.
  DisableDebugging(module_op);

  std::vector<std::string> skipping_aggregator_ops;
  if (!force_regenerate_calibration_data) {
    TF_ASSIGN_OR_RETURN(const CalibrationStatisticsFlatMap statistics_map,
                        ReadStatistics(calibration_data_dir));
    absl::c_for_each(statistics_map, [&](const auto& iter) {
      return skipping_aggregator_ops.push_back(iter.first);
    });
  }

  return RunPasses(
      /*name=*/
      CalibrationComponent::kName,
      /*add_passes_func=*/
      [calibration_data_dir, &skipping_aggregator_ops](PassManager& pm) {
        pm.addPass(CreateInsertCalibrationStatisticsSaverPass(
            calibration_data_dir, skipping_aggregator_ops));
      },
      ctx, module_op);
}

CalibrationComponent::CalibrationComponent(
    absl::Nonnull<MLIRContext*> ctx,
    absl::Nonnull<const PyFunctionLibrary*> py_function_lib,
    const absl::string_view src_saved_model_path,
    absl::flat_hash_map<FunctionName, FunctionAlias> function_aliases,
    std::unordered_set<std::string> tags,
    absl::flat_hash_map<std::string, SignatureDef> signature_def_map,
    std::vector<std::string> signature_keys)
    : ctx_(ABSL_DIE_IF_NULL(ctx)),                          // Crash OK
      py_function_lib_(ABSL_DIE_IF_NULL(py_function_lib)),  // Crash OK
      src_saved_model_path_(src_saved_model_path),
      function_aliases_(std::move(function_aliases)),
      tags_(std::move(tags)),
      signature_def_map_(std::move(signature_def_map)),
      signature_keys_(std::move(signature_keys)) {}

absl::Status CalibrationComponent::ExportToSavedModel(
    ModuleOp module_op, absl::string_view calibration_data_dir,
    const bool force_regenerate_calibration_data,
    const absl::string_view dst_saved_model_path) {
  TF_ASSIGN_OR_RETURN(const std::string checkpoint_dir, GetLocalTmpFileName());

  // Clone ModuleOp and function aliases so changes in this pipeline won't
  // be reflected in the original values.
  mlir::OwningOpRef<mlir::ModuleOp> cloned_module_ref(module_op.clone());

  TF_RETURN_IF_ERROR(RunCalibrationPasses(*cloned_module_ref, *ctx_,
                                          calibration_data_dir,
                                          force_regenerate_calibration_data));

  const bool is_calibration_required =
      IsCalibrationRequired(*cloned_module_ref);
  if (!is_calibration_required) return absl::OkStatus();

  // `duplicate_shape_determining_constants = false` because the
  // resulting graph of this step is not expected to be loaded on TPU.
  const ExportOptions export_opts = {
      /*duplicate_shape_determining_constants=*/false,
      /*unfreeze_constants=*/false, checkpoint_dir,
      /*debug_name=*/absl::StrCat(kName, kExportStepSuffix)};

  TF_ASSIGN_OR_RETURN(const SmallVector<AssetFileDef> asset_file_defs,
                      RunExportPasses(export_opts, *ctx_, *cloned_module_ref));

  TF_ASSIGN_OR_RETURN(ExportedModel exported_model,
                      ConvertMlirModuleToExportedModel(
                          *cloned_module_ref, checkpoint_dir, function_aliases_,
                          {asset_file_defs.begin(), asset_file_defs.end()}));

  py_function_lib_->SaveExportedModel(dst_saved_model_path, exported_model,
                                      src_saved_model_path_, tags_,
                                      signature_def_map_);

  return absl::OkStatus();
}

absl::StatusOr<ModuleOp> CalibrationComponent::Run(
    ModuleOp module_op, const QuantizationConfig& config) {
  // Export the calibration model to SavedModel.
  TF_ASSIGN_OR_RETURN(const std::string calibration_saved_model_dir,
                      CreateTmpDir());

  std::string calibration_data_dir =
      config.calibration_options().calibration_data_dir();
  if (calibration_data_dir.empty()) {
    TF_ASSIGN_OR_RETURN(calibration_data_dir, CreateTmpDir());
  }

  TF_RETURN_IF_ERROR(ExportToSavedModel(
      module_op, calibration_data_dir,
      config.calibration_options().force_regenerate_calibration_data(),
      calibration_saved_model_dir));

  TF_ASSIGN_OR_RETURN(std::vector<std::string> calibration_saved_model_files,
                      ListDirectory(calibration_saved_model_dir));
  if (!calibration_saved_model_files.empty()) {
    // Translate `RepresentativeDatasetConfig`s to signature key ->
    // `RepresentativeDatasetFile` mapping.
    const auto dataset_configs =
        config.calibration_options().representative_datasets();
    const std::vector<RepresentativeDatasetConfig> dataset_config_vector(
        dataset_configs.begin(), dataset_configs.end());
    TF_ASSIGN_OR_RETURN(
        const auto representative_dataset_file_map,
        CreateRepresentativeDatasetFileMap(dataset_config_vector));

    // Run calibration on the exported model.
    if (py_function_lib_->RunCalibration(
            calibration_saved_model_dir, signature_keys_, tags_,
            /*force_graph_mode_calibration=*/true,
            representative_dataset_file_map) == std::nullopt) {
      return absl::InternalError(
          "CalibrationComponent error: Failed to run calibration.");
    }
  }

  if (absl::Status status = AddCalibrationStatistics(
          module_op, calibration_data_dir, config.calibration_options(),
          *py_function_lib_);
      !status.ok()) {
    LOG(WARNING) << "Some CustomAggregator ops do not have min or max "
                    "values. Parts of the graph are not quantized. "
                 << status;
  }

  return module_op;
}

}  // namespace mlir::quant::stablehlo

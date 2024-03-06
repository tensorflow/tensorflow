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
#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/component.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/component.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/context.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/post_calibration.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/pre_calibration.h"
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
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace mlir::quant::stablehlo {
namespace {

using ::stablehlo::quantization::QuantizationConfig;
using ::stablehlo::quantization::io::GetLocalTmpFileName;
using ::tensorflow::AssetFileDef;
using ::tensorflow::MLIRImportOptions;
using ::tensorflow::SavedModelBundle;
using ::tensorflow::SavedModelSignatureDefsToMlirImport;
using ::tensorflow::SignatureDef;
using ::tensorflow::quantization::ExportedModel;
using ::tensorflow::quantization::PreprocessAndFreezeGraph;
using ::tensorflow::quantization::PyFunctionLibrary;
using ::tensorflow::quantization::RunPasses;
using ::tensorflow::quantization::UnfreezeConstantsAndSaveVariables;

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

// Represents a pair of `mlir::ModuleOp` and `tensorflow::SavedModelBundle`. The
// SavedModelBundle complements the imported ModuleOp by providing access to
// `tensorflow::Session` which may be useful when reading values from resources
// (e.g. `TF::VarHandleOp`s).
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

absl::StatusOr<ModuleOp> ImportSavedModel(
    const absl::string_view saved_model_path,
    const std::vector<std::string>& signature_keys,
    const std::unordered_set<std::string>& tags,
    const QuantizationConfig& quantization_config,
    const absl::flat_hash_map<FunctionName, FunctionAlias>& function_aliases,
    MLIRContext& ctx ABSL_ATTRIBUTE_LIFETIME_BOUND) {
  TF_ASSIGN_OR_RETURN(
      ImportedMlirModuleOp imported_module,
      SavedModelToMlirModuleOp(saved_model_path, tags, signature_keys, ctx));
  auto [module_op, saved_model_bundle] = std::move(imported_module);

  const absl::flat_hash_map<FunctionName, FunctionAlias>
      updated_function_aliases =
          UpdateFunctionAliases(function_aliases, module_op);

  // Collect the names of the functions that have aliases so that they may not
  // be inlined.
  absl::flat_hash_set<std::string> aliased_function_names;
  absl::c_for_each(updated_function_aliases, [&](const auto& aliases) {
    return aliased_function_names.insert(aliases.first);
  });

  TF_RETURN_IF_ERROR(PreprocessAndFreezeGraph(
      /*mlir_dump_file_prefix=*/PreCalibrationComponent::kName,
      /*is_inliner_run=*/true, /*noinline_functions=*/aliased_function_names,
      module_op, &ctx,
      saved_model_bundle == nullptr ? nullptr
                                    : saved_model_bundle->GetSession(),
      /*run_tf_to_stablehlo=*/true, /*deserialize_xla_call_module=*/false));
  return module_op;
}

absl::StatusOr<ExportedModel> CreateExportedModel(
    const std::vector<std::string>& signature_keys,
    const std::unordered_set<std::string>& tags,
    const QuantizationConfig& quantization_config,
    const absl::flat_hash_map<FunctionName, FunctionAlias>& function_aliases,
    MLIRContext& ctx ABSL_ATTRIBUTE_LIFETIME_BOUND, ModuleOp module_op) {
  TF_ASSIGN_OR_RETURN(const std::string checkpoint_dir, GetLocalTmpFileName());
  const ExportOptions export_opts = {
      /*duplicate_shape_determining_constants=*/true,
      /*unfreeze_constants=*/false, checkpoint_dir,
      /*debug_name=*/
      absl::StrCat(PostCalibrationComponent::kName, kExportStepSuffix)};

  TF_ASSIGN_OR_RETURN(const SmallVector<AssetFileDef> asset_file_defs,
                      RunExportPasses(export_opts, ctx, module_op));

  const absl::flat_hash_map<FunctionName, FunctionAlias>
      updated_function_aliases =
          UpdateFunctionAliases(function_aliases, module_op);

  return ConvertMlirModuleToExportedModel(
      module_op, checkpoint_dir, updated_function_aliases,
      {asset_file_defs.begin(), asset_file_defs.end()});
}

}  // namespace

StaticRangePtqComponent::StaticRangePtqComponent(
    absl::Nonnull<MLIRContext*> ctx,
    absl::Nonnull<const PyFunctionLibrary*> py_function_library,
    const absl::string_view src_saved_model_path,
    std::vector<std::string> signature_keys,
    std::unordered_set<std::string> tags,
    absl::flat_hash_map<std::string, SignatureDef> signature_def_map,
    absl::flat_hash_map<FunctionName, FunctionAlias> function_aliases)
    : ctx_(ctx) {
  // Initialize the three sub-components.
  sub_components_[0] = std::make_unique<PreCalibrationComponent>(ctx_);
  sub_components_[1] = std::make_unique<CalibrationComponent>(
      ctx_, py_function_library, src_saved_model_path,
      std::move(function_aliases), std::move(tags),
      std::move(signature_def_map), std::move(signature_keys));
  sub_components_[2] = std::make_unique<PostCalibrationComponent>(ctx_);
}

absl::StatusOr<ModuleOp> StaticRangePtqComponent::Run(
    ModuleOp module_op, const QuantizationConfig& config) {
  // Runs sub-components in sequence: PreCalibrationComponent ->
  // CalibrationComponent -> PostCalibrationComponents.
  for (std::unique_ptr<Component>& sub_component : sub_components_) {
    TF_ASSIGN_OR_RETURN(module_op, sub_component->Run(module_op, config));
  }

  return module_op;
}

// TODO: b/317167427 - Enable debugger.
absl::Status QuantizeStaticRangePtq(
    const absl::string_view src_saved_model_path,
    const absl::string_view dst_saved_model_path,
    QuantizationConfig quantization_config,
    const std::vector<std::string>& signature_keys,
    const absl::flat_hash_map<std::string, SignatureDef>& signature_def_map,
    const absl::flat_hash_map<FunctionName, FunctionAlias>& function_aliases,
    const PyFunctionLibrary& py_function_library) {
  std::unordered_set<std::string> tags;
  tags.insert(quantization_config.tf_saved_model().tags().begin(),
              quantization_config.tf_saved_model().tags().end());
  if (!quantization_config.has_calibration_options()) {
    *quantization_config.mutable_calibration_options() =
        GetDefaultCalibrationOptions();
  }

  std::unique_ptr<MLIRContext> ctx = CreateMlirContextForQuantization();

  TF_ASSIGN_OR_RETURN(
      ModuleOp module_op,
      ImportSavedModel(src_saved_model_path, signature_keys, tags,
                       quantization_config, function_aliases, *ctx));

  StaticRangePtqComponent static_range_ptq_component(
      ctx.get(), &py_function_library, src_saved_model_path, signature_keys,
      tags, signature_def_map, function_aliases);
  TF_ASSIGN_OR_RETURN(module_op, static_range_ptq_component.Run(
                                     module_op, quantization_config));

  TF_ASSIGN_OR_RETURN(
      const ExportedModel post_calibrated_exported_model,
      CreateExportedModel(signature_keys, tags, quantization_config,
                          function_aliases, *ctx, module_op));

  // Remove the `tpu` tag for exporting because the output quantized model is
  // essentially a CPU model.
  tags.erase("tpu");

  py_function_library.SaveExportedModel(
      dst_saved_model_path, post_calibrated_exported_model,
      src_saved_model_path, tags, signature_def_map);

  return absl::OkStatus();
}

}  // namespace mlir::quant::stablehlo

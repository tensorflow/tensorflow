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
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/component.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/component.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/context.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_post_calibration.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_pre_calibration.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_saved_model_export.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_saved_model_import.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/types.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"

namespace mlir::quant::stablehlo {

using ::mlir::tf_quant::stablehlo::CreateExportedModel;
using ::mlir::tf_quant::stablehlo::GetFunctionAliases;
using ::mlir::tf_quant::stablehlo::ImportSavedModel;
using ::mlir::tf_quant::stablehlo::PostCalibrationComponent;
using ::mlir::tf_quant::stablehlo::PreCalibrationComponent;
using ::stablehlo::quantization::QuantizationConfig;
using ::tensorflow::SignatureDef;
using ::tensorflow::quantization::ExportedModel;
using ::tensorflow::quantization::PyFunctionLibrary;

StaticRangePtqComponent::StaticRangePtqComponent(
    MLIRContext* absl_nonnull ctx,
    const PyFunctionLibrary* absl_nonnull py_function_library,
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
    const QuantizationConfig& quantization_config,
    const std::vector<std::string>& signature_keys,
    const absl::flat_hash_map<std::string, SignatureDef>& signature_def_map,
    const PyFunctionLibrary& py_function_library) {
  std::unordered_set<std::string> tags;
  tags.insert(quantization_config.tf_saved_model().tags().begin(),
              quantization_config.tf_saved_model().tags().end());

  std::unique_ptr<MLIRContext> ctx = CreateMlirContextForQuantization();

  absl::StatusOr<absl::flat_hash_map<FunctionName, FunctionAlias>>
      function_aliases = GetFunctionAliases(src_saved_model_path, tags);
  if (!function_aliases.ok()) {
    return absl::InternalError(absl::StrCat(
        "Failed to get function alias: ", function_aliases.status().message()));
  }

  TF_ASSIGN_OR_RETURN(
      OwningOpRef<ModuleOp> module,
      ImportSavedModel(src_saved_model_path, signature_keys, tags,
                       quantization_config, PreCalibrationComponent::kName,
                       *function_aliases, *ctx));

  StaticRangePtqComponent static_range_ptq_component(
      ctx.get(), &py_function_library, src_saved_model_path, signature_keys,
      tags, signature_def_map, *function_aliases);
  TF_ASSIGN_OR_RETURN(
      *module, static_range_ptq_component.Run(*module, quantization_config));

  TF_ASSIGN_OR_RETURN(
      const ExportedModel post_calibrated_exported_model,
      CreateExportedModel(signature_keys, tags, quantization_config,
                          PostCalibrationComponent::kName, *function_aliases,
                          *ctx, *module));

  // Remove the `tpu` tag for exporting because the output quantized model is
  // essentially a CPU model.
  tags.erase("tpu");

  py_function_library.SaveExportedModel(
      dst_saved_model_path, post_calibrated_exported_model,
      src_saved_model_path, tags, signature_def_map);

  return absl::OkStatus();
}

}  // namespace mlir::quant::stablehlo

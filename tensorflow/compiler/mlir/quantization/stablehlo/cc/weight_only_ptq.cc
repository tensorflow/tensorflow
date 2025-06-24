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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/weight_only_ptq.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/config.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/context.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/pass_pipeline.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/saved_model_export.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/saved_model_import.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/types.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/instrumentations/save_report.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace mlir::quant::stablehlo {

using ::mlir::quant::stablehlo::AddWeightOnlyQuantizationPasses;
using ::stablehlo::quantization::GetReportFilePath;
using ::stablehlo::quantization::QuantizationConfig;
using ::tensorflow::SignatureDef;
using ::tensorflow::quantization::ExportedModel;
using ::tensorflow::quantization::PyFunctionLibrary;
using ::tensorflow::quantization::RunPasses;

WeightOnlyPtqComponent::WeightOnlyPtqComponent(MLIRContext* absl_nonnull ctx)
    : ctx_(ABSL_DIE_IF_NULL(ctx)) {}  // Crash OK

absl::StatusOr<ModuleOp> WeightOnlyPtqComponent::Run(
    ModuleOp module_op, const QuantizationConfig& config) {
  TF_RETURN_IF_ERROR(RunPasses(
      kName, /*add_passes_func=*/
      [&config](PassManager& pm) {
        // Add instrumentation to save quantization report after quantization.
        pm.addInstrumentation(
            std::make_unique<SaveQuantizationReportInstrumentation>(
                GetReportFilePath(config)));

        AddWeightOnlyQuantizationPasses(pm, config.specs(),
                                        config.pipeline_config(),
                                        config.debugger_config());
      },
      *ctx_, module_op));
  return module_op;
}

absl::Status QuantizeWeightOnlyPtq(
    const absl::string_view src_saved_model_path,
    const absl::string_view dst_saved_model_path,
    QuantizationConfig quantization_config,
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
      auto module,
      ImportSavedModel(src_saved_model_path, signature_keys, tags,
                       quantization_config, WeightOnlyPtqComponent::kName,
                       *function_aliases, *ctx));

  WeightOnlyPtqComponent weight_only_ptq_component(ctx.get());
  TF_ASSIGN_OR_RETURN(
      *module, weight_only_ptq_component.Run(*module, quantization_config));

  TF_ASSIGN_OR_RETURN(
      const ExportedModel post_calibrated_exported_model,
      CreateExportedModel(signature_keys, tags, quantization_config,
                          WeightOnlyPtqComponent::kName, *function_aliases,
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

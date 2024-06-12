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

#include "tensorflow/compiler/mlir/lite/quantization/lite/quantize_model.h"

#include <optional>
#include <string>
#include <unordered_set>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/debug/debug.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/c/c_api_types.h"

namespace mlir {
namespace lite {

std::string TfLiteToMlir(const absl::string_view tflite_op_name) {
  llvm::StringRef op_name(tflite_op_name.data(), tflite_op_name.size());
  return llvm::Twine("tfl.", op_name.lower()).str();
}

// TODO(fengliuai): check the result for `fully_quantize` flag.
TfLiteStatus QuantizeModel(
    const absl::string_view model_buffer, const tflite::TensorType &input_type,
    const tflite::TensorType &output_type,
    const tflite::TensorType &inference_type,
    const std::unordered_set<std::string> &operator_names,
    bool disable_per_channel, bool fully_quantize, std::string &output_buffer,
    bool verify_numeric, bool whole_model_verify, bool legacy_float_scale,
    const absl::flat_hash_set<std::string> &denylisted_ops,
    const absl::flat_hash_set<std::string> &denylisted_nodes,
    const bool enable_variable_quantization,
    bool disable_per_channel_for_dense_layers,
    const std::optional<const tensorflow::converter::DebugOptions>
        &debug_options) {
  // Translate TFLite names to mlir op names.
  absl::flat_hash_set<std::string> denylisted_mlir_op_names;
  for (const auto& entry : denylisted_ops) {
    denylisted_mlir_op_names.insert(TfLiteToMlir(entry));
  }

  DialectRegistry registry;
  registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  MLIRContext context(registry);
  StatusScopedDiagnosticHandler statusHandler(&context,
                                              /*propagate=*/true);

  OwningOpRef<mlir::ModuleOp> module = tflite::FlatBufferToMlir(
      model_buffer, &context, UnknownLoc::get(&context));
  if (!module) {
    LOG(ERROR) << "Couldn't import flatbuffer to MLIR.";
    return kTfLiteError;
  }

  // Apply quantization passes.
  PassManager pm((*module)->getName(), OpPassManager::Nesting::Implicit);
  if (debug_options.has_value()) {
    // Add debugging instrumentation
    tensorflow::InitPassManager(pm, debug_options.value());
  }
  quant::QuantizationSpecs quant_specs;
  quant_specs.inference_type = tflite::TflTypeToTfType(inference_type);
  quant_specs.post_training_quantization = true;
  quant_specs.disable_per_channel = disable_per_channel;
  quant_specs.disable_per_channel_for_dense_layers =
      disable_per_channel_for_dense_layers;
  quant_specs.verify_numeric = verify_numeric;
  quant_specs.whole_model_verify = whole_model_verify;
  quant_specs.legacy_float_scale = legacy_float_scale;
  quant_specs.ops_blocklist = denylisted_mlir_op_names;
  quant_specs.nodes_blocklist = denylisted_nodes;
  quant_specs.enable_mlir_variable_quantization = enable_variable_quantization;

  llvm::dbgs() << "fully_quantize: " << fully_quantize
               << ", inference_type: " << quant_specs.inference_type
               << ", input_inference_type: "
               << tflite::EnumNameTensorType(input_type)
               << ", output_inference_type: "
               << tflite::EnumNameTensorType(output_type) << "\n";
  mlir::Builder mlir_builder(&context);
  mlir::Type input_mlir_type =
      tflite::ConvertElementType(input_type, mlir_builder);
  mlir::Type output_mlir_type =
      tflite::ConvertElementType(output_type, mlir_builder);

  if (fully_quantize) {
    input_mlir_type = tflite::ConvertElementType(inference_type, mlir_builder);
    output_mlir_type = input_mlir_type;
  }

  tensorflow::AddQuantizationPasses(mlir::TFL::PassConfig(quant_specs), pm);
  pm.addPass(TFL::CreateModifyIONodesPass(input_mlir_type, output_mlir_type));
  // If the first or final ops are not quantized, remove QDQ.
  pm.addPass(TFL::CreatePostQuantizeRemoveQDQPass());
  if (failed(pm.run(module.get()))) {
    const std::string err(statusHandler.ConsumeStatus().message());
    LOG(ERROR) << "Failed to quantize: " << err;
    return kTfLiteError;
  }

  // Export the results.
  tflite::FlatbufferExportOptions options;
  options.toco_flags.set_force_select_tf_ops(false);
  options.toco_flags.set_enable_select_tf_ops(true);
  options.toco_flags.set_allow_custom_ops(true);
  if (!tflite::MlirToFlatBufferTranslateFunction(module.get(), options,
                                                 &output_buffer)) {
    LOG(ERROR) << "Failed to export MLIR to flatbuffer.";
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace lite
}  // namespace mlir

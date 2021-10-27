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

#include <string>

#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mlir {
namespace lite {

std::string TfLiteToMlir(const absl::string_view tflite_op_name) {
  llvm::StringRef op_name(tflite_op_name.data(), tflite_op_name.size());
  return llvm::Twine("tfl.", op_name.lower()).str();
}

// TODO(fengliuai): check the result for `fully_quantize` flag.
TfLiteStatus QuantizeModel(
    const tflite::ModelT& input_model, const tflite::TensorType& input_type,
    const tflite::TensorType& output_type,
    const tflite::TensorType& inference_type,
    const std::unordered_set<std::string>& operator_names,
    bool disable_per_channel, bool fully_quantize,
    flatbuffers::FlatBufferBuilder* builder,
    tflite::ErrorReporter* error_reporter, bool verify_numeric,
    bool whole_model_verify, bool legacy_float_scale,
    const StringSet& denylisted_ops, const StringSet& denylisted_nodes) {
  // Translate TFLite names to mlir op names.
  StringSet denylisted_mlir_op_names;
  for (const auto& entry : denylisted_ops) {
    denylisted_mlir_op_names.insert(TfLiteToMlir(entry));
  }

  DialectRegistry registry;
  registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  MLIRContext context(registry);
  StatusScopedDiagnosticHandler statusHandler(&context,
                                              /*propagate=*/true);

  // Import input_model to a MLIR module
  flatbuffers::FlatBufferBuilder input_builder;
  flatbuffers::Offset<tflite::Model> input_model_location =
      tflite::Model::Pack(input_builder, &input_model);
  tflite::FinishModelBuffer(input_builder, input_model_location);

  std::string serialized_model(
      reinterpret_cast<const char*>(input_builder.GetBufferPointer()),
      input_builder.GetSize());

  OwningModuleRef module = tflite::FlatBufferToMlir(serialized_model, &context,
                                                    UnknownLoc::get(&context));
  if (!module) {
    error_reporter->Report("Couldn't import flatbuffer to MLIR.");
    return kTfLiteError;
  }

  // Apply quantization passes.
  PassManager pm(module->getContext(), OpPassManager::Nesting::Implicit);
  TFL::QuantizationSpecs quant_specs;
  quant_specs.inference_type = tflite::TflTypeToTfType(inference_type);
  quant_specs.post_training_quantization = true;
  quant_specs.disable_per_channel = disable_per_channel;
  quant_specs.verify_numeric = verify_numeric;
  quant_specs.whole_model_verify = whole_model_verify;
  quant_specs.legacy_float_scale = legacy_float_scale;

  llvm::dbgs() << "fully_quantize: " << fully_quantize
               << ", inference_type: " << quant_specs.inference_type
               << ", input_inference_type: " << input_type
               << ", output_inference_type: " << output_type << "\n";
  mlir::Builder mlir_builder(&context);
  mlir::Type input_mlir_type =
      tflite::ConvertElementType(input_type, mlir_builder);
  mlir::Type output_mlir_type =
      tflite::ConvertElementType(output_type, mlir_builder);

  if (fully_quantize) {
    input_mlir_type = tflite::ConvertElementType(inference_type, mlir_builder);
    output_mlir_type = input_mlir_type;
  }

  pm.addPass(TFL::CreatePrepareQuantizePass(quant_specs));
  pm.addPass(TFL::CreateQuantizePass(quant_specs, denylisted_mlir_op_names,
                                     denylisted_nodes));
  pm.addPass(TFL::CreatePostQuantizePass(/*emit_quant_adaptor_ops=*/true));
  pm.addPass(TFL::CreateOptimizeOpOrderPass());
  pm.addPass(TFL::CreateModifyIONodesPass(input_mlir_type, output_mlir_type));
  // If the first or final ops are not quantized, remove QDQ.
  pm.addPass(TFL::CreatePostQuantizeRemoveQDQPass());
  if (failed(pm.run(module.get()))) {
    const std::string& err = statusHandler.ConsumeStatus().error_message();
    error_reporter->Report("Failed to quantize: %s", err.c_str());
    return kTfLiteError;
  }

  // Export the results to the builder
  std::string result;
  tflite::FlatbufferExportOptions options;
  options.toco_flags.set_force_select_tf_ops(false);
  options.toco_flags.set_enable_select_tf_ops(true);
  options.toco_flags.set_allow_custom_ops(true);
  if (!tflite::MlirToFlatBufferTranslateFunction(module.get(), options,
                                                 &result)) {
    error_reporter->Report("Failed to export MLIR to flatbuffer.");
    return kTfLiteError;
  }
  builder->PushFlatBuffer(reinterpret_cast<const uint8_t*>(result.data()),
                          result.size());

  return kTfLiteOk;
}

}  // namespace lite
}  // namespace mlir

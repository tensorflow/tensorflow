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

#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mlir {
namespace lite {

// TODO(fengliuai): check the result for `fully_quantize` flag.
TfLiteStatus QuantizeModel(
    const tflite::ModelT& input_model, const tflite::TensorType& input_type,
    const tflite::TensorType& output_type,
    const tflite::TensorType& inference_type,
    const std::unordered_set<std::string>& operator_names,
    bool disable_per_channel, bool fully_quantize,
    flatbuffers::FlatBufferBuilder* builder,
    tflite::ErrorReporter* error_reporter) {
  // TODO(b/142502494): remove this restriction by improving the `emit_adaptor`
  // flag
  if (input_type != output_type) {
    error_reporter->Report("Required same input type and output type.");
    return kTfLiteError;
  }

  MLIRContext context;
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

  // Apply quantization passes
  PassManager pm(module->getContext());
  TFL::QuantizationSpecs quant_specs;
  quant_specs.inference_type = tflite::TflTypeToTfType(inference_type);
  quant_specs.post_training_quantization = true;
  quant_specs.disable_per_channel = disable_per_channel;

  bool emit_adaptor = false;
  auto input_tf_type = tflite::TflTypeToTfType(input_type);
  if (input_tf_type == tensorflow::DT_FLOAT) {
    emit_adaptor = true;
  } else if (input_tf_type == tensorflow::DT_UINT8 ||
             input_tf_type == tensorflow::DT_INT8 ||
             input_tf_type == tensorflow::DT_INT16) {
    quant_specs.inference_type = input_tf_type;
  }

  pm.addPass(TFL::CreatePrepareQuantizePass(quant_specs));
  pm.addPass(TFL::CreateQuantizePass());
  pm.addPass(TFL::CreatePostQuantizePass(emit_adaptor));

  if (failed(pm.run(module.get()))) {
    const std::string& err = statusHandler.ConsumeStatus().error_message();
    error_reporter->Report("Failed to quantize: %s", err.c_str());
    return kTfLiteError;
  }

  // Export the results to the builder
  std::string result;
  if (tflite::MlirToFlatBufferTranslateFunction(
          module.get(), &result, /*emit_builtin_tflite_ops=*/true,
          /*emit_select_tf_ops=*/true, /*emit_custom_ops=*/true)) {
    error_reporter->Report("Failed to export MLIR to flatbuffer.");
    return kTfLiteError;
  }
  builder->PushFlatBuffer(reinterpret_cast<const uint8_t*>(result.data()),
                          result.size());

  return kTfLiteOk;
}

}  // namespace lite
}  // namespace mlir

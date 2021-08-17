/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/sparsity/sparsify_model.h"

#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mlir {
namespace lite {

TfLiteStatus SparsifyModel(const tflite::ModelT& input_model,
                           flatbuffers::FlatBufferBuilder* builder,
                           tflite::ErrorReporter* error_reporter) {
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

  PassManager pm(module->getContext(), OpPassManager::Nesting::Implicit);
  pm.addPass(TFL::CreateDenseToSparsePass());

  if (failed(pm.run(module.get()))) {
    const std::string& err = statusHandler.ConsumeStatus().error_message();
    error_reporter->Report("Failed to sparsify: %s", err.c_str());
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

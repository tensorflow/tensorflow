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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_STABLEHLO_TFL_PIPELINE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_STABLEHLO_TFL_PIPELINE_H_

#include <string>

#include "absl/status/status.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"

namespace mlir::TFL {

// Converts StableHLO MLIR module to TFLite flatbuffer streamed directly to an
// output stream.
absl::Status ConvertStableHloToTFLite(
    mlir::ModuleOp module, const tflite::ConverterFlags& converter_flags,
    const mlir::TFL::PassConfig& pass_config,
    llvm::raw_pwrite_stream& export_stream);

}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_STABLEHLO_TFL_PIPELINE_H_

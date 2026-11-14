/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_MLIR_BYTECODE_TO_TFLITE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_MLIR_BYTECODE_TO_TFLITE_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/compiler/mlir/lite/converter_v2_options.pb.h"

// Entry points of the Converter V2 path, which converts StableHLO programs
// exported by PyTorch and JAX (MLIR bytecode plus decoupled weights, see
// `LoadSlimModel`) to a TFLite flatbuffer. This path is intended to avoid
// depending on TensorFlow graph libraries.

namespace tensorflow::internal {

// Converts the slim model in `model_dir` to a TFLite flatbuffer streamed
// directly to `export_stream`.
absl::Status ConvertMlirBytecodeToTFLite(
    const tflite::ConverterV2Options& options, absl::string_view model_dir,
    llvm::raw_pwrite_stream& export_stream);

// Converts the slim model in `model_dir` to a TFLite flatbuffer written to
// `output_file_path`.
absl::Status ConvertMlirBytecodeToTFLite(
    const tflite::ConverterV2Options& options, absl::string_view model_dir,
    absl::string_view output_file_path);

}  // namespace tensorflow::internal

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_MLIR_BYTECODE_TO_TFLITE_H_

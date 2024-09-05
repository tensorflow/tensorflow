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

// Adaptor functions for StableHLO Quantizer.
// Provides simpler interfaces when integrating StableHLO Quantizer into TFLite
// Converter.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_STABLEHLO_QUANTIZATION_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_STABLEHLO_QUANTIZATION_H_

#include <string>
#include <unordered_set>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"

namespace tensorflow {

// Runs quantization on `module_op`. `saved_model_bundle` is required to
// retrieve information about the original model (e.g. signature def mapping)
// because quantization requires exporting the intermediate `ModuleOp` back to
// SavedModel for calibration. Similarly, `saved_model_dir` is required to
// access the assets of the original model. `saved_model_tags` uniquely
// identifies the `MetaGraphDef`. `quantization_config` determines the behavior
// of StableHLO Quantizer. `quantization_py_function_lib` contains python
// implementations of certain APIs that are required for calibration.
// `module_op` is the input graph to be quantized and it should contain
// StableHLO ops.
//
// Returns a quantized `ModuleOp` in StableHLO, potentially wrapped inside a
// XlaCallModuleOp. Returns a non-OK status if quantization fails, or any of
// `saved_model_bundle` or `quantization_py_function_lib` is a nullptr.
absl::StatusOr<mlir::ModuleOp> RunQuantization(
    const SavedModelBundle* saved_model_bundle,
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& saved_model_tags,
    const stablehlo::quantization::QuantizationConfig& quantization_config,
    const tensorflow::quantization::PyFunctionLibrary*
        quantization_py_function_lib,
    mlir::ModuleOp module_op);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_STABLEHLO_QUANTIZATION_H_

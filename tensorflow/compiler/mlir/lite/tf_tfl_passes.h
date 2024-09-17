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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_PASSES_H_

#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"

namespace tensorflow {

// Add the TF to TFLite passes, specified in the pass_config, into a
// pass_manager. The session object will be provided when the TF MLIR is
// imported from saved model version one and utilized for capturing resource
// variables. If the `saved_model_dir` directory path is provided, then the
// `tf_saved_model.asset` ops will be freezed.
void AddTFToTFLConversionPasses(llvm::StringRef saved_model_dir,
                                const tflite::ConverterFlags& converter_flags,
                                const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager);

// Adds the first portion of StableHLO->TF passes happening before quantization.
// The `pass_manager` that runs on a `mlir::ModuleOp` expects a graph containing
// a `mlir::TF::XlaCallModuleOp` with serialized StableHLO module. The resulting
// `mlir::ModuleOp` after running these passes will be an MHLO module, or a
// StableHLO module if `pass_config.enable_stablehlo_quantizer` is `true`. This
// is because StableHLO Quantizer accepts StableHLO modules.
void AddPreQuantizationStableHloToTfPasses(
    mlir::StringRef entry_function_name,
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager& pass_manager);

// Adds the second portion of StableHlo->TF passes happening after quantization.
// The input module is expected to be an MHLO module, or a quantized StableHLO
// graph (expressed as `mlir::TF::XlaCallModuleOp`s) if
// `pass_config.enable_stablehlo_quantizer` is `true`.
void AddPostQuantizationStableHloToTfPasses(
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager& pass_manager);

// This is the early part of the conversion in isolation. This enables a caller
// to inject more information in the middle of the conversion before resuming it
// (like freezing variables for example).
void AddPreVariableFreezingTFToTFLConversionPasses(
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager* pass_manager);

// This is the later part of the conversion in isolation. This enables a caller
// to resume the conversion after injecting more information in the middle of
// it.
void AddPostVariableFreezingTFToTFLConversionPasses(
    llvm::StringRef saved_model_dir,
    const tflite::ConverterFlags& converter_flags,
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager* pass_manager);

// Adds the passes that freeze variables from global tensors and unfreeze
// mutable global tensors. `pass_config` is used to determine whether to freeze
// variables and `pass_manager` will be populated with the passes to run.
void AddVariableFreezingFromGlobalTensorsPasses(
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager* pass_manager);

// Simplified API for TF->TFLite conversion with default flags.
void AddTFToTFLConversionPasses(const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager);

// Add the Quantization passes, specified in the pass_config, into a pass
// manager.
void AddQuantizationPasses(const mlir::TFL::PassConfig& pass_config,
                           mlir::OpPassManager& pass_manager);

// Add the DynamicRangeQuantization passes, specified in the pass_config, into a
// pass manager.
void AddDynamicRangeQuantizationPasses(const mlir::TFL::PassConfig& pass_config,
                                       mlir::OpPassManager& pass_manager);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TF_TFL_PASSES_H_

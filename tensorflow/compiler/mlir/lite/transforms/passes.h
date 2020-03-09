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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_

#include <memory>

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class FuncOp;
class ModuleOp;
template <typename T>
class OpPassBase;

namespace TFL {
class QuantizationSpecs;

// Creates an instance of the TensorFlow Lite dialect LegalizeTF pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateLegalizeTFPass();

// Creates an instance of the TensorFlow Lite dialect Optimize pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateOptimizePass();

// Creates an instance of the TensorFlow Lite dialect PrepareTF pass.
std::unique_ptr<OpPassBase<FuncOp>> CreatePrepareTFPass(
    bool unfold_batch_matmul);

// Creates an instance of the TensorFlow Lite dialect LowerStaticTensorList
// pass.
std::unique_ptr<OpPassBase<ModuleOp>> CreateLowerStaticTensorListPass();

// Creates an instance of the TensorFlow Lite dialect Quantize pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateQuantizePass();

// Creates an instance of the TensorFlow Lite dialect PrepareQuantize pass.
std::unique_ptr<OpPassBase<FuncOp>> CreatePrepareQuantizePass(
    const QuantizationSpecs& quant_specs);

// Creates an instance of the TensorFlow Lite dialect PostQuantize pass.
std::unique_ptr<OpPassBase<FuncOp>> CreatePostQuantizePass(
    bool emit_quant_adaptor_ops);

// Creates an instance of the TensorFlow Lite dialect TrimFunctions
// pass.
std::unique_ptr<OpPassBase<ModuleOp>> CreateTrimFunctionsPass(
    llvm::ArrayRef<std::string> trim_funcs_whitelist);

// Creates an instance of the TensorFlow Lite dialect PrepareCompositeFunctions
// pass.
std::unique_ptr<OpPassBase<ModuleOp>> CreatePrepareCompositeFunctionsPass();

// Creates an instance of the TensorFlow Lite dialect ExtractOphint pass.
std::unique_ptr<OpPassBase<ModuleOp>> CreateExtractOphintPass();

// Creates an instance of the TensorFlow Lite dialect LegalizeOphintFuncOpPass
// pass. The composite op is created from the ophint extraction pass.
std::unique_ptr<OpPassBase<ModuleOp>> CreateLegalizeOphintFuncOpPass();

// Creates an instance of the TensorFlow Lite dialect SplitMergedOperandsPass.
std::unique_ptr<OpPassBase<FuncOp>> CreateSplitMergedOperandsPass();

// Creates an instance of the TensorFlow Lite dialect OptimizeFunctionalOpsPass.
std::unique_ptr<OpPassBase<ModuleOp>> CreateOptimizeFunctionalOpsPass();

// Creates an instance of the TensorFlow Lite dialect pass to add default
// quantization parameters.
std::unique_ptr<OpPassBase<FuncOp>> CreateDefaultQuantParamsPass(
    double default_min, double default_max);

// Creates an instance of the TensorFlow Lite dialect pass to convert dense
// tensor to sparse format.
std::unique_ptr<OpPassBase<FuncOp>> CreateDenseToSparsePass();

// Creates function pass to legalize TF While to TFL While.
std::unique_ptr<OpPassBase<FuncOp>> CreateLegalizeTFWhilePass();

// Creates an instance of the TensorFlow Lite dialect WhileOp outline pass.
std::unique_ptr<OpPassBase<ModuleOp>> CreateWhileOutlinePass();

// Verifies runtime supports types used.
std::unique_ptr<OpPassBase<FuncOp>> CreateRuntimeTypeVerifyPass();

}  // namespace TFL

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_

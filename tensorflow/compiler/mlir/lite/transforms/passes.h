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
class OperationPass;

namespace TFL {
class QuantizationSpecs;

// Creates an instance of the TensorFlow Lite dialect LegalizeTF pass.
// When the given run_tfl_runtime_verification value is true, it will check each
// TFL builtin op towards the TFL runtime capability and the incompatible TF ops
// will be left in the graph without getting legalized.
std::unique_ptr<OperationPass<FuncOp>> CreateLegalizeTFPass(
    bool run_tfl_runtime_verification);

// Creates an instance of the TensorFlow Lite dialect Optimize pass.
std::unique_ptr<OperationPass<FuncOp>> CreateOptimizePass();

// Creates an instance of the TensorFlow Lite dialect PrepareTF pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePrepareTFPass(
    bool unfold_batch_matmul);

// Creates an instance of the TensorFlow Lite dialect LowerStaticTensorList
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateLowerStaticTensorListPass();

// Creates an instance of the TensorFlow Lite dialect Quantize pass.
std::unique_ptr<OperationPass<FuncOp>> CreateQuantizePass();

// Creates an instance of the TensorFlow Lite dialect PrepareQuantize pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePrepareQuantizePass(
    const QuantizationSpecs& quant_specs);

// Creates an instance of the TensorFlow Lite dialect PostQuantize pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePostQuantizePass(
    bool emit_quant_adaptor_ops);

// Creates an instance of the TensorFlow Lite dialect TrimFunctions
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateTrimFunctionsPass(
    llvm::ArrayRef<std::string> trim_funcs_whitelist);

// Creates an instance of the TensorFlow Lite dialect PrepareCompositeFunctions
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreatePrepareCompositeFunctionsPass();

// Creates an instance of the TensorFlow Lite dialect SplitMergedOperandsPass.
std::unique_ptr<OperationPass<FuncOp>> CreateSplitMergedOperandsPass();

// Creates an instance of the TensorFlow Lite dialect OptimizeFunctionalOpsPass.
std::unique_ptr<OperationPass<ModuleOp>> CreateOptimizeFunctionalOpsPass();

// Creates an instance of the TensorFlow Lite dialect pass to add default
// quantization parameters.
std::unique_ptr<OperationPass<FuncOp>> CreateDefaultQuantParamsPass(
    double default_min, double default_max, bool is_signed);

// Creates an instance of the TensorFlow Lite dialect pass to convert dense
// tensor to sparse format.
std::unique_ptr<OperationPass<FuncOp>> CreateDenseToSparsePass();

// Creates function pass to legalize TF While to TFL While.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeTFWhilePass();

// Creates an instance of the TensorFlow Lite dialect WhileOp outline pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateWhileOutlinePass();

// Verifies runtime constraints.
std::unique_ptr<OperationPass<FuncOp>> CreateRuntimeVerifyPass();

}  // namespace TFL

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_

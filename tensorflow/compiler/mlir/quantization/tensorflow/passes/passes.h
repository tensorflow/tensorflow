/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/util.h"

namespace mlir {
namespace quant {

// Creates a main function if it doesn't exist in the module. This is a
// workaround to make ConvertMlirToGraphdef work for multi-signatures graphs.
// TODO(b/204265523): Removes this pass after the exporting MLIR to SavedModel
// path is available.
std::unique_ptr<OperationPass<ModuleOp>> CreateInsertMainFunctionPass();

// Converts FakeQuant ops to quant.qcast and quant.dcast (QDQ) pairs.
std::unique_ptr<OperationPass<func::FuncOp>> CreateConvertFakeQuantToQdqPass();

// Lifts the quantizable spots as composite functions.
std::unique_ptr<OperationPass<ModuleOp>>
CreateLiftQuantizableSpotsAsFunctionsPass();

// Apply graph optimizations such as fusing and constant folding to prepare
// lifting.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareLiftingPass();

// Replaces tf.CustomAggregator ops with quant.Stats ops for finalizing the
// calibration procedure.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertCustomAggregationOpToQuantStatsPass();

// Issues IDs of custom aggregation ops for preparing the calibration procedure.
std::unique_ptr<OperationPass<ModuleOp>>
CreateIssueIDsOfCustomAggregationOpsPass();

// Inserts quantized function library.
std::unique_ptr<OperationPass<ModuleOp>> CreateInsertQuantizedFunctionsPass(
    QuantizationMethod quantization_method);

// Inserts custom aggregation operators for the calibration procedure.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateInsertCustomAggregationOpsPass();

// Replaces composite functions with quantized composite functions. After this
// pass runs, functions in the given graph will be replaced with their quantized
// versions. By doing so, the quantization will be applied to the given input.
std::unique_ptr<OperationPass<ModuleOp>> CreateQuantizeCompositeFunctionsPass(
    QuantizationMethod quantization_method);

// Converts dequantize-(quantizable) call-quantize pattern to a single call op
// that has quantized input and output types. It is expected for this pass to
// emit illegal IR with unsupported quantized input and output types. The
// pass following immediately after this one will be responsible for legalizing
// input and output types by unwrapping quantization parameters.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass();

// Creates an instance of the PrepareQuantize pass, which will perfrom similar
// transformations as TFL::PrepareQuantizePass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizePass(
    QuantizationMethod quantization_method);

// Creates an instance of the PostQuantize pass, which will remove unnecessary
// ops from the final quantized graph.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizePass();

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_PASSES_H_

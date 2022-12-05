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
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"

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
CreateLiftQuantizableSpotsAsFunctionsPass(OpSet target_opset);

// Apply graph optimizations such as fusing and constant folding to prepare
// lifting.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareLiftingPass();

// Lifts the dynamic range quantizable spots as composite functions.
std::unique_ptr<OperationPass<ModuleOp>>
CreateLiftQuantizableSpotsAsFunctionsDRQPass(int min_num_elements_for_weights);

// Replaces tf.CustomAggregator ops with quant.Stats ops for finalizing the
// calibration procedure.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertCustomAggregationOpToQuantStatsPass();

// Issues IDs of custom aggregation ops for preparing the calibration procedure.
std::unique_ptr<OperationPass<ModuleOp>>
CreateIssueIDsOfCustomAggregationOpsPass();

// Inserts quantized function library.
std::unique_ptr<OperationPass<ModuleOp>> CreateInsertQuantizedFunctionsPass(
    QuantizationMethod quantization_method, OpSet target_opset);

// Inserts custom aggregation operators for the calibration procedure.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateInsertCustomAggregationOpsPass();

// Replaces composite functions with quantized composite functions. After this
// pass runs, functions in the given graph will be replaced with their quantized
// versions. By doing so, the quantization will be applied to the given input.
std::unique_ptr<OperationPass<ModuleOp>> CreateQuantizeCompositeFunctionsPass(
    QuantizationMethod quantization_method, OpSet target_opset,
    bool enable_per_channel_quantization);

// Converts dequantize-(quantizable) call-quantize pattern to a single call op
// that has quantized input and output types. It is expected for this pass to
// emit illegal IR with unsupported quantized input and output types. The
// pass following immediately after this one will be responsible for legalizing
// input and output types by unwrapping quantization parameters.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass();

// Overloading of CreateQuantizePass which takes QuantizationSpecs.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass(
    QuantizationSpecs quant_specs, OpSet target_opset);

// Creates an instance of the PrepareQuantize pass, which will perfrom similar
// transformations as TFL::PrepareQuantizePass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizePass(
    QuantizationMethod quantization_method);

// Creates an instance of the PrepareQuantizeDRQ pass, which will
// perfrom similar transformations as TFL::PrepareQuantizeDynamicRangePass.
std::unique_ptr<OperationPass<ModuleOp>> CreatePrepareQuantizeDRQPass(
    const QuantizationSpecs& quant_specs, OpSet op_set);

// Creates an instance of the PostQuantize pass, which will remove unnecessary
// ops from the final quantized graph.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizePass();

// Creates an instance of the ConvertTFQuantOpsToMHLOPass pass, which will
// convert TF uniform quantized ops to the corresponding quantized MHLO ops.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertTFQuantOpsToMHLOPass();

// Applies optimization patterns after quantization.
std::unique_ptr<OperationPass<mlir::func::FuncOp>> CreateOptimizePass();

// Creates an instance of the ReplaceCastHacksWithTFXLAOpsPass, which will
// replace mixed-type convolution and matmul cast hacks by XLA Conv2DOp and
// MatmulOp.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateReplaceCastHacksWithTFXLAOpsPass();

// Creates a pass that moves & merges initializer function's ops into the @main
// function. This pass should be run on a valid tf_executor dialect. The control
// output of the initializer function for non-variable resource initialization
// will be passed on as a dependency to a new `tf.NoOp`, whose control output
// will be merged into the main function's FetchOp. The initializer functions
// will be removed.
std::unique_ptr<OperationPass<ModuleOp>>
CreateMergeInitializerFunctionOpsToMainPass();

// Creates a pass that "unfreezes" ConstOps into variables. Each ConstOp's use
// will be replaced by a VarHandleOp -> ReadVariableOp pattern. The newly
// created variables will be initialized in the session initializer function via
// AssignVariableOps.
std::unique_ptr<OperationPass<ModuleOp>> CreateUnfreezeConstantsPass();

// Creates a plass that duplicates constants that affect the shape of a tensor
// after some computation.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateDuplicateShapeDeterminingConstantsPass();

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_PASSES_H_

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace odml {

// Creates a pass which unfuses MHLO batch norm inference op into arithmetic
// ops.
std::unique_ptr<Pass> createUnfuseBatchNormPass();

// Creates a pass which constant folds broadcast_in_dim op conditionally.
std::unique_ptr<Pass> createFoldBroadcastPass();

// Creates a pass which fuses MHLO binary element-wise ops and convolution op.
std::unique_ptr<Pass> createFuseConvolutionPass();

// Creates a pass which applies various optimizations on MHLO IR.
std::unique_ptr<Pass> createOptimizePass();

// Creates a pass that finds quantization patterns and compose them to uniform
// quantized types.
std::unique_ptr<OperationPass<ModuleOp>>
CreateComposeUniformQuantizedTypePass();

// Creates a pass that finds stablehlo ops that accept or produce uniform
// quantized typed tensors and converts them to equivalent ops in the TFLite
// dialect.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateUniformQuantizedStableHloToTflPass();

// Create a pass that commute transposes through specific ops
std::unique_ptr<OperationPass<ModuleOp>> CreateTransposeCommuteOpsPass();

// Create a pass that legalizes MHLO to TF dialect.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeHloToTfPass();

// Creates a pass which replaces a splat constant tensor with a BroadcastInDim
// op.
std::unique_ptr<OperationPass<ModuleOp>> CreateUnfoldSplatConstantPass();

// Create a pass that legalizes MHLO to TFLite dialect.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeHloToTfLitePass();

// Creates a pass that lowers stablehlo composite ops to tflite ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateCompositeLoweringPass();

// Adds the HLO to TF rewrite patterns to the specified pattern list.
void PopulateLegalizeHloToTfPatterns(RewritePatternSet* patterns,
                                     MLIRContext* context);

// Adds the HLO to TFLite rewrite patterns to the specified pattern list.
void PopulateLegalizeHloToTFLitePatterns(RewritePatternSet* patterns,
                                         MLIRContext* context);

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h.inc"

}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_PASSES_H_

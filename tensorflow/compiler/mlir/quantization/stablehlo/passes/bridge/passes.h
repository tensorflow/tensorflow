/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_BRIDGE_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_BRIDGE_PASSES_H_

#include <memory>

#define GEN_PASS_DECL
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir::quant::stablehlo {

// Legalizes from MHLO quantized ops with MHLO quant types to MHLO primitive ops
// like int ops.
std::unique_ptr<OperationPass<func::FuncOp>> createConvertMHLOQuantToIntPass(
    bool legalize_chlo = true);

// Creates an instance of the ConvertTFQuantOpsToMHLOPass pass, which will
// convert TF uniform quantized ops to the corresponding quantized MHLO ops.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertTFQuantOpsToMHLOPass();

// TODO(b/288094093): Migrate uniform quantization legalization in a separate
// pass.
void PopulateLegalizeTfQuantizationPatterns(MLIRContext *context,
                                            RewritePatternSet *patterns);

// Creates an instance of the ConvertTFQuantTypes pass, which will convert TF
// qint types to int types and surround TF UniformQuantized ops with qint <->
// int casts.
std::unique_ptr<OperationPass<func::FuncOp>> CreateConvertTFQuantTypesPass();

// Creates an instance of the VerifyQuantLegalization pass, which verifies all
// quant ops and types are lowered.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateVerifyQuantLegalizationPass();

// Add all passes for lowering TF quant ops and types to MHLO int.
void AddQuantizationLoweringPasses(mlir::OpPassManager &pm);

// Creates an instance of OptimizeIntGraphPass, which optimizes the int graph
// lowered from the quantized graph.
std::unique_ptr<OperationPass<func::FuncOp>> CreateOptimizeIntGraphPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_CONVERTMHLOQUANTTOINT
#define GEN_PASS_DECL_CONVERTTFQUANTOPSTOMHLO
#define GEN_PASS_DECL_CONVERTTFQUANTTYPES
#define GEN_PASS_DECL_VERIFYQUANTLEGALIZATION
#define GEN_PASS_DECL_OPTIMIZEINTGRAPH
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h.inc"
}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_PASSES_BRIDGE_PASSES_H_

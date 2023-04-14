/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_PASSES_H_

#include <memory>
#include <optional>
#include <string>
#include <unordered_set>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {

namespace quant {
class QuantizationDialect;
}

namespace quantfork {
class QuantizationForkDialect;
}

namespace TFL {
class TFLDialect;
}

namespace tosa {
class TosaDialect;

void populateLegalizeTFPatterns(MLIRContext* ctx, RewritePatternSet& patterns);
void populateLegalizeTFLPatterns(MLIRContext* ctx, RewritePatternSet& patterns);

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTFPass();
std::unique_ptr<OperationPass<func::FuncOp>> createFuseBiasTFPass();

// `disabledPatterns` is a set of labels used to filter out input patterns with
// a debug label or debug name in this set.
// `enabledPatterns` is a set of labels used to filter out input patterns that
//  do not have one of the labels in this set.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTFLPass(
    ArrayRef<std::string> disabled_patterns = std::nullopt,
    ArrayRef<std::string> enabled_patterns = std::nullopt);

std::unique_ptr<OperationPass<ModuleOp>> createLowerGlobalTensorsPass();
std::unique_ptr<OperationPass<ModuleOp>> createRetainCallOnceFuncsPass();
std::unique_ptr<OperationPass<ModuleOp>> createStripModuleMetadataPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTFLUint8Pass();
std::unique_ptr<OperationPass<func::FuncOp>> createDequantizeTFLSoftmaxPass();
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTFTFLPass();
std::unique_ptr<OperationPass<func::FuncOp>> createLowerComplexTypesPass();
std::unique_ptr<OperationPass<func::FuncOp>> createStripFunctionMetadataPass();
std::unique_ptr<OperationPass<func::FuncOp>> createStripQuantTypesPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#define GEN_PASS_DECL_TOSALEGALIZETFPASS
#define GEN_PASS_DECL_TOSALEGALIZETFLPASS
#define GEN_PASS_DECL_TOSALEGALIZETFTFLPASS
#define GEN_PASS_DECL_TOSAFUSEBIASTFPASS
#define GEN_PASS_DECL_TOSACONVERTTFLUINT8PASS
#define GEN_PASS_DECL_TOSASTRIPQUANTTYPESPASS
#define GEN_PASS_DECL_TOSALOWERCOMPLEXTYPESPASS
#define GEN_PASS_DECL_TOSADEQUANTIZETFLSOFTMAXPASS
#define GEN_PASS_DECL_LOWERGLOBALTENSORS
#define GEN_PASS_DECL_RETAINCALLONCEFUNCS
#define GEN_PASS_DECL_STRIPFUNCTIONMETADATA
#define GEN_PASS_DECL_STRIPMODULEMETADATA

#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

}  // namespace tosa
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_PASSES_H_

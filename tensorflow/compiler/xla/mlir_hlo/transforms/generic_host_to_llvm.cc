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
#include <memory>
#include <utility>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "transforms/passes.h"

namespace mlir {
namespace hlo {

using func::FuncOp;

void createGenericHostToLLVMPipeline(OpPassManager& pm) {
  // Convert all linalg operations to parallel loops.
  pm.addNestedPass<FuncOp>(createConvertLinalgToParallelLoopsPass());
  // Canonicalize generated scf.parallel operations to remove single iterations.
  pm.addPass(createCanonicalizerPass());

  // Expand math operations into std/arith dialect operations.
  pm.addNestedPass<FuncOp>(arith::createArithExpandOpsPass());
  pm.addNestedPass<FuncOp>(memref::createExpandOpsPass());
  pm.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());

  pm.addPass(createConvertLinalgToLLVMPass());
  pm.addPass(createConvertSCFToCFPass());

  ConvertMathToLLVMPassOptions mathOpts;
  mathOpts.approximateLog1p = false;
  pm.addPass(createConvertMathToLLVMPass(mathOpts));
  pm.addPass(createConvertMathToLibmPass());

  // Convert everything else to LLVM dialect.
  ConvertVectorToLLVMPassOptions vectorOpts;
  pm.addPass(createConvertVectorToLLVMPass(vectorOpts));
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertComplexToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Prepare module for translation to LLVM.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

}  // namespace hlo
}  // namespace mlir

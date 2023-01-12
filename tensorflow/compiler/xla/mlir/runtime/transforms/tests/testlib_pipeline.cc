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

#include "tensorflow/compiler/xla/mlir/runtime/transforms/tests/testlib_pipeline.h"

#include <utility>

#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"  // from @llvm-project
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Async/IR/Async.h"  // from @llvm-project
#include "mlir/Dialect/Async/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/runtime/transforms/compiler.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/passes.h"

namespace xla {
namespace runtime {

void RegisterXlaRuntimeTestlibDialects(DialectRegistry& dialects) {
  // Register MLIR dialects supported by the Xla runtime tests.
  dialects->insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                   mlir::scf::SCFDialect, mlir::func::FuncDialect,
                   mlir::memref::MemRefDialect, RuntimeDialect>();

  // Register MLIR dialects that can be translated to LLVM IR.
  registerLLVMDialectTranslation(*dialects);
}

void CreateXlaRuntimeTestlibPipeline(PassManager& passes) {
  passes->addPass(mlir::createConvertSCFToCFPass());
  passes->addPass(mlir::createAsyncFuncToAsyncRuntimePass());

  // Export functions to the XLA runtime.
  passes->addPass(CreateExportRuntimeFunctionsPass());
  passes->addPass(CreateConvertAssertsPass());

  // Lower from high level async operations to async runtime.
  passes->addPass(mlir::createAsyncToAsyncRuntimePass());

  // Add async.runtime reference counting operations.
  passes->addPass(mlir::createAsyncRuntimePolicyBasedRefCountingPass());

  // Convert runtime operations and custom calls to LLVM dialect.
  ConvertRuntimeToLLvmOpts rt_to_llvm_opts;
  passes->addPass(CreateConvertRuntimeToLLVMPass(std::move(rt_to_llvm_opts)));

  // Convert async runtime operations to LLVM dialect.
  passes->addPass(mlir::createConvertAsyncToLLVMPass());

  // Convert everything else to LLVM dialect.
  passes->addPass(mlir::createMemRefToLLVMConversionPass());
  passes->addPass(mlir::createConvertFuncToLLVMPass());
  passes->addPass(mlir::createReconcileUnrealizedCastsPass());

  // Clean up IR before translating it to LLVM.
  passes->addPass(mlir::createCSEPass());
  passes->addPass(mlir::createCanonicalizerPass());
}

}  // namespace runtime
}  // namespace xla

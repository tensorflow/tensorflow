//===- Common.h - Linalg  dialect RangeOp operation -----------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef LINALG_COMMON_H_
#define LINALG_COMMON_H_

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

namespace linalg {
namespace common {

////////////////////////////////////////////////////////////////////////////////
// Define a few boilerplate objects used across all linalg examples.
////////////////////////////////////////////////////////////////////////////////

// The unique MLIRContext, similar to an llvm::Context.
inline mlir::MLIRContext &globalContext() {
  static mlir::MLIRContext context;
  return context;
}

// The unique Module, similar to an llvm::Module.
inline mlir::Module &globalModule() {
  static mlir::Module module(&globalContext());
  return module;
}

/// Shortcut notation for types that we use globally.
/// The index type is the type that must be used with affine operations:
///   (`affine.apply`, `affine.for`, `affine.load`, `affine.store`).
inline mlir::IndexType indexType() {
  return mlir::IndexType::get(&globalContext());
}

/// Common f32 type.
inline mlir::FloatType f32Type() {
  return mlir::FloatType::getF32(&globalContext());
}

/// A 2-D abstraction over a flat contiguous memory region of f32 with symbolic
/// sizes.
template <int N>
inline mlir::MemRefType floatMemRefType(unsigned memorySpace = 0) {
  llvm::SmallVector<int64_t, 4> shape(N, -1);
  return mlir::MemRefType::get(shape, f32Type(), {}, memorySpace);
}

/// The simple function, taking 4 parameters of type index, that we will use
/// throughout this tutorial:
///
/// ```mlir
///    func @name(%M: index, %N: index, %K: index, %P: index)
/// ```
inline mlir::Function *makeFunction(llvm::StringRef name,
                                    llvm::ArrayRef<mlir::Type> resultTypes) {
  auto &ctx = globalContext();
  auto *function =
      new mlir::Function(mlir::UnknownLoc::get(&ctx), name,
                         mlir::FunctionType::get({indexType(), indexType(),
                                                  indexType(), indexType()},
                                                 resultTypes, &ctx));
  function->addEntryBlock();
  globalModule().getFunctions().push_back(function);
  return function;
}

/// A basic pass manager pre-populated with cleanup passes.
inline mlir::PassManager &cleanupPassManager() {
  static bool inited = false;
  static mlir::PassManager pm;
  if (!inited) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSimplifyAffineStructuresPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    inited = true;
  }
  return pm;
}

/// A simple function to verify and cleanup the IR before printing it to
/// llvm::outs() for FileCheck'ing.
/// If an error occurs, dump to llvm::errs() and do not print to llvm::outs()
/// which will make the associated FileCheck test fail.
inline void cleanupAndPrintFunction(mlir::Function *f) {
  bool printToOuts = true;
  auto check = [f, &printToOuts](mlir::LogicalResult result) {
    if (failed(result)) {
      f->dump();
      llvm::errs() << "Failure!\n";
      printToOuts = false;
    }
  };
  check(f->getModule()->verify());
  check(cleanupPassManager().run(f->getModule()));
  if (printToOuts)
    f->print(llvm::outs());
}

/// Helper class to sugar building loop nests from indexings that appear in
/// ViewOp and SliceOp.
class LoopNestRangeBuilder {
public:
  LoopNestRangeBuilder(llvm::ArrayRef<mlir::edsc::ValueHandle *> ivs,
                       llvm::ArrayRef<mlir::edsc::ValueHandle> indexings);
  LoopNestRangeBuilder(llvm::ArrayRef<mlir::edsc::ValueHandle *> ivs,
                       llvm::ArrayRef<mlir::Value *> indexings);
  mlir::edsc::ValueHandle
  operator()(llvm::ArrayRef<mlir::edsc::CapturableHandle> stmts);

private:
  llvm::SmallVector<mlir::edsc::LoopBuilder, 4> loops;
};

} // namespace common
} // namespace linalg

#endif // LINALG_COMMON_H_

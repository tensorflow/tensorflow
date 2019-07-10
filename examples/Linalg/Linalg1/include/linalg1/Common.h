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

#ifndef LINALG1_COMMON_H_
#define LINALG1_COMMON_H_

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Verifier.h"
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

/// A 2-D abstraction over a flat contiguous memory region of f32 with symbolic
/// sizes.
template <int N>
inline mlir::MemRefType floatMemRefType(mlir::MLIRContext *context,
                                        unsigned memorySpace = 0) {
  llvm::SmallVector<int64_t, 4> shape(N, -1);
  auto f32 = mlir::FloatType::getF32(context);
  return mlir::MemRefType::get(shape, f32, {}, memorySpace);
}

/// A basic function builder
inline mlir::FuncOp makeFunction(mlir::ModuleOp module, llvm::StringRef name,
                                 llvm::ArrayRef<mlir::Type> types,
                                 llvm::ArrayRef<mlir::Type> resultTypes) {
  auto *context = module.getContext();
  auto function = mlir::FuncOp::create(
      mlir::UnknownLoc::get(context), name,
      mlir::FunctionType::get({types}, resultTypes, context));
  function.addEntryBlock();
  module.push_back(function);
  return function;
}

/// A basic pass manager pre-populated with cleanup passes.
inline std::unique_ptr<mlir::PassManager> cleanupPassManager() {
  std::unique_ptr<mlir::PassManager> pm(new mlir::PassManager());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createSimplifyAffineStructuresPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createCanonicalizerPass());
  return pm;
}

/// A simple function to verify and cleanup the IR before printing it to
/// llvm::outs() for FileCheck'ing.
/// If an error occurs, dump to llvm::errs() and do not print to llvm::outs()
/// which will make the associated FileCheck test fail.
inline void cleanupAndPrintFunction(mlir::FuncOp f) {
  bool printToOuts = true;
  auto check = [&f, &printToOuts](mlir::LogicalResult result) {
    if (failed(result)) {
      f.emitError("Verification and cleanup passes failed");
      printToOuts = false;
    }
  };
  auto pm = cleanupPassManager();
  check(mlir::verify(f.getParentOfType<mlir::ModuleOp>()));
  check(pm->run(f.getParentOfType<mlir::ModuleOp>()));
  if (printToOuts)
    f.print(llvm::outs());
}

/// Helper class to sugar building loop nests from indexings that appear in
/// ViewOp and SliceOp.
class LoopNestRangeBuilder {
public:
  LoopNestRangeBuilder(llvm::ArrayRef<mlir::edsc::ValueHandle *> ivs,
                       llvm::ArrayRef<mlir::edsc::ValueHandle> indexings);
  LoopNestRangeBuilder(llvm::ArrayRef<mlir::edsc::ValueHandle *> ivs,
                       llvm::ArrayRef<mlir::Value *> indexings);
  mlir::edsc::ValueHandle operator()(std::function<void(void)> fun = nullptr);

private:
  llvm::SmallVector<mlir::edsc::LoopBuilder, 4> loops;
};

} // namespace common
} // namespace linalg

#endif // LINALG1_COMMON_H_

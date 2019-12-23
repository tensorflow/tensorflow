//===- OptUtils.cpp - MLIR Execution Engine optimization pass utilities ---===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the utility functions to trigger LLVM optimizations from
// MLIR Execution Engine.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/OptUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <climits>
#include <mutex>

// Run the module and function passes managed by the module manager.
static void runPasses(llvm::legacy::PassManager &modulePM,
                      llvm::legacy::FunctionPassManager &funcPM,
                      llvm::Module &m) {
  funcPM.doInitialization();
  for (auto &func : m) {
    funcPM.run(func);
  }
  funcPM.doFinalization();
  modulePM.run(m);
}

// Initialize basic LLVM transformation passes under lock.
void mlir::initializeLLVMPasses() {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);

  auto &registry = *llvm::PassRegistry::getPassRegistry();
  llvm::initializeCore(registry);
  llvm::initializeTransformUtils(registry);
  llvm::initializeScalarOpts(registry);
  llvm::initializeIPO(registry);
  llvm::initializeInstCombine(registry);
  llvm::initializeAggressiveInstCombine(registry);
  llvm::initializeAnalysis(registry);
  llvm::initializeVectorization(registry);
}

// Populate pass managers according to the optimization and size levels.
// This behaves similarly to LLVM opt.
static void populatePassManagers(llvm::legacy::PassManager &modulePM,
                                 llvm::legacy::FunctionPassManager &funcPM,
                                 unsigned optLevel, unsigned sizeLevel,
                                 llvm::TargetMachine *targetMachine) {
  llvm::PassManagerBuilder builder;
  builder.OptLevel = optLevel;
  builder.SizeLevel = sizeLevel;
  builder.Inliner = llvm::createFunctionInliningPass(
      optLevel, sizeLevel, /*DisableInlineHotCallSite=*/false);
  builder.LoopVectorize = optLevel > 1 && sizeLevel < 2;
  builder.SLPVectorize = optLevel > 1 && sizeLevel < 2;
  builder.DisableUnrollLoops = (optLevel == 0);

  if (targetMachine) {
    // Add pass to initialize TTI for this specific target. Otherwise, TTI will
    // be initialized to NoTTIImpl by default.
    modulePM.add(createTargetTransformInfoWrapperPass(
        targetMachine->getTargetIRAnalysis()));
    funcPM.add(createTargetTransformInfoWrapperPass(
        targetMachine->getTargetIRAnalysis()));
  }

  builder.populateModulePassManager(modulePM);
  builder.populateFunctionPassManager(funcPM);
}

// Create and return a lambda that uses LLVM pass manager builder to set up
// optimizations based on the given level.
std::function<llvm::Error(llvm::Module *)>
mlir::makeOptimizingTransformer(unsigned optLevel, unsigned sizeLevel,
                                llvm::TargetMachine *targetMachine) {
  return [optLevel, sizeLevel, targetMachine](llvm::Module *m) -> llvm::Error {
    llvm::legacy::PassManager modulePM;
    llvm::legacy::FunctionPassManager funcPM(m);
    populatePassManagers(modulePM, funcPM, optLevel, sizeLevel, targetMachine);
    runPasses(modulePM, funcPM, *m);

    return llvm::Error::success();
  };
}

// Create and return a lambda that is given a set of passes to run, plus an
// optional optimization level to pre-populate the pass manager.
std::function<llvm::Error(llvm::Module *)> mlir::makeLLVMPassesTransformer(
    llvm::ArrayRef<const llvm::PassInfo *> llvmPasses,
    llvm::Optional<unsigned> mbOptLevel, llvm::TargetMachine *targetMachine,
    unsigned optPassesInsertPos) {
  return [llvmPasses, mbOptLevel, optPassesInsertPos,
          targetMachine](llvm::Module *m) -> llvm::Error {
    llvm::legacy::PassManager modulePM;
    llvm::legacy::FunctionPassManager funcPM(m);

    bool insertOptPasses = mbOptLevel.hasValue();
    for (unsigned i = 0, e = llvmPasses.size(); i < e; ++i) {
      const auto *passInfo = llvmPasses[i];
      if (!passInfo->getNormalCtor())
        continue;

      if (insertOptPasses && optPassesInsertPos == i) {
        populatePassManagers(modulePM, funcPM, mbOptLevel.getValue(), 0,
                             targetMachine);
        insertOptPasses = false;
      }

      auto *pass = passInfo->createPass();
      if (!pass)
        return llvm::make_error<llvm::StringError>(
            "could not create pass " + passInfo->getPassName(),
            llvm::inconvertibleErrorCode());
      modulePM.add(pass);
    }

    if (insertOptPasses)
      populatePassManagers(modulePM, funcPM, mbOptLevel.getValue(), 0,
                           targetMachine);

    runPasses(modulePM, funcPM, *m);
    return llvm::Error::success();
  };
}

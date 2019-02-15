//===- OptUtils.cpp - MLIR Execution Engine optimization pass utilities ---===//
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
//
// This file implements the utility functions to trigger LLVM optimizations from
// MLIR Execution Engine.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/OptUtils.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <mutex>

// Run the module and function passes managed by the module manager.
static void runPasses(llvm::legacy::PassManager &modulePM,
                      llvm::legacy::FunctionPassManager &funcPM,
                      llvm::Module &m) {
  for (auto &func : m) {
    funcPM.run(func);
  }
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

// Create and return a lambda that uses LLVM pass manager builder to set up
// optimizations based on the given level.
std::function<llvm::Error(llvm::Module *)>
mlir::makeOptimizingTransformer(unsigned optLevel, unsigned sizeLevel) {
  return [optLevel, sizeLevel](llvm::Module *m) -> llvm::Error {
    llvm::PassManagerBuilder builder;
    builder.OptLevel = optLevel;
    builder.SizeLevel = sizeLevel;
    builder.Inliner = llvm::createFunctionInliningPass(
        optLevel, sizeLevel, /*DisableInlineHotCallSite=*/false);

    llvm::legacy::PassManager modulePM;
    llvm::legacy::FunctionPassManager funcPM(m);
    builder.populateModulePassManager(modulePM);
    builder.populateFunctionPassManager(funcPM);
    runPasses(modulePM, funcPM, *m);

    return llvm::Error::success();
  };
}

// Create and return a lambda that leverages LLVM PassInfo command line parser
// to construct passes given the command line flags that come from the given
// string rather than from the command line.
std::function<llvm::Error(llvm::Module *)>
mlir::makeLLVMPassesTransformer(std::string config) {
  return [config](llvm::Module *m) -> llvm::Error {
    static llvm::cl::list<const llvm::PassInfo *, bool, llvm::PassNameParser>
        llvmPasses(llvm::cl::desc("LLVM optimizing passes to run"));
    llvm::BumpPtrAllocator allocator;
    llvm::StringSaver saver(allocator);
    llvm::SmallVector<const char *, 16> args;
    args.push_back(""); // inject dummy program name
    llvm::cl::TokenizeGNUCommandLine(config, saver, args);
    llvm::cl::ParseCommandLineOptions(args.size(), args.data());

    llvm::legacy::PassManager modulePM;

    for (const auto *passInfo : llvmPasses) {
      if (!passInfo->getNormalCtor())
        continue;

      auto *pass = passInfo->createPass();
      if (!pass)
        return llvm::make_error<llvm::StringError>(
            "could not create pass " + passInfo->getPassName(),
            llvm::inconvertibleErrorCode());

      modulePM.add(pass);
    }

    modulePM.run(*m);
    return llvm::Error::success();
  };
}

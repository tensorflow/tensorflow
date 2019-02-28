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
#include <climits>
#include <mutex>

// A category for options that should be passed to -llvm-opts.
static llvm::cl::OptionCategory
    optFlags("opt-like flags (pass to -llvm-ops=\"\")");

// LLVM pass configuration CLI flag.
static llvm::cl::list<const llvm::PassInfo *, bool, llvm::PassNameParser>
    llvmPasses(llvm::cl::desc("LLVM optimizing passes to run"),
               llvm::cl::cat(optFlags));

// CLI variables for -On options.
static llvm::cl::opt<bool> optO0("O0", llvm::cl::desc("Run opt O0 passes"),
                                 llvm::cl::cat(optFlags));
static llvm::cl::opt<bool> optO1("O1", llvm::cl::desc("Run opt O1 passes"),
                                 llvm::cl::cat(optFlags));
static llvm::cl::opt<bool> optO2("O2", llvm::cl::desc("Run opt O2 passes"),
                                 llvm::cl::cat(optFlags));
static llvm::cl::opt<bool> optO3("O3", llvm::cl::desc("Run opt O3 passes"),
                                 llvm::cl::cat(optFlags));

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
                                 unsigned optLevel, unsigned sizeLevel) {
  llvm::PassManagerBuilder builder;
  builder.OptLevel = optLevel;
  builder.SizeLevel = sizeLevel;
  builder.Inliner = llvm::createFunctionInliningPass(
      optLevel, sizeLevel, /*DisableInlineHotCallSite=*/false);
  builder.LoopVectorize = optLevel > 1 && sizeLevel < 2;
  builder.SLPVectorize = optLevel > 1 && sizeLevel < 2;
  builder.DisableUnrollLoops = (optLevel == 0);

  builder.populateModulePassManager(modulePM);
  builder.populateFunctionPassManager(funcPM);
}

// Create and return a lambda that uses LLVM pass manager builder to set up
// optimizations based on the given level.
std::function<llvm::Error(llvm::Module *)>
mlir::makeOptimizingTransformer(unsigned optLevel, unsigned sizeLevel) {
  return [optLevel, sizeLevel](llvm::Module *m) -> llvm::Error {

    llvm::legacy::PassManager modulePM;
    llvm::legacy::FunctionPassManager funcPM(m);
    populatePassManagers(modulePM, funcPM, optLevel, sizeLevel);
    runPasses(modulePM, funcPM, *m);

    return llvm::Error::success();
  };
}

// Check if the opt flag is set and if it was located before `pos`.  If so,
// popuplate the module and the function pass managers with the passes
// corresponding to the `level` of optimization and reset the flag.
static void
populatePassManagersOptLevel(llvm::cl::opt<bool> &opt, unsigned level,
                             unsigned pos, llvm::legacy::PassManager &modulePM,
                             llvm::legacy::FunctionPassManager &funcPM) {
  if (opt && opt.getPosition() < pos) {
    opt = false;
    populatePassManagers(modulePM, funcPM, level, /*sizeLevel=*/0);
  }
}

// Create and return a lambda that leverages LLVM PassInfo command line parser
// to construct passes given the command line flags that come from the given
// string rather than from the command line.
std::function<llvm::Error(llvm::Module *)>
mlir::makeLLVMPassesTransformer(std::string config) {
  return [config](llvm::Module *m) -> llvm::Error {
    // Storage for -On flags, the index in this array corresponds to the
    // optimization level.  Do not add anything else.
    llvm::SmallVector<std::reference_wrapper<llvm::cl::opt<bool>>, 4> optFlags{
        optO0, optO1, optO2, optO3};

    llvm::BumpPtrAllocator allocator;
    llvm::StringSaver saver(allocator);
    llvm::SmallVector<const char *, 16> args;
    args.push_back(""); // inject dummy program name
    llvm::cl::TokenizeGNUCommandLine(config, saver, args);
    llvm::cl::ParseCommandLineOptions(args.size(), args.data());

    llvm::legacy::PassManager modulePM;
    llvm::legacy::FunctionPassManager funcPM(m);

    for (unsigned i = 0, e = llvmPasses.size(); i < e; ++i) {
      const auto *passInfo = llvmPasses[i];
      if (!passInfo->getNormalCtor())
        continue;

      // If there is any of -On flags textually before this pass flag, populate
      // the pass managers with the corresponding passes and reset the flag.
      for (unsigned j = 0; j < 4; ++j)
        populatePassManagersOptLevel(
            optFlags[j].get(), j, llvmPasses.getPosition(i), modulePM, funcPM);

      auto *pass = passInfo->createPass();
      if (!pass)
        return llvm::make_error<llvm::StringError>(
            "could not create pass " + passInfo->getPassName(),
            llvm::inconvertibleErrorCode());

      modulePM.add(pass);
    }
    // Populate the pass managers with passes corresponding to the -On flags
    // that have not been used yet.  Use UINT_MAX as the position index before
    // which the -On flag should appear as an always-true marker.
    for (unsigned j = 0; j < 4; ++j)
      populatePassManagersOptLevel(optFlags[j].get(), j, /*pos=*/UINT_MAX,
                                   modulePM, funcPM);

    // Run the -On function passes, then all the other passes.  Note that
    // manually requested function passes were added to modulePM and will be
    // executed in order with manual/-On module passes.  The function pass
    // manager is only populated in reaction to -On flags with passes that are
    // supposed to run "as soon as functions are created" according to the doc.
    // This behavior is identical to that of LLVM's "opt" tool.
    runPasses(modulePM, funcPM, *m);
    return llvm::Error::success();
  };
}

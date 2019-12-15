//===- PassManagerOptions.cpp - PassManager Command Line Options ----------===//
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

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlir;

namespace {
struct PassManagerOptions {
  //===--------------------------------------------------------------------===//
  // Crash Reproducer Generator
  //===--------------------------------------------------------------------===//
  llvm::cl::opt<std::string> reproducerFile{
      "pass-pipeline-crash-reproducer",
      llvm::cl::desc("Generate a .mlir reproducer file at the given output path"
                     " if the pass manager crashes or fails")};

  //===--------------------------------------------------------------------===//
  // Multi-threading
  //===--------------------------------------------------------------------===//
  llvm::cl::opt<bool> disableThreads{
      "disable-pass-threading",
      llvm::cl::desc("Disable multithreading in the pass manager"),
      llvm::cl::init(false)};

  //===--------------------------------------------------------------------===//
  // IR Printing
  //===--------------------------------------------------------------------===//
  PassPipelineCLParser printBefore{"print-ir-before",
                                   "Print IR before specified passes"};
  PassPipelineCLParser printAfter{"print-ir-after",
                                  "Print IR after specified passes"};
  llvm::cl::opt<bool> printBeforeAll{
      "print-ir-before-all", llvm::cl::desc("Print IR before each pass"),
      llvm::cl::init(false)};
  llvm::cl::opt<bool> printAfterAll{"print-ir-after-all",
                                    llvm::cl::desc("Print IR after each pass"),
                                    llvm::cl::init(false)};
  llvm::cl::opt<bool> printAfterChange{
      "print-ir-after-change",
      llvm::cl::desc(
          "When printing the IR after a pass, only print if the IR changed"),
      llvm::cl::init(false)};
  llvm::cl::opt<bool> printModuleScope{
      "print-ir-module-scope",
      llvm::cl::desc("When printing IR for print-ir-[before|after]{-all} "
                     "always print the top-level module operation"),
      llvm::cl::init(false)};

  /// Add an IR printing instrumentation if enabled by any 'print-ir' flags.
  void addPrinterInstrumentation(PassManager &pm);

  //===--------------------------------------------------------------------===//
  // Pass Timing
  //===--------------------------------------------------------------------===//
  llvm::cl::opt<bool> passTiming{
      "pass-timing",
      llvm::cl::desc("Display the execution times of each pass")};
  llvm::cl::opt<PassDisplayMode> passTimingDisplayMode{
      "pass-timing-display",
      llvm::cl::desc("Display method for pass timing data"),
      llvm::cl::init(PassDisplayMode::Pipeline),
      llvm::cl::values(
          clEnumValN(PassDisplayMode::List, "list",
                     "display the results in a list sorted by total time"),
          clEnumValN(PassDisplayMode::Pipeline, "pipeline",
                     "display the results with a nested pipeline view"))};

  //===--------------------------------------------------------------------===//
  // Pass Statistics
  //===--------------------------------------------------------------------===//
  llvm::cl::opt<bool> passStatistics{
      "pass-statistics", llvm::cl::desc("Display the statistics of each pass")};
  llvm::cl::opt<PassDisplayMode> passStatisticsDisplayMode{
      "pass-statistics-display",
      llvm::cl::desc("Display method for pass statistics"),
      llvm::cl::init(PassDisplayMode::Pipeline),
      llvm::cl::values(
          clEnumValN(
              PassDisplayMode::List, "list",
              "display the results in a merged list sorted by pass name"),
          clEnumValN(PassDisplayMode::Pipeline, "pipeline",
                     "display the results with a nested pipeline view"))};

  /// Add a pass timing instrumentation if enabled by 'pass-timing' flags.
  void addTimingInstrumentation(PassManager &pm);
};
} // end anonymous namespace

static llvm::ManagedStatic<llvm::Optional<PassManagerOptions>> options;

/// Add an IR printing instrumentation if enabled by any 'print-ir' flags.
void PassManagerOptions::addPrinterInstrumentation(PassManager &pm) {
  std::function<bool(Pass *, Operation *)> shouldPrintBeforePass;
  std::function<bool(Pass *, Operation *)> shouldPrintAfterPass;

  // Handle print-before.
  if (printBeforeAll) {
    // If we are printing before all, then just return true for the filter.
    shouldPrintBeforePass = [](Pass *, Operation *) { return true; };
  } else if (printBefore.hasAnyOccurrences()) {
    // Otherwise if there are specific passes to print before, then check to see
    // if the pass info for the current pass is included in the list.
    shouldPrintBeforePass = [&](Pass *pass, Operation *) {
      auto *passInfo = pass->lookupPassInfo();
      return passInfo && printBefore.contains(passInfo);
    };
  }

  // Handle print-after.
  if (printAfterAll) {
    // If we are printing after all, then just return true for the filter.
    shouldPrintAfterPass = [](Pass *, Operation *) { return true; };
  } else if (printAfter.hasAnyOccurrences()) {
    // Otherwise if there are specific passes to print after, then check to see
    // if the pass info for the current pass is included in the list.
    shouldPrintAfterPass = [&](Pass *pass, Operation *) {
      auto *passInfo = pass->lookupPassInfo();
      return passInfo && printAfter.contains(passInfo);
    };
  }

  // If there are no valid printing filters, then just return.
  if (!shouldPrintBeforePass && !shouldPrintAfterPass)
    return;

  // Otherwise, add the IR printing instrumentation.
  pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                      printModuleScope, printAfterChange, llvm::errs());
}

/// Add a pass timing instrumentation if enabled by 'pass-timing' flags.
void PassManagerOptions::addTimingInstrumentation(PassManager &pm) {
  if (passTiming)
    pm.enableTiming(passTimingDisplayMode);
}

void mlir::registerPassManagerCLOptions() {
  // Reset the options instance if it hasn't been enabled yet.
  if (!options->hasValue())
    options->emplace();
}

void mlir::applyPassManagerCLOptions(PassManager &pm) {
  // Generate a reproducer on crash/failure.
  if ((*options)->reproducerFile.getNumOccurrences())
    pm.enableCrashReproducerGeneration((*options)->reproducerFile);

  // Disable multi-threading.
  if ((*options)->disableThreads)
    pm.disableMultithreading();

  // Enable statistics dumping.
  if ((*options)->passStatistics)
    pm.enableStatistics((*options)->passStatisticsDisplayMode);

  // Add the IR printing instrumentation.
  (*options)->addPrinterInstrumentation(pm);

  // Note: The pass timing instrumentation should be added last to avoid any
  // potential "ghost" timing from other instrumentations being unintentionally
  // included in the timing results.
  (*options)->addTimingInstrumentation(pm);
}

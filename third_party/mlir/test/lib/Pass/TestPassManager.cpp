//===- TestPassManager.cpp - Test pass manager functionality --------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace {
struct TestModulePass : public ModulePass<TestModulePass> {
  void runOnModule() final {}
};
struct TestFunctionPass : public FunctionPass<TestFunctionPass> {
  void runOnFunction() final {}
};
class TestOptionsPass : public FunctionPass<TestOptionsPass> {
public:
  struct Options : public PassPipelineOptions<Options> {
    ListOption<int> listOption{*this, "list",
                               llvm::cl::MiscFlags::CommaSeparated,
                               llvm::cl::desc("Example list option")};
    ListOption<std::string> stringListOption{
        *this, "string-list", llvm::cl::MiscFlags::CommaSeparated,
        llvm::cl::desc("Example string list option")};
    Option<std::string> stringOption{*this, "string",
                                     llvm::cl::desc("Example string option")};
  };
  TestOptionsPass() = default;
  TestOptionsPass(const TestOptionsPass &) {}
  TestOptionsPass(const Options &options) {
    listOption->assign(options.listOption.begin(), options.listOption.end());
    stringOption.setValue(options.stringOption);
    stringListOption->assign(options.stringListOption.begin(),
                             options.stringListOption.end());
  }

  void runOnFunction() final {}

  ListOption<int> listOption{*this, "list", llvm::cl::MiscFlags::CommaSeparated,
                             llvm::cl::desc("Example list option")};
  ListOption<std::string> stringListOption{
      *this, "string-list", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("Example string list option")};
  Option<std::string> stringOption{*this, "string",
                                   llvm::cl::desc("Example string option")};
};

/// A test pass that always aborts to enable testing the crash recovery
/// mechanism of the pass manager.
class TestCrashRecoveryPass : public OperationPass<TestCrashRecoveryPass> {
  void runOnOperation() final { abort(); }
};

/// A test pass that contains a statistic.
struct TestStatisticPass : public OperationPass<TestStatisticPass> {
  TestStatisticPass() = default;
  TestStatisticPass(const TestStatisticPass &) {}

  Statistic opCount{this, "num-ops", "Number of operations counted"};

  void runOnOperation() final {
    getOperation()->walk([&](Operation *) { ++opCount; });
  }
};
} // end anonymous namespace

static void testNestedPipeline(OpPassManager &pm) {
  // Nest a module pipeline that contains:
  /// A module pass.
  auto &modulePM = pm.nest<ModuleOp>();
  modulePM.addPass(std::make_unique<TestModulePass>());
  /// A nested function pass.
  auto &nestedFunctionPM = modulePM.nest<FuncOp>();
  nestedFunctionPM.addPass(std::make_unique<TestFunctionPass>());

  // Nest a function pipeline that contains a single pass.
  auto &functionPM = pm.nest<FuncOp>();
  functionPM.addPass(std::make_unique<TestFunctionPass>());
}

static void testNestedPipelineTextual(OpPassManager &pm) {
  (void)parsePassPipeline("test-pm-nested-pipeline", pm);
}

static PassRegistration<TestOptionsPass>
    reg("test-options-pass", "Test options parsing capabilities");

static PassRegistration<TestModulePass>
    unusedMP("test-module-pass", "Test a module pass in the pass manager");
static PassRegistration<TestFunctionPass>
    unusedFP("test-function-pass", "Test a function pass in the pass manager");

static PassRegistration<TestCrashRecoveryPass>
    unusedCrashP("test-pass-crash",
                 "Test a pass in the pass manager that always crashes");

static PassRegistration<TestStatisticPass> unusedStatP("test-stats-pass",
                                                       "Test pass statistics");

static PassPipelineRegistration<>
    unused("test-pm-nested-pipeline",
           "Test a nested pipeline in the pass manager", testNestedPipeline);
static PassPipelineRegistration<>
    unusedTextual("test-textual-pm-nested-pipeline",
                  "Test a nested pipeline in the pass manager",
                  testNestedPipelineTextual);
static PassPipelineRegistration<>
    unusedDump("test-dump-pipeline",
               "Dumps the pipeline build so far for debugging purposes",
               [](OpPassManager &pm) {
                 pm.printAsTextualPipeline(llvm::errs());
                 llvm::errs() << "\n";
               });

static PassPipelineRegistration<TestOptionsPass::Options>
    registerOptionsPassPipeline(
        "test-options-pass-pipeline",
        "Parses options using pass pipeline registration",
        [](OpPassManager &pm, const TestOptionsPass::Options &options) {
          pm.addPass(std::make_unique<TestOptionsPass>(options));
        });

//===- TestPassManager.cpp - Test pass manager functionality --------------===//
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
} // namespace

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

static PassRegistration<TestModulePass>
    unusedMP("test-module-pass", "Test a module pass in the pass manager");
static PassRegistration<TestFunctionPass>
    unusedFP("test-function-pass", "Test a function pass in the pass manager");

static PassPipelineRegistration
    unused("test-pm-nested-pipeline",
           "Test a nested pipeline in the pass manager", testNestedPipeline);
static PassPipelineRegistration
    unusedTextual("test-textual-pm-nested-pipeline",
                  "Test a nested pipeline in the pass manager",
                  testNestedPipelineTextual);

//===- AnalysisManagerTest.cpp - AnalysisManager unit tests ---------------===//
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

#include "mlir/Pass/AnalysisManager.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::detail;

namespace {
/// Minimal class definitions for two analyses.
struct MyAnalysis {
  MyAnalysis(Operation *) {}
};
struct OtherAnalysis {
  OtherAnalysis(Operation *) {}
};

TEST(AnalysisManagerTest, FineGrainModuleAnalysisPreservation) {
  MLIRContext context;

  // Test fine grain invalidation of the module analysis manager.
  OwningModuleRef module(ModuleOp::create(UnknownLoc::get(&context)));
  ModuleAnalysisManager mam(*module, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;

  // Query two different analyses, but only preserve one before invalidating.
  am.getAnalysis<MyAnalysis>();
  am.getAnalysis<OtherAnalysis>();

  detail::PreservedAnalyses pa;
  pa.preserve<MyAnalysis>();
  am.invalidate(pa);

  // Check that only MyAnalysis is preserved.
  EXPECT_TRUE(am.getCachedAnalysis<MyAnalysis>().hasValue());
  EXPECT_FALSE(am.getCachedAnalysis<OtherAnalysis>().hasValue());
}

TEST(AnalysisManagerTest, FineGrainFunctionAnalysisPreservation) {
  MLIRContext context;
  Builder builder(&context);

  // Create a function and a module.
  OwningModuleRef module(ModuleOp::create(UnknownLoc::get(&context)));
  FuncOp func1 =
      FuncOp::create(builder.getUnknownLoc(), "foo",
                     builder.getFunctionType(llvm::None, llvm::None));
  module->push_back(func1);

  // Test fine grain invalidation of the function analysis manager.
  ModuleAnalysisManager mam(*module, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;
  AnalysisManager fam = am.slice(func1);

  // Query two different analyses, but only preserve one before invalidating.
  fam.getAnalysis<MyAnalysis>();
  fam.getAnalysis<OtherAnalysis>();

  detail::PreservedAnalyses pa;
  pa.preserve<MyAnalysis>();
  fam.invalidate(pa);

  // Check that only MyAnalysis is preserved.
  EXPECT_TRUE(fam.getCachedAnalysis<MyAnalysis>().hasValue());
  EXPECT_FALSE(fam.getCachedAnalysis<OtherAnalysis>().hasValue());
}

TEST(AnalysisManagerTest, FineGrainChildFunctionAnalysisPreservation) {
  MLIRContext context;
  Builder builder(&context);

  // Create a function and a module.
  OwningModuleRef module(ModuleOp::create(UnknownLoc::get(&context)));
  FuncOp func1 =
      FuncOp::create(builder.getUnknownLoc(), "foo",
                     builder.getFunctionType(llvm::None, llvm::None));
  module->push_back(func1);

  // Test fine grain invalidation of a function analysis from within a module
  // analysis manager.
  ModuleAnalysisManager mam(*module, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;

  // Check that the analysis cache is initially empty.
  EXPECT_FALSE(am.getCachedChildAnalysis<MyAnalysis>(func1).hasValue());

  // Query two different analyses, but only preserve one before invalidating.
  am.getChildAnalysis<MyAnalysis>(func1);
  am.getChildAnalysis<OtherAnalysis>(func1);

  detail::PreservedAnalyses pa;
  pa.preserve<MyAnalysis>();
  am.invalidate(pa);

  // Check that only MyAnalysis is preserved.
  EXPECT_TRUE(am.getCachedChildAnalysis<MyAnalysis>(func1).hasValue());
  EXPECT_FALSE(am.getCachedChildAnalysis<OtherAnalysis>(func1).hasValue());
}

} // end namespace

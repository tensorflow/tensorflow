//===- Vectorize.cpp - Vectorize Pass Impl ----------------------*- C++ -*-===//
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
// This file implements vectorization of loops, operations and data types to
// a target-independent, n-D virtual vector abstraction.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/MLFunctionMatcher.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Transforms/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

static cl::list<unsigned> clVirtualVectorSize(
    "virtual-vector-size",
    cl::desc("Specify n-D virtual vector size for vectorization"),
    cl::ZeroOrMore);

namespace {

struct Vectorize : public FunctionPass {
  PassResult runOnMLFunction(MLFunction *f) override;

  // Thread-safe RAII contexts local to pass, BumpPtrAllocator freed on exit.
  MLFunctionMatcherContext MLContext;
};

} // end anonymous namespace

PassResult Vectorize::runOnMLFunction(MLFunction *f) {
  using matcher::Doall;
  /// TODO(ntv): support at least 4 cases for each load/store:
  ///   1. invariant along the loop index -> 1-D vectorizable with broadcast
  ///   2. contiguous along the fastest varying dimension wrt the loop index
  ///     -> a. 1-D vectorizable via stripmine/sink if loop is not innermost
  ///     -> b. 1-D vectorizable if loop is innermost
  ///   3. contiguous along non-fastest varying dimension wrt the loop index
  ///     -> needs data layout + copy to vectorize 1-D
  ///   4. not contiguous => not vectorizable
  auto pointwiseLike = Doall();
  auto &matches = pointwiseLike.match(f);
  for (auto loop : matches) {
    auto *doall = cast<ForStmt>(loop.first);
    if (!isVectorizableLoop(*doall)) {
      outs() << "\nNon-vectorizable loop: ";
      doall->print(outs());
      continue;
    }
    outs() << "\nVectorizable loop: ";
    doall->print(outs());
  }
  return PassResult::Success;
}

FunctionPass *mlir::createVectorizePass() { return new Vectorize(); }

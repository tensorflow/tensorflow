//===- Canonicalizer.cpp - Canonicalize MLIR operations -------------------===//
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
// This transformation pass converts operations into their canonical forms by
// folding constants, applying operation identity transformations etc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/Pass.h"
#include "mlir/Transforms/Passes.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// The actual Canonicalizer Pass.
//===----------------------------------------------------------------------===//

namespace {

/// Canonicalize operations in functions.
struct Canonicalizer : public FunctionPass {
  PassResult runOnCFGFunction(CFGFunction *f) override;
  PassResult runOnMLFunction(MLFunction *f) override;
  PassResult runOnFunction(Function *fn);
};
} // end anonymous namespace


PassResult Canonicalizer::runOnCFGFunction(CFGFunction *fn) {
  return runOnFunction(fn);
}

PassResult Canonicalizer::runOnMLFunction(MLFunction *fn) {
  return runOnFunction(fn);
}

PassResult Canonicalizer::runOnFunction(Function *fn) {
  auto *context = fn->getContext();
  OwningPatternList patterns;

  // TODO: Instead of adding all known patterns from the whole system lazily add
  // and cache the canonicalization patterns for ops we see in practice when
  // building the worklist.  For now, we just grab everything.
  for (auto *op : fn->getContext()->getRegisteredOperations())
    op->getCanonicalizationPatterns(patterns, context);

  applyPatternsGreedily(fn, std::move(patterns));
  return success();
}

/// Create a Canonicalizer pass.
FunctionPass *mlir::createCanonicalizerPass() { return new Canonicalizer(); }

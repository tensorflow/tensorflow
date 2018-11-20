//===- VectorizerTestPass.cpp - VectorizerTestPass Pass Impl ----*- C++
//-*-====================//
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
// This file implements a simple testing pass for vectorization functionality.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/MLFunctionMatcher.h"
#include "mlir/Analysis/VectorAnalysis.h"
#include "mlir/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

using llvm::outs;
using llvm::cl::desc;
using llvm::cl::list;
using llvm::cl::ZeroOrMore;

static list<int> clTestVectorMultiplicity(
    "vector-multiplicity", desc("Specify the HW vector size for vectorization"),
    ZeroOrMore);

#define DEBUG_TYPE "vectorizer-test"

namespace {

struct VectorizerTestPass : public FunctionPass {
  VectorizerTestPass() : FunctionPass(&VectorizerTestPass::passID) {}

  PassResult runOnMLFunction(MLFunction *f) override;
  void testVectorMultiplicity(MLFunction *f);

  // Thread-safe RAII contexts local to pass, BumpPtrAllocator freed on exit.
  MLFunctionMatcherContext MLContext;

  static char passID;
};

} // end anonymous namespace

char VectorizerTestPass::passID = 0;

void VectorizerTestPass::testVectorMultiplicity(MLFunction *f) {
  using matcher::Op;
  SmallVector<int, 8> shape(clTestVectorMultiplicity.begin(),
                            clTestVectorMultiplicity.end());
  auto subVectorType = VectorType::get(shape, Type::getF32(f->getContext()));
  // Only filter statements that operate on a strict super-vector and have one
  // return. This makes testing easier.
  auto filter = [subVectorType](const Statement &stmt) {
    outs() << "\ntest: " << stmt << " ";
    auto *opStmt = dyn_cast<OperationStmt>(&stmt);
    if (!opStmt) {
      return false;
    }
    if (!matcher::operatesOnStrictSuperVectors(*opStmt, subVectorType)) {
      return false;
    }
    if (opStmt->getNumResults() != 1) {
      return false;
    }
    return true;
  };
  auto pat = Op(filter);
  auto matches = pat.match(f);
  for (auto m : matches) {
    auto *opStmt = cast<OperationStmt>(m.first);
    // This is a unit test that only checks and prints shape ratio.
    // As a consequence we write only Ops with a single return type for the
    // purpose of this test. If we need to test more intricate behavior in the
    // future we can always extend.
    auto superVectorType = opStmt->getResult(0)->getType().cast<VectorType>();
    auto multiplicity = shapeRatio(superVectorType, subVectorType);
    assert(multiplicity.hasValue() && "Expected multiplicity");
    outs() << "\nmatched: " << *opStmt << " with multiplicity: ";
    interleaveComma(MutableArrayRef<unsigned>(*multiplicity), outs());
  }
}

PassResult VectorizerTestPass::runOnMLFunction(MLFunction *f) {
  if (!clTestVectorMultiplicity.empty()) {
    testVectorMultiplicity(f);
  }

  return PassResult::Success;
}

FunctionPass *mlir::createVectorizerTestPass() {
  return new VectorizerTestPass();
}

static PassRegistration<VectorizerTestPass>
    pass("vectorizer-test", "Tests vectorizer standalone functionality.");

#undef DEBUG_TYPE

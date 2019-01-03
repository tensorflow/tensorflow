//===- VectorizerTestPass.cpp - VectorizerTestPass Pass Impl --------------===//
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

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/MLFunctionMatcher.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/VectorAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vectorizer-test"

using namespace mlir;

using llvm::outs;
using llvm::SetVector;

using functional::map;

static llvm::cl::list<int> clTestVectorShapeRatio(
    "vector-shape-ratio",
    llvm::cl::desc("Specify the HW vector size for vectorization"),
    llvm::cl::ZeroOrMore);
static llvm::cl::opt<bool> clTestForwardSlicingAnalysis(
    "forward-slicing",
    llvm::cl::desc(
        "Specify to enable testing forward static slicing and topological sort "
        "functionalities"));
static llvm::cl::opt<bool> clTestBackwardSlicingAnalysis(
    "backward-slicing",
    llvm::cl::desc("Specify to enable testing backward static slicing and "
                   "topological sort functionalities"));
static llvm::cl::opt<bool> clTestSlicingAnalysis(
    "slicing",
    llvm::cl::desc(
        "Specify to enable testing static slicing and topological sort "
        "functionalities"));
static llvm::cl::opt<bool> clTestComposeMaps(
    "compose-maps",
    llvm::cl::desc(
        "Specify to enable testing the composition of AffineMap where each "
        "AffineMap in the composition is specified as the affine_map attribute "
        "in a constant op."));

namespace {

struct VectorizerTestPass : public FunctionPass {
  static constexpr auto kTestAffineMapOpName = "test_affine_map";
  static constexpr auto kTestAffineMapAttrName = "affine_map";
  VectorizerTestPass() : FunctionPass(&VectorizerTestPass::passID) {}

  PassResult runOnFunction(Function *f) override;
  void testVectorShapeRatio(Function *f);
  void testForwardSlicing(Function *f);
  void testBackwardSlicing(Function *f);
  void testSlicing(Function *f);
  void testComposeMaps(Function *f);

  // Thread-safe RAII contexts local to pass, BumpPtrAllocator freed on exit.
  MLFunctionMatcherContext MLContext;

  static char passID;
};

} // end anonymous namespace

char VectorizerTestPass::passID = 0;

void VectorizerTestPass::testVectorShapeRatio(Function *f) {
  using matcher::Op;
  SmallVector<int, 8> shape(clTestVectorShapeRatio.begin(),
                            clTestVectorShapeRatio.end());
  auto subVectorType = VectorType::get(shape, Type::getF32(f->getContext()));
  // Only filter instructions that operate on a strict super-vector and have one
  // return. This makes testing easier.
  auto filter = [subVectorType](const Instruction &inst) {
    auto *opInst = dyn_cast<OperationInst>(&inst);
    if (!opInst) {
      return false;
    }
    assert(subVectorType.getElementType() ==
               Type::getF32(subVectorType.getContext()) &&
           "Only f32 supported for now");
    if (!matcher::operatesOnSuperVectors(*opInst, subVectorType)) {
      return false;
    }
    if (opInst->getNumResults() != 1) {
      return false;
    }
    return true;
  };
  auto pat = Op(filter);
  auto matches = pat.match(f);
  for (auto m : matches) {
    auto *opInst = cast<OperationInst>(m.first);
    // This is a unit test that only checks and prints shape ratio.
    // As a consequence we write only Ops with a single return type for the
    // purpose of this test. If we need to test more intricate behavior in the
    // future we can always extend.
    auto superVectorType = opInst->getResult(0)->getType().cast<VectorType>();
    auto ratio = shapeRatio(superVectorType, subVectorType);
    if (!ratio.hasValue()) {
      opInst->emitNote("NOT MATCHED");
    } else {
      outs() << "\nmatched: " << *opInst << " with shape ratio: ";
      interleaveComma(MutableArrayRef<unsigned>(*ratio), outs());
    }
  }
}

static std::string toString(Instruction *inst) {
  std::string res;
  auto os = llvm::raw_string_ostream(res);
  inst->print(os);
  return res;
}

static MLFunctionMatches matchTestSlicingOps(Function *f) {
  // Just use a custom op name for this test, it makes life easier.
  constexpr auto kTestSlicingOpName = "slicing-test-op";
  using functional::map;
  using matcher::Op;
  // Match all OpInstructions with the kTestSlicingOpName name.
  auto filter = [](const Instruction &inst) {
    const auto &opInst = cast<OperationInst>(inst);
    return opInst.getName().getStringRef() == kTestSlicingOpName;
  };
  auto pat = Op(filter);
  return pat.match(f);
}

void VectorizerTestPass::testBackwardSlicing(Function *f) {
  auto matches = matchTestSlicingOps(f);
  for (auto m : matches) {
    SetVector<Instruction *> backwardSlice;
    getBackwardSlice(m.first, &backwardSlice);
    auto strs = map(toString, backwardSlice);
    outs() << "\nmatched: " << *m.first << " backward static slice: ";
    for (const auto &s : strs) {
      outs() << "\n" << s;
    }
  }
}

void VectorizerTestPass::testForwardSlicing(Function *f) {
  auto matches = matchTestSlicingOps(f);
  for (auto m : matches) {
    SetVector<Instruction *> forwardSlice;
    getForwardSlice(m.first, &forwardSlice);
    auto strs = map(toString, forwardSlice);
    outs() << "\nmatched: " << *m.first << " forward static slice: ";
    for (const auto &s : strs) {
      outs() << "\n" << s;
    }
  }
}

void VectorizerTestPass::testSlicing(Function *f) {
  auto matches = matchTestSlicingOps(f);
  for (auto m : matches) {
    SetVector<Instruction *> staticSlice = getSlice(m.first);
    auto strs = map(toString, staticSlice);
    outs() << "\nmatched: " << *m.first << " static slice: ";
    for (const auto &s : strs) {
      outs() << "\n" << s;
    }
  }
}

bool customOpWithAffineMapAttribute(const Instruction &inst) {
  const auto &opInst = cast<OperationInst>(inst);
  return opInst.getName().getStringRef() ==
         VectorizerTestPass::kTestAffineMapOpName;
}

void VectorizerTestPass::testComposeMaps(Function *f) {
  using matcher::Op;
  auto pattern = Op(customOpWithAffineMapAttribute);
  auto matches = pattern.match(f);
  SmallVector<AffineMap, 4> maps;
  maps.reserve(matches.size());
  std::reverse(matches.begin(), matches.end());
  for (auto m : matches) {
    auto *opInst = cast<OperationInst>(m.first);
    auto map = opInst->getAttr(VectorizerTestPass::kTestAffineMapAttrName)
                   .cast<AffineMapAttr>()
                   .getValue();
    maps.push_back(map);
  }
  AffineMap res;
  for (auto m : maps) {
    res = res ? composeUnboundedMaps(res, m) : m;
  }
  res.print(outs() << "\nComposed map: ");
}

PassResult VectorizerTestPass::runOnFunction(Function *f) {
  // Only support single block functions at this point.
  if (f->getBlocks().size() != 1)
    return success();

  if (!clTestVectorShapeRatio.empty()) {
    testVectorShapeRatio(f);
  }
  if (clTestForwardSlicingAnalysis) {
    testForwardSlicing(f);
  }
  if (clTestBackwardSlicingAnalysis) {
    testBackwardSlicing(f);
  }
  if (clTestSlicingAnalysis) {
    testSlicing(f);
  }
  if (clTestComposeMaps) {
    testComposeMaps(f);
  }
  return PassResult::Success;
}

FunctionPass *mlir::createVectorizerTestPass() {
  return new VectorizerTestPass();
}

static PassRegistration<VectorizerTestPass>
    pass("vectorizer-test", "Tests vectorizer standalone functionality.");

#undef DEBUG_TYPE

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
#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/VectorAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/Passes.h"
#include "third_party/llvm/llvm/include/llvm/ADT/STLExtras.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vectorizer-test"

using namespace mlir;

using llvm::outs;
using llvm::SetVector;

using functional::map;

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

static llvm::cl::list<int> clTestVectorShapeRatio(
    "vector-shape-ratio",
    llvm::cl::desc("Specify the HW vector size for vectorization"),
    llvm::cl::ZeroOrMore, llvm::cl::cat(clOptionsCategory));
static llvm::cl::opt<bool> clTestForwardSlicingAnalysis(
    "forward-slicing",
    llvm::cl::desc("Enable testing forward static slicing and topological sort "
                   "functionalities"),
    llvm::cl::cat(clOptionsCategory));
static llvm::cl::opt<bool> clTestBackwardSlicingAnalysis(
    "backward-slicing",
    llvm::cl::desc("Enable testing backward static slicing and "
                   "topological sort functionalities"),
    llvm::cl::cat(clOptionsCategory));
static llvm::cl::opt<bool> clTestSlicingAnalysis(
    "slicing",
    llvm::cl::desc("Enable testing static slicing and topological sort "
                   "functionalities"),
    llvm::cl::cat(clOptionsCategory));
static llvm::cl::opt<bool> clTestComposeMaps(
    "compose-maps",
    llvm::cl::desc(
        "Enable testing the composition of AffineMap where each "
        "AffineMap in the composition is specified as the affine_map attribute "
        "in a constant op."),
    llvm::cl::cat(clOptionsCategory));
static llvm::cl::opt<bool> clTestNormalizeMaps(
    "normalize-maps",
    llvm::cl::desc(
        "Enable testing the normalization of AffineAffineApplyOp "
        "where each AffineAffineApplyOp in the composition is a single output "
        "instruction."),
    llvm::cl::cat(clOptionsCategory));

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
  void testNormalizeMaps(Function *f);

  static char passID;
};

} // end anonymous namespace

char VectorizerTestPass::passID = 0;

void VectorizerTestPass::testVectorShapeRatio(Function *f) {
  using matcher::Op;
  SmallVector<int64_t, 8> shape(clTestVectorShapeRatio.begin(),
                                clTestVectorShapeRatio.end());
  auto subVectorType = VectorType::get(shape, Type::getF32(f->getContext()));
  // Only filter instructions that operate on a strict super-vector and have one
  // return. This makes testing easier.
  auto filter = [subVectorType](const Instruction &inst) {
    assert(subVectorType.getElementType() ==
               Type::getF32(subVectorType.getContext()) &&
           "Only f32 supported for now");
    if (!matcher::operatesOnSuperVectors(inst, subVectorType)) {
      return false;
    }
    if (inst.getNumResults() != 1) {
      return false;
    }
    return true;
  };
  auto pat = Op(filter);
  SmallVector<NestedMatch, 8> matches;
  pat.match(f, &matches);
  for (auto m : matches) {
    auto *opInst = m.getMatchedInstruction();
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

static NestedPattern patternTestSlicingOps() {
  // Just use a custom op name for this test, it makes life easier.
  constexpr auto kTestSlicingOpName = "slicing-test-op";
  using functional::map;
  using matcher::Op;
  // Match all OpInstructions with the kTestSlicingOpName name.
  auto filter = [](const Instruction &inst) {
    return inst.getName().getStringRef() == kTestSlicingOpName;
  };
  return Op(filter);
}

void VectorizerTestPass::testBackwardSlicing(Function *f) {
  SmallVector<NestedMatch, 8> matches;
  patternTestSlicingOps().match(f, &matches);
  for (auto m : matches) {
    SetVector<Instruction *> backwardSlice;
    getBackwardSlice(m.getMatchedInstruction(), &backwardSlice);
    auto strs = map(toString, backwardSlice);
    outs() << "\nmatched: " << *m.getMatchedInstruction()
           << " backward static slice: ";
    for (const auto &s : strs) {
      outs() << "\n" << s;
    }
  }
}

void VectorizerTestPass::testForwardSlicing(Function *f) {
  SmallVector<NestedMatch, 8> matches;
  patternTestSlicingOps().match(f, &matches);
  for (auto m : matches) {
    SetVector<Instruction *> forwardSlice;
    getForwardSlice(m.getMatchedInstruction(), &forwardSlice);
    auto strs = map(toString, forwardSlice);
    outs() << "\nmatched: " << *m.getMatchedInstruction()
           << " forward static slice: ";
    for (const auto &s : strs) {
      outs() << "\n" << s;
    }
  }
}

void VectorizerTestPass::testSlicing(Function *f) {
  SmallVector<NestedMatch, 8> matches;
  patternTestSlicingOps().match(f, &matches);
  for (auto m : matches) {
    SetVector<Instruction *> staticSlice = getSlice(m.getMatchedInstruction());
    auto strs = map(toString, staticSlice);
    outs() << "\nmatched: " << *m.getMatchedInstruction() << " static slice: ";
    for (const auto &s : strs) {
      outs() << "\n" << s;
    }
  }
}

static bool customOpWithAffineMapAttribute(const Instruction &inst) {
  return inst.getName().getStringRef() ==
         VectorizerTestPass::kTestAffineMapOpName;
}

void VectorizerTestPass::testComposeMaps(Function *f) {
  using matcher::Op;
  auto pattern = Op(customOpWithAffineMapAttribute);
  SmallVector<NestedMatch, 8> matches;
  pattern.match(f, &matches);
  SmallVector<AffineMap, 4> maps;
  maps.reserve(matches.size());
  for (auto m : llvm::reverse(matches)) {
    auto *opInst = m.getMatchedInstruction();
    auto map = opInst->getAttr(VectorizerTestPass::kTestAffineMapAttrName)
                   .cast<AffineMapAttr>()
                   .getValue();
    maps.push_back(map);
  }
  AffineMap res;
  for (auto m : maps) {
    res = res ? res.compose(m) : m;
  }
  simplifyAffineMap(res).print(outs() << "\nComposed map: ");
}

static bool affineApplyOp(const Instruction &inst) {
  return inst.isa<AffineApplyOp>();
}

static bool singleResultAffineApplyOpWithoutUses(const Instruction &inst) {
  auto app = inst.dyn_cast<AffineApplyOp>();
  return app && app->use_empty();
}

void VectorizerTestPass::testNormalizeMaps(Function *f) {
  using matcher::Op;

  // Save matched AffineApplyOp that all need to be erased in the end.
  auto pattern = Op(affineApplyOp);
  SmallVector<NestedMatch, 8> toErase;
  pattern.match(f, &toErase);
  {
    // Compose maps.
    auto pattern = Op(singleResultAffineApplyOpWithoutUses);
    SmallVector<NestedMatch, 8> matches;
    pattern.match(f, &matches);
    for (auto m : matches) {
      auto app = m.getMatchedInstruction()->cast<AffineApplyOp>();
      FuncBuilder b(m.getMatchedInstruction());
      SmallVector<Value *, 8> operands(app->getOperands());
      makeComposedAffineApply(&b, app->getLoc(), app->getAffineMap(), operands);
    }
  }
  // We should now be able to erase everything in reverse order in this test.
  for (auto m : llvm::reverse(toErase)) {
    m.getMatchedInstruction()->erase();
  }
}

PassResult VectorizerTestPass::runOnFunction(Function *f) {
  // Thread-safe RAII local context, BumpPtrAllocator freed on exit.
  NestedPatternContext mlContext;

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
  if (clTestNormalizeMaps) {
    testNormalizeMaps(f);
  }
  return PassResult::Success;
}

FunctionPass *mlir::createVectorizerTestPass() {
  return new VectorizerTestPass();
}

static PassRegistration<VectorizerTestPass>
    pass("vectorizer-test", "Tests vectorizer standalone functionality.");

#undef DEBUG_TYPE

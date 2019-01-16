//===- LowerTestPass.cpp - Test pass for lowering EDSC --------------------===//
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

#include "mlir/EDSC/MLIREmitter.h"
#include "mlir/EDSC/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/InstVisitor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
// Testing pass to lower EDSC.
struct LowerEDSCTestPass : public FunctionPass {
  LowerEDSCTestPass() : FunctionPass(&LowerEDSCTestPass::passID) {}
  PassResult runOnFunction(Function *f) override;

  static char passID;
};
} // end anonymous namespace

char LowerEDSCTestPass::passID = 0;

// TODO: These should be in a common library.
static bool isDynamicSize(int size) { return size < 0; }

static SmallVector<Value *, 8> getMemRefSizes(FuncBuilder *b, Location loc,
                                              Value *memRef) {
  auto memRefType = memRef->getType().cast<MemRefType>();
  SmallVector<Value *, 8> res;
  res.reserve(memRefType.getShape().size());
  unsigned countSymbolicShapes = 0;
  for (int size : memRefType.getShape()) {
    if (isDynamicSize(size)) {
      res.push_back(b->create<DimOp>(loc, memRef, countSymbolicShapes++));
    } else {
      res.push_back(b->create<ConstantIndexOp>(loc, size));
    }
  }
  return res;
}

SmallVector<edsc::Bindable, 8> makeBoundSizes(edsc::MLIREmitter *emitter,
                                              Value *memRef) {
  MemRefType memRefType = memRef->getType().cast<MemRefType>();
  auto memRefSizes = edsc::makeBindables(memRefType.getShape().size());
  auto memrefSizeValues =
      getMemRefSizes(emitter->getBuilder(), emitter->getLocation(), memRef);
  assert(memrefSizeValues.size() == memRefSizes.size());
  emitter->bindZipRange(llvm::zip(memRefSizes, memrefSizeValues));
  return memRefSizes;
}

#include "mlir/EDSC/reference-impl.inc"

PassResult LowerEDSCTestPass::runOnFunction(Function *f) {
  f->walkOps([](OperationInst *op) {
    if (op->getName().getStringRef() == "dump") {
      printRefImplementation(op->getAttrOfType<FunctionAttr>("fn").getValue());
    }
  });
  return success();
}

static PassRegistration<LowerEDSCTestPass> pass("lower-edsc-test",
                                                "Lower EDSC test pass");

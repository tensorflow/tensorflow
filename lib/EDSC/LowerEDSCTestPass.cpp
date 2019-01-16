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

#include "mlir/EDSC/reference-impl.inc"

PassResult LowerEDSCTestPass::runOnFunction(Function *f) {
  f->walkOps([](OperationInst *op) {
    if (op->getName().getStringRef() == "print") {
      auto opName = op->getAttrOfType<StringAttr>("op");
      if (!opName) {
        op->emitOpError("no 'op' attribute provided for print");
        return;
      }
      auto function = op->getAttrOfType<FunctionAttr>("fn");
      if (!function) {
        op->emitOpError("no 'fn' attribute provided for print");
        return;
      }
      printRefImplementation(opName.getValue(), function.getValue());
    }
  });
  return success();
}

static PassRegistration<LowerEDSCTestPass> pass("lower-edsc-test",
                                                "Lower EDSC test pass");

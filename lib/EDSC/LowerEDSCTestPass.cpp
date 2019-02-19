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
  // Inject a EDSC-constructed list of blocks.
  if (f->getName().strref() == "blocks") {
    using namespace edsc::op;

    FuncBuilder builder(f);
    edsc::ScopedEDSCContext context;
    auto type = builder.getIntegerType(32);
    edsc::Expr arg1(type), arg2(type), arg3(type), arg4(type);

    auto b1 =
        edsc::block({arg1, arg2}, {type, type}, {arg1 + arg2, edsc::Return()});
    auto b2 =
        edsc::block({arg3, arg4}, {type, type}, {arg3 - arg4, edsc::Return()});

    edsc::MLIREmitter(&builder, f->getLoc()).emitBlock(b1).emitBlock(b2);
  }

  // Inject a EDSC-constructed `for` loop with bounds coming from function
  // arguments.
  if (f->getName().strref() == "dynamic_for_func_args") {
    assert(!f->getBlocks().empty() && "dynamic_for should not be empty");
    FuncBuilder builder(&f->getBlocks().front(),
                        f->getBlocks().front().begin());
    assert(f->getNumArguments() == 2 && "dynamic_for expected 4 arguments");
    for (const auto *arg : f->getArguments()) {
      (void)arg;
      assert(arg->getType().isIndex() &&
             "dynamic_for expected index arguments");
    }

    Type index = IndexType::get(f->getContext());
    edsc::ScopedEDSCContext context;
    edsc::Expr lb(index), ub(index), step(index);
    auto loop = edsc::For(lb, ub, step, {});
    edsc::MLIREmitter(&builder, f->getLoc())
        .bind(edsc::Bindable(lb), f->getArgument(0))
        .bind(edsc::Bindable(ub), f->getArgument(1))
        .bindConstant<ConstantIndexOp>(edsc::Bindable(step), 3)
        .emitStmt(loop);
    return success();
  }

  // Inject a EDSC-constructed `for` loop with non-constant bounds that are
  // obtained from AffineApplyOp (also constructed using EDSC operator
  // overloads).
  if (f->getName().strref() == "dynamic_for") {
    assert(!f->getBlocks().empty() && "dynamic_for should not be empty");
    FuncBuilder builder(&f->getBlocks().front(),
                        f->getBlocks().front().begin());
    assert(f->getNumArguments() == 4 && "dynamic_for expected 4 arguments");
    for (const auto *arg : f->getArguments()) {
      (void)arg;
      assert(arg->getType().isIndex() &&
             "dynamic_for expected index arguments");
    }

    Type index = IndexType::get(f->getContext());
    edsc::ScopedEDSCContext context;
    edsc::Expr lb1(index), lb2(index), ub1(index), ub2(index), step(index);
    using namespace edsc::op;
    auto lb = lb1 - lb2;
    auto ub = ub1 + ub2;
    auto loop = edsc::For(lb, ub, step, {});
    edsc::MLIREmitter(&builder, f->getLoc())
        .bind(edsc::Bindable(lb1), f->getArgument(0))
        .bind(edsc::Bindable(lb2), f->getArgument(1))
        .bind(edsc::Bindable(ub1), f->getArgument(2))
        .bind(edsc::Bindable(ub2), f->getArgument(3))
        .bindConstant<ConstantIndexOp>(edsc::Bindable(step), 2)
        .emitStmt(loop);

    return success();
  }

  f->walk([](Instruction *op) {
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

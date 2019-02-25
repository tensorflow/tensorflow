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
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
// Testing pass to lower EDSC.
struct LowerEDSCTestPass : public FunctionPass {
  LowerEDSCTestPass() : FunctionPass(&LowerEDSCTestPass::passID) {}
  PassResult runOnFunction(Function *f) override;

  constexpr static PassID passID = {};
};
} // end anonymous namespace

#include "mlir/EDSC/reference-impl.inc"

PassResult LowerEDSCTestPass::runOnFunction(Function *f) {
  // Inject a EDSC-constructed infinite loop implemented by mutual branching
  // between two blocks, following the pattern:
  //
  //       br ^bb1
  //    ^bb1:
  //       br ^bb2
  //    ^bb2:
  //       br ^bb1
  //
  // Use blocks with arguments.
  if (f->getName().strref() == "blocks") {
    using namespace edsc::op;

    FuncBuilder builder(f);
    edsc::ScopedEDSCContext context;
    // Declare two blocks.  Note that we must declare the blocks before creating
    // branches to them.
    auto type = builder.getIntegerType(32);
    edsc::Expr arg1(type), arg2(type), arg3(type), arg4(type), r(type);
    edsc::StmtBlock b1 = edsc::block({arg1, arg2}, {}),
                    b2 = edsc::block({arg3, arg4}, {});
    auto c1 = edsc::constantInteger(type, 42);
    auto c2 = edsc::constantInteger(type, 1234);

    // Make an infinite loops by branching between the blocks.  Note that copy-
    // assigning a block won't work well with branches, update the body instead.
    b1.set({r = arg1 + arg2, edsc::Branch(b2, {arg1, r})});
    b2.set({edsc::Branch(b1, {arg3, arg4})});
    auto instr = edsc::Branch(b2, {c1, c2});

    // Remove the existing 'return' from the function, reset the builder after
    // the instruction iterator invalidation and emit a branch to b2.  This
    // should also emit blocks b2 and b1 that appear as successors to the
    // current block after the branch instruction is insterted.
    f->begin()->clear();
    builder.setInsertionPoint(&*f->begin(), f->begin()->begin());
    edsc::MLIREmitter(&builder, f->getLoc()).emitStmt(instr);
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

    using namespace edsc::op;
    Type index = IndexType::get(f->getContext());
    edsc::ScopedEDSCContext context;
    edsc::Expr lb(index), ub(index), step(index);
    step = edsc::constantInteger(index, 3);
    auto loop = edsc::For(lb, ub, step, {lb * step + ub, step + lb});
    edsc::MLIREmitter(&builder, f->getLoc())
        .bind(edsc::Bindable(lb), f->getArgument(0))
        .bind(edsc::Bindable(ub), f->getArgument(1))
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
  if (f->getName().strref() == "max_min_for") {
    assert(!f->getBlocks().empty() && "max_min_for should not be empty");
    FuncBuilder builder(&f->getBlocks().front(),
                        f->getBlocks().front().begin());
    assert(f->getNumArguments() == 4 && "max_min_for expected 4 arguments");
    assert(std::all_of(f->args_begin(), f->args_end(),
                       [](const Value *s) { return s->getType().isIndex(); }) &&
           "max_min_for expected index arguments");

    edsc::ScopedEDSCContext context;
    edsc::Expr lb1(f->getArgument(0)->getType());
    edsc::Expr lb2(f->getArgument(1)->getType());
    edsc::Expr ub1(f->getArgument(2)->getType());
    edsc::Expr ub2(f->getArgument(3)->getType());
    edsc::Expr iv(builder.getIndexType());
    edsc::Expr step = edsc::constantInteger(builder.getIndexType(), 1);
    auto loop =
        edsc::MaxMinFor(edsc::Bindable(iv), {lb1, lb2}, {ub1, ub2}, step, {});
    edsc::MLIREmitter(&builder, f->getLoc())
        .bind(edsc::Bindable(lb1), f->getArgument(0))
        .bind(edsc::Bindable(lb2), f->getArgument(1))
        .bind(edsc::Bindable(ub1), f->getArgument(2))
        .bind(edsc::Bindable(ub2), f->getArgument(3))
        .emitStmt(loop);

    return success();
  }
  if (f->getName().strref() == "call_indirect") {
    assert(!f->getBlocks().empty() && "call_indirect should not be empty");
    FuncBuilder builder(&f->getBlocks().front(),
                        f->getBlocks().front().begin());
    Function *callee = f->getModule()->getNamedFunction("callee");
    Function *calleeArgs = f->getModule()->getNamedFunction("callee_args");
    Function *secondOrderCallee =
        f->getModule()->getNamedFunction("second_order_callee");
    assert(callee && calleeArgs && secondOrderCallee &&
           "could not find required declarations");

    auto funcRetIndexType = builder.getFunctionType({}, builder.getIndexType());

    edsc::ScopedEDSCContext context;
    edsc::Expr func(callee->getType()), funcArgs(calleeArgs->getType()),
        secondOrderFunc(secondOrderCallee->getType());
    auto stmt = edsc::call(func, {});
    auto chainedCallResult =
        edsc::call(edsc::call(secondOrderFunc, funcRetIndexType, {func}),
                   builder.getIndexType(), {});
    auto argsCall =
        edsc::call(funcArgs, {chainedCallResult, chainedCallResult});
    edsc::MLIREmitter(&builder, f->getLoc())
        .bindConstant<ConstantOp>(edsc::Bindable(func),
                                  builder.getFunctionAttr(callee))
        .bindConstant<ConstantOp>(edsc::Bindable(funcArgs),
                                  builder.getFunctionAttr(calleeArgs))
        .bindConstant<ConstantOp>(edsc::Bindable(secondOrderFunc),
                                  builder.getFunctionAttr(secondOrderCallee))
        .emitStmt(stmt)
        .emitStmt(chainedCallResult)
        .emitStmt(argsCall);

    return success();
  }

  // Inject an EDSC-constructed computation that assigns Stmt and uses the LHS.
  if (f->getName().strref().contains("assignments")) {
    FuncBuilder builder(f);
    edsc::ScopedEDSCContext context;
    edsc::MLIREmitter emitter(&builder, f->getLoc());

    edsc::Expr zero = emitter.zero();
    edsc::Expr one = emitter.one();
    auto args = emitter.makeBoundFunctionArguments(f);
    auto views = emitter.makeBoundMemRefViews(args.begin(), args.end());

    Type indexType = builder.getIndexType();
    edsc::Expr i(indexType);
    edsc::Expr A = args[0], B = args[1], C = args[2];
    edsc::Expr M = views[0].dim(0);
    // clang-format off
    using namespace edsc::op;
    edsc::Stmt scalarA, scalarB, tmp;
    auto block = edsc::block({
      For(i, zero, M, one, {
        scalarA = load(A, {i}),
        scalarB = load(B, {i}),
        tmp = scalarA * scalarB,
        store(tmp, C, {i})
      }),
    });
    // clang-format on

    emitter.emitStmts(block.getBody());
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

//===- APITest.cpp - Test for EDSC APIs -----------------------------------===//
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

// RUN: %p/api-test | FileCheck %s

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/EDSC/MLIREmitter.h"
#include "mlir/EDSC/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Transforms/LoopUtils.h"

#include "Test.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static MLIRContext &globalContext() {
  static thread_local MLIRContext context;
  return context;
}

static std::unique_ptr<Function> makeFunction(StringRef name,
                                              ArrayRef<Type> results = {},
                                              ArrayRef<Type> args = {}) {
  auto &ctx = globalContext();
  auto function = llvm::make_unique<Function>(
      UnknownLoc::get(&ctx), name, FunctionType::get(args, results, &ctx));
  function->addEntryBlock();
  return function;
}

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
TEST_FUNC(blocks) {
  using namespace edsc::op;

  auto f = makeFunction("blocks");
  FuncBuilder builder(f.get());
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
  auto op = edsc::Branch(b2, {c1, c2});

  // Emit a branch to b2.  This should also emit blocks b2 and b1 that appear as
  // successors to the current block after the branch operation is insterted.
  edsc::MLIREmitter(&builder, f->getLoc()).emitStmt(op);

  // clang-format off
  // CHECK-LABEL: @blocks
  // CHECK:        %c42_i32 = constant 42 : i32
  // CHECK-NEXT:   %c1234_i32 = constant 1234 : i32
  // CHECK-NEXT:   br ^bb1(%c42_i32, %c1234_i32 : i32, i32)
  // CHECK-NEXT: ^bb1(%0: i32, %1: i32):   // 2 preds: ^bb0, ^bb2
  // CHECK-NEXT:   br ^bb2(%0, %1 : i32, i32)
  // CHECK-NEXT: ^bb2(%2: i32, %3: i32):   // pred: ^bb1
  // CHECK-NEXT:   %4 = addi %2, %3 : i32
  // CHECK-NEXT:   br ^bb1(%2, %4 : i32, i32)
  // CHECK-NEXT: }
  // clang-format on
  f->print(llvm::outs());
}

// Inject two EDSC-constructed blocks with arguments and a conditional branch
// operation that transfers control to these blocks.
TEST_FUNC(cond_branch) {
  auto f =
      makeFunction("cond_branch", {}, {IntegerType::get(1, &globalContext())});

  FuncBuilder builder(f.get());
  edsc::ScopedEDSCContext context;
  auto i1 = builder.getIntegerType(1);
  auto i32 = builder.getIntegerType(32);
  auto i64 = builder.getIntegerType(64);
  edsc::Expr arg1(i32), arg2(i64), arg3(i32);
  // Declare two blocks with different numbers of arguments.
  edsc::StmtBlock b1 = edsc::block({arg1}, {edsc::Return()}),
                  b2 = edsc::block({arg2, arg3}, {edsc::Return()});
  edsc::Expr funcArg(i1);

  // Inject the conditional branch.
  auto condBranch = edsc::CondBranch(
      funcArg, b1, {edsc::constantInteger(i32, 32)}, b2,
      {edsc::constantInteger(i64, 64), edsc::constantInteger(i32, 42)});

  builder.setInsertionPoint(&*f->begin(), f->begin()->begin());
  edsc::MLIREmitter(&builder, f->getLoc())
      .bind(edsc::Bindable(funcArg), f->getArgument(0))
      .emitStmt(condBranch);

  // clang-format off
  // CHECK-LABEL: @cond_branch
  // CHECK:        %c0 = constant 0 : index
  // CHECK-NEXT:   %c1 = constant 1 : index
  // CHECK-NEXT:   %c32_i32 = constant 32 : i32
  // CHECK-NEXT:   %c64_i64 = constant 64 : i64
  // CHECK-NEXT:   %c42_i32 = constant 42 : i32
  // CHECK-NEXT:   cond_br %arg0, ^bb1(%c32_i32 : i32), ^bb2(%c64_i64, %c42_i32 : i64, i32)
  // CHECK-NEXT: ^bb1(%0: i32):   // pred: ^bb0
  // CHECK-NEXT:   return
  // CHECK-NEXT: ^bb2(%1: i64, %2: i32):  // pred: ^bb0
  // CHECK-NEXT:   return
  // clang-format on
  f->print(llvm::outs());
}

// Inject a EDSC-constructed `affine.for` loop with bounds coming from function
// arguments.
TEST_FUNC(dynamic_for_func_args) {
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("dynamic_for_func_args", {}, {indexType, indexType});
  FuncBuilder builder(f.get());

  using namespace edsc::op;
  Type index = IndexType::get(f->getContext());
  edsc::ScopedEDSCContext context;
  edsc::Expr lb(index), ub(index), step(index);
  step = edsc::constantInteger(index, 3);
  auto loop = edsc::For(lb, ub, step, {lb * step + ub, step + lb});
  edsc::MLIREmitter(&builder, f->getLoc())
      .bind(edsc::Bindable(lb), f->getArgument(0))
      .bind(edsc::Bindable(ub), f->getArgument(1))
      .emitStmt(loop)
      .emitStmt(edsc::Return());

  // clang-format off
  // CHECK-LABEL: func @dynamic_for_func_args(%arg0: index, %arg1: index) {
  // CHECK:  affine.for %i0 = (d0) -> (d0)(%arg0) to (d0) -> (d0)(%arg1) step 3 {
  // CHECK:  {{.*}} = affine.apply ()[s0] -> (s0 * 3)()[%arg0]
  // CHECK:  {{.*}} = affine.apply ()[s0, s1] -> (s1 + s0 * 3)()[%arg0, %arg1]
  // CHECK:  {{.*}} = affine.apply ()[s0] -> (s0 + 3)()[%arg0]
  // clang-format on
  f->print(llvm::outs());
}

// Inject a EDSC-constructed `affine.for` loop with non-constant bounds that are
// obtained from AffineApplyOp (also constructed using EDSC operator
// overloads).
TEST_FUNC(dynamic_for) {
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("dynamic_for", {},
                        {indexType, indexType, indexType, indexType});
  FuncBuilder builder(f.get());

  edsc::ScopedEDSCContext context;
  edsc::Expr lb1(indexType), lb2(indexType), ub1(indexType), ub2(indexType),
      step(indexType);
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

  // clang-format off
  // CHECK-LABEL: func @dynamic_for(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
  // CHECK:        %0 = affine.apply ()[s0, s1] -> (s0 - s1)()[%arg0, %arg1]
  // CHECK-NEXT:   %1 = affine.apply ()[s0, s1] -> (s0 + s1)()[%arg2, %arg3]
  // CHECK-NEXT:   affine.for %i0 = (d0) -> (d0)(%0) to (d0) -> (d0)(%1) step 2 {
  // clang-format on
  f->print(llvm::outs());
}

// Inject a EDSC-constructed empty `affine.for` loop with max/min bounds that
// corresponds to
//
//     for max(%arg0, %arg1) to (%arg2, %arg3) step 1
//
TEST_FUNC(max_min_for) {
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("max_min_for", {},
                        {indexType, indexType, indexType, indexType});
  FuncBuilder builder(f.get());

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

  // clang-format off
  // CHECK-LABEL: func @max_min_for(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
  // CHECK:  affine.for %i0 = max (d0, d1) -> (d0, d1)(%arg0, %arg1) to min (d0, d1) -> (d0, d1)(%arg2, %arg3) {
  // clang-format on
  f->print(llvm::outs());
}

// Inject EDSC-constructed chain of indirect calls that corresponds to
//
//     @callee()
//     var x = @second_order_callee(@callee)
//     @callee_args(x, x)
//
TEST_FUNC(call_indirect) {
  auto indexType = IndexType::get(&globalContext());
  auto callee = makeFunction("callee");
  auto calleeArgs = makeFunction("callee_args", {}, {indexType, indexType});
  auto secondOrderCallee =
      makeFunction("second_order_callee",
                   {FunctionType::get({}, {indexType}, &globalContext())},
                   {FunctionType::get({}, {}, &globalContext())});
  auto f = makeFunction("call_indirect");
  FuncBuilder builder(f.get());

  auto funcRetIndexType = builder.getFunctionType({}, builder.getIndexType());

  edsc::ScopedEDSCContext context;
  edsc::Expr func(callee->getType()), funcArgs(calleeArgs->getType()),
      secondOrderFunc(secondOrderCallee->getType());
  auto stmt = edsc::call(func, {});
  auto chainedCallResult =
      edsc::call(edsc::call(secondOrderFunc, funcRetIndexType, {func}),
                 builder.getIndexType(), {});
  auto argsCall = edsc::call(funcArgs, {chainedCallResult, chainedCallResult});
  edsc::MLIREmitter(&builder, f->getLoc())
      .bindConstant<ConstantOp>(edsc::Bindable(func),
                                builder.getFunctionAttr(callee.get()))
      .bindConstant<ConstantOp>(edsc::Bindable(funcArgs),
                                builder.getFunctionAttr(calleeArgs.get()))
      .bindConstant<ConstantOp>(
          edsc::Bindable(secondOrderFunc),
          builder.getFunctionAttr(secondOrderCallee.get()))
      .emitStmt(stmt)
      .emitStmt(chainedCallResult)
      .emitStmt(argsCall);

  // clang-format off
  // CHECK-LABEL: @call_indirect
  // CHECK: %f = constant @callee : () -> ()
  // CHECK: %f_0 = constant @callee_args : (index, index) -> ()
  // CHECK: %f_1 = constant @second_order_callee : (() -> ()) -> (() -> index)
  // CHECK: call_indirect %f() : () -> ()
  // CHECK: %0 = call_indirect %f_1(%f) : (() -> ()) -> (() -> index)
  // CHECK: %1 = call_indirect %0() : () -> index
  // CHECK: call_indirect %f_0(%1, %1) : (index, index) -> ()
  // clang-format on
  f->print(llvm::outs());
}

// Inject EDSC-constructed 1-D pointwise-add loop with assignments to scalars,
// `dim` indicates the shape of the memref storing the values.
static std::unique_ptr<Function> makeAssignmentsFunction(int dim) {
  auto memrefType =
      MemRefType::get({dim}, FloatType::getF32(&globalContext()), {}, 0);
  auto f =
      makeFunction("assignments", {}, {memrefType, memrefType, memrefType});
  FuncBuilder builder(f.get());

  edsc::ScopedEDSCContext context;
  edsc::MLIREmitter emitter(&builder, f->getLoc());

  edsc::Expr zero = emitter.zero();
  edsc::Expr one = emitter.one();
  auto args = emitter.makeBoundFunctionArguments(f.get());
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

  return f;
}

TEST_FUNC(assignments_1) {
  auto f = makeAssignmentsFunction(4);

  // clang-format off
  // CHECK-LABEL: func @assignments(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>) {
  // CHECK: affine.for %[[iv:.*]] = 0 to 4 {
  // CHECK:   %[[a:.*]] = load %arg0[%[[iv]]] : memref<4xf32>
  // CHECK:   %[[b:.*]] = load %arg1[%[[iv]]] : memref<4xf32>
  // CHECK:   %[[tmp:.*]] = mulf %[[a]], %[[b]] : f32
  // CHECK:   store %[[tmp]], %arg2[%[[iv]]] : memref<4xf32>
  // clang-format on
  f->print(llvm::outs());
}

TEST_FUNC(assignments_2) {
  auto f = makeAssignmentsFunction(-1);

  // clang-format off
  // CHECK-LABEL: func @assignments(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  // CHECK: affine.for %[[iv:.*]] = {{.*}} to {{.*}} {
  // CHECK:   %[[a:.*]] = load %arg0[%[[iv]]] : memref<?xf32>
  // CHECK:   %[[b:.*]] = load %arg1[%[[iv]]] : memref<?xf32>
  // CHECK:   %[[tmp:.*]] = mulf %[[a]], %[[b]] : f32
  // CHECK:   store %[[tmp]], %arg2[%[[iv]]] : memref<?xf32>
  // clang-format on
  f->print(llvm::outs());
}

int main() {
  RUN_TESTS();
  return 0;
}

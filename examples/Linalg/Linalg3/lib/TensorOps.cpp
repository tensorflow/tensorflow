//===- TensorOps.cpp - Implementation of the linalg TensorOps operation ---===//
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
// This file implements a simple IR operation to create new tensor computation
// operations in the linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "linalg1/Analysis.h"
#include "linalg1/Common.h"
#include "linalg3/Intrinsics.h"
#include "linalg3/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace linalg;
using namespace linalg::intrinsics;

//////////////////////////////////////////////////////////////////////////////
// Implementation of DotOp.
//////////////////////////////////////////////////////////////////////////////
SmallVector<AffineMap, 8> linalg::DotOp::loopsToOperandRangeMaps() {
  // A(K), B(K), C()
  assert(getRanges(*this).size() == 2);
  auto *context = ScopedContext::getContext();
  auto d0 = getAffineDimExpr(0, context); // K
  // A(K), B(K), C()
  //   (d0) -> (d0, d0)(%k)
  return SmallVector<AffineMap, 8>{AffineMap::get(1, 0, {d0}, {}), // A(K)
                                   AffineMap::get(1, 0, {d0}, {}), // B(K)
                                   AffineMap()};                   // C()
}

void linalg::DotOp::emitScalarImplementation(
    llvm::ArrayRef<Value *> parallelIvs, llvm::ArrayRef<Value *> reductionIvs) {
  using IndexedValue = TemplatedIndexedValue<linalg::intrinsics::load,
                                             linalg::intrinsics::store>;
  assert(reductionIvs.size() == 1);
  auto innermostLoop = getForInductionVarOwner(reductionIvs.back());
  auto *body = innermostLoop.getBody();
  using edsc::op::operator+;
  using edsc::op::operator*;
  using edsc::op::operator==;
  using edsc::intrinsics::select;
  ScopedContext scope( // account for affine.terminator in loop.
      FuncBuilder(body, std::prev(body->end(), 1)), innermostLoop.getLoc());
  FloatType fTy = getOperand(0)
                      ->getType()
                      .cast<ViewType>()
                      .getElementType()
                      .cast<FloatType>();
  IndexHandle zero(constant_index(0));
  ValueHandle zerof =
      constant_float(llvm::APFloat::getZero(fTy.getFloatSemantics()), fTy);
  IndexHandle r_i(reductionIvs[0]);
  IndexedValue A(getOperand(0)), B(getOperand(1)), C(getOperand(2));
  ValueHandle cond = (r_i == zero);
  ValueHandle scalarC = select(cond, zerof, *C());
  C() = scalarC + A(r_i) * B(r_i);
}

//////////////////////////////////////////////////////////////////////////////
// Implementation of MatvecOp.
//////////////////////////////////////////////////////////////////////////////
SmallVector<AffineMap, 8> linalg::MatvecOp::loopsToOperandRangeMaps() {
  // A(M, K), B(K), C(M)
  assert(getRanges(*this).size() == 4);
  auto *context = ScopedContext::getContext();
  auto d0 = getAffineDimExpr(0, context); // M
  auto d1 = getAffineDimExpr(1, context); // K
  // A(M, K), B(K), C(M)
  //   (d0, d1) -> (d0, d1, d1, d0)(%m, %k)
  return SmallVector<AffineMap, 8>{
      AffineMap::get(2, 0, {d0, d1}, {}), // A(M, K)
      AffineMap::get(2, 0, {d1}, {}),     // B(K)
      AffineMap::get(2, 0, {d0}, {})};    // C(M)
}

// The body expression for matvec is: C(i) = scalarC + A(i, r_j) * B(r_j)
// The body expression for dot is: C() = A(r_i) * B(r_i);
// So we must drop the `i` loop from the matvec.
void linalg::MatvecOp::writeAsFinerGrainTensorContraction() {
  auto *op = getOperation();
  auto *vA(getInputView(0)), *vB(getInputView(1)), *vC(getOutputView(0));
  auto indexingPosPair = getViewRootIndexing(vA, 0);
  assert(
      llvm::isa_and_nonnull<RangeOp>(indexingPosPair.first->getDefiningOp()));
  // clang-format off
  ScopedContext scope(FuncBuilder(op), op->getLoc());
  IndexHandle i;
  using linalg::common::LoopNestRangeBuilder;
  LoopNestRangeBuilder(&i, ValueHandle(indexingPosPair.first))({
    [&i, &vA, &vB, &vC]() {
      ValueHandle sliceA = slice(vA, i, 0);
      ValueHandle sliceC = slice(vC, i, 0);
      dot(sliceA, vB, sliceC);
      /// NestedBuilders expect handles, we thus return an IndexHandle.
      return IndexHandle();
    }()
  });
  // clang-format on
}

void linalg::MatvecOp::emitScalarImplementation(
    llvm::ArrayRef<Value *> parallelIvs, llvm::ArrayRef<Value *> reductionIvs) {
  using IndexedValue = TemplatedIndexedValue<linalg::intrinsics::load,
                                             linalg::intrinsics::store>;
  assert(reductionIvs.size() == 1);
  auto innermostLoop = getForInductionVarOwner(reductionIvs.back());
  auto *body = innermostLoop.getBody();
  using edsc::op::operator+;
  using edsc::op::operator*;
  using edsc::op::operator==;
  using edsc::intrinsics::select;
  ScopedContext scope( // account for affine.terminator in loop.
      FuncBuilder(body, std::prev(body->end(), 1)), innermostLoop.getLoc());
  FloatType fTy = getOperand(0)
                      ->getType()
                      .cast<ViewType>()
                      .getElementType()
                      .cast<FloatType>();
  IndexHandle i(parallelIvs[0]), r_j(reductionIvs[0]);
  IndexedValue A(getOperand(0)), B(getOperand(1)), C(getOperand(2));
  IndexHandle zero(constant_index(0));
  ValueHandle zerof =
      constant_float(llvm::APFloat::getZero(fTy.getFloatSemantics()), fTy);
  ValueHandle cond = (r_j == zero);
  ValueHandle scalarC = select(cond, zerof, *C(i));
  C(i) = scalarC + A(i, r_j) * B(r_j);
}

//////////////////////////////////////////////////////////////////////////////
// Implementation of Matmul.
//////////////////////////////////////////////////////////////////////////////
SmallVector<AffineMap, 8> linalg::MatmulOp::loopsToOperandRangeMaps() {
  // A(M, K), B(K, N), C(M, N)
  assert(getRanges(*this).size() == 6);
  auto *context = ScopedContext::getContext();
  auto d0 = getAffineDimExpr(0, context); // M
  auto d1 = getAffineDimExpr(1, context); // N
  auto d2 = getAffineDimExpr(2, context); // K
  // A(M, K), B(K, N), C(M, N):
  //   (d0, d1, d2) -> (d0, d2, d2, d1, d0, d1)(%m, %n, %k)
  return SmallVector<AffineMap, 8>{
      AffineMap::get(3, 0, {d0, d2}, {}), // A(M, K)
      AffineMap::get(3, 0, {d2, d1}, {}), // B(K, N)
      AffineMap::get(3, 0, {d0, d1}, {})  // C(M, N)
  };
}

// The body expression for matmul is: C(i, j) = scalarC + A(i, r_k) * B(r_k, j)
// The body expression for matvec is: C(i) = scalarC + A(i, r_j) * B(r_j)
// So we must drop the `j` loop from the matmul.
// This is fine because parallel dimensions permute: we can just do it
// declaratively.
void linalg::MatmulOp::writeAsFinerGrainTensorContraction() {
  auto *op = getOperation();
  auto *vA(getInputView(0)), *vB(getInputView(1)), *vC(getOutputView(0));
  auto indexingPosPair = getViewRootIndexing(vB, 1);
  assert(
      llvm::isa_and_nonnull<RangeOp>(indexingPosPair.first->getDefiningOp()));
  using linalg::common::LoopNestRangeBuilder;
  // clang-format off
  ScopedContext scope(FuncBuilder(op), op->getLoc());
  IndexHandle j;
  LoopNestRangeBuilder(&j, ValueHandle(indexingPosPair.first))({
    [&j, &vA, &vB, &vC]() {
      ValueHandle sliceB = slice(vB, j, 1);
      ValueHandle sliceC = slice(vC, j, 1);
      matvec(vA, sliceB, sliceC);
      /// NestedBuilders expect handles, we thus return an IndexHandle.
      return IndexHandle();
    }()
  });
  // clang-format on
}

void linalg::MatmulOp::emitScalarImplementation(
    llvm::ArrayRef<Value *> parallelIvs, llvm::ArrayRef<Value *> reductionIvs) {
  using IndexedValue = TemplatedIndexedValue<linalg::intrinsics::load,
                                             linalg::intrinsics::store>;
  assert(reductionIvs.size() == 1);
  auto innermostLoop = getForInductionVarOwner(reductionIvs.back());
  auto *body = innermostLoop.getBody();
  using edsc::op::operator+;
  using edsc::op::operator*;
  using edsc::op::operator==;
  using edsc::intrinsics::select;
  ScopedContext scope( // account for affine.terminator in loop.
      FuncBuilder(body, std::prev(body->end(), 1)), innermostLoop.getLoc());
  FloatType fTy = getOperand(0)
                      ->getType()
                      .cast<ViewType>()
                      .getElementType()
                      .cast<FloatType>();
  IndexHandle i(parallelIvs[0]), j(parallelIvs[1]), r_k(reductionIvs[0]);
  IndexedValue A(getOperand(0)), B(getOperand(1)), C(getOperand(2));
  IndexHandle zero(constant_index(0));
  ValueHandle zerof =
      constant_float(llvm::APFloat::getZero(fTy.getFloatSemantics()), fTy);
  ValueHandle cond = r_k == zero;
  ValueHandle scalarC = select(cond, zerof, *C(i, j));
  C(i, j) = scalarC + A(i, r_k) * B(r_k, j);
}

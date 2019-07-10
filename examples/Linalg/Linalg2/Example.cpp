//===- Example.cpp - Our running example ----------------------------------===//
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

// RUN: %p/test | FileCheck %s

#include "TestHarness.h"
#include "linalg1/Common.h"
#include "linalg1/Dialect.h"
#include "linalg2/Intrinsics.h"
#include "linalg2/Ops.h"
#include "linalg2/Transforms.h"
#include "mlir/IR/Function.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace linalg;
using namespace linalg::common;
using namespace linalg::intrinsics;

TEST_FUNC(linalg_ops) {
  MLIRContext context;
  OwningModuleRef module = ModuleOp::create(UnknownLoc::get(&context));
  auto indexType = mlir::IndexType::get(&context);
  mlir::FuncOp f = makeFunction(*module, "linalg_ops",
                                {indexType, indexType, indexType}, {});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());

  // clang-format off
  ValueHandle M(f.getArgument(0)), N(f.getArgument(1)), K(f.getArgument(2)),
    rM = range(constant_index(0), M, constant_index(1)),
    rN = range(constant_index(0), N, constant_index(1)),
    rK = range(constant_index(0), K, constant_index(1)),
    vA = view(alloc(floatMemRefType<2>(&context), {M, K}), {rM ,rK}),
    vB = view(alloc(floatMemRefType<2>(&context), {K, N}), {rK, rN}),
    vC = view(alloc(floatMemRefType<2>(&context), {M, N}), {rM, rN}),
    sB = slice(vB, constant_index(0), 1),
    sC = slice(vC, constant_index(0), 1),
    sA = slice(vA, constant_index(0), 0),
   ssC = slice(sC, constant_index(0), 0);
  matmul(vA, vB, vC);
  matvec(vA, sB, sC);
  dot(sA, sB, ssC);
  ret();
  // CHECK-LABEL: func @linalg_ops(%arg0: index, %arg1: index, %arg2: index) {
  //       CHECK: {{.*}} = linalg.slice {{.*}}[{{.*}}] {dim = 1} : !linalg.view<?x?xf32>, index
  //  CHECK-NEXT: {{.*}} = linalg.slice {{.*}}[{{.*}}] {dim = 1} : !linalg.view<?x?xf32>, index
  //  CHECK-NEXT: {{.*}} = linalg.slice {{.*}}[{{.*}}] {dim = 0} : !linalg.view<?x?xf32>, index
  //  CHECK-NEXT: {{.*}} = linalg.slice {{.*}}[{{.*}}] {dim = 0} : !linalg.view<?xf32>, index
  //       CHECK: linalg.matmul({{.*}}, {{.*}}, {{.*}}) : !linalg.view<?x?xf32>
  //  CHECK-NEXT: linalg.matvec({{.*}}, {{.*}}, {{.*}}) : !linalg.view<?xf32>
  //  CHECK-NEXT: linalg.dot({{.*}}, {{.*}}, {{.*}}) : !linalg.view<f32>
  // clang-format on

  cleanupAndPrintFunction(f);
}

TEST_FUNC(linalg_ops_folded_slices) {
  MLIRContext context;
  OwningModuleRef module = ModuleOp::create(UnknownLoc::get(&context));
  auto indexType = mlir::IndexType::get(&context);
  mlir::FuncOp f = makeFunction(*module, "linalg_ops_folded_slices",
                                {indexType, indexType, indexType}, {});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());

  // clang-format off
  ValueHandle M(f.getArgument(0)), N(f.getArgument(1)), K(f.getArgument(2)),
    rM = range(constant_index(0), M, constant_index(1)),
    rN = range(constant_index(0), N, constant_index(1)),
    rK = range(constant_index(0), K, constant_index(1)),
    vA = view(alloc(floatMemRefType<2>(&context), {M, K}), {rM, rK}),
    vB = view(alloc(floatMemRefType<2>(&context), {K, N}), {rK, rN}),
    vC = view(alloc(floatMemRefType<2>(&context), {M, N}), {rM, rN}),
    sB = slice(vB, constant_index(0), 1),
    sC = slice(vC, constant_index(0), 1),
    sA = slice(vA, constant_index(0), 0),
   ssC = slice(sC, constant_index(0), 0);
  matmul(vA, vB, vC);
  matvec(vA, sB, sC);
  dot(sA, sB, ssC);
  ret();
  // CHECK-LABEL: func @linalg_ops_folded_slices(%arg0: index, %arg1: index, %arg2: index) {
  //   CHECK-NOT: linalg.slice
  //       CHECK: linalg.matmul({{.*}}, {{.*}}, {{.*}}) : !linalg.view<?x?xf32>
  //  CHECK-NEXT: linalg.matvec({{.*}}, {{.*}}, {{.*}}) : !linalg.view<?xf32>
  //  CHECK-NEXT: linalg.dot({{.*}}, {{.*}}, {{.*}}) : !linalg.view<f32>
  // clang-format on

  f.walk<SliceOp>([](SliceOp slice) {
    auto *sliceResult = slice.getResult();
    auto viewOp = emitAndReturnFullyComposedView(sliceResult);
    sliceResult->replaceAllUsesWith(viewOp.getResult());
    slice.erase();
  });

  cleanupAndPrintFunction(f);
}
int main() {
  mlir::registerDialect<linalg::LinalgDialect>();
  RUN_TESTS();
  return 0;
}
